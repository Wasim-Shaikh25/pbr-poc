# Whole-model quantizer WITH cross-layer drift correction ("teacher targets").
#
# Same proven per-tensor method as quantize_full_model.py (shared rotation + one
# global VQ codebook fit + GPTQ-style column feedback), plus one new idea aimed
# squarely at the whole-model failure mode:
#
#   Plain sequential quant makes layer l see the DRIFTED input x_q that the
#   partly-quantized model actually produces, and reproduces W . x_q. It faithfully
#   carries upstream quantization drift forward -- and 24 layers of that compounds
#   (the +36% PPL hole, which coherence_probe.py showed is NOT coherent error
#   build-up, so it must be this forward-propagating drift + nonlinear amplification).
#
#   Instead, quantize each layer to OUTPUT THE CLEAN TARGET  x_clean . W^T  despite
#   receiving the drifted input x_q -- using the quantization freedom to actively
#   cancel accumulated drift. This is delta-sigma / error-feedback noise shaping
#   applied across layers, not just across columns.
#
#   Least squares gives an "effective weight" to quantize instead of W:
#       C = x_q^T x_clean / n     (in x in cross-covariance, drifted vs clean input)
#       H = x_q^T x_q     / n     (the usual GPTQ Hessian, from the drifted input)
#       W_eff = W . C^T . H^{-1}   (with damped H^{-1}; == W exactly when x_q==x_clean)
#   Quantizing W_eff under H reproduces the clean target as closely as the quantizer
#   and the irreducible "orthogonal residual" (drift outside x_q's row space) allow.
#
# Needs two model copies in memory (clean + being-quantized). No resume (kept simple
# on purpose). Usage:
#   PBR_STAGES=512,256,64 PBR_SAMPLES=32 python quantize_teacher.py ./qwen05b ./out
#   PBR_TEACHER=0 ... reverts to plain sequential (for an apples-to-apples A/B).
import os, sys, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
STAGES = [int(x) for x in os.environ.get("PBR_STAGES", "512,256,64").split(",")]
SAMPLES = int(os.environ.get("PBR_SAMPLES", "32"))
SEQ = int(os.environ.get("PBR_SEQ", "512"))
TEACHER = os.environ.get("PBR_TEACHER", "1") == "1"
DAMP = float(os.environ.get("PBR_DAMP", "0.01"))
TEACHER_DAMP = float(os.environ.get("PBR_TEACHER_DAMP", "0.05"))  # heavier: W_eff wants a stable H^{-1}
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
D, K = 8, 256

print(f"stages={STAGES} teacher={TEACHER} samples={SAMPLES} seq={SEQ}", flush=True)
tok = AutoTokenizer.from_pretrained(src)
print("loading quant model ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
clean = None
if TEACHER:
    print("loading clean (teacher) model ...", flush=True)
    clean = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
layers = model.model.layers
n_layers = len(layers)
which = [int(x) for x in os.environ["PBR_LAYERS"].split(",")] if os.environ.get("PBR_LAYERS") else list(range(n_layers))

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)
rot_cache = {}
def rotation(n):
    if n not in rot_cache: rot_cache[n] = ortho(n, n)
    return rot_cache[n]

CHUNK = 50000
def assign_rows(V, C, chunk=CHUNK):
    return np.concatenate([((V[i:i+chunk]**2).sum(1)[:, None] - 2*V[i:i+chunk]@C.T
                             + (C**2).sum(1)[None]).argmin(1) for i in range(0, len(V), chunk)])
def kmeans(V, k, it=10, chunk=CHUNK):
    rng = np.random.default_rng(0)
    C = V[rng.choice(len(V), k, replace=False)]
    for _ in range(it):
        a = assign_rows(V, C, chunk)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = V[m].mean(0)
    return C

def quantize_matrix(W, H, stages):
    """Rotation + global VQ fit + GPTQ column feedback. W is the (already effective) weight."""
    O, I = W.shape
    Ro, Ri = rotation(O), rotation(I)
    Wr = Ro @ W @ Ri
    Hr = Ri.T @ H @ Ri
    Hr = Hr + DAMP * np.mean(np.diag(Hr)) * np.eye(I)
    U = np.linalg.cholesky(np.linalg.inv(Hr)).T
    s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12
    pool = (Wr / s).reshape(-1, D)
    books, Rres = [], pool.copy()
    for k in stages:
        C = kmeans(Rres, k); a = assign_rows(Rres, C)
        books.append(C); Rres = Rres - C[a]
    T = Wr.copy(); Q = np.zeros_like(T)
    for j in range(0, I, D):
        b = slice(j, j + D); R = T[:, b] / s; q = np.zeros_like(R)
        for C in books:
            a = ((R**2).sum(1)[:, None] - 2*R@C.T + (C**2).sum(1)[None]).argmin(1)
            q += C[a]; R -= C[a]
        Q[:, b] = q * s
        if j + D < I:
            T[:, j+D:] -= (T[:, b] - Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+D:])
    return Ro.T @ Q @ Ri.T

class StopForward(Exception): pass

def capture_inputs(m, li, targets, batch):
    """Run model m forward up to layer li, capturing each target submodule's input."""
    block = m.model.layers[li]
    buf = {name: [] for name in targets}
    hooks = [block.get_submodule(name).register_forward_hook(
                lambda mod, inp, o, name=name: buf[name].append(inp[0].detach().double()))
             for name in targets]
    stop = block.register_forward_hook(lambda mod, i, o: (_ for _ in ()).throw(StopForward()))
    with torch.no_grad():
        for i in range(0, SAMPLES, 8):
            try: m(batch[i:i+8])
            except StopForward: pass
    stop.remove()
    for h in hooks: h.remove()
    return {name: torch.cat(buf[name]).reshape(-1, buf[name][0].shape[-1]).numpy() for name in targets}

# calibration batch
from datasets import load_dataset
try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(_wt["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids
starts = np.random.default_rng(0).integers(0, ids.shape[1] - SEQ, SAMPLES)
batch = torch.cat([ids[:, s:s+SEQ] for s in starts], dim=0)

def exists(block, name):
    try: block.get_submodule(name); return True
    except AttributeError: return False

t0 = time.time()
for li in which:
    block = layers[li]
    targets = [n for n in TARGETS if exists(block, n)]
    Xq = capture_inputs(model, li, targets, batch)          # drifted inputs (quant model)
    Xc = capture_inputs(clean, li, targets, batch) if TEACHER else None  # clean inputs
    for name in targets:
        mod = block.get_submodule(name)
        W = mod.weight.detach().double().numpy()
        xq = Xq[name]; n = len(xq)
        H = xq.T @ xq / n
        if TEACHER:
            xc = Xc[name]
            Hc = H + TEACHER_DAMP * np.mean(np.diag(H)) * np.eye(H.shape[0])
            Ceff = xq.T @ xc / n
            W_eff = W @ Ceff.T @ np.linalg.inv(Hc)
        else:
            W_eff = W
        Wq = quantize_matrix(W_eff, H, STAGES)
        with torch.no_grad(): mod.weight.copy_(torch.from_numpy(Wq).to(mod.weight.dtype))
        del xq
    print(f"layer {li:2d}/{n_layers-1} quantized {len(targets)} tensors [{time.time()-t0:6.1f}s]", flush=True)

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"\nSaved to {out}  (teacher={TEACHER}, stages={STAGES}, ~{sum(np.log2(k) for k in STAGES)/D:.3f} bpw)")
print(f"Eval:  python eval_ppl.py {out}")
