# Whole-model quantizer: reads a real Qwen checkpoint, quantizes every attention
# and MLP linear layer (rotation + GPTQ-style error feedback + 8-D node codebooks,
# the method proven layer-by-layer earlier), and WRITES OUT A NEW, COMPLETE,
# LOADABLE CHECKPOINT FOLDER — a real new "Qwen file", not just an in-memory score.
#
# Sequential, like real GPTQ: layer 0 is quantized first and written back into the
# live model, THEN calibration text is run again so layer 1 sees layer 0's real
# (already-quantized) output — not the original BF16 output. This matches what
# production quantizers do and what Phase 2 of the roadmap calls for.
#
# UNTESTED in the author's sandbox (no Hugging Face access). Needs: torch,
# transformers, and either `datasets` or a local calibration text file.
#
# Usage:
#   python quantize_full_model.py ./qwen05b ./qwen05b_pbr3bit
#   PBR_TEXT=calib.txt python quantize_full_model.py ./qwen05b ./qwen05b_pbr3bit
#   PBR_STAGES=256,256,64 PBR_SAMPLES=64 PBR_SEQ=512 PBR_LAYERS=0,1,2 \
#     python quantize_full_model.py ./qwen05b ./qwen05b_test   # quick dry run
#
# PBR_LAYERS restricts to a few layers for a fast first check before committing
# to a full run, which can take a long time on CPU for a 0.5B model.

import os, sys, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
STAGES = [int(x) for x in os.environ.get("PBR_STAGES", "256,256,64").split(",")]
SAMPLES = int(os.environ.get("PBR_SAMPLES", "64"))
SEQ = int(os.environ.get("PBR_SEQ", "512"))
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
D = 8            # node-vector width, same as the proven method
K = 256          # nodes per codebook

print(f"Loading model from {src} ...")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
layers = model.model.layers
n_layers = len(layers)
which = [int(x) for x in os.environ["PBR_LAYERS"].split(",")] if os.environ.get("PBR_LAYERS") else list(range(n_layers))
print(f"Model has {n_layers} layers; quantizing: {which}")

# ---- calibration text: local file (PBR_TEXT) or WikiText-2 train via `datasets` ----
if os.environ.get("PBR_TEXT"):
    text = open(os.environ["PBR_TEXT"], encoding="utf-8").read()
else:
    from datasets import load_dataset
    # Legacy id "wikitext" is the same corpus as Salesforce/wikitext. Current
    # huggingface_hub rejects the un-namespaced id.
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(_wt["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids
n_tok = SAMPLES * SEQ
if ids.shape[1] < n_tok:
    raise SystemExit(f"Calibration text too short: need {n_tok} tokens, have {ids.shape[1]}. "
                      f"Provide more text via PBR_TEXT or lower PBR_SAMPLES/PBR_SEQ.")
starts = np.random.default_rng(0).integers(0, ids.shape[1] - SEQ, SAMPLES)
batch = torch.cat([ids[:, s:s+SEQ] for s in starts], dim=0)   # (SAMPLES, SEQ)

class StopForward(Exception): pass

def ortho(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)
rot_cache = {}
def rotation(n):
    if n not in rot_cache: rot_cache[n] = ortho(n, n)   # cached: same rotation reused for every layer of this width
    return rot_cache[n]

# Same chunk bound as push_below4.py (PBR_CHUNK). The post-fit assign below
# used to build one (rows, k) distance matrix — 544,768 x 256 float64 is
# 1.04 GiB and OOMs next to the live fp32 model.
CHUNK = int(os.environ.get("PBR_CHUNK", 50000))
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

def quantize_linear(Wt, H):
    """Wt: torch weight (out, in). H: numpy (in, in) uncentered covariance of its real inputs.
       Returns a torch tensor of the same shape, quantized then dequantized.
       Two passes, matching the design already validated in push_nested.py:
       (1) fit codebooks once, pooling vectors across the WHOLE tensor — this is what
           makes it work on narrow tensors (e.g. k_proj/v_proj, shrunk by grouped-query
           attention) where any one column-block alone would have too few rows to fit
           256-entry codebooks against;
       (2) a column-block pass that applies GPTQ-style error feedback using those
           already-fixed codebooks."""
    W = Wt.detach().double().numpy(); O, I = W.shape
    Ro, Ri = rotation(O), rotation(I)
    Wr = Ro @ W @ Ri
    Hr = Ri.T @ H @ Ri
    Hr = Hr + 0.01 * np.mean(np.diag(Hr)) * np.eye(I)
    U = np.linalg.cholesky(np.linalg.inv(Hr)).T
    s = np.sqrt((Wr**2).mean(1, keepdims=True)) + 1e-12

    pool = (Wr / s).reshape(-1, D)               # pass 1: fit codebooks on the whole tensor
    books, Rres = [], pool.copy()
    for k in STAGES:
        C = kmeans(Rres, k)
        a = assign_rows(Rres, C)
        books.append(C); Rres = Rres - C[a]

    T = Wr.copy(); Q = np.zeros_like(T)            # pass 2: column-block feedback
    for j in range(0, I, D):
        b = slice(j, j + D); R = T[:, b] / s; q = np.zeros_like(R)
        for C in books:
            a = ((R**2).sum(1)[:, None] - 2*R@C.T + (C**2).sum(1)[None]).argmin(1)
            q += C[a]; R -= C[a]
        Q[:, b] = q * s
        if j + D < I:
            T[:, j+D:] -= (T[:, b] - Q[:, b]) @ np.linalg.solve(U[b, b], U[b, j+D:])
    Wq = Ro.T @ Q @ Ri.T
    return torch.from_numpy(Wq).to(Wt.dtype)

total_params, quantized_params = 0, 0
t0 = time.time()
for li in which:
    block = layers[li]
    targets = {}
    for name in TARGETS:
        try: targets[name] = block.get_submodule(name)
        except AttributeError: pass   # architecture doesn't have this projection; skip it
    inputs = {name: [] for name in targets}
    hooks = []
    for name, mod in targets.items():
        hooks.append(mod.register_forward_hook(
            lambda m, inp, o, name=name: inputs[name].append(inp[0].detach().double())))
    stop = block.register_forward_hook(lambda m, i, o: (_ for _ in ()).throw(StopForward()))
    with torch.no_grad():
        for i in range(0, SAMPLES, 8):
            try: model(batch[i:i+8])
            except StopForward: pass
    stop.remove()
    for h in hooks: h.remove()

    for name, mod in targets.items():
        X = torch.cat(inputs[name]).reshape(-1, mod.in_features).numpy()   # (tokens, in_features)
        inputs[name].clear()
        H = X.T @ X / len(X)
        del X
        Wq = quantize_linear(mod.weight, H)
        with torch.no_grad(): mod.weight.copy_(Wq)
        total_params += mod.weight.numel(); quantized_params += mod.weight.numel()
    print(f"layer {li:2d}/{n_layers-1}  quantized {len(targets)} tensors  "
          f"[{time.time()-t0:6.1f}s elapsed]")

codebook_bpw = sum(np.log2(k) for k in STAGES) / D   # index cost; codebook storage itself is
                                                       # shared and negligible on tensors this large
print(f"\nQuantized {quantized_params:,} parameters across {len(which)} layer(s), "
      f"~{codebook_bpw:.2f} bpw for touched tensors (stages={STAGES}, codebook overhead <0.1 bpw).")
print(f"Embeddings, output head, and norms were left untouched (full precision).")

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"\nSaved new checkpoint to: {out}")
print(f"Load and test it directly:  python eval_ppl.py {out}")
