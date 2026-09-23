# FAST rank sweep (#2). One forward pass over the VQ-quantized model accumulates each
# tensor's activation Gram H (float32, low RAM); compute the rank-64 low-rank-correction
# factors ONCE per tensor; then eval WikiText-2 PPL at ranks 8/16/32/64 by truncating +
# 8-bit quantizing the factors. Approximate (upstream not re-corrected during collection),
# but the correction is small so the ranking is reliable. Compares to VQ@3.125 = 16.829.
#
# Usage: python onepass_sweep.py <orig_dir> <quantized_dir>
import sys, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
orig_dir, q_dir = sys.argv[1], sys.argv[2]
SAMPLES, SEQ = int(os.environ.get("PBR_SAMPLES", 64)), int(os.environ.get("PBR_SEQ", 512))
RANKS = [8, 16, 32, 64]
TARGETS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

def q8(X):
    m = np.abs(X).max()
    return np.round(X / m * 127) * (m / 127) if m > 0 else X

tok = AutoTokenizer.from_pretrained(orig_dir)
print(f"loading quantized model {q_dir} ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(q_dir, dtype=torch.float32).eval()
layers = model.model.layers; n_layers = len(layers)
qsf = os.path.join(q_dir, "model.safetensors"); osf = os.path.join(orig_dir, "model.safetensors")

from datasets import load_dataset
try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n\n".join(_wt["text"][:4000]), return_tensors="pt").input_ids
starts = np.random.default_rng(0).integers(0, ids.shape[1] - SEQ, SAMPLES)
batch = torch.cat([ids[:, s:s+SEQ] for s in starts], dim=0)

# ---- one forward pass: accumulate H per tensor (float32) ----
Hacc = {}   # key -> (sum XtX float32, count)
keymap = {}  # module -> weight key
def mk_hook(key):
    def hook(m, inp, o):
        x = inp[0].detach().reshape(-1, m.in_features).float()
        if key in Hacc: Hacc[key][0] += (x.T @ x).numpy(); Hacc[key][1] += x.shape[0]
        else: Hacc[key] = [(x.T @ x).numpy().astype(np.float32), x.shape[0]]
    return hook
hooks = []
for L in range(n_layers):
    for name in TARGETS:
        try: mod = layers[L].get_submodule(name)
        except AttributeError: continue
        key = f"model.layers.{L}.{name}.weight"; keymap[key] = mod
        hooks.append(mod.register_forward_hook(mk_hook(key)))
t0 = time.time()
with torch.no_grad():
    for i in range(0, SAMPLES, 8):
        model(batch[i:i+8])
for h in hooks: h.remove()
print(f"activation pass done [{time.time()-t0:.0f}s], {len(Hacc)} tensors", flush=True)

# ---- rank-64 factors per tensor ----
FAC = {}   # key -> (Pc[:, :64], Sc[:64], Qc[:64])  float32
with safe_open(osf, "pt") as fo, safe_open(qsf, "pt") as fq:
    for key in list(Hacc.keys()):
        H = (Hacc[key][0] / Hacc[key][1]).astype(np.float64)
        Worig = fo.get_tensor(key).float().numpy().astype(np.float64)
        Wq = fq.get_tensor(key).float().numpy().astype(np.float64)
        ev, Ug = np.linalg.eigh(H); ev = ev[::-1]; Ug = Ug[:, ::-1]
        sq = np.sqrt(np.clip(ev, 0, None)); inv = np.where(sq > 1e-9, 1.0/sq, 0.0)
        E = Worig - Wq
        B = (E @ Ug) * sq[None, :]
        P, S, Qt = np.linalg.svd(B, full_matrices=False)
        r64 = min(64, len(S))
        C = ((P[:, :r64] * S[:r64]) @ Qt[:r64] * inv[None, :]) @ Ug.T
        Pc, Sc, Qc = np.linalg.svd(C, full_matrices=False)
        r64 = min(64, len(Sc))
        FAC[key] = (Pc[:, :r64].astype(np.float32), Sc[:r64].astype(np.float32), Qc[:r64].astype(np.float32))
        del Hacc[key]
print(f"factors computed [{time.time()-t0:.0f}s]", flush=True)

def eval_ppl():
    try: _te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tids = tok("\n\n".join(_te["text"]), return_tensors="pt").input_ids
    seq, nll, n = 2048, 0.0, 0
    with torch.no_grad():
        for i in range(0, tids.shape[1] - 1, seq):
            x = tids[:, i:i+seq]
            if x.shape[1] < 2: break
            nll += model(x, labels=x).loss.float().item() * (x.shape[1]-1); n += x.shape[1]-1
    return float(np.exp(nll / n))

# bits per weight added by rank r
def bpw_add(r):
    num = sum(r * (FAC[k][0].shape[0] + FAC[k][2].shape[1]) * 8 for k in FAC)
    den = sum(FAC[k][0].shape[0] * FAC[k][2].shape[1] for k in FAC)
    return num / den

print(f"\n{'rank':>6}{'+bpw':>8}{'total bpw':>11}{'PPL':>9}   (VQ@3.125=16.829, baseline=14.247)")
for r in RANKS:
    with safe_open(qsf, "pt") as fq:
        for key, mod in keymap.items():
            Wq = fq.get_tensor(key).float().numpy().astype(np.float64)
            Pc, Sc, Qc = FAC[key]; rr = min(r, len(Sc))
            Uf = q8(Pc[:, :rr] * np.sqrt(Sc[:rr])); Vf = q8(np.sqrt(Sc[:rr])[:, None] * Qc[:rr])
            with torch.no_grad(): mod.weight.copy_(torch.from_numpy(Wq + Uf @ Vf).to(mod.weight.dtype))
    ppl = eval_ppl(); db = bpw_add(r)
    print(f"{r:>6}{db:>8.3f}{3.125+db:>11.3f}{ppl:>9.3f}", flush=True)
