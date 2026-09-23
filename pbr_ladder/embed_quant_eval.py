# The overlooked lever: quantize the TOKEN EMBEDDINGS (tied = lm_head), which we left at
# fp16 (27% of a 0.5B model). Take our VQ-projection model, quantize embed at 8/6/5/4-bit
# (per-row min-max), eval WikiText-2 PPL, and report TOTAL effective bpw over ALL params
# -> the honest, GGUF-comparable metric. Goal: get under q2_k's 5.27 bpw / 16.527 PPL.
#
# Usage: python embed_quant_eval.py <orig_dir> <quantized_dir>
import sys, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset
orig_dir, q_dir = sys.argv[1], sys.argv[2]
BITSET = [16, 8, 6, 5, 4]
EKEY = "model.embed_tokens.weight"

tok = AutoTokenizer.from_pretrained(orig_dir)
model = AutoModelForCausalLM.from_pretrained(q_dir, dtype=torch.float32).eval()
with safe_open(os.path.join(orig_dir, "model.safetensors"), "pt") as f:
    embed_orig = f.get_tensor(EKEY).float().numpy().astype(np.float64)   # (vocab, hidden)
emb_mod = model.get_submodule("model.embed_tokens")

try: _te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
except Exception: _te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tids = tok("\n\n".join(_te["text"]), return_tensors="pt").input_ids
def ppl():
    seq, nll, n = 2048, 0.0, 0
    with torch.no_grad():
        for i in range(0, tids.shape[1]-1, seq):
            x = tids[:, i:i+seq]
            if x.shape[1] < 2: break
            nll += model(x, labels=x).loss.float().item()*(x.shape[1]-1); n += x.shape[1]-1
    return float(np.exp(nll/n))

def qrow(W, b):                       # per-row asymmetric min-max b-bit
    if b >= 16: return W
    lo = W.min(1, keepdims=True); hi = W.max(1, keepdims=True)
    L = (1 << b) - 1; sc = (hi - lo) / L; sc[sc == 0] = 1.0
    return lo + np.round((W - lo) / sc) * sc

embed = 151936 * 896                  # tied embed/lm_head param count
proj = 357826560                      # projections we VQ-quantized
PROJ_BPW = float(os.environ.get("PBR_PROJ_BPW", "3.125"))
def eff_bpw(embed_bits, proj_bpw=PROJ_BPW):
    return (embed * embed_bits + proj * proj_bpw) / (embed + proj)
def total_mb(embed_bits, proj_bpw=PROJ_BPW):
    return (embed * embed_bits + proj * proj_bpw) / 8 / 1048576

print(f"proj@{PROJ_BPW} bpw. GGUF refs: q2_k 5.27bpw/16.527, q3_k_m 5.49bpw/15.722 ; base 14.247")
print(f"{'embed bits':>11}{'eff bpw(all)':>14}{'total MB':>10}{'PPL':>10}")
for b in BITSET:
    Wq = qrow(embed_orig, b)
    with torch.no_grad(): emb_mod.weight.copy_(torch.from_numpy(Wq).to(emb_mod.weight.dtype))
    print(f"{b:>11}{eff_bpw(b):>14.3f}{total_mb(b):>10.1f}{ppl():>10.3f}", flush=True)
