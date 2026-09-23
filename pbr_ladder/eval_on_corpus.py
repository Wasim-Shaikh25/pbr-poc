# Cross-domain calibration check. Evaluate OUR WikiText-calibrated model (proj@4.0 +
# embed@8) AND the GGUF quants on a DIFFERENT domain (C4 web text we never calibrated on),
# identical harness. If our margin over GGUF holds on C4, the win isn't WikiText overfit.
#
# Usage: python eval_on_corpus.py <orig_dir> <our_quant_dir> [corpus=c4]
import sys, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset
orig_dir, our_dir = sys.argv[1], sys.argv[2]
corpus = sys.argv[3] if len(sys.argv) > 3 else "c4"
GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
tok = AutoTokenizer.from_pretrained(orig_dir)

# ---- build eval text for the chosen corpus (~300k tokens, matching WikiText test size) ----
def build_text():
    if corpus == "wikitext":
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(d["text"])
    if corpus == "ptb":
        d = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        return "\n\n".join(d["sentence"])
    # C4 en validation, streamed
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf, ntok = [], 0
    for ex in ds:
        buf.append(ex["text"]); ntok += len(ex["text"].split())
        if ntok > 260000: break            # ~word count -> ~300k+ tokens
    return "\n\n".join(buf)

text = build_text()
tids = tok(text, return_tensors="pt").input_ids
print(f"corpus={corpus}  tokens={tids.shape[1]:,}", flush=True)

def ppl(model):
    seq, nll, n = 2048, 0.0, 0
    with torch.no_grad():
        for i in range(0, tids.shape[1]-1, seq):
            x = tids[:, i:i+seq]
            if x.shape[1] < 2: break
            nll += model(x, labels=x).loss.float().item()*(x.shape[1]-1); n += x.shape[1]-1
    return float(np.exp(nll/n))

def qrow(W, b):
    lo = W.min(1, keepdims=True); hi = W.max(1, keepdims=True)
    L = (1 << b) - 1; sc = (hi - lo) / L; sc[sc == 0] = 1.0
    return lo + np.round((W - lo) / sc) * sc

results = []
# ---- our model: proj@4.0 + embed@8 ----
model = AutoModelForCausalLM.from_pretrained(our_dir, dtype=torch.float32).eval()
with safe_open(os.path.join(orig_dir, "model.safetensors"), "pt") as f:
    emb = f.get_tensor("model.embed_tokens.weight").float().numpy().astype(np.float64)
model.get_submodule("model.embed_tokens").weight.data.copy_(torch.from_numpy(qrow(emb, 8)).float())
results.append(("OURS proj@4.0+embed@8 (5.10bpw)", ppl(model))); del model

# ---- GGUF quants ----
for q in ["q2_k", "q3_k_m", "q4_k_m"]:
    fn = f"qwen2.5-0.5b-instruct-{q}.gguf"
    m = AutoModelForCausalLM.from_pretrained(GGUF_REPO, gguf_file=fn, dtype=torch.float32).eval()
    results.append((f"GGUF {q}", ppl(m))); del m

print(f"\n=== {corpus.upper()} PPL (cross-domain: calibrated on WikiText, tested on {corpus}) ===")
for name, p in results:
    print(f"  {name:<34} {p:8.3f}", flush=True)
