# Compare against REAL, official GGUF quants (downloaded, not self-made). Load each GGUF
# into transformers (dequantized to torch), compute its true bpw from file size, and run
# the SAME WikiText-2 test PPL harness that produced our VQ numbers -> directly comparable.
#
# Usage: python gguf_eval.py q2_k q3_k_m q4_k_m
import sys, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
quants = sys.argv[1:] or ["q2_k", "q3_k_m", "q4_k_m"]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
try: _te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
except Exception: _te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tids = tok("\n\n".join(_te["text"]), return_tensors="pt").input_ids

def ppl(model):
    seq, nll, n = 2048, 0.0, 0
    with torch.no_grad():
        for i in range(0, tids.shape[1] - 1, seq):
            x = tids[:, i:i+seq]
            if x.shape[1] < 2: break
            nll += model(x, labels=x).loss.float().item() * (x.shape[1]-1); n += x.shape[1]-1
    return float(np.exp(nll / n))

print(f"{'quant':<12}{'bpw':>8}{'PPL':>10}   (ours: VQ@3.125=16.829, VQ+corr@3.322=16.551; base=14.247)")
for q in quants:
    fn = f"qwen2.5-0.5b-instruct-{q}.gguf"
    path = hf_hub_download(REPO, fn)
    sz = os.path.getsize(path)
    model = AutoModelForCausalLM.from_pretrained(REPO, gguf_file=fn, dtype=torch.float32).eval()
    nparam = sum(p.numel() for p in model.parameters())
    bpw = sz * 8.0 / nparam
    p = ppl(model)
    print(f"{q:<12}{bpw:>8.3f}{p:>10.3f}", flush=True)
    del model
