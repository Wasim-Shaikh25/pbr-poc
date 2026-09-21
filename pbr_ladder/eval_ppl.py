# WikiText-2 TEST perplexity of a model, optionally after replacing some weights.
# Usage: python eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct [replacements.npz]
#   replacements.npz: keys = parameter names (e.g. model.layers.12.mlp.gate_proj.weight), values = float arrays
# UNTESTED in the author's sandbox (no Hugging Face access). Needs: torch, transformers, datasets.
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
if len(sys.argv) > 2:
    rep = np.load(sys.argv[2]); params = dict(model.named_parameters())
    for k in rep.files:
        with torch.no_grad():
            params[k].copy_(torch.from_numpy(rep[k]).to(params[k].dtype))
        print("replaced", k)
text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids
seq, nll, n = 2048, 0.0, 0
with torch.no_grad():
    for i in range(0, ids.shape[1] - 1, seq):
        x = ids[:, i:i+seq]
        if x.shape[1] < 2: break
        loss = model(x, labels=x).loss.float()
        nll += loss.item() * (x.shape[1] - 1); n += x.shape[1] - 1
print(f"WikiText-2 test perplexity: {np.exp(nll / n):.3f}")
