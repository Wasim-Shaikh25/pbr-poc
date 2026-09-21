# Capture real input activations for one linear layer (calibration data for push_*.py).
# Usage: python capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.gate_proj acts_L12_gate.npy
# UNTESTED in the author's sandbox (no Hugging Face access). Needs: torch, transformers, datasets.
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
name, layer, module, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tok = AutoTokenizer.from_pretrained(name)
import os
dtype = torch.float32 if os.environ.get("PBR_CPU_FP32") else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).eval()
target = model.model.layers[layer].get_submodule(module)
import os
if os.environ.get("PBR_TEXT"):   # local text file instead of downloading WikiText-2 (train split)
    text = open(os.environ["PBR_TEXT"], encoding="utf-8").read()
else:
    from datasets import load_dataset
    # Legacy id "wikitext" is the same corpus as Salesforce/wikitext (WikiText-2
    # raw v1). Current huggingface_hub rejects the un-namespaced id.
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(_wt["text"][:3000])
ids = tok(text, return_tensors="pt").input_ids[:, :16384]   # calibration = TRAIN split only
rows = []
hook = target.register_forward_hook(
    lambda m, inp, o: rows.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()))
with torch.no_grad():
    for i in range(0, ids.shape[1], 512):
        model(ids[:, i:i+512])
hook.remove()
A = torch.cat(rows).numpy()
A = A[np.random.default_rng(0).choice(len(A), min(4096, len(A)), replace=False)]
np.save(out, A)
print("saved", out, A.shape)
