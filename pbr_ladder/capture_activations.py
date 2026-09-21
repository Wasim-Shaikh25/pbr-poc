# Capture real input activations for one linear layer (calibration data for push_*.py).
# Usage: python capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.gate_proj acts_L12_gate.npy
# UNTESTED in the author's sandbox (no Hugging Face access). Needs: torch, transformers, datasets.
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
name, layer, module, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
target = model.model.layers[layer].get_submodule(module)
text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:3000])
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
