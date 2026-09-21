"""Capture calibration activations for the Phase 0/1 tensors.

Same procedure as pbr_ladder/capture_activations.py, one model load:
WikiText-2 train[:3000], first 16384 tokens, chunks of 512, then a
seed-0 subsample of 4096 rows per tensor. Model dtype is fp32 when
PBR_CPU_FP32 is set (this script forces that).
"""
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PBR_CPU_FP32"] = "1"
NAME = "/workspace/pbr_ladder/qwen05b"
OUT_DIR = "/workspace/pbr_ladder"

LAYERS_PHASE0 = (2, 12, 21)
PHASE0_MODULES = ("mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj")
PHASE1_MODULES = (
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
SHORT = {
    "mlp.gate_proj": "gate",
    "mlp.up_proj": "up",
    "mlp.down_proj": "down",
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k",
    "self_attn.v_proj": "v",
    "self_attn.o_proj": "o",
}


def targets():
    out = []
    for layer in LAYERS_PHASE0:
        for module in PHASE0_MODULES:
            out.append((layer, module))
        if layer == 12:
            for module in PHASE1_MODULES:
                out.append((layer, module))
    return out


def main():
    t0 = time.time()
    print("loading model", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:3000]
    )
    ids = tok(text, return_tensors="pt").input_ids[:, :16384]
    print("calib tokens", ids.shape[1], "elapsed", round(time.time() - t0, 1), flush=True)

    rows = {}
    hooks = []
    for layer, module in targets():
        key = (layer, module)
        target = model.model.layers[layer].get_submodule(module)
        rows[key] = []

        def make_hook(key):
            def hook(m, inp, o):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
                rows[key].append(x)

            return hook

        hooks.append(target.register_forward_hook(make_hook(key)))

    with torch.no_grad():
        for i in range(0, ids.shape[1], 512):
            model(ids[:, i : i + 512])
            print("chunk", i, "elapsed", round(time.time() - t0, 1), flush=True)
    for h in hooks:
        h.remove()

    for (layer, module), parts in rows.items():
        A = torch.cat(parts).numpy()
        n = len(A)
        A = A[np.random.default_rng(0).choice(n, min(4096, n), replace=False)]
        out = os.path.join(OUT_DIR, f"acts_L{layer}_{SHORT[module]}.npy")
        np.save(out, A)
        print("saved", out, A.shape, A.dtype, "from_rows", n, flush=True)

    # Shared-input checks (same forward, so these should match exactly).
    def load(layer, short):
        return np.load(os.path.join(OUT_DIR, f"acts_L{layer}_{short}.npy"))

    g, u = load(12, "gate"), load(12, "up")
    q, k, v = load(12, "q"), load(12, "k"), load(12, "v")
    print("L12 gate==up", np.array_equal(g, u), "max_abs", float(np.max(np.abs(g - u))), flush=True)
    print("L12 q==k", np.array_equal(q, k), "q==v", np.array_equal(q, v), flush=True)
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
