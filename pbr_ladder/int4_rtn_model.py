# Whole-model INT4 RTN baseline (group-wise absmax), the standard reference point.
# Quantizes every attention/MLP linear to signed int4, group size g along input dim.
# Writes a loadable checkpoint. Fast (no codebooks). Usage:
#   python int4_rtn_model.py ./qwen05b ./qwen05b_int4   [group_size]
import sys, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
src, out = sys.argv[1], sys.argv[2]
g = int(sys.argv[3]) if len(sys.argv) > 3 else 128
BITS = 4
TARGETS = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
           "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
def rtn(W, bits=BITS, g=g):
    O, I = W.shape; L = 2**bits
    pad = (-I) % g
    if pad: W = np.concatenate([W, np.zeros((O, pad))], 1)
    Z = W.reshape(O, -1, g); s = np.abs(Z).max(-1, keepdims=True)/((L-1)/2) + 1e-12
    Q = (np.clip(np.round(Z/s - .5), -L/2, L/2-1) + .5) * s
    Q = Q.reshape(O, -1)[:, :I]
    return Q
n = 0
for li, block in enumerate(model.model.layers):
    for t in TARGETS:
        try: mod = block.get_submodule(t)
        except AttributeError: continue
        W = mod.weight.detach().double().numpy()
        with torch.no_grad(): mod.weight.copy_(torch.from_numpy(rtn(W)).to(mod.weight.dtype))
        n += 1
    print(f"layer {li} int4", flush=True)
bpw = BITS + 16.0/g   # int4 + fp16 scale per group
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
print(f"\nINT4 RTN g={g} -> {bpw:.3f} bpw, {n} tensors. Saved {out}")
