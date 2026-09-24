#!/usr/bin/env python
"""
Phase 0 — incoherence-rotation SIGNAL spike (no kernel, no training).

Question: does a randomized Hadamard rotation applied to a weight matrix BEFORE
quantization actually reduce quantization error at 2-3 bits? This is the gate
that decides whether the whole sub-3-bit rotation direction is worth building.

Method (pure linear algebra, CPU):
  - load real Linear weights from Qwen2.5-0.5B-Instruct (cached locally)
  - baseline: per-group uniform quant of W  -> rel error ||W-Q(W)||/||W||
  - rotated : W' = RHT_out @ W @ RHT_in  (block randomized Hadamard, orthogonal)
              then per-group uniform quant of W'.  Because the rotation is
              orthogonal, ||W'-Q(W')|| == ||W - back-rotated Q(W')||, so the
              rotated-domain error IS the original-domain reconstruction error.
  - also report an "ideal" full random-orthogonal rotation on the input dim as
    an upper bound on the achievable benefit.
  - outlier stats (kurtosis, max/std) show WHY rotation helps: it flattens the
    heavy tails that eat a 2-bit budget.

If rotated error << baseline error at 2-3 bit -> SIGNAL. Build Phase 1/2.
"""
import numpy as np, scipy.linalg, time
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BITS = [2, 3, 4]
GS = 64          # quant group size (along input dim), llama.cpp-ish
HB = 128         # Hadamard block size (deployable; must divide the dim)

def had(n):  # normalized (orthogonal) Hadamard
    return scipy.linalg.hadamard(n).astype(np.float32) / np.sqrt(n)

def block_rht(M, axis, B, seed):
    """Apply randomized Hadamard (H @ diag(sign)) in blocks of size B along axis."""
    rng = np.random.default_rng(seed)
    X = M if axis == 1 else M.T          # operate along last dim
    r, n = X.shape
    if n % B != 0:                        # fall back to largest power-of-2 divisor
        b = 1
        while b * 2 <= n and n % (b * 2) == 0:
            b *= 2
        B = max(2, b)
    s = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    X = X * s
    H = had(B)
    Xb = X.reshape(r, n // B, B) @ H.T
    Y = Xb.reshape(r, n)
    return Y if axis == 1 else Y.T

def rand_orth(n, seed):
    g = np.random.default_rng(seed).standard_normal((n, n)).astype(np.float32)
    q, _ = np.linalg.qr(g)
    return q

def quant_dequant(W, bits, gs=GS):
    """Per-row, per-group uniform asymmetric quant (min/max), like a real quantizer."""
    o, i = W.shape
    if i % gs:  # pad-safe: use whole row if not divisible
        gs = i
    levels = (1 << bits) - 1
    Wr = W.reshape(o, i // gs, gs)
    mn = Wr.min(2, keepdims=True); mx = Wr.max(2, keepdims=True)
    scale = (mx - mn) / levels; scale[scale == 0] = 1.0
    q = np.clip(np.round((Wr - mn) / scale), 0, levels)
    return (q * scale + mn).reshape(o, i)

def relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12))

def stats(W):
    w = W.ravel(); m2 = np.mean(w**2)
    kurt = float(np.mean(w**4) / (m2**2 + 1e-12))       # Gaussian=3; higher=heavier tails
    return kurt, float(np.abs(w).max() / (np.sqrt(m2) + 1e-12))

print("loading", MODEL, "...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="float32")
sd = model.state_dict()
print(f"loaded in {time.time()-t0:.1f}s")

TARGETS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.12.self_attn.v_proj.weight",
    "model.layers.12.mlp.down_proj.weight",
]

print(f"\ngroup size={GS}, Hadamard block={HB}\n")
hdr = f"{'layer':<40} {'bits':>4} {'base_err':>9} {'rot_err':>9} {'ideal_err':>9} {'gain%':>7}"
print(hdr); print("-"*len(hdr))
agg = {b: [] for b in BITS}
for name in TARGETS:
    if name not in sd:
        print("  (missing)", name); continue
    W = sd[name].detach().numpy().astype(np.float32)
    kurt, mxr = stats(W)
    # rotated weight (two-sided block Hadamard)
    Wr = block_rht(block_rht(W, 1, HB, 1), 0, HB, 2)
    kurt_r, mxr_r = stats(Wr)
    # ideal: full random-orthogonal on input dim only (bounded cost)
    o, i = W.shape
    Ri = rand_orth(i, 3) if i <= 2048 else None
    Wi = (W @ Ri) if Ri is not None else None
    for b in BITS:
        e_base = relerr(W, quant_dequant(W, b))
        e_rot = relerr(Wr, quant_dequant(Wr, b))
        e_ideal = relerr(Wi, quant_dequant(Wi, b)) if Wi is not None else float('nan')
        gain = 100.0 * (e_base - e_rot) / (e_base + 1e-12)
        agg[b].append(gain)
        print(f"{name:<40} {b:>4} {e_base:>9.4f} {e_rot:>9.4f} {e_ideal:>9.4f} {gain:>6.1f}%")
    print(f"    outliers: kurtosis {kurt:>6.1f} -> {kurt_r:>6.1f} | max/std {mxr:>5.1f} -> {mxr_r:>5.1f}")

print("\n=== mean gain from block-Hadamard rotation (rel-error reduction) ===")
for b in BITS:
    g = np.mean(agg[b]) if agg[b] else float('nan')
    print(f"  {b}-bit: {g:+.1f}%  ({'SIGNAL' if g > 5 else 'weak' if g > 0 else 'NONE/NEG'})")
print("\nRead: positive gain at 2-3 bit = rotation reduces quant error -> Phase 1 justified.")
print("Kurtosis dropping toward ~3 (Gaussian) is the mechanism: outliers flattened.")
