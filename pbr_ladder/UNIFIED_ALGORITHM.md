# PBR-Unified: one algorithm from the parts that survived testing

This consolidates every piece that actually worked into a single pipeline, and
drops every piece that didn't — with the evidence for each decision from this
project's real Qwen2.5-0.5B runs (`stability_findings.md`, consolidated doc).

## Design principle discovered this session

> **Flat *within* a tensor, adaptive *across* layers.**

The rotation (incoherence processing) makes a *uniform* quantizer optimal by
delocalizing outliers — it crushes per-channel importance spread from 286x to
4.8x and per-row energy spread from 15x to 1.3x. So every scheme that tries to be
clever *inside* a rotated tensor (adaptive per-group bits, outlier split,
drift-correction) is dominated by simply spending the bit on more flat VQ depth.
The only productive adaptivity is at a granularity *coarser* than the rotation —
the per-layer budget, guided by where depth actually hurts (the ends).

## The pipeline

Per linear layer, in sequential order (layer L sees already-quantized inputs):

1. **Rotate** `W -> Ro W Ri` with a **randomized Hadamard transform** (RHT):
   structured H_2^k + a seed-derived sign vector, padded to the next power of two.
   - *Free to store* (regenerated from the seed), like the random-QR rotation.
   - *Quality-neutral* vs random-QR (§7.1: 14.71 vs 14.73 dB).
   - **O(n log n)** to apply instead of O(n^2) -> the decode-speed win, and matches
     QuIP#/QuaRot's production choice.
2. **Fit one global codebook set** (pool D=8 vectors across the WHOLE tensor):
   S residual stages, K=256 each. Global pooling is what makes narrow k/v
   projections work. Flat allocation — proven optimal on top of rotation.
3. **Assign codes with beam search** (beam B, AQLM-style): keep the B best
   cumulative stage-combinations per vector instead of greedy nearest-per-stage.
   Same stored bits -> free quality. [benefit measured in beam_vq.py]
4. **GPTQ error feedback** across column blocks (Cholesky of the rotated Hessian).
5. **Per-layer budget** from a one-shot accumulation-curve pass (`eval_accum.py`):
   rich end layers (L0-2, L21-23), lean stable middle (L3-20). Delivered via the
   driver's existing `PBR_LAYER_STAGES`.

## Storage (the compressed file)

Per tensor: S codebooks (S x 256 x 8 x fp16 ~ 12 KB, shared across all rows),
one fp16 scale per row, and the stage indices (= the bpw). No dense rotation
matrix (RHT is a seed + sign bits). Codebook overhead <0.1 bpw on wide tensors.

## Decode (easy + fast — the runtime path)

To reconstruct a weight block:
1. For each of S stages: **table lookup** code index -> an 8-vector from the
   codebook (no arithmetic, pure gather). Sum the S vectors.
2. Multiply by the per-row scale.
3. Apply the **inverse RHT** (fast Walsh-Hadamard + sign flip), O(n log n).

This is a LUT + a butterfly transform — GPU/SIMD-friendly, no dense matmul for
dequant, the same shape as QuIP#/QTIP decoders. Ternary/int decoders can't beat
a codebook LUT on quality-per-bit; the RHT keeps the transform cheap.

## What was DROPPED, and why (so nobody re-adds it)

| Dropped | Evidence |
|---|---|
| Per-layer *rotation* variety | errors already incoherent, rho~=0 (coherence_probe) |
| Drift-correction W_eff | worse on every real tensor (validate_teacher) |
| Adaptive per-group bits | -0.6 dB, rotation flattens importance (adaptive_waterfill) |
| Outlier split (adaptive->rotate) | ~1 dB/bpw, dominated by flat VQ (outlier_split) |
| KLT / learned / shared rotation | storage tax; ruled out in consolidated doc §6 |

## Why this is competitive

- Whole-model **3.875 bpw -> +7.4% PPL**, vs standard **INT4 RTN +46%** at 4.125
  bpw: ~6x better degradation at fewer bits.
- Beam search + per-layer budget are the two levers expected to move the flat
  +7.4% toward the <5% gate at <4 bpw. [full-run validation pending]

## Reference implementation

`quantize_full_model.py` already carries: sequential calibration, global-pool VQ,
GPTQ feedback, `PBR_LAYER_STAGES` (per-layer budget), `PBR_INT4` (baseline mode),
`PBR_INIT_CKPT` (partial re-quant). Remaining to fold in for the full unified run:
RHT rotation (`push_hadamard.py` math + power-of-two padding) and beam assignment
(`beam_vq.py`) as driver flags `PBR_HADAMARD` / `PBR_BEAM`.
