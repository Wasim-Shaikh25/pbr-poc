# PBR-H95 Phase A — Qwen mantissa quantization probe

Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Policy estimate (first-experiment map)

- Protected tensors (emb/lm/norm/bias/first/last): **7** bits
- Large attn/MLP: **5** bits
- Words: **494032768**
- Avg retained mantissa bits: **5.6721**
- Est. total BPW (1 + 2.62 + avg_mant): **9.2921**
- Quantize idempotent on all tensors: **True**

## Uniform ladder (diagnostic)

| keep | frac words changed | mse (f32 diag) | est total BPW |
| ---: | ---: | ---: | ---: |
| 7 | 0.0000 | 0.000e+00 | 10.62 |
| 6 | 0.5000 | 2.216e-08 | 9.62 |
| 5 | 0.7500 | 8.828e-08 | 8.62 |
| 4 | 0.8750 | 2.818e-07 | 7.62 |
| 3 | 0.9375 | 9.752e-07 | 6.62 |

## Honesty

Phase A only: deterministic quantization + estimated storage BPW. est_total_bpw uses RAW mantissa packing and a 2.62 exp reference — not a physical container encode. Does NOT measure model-quality retention; >=95% requires held-out eval (Phase B/C). Not a <=4 BPW claim.

Lossless reference remains ~10.62 BPW. H95 aims for lower BPW only with measured ≥95% quality later.

