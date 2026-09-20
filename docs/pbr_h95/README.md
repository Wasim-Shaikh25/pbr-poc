# PBR-H95 — Adaptive Mantissa Quantization (Phase A)

Lossy path: deterministically quantize BF16 mantissas to `k` retained bits, then store the quantized checkpoint exactly. Goal (later phases): minimize BPW subject to **≥95% of BF16 model-quality score** on held-out eval — not weight similarity.

Guide: `PBR_H95_Adaptive_Mantissa_Quantization_Guide.pdf`

## Phase A (this PR)

- `pbr_h95/quantize.py` — round-to-nearest mantissa keep-bits; Inf/NaN exact; signed zeros exact
- `pbr_h95/policy.py` — first-experiment map (emb/lm/norm/bias/first/last → 7; large attn/MLP → 5)
- `scripts/run_pbr_h95_phase_a.py` — full-model policy probe + uniform ladder k=7..3
- Artifacts: `artifacts/pbr_h95/phase_a_qwen.{json,md}`

### Qwen2.5-0.5B-Instruct (measured)

| Mode | Est. total BPW | Notes |
| ---: | ---: | --- |
| Policy (7/5) | **9.29** | avg mant bits 5.67; RAW pack + 2.62 exp ref |
| Uniform k=7 | 10.62 | lossless mantissa reference |
| Uniform k=5 | 8.62 | |
| Uniform k=3 | 6.62 | |

Quantize idempotent on all tensors: **True**.

### Honesty

- Phase A does **not** measure ≥95% quality — that needs held-out eval (B/C).
- `est_total_bpw` is RAW mantissa packing + reference exp rate, not a physical H95 container.
- Not a ≤4 BPW product claim. Lossless PBR-E reference remains ~10.6 BPW.

## Run

```bash
python scripts/run_pbr_h95_phase_a.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct
```
