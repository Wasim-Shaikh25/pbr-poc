# PBR-H95 — Adaptive Mantissa Quantization

Lossy path: deterministically quantize BF16 mantissas to `k` retained bits, then store the quantized checkpoint exactly. Goal (later phases): minimize BPW subject to **≥95% of BF16 model-quality score** on held-out eval — not weight similarity.

Guide: `PBR_H95_Adaptive_Mantissa_Quantization_Guide.pdf`

## Phase A

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

### Honesty (Phase A)

- Phase A does **not** measure ≥95% quality — that needs held-out eval (B/C).
- `est_total_bpw` is RAW mantissa packing + reference exp rate, not a physical H95 container.
- Not a ≤4 BPW product claim. Lossless PBR-E reference remains ~10.6 BPW.

## Phase B (layer / tensor-family sensitivity)

- `pbr_h95/eval_nll.py` — mean token NLL / PPL under teacher forcing
- `pbr_h95/apply_policy.py` — apply keep-bits maps onto a live BF16 state_dict
- `pbr_h95/policy.py` — family helpers (`embed`, `norm`, `attn_mid`, `mlp_mid`, `first_block`, `last_block`; `lm_head_tied` ≡ embed when tied)
- `scripts/run_pbr_h95_phase_b.py` — calibration sweep + utility ranking
- Calibration corpus: `docs/pbr_h95/calibration_v1.json` (`calib-v1`)
- Artifacts: `artifacts/pbr_h95/phase_b_qwen.{json,md}`

### Metric (honest labeling)

- Score = mean token NLL on **calib-v1** (fixed in-repo English passages); `ppl = exp(mean_nll)`.
- `ppl_retention = bf16_ppl / quant_ppl` (reported for ranking only).
- **Calibration proxy only — not held-out eval.** Phase C owns the ≥95% held-out gate.
- Do **not** claim ≥95% GO from Phase B numbers.

### Non-claims

- No held-out ≥95% quality acceptance.
- No ≤4 BPW product claim.
- Utility rankings are calibration-local and may not transfer to held-out sets.

## Run

```bash
# Phase A (storage / numeric probe)
python scripts/run_pbr_h95_phase_a.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct

# Phase B (calibration sensitivity; CPU; needs torch+transformers)
python scripts/run_pbr_h95_phase_b.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --max-length 128
```
