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
- `scripts/run_pbr_h95_phase_b.py` — calibration sweep + utility ranking (`--calib` selects corpus)
- Calibration corpora:
  - `docs/pbr_h95/calibration_v1.json` (`calib-v1`, historical; ~155 scored tokens @ max_length=128)
  - `docs/pbr_h95/calibration_v2.json` (`calib-v2`, enlarged; includes all v1 texts + longer passages; target ≥1500 scored tokens @ max_length=256)
- Artifacts: `artifacts/pbr_h95/phase_b_qwen.{json,md}` (v1) and `artifacts/pbr_h95/phase_b_qwen_calib_v2.{json,md}` (v2)

### Metric (honest labeling)

- Score = mean token NLL on the chosen calib corpus; `ppl = exp(mean_nll)`.
- `ppl_retention = bf16_ppl / quant_ppl` (reported for ranking only).
- **Calibration proxy only — not held-out eval.** Phase C owns the ≥95% held-out gate.
- Do **not** claim ≥95% GO from Phase B numbers.

### Non-claims

- No held-out ≥95% quality acceptance from Phase B.
- No ≤4 BPW product claim.
- Utility rankings are calibration-local and may not transfer to held-out sets.

## Phase C (greedy precision allocation + held-out gate)

- `scripts/run_pbr_h95_phase_c.py` — greedy global precision allocation over layer-band (or family) units
- Held-out corpus: `docs/pbr_h95/heldout_v1.json` (`heldout-v1`; **disjoint** from calib; never used for search)
- Search uses **calib-v2 only**; final **GO/NO-GO** uses **heldout-v1** `ppl_retention ≥ 0.95`
- Artifacts: `artifacts/pbr_h95/phase_c_qwen.{json,md}`

### Honesty (Phase C — must read)

- Held-out set is small/in-repo — **not** a production LM benchmark (no MMLU/HellaSwag/etc.).
- ppl_retention on heldout-v1 is a **proxy** quality score for this PoC.
- Claim GO only if heldout ppl_retention ≥ 0.95 AND map is reported; if <0.95, say NO-GO / needs restore.
- Not a physical container BPW; est only.
- Not ≤4 BPW product claim.

### Search sketch

1. Start all-protect keep=7 (BF16 mantissa).
2. Decision units: mid-MLP / mid-Attn layer bands (layers 1–7, 8–15, 16–22) when runtime allows; else family-level `mlp_mid` / `attn_mid`.
3. Greedy: lower one unit by one candidate keep (6→5→4) maximizing `bytes_saved / max(Δnll, 1e-6)` subject to calib `ppl_retention ≥ 0.97`.
4. Freeze map; evaluate held-out; also report policy_7_5 and uniform_mid_k{6,5,4} on held-out for Pareto reference.

## Run

```bash
# Phase A (storage / numeric probe)
PYTHONPATH=. python scripts/run_pbr_h95_phase_a.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct

# Phase B on calib-v1 (historical)
PYTHONPATH=. python scripts/run_pbr_h95_phase_b.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --max-length 128

# Phase B on calib-v2 (enlarged)
PYTHONPATH=. python scripts/run_pbr_h95_phase_b.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --max-length 256 \
  --out artifacts/pbr_h95/phase_b_qwen_calib_v2.json \
  --md artifacts/pbr_h95/phase_b_qwen_calib_v2.md

# Phase C (greedy + held-out gate)
PYTHONPATH=. python scripts/run_pbr_h95_phase_c.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct \
  --calib docs/pbr_h95/calibration_v2.json \
  --heldout docs/pbr_h95/heldout_v1.json \
  --max-length 256 \
  --units bands
```
