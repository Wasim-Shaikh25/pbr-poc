# PBR-H95 Phase C — Greedy precision allocation (Qwen)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU)
Calibration (search): **calib-v2** (`docs/pbr_h95/calibration_v2.json`), tokens=1925, max_length=256
Held-out (gate): **heldout-v1** (`docs/pbr_h95/heldout_v1.json`), tokens=1094, max_length=256
Decision units: **bands** (mlp_band_1_7, mlp_band_8_15, mlp_band_16_22, attn_band_1_7, attn_band_8_15, attn_band_16_22)

## Honesty

- Held-out set is small/in-repo — **not** a production LM benchmark (no MMLU/HellaSwag/etc.).
- ppl_retention on heldout-v1 is a **proxy** quality score for this PoC.
- Claim GO only if heldout ppl_retention ≥ 0.95 AND map is reported; if <0.95, say NO-GO / needs restore.
- Not a physical container BPW; est only.
- Not ≤4 BPW product claim.

## Gate decision

- heldout ppl_retention: **0.9932** (threshold 0.95)
- calib ppl_retention (frozen): **0.9935** (search proxy only)
- **Decision: GO**

## Frozen map (unit keeps)

Protected emb/norm/bias/first/last remain at keep=7.

| unit | keep |
| --- | ---: |
| attn_band_16_22 | 4 |
| attn_band_1_7 | 4 |
| attn_band_8_15 | 4 |
| mlp_band_16_22 | 4 |
| mlp_band_1_7 | 4 |
| mlp_band_8_15 | 4 |

- bytes_saved vs BF16 mantissa: **123002880**
- avg keep bits: **5.0082**
- est_total_bpw (=1+2.62+avg_keep): **8.63** (est only)
- keep_hist (tensor count): `{'0': 0, '1': 0, '2': 0, '3': 0, '4': 154, '5': 0, '6': 0, '7': 137}`

## Greedy search steps (calib-only)

| step | unit | keep | calib ret | Δnll | bytes_saved | utility |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | mlp_band_16_22 | 6 | 0.9999 | +0.000130 | 11440128 | 88335650395.15 |
| 2 | attn_band_16_22 | 6 | 0.9995 | +0.000497 | 13045760 | 26270143180.20 |
| 3 | attn_band_1_7 | 6 | 0.9972 | +0.002782 | 14651392 | 5265662879.69 |
| 4 | attn_band_1_7 | 5 | 0.9985 | +0.001540 | 16257024 | 10556527945.59 |
| 5 | mlp_band_8_15 | 6 | 0.9982 | +0.001784 | 29331456 | 16442849770.08 |
| 6 | mlp_band_8_15 | 5 | 0.9989 | +0.001141 | 42405888 | 37174238506.29 |
| 7 | mlp_band_1_7 | 6 | 0.9988 | +0.001217 | 53846016 | 44245439696.14 |
| 8 | mlp_band_1_7 | 5 | 0.9975 | +0.002514 | 65286144 | 25968995834.12 |
| 9 | mlp_band_8_15 | 4 | 0.9987 | +0.001301 | 78360576 | 60219338468.91 |
| 10 | attn_band_8_15 | 6 | 0.9982 | +0.001789 | 80195584 | 44833727829.74 |
| 11 | attn_band_8_15 | 5 | 0.9978 | +0.002163 | 82030592 | 37929861332.95 |
| 12 | attn_band_8_15 | 4 | 0.9992 | +0.000789 | 83865600 | 106347854690.16 |
| 13 | attn_band_16_22 | 5 | 0.9983 | +0.001737 | 85471232 | 49211763151.47 |
| 14 | attn_band_16_22 | 4 | 0.9996 | +0.000357 | 87076864 | 243880173081.47 |
| 15 | attn_band_1_7 | 4 | 0.9959 | +0.004148 | 88682496 | 21378120357.38 |
| 16 | mlp_band_16_22 | 5 | 0.9991 | +0.000945 | 100122624 | 105925256126.81 |
| 17 | mlp_band_16_22 | 4 | 0.9996 | +0.000413 | 111562752 | 270122587942.69 |
| 18 | mlp_band_1_7 | 4 | 0.9933 | +0.006750 | 123002880 | 18221488592.40 |

## Pareto / reference table

| name | calib ret | heldout ret | bytes_saved | avg_keep | est BPW |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen_greedy | 0.9935 | 0.9932 | 123002880 | 5.0082 | 8.63 |
| policy_7_5 | 1.0009 | 0.9850 | 82001920 | 5.6721 | 9.29 |
| uniform_mid_k6 | 0.9986 | 0.9978 | 41000960 | 6.3361 | 9.96 |
| uniform_mid_k5 | 1.0003 | 0.9850 | 82001920 | 5.6721 | 9.29 |
| uniform_mid_k4 | 0.9933 | 0.9932 | 123002880 | 5.0082 | 8.63 |

## Baselines (BF16)

- calib mean_nll=3.160408, ppl=23.5802, tokens=1925
- heldout mean_nll=3.664148, ppl=39.0229, tokens=1094

## Shortcuts / notes

- Greedy trials used **stepwise** next-lower keep only (7→6→5→4), not all jumps in one round, to keep band search under the CPU time budget.
- Candidate keeps for this run: 6, 5, 4 (k=3 not tried; calib retention remained healthy through k=4).
- Frozen band map coincides with uniform_mid_k4 (all mid mlp/attn bands at keep=4).
- Search floor (calib): ppl_retention ≥ 0.97 (softer than held-out 0.95).
- Candidate keeps tried when lowering: [6, 5, 4].
- Wall time: **1481.24s**

