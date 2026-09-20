# PBR-H95 Phase B — Layer/tensor sensitivity (Qwen)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU)
Calibration: **calib-v1** (`docs/pbr_h95/calibration_v1.json`), max_length=128, n_texts=12

## Honesty

CALIBRATION PROXY ONLY — not held-out evaluation. Scores use the fixed in-repo corpus calib-v1 under teacher-forcing mean token NLL / perplexity. Phase C owns the ≥95% held-out quality gate. Do NOT claim ≥95% GO from Phase B. ppl_retention = bf16_ppl / quant_ppl is reported for ranking, not acceptance.

## Baseline (BF16)

- mean_nll: **2.941669**
- ppl (=exp(mean_nll)): **18.9475**
- tokens scored: **155**

## Candidates

| name | mean_nll | ppl | Δnll | ppl_retention | bytes_saved | utility (B/Δnll) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| policy_7_5 | 2.936809 | 18.8556 | -0.004861 | 1.0049 | 82001920 | 82001920000000000.00 |
| uniform_mid_k6 | 2.939576 | 18.9078 | -0.002094 | 1.0021 | 41000960 | 41000960000000000.00 |
| uniform_mid_k5 | 2.936809 | 18.8556 | -0.004861 | 1.0049 | 82001920 | 82001920000000000.00 |
| uniform_mid_k4 | 2.939486 | 18.9061 | -0.002184 | 1.0022 | 123002880 | 123002880000000000.00 |
| family_embed_k5 | 2.942353 | 18.9604 | +0.000684 | 0.9993 | 34033664 | 49769366267.33 |
| family_lm_head_tied_k5 | 2.942353 | 18.9604 | +0.000684 | 0.9993 | 34033664 | 49769366267.33 |
| family_norm_k5 | 2.942295 | 18.9593 | +0.000626 | 0.9994 | 10976 | 17543431.52 |
| family_attn_mid_k5 | 2.946613 | 19.0413 | +0.004943 | 0.9951 | 10098880 | 2042940404.67 |
| family_mlp_mid_k5 | 2.939904 | 18.9140 | -0.001766 | 1.0018 | 71909376 | 71909375999999992.00 |
| family_first_block_k5 | 2.941163 | 18.9379 | -0.000507 | 1.0005 | 3727648 | 3727648000000000.00 |
| family_last_block_k5 | 2.939305 | 18.9027 | -0.002364 | 1.0024 | 3727648 | 3727648000000000.00 |
| policy_7_5_mlp_mid_k4 | 2.953009 | 19.1635 | +0.011340 | 0.9887 | 117956608 | 10401885207.55 |

### Policy 7/5

- ppl_retention: **1.0049** (calibration proxy — not a ≥95% GO)
- delta_nll: **-0.004861**
- mantissa bytes saved vs BF16: **82001920**

### Top-3 sensitive families by Δnll (higher = more quality loss)

- **attn_mid**: Δnll=+0.004943, bytes_saved=10098880
- **embed**: Δnll=+0.000684, bytes_saved=34033664
- **norm**: Δnll=+0.000626, bytes_saved=10976

### Top-3 families by utility (bytes_saved / max(Δnll, 1e-9))

_Note: Δnll≤0 ⇒ huge utility (no measured calib loss on this proxy)._

- **mlp_mid**: util=71909375999999992.00, Δnll=-0.001766, bytes_saved=71909376
- **first_block**: util=3727648000000000.00, Δnll=-0.000507, bytes_saved=3727648
- **last_block**: util=3727648000000000.00, Δnll=-0.002364, bytes_saved=3727648

### Top-3 by utility among families with Δnll > 0

- **embed**: util=49769366267.33, Δnll=+0.000684, bytes_saved=34033664
- **attn_mid**: util=2042940404.67, Δnll=+0.004943, bytes_saved=10098880
- **norm**: util=17543431.52, Δnll=+0.000626, bytes_saved=10976

## Notes

- `tie_word_embeddings=true`: `lm_head_tied` ablation is identical to `embed` (same storage).
- CPU eval pinned to 1 thread for stable calibration deltas.
- `bytes_saved` = Σ (7−keep)×n_words/8 (mantissa bits only vs BF16).
- Protected tensors (emb/norm/bias/first/last) remain at keep=7 in policy and uniform-mid maps.
- Wall time: **76.85s**

