# PBR-H95 Phase B — Layer/tensor sensitivity (Qwen)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU)
Calibration: **calib-v2** (`docs/pbr_h95/calibration_v2.json`), max_length=256, n_texts=58

## Honesty

CALIBRATION PROXY ONLY — not held-out evaluation. Scores use the fixed in-repo corpus calib-v2 under teacher-forcing mean token NLL / perplexity. Larger corpora (e.g. calib-v2) reduce noise relative to calib-v1 but remain calibration proxies. Phase C owns the ≥95% held-out quality gate. Do NOT claim ≥95% GO from Phase B. ppl_retention = bf16_ppl / quant_ppl is reported for ranking, not acceptance.

## Baseline (BF16)

- mean_nll: **3.160408**
- ppl (=exp(mean_nll)): **23.5802**
- tokens scored: **1925**

## Candidates

| name | mean_nll | ppl | Δnll | ppl_retention | bytes_saved | utility (B/Δnll) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| policy_7_5 | 3.160378 | 23.5795 | -0.000030 | 1.0000 | 82001920 | 82001920000000000.00 |
| uniform_mid_k6 | 3.161692 | 23.6105 | +0.001283 | 0.9987 | 41000960 | 31947216792.91 |
| uniform_mid_k5 | 3.160071 | 23.5723 | -0.000338 | 1.0003 | 82001920 | 82001920000000000.00 |
| uniform_mid_k4 | 3.167159 | 23.7399 | +0.006750 | 0.9933 | 123002880 | 18221488592.40 |
| family_embed_k5 | 3.161360 | 23.6027 | +0.000952 | 0.9990 | 34033664 | 35751105247.89 |
| family_lm_head_tied_k5 | 3.161432 | 23.6044 | +0.001023 | 0.9990 | 34033664 | 33255484121.56 |
| family_norm_k5 | 3.163403 | 23.6509 | +0.002995 | 0.9970 | 10976 | 3665014.61 |
| family_attn_mid_k5 | 3.166172 | 23.7165 | +0.005764 | 0.9943 | 10098880 | 1752146114.20 |
| family_mlp_mid_k5 | 3.158621 | 23.5381 | -0.001787 | 1.0018 | 71909376 | 71909375999999992.00 |
| family_first_block_k5 | 3.161748 | 23.6118 | +0.001340 | 0.9987 | 3727648 | 2782020018.86 |
| family_last_block_k5 | 3.156687 | 23.4926 | -0.003722 | 1.0037 | 3727648 | 3727648000000000.00 |
| policy_7_5_mlp_mid_k4 | 3.167703 | 23.7529 | +0.007294 | 0.9927 | 117956608 | 16170695767.71 |

### Policy 7/5

- ppl_retention: **1.0000** (calibration proxy — not a ≥95% GO)
- delta_nll: **-0.000030**
- mantissa bytes saved vs BF16: **82001920**

### Top-3 sensitive families by Δnll (higher = more quality loss)

- **attn_mid**: Δnll=+0.005764, bytes_saved=10098880
- **norm**: Δnll=+0.002995, bytes_saved=10976
- **first_block**: Δnll=+0.001340, bytes_saved=3727648

### Top-3 families by utility (bytes_saved / max(Δnll, 1e-9))

_Note: Δnll≤0 ⇒ huge utility (no measured calib loss on this proxy)._

- **mlp_mid**: util=71909375999999992.00, Δnll=-0.001787, bytes_saved=71909376
- **last_block**: util=3727648000000000.00, Δnll=-0.003722, bytes_saved=3727648
- **embed**: util=35751105247.89, Δnll=+0.000952, bytes_saved=34033664

### Top-3 by utility among families with Δnll > 0

- **embed**: util=35751105247.89, Δnll=+0.000952, bytes_saved=34033664
- **first_block**: util=2782020018.86, Δnll=+0.001340, bytes_saved=3727648
- **attn_mid**: util=1752146114.20, Δnll=+0.005764, bytes_saved=10098880


## Brief compare to calib-v1 (Phase B prior)

| | calib-v1 | calib-v2 |
| --- | ---: | ---: |
| tokens scored | 155 | 1925 |
| max_length | 128 | 256 |
| policy_7_5 ppl_retention | 1.0049 | 1.0000 |
| policy_7_5 Δnll | -0.004861 | -0.000030 |

Top sensitive by Δnll (v1): attn_mid (+0.0049), embed (+0.0007), norm (+0.0006)
Top sensitive by Δnll (v2): attn_mid (+0.0058), norm (+0.0030), first_block (+0.0013)

Larger corpus reduces noise; rankings remain a **calibration proxy** (not held-out).
## Notes

- `tie_word_embeddings=true`: `lm_head_tied` ablation is identical to `embed` (same storage).
- CPU eval pinned to 1 thread for stable calibration deltas.
- `bytes_saved` = Σ (7−keep)×n_words/8 (mantissa bits only vs BF16).
- Protected tensors (emb/norm/bias/first/last) remain at keep=7 in policy and uniform-mid maps.
- Wall time: **216.62s**

