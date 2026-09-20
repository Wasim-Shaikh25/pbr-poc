# PBR-H95Q stack follow-up — Track1 / Track2 / Track3 + stacks

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU, threads=1)
Calibration: **calib-v2** tokens=1925, max_length=256
Held-out: **heldout-v1** tokens=1094, max_length=256
Wall time: **499.63s**

## Honesty

- Proxy PPL on in-repo calib-v2 / heldout-v1 only — not a production LM benchmark.
- heldout ppl_retention ≥ 0.95 = proxy GO for that map; else NO-GO. Not MMLU/HellaSwag.
- packed_K_total_bpw = 1 sign + 2.62 exp ref + avg packed K (+ exception map_bpw for C).
- Candidate C map format: per-tensor bitmap (1 bit/unit) OR absolute indices; auto=min; + correction bits.
- Embed frequency tiers fit on calib-v2 only — risk of calib overfitting for rare tokens.
- Not a physical container encode; not a ≤8 BPW product claim unless packed≤8 AND heldout≥0.95.
- Original BF16 need not round-trip; quantized reference MUST (Q idempotent on samples).

## BF16 baseline

- calib mean_nll=3.160408 ppl=23.5802
- heldout mean_nll=3.664652 ppl=39.0426

## Results matrix

| Config | track | packed BPW | map BPW | avg K | calib ret | heldout ret | proxy | ≤8? | dual-gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| REF_B1_mlp_k4 | REF | 8.7099 | 0.00000 | 5.0899 | 0.9931 | 0.9944 | proxy_GO | False | False |
| T1_B1_embed_default | T1 | 7.8888 | 0.00000 | 4.2688 | 0.9819 | 0.9911 | proxy_GO | True | True |
| T1_B1_embed_aggressive | T1 | 7.6134 | 0.00000 | 3.9934 | 0.9905 | 0.9897 | proxy_GO | True | True |
| T1_B1_embed_conservative | T1 | 7.8937 | 0.00000 | 4.2737 | 0.9807 | 0.9923 | proxy_GO | True | True |
| T2_C_row_mag_0.02 | T2 | 8.6632 | 0.00012 | 5.0431 | 0.9941 | 0.9911 | proxy_GO | False | False |
| T2_C_row_mag_0.05 | T2 | 8.7159 | 0.00030 | 5.0956 | 0.9957 | 0.9912 | proxy_GO | False | False |
| T2_C_row_mag_0.10 | T2 | 8.8035 | 0.00047 | 5.1830 | 0.9958 | 0.9922 | proxy_GO | False | False |
| T2_C_channel_mag_0.05 | T2 | 8.7159 | 0.00018 | 5.0958 | 0.9914 | 0.9894 | proxy_GO | False | False |
| T3_mlp_all_k3 | T3 | 8.1277 | 0.00000 | 4.5077 | 0.9687 | 0.9454 | proxy_NO_GO | False | False |
| T3_mlp_band_1_7_k3 | T3 | 8.5246 | 0.00000 | 4.9046 | 0.9955 | 1.0000 | proxy_GO | False | False |
| T3_mlp_band_8_15_k3 | T3 | 8.4982 | 0.00000 | 4.8782 | 0.9908 | 0.9939 | proxy_GO | False | False |
| T3_mlp_band_16_22_k3 | T3 | 8.5246 | 0.00000 | 4.9046 | 0.9678 | 0.9464 | proxy_NO_GO | False | False |
| T3_mlp_robust50_k3 | T3 | 8.4188 | 0.00000 | 4.7988 | 0.9904 | 0.9937 | proxy_GO | False | False |
| S1_B1_aggr_embed_band815_k3 | STACK | 7.4017 | 0.00000 | 3.7817 | 0.9898 | 0.9900 | proxy_GO | True | True |
| S2_C_k3_base_row05_k7 | STACK | 8.1628 | 0.00030 | 4.5425 | 0.9609 | 0.9384 | proxy_NO_GO | False | False |

## Policy rules

- **REF_B1_mlp_k4**: Reference: plain B1 (mlp@K4 attn@K5 emb@K7) for apples-to-apples.
- **T1_B1_embed_default**: Track1: B1 body (mlp_mid@K4, attn_mid@K5, norm/bias/first/last@K7) + H95E default frequency embed row tiers (fit calib-v2 only).
  - Embed tiers (default): avg_row_K=4.0204 hist={'0': 0, '1': 0, '2': 0, '3': 271, '4': 150489, '5': 0, '6': 162, '7': 1014}
- **T1_B1_embed_aggressive**: Track1: B1 body + H95E aggressive embed frequency tiers.
  - Embed tiers (aggressive): avg_row_K=3.0207 hist={'0': 0, '1': 0, '2': 0, '3': 150925, '4': 0, '5': 397, '6': 100, '7': 514}
- **T1_B1_embed_conservative**: Track1: B1 body + H95E conservative embed frequency tiers.
  - Embed tiers (conservative): avg_row_K=4.0380 hist={'0': 0, '1': 0, '2': 0, '3': 271, '4': 149651, '5': 0, '6': 0, '7': 2014}
- **T2_C_row_mag_0.02**: Track2: aggressive body mlp@K4 attn@K4 emb@K7; restore top-2% |w| mlp rows to K7; charge bitmap/index map + correction bits.
  - C: mode=row_magnitude frac=0.02 recovered=4664/233728 map_bits=59444 schemes={'absolute_indices': 66}
- **T2_C_row_mag_0.05**: Track2: aggressive body; top-5% |w| mlp rows → K7 + honest map cost.
  - C: mode=row_magnitude frac=0.05 recovered=11682/233728 map_bits=148896 schemes={'absolute_indices': 66}
- **T2_C_row_mag_0.10**: Track2: aggressive body; top-10% |w| mlp rows → K7 + honest map cost.
  - C: mode=row_magnitude frac=0.1 recovered=23364/233728 map_bits=233728 schemes={'bitmap': 66}
- **T2_C_channel_mag_0.05**: Track2: aggressive body; top-5% |w| mlp channels → K7 + honest map cost.
  - C: mode=channel_magnitude frac=0.05 recovered=7326/146432 map_bits=89298 schemes={'absolute_indices': 66}
- **T3_mlp_all_k3**: Track3: B1 body with all mlp_mid → K3 (attn@K5).
  - K3 variant: {'variant': 'all_mlp_k3'}
- **T3_mlp_band_1_7_k3**: Track3: B1 body; mlp layers 1–7 → K3; other mlp_mid@K4.
  - K3 variant: {'variant': 'band_1_7', 'band': 'mlp_band_1_7'}
- **T3_mlp_band_8_15_k3**: Track3: B1 body; mlp layers 8–15 → K3; other mlp_mid@K4.
  - K3 variant: {'variant': 'band_8_15', 'band': 'mlp_band_8_15'}
- **T3_mlp_band_16_22_k3**: Track3: B1 body; mlp layers 16–22 → K3; other mlp_mid@K4.
  - K3 variant: {'variant': 'band_16_22', 'band': 'mlp_band_16_22'}
- **T3_mlp_robust50_k3**: Track3: B1 body; per mlp_mid matrix lowest-|w| 50% rows@K3 rest@K4.
  - K3 variant: {'variant': 'robust50_rows', 'k3_words': 143818752, 'k4_words': 143818752, 'n_tensors': 66, 'details_head': [{'name': 'model.layers.1.mlp.gate_proj.weight', 'n_rows': 4864, 'n_k3': 2432}, {'name': 'model.layers.1.mlp.up_proj.weight', 'n_rows': 4864, 'n_k3': 2432}, {'name': 'model.layers.1.mlp.down_proj.weight', 'n_rows': 896, 'n_k3': 448}, {'name': 'model.layers.2.mlp.gate_proj.weight', 'n_rows': 4864, 'n_k3': 2432}, {'name': 'model.layers.2.mlp.up_proj.weight', 'n_rows': 4864, 'n_k3': 2432}, {'name': 'model.layers.2.mlp.down_proj.weight', 'n_rows': 896, 'n_k3': 448}], 'note': 'Per mlp_mid matrix: lowest-|w| 50% rows @K3, rest @K4; no exception map (keeps are explicit per-row schedule, charged in avg K).'}
- **S1_B1_aggr_embed_band815_k3**: Stack: B1 + aggressive embed tiers + mlp band 8–15 @K3.
  - Embed tiers (aggressive): avg_row_K=3.0207 hist=None
  - K3 variant: {'variant': 'band_8_15', 'band': 'mlp_band_8_15'}
- **S2_C_k3_base_row05_k7**: Stack: C with mlp@K3 attn@K4 base + top-5% |w| mlp row recovery to K7 (+ map).
  - C: mode=row_magnitude frac=0.05 recovered=11682/233728 map_bits=148896 schemes={'absolute_indices': 66}
  - K3 variant: {'variant': 'k3_base_with_sparse_k7'}

## Candidate C map format

Per targeted 2D mlp weight: select top-|w|-mean rows **or** channels at `recover_frac`. Exception map = cheaper of (a) bitmap 1 bit/unit or (b) absolute indices `n_recover * ceil(log2(n_units))`. Correction stream = `(K_recover - K_base) * recovered_words` extra mantissa bits, folded into avg K. Packed total = 1 + 2.62 + adj_avg_K + map_bpw. Not a serialized container.

## Idempotence

- **REF_B1_mlp_k4**: checked=6 all_ok=True
- **T1_B1_embed_default**: checked=6 all_ok=True
- **T1_B1_embed_aggressive**: checked=6 all_ok=True
- **T1_B1_embed_conservative**: checked=6 all_ok=True
- **T2_C_row_mag_0.02**: checked=6 all_ok=True
- **T2_C_row_mag_0.05**: checked=6 all_ok=True
- **T2_C_row_mag_0.10**: checked=6 all_ok=True
- **T2_C_channel_mag_0.05**: checked=6 all_ok=True
- **T3_mlp_all_k3**: checked=6 all_ok=True
- **T3_mlp_band_1_7_k3**: checked=6 all_ok=True
- **T3_mlp_band_8_15_k3**: checked=6 all_ok=True
- **T3_mlp_band_16_22_k3**: checked=6 all_ok=True
- **T3_mlp_robust50_k3**: checked=6 all_ok=True
- **S1_B1_aggr_embed_band815_k3**: checked=6 all_ok=True
- **S2_C_k3_base_row05_k7**: checked=6 all_ok=True

## Summary

- Any packed ≤8: **True**
- Any dual-gate (≤8 AND heldout≥0.95): **True**
- Dual-gate names: ['T1_B1_embed_default', 'T1_B1_embed_aggressive', 'T1_B1_embed_conservative', 'S1_B1_aggr_embed_band815_k3']
- Proxy-GO names: ['REF_B1_mlp_k4', 'T1_B1_embed_default', 'T1_B1_embed_aggressive', 'T1_B1_embed_conservative', 'T2_C_row_mag_0.02', 'T2_C_row_mag_0.05', 'T2_C_row_mag_0.10', 'T2_C_channel_mag_0.05', 'T3_mlp_band_1_7_k3', 'T3_mlp_band_8_15_k3', 'T3_mlp_robust50_k3', 'S1_B1_aggr_embed_band815_k3']
- Best ≤8-or-closest: {'name': 'S1_B1_aggr_embed_band815_k3', 'packed_K_total_bpw': 7.4017, 'heldout_ppl_retention': 0.9900129451418743, 'kind': 'dual_gate', 'dual_gate_pass': True}
- Best dual-gate / proxy-GO by packed: {'name': 'S1_B1_aggr_embed_band815_k3', 'packed_K_total_bpw': 7.4017, 'heldout_ppl_retention': 0.9900129451418743, 'kind': 'dual_gate'}
- Shortcuts: ['entropy diagnostics skipped (packed-K + map honesty only)', 'C recovery salience = |w| magnitude (no per-row calib sensitivity sweep)', 'no physical container bytes; map_bpw estimated', 'embed tiers fit calib-v2 only; stack used schedule=aggressive']

