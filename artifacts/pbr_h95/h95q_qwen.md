# PBR-H95Q — Mixed-precision first run (A / B1 / B2)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (local BF16, CPU)
Calibration: **calib-v2** tokens=1925, max_length=256
Held-out: **heldout-v1** tokens=1094, max_length=256
Wall time: **145.67s**

## Honesty

- Proxy PPL on in-repo calib-v2 / heldout-v1 only — not a production LM benchmark.
- heldout ppl_retention ≥ 0.95 = proxy GO for that map; else NO-GO. Not MMLU/HellaSwag.
- packed_K_total_bpw = 1 sign + 2.62 exp ref + avg packed K — not a physical container.
- entropy_lb_total_bpw replaces packed K with empirical H(retained symbols); codecs need table cost.
- Reject rANS unless gain_vs_packed > 0 after table cost (see entropy diagnostics).
- Original BF16 need not round-trip; quantized reference MUST (Q idempotent).
- Not a ≤8 BPW product claim unless packed rate and proxy quality both support it.

## BF16 baseline

- calib mean_nll=3.160408 ppl=23.5802
- heldout mean_nll=3.664399 ppl=39.0327

## Candidates

| Candidate | packed BPW | avg K | entropy-lb BPW | calib ret | heldout ret | proxy | ≤8 packed? |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| h95q_A_7_5 | 9.2921 | 5.6721 | 9.2652 | 1.0003 | 0.9848 | proxy_GO | False |
| h95q_B1_mlp_k4 | 8.7099 | 5.0899 | 8.6845 | 0.9927 | 0.9945 | proxy_GO | False |
| h95q_B2_embed_k5 | 8.8227 | 5.2027 | 8.7963 | 1.0003 | 0.9905 | proxy_GO | False |
| h95q_C_sketch_k4_row_recover | 8.7161 | 5.0956 | 8.603 | 0.9957 | 0.9909 | proxy_GO | False |

## Policy rules

- **h95q_A_7_5**: Candidate A (H95 control): emb/lm/norm/bias/first/last → K7; mid attn/mlp (non-bias) → K5. Packed retained bits. Reproduce ~9.29 total est BPW.
- **h95q_B1_mlp_k4**: Candidate B1: same as A, but low-sensitivity mlp_mid (layers 1..22, non-bias) → K4; attn_mid stays K5; emb/norm/bias/first/last stay K7.
- **h95q_B2_embed_k5**: Candidate B2: embedding (tied lm_head) → K5; attn_mid → K6 (sensitive); mlp_mid → K5; norm/bias/first/last → K7.
- **h95q_C_sketch_k4_row_recover**: Optional C sketch: body like Phase-C mid@K4 with emb/norm/bias/first/last@K7, then restore top-magnitude fraction of mlp_mid rows to K7 (magnitude sparse recovery).
  - C sketch: recover_frac=0.05, recovered_rows=11682/233728, map_bpw=0.000473, packed+map=8.7161

## Family keep breakdown (word-weighted)

### h95q_A_7_5

| family | n_words | avg_keep |
| --- | ---: | ---: |
| attn_mid | 40395520 | 5.0013 |
| embed | 136134656 | 7.0000 |
| first_block | 14910592 | 7.0000 |
| last_block | 14910592 | 7.0000 |
| mlp_mid | 287637504 | 5.0000 |
| norm | 43904 | 7.0000 |

### h95q_B1_mlp_k4

| family | n_words | avg_keep |
| --- | ---: | ---: |
| attn_mid | 40395520 | 5.0013 |
| embed | 136134656 | 7.0000 |
| first_block | 14910592 | 7.0000 |
| last_block | 14910592 | 7.0000 |
| mlp_mid | 287637504 | 4.0000 |
| norm | 43904 | 7.0000 |

### h95q_B2_embed_k5

| family | n_words | avg_keep |
| --- | ---: | ---: |
| attn_mid | 40395520 | 6.0006 |
| embed | 136134656 | 5.0000 |
| first_block | 14910592 | 7.0000 |
| last_block | 14910592 | 7.0000 |
| mlp_mid | 287637504 | 5.0000 |
| norm | 43904 | 7.0000 |

### h95q_C_sketch_k4_row_recover

| family | n_words | avg_keep |
| --- | ---: | ---: |
| attn_mid | 40395520 | 4.0019 |
| embed | 136134656 | 7.0000 |
| first_block | 14910592 | 7.0000 |
| last_block | 14910592 | 7.0000 |
| mlp_mid | 287637504 | 4.0000 |
| norm | 43904 | 7.0000 |

## Entropy diagnostics (summary)

- **h95q_A_7_5**: avg H=5.6452 vs packed K=5.6721 (gain +0.0269); rANS-accept tensors=171/290
  - attn_mid: H=4.9750 K=5.0013 gain=+0.0262 rans_ok=88
  - embed: H=6.9720 K=7.0000 gain=+0.0280 rans_ok=1
  - first_block: H=6.9718 K=7.0000 gain=+0.0282 rans_ok=7
  - last_block: H=6.9714 K=7.0000 gain=+0.0286 rans_ok=7
  - mlp_mid: H=4.9737 K=5.0000 gain=+0.0263 rans_ok=66
  - norm: H=6.5838 K=7.0000 gain=+0.4162 rans_ok=2
- **h95q_B1_mlp_k4**: avg H=5.0645 vs packed K=5.0899 (gain +0.0254); rANS-accept tensors=171/290
  - attn_mid: H=4.9750 K=5.0013 gain=+0.0262 rans_ok=88
  - embed: H=6.9720 K=7.0000 gain=+0.0280 rans_ok=1
  - first_block: H=6.9718 K=7.0000 gain=+0.0282 rans_ok=7
  - last_block: H=6.9714 K=7.0000 gain=+0.0286 rans_ok=7
  - mlp_mid: H=3.9764 K=4.0000 gain=+0.0236 rans_ok=66
  - norm: H=6.5838 K=7.0000 gain=+0.4162 rans_ok=2
- **h95q_B2_embed_k5**: avg H=5.1763 vs packed K=5.2027 (gain +0.0264); rANS-accept tensors=171/290
  - attn_mid: H=5.9728 K=6.0006 gain=+0.0278 rans_ok=88
  - embed: H=4.9742 K=5.0000 gain=+0.0258 rans_ok=1
  - first_block: H=6.9718 K=7.0000 gain=+0.0282 rans_ok=7
  - last_block: H=6.9714 K=7.0000 gain=+0.0286 rans_ok=7
  - mlp_mid: H=4.9737 K=5.0000 gain=+0.0263 rans_ok=66
  - norm: H=6.5838 K=7.0000 gain=+0.4162 rans_ok=2
- **h95q_C_sketch_k4_row_recover**: avg H=4.9830 vs packed K=5.0082 (gain +0.0251); rANS-accept tensors=171/290
  - attn_mid: H=3.9783 K=4.0019 gain=+0.0235 rans_ok=88
  - embed: H=6.9720 K=7.0000 gain=+0.0280 rans_ok=1
  - first_block: H=6.9718 K=7.0000 gain=+0.0282 rans_ok=7
  - last_block: H=6.9714 K=7.0000 gain=+0.0286 rans_ok=7
  - mlp_mid: H=3.9764 K=4.0000 gain=+0.0236 rans_ok=66
  - norm: H=6.5838 K=7.0000 gain=+0.4162 rans_ok=2

## Idempotence

- **h95q_A_7_5**: checked=8 all_ok=True
- **h95q_B1_mlp_k4**: checked=8 all_ok=True
- **h95q_B2_embed_k5**: checked=8 all_ok=True
- **h95q_C_sketch_k4_row_recover**: checked=8 all_ok=True

## Summary decision

- Any candidate ≤8 packed BPW: **False**
- Proxy-GO candidates: ['h95q_A_7_5', 'h95q_B1_mlp_k4', 'h95q_B2_embed_k5', 'h95q_C_sketch_k4_row_recover']
- Best packed among proxy-GO: {'name': 'h95q_B1_mlp_k4', 'packed_K_total_bpw': 8.7099, 'heldout_ppl_retention': 0.99447156504464}
- Shortcuts: ['entropy subsampled to ≤250000 words/tensor', 'tiny C sketch only (row-magnitude recover); no D codecs', 'rANS is table-cost estimate only — no real encode bytes']

