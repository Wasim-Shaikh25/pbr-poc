# Matrix mantissa V1–V3 — pre-Qwen qualification

Matrix mantissa codec V1–V3 pre-Qwen qualification. Exact uint16 / 7-bit mantissa restore. Complete bytes include headers, directory, residual payloads, and padding. Do not claim Qwen compression. No ≤4 BPW claim.

**Do not claim Qwen compression yet.** These gates are synthetic / exactness only.
No ≤4 BPW product claim.

Overall: **PASS**

| gate | result |
| --- | --- |
| A | PASS |
| B | PASS |
| C | PASS |
| D | PASS |

## Checklist

### Gate A

- [x] `residual_pairs_128x128` — 16384 pairs (pred, actual) in 0..127; restore(pred, (actual-pred) mod 128)
- [x] `bf16_roundtrip_65536_v1` — complete_bytes=131110 strategy=ALL_RAW sha=6756cf54400b3908
- [x] `bf16_roundtrip_65536_v2` — complete_bytes=131110 strategy=ALL_RAW sha=6756cf54400b3908
- [x] `bf16_roundtrip_65536_v3` — complete_bytes=131110 strategy=ALL_RAW sha=6756cf54400b3908
- [x] `specials_no_fp` — ±0, ±Inf, NaN payloads, subnormals as uint16
- [x] `boundary_lengths` — n=8
- [x] `every_candidate_exact` — n_cands=184 failed=[]

### Gate B

- [x] `uniform_raw_wins` — strategy=ALL_RAW pred_counts={'RAW': 1} bytes=3602 all_raw=3602
- [x] `constant_predictor_wins` — strategy=TILED pred_counts={'PREVIOUS': 16} bytes=290
- [x] `row_ramp_exposes_UP` — UP=998 LEFT=2530 auto={'LEFT': 4, 'UP': 12}
- [x] `col_ramp_exposes_LEFT` — LEFT=998 UP=2530 auto={'UP': 4, 'LEFT': 12}
- [x] `v2_AVG_exact` — TILED
- [x] `v3_PAETH_exact` — TILED
- [x] `v3_EXP_LEFT_exact` — bytes=934 needs_exp=True

### Gate C

- [x] `all_raw_omits_tile_directory` — strategy=ALL_RAW n_tiles=0 flags=0 bytes=2034 hdr=b'MM7\x01'/3
- [x] `unused_exp_dep_omitted` — EXP_LEFT not charged on uniform ALL_RAW
- [x] `complete_bytes_match_blob`
- [x] `raw_wins_ties` — strategy=ALL_RAW counts={'RAW': 1} bytes=242
- [x] `tails_preserved` — n_tiles=4 expected 16x16 + 16x4 + 4x16 + 4x4
- [x] `unused_all_raw_plane_omitted_when_tiled_wins` — strategy=TILED n_tiles=4 bytes=86

### Gate D

- [x] `empty` — CodecError
- [x] `bad_magic` — CodecError
- [x] `truncated_header` — CodecError
- [x] `truncated_raw` — CodecError
- [x] `overrun_tile` — CodecError
- [x] `exp_left_missing_exp` — needs_exp=True

## Notes

The user PDF was not on this VM; predictors and traversals follow the task brief.
See `docs/Matrix_Mantissa_Codec_Guide.md`.
