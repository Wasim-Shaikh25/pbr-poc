# PBR-Q4 E2 mixed groupwise Q + E3−E2 ablation

## Dual-gate verdict

| Gate | Target | Measured | Verdict |
| --- | --- | ---: | --- |
| Practical physical rate | ≤ **5.0** BPW | **6.283611** | **FAIL** |
| Stretch physical rate | ≤ **4.75** BPW | **6.283611** | **FAIL** |
| Held-out proxy (hard min) | ≥ **0.95** | **0.957372** | **PASS** |
| Held-out proxy (aim) | ≥ **0.97** | **0.957372** | **FAIL** |
| Exact decode of this Q-ref | SHA match | `141c08c657be07fb6b4c666482d4b617ea49ba0986581557c4abf5423b46cdac` | **PASS** |

Policy: `e2_q6_tight` — new quantized reference SHA `141c08c657be07fb6b4c666482d4b617ea49ba0986581557c4abf5423b46cdac`.
Not S1/#17/#18 compatible. Wall time: **397.5s**

## Critical ablation (matrix net contribution)

```
E2 physical BPW  6.283588
E3 physical BPW  6.283611
Δ = E2 − E3      -2.3e-05   (below_useful)
```

Milestones: useful ≥ 0.03, good ≥ 0.05, strong ≥ 0.1. **Hit: below_useful.**

E3 selected **0 / 1,933,864** XY tiles (net margin `max(16 bits, 0.02×raw_tile_bits)` after complete costs). The −0.000023 BPW is **1408 header bytes** (E3 vs E2 JSON), not a matrix win. Product default remains packed.

## Overhead split (E3 winner)

| part | bytes | BPW |
| --- | ---: | ---: |
| weight payload | 373,928,064 | 6.055114 |
| scales (f16+zp) | 11,578,839 | 0.187499 |
| precision map | 456,185 | 0.007387 |
| codec meta (maps/flags) | 0 | 0.0 |
| container | 250,880 | 0.004063 |

## Honesty

- New mixed groupwise quantized reference — NOT bit-exact with H95Q-S1 / PR #17 / PR #18.
- Decode is bit-exact to this quantized reference, not to original BF16.
- actual_bpw = physical file_bytes * 8 / n_weights (header, scales, maps, flags, specials).
- E2 is packed mixed-Q. E3 is the same Q-ref + Phase-1 selective tiles with net margin.
- Product default is packed. Matrix family is stored only if it saves ≥ max(16 bits, 0.02×raw_tile_bits) after ALL costs.
- Quality is ppl_retention = bf16_ppl / quant_ppl on in-repo calib-v2 / heldout-v1 (proxy, not MMLU).
- Hard quality floor: heldout ≥0.95. Development aim ≥0.97.
- Practical rate target ≤5.0 BPW; stretch ≤4.75. Misses are reported as FAIL, not rounded away.
- Structured channel-group protect; no sparse per-weight exceptions as a quality lever (NaN/Inf only).
- Do not invent more X/Y modes. Do not merge/reopen #15/#17/#18 as the product path.
- S1 v2 7.420820 / #17 Q6 6.191 / #18 hybrid 7.765 are comparison baselines only.
- Not a production mobile runtime.

## vs baselines

| Container | file_bytes | actual_bpw | Q(W) |
| --- | ---: | ---: | --- |
| H95Q-S1 v2 (PR #14/#16) | 458,266,017 | **7.420820** | S1 SHA |
| PR #17 `restore_q6` | — | **6.191189** | Q6 (~heldout 0.967469) |
| PR #18 `h_quality` | — | **7.765222** | all-H95 (~heldout 0.980986) |
| This E2 packed | 388,037,300 | **6.283588** | this SHA |
| This E3 selective | 388,038,708 | **6.283611** | same SHA |

## Mode mix (E3)

- Tiles: 1933864 — X/Y 0 / packed 1933864
- XY mode hist: `{}`
- Predictors: `{}`
- Traversals: `{}`
- Specials (NaN/Inf only): 0
- Saved vs packed codes: 0 B

## Quality (proxy NLL)

| Split | BF16 ppl | Q ppl | retention |
| --- | ---: | ---: | ---: |
| calib-v2 | 23.5690 | 25.152702 | 0.937035 |
| heldout-v1 | 39.0321 | 40.770038 | 0.957372 |

## Pareto / backoff

| policy | packed-est BPW | E2 BPW | E3 BPW | calib ret | heldout ret | encoded? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| e2_stretch_4_75 | 4.75 | — | — | 0.7528 | 0.7919 | no |
| e2_budget_5_00 | 5.000 | — | — | 0.7744 | 0.8040 | no |
| e2_backoff_5_50 | 5.50 | — | — | 0.8124 | 0.7933 | no |
| e2_backoff_6_00 | 6.00 | — | — | 0.9080 | 0.9222 | no |
| **e2_q6_tight** | 6.25 | **6.283588** | **6.283611** | 0.9370 | **0.9574** | **yes** |

Q4-class maps hit the ≤5.0 packed-est band and fail heldout (0.79–0.80), same root cause as #17: groupwise INT on mid-MLP/attn is not saved by channel-group protect at this budget. Q5 floor (no Q4) reaches 0.922 — close miss. First ≥0.95 is Q6-floor + light structured Q8 (`e2_q6_tight`).

## Exactness

- SHA-256 quantized ref: `141c08c657be07fb6b4c666482d4b617ea49ba0986581557c4abf5423b46cdac`
- E2 decode(encode(Q)) == Q: **TRUE**
- E3 decode(encode(Q)) == Q: **TRUE**
- Subprocess decode SHA: **TRUE**

## Family mix (winner packed-est)

| family | n_words | packed-est BPW | specials |
| --- | ---: | ---: | ---: |
| attn_first | 1,835,008 | 7.1964 | 0 |
| attn_last | 1,835,008 | 6.5871 | 0 |
| attn_mid | 40,370,176 | 6.5175 | 0 |
| embed | 136,134,656 | 6.2811 | 0 |
| mlp_first | 13,074,432 | 6.194 | 0 |
| mlp_last | 13,074,432 | 6.194 | 0 |
| mlp_late | 91,521,024 | 6.1943 | 0 |
| mlp_mid | 196,116,480 | 6.194 | 0 |
| norm_bias | 71,552 | 8.4526 | 0 |

Bit-width word histogram:

- 6: 480,428,032
- 8: 13,602,432
- bf16: 2,304
