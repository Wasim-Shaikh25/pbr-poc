# PBR-Q4 quantize ↔ selective 256-node X/Y co-design

## Dual-gate verdict

| Gate | Target | Measured | Verdict |
| --- | --- | ---: | --- |
| Practical physical rate | ≤ **5.0** BPW | **6.191189** | **FAIL** |
| Stretch physical rate | ≤ **4.5** BPW | **6.191189** | **FAIL** |
| Held-out proxy (hard min) | ≥ **0.95** | **0.967469** | **PASS** |
| Held-out proxy (aim) | ≥ **0.97** | **0.967469** | **FAIL** |
| Exact decode of this Q-ref | SHA match | `cc37d0ab…` | **PASS** |

Q4/Q5 mixed policies do hit ≤5.0 packed-estimate (stretch 4.38, practical 4.95) but held-out retention stays **0.70–0.91**. Uniform groupwise Q6 is the first map on this ladder with held-out ≥0.95, at **6.191** physical BPW. That is an honest miss of the rate gate, not a rounded 5.0.

Policy: `restore_q6` — new quantized reference SHA `cc37d0ab085a1fea3e95b152965e087fb70a93b11e568905d5c084b38dca27ad`.
Not S1-compatible. Wall time: **1602.5s**

## Honesty

- New groupwise quantized reference — NOT bit-exact with H95Q-S1 and not S1-compatible.
- actual_bpw = physical file_bytes * 8 / n_weights (header, scales, maps, flags, outliers, checksums).
- Packed codes are the product default. X/Y is stored only if complete physical cost is strictly smaller.
- Quality is ppl_retention = bf16_ppl / quant_ppl on in-repo calib-v2 / heldout-v1 (proxy, not MMLU).
- Hard quality floor: heldout ≥0.95. Development aim ≥0.97. If aggressive misses quality, precision is restored.
- Practical rate target ≤5.0 BPW; stretch ≤4.5. Misses are reported as FAIL, not rounded away.
- S1 v2 7.420820 BPW is a comparison baseline only.
- Not a production mobile runtime.

## Physical rate vs H95Q-S1 v2

| Container | file_bytes | actual_bpw | exact Q(W)? | notes |
| --- | ---: | ---: | --- | --- |
| H95Q-S1 v2 (PR #14) | 458,266,017 | **7.420820** | S1 SHA | mantissa-keep + exp |
| This PQ4X (`restore_q6`) | 382,331,252 | **6.191189** | this SHA | groupwise + selective X/Y |

## Mode mix

- Tiles: 1929536 — X/Y 1183 / packed 1928353
- Tensors with at least one X/Y blob: 1
- XY mode hist: `{'XY_RANS': 20, 'XY_RUN': 1163}`
- Predictors: `{'PAETH': 26, 'UP': 2, 'PREVIOUS': 1155}`
- Traversals: `{'ROW': 28, 'COL': 1155}`
- Outliers: 20677
- Saved vs packed codes: 143402 B

## Quality (proxy NLL)

| Split | BF16 ppl | Q ppl | retention |
| --- | ---: | ---: | ---: |
| calib-v2 | 23.5690 | 24.89808 | 0.946618 |
| heldout-v1 | 39.0321 | 40.344554 | 0.967469 |

## Pareto / backoff

| policy | packed-est BPW | actual BPW | calib ret | heldout ret | encoded? |
| --- | ---: | ---: | ---: | ---: | --- |
| stretch | 4.379692 | — | 0.722526 | 0.705285 | False |
| practical | 4.950154 | — | 0.795275 | 0.821217 | False |
| safe | 5.178901 | — | 0.816245 | 0.848149 | False |
| restore | 5.562538 | — | 0.899478 | 0.911217 | False |
| restore_q6 | 6.19093 | 6.191189 | 0.946618 | 0.967469 | True |

## Exactness

- SHA-256 quantized ref: `cc37d0ab085a1fea3e95b152965e087fb70a93b11e568905d5c084b38dca27ad`
- decode(encode(Q)) == Q: **TRUE**
- Subprocess decode SHA: **TRUE**
- file_bytes == physical size: **YES** (`382331252`)

## Family mix (winner)

| family | n_words | packed-est BPW | outliers |
| --- | ---: | ---: | ---: |
| attn_first | 1,835,008 | 6.189 | 57 |
| attn_last | 1,835,008 | 6.1897 | 84 |
| attn_mid | 40,370,176 | 6.1894 | 1,586 |
| embed | 136,134,656 | 6.1897 | 6,142 |
| mlp_first | 13,074,432 | 6.1898 | 614 |
| mlp_last | 13,074,432 | 6.1895 | 544 |
| mlp_late | 91,521,024 | 6.1895 | 3,793 |
| mlp_mid | 196,116,480 | 6.1894 | 7,857 |
| norm_bias | 71,552 | 16.0 | 0 |
