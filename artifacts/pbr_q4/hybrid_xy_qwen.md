# PBR hybrid H95 + groupwise INT + selective X/Y

**Dual-gate: FAIL** — heldout proxy **0.981 ≥ 0.95** (PASS) but physical **7.765 ≰ 5.0** (FAIL).

## Why the INT splits missed

Groupwise INT on embed / mid-MLP still lacks per-weight exponents. Measured heldout:

| policy | packed-est BPW | heldout | note |
| --- | ---: | ---: | --- |
| h_stretch (mostly Q4 INT) | 4.683 | 0.717 | rate OK, quality collapse |
| h_mix (Q4 + late Q5 + H95 attn) | 5.001 | 0.731 | ~5.0 est, still collapse |
| h_practical (user split: late/attn H95) | 5.461 | 0.753 | H95 on late/attn was not enough |
| h_restore (Q5 INT + more H95) | 6.504 | 0.870 | closer, still <0.95 |
| **h_quality (all H95, encoded)** | 7.763 | **0.981** | quality backoff; rate miss |

So a mix that is cheap enough for ≤5.0 is too lossy for ≥0.95 on this proxy. The encoded container is the first ladder step that clears quality.

## Dual-gate verdict

| Gate | Target | Measured | Verdict |
| --- | --- | ---: | --- |
| Practical physical rate | ≤ **5.0** BPW | **7.765222** | **FAIL** |
| Stretch physical rate | ≤ **4.5** BPW | **7.765222** | **FAIL** |
| Held-out proxy (hard min) | ≥ **0.95** | **0.980986** | **PASS** |
| Held-out proxy (aim) | ≥ **0.97** | **0.980986** | **PASS** |
| Exact decode of this Q-ref | SHA match | `900f4535f7acf6958706607fea8fc96e39de682a645de4b1fba943ea44db6ca3` | **PASS** |

Policy: `h_quality` — new quantized reference SHA `900f4535f7acf6958706607fea8fc96e39de682a645de4b1fba943ea44db6ca3`.
Not S1. Not PR #17 Q6. Wall time: **444.7s**

## Honesty

- New hybrid quantized reference — NOT bit-exact with H95Q-S1 and not PR #17 restore_q6.
- actual_bpw = physical file_bytes * 8 / n_weights (header, H95 streams, INT scales, maps, flags, outliers).
- Packed codes are the product default. X/Y is stored only if complete physical cost is strictly smaller.
- H95 tensors keep per-weight exponents (adaptive exact). INT tensors share a scale+zp per group.
- Quality is ppl_retention = bf16_ppl / quant_ppl on in-repo calib-v2 / heldout-v1 (proxy, not MMLU).
- Hard quality floor: heldout ≥0.95. Development aim ≥0.97. Misses are FAIL, not rounded away.
- Practical rate target ≤5.0 BPW; stretch ≤4.5. Misses are reported as FAIL.
- S1 v2 7.420820 BPW and PR #17 6.191189 BPW are comparison baselines only.
- Not a production mobile runtime.

## Physical rate vs S1 7.42 and PR #17 6.19

| Container | file_bytes | actual_bpw | Q(W) |
| --- | ---: | ---: | --- |
| H95Q-S1 v2 (PR #14/#16) | 458,266,017 | **7.420820** | S1 SHA `eda64747…` |
| PR #17 `restore_q6` | 382,331,252 | **6.191189** | Q6 SHA `cc37d0ab…` |
| This HYBX (`h_quality`) | 479,534,240 | **7.765222** | this SHA |

## Byte mix (winner)

- H95 payload: 479,289,248 B (99.95% of file); words 494,032,768 (100.00%)
- INT payload: 0 B (0.00% of file); words 0 (0.00%)
- Raw BF16 payload: 0 B
- X/Y tiles: 1177 / 1934008 (0.0609%) — packed default
- Outliers: 0
- Saved vs packed codes: 49436 B
- XY mode hist: `{'XY_RANS': 1135, 'XY_RUN': 42}`

## Quality (proxy NLL)

| Split | BF16 ppl | Q ppl | retention |
| --- | ---: | ---: | ---: |
| calib-v2 | 23.5690 | 23.807998 | 0.98996 |
| heldout-v1 | 39.0321 | 39.788667 | 0.980986 |

PR #17 heldout proxy (reference): **0.967469** on restore_q6.

## Pareto / backoff

| policy | packed-est BPW | actual BPW | calib ret | heldout ret | encoded? |
| --- | ---: | ---: | ---: | ---: | --- |
| h_stretch | 4.682824 | — | 0.750762 | 0.717422 | False |
| h_mix | 5.001128 | — | 0.74898 | 0.730927 | False |
| h_practical | 5.460842 | — | 0.75446 | 0.752981 | False |
| h_restore | 6.504057 | — | 0.884058 | 0.870438 | False |
| h_quality | 7.76274 | 7.765222 | 0.98996 | 0.980986 | True |

## Exactness

- SHA-256 quantized ref: `900f4535f7acf6958706607fea8fc96e39de682a645de4b1fba943ea44db6ca3`
- decode(encode(Q)) == Q: **TRUE**
- Subprocess decode SHA: **TRUE**
- Distinct from S1 `eda64747…`: **YES**
- Distinct from PR #17 `cc37d0ab…`: **YES**
- file_bytes == physical size: **YES** (`479534240`)

## Family mix (winner)

| family | kind | n_words | packed-est BPW | outliers |
| --- | --- | ---: | ---: | ---: |
| attn_first | h95 | 1,835,008 | 9.62 | 0 |
| attn_last | h95 | 1,835,008 | 10.62 | 0 |
| attn_mid | h95 | 40,370,176 | 8.62 | 0 |
| embed | h95 | 136,134,656 | 6.62 | 0 |
| mlp_first | h95 | 13,074,432 | 9.62 | 0 |
| mlp_last | h95 | 13,074,432 | 10.62 | 0 |
| mlp_late | h95 | 91,521,024 | 8.62 | 0 |
| mlp_mid | h95 | 196,116,480 | 7.62 | 0 |
| norm_bias | h95 | 71,552 | 10.6202 | 0 |
