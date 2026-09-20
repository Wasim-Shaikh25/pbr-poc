# Path to 50% (≤8.0 complete BPW)

Path to 50% (≤8.0 complete BPW, bit-exact) on Qwen linear/attention weights. Complete bytes include tables, streams, headers. Not a 1–2 GB / 8 GB claim. Do not report 50% unless measured complete BPW is ≤8.0.

## Result

Target **≤8.0 complete BPW**, bit-exact, Qwen2.5-0.5B-Instruct Stage 1B set (**42** tensors, **83836928** words).

**Best measured: 10.5627 BPW** (110693129 B), method `tensor_mixture_argmin`, exact **PASS**. Hits ≤8.0: **NO**.

Unigram floor H(uint16) = **10.5388 BPW**. Bits still missing to 8.0 after that floor: **2.5388**. No i.i.d. or single-context table can close a 2.5-bit gap when H(M|exp) stays **6.9325** / 7.

## Bounds (not bitstreams)

| quantity | bits |
| --- | ---: |
| H(uint16) unigram floor | 10.5388 |
| H(sign)+H(exp)+H(M) | 10.5842 |
| H(sign)+H(exp)+H(M\|exp) | 10.5449 |
| H(exp)+H(SM\|exp) | 10.5448 |
| H(sign\|exp) | 1.0000 |
| H(M\|exp) | 6.9325 |
| top-255 value coverage | 32.11% |
| escape-255 ideal BPW | 18.8618 |
| 16×16 exact tile dup rate | 0.000000 |

## Breakdown (sign / exp / mant / tables)

PBR-E stores sign+mantissa packed raw (8 bits) and rANS-codes exponents. `em_expcond_m` keeps exp rANS, packs sign as 1-bit, and rANS-codes the 7-bit mantissa with 256 exp-conditional tables (the zoo lever, realized as a bitstream).

| piece | ideal (NLL) | PBR-E rANS | best single (`em_expcond_m`) |
| --- | ---: | ---: | ---: |
| sign | 1.0000 | 1.000 packed in SM | 1.000 packed bits |
| exponent | 2.6124 | ~2.616 bitstream | ~2.616 bitstream |
| mantissa | 6.9325 (H(M\|exp)) | 7.000 raw | ~6.93 rANS + tables |
| tables + tile headers | (in totals) | counted | counted |
| **total** | **10.5449** | **10.6161** | **10.5737** |

Mixture argmin (per-tensor min of PBR-E / EM variants / optional nibble) is **10.5627 BPW**. Overhead vs the 10.5449 NLL bound is rANS + tables + headers, not a hidden 2.5-bit reservoir.

## Measured complete BPW (bit-exact)

| method | enc B | BPW | vs 10.616 | ≤8.0 | exact |
| --- | ---: | ---: | ---: | --- | --- |
| `tensor_mixture_argmin` | 110693129 | 10.5627 | -0.0534 | no | PASS |
| `em_expcond_m` | 110808763 | 10.5737 | -0.0424 | no | PASS |
| `em_uncond_m` | 110977882 | 10.5899 | -0.0262 | no | PASS |
| `shared_em_expcond_m` | 110978672 | 10.5900 | -0.0261 | no | PASS |
| `shared_em_sm_expcond` | 110997100 | 10.5917 | -0.0244 | no | PASS |
| `em_sm_uncond` | 111001393 | 10.5921 | -0.0240 | no | PASS |
| `em_sm_expcond` | 111029601 | 10.5948 | -0.0213 | no | PASS |
| `pbre_exp_rans` | 111253029 | 10.6161 | +0.0000 | no | PASS |
| `zlib_raw_uint16_baseline` | 133488038 | 12.7379 | +2.1218 | no | n/a |
| `optional_nibble_16` | 173242202 | 16.5314 | +5.9153 | no | PASS |
| `per_tensor_escape255` | 196536252 | 18.7541 | +8.1380 | no | PASS |
| `global_escape255` | 197665992 | 18.8619 | +8.2458 | no | PASS |

### Optional nibble / mixture notes

- Optional nibble tile usage: `{'raw': 327488}`. Nibble-admitted tiles: 0 / 327488.
- Tensor-level argmin choices: `{'em_expcond_m': 17, 'pbre_exp_rans': 8, 'em_uncond_m': 17}`.
- Shared-table sidecar is counted once in `shared_*` rows.

## Why 8.0 failed

The 50% target is **not** a missing tile codec. On this dense Qwen sample:

1. **i.i.d. floor is 10.54 BPW.** rANS on uint16 itself cannot beat that. 8.0 is **2.5 bits/weight** below the unigram entropy of the words.
2. **Context does not appear.** H(M|exp) is 6.93/7 (zoo / Phase A). H(sign|exp) is 1.0000 (sign is already ~1 bit). Spatial XOR, CTW/AR/IDF, and PBR-4 mandatory c4 all made complete bytes worse.
3. **Optional nibble does not fire.** Mandatory 4-bit side info was the PBR-4 failure mode (~20.6 BPW). Making c4 optional only helps tiles that share a high-12 prototype; those tiles are essentially absent on these MLP/attention walls.
4. **Global dictionaries do not concentrate.** Top-255 coverage 32.11% implies escape-dict BPW 18.86, worse than PBR-E.
5. **Cross-tensor tile copies are ~0.** Shared tables save header bytes (tens of KB), not 2.5 bits/weight.

**Best honest complete BPW: 10.5627** (`tensor_mixture_argmin`), exact PASS. That is DF11-class (~30% vs raw 16), not 50%. Related-checkpoint XOR+rANS at 8.36 remains the only measured number near 8, and only if the base checkpoint is already stored (not standalone).

zlib on raw uint16 is a baseline, not PBR.

Elapsed 780.3 s. This is not a 1–2 GB / 8 GB result.
