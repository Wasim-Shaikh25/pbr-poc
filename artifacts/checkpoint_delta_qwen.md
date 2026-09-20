# Exact base→finetune delta

Exact base→finetune delta (related-checkpoint residual). Bit-exact uint16 only. Complete bytes include JSON metadata and every coder table. Standalone BPW is the finetune alone (PBR-E rANS). Delta-only BPW assumes the base checkpoint is already present. If the base must be shipped too, total = base PBR-E + delta. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW.

## Pair

- Base: `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Finetune / target: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Architecture: same Qwen2 config (hidden 896, 24 layers, 14 heads, vocab 151936, BF16).
- Paired 16-bit tensors: **290** (by name+shape)
- Unmatched base: none
- Unmatched target: none
- Words: **494032768**  original bytes: **988065536**

## Weight-change stats (bit-exact XOR)

| metric | value |
| --- | ---: |
| % weights unchanged | 2.0676 |
| % weights changed | 97.9324 |
| mean XOR popcount (of 16 bits) | 3.7060 |
| XOR Hamming fraction | 0.2316 |
| H(uint16 XOR) | 8.0888 |
| H(sign XOR) | 0.1885 |
| H(exp XOR) | 1.4133 |
| H(mant XOR) | 6.7471 |
| sign equal fraction | 0.9710 |
| exp equal fraction | 0.7529 |
| mantissa equal fraction | 0.0218 |

## Encoded size (complete container, all metadata)

Standalone BPW is the finetune encoded alone with PBR-E rANS. Delta-only BPW is the residual payload assuming the base is already present. **Bundle** = base PBR-E + delta (honest cost if the base must be shipped too).

| method | complete B | delta-only BPW | bundle B | bundle BPW | exact |
| --- | ---: | ---: | ---: | ---: | --- |
| 1. target PBR-E rANS (standalone) | 655839276 | n/a (standalone) | 655839276 | 10.6202 | PASS |
| 2a. uint16 XOR + zlib | 644202227 | 10.4317 | 1300022374 | 21.0516 | PASS |
| 2b. uint16 XOR + rANS lo/hi | 516005521 | 8.3558 | 1171825668 | 18.9757 | PASS |
| 3. sign/exp/mantissa field deltas | 546290439 | 8.8462 | 1202110586 | 19.4661 | PASS |
| 4. sparse changed-position patches | 1029449871 | 16.6701 | 1685270018 | 27.2900 | PASS |
| 5. default XOR residual + dict/raw | 988076895 | 16.0002 | 1643897042 | 26.6200 | PASS |

| base PBR-E rANS (for the bundle) | 655820147 | n/a | 655820147 | 10.6199 | PASS |

- Standalone finetune BPW: **10.6202**
- Best delta method: **xor_rans** (516005521 B, 8.3558 delta-only BPW)
- Reconstruct(target) from base+delta: **PASS** (uint16 / SHA-256)

Shipping the base *and* a delta is **not** cheaper than shipping the finetune standalone unless the delta is smaller than `standalone − base_PBR-E` (it is not a way to hide a second full checkpoint). Not ≤4 BPW.

Elapsed encode+verify: 2462.5 s.
