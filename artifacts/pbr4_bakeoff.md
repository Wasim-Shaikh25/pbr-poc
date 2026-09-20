# PBR-4 structured nibble bakeoff

PBR-4 structured nibble + node formulas. W[i] = F(node(i), c4[i]) XOR R[i]. Complete bytes = |S| + 4N/8 + |R| + metadata. Bit-exact uint16 BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW unless measured total complete BPW is ≤4. c4 is stored 4-bit side info; coordinates are already known.

- repo: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **42**  words: **83836928**
- best setting: `16x16`  **20.580** BPW complete
- |S|=6724733  c4=41918464 B (formal 4N/8)  |R|=163750642  meta=3279886
- % weights with R=0: **8.59%**
- vs raw 16: **no**  vs PBR-E 10.616: **no**  vs zoo 10.585: **no**
- Stretch ≤4: **no**
- H(c4) (16x16 mix, nats→bits): **3.1423** bits/weight (entropy-coding c4 cannot beat the formal 4-bit charge by much if this is ~4)
- wall 707.1s

Formal complete bytes = |S| + 4N/8 + |R| + metadata (node headers + container JSON). F families compete per block: constant prototype, K≤16 palette, affine (a·r+b·c+d·c4+e) mod 2^16, nibble insert at shift 0/4/8/12, shared sign/exp + 4-bit mantissa nibble, 16 templates on planar XOR. Positions are not a data store.

| setting | |S| | c4 B | |R| | meta | formal B | BPW | % R=0 | vs 10.616 | vs 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16x16 | 6724733 | 41918464 | 163750642 | 3279886 | 215673725 | 20.580 | 8.59% | +9.964 | no |
| per_tensor_min | 6724733 | 41918464 | 163750642 | 3279886 | 215673725 | 20.580 | 8.59% | +9.964 | no |
| 8x8 | 312370 | 41918464 | 167673898 | 13104526 | 223009258 | 21.280 | 0.52% | +10.664 | no |
| 1x64 | 335753 | 41918464 | 167673898 | 13104526 | 223032641 | 21.283 | 0.55% | +10.667 | no |

`choose_tile_hw(256)` is 16×16 on every tensor in this set. Adaptive 16→8 never beat the parent (same generators and hit rate as 16×16). Affine and SE-nibble did not win any node on the 16×16 mix.

Winning-setting family mix (node counts):
- `palette16`: 203664
- `inherit_root`: 123702
- `templates16`: 92
- `const_proto`: 25
- `nibble_insert`: 5

Exact slice (2048 words): **PASS**. Encode+decode 0.03s.

**Stretch ≤4 BPW total: no.** Do not report PBR-4 as a 4 BPW codec.

**Does not beat PBR-E 10.616 BPW.** Node formulas miss on dense LLM tiles, so |R| stays near raw 16-bit and the forced 4-bit c4 stream is extra overhead.

`choose_tile_hw(256)` matched 16×16 on this Qwen set; adaptive 16→8 never beat the parent.

This is not a 1–2 GB / 8 GB result. The mantissa zoo (CTW/GBDT/AR/IDF) is unchanged.
