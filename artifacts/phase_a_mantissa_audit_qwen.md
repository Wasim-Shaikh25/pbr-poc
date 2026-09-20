# Phase A mantissa audit (Qwen/Qwen2.5-0.5B-Instruct)

Phase A mantissa audit. Held-out code length includes Huffman-table bytes. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW. Gate: mantissa complete BPW < 6.5 after predictor/table overhead.

- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **42**
- words: **83836928**
- original bytes: **167673856**
- H(sign)=1.0000  H(exp)=2.6124  H(M)=6.9719
- uncond mantissa complete BPW: **6.9772**
- best method: `H(M|exp)` at **6.9553** mantissa BPW
- implied total (best mant + H(sign) + H(exp)): **10.5677** BPW
- gate (mantissa complete BPW < 6.5): **MISS**

Methods ranked by held-out **complete** mantissa BPW (NLL + table bytes). MI is H(M) − H(M|ctx) on the same held-out split (ideal, no tables).

| method | ideal BPW | complete BPW | MI vs H(M) | tensors improved | beats 6.5? |
| --- | ---: | ---: | ---: | ---: | --- |
| H(M|exp) | 6.9332 | 6.9553 | 0.0388 | 18/42 | no |
| H(M|sign,exp) | 6.9383 | 6.9618 | 0.0337 | 17/42 | no |
| H(M) | 6.9721 | 6.9772 | 0.0000 | 0/42 | no |
| H(M|prev M) | 6.9839 | 6.9772 | -0.0119 | 0/42 | no |
| H(M|prev row) | 6.9843 | 6.9772 | -0.0123 | 0/42 | no |
| gray(M) | 6.9721 | 6.9772 | 0.0000 | 0/42 | no |
| H(M|prev layer M) | 6.9842 | 6.9773 | -0.0121 | 0/35 | no |
| H(M|exp, prev layer M) | 7.5724 | 6.9783 | -0.6003 | 0/35 | no |
| H(M|exp, prev row) | 7.5707 | 6.9784 | -0.5986 | 0/42 | no |
| H(M|exp, prev M) | 7.5685 | 6.9789 | -0.5965 | 0/42 | no |
| mod_delta_prev_row(M) | 6.9998 | 7.0050 | -0.0278 | 0/42 | no |
| mod_delta_prev(M) | 6.9998 | 7.0050 | -0.0278 | 0/42 | no |
| bitplane_transpose(M) | 6.9734 | 7.0455 | -0.0014 | 0/42 | no |
| haar_row_pairs(M) | 7.2097 | 7.2251 | -0.2376 | 0/42 | no |

Anything that does not beat unconditional H(M) complete BPW is **rejected** (including table/predictor overhead). Spatial / grammar / tile methods are not claimed here. This is not a ≤4 BPW or 1–2 GB / 8 GB result.

**Honest negative:** no compact context or cheap reversible transform reached mantissa complete BPW < 6.5 on this sample.
