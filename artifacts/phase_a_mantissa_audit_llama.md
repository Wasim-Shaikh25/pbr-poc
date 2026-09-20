# Phase A mantissa audit (unsloth/Llama-3.2-1B-Instruct)

Phase A mantissa audit. Held-out code length includes Huffman-table bytes. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW. Gate: mantissa complete BPW < 6.5 after predictor/table overhead.

- revision: `5a8abab4a5d6f164389b1079fb721cfab8d7126c`
- tensors: **11**
- words: **83886080**
- original bytes: **167772160**
- H(sign)=1.0000  H(exp)=2.5959  H(M)=6.9661
- uncond mantissa complete BPW: **6.9676**
- best method: `H(M|exp)` at **6.9323** mantissa BPW
- implied total (best mant + H(sign) + H(exp)): **10.5282** BPW
- gate (mantissa complete BPW < 6.5): **MISS**

Methods ranked by held-out **complete** mantissa BPW (NLL + table bytes). MI is H(M) − H(M|ctx) on the same held-out split (ideal, no tables).

| method | ideal BPW | complete BPW | MI vs H(M) | tensors improved | beats 6.5? |
| --- | ---: | ---: | ---: | ---: | --- |
| H(M|exp) | 6.9155 | 6.9323 | 0.0507 | 10/11 | no |
| H(M|sign,exp) | 6.9170 | 6.9358 | 0.0492 | 7/11 | no |
| H(M) | 6.9663 | 6.9676 | 0.0000 | 0/11 | no |
| H(M|prev M) | 6.9682 | 6.9676 | -0.0019 | 0/11 | no |
| H(M|prev row) | 6.9682 | 6.9676 | -0.0019 | 0/11 | no |
| gray(M) | 6.9663 | 6.9676 | 0.0000 | 0/11 | no |
| H(M|prev layer M) | 6.9699 | 6.9691 | -0.0025 | 0/4 | no |
| bitplane_transpose(M) | 6.9675 | 6.9863 | -0.0012 | 0/11 | no |
| mod_delta_prev_row(M) | 6.9997 | 7.0011 | -0.0335 | 0/11 | no |
| mod_delta_prev(M) | 6.9997 | 7.0011 | -0.0335 | 0/11 | no |
| H(M|exp, prev M) | 7.0785 | 7.0431 | -0.1122 | 0/11 | no |
| H(M|exp, prev row) | 7.0783 | 7.0433 | -0.1120 | 0/11 | no |
| H(M|exp, prev layer M) | 7.1289 | 7.0555 | -0.1615 | 0/4 | no |
| haar_row_pairs(M) | 7.2096 | 7.2136 | -0.2433 | 0/11 | no |

Anything that does not beat unconditional H(M) complete BPW is **rejected** (including table/predictor overhead). Spatial / grammar / tile methods are not claimed here. This is not a ≤4 BPW or 1–2 GB / 8 GB result.

**Honest negative:** no compact context or cheap reversible transform reached mantissa complete BPW < 6.5 on this sample.
