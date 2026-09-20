# Phase A mantissa audit (Qwen/Qwen2.5-0.5B-Instruct)

Phase A mantissa audit. Held-out code length includes Huffman-table bytes. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW. Gate: mantissa complete BPW < 6.5 after predictor/table overhead.

- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **4**
- words: **149209088**
- original bytes: **298418176**
- H(sign)=1.0000  H(exp)=2.5837  H(M)=6.9722
- uncond mantissa complete BPW: **6.9728**
- best method: `H(M|exp)` at **6.9443** mantissa BPW
- implied total (best mant + H(sign) + H(exp)): **10.5280** BPW
- gate (mantissa complete BPW < 6.5): **MISS**

Methods ranked by held-out **complete** mantissa BPW (NLL + table bytes). MI is H(M) − H(M|ctx) on the same held-out split (ideal, no tables).

| method | ideal BPW | complete BPW | MI vs H(M) | tensors improved | beats 6.5? |
| --- | ---: | ---: | ---: | ---: | --- |
| H(M|exp) | 6.9334 | 6.9443 | 0.0392 | 4/4 | no |
| H(M|sign,exp) | 6.9335 | 6.9450 | 0.0390 | 4/4 | no |
| H(M|exp, prev M) | 6.9624 | 6.9642 | 0.0101 | 1/4 | no |
| H(M|exp, prev row) | 6.9646 | 6.9643 | 0.0079 | 1/4 | no |
| H(M) | 6.9725 | 6.9728 | 0.0000 | 0/4 | no |
| H(M|prev M) | 6.9726 | 6.9728 | -0.0001 | 0/4 | no |
| H(M|prev row) | 6.9723 | 6.9728 | 0.0002 | 0/4 | no |
| gray(M) | 6.9725 | 6.9728 | 0.0000 | 0/4 | no |
| bitplane_transpose(M) | 6.9731 | 6.9770 | -0.0006 | 0/4 | no |
| mod_delta_prev_row(M) | 6.9975 | 6.9978 | -0.0250 | 0/4 | no |
| mod_delta_prev(M) | 6.9998 | 7.0000 | -0.0273 | 0/4 | no |
| haar_row_pairs(M) | 7.2077 | 7.2086 | -0.2352 | 0/4 | no |
| H(M|prev layer M) | — | — | — | skipped | no |

Anything that does not beat unconditional H(M) complete BPW is **rejected** (including table/predictor overhead). Spatial / grammar / tile methods are not claimed here. This is not a ≤4 BPW or 1–2 GB / 8 GB result.

**Honest negative:** no compact context or cheap reversible transform reached mantissa complete BPW < 6.5 on this sample.
