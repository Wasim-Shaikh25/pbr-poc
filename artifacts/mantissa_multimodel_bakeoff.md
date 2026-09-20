# Mantissa multimodel bakeoff

Mantissa multimodel bakeoff (CTW/PPM, histogram GBDT, tiny AR, IDF-lite, mixture). Held-out last 20% of rows. Complete BPW includes every model/table/index byte. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW unless the measured total complete BPW is ≤4.

- repo: `Qwen/Qwen2.5-0.5B-Instruct`  revision `7ae557604adf67be50417f59c2c2f167def9a775`
- tensors: **42**  words: **83836928**  holdout: **16767104**
- H(sign)=1.0000  H(exp)=2.6124  H(M)=6.9719
- PBR-E-style total on this split (sign raw + exp rANS + mant raw 7 + headers): **10.6470** BPW
- full-set PBR-E rANS reference (42 tensors, all words): **10.616** BPW
- best mantissa complete: `mixture_argmin` **6.9380**
- best implied total: **10.5850** BPW
- Phase A gate (mant complete < 6.5): **MISS**
- Useful (total < PBR-E split): **yes**
- Stretch (total ≤ 4): **no**
- fit 12.7s  wall 258.1s  reservoir 250000

Complete mantissa BPW amortizes a **single global model** over the eval set (codec view). `holdout_charged` dumps the whole model on the 20% holdout only (Phase A-harsh). Totals = sign raw 1.0 + exp rANS complete + mant complete + amortized tile headers.

| method | ideal mant | complete mant | holdout-charged | total BPW | vs PBR-E | model B | gate 6.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mixture_argmin | 6.9316 | 6.9380 | 6.9636 | 10.5850 | -0.0620 | 67124 | no |
| phase_a_H(M|exp) | 6.9327 | 6.9390 | 6.9640 | 10.5860 | -0.0610 | 65536 | no |
| gbdt_hist_16x2 | 6.9605 | 6.9607 | 6.9613 | 10.6077 | -0.0393 | 1588 | no |
| uncond_rANS | 6.9720 | 6.9720 | 6.9722 | 10.6190 | -0.0280 | 388 | no |
| bit_markov_d4 | 6.9726 | 6.9727 | 6.9728 | 10.6197 | -0.0273 | 454 | no |
| bit_markov_d8 / CTW-ctx | 6.9726 | 6.9733 | 6.9761 | 10.6203 | -0.0267 | 7174 | no |
| ctw_bitmix_d0+4+8 | 6.9726 | 6.9734 | 6.9763 | 10.6203 | -0.0266 | 7662 | no |
| ppm_hash_o2 | 6.9727 | 6.9852 | 7.0352 | 10.6322 | -0.0148 | 131076 | no |
| idf_lite_coupling | 6.9877 | 6.9878 | 6.9881 | 10.6348 | -0.0122 | 774 | no |
| tiny_ar_h32 (~64KB budget) | 6.9928 | 6.9936 | 6.9970 | 10.6406 | -0.0064 | 8776 | no |
| tiny_ar_h96 (~256KB budget) | 6.9934 | 6.9958 | 7.0057 | 10.6428 | -0.0042 | 25800 | no |
| raw_M | 7.0000 | 7.0000 | 7.0000 | 10.6470 | +0.0000 | 0 | no |

Mixture used models: exp_cond, gbdt.

Exact slice (4864 words): **PASS**. Encode+decode 4.41s (1104 words/s; Python bit-rANS, not a production speed claim).

Slice SHA restore of the original BF16 words passed for uncond rANS, bit-Markov, GBDT, tiny AR, and IDF-lite.

**Stretch ≤4 BPW total: no.** Do not report these numbers as a 4 BPW codec.

**Phase A gate MISS:** no zoo model coded held-out mantissas below 6.5 complete BPW after model bytes.

A sub-0.1 BPW total trim versus raw-mantissa PBR-E is **entropy-coding M** (uncond or H(M|exp) tables), DF11-class, not a new mantissa principle. CTW / AR / IDF did not beat that table.

This is not a 1–2 GB / 8 GB result.
