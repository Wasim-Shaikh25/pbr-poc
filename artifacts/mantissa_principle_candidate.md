# Mantissa principle candidate (negative)

Mantissa multimodel bakeoff (CTW/PPM, histogram GBDT, tiny AR, IDF-lite, mixture). Held-out last 20% of rows. Complete BPW includes every model/table/index byte. Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW unless the measured total complete BPW is ≤4.

**Verdict:** negative.

## Proposed principle (one paragraph)

A combined CTW/PPM bit-context mixer, histogram GBDT on causal (exp, sign, prev, prev-row) features, tiny one-hidden-layer AR MLPs (actual stored size 8776 B and 25800 B, under the 64KB/256KB budgets), and a small integer additive coupling stack (IDF-lite) were trained on the first 80% of rows of the Qwen Stage 1B 42-tensor set and scored on the held-out 20%. A per-tensor argmin mixture paid only for the union of selected models. None of these reduced held-out mantissa complete BPW below the Phase A gate of 6.5. CTW, PPM, AR, and IDF all sat on the unigram (H(M)=6.97/7). The only total-BPW trim versus raw-mantissa PBR-E (~0.06 BPW) is entropy-coding M, especially 256 exp-conditional tables — the same DF11-class move already used on exponents, not a new principle. Learned models did not beat H(M|exp).

## Why this is not just DF11

This is not a DF11 replacement. DF11/PBR-E already entropy-codes the low-entropy exponent. Coding the mantissa with a 128-way (or 256×128) table is still DF11. The missing ~7 mantissa bits are residual entropy of dense trained weights, not a coding-format problem. Extra neural/tree/CTW tables did not buy a new compressible axis.

## Complete BPW breakdown

- sign (raw): 1.0000 BPW
- exp (rANS complete): 2.6469 BPW
- mantissa (`mixture_argmin` complete): 6.9380 BPW
- headers (amortized): 0.0001 BPW
- **total: 10.5850 BPW**
- model bytes (mixture_argmin): 67124
- PBR-E split total: 10.6470 BPW


## Failure modes

- Dense LLM BF16 mantissas with H(M)≈6.97 have no spare spatial Markov structure (Phase A already saw prev/row/layer raise entropy).
- Global trees/MLPs trained on 250k reservoir tokens rediscover the unigram; feature dependence is <0.05 bits (same scale as H(M|exp)).
- Charging a 1MB net to the holdout would look even worse; the budgets here were much smaller and still did not help.
- Bits-back / sign-fold: signs are already ~1 bit of entropy; no gauge to steal.
- Do not scale these BPW numbers to an 8 GB checkpoint or claim ≤4 BPW.

## PBR-4 structured nibble + node formulas

PBR-4 structured nibble + node formulas. W[i] = F(node(i), c4[i]) XOR R[i]. Complete bytes = |S| + 4N/8 + |R| + metadata. Bit-exact uint16 BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW unless measured total complete BPW is ≤4. c4 is stored 4-bit side info; coordinates are already known.

**Verdict:** negative.

### Proposed principle (one paragraph)

PBR-4 stores a compact generator S at each node of a shallow block tree (root → 8×8 / 16×16 / 64 / 256 tiles, optional 16→8 split) and exactly 4 bits c4 per weight. Reconstruction is W[i] = F(node(i), c4[i]) XOR R[i]. On the Qwen Stage 1B 42-tensor set the best complete setting `16x16` measured **20.580 BPW** (|S|=6724733, c4=41918464 B, |R|=163750642, meta=3279886) with 8.59% of weights having R=0. Stretch ≤4: no. Beats PBR-E 10.62: no. c4 is stored side information; coordinates are already known to encoder and decoder.

### Why this is not just DF11

This is not DF11 exponent rANS and not a mantissa unigram table. The hypothesis was that a 4-bit local choice among prototypes/templates plus a cheap integer formula on (row, col) would hit often enough that sparse R would drop the complete size toward ~4 BPW or at least below PBR-E. The measurement is the test of that hypothesis.

### Complete BPW breakdown

- |S| node generators: 6724733 B (0.6417 BPW)
- c4 stream (formal 4N/8): 41918464 B (4.0000 BPW)
- |R| exact residual: 163750642 B (15.6256 BPW)
- metadata: 3279886 B (0.3130 BPW)
- **total: 20.5803 BPW**
- % R=0: 8.59%
- PBR-E rANS reference: 10.616 BPW
- zoo best total: 10.585 BPW
- H(c4): 3.1423 bits (ANS-on-c4 would not remove the formal 4-bit floor)


### Failure modes

- Dense BF16 tiles have nearly unique uint16 values, so a 16-entry palette covers only ~8.6% with R=0 on the winning setting.
- Affine formulas on (r,c) in the uint16 ring do not match trained weight bits.
- Shared sign/exp per 16×16 is rare; SE-nibble then pays 4+3 mantissa bits plus a dense XOR residual for exponent mismatches.
- Formal 4N/8 c4 is a floor of 4 BPW before S, R, and metadata. If F misses, total is 4 + ~16 residual bits.
- Do not scale these BPW numbers to an 8 GB checkpoint or claim ≤4 BPW.

