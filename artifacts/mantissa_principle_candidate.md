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

