# Expert analysis: why VQ ≈ scalar, and the escape routes

Two independent expert passes (a mathematics/rate-distortion analysis and a
statistical-physics analysis) CONVERGED on the same diagnosis. Recorded here
because it reframes the whole method.

## The convergent diagnosis (why our VQ doesn't beat scalar at ~4 bpw)

A vector quantizer's advantage over optimal scalar decomposes (Gersho–Gray) into:
1. **Shape gain** — non-uniform marginal. Optimal SCALAR (Lloyd-Max / entropy
   coding) already captures this. VQ gets no edge.
2. **Memory gain** — correlation between the D weights in a vector. **Only VQ can
   get this — but the incoherence ROTATION whitens the weights to i.i.d. Gaussian,
   driving memory gain to ~0.** We Gaussianize away our own advantage.
3. **Space-filling gain** — Voronoi-cell geometry. The only term left, and it is
   **bounded: 1.53 dB (0.25 bit/weight) as D→∞; only ~0.65 dB at D=8 (E8 lattice).**
   Greedy k-means RVQ captures a fraction of even that.

So there is at most ~0.65 dB for a memoryless D=8 VQ over strong scalar — the
provable reason VQ (+6–7%) ≈ scalar (+5.8%) at 4 bpw. Same mechanism that killed
adaptive allocation (rotation flattens importance) kills VQ (rotation flattens
correlation). **This is a ceiling, not a tuning miss.**

## Escape routes (ranked, each with a cheap go/no-go)

1. **Trellis-coded quantization (QTIP-style)** — raise effective dimension via a
   convolutional code + Viterbi encode; bitshift + tiny-LUT decode (cheaper than
   our 256-table). Captures ~full 1.5 dB space-filling gain -> a real SUB-4-BIT
   win. Training-free. *Exists (QTIP NeurIPS'24); adopt, don't reinvent.*
   Test: Viterbi-encode one rotated tensor at 2^L states, matched bpw, output SQNR.
2. **Partial rotation** *(novel)* — rotate just enough to tame outliers for GPTQ
   while leaving residual correlation for VQ to harvest as memory gain. Nobody
   frames rotation strength as a VQ-gain vs outlier-flattening dial.
   Test: sweep rotation strength α; plot VQ-over-scalar SQNR gain vs α.
3. **Better assignment: Hessian-Mahalanobis metric + finite-T simulated-annealing
   ceiling** — beam already gave +0.65 dB free, so the T=0 assignment is suboptimal;
   SA measures the ground-state ceiling. Go/no-go: condition number of rotated 8x8
   Hessian block (if ~1, isotropic -> Mahalanobis == Euclidean, skip).
4. **Nested-lattice / successive-refinement codebook** — the multi-quality single
   file capability (decode one file at 2/3/4 bpw), near-free on a Gaussian source
   (Equitz–Cover). This is a CAPABILITY scalar lacks, not a matched-bpw win.
5. **Leech Λ24 lattice** instead of k-means/E8 — 1.03 dB ceiling (vs 0.65 at D=8),
   codebook-free closed-form decode. Incremental.
6. **Degeneracy-manifold cross-layer steering** *(novel — the salvageable drift
   correction)* — among equal-distortion beam candidates, pick the one whose output
   error ANTI-aligns with accumulated cross-layer error. Distortion-neutral (unlike
   the failed W_eff). Go/no-go: corr(per-layer error increment, accumulated error);
   if ~0 (the depth walk is already incoherent, exponent 0.06), no gain.

## Bottom line
- As a *memoryless* VQ, this method is capped at ~parity with scalar. Provable.
- The genuine advantages are: **trellis** (sub-4-bit quality), **nested lattice**
  (multi-quality capability), and possibly **partial rotation** (novel, untested).
- Every next step leads with a cheap measurement that can kill the idea first.

Sources: QTIP (arxiv 2406.11235), QuIP# (2402.04396), NestQuant (2502.09720),
CDQuant (2406.17542), Matryoshka Quant (2502.06786), Gish–Pierce / Gersho–Gray.
