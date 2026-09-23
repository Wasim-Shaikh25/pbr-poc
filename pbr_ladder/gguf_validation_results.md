# GGUF Recipe — Small Pipeline Validation (2026-09-23)

**Goal:** prove the recipe-on-GGUF pipeline works end-to-end on consumer hardware (no kernel,
no infra) and honestly measure whether our recipe beats a *well-tuned* stock GGUF.

**Setup:** Qwen2.5-0.5B-Instruct fp16 GGUF → `llama-imatrix` (32 chunks of our calib slice) →
`llama-quantize` Q3_K_M (three ways) → `llama-perplexity` (same 300 KB WikiText slice, ctx=512).
llama.cpp b11138 CPU build. Runner: `run_validation.sh`. Model params ≈ 494 M.

## Results
| config | PPL (slice, ctx512) | size | eff bpw | vs stock |
|---|---|---|---|---|
| stock Q3_K_M (no imatrix) | 16.549 | 432.0 MB | ~7.0 | — |
| **+ imatrix** (our importance) | **16.509** | 432.0 MB | ~7.0 | **−0.04, FREE (same size)** |
| + imatrix + q8_0 embed/output | 16.468 | 500.1 MB | ~8.1 | −0.08 but **+68 MB (+16%)** |

## Honest findings
1. **The pipeline works end-to-end on a laptop.** base → imatrix → quantize → PPL → packed
   GGUF (mmap tunnel automatic). This is the main win: the deployment path is real and cheap.
2. **imatrix is a free win** (−0.04 PPL at identical size). Keep it always.
3. **Forcing embeddings UP to q8_0 is NOT worth it here** (+68 MB for −0.04 more PPL). Our
   earlier "embedding lever" only helps when embeddings are being *over-spent at fp16* — a
   well-tuned stock GGUF (k-quants) already quantizes embeddings sensibly, so the lever mostly
   vanishes against a *good* baseline. This tempers the earlier "beat GGUF" claim: that win was
   largely vs downloaded defaults / our own fp16-embed mistake, not vs a tuned baseline.
4. **Edge over well-tuned stock GGUF is SMALL so far** (imatrix aside). No free lunch here.
5. **Scale thesis confirmed empirically:** on 0.5B, even "3-bit" Q3_K_M is ~7 eff bpw because
   embeddings (~27.5% of params) dominate. Sub-4-bit-at-quality is impossible on 0.5B and only
   becomes real at 3B–7B where the embedding tax shrinks. This is why P4 (scale) is mandatory.

## Caveats
- Tiny 300 KB eval slice, ctx=512 → PPL values (~16.5) are NOT comparable to our earlier
  full-WikiText seq-2048 numbers (~15.2). Relative comparison across the 3 configs is valid.
- Not matched-size: the recipe row is bigger. A fair "win" test = beat 16.509 at ≤432 MB via
  per-tensor `--tensor-type` allocation (net-neutral). Not yet done.

## IQ (codebook) + below-3-bit results (2026-09-23, same slice/ctx)
| config | PPL | size | eff bpw |
|---|---|---|---|
| **Q3_K_M + imatrix (scalar)** | **16.509** | 432 MB | 7.00 |
| IQ3_M stock | 17.314 | 419 MB | 6.79 |
| IQ3_M + imatrix | 16.933 | 419 MB | 6.79 |
| IQ3_M + imatrix + q8 embed | 16.878 | 487 MB | 7.89 |
| IQ2_M + imatrix (below-3-bit) | 19.217 | 405 MB | 6.56 |
| IQ2_M + imatrix + q8 embed | 19.167 | 473 MB | 7.66 |

### Findings (sobering but decisive)
1. **The IQ-codebook thesis FAILED on 0.5B.** Scalar Q3_K_M+imatrix (16.509) BEATS IQ3_M+imatrix
   (16.933) at ~equal size. IQ formats shine at 2-bit, not 3-bit; at 3-bit tuned k-quants win.
2. **imatrix is the one consistent free win** (−0.38 PPL on IQ3, −0.04 on Q3_K_M; no size cost).
   But imatrix is llama.cpp's, not ours — so it is NOT a differentiator.
3. **q8-embed lever: still not worth it** (+55–68 MB for ~0.05 PPL). Confirms prior run.
4. **Below-3-bit on 0.5B is a losing trade:** IQ2_M tanks PPL to 19.2 while saving only ~27 MB
   vs Q3_K_M (embeddings dominate, so cutting weight bits barely shrinks the file).
5. **Net: at 0.5B we have NO quant-quality edge over a well-tuned stock GGUF beyond stock imatrix.**
   The only untested lever is **fused rotation** (QuaRot-style) — but its benefit is largest at
   2-bit, which is pointless on 0.5B. So rotation, too, only matters at scale.

## DECISION: stop optimizing on 0.5B — it is the wrong testbed
Every result reconfirms the scale thesis: 0.5B is embedding-dominated, so sub-4-bit is neither
achievable nor meaningful here. The next real experiment MUST be at **3B** (free Colab/Kaggle),
where embeddings shrink to ~4% and low-bit + rotation + imatrix finally have room to matter.
Only there can the ≥98% story be honestly evaluated.

## Next
- Build the **fused-rotation offline step** (QuaRot/Hadamard baked into weights → standard GGUF).
- Run the whole recipe on **3B** on free Colab/Kaggle; that is the decisive test, not 0.5B.
