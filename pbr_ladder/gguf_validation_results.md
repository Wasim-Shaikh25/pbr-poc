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

## Next
- Matched-size allocation test (`--tensor-type`) at ≤432 MB — can smarter per-layer bits beat
  imatrix-only at the same size? This is the real remaining lever on 0.5B.
- Repeat on a 3B model (free Colab/Kaggle) where sub-4-bit-at-98% actually lives.
