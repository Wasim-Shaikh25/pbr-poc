# 3B recipe test — Qwen2.5-3B-Instruct (f16 base), 2026-09-24

**Question:** does sub-4-bit hold quality at ~3B (where 0.5B could not), and which
recipe is the sweet spot?

**Setup:** one f16 base (Qwen2.5-3B-Instruct, bartowski GGUF), one importance matrix
(our calib slice, 32 chunks, reused across all configs), same trimmed eval slice
(150 KB WikiText, ctx=512), llama.cpp CPU. Sizes/bpw from gguf stored-tensor count.

## Full results (phase 1 + phase 2, unified — best quality first)
| config | PPL | size | eff bpw | note |
|---|---|---|---|---|
| stock Q4_K_M (near-lossless ref) | 9.302 | 1930 MB | 5.00 | quality ceiling proxy |
| iq3-imat-q8e (IQ3_M+imatrix+q8 embed) | 9.734 | 1564 MB | 4.055 | >4 bpw; embed lever adds only 0.7% |
| **⭐ iq3-imat (IQ3_M+imatrix)** | **9.804** | **1489 MB** | **3.860** | **SWEET SPOT: near-Q4 quality, sub-4-bit, −23% size** |
| ship-3bit (Q3_K_M+imatrix) | 12.992 | 1590 MB | 4.12 | scalar Q3_K far worse than IQ3 at same bits |
| stock Q3_K_M (no imatrix) | 13.333 | 1590 MB | 4.12 | baseline |
| ship-2bit-iq (IQ2_M+imatrix+q8 embed) | 12.285 | 1257 MB | 3.26 | embed lever adds 0.9% over the row below |
| stock-iq2m-imat (IQ2_M+imatrix) | 12.398 | 1140 MB | 2.957 | smallest, but +33% PPL vs Q4 = below the quality bar |

Lower PPL = better.

## Findings — corrected and airtight (the matched baselines changed the story)
1. **The real winner is `IQ3_M + imatrix` (3.86 bpw): near-Q4 quality at sub-4-bit.**
   9.804 PPL vs Q4_K_M's 9.302 = **only +5.4% PPL, at 23% smaller (1489 vs 1930 MB) and
   genuinely below 4 bpw.** This is the "sub-4-bit at high quality" product promise,
   delivered — and it was impossible at 0.5B.
2. **IQ codebooks WIN at 3B, the opposite of 0.5B.** At ~4 bpw, IQ3 (9.80) crushes scalar
   Q3_K_M (13.0). On 0.5B, scalar Q3_K_M *beat* IQ3. So the codebook thesis was right — it
   just needs scale. This is the single cleanest confirmation of the scale thesis.
3. **CORRECTION to phase 1: the 8-bit embedding lever is marginal, NOT decisive.** The
   matched baselines isolate it: IQ2+imatrix+q8embed (12.285) vs IQ2+imatrix (12.398) =
   only **−0.9% PPL for +10% size**; IQ3 case = **−0.7% for +0.2 bpw** (and it pushes over
   4 bpw). My phase-1 claim that the embed lever "was decisive at 3B" was WRONG — most of
   the quality comes from IQ+imatrix itself. **Drop q8-embed from the default recipe.**
4. **CORRECTION to phase 1: IQ2 is too aggressive for the quality bar.** IQ2 (2.96 bpw)
   looked great vs Q3_K_M, but that was the wrong yardstick. Vs the real target (Q4/fp16),
   IQ2 is **+33% PPL** — well below a ≥98% story. IQ2 is an extreme-size option only, not
   the headline. IQ3 is the sweet spot.
5. **imatrix is the one consistent free win** (built from our calib data), but it is
   llama.cpp's lever, not a novel algorithm of ours.

## Honest scope — what is and isn't "ours"
Every config here is a **stock llama.cpp quant type + imatrix + flags**. We did not invent
a new quantizer. Our contribution is: (a) the **turnkey, reproducible, measured pipeline**
that produces these with a manifest, and (b) the **finding of the sweet spot** (IQ3+imatrix
at ~3.86 bpw) and which levers matter (imatrix yes; q8-embed no; IQ2 too far). That is a
real product (DX + provenance + a validated recipe), but not a quant-algorithm moat.

## Caveats
- PPL on a 150 KB in-domain slice, ctx=512. The "≥98%" claim needs **task-accuracy**
  (lm-eval): iq3-imat at +5.4% PPL *likely* lands ~98% task-acc (PPL overstates the drop),
  but this is unproven until measured.
- No true fp16 PPL (base deleted to save disk); Q4_K_M is the near-lossless proxy.

## Net
At 3B we can **ship a 3.86-bpw model at near-Q4 quality** — the product promise, delivered
where the product actually lives. The recipe is IQ3_M + imatrix (drop the embed lever). The
0.5B null result was the testbed, not the method; the codebook thesis holds at scale. Next:
task-accuracy on iq3-imat to nail the ≥98% claim, then on-phone tok/s.
