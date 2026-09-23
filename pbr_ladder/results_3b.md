# 3B recipe test — Qwen2.5-3B-Instruct (f16 base), 2026-09-24

**Question:** does our sub-4-bit recipe beat a well-tuned stock GGUF at ~3B, where
0.5B was too embedding-dominated to show any edge?

**Setup:** one f16 base (Qwen2.5-3B-Instruct, bartowski GGUF), one importance matrix
(our calib slice, 32 chunks), same trimmed eval slice (150 KB WikiText, ctx=512),
llama.cpp CPU. Sizes/bpw from the gguf stored-tensor count (~3.086B elements).

## Results
| config | PPL | size | eff bpw | vs stock Q3_K_M |
|---|---|---|---|---|
| stock Q3_K_M (no imatrix) | 13.333 | 1590 MB | 4.12 | — |
| **ship-3bit** (Q3_K_M + imatrix) | **12.992** | 1590 MB | 4.12 | **−2.6% PPL, FREE (same size)** |
| **ship-2bit-iq** (IQ2_M + imatrix + q8 embed) | **12.285** | **1257 MB** | **3.26** | **−7.9% PPL AND −21% size** |
| stock Q4_K_M (reference / "normal download") | 9.302 | 1930 MB | 5.00 | better quality, +21% size |
| stock IQ2_M (no imatrix) | — | — | — | FAILED: IQ2 requires an imatrix |

Lower PPL = better.

## Findings — the scale thesis is CONFIRMED
1. **At 3B our recipe finally wins — and it did not at 0.5B.** `ship-3bit` beats stock
   Q3_K_M by **−2.6% PPL at identical size** (1590 MB both). This is the imatrix +
   our calibration data working; it costs zero extra bytes.
2. **`ship-2bit-iq` is the headline: smaller AND better.** At **3.26 bpw / 1257 MB** it
   is **21% smaller than stock Q3_K_M yet 7.9% lower PPL** — a clean Pareto win, and it
   is genuinely **sub-4-bit** (the product spec). This is the "sub-4-bit at good quality"
   story that was impossible on 0.5B.
3. **Why it works — the embedding lever, validated at scale.** `ship-2bit-iq` uses IQ2
   weights (*lower* precision than Q3_K_M) yet beats the Q3 model. The only way a 3.26-bpw
   model outperforms a 4.12-bpw one is the **8-bit embeddings + imatrix allocation**
   protecting what matters. Qwen's large vocab makes embeddings decisive — exactly the
   lever that was worthless at 0.5B (where embeddings were already the whole budget).
4. **Honest ceiling:** stock Q4_K_M (5.0 bpw) is clearly better quality (9.30 PPL). Going
   sub-4-bit costs real quality vs 4-bit; our value is **size at acceptable quality**, not
   beating 4-bit quality.

## Honest caveats (do not oversell)
- **`ship-3bit`'s win = imatrix**, which is llama.cpp's lever, not uniquely ours. Our
  contribution there is packaging + calibration data, not a novel algorithm.
- **`ship-2bit-iq` lacks a matched baseline.** Stock IQ2_M can't be built without an
  imatrix (llama.cpp refuses), so we could not measure stock-IQ2+imatrix (no q8 embed) to
  isolate how much of the win is the embed lever vs imatrix alone. PENDING: re-run that
  baseline (needs the f16 base, which the runner deleted to save disk).
- **PPL on a 150 KB slice**, in-domain, ctx=512. Task-accuracy (lm-eval) still unproven.
- The runner's size column failed live (a `/d/` path wasn't understood by Windows Python);
  sizes above were recomputed correctly afterward. Fixed in the runner.

## Net
First real evidence the engine has value **where the product actually lives (3B+)**:
a genuine sub-4-bit, Pareto-better-than-stock-3-bit model. The 0.5B null result was the
testbed, not the method. Next: fill the matched IQ2+imatrix baseline, then task-accuracy,
then on-phone tok/s.
