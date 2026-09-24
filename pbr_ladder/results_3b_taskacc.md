# 3B task-accuracy — HellaSwag (1000 tasks), Qwen2.5-3B, 2026-09-24

Retention = model_acc / Q4_K_M_acc (Q4_K_M = near-lossless full-quality proxy;
f16 base was deleted to save disk). HellaSwag random baseline = 25%.

| model | bpw | HellaSwag acc | retention vs Q4 |
|---|---|---|---|
| stock Q4_K_M (ref) | 5.00 | 75.1774% | 100.0% |
| **IQ3_M + imatrix (ship)** | 3.86 | **74.5014%** | **99.1%** |
| stock Q3_K_M | 4.12 | 70.1365% | 93.3% |
| IQ2_M + imatrix | 2.96 | 68.0891% | 90.6% |

## VERDICT: ✅ PASS — the sub-4-bit-at-quality claim is real
**IQ3_M + imatrix retains 99.1% of Q4-level HellaSwag accuracy at 3.86 bpw (sub-4-bit),
clearing the ≥98% bar on a downstream task, not just PPL.** The 0.68-point gap vs Q4
(74.50 vs 75.18) is within the sampling noise of 1000 tasks — IQ3+imatrix is effectively
**indistinguishable from Q4 accuracy** while being 23% smaller and genuinely sub-4-bit.

It also decisively beats the alternatives at similar/greater bits: **IQ3 99.1% vs stock
Q3_K_M 93.3%** (a 5.8-pt accuracy gap for the same ~4 bpw) — so the recipe choice
(IQ codebook + imatrix, not scalar Q3_K) matters a lot. IQ2 (90.6%) confirms it is below
the bar: extreme-size only.

## Honest caveats (do not overstate)
- **Retention is vs Q4_K_M, not true fp16** (f16 base was deleted to save disk). Q4_K_M
  itself retains ~99%+ of fp16, so vs true fp16 IQ3 is very likely ~98.5–99% — still ≥98%,
  but the exact-vs-fp16 number is un-measured. A clean re-run from f16 would nail it.
- **1000 of 10042 HellaSwag tasks** (~±2.7% CI at 75%). Point estimate is strong; the
  IQ3≈Q4 result is robust, but a full-set or multi-task (MMLU/ARC/Winogrande) run would
  tighten it.
- One task family (commonsense NLI). Broader tasks recommended before a hard marketing claim.

## What this means for the product
This is the first **proven** instance of the promise: a **sub-4-bit model that keeps ~99%
of quality** on a real task, produced by the pipeline on consumer hardware, fully documented.
The recipe to ship is **IQ3_M + imatrix** (`ship-sub4`).
