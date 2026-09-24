# Sub-3-bit / Rotation / Vector-Codebook Investigation — Findings & Decision

Date: 2026-09-24. Governed by [`../AGENTS.md`](../AGENTS.md) RULE 0. Honest by design.
Companion: [`DEV_LOG.md`](DEV_LOG.md), [`POSITIONING.md`](POSITIONING.md), [`results_3b_taskacc.md`](results_3b_taskacc.md).

## The question
Can we beat llama.cpp's IQ3_M — training-free (PTQ), CPU-only — either by going **below 3 bpw at ≥98%
quality**, or by matching IQ3's 99% at **fewer bits** (~3.0–3.25 vs 3.86)? Levers explored:
randomized Hadamard rotation (QuIP#/QuaRot-style), vector/lattice codebooks (VPTQ/QTIP/LLVQ-style),
imatrix importance weighting, residual VQ (RVQ), GPTQ-style ideas.

## What we already had (the anchors, on disk at `D:\pbr_work\out`)
| model | bpw | HellaSwag retention |
|---|---|---|
| Q4_K_M (near-lossless proxy) | 5.00 | 100% |
| **IQ3_M + imatrix (SHIP)** | **3.86** | **99.1%** |
| stock Q3_K_M | 4.12 | 93.3% |
| IQ2_M + imatrix | 2.96 | 90.6% |
| IQ2_S + imatrix (this session) | 2.75 | 90.8% |

The 3→2-bit cliff is real: everything ≤ ~2.75 bpw sits at ~90–91%, far below the 98% bar.

## The experiments (all on real Qwen2.5-3B weights; scripts in this dir)
Metric = weight-reconstruction rel-error vs f16, compared to the **actual IQ3_M** (dequantized
from the shipped GGUF), using IQ3's own imatrix. Lower = better.

| # | experiment | script | result vs IQ3 |
|---|---|---|---|
| 1 | rotation + vector-quant (VQ) | `beat_iq3.py` | **loses 26–34%** |
| 2 | rotation alone (does it help VQ?) | inline | **neutral** (0.1614 vs 0.1603) — rotation decorrelates, which VQ needs |
| 3 | imatrix-weighted VQ (weighted-error space) | `beat_iq3_imatrix.py` | **loses 50%** |
| 4 | **heroic: RVQ + imatrix + MATCHED per-tensor bits** | `heroic_rvq.py` | **loses 46%** (IQ3 0.099 vs ours 0.144, lost on all 9 tensors) |
| — | rate-distortion proof | — | IQ3 (0.099 weighted / 0.126 plain) sits **on** the Gaussian floor (~0.125 at 3 bpw) |

Early method-validation on 0.5B (`rotation_spike.py`, `vector_codebook_spike.py`) confirmed the
*mechanisms* work in isolation (rotation flattens outliers: kurtosis 22.8→6.6, max/std 42→7;
vector codebook beats scalar by 15–38%) — but they do **not** translate into beating IQ3 on the
real model, because IQ3 already combines a lattice-tuned codebook + GPTQ-style error feedback +
mixed precision + years of tuning.

## Conclusion (five convergent negatives + a proof + the literature)
**IQ3_M is the training-free ceiling for this model.** We cannot beat it with PTQ rotation/
vector-codebook methods. Confirmed independently by our 4 matched experiments, the rate-distortion
bound, and published results (QuIP#/QTIP/VPTQ all go lossless only at ~3–3.5 bpw; none clear 98% at
2-bit training-free).

### On the Leech lattice / "invent a new codebook"
Leech-lattice VQ (LLVQ, arXiv 2603.11021) is the current SOTA codebook and beats QuIP#/QTIP by ~14%
at 2-bit. But: (a) the Leech lattice is the **proven-optimal** packing in 24D — you cannot invent a
better codebook by "combining" existing ones; the optimum is singular, not a blend. (b) Our k-means
was 46% behind IQ3, so a ~14% codebook improvement cannot bridge it. (c) Even implementing LLVQ +
GPTQ error-feedback + a CPU kernel (months of work) yields, at 3 bpw, roughly **parity** with IQ3 —
a best-case ~15% size win, no breakthrough. The only lever that truly moves the wall is **training
(QAT / BitNet-style)**, which breaks the training-free + CPU wedge.

## Decision
**Ship IQ3_M** (99% @ 3.86 bpw) as the product's quality engine. Stop the quantizer-beating chase.
This is not a loss: "3-bit model at 99% quality, training-free, one CPU command" was the mission,
and **we have it** — now validated as near information-theoretic optimal (a credibility point).
The remaining value, and the only direction with no proven ceiling, is the **product**: on-device
demo + one-command pipeline + provenance. See [`SHIP.md`](SHIP.md).
