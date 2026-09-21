# PBR-Ladder — script guide run

Source: user-supplied `PBR_Ladder_Script_Guide.docx`, plus 6 of the 9 scripts
it describes (`push_below4.py`, `push_below3.py`, `push_nested.py`,
`ladder_exact.py`, `capture_activations.py`, `eval_ppl.py`). The other 3
scripts named in the guide (`formula_probe.py`, `node_combo_q4.py`,
`node_structures.py`) were **not supplied** and are not in this repo yet.

Two separate runs exist:

1. **Synthetic environment check** — ran in this repo's cloud sandbox
   (no `huggingface.co` access there). See below.
2. **Real Qwen2.5-0.5B-Instruct run** — ran on the user's own machine
   (has HF access), following `docs/pbr_ladder/HANDOFF.md`. Raw console
   output archived at `artifacts/pbr_ladder/real_qwen05b_run.txt`. This is
   the first real-model evidence in this repo for these scripts.

## Real Qwen2.5-0.5B-Instruct run

Ran on the user's local machine (has `huggingface.co` access), not
reproduced or independently re-run in this sandbox. Raw console output:
`artifacts/pbr_ladder/real_qwen05b_run.txt` (verbatim, not summarized here
by hand — the tables below are the same numbers).

### §5.1 — tensor quality gate, 9 tensors

`gate_proj`, `down_proj`, `q_proj` at layers 2, 12, 21, with real captured
calibration activations. Gate: nodes at ≤3.3 bpw beat INT4 RTN output dB
by ≥2 dB on **most** tensors.

| Tensor | INT4 RTN dB | INT3 rot+GPTQ dB | ~2.8 bpw nodes dB | ~3.0–3.14 bpw nodes dB | Δ vs INT4 RTN |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 gate_proj | 22.43 | 20.06 | 19.69 | 21.03 | −1.40 |
| L2 down_proj | 17.89 | 36.08 | 33.04 | 34.48 | **+16.59** |
| L2 q_proj | 21.05 | 21.64 | 21.25 | 22.62 | +1.57 |
| L12 gate_proj | 23.12 | 22.31 | 21.96 | 23.30 | +0.18 |
| L12 down_proj | 19.42 | 20.37 | 19.82 | 21.28 | +1.86 |
| L12 q_proj | 21.11 | 21.44 | 21.11 | 22.49 | +1.38 |
| L21 gate_proj | 23.35 | 22.90 | 22.53 | 23.87 | +0.52 |
| L21 down_proj | 22.56 | 33.69 | 33.13 | 34.53 | **+11.97** |
| L21 q_proj | 17.33 | 20.50 | 20.11 | 21.51 | **+4.18** |

**Gate: FAIL on most tensors.** Only 3/9 (both `down_proj` layers and
L21 `q_proj`) clear the ≥2 dB bar; `gate_proj` is flat-to-negative on all
3 layers, `q_proj` is a marginal win on 2/3 layers. `down_proj` wins big
everywhere. This is a real, mixed result — not rounded toward pass.

Known issue: `push_below4.py`'s 8-D RVQ step (`rvq()` → `kmeans()`)
crashed with `numpy.core._exceptions._ArrayMemoryError` on L21
`down_proj` (tries to materialize a 544,768×256 float64 distance matrix
≈ 1.04 GiB per k-means iteration on the full-size tensor; `PBR_ROWS` can
work around this by truncating rows, at the cost of testing less of the
tensor). `push_nested.py`, `push_below3.py`, and `ladder_exact.py` all
completed on that same tensor without truncation, so the ~3.0 bpw / INT4
RTN / INT3 rot+GPTQ numbers for L21 down_proj above come from those, not
from `push_below4.py`.

### §5.2 — single-layer perplexity gate

```
baseline:                              WikiText-2 test perplexity: 14.247
L12 gate_proj swapped to 2.785 bpw:    WikiText-2 test perplexity: 14.269
```

Swap: `push_below3.py` with `PBR_SAVE_KS=256,256,64` (2.785 bpw, the
tensor's own encode from the table above, 21.96 dB output SQNR) saved via
`PBR_SAVE`/`PBR_NAME`, then loaded into the full model by `eval_ppl.py`.

Relative change: (14.269 − 14.247) / 14.247 = **+0.154%**.

**Gate: PASS** (guide requires <0.5%). This is a real, measured result on
the actual model — not scaled or extrapolated. Note it's on a tensor
(`L12 gate_proj`) that itself only marginally passed/failed the dB gate
above (+0.18 dB vs INT4 RTN) — the model-level perplexity impact of one
swapped layer was still negligible.

### What this does and doesn't support

- Real evidence that a single quantized `down_proj` or `q_proj` layer can
  swap in with no meaningful perplexity cost, at the tested bpw.
  `down_proj` is where this method's dB win is real and large; `gate_proj`
  is not a dB win on this evidence.
- **Not** a whole-model result — one layer was swapped, not all layers.
  §5.3 (whole-model quantization) and §5.4 (retention across 5 zero-shot
  tasks) from the guide were not run; that needs the driver script the
  guide describes as still to be built.
- **Not** a claim about tensors or layers not tested here.

## Synthetic environment check

Ran in this repo's cloud sandbox (no `huggingface.co` access there), just
to confirm the scripts execute correctly and are seeded/deterministic
before the real run above:

```bash
cd pbr_ladder
pip install numpy
python3 push_below4.py
python3 push_below3.py
python3 push_nested.py
python3 ladder_exact.py
```

**Result: PASS.** Output matches the guide's Appendix A exactly (seed 0,
512×896 synthetic Student-t tensor, synthetic activations with 8 outlier
channels):

| Method | bpw | output dB (measured) | output dB (Appendix A) |
| --- | ---: | ---: | ---: |
| INT3 RTN | 3.250 | 11.11 | 11.11 |
| INT3 GPTQ | 3.250 | 13.52 | 13.52 |
| INT3 rot+GPTQ | 3.250 | 18.91 | 18.91 |
| INT2 rot+GPTQ | 2.250 | 10.53 | 10.53 |
| 256+256+64 (push_below3) | 2.929 | 18.61 | 18.61 |
| Successive refinement tier 1/2/3 (push_nested) | 1.09 / 2.18 / 3.27 | 8.98 / 13.11 / 16.35 | 8.98 / 13.11 / 16.35 |
| Exact-only / ladder tier 3 + exact (ladder_exact) | 10.65 / 12.50 | bit-exact | bit-exact |

Full logs: `artifacts/pbr_ladder/push_below4_synthetic.txt`,
`artifacts/pbr_ladder/push_below3_synthetic.txt`,
`artifacts/pbr_ladder/push_nested_synthetic.txt`,
`artifacts/pbr_ladder/ladder_exact_synthetic.txt`.

This confirms the scripts run correctly. It is not itself a Qwen result —
see the real run above for that.

## `formula_probe.py`, `node_combo_q4.py`, `node_structures.py`

Not supplied, not in this repo. §3.1–3.3 of the guide (exact-formula
cover, shared-node menus, pair/quad/FP4 node structures) have not been
run, synthetic or real.

## Go/no-go gates (guide §7.2) — status

| Gate | Status |
| --- | --- |
| Environment check (synthetic, §2.3) | **PASS** — matches Appendix A |
| Tensor quality, real weights (§5.1) | **FAIL on most tensors** — 3/9 clear ≥2 dB vs INT4 RTN (both down_proj, L21 q_proj); down_proj wins big, gate_proj/q_proj mostly don't |
| Single-layer perplexity (§5.2) | **PASS** — +0.154% on L12 gate_proj swap, < 0.5% budget |
| Whole-model quantization (§5.3) | **NOT RUN** — driver script not built |
| Retention across benchmark set (§5.4) | **NOT RUN** — needs §5.3 |

## Honesty

- The tensor-quality gate is a real, mostly-FAIL result. `down_proj`
  layers show a large, consistent dB win; `gate_proj` does not, and
  `q_proj` is marginal. Don't round this up to "the method works" — it
  works clearly for one of three tested projection types so far.
- The single-layer perplexity PASS is for exactly one swapped tensor
  (`L12 mlp.gate_proj`, itself a near-zero dB-gate result) on one model.
  It is not evidence about whole-model perplexity, retention, or any
  tensor/layer not tested.
- `formula_probe.py`, `node_combo_q4.py`, `node_structures.py` were never
  supplied or run, synthetic or real.
- Raw console output for the real run is archived verbatim at
  `artifacts/pbr_ladder/real_qwen05b_run.txt` — read that, not just the
  tables above, before citing these numbers elsewhere.
