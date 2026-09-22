# Phase 4 — protect layers 8–12 (full map), full 24 layers

Whole-model rerun after the L3–7 miss (18.492). Layers 0–7, 13–18,
and 19–23 stay at `512,256,64`. Layers 8–12, the last chunk on the
Phase 4 map (alone perplexity 14.767, Δ +0.520), move to the same
`512,256,64`. Equivalently: **all layers 0–23 at `512,256,64`**.
Embeddings, the output head, and norms stay full precision.

## Verdict

**RUN IN PROGRESS.** WikiText-2 test perplexity and PASS/FAIL will be
filled after the full-24 quantize and eval finish. Gate is the printed
value ≤ **14.959** (+5% vs **14.247**). Baseline target for this run is
**14.247** again.

Expected: remaining miss after L3–7 was +3.533; L8–12 alone was only
+0.520. Clearing the gate at this rung is not expected.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per layer.
  `PBR_STAGES=512,256,64` on every layer (no `PBR_LAYER_STAGES`
  overrides).
- `PBR_CHUNK=20000`. Resume-friendly: `layer_npz` and `pbr_progress.json`
  after every layer.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length target 299078.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1,
  numpy 2.4.4. CPU, four threads.
- Checkpoint is dequantized fp32 for perplexity. Gitignored
  (`pbr_ladder/qwen05b_*/`). Not a packed file and not a size win.
- No KLT / shared rotation / permutation / outlier correction.

## Recipe

Index cost is `sum(log2(k)) / 8`, which is what the script prints.
Every layer has the same 14,909,440 linear parameters.

| Layers | n | Stages | Index bits | Index bpw |
| --- | ---: | --- | ---: | ---: |
| 0–23 | 24 | `512,256,64` | 9+8+6 = 23 | 2.875 |

Average index cost on the 357,826,560 touched parameters:

`(24 × 2.875) / 24 = 2.875` bpw.

Codebook side info estimate (`push_below3.py` formula): one layer at
`512,256,64` is 2.939 including side info, so the average is **2.939**.

| Recipe | Layers at `512,256,64` | Layers at `256,256,64` | Index bpw | Index + side |
| --- | --- | --- | ---: | ---: |
| Phase 2 flat | — | 0–23 | 2.75 | 2.798 |
| Phase 2 retry (early only) | 0–2 | 3–23 | 2.765625 | 2.816 |
| Late-block protect | 0–2 and 19–23 | 3–18 | 2.791667 | 2.845 |
| L13–18 protect | 0–2, 13–18, and 19–23 | 3–12 | 2.822917 | 2.88025 |
| L3–7 protect | 0–7, 13–18, and 19–23 | 8–12 | 2.848958 | 2.909625 |
| This run (all bumped) | 0–23 | — | 2.875 | 2.939 |

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
export PBR_CHUNK=20000
python3 eval_ppl.py ./qwen05b

PBR_STAGES=512,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_l812
python3 eval_ppl.py ./qwen05b_p4_l812
```

## Logs

- `phase4_l812_quantize.log`
- `phase4_l812_ppl_baseline.log`
- `phase4_l812_ppl_full.log`
- `phase4_l812_runner.log`
- `phase4_l812_full24.json`

## After FAIL (expected)

Phase 4 at the `512,256,64` rung is exhausted: every map chunk
(L0–2 early probe, L19–23, L13–18, L3–7, L8–12) has been raised inside
a full-24. Do not invent a next recipe or start another full-24 without
a new locked step.

Possible next-rung options (bullets only; not started):

- Higher stage (e.g. `1024,…` or `256,256,256`) on selected layers
- Outlier correction later
- Other transforms (KLT / shared rotation / permutation) if locked later
