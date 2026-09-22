# Phase 4 late-block protect — full 24 layers

Whole-model rerun after the Phase 4 map. Layers 0–2 stay at the stages
that cleared the early-block probe (`512,256,64`). Layers 19–23, the hot
chunk on that map (alone perplexity 15.447, Δ +1.200), move to the same
`512,256,64`. Layers 3–18 stay at `256,256,64`. Embeddings, the output
head, and norms stay full precision.

## Verdict

**FAIL.** WikiText-2 test perplexity is **19.039**. The gate is the
printed value ≤ **14.959** (+5% vs **14.247**). 19.039 is 4.080 over
that gate. Phase 4 is not done: the whole-model bar is still missed.

This run's unmodified baseline, same `eval_ppl.py` and the same 299078
tokens, is **14.247** again.

Phase 2's flat all-24 result at `256,256,64` on every layer was
**19.427**. The Phase 2 retry (layers 0–2 at `512,256,64`, layers 3–23
at `256,256,64`) was **19.280**. This recipe prints 0.388 lower than
the flat run and 0.241 lower than that early-only mix. Both moves are
small next to the miss: 19.039 is still much closer to 19.280 than to
14.959.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per layer.
  `PBR_STAGES=256,256,64` with per-layer overrides for layers 0–2 and
  19–23 (see the recipe table).
- `PBR_CHUNK=20000`. The chunk only splits the pooled k-means distance
  matrix. Each row's argmin is the same computation as the default
  `50000`. This was a full pass. The script wrote
  `layer_npz/layer_<i>.npz` and `pbr_progress.json` after every layer.
  All 24 finished; those files were not needed to resume.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length 299078 on the
  baseline and on the quantized checkpoint.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1,
  numpy 2.4.4. CPU, four threads.
- The saved folder is a dequantized fp32 checkpoint for perplexity.
  It is gitignored (`pbr_ladder/qwen05b_*/`). It is not a packed file
  and it is not a size win.

The quantizer reports 357,826,560 parameters touched (14,909,440 per
layer). Layer 23 finished at 2153.4s of quantizer time. Wall clock for
the quantize process was 06:40:11Z–07:16:14Z.

## Recipe

Index cost is `sum(log2(k)) / 8`, which is what the script prints.
Every layer has the same 14,909,440 linear parameters.

| Layers | n | Stages | Index bits | Index bpw |
| --- | ---: | --- | ---: | ---: |
| 0, 1, 2 | 3 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 3–18 | 16 | `256,256,64` | 8+8+6 = 22 | 2.75 |
| 19, 20, 21, 22, 23 | 5 | `512,256,64` | 9+8+6 = 23 | 2.875 |

Average index cost on the 357,826,560 touched parameters:

`(8 × 2.875 + 16 × 2.75) / 24 = 2.791667` bpw.

That is +0.041667 versus the flat Phase 2 index cost (2.75) and
+0.026042 versus the Phase 2 retry average (2.765625). It stays well
under ~4.

Codebook side info is the `push_below3.py` estimate, not a packed
checkpoint:

```
side = (sum(k) * 8 * 16 + out_features * 16) / n_params
raw  = sum(log2(k)) / 8 + side
```

`16` is fp16 bits for each codebook coordinate and for one scale per
output row. Parameter-weighted across the seven linears, one layer is
2.798 at `256,256,64` and 2.939 at `512,256,64` (same figures as
Phase 4). Narrow `k_proj` / `v_proj` are higher (3.411 and 3.821
including side info).

| Recipe | Layers at `512,256,64` | Layers at `256,256,64` | Index bpw | Index + side |
| --- | --- | --- | ---: | ---: |
| Phase 2 flat | — | 0–23 | 2.75 | 2.798 |
| Phase 2 retry (early only) | 0–2 | 3–23 | 2.765625 | 2.816 |
| This run | 0–2 and 19–23 | 3–18 | 2.791667 | 2.845 |

The index-plus-side average uses those same one-layer figures:
`(8 × 2.939 + 16 × 2.798) / 24 = 2.845`. It is not an on-disk size.

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247` on the printed
perplexity. Gate is the printed value ≤ 14.959.

| Run | What was quantized | ppl | delta | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline (this run) | nothing | 14.247 | — | — | — |
| Phase 2, flat all-24 | 0–23 at `256,256,64` | 19.427 | +5.180 | +36.359% | FAIL |
| Phase 4, early block only | 0–2 at `512,256,64`; 3–23 full precision | 14.873 | +0.626 | +4.394% | PASS |
| Phase 2 retry, early-only mix | 0–2 at `512,256,64`; 3–23 at `256,256,64` | 19.280 | +5.033 | +35.327% | FAIL |
| This run, late block protected | 0–2 and 19–23 at `512,256,64`; 3–18 at `256,256,64` | 19.039 | +4.792 | +33.635% | **FAIL** |

The Phase 2, Phase 4, and Phase 2 retry rows are those runs' logs, not
a second measurement. The baseline and the late-protect row are this
run.

19.039 − 19.427 = −0.388 perplexity versus the flat 24-layer recipe.
19.039 − 19.280 = −0.241 versus the early-only mix. That −0.241 is the
late-block bump inside an otherwise identical sequential pass (layers
0–2 already at `512,256,64`, layers 3–18 still at `256,256,64`).
19.039 − 14.959 = +4.080 versus the gate.

Protecting layers 19–23 is a partial fix. It does not bring the
quantized network under 14.959.

## Next iteration

No new mid-layer isolation was run. The Phase 4 map already scored
layers 3–7, 8–12, 13–18, and 19–23 alone at `256,256,64`, and it
already stated the fallback if a full-24 that protects 19–23 still
misses. This run is that miss.

| Order | Layers | Alone ppl | Alone Δppl | Role after this run |
| --- | --- | ---: | ---: | --- |
| done this run | 19–23 at `512,256,64`, inside the full 24 | 15.447 at `256,256,64` | +1.200 | protected; full-24 still 19.039 |
| next | 13–18 | 15.053 | +0.806 | next protect |
| then | 3–7 | 14.897 | +0.650 | after 13–18 |
| last | 8–12 | 14.767 | +0.520 | smallest isolated hit |

The next extra bits go on **layers 13–18**. Then layers 3–7. Layers
8–12 stay at `256,256,64` the longest. That order is the map's alone
deltas. It is not a forecast of how 13–18 will add once 19–23 are
already at `512,256,64`.

A follow-up full-24 that also raises 13–18 was not started. The
late-block bump, on the larger isolated chunk (+1.200), moved the full
model by only −0.241. The miss that remains is +4.080. Layers 13–18
are the next place to spend bits, and one more chunk at this rung is
not enough to clear 14.959 on the evidence in hand.

The map's bisection of 19–23 (layers 22–23 at 14.971, layers 19–21 at
14.746, both at `256,256,64` with every other layer full precision) was
not used to narrow this run. The locked recipe raised the whole late
block together.

KLT, shared rotation, permutation, and outlier correction were not
retried.

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
export PBR_CHUNK=20000
python3 eval_ppl.py ./qwen05b

PBR_STAGES=256,256,64 \
PBR_LAYER_STAGES='0:512,256,64;1:512,256,64;2:512,256,64;19:512,256,64;20:512,256,64;21:512,256,64;22:512,256,64;23:512,256,64' \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_late
python3 eval_ppl.py ./qwen05b_p4_late
```

## Logs

- `phase4_late_quantize.log` — 24 layers, 357,826,560 parameters,
  per-layer index 2.875 / 2.75 / 2.875, `QUANTIZE DONE rc=0`
- `phase4_late_ppl_baseline.log` — 14.247, 299078 tokens
- `phase4_late_ppl_full.log` — 19.039, 299078 tokens
- `phase4_late_runner.log`
- `phase4_late_block_full24.json`

## What this does not say

- No packed sub-4-bit file was written. Do not read 2.791667 bpw as an
  on-disk size. The checkpoint used for perplexity is a dequantized
  fp32 copy.
- Layers 3–18 were not given `512,256,64`. A uniform 24-layer
  `512,256,64` run was not done.
- Layers 22–23 were not raised on their own inside the full model.
- `256,256,256` was not run.
