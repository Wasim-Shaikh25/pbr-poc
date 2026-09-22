# Phase 4 — protect layers 3–7, full 24 layers

Whole-model rerun after the L13–18 miss (18.803). Layers 0–2, 13–18,
and 19–23 stay at `512,256,64`. Layers 3–7, the next chunk on the
Phase 4 map (alone perplexity 14.897, Δ +0.650), move to the same
`512,256,64`. Layers 8–12 stay at `256,256,64`. Embeddings, the output
head, and norms stay full precision.

## Verdict

**FAIL.** WikiText-2 test perplexity is **18.492**. The gate is the
printed value ≤ **14.959** (+5% vs **14.247**). 18.492 is 3.533 over
that gate. Phase 4 is not done: the whole-model bar is still missed.

This run's unmodified baseline, same `eval_ppl.py` and the same 299078
tokens, is **14.247** again.

Percent vs 14.247 is `(18.492 − 14.247) / 14.247` = **+29.796%**.

| Comparison | Other ppl | This − other |
| --- | ---: | ---: |
| Phase 4 L13–18 protect (L0–2, L13–18, and L19–23 at `512,256,64`) | 18.803 | −0.311 |
| Phase 4 late-block protect (L0–2 and L19–23 at `512,256,64`) | 19.039 | −0.547 |
| Phase 2 retry, early-only mix (L0–2 at `512,256,64`) | 19.280 | −0.788 |
| Phase 2, flat all-24 (`256,256,64` on 0–23) | 19.427 | −0.935 |
| Gate | 14.959 | +3.533 |

18.492 is 0.311 lower than the L13–18 full-24 and 0.935 lower than
the flat 24-layer recipe. Both moves are small next to the miss: 18.492
is still much closer to 18.803 than to 14.959.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per layer.
  `PBR_STAGES=256,256,64` with per-layer overrides for layers 0–7,
  13–18, and 19–23 (see the recipe table).
- `PBR_CHUNK=20000`. The chunk only splits the pooled k-means distance
  matrix. Each row's argmin is the same computation as the default
  `50000`. This was a full pass. The script wrote
  `layer_npz/layer_<i>.npz` and `pbr_progress.json` after every layer.
  All 24 finished. Resume was available and was not needed: the output
  directory was empty when quantization started. One attempt, rc=0.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length 299078 on the
  baseline and on the quantized checkpoint.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1,
  numpy 2.4.4. CPU, four threads.
- The saved folder is a dequantized fp32 checkpoint for perplexity
  (`model.safetensors` is 1.9 GB; the original bf16 file is 943 MB).
  All 290 tensors are dtype float32, including a layer-3 weight (raised
  to `512,256,64`) and a layer-8 weight (left at `256,256,64`). It is
  gitignored (`pbr_ladder/qwen05b_*/`). The file is a dequantized fp32
  copy. It is a perplexity checkpoint.

The quantizer reports 357,826,560 parameters touched (14,909,440 per
layer). Layer 23 finished at 3484.4s of quantizer time. Wall clock for
the quantize process was 09:14:38Z–10:12:56Z.

## Recipe

Index cost is `sum(log2(k)) / 8`, which is what the script prints.
Every layer has the same 14,909,440 linear parameters.

| Layers | n | Stages | Index bits | Index bpw |
| --- | ---: | --- | ---: | ---: |
| 0, 1, 2 | 3 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 3, 4, 5, 6, 7 | 5 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 8, 9, 10, 11, 12 | 5 | `256,256,64` | 8+8+6 = 22 | 2.75 |
| 13, 14, 15, 16, 17, 18 | 6 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 19, 20, 21, 22, 23 | 5 | `512,256,64` | 9+8+6 = 23 | 2.875 |

Average index cost on the 357,826,560 touched parameters:

`(19 × 2.875 + 5 × 2.75) / 24 = 2.848958` bpw.

That is +0.098958 versus the flat Phase 2 index cost (2.75),
+0.083333 versus the Phase 2 retry average (2.765625), +0.057292
versus the late-block average (2.791667), and +0.026042 versus the
L13–18 average (2.822917). It stays well under ~4.

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
| Late-block protect | 0–2 and 19–23 | 3–18 | 2.791667 | 2.845 |
| L13–18 protect | 0–2, 13–18, and 19–23 | 3–12 | 2.822917 | 2.88025 |
| This run | 0–7, 13–18, and 19–23 | 8–12 | 2.848958 | 2.909625 |

The index-plus-side average uses those same one-layer figures:
`(19 × 2.939 + 5 × 2.798) / 24 = 2.909625`. It is an estimate of
codebook side info added to the printed index cost.

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247` on the printed
perplexity. Gate is the printed value ≤ 14.959.

| Run | What was quantized | ppl | delta | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline (this run) | nothing | 14.247 | — | — | — |
| Phase 2, flat all-24 | 0–23 at `256,256,64` | 19.427 | +5.180 | +36.359% | FAIL |
| Phase 4, early block only | 0–2 at `512,256,64`; 3–23 full precision | 14.873 | +0.626 | +4.394% | PASS |
| Phase 2 retry, early-only mix | 0–2 at `512,256,64`; 3–23 at `256,256,64` | 19.280 | +5.033 | +35.327% | FAIL |
| Late-block protect | 0–2 and 19–23 at `512,256,64`; 3–18 at `256,256,64` | 19.039 | +4.792 | +33.635% | FAIL |
| L13–18 also protected | 0–2, 13–18, and 19–23 at `512,256,64`; 3–12 at `256,256,64` | 18.803 | +4.556 | +31.979% | FAIL |
| This run, L3–7 also protected | 0–7, 13–18, and 19–23 at `512,256,64`; 8–12 at `256,256,64` | 18.492 | +4.245 | +29.796% | **FAIL** |

The Phase 2, Phase 4, Phase 2 retry, late-block, and L13–18 rows are
those runs' logs, not a second measurement. The baseline and the L3–7
row are this run.

18.492 − 19.427 = −0.935 perplexity versus the flat 24-layer recipe.
18.492 − 19.280 = −0.788 versus the early-only mix.
18.492 − 19.039 = −0.547 versus the late-block protect.
18.492 − 18.803 = −0.311 versus the L13–18 protect. That −0.311 is
the layers 3–7 bump inside an otherwise identical sequential pass
(layers 0–2, 13–18, and 19–23 already at `512,256,64`, layers 8–12
still at `256,256,64`). 18.492 − 14.959 = +3.533 versus the gate.

Protecting layers 3–7 is a partial fix, a bit larger than protecting
13–18 (−0.311 here, −0.236 there) and protecting 19–23 (−0.241). It
does not bring the quantized network under 14.959.

## Next iteration

No new mid-layer isolation was run. The Phase 4 map already scored
layers 3–7, 8–12, 13–18, and 19–23 alone at `256,256,64`. After the
L13–18 miss, that map's next chunk was 3–7. This run is that chunk,
inside the full 24, and it still misses.

| Order | Layers | Alone ppl | Alone Δppl | Role after this run |
| --- | --- | ---: | ---: | --- |
| done, prior full-24 | 19–23 at `512,256,64` | 15.447 at `256,256,64` | +1.200 | protected; that full-24 was 19.039 |
| done, prior full-24 | 13–18 at `512,256,64` | 15.053 at `256,256,64` | +0.806 | protected; that full-24 was 18.803 |
| done this run | 3–7 at `512,256,64`, inside the full 24 | 14.897 at `256,256,64` | +0.650 | protected; full-24 now 18.492 |
| last | 8–12 | 14.767 | +0.520 | smallest isolated hit; still at `256,256,64` |

The remaining mildest chunk is **layers 8–12**. That order is the
map's alone deltas. It is not a forecast of how 8–12 will add once
3–7, 13–18, and 19–23 are already at `512,256,64`.

A follow-up full-24 that also raises 8–12 was not started. The 3–7
bump, on an isolated delta of +0.650, moved the full model by −0.311.
The miss that remains is +3.533. Layers 8–12 are the last place on
this map to spend bits, and one more chunk at this rung is not enough
to clear 14.959 on the evidence in hand.

KLT, shared rotation, permutation, and outlier correction were not
retried. The quantizer's existing per-tensor rotation is the same code
path as the 18.803 run.

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
export PBR_CHUNK=20000
python3 eval_ppl.py ./qwen05b

PBR_STAGES=256,256,64 \
PBR_LAYER_STAGES='0:512,256,64;1:512,256,64;2:512,256,64;3:512,256,64;4:512,256,64;5:512,256,64;6:512,256,64;7:512,256,64;13:512,256,64;14:512,256,64;15:512,256,64;16:512,256,64;17:512,256,64;18:512,256,64;19:512,256,64;20:512,256,64;21:512,256,64;22:512,256,64;23:512,256,64' \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_l37
python3 eval_ppl.py ./qwen05b_p4_l37
```

## Logs

- `phase4_l37_quantize.log` — 24 layers, 357,826,560 parameters,
  per-layer index 2.875 / 2.75 / 2.875, `QUANTIZE ATTEMPT 1 rc=0`
- `phase4_l37_ppl_baseline.log` — 14.247, 299078 tokens
- `phase4_l37_ppl_full.log` — 18.492, 299078 tokens
- `phase4_l37_runner.log`
- `phase4_l37_full24.json`

## What this does not say

- No packed sub-4-bit file was written. 2.848958 bpw is the printed
  index cost, averaged over the touched parameters. The checkpoint
  used for perplexity is a dequantized fp32 copy.
- Layers 8–12 were left at `256,256,64`. A uniform 24-layer
  `512,256,64` run was not done.
- Layers 8–12 were not raised inside this full model. That is the
  last chunk on the map, and it was not started here.
- `256,256,256` was not run.
