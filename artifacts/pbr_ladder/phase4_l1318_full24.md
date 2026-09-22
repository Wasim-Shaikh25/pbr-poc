# Phase 4 — protect layers 13–18, full 24 layers

Whole-model rerun after the late-block miss (19.039). Layers 0–2 and
layers 19–23 stay at `512,256,64`. Layers 13–18, the next chunk on the
Phase 4 map (alone perplexity 15.053, Δ +0.806), move to the same
`512,256,64`. Layers 3–12 stay at `256,256,64`. Embeddings, the output
head, and norms stay full precision.

## Verdict

**FAIL.** WikiText-2 test perplexity is **18.803**. The gate is the
printed value ≤ **14.959** (+5% vs **14.247**). 18.803 is 3.844 over
that gate. Phase 4 is not done: the whole-model bar is still missed.

This run's unmodified baseline, same `eval_ppl.py` and the same 299078
tokens, is **14.247** again.

Percent vs 14.247 is `(18.803 − 14.247) / 14.247` = **+31.979%**.

| Comparison | Other ppl | This − other |
| --- | ---: | ---: |
| Phase 4 late-block protect (L0–2 and L19–23 at `512,256,64`) | 19.039 | −0.236 |
| Phase 2 retry, early-only mix (L0–2 at `512,256,64`) | 19.280 | −0.477 |
| Phase 2, flat all-24 (`256,256,64` on 0–23) | 19.427 | −0.624 |
| Gate | 14.959 | +3.844 |

18.803 is 0.236 lower than the late-block full-24 and 0.624 lower than
the flat 24-layer recipe. Both moves are small next to the miss: 18.803
is still much closer to 19.039 than to 14.959.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per layer.
  `PBR_STAGES=256,256,64` with per-layer overrides for layers 0–2,
  13–18, and 19–23 (see the recipe table).
- `PBR_CHUNK=20000`. The chunk only splits the pooled k-means distance
  matrix. Each row's argmin is the same computation as the default
  `50000`. This was a full pass. The script wrote
  `layer_npz/layer_<i>.npz` and `pbr_progress.json` after every layer.
  All 24 finished. Resume was available and was not needed: the output
  directory was empty when quantization started.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length 299078 on the
  baseline and on the quantized checkpoint.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1,
  numpy 2.4.4. CPU, four threads.
- The saved folder is a dequantized fp32 checkpoint for perplexity
  (`model.safetensors` is 1.9 GB; the original bf16 file is 943 MB).
  Checked tensors are dtype F32, including a layer-13 weight. It is
  gitignored (`pbr_ladder/qwen05b_*/`). It is not a packed file and it
  is not a size win.

The baseline log was written first (14.247). A wrapper `exit` stopped
that process before quantization. The second invocation kept that
baseline log and started quantization from scratch. It did not reload
another recipe's layers.

The quantizer reports 357,826,560 parameters touched (14,909,440 per
layer). Layer 23 finished at 3056.3s of quantizer time. Wall clock for
the quantize process was 07:52:53Z–08:44:01Z.

## Recipe

Index cost is `sum(log2(k)) / 8`, which is what the script prints.
Every layer has the same 14,909,440 linear parameters.

| Layers | n | Stages | Index bits | Index bpw |
| --- | ---: | --- | ---: | ---: |
| 0, 1, 2 | 3 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 3–12 | 10 | `256,256,64` | 8+8+6 = 22 | 2.75 |
| 13, 14, 15, 16, 17, 18 | 6 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 19, 20, 21, 22, 23 | 5 | `512,256,64` | 9+8+6 = 23 | 2.875 |

Average index cost on the 357,826,560 touched parameters:

`(14 × 2.875 + 10 × 2.75) / 24 = 2.822917` bpw.

That is +0.072917 versus the flat Phase 2 index cost (2.75),
+0.057292 versus the Phase 2 retry average (2.765625), and +0.031250
versus the late-block average (2.791667). It stays well under ~4.

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
| This run | 0–2, 13–18, and 19–23 | 3–12 | 2.822917 | 2.88025 |

The index-plus-side average uses those same one-layer figures:
`(14 × 2.939 + 10 × 2.798) / 24 = 2.88025`. It is not an on-disk size.

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
| This run, L13–18 also protected | 0–2, 13–18, and 19–23 at `512,256,64`; 3–12 at `256,256,64` | 18.803 | +4.556 | +31.979% | **FAIL** |

The Phase 2, Phase 4, Phase 2 retry, and late-block rows are those
runs' logs, not a second measurement. The baseline and the L13–18 row
are this run.

18.803 − 19.427 = −0.624 perplexity versus the flat 24-layer recipe.
18.803 − 19.280 = −0.477 versus the early-only mix.
18.803 − 19.039 = −0.236 versus the late-block protect. That −0.236 is
the layers 13–18 bump inside an otherwise identical sequential pass
(layers 0–2 and 19–23 already at `512,256,64`, layers 3–12 still at
`256,256,64`). 18.803 − 14.959 = +3.844 versus the gate.

Protecting layers 13–18 is a partial fix of the same size as protecting
19–23 (−0.236 here, −0.241 there). It does not bring the quantized
network under 14.959.

## Next iteration

No new mid-layer isolation was run. The Phase 4 map already scored
layers 3–7, 8–12, 13–18, and 19–23 alone at `256,256,64`. After the
late-block miss, that map's next chunk was 13–18. This run is that
chunk, inside the full 24, and it still misses.

| Order | Layers | Alone ppl | Alone Δppl | Role after this run |
| --- | --- | ---: | ---: | --- |
| done, prior full-24 | 19–23 at `512,256,64` | 15.447 at `256,256,64` | +1.200 | protected; that full-24 was 19.039 |
| done this run | 13–18 at `512,256,64`, inside the full 24 | 15.053 at `256,256,64` | +0.806 | protected; full-24 now 18.803 |
| next | 3–7 | 14.897 | +0.650 | next protect |
| last | 8–12 | 14.767 | +0.520 | smallest isolated hit |

The next extra bits go on **layers 3–7**. Layers 8–12 stay at
`256,256,64` the longest. That order is the map's alone deltas. It is
not a forecast of how 3–7 will add once 13–18 and 19–23 are already at
`512,256,64`.

A follow-up full-24 that also raises 3–7 was not started. The 13–18
bump, on an isolated delta of +0.806, moved the full model by only
−0.236. The miss that remains is +3.844. Layers 3–7 are the next place
to spend bits, and one more chunk at this rung is not enough to clear
14.959 on the evidence in hand.

KLT, shared rotation, permutation, and outlier correction were not
retried. The quantizer's existing per-tensor rotation is the same code
path as the 19.039 run.

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
export PBR_CHUNK=20000
python3 eval_ppl.py ./qwen05b

PBR_STAGES=256,256,64 \
PBR_LAYER_STAGES='0:512,256,64;1:512,256,64;2:512,256,64;13:512,256,64;14:512,256,64;15:512,256,64;16:512,256,64;17:512,256,64;18:512,256,64;19:512,256,64;20:512,256,64;21:512,256,64;22:512,256,64;23:512,256,64' \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_l1318
python3 eval_ppl.py ./qwen05b_p4_l1318
```

## Logs

- `phase4_l1318_quantize.log` — 24 layers, 357,826,560 parameters,
  per-layer index 2.875 / 2.75 / 2.875, `QUANTIZE DONE rc=0`
- `phase4_l1318_ppl_baseline.log` — 14.247, 299078 tokens
- `phase4_l1318_ppl_full.log` — 18.803, 299078 tokens
- `phase4_l1318_runner.log`
- `phase4_l1318_full24.json`

## What this does not say

- No packed sub-4-bit file was written. Do not read 2.822917 bpw as an
  on-disk size. The checkpoint used for perplexity is a dequantized
  fp32 copy.
- Layers 3–12 were not given `512,256,64`. A uniform 24-layer
  `512,256,64` run was not done.
- Layers 3–7 were not raised inside this full model. That is the next
  chunk, and it was not started here.
- `256,256,256` was not run.
