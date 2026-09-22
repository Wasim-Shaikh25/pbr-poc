# Phase 2 retry — full 24 layers, mixed early-block bits

Whole-model rerun after Phase 4. Layers 0–2 use the stages that cleared
the probe gate on that block alone (`512,256,64` → 14.873). Layers 3–23
stay at the Phase 2 stages `256,256,64`. Embeddings, the output head, and
norms stay full precision.

## Verdict

**FAIL.** WikiText-2 test perplexity is **19.280**. The gate is the
printed value ≤ **14.959** (+5% vs **14.247**). 19.280 is 4.321 over
that gate.

This run's unmodified baseline, same `eval_ppl.py` and the same 299078
tokens, is **14.247** again.

Phase 2's flat all-24 result at `256,256,64` on every layer was
**19.427**. This mixed recipe prints 0.147 lower than that (19.280).
That is a small move next to the miss: 19.280 is still much closer to
19.427 than to 14.959. Clearing 14.959 on layers 0–2 while the other
layers stayed full precision (Phase 4, 14.873) does not carry over once
layers 3–23 are quantized.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per layer.
  `PBR_STAGES=256,256,64` with
  `PBR_LAYER_STAGES=0:512,256,64;1:512,256,64;2:512,256,64`.
- `PBR_CHUNK=20000`. The chunk only splits the pooled k-means distance
  matrix. Each row's argmin is the same computation as the default
  `50000`. No partial checkpoint from the earlier killed attempt was
  on this machine, so this was a full pass. The script wrote
  `layer_npz/layer_<i>.npz` and `pbr_progress.json` after every layer
  (all 24 finished; those files were not needed to resume).
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length 299078 on the
  baseline and on the quantized checkpoint.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1,
  numpy 2.4.4.
- The saved folder is a dequantized fp32 checkpoint for perplexity
  (`model.safetensors` is 1.9 GB; the original bf16 file is 943 MB).
  It is gitignored (`pbr_ladder/qwen05b_*/`). It is not a packed file
  and it is not a size win.

The quantizer reports 357,826,560 parameters touched (14,909,440 per
layer). Layer 23 finished at 2100.4s of quantizer time. Wall clock for
the quantize process was 01:23:36Z–01:58:47Z.

## Recipe and estimated bpw

Index cost is `sum(log2(k)) / 8`, which is what the script prints:

| Layers | Stages | Index bits | Index bpw |
| --- | --- | ---: | ---: |
| 0, 1, 2 | `512,256,64` | 9+8+6 = 23 | 2.875 |
| 3–23 | `256,256,64` | 8+8+6 = 22 | 2.75 |

Every layer has the same 14,909,440 linear parameters, so the average
index cost on the 357,826,560 touched parameters is

`(3 × 2.875 + 21 × 2.75) / 24 = 2.765625` bpw.

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

| Recipe | Layers | Index bpw | Index + side |
| --- | --- | ---: | ---: |
| Phase 2 flat | 0–23 at `256,256,64` | 2.75 | 2.798 |
| This retry | 0–2 at `512,256,64`, 3–23 at `256,256,64` | 2.765625 | 2.816 |

The mixed average is +0.015625 index bpw versus the flat Phase 2
recipe. That is the whole rate change.

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247` on the printed
perplexity. Gate is the printed value ≤ 14.959.

| Run | What was quantized | ppl | delta | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline (this run) | nothing | 14.247 | — | — | — |
| Phase 2, flat all-24 | 0–23 at `256,256,64` | 19.427 | +5.180 | +36.359% | FAIL |
| Phase 4, early block only | 0–2 at `512,256,64`; 3–23 full precision | 14.873 | +0.626 | +4.394% | PASS |
| This retry, full 24 mixed | 0–2 at `512,256,64`; 3–23 at `256,256,64` | 19.280 | +5.033 | +35.327% | **FAIL** |

The Phase 2 and Phase 4 rows are those runs' logs, not a second
measurement. The baseline and the mixed row are this run.

19.280 − 19.427 = −0.147 perplexity versus the flat 24-layer recipe.
19.280 − 14.959 = +4.321 versus the gate. The early-block bump that
was enough for layers 0–2 in isolation does not pull the quantized
rest of the network under 14.959.

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
export PBR_CHUNK=20000
python3 eval_ppl.py ./qwen05b

PBR_STAGES=256,256,64 \
PBR_LAYER_STAGES='0:512,256,64;1:512,256,64;2:512,256,64' \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p2r_mixed
python3 eval_ppl.py ./qwen05b_p2r_mixed
```

## Logs

- `phase2_retry_quantize.log` — 24 layers, 357,826,560 parameters,
  per-layer index 2.875 / 2.75, `QUANTIZE DONE rc=0`
- `phase2_retry_ppl_full.log` — 19.280, 299078 tokens
- `phase2_retry_ppl_baseline.log` — 14.247, 299078 tokens
- `phase2_retry_full24_mixed.json`

## What this does not say

- No packed sub-4-bit file was written. Do not read 2.765625 bpw as an
  on-disk size. The checkpoint used for perplexity is larger than the
  original bf16 file.
- Layers 3–23 were not given `512,256,64`. A uniform 24-layer
  `512,256,64` run was not done.
- Layer 3 was not measured on its own.
- `256,256,256` was not run.
