# Phase 4 — early-block bit probe (layers 0–2)

First probe after Phase 2b. Layers 0–2 at stages `256,256,64` measured
WikiText-2 test perplexity **15.005** (+5.320% vs 14.247), 0.046 over the
5% cap of **14.959**. This run raises bits on that block only. Layers 3–23,
embeddings, the output head, and norms stay full precision. No 24-layer run.

## Verdict

**PASS.** Layers 0, 1, and 2 together at stages `512,256,64` print
**14.873**, which is ≤ 14.959.

The smaller-rate variant (layer 0 at `512,256,64`, layers 1 and 2 still
`256,256,64`) also prints **14.951** ≤ 14.959. `256,256,256` and a fourth
residual stage were not run.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per selected layer.
- `PBR_LAYER_STAGES` (new, optional) overrides `PBR_STAGES` for named
  layers inside one sequential pass. Unset, every selected layer uses
  `PBR_STAGES`, same as Phase 1/2.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1 test
  via `Salesforce/wikitext`. Tokenized length 299078 on every run here.
- This run's unmodified baseline: **14.247**.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1, numpy 2.4.4.
- Saved folders are dequantized fp32 checkpoints for perplexity. They are
  gitignored (`pbr_ladder/qwen05b_*/`). They are not a packed file and
  they are not a size win.

Layer-0 weight shapes, read from `qwen05b/model.safetensors` with
torch (bf16): `q_proj` 896×896, `k_proj` 128×896, `v_proj` 128×896,
`o_proj` 896×896, `gate_proj` 4864×896, `up_proj` 4864×896,
`down_proj` 896×4864. Same seven shapes on layers 1 and 2 (Qwen2.5-0.5B
blocks are uniform). The quantizer reports 14,909,440 parameters per
layer and 44,728,320 for the three-layer runs.

## Recipe

`512,256,64` is the same residual path as Phase 2 (8-D nodes, one pooled
k-means per stage, column-block GPTQ feedback). The first codebook grows
from 256 to 512. That is one extra index bit on the first stage.

Index cost is `sum(log2(k)) / 8`:

| Stages | Index bits | Index bpw | vs `256,256,64` |
| --- | ---: | ---: | ---: |
| `256,256,64` | 8+8+6 = 22 | 2.75 | — |
| `512,256,64` | 9+8+6 = 23 | 2.875 | +0.125 |

`256,256,256` would be 3.00 index bpw (+0.25). A fourth stage of 16
(`256,256,64,16`) would be 3.25. Those were the next rungs if 2.875
missed 14.959. It did not miss.

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247` on the printed perplexity.
Gate is the printed value ≤ 14.959.

| Run | Layers 0 / 1 / 2 stages | ppl | delta | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — | — |
| L0–L2, Phase 2 dry run | `256,256,64` on all three | 15.005 | +0.758 | +5.320% | FAIL |
| L0 bumped, L1+L2 default | `512,256,64` / `256,256,64` / `256,256,64` | 14.951 | +0.704 | +4.941% | **PASS** |
| L0–L2 bumped | `512,256,64` on all three | 14.873 | +0.626 | +4.394% | **PASS** |

The Phase 2 dry-run row is the existing log, not a second measurement.
The other three rows are this run.

Both new points clear the probe gate. The layer-0-only bump is the
smaller rate change (+0.125 index bpw on one layer, early-block index
average 2.7917). The uniform bump spends +0.125 index bpw on every early
layer (2.875) and prints a lower perplexity (14.873 vs 14.951).

## Estimated bpw (not a file size)

Index bpw is what `quantize_full_model.py` prints. Codebook side info is
the `push_below3.py` estimate, not a packed checkpoint:

```
side = (sum(k) * 8 * 16 + out_features * 16) / n_params
raw  = sum(log2(k)) / 8 + side
```

`16` is fp16 bits for each codebook coordinate and for one scale per
output row. Wide tensors sit under +0.1 bpw of side info. The narrow
`k_proj` / `v_proj` (128×896) do not: the same codebooks are large next
to 114,688 parameters. Phase 1 already reported that
(`[256,256,64]` k/v raw ≈ 3.411).

| Tensor | params | `256,256,64` index | + side | `512,256,64` index | + side |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_proj 896×896 | 802,816 | 2.750 | 2.860 | 2.875 | 3.026 |
| k_proj 128×896 | 114,688 | 2.750 | 3.411 | 2.875 | 3.821 |
| v_proj 128×896 | 114,688 | 2.750 | 3.411 | 2.875 | 3.821 |
| o_proj 896×896 | 802,816 | 2.750 | 2.860 | 2.875 | 3.026 |
| gate_proj 4864×896 | 4,358,144 | 2.750 | 2.785 | 2.875 | 2.917 |
| up_proj 4864×896 | 4,358,144 | 2.750 | 2.785 | 2.875 | 2.917 |
| down_proj 896×4864 | 4,358,144 | 2.750 | 2.770 | 2.875 | 2.903 |
| one layer, parameter-weighted | 14,909,440 | 2.750 | 2.798 | 2.875 | 2.939 |

Early block (three equal layers, 44,728,320 parameters):

| Recipe | Index bpw | Index + side (weighted) |
| --- | ---: | ---: |
| All three `256,256,64` (Phase 2) | 2.750 | 2.798 |
| L0 `512,256,64`, L1 and L2 `256,256,64` | 2.7917 | 2.845 |
| All three `512,256,64` | 2.875 | 2.939 |

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
python3 eval_ppl.py ./qwen05b

PBR_LAYERS=0,1,2 PBR_STAGES=512,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_L012_s512
python3 eval_ppl.py ./qwen05b_p4_L012_s512

PBR_LAYERS=0,1,2 PBR_STAGES=256,256,64 PBR_LAYER_STAGES=0:512,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_p4_L0hi_L12def
python3 eval_ppl.py ./qwen05b_p4_L0hi_L12def
```

## Logs

- `phase4_ppl_baseline.log` — 14.247
- `phase4_quantize_L012_s512.log`, `phase4_ppl_L012_s512.log` — 14.873
- `phase4_quantize_L0hi_L12def.log`, `phase4_ppl_L0hi_L12def.log` — 14.951
- `phase4_runner.log` — the shape dump in that log dies on a numpy
  bf16 read (`data type 'bfloat16' not understood`). Quantize and eval
  still ran. Shapes above are from a later torch read of the same file.
- `phase4_early_block_bits.json`

## What this does not say

- The 24-layer model was not rerun. Phase 2 at `256,256,64` on all 24
  layers was 19.427. Clearing 14.959 on layers 0–2 does not move that
  number.
- Layer 3 was not quantized.
- No packed sub-4-bit file was written. Do not read the bpw table as an
  on-disk size.
