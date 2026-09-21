# Phase 2b — isolate layers 0, 1, and 2

Diagnostic split of the Phase 2 dry run (layers 0–2 together, WikiText-2
test perplexity 15.005, +5.320% vs 14.247). No pass/fail gate.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`, defaults
  `PBR_STAGES=256,256,64`, `PBR_SAMPLES=64`, `PBR_SEQ=512`.
- Seven linears per selected layer: `q/k/v/o_proj`, `gate/up/down_proj`.
- Multi-layer runs are sequential: each later layer is calibrated on the
  live model after earlier selected layers have been written back.
- Eval: `pbr_ladder/eval_ppl.py` with `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext` (the legacy `wikitext` id is rejected by
  current `huggingface_hub`; the script falls through). Tokenized length
  299078, the same join as the 14.247 baseline.
- Saved checkpoints are dequantized fp32 folders for perplexity. They are
  gitignored (`pbr_ladder/qwen05b_*/`) and are not a packed sub-4-bit file.
- This run's unmodified baseline: **14.247**.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1, numpy 2.4.4.
  CPU, `PBR_CPU_FP32=1`.

Commands (from `pbr_ladder/`):

```bash
PBR_LAYERS=0 python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_L0
PBR_LAYERS=1 python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_L1
PBR_LAYERS=2 python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_L2
PBR_LAYERS=0,1 python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_L0L1
PBR_LAYERS=1,2 python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_L1L2
PBR_CPU_FP32=1 python3 eval_ppl.py ./qwen05b
PBR_CPU_FP32=1 python3 eval_ppl.py ./qwen05b_iso_L0
# ... same eval for L1, L2, L0L1, L1L2
```

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247`.

| Run | Source | ppl | delta ppl | vs 14.247 |
| --- | --- | ---: | ---: | ---: |
| Baseline | this run | 14.247 | — | — |
| L0 only | this run | 14.528 | +0.281 | +1.972% |
| L1 only | this run | 14.445 | +0.198 | +1.390% |
| L2 only | this run | 14.467 | +0.220 | +1.544% |
| L0+L1 | this run | 14.702 | +0.455 | +3.194% |
| L1+L2 | this run | 14.674 | +0.427 | +2.997% |
| L0+L1+L2 | Phase 2 log | 15.005 | +0.758 | +5.320% |
| L12 only | Phase 1 log | 14.354 | +0.107 | +0.751% |
| All 24 | Phase 2 log | 19.427 | +5.180 | +36.359% |

Script-reported size: 14,909,440 parameters per layer, 29,818,880 per
pair, index cost ~2.75 bpw on touched tensors (codebook overhead reported
as <0.1 bpw). Embeddings, head, and norms stayed full precision.

Raw logs:

- `phase2b_quantize_L0.log`, `phase2b_ppl_L0.log`
- `phase2b_quantize_L1.log`, `phase2b_ppl_L1.log`
- `phase2b_quantize_L2.log`, `phase2b_ppl_L2.log`
- `phase2b_quantize_L0L1.log`, `phase2b_ppl_L0L1.log`
- `phase2b_quantize_L1L2.log`, `phase2b_ppl_L1L2.log`
- `phase2b_ppl_baseline.log`, `phase2b_runner.log`

## Additivity

| Comparison | Sum of isolated deltas | Measured | Measured − sum |
| --- | ---: | ---: | ---: |
| L0+L1 | +0.479 | +0.455 | −0.024 |
| L1+L2 | +0.418 | +0.427 | +0.009 |
| L0+L1+L2 | +0.699 | +0.758 | +0.059 |

The triple is a bit worse than the sum of the three single-layer runs.
That extra +0.059 ppl is 7.8% of the +0.758 ppl triple. The pairs are
within 0.03 ppl of the sum of their singles.

A single-layer perplexity is the damage when that layer is quantized and
everything upstream is still full precision. It is a separate measurement
from that layer's marginal effect inside the sequential triple.

Share of the triple's +0.758 ppl, using isolated deltas as the accounting
(these shares are not a causal split of the sequential run):

| Piece | Isolated delta | Share of +0.758 |
| --- | ---: | ---: |
| L0 | +0.281 | 37.1% |
| L1 | +0.198 | 26.1% |
| L2 | +0.220 | 29.0% |
| Sequential extra | +0.059 | 7.8% |

## Phase 4 recommendation

Layers 0, 1, and 2 are the same kind of hit (+1.390% to +1.972%). Layer 0
is the largest. The +5.320% dry run is those three hits stacked.

Phase 4 should raise bits on the early block — layers 0, 1, and 2, and
layer 3 with them, because layer 3 was not isolated in this run — and
keep stages `256,256,64` on the middle. The middle evidence we have is
layer 12 at +0.751%.

If only one early layer can move off ~3-bit, move layer 0 first. One
layer left at 16-bit instead of ~2.75 bpw changes the index-cost average
across the 24 equal-sized linear blocks from 2.75 bpw to ~3.30 bpw. Two
layers at 16-bit → ~3.85 bpw. Three → ~4.41 bpw. Leaving all three of
L0–L2 at 16-bit spends more rate than this dry run needs: 15.005 is 0.046
ppl over the 5% cap of 14.959, and the measured L1+L2-only point is
already 14.674. The next experiment is a modest bit increase on layers
0–2, on this same WikiText-2 test. It was not run here.

These numbers are per-configuration perplexities of dequantized fp32
checkpoints. They are not a whole-model compression result.
