# Phase 4 map — isolate layers 3–23 in four chunks

Diagnostic only. No pass/fail gate. The question is which block of
layers 3–23, quantized alone at stages `256,256,64`, moves WikiText-2
test perplexity the most.

**Hot chunk: test_d, layers 19–23.** Alone perplexity **15.447**,
Δppl **+1.200** (+8.423% vs 14.247). That is the largest of the four
isolated deltas. It is also the largest per layer (+0.240), and the
chunk has five layers, one fewer than test_c.

## Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local dir `pbr_ladder/qwen05b`.
- Quantizer: `pbr_ladder/quantize_full_model.py`. Stages
  `PBR_STAGES=256,256,64` on every selected layer. Defaults
  `PBR_SAMPLES=64`, `PBR_SEQ=512`. Seven linears per selected layer.
- Each run quantizes only its own layer list. Every other layer, the
  embeddings, the output head, and the norms stay full precision.
- Inside a chunk the pass is sequential: a later selected layer is
  calibrated on the live model after earlier selected layers in that
  same chunk have been written back.
- Eval: `pbr_ladder/eval_ppl.py`, `PBR_CPU_FP32=1`. WikiText-2 raw v1
  test via `Salesforce/wikitext`. Tokenized length 299078 on the
  baseline and on all four chunk evals.
- This run's unmodified baseline: **14.247**.
- Stack: torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1, numpy 2.4.4.
  CPU, four threads.
- Saved folders are dequantized fp32 checkpoints for perplexity. They
  are gitignored (`pbr_ladder/qwen05b_*/`). They are not a packed file
  and they are not a size win. Script-reported index cost on the
  touched tensors is ~2.75 bpw.

Each layer is 14,909,440 parameters (the same seven shapes as Phase 2).
Five-layer chunks touch 74,547,200 parameters. The six-layer chunk
touches 89,456,640.

## Results

Percent vs 14.247 is `(ppl − 14.247) / 14.247` on the printed
perplexity. Δ per layer is that chunk's Δppl divided by its layer
count.

| Chunk | Layers | n | ppl | Δppl | % vs 14.247 | Δ per layer |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (this run) | — | 0 | 14.247 | — | — | — |
| test_a | 3, 4, 5, 6, 7 | 5 | 14.897 | +0.650 | +4.562% | +0.130 |
| test_b | 8, 9, 10, 11, 12 | 5 | 14.767 | +0.520 | +3.650% | +0.104 |
| test_c | 13, 14, 15, 16, 17, 18 | 6 | 15.053 | +0.806 | +5.657% | +0.134 |
| **test_d** | **19, 20, 21, 22, 23** | **5** | **15.447** | **+1.200** | **+8.423%** | **+0.240** |

Share of the four-chunk sum (+3.176 ppl), using these isolated deltas:

| Chunk | Δppl | Share of +3.176 |
| --- | ---: | ---: |
| test_a | +0.650 | 20.47% |
| test_b | +0.520 | 16.37% |
| test_c | +0.806 | 25.38% |
| test_d | +1.200 | 37.78% |

test_b is the mild chunk. It contains layer 12, whose Phase 1
single-layer point was 14.354 (Δ +0.107). Five layers at about that
size land on +0.520. test_a and test_c sit together near +0.13 per
layer. test_d is about 1.8× test_c per layer.

For scale, the 5% line used in earlier gates is 14.959 (Δ +0.712).
test_a (+0.650) and test_b (+0.520) sit under that line by themselves.
test_c (+0.806) and test_d (+1.200) sit over it. That is a scale
note, not a gate on this map.

## Sum of chunk deltas vs the full-model gap

The four isolated deltas add to **+3.176 ppl**.

The full-model numbers already on record are a different experiment:
one sequential pass that quantizes early layers too. Comparing the sum
to those numbers mixes two effects (early-layer quantization, and
calibration of later layers on already-quantized earlier outputs).

| Envelope | What was quantized | Δppl | Chunk sum / envelope |
| --- | --- | ---: | ---: |
| Phase 2 retry, 19.280 − 14.247 | L0–2 at `512,256,64` and L3–23 at `256,256,64`, one sequential pass | +5.033 | 63.1% |
| 19.280 − 14.873 | Same early recipe on both sides. 14.873 is L0–2 only at `512,256,64`. The +4.407 is the rest of that sequential pass | +4.407 | 72.1% |
| 19.427 − 15.005 | Flat `256,256,64`. 15.005 is L0–2 only. The +4.422 is the rest of that sequential pass | +4.422 | 71.8% |
| This map | Sum of four chunks, each with every other layer full precision | +3.176 | — |

+5.033 counts layers 0–2 as well as 3–23, so it is an upper envelope,
not an L3–23 total. The two “rest of the sequential pass” rows
(+4.407 and +4.422) are the closer L3–23-ish figures, and they still
confound: in those full runs layer 3 is calibrated on quantized
layers 0–2, and each later layer sees the quantized block above it.
Each chunk here was calibrated with full-precision upstream weights.

The chunk sum is about 72% of those L3–23-ish envelopes. The leftover
is about **+1.23 ppl** (4.407 − 3.176) on the mixed recipe and
**+1.246 ppl** (4.422 − 3.176) on the flat recipe. That leftover is
the sequential interaction, not a missing fifth chunk. Layers 0–2 at
`256,256,64` were already measured at Δ +0.758 (perplexity 15.005).
Adding that isolated early delta to the chunk sum gives +3.934, still
short of the flat all-24 delta of +5.180 (19.427 − 14.247) by +1.246.

## Recommendation

Spend the next extra bits on **layers 19–23 first**. That is the hot
chunk on both the absolute delta and the per-layer delta.

Keep stages `256,256,64` on layers 8–12 the longest. That chunk is the
smallest isolated hit. Layers 3–7 and 13–18 are the middle of this map
(about +0.13 ppl per layer). They are real damage, and they are not
the first place to add bits.

Layers 0–2 remain the early-block story from Phase 2b and Phase 4.
Their isolated per-layer hit at `256,256,64` was about +0.253, in the
same range as test_d's +0.240, and the `512,256,64` bump on 0–2 already
cleared 14.959 for that block alone (14.873). The full model stayed at
19.280. This map says the leftover is concentrated at the end of the
stack, not spread evenly through layers 3–23.

Closing one chunk will not reproduce the full-model gap. The four
isolated deltas sum to +3.176, and the sequential L3–23-ish envelopes
are about +4.4. A bit bump on layers 19–23 is the first L3–23 spend,
scored later with the same WikiText-2 test. It was not run here.

## Commands

From `pbr_ladder/`:

```bash
export PBR_CPU_FP32=1
python3 eval_ppl.py ./qwen05b

PBR_LAYERS=3,4,5,6,7 PBR_STAGES=256,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_test_a
python3 eval_ppl.py ./qwen05b_iso_test_a

PBR_LAYERS=8,9,10,11,12 PBR_STAGES=256,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_test_b
python3 eval_ppl.py ./qwen05b_iso_test_b

PBR_LAYERS=13,14,15,16,17,18 PBR_STAGES=256,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_test_c
python3 eval_ppl.py ./qwen05b_iso_test_c

PBR_LAYERS=19,20,21,22,23 PBR_STAGES=256,256,64 \
  python3 quantize_full_model.py ./qwen05b ./qwen05b_iso_test_d
python3 eval_ppl.py ./qwen05b_iso_test_d
```

## Logs

- `phase4map_ppl_baseline.log` — 14.247
- `phase4map_quantize_test_a.log`, `phase4map_ppl_test_a.log` — 14.897
- `phase4map_quantize_test_b.log`, `phase4map_ppl_test_b.log` — 14.767
- `phase4map_quantize_test_c.log`, `phase4map_ppl_test_c.log` — 15.053
- `phase4map_quantize_test_d.log`, `phase4map_ppl_test_d.log` — 15.447
- `phase4map_runner.log`
- `phase4_l3_23_chunk_map.json`

## What this does not say

These four perplexities are dequantized fp32 checkpoints. There is no
packed sub-4-bit file and no size win. The map does not re-score KLT,
shared rotation, per-channel scale, permutation, or outlier correction.
It does not re-run the 24-layer model.
