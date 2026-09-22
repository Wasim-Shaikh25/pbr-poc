# PBR-Ladder — script guide run, now including real Qwen2.5-0.5B-Instruct results

**Status as of 2026-09-22:** Phase 4 iterate stopped after PR #33 (full-24
PPL **18.492**, gate **14.959**). L8–12 all-bump cancelled — no result.
Operational handoff for the next session: `docs/pbr_ladder/GROK_HANDOFF.md`.

Source: user-supplied `PBR_Ladder_Script_Guide.docx`, all 9 scripts it
describes (`push_below4.py`, `push_below3.py`, `push_nested.py`,
`ladder_exact.py`, `capture_activations.py`, `eval_ppl.py`,
`formula_probe.py`, `node_combo_q4.py`, `node_structures.py` — the last 3
were supplied later, see `docs/pbr_ladder/HANDOFF.md`).

Real-Qwen numbers below were produced on a local machine with
`huggingface.co` access (not the cloud sandbox — see HANDOFF.md for why),
running Qwen/Qwen2.5-0.5B-Instruct on CPU. Raw console output:
`pbr_ladder/real_tensor_results.txt` (push_below4/push_nested/push_below3/
ladder_exact on 9 real tensors) and `pbr_ladder/eval_ppl_results.txt`
(baseline vs. single-layer-swapped perplexity).

Note: the model download, activation captures (`.npy`, ~300MB total),
cloned model directory (`qwen05b/`, ~1GB), and the saved quantized layer
(`.npz`, 17MB) are **not committed** — they're reproducible via the
"To actually get real-Qwen numbers" steps below and are gitignored to keep
the repo small. Only the code and the raw text logs are committed.

## What ran

Only the synthetic-surrogate self-check (guide §2.3), which needs nothing
but `numpy`:

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

This confirms the scripts run correctly and are seeded/deterministic. **It
is not a Qwen result** — same caveat the guide itself states in its status
line: "every result in this guide comes from a synthetic Qwen-like tensor."

## Real-Qwen2.5-0.5B-Instruct results (guide §5.1–§5.2)

9 tensors tested: `mlp.gate_proj`, `mlp.down_proj`, `self_attn.q_proj` ×
layers 2, 12, 21. Calibration activations captured from the WikiText-2
train split (16384 tokens, 4096-row subsample per tensor) via
`capture_activations.py`. Perplexity measured on the full WikiText-2 test
split.

Note on precision: the guide's scripts default to `torch_dtype=bfloat16`
for the real-model calls. On this CPU (no native bf16 acceleration),
bf16 matmuls measured **~240x slower** than fp32 (benchmarked directly:
20 iterations of a 512×896×4864 matmul, 0.32s fp32 vs 75.5s bf16) — a
single activation capture that should take ~2 min was on track to take
over an hour. `capture_activations.py` and `eval_ppl.py` were given a
`PBR_CPU_FP32=1` env var to opt into fp32 on CPU; this is a performance
fix only, not a change to the quantization methods being measured. The
two L12 tensors captured before this fix (`acts_L12_gate.npy`,
`acts_L12_down.npy`) used bf16 and are still valid (bf16 only adds minor
precision loss to the captured activations, it doesn't invalidate them).

### §5.1 tensor-quality gate — beat INT4 RTN output dB by ≥2dB at ≤3.3bpw

**FAIL on most tensors.** Comparing the 3-stage residual-VQ result
(`push_below3.py`, `[256,256,256]`, ~3.0–3.14 bpw) against INT4 RTN
(4.25 bpw) output dB, per tensor:

| Tensor | nodes bpw | nodes dB | INT4 RTN dB | Δ | Gate (≥2dB) |
| --- | ---: | ---: | ---: | ---: | --- |
| L12 gate_proj | 3.040 | 23.30 | 23.12 | +0.18 | FAIL |
| L12 down_proj | 3.026 | 21.28 | 19.42 | +1.86 | FAIL |
| L12 q_proj | 3.140 | 22.49 | 21.11 | +1.38 | FAIL |
| L2 gate_proj | 3.040 | 21.03 | 22.43 | -1.40 | FAIL |
| L2 down_proj | 3.026 | 34.48 | 17.89 | +16.59 | **PASS** |
| L2 q_proj | 3.140 | 22.62 | 21.05 | +1.57 | FAIL |
| L21 gate_proj | 3.040 | 23.87 | 23.35 | +0.52 | FAIL |
| L21 down_proj | 3.026 | 34.53 | 22.56 | +11.97 | **PASS** |
| L21 q_proj | 3.140 | 21.51 | 17.33 | +4.18 | **PASS** |

Only 3/9 tensors clear the ≥2dB bar, all `down_proj`. `gate_proj` and
`q_proj` are marginal (+0.18 to +1.57) or negative (L2 gate_proj, -1.40).
Full method sweep (INT3/INT2 RTN/GPTQ/rot+GPTQ, all `push_below4.py`/
`push_nested.py`/`push_below3.py`/`ladder_exact.py` output) for all 9
tensors: `pbr_ladder/real_tensor_results.txt`.

### Phase 0 — were those 9 tensors measured on real activations?

**PASS.** The roadmap flagged the high `down_proj` scores at layers 2 and
21 as a possible synthetic-activation fallback (a `push_*.py` run with no
third `.npy` argument). They were re-captured and re-measured.

Procedure: WikiText-2 raw v1 train, first 16384 tokens, chunks of 512,
seed-0 subsample of 4096 rows, model in fp32 (`PBR_CPU_FP32=1`). One
batched capture (same hook and subsample as `capture_activations.py`)
was checked against a separate `capture_activations.py` run on
`L12 self_attn.q_proj`: the `.npy` files are bit-identical. Raw log:
`artifacts/pbr_ladder/capture_phase0.log`.

`push_below4.py` was then re-run with the third argument set, on all 9
tensors (`PBR_CHUNK=50000`). Every tensor whose prior `push_below4` block
completed matches the logged table to the printed precision (bpw / weight
dB / output dB). `L21 mlp.down_proj` had no prior `push_below4` table —
that run died in `kmeans` with `Unable to allocate 1.04 GiB` before the
chunked-distance fix. It now completes. Its `INT3 rot+GPTQ` output dB
(33.69) matches the reference line already printed by the prior
`push_nested` block on that tensor, so that prior block was on the same
activations.

`push_below3.py` `[256, 256, 256]` vs INT4 RTN was re-run on the same 9
real activation files. Every gate-table cell above reproduced exactly,
including the two high `down_proj` scores:

| Tensor | re-measured nodes dB | re-measured INT4 RTN dB | vs logged table |
| --- | ---: | ---: | --- |
| L2 down_proj | 34.48 | 17.89 | match |
| L21 down_proj | 34.53 | 22.56 | match |
| other 7 tensors | (see §5.1) | (see §5.1) | match |

The high layer-2 / layer-21 `down_proj` dB numbers are real captured
activations, not the synthetic fallback. They stay in the decision table.
`L21 down_proj` `push_below4` (previously crashed), real acts:

| method | bpw | weight dB | output dB |
| --- | ---: | ---: | ---: |
| INT3 RTN | 3.250 | 12.94 | 15.03 |
| INT3 GPTQ | 3.250 | 8.65 | 29.03 |
| INT3 rot+GPTQ | 3.250 | 11.43 | 33.69 |
| 3x256 8-D nodes, rot | 3.026 | 14.54 | 17.73 |
| INT2 RTN | 2.250 | 5.39 | 6.94 |
| INT2 GPTQ | 2.250 | -0.34 | 20.62 |
| INT2 rot+GPTQ | 2.250 | 3.51 | 25.38 |
| 2x256 8-D nodes, rot | 2.018 | 9.81 | 12.56 |

Raw logs: `artifacts/pbr_ladder/phase0_push_below4.log`. The 3x256 node
row above is residual VQ without the column-feedback pass; the 34.53 dB
gate-table number is the feedback encode from `push_below3`, same
activations.

### §5.2 single-layer perplexity gate — swap changes perplexity by <0.5%

**PASS.** Layer tested: `model.layers.12.mlp.gate_proj.weight`, saved at
2.785 bpw (`[256,256,64]` stage config, via `push_below3.py`'s
`PBR_SAVE`).

| Run | WikiText-2 test perplexity |
| --- | ---: |
| Baseline (unmodified Qwen2.5-0.5B-Instruct) | 14.247 |
| L12 gate_proj swapped to 2.785 bpw | 14.269 |

Relative change: **+0.154%**, under the 0.5% gate — despite this
particular config (2.785 bpw) sitting *below* INT4 RTN's output dB on
this tensor (21.96 vs 23.12 dB, see `[256,256,64]` row in
`real_tensor_results.txt`). Note this tests one layer at one bpw point;
it is not a whole-model claim and does not establish where the
perplexity gate would actually break for this or other layers — see
"What wasn't tested" below. Raw output: `pbr_ladder/eval_ppl_results.txt`.

### Phase 1 — entire layer 12

Remaining layer-12 matrices (`up_proj`, `k_proj`, `v_proj`, `o_proj`) were
measured with the same real-activation procedure. `up_proj` shares its
input with `gate_proj`, and `k_proj`/`v_proj` share theirs with `q_proj`
(captured arrays are bit-identical). Raw log:
`artifacts/pbr_ladder/phase1_L12_tensors.log`.

`push_below3` `[256,256,256]` vs INT4 RTN, same ≥2 dB at ≤3.3 bpw bar as
§5.1. `k_proj` and `v_proj` are narrow (128×896), so the codebook side
info pushes `[256,256,256]` to 3.875 bpw, over the 3.3 bpw cap.

| Tensor | nodes bpw | nodes dB | INT4 RTN dB | Δ | Gate (≥2dB @ ≤3.3bpw) |
| --- | ---: | ---: | ---: | ---: | --- |
| L12 up_proj | 3.040 | 18.78 | 18.45 | +0.33 | FAIL |
| L12 k_proj | 3.875 | 25.44 | 23.92 | +1.52 | FAIL |
| L12 v_proj | 3.875 | 21.06 | 19.08 | +1.98 | FAIL |
| L12 o_proj | 3.140 | 19.73 | 18.26 | +1.47 | FAIL |

**Whole layer 12 perplexity: PASS.** All seven attention/MLP linears in
layer 12 were replaced together by `quantize_full_model.py`
(`PBR_LAYERS=12`, stages `256,256,64`, sequential GPTQ-style feedback).
The script reports ~2.75 bpw index cost on 14,909,440 parameters;
codebook side info is under 0.1 bpw on average and about 0.66 bpw on the
two narrow `k`/`v` tensors (`[256,256,64]` row: 3.411 bpw). The saved
folder is a **dequantized fp32 checkpoint for perplexity**, not a smaller
file. This run's baseline matches the logged 14.247 (same WikiText-2
test join, 299078 tokens, `PBR_CPU_FP32=1`).

| Run | WikiText-2 test perplexity |
| --- | ---: |
| Baseline (this run, unmodified) | 14.247 |
| Layer 12, all 7 linears, stages 256,256,64 | 14.354 |

Relative change: **+0.751%** `(14.354 − 14.247) / 14.247`, under the 1%
Phase 1 gate. Every one of the seven matrices still fails the §5.1 dB
bar; the perplexity gate and the dB gate are not the same evidence.
Logs: `artifacts/pbr_ladder/phase1_quantize_L12.log`,
`artifacts/pbr_ladder/phase1_ppl_baseline.log`,
`artifacts/pbr_ladder/phase1_ppl_L12.log`.

### Phase 2 — all 24 layers

**FAIL.** Phase 1 passed, so the driver was run. Same stages
`256,256,64`, embeddings / output head / norms left full precision.
`quantize_full_model.py` again writes a dequantized fp32 checkpoint
(1.9 GB on disk vs 943 MB for the original bf16 file). That file is for
perplexity only.

Dry run, layers 0–2 only (the other 21 layers stay full precision):

| Run | WikiText-2 test perplexity | vs 14.247 |
| --- | ---: | ---: |
| Layers 0, 1, 2 | 15.005 | +5.320% |

Full model, all 24 layers, 357,826,560 parameters touched, script-reported
index cost ~2.75 bpw:

| Run | WikiText-2 test perplexity | vs 14.247 |
| --- | ---: | ---: |
| Baseline | 14.247 | — |
| All 24 layers | 19.427 | +36.359% |

The 5% gate is perplexity ≤ 14.959. 19.427 misses it. Three early layers
alone are already at +5.320%, and layer 12 alone was +0.751%, so the
per-layer losses add up. This does not support a whole-model claim.
Logs: `artifacts/pbr_ladder/phase2_quantize_L012.log`,
`artifacts/pbr_ladder/phase2_ppl_L012.log`,
`artifacts/pbr_ladder/phase2_quantize_full.log`,
`artifacts/pbr_ladder/phase2_ppl_full.log`.

### Phase 2b — isolate layers 0, 1, and 2

Diagnostic only. There is no pass/fail gate on this split. Same
`quantize_full_model.py` recipe as Phase 1/2: stages `256,256,64`, the
seven attention/MLP linears, sequential GPTQ-style feedback when more
than one layer is selected. Every other layer, the embeddings, the output
head, and the norms stay full precision. The saved folders are
dequantized fp32 checkpoints used for perplexity. This run's unmodified
baseline is again **14.247** (WikiText-2 raw v1 test, 299078 tokens,
`PBR_CPU_FP32=1`).

| Run | Layers quantized | WikiText-2 test ppl | vs 14.247 |
| --- | --- | ---: | ---: |
| Baseline (this run) | — | 14.247 | — |
| L0 only | 0 | 14.528 | +1.972% |
| L1 only | 1 | 14.445 | +1.390% |
| L2 only | 2 | 14.467 | +1.544% |
| L0+L1 | 0, 1 | 14.702 | +3.194% |
| L1+L2 | 1, 2 | 14.674 | +2.997% |
| L0+L1+L2 (Phase 2 dry run) | 0, 1, 2 | 15.005 | +5.320% |
| L12 only (Phase 1) | 12 | 14.354 | +0.751% |
| All 24 (Phase 2) | 0–23 | 19.427 | +36.359% |

L0, L1, and L2 rows and the baseline are this run. The L0–2, L12, and
all-24 rows are the Phase 1/2 logs, repeated here for the comparison.
Each single layer touches 14,909,440 parameters (script-reported ~2.75
bpw index cost). Each pair touches 29,818,880.

Layers 0, 1, and 2 share the +5.320%. Isolated deltas are +0.281 ppl
(L0), +0.198 ppl (L1), and +0.220 ppl (L2). L0 is the largest of the
three, about 1.4× L1. The sum of those three isolated deltas is +0.699
ppl. The sequential L0–2 run is +0.758 ppl. The remaining +0.059 ppl is
the sequential interaction: a later layer in a multi-layer run is
calibrated on already-quantized earlier outputs, while a single-layer
run sees full-precision upstream activations. That interaction is about
8% of the triple's damage.

The pairs sit on the same line. L0+L1 measured +0.455 ppl against +0.479
from adding the two isolated deltas. L1+L2 measured +0.427 ppl against
+0.418. Layer 12 alone is +0.107 ppl (+0.751%), smaller than any of
L0, L1, or L2.

**Phase 4 targeting.** Put the extra bits on the early block: layers 0,
1, and 2, with layer 3 in that same group (layer 3 was not isolated
here). Inside that block, layer 0 is the first place to add bits. Keep
stages `256,256,64` on the middle, where layer 12 already measured
+0.751%.

Index-cost illustration on the 357,826,560 linear parameters, if an early
layer stayed at 16-bit instead of the ~2.75 bpw index cost: one layer
moves the block average to ~3.30 bpw, two layers to ~3.85 bpw, three
layers to ~4.41 bpw. One layer is a small move on a 0.5B model. Three
layers left at 16-bit is a large move next to a dry run that sits 0.046
ppl over the 5% cap of 14.959 (the measured L1+L2-only point is already
14.674). The proportionate next experiment is a modest bit increase on
layers 0–2, scored with the same WikiText-2 test perplexity. That probe
is Phase 4 below.

Write-up and raw logs: `artifacts/pbr_ladder/phase2b_l0_l1_l2_isolation.md`,
`artifacts/pbr_ladder/phase2b_l0_l1_l2_isolation.json`,
`artifacts/pbr_ladder/phase2b_quantize_*.log`,
`artifacts/pbr_ladder/phase2b_ppl_*.log`.

### Phase 4 — modest bit increase on layers 0–2

**PASS** against the probe gate (perplexity ≤ 14.959). Same model, same
`eval_ppl.py` (`PBR_CPU_FP32=1`, WikiText-2 raw v1 test, 299078 tokens).
This run's baseline is again **14.247**. Layers 3–23 stay full precision.
No 24-layer run. Checkpoints are still dequantized fp32.

The bump is stages `512,256,64` instead of `256,256,64`: the first 8-D
codebook grows from 256 to 512. Index cost goes from 2.75 bpw to 2.875
bpw (`(9+8+6)/8`). Same rotation, pooled k-means, and column-block
feedback. `256,256,256` (3.00 index bpw) was the next rung and was not
run, because 2.875 already cleared 14.959.

| Run | Stages on layers 0, 1, 2 | WikiText-2 test ppl | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — |
| Phase 2 dry run | `256,256,64` on all three | 15.005 | +5.320% | FAIL |
| L0 bumped, L1 and L2 default | `512,256,64` / `256,256,64` / `256,256,64` | 14.951 | +4.941% | **PASS** |
| L0–L2 bumped | `512,256,64` on all three | 14.873 | +4.394% | **PASS** |

The Phase 2 row is the earlier log. The other rows are this run.
Percent is `(ppl − 14.247) / 14.247`.

Estimated rate on one early layer (index, and index plus the
`push_below3.py` fp16 codebook/scale side info), parameter-weighted
across the seven linears: **2.750 / 2.798** at `256,256,64`, **2.875 /
2.939** at `512,256,64`. The three-layer mix (only layer 0 bumped)
averages **2.7917 / 2.845**. Narrow `k`/`v` stay well above that average
(3.411 → 3.821 including side info). This is not an on-disk size.

`quantize_full_model.py` takes optional `PBR_LAYER_STAGES`
(`0:512,256,64`) so one sequential pass can mix recipes. Unset, behavior
matches Phase 1/2.

Write-up, per-tensor bpw table, and commands:
`artifacts/pbr_ladder/phase4_early_block_bits.md`,
`artifacts/pbr_ladder/phase4_early_block_bits.json`. Logs:
`phase4_quantize_*.log`, `phase4_ppl_*.log`.

### Phase 2 retry — full 24 layers at the Phase 4 mix

**FAIL** against the same 5% gate (perplexity ≤ 14.959). Layers 0–2 at
`512,256,64`, layers 3–23 at `256,256,64`. Same `quantize_full_model.py`
and `eval_ppl.py` (`PBR_CPU_FP32=1`, WikiText-2 raw v1 test, 299078
tokens). This run's baseline is again **14.247**. The checkpoint is
dequantized fp32 (1.9 GB) for perplexity, larger than the original
943 MB bf16 file, not a packed file and not a size win.

`PBR_CHUNK=20000` only splits the pooled distance matrix; the per-row
argmin matches the default chunk. No earlier partial checkpoint was
available, so this was a full pass. The script can resume from
`pbr_progress.json` after a later kill; this run finished all 24 layers
without needing that.

| Run | Recipe | WikiText-2 test ppl | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — |
| Phase 2 flat all-24 (prior log) | `256,256,64` on 0–23 | 19.427 | +36.359% | FAIL |
| Phase 4 early block only (prior log) | `512,256,64` on 0–2; 3–23 full precision | 14.873 | +4.394% | PASS |
| This retry | `512,256,64` on 0–2; `256,256,64` on 3–23 | 19.280 | +35.327% | **FAIL** |

357,826,560 parameters touched. Average index cost is **2.765625** bpw
(`(3 × 2.875 + 21 × 2.75) / 24`). Adding the `push_below3.py` fp16
codebook/scale side info puts the same average at **2.816**. Phase 2
flat was 2.75 index / 2.798 with side info. The perplexity gap versus
that flat run is −0.147 (19.280 vs 19.427). The gap versus the gate is
+4.321. The early-block bump does not bring the full model under 14.959.

Write-up: `artifacts/pbr_ladder/phase2_retry_full24_mixed.md` and
`.json`. Logs: `phase2_retry_quantize.log`,
`phase2_retry_ppl_baseline.log`, `phase2_retry_ppl_full.log`.

### Phase 4 map — layers 3–23 in four chunks

Diagnostic only. No pass/fail gate. Same `quantize_full_model.py`
recipe as the Phase 2 default: stages `256,256,64` on the selected
layers, the seven attention/MLP linears, sequential feedback inside
the chunk. Every other layer stays full precision. Same `eval_ppl.py`
(`PBR_CPU_FP32=1`, WikiText-2 raw v1 test, 299078 tokens). This run's
baseline is again **14.247**. Checkpoints are dequantized fp32 for
perplexity, not a packed file and not a size win.

**Hot chunk: layers 19–23 (test_d), perplexity 15.447, Δ +1.200
(+8.423%).** That is the largest alone delta, and the largest per
layer (+0.240).

| Chunk | Layers | ppl | Δppl | vs 14.247 |
| --- | --- | ---: | ---: | ---: |
| Baseline (this run) | — | 14.247 | — | — |
| test_a | 3–7 | 14.897 | +0.650 | +4.562% |
| test_b | 8–12 | 14.767 | +0.520 | +3.650% |
| test_c | 13–18 | 15.053 | +0.806 | +5.657% |
| test_d | 19–23 | 15.447 | +1.200 | +8.423% |

The four isolated deltas sum to **+3.176**. The Phase 2 retry full
model is 19.280 (Δ +5.033 vs 14.247), which also quantizes layers 0–2,
so that gap is not an L3–23 total. Subtracting the early-block-only
point (14.873) leaves **+4.407** for the rest of that sequential pass.
The flat pair is 19.427 − 15.005 = **+4.422**. The chunk sum is 72.1%
of +4.407 and 71.8% of +4.422. The leftover ~+1.23 ppl is sequential
interaction: in the full runs, later layers are calibrated on
already-quantized earlier layers. Each chunk here saw full-precision
upstream weights.

**Where to spend the next extra bits.** Layers **22 and 23** first,
then 19–21. The four-way hot chunk is 19–23. Splitting it in order:
layers 19–21 print 14.746 (Δ +0.499, +0.166 per layer) and layers
22–23 print 14.971 (Δ +0.724, +0.362 per layer). Those two halves sum
to +1.223, against +1.200 on layers 19–23 together. Layers 8–12 are
the smallest isolated hit (+0.520, and they contain the Phase 1
layer-12 point). Layers 3–7 and 13–18 are about +0.13 ppl per layer.
A bump on 22–23 was not run here. One chunk's fix does not close the
full-model gap. This map also did not run a 24-layer model and did
not score layers 19–23 at `512,256,64`. If a separate full-24 that
protects that whole late block at `512,256,64` still misses, the next
chunk on this table is **layers 13–18** (Δ +0.806), then layers 3–7
(Δ +0.650), then layers 8–12 (Δ +0.520). That order is the four
alone deltas, not a forecast of those chunks inside an already
re-bit late block.

Write-up: `artifacts/pbr_ladder/phase4_l3_23_chunk_map.md` and `.json`.
Logs: `phase4map_quantize_test_*.log`, `phase4map_ppl_*.log`.

The full-24 that protects layers 19–23 at `512,256,64` was run next.
It misses. See the late-block section below.

### Phase 4 late-block protect — full 24 layers

**FAIL** against the same 5% gate (perplexity ≤ 14.959). Layers 0–2
and layers 19–23 at `512,256,64`, layers 3–18 at `256,256,64`. Same
`quantize_full_model.py` and `eval_ppl.py` (`PBR_CPU_FP32=1`,
WikiText-2 raw v1 test, 299078 tokens). This run's baseline is again
**14.247**. The checkpoint is dequantized fp32 for perplexity. It is
not a packed file and it is not a size win.

`PBR_CHUNK=20000` only splits the pooled distance matrix; the per-row
argmin matches the default chunk. This was a full pass. The script can
resume from `pbr_progress.json` after a later kill; this run finished
all 24 layers without needing that. Phase 4 is done only if the
whole-model perplexity is ≤ 14.959. It is not.

| Layers | Stages | Index bpw |
| --- | --- | ---: |
| 0–2 | `512,256,64` | 2.875 |
| 3–18 | `256,256,64` | 2.75 |
| 19–23 | `512,256,64` | 2.875 |

357,826,560 parameters touched. Average index cost is **2.791667** bpw
(`(8 × 2.875 + 16 × 2.75) / 24`). Adding the `push_below3.py` fp16
codebook/scale side info puts the same average at **2.845**. Phase 2
flat was 2.75 index / 2.798 with side info. The Phase 2 retry
(early-only mix) was 2.765625 / 2.816. 2.791667 stays well under ~4.
None of these bpw figures is an on-disk size.

| Run | Recipe | WikiText-2 test ppl | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — |
| Phase 2 flat all-24 (prior log) | `256,256,64` on 0–23 | 19.427 | +36.359% | FAIL |
| Phase 2 retry, early-only mix (prior log) | `512,256,64` on 0–2; `256,256,64` on 3–23 | 19.280 | +35.327% | FAIL |
| This run | `512,256,64` on 0–2 and 19–23; `256,256,64` on 3–18 | 19.039 | +33.635% | **FAIL** |

19.039 − 19.427 = −0.388 versus the flat recipe. 19.039 − 19.280 =
−0.241 versus the early-only mix, which is the late-block bump with
layers 0–2 already at `512,256,64`. 19.039 − 14.959 = +4.080 versus
the gate. Protecting the hot chunk is a partial fix.

No new mid-layer isolation was run. The Phase 4 map already has that
ranking, and it already named the fallback if this full-24 missed.
The next extra bits go on **layers 13–18** (alone 15.053, Δ +0.806),
then layers 3–7 (Δ +0.650), then layers 8–12 (Δ +0.520). That
follow-up full-24 was not started here. The late-block bump moved the
full model by −0.241, and the miss that remains is +4.080, so one more
chunk at this rung is not enough to clear the gate on the evidence in
hand. KLT, rotation, and permutation were not retried.

Write-up: `artifacts/pbr_ladder/phase4_late_block_full24.md` and
`.json`. Logs: `phase4_late_quantize.log`,
`phase4_late_ppl_baseline.log`, `phase4_late_ppl_full.log`.

The follow-up that also protects layers 13–18 was run next. It also
misses. See the section below.

### Phase 4 — protect layers 13–18, full 24 layers

**FAIL** against the same 5% gate (perplexity ≤ 14.959). Layers 0–2,
13–18, and 19–23 at `512,256,64`, layers 3–12 at `256,256,64`. Same
`quantize_full_model.py` and `eval_ppl.py` (`PBR_CPU_FP32=1`,
WikiText-2 raw v1 test, 299078 tokens). This run's baseline is again
**14.247**. The checkpoint is dequantized fp32 for perplexity (1.9 GB
vs the original 943 MB bf16 file). It is not a packed file and it is
not a size win.

`PBR_CHUNK=20000` only splits the pooled distance matrix; the per-row
argmin matches the default chunk. This was a full pass from an empty
output directory. The script can resume from `pbr_progress.json` after
a later kill; this run finished all 24 layers without needing that.
Phase 4 is done only if the whole-model perplexity is ≤ 14.959. It is
not.

| Layers | Stages | Index bpw |
| --- | --- | ---: |
| 0–2 | `512,256,64` | 2.875 |
| 3–12 | `256,256,64` | 2.75 |
| 13–18 | `512,256,64` | 2.875 |
| 19–23 | `512,256,64` | 2.875 |

357,826,560 parameters touched. Average index cost is **2.822917** bpw
(`(14 × 2.875 + 10 × 2.75) / 24`). Adding the `push_below3.py` fp16
codebook/scale side info puts the same average at **2.88025**. Phase 2
flat was 2.75 index / 2.798 with side info. The Phase 2 retry
(early-only mix) was 2.765625 / 2.816. The late-block protect was
2.791667 / 2.845. 2.822917 stays well under ~4. None of these bpw
figures is an on-disk size.

| Run | Recipe | WikiText-2 test ppl | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — |
| Phase 2 flat all-24 (prior log) | `256,256,64` on 0–23 | 19.427 | +36.359% | FAIL |
| Phase 2 retry, early-only mix (prior log) | `512,256,64` on 0–2; `256,256,64` on 3–23 | 19.280 | +35.327% | FAIL |
| Late-block protect (prior log) | `512,256,64` on 0–2 and 19–23; `256,256,64` on 3–18 | 19.039 | +33.635% | FAIL |
| This run | `512,256,64` on 0–2, 13–18, and 19–23; `256,256,64` on 3–12 | 18.803 | +31.979% | **FAIL** |

18.803 − 19.427 = −0.624 versus the flat recipe. 18.803 − 19.280 =
−0.477 versus the early-only mix. 18.803 − 19.039 = −0.236 versus the
late-block protect, which is the layers 13–18 bump with layers 0–2 and
19–23 already at `512,256,64`. 18.803 − 14.959 = +3.844 versus the
gate. Protecting the next chunk is a partial fix of about the same
size as the late-block bump (−0.241).

No new mid-layer isolation was run. The Phase 4 map already has that
ranking. The next extra bits go on **layers 3–7** (alone 14.897,
Δ +0.650), then layers 8–12 (Δ +0.520). That follow-up full-24 was not
started here. The 13–18 bump moved the full model by −0.236, and the
miss that remains is +3.844, so one more chunk at this rung is not
enough to clear the gate on the evidence in hand. KLT, rotation, and
permutation were not retried.

Write-up: `artifacts/pbr_ladder/phase4_l1318_full24.md` and `.json`.
Logs: `phase4_l1318_quantize.log`, `phase4_l1318_ppl_baseline.log`,
`phase4_l1318_ppl_full.log`.

The follow-up that also protects layers 3–7 was run next. It also
misses. See the section below.

### Phase 4 — protect layers 3–7, full 24 layers

**FAIL** against the same 5% gate (perplexity ≤ 14.959). Layers 0–7,
13–18, and 19–23 at `512,256,64`, layers 8–12 at `256,256,64`. Same
`quantize_full_model.py` and `eval_ppl.py` (`PBR_CPU_FP32=1`,
WikiText-2 raw v1 test, 299078 tokens). This run's baseline is again
**14.247**. The checkpoint is dequantized fp32 for perplexity (1.9 GB
vs the original 943 MB bf16 file). All 290 saved tensors are float32.
It is a perplexity checkpoint.

`PBR_CHUNK=20000` only splits the pooled distance matrix; the per-row
argmin matches the default chunk. This was a full pass from an empty
output directory. The script can resume from `pbr_progress.json` after
a later kill; this run finished all 24 layers on the first attempt
without needing that. Phase 4 is done only if the whole-model
perplexity is ≤ 14.959. It is not.

| Layers | Stages | Index bpw |
| --- | --- | ---: |
| 0–7 | `512,256,64` | 2.875 |
| 8–12 | `256,256,64` | 2.75 |
| 13–18 | `512,256,64` | 2.875 |
| 19–23 | `512,256,64` | 2.875 |

357,826,560 parameters touched. Average index cost is **2.848958** bpw
(`(19 × 2.875 + 5 × 2.75) / 24`). Adding the `push_below3.py` fp16
codebook/scale side info puts the same average at **2.909625**. Phase 2
flat was 2.75 index / 2.798 with side info. The Phase 2 retry
(early-only mix) was 2.765625 / 2.816. The late-block protect was
2.791667 / 2.845. The L13–18 protect was 2.822917 / 2.88025.
2.848958 stays well under ~4. These bpw figures are index-cost
estimates.

| Run | Recipe | WikiText-2 test ppl | vs 14.247 | vs 14.959 |
| --- | --- | ---: | ---: | --- |
| Baseline (this run) | full precision | 14.247 | — | — |
| Phase 2 flat all-24 (prior log) | `256,256,64` on 0–23 | 19.427 | +36.359% | FAIL |
| Phase 2 retry, early-only mix (prior log) | `512,256,64` on 0–2; `256,256,64` on 3–23 | 19.280 | +35.327% | FAIL |
| Late-block protect (prior log) | `512,256,64` on 0–2 and 19–23; `256,256,64` on 3–18 | 19.039 | +33.635% | FAIL |
| L13–18 protect (prior log) | `512,256,64` on 0–2, 13–18, and 19–23; `256,256,64` on 3–12 | 18.803 | +31.979% | FAIL |
| This run | `512,256,64` on 0–7, 13–18, and 19–23; `256,256,64` on 8–12 | 18.492 | +29.796% | **FAIL** |

18.492 − 19.427 = −0.935 versus the flat recipe. 18.492 − 19.280 =
−0.788 versus the early-only mix. 18.492 − 19.039 = −0.547 versus the
late-block protect. 18.492 − 18.803 = −0.311 versus the L13–18
protect, which is the layers 3–7 bump with layers 0–2, 13–18, and
19–23 already at `512,256,64`. 18.492 − 14.959 = +3.533 versus the
gate. Protecting the next chunk is a partial fix, a bit larger than
the L13–18 bump (−0.236).

No new mid-layer isolation was run. The Phase 4 map already has that
ranking. The remaining mildest chunk is **layers 8–12** (alone 14.767,
Δ +0.520). That follow-up full-24 was not started here. The 3–7 bump
moved the full model by −0.311, and the miss that remains is +3.533,
so one more chunk at this rung is not enough to clear the gate on the
evidence in hand. KLT, rotation, and permutation were not retried.

Write-up: `artifacts/pbr_ladder/phase4_l37_full24.md` and `.json`.
Logs: `phase4_l37_quantize.log`, `phase4_l37_ppl_baseline.log`,
`phase4_l37_ppl_full.log`.

## What wasn't tested

- **A smaller on-disk format, and any comparison to GGUF / AQLM / QuIP#.**
  Phase 2's perplexity gate failed. The checkpoints above are dequantized
  fp32 copies used to measure perplexity.
- **Perplexity-vs-bpw curve**: the §5.1 sweep shows bpw options from
  `[256,256]` (~2.0bpw) up to `[256,256,256]` (~3.0-3.14bpw) per tensor in
  `real_tensor_results.txt`. End-to-end perplexity exists for one matrix
  (L12 gate_proj at 2.785 bpw), for all seven layer-12 linears together
  at stages `256,256,64` (Phase 1), for layers 0–2 at that recipe
  (Phase 2b, 15.005), for layers 0–2 at `512,256,64` plus the
  layer-0-only bump (Phase 4, 14.873 and 14.951), and for all 24 layers
  at the mixed recipe (Phase 2 retry, 19.280), for all 24 layers
  with layers 0–2 and 19–23 at `512,256,64` and layers 3–18 at
  `256,256,64` (Phase 4 late-block protect, 19.039), for all 24 layers
  with layers 0–2, 13–18, and 19–23 at `512,256,64` and layers 3–12 at
  `256,256,64` (Phase 4 L13–18 protect, 18.803), for all 24 layers
  with layers 0–7, 13–18, and 19–23 at `512,256,64` and layers 8–12 at
  `256,256,64` (this run, 18.492), and for four
  L3–23 chunks at `256,256,64` with every other layer full precision
  (Phase 4 map: 14.897, 14.767, 15.053, 15.447; the layer 19–23
  split is 14.746 and 14.971). Other rungs
  (`256,256,256` and above) were not scored. A uniform 24-layer
  `512,256,64` run was not scored. Per-tensor perplexity at the new
  stages is unknown.
- **Layer 3 on its own, and layers 8–12 at `512,256,64`.** The Phase 4
  map scores layers 3–7, 8–12, 13–18, and 19–23 separately at
  `256,256,64`, and it splits 19–23 into 19–21 and 22–23. The
  late-block full-24 raises bits on 0–2 and 19–23 together and leaves
  3–18 at `256,256,64` (perplexity 19.039). The next full-24 also
  raises 13–18 (perplexity 18.803) and leaves 3–12 at `256,256,64`.
  This full-24 also raises 3–7 (perplexity 18.492) and leaves 8–12 at
  `256,256,64`. It does not raise 22–23 on their own inside the full
  model. The last chunk on the map is
  layers 8–12. Phase 2's flat all-24 number remains 19.427. The
  early-only mixed full-24 number is 19.280. The late-block number is
  19.039. The L13–18 number is 18.803.
- `formula_probe.py`, `node_combo_q4.py`, `node_structures.py` were run
  only against synthetic data (their self-check), not real Qwen tensors —
  the guide's real-tensor procedure (§3.1–3.7) for these 3 wasn't
  specified beyond the synthetic case in the sandbox writeup this
  replaces.

## Go/no-go gates (guide §7.2) — status

| Gate | Status |
| --- | --- |
| Environment check (synthetic, §2.3) | **PASS** — matches Appendix A |
| Phase 0 — 9-tensor dB numbers are real activations | **PASS** — see Phase 0 section. High L2/L21 `down_proj` dB reproduced. L21 `down_proj` `push_below4` crash is fixed. |
| Tensor quality, real weights (§5.1, ≥2dB over INT4 RTN @ ≤3.3bpw) | **FAIL on 6/9 tensors** — see table above |
| Single-matrix perplexity (§5.2, <0.5% change) | **PASS** — +0.154% on L12 gate_proj @ 2.785bpw. One matrix only. |
| Phase 1 — entire layer 12 perplexity (<1%) | **PASS** — 14.247 → 14.354 (+0.751%) with all 7 L12 linears at stages 256,256,64. dB gate still fails on those tensors. |
| Whole-model retention (Phase 2, within 5% of 14.247) | **FAIL** — all 24 layers: 14.247 → 19.427 (+36.359%). Dry run layers 0–2: 15.005 (+5.320%). |
| Phase 2b — isolate L0 / L1 / L2 | **diagnostic, no gate** — L0 14.528 (+1.972%), L1 14.445 (+1.390%), L2 14.467 (+1.544%). The three share the +5.320%. |
| Phase 4 — L0–L2 at a modest bit bump (≤ 14.959) | **PASS** — all three layers at `512,256,64`: 14.873 (+4.394%). L0 only at `512,256,64`, L1 and L2 still `256,256,64`: 14.951 (+4.941%). Not a 24-layer result. |
| Whole-model retention, Phase 2 retry (mixed bits, ≤ 14.959) | **FAIL** — layers 0–2 at `512,256,64`, layers 3–23 at `256,256,64`: 14.247 → 19.280 (+35.327%). Gate is 14.959. Flat Phase 2 all-24 was 19.427. Average index bpw 2.765625. Dequantized fp32 checkpoint, not a size win. |
| Phase 4 map — L3–23 in four chunks at `256,256,64` | **diagnostic, no gate** — test_a L3–7 14.897 (+4.562%), test_b L8–12 14.767 (+3.650%), test_c L13–18 15.053 (+5.657%), test_d L19–23 15.447 (+8.423%). Hot chunk is layers 19–23. Bisect: L19–21 14.746 (+3.502%), L22–23 14.971 (+5.082%). Sum of the four isolated deltas +3.176. Dequantized fp32, not a size win. |
| Whole-model retention, Phase 4 late-block protect (≤ 14.959) | **FAIL** — layers 0–2 and 19–23 at `512,256,64`, layers 3–18 at `256,256,64`: 14.247 → 19.039 (+33.635%). Gate is 14.959. Flat Phase 2 all-24 was 19.427. Early-only mixed full-24 was 19.280. Average index bpw 2.791667. Dequantized fp32 checkpoint, not a size win. Phase 4 is not done. |
| Whole-model retention, Phase 4 L13–18 protect (≤ 14.959) | **FAIL** — layers 0–2, 13–18, and 19–23 at `512,256,64`, layers 3–12 at `256,256,64`: 14.247 → 18.803 (+31.979%). Gate is 14.959. Late-block full-24 was 19.039. Early-only mixed full-24 was 19.280. Flat Phase 2 all-24 was 19.427. Average index bpw 2.822917. Dequantized fp32 checkpoint, not a size win. Phase 4 is not done. Next chunk on the map is layers 3–7. |
| Whole-model retention, Phase 4 L3–7 protect (≤ 14.959) | **FAIL** — layers 0–7, 13–18, and 19–23 at `512,256,64`, layers 8–12 at `256,256,64`: 14.247 → 18.492 (+29.796%). Gate is 14.959. L13–18 full-24 was 18.803. Late-block full-24 was 19.039. Early-only mixed full-24 was 19.280. Flat Phase 2 all-24 was 19.427. Average index bpw 2.848958. Dequantized fp32 checkpoint for perplexity. Phase 4 is not done. The last chunk on the map is layers 8–12. |

## To reproduce

```bash
cd pbr_ladder
pip install numpy safetensors torch transformers datasets huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='qwen05b')"
export PBR_CPU_FP32=1   # skip this if your machine has real bf16 acceleration (e.g. a GPU)
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.gate_proj acts_L12_gate.npy
M=qwen05b/model.safetensors; T=model.layers.12.mlp.gate_proj.weight; A=acts_L12_gate.npy
python3 push_below4.py $M $T $A
python3 push_below3.py $M $T $A
PBR_SAVE=L12_gate_293.npz PBR_NAME=$T PBR_SAVE_KS=256,256,64 python3 push_below3.py $M $T $A
python3 eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct
python3 eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct L12_gate_293.npz
```

## Honesty

- All numbers above are from real Qwen2.5-0.5B-Instruct runs on this
  machine, reported as-is from raw console output (see
  `pbr_ladder/real_tensor_results.txt`, `pbr_ladder/eval_ppl_results.txt`)
  — nothing rounded in the network's favor.
- The §5.1 tensor-quality gate genuinely fails on 6 of 9 tensors tested;
  this is not glossed over because the perplexity gate passed.
  A layer that fails the dB gate can still pass the perplexity gate (as
  seen here) — the two are not interchangeable evidence.
- The §5.2 perplexity PASS is a single layer, single bpw point. It does
  not extrapolate to a whole-model claim.
- Phase 2b perplexities are the printed `eval_ppl.py` lines
  (14.528, 14.445, 14.467, 14.702, 14.674), with this run's baseline
  again 14.247. Percentages are `(ppl − 14.247) / 14.247`. The L0–2,
  L12, and all-24 figures in that table are the earlier Phase 1/2 logs,
  not a second measurement. Checkpoints used for these perplexities are
  dequantized fp32.
- Phase 4 perplexities are the printed lines 14.247, 14.873, and 14.951.
  The 15.005 row in that table is the Phase 2 log. bpw figures with
  side info are the `push_below3.py` formula on the measured shapes,
  not a packed-file size.
- Phase 2 retry perplexities are the printed lines 14.247 (baseline)
  and 19.280 (full 24-layer mix), both at 299078 tokens. The 19.427
  and 14.873 rows in that table are the Phase 2 and Phase 4 logs.
  2.765625 is the parameter-weighted average of the printed index
  costs. 2.816 adds estimated codebook side info. Neither number is
  an on-disk size.
- Phase 4 map perplexities are the printed lines 14.247, 14.897,
  14.767, 15.053, and 15.447, each at 299078 tokens. The bisection of
  layers 19–23 printed 14.746 (layers 19–21) and 14.971 (layers 22–23),
  also at 299078 tokens. Percentages are `(ppl − 14.247) / 14.247`.
  The 19.280, 14.873, 19.427, and 15.005 figures used in the
  additivity comparison are the earlier Phase 2, Phase 4, and Phase 2
  retry logs. The chunk checkpoints are dequantized fp32.
- Phase 4 late-block perplexities are the printed lines 14.247
  (baseline) and 19.039 (full 24-layer mix), both at 299078 tokens.
  The 19.427 and 19.280 rows in that table are the Phase 2 and Phase 2
  retry logs. 2.791667 is `(8 × 2.875 + 16 × 2.75) / 24` on the printed
  index costs. 2.845 adds estimated codebook side info. Neither number
  is an on-disk size. Percent is `(19.039 − 14.247) / 14.247`.
- Phase 4 L13–18 protect perplexities are the printed lines 14.247
  (baseline) and 18.803 (full 24-layer mix), both at 299078 tokens.
  The 19.427, 19.280, and 19.039 rows in that table are the Phase 2,
  Phase 2 retry, and late-block logs. 2.822917 is
  `(14 × 2.875 + 10 × 2.75) / 24` on the printed index costs. 2.88025
  adds estimated codebook side info. Neither number is an on-disk
  size. Percent is `(18.803 − 14.247) / 14.247`.
- Phase 4 L3–7 protect perplexities are the printed lines 14.247
  (baseline) and 18.492 (full 24-layer mix), both at 299078 tokens.
  The 19.427, 19.280, 19.039, and 18.803 rows in that table are the
  Phase 2, Phase 2 retry, late-block, and L13–18 logs. 2.848958 is
  `(19 × 2.875 + 5 × 2.75) / 24` on the printed index costs. 2.909625
  adds estimated codebook side info. Neither number is an on-disk
  size. Percent is `(18.492 − 14.247) / 14.247`.
