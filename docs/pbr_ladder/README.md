# PBR-Ladder — script guide run, now including real Qwen2.5-0.5B-Instruct results

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

## What wasn't tested

- **A smaller on-disk format, and any comparison to GGUF / AQLM / QuIP#.**
  Phase 2's perplexity gate failed. The checkpoints above are dequantized
  fp32 copies used to measure perplexity.
- **Perplexity-vs-bpw curve**: the §5.1 sweep shows bpw options from
  `[256,256]` (~2.0bpw) up to `[256,256,256]` (~3.0-3.14bpw) per tensor in
  `real_tensor_results.txt`. End-to-end perplexity exists for one matrix
  (L12 gate_proj at 2.785 bpw), for all seven layer-12 linears together
  at stages `256,256,64` (Phase 1), for layers 0–2 at that recipe
  (Phase 2b, 15.005), and for layers 0–2 at `512,256,64` plus the
  layer-0-only bump (Phase 4, 14.873 and 14.951). Other rungs
  (`256,256,256` and above) were not scored. Per-tensor perplexity at
  the new stages is unknown.
- **Layer 3, and the other 21 layers at the bumped recipe.** Phase 4
  quantizes only layers 0–2. Layer 3 was not run on its own. The
  24-layer model was not rerun; Phase 2's 19.427 stands at stages
  `256,256,64`.
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
