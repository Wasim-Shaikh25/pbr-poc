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

## What wasn't tested

- **Whole-model retention** (guide §5.3–§5.4): only one layer, one tensor,
  one bpw point was swapped and perplexity-checked. No claim is made
  about swapping all layers, or about retention at other bpw points.
- **Perplexity-vs-bpw curve**: the §5.1 sweep shows bpw options from
  `[256,256]` (~2.0bpw) up to `[256,256,256]` (~3.0-3.14bpw) per tensor in
  `real_tensor_results.txt`, but only one of those points (2.785bpw,
  L12 gate_proj) was perplexity-tested end-to-end. Where the 0.5% gate
  actually breaks, per tensor, is unknown.
- `formula_probe.py`, `node_combo_q4.py`, `node_structures.py` were run
  only against synthetic data (their self-check), not real Qwen tensors —
  the guide's real-tensor procedure (§3.1–3.7) for these 3 wasn't
  specified beyond the synthetic case in the sandbox writeup this
  replaces.

## Go/no-go gates (guide §7.2) — status

| Gate | Status |
| --- | --- |
| Environment check (synthetic, §2.3) | **PASS** — matches Appendix A |
| Tensor quality, real weights (§5.1, ≥2dB over INT4 RTN @ ≤3.3bpw) | **FAIL on 6/9 tensors** — see table above |
| Single-layer perplexity (§5.2, <0.5% change) | **PASS** — +0.154% on L12 gate_proj @ 2.785bpw |
| Whole-model retention (§5.3–§5.4) | **NOT RUN** — see "What wasn't tested" |

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
