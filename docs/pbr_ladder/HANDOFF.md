# Handoff: PBR-Ladder real-Qwen run (do this on a machine with HF access)

Context: this repo's cloud sandbox cannot reach `huggingface.co` (blocked
by the sandbox's egress proxy policy — confirmed, not a code issue). All
real-model steps below need to run on your own machine, or any environment
that can reach Hugging Face.

Branch: `claude/poc-results-x1jeoa` (already pushed). Pull it first:

```bash
git clone https://github.com/Wasim-Shaikh25/pbr-poc.git
cd pbr-poc
git checkout claude/poc-results-x1jeoa
git pull
```

## Current status (done in the sandbox, no action needed)

- `pbr_ladder/` has 6 of the 9 guide scripts: `push_below4.py`,
  `push_below3.py`, `push_nested.py`, `ladder_exact.py`,
  `capture_activations.py`, `eval_ppl.py`.
- Missing from the guide (not supplied yet): `formula_probe.py`,
  `node_combo_q4.py`, `node_structures.py`.
- All 4 non-helper scripts pass their synthetic self-check, output
  matches the guide's Appendix A exactly. Logs in `artifacts/pbr_ladder/`,
  full writeup in `docs/pbr_ladder/README.md`.
- No real-Qwen numbers exist anywhere in the repo yet. Nothing has been
  fabricated — don't let a next session assume otherwise.

## What to run, in order

### 0. Setup

```bash
cd pbr-poc/pbr_ladder
pip install numpy safetensors torch transformers datasets huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir qwen05b
```

List tensor names to confirm the layer names below are right:

```bash
python3 -c "
from safetensors import safe_open
f = safe_open('qwen05b/model.safetensors', 'pt')
[print(k, f.get_slice(k).get_shape()) for k in f.keys() if 'layers.12.' in k]
"
```

### 1. Environment check (should already match — sanity only)

```bash
python3 push_below4.py
```
Expect: `INT3 RTN 11.11`, `INT3 GPTQ 13.52`, `INT3 rot+GPTQ 18.91` (matches
Appendix A / already-committed synthetic logs).

### 2. Capture real calibration activations

```bash
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.gate_proj acts_L12_gate.npy
```
(gate_proj and up_proj share the same input — one capture covers both.
down_proj needs its own capture: `mlp.down_proj`.)

### 3. Real-tensor tests (guide §5.1) — at least 9 tensors

`gate_proj`, `down_proj`, `q_proj` × layers 2, 12, 21 (or whatever layers
you choose — 0–23 for the 0.5B model). Example for layer 12 gate_proj:

```bash
M=qwen05b/model.safetensors
T=model.layers.12.mlp.gate_proj.weight
A=acts_L12_gate.npy
python3 push_below4.py $M $T $A
python3 push_nested.py $M $T $A
python3 push_below3.py $M $T $A
python3 ladder_exact.py $M $T $A
```

Gate (guide §7.2): nodes at ≤3.3 bpw should beat INT4 RTN output dB by
≥2 dB on most tensors.

### 4. Save a quantized layer and check single-layer perplexity (guide §5.2)

```bash
PBR_SAVE=L12_gate_293.npz PBR_NAME=model.layers.12.mlp.gate_proj.weight PBR_SAVE_KS=256,256,64 \
  python3 push_below3.py $M $T $A

python3 eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct                  # baseline
python3 eval_ppl.py Qwen/Qwen2.5-0.5B-Instruct L12_gate_293.npz # one layer swapped
```

Gate: swapping one layer changes perplexity by less than 0.5%.

## Report back

When you have real numbers, either:
- push a new commit on `claude/poc-results-x1jeoa` (or a follow-up branch)
  with the raw output + an update to `docs/pbr_ladder/README.md`'s gate
  table, or
- paste the raw console output back into the Claude session and ask for
  it to be written up — don't summarize/round the numbers by hand, paste
  the actual script output.

Keep the same claims discipline as the rest of this repo (see guide §9,
and other `docs/*/README.md` files): report PASS/FAIL against the stated
gates, don't round in the network's favor, and don't extrapolate a
single-layer result into a whole-model claim.
