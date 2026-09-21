# Handoff to Grok: PBR-Ladder Phase 0–2 (whole-model quantization)

This session (Claude) is stepping back; a Grok-driven session is taking
over from here. This file is the complete state transfer: what's proven,
what's not, what just landed, and the immediate next steps per
`docs/pbr_ladder/PBR_Ladder_Whats_Next.docx` (the roadmap doc — read that
first for the full 9-phase plan, this file is just the operational
handoff).

## What's true right now

- One tensor (`model.layers.12.mlp.gate_proj.weight`), quantized to
  ~2.785 bpw, swapped into the real Qwen2.5-0.5B-Instruct model: baseline
  perplexity 14.247 → 14.269 (+0.154%), well under the 0.5% gate. **This
  is the only real, whole-model-level result that exists.**
- 9 tensors (`gate_proj`/`down_proj`/`q_proj` × layers 2/12/21) have
  per-tensor output-dB numbers against real captured activations. Tensor
  quality gate (≥2dB over INT4 RTN @ ≤3.3bpw): **FAIL on 6/9**, only the
  3 `down_proj` tensors clear it clearly (+11.97 to +16.59dB); `gate_proj`
  is flat/negative, `q_proj` marginal. Full numbers and raw logs:
  `docs/pbr_ladder/README.md`, `pbr_ladder/real_tensor_results.txt`.
- Everything else in the model — the other 23 layers, `up_proj`/
  `k_proj`/`v_proj`/`o_proj`, embeddings, output head, norms — is
  **untouched, full precision, unmeasured.**

Do not let "0.15% perplexity change" get generalized to "the whole model
compresses cleanly." It hasn't been tested yet. See the roadmap doc §1.2
and §5 for the exact claims-discipline line to hold.

## What landed in this push

1. **`pbr_ladder/quantize_full_model.py`** (new, user-supplied, UNTESTED
   here — no `huggingface.co` access in this sandbox, same limitation as
   every other real-model script in this repo). This is the Phase 2
   driver: loops every attention/MLP linear layer across all 24 layers,
   quantizes each (rotation + GPTQ-style error feedback + 8-D residual
   node codebooks — the same method already proven layer-by-layer),
   **sequentially** (layer *n* is quantized and written back before
   calibration text is re-run for layer *n+1*, so later layers see real
   already-quantized inputs, matching production GPTQ practice).

   **Important:** this script saves a **full-precision, dequantized**
   checkpoint via `model.save_pretrained(..., safe_serialization=True)`.
   The output folder is for perplexity measurement (Phase 2's gate), not
   a compressed file — it will NOT be smaller on disk than the original.
   Building an actual smaller file format is Phase 7, not this script.
   Don't let anyone read "saved a new checkpoint" as "shrunk the file."

2. **`docs/pbr_ladder/PBR_Ladder_Whats_Next.docx`** — the full 9-phase
   roadmap (source of truth for scope/gates/claims discipline). Read
   this before doing anything below.

3. **Bug fix in `pbr_ladder/push_below4.py`**: its `kmeans`/`rvq`
   functions built a full `(rows, 256)` float64 distance matrix in one
   shot — on `L21 down_proj` (544,768 rows after the 8-D reshape) that's
   ~1.04 GiB per allocation and previously crashed with
   `numpy.core._exceptions._ArrayMemoryError` (see the traceback in
   `pbr_ladder/real_tensor_results.txt`, `L21 mlp.down_proj` section).
   Fixed by chunking those distance-matrix computations (`PBR_CHUNK`,
   default 50000 rows/batch — same pattern `quantize_full_model.py`'s own
   `kmeans` already uses). **Verified bit-identical** on the synthetic
   self-check (`python3 push_below4.py` with no args still prints exactly
   `INT3 RTN 11.11`, `INT3 GPTQ 13.52`, `INT3 rot+GPTQ 18.91`, matching
   the guide's Appendix A) — this is a pure memory-behavior fix, not a
   numeric change. Also smoke-tested on a synthetic tensor sized to match
   the crash (896×4864, same shape as the real `down_proj` that OOM'd)
   without running out of memory.

## Phase 0 — one open item, not resolved here

The roadmap doc (§3, Phase 0) flags: "the unusually high scores on
`down_proj` at layers 2 and 21 suggest synthetic activations were likely
used there" — i.e. confirm whether `push_below4`/`push_nested`/
`push_below3`'s real-tensor runs for **all 9 tensors** actually used real
captured activations (`capture_activations.py` output) or fell back to
the synthetic default (which happens automatically if a script is called
without a 3rd `.npy` argument).

**This session could not resolve it** — the commit history and PR body
for the real-tensor run (`docs/pbr_ladder/README.md`, PR #21) say real
activations were used for all 9, but the actual shell commands/activation
files aren't in the repo (they're gitignored, `pbr_ladder/*.npy`) to
check directly. First action for the Grok session: re-verify by either
(a) asking whoever ran it to confirm/paste the exact commands used per
tensor, or (b) re-running `capture_activations.py` for all 9 tensor/layer
combos and re-doing the 9-tensor sweep from scratch, discarding the old
numbers if they turn out to have used synthetic activations anywhere.
Don't build Phase 1/2 conclusions on numbers that might be synthetic
without settling this first — that's exactly what Phase 0's gate is for.

## Immediate next steps (Phase 1, then Phase 2)

Needs a machine with `huggingface.co` access — this sandbox doesn't have
it (blocked by egress policy, confirmed repeatedly, see
`docs/pbr_ladder/HANDOFF.md`).

**Phase 1 — finish layer 12** (a few days, mostly compute):
```bash
cd pbr_ladder
# up_proj, k_proj, v_proj, o_proj on layer 12, same procedure as gate_proj/q_proj already used
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 mlp.up_proj acts_L12_up.npy
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 self_attn.k_proj acts_L12_k.npy
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 self_attn.v_proj acts_L12_v.npy
python3 capture_activations.py Qwen/Qwen2.5-0.5B-Instruct 12 self_attn.o_proj acts_L12_o.npy
# gate_proj/up_proj share input (already captured as acts_L12_gate.npy from the §5.1 run)
M=qwen05b/model.safetensors
python3 push_below4.py $M model.layers.12.mlp.up_proj.weight acts_L12_gate.npy
python3 push_below4.py $M model.layers.12.self_attn.k_proj.weight acts_L12_k.npy
python3 push_below4.py $M model.layers.12.self_attn.v_proj.weight acts_L12_v.npy
python3 push_below4.py $M model.layers.12.self_attn.o_proj.weight acts_L12_o.npy
```
Then replace **all** layer-12 matrices at once (not just gate_proj) and
re-measure whole-model perplexity — gate: <1% change. There's no ready
"replace all matrices in one layer" script yet; `quantize_full_model.py`
with `PBR_LAYERS=12` does this as a side effect (it quantizes every
targeted matrix in the given layer(s)) — reasonable to reuse for Phase 1
instead of writing a separate one-layer-only tool:
```bash
PBR_LAYERS=12 python3 quantize_full_model.py qwen05b qwen05b_L12_only
python3 eval_ppl.py qwen05b_L12_only   # compare against the 14.247 baseline
```

**Phase 2 — whole model** (~a week, compute-bound), only after Phase 1's
gate passes:
```bash
# quick dry run first (per the script's own header) before committing to a full run
PBR_LAYERS=0,1,2 python3 quantize_full_model.py qwen05b qwen05b_test
python3 eval_ppl.py qwen05b_test

# full run
python3 quantize_full_model.py qwen05b qwen05b_pbr3bit
python3 eval_ppl.py qwen05b_pbr3bit
```
Gate: whole-model perplexity within 5% of the 14.247 baseline.

## Claims discipline (carry this forward)

Same rule as everywhere else in this repo: only claim what the current
phase has actually measured. The roadmap doc §5 has the exact fair/unfair
claim table per phase — follow it. In particular, right now the only fair
claim is "one MLP matrix in one layer, ~2.8 bpw, <0.2% perplexity change,
measured on the real model." Nothing about the whole model, nothing about
2-bit budgets, nothing about beating GGUF/AQLM/QuIP#, nothing about a
real loadable compressed file — all of that is future phases.
