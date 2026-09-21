# PBR-Q4 E2 mixed groupwise Q + E3−E2 ablation

This is a **new** quantized reference for Qwen2.5-0.5B-Instruct. It is not
H95Q-S1, not PR #15 always-on X/Y, and not PR #17/#18. Decode is
**bit-exact to this quantized reference**, not to original BF16.

Root cause already measured on #17/#18: naive groupwise INT on embed /
mid-MLP kills the held-out proxy. E2 therefore uses
**importance-structured protect** at tensor → projection → contiguous
output-channel groups (default 16 rows). Q4 is used only where sensitivity
allows. Sparse per-weight exceptions are **not** a quality lever (NaN/Inf
overlays only).

## E2 — mixed groupwise Q

- Bits: embed/attn/MLP body `{Q4,Q5,Q6}`; norms/biases `{Q8,BF16}`;
  protected channels `{Q6,Q8,BF16}`.
- Group size 128 along the input axis; per-row (output-channel) mixed bits.
- Assignment: start at family min, spend a packed-bit budget on the best
  importance / extra-bits upgrades (H95 family weights + row L2 + calib
  activation scale + embed token frequency).
- Optional E2b: among scales with `E ≤ (1+ε) E_min` for
  `ε ∈ {0, 0.001, 0.0025, 0.005, 0.01}`, pick lowest LEFT-residual / std
  code-cost proxy. Default `ε=0.0025` (quality dominates).
- Packed estimate target ≤ **5.0** BPW (stretch ≤ **4.75**). Held-out
  proxy ≥ **0.95** (aim ≥ **0.97**). If the dual gate fails, the PR still
  opens with an honest Pareto (same bar as #17/#18).

## E3 — same Q-ref + selective lossless tiles

Reuses the PR #16 complete-cost rule on **E2 codes**, with a net margin:

- Product default = packed
- Phase-1 modes only: packed, LEFT / UP / PAETH (row traversal), BITPLANE /
  RUN / rANS. No PAIR, no extra traversals, no 12-mode expansion.
- Matrix family only if it saves `≥ max(16 bits, 0.02 × raw_tile_bits)`
  after **all** mode / table / pad / map costs.

## Critical ablation

```
E2 physical BPW
E3 physical BPW
Δ = E2 − E3   (matrix net contribution)
```

Milestones: useful ≥ 0.03, good ≥ 0.05, strong ≥ 0.10.

Overhead split: weight payload / scales / precision map / codec meta /
container.

## Run

```bash
PYTHONPATH=. python scripts/run_pbr_q4_e2_mixed.py
```

Artifacts: `artifacts/pbr_q4/e2_mixed_qwen.{json,md}`. Large `.e2mx` /
`.e3mx` blobs are gitignored.

```bash
python -m pbr_q4.e2_mixed.container artifacts/pbr_q4/containers/PBR-Q4-e2_budget_5_00.e3mx --expect-sha <sha>
```

## Measured (Qwen2.5-0.5B-Instruct)

Winner `e2_q6_tight` (Q6 floor + light structured Q8/BF16 protect):

```
E2 physical BPW  6.283588
E3 physical BPW  6.283611
Δ = E2 − E3      -0.000023   (below_useful; 0 XY tiles)
```

| Gate | Target | Measured | Verdict |
| --- | --- | ---: | --- |
| Practical rate | ≤ 5.0 | 6.283611 | **FAIL** |
| Stretch rate | ≤ 4.75 | 6.283611 | **FAIL** |
| Held-out ≥ 0.95 | 0.95 | 0.957372 | **PASS** |
| Held-out aim 0.97 | 0.97 | 0.957372 | **FAIL** |
| Exact Q-ref decode | SHA | `141c08c6…3b46cdac` | **PASS** |

Q4-class maps under a ≤5.0 packed-est still fail heldout (~0.79–0.80). Q5 floor reaches 0.922. First ≥0.95 is Q6-class, same qualitative wall as PR #17.

## Honesty

- In-repo calib-v2 / heldout-v1 PPL proxy — not MMLU.
- Physical BPW from file size, not rounded to 5.0.
- Not a production mobile runtime.
