# Hybrid H95 mantissa-keep + groupwise INT + selective X/Y

Follow-up after PR #17 (leave that PR alone). Pure groupwise INT lacks
per-weight exponents, so Q6 holds quality (heldout 0.967) at **6.191 BPW**
while Q4/Q5 hit the size gate and collapse heldout to ~0.70–0.82.

## What this is

A **new** quantized reference (`HYBX` v1):

- **Protected:** H95-style mantissa-keep with adaptive exact per-weight
  exponents (late MLP, attention, last block, norms; sparse INT outliers).
- **Aggressive:** groupwise Q4/Q5 INT (mid MLP, first MLP, embed).
- **Selective X/Y:** packed default; emit a 256-node tile only if the
  complete physical cost (payload + flag + map share) is strictly smaller.

Not S1. Not PR #17 `restore_q6`. Decode must match *this* SHA.

## Hard gates

1. Physical `actual_bpw = file_bytes * 8 / n_weights` ≤ **5.0** (stretch ≤4.5).
2. Held-out proxy (calib-v2 search / heldout-v1 gate) ≥ **0.95** (aim ≥0.97).
3. Exact decode of this Q-ref.
4. Honest FAIL if either rate or quality misses — still publish the Pareto table.

## Code

- `pbr_q4/hybrid/policy.py` — H95 vs INT slots from H95 sensitivity
- `pbr_q4/hybrid/int_quant.py` — groupwise INT, MSE-first scale search,
  spatial near-tie (λ small)
- `pbr_q4/hybrid/quantize.py` — dispatch H95 / INT / raw
- `pbr_q4/hybrid/container.py` — `HYBX` encode/decode (`python -m pbr_q4.hybrid.container`)
- `scripts/run_pbr_q4_hybrid.py` — full Qwen2.5-0.5B-Instruct
- Tests: `tests/test_pbr_q4_hybrid.py`

## Run

```bash
PYTHONPATH=. python scripts/run_pbr_q4_hybrid.py
```

Policies (auto ladder, cheapest first): `h_stretch`, `h_mix`, `h_practical`,
`h_restore`, `h_quality`. First map with heldout ≥0.95 is encoded.

## Honesty

- Proxy PPL on in-repo calib-v2 / heldout-v1 only — not MMLU.
- File-size BPW only. Packed-est rows in the Pareto table are labeled estimates.
- Not a production mobile runtime.
