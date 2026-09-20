# PBR-Q4 / H95Q-S1 + 256-node X/Y post-codec

This line **does not** re-quantize Qwen from BF16 into a new groupwise-Q4
policy. It takes the frozen **H95Q-S1** quantized reference (container v2,
physical **7.42082 BPW**, SHA
`eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de`) and
re-encodes its already-quantized mantissa K-codes with the 16×16 / 256-node
X/Y matrix family on **every** eligible tile.

H95Q S1 / container v2 code is unchanged.

## Hard rules

- Start from S1’s exact quantized words. Decode must match the frozen SHA.
- Every tile is stored as a matrix-family mode: `XY_MATRIX` / `XY_RANS` /
  `XY_BITPLANE` / `XY_PAIR` / `XY_RUN` with predictors LEFT / UP / AVG /
  PAETH / PREVIOUS and row/column (serpentine) traversals.
- Packed-K size is a **baseline metric only**. It is never the container path.
- `actual_bpw = file_bytes * 8 / n_weights` from the physical `.h95x` file.
- Quality is not re-run when the SHA matches; S1’s heldout proxy 0.990 still applies.

## Code

- `pbr_q4/predictors.py` — causal X/Y predictors + residuals
- `pbr_q4/codecs.py` — matrix-family tile codecs + all-tile encode
- `pbr_q4/container.py` — `H95X` encode/decode (`python -m pbr_q4.container`)
- `pbr_q4/s1_source.py` — load S1 from a frozen `.h95q` or rebuild from the model
- `scripts/run_pbr_q4_s1_xy.py` — full S1 encode + artifacts
- Tests: `tests/test_pbr_q4_s1_xy.py`

## Run

```bash
# Preferred: transcode a frozen S1 container (v2 or v1)
PYTHONPATH=. python scripts/run_pbr_q4_s1_xy.py \
  --s1-container artifacts/pbr_h95/containers/H95Q-S1-v2.h95q

# Or rebuild Q(W) from the checkpoint (refuses if SHA drifts)
PYTHONPATH=. python scripts/run_pbr_q4_s1_xy.py \
  --model-dir outputs/models/Qwen__Qwen2.5-0.5B-Instruct
```

Artifacts: `artifacts/pbr_q4/s1_xy_qwen.{json,md}`. Large `.h95x` blobs are gitignored.

## Honesty

- Not a production mobile runtime.
- Not a claim that X/Y beats packed-K on real S1 tiles; the wire path is X/Y
  even when it is a few bytes larger (tile flags / predictor IDs).
- Do not invent BPW numbers — read them from the physical file after encode.
