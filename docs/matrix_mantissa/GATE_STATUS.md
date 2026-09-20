# Matrix Mantissa V1–V3 — gate status (user package + Gate D)

Source: user-supplied `pbr_poc.py` / `pbr_edge_test.py`, plus Gate D container/decoder on PR #1.

## Suite result

```
python3 pbr_edge_test.py
→ 10/10 PASS (Gate D included)
```

Report: `artifacts/pbr_edge_results.json`

| Gate | Status | Notes |
| --- | --- | --- |
| A–C (edge suite) | **PASS** | Residual / BF16 / pack / predictors / selection / determinism |
| D | **PASS** | Standalone `encode_tile_blob` / `decode_tile_blob`; rejects truncate, trailing bytes, bad CRC, bad mode/dims/ids, corrupt zlib |

## Explicit non-claims

- No Qwen compression numbers in this PR
- No ≤4 BPW claim
- Uniform random → raw fallback (honest)

## Run

```bash
pip install numpy
python3 pbr_edge_test.py
```
