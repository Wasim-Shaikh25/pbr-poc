# Matrix Mantissa V1–V3 — gate status (user package)

Source: user-supplied package (`pbr_poc.py`, `pbr_edge_test.py`), not the cloud-agent reconstruction.

## Suite result (local, 2026-09-20 IST)

```
python3 pbr_edge_test.py
→ 10/10 PASS
```

Report: `artifacts/pbr_edge_results.json`

| Gate | Status | Notes |
| --- | --- | --- |
| A–C (edge suite) | **PASS** | Residual / BF16 / pack / predictors / selection / determinism / framing |
| D | **still REQUIRED** | Standalone corruption decoder not in this PoC; suite explicitly flags it |

## Explicit non-claims

- No Qwen compression numbers in this PR
- No ≤4 BPW claim
- Uniform random → raw fallback (honest)

## Run

```bash
pip install numpy
python3 pbr_edge_test.py
```
