# pbr-poc — Matrix Mantissa V1–V3 (pre-Qwen package)

User-supplied matrix mantissa codec + edge qualification suite.

## Quick start

```bash
pip install numpy
python3 pbr_edge_test.py
```

Exits non-zero on any failure. Writes `pbr_edge_results.json` in the cwd (also mirrored under `artifacts/`).

## Layout

| path | role |
| --- | --- |
| `pbr_poc.py` | V1–V3 matrix mantissa candidates |
| `pbr_edge_test.py` | Pre-Qwen edge suite |
| `artifacts/pbr_edge_results.json` | Last PASS report |
| `docs/GATE_STATUS.md` | Gate A–D summary |
| `docs/EDGE_TEST_EXECUTION.txt` | Prior execution log |
| `docs/README_QUICKSTART.md` | User quick start |
| `docs/PBR_Matrix_Mantissa_PreQwen_Guide.docx` | Qualification guide (if present) |

## Gates

See `docs/GATE_STATUS.md`. **Gate D (standalone corruption decoder) remains required** before any Qwen claim.
