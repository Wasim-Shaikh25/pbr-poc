# pbr-poc

## Matrix mantissa codec V1–V3 (pre-Qwen qualification)

7-bit matrix mantissa codec with spatial predictors. **Do not claim Qwen
compression yet.** No ≤4 BPW product claim. Exact `uint16` / mantissa restore
only; complete serialized bytes include headers, directory, payloads, padding.

```bash
python3 -m pip install -e ".[dev]"
python3 pbr_edge_test.py
# or
python3 -m pytest tests/test_matrix_mantissa.py
```

| version | tiles | predictors | traversals |
| --- | --- | --- | --- |
| V1 | 16×16 | LEFT, UP, PREVIOUS | row, serpentine |
| V2 | 8×8, 8×16, 16×16, 16×32 | + AVG | + column |
| V3 | + 32×32 | + Paeth, exp-conditioned LEFT | + column-serpentine |

RAW 7-bit is the mandatory fallback and wins ties. Reports:
`artifacts/matrix_mantissa_edge_results.{json,md}`.
