# User Matrix Mantissa attachments — landed

The user-supplied Matrix Mantissa package is now on this branch. Files
were copied **unchanged** from this Cloud Agent’s uploads. The codec was
**not reinvented**.

## Mapping (10/10 files)

| upload | repo path |
| --- | --- |
| `pbr_poc.py` | `pbr_poc.py` (byte-identical; sha256 `5ea6069d…d8bb46fe`) |
| `pbr_edge_test.py` | `pbr_edge_test.py` (byte-identical; sha256 `a6f16719…8389bff7`) |
| `pbr_edge_results.json` | `artifacts/pbr_edge_results.json` and repo-root copy |
| `EDGE_TEST_EXECUTION.txt` | `docs/EDGE_TEST_EXECUTION.txt` |
| `README_QUICKSTART.md` | `docs/README_QUICKSTART.md` |
| `GATE_STATUS.md` | `docs/GATE_STATUS.md` |
| `README.md` | root `README.md` |
| `requirements.txt` | `requirements.txt` |
| `PBR_Matrix_Mantissa_PreQwen_Guide.docx` | `docs/PBR_Matrix_Mantissa_PreQwen_Guide.docx` |

## Verification (local)

```
pip install -r requirements.txt
python3 pbr_edge_test.py
→ 10/10 PASS, exit 0
```

## Gates

| Gate | Status |
| --- | --- |
| A–C (user edge suite) | **PASS** (10/10) |
| D | **still REQUIRED** (standalone decoder / corruption rejection is not in this PoC; the suite flags it) |

## Explicit non-claims

- No Qwen compression numbers
- No ≤4 BPW claim
- Uniform random → raw fallback (honest)

Do **not merge** this PR; leave for the review bot.

PR #2 (`cursor/matrix-mantissa-v1-v3-a89e`) is a separate reconstruction
and is not a substitute for this user package — leave it alone.
