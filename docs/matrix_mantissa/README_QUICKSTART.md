# PBR Matrix Codec Pre-Qwen Test — Quick Start

1. Install: `python -m pip install -r requirements.txt`
2. Run: `python pbr_edge_test.py`
3. Require exit code 0 and review `pbr_edge_results.json`.
4. Complete the standalone decoder/corruption gates listed in the Word guide before a production claim.
5. Only then run a small representative Qwen subset.

The supplied suite tests 16,384 residual pairs, all 65,536 BF16 words, boundary lengths, special BF16 values, all V1–V3 matrix candidates, overflow boundaries, synthetic routing, determinism, and local framing.
