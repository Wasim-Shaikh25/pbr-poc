# Matrix mantissa codec V1–V3 — implementation note

The user-attached **Pre-Qwen Qualification Guide (Matrix Mantissa Codec)**
PDF was **not present** on this Cloud Agent VM. This file records the
implemented contract so gates are reproducible without the PDF.

This work is **pre-Qwen**. Do not claim Qwen compression. No ≤4 BPW claim.

## Field

BF16 words stay `uint16`. Sign / exponent / 7-bit mantissa are split with
shifts and masks only (no FP32). Residuals are **modulo 128**:

```
res  = (actual - pred) mod 128
rec  = (pred + res)   mod 128
```

RAW stores the 7-bit mantissa (mandatory fallback) and **wins ties**.

## V1

- Tiles: 16×16 (edge tails keep actual `h×w`)
- Predictors: LEFT, UP, PREVIOUS
- Traversals: row, row-serpentine
- Unavailable neighbors (tile edge or not yet decoded) predict 0

## V2

V1 plus:

- Tiles: 8×8, 8×16, 16×16, 16×32
- Predictor: AVG = `(LEFT+UP)//2`
- Traversal: column

## V3

V2 plus:

- Tiles: 32×32
- Predictor: Paeth (PNG selection on integers)
- Predictor: exp-conditioned LEFT (LEFT only when `exp == left_exp`)
- Traversal: column-serpentine

## Selection

For each allowed tile shape, every tile encodes **all** legal candidates,
decodes them, and keeps min **complete** bytes (12-byte tile header +
payload). Residual payloads: ZERO, CONST, k-bit PACK, SPARSE (default 0).
A whole-plane ALL_RAW blob (no tile directory, no codebook, no exp flag)
competes once; unused dependencies are omitted.

## Gates

See `pbr_edge_test.py` → `artifacts/matrix_mantissa_edge_results.md`.
