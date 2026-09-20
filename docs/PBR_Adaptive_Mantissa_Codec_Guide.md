# Adaptive 7-bit mantissa codec — implementation note

The user-attached PDF
`PBR_Adaptive_Mantissa_Codec_Guide.pdf` was **not present** in this Cloud
Agent VM (searched uploads, `/home/box/agent-data/...`, and the named
content hash). This file records what the PoC implements so the run is
reproducible without the PDF.

Place the original PDF next to this file as
`docs/PBR_Adaptive_Mantissa_Codec_Guide.pdf` if you have it.

## Block (default 256 mantissas)

Candidates, each encoded, decoded, and compared on **complete** bytes
(payload + local padding):

1. **RAW** — 7 bits per mantissa. Mandatory fallback.
2. **Huffman + ESCAPE** — canonical Huffman trained on a freeze prefix
   (first 32 blocks). Unseen symbols: ESCAPE code + 7 raw bits.
3. **CONTEXT** — 1-bit hit / 7-bit miss with exact feedback.
   `predict(prev_m, exp, pos, rule)` in `pbr_adaptive_mantissa/predict.py`:
   copy-prev, local position, prev+1, exp&127, xor mix, integer average.
   The freeze region picks the rule with the highest hit rate.

## Tensor

Compare **ALL_RAW**, **ALL_HUFFMAN**, **ALL_CONTEXT**, **MIXED**
(per-block argmin). Shared Huffman table and 2-bit mode directory are
charged **once**, and **only if used**. ALL_RAW never serializes a codebook.

## Full BF16 bundle

`sign` raw 1-bit + `exp` rANS (freq table + stream) + adaptive mantissa.
`BPW = 8 × complete_bundle_bytes / n_words`.
