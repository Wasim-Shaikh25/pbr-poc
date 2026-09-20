# Blocker diagnosis (Qwen Stage 1B set)

Lossless BF16 only. Not a 1–2 GB / 8 GB claim.

Tensors: **42**. Words: **83836928**.

| field | weighted entropy (bits) |
| --- | ---: |
| H(uint16) | 10.5388 |
| H(exponent) | 2.6124 |
| H(mantissa) | 6.9719 |
| H(sign) | 1.0000 |
| H(uint16 prev_value residual) | 11.1121 |
| H(uint16 prev_row residual) | 11.1221 |
| H(exponent prev_value residual) | 3.1161 |
| H(exponent prev_row residual) | 3.1269 |

Tile-256 fraction where spatial residual entropy beats *raw uint16* (not vs exponent Huffman):

| | ideal (ignore meta) | complete container |
| --- | ---: | ---: |
| uint16 prev_value | 1.0000 | 0.0000 |
| exponent prev_value | 1.0000 | 1.0000 |

Cross-tensor exact duplicate tiles:

| tile | tiles | unique | dup rate | same-role dup rate |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1309952 | 1309952 | 0.000000 | 0.000000 |
| 256 | 327488 | 327488 | 0.000000 | 0.000000 |
| 1024 | 81872 | 81872 | 0.000000 | 0.000000 |

Mantissa entropy stays near 7 bits. On this Qwen sample, uint16 spatial residuals are *higher* than H(uint16), and exponent spatial residuals are *higher* than H(exp). Exact duplicate tiles are zero at 64/256/1024 (including same-role across layers). uint16 spatial beats raw only in the ideal no-codebook sense; complete container cost loses on every tile.
