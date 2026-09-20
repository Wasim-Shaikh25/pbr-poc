# Blocker diagnosis (unsloth/Llama-3.2-1B-Instruct)

Lossless BF16 only. Not a 1–2 GB / 8 GB claim.

Tensors: **11**. Words: **83886080**.

| field | weighted entropy (bits) |
| --- | ---: |
| H(uint16) | 10.5090 |
| H(exponent) | 2.5959 |
| H(mantissa) | 6.9661 |
| H(sign) | 1.0000 |
| H(uint16 prev_value residual) | 11.0997 |
| H(uint16 prev_row residual) | 11.1004 |
| H(exponent prev_value residual) | 3.1021 |
| H(exponent prev_row residual) | 3.1031 |

Tile-256 fraction where spatial residual entropy beats *raw uint16* (not vs exponent Huffman):

| | ideal (ignore meta) | complete container |
| --- | ---: | ---: |
| uint16 prev_value | 1.0000 | 0.0000 |
| exponent prev_value | 1.0000 | 1.0000 |

Cross-tensor exact duplicate tiles:

| tile | tiles | unique | dup rate | same-role dup rate |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1310720 | 1310720 | 0.000000 | 0.000000 |
| 256 | 327680 | 327680 | 0.000000 | 0.000000 |
| 1024 | 81920 | 81920 | 0.000000 | 0.000000 |

On this Llama-3.2-1B sample, uint16 spatial residuals are *higher* than H(uint16), and exponent spatial residuals are *higher* than H(exp). Exact duplicate tiles are zero at 64/256/1024 (including same-role across layers). uint16 spatial beats raw only in the ideal no-codebook sense; complete container cost loses on every tile.
