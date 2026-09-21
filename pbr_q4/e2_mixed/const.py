"""Constants for E2 mixed groupwise Q + E3 selective lossless tiles.

E2 is a *new* quantized reference (not H95Q-S1, not PR #17/#18).
Decode is bit-exact to this quantized reference, not to BF16.
"""

from __future__ import annotations

from pbr_q4.const import (
    MAP_BITMAP,
    MAP_RLE,
    MAP_SPARSE,
    MODE_PLANE,
    MODE_RANS,
    MODE_RUN,
    PRED_LEFT,
    PRED_PAETH,
    PRED_UP,
    TILE,
    TRAV_ROW,
)

MAGIC_E2 = b"E2MX"
MAGIC_E3 = b"E3MX"
VERSION = 1
CHECKSUM_SHA256 = 1
FLAGS_LITTLE_ENDIAN = 0x0001
ALIGN = 64

GROUP_SIZE = 128
CHANNEL_GROUP = 16  # structured protect: contiguous output-channel blocks

# Family body bits. Protected channels may additionally use Q8 / BF16.
BODY_BITS = (4, 5, 6)
PROTECT_BITS = (6, 8, 16)
NORM_BITS = (8, 16)

BIT_CODE = {4: 0, 5: 1, 6: 2, 8: 3, 16: 4}
CODE_BIT = {v: k for k, v in BIT_CODE.items()}

# E2b accuracy slack grid. Quality dominates: operating point is 0.0025.
E2B_EPSILONS = (0.0, 0.001, 0.0025, 0.005, 0.01)
E2B_EPS_DEFAULT = 0.0025

# E3 Phase-1 mode set — do not expand.
E3_PREDS = (PRED_LEFT, PRED_UP, PRED_PAETH)
E3_TRAV = TRAV_ROW
E3_CODECS = (MODE_PLANE, MODE_RUN, MODE_RANS)

# Net margin: matrix only if it saves at least this after complete costs.
E3_MIN_SAVE_BITS = 16
E3_MIN_SAVE_FRAC = 0.02

TARGET_PRACTICAL = 5.0
TARGET_STRETCH = 4.75
QUALITY_MIN = 0.95
QUALITY_AIM = 0.97

S1_V2_ACTUAL_BPW = 7.42082
S1_V2_FILE_BYTES = 458266017
S1_N_WEIGHTS = 494032768
PR17_Q6_BPW = 6.191189
PR17_Q6_HELDOUT = 0.967469
PR18_H95_BPW = 7.765222
PR18_H95_HELDOUT = 0.980986

DELTA_USEFUL = 0.03
DELTA_GOOD = 0.05
DELTA_STRONG = 0.10

NUM_LAYERS_QWEN_05B = 24
LATE_MLP_FROM = 16
