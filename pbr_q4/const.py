"""Wire IDs for the S1 + selective 256-node X/Y post-codec.

Product path: packed-K (existing S1 stream packing) is the default. A
matrix-family tile is emitted only when its *complete* physical cost —
payload + flag + that tile's share of the mode map — is strictly smaller
than packed-K of the true (unpadded) tile, and the tensor-level hybrid
blob is still strictly smaller than the whole-tensor packed stream.
"""

from __future__ import annotations

TILE = 16
NODES = TILE * TILE  # 256

# 0 is not a per-tile flag on the selective wire (packed tiles have no flag).
MODE_PACKED = 0
MODE_MATRIX = 1
MODE_RANS = 2
MODE_PLANE = 3
MODE_PAIR = 4
MODE_RUN = 5

MODE_NAMES = {
    MODE_MATRIX: "XY_MATRIX",
    MODE_RANS: "XY_RANS",
    MODE_PLANE: "XY_BITPLANE",
    MODE_PAIR: "XY_PAIR",
    MODE_RUN: "XY_RUN",
}

MATRIX_FAMILY_MODES = frozenset(MODE_NAMES)

PRED_PREVIOUS = 0
PRED_LEFT = 1
PRED_UP = 2
PRED_AVG = 3
PRED_PAETH = 4

PRED_NAMES = {
    PRED_PREVIOUS: "PREVIOUS",
    PRED_LEFT: "LEFT",
    PRED_UP: "UP",
    PRED_AVG: "AVG",
    PRED_PAETH: "PAETH",
}

TRAV_ROW = 0
TRAV_ROW_SERP = 1
TRAV_COL = 2
TRAV_COL_SERP = 3

TRAV_NAMES = {
    TRAV_ROW: "ROW",
    TRAV_ROW_SERP: "ROW_SERP",
    TRAV_COL: "COL",
    TRAV_COL_SERP: "COL_SERP",
}

# Tie-break order when complete bytes are equal (simpler first).
MODE_TIE_ORDER = (MODE_MATRIX, MODE_PLANE, MODE_RUN, MODE_PAIR, MODE_RANS)

# Search space: every predictor × every traversal (decoder-causal).
SEARCH_SPACE = tuple(
    (trav, pred)
    for trav in (TRAV_ROW, TRAV_ROW_SERP, TRAV_COL, TRAV_COL_SERP)
    for pred in (PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH)
)

FROZEN_S1_SHA = "eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de"
S1_V2_ACTUAL_BPW = 7.42082
S1_V2_FILE_BYTES = 458266017
S1_N_WEIGHTS = 494032768
PR15_ALL_TILE_BPW = 7.45389
PR15_ALL_TILE_FILE_BYTES = 460308254
PRIOR_HELDOUT_PROXY = 0.990

# Tile-map encodings (selective mantissa blob).
MAP_SPARSE = 0
MAP_BITMAP = 1
MAP_RLE = 2
