"""Spatial predictors on 7-bit mantissas. Integer-only; no FP.

The Pre-Qwen Qualification Guide PDF was not on this VM. Rules follow the
user brief: LEFT, UP, PREVIOUS (V1); AVG (V2); Paeth and exp-conditioned LEFT (V3).

Unavailable neighbors (tile edge or not yet decoded in this traversal) are 0.
"""

from __future__ import annotations

PRED_RAW = 0
PRED_LEFT = 1
PRED_UP = 2
PRED_PREV = 3
PRED_AVG = 4
PRED_PAETH = 5
PRED_EXP_LEFT = 6

PRED_NAMES = {
    PRED_RAW: "RAW",
    PRED_LEFT: "LEFT",
    PRED_UP: "UP",
    PRED_PREV: "PREVIOUS",
    PRED_AVG: "AVG",
    PRED_PAETH: "PAETH",
    PRED_EXP_LEFT: "EXP_LEFT",
}


def paeth(left: int, up: int, up_left: int) -> int:
    """PNG Paeth on 7-bit values; selection uses signed int, not mod-128 wrap."""
    a = int(left) & 0x7F
    b = int(up) & 0x7F
    c = int(up_left) & 0x7F
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def predict(
    pred_id: int,
    *,
    left: int,
    up: int,
    up_left: int,
    prev: int,
    exp: int,
    left_exp: int,
    has_left: bool,
    has_up: bool,
    has_up_left: bool,
) -> int:
    """Causal predictor. ``has_*`` is false when the neighbor is not yet decoded."""
    if pred_id == PRED_RAW:
        return 0
    if pred_id == PRED_LEFT:
        return (int(left) & 0x7F) if has_left else 0
    if pred_id == PRED_UP:
        return (int(up) & 0x7F) if has_up else 0
    if pred_id == PRED_PREV:
        return int(prev) & 0x7F
    if pred_id == PRED_AVG:
        a = (int(left) & 0x7F) if has_left else 0
        b = (int(up) & 0x7F) if has_up else 0
        return ((a + b) >> 1) & 0x7F
    if pred_id == PRED_PAETH:
        a = (int(left) & 0x7F) if has_left else 0
        b = (int(up) & 0x7F) if has_up else 0
        c = (int(up_left) & 0x7F) if has_up_left else 0
        return paeth(a, b, c)
    if pred_id == PRED_EXP_LEFT:
        if has_left and (int(exp) & 0xFF) == (int(left_exp) & 0xFF):
            return int(left) & 0x7F
        return 0
    raise ValueError(f"unknown predictor {pred_id}")
