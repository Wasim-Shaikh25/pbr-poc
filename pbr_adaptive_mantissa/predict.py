"""CONTEXT predictors for the adaptive 7-bit mantissa codec.

The user-attached implementation-guide PDF was not present on this agent VM
(``docs/PBR_Adaptive_Mantissa_Codec_Guide.pdf``). These rules follow the brief:
predict from previous **exact** mantissa, exponent, and local block position.
Integer-only; no FP rounding. Exact feedback uses already-decoded mantissas.
"""

from __future__ import annotations

import numpy as np

N_RULES = 6
RULE_COPY = 0
RULE_POS = 1
RULE_RAMP = 2
RULE_EXP = 3
RULE_XOR = 4
RULE_AVG = 5

RULE_NAMES = {
    RULE_COPY: "copy_prev",
    RULE_POS: "local_pos",
    RULE_RAMP: "prev_plus_one",
    RULE_EXP: "exp_low7",
    RULE_XOR: "prev_xor_exp_xor_pos",
    RULE_AVG: "avg_prev_exp",
}


def predict(prev_m: int, exp: int, pos: int, rule: int) -> int:
    """Scalar predictor. ``pos`` is the local index inside the 256-wide block."""
    prev_m = int(prev_m) & 0x7F
    exp7 = int(exp) & 0x7F
    pos7 = int(pos) & 0x7F
    if rule == RULE_COPY:
        return prev_m
    if rule == RULE_POS:
        return pos7
    if rule == RULE_RAMP:
        return (prev_m + 1) & 0x7F
    if rule == RULE_EXP:
        return exp7
    if rule == RULE_XOR:
        return (prev_m ^ exp7 ^ pos7) & 0x7F
    if rule == RULE_AVG:
        return ((prev_m + exp7) >> 1) & 0x7F
    raise ValueError(f"unknown CONTEXT rule {rule}")


def predict_vec(prev_m: np.ndarray, exp: np.ndarray, pos: np.ndarray, rule: int) -> np.ndarray:
    """Vectorized ``predict`` (same integer rules)."""
    prev = np.ascontiguousarray(prev_m, dtype=np.uint8).ravel() & np.uint8(0x7F)
    exp7 = np.ascontiguousarray(exp, dtype=np.uint8).ravel() & np.uint8(0x7F)
    pos7 = np.ascontiguousarray(pos, dtype=np.uint8).ravel() & np.uint8(0x7F)
    if prev.size != exp7.size or prev.size != pos7.size:
        raise ValueError("predict_vec length mismatch")
    if rule == RULE_COPY:
        return prev
    if rule == RULE_POS:
        return pos7
    if rule == RULE_RAMP:
        return (prev + np.uint8(1)) & np.uint8(0x7F)
    if rule == RULE_EXP:
        return exp7
    if rule == RULE_XOR:
        return (prev ^ exp7 ^ pos7) & np.uint8(0x7F)
    if rule == RULE_AVG:
        s = prev.astype(np.uint16) + exp7.astype(np.uint16)
        return np.right_shift(s, 1).astype(np.uint8) & np.uint8(0x7F)
    raise ValueError(f"unknown CONTEXT rule {rule}")


def causal_prev(mant: np.ndarray) -> np.ndarray:
    """Exact previous mantissa in raster order; 0 at the first weight."""
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    prev = np.zeros_like(m)
    if m.size:
        prev[1:] = m[:-1]
    return prev


def local_pos(n: int, block_size: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n < 0")
    if block_size < 1:
        raise ValueError("block_size < 1")
    return (np.arange(n, dtype=np.int32) % int(block_size)).astype(np.uint8)


def pick_rule(
    mant: np.ndarray,
    exp: np.ndarray,
    block_size: int,
) -> tuple[int, float]:
    """Choose the freeze-region rule with the highest hit rate."""
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if m.size == 0:
        return RULE_COPY, 0.0
    prev = causal_prev(m)
    pos = local_pos(int(m.size), block_size)
    best_rule = RULE_COPY
    best_hit = -1.0
    n = float(m.size)
    for rule in range(N_RULES):
        pred = predict_vec(prev, e, pos, rule)
        hit = float(np.count_nonzero(pred == m)) / n
        if hit > best_hit:
            best_hit = hit
            best_rule = rule
    return best_rule, best_hit
