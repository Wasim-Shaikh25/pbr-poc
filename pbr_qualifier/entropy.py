"""Cheap uint16 / component / residual entropy estimates."""

from __future__ import annotations

import numpy as np

from pbr_codecs.xor_predictor import residuals_prev_row, residuals_prev_value
from pbr_core.bf16 import split_components


def shannon_entropy(words: np.ndarray) -> float:
    flat = np.ascontiguousarray(words).ravel()
    if flat.size == 0:
        return 0.0
    _unique, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-(probs * np.log2(probs)).sum())


def component_entropy(words: np.ndarray) -> dict[str, float]:
    sign, exp, mant = split_components(words)
    return {
        "sign_bits": shannon_entropy(sign),
        "exponent_bits": shannon_entropy(exp),
        "mantissa_bits": shannon_entropy(mant),
        "word_bits": shannon_entropy(words),
    }


def residual_entropy(words: np.ndarray) -> dict[str, float]:
    matrix = np.ascontiguousarray(words, dtype=np.uint16)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return {
        "prev_value_bits": shannon_entropy(residuals_prev_value(matrix)),
        "prev_row_bits": shannon_entropy(residuals_prev_row(matrix)),
    }


def unique_coverage(words: np.ndarray, k: int = 4) -> dict[str, float | int]:
    flat = np.ascontiguousarray(words).ravel()
    if flat.size == 0:
        return {"unique": 0, "top1_coverage": 1.0, "top4_coverage": 1.0}
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    top1 = float(counts[order[0]] / flat.size)
    topk = float(counts[order][:k].sum() / flat.size)
    return {
        "unique": int(unique.size),
        "top1_coverage": top1,
        "top4_coverage": topk,
    }


def is_high_entropy(word_bits: float, residual_bits: float) -> bool:
    """Hopeless-enough to skip extra searches; still sample-encode once."""
    return word_bits >= 15.5 and residual_bits >= 15.0
