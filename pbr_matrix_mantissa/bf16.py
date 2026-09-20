"""BF16 field split/join without floating-point conversion."""

from __future__ import annotations

import numpy as np

SIGN_SHIFT = 15
EXP_SHIFT = 7
EXP_MASK = 0xFF
MANT_MASK = 0x7F


def view_uint16(array: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype.itemsize != 2:
        raise TypeError(
            f"need a 16-bit dtype, got {arr.dtype}; refusing FP32 conversion"
        )
    return arr.view(np.uint16)


def split_components(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = view_uint16(words)
    sign = (w >> np.uint16(SIGN_SHIFT)).astype(np.uint8)
    exp = ((w >> np.uint16(EXP_SHIFT)) & np.uint16(EXP_MASK)).astype(np.uint8)
    mant = (w & np.uint16(MANT_MASK)).astype(np.uint8)
    return sign, exp, mant


def join_components(
    sign: np.ndarray, exponent: np.ndarray, mantissa: np.ndarray
) -> np.ndarray:
    s = np.asarray(sign, dtype=np.uint16) & np.uint16(1)
    e = np.asarray(exponent, dtype=np.uint16) & np.uint16(EXP_MASK)
    m = np.asarray(mantissa, dtype=np.uint16) & np.uint16(MANT_MASK)
    return (s << np.uint16(SIGN_SHIFT)) | (e << np.uint16(EXP_SHIFT)) | m


def special_payload_words() -> np.ndarray:
    return np.array(
        [
            0x0000,  # +0
            0x8000,  # -0
            0x7F80,  # +Inf
            0xFF80,  # -Inf
            0x7FC1,  # qNaN payload
            0x7F81,  # another NaN
            0x0001,  # +subnormal
            0x8001,  # -subnormal
        ],
        dtype=np.uint16,
    )
