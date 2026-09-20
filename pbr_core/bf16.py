"""BF16 bit-pattern helpers that never convert through FP32.

PBR archives the exact 16-bit word. Sign, exponent, mantissa,
signed zero, subnormals, infinities, and NaN payloads must survive.
"""

from __future__ import annotations

import numpy as np

BF16_DTYPE_TAG = "bf16_uint16"

# IEEE-like BF16 layout: 1 sign, 8 exponent, 7 mantissa.
SIGN_SHIFT = 15
EXP_SHIFT = 7
EXP_MASK = 0xFF
MANT_MASK = 0x7F


def view_uint16(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous uint16 view of *array* without numeric conversion.

    Accepts uint16 already, or any dtype whose itemsize is 2 (e.g. float16).
    BF16 stored as a 2-byte dtype is viewed, never cast through float32.
    """
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype.itemsize != 2:
        raise TypeError(
            f"PBR archival path requires a 16-bit dtype, got {arr.dtype}. "
            "Refusing to convert through a wider float type."
        )
    return arr.view(np.uint16)


def make_bf16_bits(
    sign: np.ndarray | int,
    exponent: np.ndarray | int,
    mantissa: np.ndarray | int,
) -> np.ndarray:
    """Compose BF16 words from integer component fields (no floating math)."""
    s = np.asarray(sign, dtype=np.uint16) & np.uint16(1)
    e = np.asarray(exponent, dtype=np.uint16) & np.uint16(EXP_MASK)
    m = np.asarray(mantissa, dtype=np.uint16) & np.uint16(MANT_MASK)
    return (s << np.uint16(SIGN_SHIFT)) | (e << np.uint16(EXP_SHIFT)) | m


def split_components(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split uint16 BF16 words into sign, exponent, mantissa fields."""
    w = view_uint16(words)
    sign = (w >> np.uint16(SIGN_SHIFT)).astype(np.uint8)
    exp = ((w >> np.uint16(EXP_SHIFT)) & np.uint16(EXP_MASK)).astype(np.uint8)
    mant = (w & np.uint16(MANT_MASK)).astype(np.uint8)
    return sign, exp, mant


def join_components(
    sign: np.ndarray, exponent: np.ndarray, mantissa: np.ndarray
) -> np.ndarray:
    return make_bf16_bits(sign, exponent, mantissa)


def pack_sign_mantissa(sign: np.ndarray, mantissa: np.ndarray) -> np.ndarray:
    """Pack sign (MSB) + 7-bit mantissa into one byte per weight."""
    s = np.asarray(sign, dtype=np.uint8) & np.uint8(1)
    m = np.asarray(mantissa, dtype=np.uint8) & np.uint8(MANT_MASK)
    return (s << np.uint8(7)) | m


def unpack_sign_mantissa(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(packed, dtype=np.uint8)
    return (p >> np.uint8(7)).astype(np.uint8), (p & np.uint8(MANT_MASK))


def special_payload_words() -> np.ndarray:
    """Canonical bit patterns that an FP32 detour would be likely to destroy."""
    return np.array(
        [
            0x0000,  # +0
            0x8000,  # -0
            0x7F80,  # +Inf
            0xFF80,  # -Inf
            0x7FC1,  # qNaN with nonzero payload
            0x7F81,  # another NaN payload
            0x0001,  # +subnormal
            0x8001,  # -subnormal
        ],
        dtype=np.uint16,
    )
