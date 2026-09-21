"""BF16 ↔ float32 views and signed ↔ stored Q-code maps.

The quantized reference is BF16 uint16 words. Dequant produces float32,
then rounds to BF16 with the IEEE-style round-to-nearest-even used by
most BF16 toolchains (top 16 bits of FP32 plus a 7fff+lsb bias).
"""

from __future__ import annotations

import numpy as np

from pbr_core.safetensors_io import logical_2d_shape


def bf16_u16_to_f32(words: np.ndarray) -> np.ndarray:
    """Interpret BF16 uint16 words as float32 (lossless widen)."""
    u = np.ascontiguousarray(words, dtype=np.uint16)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 uint16 (nearest-even)."""
    x = np.ascontiguousarray(values, dtype=np.float32)
    u32 = x.view(np.uint32)
    # Preserve NaN / Inf payloads: just take the top 16 after rounding.
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    # uint32 add can wrap on NaN payloads near 0xFFFFFFFF; use uint64.
    rounded = u32.astype(np.uint64) + bias.astype(np.uint64)
    return (rounded >> np.uint64(16)).astype(np.uint16)


def qmax_for_bits(bits: int) -> int:
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be 2..8, got {bits}")
    return (1 << (bits - 1)) - 1


def signed_to_stored(q: np.ndarray, bits: int) -> np.ndarray:
    """Map signed codes in [-qmax, qmax] to 0 .. 2*qmax (fits in *bits* bits)."""
    qmax = qmax_for_bits(bits)
    stored = np.asarray(q, dtype=np.int32) + qmax
    if np.any(stored < 0) or np.any(stored > 2 * qmax):
        raise ValueError("signed code out of range")
    return stored.astype(np.uint16)


def stored_to_signed(stored: np.ndarray, bits: int) -> np.ndarray:
    qmax = qmax_for_bits(bits)
    return np.asarray(stored, dtype=np.int32) - qmax


def as_rank2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    a = np.ascontiguousarray(arr)
    orig = tuple(int(x) for x in a.shape)
    rows, cols = logical_2d_shape(orig)
    if a.size != rows * cols:
        raise ValueError(f"shape {orig} is not {rows}x{cols}")
    return a.reshape(rows, cols), orig, rows, cols


def group_ids(rows: int, cols: int, group_size: int) -> tuple[np.ndarray, int, int]:
    """Return (gid[rows, cols], n_gc, n_groups) grouping along the last axis."""
    if group_size <= 0:
        raise ValueError(group_size)
    n_gc = (cols + group_size - 1) // group_size if cols else 0
    n_groups = rows * n_gc
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.int32), n_gc, n_groups
    cols_i = np.arange(cols, dtype=np.int32)
    gid = (np.arange(rows, dtype=np.int32)[:, None] * n_gc) + (cols_i // group_size)
    return gid, n_gc, n_groups
