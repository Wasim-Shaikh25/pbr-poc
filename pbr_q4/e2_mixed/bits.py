"""BF16 views, group ids, and packed-bit estimators for mixed groupwise Q."""

from __future__ import annotations

import numpy as np

from pbr_core.safetensors_io import logical_2d_shape
from pbr_q4.e2_mixed.const import GROUP_SIZE


def bf16_u16_to_f32(words: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(words, dtype=np.uint16)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 uint16 (nearest-even)."""
    x = np.ascontiguousarray(values, dtype=np.float32)
    u32 = x.view(np.uint32)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = u32.astype(np.uint64) + bias.astype(np.uint64)
    return (rounded >> np.uint64(16)).astype(np.uint16)


def qmax_for_bits(bits: int) -> int:
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be 2..8, got {bits}")
    return (1 << (bits - 1)) - 1


def as_rank2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    a = np.ascontiguousarray(arr)
    orig = tuple(int(x) for x in a.shape)
    rows, cols = logical_2d_shape(orig)
    if a.size != rows * cols:
        raise ValueError(f"shape {orig} is not {rows}x{cols}")
    return a.reshape(rows, cols), orig, rows, cols


def group_ids(rows: int, cols: int, group_size: int = GROUP_SIZE) -> tuple[np.ndarray, int, int]:
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


def n_groups_for(rows: int, cols: int, group_size: int = GROUP_SIZE) -> int:
    if rows <= 0 or cols <= 0:
        return 0
    n_gc = (cols + group_size - 1) // group_size
    return rows * n_gc


def packed_field_bits(
    n_weights: int,
    bits: int,
    n_groups: int,
    *,
    include_scales: bool = True,
) -> int:
    """Physical packed-field bits for one uniform-bit block (no container header)."""
    if bits >= 16:
        return int(n_weights) * 16
    code_bits = int(n_weights) * int(bits)
    if not include_scales:
        return code_bits
    return code_bits + int(n_groups) * (16 + 8)


def mse_scale(bits: int) -> float:
    """Relative quantization-MSE proxy (smaller is better). BF16 → 0."""
    if bits >= 16:
        return 0.0
    q = (1 << int(bits)) - 1
    return 1.0 / float(q * q)
