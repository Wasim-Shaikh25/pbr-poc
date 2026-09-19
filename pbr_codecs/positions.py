"""Sparse position list vs bitmap. Encoder picks the cheaper complete encoding."""

from __future__ import annotations

import math
import struct

import numpy as np


def position_width(n: int) -> int:
    if n <= 256:
        return 1
    if n <= 65536:
        return 2
    return 4


def pack_positions(positions: np.ndarray, n: int) -> bytes:
    width = position_width(n)
    count = int(positions.size)
    header = struct.pack("<I", count)
    if width == 1:
        return header + np.asarray(positions, dtype="<u1").tobytes()
    if width == 2:
        return header + np.asarray(positions, dtype="<u2").tobytes()
    return header + np.asarray(positions, dtype="<u4").tobytes()


def unpack_positions(data: bytes, n: int, offset: int = 0) -> tuple[np.ndarray, int]:
    (count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    width = position_width(n)
    nbytes = count * width
    blob = data[offset : offset + nbytes]
    if width == 1:
        pos = np.frombuffer(blob, dtype="<u1", count=count).astype(np.uint32)
    elif width == 2:
        pos = np.frombuffer(blob, dtype="<u2", count=count).astype(np.uint32)
    else:
        pos = np.frombuffer(blob, dtype="<u4", count=count).astype(np.uint32)
    return pos.copy(), offset + nbytes


def pack_bitmap(mask: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(mask, dtype=bool).ravel()
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_bitmap(data: bytes, n: int, offset: int = 0) -> tuple[np.ndarray, int]:
    nbytes = math.ceil(n / 8)
    bits = np.frombuffer(data[offset : offset + nbytes], dtype=np.uint8)
    mask = np.unpackbits(bits, bitorder="little", count=n).astype(bool)
    return mask, offset + nbytes


def list_bytes(n_exceptions: int, n: int) -> int:
    return 4 + n_exceptions * position_width(n)


def bitmap_bytes(n: int) -> int:
    return math.ceil(n / 8)


def choose_position_encoding(exception_mask: np.ndarray) -> str:
    n = int(exception_mask.size)
    n_exc = int(np.count_nonzero(exception_mask))
    if list_bytes(n_exc, n) <= bitmap_bytes(n):
        return "list"
    return "bitmap"
