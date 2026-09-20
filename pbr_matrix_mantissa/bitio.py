"""LSB-first bit packing (1–7 bit symbols)."""

from __future__ import annotations

import numpy as np


def ceil_bytes(nbits: int) -> int:
    return (int(nbits) + 7) // 8


def pack_lsb(values: np.ndarray, nbits: int) -> bytes:
    if nbits < 1 or nbits > 7:
        raise ValueError(f"nbits must be 1..7, got {nbits}")
    v = np.ascontiguousarray(values, dtype=np.uint32).ravel()
    n = int(v.size)
    if n == 0:
        return b""
    mask = (1 << nbits) - 1
    acc = 0
    filled = 0
    out = bytearray()
    for x in v.tolist():
        acc |= (int(x) & mask) << filled
        filled += nbits
        while filled >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            filled -= 8
    if filled:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_lsb(data: bytes, count: int, nbits: int) -> np.ndarray:
    if nbits < 1 or nbits > 7:
        raise ValueError(f"nbits must be 1..7, got {nbits}")
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    need = ceil_bytes(count * nbits)
    if len(data) < need:
        raise ValueError("truncated bit-packed payload")
    mask = (1 << nbits) - 1
    out = np.empty(count, dtype=np.uint8)
    acc = 0
    filled = 0
    bi = 0
    for i in range(count):
        while filled < nbits:
            acc |= data[bi] << filled
            filled += 8
            bi += 1
        out[i] = acc & mask
        acc >>= nbits
        filled -= nbits
    return out
