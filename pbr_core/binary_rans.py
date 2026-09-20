"""Binary rANS for causal bit streams with a per-bit p(1) (or a static p).

Used by mantissa-model exact codecs. Complete cost includes the 4-byte state.
Alphabet {0,1}; frequencies sum to M = 4096.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.rans import M, RANS_L, SCALE_BITS

_MIN_F = 1


def _freq_pair(p1: int) -> tuple[int, int]:
    f1 = int(np.clip(p1, _MIN_F, M - _MIN_F))
    return M - f1, f1


def p_to_freq1(p: np.ndarray | float) -> np.ndarray:
    """Map a probability in (0,1) onto a rANS bin count in 1..M-1."""
    arr = np.asarray(p, dtype=np.float64)
    f = np.rint(arr * M).astype(np.int64)
    return np.clip(f, _MIN_F, M - _MIN_F).astype(np.int64)


def binary_rans_encode(bits: np.ndarray, freq1: np.ndarray | int) -> bytes:
    """Encode bits (0/1). ``freq1`` is a scalar or one bin-count per bit."""
    flat = np.ascontiguousarray(bits, dtype=np.uint8).ravel()
    n = int(flat.size)
    if n == 0:
        return struct.pack("<I", RANS_L)
    if np.isscalar(freq1) or (isinstance(freq1, np.ndarray) and freq1.ndim == 0):
        f1 = np.full(n, int(freq1), dtype=np.int64)
    else:
        f1 = np.ascontiguousarray(freq1, dtype=np.int64).ravel()
        if f1.size != n:
            raise ValueError("freq1 length must match bits")
    f1 = np.clip(f1, _MIN_F, M - _MIN_F)
    overflow = bytearray()
    x = RANS_L
    raw = flat.tobytes()
    for i in range(n - 1, -1, -1):
        bit = raw[i] & 1
        f1i = int(f1[i])
        f0 = M - f1i
        f = f1i if bit else f0
        start = 0 if bit == 0 else f0
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            overflow.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + start
    overflow.reverse()
    return struct.pack("<I", x & 0xFFFFFFFF) + bytes(overflow)


def binary_rans_decode(blob: bytes, count: int, freq1: np.ndarray | int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    if np.isscalar(freq1) or (isinstance(freq1, np.ndarray) and freq1.ndim == 0):
        f1 = np.full(count, int(freq1), dtype=np.int64)
    else:
        f1 = np.ascontiguousarray(freq1, dtype=np.int64).ravel()
        if f1.size != count:
            raise ValueError("freq1 length must match count")
    f1 = np.clip(f1, _MIN_F, M - _MIN_F)
    x = struct.unpack_from("<I", blob, 0)[0]
    pos = 4
    nblob = len(blob)
    out = np.empty(count, dtype=np.uint8)
    mask = M - 1
    for i in range(count):
        cf = x & mask
        f1i = int(f1[i])
        f0 = M - f1i
        bit = 0 if cf < f0 else 1
        f = f1i if bit else f0
        start = 0 if bit == 0 else f0
        x = f * (x >> SCALE_BITS) + cf - start
        out[i] = bit
        while x < RANS_L and pos < nblob:
            x = (x << 8) | blob[pos]
            pos += 1
    return out
