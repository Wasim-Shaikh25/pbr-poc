"""Byte-renormalized 32-bit rANS for small alphabets (BF16 exponents).

Complete cost includes the frequency table. Bit-exact symbol roundtrip.
"""

from __future__ import annotations

import struct

import numpy as np

RANS_L = 1 << 23
SCALE_BITS = 12
M = 1 << SCALE_BITS  # 4096


def normalize_counts(counts: np.ndarray, total: int = M) -> np.ndarray:
    """Map raw counts onto a table that sums to ``total``; keep support."""
    raw = np.zeros(256, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64).ravel()
    n = min(int(c.size), 256)
    raw[:n] = c[:n]
    s = int(raw.sum())
    freq = np.zeros(256, dtype=np.int64)
    if s <= 0:
        freq[0] = total
        return freq
    freq = (raw * total) // s
    zeros = (raw > 0) & (freq == 0)
    freq[zeros] = 1
    diff = total - int(freq.sum())
    if diff != 0:
        order = np.argsort(-raw)
        i = 0
        while diff != 0 and i < 256:
            idx = int(order[i % 256])
            if raw[idx] == 0 and freq[idx] == 0:
                i += 1
                continue
            if diff > 0:
                freq[idx] += 1
                diff -= 1
            elif freq[idx] > 1:
                freq[idx] -= 1
                diff += 1
            i += 1
        if diff != 0:
            # Last-resort: dump remainder on a used bin.
            used = int(np.argmax(freq))
            freq[used] += diff
    return freq


def _cumul(freq: np.ndarray) -> np.ndarray:
    c = np.zeros(257, dtype=np.int64)
    c[1:] = np.cumsum(freq)
    return c


def _symbol_lut(freq: np.ndarray) -> np.ndarray:
    lut = np.empty(M, dtype=np.uint8)
    pos = 0
    for s in range(256):
        f = int(freq[s])
        if f:
            lut[pos : pos + f] = s
            pos += f
    if pos < M:
        lut[pos:] = lut[pos - 1] if pos else 0
    return lut


def dump_freq_table(freq: np.ndarray) -> bytes:
    items = [(i, int(freq[i])) for i in range(256) if int(freq[i]) > 0]
    blob = struct.pack("<HH", SCALE_BITS, len(items))
    for sym, f in items:
        blob += struct.pack("<BH", sym, f)
    return blob


def load_freq_table(data: bytes, offset: int = 0) -> tuple[np.ndarray, int]:
    scale, count = struct.unpack_from("<HH", data, offset)
    offset += 4
    if scale != SCALE_BITS:
        raise ValueError(f"rANS scale_bits {scale} != {SCALE_BITS}")
    freq = np.zeros(256, dtype=np.int64)
    for _ in range(count):
        sym, f = struct.unpack_from("<BH", data, offset)
        offset += 3
        freq[sym] = f
    if int(freq.sum()) != M:
        raise ValueError(f"rANS freq sum {int(freq.sum())} != {M}")
    return freq, offset


def rans_encode(symbols: np.ndarray, freq: np.ndarray) -> bytes:
    """Encode ``symbols`` (uint8). Layout: little-endian state then overflow bytes."""
    cumul = _cumul(freq)
    overflow = bytearray()
    x = RANS_L
    # Iterate the compact byte buffer so large tensors do not materialize a
    # Python list of symbols (embedding-scale streams are 1e8+ bytes).
    raw = np.ascontiguousarray(symbols, dtype=np.uint8).ravel().tobytes()
    for s in reversed(raw):
        f = int(freq[s])
        start = int(cumul[s])
        x_max = ((RANS_L >> SCALE_BITS) << 8) * f
        while x >= x_max:
            overflow.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + start
    state = struct.pack("<I", x & 0xFFFFFFFF)
    overflow.reverse()
    return state + bytes(overflow)


def rans_decode(blob: bytes, count: int, freq: np.ndarray) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    cumul = _cumul(freq)
    lut = _symbol_lut(freq)
    if len(blob) < 4:
        raise ValueError("rANS blob too short")
    x = struct.unpack_from("<I", blob, 0)[0]
    pos = 4
    nblob = len(blob)
    out = np.empty(count, dtype=np.uint8)
    mask = M - 1
    for i in range(count):
        cf = x & mask
        s = int(lut[cf])
        f = int(freq[s])
        start = int(cumul[s])
        x = f * (x >> SCALE_BITS) + cf - start
        out[i] = s
        while x < RANS_L and pos < nblob:
            x = (x << 8) | blob[pos]
            pos += 1
    return out


def table_from_symbols(symbols: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
    counts = np.bincount(flat.astype(np.int64), minlength=256)
    return normalize_counts(counts)
