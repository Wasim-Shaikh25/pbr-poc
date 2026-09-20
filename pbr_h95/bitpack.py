"""K-bit mantissa / sign / exp stream packers for H95Q containers.

MSB-first within each symbol; streams padded to a byte boundary at the end.
Pure numpy — decode-only imports stay free of torch.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_CHUNK = 1 << 20


def pack_kbit(values: np.ndarray, k: int, *, chunk: int = _DEFAULT_CHUNK) -> bytes:
    if k < 0 or k > 16:
        raise ValueError(f"k must be 0..16, got {k}")
    if k == 0:
        return b""
    vals = np.ascontiguousarray(values, dtype=np.uint16).ravel()
    n = int(vals.size)
    if n == 0:
        return b""
    if k == 8:
        return vals.astype(np.uint8, copy=False).tobytes()
    if k == 1:
        return np.packbits((vals & np.uint16(1)).astype(np.uint8), bitorder="big").tobytes()

    mask = np.uint16((1 << k) - 1)
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint16)
    out = bytearray()
    leftover = np.zeros(0, dtype=np.uint8)
    for start in range(0, n, chunk):
        block = vals[start : start + chunk] & mask
        bits = ((block[:, None] >> shifts[None, :]) & np.uint16(1)).astype(np.uint8).reshape(-1)
        if leftover.size:
            bits = np.concatenate([leftover, bits])
        nbytes = int(bits.size) // 8
        if nbytes:
            out.extend(np.packbits(bits[: nbytes * 8], bitorder="big").tobytes())
        leftover = bits[nbytes * 8 :]
    if leftover.size:
        out.extend(np.packbits(leftover, bitorder="big").tobytes())
    return bytes(out)


def unpack_kbit(data: bytes, n: int, k: int) -> np.ndarray:
    if k < 0 or k > 16:
        raise ValueError(f"k must be 0..16, got {k}")
    if n < 0:
        raise ValueError(n)
    if k == 0:
        return np.zeros(n, dtype=np.uint16)
    if n == 0:
        return np.zeros(0, dtype=np.uint16)
    need = (n * k + 7) // 8
    if len(data) < need:
        raise ValueError(f"short bitstream: need {need} bytes, got {len(data)}")
    if k == 8:
        return np.frombuffer(memoryview(data)[:n], dtype=np.uint8).astype(np.uint16).copy()
    arr = np.frombuffer(memoryview(data)[:need], dtype=np.uint8)
    all_bits = np.unpackbits(arr, bitorder="big")[: n * k].reshape(n, k)
    out = np.zeros(n, dtype=np.uint16)
    for i in range(k):
        out |= all_bits[:, i].astype(np.uint16) << np.uint16(k - 1 - i)
    return out


def pack_kept_mantissas(mant_full: np.ndarray, keep_bits: int) -> bytes:
    if keep_bits == 0:
        return b""
    if keep_bits < 0 or keep_bits > 7:
        raise ValueError(f"keep_bits must be 0..7, got {keep_bits}")
    m = np.asarray(mant_full, dtype=np.uint16).ravel()
    removed = 7 - keep_bits
    kept = (m >> np.uint16(removed)) & np.uint16((1 << keep_bits) - 1)
    return pack_kbit(kept, keep_bits)


def unpack_kept_mantissas(data: bytes, n: int, keep_bits: int) -> np.ndarray:
    return unpack_kbit(data, n, keep_bits)


def pack_sign_bitplane(sign: np.ndarray) -> bytes:
    s = np.asarray(sign, dtype=np.uint8).ravel() & np.uint8(1)
    return np.packbits(s, bitorder="big").tobytes()


def unpack_sign_bitplane(data: bytes, n: int) -> np.ndarray:
    need = (n + 7) // 8
    if len(data) < need:
        raise ValueError(f"short sign plane: need {need}, got {len(data)}")
    bits = np.unpackbits(np.frombuffer(memoryview(data)[:need], dtype=np.uint8), bitorder="big")
    return bits[:n].astype(np.uint8)


def pack_exp_u8(exp: np.ndarray) -> bytes:
    return np.asarray(exp, dtype=np.uint8).ravel().tobytes()


def unpack_exp_u8(data: bytes, n: int) -> np.ndarray:
    if len(data) < n:
        raise ValueError(f"short exp stream: need {n}, got {len(data)}")
    return np.frombuffer(memoryview(data)[:n], dtype=np.uint8).copy()


def packed_bytes_for_k(n: int, k: int) -> int:
    if k <= 0 or n <= 0:
        return 0
    return (n * k + 7) // 8
