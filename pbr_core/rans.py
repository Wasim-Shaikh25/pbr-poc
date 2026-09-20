"""Byte-renormalized 32-bit rANS for small alphabets (BF16 exponents).

Complete cost includes the frequency table. Bit-exact symbol roundtrip.
"""

from __future__ import annotations

import os
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


def rans_encode_python(symbols: np.ndarray, freq: np.ndarray) -> bytes:
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


def rans_encode_c(symbols: np.ndarray, freq: np.ndarray) -> bytes | None:
    lib = _c_lib()
    if lib is None or not hasattr(lib, "pbr_rans_encode"):
        return None
    sym = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
    n = int(sym.size)
    freq64 = np.ascontiguousarray(freq, dtype=np.int64)
    if freq64.size < 256:
        padded = np.zeros(256, dtype=np.int64)
        padded[: freq64.size] = freq64
        freq64 = padded
    cumul = _cumul(freq64)
    freq32 = np.ascontiguousarray(freq64, dtype=np.uint32)
    cumul32 = np.ascontiguousarray(cumul, dtype=np.uint32)
    # 4 + n + slack for renormalization bursts
    out_cap = 4 + n + max(n // 8, 64) + 1024
    out = np.empty(out_cap, dtype=np.uint8)
    import ctypes

    rc = lib.pbr_rans_encode(
        sym.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n),
        freq32.ctypes.data_as(ctypes.c_void_p),
        cumul32.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(out_cap),
    )
    if rc < 0:
        return None
    return bytes(out[:rc])


def rans_encode(symbols: np.ndarray, freq: np.ndarray) -> bytes:
    if rans_impl() == "c":
        blob = rans_encode_c(symbols, freq)
        if blob is not None:
            return blob
    return rans_encode_python(symbols, freq)


def rans_decode_python(blob: bytes, count: int, freq: np.ndarray) -> np.ndarray:
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


_C_LIB = None
_C_LIB_KEY: tuple[float, float] | None = None


def _c_lib():
    """Compile/load librans. None if gcc/ctypes is unavailable."""
    global _C_LIB, _C_LIB_KEY
    import ctypes
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent
    src = root / "rans_fast.c"
    so = root / "_rans_fast.so"
    if not src.is_file():
        return None
    src_mtime = src.stat().st_mtime
    so_mtime = so.stat().st_mtime if so.is_file() else 0.0
    key = (src_mtime, so_mtime)
    if _C_LIB is not None and _C_LIB_KEY == key and so_mtime >= src_mtime:
        return _C_LIB
    need = (not so.is_file()) or so_mtime < src_mtime
    if need:
        gcc = os.environ.get("CC", "gcc")
        base = [gcc, "-O3", "-std=c99", "-shared", "-fPIC", "-funroll-loops"]
        cmds = [base + ["-march=native", "-o", str(so), str(src)], base + ["-o", str(so), str(src)]]
        compiled = False
        for cmd in cmds:
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                compiled = True
                break
            except (OSError, subprocess.CalledProcessError):
                continue
        if not compiled:
            _C_LIB = None
            _C_LIB_KEY = None
            return None
        so_mtime = so.stat().st_mtime
        key = (src_mtime, so_mtime)
    try:
        lib = ctypes.CDLL(str(so))
    except OSError:
        _C_LIB = None
        _C_LIB_KEY = None
        return None
    try:
        lib.pbr_rans_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.pbr_rans_decode.restype = ctypes.c_int
    except AttributeError:
        _C_LIB = None
        _C_LIB_KEY = None
        return None
    if hasattr(lib, "pbr_rans_decode_u32"):
        lib.pbr_rans_decode_u32.argtypes = list(lib.pbr_rans_decode.argtypes)
        lib.pbr_rans_decode_u32.restype = ctypes.c_int
    if hasattr(lib, "pbr_join_bf16"):
        lib.pbr_join_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.pbr_join_bf16.restype = ctypes.c_int
    if hasattr(lib, "pbr_huffman_decode"):
        lib.pbr_huffman_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.pbr_huffman_decode.restype = ctypes.c_int
    if hasattr(lib, "pbr_decode_exp_rans_tile"):
        lib.pbr_decode_exp_rans_tile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.pbr_decode_exp_rans_tile.restype = ctypes.c_int
    if hasattr(lib, "pbr_rans_encode"):
        lib.pbr_rans_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.pbr_rans_encode.restype = ctypes.c_int
    _C_LIB = lib
    _C_LIB_KEY = key
    return lib


def rans_impl() -> str:
    forced = os.environ.get("PBR_RANS_IMPL", "").strip().lower()
    if forced in {"python", "py"}:
        return "python"
    if _c_lib() is not None:
        return "c"
    return "python"


def rans_decode_c(blob: bytes, count: int, freq: np.ndarray) -> np.ndarray:
    lib = _c_lib()
    if lib is None:
        return rans_decode_python(blob, count, freq)
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    if len(blob) < 4:
        raise ValueError("rANS blob too short")
    freq64 = np.ascontiguousarray(freq, dtype=np.int64)
    if freq64.size < 256:
        padded = np.zeros(256, dtype=np.int64)
        padded[: freq64.size] = freq64
        freq64 = padded
    cumul = _cumul(freq64)
    lut = np.ascontiguousarray(_symbol_lut(freq64), dtype=np.uint8)
    freq32 = np.ascontiguousarray(freq64, dtype=np.uint32)
    cumul32 = np.ascontiguousarray(cumul, dtype=np.uint32)
    out = np.empty(count, dtype=np.uint8)
    blob_u8 = np.frombuffer(memoryview(blob), dtype=np.uint8)
    if not blob_u8.flags.c_contiguous:
        blob_u8 = np.ascontiguousarray(blob_u8)
    import ctypes

    fn = getattr(lib, "pbr_rans_decode_u32", None) or lib.pbr_rans_decode
    freq_p = freq32 if fn is getattr(lib, "pbr_rans_decode_u32", None) else freq64
    cumul_p = cumul32 if fn is getattr(lib, "pbr_rans_decode_u32", None) else cumul
    rc = fn(
        blob_u8.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(int(blob_u8.size)),
        ctypes.c_int(int(count)),
        freq_p.ctypes.data_as(ctypes.c_void_p),
        cumul_p.ctypes.data_as(ctypes.c_void_p),
        lut.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError(f"pbr_rans_decode rc={rc}")
    return out


def join_bf16_u16(exp: np.ndarray, packed_sm: np.ndarray) -> np.ndarray | None:
    """C join of exponent + packed SM into uint16 BF16 words. None if no lib."""
    lib = _c_lib()
    if lib is None or not hasattr(lib, "pbr_join_bf16"):
        return None
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    sm = np.ascontiguousarray(packed_sm, dtype=np.uint8).ravel()
    n = int(e.size)
    if int(sm.size) != n:
        raise ValueError("join_bf16 length mismatch")
    out = np.empty(n, dtype=np.uint16)
    import ctypes

    rc = lib.pbr_join_bf16(
        e.ctypes.data_as(ctypes.c_void_p),
        sm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    if rc != 0:
        return None
    return out


def huffman_decode_c(
    data: bytes,
    count: int,
    max_len: int,
    lut_sym: np.ndarray,
    lut_nbits: np.ndarray,
) -> np.ndarray | None:
    lib = _c_lib()
    if lib is None or not hasattr(lib, "pbr_huffman_decode"):
        return None
    packed = (lut_sym.astype(np.uint16) & np.uint16(0xFF)) | (lut_nbits.astype(np.uint16) << 8)
    packed = np.ascontiguousarray(packed, dtype=np.uint16)
    buf = np.frombuffer(memoryview(data), dtype=np.uint8)
    if not buf.flags.c_contiguous:
        buf = np.ascontiguousarray(buf)
    out = np.empty(count, dtype=np.uint8)
    import ctypes

    rc = lib.pbr_huffman_decode(
        buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(int(buf.size)),
        ctypes.c_int(int(count)),
        ctypes.c_int(int(max_len)),
        packed.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    if rc != 0:
        return None
    return out


def decode_exp_rans_tile_c(payload: bytes | memoryview | np.ndarray, n: int) -> np.ndarray | None:
    """Fused C: parse PBR-E exp-rANS tile payload → uint16 words. None if unavailable."""
    lib = _c_lib()
    if lib is None or not hasattr(lib, "pbr_decode_exp_rans_tile") or rans_impl() != "c":
        return None
    if n < 0:
        raise ValueError("negative tile length")
    if n == 0:
        return np.zeros(0, dtype=np.uint16)
    if isinstance(payload, np.ndarray):
        buf = np.ascontiguousarray(payload, dtype=np.uint8).ravel()
    else:
        buf = np.frombuffer(memoryview(payload), dtype=np.uint8)
        if not buf.flags.c_contiguous:
            buf = np.ascontiguousarray(buf)
    out = np.empty(int(n), dtype=np.uint16)
    import ctypes

    rc = lib.pbr_decode_exp_rans_tile(
        buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(int(buf.size)),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    if rc != 0:
        return None
    return out


def rans_decode(blob: bytes, count: int, freq: np.ndarray) -> np.ndarray:
    if rans_impl() == "c":
        return rans_decode_c(blob, count, freq)
    return rans_decode_python(blob, count, freq)


def table_from_symbols(symbols: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
    counts = np.bincount(flat.astype(np.int64), minlength=256)
    return normalize_counts(counts)
