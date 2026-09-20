"""PBR-EM: exponent rANS plus mantissa (or SM) entropy coding.

Standalone field codecs. Not on STAGE1A_CODECS. Complete bytes include
every table, stream, packed sign bits, and one tile header. Bit-exact.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.bf16_exp_huffman import PRED_NONE, PRED_PREV_EXP, _exp_from_residuals, _exp_residuals
from pbr_core.bf16 import join_components, pack_sign_mantissa, split_components, unpack_sign_mantissa
from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode, table_from_symbols
from pbr_core.types import TILE_HEADER_BYTES

DISCLAIMER = (
    "PBR-EM field codecs (exp rANS + mantissa/SM rANS). Complete bytes include "
    "tables and one tile header. Lossless BF16 only. Not a 1–2 GB / 8 GB claim."
)

TAG_UNCOND_M = 0
TAG_EXPCOND_M = 1
TAG_SM_UNCOND = 2
TAG_SM_EXPCOND = 3


def pack_bits01(bits: np.ndarray) -> bytes:
    b = np.ascontiguousarray(bits, dtype=np.uint8).ravel() & np.uint8(1)
    return np.packbits(b, bitorder="little").tobytes()


def unpack_bits01(data: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")
    if int(bits.size) < n:
        raise ValueError("sign-bit payload truncated")
    return bits[:n].astype(np.uint8)


def dump_rans(symbols: np.ndarray, freq: np.ndarray | None = None) -> bytes:
    sym = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
    table = freq if freq is not None else table_from_symbols(sym)
    book = dump_freq_table(table)
    stream = rans_encode(sym, table)
    return struct.pack("<HI", len(book), len(stream)) + book + stream


def load_rans(data: bytes, n: int, offset: int = 0) -> tuple[np.ndarray, int]:
    _book_len, stream_len = struct.unpack_from("<HI", data, offset)
    offset += 6
    freq, book_end = load_freq_table(data, offset)
    stream = data[book_end : book_end + stream_len]
    if len(stream) != stream_len:
        raise ValueError("rANS stream truncated")
    return rans_decode(stream, n, freq), book_end + stream_len


def dump_rans_body(symbols: np.ndarray, freq: np.ndarray) -> bytes:
    """Stream only (shared table lives in a sidecar)."""
    stream = rans_encode(np.ascontiguousarray(symbols, dtype=np.uint8).ravel(), freq)
    return struct.pack("<I", len(stream)) + stream


def load_rans_body(data: bytes, n: int, freq: np.ndarray, offset: int = 0) -> tuple[np.ndarray, int]:
    (stream_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    stream = data[offset : offset + stream_len]
    if len(stream) != stream_len:
        raise ValueError("shared rANS stream truncated")
    return rans_decode(stream, n, freq), offset + stream_len


def _exp_stream(exp: np.ndarray, pred: int) -> np.ndarray:
    return exp if pred == PRED_NONE else _exp_residuals(exp)


def choose_exp_pred(exp: np.ndarray) -> tuple[int, bytes]:
    raw = _exp_stream(exp, PRED_NONE)
    prev = _exp_stream(exp, PRED_PREV_EXP)
    a = dump_rans(raw)
    b = dump_rans(prev)
    if len(a) <= len(b):
        return PRED_NONE, a
    return PRED_PREV_EXP, b


def encode_exp(exp: np.ndarray, pred: int | None = None) -> tuple[int, bytes]:
    # Prev-exp residuals have *higher* entropy than raw exponents on dense LLMs
    # (blocker diagnosis H=3.12 vs 2.61). Default is raw; caller may still force prev.
    if pred is None:
        pred = PRED_NONE
    return pred, dump_rans(_exp_stream(exp, pred))


def decode_exp(data: bytes, n: int, pred: int, offset: int = 0) -> tuple[np.ndarray, int]:
    stream, end = load_rans(data, n, offset)
    exp = stream if pred == PRED_NONE else _exp_from_residuals(stream)
    return exp, end


def encode_mant_expcond(mant: np.ndarray, exp: np.ndarray, freqs: list[np.ndarray] | None = None) -> bytes:
    """256 rANS groups, raster order within each exponent."""
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    counts = np.bincount(e.astype(np.int64), minlength=256)
    grouped = m[np.argsort(e, kind="mergesort")]
    parts = [struct.pack("<H", 256)]
    start = 0
    for ev in range(256):
        k = int(counts[ev])
        sel = grouped[start : start + k]
        start += k
        if k == 0:
            parts.append(struct.pack("<I", 0))
            continue
        freq = freqs[ev] if freqs is not None else None
        blob = dump_rans(sel) if freq is None else dump_rans(sel, freq)
        parts.append(struct.pack("<I", len(blob)) + blob)
    return b"".join(parts)


def decode_mant_expcond(data: bytes, exp: np.ndarray, offset: int = 0) -> tuple[np.ndarray, int]:
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    n = int(e.size)
    (n_groups,) = struct.unpack_from("<H", data, offset)
    offset += 2
    if n_groups != 256:
        raise ValueError(f"expcond groups {n_groups} != 256")
    counts = np.bincount(e.astype(np.int64), minlength=256)
    order = np.argsort(e, kind="mergesort")
    grouped = np.empty(n, dtype=np.uint8)
    start = 0
    for ev in range(256):
        (ln,) = struct.unpack_from("<I", data, offset)
        offset += 4
        blob = data[offset : offset + ln]
        offset += ln
        k = int(counts[ev])
        if ln == 0:
            if k:
                raise ValueError(f"empty expcond group {ev} but {k} positions")
            start += k
            continue
        rec, end = load_rans(blob, k, 0)
        if end != ln:
            raise ValueError("expcond group trailing bytes")
        grouped[start : start + k] = rec
        start += k
    out = np.empty(n, dtype=np.uint8)
    out[order] = grouped
    return out, offset


def encode_sm_expcond(sm: np.ndarray, exp: np.ndarray, freqs: list[np.ndarray] | None = None) -> bytes:
    return encode_mant_expcond(sm, exp, freqs)


def decode_sm_expcond(data: bytes, exp: np.ndarray, offset: int = 0) -> tuple[np.ndarray, int]:
    return decode_mant_expcond(data, exp, offset)


def encode_tensor_em(
    words: np.ndarray,
    *,
    tag: int = TAG_EXPCOND_M,
    exp_freq: np.ndarray | None = None,
    mant_freq: np.ndarray | None = None,
    sm_freq: np.ndarray | None = None,
    group_freqs: list[np.ndarray] | None = None,
) -> bytes:
    """Encode one tensor. Shared freqs omit per-tensor tables via dump_rans(..., freq)."""
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    sign, exp, mant = split_components(flat)
    pred, exp_blob = encode_exp(exp) if exp_freq is None else (PRED_NONE, dump_rans(_exp_stream(exp, PRED_NONE), exp_freq))
    if tag == TAG_UNCOND_M:
        mant_blob = dump_rans(mant, mant_freq)
        sign_blob = pack_bits01(sign)
        body = struct.pack("<I", len(sign_blob)) + sign_blob + mant_blob
    elif tag == TAG_EXPCOND_M:
        mant_blob = encode_mant_expcond(mant, exp, group_freqs)
        sign_blob = pack_bits01(sign)
        body = struct.pack("<I", len(sign_blob)) + sign_blob + mant_blob
    elif tag == TAG_SM_UNCOND:
        sm = pack_sign_mantissa(sign, mant)
        body = dump_rans(sm, sm_freq)
    elif tag == TAG_SM_EXPCOND:
        sm = pack_sign_mantissa(sign, mant)
        body = encode_sm_expcond(sm, exp, group_freqs)
    else:
        raise ValueError(f"unknown PBR-EM tag {tag}")
    return struct.pack("<BBI", tag, pred, len(exp_blob)) + exp_blob + body


def decode_tensor_em(payload: bytes, n: int) -> np.ndarray:
    tag, pred, exp_len = struct.unpack_from("<BBI", payload, 0)
    offset = 6
    exp_blob = payload[offset : offset + exp_len]
    exp, _end = decode_exp(exp_blob, n, pred, 0)
    offset += exp_len
    if tag in (TAG_UNCOND_M, TAG_EXPCOND_M):
        (sign_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        sign = unpack_bits01(payload[offset : offset + sign_len], n)
        offset += sign_len
        if tag == TAG_UNCOND_M:
            mant, _ = load_rans(payload, n, offset)
        else:
            mant, _ = decode_mant_expcond(payload, exp, offset)
        return join_components(sign, exp, mant)
    if tag == TAG_SM_UNCOND:
        sm, _ = load_rans(payload, n, offset)
        sign, mant = unpack_sign_mantissa(sm)
        return join_components(sign, exp, mant)
    if tag == TAG_SM_EXPCOND:
        sm, _ = decode_sm_expcond(payload, exp, offset)
        sign, mant = unpack_sign_mantissa(sm)
        return join_components(sign, exp, mant)
    raise ValueError(f"unknown PBR-EM tag {tag}")


def dump_shared_tables(exp_freq: np.ndarray, group_freqs: list[np.ndarray]) -> bytes:
    exp_book = dump_freq_table(exp_freq)
    out = bytearray(struct.pack("<I", len(exp_book)) + exp_book)
    out += struct.pack("<H", len(group_freqs))
    for freq in group_freqs:
        book = dump_freq_table(freq)
        out += struct.pack("<H", len(book)) + book
    return bytes(out)


def load_shared_tables(data: bytes) -> tuple[np.ndarray, list[np.ndarray], int]:
    (exp_len,) = struct.unpack_from("<I", data, 0)
    exp_freq, _ = load_freq_table(data, 4)
    offset = 4 + exp_len
    (n_groups,) = struct.unpack_from("<H", data, offset)
    offset += 2
    groups: list[np.ndarray] = []
    for _ in range(n_groups):
        (ln,) = struct.unpack_from("<H", data, offset)
        offset += 2
        freq, end = load_freq_table(data, offset)
        if end - offset != ln:
            raise ValueError("shared group table length mismatch")
        offset = end
        groups.append(freq)
    return exp_freq, groups, offset


def encode_mant_expcond_shared(mant: np.ndarray, exp: np.ndarray, freqs: list[np.ndarray]) -> bytes:
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel()
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    counts = np.bincount(e.astype(np.int64), minlength=256)
    grouped = m[np.argsort(e, kind="mergesort")]
    parts = [struct.pack("<H", 256)]
    start = 0
    for ev in range(256):
        k = int(counts[ev])
        sel = grouped[start : start + k]
        start += k
        if k == 0:
            parts.append(struct.pack("<I", 0))
            continue
        body = dump_rans_body(sel, freqs[ev])
        parts.append(struct.pack("<I", len(body)) + body)
    return b"".join(parts)


def decode_mant_expcond_shared(
    data: bytes, exp: np.ndarray, freqs: list[np.ndarray], offset: int = 0
) -> tuple[np.ndarray, int]:
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    n = int(e.size)
    (n_groups,) = struct.unpack_from("<H", data, offset)
    offset += 2
    counts = np.bincount(e.astype(np.int64), minlength=256)
    order = np.argsort(e, kind="mergesort")
    grouped = np.empty(n, dtype=np.uint8)
    start = 0
    for ev in range(n_groups):
        (ln,) = struct.unpack_from("<I", data, offset)
        offset += 4
        blob = data[offset : offset + ln]
        offset += ln
        k = int(counts[ev]) if ev < counts.size else 0
        if ln == 0:
            if k:
                raise ValueError(f"empty shared group {ev} but {k} positions")
            start += k
            continue
        rec, end = load_rans_body(blob, k, freqs[ev], 0)
        if end != ln:
            raise ValueError("shared group trailing bytes")
        grouped[start : start + k] = rec
        start += k
    out = np.empty(n, dtype=np.uint8)
    out[order] = grouped
    return out, offset


def encode_tensor_em_shared(
    words: np.ndarray,
    *,
    exp_freq: np.ndarray,
    group_freqs: list[np.ndarray],
    tag: int = TAG_EXPCOND_M,
) -> bytes:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    sign, exp, mant = split_components(flat)
    exp_blob = dump_rans_body(_exp_stream(exp, PRED_NONE), exp_freq)
    if tag == TAG_EXPCOND_M:
        sign_blob = pack_bits01(sign)
        mant_blob = encode_mant_expcond_shared(mant, exp, group_freqs)
        rest = struct.pack("<I", len(sign_blob)) + sign_blob + mant_blob
    elif tag == TAG_SM_EXPCOND:
        sm = pack_sign_mantissa(sign, mant)
        rest = encode_mant_expcond_shared(sm, exp, group_freqs)
    else:
        raise ValueError("shared encoder supports expcond tags only")
    return struct.pack("<BBI", tag, PRED_NONE, len(exp_blob)) + exp_blob + rest


def decode_tensor_em_shared(
    payload: bytes, n: int, exp_freq: np.ndarray, group_freqs: list[np.ndarray]
) -> np.ndarray:
    tag, pred, exp_len = struct.unpack_from("<BBI", payload, 0)
    offset = 6
    exp_blob = payload[offset : offset + exp_len]
    stream, _ = load_rans_body(exp_blob, n, exp_freq, 0)
    exp = stream if pred == PRED_NONE else _exp_from_residuals(stream)
    offset += exp_len
    if tag == TAG_EXPCOND_M:
        (sign_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        sign = unpack_bits01(payload[offset : offset + sign_len], n)
        offset += sign_len
        mant, _ = decode_mant_expcond_shared(payload, exp, group_freqs, offset)
        return join_components(sign, exp, mant)
    if tag == TAG_SM_EXPCOND:
        sm, _ = decode_mant_expcond_shared(payload, exp, group_freqs, offset)
        sign, mant = unpack_sign_mantissa(sm)
        return join_components(sign, exp, mant)
    raise ValueError(f"shared decoder: bad tag {tag}")


def complete_bytes(payload: bytes) -> int:
    return TILE_HEADER_BYTES + len(payload)


def roundtrip_em(words: np.ndarray, **kwargs) -> tuple[bytes, np.ndarray]:
    payload = encode_tensor_em(words, **kwargs)
    rec = decode_tensor_em(payload, int(np.asarray(words).size))
    rec = rec.reshape(np.asarray(words).shape)
    return payload, rec
