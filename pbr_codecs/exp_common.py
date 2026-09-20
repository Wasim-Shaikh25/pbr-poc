"""Shared lossless exponent-stream helpers (Huffman + packed sign/mantissa)."""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.bf16 import pack_sign_mantissa, unpack_sign_mantissa
from pbr_core.huffman import dump_table, load_table, table_from_symbols

PRED_NONE = 0
PRED_PREV = 1
PRED_PREV_ROW = 2
PRED_PROTO = 3
PRED_HIER = 4
PRED_CROSS_EXP = 5


def most_frequent_u8(values: np.ndarray) -> int:
    flat = np.ascontiguousarray(values, dtype=np.uint8).ravel()
    if flat.size == 0:
        return 0
    unique, counts = np.unique(flat, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def exp_prev_residuals(exp: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if flat.size == 0:
        return flat
    out = np.empty_like(flat)
    out[0] = flat[0]
    if flat.size > 1:
        out[1:] = np.bitwise_xor(flat[1:], flat[:-1])
    return out


def exp_from_prev_residuals(residuals: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(residuals, dtype=np.uint8).ravel()
    if flat.size == 0:
        return flat
    return np.bitwise_xor.accumulate(flat.astype(np.uint16)).astype(np.uint8)


def exp_prev_row_residuals(exp_2d: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(exp_2d, dtype=np.uint8)
    pred = np.zeros_like(tile)
    if tile.shape[0] > 1:
        pred[1:, :] = tile[:-1, :]
    return tile ^ pred


def exp_from_prev_row(residuals_2d: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(residuals_2d, dtype=np.uint8)
    if tile.size == 0:
        return tile
    return np.bitwise_xor.accumulate(tile.astype(np.uint16), axis=0).astype(np.uint8)


def pack_huffman_plus_sm(
    stream: np.ndarray,
    sign: np.ndarray,
    mant: np.ndarray,
    *,
    header: bytes,
) -> bytes:
    table = table_from_symbols(stream)
    codebook = dump_table(table)
    bitstream = table.encode_symbols(stream)
    packed_sm = pack_sign_mantissa(sign, mant).tobytes()
    return (
        header
        + struct.pack("<HI", len(codebook), len(bitstream))
        + codebook
        + bitstream
        + packed_sm
    )


def unpack_huffman_plus_sm(
    payload: bytes, n: int, *, header_len: int
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray]:
    header = payload[:header_len]
    book_len, bit_len = struct.unpack_from("<HI", payload, header_len)
    offset = header_len + 6
    table, book_end = load_table(payload, offset)
    if book_end - offset != book_len:
        raise ValueError("exponent codebook length mismatch")
    bitstream = payload[book_end : book_end + bit_len]
    packed_sm = payload[book_end + bit_len :]
    if len(packed_sm) != n:
        raise ValueError(f"sign/mantissa length {len(packed_sm)} != {n}")
    stream = table.decode_symbols(bitstream, n)
    sign, mant = unpack_sign_mantissa(np.frombuffer(packed_sm, dtype=np.uint8))
    return header, stream, sign, mant
