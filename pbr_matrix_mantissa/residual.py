"""Modulo-128 residuals and compact lossless residual payloads."""

from __future__ import annotations

import struct

import numpy as np

from pbr_matrix_mantissa.bitio import pack_lsb, unpack_lsb
from pbr_matrix_mantissa.errors import CodecError

MOD = 128
RES_ZERO = 0
RES_CONST = 1
RES_PACK = 2
RES_SPARSE = 3

RES_NAMES = {
    RES_ZERO: "ZERO",
    RES_CONST: "CONST",
    RES_PACK: "PACK",
    RES_SPARSE: "SPARSE",
}


def residual(actual: int, pred: int) -> int:
    return (int(actual) - int(pred)) & 0x7F


def restore(pred: int, res: int) -> int:
    return (int(pred) + int(res)) & 0x7F


def encode_residuals(res: np.ndarray) -> tuple[int, bytes]:
    """Pick the shortest residual payload. Ties keep the lower kind id (ZERO first)."""
    flat = np.ascontiguousarray(res, dtype=np.uint8).ravel() & np.uint8(0x7F)
    n = int(flat.size)
    cands: list[tuple[int, bytes]] = []
    if n == 0 or not np.any(flat):
        cands.append((RES_ZERO, b""))
        return cands[0]

    uniq = np.unique(flat)
    if uniq.size == 1:
        cands.append((RES_CONST, bytes([int(uniq[0])])))

    mx = int(flat.max())
    k = mx.bit_length()
    k = max(k, 1)
    packed = pack_lsb(flat, k)
    cands.append((RES_PACK, bytes([k]) + packed))

    if n <= 65535:
        nz = np.flatnonzero(flat)
        body = bytearray(struct.pack("<H", int(nz.size)))
        for i in nz.tolist():
            body += struct.pack("<HB", int(i), int(flat[i]))
        cands.append((RES_SPARSE, bytes(body)))

    return min(cands, key=lambda item: (len(item[1]), item[0]))


def decode_residuals(payload: bytes, count: int, kind: int) -> np.ndarray:
    if count < 0:
        raise CodecError("negative residual count")
    if kind == RES_ZERO:
        return np.zeros(count, dtype=np.uint8)
    if kind == RES_CONST:
        if len(payload) < 1:
            raise CodecError("truncated CONST residual")
        return np.full(count, payload[0] & 0x7F, dtype=np.uint8)
    if kind == RES_PACK:
        if len(payload) < 1:
            raise CodecError("truncated PACK residual")
        k = int(payload[0])
        if k < 1 or k > 7:
            raise CodecError(f"bad PACK width {k}")
        return unpack_lsb(payload[1:], count, k)
    if kind == RES_SPARSE:
        if len(payload) < 2:
            raise CodecError("truncated SPARSE residual")
        (nnz,) = struct.unpack_from("<H", payload, 0)
        need = 2 + 3 * int(nnz)
        if len(payload) < need:
            raise CodecError("truncated SPARSE residual body")
        out = np.zeros(count, dtype=np.uint8)
        off = 2
        for _ in range(int(nnz)):
            idx, val = struct.unpack_from("<HB", payload, off)
            off += 3
            if idx >= count:
                raise CodecError("SPARSE index out of range")
            out[idx] = val & 0x7F
        return out
    raise CodecError(f"unknown residual kind {kind}")
