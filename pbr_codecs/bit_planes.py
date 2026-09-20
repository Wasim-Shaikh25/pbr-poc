"""Lossless 16-plane encoding of uint16 tiles.

Each plane is all-zero, all-one, sparse minority-bit positions, or packed raw.
Admission requires a complete-byte saving versus raw uint16.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.positions import pack_bitmap, pack_positions, unpack_bitmap, unpack_positions
from pbr_core.types import (
    MODE_BITPLANES,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)

PLANE_ZERO = 0
PLANE_ONE = 1
PLANE_SPARSE = 2
PLANE_RAW = 3


def _plane_bits(words: np.ndarray, bit: int) -> np.ndarray:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    return ((flat >> np.uint16(bit)) & np.uint16(1)).astype(np.uint8)


def _encode_plane(bits: np.ndarray) -> bytes:
    n = int(bits.size)
    ones = int(bits.sum())
    if ones == 0:
        return bytes([PLANE_ZERO])
    if ones == n:
        return bytes([PLANE_ONE])
    raw = bytes([PLANE_RAW]) + pack_bitmap(bits.astype(bool))
    default = 0 if ones <= n - ones else 1
    mask = bits != default
    positions = np.nonzero(mask)[0].astype(np.uint32)
    sparse = bytes([PLANE_SPARSE, default]) + pack_positions(positions, n)
    return sparse if len(sparse) < len(raw) else raw


def _decode_plane(data: bytes, n: int, offset: int) -> tuple[np.ndarray, int]:
    kind = data[offset]
    offset += 1
    if kind == PLANE_ZERO:
        return np.zeros(n, dtype=np.uint8), offset
    if kind == PLANE_ONE:
        return np.ones(n, dtype=np.uint8), offset
    if kind == PLANE_RAW:
        mask, offset = unpack_bitmap(data, n, offset)
        return mask.astype(np.uint8), offset
    if kind != PLANE_SPARSE:
        raise ValueError(f"unknown bit-plane kind {kind}")
    default = data[offset]
    offset += 1
    positions, offset = unpack_positions(data, n, offset)
    out = np.full(n, default, dtype=np.uint8)
    if positions.size:
        out[positions] = np.uint8(1 - default)
    return out, offset


class BitPlanesCodec:
    name = "bit_planes"
    mode_id = MODE_BITPLANES

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        if flat.size == 0:
            return None
        # Cheap reject: if no plane is constant, 16 packed planes == raw 16-bit.
        ones = [int(_plane_bits(flat, b).sum()) for b in range(16)]
        n = int(flat.size)
        if not any(c == 0 or c == n for c in ones):
            return None
        payload = b"".join(_encode_plane(_plane_bits(flat, b)) for b in range(16))
        if TILE_HEADER_BYTES + len(payload) >= TILE_HEADER_BYTES + n * 2:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        n = encoded.rows * encoded.cols
        offset = 0
        acc = np.zeros(n, dtype=np.uint16)
        for bit in range(16):
            plane, offset = _decode_plane(encoded.payload, n, offset)
            acc |= plane.astype(np.uint16) << np.uint16(bit)
        return acc.reshape(encoded.rows, encoded.cols)
