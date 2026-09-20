"""Exact repeated-value dictionary. Admission requires net byte savings vs raw."""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.bitio import bits_needed, pack_ids, unpack_ids
from pbr_core.hashing import words_to_bytes
from pbr_core.types import MODE_VALUE_DICT, TILE_HEADER_BYTES, CostEstimate, EncodedBlock, EncodeContext

# Keep the palette small so index bits stay cheap.
MAX_PALETTE = 16


class ValueDictCodec:
    name = "value_dict"
    mode_id = MODE_VALUE_DICT

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        if flat.size == 0:
            return None
        unique = np.unique(flat)
        k = int(unique.size)
        if k < 2 or k > MAX_PALETTE:
            return None
        nbits = bits_needed(k)
        palette = np.sort(unique)
        index = {int(v): i for i, v in enumerate(palette)}
        ids = np.array([index[int(v)] for v in flat], dtype=np.uint16)
        packed = pack_ids(ids, nbits)
        payload = struct.pack("<BB", k, nbits) + words_to_bytes(palette) + packed
        raw_bytes = TILE_HEADER_BYTES + flat.size * 2
        if TILE_HEADER_BYTES + len(payload) >= raw_bytes:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        k, nbits = struct.unpack_from("<BB", encoded.payload, 0)
        palette = np.frombuffer(encoded.payload, dtype="<u2", count=k, offset=2).astype(np.uint16)
        packed = encoded.payload[2 + 2 * k :]
        n = encoded.rows * encoded.cols
        ids = unpack_ids(packed, n, nbits)
        return palette[ids].reshape(encoded.rows, encoded.cols)
