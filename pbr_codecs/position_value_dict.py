"""Default + position-set of exceptions stored as a small value dictionary.

Admitted only when complete bytes beat raw uint16.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.positions import (
    pack_bitmap,
    pack_positions,
    unpack_bitmap,
    unpack_positions,
)
from pbr_core.bitio import bits_needed, pack_ids, unpack_ids
from pbr_core.hashing import words_to_bytes
from pbr_core.types import (
    MODE_POS_VALUE,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)

MAX_EXC_PALETTE = 16
KIND_LIST = 0
KIND_BITMAP = 1


class PositionValueDictCodec:
    name = "position_value_dict"
    mode_id = MODE_POS_VALUE

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        n = int(flat.size)
        if n == 0:
            return None
        unique, counts = np.unique(flat, return_counts=True)
        default = int(unique[int(np.argmax(counts))])
        mask = flat != np.uint16(default)
        n_exc = int(mask.sum())
        if n_exc == 0 or n_exc == n:
            return None
        exc_vals = flat[mask]
        exc_unique = np.unique(exc_vals)
        if exc_unique.size < 1 or exc_unique.size > MAX_EXC_PALETTE:
            return None
        palette = np.sort(exc_unique)
        index = {int(v): i for i, v in enumerate(palette)}
        ids = np.array([index[int(v)] for v in exc_vals], dtype=np.uint16)
        nbits = bits_needed(int(palette.size))
        packed_ids = pack_ids(ids, nbits)
        positions = np.nonzero(mask)[0].astype(np.uint32)
        list_body = pack_positions(positions, n) + packed_ids
        bitmap_body = pack_bitmap(mask) + packed_ids
        if len(list_body) <= len(bitmap_body):
            kind, body = KIND_LIST, list_body
        else:
            kind, body = KIND_BITMAP, bitmap_body
        payload = (
            struct.pack("<BHBB", kind, default, nbits, int(palette.size))
            + words_to_bytes(palette)
            + body
        )
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
        kind, default, nbits, k = struct.unpack_from("<BHBB", encoded.payload, 0)
        offset = 5
        palette = np.frombuffer(encoded.payload, dtype="<u2", count=k, offset=offset).astype(np.uint16)
        offset += 2 * k
        n = encoded.rows * encoded.cols
        body = encoded.payload[offset:]
        if kind == KIND_LIST:
            positions, pos_end = unpack_positions(body, n, 0)
            ids = unpack_ids(body[pos_end:], int(positions.size), nbits)
            out = np.full(n, default, dtype=np.uint16)
            if positions.size:
                out[positions] = palette[ids]
            return out.reshape(encoded.rows, encoded.cols)
        if kind != KIND_BITMAP:
            raise ValueError(f"unknown position_value_dict kind {kind}")
        mask, mid = unpack_bitmap(body, n, 0)
        n_exc = int(mask.sum())
        ids = unpack_ids(body[mid:], n_exc, nbits)
        out = np.full(n, default, dtype=np.uint16)
        if n_exc:
            out[mask] = palette[ids]
        return out.reshape(encoded.rows, encoded.cols)
