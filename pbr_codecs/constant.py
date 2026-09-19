"""Constant-block mode: one exact uint16 repeated across the tile."""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.types import MODE_CONSTANT, CostEstimate, EncodedBlock, EncodeContext


class ConstantCodec:
    name = "constant"
    mode_id = MODE_CONSTANT

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        if flat.size == 0:
            return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=b"")
        first = int(flat[0])
        if not bool(np.all(flat == first)):
            return None
        return EncodedBlock(
            mode_id=self.mode_id,
            mode_name=self.name,
            payload=struct.pack("<H", first),
        )

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        if encoded.rows * encoded.cols == 0:
            return np.zeros((encoded.rows, encoded.cols), dtype=np.uint16)
        (value,) = struct.unpack("<H", encoded.payload)
        return np.full((encoded.rows, encoded.cols), value, dtype=np.uint16)
