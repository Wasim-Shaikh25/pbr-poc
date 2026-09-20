"""Raw BF16 fallback: exact little-endian uint16 words. Always available."""

from __future__ import annotations

import numpy as np

from pbr_core.hashing import words_to_bytes
from pbr_core.types import MODE_RAW, CostEstimate, EncodedBlock, EncodeContext


class RawCodec:
    name = "raw_bf16"
    mode_id = MODE_RAW

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock:
        del context
        payload = words_to_bytes(words)
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        return self.encode(words, context).to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        n = encoded.rows * encoded.cols
        arr = np.frombuffer(encoded.payload, dtype="<u2", count=n)
        return np.array(arr, dtype=np.uint16, copy=True).reshape(encoded.rows, encoded.cols)
