"""Exact duplicate-tile references and previous-tile XOR patches."""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.residual import best_residual, decode_residuals
from pbr_core.hashing import words_to_bytes
from pbr_core.types import (
    MODE_DUP_REF,
    MODE_REF_PREV,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)


class DuplicateRefCodec:
    name = "duplicate_ref"
    mode_id = MODE_DUP_REF

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        if context is None:
            return None
        raw = words_to_bytes(words)
        ref = context.seen_exact.get(raw)
        if ref is None or ref == context.tile_index:
            return None
        return EncodedBlock(
            mode_id=self.mode_id,
            mode_name=self.name,
            payload=struct.pack("<I", int(ref)),
        )

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del encoded, context
        raise RuntimeError("duplicate_ref decode is handled by the tensor decoder")


class RefPrevTileCodec:
    """Near-duplicate of the immediately previous same-shaped tile."""

    name = "ref_prev_tile"
    mode_id = MODE_REF_PREV

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        if context is None or context.prev_tile is None:
            return None
        prev = context.prev_tile
        if prev.shape != words.shape:
            return None
        residuals = np.ascontiguousarray(words, dtype=np.uint16) ^ prev
        res_name, res_payload = best_residual(residuals)
        payload = res_payload
        raw_cost = TILE_HEADER_BYTES + words.size * 2
        if TILE_HEADER_BYTES + len(payload) >= raw_cost:
            return None
        return EncodedBlock(
            mode_id=self.mode_id,
            mode_name=f"{self.name}+{res_name}",
            payload=payload,
        )

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        if context is None or context.prev_tile is None:
            raise ValueError("ref_prev_tile requires a previous decoded tile")
        residuals = decode_residuals(encoded.payload, encoded.rows, encoded.cols)
        return residuals ^ context.prev_tile
