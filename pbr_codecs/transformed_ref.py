"""Reference a previous tile after a cheap exact transform + XOR patch."""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.residual import best_residual, decode_residuals
from pbr_codecs.transforms import XF_NAMES, apply_transform
from pbr_core.hashing import words_to_bytes
from pbr_core.types import (
    MODE_XFORM_REF,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)


def _index_latest(context: EncodeContext) -> None:
    """Index only the newest stored tile (O(1) per tile, not O(n²))."""
    if not context.seen_tiles:
        return
    idx = len(context.seen_tiles) - 1
    tile = context.seen_tiles[idx]
    ident = apply_transform(tile, 0)
    if ident is not None and context.seen_xf.get(words_to_bytes(ident)) == (idx, 0):
        return
    for xf_id in XF_NAMES:
        xf = apply_transform(tile, xf_id)
        if xf is None:
            continue
        context.seen_xf.setdefault(words_to_bytes(xf), (idx, xf_id))


class TransformedRefCodec:
    name = "transformed_ref"
    mode_id = MODE_XFORM_REF

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        if context is None or not context.seen_tiles:
            return None
        _index_latest(context)
        matrix = np.ascontiguousarray(words, dtype=np.uint16)
        raw_cost = TILE_HEADER_BYTES + matrix.size * 2
        best: EncodedBlock | None = None

        exact = context.seen_xf.get(words_to_bytes(matrix))
        if exact is not None and exact[0] != context.tile_index:
            payload = struct.pack("<BI", exact[1], exact[0])
            best = EncodedBlock(
                mode_id=self.mode_id,
                mode_name=f"{self.name}+{XF_NAMES[exact[1]]}+exact",
                payload=payload,
            )

        prev = context.prev_tile
        if prev is not None:
            for xf_id, xf_name in XF_NAMES.items():
                pred = apply_transform(prev, xf_id)
                if pred is None or pred.shape != matrix.shape:
                    continue
                residuals = matrix ^ pred
                res_name, res_payload = best_residual(residuals)
                payload = struct.pack("<BI", xf_id, int(context.prev_index or 0)) + res_payload
                if TILE_HEADER_BYTES + len(payload) >= raw_cost:
                    continue
                cand = EncodedBlock(
                    mode_id=self.mode_id,
                    mode_name=f"{self.name}+{xf_name}+{res_name}",
                    payload=payload,
                )
                if best is None or cand.total_bytes < best.total_bytes:
                    best = cand
        if best is None or best.total_bytes >= raw_cost:
            return None
        return best

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del encoded, context
        raise RuntimeError("transformed_ref decode is handled by the tensor decoder")
