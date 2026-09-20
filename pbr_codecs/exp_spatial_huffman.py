"""Spatial predictors on the BF16 exponent byte, then Huffman.

Predictors never touch the mantissa. Sign+mantissa stay packed raw.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.exp_common import (
    PRED_NONE,
    PRED_PREV,
    PRED_PREV_ROW,
    PRED_PROTO,
    exp_from_prev_residuals,
    exp_from_prev_row,
    exp_prev_residuals,
    exp_prev_row_residuals,
    most_frequent_u8,
    pack_huffman_plus_sm,
    unpack_huffman_plus_sm,
)
from pbr_core.bf16 import join_components, split_components
from pbr_core.types import (
    MODE_EXP_SPATIAL,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)
from pbr_qualifier.entropy import shannon_entropy

_PRED_NAMES = {
    PRED_NONE: "exp_spatial_huffman",
    PRED_PREV: "exp_spatial_huffman+prev",
    PRED_PREV_ROW: "exp_spatial_huffman+prev_row",
    PRED_PROTO: "exp_spatial_huffman+proto",
}


def _candidate_streams(exp_2d: np.ndarray) -> list[tuple[int, np.ndarray, bytes]]:
    exp = np.ascontiguousarray(exp_2d, dtype=np.uint8)
    h_base = shannon_entropy(exp)
    out: list[tuple[int, np.ndarray, bytes]] = [
        (PRED_NONE, exp.ravel(), struct.pack("<B", PRED_NONE)),
    ]
    prev = exp_prev_residuals(exp)
    if shannon_entropy(prev) <= h_base - 0.03:
        out.append((PRED_PREV, prev, struct.pack("<B", PRED_PREV)))
    if exp.shape[0] > 1:
        row = exp_prev_row_residuals(exp)
        if shannon_entropy(row) <= h_base - 0.03:
            out.append((PRED_PREV_ROW, row.ravel(), struct.pack("<B", PRED_PREV_ROW)))
    proto = most_frequent_u8(exp)
    proto_stream = (exp.ravel() ^ np.uint8(proto))
    if shannon_entropy(proto_stream) <= h_base - 0.03:
        out.append((PRED_PROTO, proto_stream, struct.pack("<BB", PRED_PROTO, proto)))
    return out


class ExpSpatialHuffmanCodec:
    """PBR spatial predictors restricted to the 8-bit exponent field."""

    name = "exp_spatial_huffman"
    mode_id = MODE_EXP_SPATIAL

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        matrix = np.ascontiguousarray(words, dtype=np.uint16)
        if matrix.size == 0:
            return None
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        sign, exp, mant = split_components(matrix)
        exp_2d = exp.reshape(matrix.shape)
        best_name = self.name
        best_payload: bytes | None = None
        raw_cap = TILE_HEADER_BYTES + matrix.size * 2
        for pred, stream, header in _candidate_streams(exp_2d):
            payload = pack_huffman_plus_sm(stream, sign, mant, header=header)
            if best_payload is None or len(payload) < len(best_payload):
                best_payload = payload
                best_name = _PRED_NAMES.get(pred, self.name)
        if best_payload is None or TILE_HEADER_BYTES + len(best_payload) >= raw_cap:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=best_name, payload=best_payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        n = encoded.rows * encoded.cols
        pred = encoded.payload[0]
        header_len = 2 if pred == PRED_PROTO else 1
        header, stream, sign, mant = unpack_huffman_plus_sm(
            encoded.payload, n, header_len=header_len
        )
        if pred == PRED_NONE:
            exp = stream
        elif pred == PRED_PREV:
            exp = exp_from_prev_residuals(stream)
        elif pred == PRED_PREV_ROW:
            exp = exp_from_prev_row(stream.reshape(encoded.rows, encoded.cols)).ravel()
        elif pred == PRED_PROTO:
            proto = header[1]
            exp = stream ^ np.uint8(proto)
        else:
            raise ValueError(f"unknown exp_spatial predictor {pred}")
        words = join_components(sign, exp, mant)
        return words.reshape(encoded.rows, encoded.cols)
