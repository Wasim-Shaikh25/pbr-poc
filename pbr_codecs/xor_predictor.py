"""Causal XOR predictors from PBR-Direct.

Prediction uses already-decoded neighbours. Residuals are exact uint16 XOR.
The inverse is bitwise prefix-XOR, never floating-point arithmetic.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.residual import best_residual, decode_residuals
from pbr_core.types import (
    MODE_CONST_PRED,
    MODE_PREV_ROW,
    MODE_PREV_VALUE,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)


def residuals_prev_value(words: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    pred = np.empty_like(flat)
    if flat.size:
        pred[0] = np.uint16(0)
        if flat.size > 1:
            pred[1:] = flat[:-1]
    return (flat ^ pred).reshape(words.shape)


def residuals_prev_row(words: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(words, dtype=np.uint16)
    pred = np.zeros_like(tile)
    if tile.shape[0] > 1:
        pred[1:, :] = tile[:-1, :]
    return tile ^ pred


def residuals_const(words: np.ndarray, value: int) -> np.ndarray:
    return np.ascontiguousarray(words, dtype=np.uint16) ^ np.uint16(value)


def reconstruct_prev_value(residuals: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(residuals, dtype=np.uint16).ravel()
    if flat.size == 0:
        return flat.reshape(residuals.shape)
    return np.bitwise_xor.accumulate(flat).reshape(residuals.shape)


def reconstruct_prev_row(residuals: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(residuals, dtype=np.uint16)
    if tile.size == 0:
        return tile
    return np.bitwise_xor.accumulate(tile, axis=0)


def most_frequent_word(words: np.ndarray) -> int:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    if flat.size == 0:
        return 0
    unique, counts = np.unique(flat, return_counts=True)
    return int(unique[int(np.argmax(counts))])


class _PredictorCodec:
    name: str
    mode_id: int

    def _residuals_and_meta(self, words: np.ndarray) -> tuple[np.ndarray, bytes]:
        raise NotImplementedError

    def _reconstruct(self, residuals: np.ndarray, meta: bytes) -> np.ndarray:
        raise NotImplementedError

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock:
        del context
        residuals, meta = self._residuals_and_meta(words)
        res_name, res_payload = best_residual(residuals)
        payload = meta + res_payload
        return EncodedBlock(
            mode_id=self.mode_id,
            mode_name=f"{self.name}+{res_name}",
            payload=payload,
        )

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        return self.encode(words, context).to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        meta_len = self._meta_len()
        residuals = decode_residuals(encoded.payload[meta_len:], encoded.rows, encoded.cols)
        return self._reconstruct(residuals, encoded.payload[:meta_len])

    def _meta_len(self) -> int:
        return 0


class PrevValueCodec(_PredictorCodec):
    name = "prev_value"
    mode_id = MODE_PREV_VALUE

    def _residuals_and_meta(self, words: np.ndarray) -> tuple[np.ndarray, bytes]:
        return residuals_prev_value(words), b""

    def _reconstruct(self, residuals: np.ndarray, meta: bytes) -> np.ndarray:
        del meta
        return reconstruct_prev_value(residuals)


class PrevRowCodec(_PredictorCodec):
    name = "prev_row"
    mode_id = MODE_PREV_ROW

    def _residuals_and_meta(self, words: np.ndarray) -> tuple[np.ndarray, bytes]:
        return residuals_prev_row(words), b""

    def _reconstruct(self, residuals: np.ndarray, meta: bytes) -> np.ndarray:
        del meta
        return reconstruct_prev_row(residuals)


class ConstPredCodec(_PredictorCodec):
    """Block prototype: most frequent exact word XOR residual field."""

    name = "const_pred"
    mode_id = MODE_CONST_PRED

    def _residuals_and_meta(self, words: np.ndarray) -> tuple[np.ndarray, bytes]:
        proto = most_frequent_word(words)
        return residuals_const(words, proto), struct.pack("<H", proto)

    def _reconstruct(self, residuals: np.ndarray, meta: bytes) -> np.ndarray:
        (proto,) = struct.unpack("<H", meta)
        return residuals ^ np.uint16(proto)

    def _meta_len(self) -> int:
        return 2
