"""Cross-layer exact XOR against a previous same-shaped tensor.

The referenced tensor is *not* stored again. Complete cost is the patch
plus a 16-byte ref fingerprint. Decode requires the causal ref tensor.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.exp_common import pack_huffman_plus_sm, unpack_huffman_plus_sm
from pbr_codecs.residual import best_residual, decode_residuals
from pbr_core.bf16 import join_components, split_components
from pbr_core.hashing import sha256_words
from pbr_core.types import (
    MODE_CROSS_LAYER,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)
from pbr_qualifier.entropy import shannon_entropy

KIND_EXP = 0
KIND_UINT16 = 1


def _ref_fp(words: np.ndarray) -> bytes:
    return bytes.fromhex(sha256_words(words))[:16]


def _aligned_ref(words: np.ndarray, context: EncodeContext | None) -> np.ndarray | None:
    if context is None or context.ref_tensor is None:
        return None
    ref = np.ascontiguousarray(context.ref_tensor, dtype=np.uint16)
    matrix = np.ascontiguousarray(words, dtype=np.uint16)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.shape == matrix.shape:
        return ref
    r0, c0 = context.tile_row0, context.tile_col0
    h, w = matrix.shape
    if r0 < 0 or c0 < 0 or r0 + h > ref.shape[0] or c0 + w > ref.shape[1]:
        return None
    return ref[r0 : r0 + h, c0 : c0 + w]


class CrossLayerTileXorCodec:
    """XOR vs a previous same-role tensor; Huffman exp residual or sparse uint16."""

    name = "cross_layer_tile_xor"
    mode_id = MODE_CROSS_LAYER

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        ref = _aligned_ref(words, context)
        if ref is None:
            return None
        matrix = np.ascontiguousarray(words, dtype=np.uint16)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        fp = _ref_fp(ref)
        raw_cap = TILE_HEADER_BYTES + matrix.size * 2
        candidates: list[tuple[str, bytes]] = []

        sign, exp, mant = split_components(matrix)
        _s, ref_exp, _m = split_components(ref)
        stream = exp.ravel() ^ ref_exp.ravel()
        if shannon_entropy(stream) <= shannon_entropy(exp) - 0.03:
            header = struct.pack("<B", KIND_EXP) + fp
            payload = pack_huffman_plus_sm(stream, sign, mant, header=header)
            candidates.append((f"{self.name}+exp", payload))

        xor16 = matrix ^ ref
        res_name, res_payload = best_residual(xor16)
        payload16 = struct.pack("<B", KIND_UINT16) + fp + res_payload
        candidates.append((f"{self.name}+{res_name}", payload16))

        if not candidates:
            return None
        name, payload = min(candidates, key=lambda item: len(item[1]))
        if TILE_HEADER_BYTES + len(payload) >= raw_cap:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        if context is None or context.ref_tensor is None:
            raise ValueError("cross_layer_tile_xor requires a causal ref tensor")
        ref_full = np.ascontiguousarray(context.ref_tensor, dtype=np.uint16)
        if ref_full.ndim == 1:
            ref_full = ref_full.reshape(1, -1)
        if ref_full.shape == (encoded.rows, encoded.cols):
            ref = ref_full
        else:
            r0, c0 = context.tile_row0, context.tile_col0
            ref = ref_full[r0 : r0 + encoded.rows, c0 : c0 + encoded.cols]
            if ref.shape != (encoded.rows, encoded.cols):
                raise ValueError("cross_layer ref shape mismatch")
        kind = encoded.payload[0]
        fp = encoded.payload[1:17]
        if fp != _ref_fp(ref):
            raise ValueError("cross_layer ref fingerprint mismatch")
        if kind == KIND_UINT16:
            residuals = decode_residuals(encoded.payload[17:], encoded.rows, encoded.cols)
            return residuals ^ ref
        if kind != KIND_EXP:
            raise ValueError(f"unknown cross_layer kind {kind}")
        n = encoded.rows * encoded.cols
        _h, stream, sign, mant = unpack_huffman_plus_sm(
            encoded.payload, n, header_len=17
        )
        _s, ref_exp, _m = split_components(ref)
        exp = stream ^ ref_exp.ravel()
        return join_components(sign, exp, mant).reshape(encoded.rows, encoded.cols)
