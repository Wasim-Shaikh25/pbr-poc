"""PBR-E: Huffman-code BF16 exponents; pack sign+mantissa raw.

Complete cost includes the codebook. Bit-exact uint16 reconstruction.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.bf16 import join_components, pack_sign_mantissa, split_components, unpack_sign_mantissa
from pbr_core.huffman import dump_table, load_table, table_from_symbols
from pbr_core.types import (
    MODE_EXP_HUFFMAN,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)

PRED_NONE = 0
PRED_PREV_EXP = 1


def _exp_residuals(exp: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if flat.size == 0:
        return flat
    out = np.empty_like(flat)
    out[0] = flat[0]
    if flat.size > 1:
        out[1:] = np.bitwise_xor(flat[1:], flat[:-1])
    return out


def _exp_from_residuals(residuals: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(residuals, dtype=np.uint8).ravel()
    if flat.size == 0:
        return flat
    return np.bitwise_xor.accumulate(flat.astype(np.uint16)).astype(np.uint8)


def _encode_stream(exp: np.ndarray, sign: np.ndarray, mant: np.ndarray, pred: int) -> bytes:
    stream = exp if pred == PRED_NONE else _exp_residuals(exp)
    table = table_from_symbols(stream)
    codebook = dump_table(table)
    bitstream = table.encode_symbols(stream)
    packed_sm = pack_sign_mantissa(sign, mant).tobytes()
    return (
        struct.pack("<BHI", pred, len(codebook), len(bitstream))
        + codebook
        + bitstream
        + packed_sm
    )


class Bf16ExpHuffmanCodec:
    """DF11/NeuZip-style lossless split: entropy-code exponents only."""

    name = "bf16_exp_huffman"
    mode_id = MODE_EXP_HUFFMAN

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        if flat.size == 0:
            return None
        sign, exp, mant = split_components(flat)
        candidates = [
            (self.name, _encode_stream(exp, sign, mant, PRED_NONE)),
            (f"{self.name}+prev_exp", _encode_stream(exp, sign, mant, PRED_PREV_EXP)),
        ]
        mode_name, payload = min(candidates, key=lambda item: len(item[1]))
        raw_cap = TILE_HEADER_BYTES + flat.size * 2
        if TILE_HEADER_BYTES + len(payload) >= raw_cap:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=mode_name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        n = encoded.rows * encoded.cols
        pred, book_len, bit_len = struct.unpack_from("<BHI", encoded.payload, 0)
        offset = 7
        table, book_end = load_table(encoded.payload, offset)
        if book_end - offset != book_len:
            raise ValueError("exp-huffman codebook length mismatch")
        bitstream = encoded.payload[book_end : book_end + bit_len]
        packed_sm = encoded.payload[book_end + bit_len :]
        if len(packed_sm) != n:
            raise ValueError(
                f"exp-huffman sign/mantissa length {len(packed_sm)} != {n}"
            )
        stream = table.decode_symbols(bitstream, n)
        exp = stream if pred == PRED_NONE else _exp_from_residuals(stream)
        packed = np.frombuffer(packed_sm, dtype=np.uint8)
        from pbr_core.rans import join_bf16_u16

        joined = join_bf16_u16(exp, packed)
        if joined is not None:
            return joined.reshape(encoded.rows, encoded.cols)
        sign, mant = unpack_sign_mantissa(packed)
        words = join_components(sign, exp, mant)
        return words.reshape(encoded.rows, encoded.cols)
