"""PBR-E rANS exponents: same split as Huffman, ANS coded exponent stream."""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.bf16_exp_huffman import PRED_NONE, PRED_PREV_EXP, _exp_from_residuals, _exp_residuals
from pbr_core.bf16 import join_components, pack_sign_mantissa, split_components, unpack_sign_mantissa
from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode, table_from_symbols
from pbr_core.types import (
    MODE_EXP_RANS,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)


def _encode_stream(exp: np.ndarray, sign: np.ndarray, mant: np.ndarray, pred: int) -> bytes:
    stream = exp if pred == PRED_NONE else _exp_residuals(exp)
    freq = table_from_symbols(stream)
    table = dump_freq_table(freq)
    bitstream = rans_encode(stream, freq)
    packed_sm = pack_sign_mantissa(sign, mant).tobytes()
    return (
        struct.pack("<BHI", pred, len(table), len(bitstream))
        + table
        + bitstream
        + packed_sm
    )


class Bf16ExpRansCodec:
    """Same field split as PBR-E Huffman; rANS the exponent byte."""

    name = "bf16_exp_rans"
    mode_id = MODE_EXP_RANS

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
        freq, book_end = load_freq_table(encoded.payload, offset)
        if book_end - offset != book_len:
            raise ValueError("exp-rANS table length mismatch")
        bitstream = encoded.payload[book_end : book_end + bit_len]
        packed_sm = encoded.payload[book_end + bit_len :]
        if len(packed_sm) != n:
            raise ValueError(f"exp-rANS sign/mantissa length {len(packed_sm)} != {n}")
        stream = rans_decode(bitstream, n, freq)
        exp = stream if pred == PRED_NONE else _exp_from_residuals(stream)
        sign, mant = unpack_sign_mantissa(np.frombuffer(packed_sm, dtype=np.uint8))
        words = join_components(sign, exp, mant)
        return words.reshape(encoded.rows, encoded.cols)
