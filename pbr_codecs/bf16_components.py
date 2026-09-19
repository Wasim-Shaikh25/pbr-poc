"""BF16 sign / exponent / mantissa streams. Exact bit reconstruction."""

from __future__ import annotations

import struct

import numpy as np

from pbr_core.bf16 import join_components, split_components
from pbr_core.bitio import BitReader, BitWriter, bits_needed, pack_ids, unpack_ids
from pbr_core.types import MODE_COMPONENTS, TILE_HEADER_BYTES, CostEstimate, EncodedBlock, EncodeContext

MAX_EXP_PALETTE = 16


class ComponentsCodec:
    name = "bf16_components"
    mode_id = MODE_COMPONENTS

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
        if flat.size == 0:
            return None
        sign, exp, mant = split_components(flat)
        writer = BitWriter()
        writer.write_array(sign, 1)

        unique_exp = np.unique(exp)
        if 1 < unique_exp.size <= MAX_EXP_PALETTE:
            palette = np.sort(unique_exp)
            nbits = bits_needed(int(palette.size))
            index = {int(v): i for i, v in enumerate(palette)}
            ids = np.array([index[int(v)] for v in exp], dtype=np.uint16)
            exp_blob = struct.pack("<BB", int(palette.size), nbits) + palette.tobytes() + pack_ids(ids, nbits)
            exp_mode = 1
        else:
            exp_blob = exp.astype(np.uint8, copy=False).tobytes()
            exp_mode = 0

        writer.write_array(mant, 7)
        packed = writer.finalize()
        payload = struct.pack("<B", exp_mode) + struct.pack("<I", len(exp_blob)) + exp_blob + packed
        if TILE_HEADER_BYTES + len(payload) >= TILE_HEADER_BYTES + flat.size * 2:
            return None
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def estimate(self, words: np.ndarray, context: EncodeContext | None = None) -> CostEstimate:
        enc = self.encode(words, context)
        if enc is None:
            return CostEstimate(total_bytes=1 << 30, mode_name=self.name)
        return enc.to_cost()

    def decode(self, encoded: EncodedBlock, context: EncodeContext | None = None) -> np.ndarray:
        del context
        n = encoded.rows * encoded.cols
        exp_mode = encoded.payload[0]
        (exp_len,) = struct.unpack_from("<I", encoded.payload, 1)
        exp_blob = encoded.payload[5 : 5 + exp_len]
        packed = encoded.payload[5 + exp_len :]
        if exp_mode == 1:
            k, nbits = struct.unpack_from("<BB", exp_blob, 0)
            palette = np.frombuffer(exp_blob, dtype=np.uint8, count=k, offset=2)
            ids = unpack_ids(exp_blob[2 + k :], n, nbits)
            exp = palette[ids]
        else:
            exp = np.frombuffer(exp_blob, dtype=np.uint8, count=n)

        reader = BitReader(packed)
        sign = reader.read_array(n, 1).astype(np.uint8)
        mant = reader.read_array(n, 7).astype(np.uint8)
        words = join_components(sign, exp, mant)
        return words.reshape(encoded.rows, encoded.cols)
