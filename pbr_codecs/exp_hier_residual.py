"""Two-level exact residual on exponents: block default, then Huffman.

Defaults are themselves Huffman-coded. Sign+mantissa stay packed raw.
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.exp_common import (
    exp_from_prev_residuals,
    exp_prev_residuals,
    most_frequent_u8,
    pack_huffman_plus_sm,
    unpack_huffman_plus_sm,
)
from pbr_core.bf16 import join_components, split_components
from pbr_core.huffman import dump_table, load_table, table_from_symbols
from pbr_core.types import (
    MODE_EXP_HIER,
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
)
from pbr_qualifier.entropy import shannon_entropy

_BLOCK_SIZES = (8, 16, 32)


def block_defaults_and_residual(
    exp_2d: np.ndarray, block_h: int, block_w: int
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = int(exp_2d.shape[0]), int(exp_2d.shape[1])
    n_br = (rows + block_h - 1) // block_h
    n_bc = (cols + block_w - 1) // block_w
    defaults = np.empty((n_br, n_bc), dtype=np.uint8)
    residual = np.empty_like(exp_2d)
    for i in range(n_br):
        r0, r1 = i * block_h, min((i + 1) * block_h, rows)
        for j in range(n_bc):
            c0, c1 = j * block_w, min((j + 1) * block_w, cols)
            block = exp_2d[r0:r1, c0:c1]
            proto = most_frequent_u8(block)
            defaults[i, j] = proto
            residual[r0:r1, c0:c1] = block ^ np.uint8(proto)
    return defaults, residual


def apply_defaults(defaults: np.ndarray, block_h: int, block_w: int, rows: int, cols: int) -> np.ndarray:
    out = np.empty((rows, cols), dtype=np.uint8)
    n_br, n_bc = defaults.shape
    for i in range(n_br):
        r0, r1 = i * block_h, min((i + 1) * block_h, rows)
        for j in range(n_bc):
            c0, c1 = j * block_w, min((j + 1) * block_w, cols)
            out[r0:r1, c0:c1] = defaults[i, j]
    return out


def _encode_defaults(defaults: np.ndarray) -> bytes:
    flat = defaults.ravel()
    table = table_from_symbols(flat)
    codebook = dump_table(table)
    stream = table.encode_symbols(flat)
    return struct.pack("<HH", defaults.shape[0], defaults.shape[1]) + struct.pack(
        "<HI", len(codebook), len(stream)
    ) + codebook + stream


def _decode_defaults(data: bytes, offset: int) -> tuple[np.ndarray, int]:
    n_br, n_bc = struct.unpack_from("<HH", data, offset)
    offset += 4
    book_len, bit_len = struct.unpack_from("<HI", data, offset)
    offset += 6
    table, book_end = load_table(data, offset)
    if book_end - offset != book_len:
        raise ValueError("hier defaults codebook length mismatch")
    stream = table.decode_symbols(data[book_end : book_end + bit_len], n_br * n_bc)
    return stream.reshape(n_br, n_bc), book_end + bit_len


class ExpHierResidualCodec:
    """Block-default then local exponent residual, both entropy-coded."""

    name = "exp_hier_residual"
    mode_id = MODE_EXP_HIER

    def encode(self, words: np.ndarray, context: EncodeContext | None = None) -> EncodedBlock | None:
        del context
        matrix = np.ascontiguousarray(words, dtype=np.uint16)
        if matrix.size == 0:
            return None
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        sign, exp, mant = split_components(matrix)
        exp_2d = exp.reshape(matrix.shape)
        h_base = shannon_entropy(exp_2d)
        raw_cap = TILE_HEADER_BYTES + matrix.size * 2
        best_payload: bytes | None = None
        best_name = self.name
        for bh in _BLOCK_SIZES:
            if bh > matrix.shape[0] and bh > matrix.shape[1]:
                continue
            bw = min(bh, int(matrix.shape[1]))
            defaults, residual = block_defaults_and_residual(exp_2d, bh, bw)
            if shannon_entropy(residual) > h_base - 0.03:
                continue
            streams = [
                (residual.ravel(), f"{self.name}+{bh}x{bw}"),
            ]
            prev = exp_prev_residuals(residual)
            if shannon_entropy(prev) <= shannon_entropy(residual) - 0.03:
                streams.append((prev, f"{self.name}+{bh}x{bw}+prev"))
            defaults_blob = _encode_defaults(defaults)
            for stream, name in streams:
                header = struct.pack("<BBB", bh, bw, 1 if name.endswith("+prev") else 0)
                body = pack_huffman_plus_sm(stream, sign, mant, header=header)
                payload = body[:3] + defaults_blob + body[3:]
                # pack_huffman_plus_sm header is 3 bytes (BBB); insert defaults after it.
                if best_payload is None or len(payload) < len(best_payload):
                    best_payload = payload
                    best_name = name
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
        bh, bw, use_prev = struct.unpack_from("<BBB", encoded.payload, 0)
        defaults, mid = _decode_defaults(encoded.payload, 3)
        tail = encoded.payload[mid:]
        # unpack_huffman_plus_sm expects header + HI + book + bits + sm.
        fake = struct.pack("<BBB", bh, bw, use_prev) + tail
        _header, stream, sign, mant = unpack_huffman_plus_sm(fake, n, header_len=3)
        residual = exp_from_prev_residuals(stream) if use_prev else stream
        residual_2d = residual.reshape(encoded.rows, encoded.cols)
        proto = apply_defaults(defaults, bh, bw, encoded.rows, encoded.cols)
        exp = residual_2d ^ proto
        words = join_components(sign, exp.ravel(), mant)
        return words.reshape(encoded.rows, encoded.cols)
