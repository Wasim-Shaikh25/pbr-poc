"""Canonical Huffman for small alphabets (BF16 exponents, 0–255)."""

from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass

import numpy as np


MAX_CODE_LEN = 16


@dataclass(frozen=True)
class HuffmanTable:
    """Canonical table: symbol -> (lsb_first_code, length)."""

    lengths: dict[int, int]
    codes: dict[int, tuple[int, int]]
    max_len: int

    def encode_symbols(self, symbols: np.ndarray) -> bytes:
        if len(self.codes) == 0:
            return b""
        code_lut = np.zeros(256, dtype=np.uint32)
        len_lut = np.zeros(256, dtype=np.uint8)
        for sym, (code, nbits) in self.codes.items():
            code_lut[sym] = code
            len_lut[sym] = nbits
        flat = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
        codes = code_lut[flat].tolist()
        nbits = len_lut[flat].tolist()
        acc = 0
        filled = 0
        out = bytearray()
        append = out.append
        for code, n in zip(codes, nbits):
            if n == 0:
                continue
            acc |= code << filled
            filled += n
            while filled >= 8:
                append(acc & 0xFF)
                acc >>= 8
                filled -= 8
        if filled:
            append(acc & 0xFF)
        return bytes(out)

    def decode_symbols(self, data: bytes, count: int) -> np.ndarray:
        if count == 0:
            return np.zeros(0, dtype=np.uint8)
        if len(self.lengths) == 1:
            only = next(iter(self.lengths))
            return np.full(count, only, dtype=np.uint8)
        lut = _build_lut(self.codes, self.max_len)
        mask = (1 << self.max_len) - 1
        # Trailing pad so the last symbol can peek max_len bits.
        padded = data + b"\x00\x00\x00\x00"
        mv = memoryview(padded)
        bitpos = 0
        out = np.empty(count, dtype=np.uint8)
        for i in range(count):
            byte_i = bitpos >> 3
            shift = bitpos & 7
            peek = (mv[byte_i] | (mv[byte_i + 1] << 8) | (mv[byte_i + 2] << 16)) >> shift
            sym, nbits = lut[peek & mask]
            bitpos += nbits
            out[i] = sym
        return out


def _bit_reverse(value: int, nbits: int) -> int:
    result = 0
    for _ in range(nbits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _code_lengths(freq: dict[int, int]) -> dict[int, int]:
    items = [(f, s) for s, f in freq.items() if f > 0]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][1]: 0}

    heap: list[tuple[int, int, object]] = []
    nid = 0
    for freq_i, sym in items:
        heapq.heappush(heap, (freq_i, nid, ("leaf", sym)))
        nid += 1
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, nid, ("int", a, b)))
        nid += 1
    tree = heap[0][2]
    lengths: dict[int, int] = {}

    def walk(node: object, depth: int) -> None:
        kind = node[0]
        if kind == "leaf":
            lengths[node[1]] = depth if depth > 0 else 1
            return
        walk(node[1], depth + 1)
        walk(node[2], depth + 1)

    walk(tree, 0)
    return _limit_lengths(lengths, MAX_CODE_LEN)


def _limit_lengths(lengths: dict[int, int], max_len: int) -> dict[int, int]:
    """Kraft-inequality cap so decode LUTs stay small."""
    if not lengths or max(lengths.values()) <= max_len:
        return lengths
    limited = {s: min(l, max_len) for s, l in lengths.items()}
    # Ensure a valid prefix code: bump some lengths until Kraft ≤ 1.
    while True:
        kraft = sum(1 / (1 << l) for l in limited.values())
        if kraft <= 1.0 + 1e-12:
            return limited
        # Increase the shortest code (wastes a bit, always valid eventually).
        shortest = min(limited, key=lambda s: limited[s])
        if limited[shortest] >= max_len:
            # Fall back: every symbol max_len (not optimal, still decodable
            # only if 2^max_len >= alphabet). For 256 symbols, max_len 16 is OK.
            return {s: max_len for s in limited}
        limited[shortest] += 1


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    ordered = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for sym, length in ordered:
        code <<= length - prev_len
        # Transmit MSB first; BitWriter is LSB-first, so reverse once.
        codes[sym] = (_bit_reverse(code, length), length)
        code += 1
        prev_len = length
    return codes


def _build_lut(codes: dict[int, tuple[int, int]], max_len: int) -> list[tuple[int, int]]:
    size = 1 << max_len
    lut = [(0, max_len)] * size
    for sym, (lsb_code, nbits) in codes.items():
        step = 1 << nbits
        for extra in range(0, size, step):
            lut[lsb_code | extra] = (sym, nbits)
    return lut


def table_from_symbols(symbols: np.ndarray) -> HuffmanTable:
    flat = np.asarray(symbols).ravel()
    if flat.size == 0:
        return HuffmanTable(lengths={}, codes={}, max_len=1)
    unique, counts = np.unique(flat, return_counts=True)
    freq = {int(s): int(c) for s, c in zip(unique, counts)}
    lengths = _code_lengths(freq)
    nonzero = {s: l for s, l in lengths.items() if l > 0}
    codes = _canonical_codes(nonzero) if nonzero else {}
    max_len = max(nonzero.values()) if nonzero else 1
    return HuffmanTable(lengths=lengths, codes=codes, max_len=max_len)


def table_from_lengths(pairs: list[tuple[int, int]]) -> HuffmanTable:
    lengths = {int(s): int(l) for s, l in pairs}
    nonzero = {s: l for s, l in lengths.items() if l > 0}
    codes = _canonical_codes(nonzero) if nonzero else {}
    max_len = max(nonzero.values()) if nonzero else 1
    return HuffmanTable(lengths=lengths, codes=codes, max_len=max_len)


def dump_table(table: HuffmanTable) -> bytes:
    items = sorted(table.lengths.items())
    blob = struct.pack("<H", len(items))
    blob += bytes(sym for sym, _ in items)
    blob += bytes(length for _, length in items)
    return blob


def load_table(data: bytes, offset: int = 0) -> tuple[HuffmanTable, int]:
    (count,) = struct.unpack_from("<H", data, offset)
    offset += 2
    symbols = list(data[offset : offset + count])
    offset += count
    lengths = list(data[offset : offset + count])
    offset += count
    table = table_from_lengths(list(zip(symbols, lengths)))
    return table, offset

