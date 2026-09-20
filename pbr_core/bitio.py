"""LSB-first bit packing used by dictionary and component codecs."""

from __future__ import annotations

import numpy as np


class BitWriter:
    def __init__(self) -> None:
        self._bits = 0
        self._nbits = 0
        self._buf = bytearray()

    def write(self, value: int, nbits: int) -> None:
        if nbits < 0:
            raise ValueError("nbits must be >= 0")
        if nbits == 0:
            return
        if nbits > 32:
            raise ValueError("nbits must be <= 32")
        value &= (1 << nbits) - 1
        self._bits |= value << self._nbits
        self._nbits += nbits
        while self._nbits >= 8:
            self._buf.append(self._bits & 0xFF)
            self._bits >>= 8
            self._nbits -= 8

    def write_array(self, values: np.ndarray, nbits: int) -> None:
        for v in np.asarray(values).ravel():
            self.write(int(v), nbits)

    def finalize(self) -> bytes:
        if self._nbits:
            self._buf.append(self._bits & 0xFF)
            self._bits = 0
            self._nbits = 0
        return bytes(self._buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bitpos = 0

    def read(self, nbits: int) -> int:
        if nbits == 0:
            return 0
        result = 0
        got = 0
        total_bits = len(self._data) * 8
        while got < nbits:
            if self._bitpos >= total_bits:
                raise ValueError("BitReader overrun")
            byte_i = self._bitpos // 8
            bit_i = self._bitpos % 8
            take = min(nbits - got, 8 - bit_i)
            chunk = (self._data[byte_i] >> bit_i) & ((1 << take) - 1)
            result |= chunk << got
            self._bitpos += take
            got += take
        return result

    def read_array(self, count: int, nbits: int) -> np.ndarray:
        out = np.empty(count, dtype=np.uint32)
        for i in range(count):
            out[i] = self.read(nbits)
        return out

    def rewind(self, nbits: int) -> None:
        if nbits < 0 or self._bitpos < nbits:
            raise ValueError("BitReader rewind out of range")
        self._bitpos -= nbits


def bits_needed(n_symbols: int) -> int:
    if n_symbols <= 1:
        return 0
    return int(n_symbols - 1).bit_length()


def pack_ids(ids: np.ndarray, nbits: int) -> bytes:
    w = BitWriter()
    w.write_array(ids, nbits)
    return w.finalize()


def unpack_ids(data: bytes, count: int, nbits: int) -> np.ndarray:
    r = BitReader(data)
    return r.read_array(count, nbits).astype(np.uint16)
