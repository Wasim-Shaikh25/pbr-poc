"""PBR Stage 1A container. Reported size is the complete on-disk blob."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

from pbr_core.hashing import sha256_bytes, sha256_words
from pbr_core.types import TILE_HEADER_BYTES, EncodedBlock

MAGIC = b"PBR1"
VERSION = 1
_TILE_PREFIX = struct.Struct("<BHHIII")  # mode, rows, cols, row0, col0, payload_len


@dataclass
class TensorBlob:
    name: str
    shape: tuple[int, ...]
    dtype: str
    block_size: int
    tile_rows: int
    tile_cols: int
    n_words: int
    sha256: str
    tiles: list[EncodedBlock] = field(default_factory=list)


@dataclass
class PBRContainer:
    tensors: list[TensorBlob] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def dumps(self) -> bytes:
        header = {
            "format": "PBR-Direct Stage 1A",
            "version": VERSION,
            "note": "Controlled-method container. Not a real-model result.",
            "tensors": [
                {
                    "name": t.name,
                    "shape": list(t.shape),
                    "dtype": t.dtype,
                    "block_size": t.block_size,
                    "tile_rows": t.tile_rows,
                    "tile_cols": t.tile_cols,
                    "n_words": t.n_words,
                    "sha256": t.sha256,
                    "n_tiles": len(t.tiles),
                    "modes": [tile.mode_name for tile in t.tiles],
                }
                for t in self.tensors
            ],
            "extra": self.extra,
        }
        header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = bytearray()
        for tensor in self.tensors:
            for tile in tensor.tiles:
                payload.extend(
                    _TILE_PREFIX.pack(
                        tile.mode_id,
                        tile.rows,
                        tile.cols,
                        tile.row0,
                        tile.col0,
                        len(tile.payload),
                    )
                )
                payload.extend(tile.payload)
        return MAGIC + struct.pack("<HI", VERSION, len(header_bytes)) + header_bytes + bytes(payload)

    @classmethod
    def loads(cls, data: bytes) -> "PBRContainer":
        if data[:4] != MAGIC:
            raise ValueError(f"Bad magic: {data[:4]!r}")
        version, header_len = struct.unpack_from("<HI", data, 4)
        if version != VERSION:
            raise ValueError(f"Unsupported container version {version}")
        header_start = 10
        header_end = header_start + header_len
        header = json.loads(data[header_start:header_end].decode("utf-8"))
        offset = header_end
        tensors: list[TensorBlob] = []
        for spec in header["tensors"]:
            tiles: list[EncodedBlock] = []
            for i in range(int(spec["n_tiles"])):
                if offset + TILE_HEADER_BYTES > len(data):
                    raise ValueError(f"Truncated tile header at tile {i}")
                mode_id, rows, cols, row0, col0, plen = _TILE_PREFIX.unpack_from(data, offset)
                offset += TILE_HEADER_BYTES
                payload = data[offset : offset + plen]
                if len(payload) != plen:
                    raise ValueError(f"Truncated tile payload at tile {i}")
                offset += plen
                mode_name = spec["modes"][i] if i < len(spec.get("modes", [])) else f"mode_{mode_id}"
                tiles.append(
                    EncodedBlock(
                        mode_id=mode_id,
                        mode_name=mode_name,
                        payload=payload,
                        rows=rows,
                        cols=cols,
                        row0=row0,
                        col0=col0,
                    )
                )
            tensors.append(
                TensorBlob(
                    name=spec["name"],
                    shape=tuple(spec["shape"]),
                    dtype=spec["dtype"],
                    block_size=int(spec["block_size"]),
                    tile_rows=int(spec["tile_rows"]),
                    tile_cols=int(spec["tile_cols"]),
                    n_words=int(spec["n_words"]),
                    sha256=spec["sha256"],
                    tiles=tiles,
                )
            )
        if offset != len(data):
            raise ValueError(f"Container has {len(data) - offset} trailing bytes")
        return cls(tensors=tensors, extra=header.get("extra") or {})

    def file_sha256(self) -> str:
        """SHA-256 of the complete on-disk blob from dumps()."""
        return sha256_bytes(self.dumps())


def iter_raw_tiles(data: bytes):
    """Walk on-disk tiles without parsing the JSON header (tunnel hot path)."""
    if data[:4] != MAGIC:
        raise ValueError(f"Bad magic: {data[:4]!r}")
    version, header_len = struct.unpack_from("<HI", data, 4)
    if version != VERSION:
        raise ValueError(f"Unsupported container version {version}")
    offset = 10 + header_len
    n = len(data)
    while offset < n:
        if offset + TILE_HEADER_BYTES > n:
            raise ValueError("truncated tile header")
        mode_id, rows, cols, row0, col0, plen = _TILE_PREFIX.unpack_from(data, offset)
        offset += TILE_HEADER_BYTES
        end = offset + plen
        if end > n:
            raise ValueError("truncated tile payload")
        yield mode_id, int(rows), int(cols), int(row0), int(col0), data[offset:end]
        offset = end


def attach_geometry(block: EncodedBlock, tile_rows: int, tile_cols: int, row0: int, col0: int) -> EncodedBlock:
    block.rows = tile_rows
    block.cols = tile_cols
    block.row0 = row0
    block.col0 = col0
    return block


def raw_word_bytes(words: np.ndarray) -> int:
    return int(np.asarray(words).size * 2)


def original_sha256(words: np.ndarray) -> str:
    return sha256_words(words)
