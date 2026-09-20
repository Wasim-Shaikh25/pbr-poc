"""Shared types and the Stage 1A codec contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# On-wire tile prefix: mode_id(1) + rows(2) + cols(2) + row0(4) + col0(4) + payload_len(4)
TILE_HEADER_BYTES = 17

# Residual-kind tags stored as the first payload byte of predictor modes.
RES_RAW = 0
RES_CONSTANT = 1
RES_DEFAULT_LIST = 2
RES_DEFAULT_BITMAP = 3
RES_DICT = 4

MODE_RAW = 0
MODE_CONSTANT = 1
MODE_VALUE_DICT = 2
MODE_PREV_VALUE = 3
MODE_PREV_ROW = 4
MODE_CONST_PRED = 5
MODE_DUP_REF = 6
MODE_COMPONENTS = 7
MODE_REF_PREV = 8
MODE_EXP_HUFFMAN = 9

MODE_NAMES = {
    MODE_RAW: "raw_bf16",
    MODE_CONSTANT: "constant",
    MODE_VALUE_DICT: "value_dict",
    MODE_PREV_VALUE: "prev_value",
    MODE_PREV_ROW: "prev_row",
    MODE_CONST_PRED: "const_pred",
    MODE_DUP_REF: "duplicate_ref",
    MODE_COMPONENTS: "bf16_components",
    MODE_REF_PREV: "ref_prev_tile",
    MODE_EXP_HUFFMAN: "bf16_exp_huffman",
}

RES_NAMES = {
    RES_RAW: "raw_residual",
    RES_CONSTANT: "constant_residual",
    RES_DEFAULT_LIST: "default_list",
    RES_DEFAULT_BITMAP: "default_bitmap",
    RES_DICT: "residual_dict",
}


@dataclass(frozen=True)
class CostEstimate:
    """Complete anticipated bytes, including every decoder dependency."""

    total_bytes: int
    mode_name: str
    payload_bytes: int = 0
    metadata_bytes: int = 0


@dataclass
class EncodedBlock:
    """One encoded tile. ``total_bytes`` is the complete on-wire size."""

    mode_id: int
    mode_name: str
    payload: bytes
    metadata: bytes = b""
    rows: int = 0
    cols: int = 0
    row0: int = 0
    col0: int = 0

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    @property
    def metadata_bytes(self) -> int:
        return TILE_HEADER_BYTES + len(self.metadata)

    @property
    def total_bytes(self) -> int:
        return TILE_HEADER_BYTES + len(self.metadata) + len(self.payload)

    def to_cost(self) -> CostEstimate:
        return CostEstimate(
            total_bytes=self.total_bytes,
            mode_name=self.mode_name,
            payload_bytes=self.payload_bytes,
            metadata_bytes=self.metadata_bytes,
        )


@dataclass
class TileInfo:
    words: np.ndarray
    row0: int
    col0: int
    rows: int
    cols: int
    index: int = 0


@dataclass
class EncodeContext:
    """Causal context available to both encoder and decoder."""

    tile_index: int = 0
    ncols: int = 0
    seen_exact: dict[bytes, int] = field(default_factory=dict)
    prev_tile: np.ndarray | None = None
    prev_index: int | None = None

    def record(self, tile: np.ndarray) -> None:
        raw = np.ascontiguousarray(tile, dtype="<u2").tobytes()
        self.seen_exact.setdefault(raw, self.tile_index)
        self.prev_tile = np.array(tile, dtype=np.uint16, copy=True)
        self.prev_index = self.tile_index


class Codec(Protocol):
    name: str

    def estimate(self, words: np.ndarray, context: EncodeContext) -> CostEstimate:
        """Return complete anticipated bytes, including metadata."""

    def encode(self, words: np.ndarray, context: EncodeContext) -> EncodedBlock | None:
        """Return mode metadata and exact payload, or None if inapplicable."""

    def decode(self, encoded: EncodedBlock, context: EncodeContext) -> np.ndarray:
        """Reconstruct exact original uint16 words."""
