"""Matrix-friendly tiling. Tile geometry is stored so decode is exact."""

from __future__ import annotations

import math

import numpy as np

from pbr_core.types import TileInfo


def as_2d(words: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(words, dtype=np.uint16)
    if shape is not None:
        return arr.reshape(shape)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        return arr.reshape(arr.shape[0], int(np.prod(arr.shape[1:])))
    return arr


def choose_tile_hw(block_size: int, rows: int, cols: int) -> tuple[int, int]:
    """Pick a tile close to ``block_size`` weights that still has 2D structure.

    Prefer a multi-row strip when the matrix width itself is a good tile width.
    Otherwise use a roughly square tile so previous-row prediction is meaningful.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if cols <= block_size and cols >= 2:
        tile_rows = max(1, min(rows, block_size // cols))
        if tile_rows >= 2 or rows == 1:
            return tile_rows, cols
    side = max(1, int(math.sqrt(block_size)))
    tile_cols = min(cols, side)
    tile_rows = max(1, min(rows, block_size // max(tile_cols, 1)))
    if tile_rows * tile_cols < 1:
        return 1, 1
    return tile_rows, tile_cols


def iter_tiles(words_2d: np.ndarray, block_size: int) -> list[TileInfo]:
    rows, cols = int(words_2d.shape[0]), int(words_2d.shape[1])
    th, tw = choose_tile_hw(block_size, rows, cols)
    tiles: list[TileInfo] = []
    index = 0
    for r0 in range(0, rows, th):
        for c0 in range(0, cols, tw):
            tile = words_2d[r0 : r0 + th, c0 : c0 + tw]
            tiles.append(
                TileInfo(
                    words=tile,
                    row0=r0,
                    col0=c0,
                    rows=int(tile.shape[0]),
                    cols=int(tile.shape[1]),
                    index=index,
                )
            )
            index += 1
    return tiles


def place_tile(dest: np.ndarray, tile: np.ndarray, row0: int, col0: int) -> None:
    dest[row0 : row0 + tile.shape[0], col0 : col0 + tile.shape[1]] = tile


def tile_origins(rows: int, cols: int, block_size: int) -> list[tuple[int, int, int, int]]:
    """Return (row0, col0, height, width) for every tile without loading data."""
    if rows <= 0 or cols <= 0:
        return []
    th, tw = choose_tile_hw(block_size, rows, cols)
    out: list[tuple[int, int, int, int]] = []
    for r0 in range(0, rows, th):
        for c0 in range(0, cols, tw):
            out.append((r0, c0, min(th, rows - r0), min(tw, cols - c0)))
    return out


def tile_count(rows: int, cols: int, block_size: int) -> int:
    return len(tile_origins(rows, cols, block_size))
