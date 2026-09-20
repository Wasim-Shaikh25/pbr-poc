"""Extract strided tile samples without loading a whole large tensor."""

from __future__ import annotations

import numpy as np

from pbr_core.safetensors_io import (
    TensorSpec,
    load_uint16,
    load_uint16_region,
    load_uint16_words,
    logical_2d_shape,
)
from pbr_core.tiles import tile_count, tile_origins


def entropy_subsample(spec: TensorSpec, max_words: int = 65536) -> np.ndarray:
    """Three contiguous chunks (start / mid / end) for cheap entropy."""
    n = spec.n_words
    if n <= max_words:
        return load_uint16(spec).reshape(-1)
    chunk = max(1, max_words // 3)
    starts = [0, max(0, n // 2 - chunk // 2), max(0, n - chunk)]
    parts = [load_uint16_words(spec, s, min(chunk, n - s)) for s in starts]
    return np.concatenate(parts)


def choose_sample_origins(
    rows: int,
    cols: int,
    block_size: int,
    n_samples: int,
) -> list[tuple[int, int, int, int]]:
    origins = tile_origins(rows, cols, block_size)
    if not origins:
        return []
    if len(origins) <= n_samples:
        return origins
    if n_samples == 1:
        return [origins[0]]
    idxs = np.linspace(0, len(origins) - 1, n_samples)
    picked = []
    seen: set[int] = set()
    for raw in idxs:
        i = int(round(float(raw)))
        i = min(max(i, 0), len(origins) - 1)
        if i in seen:
            continue
        seen.add(i)
        picked.append(origins[i])
    return picked


def load_sample_tiles(
    spec: TensorSpec,
    *,
    block_size: int,
    n_samples: int,
    max_full_words: int,
) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Return ``(sample_matrix, tile_list, strategy)``.

    Small tensors are encoded in full. Large tensors contribute a strided
    stack of tiles so previous-row geometry is preserved inside each tile.
    """
    rows, cols = logical_2d_shape(spec.shape)
    n_tiles = tile_count(rows, cols, block_size)
    if spec.n_words <= max_full_words or n_tiles <= n_samples:
        matrix = load_uint16(spec)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim > 2:
            matrix = matrix.reshape(rows, cols)
        tiles = _split_loaded(matrix, block_size)
        return np.ascontiguousarray(matrix, dtype=np.uint16), tiles, "full"

    origins = choose_sample_origins(rows, cols, block_size, n_samples)
    tiles = [load_uint16_region(spec, r0, c0, h, w) for r0, c0, h, w in origins]
    sample = _stack_tiles(tiles)
    return sample, tiles, "strided_tiles"


def _split_loaded(matrix: np.ndarray, block_size: int) -> list[np.ndarray]:
    from pbr_core.tiles import iter_tiles

    return [t.words.copy() for t in iter_tiles(matrix, block_size)]


def _stack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    if not tiles:
        return np.zeros((0, 0), dtype=np.uint16)
    width = max(t.shape[1] for t in tiles)
    parts = []
    for tile in tiles:
        if tile.shape[1] == width:
            parts.append(tile)
            continue
        padded = np.zeros((tile.shape[0], width), dtype=np.uint16)
        padded[:, : tile.shape[1]] = tile
        parts.append(padded)
    return np.concatenate(parts, axis=0)
