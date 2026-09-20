"""16×16 / 256-node tiling and causal traversals."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from pbr_q4.const import TILE, TRAV_COL, TRAV_COL_SERP, TRAV_ROW, TRAV_ROW_SERP


def tile_boxes(
    rows: int, cols: int, *, th: int = TILE, tw: int = TILE
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (row0, col0, h, w) for every tile covering *rows* × *cols*."""
    if rows < 0 or cols < 0:
        raise ValueError("negative shape")
    if rows == 0 or cols == 0:
        return
    for r0 in range(0, rows, th):
        rh = min(th, rows - r0)
        for c0 in range(0, cols, tw):
            cw = min(tw, cols - c0)
            yield r0, c0, rh, cw


def n_tiles(rows: int, cols: int, *, th: int = TILE, tw: int = TILE) -> int:
    if rows <= 0 or cols <= 0:
        return 0
    return ((rows + th - 1) // th) * ((cols + tw - 1) // tw)


def as_full_tiles(arr: np.ndarray, *, th: int = TILE, tw: int = TILE) -> np.ndarray:
    """View a rank-2 array whose dims are multiples of the tile as (T, th, tw)."""
    a = np.ascontiguousarray(arr)
    if a.ndim != 2:
        raise ValueError("as_full_tiles expects rank-2")
    r, c = int(a.shape[0]), int(a.shape[1])
    if r % th or c % tw:
        raise ValueError(f"shape {a.shape} is not aligned to {th}x{tw}")
    return a.reshape(r // th, th, c // tw, tw).swapaxes(1, 2).reshape(-1, th, tw)


def scatter_full_tiles(
    tiles: np.ndarray, rows: int, cols: int, *, th: int = TILE, tw: int = TILE
) -> np.ndarray:
    t = np.ascontiguousarray(tiles)
    nr, nc = rows // th, cols // tw
    return t.reshape(nr, nc, th, tw).swapaxes(1, 2).reshape(rows, cols)


def traversal_coords(h: int, w: int, trav: int) -> list[tuple[int, int]]:
    """Decoder order: only previously yielded nodes are reconstructed."""
    if h < 0 or w < 0:
        raise ValueError("negative tile")
    if h == 0 or w == 0:
        return []
    if trav == TRAV_ROW:
        return [(y, x) for y in range(h) for x in range(w)]
    if trav == TRAV_ROW_SERP:
        out: list[tuple[int, int]] = []
        for y in range(h):
            xs = range(w) if (y % 2 == 0) else range(w - 1, -1, -1)
            out.extend((y, x) for x in xs)
        return out
    if trav == TRAV_COL:
        return [(y, x) for x in range(w) for y in range(h)]
    if trav == TRAV_COL_SERP:
        out = []
        for x in range(w):
            ys = range(h) if (x % 2 == 0) else range(h - 1, -1, -1)
            out.extend((y, x) for y in ys)
        return out
    raise ValueError(f"unknown traversal {trav}")


def coords_to_order_index(h: int, w: int, trav: int) -> np.ndarray:
    """Map (y, x) → traversal index. Shape (h, w)."""
    idx = np.empty((h, w), dtype=np.int32)
    for i, (y, x) in enumerate(traversal_coords(h, w, trav)):
        idx[y, x] = i
    return idx


def gather_traversal(tile: np.ndarray, trav: int) -> np.ndarray:
    """Flatten *tile* in traversal order."""
    h, w = int(tile.shape[0]), int(tile.shape[1])
    coords = traversal_coords(h, w, trav)
    if not coords:
        return np.zeros(0, dtype=tile.dtype)
    ys = np.fromiter((c[0] for c in coords), dtype=np.intp, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.intp, count=len(coords))
    return np.ascontiguousarray(tile)[ys, xs]


def scatter_traversal(flat: np.ndarray, h: int, w: int, trav: int, dtype=np.uint8) -> np.ndarray:
    out = np.zeros((h, w), dtype=dtype)
    coords = traversal_coords(h, w, trav)
    if len(coords) != int(np.asarray(flat).size):
        raise ValueError("traversal scatter size mismatch")
    ys = np.fromiter((c[0] for c in coords), dtype=np.intp, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.intp, count=len(coords))
    out[ys, xs] = np.asarray(flat).ravel()
    return out


def reshape_1d_as_row(arr: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        return a
    return a.reshape(-1, a.shape[-1]) if a.size else a.reshape(0, 0)
