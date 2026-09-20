"""Tile traversals. Coordinates are (row, col) in tile-local space."""

from __future__ import annotations

from functools import lru_cache

TRAV_ROW = 0
TRAV_SERP = 1
TRAV_COL = 2
TRAV_COL_SERP = 3

TRAV_NAMES = {
    TRAV_ROW: "row",
    TRAV_SERP: "serpentine",
    TRAV_COL: "column",
    TRAV_COL_SERP: "column_serpentine",
}


@lru_cache(maxsize=512)
def coord_list(h: int, w: int, trav: int) -> tuple[tuple[int, int], ...]:
    if h < 0 or w < 0:
        raise ValueError("negative tile size")
    out: list[tuple[int, int]] = []
    if trav == TRAV_ROW:
        for r in range(h):
            for c in range(w):
                out.append((r, c))
    elif trav == TRAV_SERP:
        for r in range(h):
            cols = range(w) if (r % 2) == 0 else range(w - 1, -1, -1)
            for c in cols:
                out.append((r, c))
    elif trav == TRAV_COL:
        for c in range(w):
            for r in range(h):
                out.append((r, c))
    elif trav == TRAV_COL_SERP:
        for c in range(w):
            rows = range(h) if (c % 2) == 0 else range(h - 1, -1, -1)
            for r in rows:
                out.append((r, c))
    else:
        raise ValueError(f"unknown traversal {trav}")
    return tuple(out)
