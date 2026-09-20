"""Duplicate / near-duplicate tile stats on cheap samples."""

from __future__ import annotations

import hashlib

import numpy as np


def _popcount16(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint32)
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    return (((x + (x >> 4)) & 0x0F0F) * 0x0101) >> 8


def tile_repetition(tiles: list[np.ndarray]) -> dict[str, float | int]:
    if not tiles:
        return {
            "sample_tiles": 0,
            "unique_tiles": 0,
            "exact_duplicate_rate": 0.0,
            "mean_xor_popcount_vs_prev": 0.0,
        }
    hashes: list[bytes] = []
    pop_vs_prev: list[float] = []
    prev: np.ndarray | None = None
    for tile in tiles:
        raw = np.ascontiguousarray(tile, dtype="<u2").tobytes()
        hashes.append(hashlib.sha256(raw).digest())
        if prev is not None and prev.shape == tile.shape:
            xor = prev ^ np.ascontiguousarray(tile, dtype=np.uint16)
            pop_vs_prev.append(float(np.mean(_popcount16(xor))))
        prev = np.ascontiguousarray(tile, dtype=np.uint16)
    unique = len(set(hashes))
    return {
        "sample_tiles": len(tiles),
        "unique_tiles": unique,
        "exact_duplicate_rate": float(1.0 - unique / len(tiles)),
        "mean_xor_popcount_vs_prev": float(np.mean(pop_vs_prev)) if pop_vs_prev else 0.0,
    }
