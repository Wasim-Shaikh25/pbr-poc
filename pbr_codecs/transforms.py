"""Cheap exact tile transforms for hierarchical near-duplicate refs."""

from __future__ import annotations

import numpy as np

XF_IDENTITY = 0
XF_TRANSPOSE = 1
XF_SIGN_XOR = 2
XF_BYTESWAP = 3

XF_NAMES = {
    XF_IDENTITY: "id",
    XF_TRANSPOSE: "transpose",
    XF_SIGN_XOR: "sign_xor",
    XF_BYTESWAP: "byteswap",
}


def apply_transform(words: np.ndarray, xf_id: int) -> np.ndarray | None:
    tile = np.ascontiguousarray(words, dtype=np.uint16)
    if xf_id == XF_IDENTITY:
        return tile
    if xf_id == XF_TRANSPOSE:
        if tile.ndim != 2 or tile.shape[0] != tile.shape[1]:
            return None
        return np.ascontiguousarray(tile.T)
    if xf_id == XF_SIGN_XOR:
        return tile ^ np.uint16(0x8000)
    if xf_id == XF_BYTESWAP:
        return ((tile & np.uint16(0x00FF)) << np.uint16(8)) | (tile >> np.uint16(8))
    raise ValueError(f"unknown transform id {xf_id}")


def invert_transform(words: np.ndarray, xf_id: int) -> np.ndarray:
    # All listed transforms are involutions.
    out = apply_transform(words, xf_id)
    if out is None:
        raise ValueError(f"cannot invert transform {xf_id} on shape {words.shape}")
    return out
