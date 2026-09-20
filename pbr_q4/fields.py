"""Extract / rebuild H95Q-S1 quantized fields (sign, exp, K-bit mantissa).

The numerical values are the frozen S1 quantized reference. This module does
not re-quantize from original BF16; it only splits already-quantized words.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pbr_core.bf16 import join_components, split_components


def kept_from_mant(mant: np.ndarray, keep: int) -> np.ndarray:
    m = np.asarray(mant, dtype=np.uint16).ravel()
    k = int(keep)
    if k <= 0:
        return np.zeros(m.shape, dtype=np.uint16)
    if k >= 7:
        return m & np.uint16(0x7F)
    removed = 7 - k
    return (m >> np.uint16(removed)) & np.uint16((1 << k) - 1)


def mant_from_kept(kept: np.ndarray, keep: int) -> np.ndarray:
    k = int(keep)
    g = np.asarray(kept, dtype=np.uint16).ravel()
    if k <= 0:
        return np.zeros(g.shape, dtype=np.uint16)
    if k >= 7:
        return g & np.uint16(0x7F)
    removed = 7 - k
    return (g << np.uint16(removed)) & np.uint16(0x7F)


def split_quantized(words: np.ndarray, keep: int) -> dict[str, Any]:
    """Split a uniform-keep quantized tensor into fields + dirty-bit exceptions."""
    w = np.ascontiguousarray(words, dtype=np.uint16)
    shape = tuple(int(x) for x in w.shape)
    flat = w.ravel()
    sign, exp, mant = split_components(flat)
    k = int(keep)
    kept = kept_from_mant(mant, k)
    recon = join_components(sign, exp, mant_from_kept(kept, k))
    dirty = recon != flat
    idx = np.flatnonzero(dirty).astype(np.int64)
    return {
        "shape": shape,
        "sign": sign.astype(np.uint8),
        "exp": exp.astype(np.uint8),
        "kept": kept,
        "keep": k,
        "exception_indices": idx,
        "exception_words": flat[idx].astype(np.uint16) if idx.size else np.zeros(0, dtype=np.uint16),
    }


def split_embed_tiers(words: np.ndarray, row_keeps: np.ndarray) -> dict[str, Any]:
    """Split a rank-2 embed matrix with per-row keep bits."""
    w = np.ascontiguousarray(words, dtype=np.uint16)
    if w.ndim != 2:
        raise ValueError("embed tiers require rank-2")
    rows, cols = int(w.shape[0]), int(w.shape[1])
    rk = np.asarray(row_keeps, dtype=np.int8)
    if int(rk.shape[0]) != rows:
        raise ValueError("row_keeps / rows mismatch")
    sign, exp, mant = split_components(w.ravel())
    sign2 = sign.reshape(rows, cols)
    exp2 = exp.reshape(rows, cols)
    mant2 = mant.reshape(rows, cols)
    groups: dict[int, dict[str, Any]] = {}
    recon = np.empty((rows, cols), dtype=np.uint16)
    for k in sorted({int(x) for x in rk.tolist()}):
        idx = np.flatnonzero(rk == k).astype(np.int32)
        block_m = mant2[idx]
        kept = kept_from_mant(block_m, k if k >= 0 else 0).reshape(idx.size, cols)
        store_k = 7 if k >= 7 else max(int(k), 0)
        groups[int(k)] = {
            "k": store_k,
            "row_indices": idx,
            "kept": kept,
            "n_rows": int(idx.size),
            "n_weights": int(idx.size * cols),
        }
        rec_m = mant_from_kept(kept.ravel(), store_k).reshape(idx.size, cols)
        rec = join_components(sign2[idx].ravel(), exp2[idx].ravel(), rec_m.ravel())
        recon[idx] = rec.reshape(idx.size, cols)
    dirty = recon.ravel() != w.ravel()
    eidx = np.flatnonzero(dirty).astype(np.int64)
    return {
        "shape": (rows, cols),
        "sign": sign.astype(np.uint8),
        "exp": exp.astype(np.uint8),
        "row_keeps": rk,
        "groups": groups,
        "exception_indices": eidx,
        "exception_words": w.ravel()[eidx].astype(np.uint16) if eidx.size else np.zeros(0, dtype=np.uint16),
    }


def join_uniform(
    sign: np.ndarray,
    exp: np.ndarray,
    kept: np.ndarray,
    keep: int,
    shape: tuple[int, ...],
    *,
    exception_indices: np.ndarray | None = None,
    exception_words: np.ndarray | None = None,
) -> np.ndarray:
    words = join_components(sign, exp, mant_from_kept(kept, keep))
    if exception_indices is not None and exception_words is not None and int(np.asarray(exception_indices).size):
        words = words.copy()
        words[np.asarray(exception_indices, dtype=np.int64)] = np.asarray(exception_words, dtype=np.uint16)
    return words.astype(np.uint16).reshape(shape)


def join_embed_tiers(
    sign: np.ndarray,
    exp: np.ndarray,
    row_keeps: np.ndarray,
    groups: dict[int, dict[str, Any]],
    shape: tuple[int, int],
    *,
    exception_indices: np.ndarray | None = None,
    exception_words: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = shape
    out = np.empty((rows, cols), dtype=np.uint16)
    sign2 = np.asarray(sign, dtype=np.uint8).reshape(rows, cols)
    exp2 = np.asarray(exp, dtype=np.uint8).reshape(rows, cols)
    for k, g in groups.items():
        idx = np.asarray(g["row_indices"], dtype=np.int32)
        kept = np.asarray(g["kept"], dtype=np.uint16)
        store_k = int(g["k"])
        rec_m = mant_from_kept(kept.ravel(), store_k)
        rec = join_components(sign2[idx].ravel(), exp2[idx].ravel(), rec_m)
        out[idx] = rec.reshape(idx.size, cols)
    words = out.ravel()
    if exception_indices is not None and exception_words is not None and int(np.asarray(exception_indices).size):
        words = words.copy()
        words[np.asarray(exception_indices, dtype=np.int64)] = np.asarray(exception_words, dtype=np.uint16)
        out = words.reshape(rows, cols)
    return out
