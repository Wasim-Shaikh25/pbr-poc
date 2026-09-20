"""Causal X/Y predictors and residual maps for 256-node tiles.

The decoder may use only already-reconstructed neighbors. WEST/NORTH fall
back to PREVIOUS (or 0 at the first node) when that neighbor is not yet
decoded in the selected traversal.
"""

from __future__ import annotations

import numpy as np

from pbr_q4.const import (
    PRED_AVG,
    PRED_LEFT,
    PRED_PAETH,
    PRED_PREVIOUS,
    PRED_UP,
    TRAV_COL,
    TRAV_COL_SERP,
    TRAV_ROW,
    TRAV_ROW_SERP,
)
from pbr_q4.tiles import traversal_coords


def _mask(nbits: int) -> int:
    if nbits < 0 or nbits > 8:
        raise ValueError(f"nbits must be 0..8, got {nbits}")
    return (1 << nbits) - 1 if nbits else 0


def _seen_left(trav: int, y: int, x: int) -> bool:
    """True iff (y, x-1) is visited before (y, x) under *trav* (x>0 assumed)."""
    if x <= 0:
        return False
    if trav == TRAV_ROW:
        return True
    if trav == TRAV_ROW_SERP:
        return (y % 2) == 0
    if trav == TRAV_COL:
        return True  # previous columns are complete
    if trav == TRAV_COL_SERP:
        return True
    return False


def _seen_up(trav: int, y: int, x: int) -> bool:
    if y <= 0:
        return False
    if trav == TRAV_ROW:
        return True
    if trav == TRAV_ROW_SERP:
        return True  # previous rows complete
    if trav == TRAV_COL:
        return True
    if trav == TRAV_COL_SERP:
        return (x % 2) == 0
    return False


def _seen_nw(trav: int, y: int, x: int) -> bool:
    if y <= 0 or x <= 0:
        return False
    # NW is on the previous row and previous column — decoded once that row/col is done.
    if trav in (TRAV_ROW, TRAV_ROW_SERP):
        return True
    if trav == TRAV_COL:
        return True
    if trav == TRAV_COL_SERP:
        return True
    return False


def predict_one(
    grid: np.ndarray,
    y: int,
    x: int,
    *,
    pred: int,
    prev: int,
    trav: int,
    nbits: int,
) -> int:
    """Return the causal prediction at (y, x). *grid* holds reconstructed nodes."""
    h, w = int(grid.shape[0]), int(grid.shape[1])
    left = int(grid[y, x - 1]) if x > 0 and _seen_left(trav, y, x) else None
    up = int(grid[y - 1, x]) if y > 0 and _seen_up(trav, y, x) else None
    nw = int(grid[y - 1, x - 1]) if y > 0 and x > 0 and _seen_nw(trav, y, x) else None

    if pred == PRED_PREVIOUS:
        return prev
    if pred == PRED_LEFT:
        return left if left is not None else prev
    if pred == PRED_UP:
        return up if up is not None else prev
    if pred == PRED_AVG:
        if left is not None and up is not None:
            return (left + up) // 2
        if left is not None:
            return left
        if up is not None:
            return up
        return prev
    if pred == PRED_PAETH:
        if left is None and up is None:
            return prev
        if left is None:
            return up if up is not None else prev
        if up is None:
            return left
        est = left + up - (nw if nw is not None else 0)
        cand = [(abs(est - left), 0, left), (abs(est - up), 1, up)]
        if nw is not None:
            cand.append((abs(est - nw), 2, nw))
        cand.sort()
        return cand[0][2]
    raise ValueError(f"unknown predictor {pred}")


def residual_tile(codes: np.ndarray, *, pred: int, trav: int, nbits: int) -> np.ndarray:
    """Return residuals in *spatial* layout: (actual - pred) mod 2^nbits."""
    tile = np.ascontiguousarray(codes)
    h, w = int(tile.shape[0]), int(tile.shape[1])
    mask = _mask(nbits)
    out = np.zeros((h, w), dtype=np.uint8)
    recon = np.zeros((h, w), dtype=np.uint16)
    prev = 0
    for y, x in traversal_coords(h, w, trav):
        p = predict_one(recon, y, x, pred=pred, prev=prev, trav=trav, nbits=nbits)
        actual = int(tile[y, x]) & mask
        out[y, x] = (actual - p) & mask
        recon[y, x] = actual
        prev = actual
    return out


def reconstruct_tile(
    residuals: np.ndarray, *, pred: int, trav: int, nbits: int
) -> np.ndarray:
    res = np.ascontiguousarray(residuals)
    h, w = int(res.shape[0]), int(res.shape[1])
    mask = _mask(nbits)
    recon = np.zeros((h, w), dtype=np.uint16)
    prev = 0
    for y, x in traversal_coords(h, w, trav):
        p = predict_one(recon, y, x, pred=pred, prev=prev, trav=trav, nbits=nbits)
        actual = (p + int(res[y, x])) & mask
        recon[y, x] = actual
        prev = actual
    return recon.astype(np.uint8)


def residual_flat(codes: np.ndarray, *, pred: int, trav: int, nbits: int) -> np.ndarray:
    """Residuals in traversal order (for RUN / rANS / pair-along-path)."""
    spatial = residual_tile(codes, pred=pred, trav=trav, nbits=nbits)
    coords = traversal_coords(int(spatial.shape[0]), int(spatial.shape[1]), trav)
    if not coords:
        return np.zeros(0, dtype=np.uint8)
    ys = np.fromiter((c[0] for c in coords), dtype=np.intp, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.intp, count=len(coords))
    return spatial[ys, xs]


def reconstruct_from_flat(
    flat_res: np.ndarray, h: int, w: int, *, pred: int, trav: int, nbits: int
) -> np.ndarray:
    spatial = np.zeros((h, w), dtype=np.uint8)
    coords = traversal_coords(h, w, trav)
    if len(coords) != int(np.asarray(flat_res).size):
        raise ValueError("flat residual size mismatch")
    ys = np.fromiter((c[0] for c in coords), dtype=np.intp, count=len(coords))
    xs = np.fromiter((c[1] for c in coords), dtype=np.intp, count=len(coords))
    spatial[ys, xs] = np.asarray(flat_res, dtype=np.uint8).ravel()
    return reconstruct_tile(spatial, pred=pred, trav=trav, nbits=nbits)


def score_residual(res: np.ndarray) -> tuple[float, float, int]:
    """Return (zero_rate, unique_count, n) for candidate ranking."""
    flat = np.asarray(res).ravel()
    n = int(flat.size)
    if n == 0:
        return 1.0, 0, 0
    z = float(np.mean(flat == 0))
    uniq = int(np.unique(flat).size)
    return z, uniq, n


def _mask_arr(tiles: np.ndarray, nbits: int) -> tuple[np.ndarray, int]:
    mask = _mask(nbits)
    t = np.ascontiguousarray(tiles, dtype=np.uint16) & np.uint16(mask)
    return t, mask


def batched_residual_maps(tiles: np.ndarray, nbits: int) -> list[tuple[int, int, np.ndarray]]:
    """Vectorized residual maps for a stack of full tiles ``(T, H, W)``.

    Covers all 4 traversals × 5 predictors. Geometry matches ``residual_tile``.
    """
    t, mask = _mask_arr(tiles, nbits)
    if t.ndim != 3:
        raise ValueError("batched tiles must be (T,H,W)")
    T, H, W = t.shape
    out: list[tuple[int, int, np.ndarray]] = []
    ti = t.astype(np.int32)

    def _res(pred_grid: np.ndarray) -> np.ndarray:
        return ((ti - pred_grid.astype(np.int32)) & mask).astype(np.uint8)

    # --- ROW / COL spatial neighbors (decoded in both ROW and COL order) ---
    left = np.zeros_like(ti)
    left[:, :, 1:] = ti[:, :, :-1]
    up = np.zeros_like(ti)
    up[:, 1:, :] = ti[:, :-1, :]
    nw = np.zeros_like(ti)
    nw[:, 1:, 1:] = ti[:, :-1, :-1]

    # PREVIOUS along ROW = left, with first-of-row = last of previous row
    prev_row = left.copy()
    if H > 1 and W > 0:
        prev_row[:, 1:, 0] = ti[:, :-1, -1]

    prev_col = up.copy()
    if W > 1 and H > 0:
        prev_col[:, 0, 1:] = ti[:, -1, :-1]

    # ROW_SERP PREVIOUS
    prev_rserp = np.zeros_like(ti)
    prev_rserp[:, 0::2, 1:] = ti[:, 0::2, :-1]
    if H > 1:
        prev_rserp[:, 1::2, :-1] = ti[:, 1::2, 1:]
        n_odd = (H + 1) // 2 - (1 if H % 2 == 0 else 0)
        # start of odd row (rightmost) = end of previous even row (rightmost)
        odd = np.arange(1, H, 2)
        prev_rserp[:, odd, -1] = ti[:, odd - 1, -1]
        even2 = np.arange(2, H, 2)
        if even2.size:
            prev_rserp[:, even2, 0] = ti[:, even2 - 1, 0]

    # COL_SERP PREVIOUS
    prev_cserp = np.zeros_like(ti)
    prev_cserp[:, 1:, 0::2] = ti[:, :-1, 0::2]
    if W > 1:
        prev_cserp[:, :-1, 1::2] = ti[:, 1:, 1::2]
        oddc = np.arange(1, W, 2)
        prev_cserp[:, -1, oddc] = ti[:, -1, oddc - 1]
        even2c = np.arange(2, W, 2)
        if even2c.size:
            prev_cserp[:, 0, even2c] = ti[:, 0, even2c - 1]

    avg = (left + up) // 2
    avg_row = avg.copy()
    avg_row[:, 0, :] = left[:, 0, :]
    avg_row[:, :, 0] = up[:, :, 0]

    est = left + up - nw
    dL = np.abs(est - left)
    dU = np.abs(est - up)
    dNW = np.abs(est - nw)
    paeth = left.copy()
    use_u = (dU < dL) & (dU <= dNW)
    use_nw = (dNW < dL) & (dNW < dU)
    paeth = np.where(use_u, up, paeth)
    paeth = np.where(use_nw, nw, paeth)
    paeth_row = paeth.copy()
    paeth_row[:, 0, :] = left[:, 0, :]
    paeth_row[:, :, 0] = up[:, :, 0]

    up_avail = np.zeros((T, H, W), dtype=bool)
    up_avail[:, 1:, :] = True
    left_avail = np.zeros((T, H, W), dtype=bool)
    left_avail[:, :, 1:] = True
    # Serpentine AVG/PAETH stay on the sequential path (partial / tests).
    maps = [
        (TRAV_ROW, PRED_PREVIOUS, prev_row),
        (TRAV_ROW, PRED_LEFT, prev_row),
        (TRAV_ROW, PRED_UP, np.where(up_avail, up, prev_row)),
        (TRAV_ROW, PRED_AVG, avg_row),
        (TRAV_ROW, PRED_PAETH, paeth_row),
        (TRAV_COL, PRED_PREVIOUS, prev_col),
        (TRAV_COL, PRED_LEFT, np.where(left_avail, left, prev_col)),
        (TRAV_COL, PRED_UP, prev_col),
        (TRAV_COL, PRED_AVG, avg_row),
        (TRAV_COL, PRED_PAETH, paeth_row),
        (TRAV_ROW_SERP, PRED_PREVIOUS, prev_rserp),
        (TRAV_ROW_SERP, PRED_LEFT, prev_rserp),
        (TRAV_ROW_SERP, PRED_UP, np.where(up_avail, up, prev_rserp)),
        (TRAV_COL_SERP, PRED_PREVIOUS, prev_cserp),
        (TRAV_COL_SERP, PRED_LEFT, np.where(left_avail, left, prev_cserp)),
        (TRAV_COL_SERP, PRED_UP, prev_cserp),
    ]
    for trav, pred, grid in maps:
        out.append((trav, pred, _res(grid)))
    return out


def pick_best_geometry_batched(tiles: np.ndarray, nbits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trav[T], pred[T], residual[T,H,W]) for a full-tile stack."""
    maps = batched_residual_maps(tiles, nbits)
    T = int(tiles.shape[0])
    best_z = np.full(T, -1.0)
    best_u = np.full(T, 10**9)
    best_pred = np.full(T, 99, dtype=np.int16)
    best_trav = np.full(T, 99, dtype=np.int16)
    best_res = np.empty(tiles.shape, dtype=np.uint8)
    for trav, pred, res in maps:
        z = np.mean(res == 0, axis=(1, 2))
        # unique count is expensive; use a cheap proxy (sum of occupancy)
        # plus zero-rate. Tie-break pred, trav.
        better = (
            (z > best_z + 1e-12)
            | ((np.abs(z - best_z) <= 1e-12) & (pred < best_pred))
            | ((np.abs(z - best_z) <= 1e-12) & (pred == best_pred) & (trav < best_trav))
        )
        if not np.any(better):
            continue
        best_z = np.where(better, z, best_z)
        best_pred = np.where(better, pred, best_pred)
        best_trav = np.where(better, trav, best_trav)
        best_res[better] = res[better]
    return best_trav.astype(np.uint8), best_pred.astype(np.uint8), best_res


def pick_best_geometry(codes: np.ndarray, nbits: int) -> tuple[int, int, np.ndarray]:
    """Choose (trav, pred, spatial_residual) with max zero-rate then min unique.

    Tie-break: TRAV_ROW / PRED_PREVIOUS (stable, causal, cheap).
    """
    tile = np.ascontiguousarray(codes)
    best: tuple[int, int, np.ndarray] | None = None
    best_key: tuple = (1.0, 10**9, 99, 99)
    for trav, pred in (
        (t, p)
        for t in (TRAV_ROW, TRAV_ROW_SERP, TRAV_COL, TRAV_COL_SERP)
        for p in (PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH)
    ):
        res = residual_tile(tile, pred=pred, trav=trav, nbits=nbits)
        z, uniq, _n = score_residual(res)
        key = (-z, uniq, pred, trav)
        if key < best_key:
            best_key = key
            best = (trav, pred, res)
    assert best is not None
    return best
