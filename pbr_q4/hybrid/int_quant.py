"""Groupwise INT quantizer with MSE-first, spatial near-tie scale search.

For each group, several scales compete on reconstruction MSE. Among
near-ties (MSE ≤ ``mse_tie_ratio`` × best), the pick that lowers LEFT/UP
neighbor-code disagreement wins. A materially worse reconstruction never
wins on spatial score alone (λ is a tie-break, quality dominates).

Sparse high-error positions are stored as original BF16 words. Decode of
*this* quantized reference is exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pbr_q4.hybrid.bits import (
    as_rank2,
    bf16_u16_to_f32,
    f32_to_bf16_u16,
    group_ids,
    min_scale,
    qmax_for_bits,
)

_SCALE_FACTORS = np.array([0.94, 1.00, 1.12], dtype=np.float32)
_CHUNK_ROWS = 4096


def _special_mask_f32(x: np.ndarray) -> np.ndarray:
    return ~np.isfinite(x)


def _chunk_candidates(
    rows_f32: np.ndarray,
    valid: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scales/zp/stored/mse for a row chunk.

    shapes: scales/zp/mse (n_cand, R, n_gc), stored (n_cand, R, n_gc, G).
    """
    r, c = int(rows_f32.shape[0]), int(rows_f32.shape[1])
    g = int(group_size)
    n_gc = (c + g - 1) // g
    cpad = n_gc * g
    qmax = qmax_for_bits(bits)
    qrange = np.float32(2 * qmax)
    buf = np.zeros((r, cpad), dtype=np.float32)
    vbuf = np.zeros((r, cpad), dtype=bool)
    buf[:, :c] = rows_f32
    vbuf[:, :c] = valid
    groups = buf.reshape(r, n_gc, g)
    v_g = vbuf.reshape(r, n_gc, g)
    masked = np.where(v_g, groups, np.nan)
    with np.errstate(invalid="ignore"):
        wmin = np.nanmin(masked, axis=-1).astype(np.float32)
        wmax = np.nanmax(masked, axis=-1).astype(np.float32)
    wmin = np.where(np.isfinite(wmin), wmin, 0.0)
    wmax = np.where(np.isfinite(wmax), wmax, 0.0)
    absmax = np.maximum(np.maximum(np.abs(wmin), np.abs(wmax)), min_scale())

    span = np.maximum(wmax - wmin, min_scale())
    mm_scale = (span / qrange)[None, ...] * _SCALE_FACTORS[:, None, None]
    mm_scale = np.maximum(mm_scale, min_scale())
    mm_zp = np.rint((-wmin[None, ...] / mm_scale)).astype(np.int16)
    mm_zp = np.clip(mm_zp, 0, int(qrange))

    sym_scale = (absmax / np.float32(qmax))[None, ...] * _SCALE_FACTORS[:, None, None]
    sym_scale = np.maximum(sym_scale, min_scale())
    sym_zp = np.full(sym_scale.shape, qmax, dtype=np.int16)

    scales = np.concatenate([sym_scale, mm_scale], axis=0)
    zp = np.concatenate([sym_zp, mm_zp], axis=0)
    q = np.clip(
        np.rint(groups[None] / scales[..., None] + zp.astype(np.float32)[..., None]),
        0,
        int(qrange),
    ).astype(np.int16)
    recon = (q.astype(np.float32) - zp.astype(np.float32)[..., None]) * scales[..., None]
    diff2 = np.where(v_g[None], (groups[None] - recon) ** 2, 0.0)
    denom = np.maximum(v_g.sum(axis=-1).astype(np.float32), 1.0)
    mse = diff2.sum(axis=-1) / denom[None, :, :]
    return scales, zp, q.astype(np.uint16), mse


def _spatial_chunk(
    stored: np.ndarray,
    valid: np.ndarray,
    group_size: int,
    prev_ref: np.ndarray | None,
) -> np.ndarray:
    """LEFT + UP disagreement per candidate / row / group.

    *stored* is (n_cand, R, Cpad). *valid* is (R, C).
    *prev_ref* is (Cpad,) codes from the previous chunk's last chosen row.
    UP uses the previous *row's same-candidate* codes (vectorized). Quality
    still dominates via the MSE near-tie gate.
    """
    n_cand, r, cpad = stored.shape
    g = int(group_size)
    n_gc = cpad // g
    c = int(valid.shape[1])
    left = np.zeros((n_cand, r, cpad), dtype=np.float32)
    if cpad > 1:
        d = np.abs(stored[:, :, 1:].astype(np.int16) - stored[:, :, :-1].astype(np.int16))
        disagree = (stored[:, :, 1:] != stored[:, :, :-1]).astype(np.float32)
        left[:, :, 1:] = d.astype(np.float32) * np.float32(0.25) + disagree
    up = np.zeros((n_cand, r, cpad), dtype=np.float32)
    if r > 1:
        ud = np.abs(stored[:, 1:, :].astype(np.int16) - stored[:, :-1, :].astype(np.int16))
        up[:, 1:, :] = ud.astype(np.float32) * np.float32(0.25) + (
            stored[:, 1:, :] != stored[:, :-1, :]
        ).astype(np.float32)
    if prev_ref is not None:
        prev = np.asarray(prev_ref, dtype=np.uint16).reshape(1, 1, -1)
        ud0 = np.abs(stored[:, :1, :].astype(np.int16) - prev.astype(np.int16))
        up[:, :1, :] = ud0.astype(np.float32) * np.float32(0.25) + (
            stored[:, :1, :] != prev
        ).astype(np.float32)
    mask = np.zeros((r, cpad), dtype=np.float32)
    mask[:, :c] = valid.astype(np.float32)
    left *= mask[None, :, :]
    up *= mask[None, :, :]
    left_g = left.reshape(n_cand, r, n_gc, g).sum(axis=-1)
    up_g = up.reshape(n_cand, r, n_gc, g).sum(axis=-1)
    denom = np.maximum(mask.reshape(r, n_gc, g).sum(axis=-1), 1.0)
    return (left_g + up_g) / denom[None, :, :]


def pick_scales(
    mse: np.ndarray,
    spatial: np.ndarray,
    *,
    tie_ratio: float,
) -> tuple[np.ndarray, int]:
    """Among near-tie MSE candidates, minimize spatial score.

    *mse* / *spatial* are (n_cand, ...). Returns pick (same trailing shape)
    and the number of groups that did not take the unique MSE-best candidate.
    """
    best = np.min(mse, axis=0)
    near = mse <= (np.float32(tie_ratio) * best)[None, ...]
    score = np.where(near, spatial, np.float32(1e9))
    rank = score * np.float32(1000.0) + mse
    pick = np.argmin(rank, axis=0).astype(np.int32)
    mse_best = np.argmin(mse, axis=0).astype(np.int32)
    n_override = int(np.count_nonzero(pick != mse_best))
    return pick, n_override


def _select_outliers(
    err: np.ndarray,
    special: np.ndarray,
    original: np.ndarray,
    scale_of: np.ndarray,
    outlier_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    special_idx = np.flatnonzero(special.ravel())
    if outlier_frac <= 0:
        words = (
            original.ravel()[special_idx].astype(np.uint16)
            if special_idx.size
            else np.zeros(0, dtype=np.uint16)
        )
        return special_idx.astype(np.int64), words

    thr = np.float32(0.60) * np.maximum(scale_of, min_scale())
    cand = (err > thr) & (~special)
    cand_idx = np.flatnonzero(cand.ravel())
    n = int(original.size)
    cap = max(0, int(np.ceil(float(outlier_frac) * n)))
    if cand_idx.size > cap:
        e = err.ravel()[cand_idx]
        keep = np.argpartition(e, -cap)[-cap:]
        cand_idx = cand_idx[keep]
    idx = np.unique(np.concatenate([special_idx.astype(np.int64), cand_idx.astype(np.int64)]))
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint16)
    return idx, original.ravel()[idx].astype(np.uint16)


def quantize_matrix(
    words: np.ndarray,
    bits: int,
    *,
    group_size: int = 128,
    mse_tie_ratio: float = 1.02,
    outlier_frac: float = 0.001,
    name: str = "",
    family: str = "",
) -> dict[str, Any]:
    """Quantize BF16-uint16 words to groupwise INT + BF16 q-ref dict."""
    arr2, orig, rows, cols = as_rank2(np.ascontiguousarray(words, dtype=np.uint16))
    w_f32 = bf16_u16_to_f32(arr2)
    special = _special_mask_f32(w_f32)
    valid = ~special
    qmax = qmax_for_bits(bits)
    gid, n_gc, n_groups = group_ids(rows, cols, group_size)
    cpad = n_gc * group_size
    scales = np.zeros((rows, n_gc), dtype=np.float32)
    zps = np.zeros((rows, n_gc), dtype=np.uint8)
    codes_pad = np.zeros((rows, cpad), dtype=np.uint16)
    n_override = 0
    qrange = 2 * qmax
    prev_ref: np.ndarray | None = None

    for r0 in range(0, rows, _CHUNK_ROWS):
        r1 = min(rows, r0 + _CHUNK_ROWS)
        sc, zp_c, stored, mse = _chunk_candidates(
            w_f32[r0:r1], valid[r0:r1], bits, group_size
        )
        n_cand = int(stored.shape[0])
        rr = r1 - r0
        stored_rows = stored.reshape(n_cand, rr, cpad)
        spatial = _spatial_chunk(stored_rows, valid[r0:r1], group_size, prev_ref)
        pick, n_ov = pick_scales(mse, spatial, tie_ratio=mse_tie_ratio)
        n_override += n_ov
        # Advanced indexing: pick[i, j] selects candidate for row i, group j.
        ar = np.arange(n_gc, dtype=np.int32)[None, :]
        row_i = np.arange(rr, dtype=np.int32)[:, None]
        scales[r0:r1] = sc[pick, row_i, ar]
        zps[r0:r1] = zp_c[pick, row_i, ar].astype(np.uint8)
        chosen = stored[pick, row_i, ar, :]  # (rr, n_gc, G)
        codes_pad[r0:r1] = chosen.reshape(rr, cpad)
        prev_ref = codes_pad[r1 - 1]

    codes = codes_pad[:, :cols]
    scales_f16 = scales.astype(np.float16)
    zp_u8 = zps.astype(np.uint8)
    scale_of = scales_f16.astype(np.float32).reshape(-1)[gid]
    zp_of = zp_u8.astype(np.float32).reshape(-1)[gid]
    deq = (codes.astype(np.float32) - zp_of) * scale_of
    deq = np.where(special, w_f32, deq)
    q_ref = f32_to_bf16_u16(deq)
    if np.any(special):
        q_ref = q_ref.copy()
        q_ref[special] = arr2[special]

    err = np.abs(w_f32 - bf16_u16_to_f32(q_ref))
    outlier_idx, outlier_words = _select_outliers(
        err, special, arr2, scale_of, outlier_frac
    )
    if outlier_idx.size:
        q_ref = q_ref.copy()
        flat = q_ref.ravel()
        flat[outlier_idx] = outlier_words
        q_ref = flat.reshape(rows, cols)

    mse = (
        float(np.mean((w_f32[valid] - bf16_u16_to_f32(q_ref)[valid]) ** 2))
        if np.any(valid)
        else 0.0
    )
    if cols > 1:
        left_dis = float(np.mean(codes[:, 1:] != codes[:, :-1]))
    else:
        left_dis = 0.0
    if rows > 1:
        up_dis = float(np.mean(codes[1:, :] != codes[:-1, :]))
    else:
        up_dis = 0.0
    n_clipped = int(np.count_nonzero((codes == 0) | (codes == qrange)))
    return {
        "name": name,
        "shape": orig,
        "rows": rows,
        "cols": cols,
        "bits": bits,
        "keep": 0,
        "group_size": group_size,
        "n_groups": n_groups,
        "scales": scales_f16.ravel(),
        "zp": zp_u8.ravel(),
        "codes": codes.astype(np.uint16),
        "q_ref": q_ref.reshape(orig).astype(np.uint16),
        "outlier_idx": outlier_idx,
        "outlier_words": outlier_words,
        "mse": mse,
        "spatial_score": left_dis + up_dis,
        "n_spatial_overrides": n_override,
        "n_clipped": n_clipped,
        "family": family,
        "kind": "groupwise",
    }


def dequantize_int(
    *,
    bits: int,
    codes: np.ndarray,
    scales: np.ndarray,
    rows: int,
    cols: int,
    group_size: int,
    shape: tuple[int, ...],
    zp: np.ndarray | None = None,
    outlier_idx: np.ndarray | None = None,
    outlier_words: np.ndarray | None = None,
) -> np.ndarray:
    stored = np.ascontiguousarray(codes, dtype=np.uint16).reshape(rows, cols)
    gid, _n_gc, n_groups = group_ids(rows, cols, group_size)
    sc = np.ascontiguousarray(scales, dtype=np.float16).astype(np.float32).ravel()
    if int(sc.size) != n_groups:
        raise ValueError(f"scale count {sc.size} != n_groups {n_groups}")
    if zp is None:
        z = np.full(n_groups, qmax_for_bits(bits), dtype=np.float32)
    else:
        z = np.ascontiguousarray(zp, dtype=np.uint8).astype(np.float32).ravel()
        if int(z.size) != n_groups:
            raise ValueError(f"zp count {z.size} != n_groups {n_groups}")
    deq = (stored.astype(np.float32) - z[gid]) * sc[gid]
    out = f32_to_bf16_u16(deq).reshape(shape)
    if outlier_idx is not None and outlier_words is not None and int(np.asarray(outlier_idx).size):
        flat = out.ravel()
        idx = np.asarray(outlier_idx, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= flat.size):
            raise ValueError("outlier index out of range")
        flat[idx] = np.asarray(outlier_words, dtype=np.uint16)
        out = flat.reshape(shape)
    return out.astype(np.uint16)
