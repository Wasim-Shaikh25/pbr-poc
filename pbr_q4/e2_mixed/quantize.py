"""Mixed groupwise INT quantizer with structured per-row bits + E2b scale search.

Each output channel (row) has a bit width in {4,5,6,8,16}. Groups of
``GROUP_SIZE`` run along the input axis. Scale search is
accuracy-constrained: among candidates with E ≤ (1+ε) E_min, pick the
lowest estimated code cost (LEFT residual + uniqueness). Quality dominates
via a small default ε (0.0025).

Sparse per-weight exceptions are *not* used as a quality lever — only
non-finite BF16 payloads are stored as overlays. Decode of this quantized
reference is bit-exact; it is not BF16-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pbr_q4.e2_mixed.bits import (
    as_rank2,
    bf16_u16_to_f32,
    f32_to_bf16_u16,
    qmax_for_bits,
)
from pbr_q4.e2_mixed.const import E2B_EPS_DEFAULT, E2B_EPSILONS, GROUP_SIZE

_SCALE_FACTORS = np.array([0.88, 0.94, 1.00, 1.06, 1.14, 1.25], dtype=np.float32)
_MIN_SCALE = np.float32(1e-12)
_CHUNK_ROWS = 1024


@dataclass
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    rows: int
    cols: int
    row_bits: np.ndarray  # uint8[rows], 4/5/6/8/16
    group_size: int
    n_groups: int  # quantized groups only (excludes BF16 rows)
    scales: np.ndarray  # float16, n_groups
    zp: np.ndarray  # uint8, n_groups
    codes: np.ndarray  # uint16 (rows, cols); unused on BF16 rows
    q_ref: np.ndarray  # uint16 quantized reference, original shape
    special_idx: np.ndarray
    special_words: np.ndarray
    mse: float
    n_spatial_picks: int
    family: str
    projection: str
    eps: float

    def stats(self) -> dict[str, Any]:
        rb, counts = np.unique(self.row_bits, return_counts=True)
        return {
            "name": self.name,
            "shape": list(self.shape),
            "family": self.family,
            "projection": self.projection,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "n_weights": int(self.q_ref.size),
            "n_specials": int(self.special_idx.size),
            "mse": self.mse,
            "eps": self.eps,
            "row_bit_hist": {str(int(b)): int(c) for b, c in zip(rb.tolist(), counts.tolist())},
            "n_spatial_picks": self.n_spatial_picks,
        }


@dataclass
class QuantizeReport:
    tensors: list[QuantizedTensor] = field(default_factory=list)

    def by_name(self) -> dict[str, QuantizedTensor]:
        return {t.name: t for t in self.tensors}


def _code_cost(stored: np.ndarray, valid_pad: np.ndarray, group_size: int) -> np.ndarray:
    """LEFT-residual + uniqueness proxy per group, per candidate.

    *stored* is (n_cand, Cpad) uint16. *valid_pad* is (Cpad,) float mask.
    """
    n_cand, cpad = stored.shape
    g = int(group_size)
    n_gc = cpad // g
    left = np.zeros((n_cand, cpad), dtype=np.float32)
    if cpad > 1:
        d = np.abs(stored[:, 1:].astype(np.int16) - stored[:, :-1].astype(np.int16)).astype(np.float32)
        left[:, 1:] = d
    left *= valid_pad[None, :].astype(np.float32)
    left_g = left.reshape(n_cand, n_gc, g).mean(axis=-1)
    # Cheap uniqueness proxy: per-group std of stored codes (vectorized).
    blocks = stored.reshape(n_cand, n_gc, g).astype(np.float32)
    uniq = blocks.std(axis=-1) / np.float32(max(qmax_for_bits(2), 1))
    return left_g + np.float32(0.25) * uniq


def _pick_e2b(mse: np.ndarray, cost: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Among E ≤ (1+ε) E_min, minimize code cost. Quality dominates."""
    best = np.min(mse, axis=0)
    near = mse <= (np.float32(1.0 + float(eps)) * best)[None, :]
    score = np.where(near, cost, np.float32(1e9))
    rank = score * np.float32(1.0e6) + mse
    pick = np.argmin(rank, axis=0).astype(np.int32)
    mse_best = np.argmin(mse, axis=0).astype(np.int32)
    n_override = int(np.count_nonzero(pick != mse_best))
    return pick, n_override


def _chunk_candidates(
    rows_f32: np.ndarray,
    valid: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scales/zp/stored/mse. shapes: scales/zp/mse (n_cand, R, n_gc), stored (n_cand, R, n_gc, G)."""
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
        p01 = np.nanpercentile(masked, 1.0, axis=-1).astype(np.float32)
        p99 = np.nanpercentile(masked, 99.0, axis=-1).astype(np.float32)
    wmin = np.where(np.isfinite(wmin), wmin, 0.0)
    wmax = np.where(np.isfinite(wmax), wmax, 0.0)
    p01 = np.where(np.isfinite(p01), p01, wmin)
    p99 = np.where(np.isfinite(p99), p99, wmax)
    absmax = np.maximum(np.maximum(np.abs(wmin), np.abs(wmax)), _MIN_SCALE)

    def _affine(lo: np.ndarray, hi: np.ndarray, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        span = np.maximum(hi - lo, _MIN_SCALE)
        scales = (span / qrange)[None, ...] * factors[:, None, None]
        scales = np.maximum(scales, _MIN_SCALE)
        with np.errstate(invalid="ignore"):
            zp = np.rint((-lo[None, ...] / scales))
        zp = np.where(np.isfinite(zp), zp, 0)
        zp = np.clip(zp, 0, int(qrange)).astype(np.int16)
        return scales, zp

    sym_scale = (absmax / np.float32(qmax))[None, ...] * _SCALE_FACTORS[:, None, None]
    sym_scale = np.maximum(sym_scale, _MIN_SCALE)
    sym_zp = np.full(sym_scale.shape, qmax, dtype=np.int16)
    mm_scale, mm_zp = _affine(wmin, wmax, _SCALE_FACTORS)
    pct_scale, pct_zp = _affine(p01, p99, np.array([1.00, 1.10], dtype=np.float32))
    scales = np.concatenate([sym_scale, mm_scale, pct_scale], axis=0)
    zp = np.concatenate([sym_zp, mm_zp, pct_zp], axis=0)
    q = np.clip(
        np.rint(groups[None] / scales[..., None] + zp.astype(np.float32)[..., None]),
        0,
        int(qrange),
    )
    q = np.where(np.isfinite(q), q, 0).astype(np.int16)
    recon = (q.astype(np.float32) - zp.astype(np.float32)[..., None]) * scales[..., None]
    diff2 = np.where(v_g[None], (groups[None] - recon) ** 2, 0.0)
    denom = np.maximum(v_g.sum(axis=-1).astype(np.float32), 1.0)
    mse = diff2.sum(axis=-1) / denom[None, :, :]
    return scales, zp, q.astype(np.uint16), mse


def _quantize_rows(
    w_f32: np.ndarray,
    valid: np.ndarray,
    bits: int,
    group_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Quantize a block of rows at uniform *bits*. Returns scales, zp, codes, n_overrides."""
    rows, cols = int(w_f32.shape[0]), int(w_f32.shape[1])
    g = int(group_size)
    n_gc = (cols + g - 1) // g if cols else 0
    cpad = n_gc * g
    scales = np.zeros((rows, n_gc), dtype=np.float32)
    zps = np.zeros((rows, n_gc), dtype=np.uint8)
    codes_pad = np.zeros((rows, cpad), dtype=np.uint16)
    n_override = 0
    qrange = 2 * qmax_for_bits(bits)
    mask = np.zeros(cpad, dtype=np.float32)
    mask[:cols] = 1.0

    for r0 in range(0, rows, _CHUNK_ROWS):
        r1 = min(rows, r0 + _CHUNK_ROWS)
        sc, zp_c, stored, mse = _chunk_candidates(w_f32[r0:r1], valid[r0:r1], bits, g)
        n_cand = int(stored.shape[0])
        rr = r1 - r0
        stored_rows = stored.reshape(n_cand, rr, cpad)
        ar = np.arange(n_gc, dtype=np.int32)
        for i, r in enumerate(range(r0, r1)):
            vpad = mask.copy()
            vpad[:cols] = valid[r].astype(np.float32)
            cost = _code_cost(stored_rows[:, i, :], vpad, g)
            pick, n_ov = _pick_e2b(mse[:, i, :], cost, eps)
            n_override += n_ov
            scales[r] = sc[pick, i, ar]
            zps[r] = np.clip(zp_c[pick, i, ar], 0, qrange).astype(np.uint8)
            chosen = stored[pick, i, ar, :]
            codes_pad[r] = chosen.reshape(cpad)
    return scales, zps, codes_pad[:, :cols], n_override


def quantize_mixed(
    words: np.ndarray,
    row_bits: np.ndarray,
    *,
    group_size: int = GROUP_SIZE,
    eps: float = E2B_EPS_DEFAULT,
    name: str = "",
    family: str = "",
    projection: str = "",
) -> QuantizedTensor:
    """Quantize a BF16-uint16 tensor with per-row mixed bit widths."""
    arr2, orig, rows, cols = as_rank2(np.ascontiguousarray(words, dtype=np.uint16))
    rb = np.ascontiguousarray(row_bits, dtype=np.uint8).reshape(rows)
    legal = {4, 5, 6, 8, 16}
    if set(int(x) for x in np.unique(rb).tolist()) - legal:
        raise ValueError(f"illegal row bits in {name}: {np.unique(rb).tolist()}")
    w_f32 = bf16_u16_to_f32(arr2)
    special = ~np.isfinite(w_f32)
    valid = ~special
    codes = np.zeros((rows, cols), dtype=np.uint16)
    q_ref = arr2.copy()
    n_gc = (cols + int(group_size) - 1) // int(group_size) if cols else 0
    scales_by_row = np.zeros((rows, max(n_gc, 1)), dtype=np.float16)
    zp_by_row = np.zeros((rows, max(n_gc, 1)), dtype=np.uint8)
    quantized_row = np.zeros(rows, dtype=bool)
    n_override = 0
    gid = np.arange(cols, dtype=np.int32) // int(group_size) if cols else np.zeros(0, dtype=np.int32)

    for b in (4, 5, 6, 8):
        idx = np.flatnonzero(rb == np.uint8(b))
        if idx.size == 0:
            continue
        sc, zp, cd, n_ov = _quantize_rows(w_f32[idx], valid[idx], b, group_size, eps)
        n_override += n_ov
        quantized_row[idx] = True
        scales_by_row[idx, :n_gc] = sc.astype(np.float16)
        zp_by_row[idx, :n_gc] = zp.astype(np.uint8)
        codes[idx] = cd
        sc_f = sc.astype(np.float16).astype(np.float32)
        zp_f = zp.astype(np.float32)
        deq = (cd.astype(np.float32) - zp_f[:, gid]) * sc_f[:, gid]
        deq = np.where(special[idx], w_f32[idx], deq)
        q_rows = f32_to_bf16_u16(deq)
        if np.any(special[idx]):
            q_rows = q_rows.copy()
            q_rows[special[idx]] = arr2[idx][special[idx]]
        q_ref[idx] = q_rows

    n_groups = int(np.count_nonzero(quantized_row) * n_gc)
    if n_groups:
        scales = scales_by_row[quantized_row, :n_gc].ravel().astype(np.float16)
        zps = zp_by_row[quantized_row, :n_gc].ravel().astype(np.uint8)
    else:
        scales = np.zeros(0, dtype=np.float16)
        zps = np.zeros(0, dtype=np.uint8)

    special_idx = np.flatnonzero(special.ravel()).astype(np.int64)
    special_words = arr2.ravel()[special_idx].astype(np.uint16) if special_idx.size else np.zeros(0, dtype=np.uint16)
    if special_idx.size:
        flat = q_ref.ravel()
        flat[special_idx] = special_words
        q_ref = flat.reshape(rows, cols)

    if np.any(valid):
        mse = float(np.mean((w_f32[valid] - bf16_u16_to_f32(q_ref)[valid]) ** 2))
    else:
        mse = 0.0

    return QuantizedTensor(
        name=name,
        shape=orig,
        rows=rows,
        cols=cols,
        row_bits=rb.copy(),
        group_size=int(group_size),
        n_groups=int(n_groups),
        scales=scales,
        zp=zps,
        codes=codes.astype(np.uint16),
        q_ref=q_ref.reshape(orig).astype(np.uint16),
        special_idx=special_idx,
        special_words=special_words,
        mse=mse,
        n_spatial_picks=n_override,
        family=family,
        projection=projection,
        eps=float(eps),
    )


def dequantize_mixed(
    *,
    row_bits: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    zp: np.ndarray,
    rows: int,
    cols: int,
    group_size: int,
    shape: tuple[int, ...],
    special_idx: np.ndarray | None = None,
    special_words: np.ndarray | None = None,
    bf16_rows: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Rebuild the quantized-reference BF16 words from stored fields."""
    rb = np.ascontiguousarray(row_bits, dtype=np.uint8).reshape(rows)
    stored = np.ascontiguousarray(codes, dtype=np.uint16).reshape(rows, cols)
    sc_all = np.ascontiguousarray(scales, dtype=np.float16).astype(np.float32).ravel()
    zp_all = np.ascontiguousarray(zp, dtype=np.uint8).astype(np.float32).ravel()
    out = np.zeros((rows, cols), dtype=np.uint16)
    off = 0
    n_gc = (cols + group_size - 1) // group_size if cols else 0
    gid = np.arange(cols, dtype=np.int32) // int(group_size) if cols else np.zeros(0, dtype=np.int32)
    for r in range(rows):
        b = int(rb[r])
        if b >= 16:
            if bf16_rows is None or r not in bf16_rows:
                raise ValueError(f"missing BF16 payload for row {r}")
            row = np.ascontiguousarray(bf16_rows[r], dtype=np.uint16).reshape(cols)
            out[r] = row
            continue
        need = n_gc
        if off + need > sc_all.size:
            raise ValueError("scale underrun")
        sc = sc_all[off : off + need]
        z = zp_all[off : off + need]
        off += need
        deq = (stored[r].astype(np.float32) - z[gid]) * sc[gid]
        out[r] = f32_to_bf16_u16(deq)
    if off != sc_all.size:
        raise ValueError(f"scale leftover {sc_all.size - off}")
    if special_idx is not None and special_words is not None and int(np.asarray(special_idx).size):
        flat = out.ravel()
        idx = np.asarray(special_idx, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= flat.size):
            raise ValueError("special index out of range")
        flat[idx] = np.asarray(special_words, dtype=np.uint16)
        out = flat.reshape(rows, cols)
    return out.reshape(shape).astype(np.uint16)


def estimate_packed_bits(qt: QuantizedTensor) -> dict[str, int]:
    """Split packed-field bits: payload / scales / precision map / specials."""
    n = int(qt.q_ref.size)
    payload = 0
    scale_bits = int(qt.n_groups) * 24
    map_bits = int(qt.rows) * 8
    special_bits = int(qt.special_idx.size) * (32 + 16)
    for b in (4, 5, 6, 8, 16):
        rows = np.flatnonzero(qt.row_bits == b)
        if rows.size == 0:
            continue
        nw = int(rows.size) * int(qt.cols)
        payload += nw * (16 if b >= 16 else b)
    return {
        "weight_payload_bits": payload,
        "scales_bits": scale_bits,
        "precision_map_bits": map_bits,
        "specials_bits": special_bits,
        "total_bits": payload + scale_bits + map_bits + special_bits,
        "n_weights": n,
    }


def e2b_epsilon_diagnostic(mse: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    """For tests / artifacts: which ε changes the pick, and mean cost."""
    out = {}
    for eps in E2B_EPSILONS:
        pick, n_ov = _pick_e2b(mse, cost, eps)
        taken_cost = cost[pick, np.arange(cost.shape[1])]
        taken_mse = mse[pick, np.arange(mse.shape[1])]
        out[str(eps)] = {
            "n_overrides_vs_mse_min": n_ov,
            "mean_code_cost": float(np.mean(taken_cost)),
            "mean_mse": float(np.mean(taken_mse)),
        }
    return out
