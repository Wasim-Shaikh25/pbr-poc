"""Groupwise mixed-precision quantizer with compression-aware scale search.

For each group, several scales are scored by reconstruction MSE. Among
near-ties (MSE ≤ ``mse_tie_ratio`` × best), the encoder prefers the scale
that lowers LEFT/UP neighbor-code disagreement. Quality dominates: a
materially worse reconstruction never wins on spatial score alone.

Sparse high-error positions are stored as original BF16 words (charged
in the container). Decode of *this* quantized reference is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pbr_q4.codesign.bits import (
    as_rank2,
    bf16_u16_to_f32,
    f32_to_bf16_u16,
    group_ids,
    qmax_for_bits,
    signed_to_stored,
    stored_to_signed,
)
from pbr_q4.codesign.policy import CodesignPolicy

# Scale multipliers around absmax/qmax. 1.00 is the classic symmetric pick.
_SCALE_FACTORS = np.array([0.88, 0.94, 1.00, 1.06, 1.14, 1.25], dtype=np.float32)
_P99 = 0.99
_CHUNK_ROWS = 2048
_MIN_SCALE = np.float32(1e-12)


@dataclass
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    rows: int
    cols: int
    bits: int | None
    group_size: int
    n_groups: int
    scales: np.ndarray  # float16, n_groups (empty if BF16)
    codes: np.ndarray  # uint16 (rows, cols) stored codes (empty if BF16)
    q_ref: np.ndarray  # uint16 BF16 quantized reference, original shape
    outlier_idx: np.ndarray
    outlier_words: np.ndarray
    mse: float
    spatial_score: float
    n_spatial_overrides: int
    n_clipped: int
    family: str
    kind: str  # "bf16_raw" | "groupwise"

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "bits": self.bits,
            "kind": self.kind,
            "family": self.family,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "n_weights": int(self.q_ref.size),
            "n_outliers": int(self.outlier_idx.size),
            "mse": self.mse,
            "spatial_score": self.spatial_score,
            "n_spatial_overrides": self.n_spatial_overrides,
            "n_clipped": self.n_clipped,
        }


@dataclass
class QuantizeReport:
    tensors: list[QuantizedTensor] = field(default_factory=list)
    n_spatial_overrides: int = 0
    n_outliers: int = 0

    def by_name(self) -> dict[str, QuantizedTensor]:
        return {t.name: t for t in self.tensors}


def _special_mask_f32(x: np.ndarray) -> np.ndarray:
    return ~np.isfinite(x)


def _chunk_candidates(
    rows_f32: np.ndarray,
    valid: np.ndarray,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scales/q/mse for a row chunk.

    shapes: scales (n_cand, R, n_gc), q (n_cand, R, n_gc, G), mse (n_cand, R, n_gc).
    """
    r, c = int(rows_f32.shape[0]), int(rows_f32.shape[1])
    g = int(group_size)
    n_gc = (c + g - 1) // g
    cpad = n_gc * g
    qmax = qmax_for_bits(bits)
    buf = np.zeros((r, cpad), dtype=np.float32)
    vbuf = np.zeros((r, cpad), dtype=bool)
    buf[:, :c] = rows_f32
    vbuf[:, :c] = valid
    groups = buf.reshape(r, n_gc, g)
    v_g = vbuf.reshape(r, n_gc, g)
    abs_g = np.where(v_g, np.abs(groups), 0.0)
    absmax = np.max(abs_g, axis=-1).astype(np.float32)
    with np.errstate(invalid="ignore"):
        p99 = np.quantile(np.where(v_g, abs_g, np.nan), _P99, axis=-1)
    p99 = np.where(np.isfinite(p99), p99, absmax).astype(np.float32)
    absmax = np.maximum(absmax, _MIN_SCALE)
    p99 = np.maximum(p99, _MIN_SCALE)
    scales_abs = (absmax / np.float32(qmax))[None, :, :] * _SCALE_FACTORS[:, None, None]
    scales_p99 = (p99 / np.float32(qmax))[None, :, :]
    scales = np.concatenate([scales_abs, scales_p99], axis=0)
    scales = np.maximum(scales, _MIN_SCALE)
    q = np.clip(np.rint(groups[None] / scales[..., None]), -qmax, qmax).astype(np.int16)
    recon = q.astype(np.float32) * scales[..., None]
    diff2 = np.where(v_g[None], (groups[None] - recon) ** 2, 0.0)
    denom = np.maximum(v_g.sum(axis=-1).astype(np.float32), 1.0)
    mse = diff2.sum(axis=-1) / denom[None, :, :]
    return scales, q, mse


def _spatial_scores(
    stored: np.ndarray,
    valid: np.ndarray,
    prev_stored: np.ndarray | None,
    group_size: int,
) -> np.ndarray:
    """LEFT + UP disagreement (+ mean |LEFT residual|) per group, per candidate.

    *stored* is (n_cand, Cpad) uint16. *valid* is (C,) for the real columns.
    """
    n_cand, cpad = stored.shape
    g = int(group_size)
    n_gc = cpad // g
    c = int(valid.size)
    left = np.zeros((n_cand, cpad), dtype=np.float32)
    if cpad > 1:
        d = np.abs(stored[:, 1:].astype(np.int16) - stored[:, :-1].astype(np.int16))
        left[:, 1:] = d.astype(np.float32)
        disagree = (stored[:, 1:] != stored[:, :-1]).astype(np.float32)
        left[:, 1:] = left[:, 1:] * np.float32(0.25) + disagree
    up = np.zeros((n_cand, cpad), dtype=np.float32)
    if prev_stored is not None:
        prev = np.asarray(prev_stored, dtype=np.uint16).reshape(1, -1)
        if prev.shape[1] != cpad:
            raise ValueError("prev row width mismatch")
        ud = np.abs(stored.astype(np.int16) - prev.astype(np.int16))
        up = ud.astype(np.float32) * np.float32(0.25) + (stored != prev).astype(np.float32)
    # Mask padded / invalid columns.
    mask = np.zeros(cpad, dtype=np.float32)
    mask[:c] = valid.astype(np.float32)
    left *= mask[None, :]
    up *= mask[None, :]
    left_g = left.reshape(n_cand, n_gc, g).sum(axis=-1)
    up_g = up.reshape(n_cand, n_gc, g).sum(axis=-1)
    denom = np.maximum(mask.reshape(n_gc, g).sum(axis=-1), 1.0)
    return (left_g + up_g) / denom[None, :]


def _pick_scales(
    mse: np.ndarray,
    spatial: np.ndarray,
    *,
    tie_ratio: float,
) -> tuple[np.ndarray, int]:
    """Among near-tie MSE candidates, minimize spatial score.

    Returns (pick[n_gc], n_overrides) where an override is a pick that is
    not the unique MSE-best candidate.
    """
    best = np.min(mse, axis=0)
    near = mse <= (np.float32(tie_ratio) * best)[None, :]
    # Huge spatial for candidates outside the near-tie band.
    score = np.where(near, spatial, np.float32(1e9))
    # Secondary: lower MSE, then lower candidate index (1.00 factor is mid-list).
    rank = score * np.float32(1000.0) + mse
    pick = np.argmin(rank, axis=0).astype(np.int32)
    mse_best = np.argmin(mse, axis=0).astype(np.int32)
    n_override = int(np.count_nonzero(pick != mse_best))
    return pick, n_override


def quantize_matrix(
    words: np.ndarray,
    bits: int,
    *,
    group_size: int = 128,
    mse_tie_ratio: float = 1.02,
    outlier_frac: float = 0.001,
    name: str = "",
    family: str = "",
) -> QuantizedTensor:
    """Quantize a BF16-uint16 tensor to groupwise signed INT + BF16 q-ref."""
    arr2, orig, rows, cols = as_rank2(np.ascontiguousarray(words, dtype=np.uint16))
    w_f32 = bf16_u16_to_f32(arr2)
    special = _special_mask_f32(w_f32)
    valid = ~special
    qmax = qmax_for_bits(bits)
    gid, n_gc, n_groups = group_ids(rows, cols, group_size)
    cpad = n_gc * group_size
    scales = np.zeros((rows, n_gc), dtype=np.float32)
    codes_pad = np.zeros((rows, cpad), dtype=np.uint16)
    n_override = 0
    n_clipped = 0
    prev_stored: np.ndarray | None = None

    for r0 in range(0, rows, _CHUNK_ROWS):
        r1 = min(rows, r0 + _CHUNK_ROWS)
        sc, q, mse = _chunk_candidates(w_f32[r0:r1], valid[r0:r1], bits, group_size)
        stored = signed_to_stored(q, bits)  # (n_cand, R, n_gc, G)
        n_cand = int(stored.shape[0])
        rr = r1 - r0
        stored_rows = stored.reshape(n_cand, rr, cpad)
        ar = np.arange(n_gc, dtype=np.int32)
        for i, r in enumerate(range(r0, r1)):
            spatial = _spatial_scores(stored_rows[:, i, :], valid[r], prev_stored, group_size)
            pick, n_ov = _pick_scales(mse[:, i, :], spatial, tie_ratio=mse_tie_ratio)
            n_override += n_ov
            scales[r] = sc[pick, i, ar]
            chosen_q = q[pick, i, ar, :]
            n_clipped += int(np.count_nonzero(np.abs(chosen_q) >= qmax))
            codes_pad[r] = signed_to_stored(chosen_q, bits).reshape(cpad)
            prev_stored = codes_pad[r]

    codes = codes_pad[:, :cols]
    q_signed = stored_to_signed(codes, bits)
    # Q-ref must use the same float16 scales the container stores.
    scales_f16 = scales.astype(np.float16)
    scale_of = scales_f16.astype(np.float32).reshape(-1)[gid]
    deq = q_signed.astype(np.float32) * scale_of
    # Specials (NaN/Inf) always become outliers; they are not dequantized.
    deq = np.where(special, w_f32, deq)
    q_ref = f32_to_bf16_u16(deq)
    # Force specials back to the original payload.
    if np.any(special):
        q_ref = q_ref.copy()
        q_ref[special] = arr2[special]

    err = np.abs(w_f32 - bf16_u16_to_f32(q_ref))
    outlier_idx, outlier_words = _select_outliers(
        err, special, arr2, scale_of, bits, outlier_frac
    )
    if outlier_idx.size:
        q_ref = q_ref.copy()
        flat = q_ref.ravel()
        flat[outlier_idx] = outlier_words
        q_ref = flat.reshape(rows, cols)

    mse = float(np.mean((w_f32[valid] - bf16_u16_to_f32(q_ref)[valid]) ** 2)) if np.any(valid) else 0.0
    if cols > 1:
        left_dis = float(np.mean(codes[:, 1:] != codes[:, :-1]))
    else:
        left_dis = 0.0
    if rows > 1:
        up_dis = float(np.mean(codes[1:, :] != codes[:-1, :]))
    else:
        up_dis = 0.0
    return QuantizedTensor(
        name=name,
        shape=orig,
        rows=rows,
        cols=cols,
        bits=bits,
        group_size=group_size,
        n_groups=n_groups,
        scales=scales_f16.ravel(),
        codes=codes.astype(np.uint16),
        q_ref=q_ref.reshape(orig).astype(np.uint16),
        outlier_idx=outlier_idx,
        outlier_words=outlier_words,
        mse=mse,
        spatial_score=left_dis + up_dis,
        n_spatial_overrides=n_override,
        n_clipped=n_clipped,
        family=family,
        kind="groupwise",
    )


def _select_outliers(
    err: np.ndarray,
    special: np.ndarray,
    original: np.ndarray,
    scale_of: np.ndarray,
    bits: int,
    outlier_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    special_idx = np.flatnonzero(special.ravel())
    if outlier_frac <= 0:
        words = original.ravel()[special_idx].astype(np.uint16) if special_idx.size else np.zeros(0, dtype=np.uint16)
        return special_idx.astype(np.int64), words

    # High residual vs group scale (more than ~0.6 ulp of the integer grid).
    thr = np.float32(0.60) * np.maximum(scale_of, _MIN_SCALE)
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


def quantize_bf16_raw(
    words: np.ndarray,
    *,
    name: str = "",
    family: str = "norm_bias",
) -> QuantizedTensor:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    orig = tuple(int(x) for x in w.shape)
    rows, cols = (1, int(w.size)) if w.ndim <= 1 else (int(w.shape[0]), int(np.prod(w.shape[1:])))
    return QuantizedTensor(
        name=name,
        shape=orig,
        rows=rows,
        cols=cols,
        bits=None,
        group_size=0,
        n_groups=0,
        scales=np.zeros(0, dtype=np.float16),
        codes=np.zeros((0, 0), dtype=np.uint16),
        q_ref=w.copy(),
        outlier_idx=np.zeros(0, dtype=np.int64),
        outlier_words=np.zeros(0, dtype=np.uint16),
        mse=0.0,
        spatial_score=0.0,
        n_spatial_overrides=0,
        n_clipped=0,
        family=family,
        kind="bf16_raw",
    )


def dequantize_to_bf16(
    *,
    bits: int | None,
    codes: np.ndarray,
    scales: np.ndarray,
    rows: int,
    cols: int,
    group_size: int,
    shape: tuple[int, ...],
    outlier_idx: np.ndarray | None = None,
    outlier_words: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the quantized-reference BF16 words from stored fields."""
    if bits is None:
        raise ValueError("use the raw payload path for bf16_raw tensors")

    stored = np.ascontiguousarray(codes, dtype=np.uint16).reshape(rows, cols)
    q = stored_to_signed(stored, bits)
    gid, _n_gc, n_groups = group_ids(rows, cols, group_size)
    sc = np.ascontiguousarray(scales, dtype=np.float16).astype(np.float32).ravel()
    if int(sc.size) != n_groups:
        raise ValueError(f"scale count {sc.size} != n_groups {n_groups}")
    deq = q.astype(np.float32) * sc[gid]
    out = f32_to_bf16_u16(deq).reshape(shape)
    if outlier_idx is not None and outlier_words is not None and int(np.asarray(outlier_idx).size):
        flat = out.ravel()
        idx = np.asarray(outlier_idx, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= flat.size):
            raise ValueError("outlier index out of range")
        flat[idx] = np.asarray(outlier_words, dtype=np.uint16)
        out = flat.reshape(shape)
    return out.astype(np.uint16)


def quantize_tensor(
    words: np.ndarray,
    bits: int | None,
    policy: CodesignPolicy,
    *,
    name: str = "",
    family: str = "",
) -> QuantizedTensor:
    if bits is None:
        return quantize_bf16_raw(words, name=name, family=family or "norm_bias")
    outlier_frac = float(policy.outlier_frac) if bits <= int(policy.outlier_max_bits) else 0.0
    return quantize_matrix(
        words,
        int(bits),
        group_size=int(policy.group_size),
        mse_tie_ratio=float(policy.mse_tie_ratio),
        outlier_frac=outlier_frac,
        name=name,
        family=family,
    )


def estimate_packed_bits(qt: QuantizedTensor) -> int:
    """Complete packed-field bit count (no X/Y, no container header)."""
    n = int(qt.q_ref.size)
    if qt.kind == "bf16_raw" or qt.bits is None:
        return n * 16
    code_bits = n * int(qt.bits)
    scale_bits = int(qt.n_groups) * 16
    # sparse outliers: u32 index + bf16 word
    out_bits = int(qt.outlier_idx.size) * (32 + 16)
    return code_bits + scale_bits + out_bits
