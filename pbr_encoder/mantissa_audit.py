"""Phase A: held-out mantissa predictability audit (lossless, bit-exact BF16).

Counts every codebook / table byte. Success is not ≤4 BPW. The first decisive
gate is whether any *compact* context or reversible transform codes the 7-bit
mantissa below 6.5 complete BPW on held-out words (≈ <10.1 total with ~1 sign
+ ~2.6 exponent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from pbr_core.bf16 import split_components
from pbr_core.safetensors_io import is_embedding_weight, layer_index, load_uint16, tensor_role
from pbr_core.tiles import as_2d
from pbr_encoder.hf_weights import (
    PRIMARY_LICENSE,
    PRIMARY_REPO,
    inventory_from_dir,
    select_weight_specs,
)
from pbr_qualifier.entropy import shannon_entropy

MANT_ALPH = 128
HOLD_FRAC = 0.2
MIN_CTX_COUNT = 256
GATE_MANT_BPW = 6.5
# dump_table: uint16 count + symbol bytes + length bytes
TABLE_OVERHEAD = 2

DISCLAIMER = (
    "Phase A mantissa audit. Held-out code length includes Huffman-table bytes. "
    "Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW. "
    "Gate: mantissa complete BPW < 6.5 after predictor/table overhead."
)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def gray_encode(m: np.ndarray) -> np.ndarray:
    x = np.asarray(m, dtype=np.uint8) & np.uint8(0x7F)
    return (x ^ (x >> np.uint8(1))) & np.uint8(0x7F)


def gray_decode(g: np.ndarray) -> np.ndarray:
    x = np.asarray(g, dtype=np.uint8) & np.uint8(0x7F)
    x ^= x >> np.uint8(1)
    x ^= x >> np.uint8(2)
    x ^= x >> np.uint8(4)
    return x & np.uint8(0x7F)


def haar_fwd(m2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Integer lifting Haar along rows: d = a-b, s = b + (d>>1)."""
    tile = np.ascontiguousarray(m2d, dtype=np.uint8)
    cols = tile.shape[1]
    even = cols - (cols % 2)
    a = tile[:, 0:even:2].astype(np.int16)
    b = tile[:, 1:even:2].astype(np.int16)
    d = a - b
    s = b + (d >> 1)
    leftover = tile[:, even:] if cols % 2 else None
    return s, d, leftover


def haar_inv(
    s: np.ndarray, d: np.ndarray, leftover: np.ndarray | None, cols: int
) -> np.ndarray:
    b = s - (d >> 1)
    a = d + b
    rows = int(s.shape[0])
    out = np.empty((rows, cols), dtype=np.uint8)
    even = cols - (cols % 2)
    out[:, 0:even:2] = a.astype(np.uint8)
    out[:, 1:even:2] = b.astype(np.uint8)
    if leftover is not None and leftover.size:
        out[:, even:] = leftover
    return out


def _split_hold(arr: np.ndarray, hold_frac: float = HOLD_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Last hold_frac of rows (or columns if too few rows) is held-out."""
    mat = np.ascontiguousarray(arr)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    rows, cols = mat.shape
    if rows >= 5:
        n_hold = max(1, int(round(rows * hold_frac)))
        n_hold = min(n_hold, rows - 1)
        return mat[: rows - n_hold], mat[rows - n_hold :]
    n_hold = max(1, int(round(cols * hold_frac)))
    n_hold = min(n_hold, cols - 1) if cols > 1 else 0
    if n_hold == 0:
        return mat, mat[:0]
    return mat[:, : cols - n_hold], mat[:, cols - n_hold :]


def _table_bytes(n_nonzero: int) -> int:
    return TABLE_OVERHEAD + 2 * max(int(n_nonzero), 0)


def _entropy_from_counts(counts: np.ndarray) -> float:
    tot = float(counts.sum())
    if tot <= 0:
        return 0.0
    p = counts.astype(np.float64)
    p = p[p > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _row_entropies(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-context entropy and occupancy."""
    totals = counts.sum(axis=1).astype(np.float64)
    out = np.zeros(counts.shape[0], dtype=np.float64)
    mask = totals > 0
    if not np.any(mask):
        return out, totals
    rows = counts[mask].astype(np.float64)
    # Avoid 0*log0.
    safe = np.where(rows > 0, rows / totals[mask, None], 1.0)
    h = -np.sum(np.where(rows > 0, (rows / totals[mask, None]) * np.log2(safe), 0.0), axis=1)
    out[mask] = h
    return out, totals


def _bincount2(ctx: np.ndarray, y: np.ndarray, n_ctx: int, n_y: int) -> np.ndarray:
    idx = ctx.astype(np.int64) * n_y + y.astype(np.int64)
    return np.bincount(idx, minlength=n_ctx * n_y).reshape(n_ctx, n_y)


def _admit_mask(train_counts: np.ndarray, min_count: int = MIN_CTX_COUNT) -> np.ndarray:
    """Keep a per-context table only if train occupancy pays for the table."""
    n_ctx, n_y = train_counts.shape
    occ = train_counts.sum(axis=1)
    h_row, _ = _row_entropies(train_counts)
    global_counts = train_counts.sum(axis=0)
    h_g = _entropy_from_counts(global_counts)
    nz = (train_counts > 0).sum(axis=1)
    table_b = np.array([_table_bytes(int(v)) for v in nz], dtype=np.float64)
    savings_bits = occ.astype(np.float64) * np.maximum(h_g - h_row, 0.0)
    admit = (occ >= min_count) & (savings_bits / 8.0 > table_b)
    # Never admit the empty contexts.
    admit &= occ > 0
    if n_ctx == 1:
        return np.ones(1, dtype=bool)
    return admit


def _nll_bits(
    counts: np.ndarray,
    ctx: np.ndarray,
    y: np.ndarray,
    *,
    admit: np.ndarray | None = None,
    smooth: float = 1.0,
) -> float:
    """Held-out negative log2 likelihood in bits.

    smooth=1 is Laplace; smooth=0 is MLE with a tiny floor for unseen bins.
    """
    if y.size == 0:
        return 0.0
    n_y = counts.shape[1]
    g = counts.sum(axis=0).astype(np.float64)
    g_sm = g + (smooth if smooth > 0 else 0.0)
    g_tot = g_sm.sum()
    if g_tot <= 0:
        return float(y.size * np.log2(max(n_y, 2)))
    yy = np.clip(np.ascontiguousarray(y, dtype=np.int64).ravel(), 0, n_y - 1)
    c = np.clip(np.ascontiguousarray(ctx, dtype=np.int64).ravel(), 0, counts.shape[0] - 1)
    if smooth > 0:
        p_g = (g[yy] + smooth) / (g.sum() + smooth * n_y)
        sm = counts.astype(np.float64) + smooth
        row = sm.sum(axis=1)
        p_c = sm[c, yy] / row[c]
    else:
        g_sum = max(float(g.sum()), 1.0)
        p_g = np.where(g[yy] > 0, g[yy] / g_sum, 1e-12)
        row = counts.sum(axis=1).astype(np.float64)
        raw = counts[c, yy].astype(np.float64)
        p_c = np.where((row[c] > 0) & (raw > 0), raw / np.maximum(row[c], 1.0), 1e-12)
    if admit is None:
        p = p_c
    else:
        use = admit[c]
        p = np.where(use, p_c, p_g)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.log2(p).sum())


def _code_report(
    name: str,
    *,
    nll_bits: float,
    n_hold: int,
    codebook_bytes: int,
    extra_bytes: int = 0,
    n_tables: int = 1,
    kind: str = "context",
    h_full: float | None = None,
) -> dict:
    overhead_bits = 8.0 * (codebook_bytes + extra_bytes)
    complete_bits = nll_bits + overhead_bits
    n = max(int(n_hold), 1)
    return {
        "name": name,
        "kind": kind,
        "n_hold": int(n_hold),
        "ideal_bpw": nll_bits / n if n_hold else 0.0,
        "complete_bpw": complete_bits / n if n_hold else 0.0,
        "nll_bits": nll_bits,
        "codebook_bytes": int(codebook_bytes),
        "extra_predictor_bytes": int(extra_bytes),
        "n_tables": int(n_tables),
        "h_full": h_full,
        "improves_vs_uncond": None,
    }


def _uncond_report(y_train: np.ndarray, y_hold: np.ndarray, *, name: str = "H(M)") -> dict:
    y_tr = np.ascontiguousarray(y_train, dtype=np.uint8).ravel()
    y_ho = np.ascontiguousarray(y_hold, dtype=np.uint8).ravel()
    counts = np.bincount(y_tr.astype(np.int64), minlength=MANT_ALPH)
    counts = counts.reshape(1, MANT_ALPH)
    nll = _nll_bits(counts, np.zeros(y_ho.size, dtype=np.int64), y_ho, smooth=0.0)
    nll_code = _nll_bits(counts, np.zeros(y_ho.size, dtype=np.int64), y_ho, smooth=1.0)
    nz = int((counts[0] > 0).sum())
    h_full = shannon_entropy(np.concatenate([y_tr, y_ho]) if y_ho.size else y_tr)
    rec = _code_report(
        name,
        nll_bits=nll_code,
        n_hold=int(y_ho.size),
        codebook_bytes=_table_bytes(nz),
        n_tables=1,
        kind="uncond",
        h_full=h_full,
    )
    rec["ideal_bpw"] = nll / max(int(y_ho.size), 1)
    rec["nll_bits_ideal"] = nll
    return rec


def _conditional_report(
    name: str,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    ctx_hold: np.ndarray,
    y_hold: np.ndarray,
    n_ctx: int,
) -> dict:
    counts = _bincount2(ctx_train.ravel(), y_train.ravel(), n_ctx, MANT_ALPH)
    admit = _admit_mask(counts)
    y_ho = np.ascontiguousarray(y_hold).ravel()
    nll_ideal = _nll_bits(counts, np.ascontiguousarray(ctx_hold).ravel(), y_ho, smooth=0.0)
    nll_code = _nll_bits(
        counts, np.ascontiguousarray(ctx_hold).ravel(), y_ho, admit=admit, smooth=1.0
    )
    g_nz = int((counts.sum(axis=0) > 0).sum())
    ctx_nz = (counts[admit] > 0).sum(axis=1) if np.any(admit) else np.zeros(0, dtype=int)
    codebook = _table_bytes(g_nz) + int(sum(_table_bytes(int(v)) for v in ctx_nz))
    n_tables = 1 + int(admit.sum())
    # Do not double-count the single dummy context.
    if n_ctx == 1:
        codebook = _table_bytes(g_nz)
        n_tables = 1
    rec = _code_report(
        name,
        nll_bits=nll_code,
        n_hold=int(y_ho.size),
        codebook_bytes=codebook,
        n_tables=n_tables,
        kind="context",
    )
    rec["ideal_bpw"] = nll_ideal / max(int(y_ho.size), 1)
    rec["nll_bits_ideal"] = nll_ideal
    return rec


def _prev_value(m: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(m, dtype=np.uint8).ravel()
    prev = np.zeros_like(flat)
    if flat.size:
        prev[1:] = flat[:-1]
    return prev.reshape(np.asarray(m).shape)


def _prev_row(m: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(m, dtype=np.uint8)
    if tile.ndim == 1:
        tile = tile.reshape(1, -1)
    pred = np.zeros_like(tile)
    if tile.shape[0] > 1:
        pred[1:, :] = tile[:-1, :]
    return pred


def _mod_delta(m: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (np.asarray(m, dtype=np.uint8) - np.asarray(pred, dtype=np.uint8)) & np.uint8(0x7F)


def _plane_packed_bytes(m: np.ndarray) -> list[np.ndarray]:
    """Bit-plane transpose: 7 packed-byte streams (plane-major)."""
    bits = np.ascontiguousarray(m, dtype=np.uint8).ravel()
    streams: list[np.ndarray] = []
    for b in range(7):
        plane = ((bits >> b) & 1).astype(np.uint8)
        streams.append(np.packbits(plane))
    return streams


def _split_1d(arr: np.ndarray, hold_frac: float = HOLD_FRAC) -> tuple[np.ndarray, np.ndarray]:
    n = int(arr.size)
    if n < 2:
        return arr, arr[:0]
    n_hold = max(1, int(round(n * hold_frac)))
    n_hold = min(n_hold, n - 1)
    return arr[: n - n_hold], arr[n - n_hold :]


def _stream_huffman_report(name: str, streams: list[np.ndarray], n_words: int) -> dict:
    """Independent Huffman of one or more byte streams covering n_words mantissas."""
    nll = 0.0
    codebook = 0
    n_tables = 0
    n_hold_units = 0
    for stream in streams:
        tr, ho = _split_1d(stream)
        if ho.size == 0:
            continue
        # 256-ary byte alphabet.
        counts = np.bincount(tr.astype(np.int64), minlength=256).reshape(1, 256)
        nll += _nll_bits(counts, np.zeros(ho.size, dtype=np.int64), ho, smooth=1.0)
        codebook += _table_bytes(int((counts[0] > 0).sum()))
        n_tables += 1
        n_hold_units += int(ho.size)
    # Scale held-out NLL from packed-bytes back to the matching word fraction.
    hold_words = max(1, int(round(n_words * HOLD_FRAC)))
    # complete bits over hold_words, using the byte-stream NLL as the payload.
    extra = 0
    complete_bits = nll + 8.0 * (codebook + extra)
    return {
        "name": name,
        "kind": "transform",
        "n_hold": hold_words,
        "ideal_bpw": (nll / hold_words) if hold_words else 0.0,
        "complete_bpw": (complete_bits / hold_words) if hold_words else 0.0,
        "nll_bits": nll,
        "codebook_bytes": codebook,
        "extra_predictor_bytes": extra,
        "n_tables": n_tables,
        "h_full": None,
        "improves_vs_uncond": None,
        "note": f"packed-byte streams={len(streams)}; hold_bytes={n_hold_units}",
    }


def audit_mantissa_fields(
    words_2d: np.ndarray,
    *,
    prev_layer: np.ndarray | None = None,
) -> dict:
    """Held-out context + transform audit for one 2-D BF16 tensor."""
    mat = as_2d(np.ascontiguousarray(words_2d, dtype=np.uint16))
    sign, exp, mant = split_components(mat)
    sign2 = sign.reshape(mat.shape)
    exp2 = exp.reshape(mat.shape)
    mant2 = mant.reshape(mat.shape)

    s_tr, s_ho = _split_hold(sign2)
    e_tr, e_ho = _split_hold(exp2)
    m_tr, m_ho = _split_hold(mant2)

    reports: list[dict] = []
    uncond = _uncond_report(m_tr, m_ho)
    reports.append(uncond)

    # Contexts (train-only counts, held-out NLL + admitted tables).
    contexts = [
        ("H(M|exp)", e_tr, e_ho, 256),
        ("H(M|sign,exp)", (s_tr.astype(np.int64) << 8) | e_tr.astype(np.int64),
         (s_ho.astype(np.int64) << 8) | e_ho.astype(np.int64), 512),
        ("H(M|prev M)", _split_hold(_prev_value(mant2))[0], _split_hold(_prev_value(mant2))[1], 128),
        ("H(M|prev row)", _split_hold(_prev_row(mant2))[0], _split_hold(_prev_row(mant2))[1], 128),
    ]
    for name, ctx_tr, ctx_ho, n_ctx in contexts:
        reports.append(_conditional_report(name, ctx_tr, m_tr, ctx_ho, m_ho, n_ctx))

    # Combined compact contexts.
    prev_m_tr, prev_m_ho = _split_hold(_prev_value(mant2))
    prev_r_tr, prev_r_ho = _split_hold(_prev_row(mant2))
    reports.append(
        _conditional_report(
            "H(M|exp, prev M)",
            (e_tr.astype(np.int64) << 7) | prev_m_tr.astype(np.int64),
            m_tr,
            (e_ho.astype(np.int64) << 7) | prev_m_ho.astype(np.int64),
            m_ho,
            256 * 128,
        )
    )
    reports.append(
        _conditional_report(
            "H(M|exp, prev row)",
            (e_tr.astype(np.int64) << 7) | prev_r_tr.astype(np.int64),
            m_tr,
            (e_ho.astype(np.int64) << 7) | prev_r_ho.astype(np.int64),
            m_ho,
            256 * 128,
        )
    )

    if prev_layer is not None and prev_layer.shape == mat.shape:
        _ps, _pe, pm = split_components(prev_layer)
        pm2 = pm.reshape(mat.shape)
        pm_tr, pm_ho = _split_hold(pm2)
        reports.append(_conditional_report("H(M|prev layer M)", pm_tr, m_tr, pm_ho, m_ho, 128))
        reports.append(
            _conditional_report(
                "H(M|exp, prev layer M)",
                (e_tr.astype(np.int64) << 7) | pm_tr.astype(np.int64),
                m_tr,
                (e_ho.astype(np.int64) << 7) | pm_ho.astype(np.int64),
                m_ho,
                256 * 128,
            )
        )
    else:
        reports.append(
            {
                "name": "H(M|prev layer M)",
                "kind": "context",
                "n_hold": int(m_ho.size),
                "ideal_bpw": None,
                "complete_bpw": None,
                "skipped": True,
                "reason": "no same-role previous tensor with matching shape",
            }
        )

    # Reversible transforms → Huffman of transformed symbols (one table).
    g_tr, g_ho = _split_hold(gray_encode(mant2).reshape(mat.shape))
    reports.append(_uncond_report(g_tr, g_ho, name="gray(M)"))
    reports[-1]["kind"] = "transform"

    d_prev = _mod_delta(mant2, _prev_value(mant2))
    d_tr, d_ho = _split_hold(d_prev)
    reports.append(_uncond_report(d_tr, d_ho, name="mod_delta_prev(M)"))
    reports[-1]["kind"] = "transform"

    d_row = _mod_delta(mant2, _prev_row(mant2))
    r_tr, r_ho = _split_hold(d_row)
    reports.append(_uncond_report(r_tr, r_ho, name="mod_delta_prev_row(M)"))
    reports[-1]["kind"] = "transform"

    reports.append(_stream_huffman_report("bitplane_transpose(M)", _plane_packed_bytes(mant2), int(mant2.size)))

    s, d, leftover = haar_fwd(mant2)
    # Map Haar bands into 0..255 for the byte Huffman estimator.
    s_u = (s.astype(np.int16) + 64).astype(np.uint8)
    d_u = (d.astype(np.int16) + 127).astype(np.uint8)
    haar_streams = [s_u.ravel(), d_u.ravel()]
    if leftover is not None and leftover.size:
        haar_streams.append(leftover.ravel())
    haar = _stream_huffman_report("haar_row_pairs(M)", haar_streams, int(mant2.size))
    reports.append(haar)

    h_m = float(uncond["ideal_bpw"])
    for rec in reports:
        if rec.get("skipped"):
            continue
        rec["mi_vs_uncond"] = None if rec["ideal_bpw"] is None else h_m - float(rec["ideal_bpw"])
        rec["improves_vs_uncond"] = (
            rec["complete_bpw"] is not None and rec["complete_bpw"] < float(uncond["complete_bpw"]) - 1e-6
        )
        rec["beats_gate"] = rec["complete_bpw"] is not None and rec["complete_bpw"] < GATE_MANT_BPW

    h_sign = shannon_entropy(sign2)
    h_exp = shannon_entropy(exp2)
    scored = [r for r in reports if r.get("complete_bpw") is not None]
    best = min(scored, key=lambda r: r["complete_bpw"]) if scored else uncond
    return {
        "n_words": int(mat.size),
        "shape": [int(mat.shape[0]), int(mat.shape[1])],
        "H_sign": h_sign,
        "H_exp": h_exp,
        "H_mantissa_full": shannon_entropy(mant2),
        "uncond_complete_bpw": uncond["complete_bpw"],
        "best_method": best["name"],
        "best_complete_bpw": best["complete_bpw"],
        "implied_total_bpw": float(best["complete_bpw"]) + h_sign + h_exp,
        "gate_mantissa_bpw": GATE_MANT_BPW,
        "beats_gate": bool(best["complete_bpw"] < GATE_MANT_BPW),
        "methods": reports,
    }


def _weighted(rows: list[dict], key: str) -> float | None:
    num = 0.0
    den = 0.0
    for r in rows:
        val = r.get(key)
        n = r.get("n_words") or 0
        if val is None:
            continue
        num += float(val) * int(n)
        den += int(n)
    return num / den if den else None


def aggregate_audit(tensor_rows: list[dict]) -> dict:
    method_names = []
    for r in tensor_rows:
        for m in r["audit"]["methods"]:
            if m["name"] not in method_names:
                method_names.append(m["name"])
    ranked: list[dict] = []
    n_total = sum(int(r["n_words"]) for r in tensor_rows)
    for name in method_names:
        w_nll = 0.0
        w_complete = 0.0
        w_ideal = 0.0
        w_mi = 0.0
        n = 0
        n_improve = 0
        n_present = 0
        n_tables = 0
        codebook = 0
        skipped = 0
        for r in tensor_rows:
            rec = next((m for m in r["audit"]["methods"] if m["name"] == name), None)
            if rec is None or rec.get("skipped"):
                skipped += 1
                continue
            nw = int(r["n_words"])
            n += nw
            n_present += 1
            w_complete += float(rec["complete_bpw"]) * nw
            w_ideal += float(rec["ideal_bpw"]) * nw
            if rec.get("mi_vs_uncond") is not None:
                w_mi += float(rec["mi_vs_uncond"]) * nw
            n_tables += int(rec.get("n_tables") or 0)
            codebook += int(rec.get("codebook_bytes") or 0)
            if rec.get("improves_vs_uncond"):
                n_improve += 1
        if n == 0:
            ranked.append({"name": name, "skipped_all": True, "n_tensors_skipped": skipped})
            continue
        complete = w_complete / n
        ranked.append(
            {
                "name": name,
                "weighted_ideal_bpw": w_ideal / n,
                "weighted_complete_bpw": complete,
                "weighted_mi": w_mi / n,
                "tensors_present": n_present,
                "tensors_improved_vs_uncond": n_improve,
                "tensors_skipped": skipped,
                "codebook_bytes_sum": codebook,
                "n_tables_sum": n_tables,
                "beats_gate": complete < GATE_MANT_BPW,
            }
        )
    ranked.sort(key=lambda x: x.get("weighted_complete_bpw", 99.0))
    uncond = next((x for x in ranked if x["name"] == "H(M)"), ranked[0] if ranked else {})
    best = ranked[0] if ranked else {}
    h_sign = _weighted([{**r, "H_sign": r["audit"]["H_sign"]} for r in tensor_rows], "H_sign") or 1.0
    h_exp = _weighted([{**r, "H_exp": r["audit"]["H_exp"]} for r in tensor_rows], "H_exp") or 2.6
    # Recompute implied total from best mantissa complete + measured sign/exp.
    best_m = float(best.get("weighted_complete_bpw") or 7.0)
    return {
        "n_tensors": len(tensor_rows),
        "n_words": n_total,
        "original_bytes": 2 * n_total,
        "weighted_H_sign": h_sign,
        "weighted_H_exp": h_exp,
        "weighted_H_mantissa": _weighted(
            [{**r, "H_mantissa_full": r["audit"]["H_mantissa_full"]} for r in tensor_rows],
            "H_mantissa_full",
        ),
        "uncond_complete_bpw": uncond.get("weighted_complete_bpw"),
        "best_method": best.get("name"),
        "best_complete_mantissa_bpw": best_m,
        "implied_total_bpw": best_m + h_sign + h_exp,
        "gate_mantissa_bpw": GATE_MANT_BPW,
        "gate_implied_total_bpw": GATE_MANT_BPW + 1.0 + 2.6,
        "beats_gate": bool(best_m < GATE_MANT_BPW),
        "any_improves_vs_uncond": any(
            (x.get("tensors_improved_vs_uncond") or 0) > 0 and not x.get("skipped_all")
            for x in ranked
        ),
        "ranked_methods": ranked,
    }


def format_markdown(report: dict) -> str:
    s = report["summary"]
    repo = report.get("model", {}).get("repo_id", "checkpoint")
    rev = report.get("model", {}).get("revision", "")
    gate = "PASS" if s["beats_gate"] else "MISS"
    lines = [
        f"# Phase A mantissa audit ({repo})",
        "",
        DISCLAIMER,
        "",
        f"- revision: `{rev}`",
        f"- tensors: **{s['n_tensors']}**",
        f"- words: **{s['n_words']}**",
        f"- original bytes: **{s['original_bytes']}**",
        f"- H(sign)={s['weighted_H_sign']:.4f}  H(exp)={s['weighted_H_exp']:.4f}  "
        f"H(M)={s['weighted_H_mantissa']:.4f}",
        f"- uncond mantissa complete BPW: **{s['uncond_complete_bpw']:.4f}**",
        f"- best method: `{s['best_method']}` at **{s['best_complete_mantissa_bpw']:.4f}** mantissa BPW",
        f"- implied total (best mant + H(sign) + H(exp)): **{s['implied_total_bpw']:.4f}** BPW",
        f"- gate (mantissa complete BPW < {GATE_MANT_BPW}): **{gate}**",
        "",
        "Methods ranked by held-out **complete** mantissa BPW (NLL + table bytes). "
        "MI is H(M) − H(M|ctx) on the same held-out split (ideal, no tables).",
        "",
        "| method | ideal BPW | complete BPW | MI vs H(M) | tensors improved | beats 6.5? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for m in s["ranked_methods"]:
        if m.get("skipped_all"):
            lines.append(f"| {m['name']} | — | — | — | skipped | no |")
            continue
        mi = m.get("weighted_mi")
        mi_s = f"{mi:.4f}" if mi is not None else "—"
        lines.append(
            f"| {m['name']} | {m['weighted_ideal_bpw']:.4f} | {m['weighted_complete_bpw']:.4f} | "
            f"{mi_s} | {m['tensors_improved_vs_uncond']}/{m['tensors_present']} | "
            f"{'yes' if m['beats_gate'] else 'no'} |"
        )
    lines += [
        "",
        "Anything that does not beat unconditional H(M) complete BPW is **rejected** "
        "(including table/predictor overhead). Spatial / grammar / tile methods are "
        "not claimed here. This is not a ≤4 BPW or 1–2 GB / 8 GB result.",
        "",
    ]
    if not s["beats_gate"]:
        lines.append(
            "**Honest negative:** no compact context or cheap reversible transform "
            f"reached mantissa complete BPW < {GATE_MANT_BPW} on this sample."
        )
    else:
        lines.append(
            f"**Gate hit:** `{s['best_method']}` coded held-out mantissas below "
            f"{GATE_MANT_BPW} complete BPW. Still not a ≤4 BPW claim."
        )
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    specs,
    output_json: Path,
    output_md: Path,
    model_meta: dict,
) -> dict:
    by_role: dict[tuple, np.ndarray] = {}
    tensor_rows: list[dict] = []
    ordered = sorted(
        specs,
        key=lambda s: (
            layer_index(s.name) is None,
            layer_index(s.name) if layer_index(s.name) is not None else 10**9,
            s.name,
        ),
    )
    for spec in ordered:
        words = as_2d(load_uint16(spec))
        role = tensor_role(spec.name)
        key = (role, (int(words.shape[0]), int(words.shape[1])))
        prev = by_role.get(key)
        print(f"phase-a {spec.name}  {list(spec.shape)}  prev_layer={'yes' if prev is not None else 'no'}", flush=True)
        audit = audit_mantissa_fields(words, prev_layer=prev)
        by_role[key] = words
        tensor_rows.append(
            {
                "name": spec.name,
                "role": role,
                "shape": list(spec.shape),
                "n_words": int(words.size),
                "audit": audit,
            }
        )
        print(
            f"  -> H(M)={audit['H_mantissa_full']:.4f}  best={audit['best_method']}  "
            f"mant_complete={audit['best_complete_bpw']:.4f}  gate={'yes' if audit['beats_gate'] else 'no'}",
            flush=True,
        )
    summary = aggregate_audit(tensor_rows)
    report = {
        "disclaimer": DISCLAIMER,
        "model": model_meta,
        "summary": summary,
        "tensors": tensor_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(report), encoding="utf-8")
    return report


def select_remaining_specs(specs_all, already, *, max_bytes: int) -> list:
    """2-D 16-bit tensors not in the Stage 1B window (embeddings first)."""
    names = {s.name for s in already}
    rest = [
        s
        for s in specs_all
        if s.dtype in {"BF16", "F16", "FP16"} and s.ndim == 2 and s.name not in names
    ]
    rest.sort(key=lambda s: (not is_embedding_weight(s.name), -s.nbytes, s.name))
    chosen = []
    total = 0
    for spec in rest:
        if len(chosen) >= 4:
            break
        if total >= max_bytes and chosen:
            break
        if total + spec.nbytes > max_bytes and chosen:
            continue
        chosen.append(spec)
        total += spec.nbytes
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A held-out mantissa audit.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--tag", default="qwen")
    parser.add_argument(
        "--remaining",
        action="store_true",
        help="Audit leftover 2-D 16-bit tensors (embeddings first) not in the Stage 1B window.",
    )
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    tag = args.tag
    out_json = args.output_json or Path(f"artifacts/phase_a_mantissa_audit_{tag}.json")
    out_md = args.output_md or Path(f"artifacts/phase_a_mantissa_audit_{tag}.md")
    specs_all = inventory_from_dir(args.model_dir)
    chosen = select_weight_specs(
        specs_all,
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    if args.remaining:
        extra_budget = max(int(cfg.get("max_bytes", 167772160)), 400 * 1024 * 1024)
        chosen = select_remaining_specs(specs_all, chosen, max_bytes=extra_budget)
        tag = tag if tag.endswith("remaining") else f"{tag}_remaining"
        out_json = args.output_json or Path(f"artifacts/phase_a_mantissa_audit_{tag}.json")
        out_md = args.output_md or Path(f"artifacts/phase_a_mantissa_audit_{tag}.md")
    n16 = sum(1 for s in specs_all if s.dtype in {"BF16", "F16", "FP16"})
    meta = {
        "repo_id": cfg.get("repo", PRIMARY_REPO),
        "revision": cfg.get("revision", "main"),
        "license": cfg.get("license", PRIMARY_LICENSE),
        "local_dir": str(args.model_dir),
        "inventory_tensors": len(specs_all),
        "inventory_16bit": n16,
        "selected": [s.name for s in chosen],
        "remaining": bool(args.remaining),
    }
    print(DISCLAIMER)
    print(f"selected {len(chosen)} tensors from {meta['repo_id']} @ {meta['revision']}")
    for spec in chosen:
        print(f"  - {spec.name}  {list(spec.shape)}  {spec.nbytes}")
    report = run_audit(specs=chosen, output_json=out_json, output_md=out_md, model_meta=meta)
    print()
    print(format_markdown(report))
    print(f"Wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
