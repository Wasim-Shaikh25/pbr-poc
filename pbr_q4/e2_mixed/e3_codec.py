"""E3 selective lossless tile codec on E2 codes.

Phase-1 mode set only: packed (product default), LEFT / UP / PAETH on row
traversal, then BITPLANE / RUN / rANS. No PAIR, no extra traversals, no
always-on X/Y.

A matrix-family tile is stored only when its complete physical cost
(payload + 1-byte flag + share of the tensor mode map) saves at least
``max(16 bits, 0.02 × raw_tile_bits)``. If the hybrid blob misses that
margin vs whole-tensor packed, the encoder falls back to packed.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from pbr_h95.bitpack import pack_kbit, packed_bytes_for_k
from pbr_q4.codecs import (
    TileEncoding,
    _decode_residuals,
    _encode_planes,
    _encode_rans,
    _encode_run,
    _should_try_rans,
)
from pbr_q4.const import (
    MODE_MATRIX,
    MODE_NAMES,
    MODE_PLANE,
    MODE_RANS,
    MODE_RUN,
    PRED_LEFT,
    PRED_NAMES,
    PRED_PAETH,
    PRED_UP,
    TILE,
    TRAV_NAMES,
    TRAV_ROW,
)
from pbr_q4.e2_mixed.const import (
    E3_CODECS,
    E3_MIN_SAVE_BITS,
    E3_MIN_SAVE_FRAC,
    E3_PREDS,
)
from pbr_q4.predictors import batched_residual_maps, residual_tile
from pbr_q4.selective import (
    _Cand,
    _as_rank2,
    _choose_candidates,
    _empty_stats,
    _hist,
    decode_array_selective,
)
from pbr_q4.tiles import as_full_tiles, gather_traversal, tile_boxes

_ZRATE_RUN = 0.20
_ZRATE_COMPETE = 0.18


def tile_margin_bits(n_nodes: int, nbits: int) -> int:
    raw = int(n_nodes) * int(nbits)
    return max(int(E3_MIN_SAVE_BITS), int(np.ceil(E3_MIN_SAVE_FRAC * raw)))


def _encode_phase1(res: np.ndarray, *, pred: int, trav: int, nbits: int) -> TileEncoding:
    """Compete BITPLANE / RUN / rANS for a residual tile. No PAIR, no MATRIX."""
    res_u = np.ascontiguousarray(res, dtype=np.uint8)
    h, w = int(res_u.shape[0]), int(res_u.shape[1])
    n = h * w
    packed_base = packed_bytes_for_k(n, nbits)
    if n == 0 or nbits == 0:
        return TileEncoding(
            mode=MODE_PLANE,
            pred=pred,
            trav=trav,
            nbits=nbits,
            rows=h,
            cols=w,
            payload=b"",
            packed_baseline_bytes=packed_base,
        )
    flat = gather_traversal(res_u, trav)
    cand: list[tuple[int, bytes]] = [(MODE_PLANE, _encode_planes(res_u, nbits))]
    zrate = float(np.mean(flat == 0)) if flat.size else 0.0
    if zrate >= _ZRATE_RUN:
        cand.append((MODE_RUN, _encode_run(flat)))
    if _should_try_rans(flat, nbits, packed_base):
        cand.append((MODE_RANS, _encode_rans(flat)))
    survivors: list[tuple[int, bytes]] = []
    for mode, blob in cand:
        if mode not in E3_CODECS:
            continue
        try:
            rec, used = _decode_residuals(mode, blob, h, w, nbits, trav)
            if used != len(blob):
                continue
            from pbr_q4.predictors import reconstruct_from_flat, reconstruct_tile

            recon = (
                reconstruct_tile(rec.reshape(h, w), pred=pred, trav=trav, nbits=nbits)
                if mode == MODE_PLANE
                else reconstruct_from_flat(rec, h, w, pred=pred, trav=trav, nbits=nbits)
            )
            expect = reconstruct_tile(res_u, pred=pred, trav=trav, nbits=nbits)
            if np.array_equal(recon, expect):
                survivors.append((mode, blob))
        except (ValueError, struct.error, RuntimeError):
            continue
    if not survivors:
        # BITPLANE is constructed to be exact; keep it as a candidate even if
        # it will lose to packed on complete cost.
        survivors = [(MODE_PLANE, _encode_planes(res_u, nbits))]
    mode, payload = min(survivors, key=lambda t: (len(t[1]), t[0]))
    return TileEncoding(
        mode=mode,
        pred=int(pred),
        trav=int(trav),
        nbits=nbits,
        rows=h,
        cols=w,
        payload=payload,
        packed_baseline_bytes=packed_base,
    )


def encode_tile_phase1(codes: np.ndarray, nbits: int) -> TileEncoding:
    """Pick LEFT / UP / PAETH on row traversal, then Phase-1 codecs."""
    tile = np.ascontiguousarray(codes, dtype=np.uint16)
    if tile.ndim != 2:
        raise ValueError("tile must be rank-2")
    h, w = int(tile.shape[0]), int(tile.shape[1])
    packed_base = packed_bytes_for_k(h * w, nbits)
    if h * w == 0 or nbits == 0:
        return TileEncoding(
            mode=MODE_PLANE,
            pred=PRED_LEFT,
            trav=TRAV_ROW,
            nbits=nbits,
            rows=h,
            cols=w,
            payload=b"",
            packed_baseline_bytes=packed_base,
        )
    best_pred = PRED_LEFT
    best_z = -1.0
    best_res = None
    for pred in E3_PREDS:
        res = residual_tile(tile, pred=pred, trav=TRAV_ROW, nbits=nbits)
        z = float(np.mean(res == 0)) if res.size else 0.0
        if z > best_z + 1e-12 or (abs(z - best_z) <= 1e-12 and pred < best_pred):
            best_z = z
            best_pred = pred
            best_res = res
    assert best_res is not None
    return _encode_phase1(best_res, pred=best_pred, trav=TRAV_ROW, nbits=nbits)


def _candidates_aligned_phase1(arr: np.ndarray, nbits: int, th: int, tw: int) -> list[_Cand]:
    tiles_all = as_full_tiles(arr, th=th, tw=tw)
    packed_one = packed_bytes_for_k(th * tw, nbits)
    margin = tile_margin_bits(th * tw, nbits)
    Ttot = int(tiles_all.shape[0])
    out: list[_Cand] = []
    maps = batched_residual_maps(tiles_all, nbits)
    maps = [(t, p, r) for t, p, r in maps if t == TRAV_ROW and p in E3_PREDS]
    if not maps:
        return out
    T = Ttot
    best_z = np.full(T, -1.0)
    best_pred = np.full(T, 99, dtype=np.int16)
    best_res = np.empty(tiles_all.shape, dtype=np.uint8)
    for trav, pred, res in maps:
        z = np.mean(res == 0, axis=(1, 2))
        better = (z > best_z + 1e-12) | ((np.abs(z - best_z) <= 1e-12) & (pred < best_pred))
        if not np.any(better):
            continue
        best_z = np.where(better, z, best_z)
        best_pred = np.where(better, pred, best_pred)
        best_res[better] = res[better]
    compete = best_z >= _ZRATE_COMPETE
    if nbits > 0:
        uni = np.zeros(T, dtype=bool)
        for b in range(nbits):
            plane = (best_res >> np.uint8(b)) & np.uint8(1)
            flatp = plane.reshape(T, -1)
            uni |= flatp.all(axis=1) | (~flatp.any(axis=1))
        compete = compete | uni
    for i in np.flatnonzero(compete).tolist():
        enc = _encode_phase1(
            best_res[i], pred=int(best_pred[i]), trav=TRAV_ROW, nbits=nbits
        )
        if enc.mode == MODE_MATRIX:
            continue
        save_bits = (packed_one - (1 + len(enc.payload))) * 8
        if save_bits >= margin:
            out.append(_Cand(int(i), th * tw, packed_one, enc))
    return out


def encode_array_e3(
    kept: np.ndarray,
    nbits: int,
    *,
    th: int = TILE,
    tw: int = TILE,
) -> dict[str, Any]:
    """Packed default; Phase-1 matrix family only with net-margin complete cost."""
    arr, orig_shape, rows, cols = _as_rank2(kept)
    n_nodes = int(arr.size)
    if nbits < 0 or nbits > 8:
        raise ValueError(nbits)
    packed_whole_blob = pack_kbit(arr.ravel(), nbits) if nbits and n_nodes else b""
    packed_whole = len(packed_whole_blob)
    if n_nodes == 0 or nbits == 0:
        stats = _empty_stats(nbits, rows, cols, orig_shape)
        stats["blob"] = packed_whole_blob
        stats["packed_whole_bytes"] = packed_whole
        stats["blob_bytes"] = packed_whole
        stats["n_tiles"] = 0
        stats["product_default"] = "packed"
        stats["net_margin_bits"] = 0
        return stats

    boxes = list(tile_boxes(rows, cols, th=th, tw=tw))
    n_tiles = len(boxes)
    cands: list[_Cand] = []
    n_tc = (cols + tw - 1) // tw if cols else 0
    full_r = (rows // th) * th
    full_c = (cols // tw) * tw
    if full_r >= th and full_c >= tw and n_tc:
        inner_tc = full_c // tw
        for c in _candidates_aligned_phase1(arr[:full_r, :full_c], nbits, th, tw):
            tr = c.index // inner_tc
            tc = c.index % inner_tc
            cands.append(_Cand(tr * n_tc + tc, c.n_nodes, c.packed_len, c.enc))
    for i, (r0, c0, h, w) in enumerate(boxes):
        if h == th and w == tw:
            continue
        tile = arr[r0 : r0 + h, c0 : c0 + w]
        packed_len = packed_bytes_for_k(h * w, nbits)
        enc = encode_tile_phase1(tile, nbits)
        if enc.mode == MODE_MATRIX:
            continue
        save_bits = (packed_len - (1 + len(enc.payload))) * 8
        if save_bits >= tile_margin_bits(h * w, nbits):
            cands.append(_Cand(i, h * w, packed_len, enc))

    tensor_margin_bytes = max(2, int(np.ceil(tile_margin_bits(n_nodes, nbits) / 8.0)))
    # Reuse #16 complete-cost subset search, then demand the net margin on
    # the surviving hybrid blob (all map/flag/payload/pad costs included).
    chosen, hybrid, meta = _choose_candidates(arr, nbits, boxes, cands, packed_whole)
    use_hybrid = (
        bool(chosen)
        and hybrid
        and (packed_whole - len(hybrid) >= tensor_margin_bytes)
    )
    blob = hybrid if use_hybrid else packed_whole_blob
    if not use_hybrid:
        chosen = []
        meta = {
            "map_bytes": 0,
            "map_kind": None,
            "packed_stream_bytes": packed_whole,
            "flag_bytes": 0,
            "xy_payload_bytes": 0,
            "pred_hist": {},
            "trav_hist": {},
            "mode_hist": _hist(),
        }
    illegal = [
        m
        for m, c in (meta.get("mode_hist") or {}).items()
        if c and m not in MODE_NAMES.values()
    ]
    if illegal:
        raise RuntimeError(f"non-matrix XY modes in E3 histogram: {illegal}")
    if (meta.get("mode_hist") or {}).get("XY_MATRIX"):
        raise RuntimeError("E3 selected XY_MATRIX; packed is the product default")
    if (meta.get("mode_hist") or {}).get("XY_PAIR"):
        raise RuntimeError("E3 selected XY_PAIR; not in the Phase-1 set")

    return {
        "kind": "xy_sel" if use_hybrid else "packed",
        "blob": blob,
        "rows": rows,
        "cols": cols,
        "orig_shape": orig_shape,
        "nbits": nbits,
        "n_tiles": n_tiles,
        "n_xy_tiles": len(chosen) if use_hybrid else 0,
        "n_packed_tiles": n_tiles - (len(chosen) if use_hybrid else 0),
        "packed_whole_bytes": packed_whole,
        "blob_bytes": len(blob),
        "xy_payload_bytes": int(meta.get("xy_payload_bytes") or 0),
        "map_bytes": int(meta.get("map_bytes") or 0),
        "flag_bytes": int(meta.get("flag_bytes") or 0),
        "packed_stream_bytes": int(meta.get("packed_stream_bytes") or 0),
        "saved_vs_packed": packed_whole - len(blob),
        "map_kind": meta.get("map_kind"),
        "mode_hist": meta.get("mode_hist") or _hist(),
        "pred_hist": meta.get("pred_hist") or {},
        "trav_hist": meta.get("trav_hist") or {},
        "fallback_packed": not use_hybrid,
        "padded": False,
        "product_default": "packed",
        "net_margin_bits": tile_margin_bits(n_nodes, nbits) if n_nodes else 0,
        "phase1_preds": [PRED_NAMES[p] for p in E3_PREDS],
        "phase1_codecs": [MODE_NAMES[m] for m in E3_CODECS],
    }


def decode_array_e3(
    blob: bytes,
    *,
    rows: int,
    cols: int,
    nbits: int,
    th: int = TILE,
    tw: int = TILE,
) -> np.ndarray:
    """Same wire as PR #16 selective blobs (packed default / map+flags+payload)."""
    return decode_array_selective(blob, rows=rows, cols=cols, nbits=nbits, th=th, tw=tw)
