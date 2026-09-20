"""Selective X/Y vs packed-K mantissa packing.

Packed-K of the *true* (unpadded) nodes is the default. A matrix-family
candidate is stored only when it is strictly cheaper than packed-K after
charging the 1-byte XY flag and the tensor-level mode map. If the hybrid
blob is not strictly smaller than the whole-tensor packed stream, the
encoder falls back to packed-K (S1-identical mantissa bytes).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pbr_h95.bitpack import pack_kbit, packed_bytes_for_k, unpack_kbit
from pbr_q4.codecs import (
    TileEncoding,
    decode_tile,
    encode_tile,
    encode_tile_given_geometry,
)
from pbr_q4.const import (
    MAP_BITMAP,
    MAP_RLE,
    MAP_SPARSE,
    MODE_MATRIX,
    MODE_NAMES,
    PRED_NAMES,
    TILE,
    TRAV_NAMES,
)
from pbr_q4.predictors import pick_best_geometry_batched
from pbr_q4.tiles import as_full_tiles, tile_boxes

# Try expensive XY codecs only when residuals look compressible.
_ZRATE_COMPETE = 0.18


@dataclass(frozen=True)
class _Cand:
    index: int
    n_nodes: int
    packed_len: int
    enc: TileEncoding

    @property
    def xy_cost(self) -> int:
        return 1 + len(self.enc.payload)

    @property
    def savings(self) -> int:
        return self.packed_len - self.xy_cost


def encode_tile_map(n_tiles: int, xy_indices: Sequence[int]) -> bytes:
    """Encode a packed-vs-XY map; pick the shortest of sparse / bitmap / RLE."""
    if n_tiles < 0:
        raise ValueError("n_tiles")
    xy = sorted({int(i) for i in xy_indices})
    for i in xy:
        if i < 0 or i >= n_tiles:
            raise ValueError(f"xy index {i} out of range {n_tiles}")
    cands = [
        _map_sparse(n_tiles, xy),
        _map_bitmap(n_tiles, xy),
        _map_rle(n_tiles, xy),
    ]
    return min(cands, key=len)


def decode_tile_map(data: bytes, n_tiles: int) -> tuple[list[int], int]:
    if n_tiles < 0:
        raise ValueError("n_tiles")
    if not data:
        raise ValueError("empty tile map")
    kind = data[0]
    if kind == MAP_SPARSE:
        return _decode_sparse(data, n_tiles)
    if kind == MAP_BITMAP:
        return _decode_bitmap(data, n_tiles)
    if kind == MAP_RLE:
        return _decode_rle(data, n_tiles)
    raise ValueError(f"unknown tile-map kind {kind}")


def _map_sparse(n_tiles: int, xy: Sequence[int]) -> bytes:
    n_xy = len(xy)
    width = 2 if n_tiles <= 0xFFFF else 4
    dtype = "<u2" if width == 2 else "<u4"
    body = np.asarray(xy, dtype=dtype).tobytes() if n_xy else b""
    return struct.pack("<BBI", MAP_SPARSE, width, n_xy) + body


def _decode_sparse(data: bytes, n_tiles: int) -> tuple[list[int], int]:
    if len(data) < 6:
        raise ValueError("short sparse map")
    _kind, width, n_xy = struct.unpack_from("<BBI", data, 0)
    if width not in (2, 4):
        raise ValueError(f"bad sparse width {width}")
    need = 6 + n_xy * width
    if len(data) < need:
        raise ValueError("short sparse body")
    dtype = np.uint16 if width == 2 else np.uint32
    idx = np.frombuffer(memoryview(data)[6:need], dtype=dtype)
    out = [int(x) for x in idx.tolist()]
    if any(i < 0 or i >= n_tiles for i in out):
        raise ValueError("sparse index out of range")
    return out, need


def _map_bitmap(n_tiles: int, xy: Sequence[int]) -> bytes:
    mask = np.zeros(n_tiles, dtype=np.uint8)
    if xy:
        mask[np.asarray(xy, dtype=np.int64)] = 1
    bits = np.packbits(mask, bitorder="little").tobytes()
    return struct.pack("<BI", MAP_BITMAP, n_tiles) + bits


def _decode_bitmap(data: bytes, n_tiles: int) -> tuple[list[int], int]:
    if len(data) < 5:
        raise ValueError("short bitmap map")
    _kind, n = struct.unpack_from("<BI", data, 0)
    if n != n_tiles:
        raise ValueError(f"bitmap n_tiles {n} != {n_tiles}")
    nbytes = (n_tiles + 7) // 8
    need = 5 + nbytes
    if len(data) < need:
        raise ValueError("short bitmap body")
    bits = np.unpackbits(
        np.frombuffer(memoryview(data)[5:need], dtype=np.uint8), bitorder="little"
    )[:n_tiles]
    return [int(i) for i in np.flatnonzero(bits).tolist()], need


def _map_rle(n_tiles: int, xy: Sequence[int]) -> bytes:
    xy_set = set(xy)
    body = bytearray()
    n_runs = 0
    i = 0
    while i < n_tiles:
        is_xy = i in xy_set
        j = i + 1
        while j < n_tiles and ((j in xy_set) == is_xy):
            j += 1
        length = j - i
        flag = 0x80 if is_xy else 0x00
        if 1 <= length < 128:
            body.append(flag | length)
        else:
            body.append(flag)  # length field 0 ⇒ extended u32
            body.extend(struct.pack("<I", length))
        n_runs += 1
        i = j
    return struct.pack("<BI", MAP_RLE, n_runs) + bytes(body)


def _decode_rle(data: bytes, n_tiles: int) -> tuple[list[int], int]:
    if len(data) < 5:
        raise ValueError("short RLE map")
    _kind, n_runs = struct.unpack_from("<BI", data, 0)
    off = 5
    xy: list[int] = []
    t = 0
    for _ in range(int(n_runs)):
        if off >= len(data):
            raise ValueError("short RLE run")
        b = data[off]
        off += 1
        is_xy = bool(b & 0x80)
        length = b & 0x7F
        if length == 0:
            if off + 4 > len(data):
                raise ValueError("short RLE extended length")
            (length,) = struct.unpack_from("<I", data, off)
            off += 4
        if length <= 0 or t + length > n_tiles:
            raise ValueError("bad RLE length")
        if is_xy:
            xy.extend(range(t, t + length))
        t += length
    if t != n_tiles:
        raise ValueError(f"RLE covered {t} != {n_tiles}")
    return xy, off


def _as_rank2(kept: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    arr = np.ascontiguousarray(kept, dtype=np.uint16)
    orig_shape = tuple(int(x) for x in arr.shape)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(int(arr.shape[0]), -1)
    return arr, orig_shape, int(arr.shape[0]), int(arr.shape[1])


def _hist() -> dict[str, int]:
    return {n: 0 for n in MODE_NAMES.values()}


def _empty_stats(nbits: int, rows: int, cols: int, orig_shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "kind": "packed",
        "blob": b"",
        "rows": rows,
        "cols": cols,
        "orig_shape": orig_shape,
        "nbits": nbits,
        "n_tiles": 0,
        "n_xy_tiles": 0,
        "n_packed_tiles": 0,
        "packed_whole_bytes": 0,
        "blob_bytes": 0,
        "xy_payload_bytes": 0,
        "map_bytes": 0,
        "flag_bytes": 0,
        "packed_stream_bytes": 0,
        "saved_vs_packed": 0,
        "map_kind": None,
        "mode_hist": _hist(),
        "pred_hist": {},
        "trav_hist": {},
        "fallback_packed": True,
        "padded": False,
    }


def _hybrid_blob(
    arr: np.ndarray,
    nbits: int,
    boxes: list[tuple[int, int, int, int]],
    chosen: Sequence[_Cand],
) -> tuple[bytes, dict[str, Any]]:
    n_tiles = len(boxes)
    xy_indices = [c.index for c in chosen]
    xy_set = set(xy_indices)
    map_b = encode_tile_map(n_tiles, xy_indices)
    packed_nodes: list[np.ndarray] = []
    n_packed_nodes = 0
    for i, (r0, c0, h, w) in enumerate(boxes):
        if i in xy_set:
            continue
        sl = np.ascontiguousarray(arr[r0 : r0 + h, c0 : c0 + w]).ravel()
        packed_nodes.append(sl)
        n_packed_nodes += int(sl.size)
    packed_stream = (
        pack_kbit(np.concatenate(packed_nodes), nbits) if packed_nodes else b""
    )
    flags = bytearray()
    payloads = bytearray()
    mode_hist = _hist()
    pred_hist: dict[str, int] = {}
    trav_hist: dict[str, int] = {}
    for c in chosen:
        flags.append(c.enc.flag_byte())
        payloads.extend(c.enc.payload)
        mode_hist[c.enc.mode_name] = mode_hist.get(c.enc.mode_name, 0) + 1
        pn = PRED_NAMES[c.enc.pred]
        tn = TRAV_NAMES[c.enc.trav]
        pred_hist[pn] = pred_hist.get(pn, 0) + 1
        trav_hist[tn] = trav_hist.get(tn, 0) + 1
    blob = map_b + packed_stream + bytes(flags) + bytes(payloads)
    meta = {
        "map_bytes": len(map_b),
        "map_kind": map_b[0] if map_b else None,
        "packed_stream_bytes": len(packed_stream),
        "flag_bytes": len(flags),
        "xy_payload_bytes": len(payloads),
        "mode_hist": mode_hist,
        "pred_hist": pred_hist,
        "trav_hist": trav_hist,
        "n_packed_nodes": n_packed_nodes,
    }
    return blob, meta


def _choose_candidates(
    arr: np.ndarray,
    nbits: int,
    boxes: list[tuple[int, int, int, int]],
    cands: list[_Cand],
    packed_whole: int,
) -> tuple[list[_Cand], bytes, dict[str, Any]]:
    """Keep a subset of XY tiles so the hybrid blob is strictly smaller than packed."""
    viable = [c for c in cands if c.savings > 0]
    viable.sort(key=lambda c: (c.savings, -c.index))
    chosen = list(viable)
    while chosen:
        blob, meta = _hybrid_blob(arr, nbits, boxes, chosen)
        if len(blob) < packed_whole:
            return chosen, blob, meta
        # Drop the least-saving tile and retry (complete cost includes the map).
        chosen = chosen[1:]
    return [], b"", {}


def encode_array_selective(
    kept: np.ndarray, nbits: int, *, th: int = TILE, tw: int = TILE
) -> dict[str, Any]:
    """Pack *kept* codes: packed-K default, X/Y only when strictly smaller."""
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
        return stats

    boxes = list(tile_boxes(rows, cols, th=th, tw=tw))
    n_tiles = len(boxes)
    cands: list[_Cand] = []

    n_tc = (cols + tw - 1) // tw if cols else 0
    full_r = (rows // th) * th
    full_c = (cols // tw) * tw
    if full_r >= th and full_c >= tw and n_tc:
        inner_tc = full_c // tw
        for c in _candidates_aligned(arr[:full_r, :full_c], nbits, th, tw):
            tr = c.index // inner_tc
            tc = c.index % inner_tc
            cands.append(_Cand(tr * n_tc + tc, c.n_nodes, c.packed_len, c.enc))
    for i, (r0, c0, h, w) in enumerate(boxes):
        if h == th and w == tw:
            continue
        tile = arr[r0 : r0 + h, c0 : c0 + w]
        packed_len = packed_bytes_for_k(h * w, nbits)
        enc = encode_tile(tile, nbits)
        if enc.mode == MODE_MATRIX:
            continue
        if 1 + len(enc.payload) < packed_len:
            cands.append(_Cand(i, h * w, packed_len, enc))

    chosen, hybrid, meta = _choose_candidates(arr, nbits, boxes, cands, packed_whole)
    use_hybrid = bool(chosen) and hybrid and len(hybrid) < packed_whole
    blob = hybrid if use_hybrid else packed_whole_blob
    mode_hist = meta.get("mode_hist") if use_hybrid else _hist()
    if not use_hybrid:
        mode_hist = _hist()
        meta = {
            "map_bytes": 0,
            "map_kind": None,
            "packed_stream_bytes": packed_whole,
            "flag_bytes": 0,
            "xy_payload_bytes": 0,
            "pred_hist": {},
            "trav_hist": {},
        }

    out = {
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
        "mode_hist": mode_hist,
        "pred_hist": meta.get("pred_hist") or {},
        "trav_hist": meta.get("trav_hist") or {},
        "fallback_packed": not use_hybrid,
        "padded": False,
    }
    return out


def _candidates_aligned(arr: np.ndarray, nbits: int, th: int, tw: int) -> list[_Cand]:
    tiles_all = as_full_tiles(arr, th=th, tw=tw)
    packed_one = packed_bytes_for_k(th * tw, nbits)
    Ttot = int(tiles_all.shape[0])
    out: list[_Cand] = []
    chunk = 4096
    for start in range(0, Ttot, chunk):
        tiles = tiles_all[start : start + chunk]
        travs, preds, res = pick_best_geometry_batched(tiles, nbits)
        zrate = np.mean(res == 0, axis=(1, 2))
        compete = zrate >= _ZRATE_COMPETE
        if nbits > 0:
            uni = np.zeros(int(tiles.shape[0]), dtype=bool)
            for b in range(nbits):
                plane = (res >> np.uint8(b)) & np.uint8(1)
                flatp = plane.reshape(int(tiles.shape[0]), -1)
                uni |= flatp.all(axis=1) | (~flatp.any(axis=1))
            compete = compete | uni
        if not np.any(compete):
            continue
        for i in np.flatnonzero(compete).tolist():
            enc = encode_tile_given_geometry(
                res[i], pred=int(preds[i]), trav=int(travs[i]), nbits=nbits, verify=False
            )
            if enc.mode == MODE_MATRIX:
                continue
            if 1 + len(enc.payload) < packed_one:
                out.append(_Cand(start + int(i), th * tw, packed_one, enc))
    return out


def decode_array_selective(
    blob: bytes,
    *,
    rows: int,
    cols: int,
    nbits: int,
    n_nodes: int | None = None,
    th: int = TILE,
    tw: int = TILE,
) -> np.ndarray:
    """Decode a packed-K or selective-XY mantissa blob into K-codes."""
    n_expect = int(rows) * int(cols) if n_nodes is None else int(n_nodes)
    packed_len = packed_bytes_for_k(n_expect, nbits)
    if len(blob) == packed_len:
        vals = unpack_kbit(blob, n_expect, nbits)
        return vals.reshape(rows, cols).astype(np.uint16)
    if nbits <= 0:
        return np.zeros((rows, cols), dtype=np.uint16)
    boxes = list(tile_boxes(rows, cols, th=th, tw=tw))
    n_tiles = len(boxes)
    xy_indices, map_used = decode_tile_map(blob, n_tiles)
    xy_set = set(xy_indices)
    n_packed_nodes = 0
    for i, (_r0, _c0, h, w) in enumerate(boxes):
        if i not in xy_set:
            n_packed_nodes += h * w
    packed_stream_len = packed_bytes_for_k(n_packed_nodes, nbits)
    off = map_used
    packed_stream = blob[off : off + packed_stream_len]
    if len(packed_stream) != packed_stream_len:
        raise ValueError("short packed stream in XY-selective blob")
    off += packed_stream_len
    n_xy = len(xy_indices)
    flags = blob[off : off + n_xy]
    if len(flags) != n_xy:
        raise ValueError("short XY flags")
    off += n_xy
    xy_payload = blob[off:]

    packed_vals = unpack_kbit(packed_stream, n_packed_nodes, nbits) if n_packed_nodes else np.zeros(0, dtype=np.uint16)
    out = np.zeros((rows, cols), dtype=np.uint16)
    ppos = 0
    xoff = 0
    fi = 0
    for i, (r0, c0, h, w) in enumerate(boxes):
        n = h * w
        if i in xy_set:
            recon, used = decode_tile(flags[fi], xy_payload[xoff:], rows=h, cols=w, nbits=nbits)
            if used < 0 or xoff + used > len(xy_payload):
                raise ValueError("XY payload overrun")
            out[r0 : r0 + h, c0 : c0 + w] = recon.astype(np.uint16)
            xoff += used
            fi += 1
        else:
            out[r0 : r0 + h, c0 : c0 + w] = packed_vals[ppos : ppos + n].reshape(h, w)
            ppos += n
    if ppos != n_packed_nodes:
        raise ValueError("packed node count mismatch")
    if fi != n_xy:
        raise ValueError("XY flag count mismatch")
    if xoff != len(xy_payload):
        raise ValueError(f"trailing XY payload {len(xy_payload) - xoff} bytes")
    return out


def packed_equals_blob(blob: bytes, n_nodes: int, nbits: int) -> bool:
    return len(blob) == packed_bytes_for_k(n_nodes, nbits)
