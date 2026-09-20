"""Matrix-family codecs for one 256-node (or partial) tile.

Modes (all X/Y family — packed-raw is never emitted):
  XY_MATRIX   predictor residuals packed at K bits (always exact)
  XY_RANS     rANS on traversal-order residuals
  XY_BITPLANE uniform-plane flags + raw remaining planes
  XY_PAIR     local top-15 residual-pair dictionary + escapes
  XY_RUN      run-length on traversal-order residuals

Selection is min complete payload bytes; ties prefer MATRIX.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode
from pbr_core.rans import table_from_symbols as rans_table_from_symbols
from pbr_h95.bitpack import pack_kbit, packed_bytes_for_k, unpack_kbit
from pbr_q4.const import (
    MATRIX_FAMILY_MODES,
    MODE_MATRIX,
    MODE_NAMES,
    MODE_PAIR,
    MODE_PLANE,
    MODE_RANS,
    MODE_RUN,
    MODE_TIE_ORDER,
    PRED_AVG,
    PRED_LEFT,
    PRED_PAETH,
    PRED_PREVIOUS,
    PRED_UP,
    TILE,
    TRAV_ROW,
)
from pbr_q4.predictors import (
    _seen_left,
    _seen_nw,
    _seen_up,
    pick_best_geometry,
    pick_best_geometry_batched,
    reconstruct_from_flat,
    reconstruct_tile,
)
from pbr_q4.tiles import as_full_tiles, gather_traversal, scatter_traversal, tile_boxes, traversal_coords

PAIR_HORIZ = 0
PAIR_VERT = 1
PAIR_TRAV = 2
_MAX_DICT = 15
_MAX_RUN = 255


@dataclass(frozen=True)
class TileEncoding:
    mode: int
    pred: int
    trav: int
    nbits: int
    rows: int
    cols: int
    payload: bytes
    packed_baseline_bytes: int

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.mode, f"MODE_{self.mode}")

    @property
    def family(self) -> str:
        return "matrix"

    @property
    def complete_bytes(self) -> int:
        return 1 + len(self.payload)  # flags byte is charged at the tensor stream

    def flag_byte(self) -> int:
        if self.mode not in MATRIX_FAMILY_MODES:
            raise ValueError("non-matrix mode is illegal on the wire")
        return (self.mode & 7) | ((self.pred & 7) << 3) | ((self.trav & 3) << 6)


def parse_flag_byte(b: int) -> tuple[int, int, int]:
    mode = b & 7
    pred = (b >> 3) & 7
    trav = (b >> 6) & 3
    if mode not in MATRIX_FAMILY_MODES:
        raise ValueError(f"illegal tile mode {mode} (packed-raw is not a wire path)")
    return mode, pred, trav


def packed_baseline_len(n_nodes: int, nbits: int) -> int:
    return packed_bytes_for_k(n_nodes, nbits)


def unpack_kbit_batch(packed: np.ndarray, n: int, nbits: int) -> np.ndarray:
    """Inverse of :func:`pack_kbit_batch`. ``packed`` is ``(T, nbytes)``."""
    p = np.ascontiguousarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("unpack_kbit_batch expects (T, nbytes)")
    T = int(p.shape[0])
    if nbits <= 0 or T == 0 or n <= 0:
        return np.zeros((T, n), dtype=np.uint16)
    bits = np.unpackbits(p, axis=1, bitorder="big")[:, : n * nbits]
    bits = bits.reshape(T, n, nbits)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint16)
    vals = np.zeros((T, n), dtype=np.uint16)
    for i, sh in enumerate(shifts.tolist()):
        vals |= bits[:, :, i].astype(np.uint16) << np.uint16(sh)
    return vals


def reconstruct_stack(
    residuals: np.ndarray, *, pred: int, trav: int, nbits: int
) -> np.ndarray:
    """Vectorized-over-tiles sequential reconstruct. ``residuals`` is (T,H,W)."""
    res = np.ascontiguousarray(residuals, dtype=np.uint16)
    if res.ndim != 3:
        raise ValueError("reconstruct_stack expects (T,H,W)")
    T, H, W = (int(x) for x in res.shape)
    mask = (1 << nbits) - 1 if nbits else 0
    out = np.zeros((T, H, W), dtype=np.uint16)
    prev = np.zeros(T, dtype=np.int32)
    coords = traversal_coords(H, W, trav)
    for y, x in coords:
        if pred == PRED_PREVIOUS:
            p = prev
        elif pred == PRED_LEFT:
            if x > 0 and _seen_left(trav, y, x):
                p = out[:, y, x - 1].astype(np.int32)
            else:
                p = prev
        elif pred == PRED_UP:
            if y > 0 and _seen_up(trav, y, x):
                p = out[:, y - 1, x].astype(np.int32)
            else:
                p = prev
        elif pred == PRED_AVG:
            have_l = x > 0 and _seen_left(trav, y, x)
            have_u = y > 0 and _seen_up(trav, y, x)
            if have_l and have_u:
                p = (out[:, y, x - 1].astype(np.int32) + out[:, y - 1, x].astype(np.int32)) // 2
            elif have_l:
                p = out[:, y, x - 1].astype(np.int32)
            elif have_u:
                p = out[:, y - 1, x].astype(np.int32)
            else:
                p = prev
        elif pred == PRED_PAETH:
            have_l = x > 0 and _seen_left(trav, y, x)
            have_u = y > 0 and _seen_up(trav, y, x)
            have_nw = y > 0 and x > 0 and _seen_nw(trav, y, x)
            if not have_l and not have_u:
                p = prev
            elif not have_l:
                p = out[:, y - 1, x].astype(np.int32)
            elif not have_u:
                p = out[:, y, x - 1].astype(np.int32)
            else:
                L = out[:, y, x - 1].astype(np.int32)
                U = out[:, y - 1, x].astype(np.int32)
                NW = out[:, y - 1, x - 1].astype(np.int32) if have_nw else np.zeros(T, dtype=np.int32)
                est = L + U - NW
                dL = np.abs(est - L)
                dU = np.abs(est - U)
                dNW = np.abs(est - NW)
                p = L.copy()
                use_u = (dU < dL) & (dU <= dNW)
                use_nw = have_nw and ((dNW < dL) & (dNW < dU))
                p = np.where(use_u, U, p)
                if have_nw:
                    p = np.where(use_nw, NW, p)
        else:
            raise ValueError(pred)
        actual = (p + res[:, y, x].astype(np.int32)) & mask
        out[:, y, x] = actual
        prev = actual
    return out.astype(np.uint8)


def pack_kbit_batch(tiles: np.ndarray, nbits: int) -> np.ndarray:
    """Pack a ``(T, H, W)`` stack; returns ``(T, nbytes)`` uint8."""
    t = np.ascontiguousarray(tiles, dtype=np.uint16)
    if t.ndim != 3:
        raise ValueError("pack_kbit_batch expects (T,H,W)")
    T = int(t.shape[0])
    n = int(t.shape[1] * t.shape[2])
    if nbits <= 0 or T == 0:
        return np.zeros((T, 0), dtype=np.uint8)
    mask = (1 << nbits) - 1
    vals = t.reshape(T, n) & np.uint16(mask)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint16)
    bits = ((vals[:, :, None] >> shifts[None, None, :]) & np.uint16(1)).astype(np.uint8)
    bits = bits.reshape(T, n * nbits)
    pad = (-(n * nbits)) % 8
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    return np.packbits(bits, axis=1, bitorder="big")


def _encode_matrix(res: np.ndarray, nbits: int) -> bytes:
    return pack_kbit(res.ravel(), nbits)


def _decode_matrix(data: bytes, n: int, nbits: int) -> tuple[np.ndarray, int]:
    need = packed_bytes_for_k(n, nbits)
    if len(data) < need:
        raise ValueError(f"short MATRIX residual: need {need}, got {len(data)}")
    return unpack_kbit(data[:need], n, nbits), need


def _encode_planes(res: np.ndarray, nbits: int) -> bytes:
    flat = np.ascontiguousarray(res, dtype=np.uint16).ravel()
    n = int(flat.size)
    if nbits <= 0 or n == 0:
        return struct.pack("<H", 0)
    flags = 0
    chunks = bytearray()
    for i in range(nbits):
        plane = ((flat >> np.uint16(i)) & np.uint16(1)).astype(np.uint8)
        shift = 2 * i
        if np.all(plane == 0):
            flags |= 0 << shift
        elif np.all(plane == 1):
            flags |= 1 << shift
        else:
            flags |= 2 << shift
            chunks.extend(np.packbits(plane, bitorder="big").tobytes())
    return struct.pack("<H", flags) + bytes(chunks)


def _decode_planes(data: bytes, n: int, nbits: int) -> tuple[np.ndarray, int]:
    if nbits <= 0 or n == 0:
        return np.zeros(n, dtype=np.uint16), (2 if len(data) >= 2 else 0)
    if len(data) < 2:
        raise ValueError("short bitplane header")
    (flags,) = struct.unpack_from("<H", data, 0)
    off = 2
    out = np.zeros(n, dtype=np.uint16)
    plane_bytes = (n + 7) // 8
    for i in range(nbits):
        kind = (flags >> (2 * i)) & 3
        if kind == 0:
            continue
        if kind == 1:
            out |= np.uint16(1 << i)
            continue
        if kind != 2:
            raise ValueError(f"bad plane kind {kind}")
        if off + plane_bytes > len(data):
            raise ValueError("short bitplane body")
        bits = np.unpackbits(
            np.frombuffer(memoryview(data)[off : off + plane_bytes], dtype=np.uint8),
            bitorder="big",
        )[:n]
        out |= bits.astype(np.uint16) << np.uint16(i)
        off += plane_bytes
    return out, off


def _encode_run(flat: np.ndarray) -> bytes:
    vals = np.ascontiguousarray(flat, dtype=np.uint8).ravel()
    n = int(vals.size)
    if n == 0:
        return struct.pack("<H", 0)
    runs_v: list[int] = []
    runs_l: list[int] = []
    i = 0
    while i < n:
        v = int(vals[i])
        j = i + 1
        lim = min(n, i + _MAX_RUN)
        while j < lim and int(vals[j]) == v:
            j += 1
        runs_v.append(v)
        runs_l.append(j - i)
        i = j
    body = np.empty(len(runs_v) * 2, dtype=np.uint8)
    body[0::2] = np.asarray(runs_v, dtype=np.uint8)
    body[1::2] = np.asarray(runs_l, dtype=np.uint8)
    return struct.pack("<H", len(runs_v)) + body.tobytes()


def _decode_run(data: bytes, n: int) -> tuple[np.ndarray, int]:
    if len(data) < 2:
        raise ValueError("short RUN header")
    (n_runs,) = struct.unpack_from("<H", data, 0)
    need = 2 + 2 * n_runs
    if len(data) < need:
        raise ValueError("short RUN body")
    if n_runs == 0:
        if n != 0:
            raise ValueError("empty RUN for non-empty tile")
        return np.zeros(0, dtype=np.uint8), 2
    body = np.frombuffer(memoryview(data)[2:need], dtype=np.uint8)
    values = body[0::2]
    lengths = body[1::2]
    out = np.empty(n, dtype=np.uint8)
    pos = 0
    for v, L in zip(values.tolist(), lengths.tolist()):
        L = int(L)
        if L <= 0 or pos + L > n:
            raise ValueError("bad RUN length")
        out[pos : pos + L] = np.uint8(v)
        pos += L
    if pos != n:
        raise ValueError(f"RUN decoded {pos} != {n}")
    return out, need


def _encode_rans(flat: np.ndarray) -> bytes:
    stream = np.ascontiguousarray(flat, dtype=np.uint8).ravel()
    if stream.size == 0:
        return struct.pack("<II", 0, 0)
    freq = rans_table_from_symbols(stream)
    table = dump_freq_table(freq)
    bitstream = rans_encode(stream, freq)
    return struct.pack("<II", len(table), len(bitstream)) + table + bitstream


def _decode_rans(data: bytes, n: int) -> tuple[np.ndarray, int]:
    if len(data) < 8:
        raise ValueError("short rANS header")
    table_len, stream_len = struct.unpack_from("<II", data, 0)
    off = 8
    if table_len == 0 and n == 0:
        return np.zeros(0, dtype=np.uint8), 8
    freq, end = load_freq_table(data, off)
    if end - off != table_len:
        raise ValueError("rANS table length mismatch")
    bitstream = data[end : end + stream_len]
    if len(bitstream) != stream_len:
        raise ValueError("rANS stream truncated")
    return rans_decode(bitstream, n, freq), end + stream_len


def _pair_keys(a: np.ndarray, b: np.ndarray, nbits: int) -> np.ndarray:
    return (a.astype(np.uint16) << np.uint16(nbits)) | b.astype(np.uint16)


def _encode_pair(res_spatial: np.ndarray, flat_trav: np.ndarray, nbits: int) -> bytes | None:
    h, w = int(res_spatial.shape[0]), int(res_spatial.shape[1])
    n = int(res_spatial.size)
    if n < 2 or nbits <= 0:
        return None
    candidates: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    if w >= 2:
        a = res_spatial[:, 0 : (w - w % 2) : 2].ravel()
        b = res_spatial[:, 1 : (w - w % 2) : 2].ravel()
        rem = res_spatial[:, w - 1].ravel() if (w % 2) else None
        candidates.append((PAIR_HORIZ, _pair_keys(a, b, nbits), rem))
    if h >= 2:
        a = res_spatial[0 : (h - h % 2) : 2, :].ravel()
        b = res_spatial[1 : (h - h % 2) : 2, :].ravel()
        rem = res_spatial[h - 1, :].ravel() if (h % 2) else None
        candidates.append((PAIR_VERT, _pair_keys(a, b, nbits), rem))
    ft = np.ascontiguousarray(flat_trav, dtype=np.uint8).ravel()
    a = ft[0 : (n - n % 2) : 2]
    b = ft[1 : (n - n % 2) : 2]
    rem = ft[n - 1 : n] if (n % 2) else None
    candidates.append((PAIR_TRAV, _pair_keys(a, b, nbits), rem))

    best_blob: bytes | None = None
    for kind, keys, rem in candidates:
        if keys.size == 0:
            continue
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)
        top = uniq[order[:_MAX_DICT]]
        n_dict = int(top.size)
        lut = {int(v): i for i, v in enumerate(top.tolist())}
        ids = np.empty(keys.size, dtype=np.uint8)
        esc: list[int] = []
        for i, key in enumerate(keys.tolist()):
            if key in lut:
                ids[i] = lut[key]
            else:
                ids[i] = 15
                esc.append(int(key))
        blob = bytearray()
        blob.append(kind & 0xFF)
        blob.append(n_dict & 0xFF)
        blob.extend(np.asarray(top, dtype="<u2").tobytes())
        blob.extend(pack_kbit(ids, 4))
        blob.extend(struct.pack("<H", len(esc)))
        if esc:
            blob.extend(np.asarray(esc, dtype="<u2").tobytes())
        if rem is not None and rem.size:
            blob.append(1)
            blob.extend(np.asarray(rem, dtype=np.uint8).tobytes())
        else:
            blob.append(0)
        raw = bytes(blob)
        if best_blob is None or len(raw) < len(best_blob):
            best_blob = raw
    return best_blob


def _decode_pair(data: bytes, h: int, w: int, nbits: int, trav: int) -> tuple[np.ndarray, int]:
    if len(data) < 3:
        raise ValueError("short PAIR header")
    kind = data[0]
    n_dict = data[1]
    off = 2
    top = np.frombuffer(memoryview(data)[off : off + 2 * n_dict], dtype="<u2").copy()
    off += 2 * n_dict
    if kind == PAIR_HORIZ:
        n_pairs = h * (w // 2)
    elif kind == PAIR_VERT:
        n_pairs = (h // 2) * w
    elif kind == PAIR_TRAV:
        n_pairs = (h * w) // 2
    else:
        raise ValueError(f"bad pair kind {kind}")
    id_bytes = packed_bytes_for_k(n_pairs, 4)
    ids = unpack_kbit(data[off : off + id_bytes], n_pairs, 4)
    off += id_bytes
    (n_esc,) = struct.unpack_from("<H", data, off)
    off += 2
    esc = np.frombuffer(memoryview(data)[off : off + 2 * n_esc], dtype="<u2").copy()
    off += 2 * n_esc
    has_rem = data[off]
    off += 1
    rem = np.zeros(0, dtype=np.uint8)
    if has_rem:
        # remainder length depends on kind
        if kind == PAIR_HORIZ:
            rlen = h if (w % 2) else 0
        elif kind == PAIR_VERT:
            rlen = w if (h % 2) else 0
        else:
            rlen = (h * w) % 2
        rem = np.frombuffer(memoryview(data)[off : off + rlen], dtype=np.uint8).copy()
        off += rlen

    mask = (1 << nbits) - 1
    a = np.empty(n_pairs, dtype=np.uint16)
    b = np.empty(n_pairs, dtype=np.uint16)
    ei = 0
    for i, pid in enumerate(ids.tolist()):
        if int(pid) == 15:
            if ei >= esc.size:
                raise ValueError("PAIR escape underrun")
            key = int(esc[ei])
            ei += 1
        else:
            if int(pid) >= n_dict:
                raise ValueError("PAIR id out of dict")
            key = int(top[int(pid)])
        a[i] = (key >> nbits) & mask
        b[i] = key & mask
    if ei != int(esc.size):
        raise ValueError("PAIR escape leftover")

    spatial = np.zeros((h, w), dtype=np.uint8)
    if kind == PAIR_HORIZ:
        ww = w - (w % 2)
        spatial[:, 0:ww:2] = a.reshape(h, ww // 2)
        spatial[:, 1:ww:2] = b.reshape(h, ww // 2)
        if rem.size:
            spatial[:, w - 1] = rem
    elif kind == PAIR_VERT:
        hh = h - (h % 2)
        spatial[0:hh:2, :] = a.reshape(hh // 2, w)
        spatial[1:hh:2, :] = b.reshape(hh // 2, w)
        if rem.size:
            spatial[h - 1, :] = rem
    else:
        flat = np.empty(h * w, dtype=np.uint8)
        flat[0 : 2 * n_pairs : 2] = a
        flat[1 : 2 * n_pairs : 2] = b
        if rem.size:
            flat[-1] = rem[0]
        spatial = scatter_traversal(flat, h, w, trav, dtype=np.uint8)
    return spatial.ravel(), off


def _should_try_rans(flat: np.ndarray, nbits: int, packed_len: int) -> bool:
    n = int(flat.size)
    if n < 32 or nbits <= 0:
        return False
    uniq = int(np.unique(flat).size)
    if uniq > min(12, 1 << nbits):
        return False
    # Frequency table + stream cannot beat packed unless the alphabet is peaked.
    return uniq <= 8 or float(np.mean(flat == 0)) >= 0.45


def encode_tile(codes: np.ndarray, nbits: int) -> TileEncoding:
    """Encode one tile; always returns a matrix-family candidate."""
    tile = np.ascontiguousarray(codes, dtype=np.uint16)
    if tile.ndim != 2:
        raise ValueError("tile must be rank-2")
    h, w = int(tile.shape[0]), int(tile.shape[1])
    n = h * w
    if nbits < 0 or nbits > 8:
        raise ValueError(nbits)
    packed_base = packed_baseline_len(n, nbits)
    if n == 0 or nbits == 0:
        return TileEncoding(
            mode=MODE_MATRIX,
            pred=PRED_PREVIOUS,
            trav=TRAV_ROW,
            nbits=nbits,
            rows=h,
            cols=w,
            payload=b"",
            packed_baseline_bytes=packed_base,
        )

    trav, pred, res_spatial = pick_best_geometry(tile, nbits)
    flat = gather_traversal(res_spatial, trav)

    cand: list[tuple[int, bytes]] = [(MODE_MATRIX, _encode_matrix(res_spatial, nbits))]
    plane_b = _encode_planes(res_spatial, nbits)
    cand.append((MODE_PLANE, plane_b))
    zrate = float(np.mean(flat == 0)) if flat.size else 0.0
    if zrate >= 0.20:
        cand.append((MODE_RUN, _encode_run(flat)))
    if nbits <= 5 and n >= 8:
        pair_b = _encode_pair(res_spatial, flat, nbits)
        if pair_b is not None:
            cand.append((MODE_PAIR, pair_b))
    if _should_try_rans(flat, nbits, packed_base):
        cand.append((MODE_RANS, _encode_rans(flat)))

    # Reject anything that does not independently decode.
    survivors: list[tuple[int, bytes]] = []
    for mode, blob in cand:
        try:
            rec, used = _decode_residuals(mode, blob, h, w, nbits, trav)
            if used != len(blob):
                continue
            recon = (
                reconstruct_tile(rec.reshape(h, w), pred=pred, trav=trav, nbits=nbits)
                if mode in (MODE_MATRIX, MODE_PLANE, MODE_PAIR)
                else reconstruct_from_flat(rec, h, w, pred=pred, trav=trav, nbits=nbits)
            )
            expect = (tile.astype(np.uint16) & np.uint16((1 << nbits) - 1)).astype(np.uint8)
            if np.array_equal(recon, expect):
                survivors.append((mode, blob))
        except (ValueError, struct.error, RuntimeError):
            continue
    if not survivors:
        # MATRIX is constructed to be exact; this is a safety net.
        survivors = [(MODE_MATRIX, _encode_matrix(res_spatial, nbits))]

    tie = {m: i for i, m in enumerate(MODE_TIE_ORDER)}
    mode, payload = min(survivors, key=lambda t: (len(t[1]), tie.get(t[0], 99)))
    if mode not in MATRIX_FAMILY_MODES:
        raise RuntimeError("encoder attempted a non-matrix wire mode")
    return TileEncoding(
        mode=mode,
        pred=pred,
        trav=trav,
        nbits=nbits,
        rows=h,
        cols=w,
        payload=payload,
        packed_baseline_bytes=packed_base,
    )


def encode_tile_given_geometry(
    res_spatial: np.ndarray,
    *,
    pred: int,
    trav: int,
    nbits: int,
    verify: bool = False,
) -> TileEncoding:
    """Compete matrix-family codecs for an already-chosen residual tile."""
    res = np.ascontiguousarray(res_spatial, dtype=np.uint8)
    h, w = int(res.shape[0]), int(res.shape[1])
    n = h * w
    packed_base = packed_baseline_len(n, nbits)
    if n == 0 or nbits == 0:
        return TileEncoding(
            mode=MODE_MATRIX,
            pred=pred,
            trav=trav,
            nbits=nbits,
            rows=h,
            cols=w,
            payload=b"",
            packed_baseline_bytes=packed_base,
        )
    flat = gather_traversal(res, trav)
    cand: list[tuple[int, bytes]] = [(MODE_MATRIX, _encode_matrix(res, nbits))]
    cand.append((MODE_PLANE, _encode_planes(res, nbits)))
    zrate = float(np.mean(flat == 0)) if flat.size else 0.0
    if zrate >= 0.20:
        cand.append((MODE_RUN, _encode_run(flat)))
    if nbits <= 5 and n >= 8:
        pair_b = _encode_pair(res, flat, nbits)
        if pair_b is not None:
            cand.append((MODE_PAIR, pair_b))
    if _should_try_rans(flat, nbits, packed_base):
        cand.append((MODE_RANS, _encode_rans(flat)))
    if verify:
        survivors: list[tuple[int, bytes]] = []
        for mode, blob in cand:
            try:
                rec, used = _decode_residuals(mode, blob, h, w, nbits, trav)
                if used != len(blob):
                    continue
                recon = (
                    reconstruct_tile(rec.reshape(h, w), pred=pred, trav=trav, nbits=nbits)
                    if mode in (MODE_MATRIX, MODE_PLANE, MODE_PAIR)
                    else reconstruct_from_flat(rec, h, w, pred=pred, trav=trav, nbits=nbits)
                )
                expect = reconstruct_tile(res, pred=pred, trav=trav, nbits=nbits)
                if np.array_equal(recon, expect):
                    survivors.append((mode, blob))
            except (ValueError, struct.error, RuntimeError):
                continue
        cand = survivors or [(MODE_MATRIX, _encode_matrix(res, nbits))]
    tie = {m: i for i, m in enumerate(MODE_TIE_ORDER)}
    mode, payload = min(cand, key=lambda t: (len(t[1]), tie.get(t[0], 99)))
    if mode not in MATRIX_FAMILY_MODES:
        raise RuntimeError("encoder attempted a non-matrix wire mode")
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


def _decode_residuals(
    mode: int, data: bytes, h: int, w: int, nbits: int, trav: int
) -> tuple[np.ndarray, int]:
    n = h * w
    if mode == MODE_MATRIX:
        vals, used = _decode_matrix(data, n, nbits)
        return vals, used
    if mode == MODE_PLANE:
        vals, used = _decode_planes(data, n, nbits)
        return vals, used
    if mode == MODE_RUN:
        vals, used = _decode_run(data, n)
        return vals.astype(np.uint16), used
    if mode == MODE_RANS:
        vals, used = _decode_rans(data, n)
        return vals.astype(np.uint16), used
    if mode == MODE_PAIR:
        vals, used = _decode_pair(data, h, w, nbits, trav)
        return vals.astype(np.uint16), used
    raise ValueError(f"illegal mode {mode}")


def decode_tile(
    flag: int,
    data: bytes,
    *,
    rows: int,
    cols: int,
    nbits: int,
) -> tuple[np.ndarray, int]:
    mode, pred, trav = parse_flag_byte(flag)
    res, used = _decode_residuals(mode, data, rows, cols, nbits, trav)
    if mode in (MODE_RUN, MODE_RANS):
        recon = reconstruct_from_flat(res, rows, cols, pred=pred, trav=trav, nbits=nbits)
    else:
        spatial = res.reshape(rows, cols).astype(np.uint8)
        recon = reconstruct_tile(spatial, pred=pred, trav=trav, nbits=nbits)
    return recon, used


def encode_array_xy(kept: np.ndarray, nbits: int, *, th: int = TILE, tw: int = TILE) -> dict[str, Any]:
    """Tile a 2-D (or 1-D) code array and encode every tile via the X/Y family."""
    arr = np.ascontiguousarray(kept, dtype=np.uint16)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("kept codes must be rank-1 or rank-2")
    orig_rows, orig_cols = int(arr.shape[0]), int(arr.shape[1])
    pr = ((orig_rows + th - 1) // th) * th if orig_rows else 0
    pc = ((orig_cols + tw - 1) // tw) * tw if orig_cols else 0
    if (pr, pc) != (orig_rows, orig_cols) and pr > 0 and pc > 0:
        padded = np.zeros((pr, pc), dtype=np.uint16)
        padded[:orig_rows, :orig_cols] = arr
        arr = padded
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    flags = bytearray()
    payload = bytearray()
    hist: dict[str, int] = {n: 0 for n in MODE_NAMES.values()}
    pred_hist: dict[str, int] = {}
    trav_hist: dict[str, int] = {}
    packed_base = 0
    xy_payload = 0
    n_tiles = 0
    n_xy_lt = n_xy_eq = n_xy_gt = 0
    xy_win_bytes = 0
    from pbr_q4.const import PRED_NAMES, TRAV_NAMES

    def _commit(enc: TileEncoding) -> None:
        nonlocal packed_base, xy_payload, n_tiles, n_xy_lt, n_xy_eq, n_xy_gt, xy_win_bytes
        if enc.mode not in MATRIX_FAMILY_MODES:
            raise RuntimeError("tile escaped matrix family")
        flags.append(enc.flag_byte())
        payload.extend(enc.payload)
        hist[enc.mode_name] = hist.get(enc.mode_name, 0) + 1
        pn = PRED_NAMES[enc.pred]
        tn = TRAV_NAMES[enc.trav]
        pred_hist[pn] = pred_hist.get(pn, 0) + 1
        trav_hist[tn] = trav_hist.get(tn, 0) + 1
        packed_base += enc.packed_baseline_bytes
        xy_payload += len(enc.payload)
        n_tiles += 1
        plen = len(enc.payload)
        pbase = enc.packed_baseline_bytes
        if plen < pbase:
            n_xy_lt += 1
            xy_win_bytes += pbase - plen
        elif plen > pbase:
            n_xy_gt += 1
        else:
            n_xy_eq += 1

    aligned = rows % th == 0 and cols % tw == 0 and rows > 0 and cols > 0
    if aligned:
        tiles_all = as_full_tiles(arr, th=th, tw=tw)
        packed_one = packed_baseline_len(th * tw, nbits)
        chunk = 4096
        Ttot = int(tiles_all.shape[0])
        for start in range(0, Ttot, chunk):
            tiles = tiles_all[start : start + chunk]
            travs, preds, res = pick_best_geometry_batched(tiles, nbits)
            zrate = np.mean(res == 0, axis=(1, 2))
            compete = zrate >= 0.20
            if nbits > 0:
                uni = np.zeros(int(tiles.shape[0]), dtype=bool)
                for b in range(nbits):
                    plane = (res >> np.uint8(b)) & np.uint8(1)
                    flatp = plane.reshape(int(tiles.shape[0]), -1)
                    uni |= flatp.all(axis=1) | (~flatp.any(axis=1))
                compete = compete | uni
            matrix_payloads = pack_kbit_batch(res, nbits)
            for i in range(int(tiles.shape[0])):
                if compete[i]:
                    enc = encode_tile_given_geometry(
                        res[i], pred=int(preds[i]), trav=int(travs[i]), nbits=nbits, verify=False
                    )
                else:
                    enc = TileEncoding(
                        mode=MODE_MATRIX,
                        pred=int(preds[i]),
                        trav=int(travs[i]),
                        nbits=nbits,
                        rows=th,
                        cols=tw,
                        payload=matrix_payloads[i].tobytes(),
                        packed_baseline_bytes=packed_one,
                    )
                _commit(enc)
    else:
        for r0, c0, h, w in tile_boxes(rows, cols, th=th, tw=tw):
            _commit(encode_tile(arr[r0 : r0 + h, c0 : c0 + w], nbits))

    return {
        "rows": orig_rows,
        "cols": orig_cols,
        "pad_rows": rows,
        "pad_cols": cols,
        "nbits": nbits,
        "n_tiles": n_tiles,
        "flags": bytes(flags),
        "payload": bytes(payload),
        "mode_hist": hist,
        "pred_hist": pred_hist,
        "trav_hist": trav_hist,
        "packed_baseline_bytes": packed_base,
        "xy_payload_bytes": xy_payload,
        "n_tiles_xy_lt_packed": n_xy_lt,
        "n_tiles_xy_eq_packed": n_xy_eq,
        "n_tiles_xy_gt_packed": n_xy_gt,
        "xy_win_bytes_vs_packed": xy_win_bytes,
        "all_matrix_family": True,
    }


def decode_array_xy(
    flags: bytes,
    payload: bytes,
    *,
    rows: int,
    cols: int,
    nbits: int,
    th: int = TILE,
    tw: int = TILE,
    pad_rows: int | None = None,
    pad_cols: int | None = None,
) -> np.ndarray:
    pr = int(pad_rows if pad_rows is not None else rows)
    pc = int(pad_cols if pad_cols is not None else cols)
    boxes = list(tile_boxes(pr, pc, th=th, tw=tw))
    if len(flags) != len(boxes):
        raise ValueError(f"flag/tile count mismatch {len(flags)} vs {len(boxes)}")
    out = np.zeros((pr, pc), dtype=np.uint8)
    full = all(h == th and w == tw for _r, _c, h, w in boxes)
    if full and boxes:
        decoded = _decode_full_tiles(flags, payload, nbits=nbits, th=th, tw=tw)
        for i, (r0, c0, h, w) in enumerate(boxes):
            out[r0 : r0 + h, c0 : c0 + w] = decoded[i]
        return out[:rows, :cols]
    off = 0
    for i, (r0, c0, h, w) in enumerate(boxes):
        recon, used = decode_tile(flags[i], payload[off:], rows=h, cols=w, nbits=nbits)
        if used < 0 or off + used > len(payload):
            raise ValueError("tile payload overrun")
        out[r0 : r0 + h, c0 : c0 + w] = recon
        off += used
    if off != len(payload):
        raise ValueError(f"trailing tile payload {len(payload) - off} bytes")
    return out[:rows, :cols]


def _reconstruct_grouped(
    residuals: np.ndarray, preds: np.ndarray, travs: np.ndarray, nbits: int
) -> np.ndarray:
    rec = np.empty(residuals.shape, dtype=np.uint8)
    for pred in np.unique(preds).tolist():
        for trav in np.unique(travs).tolist():
            m = (preds == pred) & (travs == trav)
            if not np.any(m):
                continue
            rec[m] = reconstruct_stack(residuals[m], pred=int(pred), trav=int(trav), nbits=nbits)
    return rec


def _decode_full_tiles(flags: bytes, payload: bytes, *, nbits: int, th: int, tw: int) -> np.ndarray:
    """Decode a homogeneous 16×16 tile stack (matrix family, mixed modes)."""
    T = len(flags)
    n = th * tw
    need = packed_bytes_for_k(n, nbits)
    flag_arr = np.frombuffer(flags, dtype=np.uint8)
    modes = flag_arr & np.uint8(7)
    preds = (flag_arr >> 3) & np.uint8(7)
    travs = (flag_arr >> 6) & np.uint8(3)

    if T > 0 and int(modes.min()) == MODE_MATRIX and int(modes.max()) == MODE_MATRIX:
        if len(payload) != T * need:
            raise ValueError(f"MATRIX payload size {len(payload)} != {T * need}")
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(T, need)
        residuals = unpack_kbit_batch(packed, n, nbits).reshape(T, th, tw)
        return _reconstruct_grouped(residuals, preds, travs, nbits)

    out = np.zeros((T, th, tw), dtype=np.uint8)
    is_m = modes == np.uint8(MODE_MATRIX)
    n_mat = int(np.count_nonzero(is_m))
    mat_packed = np.empty((n_mat, need), dtype=np.uint8)
    mat_index = np.empty(n_mat, dtype=np.int64)
    off = 0
    mi = 0
    i = 0
    mv = memoryview(payload)
    while i < T:
        if bool(is_m[i]):
            j = i + 1
            while j < T and bool(is_m[j]):
                j += 1
            count = j - i
            nbytes = count * need
            if off + nbytes > len(payload):
                raise ValueError("short MATRIX payload")
            mat_packed[mi : mi + count] = np.frombuffer(mv[off : off + nbytes], dtype=np.uint8).reshape(
                count, need
            )
            mat_index[mi : mi + count] = np.arange(i, j, dtype=np.int64)
            off += nbytes
            mi += count
            i = j
            continue
        recon, used = decode_tile(int(flags[i]), payload[off:], rows=th, cols=tw, nbits=nbits)
        out[i] = recon
        off += used
        i += 1
    if off != len(payload):
        raise ValueError(f"trailing tile payload {len(payload) - off} bytes")
    if n_mat:
        residuals = unpack_kbit_batch(mat_packed, n, nbits).reshape(n_mat, th, tw)
        rec = _reconstruct_grouped(residuals, preds[mat_index], travs[mat_index], nbits)
        out[mat_index] = rec
    return out
