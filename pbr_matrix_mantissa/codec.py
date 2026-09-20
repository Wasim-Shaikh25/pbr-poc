"""Matrix 7-bit mantissa codec V1–V3.

Per tile, every legal (predictor × traversal) plus RAW is encoded, decoded in
memory, and scored on **complete** bytes (tile header + residual payload).
RAW wins ties. Unused predictors, shapes, and the exponent plane are not
serialized. Tails (partial edge tiles) are preserved.

This is a pre-Qwen qualification codec. It does not claim Qwen BPW or ≤4 BPW.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np

from pbr_matrix_mantissa.bf16 import join_components, split_components
from pbr_matrix_mantissa.bitio import ceil_bytes, pack_lsb, unpack_lsb
from pbr_matrix_mantissa.errors import CodecError
from pbr_matrix_mantissa.predict import (
    PRED_AVG,
    PRED_EXP_LEFT,
    PRED_LEFT,
    PRED_NAMES,
    PRED_PAETH,
    PRED_PREV,
    PRED_RAW,
    PRED_UP,
    predict,
)
from pbr_matrix_mantissa.residual import decode_residuals, encode_residuals, residual
from pbr_matrix_mantissa.traverse import (
    TRAV_COL,
    TRAV_COL_SERP,
    TRAV_NAMES,
    TRAV_ROW,
    TRAV_SERP,
    coord_list,
)

MAGIC = b"MM7\x01"
BUNDLE_MAGIC = b"MB16"
VERSIONS = (1, 2, 3)
_HDR = struct.Struct("<4sBIIIB")  # magic, version, rows, cols, n_tiles, flags
_TILE = struct.Struct("<HHHHBBH")  # row0, col0, h, w, pred, trav_res, payload_len
FLAG_NEEDS_EXP = 1 << 0

SHAPES = {
    1: ((16, 16),),
    2: ((8, 8), (8, 16), (16, 16), (16, 32)),
    3: ((8, 8), (8, 16), (16, 16), (16, 32), (32, 32)),
}
PREDS = {
    1: (PRED_LEFT, PRED_UP, PRED_PREV),
    2: (PRED_LEFT, PRED_UP, PRED_PREV, PRED_AVG),
    3: (PRED_LEFT, PRED_UP, PRED_PREV, PRED_AVG, PRED_PAETH, PRED_EXP_LEFT),
}
TRAVS = {
    1: (TRAV_ROW, TRAV_SERP),
    2: (TRAV_ROW, TRAV_SERP, TRAV_COL),
    3: (TRAV_ROW, TRAV_SERP, TRAV_COL, TRAV_COL_SERP),
}

DISCLAIMER = (
    "Matrix mantissa codec V1–V3 pre-Qwen qualification. Exact uint16 / 7-bit "
    "mantissa restore. Complete bytes include headers, directory, residual "
    "payloads, and padding. Do not claim Qwen compression. No ≤4 BPW claim."
)


class _Cand:
    __slots__ = ("pred", "trav", "kind", "payload", "nbytes", "is_raw")

    def __init__(
        self,
        pred: int,
        trav: int,
        kind: int,
        payload: bytes,
        nbytes: int,
        is_raw: bool,
    ) -> None:
        self.pred = pred
        self.trav = trav
        self.kind = kind
        self.payload = payload
        self.nbytes = nbytes
        self.is_raw = is_raw


def _as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {a.shape}")
    return a


def iter_tiles(rows: int, cols: int, tile_h: int, tile_w: int) -> list[tuple[int, int, int, int]]:
    if tile_h < 1 or tile_w < 1:
        raise ValueError("tile shape must be >= 1")
    out: list[tuple[int, int, int, int]] = []
    for r0 in range(0, rows, tile_h):
        for c0 in range(0, cols, tile_w):
            h = min(tile_h, rows - r0)
            w = min(tile_w, cols - c0)
            out.append((r0, c0, h, w))
    return out


def _neighbors(
    decoded: np.ndarray, out: np.ndarray, r: int, c: int
) -> tuple[int, int, int, bool, bool, bool]:
    has_left = c > 0 and bool(decoded[r, c - 1])
    has_up = r > 0 and bool(decoded[r - 1, c])
    has_ul = r > 0 and c > 0 and bool(decoded[r - 1, c - 1])
    left = int(out[r, c - 1]) if has_left else 0
    up = int(out[r - 1, c]) if has_up else 0
    ul = int(out[r - 1, c - 1]) if has_ul else 0
    return left, up, ul, has_left, has_up, has_ul


def _apply_tile(
    mant: np.ndarray,
    exp: np.ndarray | None,
    pred_id: int,
    trav: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (residuals_or_raw_values, reconstructed)."""
    h, w = int(mant.shape[0]), int(mant.shape[1])
    rec = np.zeros((h, w), dtype=np.uint8)
    stored = np.zeros((h, w), dtype=np.uint8)
    decoded = np.zeros((h, w), dtype=bool)
    prev = 0
    eplane = None if exp is None else np.ascontiguousarray(exp, dtype=np.uint8)
    for r, c in coord_list(h, w, trav):
        actual = int(mant[r, c]) & 0x7F
        if pred_id == PRED_RAW:
            stored[r, c] = actual
            rec[r, c] = actual
        else:
            left, up, ul, has_l, has_u, has_ul = _neighbors(decoded, rec, r, c)
            ev = int(eplane[r, c]) if eplane is not None else 0
            lev = int(eplane[r, c - 1]) if (eplane is not None and has_l) else 0
            pred = predict(
                pred_id,
                left=left,
                up=up,
                up_left=ul,
                prev=prev,
                exp=ev,
                left_exp=lev,
                has_left=has_l,
                has_up=has_u,
                has_up_left=has_ul,
            )
            stored[r, c] = residual(actual, pred)
            rec[r, c] = (pred + int(stored[r, c])) & 0x7F
        decoded[r, c] = True
        prev = int(rec[r, c])
    return stored, rec


def _restore_tile(
    stored: np.ndarray,
    pred_id: int,
    trav: int,
    exp: np.ndarray | None,
) -> np.ndarray:
    h, w = int(stored.shape[0]), int(stored.shape[1])
    rec = np.zeros((h, w), dtype=np.uint8)
    decoded = np.zeros((h, w), dtype=bool)
    prev = 0
    eplane = None if exp is None else np.ascontiguousarray(exp, dtype=np.uint8)
    for r, c in coord_list(h, w, trav):
        if pred_id == PRED_RAW:
            rec[r, c] = int(stored[r, c]) & 0x7F
        else:
            left, up, ul, has_l, has_u, has_ul = _neighbors(decoded, rec, r, c)
            ev = int(eplane[r, c]) if eplane is not None else 0
            lev = int(eplane[r, c - 1]) if (eplane is not None and has_l) else 0
            pred = predict(
                pred_id,
                left=left,
                up=up,
                up_left=ul,
                prev=prev,
                exp=ev,
                left_exp=lev,
                has_left=has_l,
                has_up=has_u,
                has_up_left=has_ul,
            )
            rec[r, c] = (pred + int(stored[r, c])) & 0x7F
        decoded[r, c] = True
        prev = int(rec[r, c])
    return rec


def _pack_trav_res(trav: int, kind: int) -> int:
    return (int(trav) & 0x3) | ((int(kind) & 0x3) << 2)


def _unpack_trav_res(byte: int) -> tuple[int, int]:
    return int(byte) & 0x3, (int(byte) >> 2) & 0x3


def _tile_record(row0: int, col0: int, h: int, w: int, cand: _Cand) -> bytes:
    return (
        _TILE.pack(
            row0,
            col0,
            h,
            w,
            int(cand.pred),
            _pack_trav_res(cand.trav, cand.kind),
            len(cand.payload),
        )
        + cand.payload
    )


def _score_tile(
    tile_m: np.ndarray,
    tile_e: np.ndarray | None,
    preds: tuple[int, ...],
    travs: tuple[int, ...],
    *,
    force_pred: int | None = None,
    force_trav: int | None = None,
) -> _Cand:
    h, w = int(tile_m.shape[0]), int(tile_m.shape[1])
    raw_payload = pack_lsb(tile_m, 7)
    raw = _Cand(PRED_RAW, TRAV_ROW, 0, raw_payload, _TILE.size + len(raw_payload), True)
    rec_raw = unpack_lsb(raw_payload, h * w, 7).reshape(h, w)
    if not np.array_equal(rec_raw, tile_m):
        raise CodecError("RAW tile failed byte-compare")
    if force_pred == PRED_RAW:
        return raw

    best: _Cand | None = raw if force_pred is None else None
    pred_ids = (force_pred,) if force_pred is not None else preds
    trav_ids = (force_trav,) if force_trav is not None else travs
    for pred_id in pred_ids:
        if pred_id == PRED_RAW:
            continue
        if pred_id == PRED_EXP_LEFT and tile_e is None and force_pred != PRED_EXP_LEFT:
            continue
        for trav in trav_ids:
            stored, rec = _apply_tile(tile_m, tile_e, pred_id, trav)
            if not np.array_equal(rec, tile_m):
                raise CodecError(
                    f"candidate {PRED_NAMES.get(pred_id)}/{TRAV_NAMES.get(trav)} "
                    "failed exact restore"
                )
            kind, payload = encode_residuals(stored)
            decoded_store = decode_residuals(payload, h * w, kind).reshape(h, w)
            if not np.array_equal(decoded_store, stored):
                raise CodecError("residual payload failed byte-compare")
            back = _restore_tile(decoded_store, pred_id, trav, tile_e)
            if not np.array_equal(back, tile_m):
                raise CodecError("decode(encode) mantissa mismatch")
            nbytes = _TILE.size + len(payload)
            cand = _Cand(pred_id, trav, kind, payload, nbytes, False)
            if best is None:
                best = cand
            elif (cand.nbytes, 0 if cand.is_raw else 1) < (
                best.nbytes,
                0 if best.is_raw else 1,
            ):
                best = cand
    if best is None:
        raise CodecError("no tile candidate produced a payload")
    return best


@dataclass
class MantissaEncode:
    blob: bytes
    version: int
    rows: int
    cols: int
    n_words: int
    strategy: str
    tile_shape: tuple[int, int] | None
    n_tiles: int
    complete_bytes: int
    mantissa_bpw: float
    needs_exp: bool
    pred_counts: dict[str, int] = field(default_factory=dict)
    trav_counts: dict[str, int] = field(default_factory=dict)
    shape_used: tuple[int, int] | None = None
    sha256: str = ""


def _header(version: int, rows: int, cols: int, n_tiles: int, flags: int) -> bytes:
    return _HDR.pack(MAGIC, int(version), int(rows), int(cols), int(n_tiles), int(flags))


def encode_all_raw(mant: np.ndarray, *, version: int = 1) -> MantissaEncode:
    m = _as_2d(np.ascontiguousarray(mant, dtype=np.uint8) & np.uint8(0x7F))
    rows, cols = int(m.shape[0]), int(m.shape[1])
    payload = pack_lsb(m, 7)
    blob = _header(version, rows, cols, 0, 0) + payload
    n = rows * cols
    return MantissaEncode(
        blob=blob,
        version=version,
        rows=rows,
        cols=cols,
        n_words=n,
        strategy="ALL_RAW",
        tile_shape=None,
        n_tiles=0,
        complete_bytes=len(blob),
        mantissa_bpw=(8.0 * len(blob) / n) if n else 0.0,
        needs_exp=False,
        pred_counts={"RAW": 1},
        sha256=_sha_u8(m),
    )


def _sha_u8(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    return hashlib.sha256(a.tobytes()).hexdigest()


def encode_mantissa(
    mant: np.ndarray,
    *,
    version: int = 3,
    exp: np.ndarray | None = None,
    force_shape: tuple[int, int] | None = None,
    force_pred: int | None = None,
    force_trav: int | None = None,
    allow_all_raw: bool = True,
) -> MantissaEncode:
    if version not in VERSIONS:
        raise ValueError(f"version must be 1..3, got {version}")
    m = _as_2d(np.ascontiguousarray(mant, dtype=np.uint8) & np.uint8(0x7F))
    rows, cols = int(m.shape[0]), int(m.shape[1])
    e = None
    if exp is not None:
        e = _as_2d(np.ascontiguousarray(exp, dtype=np.uint8))
        if e.shape != m.shape:
            raise ValueError("exp shape must match mantissa")
    n = rows * cols
    preds = PREDS[version]
    travs = TRAVS[version]
    shapes = SHAPES[version] if force_shape is None else (force_shape,)
    allow_raw_plane = allow_all_raw and force_pred in (None, PRED_RAW)

    best: MantissaEncode | None = None
    if allow_raw_plane and force_shape is None:
        best = encode_all_raw(m, version=version)

    for th, tw in shapes:
        tiles = iter_tiles(rows, cols, th, tw) if n else []
        records = bytearray()
        pred_counts: dict[str, int] = {}
        trav_counts: dict[str, int] = {}
        needs_exp = False
        for r0, c0, h, w in tiles:
            sl_m = m[r0 : r0 + h, c0 : c0 + w]
            sl_e = None if e is None else e[r0 : r0 + h, c0 : c0 + w]
            cand = _score_tile(
                sl_m,
                sl_e,
                preds,
                travs,
                force_pred=force_pred,
                force_trav=force_trav,
            )
            if cand.pred == PRED_EXP_LEFT:
                needs_exp = True
            pname = PRED_NAMES[cand.pred]
            tname = TRAV_NAMES[cand.trav] if cand.pred != PRED_RAW else "raw"
            pred_counts[pname] = pred_counts.get(pname, 0) + 1
            trav_counts[tname] = trav_counts.get(tname, 0) + 1
            records.extend(_tile_record(r0, c0, h, w, cand))
        flags = FLAG_NEEDS_EXP if needs_exp else 0
        blob = _header(version, rows, cols, len(tiles), flags) + bytes(records)
        enc = MantissaEncode(
            blob=blob,
            version=version,
            rows=rows,
            cols=cols,
            n_words=n,
            strategy="TILED",
            tile_shape=(th, tw),
            n_tiles=len(tiles),
            complete_bytes=len(blob),
            mantissa_bpw=(8.0 * len(blob) / n) if n else 0.0,
            needs_exp=needs_exp,
            pred_counts=pred_counts,
            trav_counts=trav_counts,
            shape_used=(th, tw),
            sha256=_sha_u8(m),
        )
        if best is None:
            best = enc
        else:
            raw_penalty = 0 if enc.strategy == "ALL_RAW" else 1
            best_penalty = 0 if best.strategy == "ALL_RAW" else 1
            if (enc.complete_bytes, raw_penalty) < (best.complete_bytes, best_penalty):
                best = enc

    if best is None:
        best = encode_all_raw(m, version=version)
    exp_for_decode = e
    if best.needs_exp and exp_for_decode is None:
        exp_for_decode = np.zeros_like(m)
    rec = decode_mantissa(best.blob, exp=exp_for_decode)
    if not np.array_equal(rec, m):
        raise CodecError("matrix mantissa round-trip failed")
    return best


def decode_mantissa(blob: bytes, *, exp: np.ndarray | None = None) -> np.ndarray:
    if len(blob) < _HDR.size:
        raise CodecError("truncated matrix-mantissa header")
    magic, version, rows, cols, n_tiles, flags = _HDR.unpack_from(blob, 0)
    if magic != MAGIC:
        raise CodecError(f"bad magic {magic!r}")
    if version not in VERSIONS:
        raise CodecError(f"unsupported version {version}")
    if rows < 0 or cols < 0:
        raise CodecError("negative dimensions")
    off = _HDR.size
    out = np.zeros((rows, cols), dtype=np.uint8)
    e = None
    if exp is not None:
        e = _as_2d(np.ascontiguousarray(exp, dtype=np.uint8))
        if e.shape != (rows, cols):
            raise CodecError("exp shape must match header rows/cols")
    if flags & FLAG_NEEDS_EXP and e is None:
        raise CodecError("blob requires exponent plane for EXP_LEFT")
    if n_tiles == 0:
        n = rows * cols
        need = ceil_bytes(n * 7)
        if len(blob) < off + need:
            raise CodecError("truncated ALL_RAW payload")
        flat = unpack_lsb(blob[off : off + need], n, 7)
        return flat.reshape(rows, cols) if n else out
    for _ in range(int(n_tiles)):
        if len(blob) < off + _TILE.size:
            raise CodecError("truncated tile directory")
        row0, col0, h, w, pred, trav_res, plen = _TILE.unpack_from(blob, off)
        off += _TILE.size
        if plen > len(blob) - off:
            raise CodecError("tile payload overruns blob")
        payload = blob[off : off + plen]
        off += plen
        if h < 1 or w < 1 or row0 < 0 or col0 < 0:
            raise CodecError("invalid tile geometry")
        if row0 + h > rows or col0 + w > cols:
            raise CodecError("tile exceeds matrix")
        trav, kind = _unpack_trav_res(trav_res)
        sl_e = None if e is None else e[row0 : row0 + h, col0 : col0 + w]
        if pred == PRED_RAW:
            stored = unpack_lsb(payload, h * w, 7).reshape(h, w)
            rec = stored
        else:
            stored = decode_residuals(payload, h * w, kind).reshape(h, w)
            rec = _restore_tile(stored, int(pred), trav, sl_e)
        out[row0 : row0 + h, col0 : col0 + w] = rec
    return out


def sha256_words(words: np.ndarray) -> str:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    return hashlib.sha256(w.astype("<u2", copy=False).tobytes()).hexdigest()


@dataclass
class BundleEncode:
    blob: bytes
    mantissa: MantissaEncode
    complete_bytes: int
    sha256: str


def encode_bf16_matrix(
    words: np.ndarray,
    *,
    version: int = 3,
    force_shape: tuple[int, int] | None = None,
    force_pred: int | None = None,
    force_trav: int | None = None,
) -> BundleEncode:
    arr = _as_2d(np.ascontiguousarray(words, dtype=np.uint16))
    sign, exp, mant = split_components(arr)
    man = encode_mantissa(
        mant,
        version=version,
        exp=exp,
        force_shape=force_shape,
        force_pred=force_pred,
        force_trav=force_trav,
    )
    sign_b = pack_lsb(sign, 1)
    exp_b = np.ascontiguousarray(exp, dtype=np.uint8).tobytes()
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    header = BUNDLE_MAGIC + struct.pack(
        "<HHIII", rows, cols, len(sign_b), len(exp_b), len(man.blob)
    )
    blob = header + sign_b + exp_b + man.blob
    rec = decode_bf16_matrix(blob)
    if not np.array_equal(rec, arr):
        raise CodecError("BF16 matrix round-trip failed")
    return BundleEncode(
        blob=blob,
        mantissa=man,
        complete_bytes=len(blob),
        sha256=sha256_words(arr),
    )


def decode_bf16_matrix(blob: bytes) -> np.ndarray:
    if len(blob) < 4 + 16:
        raise CodecError("truncated BF16 bundle")
    if blob[:4] != BUNDLE_MAGIC:
        raise CodecError(f"bad bundle magic {blob[:4]!r}")
    rows, cols, slen, elen, mlen = struct.unpack_from("<HHIII", blob, 4)
    off = 20
    end = off + slen + elen + mlen
    if len(blob) < end:
        raise CodecError("truncated BF16 bundle body")
    sign = unpack_lsb(blob[off : off + slen], rows * cols, 1).reshape(rows, cols)
    off += slen
    exp = np.frombuffer(blob[off : off + elen], dtype=np.uint8, count=rows * cols).reshape(
        rows, cols
    )
    off += elen
    mant = decode_mantissa(blob[off : off + mlen], exp=exp)
    return join_components(sign, exp, mant)
