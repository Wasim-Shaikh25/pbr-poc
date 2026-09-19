"""Exact XOR-residual encodings selected by complete byte cost, not MSE."""

from __future__ import annotations

import struct
import numpy as np

from pbr_codecs.positions import (
    pack_bitmap,
    pack_positions,
    unpack_bitmap,
    unpack_positions,
)
from pbr_core.bitio import bits_needed, pack_ids, unpack_ids
from pbr_core.hashing import words_to_bytes
from pbr_core.types import (
    RES_CONSTANT,
    RES_DEFAULT_BITMAP,
    RES_DEFAULT_LIST,
    RES_DICT,
    RES_NAMES,
    RES_RAW,
)

MAX_DICT = 4


def residual_candidates(residuals: np.ndarray) -> list[tuple[str, bytes]]:
    """Return (residual_mode_name, payload) candidates including raw."""
    flat = np.ascontiguousarray(residuals, dtype=np.uint16).ravel()
    n = int(flat.size)
    out: list[tuple[str, bytes]] = []
    out.append((RES_NAMES[RES_RAW], bytes([RES_RAW]) + words_to_bytes(flat)))
    if n == 0:
        return out

    unique, counts = np.unique(flat, return_counts=True)
    if unique.size == 1:
        out.append((RES_NAMES[RES_CONSTANT], bytes([RES_CONSTANT]) + struct.pack("<H", int(unique[0]))))
        return out

    default = int(unique[int(np.argmax(counts))])
    mask = flat != np.uint16(default)
    positions = np.nonzero(mask)[0].astype(np.uint32)
    values = flat[mask]
    value_bytes = words_to_bytes(values)

    list_payload = bytes([RES_DEFAULT_LIST]) + struct.pack("<H", default) + pack_positions(positions, n) + value_bytes
    out.append((RES_NAMES[RES_DEFAULT_LIST], list_payload))

    bitmap_payload = bytes([RES_DEFAULT_BITMAP]) + struct.pack("<H", default) + pack_bitmap(mask) + value_bytes
    out.append((RES_NAMES[RES_DEFAULT_BITMAP], bitmap_payload))

    dict_payload = _try_dict(flat, unique, counts)
    if dict_payload is not None:
        out.append((RES_NAMES[RES_DICT], dict_payload))

    # Prefer cheaper encodings; keep raw as a correctness fallback.
    out.sort(key=lambda item: len(item[1]))
    return out


def best_residual(residuals: np.ndarray) -> tuple[str, bytes]:
    cands = residual_candidates(residuals)
    return min(cands, key=lambda item: len(item[1]))


def decode_residuals(payload: bytes, rows: int, cols: int) -> np.ndarray:
    if not payload:
        return np.zeros((rows, cols), dtype=np.uint16)
    kind = payload[0]
    n = rows * cols
    body = payload[1:]
    if kind == RES_RAW:
        arr = np.frombuffer(body, dtype="<u2", count=n)
        return np.array(arr, dtype=np.uint16, copy=True).reshape(rows, cols)
    if kind == RES_CONSTANT:
        (value,) = struct.unpack("<H", body[:2])
        return np.full((rows, cols), value, dtype=np.uint16)
    if kind == RES_DEFAULT_LIST:
        (default,) = struct.unpack_from("<H", body, 0)
        positions, offset = unpack_positions(body, n, 2)
        values = np.frombuffer(body, dtype="<u2", count=len(positions), offset=offset)
        out = np.full(n, default, dtype=np.uint16)
        if len(positions):
            out[positions] = np.array(values, dtype=np.uint16, copy=True)
        return out.reshape(rows, cols)
    if kind == RES_DEFAULT_BITMAP:
        (default,) = struct.unpack_from("<H", body, 0)
        mask, offset = unpack_bitmap(body, n, 2)
        values = np.frombuffer(body, dtype="<u2", count=int(mask.sum()), offset=offset)
        out = np.full(n, default, dtype=np.uint16)
        if values.size:
            out[mask] = np.array(values, dtype=np.uint16, copy=True)
        return out.reshape(rows, cols)
    if kind == RES_DICT:
        k, nbits = struct.unpack_from("<BB", body, 0)
        palette = np.frombuffer(body, dtype="<u2", count=k, offset=2).astype(np.uint16)
        ids = unpack_ids(body[2 + 2 * k :], n, nbits)
        return palette[ids].reshape(rows, cols)
    raise ValueError(f"Unknown residual kind {kind}")


def _try_dict(flat: np.ndarray, unique: np.ndarray, counts: np.ndarray) -> bytes | None:
    k = int(unique.size)
    if k < 2 or k > MAX_DICT:
        return None
    nbits = bits_needed(k)
    # Stable palette: most common first, then numeric value.
    order = sorted(range(k), key=lambda i: (-int(counts[i]), int(unique[i])))
    palette = unique[order]
    index = {int(v): i for i, v in enumerate(palette)}
    ids = np.array([index[int(v)] for v in flat], dtype=np.uint16)
    packed = pack_ids(ids, nbits)
    return bytes([RES_DICT]) + struct.pack("<BB", k, nbits) + words_to_bytes(palette) + packed


def default_residual_stats(residuals: np.ndarray) -> dict:
    flat = np.ascontiguousarray(residuals, dtype=np.uint16).ravel()
    if flat.size == 0:
        return {"n": 0, "zero_rate": 1.0, "unique": 0, "top4_coverage": 1.0, "mean_popcount": 0.0}
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    top4 = counts[order][:4].sum() / flat.size
    pop = np.bitwise_count(flat.astype(np.uint16)) if hasattr(np, "bitwise_count") else _popcount(flat)
    return {
        "n": int(flat.size),
        "zero_rate": float(np.mean(flat == 0)),
        "unique": int(unique.size),
        "top4_coverage": float(top4),
        "mean_popcount": float(np.mean(pop)),
        "default": int(unique[int(np.argmax(counts))]),
    }


def _popcount(arr: np.ndarray) -> np.ndarray:
    # Portable 16-bit popcount without depending on numpy 2.0 bitwise_count.
    x = arr.astype(np.uint32)
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    return (((x + (x >> 4)) & 0x0F0F) * 0x0101) >> 8
