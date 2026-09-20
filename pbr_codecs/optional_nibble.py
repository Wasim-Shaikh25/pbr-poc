"""Optional 4-bit side info: store a nibble only when it beats raw / PBR-E.

Unlike PBR-4, c4 is not mandatory. A tile that cannot save more than 4 bits
per admitted nibble falls back to a cheaper codec (palette, raw, or defer).
"""

from __future__ import annotations

import struct

import numpy as np

from pbr_codecs.pbr4 import c4_bytes_for, pack_nibbles, unpack_nibbles
from pbr_codecs.positions import pack_bitmap, unpack_bitmap
from pbr_codecs.value_dictionary import ValueDictCodec
from pbr_core.types import TILE_HEADER_BYTES, EncodedBlock

DISCLAIMER = (
    "Optional nibble: 4-bit side info only when residual savings exceed 4 bits. "
    "Not classic PBR-4 (mandatory c4). Lossless BF16. Not a 1–2 GB / 8 GB claim."
)

KIND_RAW = 0
KIND_PALETTE = 1
KIND_HIGH12 = 2
KIND_MODE_HIGH = 3  # nibble only on hits of the block-mode high-12 bits
KIND_CONST = 4


def _raw_payload(words: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(words, dtype="<u2").ravel()
    return bytes([KIND_RAW]) + flat.tobytes()


def _palette_payload(words: np.ndarray) -> bytes | None:
    enc = ValueDictCodec().encode(words)
    if enc is None:
        return None
    return bytes([KIND_PALETTE]) + enc.payload


def _high12_payload(words: np.ndarray) -> bytes | None:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    high = flat >> np.uint16(4)
    low = (flat & np.uint16(0xF)).astype(np.uint8)
    uniq = np.unique(high)
    if uniq.size < 1 or uniq.size > 16:
        return None
    palette = np.sort(uniq).astype(np.uint16)
    index = {int(v): i for i, v in enumerate(palette)}
    ids = np.array([index[int(v)] for v in high], dtype=np.uint8)
    nbits = 4 if palette.size <= 16 else 8
    # 4-bit ids packed + 4-bit residual nibbles.
    packed_ids = pack_nibbles(ids)
    packed_low = pack_nibbles(low)
    return (
        bytes([KIND_HIGH12, int(palette.size)])
        + palette.tobytes()
        + packed_ids
        + packed_low
    )


def _mode_high_payload(words: np.ndarray) -> bytes | None:
    """Store 4-bit low nibble only where high-12 equals the block mode."""
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    n = int(flat.size)
    high = flat >> np.uint16(4)
    low = (flat & np.uint16(0xF)).astype(np.uint8)
    uniq, counts = np.unique(high, return_counts=True)
    mode = int(uniq[int(np.argmax(counts))])
    hit = high == np.uint16(mode)
    n_hit = int(hit.sum())
    # Optional nibble: only pay 4 bits on hits. Misses are raw uint16.
    # Admit only if that is cheaper than storing every nibble (mandatory c4)
    # and cheaper than raw.
    miss_vals = flat[~hit]
    payload = (
        bytes([KIND_MODE_HIGH])
        + struct.pack("<IH", n, mode)
        + pack_bitmap(hit)
        + pack_nibbles(low[hit])
        + miss_vals.astype("<u2").tobytes()
    )
    raw = 2 * n
    mandatory_c4 = c4_bytes_for(n) + 2 + (n - n_hit) * 2  # c4 everywhere + miss residual
    if len(payload) - 1 >= raw:
        return None
    if n_hit == 0:
        return None
    # Explicit: optional must beat mandatory-c4 on this tile.
    if len(payload) - 1 >= mandatory_c4:
        return None
    saved_vs_raw = raw - (len(payload) - 1)
    # Each admitted nibble must save > 4 bits vs paying raw on that position.
    # Equivalent check: total savings > 4 * n_hit bits.
    if saved_vs_raw * 8 <= 4 * n_hit:
        return None
    return payload


def encode_tile(words: np.ndarray) -> tuple[str, bytes]:
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    uniq = np.unique(flat)
    cands: list[tuple[str, bytes]] = [("raw", _raw_payload(words))]
    if uniq.size == 1:
        cands.append(("const", bytes([KIND_CONST]) + struct.pack("<H", int(uniq[0]))))
    pal = _palette_payload(words)
    if pal is not None:
        cands.append(("palette", pal))
    h12 = _high12_payload(words)
    if h12 is not None:
        cands.append(("high12", h12))
    mh = _mode_high_payload(words)
    if mh is not None:
        cands.append(("mode_high_optional_c4", mh))
    name, payload = min(cands, key=lambda kv: len(kv[1]))
    return name, payload


def decode_tile(payload: bytes, rows: int, cols: int) -> np.ndarray:
    kind = payload[0]
    n = rows * cols
    body = payload[1:]
    if kind == KIND_CONST:
        (value,) = struct.unpack("<H", body[:2])
        return np.full((rows, cols), value, dtype=np.uint16)
    if kind == KIND_RAW:
        arr = np.frombuffer(body, dtype="<u2", count=n)
        return np.array(arr, dtype=np.uint16, copy=True).reshape(rows, cols)
    if kind == KIND_PALETTE:
        block = EncodedBlock(mode_id=2, mode_name="value_dict", payload=body, rows=rows, cols=cols)
        return ValueDictCodec().decode(block)
    if kind == KIND_HIGH12:
        k = payload[1]
        palette = np.frombuffer(payload, dtype="<u2", count=k, offset=2).astype(np.uint16)
        pos = 2 + 2 * k
        id_bytes = c4_bytes_for(n)
        ids = unpack_nibbles(payload[pos : pos + id_bytes], n)
        pos += id_bytes
        low = unpack_nibbles(payload[pos : pos + id_bytes], n)
        high = palette[ids].astype(np.uint16)
        words = (high << np.uint16(4)) | low.astype(np.uint16)
        return words.reshape(rows, cols)
    if kind == KIND_MODE_HIGH:
        n_decl, mode = struct.unpack_from("<IH", body, 0)
        if n_decl != n:
            raise ValueError("optional nibble n mismatch")
        hit, offset = unpack_bitmap(body, n_decl, 6)
        hit = hit.astype(bool)
        n_hit = int(hit.sum())
        nib_b = c4_bytes_for(n_hit)
        low_hits = unpack_nibbles(body[offset : offset + nib_b], n_hit)
        offset += nib_b
        n_miss = n - n_hit
        miss = np.frombuffer(body, dtype="<u2", count=n_miss, offset=offset).astype(np.uint16)
        out = np.empty(n, dtype=np.uint16)
        out[hit] = (np.uint16(mode) << np.uint16(4)) | low_hits.astype(np.uint16)
        out[~hit] = miss
        return out.reshape(rows, cols)
    raise ValueError(f"unknown optional-nibble kind {kind}")


def encode_matrix(words_2d: np.ndarray, tile: int = 16) -> tuple[bytes, dict]:
    mat = np.ascontiguousarray(words_2d, dtype=np.uint16)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    th = max(1, min(tile, rows))
    tw = max(1, min(tile, cols))
    parts: list[bytes] = []
    usage: dict[str, int] = {}
    nibble_tiles = 0
    n_tiles = 0
    for r0 in range(0, rows, th):
        for c0 in range(0, cols, tw):
            tile_w = mat[r0 : r0 + th, c0 : c0 + tw]
            name, payload = encode_tile(tile_w)
            usage[name] = usage.get(name, 0) + 1
            if name in ("high12", "mode_high_optional_c4"):
                nibble_tiles += 1
            n_tiles += 1
            hdr = struct.pack("<HHII", tile_w.shape[0], tile_w.shape[1], r0, c0)
            parts.append(struct.pack("<I", len(payload)) + hdr + payload)
    blob = struct.pack("<HHHH", rows, cols, th, tw) + b"".join(parts)
    stats = {
        "n_tiles": n_tiles,
        "nibble_tiles": nibble_tiles,
        "usage": usage,
        "payload_bytes": len(blob),
        "complete_bytes": TILE_HEADER_BYTES + len(blob),
    }
    return blob, stats


def decode_matrix(blob: bytes) -> np.ndarray:
    rows, cols, _th, _tw = struct.unpack_from("<HHHH", blob, 0)
    out = np.empty((rows, cols), dtype=np.uint16)
    offset = 8
    while offset < len(blob):
        (plen,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        tr, tc, r0, c0 = struct.unpack_from("<HHII", blob, offset)
        offset += 12
        payload = blob[offset : offset + plen]
        offset += plen
        tile = decode_tile(payload, tr, tc)
        out[r0 : r0 + tr, c0 : c0 + tc] = tile
    return out
