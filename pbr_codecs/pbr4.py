"""PBR-4: structured 4-bit side info + node formulas, exact XOR residual.

Formal reconstruction (bit-exact):

    W[i] = F(node(i), c4[i]) XOR R[i]

Complete bytes = |S| + 4N/8 + |R| + metadata. Positions are already known;
c4 is stored 4-bit side information, not data "hidden in coordinates".

Standalone container magic ``PBR4``. Not on STAGE1A_CODECS.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from pbr_codecs.positions import bitmap_bytes, list_bytes, pack_bitmap
from pbr_codecs.residual import RES_CONSTANT, RES_DEFAULT_BITMAP, RES_RAW, decode_residuals
from pbr_core.hashing import words_to_bytes
from pbr_core.metrics import bits_per_weight
from pbr_core.tiles import as_2d, choose_tile_hw
from pbr_core.types import MODE_PBR4

MAGIC = b"PBR4"
VERSION = 1
CONTAINER_PREFIX = 10  # magic(4) + version u16 + header_len u32

KIND_CONST = 0
KIND_PALETTE = 1
KIND_AFFINE = 2
KIND_NIBBLE = 3
KIND_SE_NIBBLE = 4
KIND_TEMPLATES = 5
KIND_INHERIT = 6

KIND_NAMES = {
    KIND_CONST: "const_proto",
    KIND_PALETTE: "palette16",
    KIND_AFFINE: "affine_rc",
    KIND_NIBBLE: "nibble_insert",
    KIND_SE_NIBBLE: "se_nibble",
    KIND_TEMPLATES: "templates16",
    KIND_INHERIT: "inherit_root",
}

R_ZLIB = 5
R_PACK3 = 6

FLAG_ADAPTIVE = 1 << 0
MAX_K = 16
# root_p0, root_se, flags, block_h, block_w, rows, cols
_PREFIX = struct.Struct("<HHBHHII")  # 17 bytes
_NODE = struct.Struct("<BHHHHB")  # kind, r0, c0, h, w, slen  (10 bytes)

DISCLAIMER = (
    "PBR-4 structured nibble + node formulas. W[i] = F(node(i), c4[i]) XOR R[i]. "
    "Complete bytes = |S| + 4N/8 + |R| + metadata. Bit-exact uint16 BF16 only. "
    "Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW unless measured total "
    "complete BPW is ≤4. c4 is stored 4-bit side info; coordinates are already known."
)


def c4_bytes_for(n: int) -> int:
    """Formal |c4| = ceil(4N / 8)."""
    return (int(n) + 1) // 2


def pack_nibbles(c4: np.ndarray) -> bytes:
    v = np.ascontiguousarray(c4, dtype=np.uint8).ravel() & np.uint8(0x0F)
    n = int(v.size)
    out = np.zeros((n + 1) // 2, dtype=np.uint8)
    even = n - (n % 2)
    if even:
        out[: even // 2] = v[0:even:2] | (v[1:even:2] << np.uint8(4))
    if n % 2:
        out[-1] = v[-1]
    return out.tobytes()


def unpack_nibbles(data: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(n, dtype=np.uint8)
    even = n - (n % 2)
    m = even // 2
    if m:
        out[0:even:2] = b[:m] & np.uint8(0x0F)
        out[1:even:2] = (b[:m] >> np.uint8(4)) & np.uint8(0x0F)
    if n % 2:
        out[-1] = b[m] & np.uint8(0x0F)
    return out


def pack_3bit(vals: np.ndarray) -> bytes:
    v = np.ascontiguousarray(vals, dtype=np.uint8).ravel() & np.uint8(0x07)
    n = int(v.size)
    pad = (8 - n % 8) % 8
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.uint8)])
    g = v.reshape(-1, 8).astype(np.uint32)
    acc = (
        g[:, 0]
        | (g[:, 1] << 3)
        | (g[:, 2] << 6)
        | (g[:, 3] << 9)
        | (g[:, 4] << 12)
        | (g[:, 5] << 15)
        | (g[:, 6] << 18)
        | (g[:, 7] << 21)
    )
    out = np.empty(acc.size * 3, dtype=np.uint8)
    out[0::3] = acc & 0xFF
    out[1::3] = (acc >> 8) & 0xFF
    out[2::3] = (acc >> 16) & 0xFF
    return out.tobytes()


def unpack_3bit(data: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8)
    ng = (n + 7) // 8
    acc = np.zeros(ng, dtype=np.uint32)
    raw = np.zeros(ng * 3, dtype=np.uint8)
    take = min(raw.size, b.size)
    raw[:take] = b[:take]
    acc[:] = raw[0::3].astype(np.uint32) | (raw[1::3].astype(np.uint32) << 8) | (
        raw[2::3].astype(np.uint32) << 16
    )
    g = np.empty((ng, 8), dtype=np.uint8)
    for i in range(8):
        g[:, i] = (acc >> (3 * i)) & 7
    return g.ravel()[:n]


def _popcount16(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.uint32)
    v = v - ((v >> 1) & 0x5555)
    v = (v & 0x3333) + ((v >> 2) & 0x3333)
    return (((v + (v >> 4)) & 0x0F0F) * 0x0101) >> 8


def _mode_u16(flat: np.ndarray) -> int:
    uniq, counts = np.unique(flat, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _assign_palette(flat: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Exact match when possible; otherwise min Hamming distance to prototypes."""
    src = np.ascontiguousarray(flat, dtype=np.uint16).ravel()
    pal = np.ascontiguousarray(pal, dtype=np.uint16).ravel()
    idx = np.zeros(src.size, dtype=np.uint8)
    matched = np.zeros(src.size, dtype=bool)
    for i, v in enumerate(pal.tolist()):
        hit = src == np.uint16(v)
        idx[hit] = i
        matched |= hit
    if not bool(matched.all()):
        mv = src[~matched]
        xor = np.bitwise_xor(mv[:, None], pal[None, :])
        idx[~matched] = _popcount16(xor).argmin(axis=1).astype(np.uint8)
    return idx


def _mesh(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.arange(h, dtype=np.uint32)[:, None]
    c = np.arange(w, dtype=np.uint32)[None, :]
    return r, c


def _planar(tile: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    e = int(tile[0, 0])
    a = (int(tile[1, 0]) - e) & 0xFFFF if tile.shape[0] > 1 else 0
    b = (int(tile[0, 1]) - e) & 0xFFFF if tile.shape[1] > 1 else 0
    r, c = _mesh(int(tile.shape[0]), int(tile.shape[1]))
    f = (np.uint32(e) + np.uint32(a) * r + np.uint32(b) * c) & np.uint32(0xFFFF)
    return f.astype(np.uint16), a, b, e


def estimate_r_bytes(r: np.ndarray) -> int:
    n = int(r.size)
    n_miss = int(np.count_nonzero(r))
    if n_miss == 0:
        return 3
    raw = 1 + 2 * n
    list_b = 1 + 2 + list_bytes(n_miss, n) + 2 * n_miss
    bit_b = 1 + 2 + bitmap_bytes(n) + 2 * n_miss
    r3 = 1 + (3 * n + 7) // 8 if n and int(r.max()) < 8 else raw
    return min(raw, list_b, bit_b, r3)


def pack_residual(r: np.ndarray) -> bytes:
    """Sparse XOR residual (default 0), 3-bit pack, zlib, or raw — shortest wins."""
    flat = np.ascontiguousarray(r, dtype=np.uint16).ravel()
    n = int(flat.size)
    n_miss = int(np.count_nonzero(flat))
    if n == 0 or n_miss == 0:
        return bytes([RES_CONSTANT]) + struct.pack("<H", 0)
    raw = bytes([RES_RAW]) + words_to_bytes(flat)
    cands = [raw]
    if n_miss < n:
        mask = flat != 0
        bit = (
            bytes([RES_DEFAULT_BITMAP])
            + struct.pack("<H", 0)
            + pack_bitmap(mask)
            + words_to_bytes(flat[mask])
        )
        cands.append(bit)
    if n and int(flat.max()) < 8:
        cands.append(bytes([R_PACK3]) + pack_3bit(flat))
    if n < 8192:
        z = zlib.compress(words_to_bytes(flat), 3)
        if len(z) + 1 < min(len(c) for c in cands):
            cands.append(bytes([R_ZLIB]) + z)
    return min(cands, key=len)


def unpack_residual(payload: bytes, rows: int, cols: int) -> np.ndarray:
    if not payload:
        return np.zeros((rows, cols), dtype=np.uint16)
    kind = payload[0]
    n = rows * cols
    if kind == R_ZLIB:
        raw = zlib.decompress(payload[1:])
        arr = np.frombuffer(raw, dtype="<u2", count=n)
        return np.array(arr, dtype=np.uint16, copy=True).reshape(rows, cols)
    if kind == R_PACK3:
        vals = unpack_3bit(payload[1:], n)
        return vals.astype(np.uint16).reshape(rows, cols)
    return decode_residuals(payload, rows, cols)


@dataclass
class NodeEnc:
    kind: int
    s: bytes
    f: np.ndarray
    c4: np.ndarray
    row0: int
    col0: int
    rows: int
    cols: int

    def cost(self, tile: np.ndarray) -> int:
        r = tile ^ self.f
        n = int(tile.size)
        return len(self.s) + c4_bytes_for(n) + estimate_r_bytes(r) + 10


def _cand(kind: int, s: bytes, f: np.ndarray, c4: np.ndarray, tile: np.ndarray, r0: int, c0: int) -> NodeEnc:
    return NodeEnc(
        kind=kind,
        s=s,
        f=np.ascontiguousarray(f, dtype=np.uint16),
        c4=np.ascontiguousarray(c4, dtype=np.uint8),
        row0=r0,
        col0=c0,
        rows=int(tile.shape[0]),
        cols=int(tile.shape[1]),
    )


def encode_block(tile: np.ndarray, row0: int, col0: int, root_p0: int) -> NodeEnc:
    """Compete cheap F families on one block; pick min formal |S|+4n/8+|R|."""
    tile = np.ascontiguousarray(tile, dtype=np.uint16)
    h, w = int(tile.shape[0]), int(tile.shape[1])
    flat = tile.ravel()
    uniq, counts = np.unique(flat, return_counts=True)
    mode = int(uniq[int(np.argmax(counts))])
    zeros = np.zeros((h, w), dtype=np.uint8)
    cands: list[NodeEnc] = []

    f_const = np.full((h, w), mode, dtype=np.uint16)
    cands.append(_cand(KIND_CONST, struct.pack("<H", mode), f_const, zeros, tile, row0, col0))
    if root_p0 != mode:
        f_inh = np.full((h, w), root_p0, dtype=np.uint16)
        cands.append(_cand(KIND_INHERIT, b"", f_inh, zeros, tile, row0, col0))

    k = min(MAX_K, int(uniq.size))
    order = np.argsort(-counts, kind="stable")
    pal = uniq[order][:k].astype(np.uint16)
    idx = _assign_palette(flat, pal)
    f_pal = pal[idx].reshape(h, w)
    c4_pal = idx.reshape(h, w)
    cands.append(_cand(KIND_PALETTE, bytes([k]) + words_to_bytes(pal), f_pal, c4_pal, tile, row0, col0))

    for shift in (0, 4, 8, 12):
        mask = np.uint16(0xF << shift)
        base = tile & ~mask
        p = _mode_u16(base.ravel())
        c4n = ((tile >> np.uint16(shift)) & np.uint16(0xF)).astype(np.uint8)
        f_n = (np.uint16(p) & ~mask) | (c4n.astype(np.uint16) << np.uint16(shift))
        cands.append(
            _cand(KIND_NIBBLE, struct.pack("<HB", p, shift), f_n, c4n, tile, row0, col0)
        )

    se = tile >> np.uint16(7)
    se_mode = _mode_u16(se.ravel())
    c4_se = ((tile >> np.uint16(3)) & np.uint16(0xF)).astype(np.uint8)
    f_se = (np.uint16(se_mode) << np.uint16(7)) | (c4_se.astype(np.uint16) << np.uint16(3))
    cands.append(_cand(KIND_SE_NIBBLE, struct.pack("<H", se_mode), f_se, c4_se, tile, row0, col0))

    lin, a, b, e = _planar(tile)
    cands.append(_cand(KIND_AFFINE, struct.pack("<HHHH", a, b, 0, e), lin, zeros, tile, row0, col0))
    delta = (tile.astype(np.uint32) - lin.astype(np.uint32)) & np.uint32(0xFFFF)
    c4_a = (delta & np.uint32(0xF)).astype(np.uint8)
    f_a = ((lin.astype(np.uint32) + c4_a.astype(np.uint32)) & np.uint32(0xFFFF)).astype(np.uint16)
    cands.append(_cand(KIND_AFFINE, struct.pack("<HHHH", a, b, 1, e), f_a, c4_a, tile, row0, col0))

    xor_l = tile ^ lin
    u2, c2 = np.unique(xor_l, return_counts=True)
    k2 = min(MAX_K, int(u2.size))
    pal2 = u2[np.argsort(-c2, kind="stable")][:k2].astype(np.uint16)
    idx2 = _assign_palette(xor_l.ravel(), pal2)
    f_t = lin ^ pal2[idx2].reshape(h, w)
    c4_t = idx2.reshape(h, w)
    s_t = struct.pack("<HHH", a, b, e) + bytes([k2]) + words_to_bytes(pal2)
    cands.append(_cand(KIND_TEMPLATES, s_t, f_t, c4_t, tile, row0, col0))

    return min(cands, key=lambda nd: nd.cost(tile))


def apply_f(kind: int, s: bytes, c4: np.ndarray, rows: int, cols: int, root_p0: int) -> np.ndarray:
    c4 = np.ascontiguousarray(c4, dtype=np.uint8).reshape(rows, cols)
    r, c = _mesh(rows, cols)
    if kind == KIND_CONST:
        (p,) = struct.unpack("<H", s[:2])
        return np.full((rows, cols), p, dtype=np.uint16)
    if kind == KIND_INHERIT:
        return np.full((rows, cols), root_p0, dtype=np.uint16)
    if kind == KIND_PALETTE:
        k = s[0]
        pal = np.frombuffer(s, dtype="<u2", count=k, offset=1).astype(np.uint16)
        return pal[c4.ravel() % k].reshape(rows, cols)
    if kind == KIND_NIBBLE:
        p, shift = struct.unpack("<HB", s[:3])
        mask = np.uint16(0xF << shift)
        return (np.uint16(p) & ~mask) | (c4.astype(np.uint16) << np.uint16(shift))
    if kind == KIND_SE_NIBBLE:
        (se_mode,) = struct.unpack("<H", s[:2])
        return (np.uint16(se_mode) << np.uint16(7)) | (c4.astype(np.uint16) << np.uint16(3))
    if kind == KIND_AFFINE:
        a, b, d, e = struct.unpack("<HHHH", s[:8])
        base = (np.uint32(e) + np.uint32(a) * r + np.uint32(b) * c + np.uint32(d) * c4.astype(np.uint32)) & np.uint32(
            0xFFFF
        )
        return base.astype(np.uint16)
    if kind == KIND_TEMPLATES:
        a, b, e = struct.unpack_from("<HHH", s, 0)
        k = s[6]
        pal = np.frombuffer(s, dtype="<u2", count=k, offset=7).astype(np.uint16)
        lin = ((np.uint32(e) + np.uint32(a) * r + np.uint32(b) * c) & np.uint32(0xFFFF)).astype(np.uint16)
        return lin ^ pal[c4.ravel() % k].reshape(rows, cols)
    raise ValueError(f"unknown PBR-4 kind {kind}")


def _iter_grid(rows: int, cols: int, bh: int, bw: int) -> list[tuple[int, int, int, int]]:
    out = []
    for r0 in range(0, rows, bh):
        for c0 in range(0, cols, bw):
            h = min(bh, rows - r0)
            w = min(bw, cols - c0)
            out.append((r0, c0, h, w))
    return out


def _split_children(tile: np.ndarray, r0: int, c0: int, root_p0: int) -> list[NodeEnc] | None:
    h, w = int(tile.shape[0]), int(tile.shape[1])
    if h < 16 or w < 16:
        return None
    sh, sw = (h + 1) // 2, (w + 1) // 2
    if sh < 4 or sw < 4:
        return None
    kids = []
    for rr in range(0, h, sh):
        for cc in range(0, w, sw):
            sub = tile[rr : rr + sh, cc : cc + sw]
            kids.append(encode_block(sub, r0 + rr, c0 + cc, root_p0))
    return kids


@dataclass
class EncodedTensor:
    name: str
    shape: tuple[int, int]
    nodes: list[NodeEnc]
    c4: np.ndarray
    r_payload: bytes
    root_p0: int
    root_se: int
    block_h: int
    block_w: int
    adaptive: bool
    n_hit: int
    family_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def s_bytes(self) -> int:
        return 4 + sum(len(n.s) for n in self.nodes)  # root_p0 + root_se

    @property
    def c4_bytes(self) -> int:
        return c4_bytes_for(self.n_words)

    @property
    def r_bytes(self) -> int:
        return len(self.r_payload)

    @property
    def node_meta_bytes(self) -> int:
        # kind(1)+r0(2)+c0(2)+h(2)+w(2)+s_len(1) per node
        return 10 * len(self.nodes)

    def payload_bytes(self) -> bytes:
        flags = FLAG_ADAPTIVE if self.adaptive else 0
        body = bytearray(
            _PREFIX.pack(
                self.root_p0,
                self.root_se,
                flags,
                self.block_h,
                self.block_w,
                self.shape[0],
                self.shape[1],
            )
        )
        body.extend(struct.pack("<I", len(self.nodes)))
        for nd in self.nodes:
            if len(nd.s) > 255:
                raise ValueError("PBR-4 node generator exceeds 255 bytes")
            body.extend(_NODE.pack(nd.kind, nd.row0, nd.col0, nd.rows, nd.cols, len(nd.s)))
            body.extend(nd.s)
        c4b = pack_nibbles(self.c4)
        body.extend(struct.pack("<I", len(c4b)))
        body.extend(c4b)
        body.extend(struct.pack("<I", len(self.r_payload)))
        body.extend(self.r_payload)
        return bytes(body)

    def stats(self) -> dict:
        n = self.n_words
        payload = self.payload_bytes()
        node_s = sum(len(nd.s) for nd in self.nodes)
        s_b = node_s + 4
        c4_b = self.c4_bytes
        r_b = self.r_bytes
        meta = len(payload) - s_b - c4_b - r_b
        return {
            "name": self.name,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "n_words": n,
            "n_nodes": len(self.nodes),
            "block_h": self.block_h,
            "block_w": self.block_w,
            "adaptive": self.adaptive,
            "s_bytes": s_b,
            "c4_bytes": c4_b,
            "c4_bits": 4 * n,
            "r_bytes": r_b,
            "meta_bytes": meta,
            "payload_bytes": len(payload),
            "formal_bytes": len(payload),
            "formal_bpw": bits_per_weight(len(payload), n),
            "n_hit": self.n_hit,
            "hit_rate": self.n_hit / max(n, 1),
            "family_counts": dict(self.family_counts),
        }


def encode_tensor(
    words: np.ndarray,
    *,
    name: str = "tensor",
    block_h: int = 16,
    block_w: int = 16,
    adaptive: bool = False,
) -> EncodedTensor:
    arr = as_2d(words)
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    root_p0 = _mode_u16(arr.ravel()) if arr.size else 0
    se = (arr >> np.uint16(7)) if arr.size else np.zeros(0, dtype=np.uint16)
    root_se = _mode_u16(se.ravel()) if se.size else 0
    f = np.empty_like(arr)
    c4 = np.zeros((rows, cols), dtype=np.uint8)
    nodes: list[NodeEnc] = []
    family_counts: dict[str, int] = {}
    n_hit = 0

    for r0, c0, h, w in _iter_grid(rows, cols, block_h, block_w):
        tile = arr[r0 : r0 + h, c0 : c0 + w]
        parent = encode_block(tile, r0, c0, root_p0)
        chosen: list[NodeEnc] = [parent]
        if adaptive:
            kids = _split_children(tile, r0, c0, root_p0)
            if kids is not None:
                pcost = parent.cost(tile)
                kcost = sum(k.cost(tile[k.row0 - r0 : k.row0 - r0 + k.rows, k.col0 - c0 : k.col0 - c0 + k.cols]) for k in kids)
                if kcost < pcost:
                    chosen = kids
        for nd in chosen:
            f[nd.row0 : nd.row0 + nd.rows, nd.col0 : nd.col0 + nd.cols] = nd.f
            c4[nd.row0 : nd.row0 + nd.rows, nd.col0 : nd.col0 + nd.cols] = nd.c4
            n_hit += int(np.count_nonzero(nd.f == arr[nd.row0 : nd.row0 + nd.rows, nd.col0 : nd.col0 + nd.cols]))
            family_counts[KIND_NAMES[nd.kind]] = family_counts.get(KIND_NAMES[nd.kind], 0) + 1
            nodes.append(nd)

    residual = arr ^ f
    r_payload = pack_residual(residual)
    enc = EncodedTensor(
        name=name,
        shape=(rows, cols),
        nodes=nodes,
        c4=c4.ravel(),
        r_payload=r_payload,
        root_p0=root_p0,
        root_se=root_se,
        block_h=block_h,
        block_w=block_w,
        adaptive=adaptive,
        n_hit=n_hit,
        family_counts=family_counts,
    )
    return enc


def decode_payload(payload: bytes, *, root_override: tuple[int, int] | None = None) -> np.ndarray:
    if len(payload) < _PREFIX.size + 4:
        raise ValueError("PBR-4 tensor payload too short")
    root_p0, root_se, flags, bh, bw, rows, cols = _PREFIX.unpack_from(payload, 0)
    if root_override is not None:
        root_p0, root_se = root_override
    del flags, bh, bw, root_se
    offset = _PREFIX.size
    (n_nodes,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    nodes_meta = []
    for _ in range(n_nodes):
        kind, r0, c0, h, w, slen = _NODE.unpack_from(payload, offset)
        offset += _NODE.size
        s = payload[offset : offset + slen]
        offset += slen
        nodes_meta.append((kind, r0, c0, h, w, s))
    (c4_len,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    c4_bytes = payload[offset : offset + c4_len]
    offset += c4_len
    n = rows * cols
    c4_full = unpack_nibbles(c4_bytes, n).reshape(rows, cols)
    (r_len,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    r_payload = payload[offset : offset + r_len]
    if offset + r_len != len(payload):
        raise ValueError("PBR-4 payload trailing/truncated residual")
    residual = unpack_residual(r_payload, rows, cols)
    f = np.zeros((rows, cols), dtype=np.uint16)
    for kind, r0, c0, h, w, s in nodes_meta:
        c4b = c4_full[r0 : r0 + h, c0 : c0 + w]
        f[r0 : r0 + h, c0 : c0 + w] = apply_f(kind, s, c4b, h, w, root_p0)
    return f ^ residual


def decode_tensor(enc: EncodedTensor) -> np.ndarray:
    return decode_payload(enc.payload_bytes())


@dataclass
class PBR4Container:
    tensors: list[tuple[EncodedTensor, bytes]]
    extra: dict = field(default_factory=dict)

    def dumps(self) -> bytes:
        header = {
            "format": "PBR-4 structured nibble",
            "version": VERSION,
            "note": DISCLAIMER,
            "tensors": [
                {
                    "name": enc.name,
                    "shape": list(enc.shape),
                    "n_words": enc.n_words,
                    "block_h": enc.block_h,
                    "block_w": enc.block_w,
                    "adaptive": enc.adaptive,
                    "n_nodes": len(enc.nodes),
                    "payload_len": len(blob),
                    "s_bytes": enc.stats()["s_bytes"],
                    "c4_bytes": enc.c4_bytes,
                    "r_bytes": enc.r_bytes,
                    "n_hit": enc.n_hit,
                    "family_counts": enc.family_counts,
                }
                for enc, blob in self.tensors
            ],
            "extra": self.extra,
        }
        header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = bytearray()
        for _enc, blob in self.tensors:
            payload.extend(blob)
        return MAGIC + struct.pack("<HI", VERSION, len(header_bytes)) + header_bytes + bytes(payload)

    @classmethod
    def loads(cls, data: bytes) -> "PBR4Container":
        if data[:4] != MAGIC:
            raise ValueError(f"Bad PBR-4 magic: {data[:4]!r}")
        version, header_len = struct.unpack_from("<HI", data, 4)
        if version != VERSION:
            raise ValueError(f"Unsupported PBR4 version {version}")
        start = CONTAINER_PREFIX
        header = json.loads(data[start : start + header_len].decode("utf-8"))
        offset = start + header_len
        tensors: list[tuple[EncodedTensor, bytes]] = []
        for spec in header["tensors"]:
            plen = int(spec["payload_len"])
            blob = data[offset : offset + plen]
            if len(blob) != plen:
                raise ValueError("truncated PBR-4 tensor payload")
            offset += plen
            words = decode_payload(blob)
            # Rebuild a stub EncodedTensor for API symmetry (nodes empty; payload kept).
            stub = EncodedTensor(
                name=spec["name"],
                shape=(int(spec["shape"][0]), int(spec["shape"][1])),
                nodes=[],
                c4=np.zeros(0, dtype=np.uint8),
                r_payload=b"",
                root_p0=0,
                root_se=0,
                block_h=int(spec["block_h"]),
                block_w=int(spec["block_w"]),
                adaptive=bool(spec.get("adaptive")),
                n_hit=int(spec.get("n_hit", 0)),
                family_counts=dict(spec.get("family_counts") or {}),
            )
            stub._decoded = words  # type: ignore[attr-defined]
            stub._blob = blob  # type: ignore[attr-defined]
            tensors.append((stub, blob))
        if offset != len(data):
            raise ValueError(f"PBR4 container has {len(data) - offset} trailing bytes")
        return cls(tensors=tensors, extra=header.get("extra") or {})

    def complete_bytes(self) -> int:
        return len(self.dumps())


def decode_container(container: PBR4Container) -> list[np.ndarray]:
    out = []
    for enc, blob in container.tensors:
        cached = getattr(enc, "_decoded", None)
        if cached is not None:
            out.append(cached)
        else:
            out.append(decode_payload(blob))
    return out


def encode_container(items: list[EncodedTensor], extra: dict | None = None) -> PBR4Container:
    return PBR4Container(tensors=[(enc, enc.payload_bytes()) for enc in items], extra=extra or {})


class Pbr4Codec:
    """Optional tile-level wrapper. Not registered in STAGE1A_CODECS."""

    name = "pbr4_structured_nibble"
    mode_id = MODE_PBR4

    def encode(self, words, context=None):
        from pbr_core.types import EncodedBlock

        del context
        enc = encode_tensor(words, block_h=min(16, int(words.shape[0]) or 1), block_w=min(16, int(words.shape[1]) or 1))
        payload = enc.payload_bytes()
        return EncodedBlock(mode_id=self.mode_id, mode_name=self.name, payload=payload)

    def decode(self, encoded, context=None):
        del context
        return decode_payload(encoded.payload)


def setting_label(block_h: int, block_w: int, adaptive: bool) -> str:
    base = f"{block_h}x{block_w}"
    return base + ("_tree" if adaptive else "")


def choose_setting_hw(block_size: int, rows: int, cols: int) -> tuple[int, int]:
    return choose_tile_hw(block_size, rows, cols)


def shannon_nibble(c4: np.ndarray) -> float:
    flat = np.ascontiguousarray(c4, dtype=np.uint8).ravel() & 15
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat, minlength=16).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
