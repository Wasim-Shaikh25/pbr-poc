"""Adaptive 7-bit mantissa codec (RAW / Huffman+ESCAPE / CONTEXT).

Complete byte accounting: payload + local headers + shared codebook/rule,
charged only when used. ALL_RAW never includes a Huffman table.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from pbr_adaptive_mantissa.predict import (
    causal_prev,
    local_pos,
    pick_rule,
    predict_vec,
)
from pbr_core.bf16 import join_components, split_components
from pbr_core.bitio import BitReader, BitWriter
from pbr_core.hashing import sha256_words
from pbr_core.huffman import HuffmanTable, dump_table, load_table, table_from_symbols
from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode, table_from_symbols as rans_table
from pbr_encoder.verification import assert_exact

MAGIC = b"AM7\x01"
BUNDLE_MAGIC = b"PBRB"
VERSION = 1
BLOCK_SIZE = 256
ESCAPE = 128
STRAT_ALL_RAW = 0
STRAT_ALL_HUFF = 1
STRAT_ALL_CTX = 2
STRAT_MIXED = 3
STRAT_NAMES = {
    STRAT_ALL_RAW: "ALL_RAW",
    STRAT_ALL_HUFF: "ALL_HUFFMAN",
    STRAT_ALL_CTX: "ALL_CONTEXT",
    STRAT_MIXED: "MIXED",
}
MODE_RAW = 0
MODE_HUFF = 1
MODE_CTX = 2
MODE_NAMES = {MODE_RAW: "RAW", MODE_HUFF: "HUFFMAN", MODE_CTX: "CONTEXT"}
FLAG_HAS_TABLE = 1 << 0
FREEZE_BLOCKS = 32
PBRE_REF_BPW = 10.62

DISCLAIMER = (
    "Adaptive 7-bit mantissa codec PoC. Exact uint16 round-trip. Complete bytes "
    "include modes, codebook, directory, and padding. Not a ≤4 BPW or ≤8 BPW "
    "total claim unless the measured complete BPW is that low. Phase A found "
    "real LLM mantissas near 7 BPW; this codec is allowed to confirm that."
)

# magic, n_words, block_size, strat, rule, flags, table_len, n_blocks
_HDR = struct.Struct("<4sIHBBBHI")


def pack_lsb(values: np.ndarray, nbits: int) -> bytes:
    """LSB-first pack matching ``BitWriter``."""
    v = np.ascontiguousarray(values, dtype=np.uint32).ravel()
    n = int(v.size)
    if n == 0:
        return b""
    if nbits == 1:
        bits = (v & 1).astype(np.uint8)
        return np.packbits(bits, bitorder="little").tobytes()
    if nbits == 7:
        return _pack_7(v)
    w = BitWriter()
    w.write_array(v, nbits)
    return w.finalize()


def _pack_7(v: np.ndarray) -> bytes:
    m = np.ascontiguousarray(v, dtype=np.uint64).ravel() & np.uint64(0x7F)
    n = int(m.size)
    pad = (-n) % 8
    if pad:
        m = np.concatenate([m, np.zeros(pad, dtype=np.uint64)])
    g = m.reshape(-1, 8)
    acc = np.zeros(g.shape[0], dtype=np.uint64)
    for i in range(8):
        acc |= g[:, i] << np.uint64(7 * i)
    out = np.empty((g.shape[0], 7), dtype=np.uint8)
    for b in range(7):
        out[:, b] = (acc >> np.uint64(8 * b)) & np.uint64(0xFF)
    real = (n * 7 + 7) // 8
    return out.ravel()[:real].tobytes()


def unpack_lsb(data: bytes, count: int, nbits: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    if nbits == 1:
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
        return bits[:count].astype(np.uint8)
    if nbits == 7:
        return _unpack_7(data, count)
    r = BitReader(data)
    return r.read_array(count, nbits).astype(np.uint8)


def _unpack_7(data: bytes, count: int) -> np.ndarray:
    need = (count * 7 + 7) // 8
    buf = np.frombuffer(data[:need], dtype=np.uint8)
    if buf.size < need:
        raise ValueError("truncated 7-bit stream")
    pad_sym = (-count) % 8
    n_g = (count + pad_sym) // 8
    padded_bytes = n_g * 7
    raw = np.zeros(padded_bytes, dtype=np.uint8)
    raw[: buf.size] = buf
    g = raw.reshape(n_g, 7).astype(np.uint64)
    acc = np.zeros(n_g, dtype=np.uint64)
    for b in range(7):
        acc |= g[:, b] << np.uint64(8 * b)
    out = np.empty((n_g, 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = (acc >> np.uint64(7 * i)) & np.uint64(0x7F)
    return out.ravel()[:count]


def _ceil_bytes(nbits: int) -> int:
    return (int(nbits) + 7) // 8


def _huffman_with_escape(freeze: np.ndarray) -> HuffmanTable:
    freeze = np.ascontiguousarray(freeze, dtype=np.uint8).ravel() & np.uint8(0x7F)
    if freeze.size == 0:
        freeze = np.array([0], dtype=np.uint8)
    train = np.concatenate([freeze, np.array([ESCAPE], dtype=np.uint8)])
    return table_from_symbols(train)


def _huff_symbol_bits(table: HuffmanTable, mant: np.ndarray) -> np.ndarray:
    """Complete payload bits per symbol (code + 7-bit ESCAPE extra)."""
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    len_lut = np.zeros(128, dtype=np.int32)
    known = np.zeros(128, dtype=bool)
    for sym, (_code, nbits) in table.codes.items():
        if 0 <= int(sym) < 128:
            len_lut[int(sym)] = int(nbits)
            known[int(sym)] = True
    if ESCAPE not in table.codes:
        raise ValueError("Huffman table missing ESCAPE")
    esc_n = int(table.codes[ESCAPE][1])
    return np.where(known[m], len_lut[m], esc_n + 7)


def encode_huffman_escape(table: HuffmanTable, mant: np.ndarray) -> bytes:
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    known = set(s for s in table.codes if s != ESCAPE)
    if known.issuperset(range(128)) or (m.size and bool(np.all(np.isin(m, np.fromiter(known, dtype=np.uint8))))):
        return table.encode_symbols(m)
    w = BitWriter()
    codes = table.codes
    esc_code, esc_n = codes[ESCAPE]
    for v in m.tolist():
        if v in codes and v != ESCAPE:
            c, n = codes[v]
            w.write(c, n)
        else:
            w.write(esc_code, esc_n)
            w.write(int(v), 7)
    return w.finalize()


def decode_huffman_escape(table: HuffmanTable, data: bytes, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    from pbr_core.huffman import _build_lut

    lut = _build_lut(table.codes, table.max_len)
    mask = (1 << table.max_len) - 1
    padded = data + b"\x00\x00\x00\x00"
    mv = memoryview(padded)
    bitpos = 0
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        byte_i = bitpos >> 3
        shift = bitpos & 7
        peek = (mv[byte_i] | (mv[byte_i + 1] << 8) | (mv[byte_i + 2] << 16)) >> shift
        sym, nbits = lut[peek & mask]
        bitpos += nbits
        if int(sym) == ESCAPE:
            byte_i = bitpos >> 3
            shift = bitpos & 7
            peek = (mv[byte_i] | (mv[byte_i + 1] << 8) | (mv[byte_i + 2] << 16)) >> shift
            out[i] = peek & 0x7F
            bitpos += 7
        else:
            out[i] = int(sym) & 0x7F
    return out


def encode_context(
    mant: np.ndarray,
    exp: np.ndarray,
    pos: np.ndarray,
    rule: int,
    *,
    start_prev: int = 0,
) -> bytes:
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    prev = causal_prev(m)
    if m.size:
        prev[0] = np.uint8(int(start_prev) & 0x7F)
    pred = predict_vec(prev, exp, pos, rule)
    hits = pred == m
    w = BitWriter()
    m_list = m.tolist()
    hit_list = hits.tolist()
    for h, v in zip(hit_list, m_list):
        if h:
            w.write(1, 1)
        else:
            w.write(0, 1)
            w.write(int(v), 7)
    return w.finalize()


def decode_context(
    data: bytes,
    count: int,
    exp: np.ndarray,
    pos: np.ndarray,
    rule: int,
    *,
    start_prev: int = 0,
) -> np.ndarray:
    r = BitReader(data)
    out = np.empty(count, dtype=np.uint8)
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    p = np.ascontiguousarray(pos, dtype=np.uint8).ravel()
    prev = int(start_prev) & 0x7F
    from pbr_adaptive_mantissa.predict import predict as predict_s

    for i in range(count):
        hit = r.read(1)
        if hit:
            rec = predict_s(prev, int(e[i]), int(p[i]), rule)
        else:
            rec = r.read(7) & 0x7F
        out[i] = rec
        prev = rec
    return out


def _block_counts(n: int, block_size: int) -> tuple[int, np.ndarray]:
    n_blocks = (n + block_size - 1) // block_size if n else 0
    counts = np.full(max(n_blocks, 0), block_size, dtype=np.int32)
    if n_blocks:
        counts[-1] = n - (n_blocks - 1) * block_size
    return n_blocks, counts


def _reduce_block_bits(bits_per_sym: np.ndarray, block_size: int) -> np.ndarray:
    n = int(bits_per_sym.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(0, n, block_size)
    return np.add.reduceat(bits_per_sym.astype(np.int64), idx)


@dataclass
class MantissaEncode:
    blob: bytes
    n_words: int
    strategy: int
    rule: int
    block_size: int
    mode_counts: dict[str, int]
    mantissa_bytes: int
    mantissa_bpw: float
    codebook_bytes: int
    directory_bytes: int
    payload_bytes: int
    header_bytes: int
    freeze_hit_rate: float
    used_huffman_table: bool


def encode_mantissa(
    mant: np.ndarray,
    exp: np.ndarray,
    *,
    block_size: int = BLOCK_SIZE,
    freeze_blocks: int = FREEZE_BLOCKS,
) -> MantissaEncode:
    m = np.ascontiguousarray(mant, dtype=np.uint8).ravel() & np.uint8(0x7F)
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if e.size != m.size:
        raise ValueError("exp/mant length mismatch")
    n = int(m.size)
    pos = local_pos(n, block_size)
    n_blocks, counts = _block_counts(n, block_size)
    freeze_n = min(n, int(freeze_blocks) * int(block_size))
    freeze_m = m[:freeze_n]
    freeze_e = e[:freeze_n]
    rule, hit_rate = pick_rule(freeze_m, freeze_e, block_size) if freeze_n else (0, 0.0)
    table = _huffman_with_escape(freeze_m)
    codebook = dump_table(table)

    raw_bits = np.full(n, 7, dtype=np.int64)
    huff_bits = _huff_symbol_bits(table, m)
    pred = predict_vec(causal_prev(m), e, pos, rule)
    ctx_bits = np.where(pred == m, 1, 8).astype(np.int64)

    raw_b = _ceil_bytes(int(raw_bits.sum()))
    huff_b = _ceil_bytes(int(huff_bits.sum()))
    ctx_b = _ceil_bytes(int(ctx_bits.sum()))

    raw_blk = np.array([_ceil_bytes(int(c) * 7) for c in counts], dtype=np.int64)
    huff_blk = np.array(
        [_ceil_bytes(int(x)) for x in _reduce_block_bits(huff_bits, block_size)],
        dtype=np.int64,
    )
    ctx_blk = np.array(
        [_ceil_bytes(int(x)) for x in _reduce_block_bits(ctx_bits, block_size)],
        dtype=np.int64,
    )
    if n_blocks == 0:
        winner = np.zeros(0, dtype=np.int8)
    else:
        stacked = np.stack([raw_blk, huff_blk, ctx_blk], axis=0)
        winner = np.argmin(stacked, axis=0).astype(np.int8)

    dir_bytes = _ceil_bytes(n_blocks * 2) if n_blocks else 0
    mixed_payload = int(stacked.min(axis=0).sum()) if n_blocks else 0
    any_huff = bool(n_blocks and np.any(winner == MODE_HUFF))
    mixed_extra = dir_bytes + (len(codebook) if any_huff else 0)

    # Header without table: MAGIC(4)+n_words(4)+block(2)+strat(1)+rule(1)+flags(1)+table_len(2)+n_blocks(4) = 19
    base_hdr = 19
    cost = {
        STRAT_ALL_RAW: base_hdr + raw_b,
        STRAT_ALL_HUFF: base_hdr + len(codebook) + huff_b,
        STRAT_ALL_CTX: base_hdr + ctx_b,
        STRAT_MIXED: base_hdr + mixed_extra + mixed_payload,
    }
    strat = min(cost, key=cost.get)
    used_table = strat == STRAT_ALL_HUFF or (strat == STRAT_MIXED and any_huff)
    flags = FLAG_HAS_TABLE if used_table else 0
    table_blob = codebook if used_table else b""

    if strat == STRAT_ALL_RAW:
        payload = pack_lsb(m, 7)
        modes = {MODE_RAW: n_blocks, MODE_HUFF: 0, MODE_CTX: 0}
        directory = b""
    elif strat == STRAT_ALL_HUFF:
        payload = encode_huffman_escape(table, m)
        modes = {MODE_RAW: 0, MODE_HUFF: n_blocks, MODE_CTX: 0}
        directory = b""
        rec = decode_huffman_escape(table, payload, n)
        assert_exact(
            m.astype(np.uint16),
            rec.astype(np.uint16),
            label="all_huff_mant",
        )
    elif strat == STRAT_ALL_CTX:
        payload = encode_context(m, e, pos, rule, start_prev=0)
        rec = decode_context(payload, n, e, pos, rule, start_prev=0)
        assert_exact(m.astype(np.uint16), rec.astype(np.uint16), label="all_ctx_mant")
        modes = {MODE_RAW: 0, MODE_HUFF: 0, MODE_CTX: n_blocks}
        directory = b""
    else:
        directory = pack_lsb(winner, 2)
        chunks: list[bytes] = []
        off = 0
        start_prev = 0
        for bi, mode in enumerate(winner.tolist()):
            cnt = int(counts[bi])
            sl_m = m[off : off + cnt]
            sl_e = e[off : off + cnt]
            sl_p = pos[off : off + cnt]
            if mode == MODE_RAW:
                chunks.append(pack_lsb(sl_m, 7))
            elif mode == MODE_HUFF:
                blob_i = encode_huffman_escape(table, sl_m)
                rec = decode_huffman_escape(table, blob_i, cnt)
                assert_exact(sl_m.astype(np.uint16), rec.astype(np.uint16), label=f"mix_huff_{bi}")
                chunks.append(blob_i)
            else:
                blob_i = encode_context(sl_m, sl_e, sl_p, rule, start_prev=start_prev)
                rec = decode_context(blob_i, cnt, sl_e, sl_p, rule, start_prev=start_prev)
                assert_exact(sl_m.astype(np.uint16), rec.astype(np.uint16), label=f"mix_ctx_{bi}")
                chunks.append(blob_i)
            if cnt:
                start_prev = int(sl_m[-1])
            off += cnt
        payload = b"".join(chunks)
        modes = {
            MODE_RAW: int(np.count_nonzero(winner == MODE_RAW)),
            MODE_HUFF: int(np.count_nonzero(winner == MODE_HUFF)),
            MODE_CTX: int(np.count_nonzero(winner == MODE_CTX)),
        }

    header = _HDR.pack(
        MAGIC,
        n,
        int(block_size),
        int(strat),
        int(rule) if strat in {STRAT_ALL_CTX, STRAT_MIXED} else 0,
        flags,
        len(table_blob),
        n_blocks,
    )
    blob = header + table_blob + directory + payload
    restored = decode_mantissa(blob, exp=e)
    assert_exact(m.astype(np.uint16), restored.astype(np.uint16), label="mantissa_field")
    return MantissaEncode(
        blob=blob,
        n_words=n,
        strategy=strat,
        rule=rule,
        block_size=block_size,
        mode_counts={MODE_NAMES[k]: v for k, v in modes.items()},
        mantissa_bytes=len(blob),
        mantissa_bpw=(8.0 * len(blob) / n) if n else 0.0,
        codebook_bytes=len(table_blob),
        directory_bytes=len(directory),
        payload_bytes=len(payload),
        header_bytes=len(header),
        freeze_hit_rate=float(hit_rate),
        used_huffman_table=used_table,
    )


def decode_mantissa(blob: bytes, *, exp: np.ndarray) -> np.ndarray:
    if len(blob) < _HDR.size:
        raise ValueError("adaptive mantissa blob too short")
    magic, n, block_size, strat, rule, flags, table_len, n_blocks = _HDR.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad AM7 magic {magic!r}")
    off = _HDR.size
    table = None
    if flags & FLAG_HAS_TABLE:
        table, _end = load_table(blob, off)
        off = off + int(table_len)
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if int(e.size) != int(n):
        raise ValueError("decode exp length mismatch")
    pos = local_pos(n, block_size)
    if strat == STRAT_ALL_RAW:
        return unpack_lsb(blob[off:], n, 7)
    if strat == STRAT_ALL_HUFF:
        if table is None:
            raise ValueError("ALL_HUFFMAN missing table")
        return decode_huffman_escape(table, blob[off:], n)
    if strat == STRAT_ALL_CTX:
        return decode_context(blob[off:], n, e, pos, rule, start_prev=0)
    if strat != STRAT_MIXED:
        raise ValueError(f"unknown strategy {strat}")
    dir_n = _ceil_bytes(n_blocks * 2)
    winner = unpack_lsb(blob[off : off + dir_n], n_blocks, 2) if n_blocks else np.zeros(0, dtype=np.uint8)
    off += dir_n
    rest = blob[off:]
    out = np.empty(n, dtype=np.uint8)
    _, counts = _block_counts(n, block_size)
    cursor = 0
    wpos = 0
    start_prev = 0
    for bi in range(n_blocks):
        cnt = int(counts[bi])
        mode = int(winner[bi])
        sl_e = e[wpos : wpos + cnt]
        sl_p = pos[wpos : wpos + cnt]
        if mode == MODE_RAW:
            nb = _ceil_bytes(cnt * 7)
            rec = unpack_lsb(rest[cursor : cursor + nb], cnt, 7)
            cursor += nb
        elif mode == MODE_HUFF:
            if table is None:
                raise ValueError("MIXED Huffman block without table")
            rec = decode_huffman_escape(table, rest[cursor:], cnt)
            used = len(encode_huffman_escape(table, rec))
            cursor += used
        else:
            rec = decode_context(rest[cursor:], cnt, sl_e, sl_p, rule, start_prev=start_prev)
            used = len(encode_context(rec, sl_e, sl_p, rule, start_prev=start_prev))
            cursor += used
        out[wpos : wpos + cnt] = rec
        if cnt:
            start_prev = int(rec[-1])
        wpos += cnt
    return out


def encode_sign_raw(sign: np.ndarray) -> bytes:
    s = np.ascontiguousarray(sign, dtype=np.uint8).ravel() & np.uint8(1)
    return pack_lsb(s, 1)


def decode_sign_raw(data: bytes, n: int) -> np.ndarray:
    return unpack_lsb(data, n, 1) & np.uint8(1)


def encode_exp_rans(exp: np.ndarray) -> bytes:
    e = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if e.size == 0:
        return struct.pack("<I", 0)
    freq = rans_table(e)
    table = dump_freq_table(freq)
    stream = rans_encode(e, freq)
    return struct.pack("<II", len(table), len(stream)) + table + stream


def decode_exp_rans(data: bytes, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    tlen, slen = struct.unpack_from("<II", data, 0)
    table = data[8 : 8 + tlen]
    stream = data[8 + tlen : 8 + tlen + slen]
    freq, _ = load_freq_table(table, 0)
    rec = rans_decode(stream, n, freq)
    return rec.astype(np.uint8)


@dataclass
class BundleEncode:
    blob: bytes
    n_words: int
    mantissa: MantissaEncode
    sign_bytes: int
    exp_bytes: int
    total_bytes: int
    total_bpw: float
    sha256: str


def encode_bf16_bundle(words: np.ndarray, *, block_size: int = BLOCK_SIZE) -> BundleEncode:
    arr = np.ascontiguousarray(words, dtype=np.uint16)
    shape = tuple(int(x) for x in arr.shape)
    w = arr.ravel()
    sign, exp, mant = split_components(w)
    man = encode_mantissa(mant, exp, block_size=block_size)
    sign_b = encode_sign_raw(sign)
    exp_b = encode_exp_rans(exp)
    n = int(w.size)
    header = BUNDLE_MAGIC + struct.pack("<IB", n, len(shape))
    if shape:
        header += struct.pack("<" + "I" * len(shape), *shape)
    header += struct.pack("<II", len(sign_b), len(exp_b))
    blob = header + sign_b + exp_b + man.blob
    rec = decode_bf16_bundle(blob)
    vr = assert_exact(arr, rec, label="bf16_bundle")
    return BundleEncode(
        blob=blob,
        n_words=n,
        mantissa=man,
        sign_bytes=len(sign_b),
        exp_bytes=len(exp_b),
        total_bytes=len(blob),
        total_bpw=(8.0 * len(blob) / n) if n else 0.0,
        sha256=vr.original_sha256,
    )


def decode_bf16_bundle(blob: bytes) -> np.ndarray:
    if blob[:4] != BUNDLE_MAGIC:
        raise ValueError(f"bad bundle magic {blob[:4]!r}")
    n, ndims = struct.unpack_from("<IB", blob, 4)
    off = 9
    if ndims:
        shape = struct.unpack_from("<" + "I" * ndims, blob, off)
        off += 4 * ndims
    else:
        shape = (n,)
    slen, elen = struct.unpack_from("<II", blob, off)
    off += 8
    sign = decode_sign_raw(blob[off : off + slen], n)
    off += slen
    exp = decode_exp_rans(blob[off : off + elen], n)
    off += elen
    mant = decode_mantissa(blob[off:], exp=exp)
    words = join_components(sign, exp, mant)
    return words.reshape(shape)
