"""Bit-exact base→finetune checkpoint delta (related-checkpoint residual).

All methods are lossless uint16. Complete byte counts include container
JSON metadata. Success is exact restore of the target, not ≤4 BPW.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import yaml

from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_codecs.positions import (
    bitmap_bytes,
    choose_position_encoding,
    list_bytes,
    pack_bitmap,
    pack_positions,
    unpack_bitmap,
    unpack_positions,
)
from pbr_codecs.residual import best_residual, decode_residuals
from pbr_core.bf16 import join_components, split_components
from pbr_core.bitio import bits_needed
from pbr_core.types import RES_CONSTANT, RES_DEFAULT_BITMAP, RES_DEFAULT_LIST, RES_RAW
from pbr_core.hashing import sha256_words, words_to_bytes
from pbr_core.metrics import bits_per_weight
from pbr_core.rans import dump_freq_table, load_freq_table, rans_decode, rans_encode, table_from_symbols
from pbr_core.safetensors_io import TensorSpec, load_uint16
from pbr_core.types import EncodedBlock, TILE_HEADER_BYTES
from pbr_encoder.hf_weights import inventory_from_dir
from pbr_encoder.verification import ExactnessError, assert_exact

MAGIC = b"PBRD"
VERSION = 1
CONTAINER_PREFIX = 10  # magic(4) + version(u16) + header_len(u32)

KIND_RAW = 0
KIND_RANS = 1
KIND_ZLIB = 2
KIND_LIST = 3
KIND_BITMAP = 4
KIND_RESIDUAL = 5
KIND_FIELD = 6
KIND_STANDALONE = 7

SIGN_PACKBITS = 0
SIGN_LIST = 1
SIGN_BITMAP = 2
FIELD_RANS_XOR = 0
FIELD_RANS_MOD = 1
FIELD_CONSTANT = 2

DISCLAIMER = (
    "Exact base→finetune delta (related-checkpoint residual). "
    "Bit-exact uint16 only. Complete bytes include JSON metadata and every "
    "coder table. Standalone BPW is the finetune alone (PBR-E rANS). "
    "Delta-only BPW assumes the base checkpoint is already present. "
    "If the base must be shipped too, total = base PBR-E + delta. "
    "Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
)

TWO_BYTE = {"BF16", "F16", "FP16"}


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def pair_tensors(
    base_specs: list[TensorSpec], target_specs: list[TensorSpec]
) -> tuple[list[tuple[TensorSpec, TensorSpec]], list[TensorSpec], list[TensorSpec]]:
    """Pair 16-bit tensors by (name, shape). Unmatched target tensors must be stored standalone."""
    base16 = [s for s in base_specs if s.dtype in TWO_BYTE]
    tgt16 = [s for s in target_specs if s.dtype in TWO_BYTE]
    bmap = {(s.name, tuple(s.shape)): s for s in base16}
    tmap = {(s.name, tuple(s.shape)): s for s in tgt16}
    keys = sorted(set(bmap) & set(tmap))
    paired = [(bmap[k], tmap[k]) for k in keys]
    unmatched_base = [bmap[k] for k in sorted(set(bmap) - set(tmap))]
    unmatched_target = [tmap[k] for k in sorted(set(tmap) - set(bmap))]
    return paired, unmatched_base, unmatched_target


def _entropy_bincount(values: np.ndarray, minlength: int) -> float:
    flat = np.ascontiguousarray(values).ravel()
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat.astype(np.int64), minlength=minlength)
    total = float(counts.sum())
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def xor_stats(base: np.ndarray, target: np.ndarray) -> dict:
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    t = np.ascontiguousarray(target, dtype=np.uint16).ravel()
    if b.size != t.size:
        raise ValueError("base/target word counts differ")
    xor = np.bitwise_xor(b, t)
    n = int(xor.size)
    unchanged = int(np.count_nonzero(xor == 0))
    pop = np.bitwise_count(xor).astype(np.float64) if hasattr(np, "bitwise_count") else _popcount16(xor)
    sb, eb, mb = split_components(b)
    st, et, mt = split_components(t)
    sx = np.bitwise_xor(sb, st)
    ex = np.bitwise_xor(eb, et)
    mx = np.bitwise_xor(mb, mt)
    return {
        "n_words": n,
        "n_unchanged": unchanged,
        "unchanged_frac": (unchanged / n) if n else 1.0,
        "n_changed": n - unchanged,
        "mean_xor_popcount": float(pop.mean()) if n else 0.0,
        "xor_hamming_frac": float(pop.mean() / 16.0) if n else 0.0,
        "H_xor": _entropy_bincount(xor, 65536),
        "H_sign_xor": _entropy_bincount(sx, 2),
        "H_exp_xor": _entropy_bincount(ex, 256),
        "H_mant_xor": _entropy_bincount(mx, 128),
        "sign_equal_frac": float(np.mean(sx == 0)) if n else 1.0,
        "exp_equal_frac": float(np.mean(ex == 0)) if n else 1.0,
        "mant_equal_frac": float(np.mean(mx == 0)) if n else 1.0,
    }


def _popcount16(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint32)
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    return (((x + (x >> 4)) & 0x0F0F) * 0x0101) >> 8


def _rans_u8(symbols: np.ndarray) -> bytes:
    freq = table_from_symbols(symbols)
    table = dump_freq_table(freq)
    stream = rans_encode(symbols, freq)
    return struct.pack("<HI", len(table), len(stream)) + table + stream


def _unrans_u8(data: bytes, n: int, offset: int = 0) -> tuple[np.ndarray, int]:
    table_len, stream_len = struct.unpack_from("<HI", data, offset)
    offset += 6
    freq, end = load_freq_table(data, offset)
    if end - offset != table_len:
        raise ValueError("rANS table length mismatch")
    stream = data[end : end + stream_len]
    if len(stream) != stream_len:
        raise ValueError("rANS stream truncated")
    return rans_decode(stream, n, freq), end + stream_len


def encode_standalone_pbre(words: np.ndarray) -> bytes:
    """Target (or base) alone: PBR-E rANS exponents, raw fallback if rANS loses."""
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    n = int(flat.size)
    raw = bytes([KIND_RAW]) + words_to_bytes(flat)
    if n == 0:
        return raw
    codec = Bf16ExpRansCodec()
    enc = codec.encode(flat.reshape(n, 1) if n else flat)
    if enc is None:
        return raw
    rans = bytes([KIND_RANS]) + enc.payload
    # Count one conceptual tile header inside the rans payload comparison so
    # tiny tensors still prefer raw when the codebook cannot pay for itself.
    rans_complete = TILE_HEADER_BYTES + len(enc.payload)
    raw_complete = TILE_HEADER_BYTES + 2 * n
    return rans if rans_complete < raw_complete else raw


def decode_standalone_pbre(payload: bytes, n_words: int, shape: tuple[int, ...]) -> np.ndarray:
    kind = payload[0]
    body = payload[1:]
    if kind == KIND_RAW:
        words = np.frombuffer(body, dtype="<u2", count=n_words)
        return np.array(words, dtype=np.uint16, copy=True).reshape(shape)
    if kind == KIND_RANS:
        block = EncodedBlock(mode_id=0, mode_name="bf16_exp_rans", payload=body, rows=n_words, cols=1)
        flat = Bf16ExpRansCodec().decode(block)
        return np.ascontiguousarray(flat, dtype=np.uint16).reshape(shape)
    raise ValueError(f"unknown standalone kind {kind}")


def encode_xor_zlib(base: np.ndarray, target: np.ndarray) -> bytes:
    xor = np.bitwise_xor(
        np.ascontiguousarray(base, dtype=np.uint16).ravel(),
        np.ascontiguousarray(target, dtype=np.uint16).ravel(),
    )
    return bytes([KIND_ZLIB]) + zlib.compress(words_to_bytes(xor), 9)


def decode_xor_zlib(payload: bytes, base: np.ndarray) -> np.ndarray:
    if payload[:1] != bytes([KIND_ZLIB]):
        raise ValueError("xor_zlib kind mismatch")
    xor = np.frombuffer(zlib.decompress(payload[1:]), dtype="<u2")
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    if xor.size != b.size:
        raise ValueError("xor_zlib length mismatch")
    return np.bitwise_xor(b, np.array(xor, dtype=np.uint16, copy=True)).reshape(base.shape)


def encode_xor_rans(base: np.ndarray, target: np.ndarray) -> bytes:
    """Whole-word uint16 XOR, then rANS the little-endian lo/hi bytes."""
    xor = np.bitwise_xor(
        np.ascontiguousarray(base, dtype=np.uint16).ravel(),
        np.ascontiguousarray(target, dtype=np.uint16).ravel(),
    )
    lo = (xor & np.uint16(0xFF)).astype(np.uint8)
    hi = (xor >> np.uint16(8)).astype(np.uint8)
    return bytes([KIND_RANS]) + _rans_u8(lo) + _rans_u8(hi)


def decode_xor_rans(payload: bytes, base: np.ndarray) -> np.ndarray:
    if payload[:1] != bytes([KIND_RANS]):
        raise ValueError("xor_rans kind mismatch")
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    n = int(b.size)
    lo, off = _unrans_u8(payload, n, 1)
    hi, _end = _unrans_u8(payload, n, off)
    xor = lo.astype(np.uint16) | (hi.astype(np.uint16) << np.uint16(8))
    return np.bitwise_xor(b, xor).reshape(base.shape)


def _encode_sign_delta(sign_xor: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(sign_xor, dtype=np.uint8).ravel()
    n = int(flat.size)
    mask = flat != 0
    n_flip = int(np.count_nonzero(mask))
    packed = bytes([SIGN_PACKBITS]) + pack_bitmap(mask)
    list_payload = bytes([SIGN_LIST]) + pack_positions(np.nonzero(mask)[0].astype(np.uint32), n)
    bitmap_payload = bytes([SIGN_BITMAP]) + pack_bitmap(mask)
    return min((packed, list_payload, bitmap_payload), key=len)


def _decode_sign_delta(data: bytes, n: int, offset: int) -> tuple[np.ndarray, int]:
    kind = data[offset]
    offset += 1
    if kind == SIGN_PACKBITS or kind == SIGN_BITMAP:
        mask, offset = unpack_bitmap(data, n, offset)
        return mask.astype(np.uint8), offset
    if kind == SIGN_LIST:
        pos, offset = unpack_positions(data, n, offset)
        out = np.zeros(n, dtype=np.uint8)
        if pos.size:
            out[pos] = 1
        return out, offset
    raise ValueError(f"unknown sign delta kind {kind}")


def _encode_field_byte(delta: np.ndarray, *, mode: int = FIELD_RANS_XOR) -> bytes:
    flat = np.ascontiguousarray(delta, dtype=np.uint8).ravel()
    if flat.size == 0:
        return bytes([mode]) + _rans_u8(flat)
    if int(flat.min()) == int(flat.max()) and mode != FIELD_RANS_MOD:
        return bytes([FIELD_CONSTANT, int(flat[0])])
    return bytes([mode]) + _rans_u8(flat)


def _decode_field_byte(data: bytes, n: int, offset: int) -> tuple[np.ndarray, int]:
    kind = data[offset]
    offset += 1
    if kind == FIELD_CONSTANT:
        return np.full(n, data[offset], dtype=np.uint8), offset + 1
    if kind in (FIELD_RANS_XOR, FIELD_RANS_MOD):
        return _unrans_u8(data, n, offset)
    raise ValueError(f"unknown field kind {kind}")


def encode_field_split(base: np.ndarray, target: np.ndarray) -> bytes:
    """Sign / exponent / mantissa deltas, each coded on its own alphabet."""
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    t = np.ascontiguousarray(target, dtype=np.uint16).ravel()
    sb, eb, mb = split_components(b)
    st, et, mt = split_components(t)
    sign_xor = np.bitwise_xor(sb, st)
    exp_xor = np.bitwise_xor(eb, et)
    exp_mod = np.bitwise_and(et.astype(np.int16) - eb.astype(np.int16), 255).astype(np.uint8)
    mant_xor = np.bitwise_xor(mb, mt)
    mant_mod = np.bitwise_and(mt.astype(np.int16) - mb.astype(np.int16), 127).astype(np.uint8)
    sign_blob = _encode_sign_delta(sign_xor)
    exp_blob = min(
        (_encode_field_byte(exp_xor, mode=FIELD_RANS_XOR), _encode_field_byte(exp_mod, mode=FIELD_RANS_MOD)),
        key=len,
    )
    mant_blob = min(
        (_encode_field_byte(mant_xor, mode=FIELD_RANS_XOR), _encode_field_byte(mant_mod, mode=FIELD_RANS_MOD)),
        key=len,
    )
    return bytes([KIND_FIELD]) + sign_blob + exp_blob + mant_blob


def decode_field_split(payload: bytes, base: np.ndarray) -> np.ndarray:
    if payload[:1] != bytes([KIND_FIELD]):
        raise ValueError("field_split kind mismatch")
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    n = int(b.size)
    sb, eb, mb = split_components(b)
    sign_xor, off = _decode_sign_delta(payload, n, 1)
    exp_kind = payload[off]
    exp_delta, off = _decode_field_byte(payload, n, off)
    mant_kind = payload[off]
    mant_delta, _end = _decode_field_byte(payload, n, off)
    sign_t = np.bitwise_xor(sb, sign_xor)
    if exp_kind == FIELD_RANS_MOD:
        exp_t = np.bitwise_and(eb.astype(np.uint16) + exp_delta.astype(np.uint16), np.uint16(0xFF)).astype(np.uint8)
    else:
        exp_t = np.bitwise_xor(eb, exp_delta)
    if mant_kind == FIELD_RANS_MOD:
        mant_t = np.bitwise_and(mb.astype(np.uint16) + mant_delta.astype(np.uint16), np.uint16(0x7F)).astype(np.uint8)
    else:
        mant_t = np.bitwise_xor(mb, mant_delta)
    return join_components(sign_t, exp_t, mant_t).reshape(base.shape)


def encode_sparse_patch(base: np.ndarray, target: np.ndarray) -> bytes:
    """Store only differing indices + target values (list vs bitmap)."""
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel()
    t = np.ascontiguousarray(target, dtype=np.uint16).ravel()
    n = int(b.size)
    mask = b != t
    n_exc = int(np.count_nonzero(mask))
    values = words_to_bytes(t[mask]) if n_exc else b""
    if choose_position_encoding(mask) == "list":
        return bytes([KIND_LIST]) + pack_positions(np.nonzero(mask)[0].astype(np.uint32), n) + values
    return bytes([KIND_BITMAP]) + pack_bitmap(mask) + values


def decode_sparse_patch(payload: bytes, base: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(base, dtype=np.uint16).ravel().copy()
    n = int(b.size)
    kind = payload[0]
    if kind == KIND_LIST:
        pos, offset = unpack_positions(payload, n, 1)
        if pos.size:
            vals = np.frombuffer(payload, dtype="<u2", count=int(pos.size), offset=offset)
            b[pos] = np.array(vals, dtype=np.uint16, copy=True)
        return b.reshape(base.shape)
    if kind == KIND_BITMAP:
        mask, offset = unpack_bitmap(payload, n, 1)
        n_exc = int(np.count_nonzero(mask))
        if n_exc:
            vals = np.frombuffer(payload, dtype="<u2", count=n_exc, offset=offset)
            b[mask] = np.array(vals, dtype=np.uint16, copy=True)
        return b.reshape(base.shape)
    raise ValueError(f"unknown sparse kind {kind}")


def encode_xor_residual(base: np.ndarray, target: np.ndarray) -> bytes:
    """Default XOR residual + dict + raw fallback (complete-byte min)."""
    xor = np.bitwise_xor(
        np.ascontiguousarray(base, dtype=np.uint16).ravel(),
        np.ascontiguousarray(target, dtype=np.uint16).ravel(),
    )
    n = int(xor.size)
    if n == 0 or n <= 1_000_000:
        return bytes([KIND_RESIDUAL]) + best_residual(xor)[1]
    counts = np.bincount(xor.astype(np.int64), minlength=65536)
    n_unique = int(np.count_nonzero(counts))
    default = int(np.argmax(counts))
    n_exc = n - int(counts[default])
    raw_len = 1 + 2 * n
    list_len = 1 + 2 + list_bytes(n_exc, n) + 2 * n_exc
    bitmap_len = 1 + 2 + bitmap_bytes(n) + 2 * n_exc
    dict_len = 1 << 30
    if 2 <= n_unique <= 4:
        nbits = bits_needed(n_unique)
        dict_len = 1 + 2 + 2 * n_unique + math.ceil(n * nbits / 8)
    if n_unique == 1:
        blob = bytes([RES_CONSTANT]) + struct.pack("<H", default)
        return bytes([KIND_RESIDUAL]) + blob
    best_est = min(raw_len, list_len, bitmap_len, dict_len)
    if best_est == list_len:
        mask = xor != np.uint16(default)
        pos = np.nonzero(mask)[0].astype(np.uint32)
        blob = (
            bytes([RES_DEFAULT_LIST])
            + struct.pack("<H", default)
            + pack_positions(pos, n)
            + words_to_bytes(xor[mask])
        )
        return bytes([KIND_RESIDUAL]) + blob
    if best_est == bitmap_len:
        mask = xor != np.uint16(default)
        blob = (
            bytes([RES_DEFAULT_BITMAP])
            + struct.pack("<H", default)
            + pack_bitmap(mask)
            + words_to_bytes(xor[mask])
        )
        return bytes([KIND_RESIDUAL]) + blob
    if best_est == dict_len:
        return bytes([KIND_RESIDUAL]) + best_residual(xor)[1]
    return bytes([KIND_RESIDUAL, RES_RAW]) + words_to_bytes(xor)


def decode_xor_residual(payload: bytes, base: np.ndarray) -> np.ndarray:
    if payload[:1] != bytes([KIND_RESIDUAL]):
        raise ValueError("xor_residual kind mismatch")
    b = np.ascontiguousarray(base, dtype=np.uint16)
    n = int(b.size)
    xor = decode_residuals(payload[1:], n, 1).ravel()
    return np.bitwise_xor(b.ravel(), xor).reshape(base.shape)


def dumps_delta(header: dict, payloads: list[bytes]) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return MAGIC + struct.pack("<HI", VERSION, len(header_bytes)) + header_bytes + b"".join(payloads)


def loads_delta(blob: bytes) -> tuple[dict, list[bytes]]:
    if blob[:4] != MAGIC:
        raise ValueError(f"Bad delta magic: {blob[:4]!r}")
    version, header_len = struct.unpack_from("<HI", blob, 4)
    if version != VERSION:
        raise ValueError(f"Unsupported PBRD version {version}")
    start = CONTAINER_PREFIX
    header = json.loads(blob[start : start + header_len].decode("utf-8"))
    cursor = start + header_len
    payloads: list[bytes] = []
    for rec in header.get("tensors", []):
        n = int(rec["payload_len"])
        payloads.append(blob[cursor : cursor + n])
        cursor += n
    return header, payloads


def complete_container_bytes(header: dict, payload_lens: list[int]) -> int:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return CONTAINER_PREFIX + len(header_bytes) + int(sum(payload_lens))


METHOD_CODECS = {
    "standalone_pbre_rans": (encode_standalone_pbre, decode_standalone_pbre),
    "xor_zlib": (encode_xor_zlib, decode_xor_zlib),
    "xor_rans": (encode_xor_rans, decode_xor_rans),
    "field_split": (encode_field_split, decode_field_split),
    "sparse_patch": (encode_sparse_patch, decode_sparse_patch),
    "xor_residual": (encode_xor_residual, decode_xor_residual),
}

DELTA_METHODS = ("xor_zlib", "xor_rans", "field_split", "sparse_patch", "xor_residual")


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    w = float(sum(weights))
    if w <= 0:
        return 0.0
    return float(sum(v * wt for v, wt in zip(values, weights)) / w)


def evaluate_pair(
    base_dir: Path,
    target_dir: Path,
    *,
    base_meta: dict,
    target_meta: dict,
    max_tensors: int = 0,
) -> dict:
    paired, unmatched_base, unmatched_target = pair_tensors(
        inventory_from_dir(base_dir), inventory_from_dir(target_dir)
    )
    if max_tensors > 0:
        paired = paired[:max_tensors]
    t0 = time.perf_counter()
    tensor_rows: list[dict] = []
    payload_lens: dict[str, list[int]] = {m: [] for m in ("standalone_pbre_rans", "base_pbre_rans", *DELTA_METHODS)}
    header_recs: dict[str, list[dict]] = {m: [] for m in payload_lens}
    exact_flags: dict[str, bool] = {m: True for m in ("standalone_pbre_rans", *DELTA_METHODS)}
    n_words = 0
    paired_words = 0
    orig_bytes = 0
    stat_acc = {
        "n_unchanged": 0,
        "n_changed": 0,
        "popcount_sum": 0.0,
        "H_xor": 0.0,
        "H_sign_xor": 0.0,
        "H_exp_xor": 0.0,
        "H_mant_xor": 0.0,
        "sign_equal": 0.0,
        "exp_equal": 0.0,
        "mant_equal": 0.0,
    }

    print(DISCLAIMER, flush=True)
    print(
        f"pair base={base_meta.get('repo_id')}@{base_meta.get('revision', '')[:12]} "
        f"target={target_meta.get('repo_id')}@{target_meta.get('revision', '')[:12]} "
        f"paired={len(paired)} unmatched_base={len(unmatched_base)} "
        f"unmatched_target={len(unmatched_target)}",
        flush=True,
    )

    for i, (bspec, tspec) in enumerate(paired, start=1):
        base = load_uint16(bspec)
        target = load_uint16(tspec)
        n = int(target.size)
        stats = xor_stats(base, target)
        n_words += n
        paired_words += n
        orig_bytes += n * 2
        stat_acc["n_unchanged"] += stats["n_unchanged"]
        stat_acc["n_changed"] += stats["n_changed"]
        stat_acc["popcount_sum"] += stats["mean_xor_popcount"] * n
        for key, acc in (
            ("H_xor", "H_xor"),
            ("H_sign_xor", "H_sign_xor"),
            ("H_exp_xor", "H_exp_xor"),
            ("H_mant_xor", "H_mant_xor"),
            ("sign_equal_frac", "sign_equal"),
            ("exp_equal_frac", "exp_equal"),
            ("mant_equal_frac", "mant_equal"),
        ):
            stat_acc[acc] += float(stats[key]) * n

        target_sha = sha256_words(target)
        base_sha = sha256_words(base)
        print(
            f"[{i}/{len(paired)}] {tspec.name} {list(tspec.shape)} "
            f"unchanged={stats['unchanged_frac']:.4f} mean_pop={stats['mean_xor_popcount']:.3f}",
            flush=True,
        )

        base_payload = encode_standalone_pbre(base)
        tgt_payload = encode_standalone_pbre(target)
        restored_tgt = decode_standalone_pbre(tgt_payload, n, tuple(target.shape))
        try:
            assert_exact(target, restored_tgt, label=f"standalone:{tspec.name}")
        except ExactnessError:
            exact_flags["standalone_pbre_rans"] = False
            raise
        payload_lens["standalone_pbre_rans"].append(len(tgt_payload))
        payload_lens["base_pbre_rans"].append(len(base_payload))
        rec_common = {
            "name": tspec.name,
            "shape": list(int(x) for x in tspec.shape),
            "n_words": n,
            "sha256": target_sha,
        }
        header_recs["standalone_pbre_rans"].append({**rec_common, "payload_len": len(tgt_payload), "kind": "standalone"})
        header_recs["base_pbre_rans"].append(
            {**rec_common, "sha256": base_sha, "payload_len": len(base_payload), "kind": "standalone"}
        )

        method_sizes: dict[str, int] = {"standalone_pbre_rans": len(tgt_payload), "base_pbre_rans": len(base_payload)}
        for method in DELTA_METHODS:
            encode_fn, decode_fn = METHOD_CODECS[method]
            payload = encode_fn(base, target)
            restored = decode_fn(payload, base)
            try:
                vr = assert_exact(target, restored, label=f"{method}:{tspec.name}")
                exact = vr.exact
            except ExactnessError:
                exact_flags[method] = False
                raise
            payload_lens[method].append(len(payload))
            header_recs[method].append({**rec_common, "payload_len": len(payload), "kind": method})
            method_sizes[method] = len(payload)
            del payload, restored
            if not exact:
                exact_flags[method] = False

        tensor_rows.append(
            {
                "name": tspec.name,
                "shape": list(int(x) for x in tspec.shape),
                "n_words": n,
                "target_sha256": target_sha,
                "base_sha256": base_sha,
                "stats": stats,
                "payload_bytes": method_sizes,
                "exact": {m: "PASS" for m in ("standalone_pbre_rans", *DELTA_METHODS)},
            }
        )
        del base, target, restored_tgt, base_payload, tgt_payload

    for spec in unmatched_target:
        target = load_uint16(spec)
        n = int(target.size)
        n_words += n
        orig_bytes += n * 2
        payload = encode_standalone_pbre(target)
        restored = decode_standalone_pbre(payload, n, tuple(target.shape))
        assert_exact(target, restored, label=f"unmatched:{spec.name}")
        rec = {
            "name": spec.name,
            "shape": list(int(x) for x in spec.shape),
            "n_words": n,
            "sha256": sha256_words(target),
            "payload_len": len(payload),
            "kind": "unmatched_standalone",
        }
        for method in ("standalone_pbre_rans", *DELTA_METHODS):
            payload_lens[method].append(len(payload))
            header_recs[method].append(rec)
        tensor_rows.append(
            {
                "name": spec.name,
                "shape": rec["shape"],
                "n_words": n,
                "target_sha256": rec["sha256"],
                "base_sha256": None,
                "unmatched_target": True,
                "payload_bytes": {m: len(payload) for m in ("standalone_pbre_rans", *DELTA_METHODS)},
                "exact": {m: "PASS" for m in ("standalone_pbre_rans", *DELTA_METHODS)},
            }
        )
        del target, restored, payload

    elapsed = time.perf_counter() - t0
    stats_den = paired_words or 1
    method_complete: dict[str, int] = {}
    method_headers: dict[str, dict] = {}
    for method, recs in header_recs.items():
        who = target_meta if method != "base_pbre_rans" else base_meta
        header = {
            "format": "PBR-Delta",
            "version": VERSION,
            "method": method,
            "base": {"repo_id": base_meta.get("repo_id"), "revision": base_meta.get("revision")},
            "target": {"repo_id": target_meta.get("repo_id"), "revision": target_meta.get("revision")},
            "note": DISCLAIMER,
            "tensors": recs,
            "unmatched_base": [s.name for s in unmatched_base],
            "unmatched_target": [s.name for s in unmatched_target],
        }
        method_headers[method] = header
        method_complete[method] = complete_container_bytes(header, payload_lens[method])

    totals = {
        "paired_tensors": len(paired),
        "unmatched_base": [s.name for s in unmatched_base],
        "unmatched_target": [s.name for s in unmatched_target],
        "n_words": n_words,
        "paired_words": paired_words,
        "original_bytes": orig_bytes,
        "unchanged_frac": stat_acc["n_unchanged"] / stats_den,
        "changed_frac": stat_acc["n_changed"] / stats_den,
        "mean_xor_popcount": stat_acc["popcount_sum"] / stats_den,
        "xor_hamming_frac": (stat_acc["popcount_sum"] / stats_den) / 16.0,
        "H_xor": stat_acc["H_xor"] / stats_den,
        "H_sign_xor": stat_acc["H_sign_xor"] / stats_den,
        "H_exp_xor": stat_acc["H_exp_xor"] / stats_den,
        "H_mant_xor": stat_acc["H_mant_xor"] / stats_den,
        "sign_equal_frac": stat_acc["sign_equal"] / stats_den,
        "exp_equal_frac": stat_acc["exp_equal"] / stats_den,
        "mant_equal_frac": stat_acc["mant_equal"] / stats_den,
        "elapsed_s": elapsed,
        "methods": {},
    }
    standalone_b = method_complete["standalone_pbre_rans"]
    base_b = method_complete["base_pbre_rans"]
    for method in ("standalone_pbre_rans", *DELTA_METHODS):
        enc_b = method_complete[method]
        delta_only = method != "standalone_pbre_rans"
        bundle_b = (base_b + enc_b) if delta_only else enc_b
        totals["methods"][method] = {
            "payload_bytes": int(sum(payload_lens[method])),
            "complete_bytes": enc_b,
            "standalone_bpw": bits_per_weight(standalone_b, n_words),
            "delta_only_bpw": bits_per_weight(enc_b, n_words) if delta_only else None,
            "bundle_bytes": bundle_b,
            "bundle_bpw": bits_per_weight(bundle_b, n_words),
            "exact": "PASS" if exact_flags[method] else "FAIL",
        }
    totals["methods"]["base_pbre_rans"] = {
        "payload_bytes": int(sum(payload_lens["base_pbre_rans"])),
        "complete_bytes": base_b,
        "delta_only_bpw": None,
        "bundle_bytes": base_b,
        "bundle_bpw": bits_per_weight(base_b, n_words),
        "exact": "PASS",
    }
    best_delta = min(DELTA_METHODS, key=lambda m: method_complete[m])
    totals["best_delta_method"] = best_delta
    totals["best_delta_complete_bytes"] = method_complete[best_delta]
    totals["reconstruct_target"] = (
        "PASS" if all(exact_flags[m] for m in ("standalone_pbre_rans", *DELTA_METHODS)) else "FAIL"
    )

    return {
        "disclaimer": DISCLAIMER,
        "base": dict(base_meta),
        "target": dict(target_meta),
        "architecture": {
            "note": "Paired by tensor name and shape. Same-arch public Qwen2.5-0.5B pair.",
        },
        "totals": totals,
        "tensors": tensor_rows,
        "container_headers": {k: {"n_tensors": len(v["tensors"]), "complete_bytes": method_complete[k]} for k, v in method_headers.items()},
    }


def format_markdown(report: dict) -> str:
    t = report["totals"]
    base = report["base"]
    tgt = report["target"]
    methods = t["methods"]

    def row(label: str, key: str) -> str:
        m = methods[key]
        delta = m["delta_only_bpw"]
        delta_s = f"{delta:.4f}" if delta is not None else "n/a (standalone)"
        return (
            f"| {label} | {m['complete_bytes']} | {delta_s} | "
            f"{m['bundle_bytes']} | {m['bundle_bpw']:.4f} | {m['exact']} |"
        )

    lines = [
        "# Exact base→finetune delta",
        "",
        DISCLAIMER,
        "",
        "## Pair",
        "",
        f"- Base: `{base.get('repo_id')}` revision `{base.get('revision')}`",
        f"- Finetune / target: `{tgt.get('repo_id')}` revision `{tgt.get('revision')}`",
        f"- Architecture: same Qwen2 config (hidden 896, 24 layers, 14 heads, vocab 151936, BF16).",
        f"- Paired 16-bit tensors: **{t['paired_tensors']}** (by name+shape)",
        f"- Unmatched base: {t['unmatched_base'] or 'none'}",
        f"- Unmatched target: {t['unmatched_target'] or 'none'}",
        f"- Words: **{t['n_words']}**  original bytes: **{t['original_bytes']}**",
        "",
        "## Weight-change stats (bit-exact XOR)",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| % weights unchanged | {100.0 * t['unchanged_frac']:.4f} |",
        f"| % weights changed | {100.0 * t['changed_frac']:.4f} |",
        f"| mean XOR popcount (of 16 bits) | {t['mean_xor_popcount']:.4f} |",
        f"| XOR Hamming fraction | {t['xor_hamming_frac']:.4f} |",
        f"| H(uint16 XOR) | {t['H_xor']:.4f} |",
        f"| H(sign XOR) | {t['H_sign_xor']:.4f} |",
        f"| H(exp XOR) | {t['H_exp_xor']:.4f} |",
        f"| H(mant XOR) | {t['H_mant_xor']:.4f} |",
        f"| sign equal fraction | {t['sign_equal_frac']:.4f} |",
        f"| exp equal fraction | {t['exp_equal_frac']:.4f} |",
        f"| mantissa equal fraction | {t['mant_equal_frac']:.4f} |",
        "",
        "## Encoded size (complete container, all metadata)",
        "",
        "Standalone BPW is the finetune encoded alone with PBR-E rANS. "
        "Delta-only BPW is the residual payload assuming the base is already present. "
        "**Bundle** = base PBR-E + delta (honest cost if the base must be shipped too).",
        "",
        "| method | complete B | delta-only BPW | bundle B | bundle BPW | exact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        row("1. target PBR-E rANS (standalone)", "standalone_pbre_rans"),
        row("2a. uint16 XOR + zlib", "xor_zlib"),
        row("2b. uint16 XOR + rANS lo/hi", "xor_rans"),
        row("3. sign/exp/mantissa field deltas", "field_split"),
        row("4. sparse changed-position patches", "sparse_patch"),
        row("5. default XOR residual + dict/raw", "xor_residual"),
        "",
        f"| base PBR-E rANS (for the bundle) | {methods['base_pbre_rans']['complete_bytes']} | n/a | "
        f"{methods['base_pbre_rans']['complete_bytes']} | {methods['base_pbre_rans']['bundle_bpw']:.4f} | "
        f"{methods['base_pbre_rans']['exact']} |",
        "",
        f"- Standalone finetune BPW: **{methods['standalone_pbre_rans']['standalone_bpw']:.4f}**",
        f"- Best delta method: **{t['best_delta_method']}** "
        f"({t['best_delta_complete_bytes']} B, "
        f"{methods[t['best_delta_method']]['delta_only_bpw']:.4f} delta-only BPW)",
        f"- Reconstruct(target) from base+delta: **{t['reconstruct_target']}** (uint16 / SHA-256)",
        "",
        "Shipping the base *and* a delta is **not** cheaper than shipping the finetune "
        "standalone unless the delta is smaller than `standalone − base_PBR-E` (it is not "
        "a way to hide a second full checkpoint). Not ≤4 BPW.",
        "",
        f"Elapsed encode+verify: {t['elapsed_s']:.1f} s.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact base→finetune checkpoint delta.")
    parser.add_argument("--config", type=Path, default=Path("configs/poc_delta.yaml"))
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--tag", default="qwen")
    parser.add_argument("--max-tensors", type=int, default=0, help="0 = all paired 16-bit tensors")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/checkpoint_delta"))
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    base_meta = {
        "repo_id": cfg.get("base_repo", "Qwen/Qwen2.5-0.5B"),
        "revision": cfg.get("base_revision", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
        "license": cfg.get("base_license", "Apache-2.0"),
    }
    target_meta = {
        "repo_id": cfg.get("target_repo", "Qwen/Qwen2.5-0.5B-Instruct"),
        "revision": cfg.get("target_revision", "7ae557604adf67be50417f59c2c2f167def9a775"),
        "license": cfg.get("target_license", "Apache-2.0"),
    }
    base_dir = args.base_dir or Path(cfg.get("base_dir", "outputs/models/Qwen__Qwen2.5-0.5B"))
    target_dir = args.target_dir or Path(cfg.get("target_dir", "outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    if not base_dir.exists() or not target_dir.exists():
        print(f"missing checkpoint dir(s): base={base_dir} target={target_dir}")
        return 2
    report = evaluate_pair(
        base_dir,
        target_dir,
        base_meta=base_meta,
        target_meta=target_meta,
        max_tensors=args.max_tensors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    art_json = Path(f"artifacts/checkpoint_delta_{tag}.json")
    art_md = Path(f"artifacts/checkpoint_delta_{tag}.md")
    out_json = args.output_dir / f"checkpoint_delta_{tag}.json"
    out_md = args.output_dir / f"checkpoint_delta_{tag}.md"
    md = format_markdown(report)
    blob = json.dumps(report, indent=2) + "\n"
    for path in (art_json, out_json):
        path.write_text(blob, encoding="utf-8")
    for path in (art_md, out_md):
        path.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {art_json} and {art_md}")
    if report["totals"]["reconstruct_target"] != "PASS":
        print("DELTA FAIL: reconstruct(target) was not bit-exact")
        return 1
    print("DELTA PASS: reconstruct(target) from base+delta is bit-exact (SHA-256).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
