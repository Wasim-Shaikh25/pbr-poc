"""HYBX physical container: hybrid H95 + groupwise INT + selective X/Y.

Layout (little-endian)::

    [0:4]   magic b"HYBX"
    [4:6]   version u16le = 1
    [6:8]   flags u16le (bit0 = little-endian)
    [8:9]   checksum_algo u8 (1 = SHA-256)
    [9:12]  reserved
    [12:16] header_json_len u32le
    [16:]   header JSON
    then pad to 64-byte alignment
    then payload blobs; absolute file offsets live in the header

Per unique tensor
-----------------
* ``bf16_raw``: original/protected uint16 words.
* ``h95``: sign bitplane + adaptive exact exponents + packed-K or
  selective-XY mantissa (+ optional dirty-bit exceptions).
* ``groupwise``: float16 scales + uint8 zp + packed-or-XY codes +
  optional sparse BF16 outliers.

X/Y is emitted only when ``encode_array_selective`` reports a strictly
smaller complete blob than packed-K (maps + flags included). Packed
codes are the product default. This is a **new** quantized reference —
not H95Q-S1 and not PR #17 restore_q6.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pbr_h95.bitpack import pack_sign_bitplane, unpack_sign_bitplane
from pbr_h95.container_h95q import sha256_state_u16, sha256_u16
from pbr_h95.exp_codec import EXP_RAW8, decode_exp_blob, encode_exp_adaptive
from pbr_q4.const import MODE_NAMES, TILE
from pbr_q4.fields import join_uniform, split_quantized
from pbr_q4.hybrid.int_quant import dequantize_int
from pbr_q4.hybrid.policy import HybridPolicy
from pbr_q4.hybrid.quantize import QuantizedTensor
from pbr_q4.selective import decode_array_selective, encode_array_selective

MAGIC = b"HYBX"
VERSION = 1
CHECKSUM_SHA256 = 1
FLAGS_LITTLE_ENDIAN = 0x0001
ALIGN = 64
_PREFIX = struct.Struct("<4sHHB3sI")


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _merge_hist(dst: dict[str, int], src: Mapping[str, int]) -> None:
    for k, v in src.items():
        if not v:
            continue
        dst[k] = dst.get(k, 0) + int(v)


def _encode_outliers(idx: np.ndarray, words: np.ndarray) -> tuple[bytes, bytes]:
    if idx.size == 0:
        return b"", b""
    return (
        np.ascontiguousarray(idx, dtype="<u4").tobytes(),
        np.ascontiguousarray(words, dtype="<u2").tobytes(),
    )


def _decode_outliers(idx_b: bytes, words_b: bytes, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint16)
    need_i, need_w = n * 4, n * 2
    if len(idx_b) < need_i or len(words_b) < need_w:
        raise ValueError("short outlier stream")
    idx = np.frombuffer(memoryview(idx_b)[:need_i], dtype="<u4").astype(np.int64).copy()
    words = np.frombuffer(memoryview(words_b)[:need_w], dtype="<u2").copy()
    return idx, words


def _kept_rank2(kept: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    k = np.ascontiguousarray(kept, dtype=np.uint16)
    if len(shape) == 1:
        return k.reshape(1, -1)
    if len(shape) == 2:
        return k.reshape(shape)
    return k.reshape(int(shape[0]), -1)


def encode_quantized(
    tensors: list[QuantizedTensor],
    out_path: str | Path,
    *,
    policy: HybridPolicy,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a HYBX file from already-quantized tensors. Returns rate stats."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aliases = dict(aliases or {})

    qref: dict[str, np.ndarray] = {t.name: t.q_ref for t in tensors}
    ref_sha = sha256_state_u16(qref)
    total_weights = sum(int(t.q_ref.size) for t in tensors)

    chunks: list[bytes] = []
    rel = 0
    specs: list[dict[str, Any]] = []
    mode_hist: dict[str, int] = {}
    pred_hist: dict[str, int] = {}
    trav_hist: dict[str, int] = {}
    n_tiles = n_xy = n_packed = 0
    map_bytes = flag_bytes = xy_payload = 0
    packed_whole = mant_blob = 0
    n_xy_sel = 0
    n_outliers = 0
    h95_payload = 0
    int_payload = 0
    raw_payload = 0
    h95_words = 0
    int_words = 0
    raw_words = 0
    family_words: dict[str, dict[str, int]] = {}
    kind_words: dict[str, int] = {}

    def _push(blob: bytes) -> tuple[int, int]:
        nonlocal rel
        off = rel
        chunks.append(blob)
        rel += len(blob)
        return off, len(blob)

    def _account_xy(enc: dict[str, Any]) -> None:
        nonlocal n_tiles, n_xy, n_packed, map_bytes, flag_bytes, xy_payload
        nonlocal packed_whole, mant_blob, n_xy_sel
        _merge_hist(mode_hist, enc["mode_hist"])
        _merge_hist(pred_hist, enc["pred_hist"])
        _merge_hist(trav_hist, enc["trav_hist"])
        n_tiles += int(enc["n_tiles"])
        n_xy += int(enc["n_xy_tiles"])
        n_packed += int(enc["n_packed_tiles"])
        map_bytes += int(enc["map_bytes"])
        flag_bytes += int(enc["flag_bytes"])
        xy_payload += int(enc["xy_payload_bytes"])
        packed_whole += int(enc["packed_whole_bytes"])
        mant_blob += int(enc["blob_bytes"])
        if enc["kind"] == "xy_sel":
            n_xy_sel += 1

    for qt in tensors:
        fam = family_words.setdefault(
            qt.family, {"n_words": 0, "n_outliers": 0, "kind": qt.kind}
        )
        fam["n_words"] += int(qt.q_ref.size)
        fam["n_outliers"] += int(qt.outlier_idx.size)
        kind_words[qt.kind] = kind_words.get(qt.kind, 0) + int(qt.q_ref.size)
        n_outliers += int(qt.outlier_idx.size)
        item: dict[str, Any] = {
            "name": qt.name,
            "shape": list(qt.shape),
            "rows": qt.rows,
            "cols": qt.cols,
            "kind": qt.kind,
            "family": qt.family,
            "keep": qt.keep,
            "bits": qt.bits,
            "group_size": qt.group_size,
            "n_groups": qt.n_groups,
            "n_weights": int(qt.q_ref.size),
            "n_outliers": int(qt.outlier_idx.size),
            "sha256_q_ref": sha256_u16(qt.q_ref),
            "mse": qt.mse,
            "spatial_score": qt.spatial_score,
            "n_spatial_overrides": qt.n_spatial_overrides,
        }

        if qt.kind == "bf16_raw":
            raw = np.ascontiguousarray(qt.q_ref, dtype="<u2").tobytes()
            item["raw_rel"], item["raw_len"] = _push(raw)
            item["codes_kind"] = "bf16_raw"
            item["n_tiles"] = 0
            item["n_xy_tiles"] = 0
            item["n_packed_tiles"] = 0
            raw_payload += len(raw)
            raw_words += int(qt.q_ref.size)
            specs.append(item)
            continue

        if qt.kind == "h95":
            fields = split_quantized(qt.q_ref, int(qt.keep))
            sign_b = pack_sign_bitplane(fields["sign"])
            exp_res = encode_exp_adaptive(fields["exp"])
            kept2 = _kept_rank2(fields["kept"], fields["shape"])
            enc = encode_array_selective(kept2, int(qt.keep))
            idx_b, words_b = _encode_outliers(
                fields["exception_indices"], fields["exception_words"]
            )
            item["sign_rel"], item["sign_len"] = _push(sign_b)
            item["exp_rel"], item["exp_len"] = _push(exp_res.blob)
            item["exp_meta"] = exp_res.as_dict()
            item["mant_rel"], item["mant_len"] = _push(enc["blob"])
            item["out_idx_rel"], item["out_idx_len"] = _push(idx_b)
            item["out_words_rel"], item["out_words_len"] = _push(words_b)
            item["n_outliers"] = int(fields["exception_indices"].size)
            item["codes_kind"] = enc["kind"]
            item["n_tiles"] = int(enc["n_tiles"])
            item["n_xy_tiles"] = int(enc["n_xy_tiles"])
            item["n_packed_tiles"] = int(enc["n_packed_tiles"])
            item["packed_whole_bytes"] = int(enc["packed_whole_bytes"])
            item["map_bytes"] = int(enc["map_bytes"])
            item["flag_bytes"] = int(enc["flag_bytes"])
            item["xy_payload_bytes"] = int(enc["xy_payload_bytes"])
            item["fallback_packed"] = bool(enc["fallback_packed"])
            h95_payload += (
                len(sign_b) + len(exp_res.blob) + len(enc["blob"]) + len(idx_b) + len(words_b)
            )
            h95_words += int(qt.q_ref.size)
            _account_xy(enc)
            specs.append(item)
            continue

        if qt.kind != "groupwise":
            raise ValueError(f"unknown kind {qt.kind}")

        scale_b = np.ascontiguousarray(qt.scales, dtype="<f2").tobytes()
        zp_b = np.ascontiguousarray(qt.zp, dtype=np.uint8).tobytes()
        enc = encode_array_selective(qt.codes, int(qt.bits))
        idx_b, words_b = _encode_outliers(qt.outlier_idx, qt.outlier_words)
        item["scale_rel"], item["scale_len"] = _push(scale_b)
        item["zp_rel"], item["zp_len"] = _push(zp_b)
        item["codes_rel"], item["codes_len"] = _push(enc["blob"])
        item["out_idx_rel"], item["out_idx_len"] = _push(idx_b)
        item["out_words_rel"], item["out_words_len"] = _push(words_b)
        item["codes_kind"] = enc["kind"]
        item["n_tiles"] = int(enc["n_tiles"])
        item["n_xy_tiles"] = int(enc["n_xy_tiles"])
        item["n_packed_tiles"] = int(enc["n_packed_tiles"])
        item["packed_whole_bytes"] = int(enc["packed_whole_bytes"])
        item["map_bytes"] = int(enc["map_bytes"])
        item["flag_bytes"] = int(enc["flag_bytes"])
        item["xy_payload_bytes"] = int(enc["xy_payload_bytes"])
        item["fallback_packed"] = bool(enc["fallback_packed"])
        int_payload += len(scale_b) + len(zp_b) + len(enc["blob"]) + len(idx_b) + len(words_b)
        int_words += int(qt.q_ref.size)
        _account_xy(enc)
        specs.append(item)

    payload = b"".join(chunks)

    def build_header_json(payload_start: int) -> bytes:
        tensors_out = []
        for spec in specs:
            item = {
                "name": spec["name"],
                "shape": spec["shape"],
                "rows": spec["rows"],
                "cols": spec["cols"],
                "kind": spec["kind"],
                "family": spec["family"],
                "keep": spec["keep"],
                "bits": spec["bits"],
                "group_size": spec["group_size"],
                "n_groups": spec["n_groups"],
                "n_weights": spec["n_weights"],
                "n_outliers": spec["n_outliers"],
                "sha256_q_ref": spec["sha256_q_ref"],
                "codes_kind": spec["codes_kind"],
                "n_tiles": spec["n_tiles"],
                "n_xy_tiles": spec["n_xy_tiles"],
                "n_packed_tiles": spec["n_packed_tiles"],
            }
            if spec["kind"] == "bf16_raw":
                item["raw_off"] = payload_start + spec["raw_rel"]
                item["raw_len"] = spec["raw_len"]
            elif spec["kind"] == "h95":
                item["sign_off"] = payload_start + spec["sign_rel"]
                item["sign_len"] = spec["sign_len"]
                item["exp_off"] = payload_start + spec["exp_rel"]
                item["exp_len"] = spec["exp_len"]
                item["exp_meta"] = spec.get("exp_meta") or {}
                item["mant_off"] = payload_start + spec["mant_rel"]
                item["mant_len"] = spec["mant_len"]
                item["out_idx_off"] = payload_start + spec["out_idx_rel"]
                item["out_idx_len"] = spec["out_idx_len"]
                item["out_words_off"] = payload_start + spec["out_words_rel"]
                item["out_words_len"] = spec["out_words_len"]
                item["packed_whole_bytes"] = spec.get("packed_whole_bytes", 0)
                item["map_bytes"] = spec.get("map_bytes", 0)
                item["flag_bytes"] = spec.get("flag_bytes", 0)
                item["xy_payload_bytes"] = spec.get("xy_payload_bytes", 0)
                item["fallback_packed"] = spec.get("fallback_packed", True)
            else:
                item["scale_off"] = payload_start + spec["scale_rel"]
                item["scale_len"] = spec["scale_len"]
                item["zp_off"] = payload_start + spec["zp_rel"]
                item["zp_len"] = spec["zp_len"]
                item["codes_off"] = payload_start + spec["codes_rel"]
                item["codes_len"] = spec["codes_len"]
                item["out_idx_off"] = payload_start + spec["out_idx_rel"]
                item["out_idx_len"] = spec["out_idx_len"]
                item["out_words_off"] = payload_start + spec["out_words_rel"]
                item["out_words_len"] = spec["out_words_len"]
                item["packed_whole_bytes"] = spec.get("packed_whole_bytes", 0)
                item["map_bytes"] = spec.get("map_bytes", 0)
                item["flag_bytes"] = spec.get("flag_bytes", 0)
                item["xy_payload_bytes"] = spec.get("xy_payload_bytes", 0)
                item["fallback_packed"] = spec.get("fallback_packed", True)
            tensors_out.append(item)
        header_obj = {
            "format": "HYBX",
            "version": VERSION,
            "model_id": model_id,
            "policy": policy.as_dict(),
            "dtype": "bf16_uint16",
            "n_tensors": len(tensors_out),
            "n_alias_tensors": len(aliases),
            "total_weights": total_weights,
            "endianness": "little",
            "checksum_algo": "sha256",
            "tile": [TILE, TILE],
            "xy_default": "packed",
            "hard_override": "selective_xy_only_if_strictly_smaller",
            "s1_compatible": False,
            "pr17_compatible": False,
            "sha256_quantized_reference": ref_sha,
            "aliases": aliases,
            "tensors": tensors_out,
            "alignment": ALIGN,
            "payload_start": payload_start,
        }
        return json.dumps(header_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    payload_start = ALIGN
    blob = b""
    for _ in range(6):
        header_json = build_header_json(payload_start)
        prefix = _PREFIX.pack(
            MAGIC, VERSION, FLAGS_LITTLE_ENDIAN, CHECKSUM_SHA256, b"\x00\x00\x00", len(header_json)
        )
        header_end = len(prefix) + len(header_json)
        new_start = _align(header_end, ALIGN)
        pad = new_start - header_end
        if new_start == payload_start:
            blob = prefix + header_json + (b"\x00" * pad) + payload
            break
        payload_start = new_start
    else:
        raise RuntimeError("failed to stabilize HYBX header alignment")

    out_path.write_bytes(blob)
    file_bytes = len(blob)
    if file_bytes != out_path.stat().st_size:
        raise RuntimeError("written size != path size")
    actual_bpw = (file_bytes * 8.0 / total_weights) if total_weights else 0.0
    packed_bpw = (packed_whole * 8.0 / total_weights) if total_weights else 0.0
    mant_bpw = (mant_blob * 8.0 / total_weights) if total_weights else 0.0
    illegal = [m for m in mode_hist if m not in MODE_NAMES.values()]
    if illegal:
        raise RuntimeError(f"non-matrix XY modes in histogram: {illegal}")

    mix = {
        "h95_payload_bytes": h95_payload,
        "int_payload_bytes": int_payload,
        "raw_payload_bytes": raw_payload,
        "h95_words": h95_words,
        "int_words": int_words,
        "raw_words": raw_words,
        "h95_word_frac": (h95_words / total_weights) if total_weights else 0.0,
        "int_word_frac": (int_words / total_weights) if total_weights else 0.0,
        "h95_payload_frac": (h95_payload / file_bytes) if file_bytes else 0.0,
        "int_payload_frac": (int_payload / file_bytes) if file_bytes else 0.0,
        "kind_words": kind_words,
    }

    return {
        "path": str(out_path),
        "file_bytes": file_bytes,
        "header_bytes": payload_start,
        "payload_stream_bytes": len(payload),
        "overhead_bytes": file_bytes - len(payload),
        "n_weights": total_weights,
        "n_tensors": len(specs),
        "n_tiles": n_tiles,
        "n_xy_tiles": n_xy,
        "n_packed_tiles": n_packed,
        "xy_tile_frac": (n_xy / n_tiles) if n_tiles else 0.0,
        "n_xy_sel_tensors": n_xy_sel,
        "n_outliers": n_outliers,
        "actual_bpw": round(actual_bpw, 6),
        "sha256_quantized_reference": ref_sha,
        "sha256_file": _sha256_hex(blob),
        "xy_flag_bytes": flag_bytes,
        "xy_map_bytes": map_bytes,
        "xy_payload_bytes": xy_payload,
        "mant_blob_bytes": mant_blob,
        "mant_blob_bpw": round(mant_bpw, 6),
        "packed_whole_bytes": packed_whole,
        "packed_whole_bpw": round(packed_bpw, 6),
        "saved_vs_packed_code_bytes": packed_whole - mant_blob,
        "xy_mode_histogram": mode_hist,
        "xy_pred_histogram": pred_hist,
        "xy_trav_histogram": trav_hist,
        "family_words": family_words,
        "byte_mix": mix,
        "product_default_packed": True,
        "s1_compatible": False,
        "pr17_compatible": False,
        "policy": policy.name,
        "container_version": VERSION,
    }


def read_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        prefix = f.read(_PREFIX.size)
        magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        if ver != VERSION:
            raise ValueError(f"unsupported version {ver}")
        return json.loads(f.read(hdr_len).decode("utf-8"))


def _slice(data: memoryview, off: int, ln: int, name: str, label: str) -> memoryview:
    if off < 0 or ln < 0 or off + ln > len(data):
        raise ValueError(f"{label} payload out of range for {name}")
    return data[off : off + ln]


def _decode_one(spec: Mapping[str, Any], data: memoryview) -> np.ndarray:
    kind = spec["kind"]
    shape = tuple(int(x) for x in spec["shape"])
    name = spec["name"]
    if kind == "bf16_raw":
        off, ln = int(spec["raw_off"]), int(spec["raw_len"])
        raw = _slice(data, off, ln, name, "raw")
        need = int(np.prod(shape)) * 2
        if ln != need:
            raise ValueError(f"raw length {ln} != {need} for {name}")
        return np.frombuffer(raw, dtype="<u2").reshape(shape).copy()

    if kind == "h95":
        n = int(spec["n_weights"])
        keep = int(spec["keep"])
        sign = unpack_sign_bitplane(
            bytes(_slice(data, int(spec["sign_off"]), int(spec["sign_len"]), name, "sign")),
            n,
        )
        exp, _mode = decode_exp_blob(
            bytes(_slice(data, int(spec["exp_off"]), int(spec["exp_len"]), name, "exp")),
            n,
        )
        rows, cols = int(spec["rows"]), int(spec["cols"])
        blob = bytes(_slice(data, int(spec["mant_off"]), int(spec["mant_len"]), name, "mant"))
        kept2 = decode_array_selective(blob, rows=rows, cols=cols, nbits=keep, n_nodes=n)
        words = join_uniform(sign, exp, kept2.ravel()[:n], keep, shape)
        n_out = int(spec.get("n_outliers") or 0)
        if n_out:
            idx_b = bytes(
                _slice(data, int(spec["out_idx_off"]), int(spec["out_idx_len"]), name, "h95-idx")
            )
            words_b = bytes(
                _slice(
                    data, int(spec["out_words_off"]), int(spec["out_words_len"]), name, "h95-words"
                )
            )
            oidx, owords = _decode_outliers(idx_b, words_b, n_out)
            flat = words.ravel().copy()
            if np.any(oidx < 0) or np.any(oidx >= flat.size):
                raise ValueError(f"h95 exception index out of range for {name}")
            flat[oidx] = owords
            words = flat.reshape(shape)
        return words.astype(np.uint16)

    if kind != "groupwise":
        raise ValueError(f"unknown tensor kind {kind}")

    bits = int(spec["bits"])
    rows, cols = int(spec["rows"]), int(spec["cols"])
    group_size = int(spec["group_size"])
    n_groups = int(spec["n_groups"])
    s_ln = int(spec["scale_len"])
    if s_ln != n_groups * 2:
        raise ValueError(f"scale length {s_ln} != {n_groups * 2}")
    scales = np.frombuffer(
        _slice(data, int(spec["scale_off"]), s_ln, name, "scale"), dtype="<f2"
    ).copy()
    z_ln = int(spec["zp_len"])
    if z_ln != n_groups:
        raise ValueError(f"zp length {z_ln} != {n_groups}")
    zp = np.frombuffer(_slice(data, int(spec["zp_off"]), z_ln, name, "zp"), dtype=np.uint8).copy()
    blob = bytes(_slice(data, int(spec["codes_off"]), int(spec["codes_len"]), name, "codes"))
    codes = decode_array_selective(blob, rows=rows, cols=cols, nbits=bits)
    n_out = int(spec.get("n_outliers") or 0)
    idx_b = bytes(_slice(data, int(spec["out_idx_off"]), int(spec["out_idx_len"]), name, "out-idx"))
    words_b = bytes(
        _slice(data, int(spec["out_words_off"]), int(spec["out_words_len"]), name, "out-words")
    )
    oidx, owords = _decode_outliers(idx_b, words_b, n_out)
    return dequantize_int(
        bits=bits,
        codes=codes,
        scales=scales,
        zp=zp,
        rows=rows,
        cols=cols,
        group_size=group_size,
        shape=shape,
        outlier_idx=oidx,
        outlier_words=owords,
    )


def decode_container(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    if len(data) != path.stat().st_size:
        raise ValueError("read size != file size")
    magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    header = json.loads(data[16 : 16 + hdr_len].decode("utf-8"))
    if int(header.get("version", ver)) != int(ver):
        raise ValueError("header/wire version mismatch")
    if header.get("s1_compatible") or header.get("pr17_compatible"):
        raise ValueError("HYBX must not claim S1 or PR #17 compatibility")
    mv = memoryview(data)
    tensors: dict[str, np.ndarray] = {}
    n_spec = len(header["tensors"])
    for i, spec in enumerate(header["tensors"], 1):
        if i == 1 or i == n_spec or i % 40 == 0:
            print(f" hybx-decode {i}/{n_spec} {spec['name']}", flush=True)
        tensors[spec["name"]] = _decode_one(spec, mv)
    for alias, canon in (header.get("aliases") or {}).items():
        if canon in tensors:
            tensors[alias] = tensors[canon]
    last = 0
    for spec in header["tensors"]:
        for key in (
            "raw_off",
            "sign_off",
            "exp_off",
            "mant_off",
            "scale_off",
            "zp_off",
            "codes_off",
            "out_idx_off",
            "out_words_off",
        ):
            ln_key = key.replace("_off", "_len")
            if key in spec:
                last = max(last, int(spec[key]) + int(spec.get(ln_key, 0)))
    if last > len(data):
        raise ValueError("payload overruns file")
    return {
        "header": header,
        "tensors": tensors,
        "file_bytes": len(data),
        "sha256_file": _sha256_hex(data),
        "sha256_decoded": sha256_state_u16(tensors),
    }


def verify_decoded_against_reference(
    decoded: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    missing = [k for k in reference if k not in decoded]
    extra = [k for k in decoded if k not in reference]
    diffs = []
    for name, ref in reference.items():
        if name not in decoded:
            continue
        a = np.ascontiguousarray(decoded[name], dtype=np.uint16)
        b = np.ascontiguousarray(ref, dtype=np.uint16)
        if a.shape != b.shape or not np.array_equal(a, b):
            n = int(min(a.size, b.size))
            ndiff = int(np.count_nonzero(a.ravel()[:n] != b.ravel()[:n])) if n else -1
            diffs.append(
                {
                    "name": name,
                    "n_diff": ndiff,
                    "dec_shape": list(a.shape),
                    "ref_shape": list(b.shape),
                }
            )
    ok = not missing and not extra and not diffs
    return {
        "ok": ok,
        "missing": missing,
        "extra": extra,
        "n_diff_tensors": len(diffs),
        "diffs": diffs[:12],
        "sha_decoded": sha256_state_u16(decoded),
        "sha_reference": sha256_state_u16(reference),
    }


def main_decode_cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="HYBX hybrid decode-only tool")
    p.add_argument("container", type=Path)
    p.add_argument("--dump-header", action="store_true")
    p.add_argument("--expect-sha", type=str, default="")
    args = p.parse_args(argv)
    if args.dump_header:
        h = read_header(args.container)
        slim = {k: h[k] for k in h if k != "tensors"}
        json.dump(slim, sys.stdout, indent=2)
        print()
        return 0
    result = decode_container(args.container)
    h = result["header"]
    tw = int(h["total_weights"])
    print(f"decoded n_tensors={h['n_tensors']} n_weights={tw}")
    print(f"file_bytes={result['file_bytes']} actual_bpw={result['file_bytes'] * 8 / tw:.6f}")
    print(f"sha256_quantized_reference={h['sha256_quantized_reference']}")
    print(f"sha256_decoded={result['sha256_decoded']}")
    if args.expect_sha:
        ok = (
            result["sha256_decoded"] == args.expect_sha
            and h["sha256_quantized_reference"] == args.expect_sha
        )
        print(f"sha_match={ok}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main_decode_cli())
