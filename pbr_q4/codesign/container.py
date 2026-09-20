"""PQ4X physical container: groupwise Q-ref + selective 256-node X/Y.

Layout (little-endian)::

    [0:4]   magic b"PQ4X"
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
* ``groupwise``: float16 scales + packed-or-XY code blob + optional
  sparse outliers (u32 indices + BF16 words).

X/Y is emitted only when ``encode_array_selective`` reports a strictly
smaller complete blob than packed-K (maps + flags included). Packed
codes are the product default. This is a **new** quantized reference —
not H95Q-S1 compatible.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pbr_h95.container_h95q import sha256_state_u16, sha256_u16
from pbr_q4.codesign.policy import CodesignPolicy
from pbr_q4.codesign.quantize import (
    QuantizedTensor,
    dequantize_to_bf16,
    quantize_tensor,
)
from pbr_q4.const import MODE_NAMES, TILE
from pbr_q4.selective import decode_array_selective, encode_array_selective

MAGIC = b"PQ4X"
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


def encode_quantized(
    tensors: list[QuantizedTensor],
    out_path: str | Path,
    *,
    policy: CodesignPolicy,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a PQ4X file from already-quantized tensors. Returns rate stats."""
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
    family_words: dict[str, dict[str, int]] = {}

    def _push(blob: bytes) -> tuple[int, int]:
        nonlocal rel
        off = rel
        chunks.append(blob)
        rel += len(blob)
        return off, len(blob)

    for qt in tensors:
        fam = family_words.setdefault(qt.family, {"n_words": 0, "n_outliers": 0})
        fam["n_words"] += int(qt.q_ref.size)
        fam["n_outliers"] += int(qt.outlier_idx.size)
        n_outliers += int(qt.outlier_idx.size)
        item: dict[str, Any] = {
            "name": qt.name,
            "shape": list(qt.shape),
            "rows": qt.rows,
            "cols": qt.cols,
            "kind": qt.kind,
            "family": qt.family,
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
            specs.append(item)
            continue

        scale_b = np.ascontiguousarray(qt.scales, dtype="<f2").tobytes()
        item["scale_rel"], item["scale_len"] = _push(scale_b)
        enc = encode_array_selective(qt.codes, int(qt.bits))
        item["codes_rel"], item["codes_len"] = _push(enc["blob"])
        item["codes_kind"] = enc["kind"]
        item["rows"] = enc["rows"]
        item["cols"] = enc["cols"]
        item["n_tiles"] = int(enc["n_tiles"])
        item["n_xy_tiles"] = int(enc["n_xy_tiles"])
        item["n_packed_tiles"] = int(enc["n_packed_tiles"])
        item["packed_whole_bytes"] = int(enc["packed_whole_bytes"])
        item["map_bytes"] = int(enc["map_bytes"])
        item["flag_bytes"] = int(enc["flag_bytes"])
        item["xy_payload_bytes"] = int(enc["xy_payload_bytes"])
        item["fallback_packed"] = bool(enc["fallback_packed"])
        idx_b, words_b = _encode_outliers(qt.outlier_idx, qt.outlier_words)
        item["out_idx_rel"], item["out_idx_len"] = _push(idx_b)
        item["out_words_rel"], item["out_words_len"] = _push(words_b)
        specs.append(item)

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
            else:
                item["scale_off"] = payload_start + spec["scale_rel"]
                item["scale_len"] = spec["scale_len"]
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
            "format": "PQ4X",
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
        raise RuntimeError("failed to stabilize PQ4X header alignment")

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
        "product_default_packed": True,
        "s1_compatible": False,
        "policy": policy.name,
        "container_version": VERSION,
    }


def encode_container(
    state: Mapping[str, np.ndarray],
    out_path: str | Path,
    *,
    policy: CodesignPolicy,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    aliases: Mapping[str, str] | None = None,
    bits_fn=None,
    family_fn=None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[QuantizedTensor]]:
    """Quantize *state* (name → BF16 uint16) and write PQ4X."""
    from pbr_q4.codesign.policy import bits_for_name, family_label

    qts: list[QuantizedTensor] = []
    qref: dict[str, np.ndarray] = {}
    names = sorted(state.keys())
    for i, name in enumerate(names, 1):
        words = np.ascontiguousarray(state[name], dtype=np.uint16)
        bits = bits_fn(name) if bits_fn else bits_for_name(name, policy)
        fam = family_fn(name) if family_fn else family_label(name, policy)
        if i == 1 or i == len(names) or i % 20 == 0:
            print(f"  quantize {i}/{len(names)} {name} bits={bits}", flush=True)
        qt = quantize_tensor(words, bits, policy, name=name, family=fam)
        qts.append(qt)
        qref[name] = qt.q_ref
    stats = encode_quantized(qts, out_path, policy=policy, model_id=model_id, aliases=aliases)
    return stats, qref, qts


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


def _decode_one(spec: Mapping[str, Any], data: memoryview) -> np.ndarray:
    kind = spec["kind"]
    shape = tuple(int(x) for x in spec["shape"])
    if kind == "bf16_raw":
        off, ln = int(spec["raw_off"]), int(spec["raw_len"])
        if off < 0 or ln < 0 or off + ln > len(data):
            raise ValueError(f"raw payload out of range for {spec['name']}")
        need = int(np.prod(shape)) * 2
        if ln != need:
            raise ValueError(f"raw length {ln} != {need} for {spec['name']}")
        return np.frombuffer(data[off : off + ln], dtype="<u2").reshape(shape).copy()
    if kind != "groupwise":
        raise ValueError(f"unknown tensor kind {kind}")

    bits = int(spec["bits"])
    rows, cols = int(spec["rows"]), int(spec["cols"])
    group_size = int(spec["group_size"])
    n_groups = int(spec["n_groups"])

    s_off, s_ln = int(spec["scale_off"]), int(spec["scale_len"])
    if s_ln != n_groups * 2:
        raise ValueError(f"scale length {s_ln} != {n_groups * 2}")
    scales = np.frombuffer(data[s_off : s_off + s_ln], dtype="<f2").copy()

    c_off, c_ln = int(spec["codes_off"]), int(spec["codes_len"])
    if c_off < 0 or c_ln < 0 or c_off + c_ln > len(data):
        raise ValueError(f"codes payload out of range for {spec['name']}")
    blob = bytes(data[c_off : c_off + c_ln])
    codes = decode_array_selective(blob, rows=rows, cols=cols, nbits=bits)

    n_out = int(spec.get("n_outliers") or 0)
    idx_b = bytes(data[int(spec["out_idx_off"]) : int(spec["out_idx_off"]) + int(spec["out_idx_len"])])
    words_b = bytes(data[int(spec["out_words_off"]) : int(spec["out_words_off"]) + int(spec["out_words_len"])])
    oidx, owords = _decode_outliers(idx_b, words_b, n_out)
    return dequantize_to_bf16(
        bits=bits,
        codes=codes,
        scales=scales,
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
    if header.get("s1_compatible"):
        raise ValueError("PQ4X must not claim S1 compatibility")
    mv = memoryview(data)
    tensors: dict[str, np.ndarray] = {}
    n_spec = len(header["tensors"])
    for i, spec in enumerate(header["tensors"], 1):
        if i == 1 or i == n_spec or i % 40 == 0:
            print(f"  pq4x-decode {i}/{n_spec} {spec['name']}", flush=True)
        tensors[spec["name"]] = _decode_one(spec, mv)
    for alias, canon in (header.get("aliases") or {}).items():
        if canon in tensors:
            tensors[alias] = tensors[canon]
    # Trailing-byte / overlap check: payload must end at file end (padding only in header).
    last = 0
    for spec in header["tensors"]:
        for key in ("raw_off", "scale_off", "codes_off", "out_idx_off", "out_words_off"):
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
            diffs.append({"name": name, "n_diff": ndiff, "dec_shape": list(a.shape), "ref_shape": list(b.shape)})
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

    p = argparse.ArgumentParser(description="PQ4X codesign decode-only tool")
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
        ok = result["sha256_decoded"] == args.expect_sha and h["sha256_quantized_reference"] == args.expect_sha
        print(f"sha_match={ok}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main_decode_cli())
