"""H95Q physical container — magic ``H95Q``, versions 1 and 2.

Physical layout
---------------
::

    [0:4]   magic b"H95Q"
    [4:6]   version u16le (1 = raw-u8 exp, 2 = adaptive exact exp)
    [6:8]   flags u16le (bit0 = little-endian)
    [8:9]   checksum_algo u8 (1 = SHA-256)
    [9:12]  reserved
    [12:16] header_json_len u32le
    [16:]   header JSON (utf-8)
    then pad to 64-byte alignment
    then payload blobs; absolute file offsets live in the header

Streams per unique tensor
-------------------------
* sign bitplane (1 bit/weight)
* exponents:
    - v1: **raw u8** (8 bits/weight)
    - v2: adaptive exact coding (RAW8 / rANS / Huffman / delta-rANS / run-rANS)
      selected per tensor by minimum complete physical section bytes
* mantissa K-bit packed (K in 0..7), MSB-first, pad to byte per stream
* embed tiers: int8 row-keep table + mantissa groups keyed by K
* optional raw-u16 exception words when low mantissa bits survive (NaN payloads)

Decode-only imports: numpy, this module, ``pbr_h95.bitpack``, ``pbr_h95.exp_codec`` (v2),
``pbr_core.bf16``.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pbr_core.bf16 import join_components, split_components
from pbr_h95.bitpack import (
    pack_exp_u8,
    pack_kept_mantissas,
    pack_sign_bitplane,
    unpack_exp_u8,
    unpack_kept_mantissas,
    unpack_sign_bitplane,
)
from pbr_h95.exp_codec import (
    EXP_RAW8,
    MODE_NAMES,
    decode_exp_blob,
    encode_exp_adaptive,
    encode_raw8,
)

MAGIC = b"H95Q"
VERSION = 2
VERSION_V1 = 1
VERSION_V2 = 2
SUPPORTED_VERSIONS = {VERSION_V1, VERSION_V2}
CHECKSUM_SHA256 = 1
FLAGS_LITTLE_ENDIAN = 0x0001
ALIGN = 64
_PREFIX = struct.Struct("<4sHHB3sI")


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_u16(words: np.ndarray) -> str:
    arr = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    return _sha256_hex(arr.astype("<u2", copy=False).tobytes())


def sha256_state_u16(tensors: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    seen: set[int] = set()
    for name in sorted(tensors.keys()):
        w = np.ascontiguousarray(tensors[name], dtype=np.uint16).ravel()
        ptr = int(w.__array_interface__["data"][0])
        if ptr in seen:
            continue
        seen.add(ptr)
        h.update(w.astype("<u2", copy=False).tobytes())
    return h.hexdigest()


def _words_from_fields(sign: np.ndarray, exp: np.ndarray, kept: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        mant = np.zeros(np.asarray(sign).shape, dtype=np.uint16)
    elif k >= 7:
        mant = np.asarray(kept, dtype=np.uint16) & np.uint16(0x7F)
    else:
        removed = 7 - k
        mant = (np.asarray(kept, dtype=np.uint16) << np.uint16(removed)) & np.uint16(0x7F)
    return join_components(sign, exp, mant)


def _encode_one(
    words: np.ndarray,
    *,
    keep: int | None = None,
    row_keeps: np.ndarray | None = None,
    version: int = VERSION,
) -> dict[str, Any]:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    shape = tuple(int(x) for x in w.shape)
    flat = w.ravel()
    n = int(flat.size)
    sign, exp, mant = split_components(flat)
    if int(version) >= VERSION_V2:
        exp_res = encode_exp_adaptive(exp)
        exp_blob = exp_res.blob
        exp_meta = exp_res.as_dict()
    else:
        exp_blob = pack_exp_u8(exp)
        exp_meta = {
            "mode": EXP_RAW8,
            "mode_name": "EXP_RAW8",
            "n_weights": n,
            "complete_bytes": len(exp_blob),
            "complete_bpw": 8.0 if n else 0.0,
            "table_bytes": 0,
            "stream_bytes": len(exp_blob),
            "mode_id_bytes": 0,
            "length_field_bytes": 0,
        }
    out: dict[str, Any] = {
        "shape": shape,
        "n_weights": n,
        "sign": pack_sign_bitplane(sign),
        "exp": exp_blob,
        "exp_meta": exp_meta,
    }

    if row_keeps is not None:
        if len(shape) != 2:
            raise ValueError("row_keeps requires rank-2 tensor")
        rk = np.asarray(row_keeps, dtype=np.int8)
        if int(rk.shape[0]) != shape[0]:
            raise ValueError("row_keeps / rows mismatch")
        rows, cols = shape
        mant2 = mant.reshape(rows, cols)
        groups: dict[str, Any] = {}
        for k in sorted({int(x) for x in rk.tolist()}):
            idx = np.flatnonzero(rk == k).astype(np.int32)
            block = mant2[idx].reshape(-1)
            store_k = 7 if k >= 7 else max(int(k), 0)
            payload = b"" if store_k == 0 else pack_kept_mantissas(block, store_k)
            groups[str(k)] = {
                "k": store_k,
                "n_rows": int(idx.size),
                "n_weights": int(block.size),
                "row_indices": idx,
                "payload": payload,
            }
        # Reconstruct what pack/unpack would yield and store any dirty lower
        # bits (NaN payloads, etc.) as raw u16 exceptions — same rule as uniform.
        recon = np.empty_like(flat)
        mant2r = mant.reshape(rows, cols)
        sign2 = sign.reshape(rows, cols)
        exp2 = exp.reshape(rows, cols)
        for k_str, g in groups.items():
            k = int(g["k"])
            idx = g["row_indices"]
            block = mant2r[idx].reshape(-1)
            if k <= 0:
                kept = np.zeros(block.size, dtype=np.uint16)
            elif k >= 7:
                kept = block & np.uint16(0x7F)
            else:
                removed = 7 - k
                kept = (block >> np.uint16(removed)) & np.uint16((1 << k) - 1)
            s = sign2[idx].ravel()
            e = exp2[idx].ravel()
            words = _words_from_fields(s, e, kept, k).reshape(idx.size, cols)
            # scatter into flat via row indices
            for local_i, row_i in enumerate(idx.tolist()):
                recon[row_i * cols : (row_i + 1) * cols] = words[local_i]
        dirty = recon != flat
        exc_idx = np.flatnonzero(dirty).astype(np.int64)
        out.update(
            mode="embed_tiers",
            row_keeps=rk,
            mant_groups=groups,
            base_keep=None,
            exception_indices=exc_idx,
            exception_words=flat[exc_idx].astype(np.uint16) if exc_idx.size else np.zeros(0, dtype=np.uint16),
        )
        return out

    if keep is None:
        raise ValueError("keep or row_keeps required")
    k = int(keep)
    if k < 0 or k > 7:
        raise ValueError(f"keep out of range: {k}")
    if k < 7:
        lower = np.uint16((1 << (7 - k)) - 1)
        dirty = (mant & lower) != 0
        exc_idx = np.flatnonzero(dirty).astype(np.int64)
    else:
        exc_idx = np.zeros(0, dtype=np.int64)
    mant_b = b"" if k == 0 else pack_kept_mantissas(mant, k)
    out.update(
        mode="uniform",
        base_keep=k,
        mant=mant_b,
        exception_indices=exc_idx,
        exception_words=flat[exc_idx].astype(np.uint16) if exc_idx.size else np.zeros(0, dtype=np.uint16),
    )
    return out


def _decode_one(spec: dict[str, Any], data: memoryview, *, version: int = VERSION_V1) -> np.ndarray:
    n = int(spec["n_weights"])
    shape = tuple(spec["shape"])
    sign = unpack_sign_bitplane(bytes(data[spec["sign_off"] : spec["sign_off"] + spec["sign_len"]]), n)
    exp_blob = bytes(data[spec["exp_off"] : spec["exp_off"] + spec["exp_len"]])
    if int(version) >= VERSION_V2:
        exp, _mode = decode_exp_blob(exp_blob, n)
    else:
        exp = unpack_exp_u8(exp_blob, n)

    if spec["mode"] == "uniform":
        k = int(spec["base_keep"])
        mant_blob = bytes(data[spec["mant_off"] : spec["mant_off"] + spec["mant_len"]])
        if k == 0:
            words = _words_from_fields(sign, exp, np.zeros(n, dtype=np.uint16), 0)
        else:
            kept = unpack_kept_mantissas(mant_blob, n, k)
            words = _words_from_fields(sign, exp, kept, k)
        exc = spec.get("exceptions") or {}
        if int(exc.get("n", 0)):
            idx = np.frombuffer(bytes(data[exc["idx_off"] : exc["idx_off"] + exc["idx_len"]]), dtype="<i8")
            raw = np.frombuffer(bytes(data[exc["words_off"] : exc["words_off"] + exc["words_len"]]), dtype="<u2")
            words = words.copy()
            words[idx] = raw
        return words.reshape(shape)

    if spec["mode"] == "embed_tiers":
        rows, cols = shape
        rk = np.frombuffer(
            bytes(data[spec["row_keeps_off"] : spec["row_keeps_off"] + spec["row_keeps_len"]]),
            dtype=np.int8,
        )
        if rk.size != rows:
            raise ValueError("row_keeps size mismatch")
        out = np.empty((rows, cols), dtype=np.uint16)
        sign2 = sign.reshape(rows, cols)
        exp2 = exp.reshape(rows, cols)
        for g in spec["mant_groups"]:
            k = int(g["k"])
            idx = np.asarray(g["row_indices"], dtype=np.int32)
            nw = int(g["n_weights"])
            blob = bytes(data[g["off"] : g["off"] + g["len"]])
            s = sign2[idx].ravel()
            e = exp2[idx].ravel()
            if k == 0:
                block = _words_from_fields(s, e, np.zeros(nw, dtype=np.uint16), 0)
            else:
                kept = unpack_kept_mantissas(blob, nw, k)
                block = _words_from_fields(s, e, kept, k)
            out[idx] = block.reshape(idx.size, cols)
        words = out.ravel()
        exc = spec.get("exceptions") or {}
        if int(exc.get("n", 0)):
            eidx = np.frombuffer(bytes(data[exc["idx_off"] : exc["idx_off"] + exc["idx_len"]]), dtype="<i8")
            raw = np.frombuffer(bytes(data[exc["words_off"] : exc["words_off"] + exc["words_len"]]), dtype="<u2")
            words = words.copy()
            words[eidx] = raw
            out = words.reshape(shape)
        return out

    raise ValueError(f"unknown mode {spec['mode']!r}")


def encode_container(
    out_path: str | Path,
    tensors: Mapping[str, np.ndarray],
    *,
    keep_map: Mapping[str, int],
    model_id: str,
    model_hash: str = "",
    candidate: str = "",
    precision_map: dict[str, Any] | None = None,
    embed_name: str | None = None,
    embed_row_keeps: np.ndarray | None = None,
    dtype_tag: str = "bf16",
    version: int = VERSION,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    version = int(version)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported H95Q version {version}")

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[int] = set()
    aliases: dict[str, str] = {}
    ptr_canon: dict[int, str] = {}
    for name in sorted(tensors.keys()):
        w = np.ascontiguousarray(tensors[name], dtype=np.uint16)
        ptr = int(w.__array_interface__["data"][0])
        if ptr in seen:
            aliases[name] = ptr_canon[ptr]
            continue
        seen.add(ptr)
        ptr_canon[ptr] = name
        unique.append((name, w))

    encoded: list[dict[str, Any]] = []
    for name, w in unique:
        if embed_name and name == embed_name and embed_row_keeps is not None:
            enc = _encode_one(w, row_keeps=embed_row_keeps, version=version)
        else:
            k = int(keep_map.get(name, 7))
            enc = _encode_one(w, keep=max(k, 0), version=version)
        enc["name"] = name
        enc["sha256"] = sha256_u16(w)
        encoded.append(enc)

    ref_sha = sha256_state_u16({n: t for n, t in unique})
    total_weights = sum(int(e["n_weights"]) for e in encoded)

    chunks: list[bytes] = []

    def add(b: bytes) -> tuple[int, int]:
        off = sum(len(c) for c in chunks)
        chunks.append(b)
        return off, len(b)

    specs: list[dict[str, Any]] = []
    for enc in encoded:
        spec: dict[str, Any] = {
            "name": enc["name"],
            "shape": list(enc["shape"]),
            "layout": "row_major",
            "mode": enc["mode"],
            "base_keep": enc.get("base_keep"),
            "n_weights": enc["n_weights"],
            "sha256_q_ref": enc["sha256"],
            "exp_meta": enc.get("exp_meta") or {},
        }
        so, sl = add(enc["sign"])
        eo, el = add(enc["exp"])
        spec["sign_rel"], spec["sign_len"] = so, sl
        spec["exp_rel"], spec["exp_len"] = eo, el
        if version >= VERSION_V2:
            spec["exp_mode"] = int(spec["exp_meta"].get("mode", EXP_RAW8))
            spec["exp_mode_name"] = str(spec["exp_meta"].get("mode_name", "EXP_RAW8"))

        if enc["mode"] == "uniform":
            mo, ml = add(enc["mant"])
            spec["mant_rel"], spec["mant_len"] = mo, ml
            ei = enc["exception_indices"]
            ew = enc["exception_words"]
            if ei.size:
                ib = np.ascontiguousarray(ei, dtype="<i8").tobytes()
                wb = np.ascontiguousarray(ew, dtype="<u2").tobytes()
                io, il = add(ib)
                wo, wl = add(wb)
                spec["exceptions"] = {
                    "n": int(ei.size),
                    "idx_rel": io,
                    "idx_len": il,
                    "words_rel": wo,
                    "words_len": wl,
                }
            else:
                spec["exceptions"] = {"n": 0}
        else:
            rk_b = np.ascontiguousarray(enc["row_keeps"], dtype=np.int8).tobytes()
            ro, rl = add(rk_b)
            spec["row_keeps_rel"], spec["row_keeps_len"] = ro, rl
            gout = []
            for k_str, g in enc["mant_groups"].items():
                po, pl = add(g["payload"])
                gout.append(
                    {
                        "policy_k": int(k_str),
                        "k": g["k"],
                        "n_rows": g["n_rows"],
                        "n_weights": g["n_weights"],
                        "row_indices": g["row_indices"].astype(int).tolist(),
                        "rel": po,
                        "len": pl,
                    }
                )
            spec["mant_groups"] = gout
            ei = enc.get("exception_indices", np.zeros(0, dtype=np.int64))
            ew = enc.get("exception_words", np.zeros(0, dtype=np.uint16))
            if getattr(ei, "size", 0):
                ib = np.ascontiguousarray(ei, dtype="<i8").tobytes()
                wb = np.ascontiguousarray(ew, dtype="<u2").tobytes()
                io, il = add(ib)
                wo, wl = add(wb)
                spec["exceptions"] = {
                    "n": int(ei.size),
                    "idx_rel": io,
                    "idx_len": il,
                    "words_rel": wo,
                    "words_len": wl,
                }
            else:
                spec["exceptions"] = {"n": 0}
        specs.append(spec)

    payload = b"".join(chunks)
    payload_stream_bytes = len(payload)

    def build_header_json(payload_start: int) -> bytes:
        tensors_out = []
        for spec in specs:
            item = {
                "name": spec["name"],
                "shape": spec["shape"],
                "layout": spec["layout"],
                "mode": spec["mode"],
                "base_keep": spec["base_keep"],
                "n_weights": spec["n_weights"],
                "sha256_q_ref": spec["sha256_q_ref"],
                "sign_off": payload_start + spec["sign_rel"],
                "sign_len": spec["sign_len"],
                "exp_off": payload_start + spec["exp_rel"],
                "exp_len": spec["exp_len"],
            }
            if version >= VERSION_V2:
                item["exp_mode"] = spec.get("exp_mode", EXP_RAW8)
                item["exp_mode_name"] = spec.get("exp_mode_name", "EXP_RAW8")
                item["exp_complete_bytes"] = int(spec["exp_meta"].get("complete_bytes", spec["exp_len"]))
            if spec["mode"] == "uniform":
                item["mant_off"] = payload_start + spec["mant_rel"]
                item["mant_len"] = spec["mant_len"]
                exc = spec.get("exceptions") or {"n": 0}
                if exc.get("n", 0):
                    item["exceptions"] = {
                        "n": exc["n"],
                        "idx_off": payload_start + exc["idx_rel"],
                        "idx_len": exc["idx_len"],
                        "words_off": payload_start + exc["words_rel"],
                        "words_len": exc["words_len"],
                    }
                else:
                    item["exceptions"] = {"n": 0}
            else:
                item["row_keeps_off"] = payload_start + spec["row_keeps_rel"]
                item["row_keeps_len"] = spec["row_keeps_len"]
                item["mant_groups"] = [
                    {
                        "policy_k": g["policy_k"],
                        "k": g["k"],
                        "n_rows": g["n_rows"],
                        "n_weights": g["n_weights"],
                        "row_indices": g["row_indices"],
                        "off": payload_start + g["rel"],
                        "len": g["len"],
                    }
                    for g in spec["mant_groups"]
                ]
                exc = spec.get("exceptions") or {"n": 0}
                if exc.get("n", 0):
                    item["exceptions"] = {
                        "n": exc["n"],
                        "idx_off": payload_start + exc["idx_rel"],
                        "idx_len": exc["idx_len"],
                        "words_off": payload_start + exc["words_rel"],
                        "words_len": exc["words_len"],
                    }
                else:
                    item["exceptions"] = {"n": 0}
            tensors_out.append(item)

        if version >= VERSION_V2:
            exp_packing = "adaptive_exact_v2"
            exp_note = (
                "Per-tensor adaptive exact exponent coding (RAW8/rANS/Huffman/delta-rANS/run-rANS); "
                "mode chosen by minimum complete physical section bytes "
                "(payload+table+mode_id+length fields). Decoder reconstructs bit-identical Q(W)."
            )
        else:
            exp_packing = "raw_u8"
            exp_note = (
                "Exponents stored as raw 8 bits/weight for bit-exact decode. "
                "Prior estimated packed BPW used EXP_BPW_REF≈2.62; physical raw_u8 "
                "therefore sits ~5.38 BPW higher on the exponent term alone before metadata."
            )
        header_obj = {
            "format": "H95Q",
            "version": version,
            "model_id": model_id,
            "model_hash": model_hash,
            "candidate": candidate,
            "dtype": dtype_tag,
            "n_tensors": len(tensors_out),
            "n_alias_tensors": len(aliases),
            "total_weights": total_weights,
            "endianness": "little",
            "checksum_algo": "sha256",
            "exp_packing": exp_packing,
            "exp_packing_note": exp_note,
            "sign_packing": "bitplane",
            "mantissa_packing": "kbit_msb_first",
            "sha256_quantized_reference": ref_sha,
            "precision_map": precision_map or {},
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
            MAGIC, version, FLAGS_LITTLE_ENDIAN, CHECKSUM_SHA256, b"\x00\x00\x00", len(header_json)
        )
        header_end = len(prefix) + len(header_json)
        new_start = _align(header_end, ALIGN)
        pad = new_start - header_end
        if new_start == payload_start:
            blob = prefix + header_json + (b"\x00" * pad) + payload
            break
        payload_start = new_start
    else:
        raise RuntimeError("failed to stabilize H95Q header alignment")

    out_path.write_bytes(blob)
    file_bytes = len(blob)
    actual_bpw = (file_bytes * 8.0 / total_weights) if total_weights else 0.0
    stream_bpw = (payload_stream_bytes * 8.0 / total_weights) if total_weights else 0.0
    exp_bytes = sum(int(s["exp_len"]) for s in specs)
    exp_bpw = (exp_bytes * 8.0 / total_weights) if total_weights else 0.0
    mode_hist: dict[str, int] = {}
    for s in specs:
        name = str((s.get("exp_meta") or {}).get("mode_name") or s.get("exp_mode_name") or "EXP_RAW8")
        mode_hist[name] = mode_hist.get(name, 0) + 1
    exp_packing = "adaptive_exact_v2" if version >= VERSION_V2 else "raw_u8"
    return {
        "path": str(out_path),
        "file_bytes": file_bytes,
        "header_bytes": payload_start,
        "payload_stream_bytes": payload_stream_bytes,
        "overhead_bytes": file_bytes - payload_stream_bytes,
        "n_weights": total_weights,
        "n_tensors": len(specs),
        "actual_bpw": round(actual_bpw, 6),
        "stream_bpw": round(stream_bpw, 6),
        "sha256_quantized_reference": ref_sha,
        "sha256_file": _sha256_hex(blob),
        "exp_packing": exp_packing,
        "container_version": version,
        "exp_section_bytes": exp_bytes,
        "exp_complete_bpw": round(exp_bpw, 6),
        "exp_mode_histogram": mode_hist,
        "candidate": candidate,
    }


def read_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        prefix = f.read(_PREFIX.size)
        magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}")
        if ver not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported version {ver}")
        return json.loads(f.read(hdr_len).decode("utf-8"))


def decode_container(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported version {ver}")
    header = json.loads(data[16 : 16 + hdr_len].decode("utf-8"))
    # Prefer on-wire version; header JSON version should match.
    wire_ver = int(ver)
    hdr_ver = int(header.get("version", wire_ver))
    if hdr_ver != wire_ver:
        raise ValueError(f"header/wire version mismatch {hdr_ver} vs {wire_ver}")
    mv = memoryview(data)
    tensors: dict[str, np.ndarray] = {}
    for spec in header["tensors"]:
        tensors[spec["name"]] = _decode_one(spec, mv, version=wire_ver)
    for alias, canon in (header.get("aliases") or {}).items():
        if canon in tensors:
            tensors[alias] = tensors[canon]
    return {
        "header": header,
        "tensors": tensors,
        "file_bytes": len(data),
        "sha256_file": _sha256_hex(data),
    }


def verify_decoded_against_reference(
    decoded: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for name in sorted(reference.keys()):
        if name not in decoded:
            mismatches.append({"name": name, "error": "missing_in_decoded"})
            continue
        a = np.ascontiguousarray(reference[name], dtype=np.uint16).ravel()
        b = np.ascontiguousarray(decoded[name], dtype=np.uint16).ravel()
        checked += 1
        if a.shape != b.shape:
            mismatches.append(
                {"name": name, "error": "shape", "ref": list(a.shape), "dec": list(b.shape)}
            )
            continue
        if not np.array_equal(a, b):
            bad = np.flatnonzero(a != b)
            mismatches.append(
                {
                    "name": name,
                    "error": "words",
                    "n_mismatch": int(bad.size),
                    "first_idx": int(bad[0]),
                    "ref0": int(a[bad[0]]),
                    "dec0": int(b[bad[0]]),
                }
            )
    ref_sha = sha256_state_u16(reference)
    dec_sha = sha256_state_u16(decoded)
    return {
        "exact_match": len(mismatches) == 0,
        "n_tensors_checked": checked,
        "mismatches": mismatches[:20],
        "sha256_ref": ref_sha,
        "sha256_decoded": dec_sha,
        "sha_ok": ref_sha == dec_sha,
    }


def main_decode_cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="H95Q decode-only tool")
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
    if args.expect_sha:
        ok = h["sha256_quantized_reference"] == args.expect_sha
        print(f"sha_match={ok}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main_decode_cli())
