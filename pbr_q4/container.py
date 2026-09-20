"""H95X container: H95Q-S1 quantized reference + all-tile 256-node X/Y mantissas.

Physical layout
---------------
::

    [0:4]   magic b"H95X"
    [4:6]   version u16le = 1
    [6:8]   flags u16le (bit0 = little-endian)
    [8:9]   checksum_algo u8 (1 = SHA-256)
    [9:12]  reserved
    [12:16] header_json_len u32le
    [16:]   header JSON (utf-8)
    then pad to 64-byte alignment
    then payload blobs; absolute file offsets live in the header

Streams per unique tensor
-------------------------
* sign bitplane (unchanged from H95Q)
* exponents: adaptive exact (reuse ``pbr_h95.exp_codec``)
* mantissa: 16×16 X/Y matrix-family tiles (never packed-raw on the wire)
* embed tiers: int8 row-keep table + per-K X/Y groups
* optional raw-u16 exception words (NaN payloads / dirty lower bits)

Decode-only imports: numpy, this package, ``pbr_h95.bitpack``, ``pbr_h95.exp_codec``,
``pbr_core.bf16``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pbr_h95.bitpack import pack_sign_bitplane, unpack_sign_bitplane
from pbr_h95.container_h95q import sha256_state_u16, sha256_u16, verify_decoded_against_reference
from pbr_h95.exp_codec import (
    EXP_RAW8,
    decode_exp_blob,
    encode_exp_adaptive,
)
from pbr_q4.codecs import decode_array_xy, encode_array_xy
from pbr_q4.const import FROZEN_S1_SHA, MODE_NAMES, TILE
from pbr_q4.fields import join_embed_tiers, join_uniform, split_embed_tiers, split_quantized

MAGIC = b"H95X"
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


def _encode_one(
    words: np.ndarray,
    *,
    keep: int | None = None,
    row_keeps: np.ndarray | None = None,
) -> dict[str, Any]:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    if row_keeps is not None:
        fields = split_embed_tiers(w, row_keeps)
        exp_res = encode_exp_adaptive(fields["exp"])
        groups_out: dict[str, Any] = {}
        mode_hist: dict[str, int] = {}
        pred_hist: dict[str, int] = {}
        trav_hist: dict[str, int] = {}
        packed_base = 0
        xy_payload = 0
        n_tiles = 0
        n_xy_lt = n_xy_eq = n_xy_gt = 0
        xy_win_bytes = 0
        for k, g in fields["groups"].items():
            enc = encode_array_xy(g["kept"], int(g["k"]))
            if not enc["all_matrix_family"]:
                raise RuntimeError("embed group escaped matrix family")
            groups_out[str(int(k))] = {
                "k": int(g["k"]),
                "n_rows": int(g["n_rows"]),
                "n_weights": int(g["n_weights"]),
                "row_indices": g["row_indices"],
                "rows": enc["rows"],
                "cols": enc["cols"],
                "pad_rows": enc.get("pad_rows", enc["rows"]),
                "pad_cols": enc.get("pad_cols", enc["cols"]),
                "n_tiles": enc["n_tiles"],
                "flags": enc["flags"],
                "payload": enc["payload"],
            }
            _merge_hist(mode_hist, enc["mode_hist"])
            _merge_hist(pred_hist, enc["pred_hist"])
            _merge_hist(trav_hist, enc["trav_hist"])
            packed_base += int(enc["packed_baseline_bytes"])
            xy_payload += int(enc["xy_payload_bytes"])
            n_tiles += int(enc["n_tiles"])
            n_xy_lt += int(enc.get("n_tiles_xy_lt_packed") or 0)
            n_xy_eq += int(enc.get("n_tiles_xy_eq_packed") or 0)
            n_xy_gt += int(enc.get("n_tiles_xy_gt_packed") or 0)
            xy_win_bytes += int(enc.get("xy_win_bytes_vs_packed") or 0)
        return {
            "shape": fields["shape"],
            "n_weights": int(w.size),
            "mode": "embed_tiers",
            "sign": pack_sign_bitplane(fields["sign"]),
            "exp": exp_res.blob,
            "exp_meta": exp_res.as_dict(),
            "row_keeps": fields["row_keeps"],
            "mant_groups": groups_out,
            "base_keep": None,
            "exception_indices": fields["exception_indices"],
            "exception_words": fields["exception_words"],
            "xy_mode_hist": mode_hist,
            "xy_pred_hist": pred_hist,
            "xy_trav_hist": trav_hist,
            "n_tiles": n_tiles,
            "packed_baseline_bytes": packed_base,
            "xy_payload_bytes": xy_payload,
            "n_tiles_xy_lt_packed": n_xy_lt,
            "n_tiles_xy_eq_packed": n_xy_eq,
            "n_tiles_xy_gt_packed": n_xy_gt,
            "xy_win_bytes_vs_packed": xy_win_bytes,
            "all_matrix_family": True,
        }

    if keep is None:
        raise ValueError("keep or row_keeps required")
    fields = split_quantized(w, int(keep))
    exp_res = encode_exp_adaptive(fields["exp"])
    kept2 = fields["kept"].reshape(fields["shape"] if len(fields["shape"]) == 2 else (1, fields["kept"].size))
    if len(fields["shape"]) == 1:
        kept2 = fields["kept"].reshape(1, -1)
    elif len(fields["shape"]) > 2:
        kept2 = fields["kept"].reshape(fields["shape"][0], -1)
    enc = encode_array_xy(kept2, int(keep))
    if not enc["all_matrix_family"]:
        raise RuntimeError("tensor escaped matrix family")
    return {
        "shape": fields["shape"],
        "n_weights": int(w.size),
        "mode": "uniform",
        "sign": pack_sign_bitplane(fields["sign"]),
        "exp": exp_res.blob,
        "exp_meta": exp_res.as_dict(),
        "base_keep": int(keep),
        "xy": enc,
        "exception_indices": fields["exception_indices"],
        "exception_words": fields["exception_words"],
        "xy_mode_hist": enc["mode_hist"],
        "xy_pred_hist": enc["pred_hist"],
        "xy_trav_hist": enc["trav_hist"],
        "n_tiles": enc["n_tiles"],
        "packed_baseline_bytes": enc["packed_baseline_bytes"],
        "xy_payload_bytes": enc["xy_payload_bytes"],
        "n_tiles_xy_lt_packed": enc.get("n_tiles_xy_lt_packed", 0),
        "n_tiles_xy_eq_packed": enc.get("n_tiles_xy_eq_packed", 0),
        "n_tiles_xy_gt_packed": enc.get("n_tiles_xy_gt_packed", 0),
        "xy_win_bytes_vs_packed": enc.get("xy_win_bytes_vs_packed", 0),
        "all_matrix_family": True,
    }


def _decode_one(spec: dict[str, Any], data: memoryview) -> np.ndarray:
    n = int(spec["n_weights"])
    shape = tuple(spec["shape"])
    sign = unpack_sign_bitplane(bytes(data[spec["sign_off"] : spec["sign_off"] + spec["sign_len"]]), n)
    exp, _mode = decode_exp_blob(bytes(data[spec["exp_off"] : spec["exp_off"] + spec["exp_len"]]), n)

    if spec["mode"] == "uniform":
        k = int(spec["base_keep"])
        flags = bytes(data[spec["xy_flags_off"] : spec["xy_flags_off"] + spec["xy_flags_len"]])
        payload = bytes(data[spec["xy_payload_off"] : spec["xy_payload_off"] + spec["xy_payload_len"]])
        rows = int(spec["xy_rows"])
        cols = int(spec["xy_cols"])
        kept2 = decode_array_xy(
            flags,
            payload,
            rows=rows,
            cols=cols,
            nbits=k,
            pad_rows=int(spec.get("xy_pad_rows", rows)),
            pad_cols=int(spec.get("xy_pad_cols", cols)),
        )
        kept = kept2.ravel()[:n]
        words = join_uniform(sign, exp, kept, k, shape)
        exc = spec.get("exceptions") or {}
        if int(exc.get("n", 0)):
            idx = np.frombuffer(bytes(data[exc["idx_off"] : exc["idx_off"] + exc["idx_len"]]), dtype="<i8")
            raw = np.frombuffer(bytes(data[exc["words_off"] : exc["words_off"] + exc["words_len"]]), dtype="<u2")
            flat = words.ravel().copy()
            flat[idx] = raw
            words = flat.reshape(shape)
        return words

    if spec["mode"] == "embed_tiers":
        rows, cols = shape
        rk = np.frombuffer(
            bytes(data[spec["row_keeps_off"] : spec["row_keeps_off"] + spec["row_keeps_len"]]),
            dtype=np.int8,
        )
        if rk.size != rows:
            raise ValueError("row_keeps size mismatch")
        groups: dict[int, dict[str, Any]] = {}
        for g in spec["mant_groups"]:
            flags = bytes(data[g["flags_off"] : g["flags_off"] + g["flags_len"]])
            payload = bytes(data[g["payload_off"] : g["payload_off"] + g["payload_len"]])
            kept = decode_array_xy(
                flags,
                payload,
                rows=int(g["rows"]),
                cols=int(g["cols"]),
                nbits=int(g["k"]),
                pad_rows=int(g.get("pad_rows", g["rows"])),
                pad_cols=int(g.get("pad_cols", g["cols"])),
            )
            groups[int(g["policy_k"])] = {
                "k": int(g["k"]),
                "row_indices": np.asarray(g["row_indices"], dtype=np.int32),
                "kept": kept,
            }
        words = join_embed_tiers(sign, exp, rk, groups, (rows, cols))
        exc = spec.get("exceptions") or {}
        if int(exc.get("n", 0)):
            eidx = np.frombuffer(bytes(data[exc["idx_off"] : exc["idx_off"] + exc["idx_len"]]), dtype="<i8")
            raw = np.frombuffer(bytes(data[exc["words_off"] : exc["words_off"] + exc["words_len"]]), dtype="<u2")
            flat = words.ravel().copy()
            flat[eidx] = raw
            words = flat.reshape(shape)
        return words

    raise ValueError(f"unknown mode {spec['mode']!r}")


def encode_container(
    out_path: str | Path,
    tensors: Mapping[str, np.ndarray],
    *,
    keep_map: Mapping[str, int],
    model_id: str,
    model_hash: str = "",
    candidate: str = "H95Q-S1+XY",
    precision_map: dict[str, Any] | None = None,
    embed_name: str | None = None,
    embed_row_keeps: np.ndarray | None = None,
    dtype_tag: str = "bf16",
    expect_ref_sha: str | None = FROZEN_S1_SHA,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    n_unique = len(unique)
    for i, (name, w) in enumerate(unique, 1):
        print(f"  xy-encode start {i}/{n_unique} {name} shape={tuple(w.shape)}", flush=True)
        if embed_name and name == embed_name and embed_row_keeps is not None:
            enc = _encode_one(w, row_keeps=embed_row_keeps)
        else:
            k = int(keep_map.get(name, 7))
            enc = _encode_one(w, keep=max(k, 0))
        enc["name"] = name
        enc["sha256"] = sha256_u16(w)
        if not enc.get("all_matrix_family", False):
            raise RuntimeError(f"{name}: tile escaped matrix family")
        encoded.append(enc)
        if i == 1 or i == n_unique or i % 20 == 0:
            print(
                f"  xy-encode {i}/{n_unique} {name} tiles={enc.get('n_tiles', 0)} "
                f"modes={enc.get('xy_mode_hist')}",
                flush=True,
            )

    ref_sha = sha256_state_u16({n: t for n, t in unique})
    if expect_ref_sha and ref_sha != expect_ref_sha:
        raise RuntimeError(
            f"S1 quantized-reference SHA drift: got {ref_sha}, expected {expect_ref_sha}"
        )
    total_weights = sum(int(e["n_weights"]) for e in encoded)

    chunks: list[bytes] = []

    def add(b: bytes) -> tuple[int, int]:
        off = sum(len(c) for c in chunks)
        chunks.append(b)
        return off, len(b)

    specs: list[dict[str, Any]] = []
    mode_hist: dict[str, int] = {}
    pred_hist: dict[str, int] = {}
    trav_hist: dict[str, int] = {}
    packed_base_total = 0
    xy_payload_total = 0
    n_tiles_total = 0
    n_xy_lt = n_xy_eq = n_xy_gt = 0
    xy_win_bytes = 0

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
            "n_tiles": enc.get("n_tiles", 0),
            "xy_mode_hist": enc.get("xy_mode_hist") or {},
        }
        so, sl = add(enc["sign"])
        eo, el = add(enc["exp"])
        spec["sign_rel"], spec["sign_len"] = so, sl
        spec["exp_rel"], spec["exp_len"] = eo, el
        spec["exp_mode"] = int(spec["exp_meta"].get("mode", EXP_RAW8))
        spec["exp_mode_name"] = str(spec["exp_meta"].get("mode_name", "EXP_RAW8"))

        if enc["mode"] == "uniform":
            xy = enc["xy"]
            fo, fl = add(xy["flags"])
            po, pl = add(xy["payload"])
            spec["xy_flags_rel"], spec["xy_flags_len"] = fo, fl
            spec["xy_payload_rel"], spec["xy_payload_len"] = po, pl
            spec["xy_rows"], spec["xy_cols"] = int(xy["rows"]), int(xy["cols"])
            spec["xy_pad_rows"] = int(xy.get("pad_rows", xy["rows"]))
            spec["xy_pad_cols"] = int(xy.get("pad_cols", xy["cols"]))
        else:
            rk_b = np.ascontiguousarray(enc["row_keeps"], dtype=np.int8).tobytes()
            ro, rl = add(rk_b)
            spec["row_keeps_rel"], spec["row_keeps_len"] = ro, rl
            gout = []
            for k_str, g in enc["mant_groups"].items():
                fo, fl = add(g["flags"])
                po, pl = add(g["payload"])
                gout.append(
                    {
                        "policy_k": int(k_str),
                        "k": g["k"],
                        "n_rows": g["n_rows"],
                        "n_weights": g["n_weights"],
                        "row_indices": g["row_indices"].astype(int).tolist(),
                        "rows": g["rows"],
                        "cols": g["cols"],
                        "pad_rows": g.get("pad_rows", g["rows"]),
                        "pad_cols": g.get("pad_cols", g["cols"]),
                        "n_tiles": g["n_tiles"],
                        "flags_rel": fo,
                        "flags_len": fl,
                        "payload_rel": po,
                        "payload_len": pl,
                    }
                )
            spec["mant_groups"] = gout

        ei = enc["exception_indices"]
        ew = enc["exception_words"]
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
        _merge_hist(mode_hist, enc.get("xy_mode_hist") or {})
        _merge_hist(pred_hist, enc.get("xy_pred_hist") or {})
        _merge_hist(trav_hist, enc.get("xy_trav_hist") or {})
        packed_base_total += int(enc.get("packed_baseline_bytes") or 0)
        xy_payload_total += int(enc.get("xy_payload_bytes") or 0)
        n_tiles_total += int(enc.get("n_tiles") or 0)
        n_xy_lt += int(enc.get("n_tiles_xy_lt_packed") or 0)
        n_xy_eq += int(enc.get("n_tiles_xy_eq_packed") or 0)
        n_xy_gt += int(enc.get("n_tiles_xy_gt_packed") or 0)
        xy_win_bytes += int(enc.get("xy_win_bytes_vs_packed") or 0)

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
                "n_tiles": spec["n_tiles"],
                "sign_off": payload_start + spec["sign_rel"],
                "sign_len": spec["sign_len"],
                "exp_off": payload_start + spec["exp_rel"],
                "exp_len": spec["exp_len"],
                "exp_mode": spec.get("exp_mode", EXP_RAW8),
                "exp_mode_name": spec.get("exp_mode_name", "EXP_RAW8"),
                "exp_complete_bytes": int(spec["exp_meta"].get("complete_bytes", spec["exp_len"])),
            }
            if spec["mode"] == "uniform":
                item["xy_rows"] = spec["xy_rows"]
                item["xy_cols"] = spec["xy_cols"]
                item["xy_pad_rows"] = spec.get("xy_pad_rows", spec["xy_rows"])
                item["xy_pad_cols"] = spec.get("xy_pad_cols", spec["xy_cols"])
                item["xy_flags_off"] = payload_start + spec["xy_flags_rel"]
                item["xy_flags_len"] = spec["xy_flags_len"]
                item["xy_payload_off"] = payload_start + spec["xy_payload_rel"]
                item["xy_payload_len"] = spec["xy_payload_len"]
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
                        "rows": g["rows"],
                        "cols": g["cols"],
                        "pad_rows": g.get("pad_rows", g["rows"]),
                        "pad_cols": g.get("pad_cols", g["cols"]),
                        "n_tiles": g["n_tiles"],
                        "flags_off": payload_start + g["flags_rel"],
                        "flags_len": g["flags_len"],
                        "payload_off": payload_start + g["payload_rel"],
                        "payload_len": g["payload_len"],
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

        header_obj = {
            "format": "H95Q-S1-XY",
            "version": VERSION,
            "model_id": model_id,
            "model_hash": model_hash,
            "candidate": candidate,
            "dtype": dtype_tag,
            "n_tensors": len(tensors_out),
            "n_alias_tensors": len(aliases),
            "total_weights": total_weights,
            "endianness": "little",
            "checksum_algo": "sha256",
            "exp_packing": "adaptive_exact_v2",
            "sign_packing": "bitplane",
            "mantissa_packing": "xy_256_node_matrix_family",
            "tile": [TILE, TILE],
            "hard_override": "all_tiles_matrix_family_never_packed_raw",
            "packed_k_is_baseline_metric_only": True,
            "sha256_quantized_reference": ref_sha,
            "frozen_s1_sha": FROZEN_S1_SHA,
            "precision_map": precision_map or {},
            "aliases": aliases,
            "tensors": tensors_out,
            "alignment": ALIGN,
            "payload_start": payload_start,
            "n_tiles": n_tiles_total,
            "xy_mode_histogram": mode_hist,
            "xy_pred_histogram": pred_hist,
            "xy_trav_histogram": trav_hist,
            "packed_baseline_bytes": packed_base_total,
            "xy_payload_bytes": xy_payload_total,
            "tile_win_stats": {
                "n_tiles_xy_lt_packed": n_xy_lt,
                "n_tiles_xy_eq_packed": n_xy_eq,
                "n_tiles_xy_gt_packed": n_xy_gt,
                "xy_win_bytes_vs_packed": xy_win_bytes,
            },
            "note": (
                "Post-codec of frozen H95Q-S1 quantized words. "
                "Does not re-quantize from BF16. "
                "Every eligible tile uses the 256-node X/Y matrix family."
            ),
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
        raise RuntimeError("failed to stabilize H95X header alignment")

    out_path.write_bytes(blob)
    file_bytes = len(blob)
    actual_bpw = (file_bytes * 8.0 / total_weights) if total_weights else 0.0
    stream_bpw = (payload_stream_bytes * 8.0 / total_weights) if total_weights else 0.0
    exp_bytes = sum(int(s["exp_len"]) for s in specs)
    exp_bpw = (exp_bytes * 8.0 / total_weights) if total_weights else 0.0
    flags_bytes = 0
    for s in specs:
        if s["mode"] == "uniform":
            flags_bytes += int(s["xy_flags_len"])
        else:
            flags_bytes += sum(int(g["flags_len"]) for g in s["mant_groups"])
    packed_base_bpw = (packed_base_total * 8.0 / total_weights) if total_weights else 0.0
    xy_payload_bpw = (xy_payload_total * 8.0 / total_weights) if total_weights else 0.0
    illegal = [m for m in mode_hist if m not in MODE_NAMES.values()]
    if illegal:
        raise RuntimeError(f"non-matrix modes in histogram: {illegal}")

    return {
        "path": str(out_path),
        "file_bytes": file_bytes,
        "header_bytes": payload_start,
        "payload_stream_bytes": payload_stream_bytes,
        "overhead_bytes": file_bytes - payload_stream_bytes,
        "n_weights": total_weights,
        "n_tensors": len(specs),
        "n_tiles": n_tiles_total,
        "actual_bpw": round(actual_bpw, 6),
        "stream_bpw": round(stream_bpw, 6),
        "sha256_quantized_reference": ref_sha,
        "sha256_file": _sha256_hex(blob),
        "exp_packing": "adaptive_exact_v2",
        "mantissa_packing": "xy_256_node_matrix_family",
        "container_version": VERSION,
        "exp_section_bytes": exp_bytes,
        "exp_complete_bpw": round(exp_bpw, 6),
        "xy_flag_bytes": flags_bytes,
        "xy_payload_bytes": xy_payload_total,
        "xy_payload_bpw": round(xy_payload_bpw, 6),
        "packed_baseline_bytes": packed_base_total,
        "packed_baseline_bpw": round(packed_base_bpw, 6),
        "xy_vs_packed_payload_bytes": xy_payload_total - packed_base_total,
        "xy_mode_histogram": mode_hist,
        "xy_pred_histogram": pred_hist,
        "xy_trav_histogram": trav_hist,
        "n_tiles_xy_lt_packed": n_xy_lt,
        "n_tiles_xy_eq_packed": n_xy_eq,
        "n_tiles_xy_gt_packed": n_xy_gt,
        "xy_win_bytes_vs_packed": xy_win_bytes,
        "all_tiles_matrix_family": True,
        "candidate": candidate,
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


def decode_container(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    header = json.loads(data[16 : 16 + hdr_len].decode("utf-8"))
    if int(header.get("version", ver)) != int(ver):
        raise ValueError("header/wire version mismatch")
    mv = memoryview(data)
    tensors: dict[str, np.ndarray] = {}
    n_spec = len(header["tensors"])
    for i, spec in enumerate(header["tensors"], 1):
        if i == 1 or i == n_spec or i % 20 == 0:
            print(f"  xy-decode {i}/{n_spec} {spec['name']}", flush=True)
        tensors[spec["name"]] = _decode_one(spec, mv)
    for alias, canon in (header.get("aliases") or {}).items():
        if canon in tensors:
            tensors[alias] = tensors[canon]
    return {
        "header": header,
        "tensors": tensors,
        "file_bytes": len(data),
        "sha256_file": _sha256_hex(data),
    }


def assert_all_tiles_matrix_family(header: Mapping[str, Any]) -> None:
    if header.get("mantissa_packing") != "xy_256_node_matrix_family":
        raise AssertionError("mantissa_packing is not the X/Y matrix family")
    if not header.get("packed_k_is_baseline_metric_only"):
        raise AssertionError("packed-K must be labeled baseline-only")


def main_decode_cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="H95X (S1+X/Y) decode-only tool")
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
    print(f"mantissa_packing={h.get('mantissa_packing')}")
    dec_sha = sha256_state_u16(result["tensors"])
    print(f"sha256_decoded={dec_sha}")
    if args.expect_sha:
        ok = dec_sha == args.expect_sha and h["sha256_quantized_reference"] == args.expect_sha
        print(f"sha_match={ok}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main_decode_cli())
