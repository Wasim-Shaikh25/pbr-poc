"""E2 packed / E3 selective physical containers for mixed groupwise Q.

Layout (little-endian)::

    [0:4]   magic b"E2MX" (packed) or b"E3MX" (selective tiles)
    [4:6]   version u16le = 1
    [6:8]   flags u16le (bit0 = little-endian)
    [8:9]   checksum_algo u8 (1 = SHA-256)
    [9:12]  reserved
    [12:16] header_json_len u32le
    [16:]   header JSON
    then pad to 64-byte alignment
    then payload blobs

Per unique tensor
-----------------
* uint8 row_bits (precision map)
* optional BF16 raw rows (row indices + words)
* f16 scales + u8 zp for quantized groups (row-major, skipping BF16 rows)
* packed codes (E2) or selective Phase-1 tiles (E3) **per bitwidth band**
* optional specials (NaN/Inf only) — not a quality lever

Product default for E3 is packed. Matrix family is used only with net
margin after complete costs. Decode of this quantized reference is
bit-exact; it is not BF16-exact.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pbr_h95.bitpack import pack_kbit, packed_bytes_for_k
from pbr_h95.container_h95q import sha256_state_u16, sha256_u16
from pbr_q4.const import MODE_NAMES, TILE
from pbr_q4.e2_mixed.const import (
    ALIGN,
    CHECKSUM_SHA256,
    FLAGS_LITTLE_ENDIAN,
    GROUP_SIZE,
    MAGIC_E2,
    MAGIC_E3,
    VERSION,
)
from pbr_q4.e2_mixed.e3_codec import decode_array_e3, encode_array_e3
from pbr_q4.e2_mixed.quantize import QuantizedTensor, dequantize_mixed

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


def encode_quantized(
    tensors: list[QuantizedTensor],
    out_path: str | Path,
    *,
    phase: str,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    policy_name: str = "",
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write E2 packed or E3 selective container. Same Q-ref either way."""
    if phase not in ("E2", "E3"):
        raise ValueError(phase)
    magic = MAGIC_E2 if phase == "E2" else MAGIC_E3
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aliases = dict(aliases or {})

    qref = {t.name: t.q_ref for t in tensors}
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
    packed_whole = codes_blob = 0
    n_xy_sel = 0
    n_specials = 0
    weight_payload = scales_bytes = prec_map_bytes = specials_bytes = 0
    family_words: dict[str, dict[str, int]] = {}

    def _push(blob: bytes) -> tuple[int, int]:
        nonlocal rel
        off = rel
        chunks.append(blob)
        rel += len(blob)
        return off, len(blob)

    for qt in tensors:
        fam = family_words.setdefault(qt.family, {"n_words": 0, "n_specials": 0})
        fam["n_words"] += int(qt.q_ref.size)
        fam["n_specials"] += int(qt.special_idx.size)
        n_specials += int(qt.special_idx.size)
        item: dict[str, Any] = {
            "name": qt.name,
            "shape": list(qt.shape),
            "rows": qt.rows,
            "cols": qt.cols,
            "family": qt.family,
            "projection": qt.projection,
            "group_size": qt.group_size,
            "n_groups": qt.n_groups,
            "n_weights": int(qt.q_ref.size),
            "n_specials": int(qt.special_idx.size),
            "sha256_q_ref": sha256_u16(qt.q_ref),
            "mse": qt.mse,
            "eps": qt.eps,
            "bands": [],
        }
        rb = np.ascontiguousarray(qt.row_bits, dtype=np.uint8)
        item["row_bits_rel"], item["row_bits_len"] = _push(rb.tobytes())
        prec_map_bytes += int(item["row_bits_len"])

        # BF16 protected rows (structured, not sparse weights).
        bf16_idx = np.flatnonzero(rb >= 16).astype(np.uint32)
        if bf16_idx.size:
            arr2 = np.ascontiguousarray(qt.q_ref, dtype=np.uint16).reshape(qt.rows, qt.cols)
            bf16_words = arr2[bf16_idx]
            idx_b = np.ascontiguousarray(bf16_idx, dtype="<u4").tobytes()
            words_b = np.ascontiguousarray(bf16_words, dtype="<u2").tobytes()
        else:
            idx_b = b""
            words_b = b""
        item["bf16_idx_rel"], item["bf16_idx_len"] = _push(idx_b)
        item["bf16_words_rel"], item["bf16_words_len"] = _push(words_b)
        weight_payload += len(words_b)

        scale_b = np.ascontiguousarray(qt.scales, dtype="<f2").tobytes()
        zp_b = np.ascontiguousarray(qt.zp, dtype=np.uint8).tobytes()
        item["scale_rel"], item["scale_len"] = _push(scale_b)
        item["zp_rel"], item["zp_len"] = _push(zp_b)
        scales_bytes += len(scale_b) + len(zp_b)

        codes2 = np.ascontiguousarray(qt.codes, dtype=np.uint16).reshape(qt.rows, qt.cols)
        bands: list[dict[str, Any]] = []
        for b in (4, 5, 6, 8):
            idx = np.flatnonzero(rb == np.uint8(b)).astype(np.int32)
            if idx.size == 0:
                continue
            sub = codes2[idx]
            if phase == "E3":
                enc = encode_array_e3(sub, int(b))
                blob = enc["blob"]
                kind = enc["kind"]
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
                codes_blob += int(enc["blob_bytes"])
                if enc["kind"] == "xy_sel":
                    n_xy_sel += 1
                band_meta = {
                    "n_tiles": int(enc["n_tiles"]),
                    "n_xy_tiles": int(enc["n_xy_tiles"]),
                    "n_packed_tiles": int(enc["n_packed_tiles"]),
                    "map_bytes": int(enc["map_bytes"]),
                    "flag_bytes": int(enc["flag_bytes"]),
                    "fallback_packed": bool(enc["fallback_packed"]),
                    "packed_whole_bytes": int(enc["packed_whole_bytes"]),
                }
            else:
                blob = pack_kbit(sub.ravel(), int(b)) if b and sub.size else b""
                kind = "packed"
                packed_whole += len(blob)
                codes_blob += len(blob)
                n_tiles += 0
                band_meta = {
                    "n_tiles": 0,
                    "n_xy_tiles": 0,
                    "n_packed_tiles": 0,
                    "map_bytes": 0,
                    "flag_bytes": 0,
                    "fallback_packed": True,
                    "packed_whole_bytes": len(blob),
                }
            item_band: dict[str, Any] = {
                "bits": int(b),
                "n_rows": int(idx.size),
                "kind": kind,
                **band_meta,
            }
            idx_bytes = np.ascontiguousarray(idx.astype(np.uint32), dtype="<u4").tobytes()
            item_band["row_idx_rel"], item_band["row_idx_len"] = _push(idx_bytes)
            item_band["codes_rel"], item_band["codes_len"] = _push(blob)
            weight_payload += len(blob)
            bands.append(item_band)
        item["bands"] = bands

        if qt.special_idx.size:
            sidx = np.ascontiguousarray(qt.special_idx, dtype="<u4").tobytes()
            sw = np.ascontiguousarray(qt.special_words, dtype="<u2").tobytes()
        else:
            sidx = b""
            sw = b""
        item["special_idx_rel"], item["special_idx_len"] = _push(sidx)
        item["special_words_rel"], item["special_words_len"] = _push(sw)
        specials_bytes += len(sidx) + len(sw)
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
                "family": spec["family"],
                "projection": spec["projection"],
                "group_size": spec["group_size"],
                "n_groups": spec["n_groups"],
                "n_weights": spec["n_weights"],
                "n_specials": spec["n_specials"],
                "sha256_q_ref": spec["sha256_q_ref"],
                "row_bits_off": payload_start + spec["row_bits_rel"],
                "row_bits_len": spec["row_bits_len"],
                "bf16_idx_off": payload_start + spec["bf16_idx_rel"],
                "bf16_idx_len": spec["bf16_idx_len"],
                "bf16_words_off": payload_start + spec["bf16_words_rel"],
                "bf16_words_len": spec["bf16_words_len"],
                "scale_off": payload_start + spec["scale_rel"],
                "scale_len": spec["scale_len"],
                "zp_off": payload_start + spec["zp_rel"],
                "zp_len": spec["zp_len"],
                "special_idx_off": payload_start + spec["special_idx_rel"],
                "special_idx_len": spec["special_idx_len"],
                "special_words_off": payload_start + spec["special_words_rel"],
                "special_words_len": spec["special_words_len"],
                "bands": [],
            }
            for band in spec["bands"]:
                item["bands"].append(
                    {
                        "bits": band["bits"],
                        "n_rows": band["n_rows"],
                        "kind": band["kind"],
                        "n_tiles": band["n_tiles"],
                        "n_xy_tiles": band["n_xy_tiles"],
                        "n_packed_tiles": band["n_packed_tiles"],
                        "map_bytes": band["map_bytes"],
                        "flag_bytes": band["flag_bytes"],
                        "fallback_packed": band["fallback_packed"],
                        "packed_whole_bytes": band["packed_whole_bytes"],
                        "row_idx_off": payload_start + band["row_idx_rel"],
                        "row_idx_len": band["row_idx_len"],
                        "codes_off": payload_start + band["codes_rel"],
                        "codes_len": band["codes_len"],
                    }
                )
            tensors_out.append(item)
        header_obj = {
            "format": magic.decode("ascii"),
            "version": VERSION,
            "phase": phase,
            "model_id": model_id,
            "policy": policy_name,
            "dtype": "bf16_uint16",
            "n_tensors": len(tensors_out),
            "n_alias_tensors": len(aliases),
            "total_weights": total_weights,
            "endianness": "little",
            "checksum_algo": "sha256",
            "tile": [TILE, TILE],
            "xy_default": "packed",
            "group_size": GROUP_SIZE,
            "s1_compatible": False,
            "pr17_compatible": False,
            "pr18_compatible": False,
            "sha256_quantized_reference": ref_sha,
            "aliases": aliases,
            "tensors": tensors_out,
            "alignment": ALIGN,
            "payload_start": payload_start,
            "hard_override": (
                "packed"
                if phase == "E2"
                else "selective_phase1_only_if_net_margin"
            ),
            "e3_net_margin": "max(16 bits, 0.02 * raw_tile_bits) after all costs",
            "quality_claim": "bit-exact to this quantized reference, not BF16-exact",
        }
        return json.dumps(header_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    payload_start = ALIGN
    blob = b""
    for _ in range(6):
        header_json = build_header_json(payload_start)
        prefix = _PREFIX.pack(
            magic, VERSION, FLAGS_LITTLE_ENDIAN, CHECKSUM_SHA256, b"\x00\x00\x00", len(header_json)
        )
        header_end = len(prefix) + len(header_json)
        new_start = _align(header_end, ALIGN)
        pad = new_start - header_end
        if new_start == payload_start:
            blob = prefix + header_json + (b"\x00" * pad) + payload
            break
        payload_start = new_start
    else:
        raise RuntimeError("failed to stabilize E2/E3 header alignment")

    out_path.write_bytes(blob)
    file_bytes = len(blob)
    if file_bytes != out_path.stat().st_size:
        raise RuntimeError("written size != path size")
    actual_bpw = (file_bytes * 8.0 / total_weights) if total_weights else 0.0
    header_bytes = payload_start
    container_over = file_bytes - len(payload)

    illegal = [m for m in mode_hist if m not in MODE_NAMES.values()]
    if illegal:
        raise RuntimeError(f"non-matrix XY modes in histogram: {illegal}")
    if mode_hist.get("XY_MATRIX"):
        raise RuntimeError("XY_MATRIX must not be selected")
    if mode_hist.get("XY_PAIR"):
        raise RuntimeError("XY_PAIR is not in the Phase-1 mode set")

    def _bpw(nbytes: int) -> float:
        return round((nbytes * 8.0 / total_weights), 6) if total_weights else 0.0

    return {
        "path": str(out_path),
        "phase": phase,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "payload_stream_bytes": len(payload),
        "n_weights": total_weights,
        "n_tensors": len(specs),
        "n_tiles": n_tiles,
        "n_xy_tiles": n_xy,
        "n_packed_tiles": n_packed,
        "n_xy_sel_tensors": n_xy_sel,
        "n_specials": n_specials,
        "actual_bpw": round(actual_bpw, 6),
        "sha256_quantized_reference": ref_sha,
        "sha256_file": _sha256_hex(blob),
        "xy_flag_bytes": flag_bytes,
        "xy_map_bytes": map_bytes,
        "xy_payload_bytes": xy_payload,
        "codes_blob_bytes": codes_blob,
        "packed_whole_bytes": packed_whole,
        "saved_vs_packed_code_bytes": packed_whole - codes_blob,
        "xy_mode_histogram": mode_hist,
        "xy_pred_histogram": pred_hist,
        "xy_trav_histogram": trav_hist,
        "family_words": family_words,
        "product_default_packed": True,
        "s1_compatible": False,
        "policy": policy_name,
        "container_version": VERSION,
        "overhead_split": {
            "weight_payload_bytes": weight_payload,
            "scales_bytes": scales_bytes,
            "precision_map_bytes": prec_map_bytes,
            "codec_meta_bytes": map_bytes + flag_bytes,
            "container_bytes": container_over + specials_bytes,
            "weight_payload_bpw": _bpw(weight_payload),
            "scales_bpw": _bpw(scales_bytes),
            "precision_map_bpw": _bpw(prec_map_bytes),
            "codec_meta_bpw": _bpw(map_bytes + flag_bytes),
            "container_bpw": _bpw(container_over + specials_bytes),
        },
    }


def _slice(data: memoryview, off: int, ln: int) -> bytes:
    if off < 0 or ln < 0 or off + ln > len(data):
        raise ValueError("payload out of range")
    return bytes(data[off : off + ln])


def _decode_one(spec: Mapping[str, Any], data: memoryview) -> np.ndarray:
    shape = tuple(int(x) for x in spec["shape"])
    rows, cols = int(spec["rows"]), int(spec["cols"])
    group_size = int(spec["group_size"])
    n_groups = int(spec["n_groups"])
    rb = np.frombuffer(_slice(data, int(spec["row_bits_off"]), int(spec["row_bits_len"])), dtype=np.uint8).copy()
    if int(rb.size) != rows:
        raise ValueError(f"row_bits length {rb.size} != rows {rows}")

    bf16_n = int(spec["bf16_idx_len"]) // 4
    bf16_idx = np.frombuffer(_slice(data, int(spec["bf16_idx_off"]), int(spec["bf16_idx_len"])), dtype="<u4").astype(np.int64).copy()
    bf16_words = np.frombuffer(_slice(data, int(spec["bf16_words_off"]), int(spec["bf16_words_len"])), dtype="<u2").copy()
    if bf16_n and (bf16_words.size != bf16_n * cols):
        raise ValueError("BF16 payload size mismatch")
    bf16_rows: dict[int, np.ndarray] = {}
    if bf16_n:
        words2 = bf16_words.reshape(bf16_n, cols)
        for i, r in enumerate(bf16_idx.tolist()):
            bf16_rows[int(r)] = words2[i]

    scales = np.frombuffer(_slice(data, int(spec["scale_off"]), int(spec["scale_len"])), dtype="<f2").copy()
    zp = np.frombuffer(_slice(data, int(spec["zp_off"]), int(spec["zp_len"])), dtype=np.uint8).copy()
    if int(scales.size) != n_groups or int(zp.size) != n_groups:
        raise ValueError(f"scale/zp count {scales.size}/{zp.size} != n_groups {n_groups}")

    codes = np.zeros((rows, cols), dtype=np.uint16)
    for band in spec["bands"]:
        b = int(band["bits"])
        n_rows = int(band["n_rows"])
        idx = np.frombuffer(_slice(data, int(band["row_idx_off"]), int(band["row_idx_len"])), dtype="<u4").astype(np.int64).copy()
        if int(idx.size) != n_rows:
            raise ValueError("band row index count mismatch")
        blob = _slice(data, int(band["codes_off"]), int(band["codes_len"]))
        kind = band.get("kind") or "packed"
        if kind == "packed":
            need = packed_bytes_for_k(n_rows * cols, b)
            if len(blob) != need:
                raise ValueError(f"packed band length {len(blob)} != {need}")
            from pbr_h95.bitpack import unpack_kbit

            vals = unpack_kbit(blob, n_rows * cols, b).reshape(n_rows, cols)
        else:
            vals = decode_array_e3(blob, rows=n_rows, cols=cols, nbits=b)
        codes[idx] = vals

    n_sp = int(spec.get("n_specials") or 0)
    sidx = np.frombuffer(_slice(data, int(spec["special_idx_off"]), int(spec["special_idx_len"])), dtype="<u4").astype(np.int64).copy()
    swords = np.frombuffer(_slice(data, int(spec["special_words_off"]), int(spec["special_words_len"])), dtype="<u2").copy()
    if n_sp and (int(sidx.size) != n_sp or int(swords.size) != n_sp):
        raise ValueError("specials length mismatch")

    return dequantize_mixed(
        row_bits=rb,
        codes=codes,
        scales=scales,
        zp=zp,
        rows=rows,
        cols=cols,
        group_size=group_size,
        shape=shape,
        special_idx=sidx if n_sp else None,
        special_words=swords if n_sp else None,
        bf16_rows=bf16_rows,
    )


def read_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        prefix = f.read(_PREFIX.size)
        magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack(prefix)
        if magic not in (MAGIC_E2, MAGIC_E3):
            raise ValueError(f"bad magic {magic!r}")
        if ver != VERSION:
            raise ValueError(f"unsupported version {ver}")
        return json.loads(f.read(hdr_len).decode("utf-8"))


def decode_container(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    if len(data) != path.stat().st_size:
        raise ValueError("read size != file size")
    magic, ver, _flags, _algo, _res, hdr_len = _PREFIX.unpack_from(data, 0)
    if magic not in (MAGIC_E2, MAGIC_E3):
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    header = json.loads(data[16 : 16 + hdr_len].decode("utf-8"))
    if header.get("s1_compatible") or header.get("pr17_compatible") or header.get("pr18_compatible"):
        raise ValueError("E2/E3 must not claim S1/#17/#18 compatibility")
    mv = memoryview(data)
    tensors: dict[str, np.ndarray] = {}
    n_spec = len(header["tensors"])
    for i, spec in enumerate(header["tensors"], 1):
        if i == 1 or i == n_spec or i % 40 == 0:
            print(f"  e2e3-decode {i}/{n_spec} {spec['name']}", flush=True)
        tensors[spec["name"]] = _decode_one(spec, mv)
    for alias, canon in (header.get("aliases") or {}).items():
        if canon in tensors:
            tensors[alias] = tensors[canon]
    last = 0
    for spec in header["tensors"]:
        for key in (
            "row_bits_off",
            "bf16_idx_off",
            "bf16_words_off",
            "scale_off",
            "zp_off",
            "special_idx_off",
            "special_words_off",
        ):
            ln_key = key.replace("_off", "_len")
            last = max(last, int(spec[key]) + int(spec.get(ln_key, 0)))
        for band in spec.get("bands") or []:
            for key in ("row_idx_off", "codes_off"):
                ln_key = key.replace("_off", "_len")
                last = max(last, int(band[key]) + int(band.get(ln_key, 0)))
    if last > len(data):
        raise ValueError("payload overruns file")
    return {
        "header": header,
        "tensors": tensors,
        "file_bytes": len(data),
        "sha256_file": _sha256_hex(data),
        "sha256_decoded": sha256_state_u16(tensors),
        "phase": header.get("phase"),
        "magic": magic,
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

    p = argparse.ArgumentParser(description="E2/E3 mixed-Q decode-only tool")
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
    print(f"decoded n_tensors={h['n_tensors']} n_weights={tw} phase={h.get('phase')}")
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
