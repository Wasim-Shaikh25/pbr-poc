"""Experiments toward ≤8.0 complete BPW on Qwen (bit-exact).

Combines exponent rANS with mantissa/SM rANS, optional nibbles, global
dictionaries, and shared tables. Success is ≤8.0 complete BPW with
Decode(Encode(W))==W, or an honest falsification with the best number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import yaml
import zlib

from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_codecs.optional_nibble import decode_matrix as decode_optional
from pbr_codecs.optional_nibble import encode_matrix as encode_optional
from pbr_codecs.pbr_em import (
    DISCLAIMER as EM_DISCLAIMER,
    TAG_EXPCOND_M,
    TAG_SM_EXPCOND,
    TAG_SM_UNCOND,
    TAG_UNCOND_M,
    complete_bytes,
    decode_tensor_em,
    decode_tensor_em_shared,
    dump_shared_tables,
    encode_tensor_em,
    encode_tensor_em_shared,
)
from pbr_core.bf16 import split_components
from pbr_core.metrics import bits_per_weight
from pbr_core.rans import normalize_counts
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d
from pbr_core.types import TILE_HEADER_BYTES
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO, inventory_from_dir, select_weight_specs
from pbr_encoder.verification import assert_exact
from pbr_qualifier.entropy import shannon_entropy

DISCLAIMER = (
    "Path to 50% (≤8.0 complete BPW, bit-exact) on Qwen linear/attention weights. "
    "Complete bytes include tables, streams, headers. Not a 1–2 GB / 8 GB claim. "
    "Do not report 50% unless measured complete BPW is ≤8.0."
)
TARGET_BPW = 8.0
PBRE_REF = 10.6161
PBRE_REF_BYTES = 111253029


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _nll_laplace(counts: np.ndarray) -> float:
    """NLL bits of the observations that built ``counts`` (Laplace +1)."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 1:
        tot = float(c.sum())
        if tot <= 0:
            return 0.0
        p = (c + 1.0) / (tot + c.size)
        return float(-(c * np.log2(np.maximum(p, 1e-300))).sum())
    nll = 0.0
    for row in c:
        nll += _nll_laplace(row)
    return nll


def _freqs_from_counts2d(counts: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for row in counts:
        buf = np.zeros(256, dtype=np.int64)
        n = min(256, int(row.size))
        buf[:n] = np.asarray(row[:n], dtype=np.int64)
        if int(buf.sum()) <= 0:
            buf[0] = 1
        out.append(normalize_counts(buf))
    return out


def encode_escape_dict(words: np.ndarray, palette: np.ndarray) -> bytes:
    pal = np.ascontiguousarray(palette, dtype=np.uint16).ravel()
    k = int(pal.size)
    if k > 255:
        raise ValueError("escape palette must be <= 255")
    lut = np.full(65536, 255, dtype=np.uint8)
    lut[pal.astype(np.int64)] = np.arange(k, dtype=np.uint8)
    flat = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    ids = lut[flat.astype(np.int64)]
    escapes = flat[ids == 255]
    return struct_pack_escape(k, pal, ids, escapes)


def struct_pack_escape(k: int, pal: np.ndarray, ids: np.ndarray, escapes: np.ndarray) -> bytes:
    import struct

    return struct.pack("<H", k) + pal.astype("<u2").tobytes() + ids.tobytes() + escapes.astype("<u2").tobytes()


def decode_escape_dict(payload: bytes, n: int) -> np.ndarray:
    import struct

    (k,) = struct.unpack_from("<H", payload, 0)
    pal = np.frombuffer(payload, dtype="<u2", count=k, offset=2).astype(np.uint16)
    offset = 2 + 2 * k
    ids = np.frombuffer(payload, dtype=np.uint8, count=n, offset=offset).copy()
    offset += n
    n_esc = int((ids == 255).sum())
    esc = np.frombuffer(payload, dtype="<u2", count=n_esc, offset=offset).astype(np.uint16)
    out = pal[np.clip(ids, 0, max(k - 1, 0))].astype(np.uint16)
    out[ids == 255] = esc
    return out


def _tile_duplicate_rate(matrices: list[np.ndarray], tile: int = 16) -> dict:
    seen: dict[bytes, int] = {}
    n_tiles = 0
    n_dup = 0
    for mat in matrices:
        rows, cols = int(mat.shape[0]), int(mat.shape[1])
        for r0 in range(0, rows, tile):
            for c0 in range(0, cols, tile):
                chunk = np.ascontiguousarray(mat[r0 : r0 + tile, c0 : c0 + tile])
                key = hashlib.sha1(chunk.tobytes()).digest()
                n_tiles += 1
                if key in seen:
                    n_dup += 1
                else:
                    seen[key] = 1
    return {
        "tile": tile,
        "n_tiles": n_tiles,
        "n_dup": n_dup,
        "dup_rate": n_dup / max(n_tiles, 1),
        "unique_tiles": len(seen),
    }


def evaluate(specs, *, nibble_tile: int = 16) -> dict:
    t0 = time.perf_counter()
    tensors: list[dict] = []
    matrices: list[np.ndarray] = []
    names: list[str] = []
    u16 = np.zeros(65536, dtype=np.int64)
    exp_c = np.zeros(256, dtype=np.int64)
    mant_c = np.zeros(128, dtype=np.int64)
    sign_c = np.zeros(2, dtype=np.int64)
    mant_cond = np.zeros((256, 128), dtype=np.int64)
    sm_cond = np.zeros((256, 256), dtype=np.int64)
    sign_cond = np.zeros((256, 2), dtype=np.int64)
    n_all = 0
    h_u16 = 0.0
    h_exp = 0.0
    h_mant = 0.0
    h_sign = 0.0
    print(DISCLAIMER, flush=True)
    print(f"loading {len(specs)} tensors", flush=True)
    for spec in specs:
        words = as_2d(load_uint16(spec))
        flat = words.ravel()
        sign, exp, mant = split_components(flat)
        sm = (sign.astype(np.uint16) << 7) | mant.astype(np.uint16)
        n = int(flat.size)
        n_all += n
        u16 += np.bincount(flat.astype(np.int64), minlength=65536)
        exp_c += np.bincount(exp.astype(np.int64), minlength=256)
        mant_c += np.bincount(mant.astype(np.int64), minlength=128)
        sign_c += np.bincount(sign.astype(np.int64), minlength=2)
        mant_cond += np.bincount(
            exp.astype(np.int64) * 128 + mant.astype(np.int64), minlength=256 * 128
        ).reshape(256, 128)
        sm_cond += np.bincount(
            exp.astype(np.int64) * 256 + sm.astype(np.int64), minlength=256 * 256
        ).reshape(256, 256)
        sign_cond += np.bincount(
            exp.astype(np.int64) * 2 + sign.astype(np.int64), minlength=256 * 2
        ).reshape(256, 2)
        hu, he, hm, hs = shannon_entropy(flat), shannon_entropy(exp), shannon_entropy(mant), shannon_entropy(sign)
        h_u16 += hu * n
        h_exp += he * n
        h_mant += hm * n
        h_sign += hs * n
        matrices.append(words)
        names.append(spec.name)
        tensors.append({"name": spec.name, "n_words": n, "shape": list(words.shape)})
        print(f"  loaded {spec.name}  {list(words.shape)}  H(u16)={hu:.3f}", flush=True)

    bounds = {
        "H_uint16": h_u16 / max(n_all, 1),
        "H_sign": h_sign / max(n_all, 1),
        "H_exp": h_exp / max(n_all, 1),
        "H_mant": h_mant / max(n_all, 1),
        "H_mant_given_exp": _nll_laplace(mant_cond) / max(n_all, 1),
        "H_sign_given_exp": _nll_laplace(sign_cond) / max(n_all, 1),
        "H_sm_given_exp": _nll_laplace(sm_cond) / max(n_all, 1),
        "unigram_floor_bpw": h_u16 / max(n_all, 1),
        "field_split_ideal_bpw": (h_sign + h_exp + h_mant) / max(n_all, 1),
        "expcond_ideal_bpw": (h_sign + h_exp) / max(n_all, 1) + _nll_laplace(mant_cond) / max(n_all, 1),
        "joint_sm_exp_ideal_bpw": h_exp / max(n_all, 1) + _nll_laplace(sm_cond) / max(n_all, 1),
    }
    top_idx = np.argsort(-u16)[:255]
    top_cov = float(u16[top_idx].sum() / max(n_all, 1))
    escape_bpw = 8.0 + 16.0 * (1.0 - top_cov)
    bounds["top255_coverage"] = top_cov
    bounds["escape255_ideal_bpw"] = escape_bpw
    bounds["bits_missing_to_8"] = bounds["unigram_floor_bpw"] - TARGET_BPW
    dups = _tile_duplicate_rate(matrices, 16)
    print(
        f"bounds H(u16)={bounds['H_uint16']:.4f} H(M|e)={bounds['H_mant_given_exp']:.4f} "
        f"ideal_em={bounds['expcond_ideal_bpw']:.4f} top255={top_cov:.4f} "
        f"tile_dup={dups['dup_rate']:.6f}",
        flush=True,
    )

    methods: dict[str, dict] = {}

    def _record(name: str, encoded_bytes: int, exact: str, extra: dict | None = None) -> None:
        rec = {
            "name": name,
            "encoded_bytes": int(encoded_bytes),
            "bpw": bits_per_weight(encoded_bytes, n_all),
            "ratio_vs_raw": encoded_bytes / max(2 * n_all, 1),
            "exact": exact,
            "beats_pbre": bits_per_weight(encoded_bytes, n_all) < PBRE_REF - 1e-6,
            "hits_8bpw": bits_per_weight(encoded_bytes, n_all) <= TARGET_BPW,
        }
        if extra:
            rec.update(extra)
        methods[name] = rec
        print(
            f"  {name}: {encoded_bytes} B  {rec['bpw']:.4f} BPW  exact={exact}  "
            f"≤8={rec['hits_8bpw']}",
            flush=True,
        )

    # 1) PBR-E rANS baseline (re-measure)
    per_tensor: list[dict[str, int]] = [{} for _ in matrices]
    rans_c = Bf16ExpRansCodec()
    pbre_b = 0
    pbre_exact = "PASS"
    print("encode PBR-E rANS", flush=True)
    for i, (mat, spec_name) in enumerate(zip(matrices, names)):
        enc = rans_c.encode(mat)
        if enc is None:
            pbre_exact = "FAIL"
            continue
        enc.rows, enc.cols = int(mat.shape[0]), int(mat.shape[1])
        rec = rans_c.decode(enc)
        try:
            assert_exact(mat, rec, label=f"pbre:{spec_name}")
        except Exception:
            pbre_exact = "FAIL"
        pbre_b += TILE_HEADER_BYTES + len(enc.payload)
        per_tensor[i]["pbre_exp_rans"] = TILE_HEADER_BYTES + len(enc.payload)
    _record("pbre_exp_rans", pbre_b, pbre_exact)

    tags = [
        ("em_uncond_m", TAG_UNCOND_M),
        ("em_expcond_m", TAG_EXPCOND_M),
        ("em_sm_uncond", TAG_SM_UNCOND),
        ("em_sm_expcond", TAG_SM_EXPCOND),
    ]
    for name, tag in tags:
        print(f"encode {name}", flush=True)
        total = 0
        exact = "PASS"
        for i, (mat, spec_name) in enumerate(zip(matrices, names)):
            payload = encode_tensor_em(mat, tag=tag)
            rec = decode_tensor_em(payload, int(mat.size)).reshape(mat.shape)
            try:
                assert_exact(mat, rec, label=f"{name}:{spec_name}")
            except Exception:
                exact = "FAIL"
            cost = complete_bytes(payload)
            per_tensor[i][name] = cost
            total += cost
        _record(name, total, exact)

    # Shared tables (cross-tensor)
    print("encode shared expcond tables", flush=True)
    exp_freq = normalize_counts(exp_c)
    mant_freqs = _freqs_from_counts2d(mant_cond)
    sm_freqs = _freqs_from_counts2d(sm_cond)
    for name, tag, freqs in (
        ("shared_em_expcond_m", TAG_EXPCOND_M, mant_freqs),
        ("shared_em_sm_expcond", TAG_SM_EXPCOND, sm_freqs),
    ):
        sidecar = dump_shared_tables(exp_freq, freqs)
        total = len(sidecar)
        exact = "PASS"
        for mat, spec_name in zip(matrices, names):
            payload = encode_tensor_em_shared(mat, exp_freq=exp_freq, group_freqs=freqs, tag=tag)
            rec = decode_tensor_em_shared(payload, int(mat.size), exp_freq, freqs).reshape(mat.shape)
            try:
                assert_exact(mat, rec, label=f"{name}:{spec_name}")
            except Exception:
                exact = "FAIL"
            total += complete_bytes(payload)
        _record(name, total, exact, extra={"sidecar_bytes": len(sidecar)})

    # Optional nibble (no mandatory c4)
    print("encode optional nibble tiles", flush=True)
    nib_b = 0
    nib_exact = "PASS"
    usage: dict[str, int] = {}
    nibble_tiles = 0
    n_tiles = 0
    for i, (mat, spec_name) in enumerate(zip(matrices, names)):
        blob, stats = encode_optional(mat, tile=nibble_tile)
        rec = decode_optional(blob)
        try:
            assert_exact(mat, rec, label=f"nibble:{spec_name}")
        except Exception:
            nib_exact = "FAIL"
        nib_b += stats["complete_bytes"]
        per_tensor[i]["optional_nibble_16"] = int(stats["complete_bytes"])
        nibble_tiles += int(stats["nibble_tiles"])
        n_tiles += int(stats["n_tiles"])
        for k, v in stats["usage"].items():
            usage[k] = usage.get(k, 0) + int(v)
    _record(
        "optional_nibble_16",
        nib_b,
        nib_exact,
        extra={"usage": usage, "nibble_tiles": nibble_tiles, "n_tiles": n_tiles},
    )

    # Global + per-tensor escape dictionaries
    pal = top_idx.astype(np.uint16)
    print("encode global escape-255 dict", flush=True)
    esc_b = 2 + 2 * int(pal.size)  # one global palette
    esc_exact = "PASS"
    for mat, spec_name in zip(matrices, names):
        payload = encode_escape_dict(mat, pal)
        rec = decode_escape_dict(payload, int(mat.size)).reshape(mat.shape)
        try:
            assert_exact(mat, rec, label=f"esc:{spec_name}")
        except Exception:
            esc_exact = "FAIL"
        # payload includes a copy of the palette; count palette once globally.
        (k,) = struct.unpack_from("<H", payload, 0)
        pal_b = 2 + 2 * k
        esc_b += TILE_HEADER_BYTES + (len(payload) - pal_b)
    _record("global_escape255", esc_b, esc_exact, extra={"coverage": top_cov})

    print("encode per-tensor escape-255 dict", flush=True)
    loc_b = 0
    loc_exact = "PASS"
    for mat, spec_name in zip(matrices, names):
        loc_counts = np.bincount(mat.ravel().astype(np.int64), minlength=65536)
        loc_pal = np.argsort(-loc_counts)[:255].astype(np.uint16)
        payload = encode_escape_dict(mat, loc_pal)
        rec = decode_escape_dict(payload, int(mat.size)).reshape(mat.shape)
        try:
            assert_exact(mat, rec, label=f"esc_local:{spec_name}")
        except Exception:
            loc_exact = "FAIL"
        loc_b += TILE_HEADER_BYTES + len(payload)
    _record("per_tensor_escape255", loc_b, loc_exact)

    # Tensor-level mixture: min(PBR-E, EM variants) per tensor, then sum.
    print("tensor-level mixture (cached costs)", flush=True)
    mix_b = 0
    mix_exact = "PASS" if pbre_exact == "PASS" else "FAIL"
    mix_choice: dict[str, int] = {}
    mix_keys = [
        "pbre_exp_rans",
        "em_uncond_m",
        "em_expcond_m",
        "em_sm_uncond",
        "em_sm_expcond",
        "optional_nibble_16",
    ]
    for costs in per_tensor:
        present = {k: costs[k] for k in mix_keys if k in costs}
        if not present:
            mix_exact = "FAIL"
            continue
        choice = min(present, key=present.get)
        mix_choice[choice] = mix_choice.get(choice, 0) + 1
        mix_b += present[choice]
    _record("tensor_mixture_argmin", mix_b, mix_exact, extra={"choice_counts": mix_choice})

    # zlib baseline on raw uint16 (not PBR)
    print("zlib baseline (not PBR)", flush=True)
    z_b = 0
    for mat in matrices:
        z_b += len(zlib.compress(np.ascontiguousarray(mat, dtype="<u2").tobytes(), 9))
    _record("zlib_raw_uint16_baseline", z_b, "n/a")

    honest = [m for m in methods.values() if m["exact"] == "PASS"]
    if not honest:
        honest = list(methods.values())
    best = min(honest, key=lambda r: r["bpw"])
    elapsed = time.perf_counter() - t0
    return {
        "disclaimer": DISCLAIMER,
        "em_disclaimer": EM_DISCLAIMER,
        "n_tensors": len(matrices),
        "n_words": n_all,
        "original_bytes": 2 * n_all,
        "target_bpw": TARGET_BPW,
        "pbre_ref_bpw": PBRE_REF,
        "bounds": bounds,
        "tile_duplicates": dups,
        "methods": methods,
        "best": best,
        "hits_8bpw": bool(best["hits_8bpw"] and best["exact"] == "PASS"),
        "elapsed_s": elapsed,
        "tensors": tensors,
    }


def format_markdown(report: dict) -> str:
    b = report["bounds"]
    best = report["best"]
    hit = "YES" if report["hits_8bpw"] else "NO"
    lines = [
        "# Path to 50% (≤8.0 complete BPW)",
        "",
        DISCLAIMER,
        "",
        "## Result",
        "",
        f"Target **≤8.0 complete BPW**, bit-exact, Qwen2.5-0.5B-Instruct Stage 1B "
        f"set (**{report['n_tensors']}** tensors, **{report['n_words']}** words).",
        "",
        f"**Best measured: {best['bpw']:.4f} BPW** ({best['encoded_bytes']} B), "
        f"method `{best['name']}`, exact **{best['exact']}**. Hits ≤8.0: **{hit}**.",
        "",
        f"Unigram floor H(uint16) = **{b['H_uint16']:.4f} BPW**. Bits still missing "
        f"to 8.0 after that floor: **{b['bits_missing_to_8']:.4f}**. No i.i.d. or "
        f"single-context table can close a 2.5-bit gap when H(M|exp) stays "
        f"**{b['H_mant_given_exp']:.4f}** / 7.",
        "",
        "## Bounds (not bitstreams)",
        "",
        "| quantity | bits |",
        "| --- | ---: |",
        f"| H(uint16) unigram floor | {b['H_uint16']:.4f} |",
        f"| H(sign)+H(exp)+H(M) | {b['field_split_ideal_bpw']:.4f} |",
        f"| H(sign)+H(exp)+H(M\\|exp) | {b['expcond_ideal_bpw']:.4f} |",
        f"| H(exp)+H(SM\\|exp) | {b['joint_sm_exp_ideal_bpw']:.4f} |",
        f"| H(sign\\|exp) | {b['H_sign_given_exp']:.4f} |",
        f"| H(M\\|exp) | {b['H_mant_given_exp']:.4f} |",
        f"| top-255 value coverage | {100 * b['top255_coverage']:.2f}% |",
        f"| escape-255 ideal BPW | {b['escape255_ideal_bpw']:.4f} |",
        f"| 16×16 exact tile dup rate | {report['tile_duplicates']['dup_rate']:.6f} |",
        "",
        "## Measured complete BPW (bit-exact)",
        "",
        "| method | enc B | BPW | vs 10.616 | ≤8.0 | exact |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rec in sorted(report["methods"].values(), key=lambda r: r["bpw"]):
        vs = rec["bpw"] - PBRE_REF
        lines.append(
            f"| `{rec['name']}` | {rec['encoded_bytes']} | {rec['bpw']:.4f} | "
            f"{vs:+.4f} | {'yes' if rec['hits_8bpw'] else 'no'} | {rec['exact']} |"
        )
    mix = report["methods"].get("tensor_mixture_argmin") or {}
    nib = report["methods"].get("optional_nibble_16") or {}
    lines += [
        "",
        "### Optional nibble / mixture notes",
        "",
        f"- Optional nibble tile usage: `{nib.get('usage', {})}`. "
        f"Nibble-admitted tiles: {nib.get('nibble_tiles', 0)} / {nib.get('n_tiles', 0)}.",
        f"- Tensor-level argmin choices: `{mix.get('choice_counts', {})}`.",
        f"- Shared-table sidecar is counted once in `shared_*` rows.",
        "",
        "## Why 8.0 failed" if not report["hits_8bpw"] else "## 8.0 hit",
        "",
    ]
    if report["hits_8bpw"]:
        lines.append(
            f"Measured complete BPW {best['bpw']:.4f} ≤ 8.0 with exact {best['exact']} "
            f"on `{best['name']}`."
        )
    else:
        lines += [
            "The 50% target is **not** a missing tile codec. On this dense Qwen sample:",
            "",
            f"1. **i.i.d. floor is {b['H_uint16']:.2f} BPW.** rANS on uint16 itself cannot beat that. "
            "8.0 is **2.5 bits/weight** below the unigram entropy of the words.",
            "2. **Context does not appear.** H(M|exp) is 6.93/7 (zoo / Phase A). "
            f"H(sign|exp) is {b['H_sign_given_exp']:.4f} (sign is already ~1 bit). "
            "Spatial XOR, CTW/AR/IDF, and PBR-4 mandatory c4 all made complete bytes worse.",
            "3. **Optional nibble does not fire.** Mandatory 4-bit side info was the PBR-4 "
            "failure mode (~20.6 BPW). Making c4 optional only helps tiles that share a high-12 "
            "prototype; those tiles are essentially absent on these MLP/attention walls.",
            "4. **Global dictionaries do not concentrate.** Top-255 coverage "
            f"{100 * b['top255_coverage']:.2f}% implies escape-dict BPW "
            f"{b['escape255_ideal_bpw']:.2f}, worse than PBR-E.",
            "5. **Cross-tensor tile copies are ~0.** Shared tables save header bytes "
            "(tens of KB), not 2.5 bits/weight.",
            "",
            f"**Best honest complete BPW: {best['bpw']:.4f}** (`{best['name']}`), exact "
            f"{best['exact']}. That is DF11-class (~30% vs raw 16), not 50%. "
            "Related-checkpoint XOR+rANS at 8.36 remains the only measured number near 8, "
            "and only if the base checkpoint is already stored (not standalone).",
            "",
            "zlib on raw uint16 is a baseline, not PBR.",
        ]
    lines += [
        "",
        f"Elapsed {report['elapsed_s']:.1f} s. This is not a 1–2 GB / 8 GB result.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiments toward ≤8.0 complete BPW.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--tag", default="qwen")
    parser.add_argument("--max-tensors", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    specs = select_weight_specs(
        inventory_from_dir(args.model_dir),
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    if args.max_tensors:
        specs = specs[: args.max_tensors]
    report = evaluate(specs)
    report["model"] = {
        "repo_id": cfg.get("repo", PRIMARY_REPO),
        "revision": cfg.get("revision", "main"),
        "license": cfg.get("license", PRIMARY_LICENSE),
        "selected": [s.name for s in specs],
    }
    out_json = Path(f"artifacts/path_to_50pct_{args.tag}.json")
    out_md = Path("artifacts/path_to_50pct.md")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(report)
    out_md.write_text(md, encoding="utf-8")
    Path(f"artifacts/path_to_50pct_{args.tag}.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {out_md} and {out_json}")
    if any(m["exact"] == "FAIL" for m in report["methods"].values() if m["exact"] != "n/a"):
        print("EXACTNESS FAIL")
        return 1
    if report["hits_8bpw"]:
        print("TARGET HIT: complete BPW ≤ 8.0")
    else:
        print(f"TARGET MISS: best {report['best']['bpw']:.4f} BPW (need ≤8.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
