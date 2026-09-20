#!/usr/bin/env python3
"""Freeze H95Q candidates into physical .h95q containers and prove exact decode.

Candidates (exact names):
  H95Q-Conservative = B1 + conservative embed tiers
  H95Q-Balanced     = B1 + aggressive embed (NO mid-MLP K3)
  H95Q-S1           = B1 + aggressive embed + mlp band 8-15 @K3

Container versions:
  --version 1  raw-u8 exponents (legacy)
  --version 2  adaptive exact exponent compression (default)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pbr_h95.apply_policy import bf16_tensor_to_u16, clone_cpu_state_dict
from pbr_h95.container_h95q import (
    VERSION_V1,
    VERSION_V2,
    decode_container,
    encode_container,
    read_header,
    sha256_state_u16,
    verify_decoded_against_reference,
)
from pbr_h95.embed_tiers import find_embed_param
from pbr_h95.h95q_stack import (
    apply_b1_plus_embed_tiers,
    apply_mlp_k3_policy,
    fit_embed_tiers_on_calib,
    rate_b1_embed_tiers,
    rate_mlp_k3,
)
from pbr_h95.quantize import quantize_bf16_mantissas

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
CONTAINER_DIR = Path("artifacts/pbr_h95/containers")

FROZEN_V1_SHA = {
    "H95Q-Conservative": "9c0247096cdcaeee1e0cedc4d734396962aa0e43fc2959b602d8ba749479bc1a",
    "H95Q-Balanced": "3610ea3ccac36ac646bea028222e3900ca75448989f869b6165e297a50461ef8",
    "H95Q-S1": "eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de",
}

V1_ACTUAL_BPW = {
    "H95Q-Conservative": 13.293241,
    "H95Q-Balanced": 13.012932,
    "H95Q-S1": 12.801215,
}

PRIOR_EST = {
    "H95Q-Conservative": {
        "stack_name": "T1_B1_embed_conservative",
        "est_bpw": 7.8937,
        "heldout_ret": 0.9923,
        "proxy_gate": "proxy_GO",
    },
    "H95Q-Balanced": {
        "stack_name": "T1_B1_embed_aggressive",
        "est_bpw": 7.6134,
        "heldout_ret": 0.9897,
        "proxy_gate": "proxy_GO",
    },
    "H95Q-S1": {
        "stack_name": "S1_B1_aggr_embed_band815_k3",
        "est_bpw": 7.4017,
        "heldout_ret": 0.9900,
        "proxy_gate": "proxy_GO",
    },
}

HONESTY_V1 = [
    "actual_bpw = physical file_bytes * 8 / n_weights (includes header/metadata).",
    "Exponents stored as raw u8 for exactness — NOT the ~2.62 BPW entropy reference used in estimated packed BPW.",
    "Therefore actual_bpw is expected >> est_bpw (~+5.38 on the exp term alone) even before metadata.",
    "Prior estimated packed BPW + heldout retention copied from H95Q-stack artifacts (proxy only).",
    "Exactness gate: decode(encode(Q(W))) == Q(W) uint16 bit-identical + matching SHA-256.",
    "Not production quality / multilingual / runtime RAM == BPW. Large .h95q blobs are gitignored.",
]

HONESTY_V2 = [
    "actual_bpw = physical file_bytes * 8 / n_weights (includes header/metadata) — file size only.",
    "v2 replaces only the raw-u8 exponent stream with adaptive exact coding (RAW8/rANS/Huffman/delta-rANS/run-rANS).",
    "Complete exponent cost = payload + probability table + mode_id + stream_length + final_state (+ section bytes).",
    "Reported exp section bytes == physical sum(exp_len); reported file bytes == physical file size.",
    "S1 precision policy / embed tiers / mid-MLP K3 / mantissa / signs / tensor order FROZEN — Q(W) SHA must match v1.",
    "Prior heldout retention 0.990 is proxy from prior stack — not re-evaluated here.",
    "Not production quality / multilingual / runtime RAM == BPW. Large .h95q blobs are gitignored.",
]


def load_texts(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [t["text"] for t in payload["texts"]]


def unique_float_state(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for name, t in sd.items():
        if not t.is_floating_point():
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        out[name] = t
    return out


def state_to_u16(sd: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {n: bf16_tensor_to_u16(t) for n, t in unique_float_state(sd).items()}


def assert_q_idempotent(
    u16_map: dict[str, np.ndarray],
    keep_map: dict[str, int],
    row_keeps: np.ndarray | None,
    embed_name: str,
) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for name, words in list(u16_map.items())[:16]:
        if name == embed_name and row_keeps is not None and words.ndim == 2:
            for r in (0, len(row_keeps) // 2, len(row_keeps) - 1):
                k = int(row_keeps[r])
                row = words[r]
                q1 = quantize_bf16_mantissas(row, k)
                q2 = quantize_bf16_mantissas(q1, k)
                checked += 1
                if not np.array_equal(q1, q2):
                    failures.append(f"{name}[row{r}]")
            continue
        k = int(keep_map.get(name, 7))
        if k < 0:
            k = 4
        flat = words.ravel()
        sample = flat if flat.size <= 4096 else flat[:4096]
        q1 = quantize_bf16_mantissas(sample, k)
        q2 = quantize_bf16_mantissas(q1, k)
        checked += 1
        if not np.array_equal(q1, q2):
            failures.append(name)
    return {"checked": checked, "all_idempotent": len(failures) == 0, "failures": failures}


def decode_in_subprocess(container_path: Path, expect_sha: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pbr_h95.container_h95q",
        str(container_path),
        "--expect-sha",
        expect_sha,
    ]
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(Path.cwd()))
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "sha_ok_subprocess": proc.returncode == 0 and "sha_match=True" in proc.stdout,
    }


def _container_name(name: str, version: int) -> str:
    if version >= VERSION_V2:
        return f"{name}-v2.h95q"
    return f"{name}.h95q"


def _report_paths(version: int) -> tuple[Path, Path]:
    if version >= VERSION_V2:
        return (
            Path("artifacts/pbr_h95/h95q_container_v2_qwen.json"),
            Path("artifacts/pbr_h95/h95q_container_v2_qwen.md"),
        )
    return (
        Path("artifacts/pbr_h95/h95q_container_qwen.json"),
        Path("artifacts/pbr_h95/h95q_container_qwen.md"),
    )


def _load_row_keeps_from_v1(v1_path: Path, header: dict, embed_name: str | None) -> np.ndarray | None:
    if not embed_name:
        return None
    spec = next(s for s in header["tensors"] if s["name"] == embed_name)
    data = v1_path.read_bytes()
    return np.frombuffer(
        data[spec["row_keeps_off"] : spec["row_keeps_off"] + spec["row_keeps_len"]],
        dtype=np.int8,
    ).copy()


def transcode_from_v1(
    name: str,
    *,
    version: int,
    container_dir: Path,
    v1_path: Path | None = None,
) -> dict[str, Any]:
    """Re-encode a frozen v1 container as v2 (or v1) without touching the model."""
    print(f"\n=== {name} (transcode from v1) ===", flush=True)
    t0 = time.perf_counter()
    src = v1_path or (container_dir / f"{name}.h95q")
    if not src.is_file():
        raise FileNotFoundError(src)

    t_dec0 = time.perf_counter()
    v1 = decode_container(src)
    wall_v1_dec = time.perf_counter() - t_dec0
    header = v1["header"]
    frozen = FROZEN_V1_SHA.get(name)
    ref_sha = header["sha256_quantized_reference"]
    if frozen and ref_sha != frozen:
        raise RuntimeError(f"v1 SHA drift for {name}: {ref_sha} != {frozen}")

    unique = {spec["name"]: v1["tensors"][spec["name"]] for spec in header["tensors"]}
    keep_map: dict[str, int] = {}
    embed_name = None
    for spec in header["tensors"]:
        if spec["mode"] == "uniform":
            keep_map[spec["name"]] = int(spec["base_keep"])
        else:
            embed_name = spec["name"]
            keep_map[spec["name"]] = -1
    row_keeps = _load_row_keeps_from_v1(src, header, embed_name)
    precision = header.get("precision_map") or {}
    idem = assert_q_idempotent(unique, keep_map, row_keeps, embed_name or "")
    wall_build = time.perf_counter() - t0

    out_path = container_dir / _container_name(name, version)
    t_enc = time.perf_counter()
    enc = encode_container(
        out_path,
        unique,
        keep_map=keep_map,
        model_id=str(header.get("model_id") or "qwen"),
        candidate=name,
        precision_map=precision,
        embed_name=embed_name,
        embed_row_keeps=row_keeps,
        version=version,
    )
    wall_enc = time.perf_counter() - t_enc
    print(
        f"  encoded file_bytes={enc['file_bytes']} actual_bpw={enc['actual_bpw']:.4f} "
        f"exp_bpw={enc.get('exp_complete_bpw', float('nan')):.4f} "
        f"modes={enc.get('exp_mode_histogram')} wall_enc={wall_enc:.1f}s",
        flush=True,
    )

    t_dec = time.perf_counter()
    decoded = decode_container(out_path)
    verify = verify_decoded_against_reference(decoded["tensors"], unique)
    # E4: vs v1 words
    word_diffs = 0
    for tname in unique:
        a = unique[tname].ravel()
        b = decoded["tensors"][tname].ravel()
        if a.shape != b.shape or not np.array_equal(a, b):
            word_diffs += int(np.sum(a != b)) if a.shape == b.shape else int(a.size)
    wall_dec = time.perf_counter() - t_dec
    if not verify["exact_match"] or not verify["sha_ok"] or word_diffs:
        raise RuntimeError(f"exactness FAILED for {name}: diffs={word_diffs} {verify}")

    if enc["sha256_quantized_reference"] != ref_sha:
        raise RuntimeError("encoded SHA != v1 reference SHA")

    sub = decode_in_subprocess(out_path, ref_sha)
    if not sub["sha_ok_subprocess"]:
        raise RuntimeError(f"subprocess decode failed: {sub}")

    prior = PRIOR_EST[name]
    actual = float(enc["actual_bpw"])
    v1_actual = float(V1_ACTUAL_BPW.get(name) or 0.0)
    exp_bpw = float(enc.get("exp_complete_bpw") or 0.0)
    row = {
        "name": name,
        "stack_name": prior["stack_name"],
        "precision_map": precision,
        "container_version": version,
        "est_bpw": prior["est_bpw"],
        "actual_bpw": actual,
        "v1_actual_bpw": v1_actual,
        "delta_vs_v1_bpw": round(actual - v1_actual, 6) if v1_actual else None,
        "stream_bpw": enc["stream_bpw"],
        "exp_complete_bpw": exp_bpw,
        "exp_section_bytes": enc.get("exp_section_bytes"),
        "exp_mode_histogram": enc.get("exp_mode_histogram") or {},
        "file_bytes": enc["file_bytes"],
        "overhead_bytes": enc["overhead_bytes"],
        "header_bytes": enc["header_bytes"],
        "payload_stream_bytes": enc["payload_stream_bytes"],
        "n_weights": enc["n_weights"],
        "n_tensors": enc["n_tensors"],
        "exact_match": True,
        "sha_ok": True,
        "word_diffs_vs_v1": 0,
        "sha256_quantized_reference": ref_sha,
        "sha256_file": enc["sha256_file"],
        "container_path": str(out_path),
        "source_v1_path": str(src),
        "exp_packing": enc["exp_packing"],
        "prior_heldout_retention": prior["heldout_ret"],
        "prior_proxy_gate": prior["proxy_gate"],
        "actual_bpw_le_8": bool(actual <= 8.0),
        "exp_bpw_le_3_2": bool(exp_bpw <= 3.20),
        "idempotence": idem,
        "subprocess_decode": {"returncode": sub["returncode"], "sha_ok": True},
        "wall_build_s": round(wall_build, 2),
        "wall_v1_decode_s": round(wall_v1_dec, 2),
        "wall_encode_s": round(wall_enc, 2),
        "wall_decode_verify_s": round(wall_dec, 2),
        "note": (
            "Transcoded from frozen v1 container; Q(W) SHA frozen. "
            "Quality retention is prior proxy (0.99), not re-eval'd."
        ),
    }
    print(
        f"  exact=YES sha=OK word_diffs=0 actual_bpw={actual:.4f} "
        f"exp_bpw={exp_bpw:.4f} <=8? {row['actual_bpw_le_8']}",
        flush=True,
    )
    del unique, decoded, v1
    gc.collect()
    return row


def run_one_from_model(
    name: str,
    model,
    baseline_sd,
    *,
    embed_name: str,
    tier_cache: dict,
    model_id: str,
    container_dir: Path,
    version: int,
) -> dict[str, Any]:
    print(f"\n=== {name} (from model, version={version}) ===", flush=True)
    t0 = time.perf_counter()

    if name == "H95Q-Conservative":
        row_keeps, tier_meta = tier_cache["conservative"]
        meta = apply_b1_plus_embed_tiers(
            model, baseline_sd=baseline_sd, row_keeps=row_keeps, embed_name=embed_name
        )
        rate = rate_b1_embed_tiers(
            meta["keep_map"], baseline_sd, embed_name=embed_name, row_keeps=row_keeps
        )
        precision = {
            "body": "B1",
            "embed_schedule": "conservative",
            "mlp_k3_bands": [],
            "tier_schedule": tier_meta.get("schedule"),
            "keep_hist_rows": tier_meta.get("keep_hist_rows"),
        }
    elif name == "H95Q-Balanced":
        row_keeps, tier_meta = tier_cache["aggressive"]
        meta = apply_b1_plus_embed_tiers(
            model, baseline_sd=baseline_sd, row_keeps=row_keeps, embed_name=embed_name
        )
        rate = rate_b1_embed_tiers(
            meta["keep_map"], baseline_sd, embed_name=embed_name, row_keeps=row_keeps
        )
        precision = {
            "body": "B1",
            "embed_schedule": "aggressive",
            "mlp_k3_bands": [],
            "note": "NO mid-MLP K3",
            "tier_schedule": tier_meta.get("schedule"),
            "keep_hist_rows": tier_meta.get("keep_hist_rows"),
        }
    elif name == "H95Q-S1":
        row_keeps, tier_meta = tier_cache["aggressive"]
        meta = apply_mlp_k3_policy(
            model,
            baseline_sd=baseline_sd,
            variant="band_8_15",
            embed_row_keeps=row_keeps,
            embed_name=embed_name,
        )
        rate = rate_mlp_k3(meta, baseline_sd, embed_name=embed_name, embed_row_keeps=row_keeps)
        precision = {
            "body": "B1",
            "embed_schedule": "aggressive",
            "mlp_k3_bands": ["mlp_band_8_15"],
            "k3_variant": meta.get("k3_variant"),
            "tier_schedule": tier_meta.get("schedule"),
            "keep_hist_rows": tier_meta.get("keep_hist_rows"),
        }
    else:
        raise KeyError(name)

    keep_map = {k: int(v) for k, v in meta["keep_map"].items()}
    u16_map = state_to_u16(model.state_dict())
    idem = assert_q_idempotent(u16_map, keep_map, row_keeps, embed_name)
    if not idem["all_idempotent"]:
        raise RuntimeError(f"Q not idempotent: {idem}")
    ref_sha = sha256_state_u16(u16_map)
    frozen = FROZEN_V1_SHA.get(name)
    if frozen and ref_sha != frozen:
        raise RuntimeError(f"quantized SHA drift for {name}: {ref_sha} != frozen {frozen}")
    wall_build = time.perf_counter() - t0

    out_path = container_dir / _container_name(name, version)
    t_enc = time.perf_counter()
    enc = encode_container(
        out_path,
        u16_map,
        keep_map=keep_map,
        model_id=model_id,
        candidate=name,
        precision_map=precision,
        embed_name=embed_name,
        embed_row_keeps=row_keeps,
        version=version,
    )
    wall_enc = time.perf_counter() - t_enc

    t_dec = time.perf_counter()
    decoded = decode_container(out_path)
    verify = verify_decoded_against_reference(decoded["tensors"], u16_map)
    wall_dec = time.perf_counter() - t_dec
    if not verify["exact_match"] or not verify["sha_ok"]:
        raise RuntimeError(f"exactness FAILED for {name}: {verify}")

    sub = decode_in_subprocess(out_path, ref_sha)
    if not sub["sha_ok_subprocess"]:
        raise RuntimeError(f"subprocess decode failed: {sub}")

    prior = PRIOR_EST[name]
    actual = float(enc["actual_bpw"])
    v1_actual = float(V1_ACTUAL_BPW.get(name) or 0.0)
    exp_bpw = float(enc.get("exp_complete_bpw") or 0.0)
    live_est = float(rate.get("packed_K_total_bpw") or 0.0)
    row = {
        "name": name,
        "stack_name": prior["stack_name"],
        "precision_map": precision,
        "container_version": version,
        "est_bpw": prior["est_bpw"],
        "est_bpw_live_recomputed": live_est,
        "actual_bpw": actual,
        "v1_actual_bpw": v1_actual,
        "delta_vs_v1_bpw": round(actual - v1_actual, 6) if v1_actual else None,
        "stream_bpw": enc["stream_bpw"],
        "exp_complete_bpw": exp_bpw,
        "exp_section_bytes": enc.get("exp_section_bytes"),
        "exp_mode_histogram": enc.get("exp_mode_histogram") or {},
        "file_bytes": enc["file_bytes"],
        "overhead_bytes": enc["overhead_bytes"],
        "header_bytes": enc["header_bytes"],
        "payload_stream_bytes": enc["payload_stream_bytes"],
        "n_weights": enc["n_weights"],
        "n_tensors": enc["n_tensors"],
        "exact_match": True,
        "sha_ok": True,
        "word_diffs_vs_v1": None,
        "sha256_quantized_reference": ref_sha,
        "sha256_file": enc["sha256_file"],
        "container_path": str(out_path),
        "exp_packing": enc["exp_packing"],
        "prior_heldout_retention": prior["heldout_ret"],
        "prior_proxy_gate": prior["proxy_gate"],
        "actual_bpw_le_8": bool(actual <= 8.0),
        "exp_bpw_le_3_2": bool(exp_bpw <= 3.20),
        "idempotence": idem,
        "subprocess_decode": {"returncode": sub["returncode"], "sha_ok": True},
        "wall_build_s": round(wall_build, 2),
        "wall_encode_s": round(wall_enc, 2),
        "wall_decode_verify_s": round(wall_dec, 2),
    }
    print(
        f"  exact=YES sha=OK actual_bpw={actual:.4f} exp_bpw={exp_bpw:.4f} <=8? {row['actual_bpw_le_8']}",
        flush=True,
    )
    del u16_map, decoded
    gc.collect()
    return row


def write_report(rows: list[dict], wall_total: float, model_dir: str, version: int) -> None:
    out_json, out_md = _report_paths(version)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    honesty = HONESTY_V2 if version >= VERSION_V2 else HONESTY_V1
    s1 = next((r for r in rows if r["name"] == "H95Q-S1"), None)
    payload = {
        "phase": "H95Q-container-v2" if version >= VERSION_V2 else "H95Q-container",
        "model_dir": model_dir,
        "container_version": version,
        "magic": "H95Q",
        "exp_packing": "adaptive_exact_v2" if version >= VERSION_V2 else "raw_u8",
        "honesty": honesty,
        "wall_seconds": round(wall_total, 2),
        "candidates": rows,
        "summary": {
            "all_exact_match": all(r["exact_match"] for r in rows),
            "all_sha_ok": all(r["sha_ok"] for r in rows),
            "s1_actual_bpw_le_8": (s1["actual_bpw_le_8"] if s1 else None),
            "s1_actual_bpw": (s1["actual_bpw"] if s1 else None),
            "s1_exp_complete_bpw": (s1.get("exp_complete_bpw") if s1 else None),
            "s1_v1_actual_bpw": (s1.get("v1_actual_bpw") if s1 else None),
            "s1_mode_histogram": (s1.get("exp_mode_histogram") if s1 else None),
            "physical_le_8_gate": (
                "PASS" if s1 and s1["actual_bpw_le_8"] else ("FAIL" if s1 else None)
            ),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# PBR-H95Q physical containerization (v{version})",
        "",
        f"Model: `{model_dir}`",
        f"Container: magic `H95Q` version **{version}**; "
        + (
            "exp packing = **adaptive exact** (per-tensor mode competition)."
            if version >= VERSION_V2
            else "exp packing = **raw u8** (exactness)."
        ),
        f"Wall time: **{wall_total:.1f}s**",
        "",
        "## Honesty",
        "",
    ]
    for h in honesty:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Results",
        "",
        "| Candidate | v1 actual | v2 actual | exp BPW | modes | file_bytes | exact | ≤8? |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for r in rows:
        modes = r.get("exp_mode_histogram") or {}
        mode_s = ",".join(f"{k}:{v}" for k, v in sorted(modes.items())) or "RAW8"
        lines.append(
            f"| {r['name']} | {r.get('v1_actual_bpw', float('nan')):.4f} | {r['actual_bpw']:.4f} | "
            f"{r.get('exp_complete_bpw', float('nan')):.4f} | {mode_s} | "
            f"{r['file_bytes']:,} | "
            f"{'yes' if r['exact_match'] else 'NO'} | {r['actual_bpw_le_8']} |"
        )
    lines += ["", "## Per-candidate notes", ""]
    for r in rows:
        lines += [
            f"### {r['name']}",
            "",
            f"- Stack: `{r['stack_name']}`",
            f"- Container: `{r['container_path']}` (gitignored if large)",
            f"- SHA-256 quantized ref: `{r['sha256_quantized_reference']}`",
            f"- SHA-256 file: `{r['sha256_file']}`",
            f"- Exp complete BPW: {r.get('exp_complete_bpw')}",
            f"- Mode histogram: `{r.get('exp_mode_histogram')}`",
            f"- Δ vs v1 actual: {r.get('delta_vs_v1_bpw')}",
            f"- Prior heldout retention (stack proxy): {r['prior_heldout_retention']:.4f}",
            f"- Word diffs vs v1 decode: {r.get('word_diffs_vs_v1')}",
            "",
        ]
    if s1 is not None:
        lines += ["## S1 physical ≤8 gate", ""]
        if s1["actual_bpw"] <= 8:
            lines += [
                f"**PASS** — H95Q-S1 `actual_bpw={s1['actual_bpw']:.4f}` ≤ 8 "
                f"(v1 was {s1.get('v1_actual_bpw')}; exp complete {s1.get('exp_complete_bpw'):.4f} BPW).",
                "",
            ]
        else:
            lines += [
                f"**FAIL** — H95Q-S1 `actual_bpw={s1['actual_bpw']:.4f}` > 8 "
                f"(exp complete {s1.get('exp_complete_bpw')}).",
                "",
            ]
    lines += [
        "## Exactness gates",
        "",
        "- Q idempotent on samples: "
        + ("PASS" if all(r["idempotence"]["all_idempotent"] for r in rows) else "FAIL"),
        "- decode(encode(Q(W))) == Q(W) uint16: "
        + ("PASS" if all(r["exact_match"] for r in rows) else "FAIL"),
        "- SHA-256(ref) == SHA-256(decoded) == frozen v1: "
        + ("PASS" if all(r["sha_ok"] for r in rows) else "FAIL"),
        "- Subprocess decode-only entrypoint: "
        + ("PASS" if all(r["subprocess_decode"]["sha_ok"] for r in rows) else "FAIL"),
        "- decode(v1)==decode(v2) word-identical: "
        + (
            "PASS"
            if all(r.get("word_diffs_vs_v1") in (0, None) for r in rows)
            and any(r.get("word_diffs_vs_v1") == 0 for r in rows)
            else ("n/a" if all(r.get("word_diffs_vs_v1") is None for r in rows) else "FAIL")
        ),
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json} and {out_md}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=["H95Q-Conservative", "H95Q-Balanced", "H95Q-S1"],
    )
    ap.add_argument("--version", type=int, default=VERSION_V2, choices=[VERSION_V1, VERSION_V2])
    ap.add_argument(
        "--from-v1",
        action="store_true",
        default=True,
        help="Transcode frozen v1 containers (default for v2; guarantees SHA freeze)",
    )
    ap.add_argument(
        "--from-model",
        action="store_true",
        help="Rebuild Q(W) from model+policy instead of transcoding v1",
    )
    args = ap.parse_args()
    if args.from_model:
        args.from_v1 = False

    t_all = time.perf_counter()
    CONTAINER_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.from_v1 and args.version >= VERSION_V2:
        for name in args.candidates:
            rows.append(
                transcode_from_v1(name, version=args.version, container_dir=CONTAINER_DIR)
            )
    else:
        print("Loading model…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()
        baseline_sd = clone_cpu_state_dict(model)
        embed_name, emb_w = find_embed_param(model)
        vocab_rows = int(emb_w.shape[0])
        calib_texts = load_texts(args.calib)

        print("Fitting embed tiers on calib-v2…", flush=True)
        tier_cache: dict[str, tuple[np.ndarray, dict]] = {}
        for sched in ("conservative", "aggressive"):
            rk, meta = fit_embed_tiers_on_calib(
                tokenizer, calib_texts, vocab_rows=vocab_rows, schedule=sched, max_length=256
            )
            tier_cache[sched] = (rk, meta)
            print(f"  {sched}: avg_row_K={float(rk.mean()):.4f}", flush=True)

        model_id = str(getattr(getattr(model, "config", None), "_name_or_path", None) or args.model_dir)
        for name in args.candidates:
            rows.append(
                run_one_from_model(
                    name,
                    model,
                    baseline_sd,
                    embed_name=embed_name,
                    tier_cache=tier_cache,
                    model_id=model_id,
                    container_dir=CONTAINER_DIR,
                    version=args.version,
                )
            )

    wall = time.perf_counter() - t_all
    write_report(rows, wall, str(args.model_dir), args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
