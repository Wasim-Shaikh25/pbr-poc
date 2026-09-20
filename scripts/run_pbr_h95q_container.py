#!/usr/bin/env python3
"""Freeze H95Q candidates into physical .h95q containers and prove exact decode.

Candidates (exact names):
  H95Q-Conservative = B1 + conservative embed tiers
  H95Q-Balanced     = B1 + aggressive embed (NO mid-MLP K3)
  H95Q-S1           = B1 + aggressive embed + mlp band 8-15 @K3
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
    decode_container,
    encode_container,
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
OUT_JSON = Path("artifacts/pbr_h95/h95q_container_qwen.json")
OUT_MD = Path("artifacts/pbr_h95/h95q_container_qwen.md")
CONTAINER_DIR = Path("artifacts/pbr_h95/containers")

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

HONESTY = [
    "actual_bpw = physical file_bytes * 8 / n_weights (includes header/metadata).",
    "Exponents stored as raw u8 for exactness — NOT the ~2.62 BPW entropy reference used in estimated packed BPW.",
    "Therefore actual_bpw is expected >> est_bpw (~+5.38 on the exp term alone) even before metadata.",
    "Prior estimated packed BPW + heldout retention copied from H95Q-stack artifacts (proxy only).",
    "Exactness gate: decode(encode(Q(W))) == Q(W) uint16 bit-identical + matching SHA-256.",
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


def run_one(
    name: str,
    model,
    baseline_sd,
    *,
    embed_name: str,
    tier_cache: dict,
    model_id: str,
    container_dir: Path,
) -> dict[str, Any]:
    print(f"\n=== {name} ===", flush=True)
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
    wall_build = time.perf_counter() - t0

    out_path = container_dir / f"{name}.h95q"
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
    )
    wall_enc = time.perf_counter() - t_enc
    print(
        f"  encoded file_bytes={enc['file_bytes']} actual_bpw={enc['actual_bpw']:.4f} "
        f"est={PRIOR_EST[name]['est_bpw']:.4f} wall_enc={wall_enc:.1f}s",
        flush=True,
    )

    t_dec = time.perf_counter()
    decoded = decode_container(out_path)
    verify = verify_decoded_against_reference(decoded["tensors"], u16_map)
    wall_dec = time.perf_counter() - t_dec
    if not verify["exact_match"] or not verify["sha_ok"]:
        raise RuntimeError(f"exactness FAILED for {name}: {verify}")

    hdr_sha = decoded["header"]["sha256_quantized_reference"]
    if hdr_sha != ref_sha:
        raise RuntimeError(f"header sha mismatch: {hdr_sha} vs {ref_sha}")

    sub = decode_in_subprocess(out_path, ref_sha)
    if not sub["sha_ok_subprocess"]:
        raise RuntimeError(f"subprocess decode failed: {sub}")

    prior = PRIOR_EST[name]
    actual = float(enc["actual_bpw"])
    live_est = float(rate.get("packed_K_total_bpw") or 0.0)
    row = {
        "name": name,
        "stack_name": prior["stack_name"],
        "precision_map": precision,
        "est_bpw": prior["est_bpw"],
        "est_bpw_live_recomputed": live_est,
        "actual_bpw": actual,
        "stream_bpw": enc["stream_bpw"],
        "file_bytes": enc["file_bytes"],
        "overhead_bytes": enc["overhead_bytes"],
        "header_bytes": enc["header_bytes"],
        "payload_stream_bytes": enc["payload_stream_bytes"],
        "n_weights": enc["n_weights"],
        "n_tensors": enc["n_tensors"],
        "exact_match": True,
        "sha_ok": True,
        "sha256_quantized_reference": ref_sha,
        "sha256_file": enc["sha256_file"],
        "container_path": str(out_path),
        "exp_packing": "raw_u8",
        "prior_heldout_retention": prior["heldout_ret"],
        "prior_proxy_gate": prior["proxy_gate"],
        "actual_bpw_le_8": bool(actual <= 8.0),
        "idempotence": idem,
        "subprocess_decode": {"returncode": sub["returncode"], "sha_ok": True},
        "wall_build_s": round(wall_build, 2),
        "wall_encode_s": round(wall_enc, 2),
        "wall_decode_verify_s": round(wall_dec, 2),
        "delta_actual_minus_est_bpw": round(actual - prior["est_bpw"], 4),
        "note_bpw_gap": (
            "Gap vs est is dominated by raw_u8 exponents (8 BPW) vs EXP_BPW_REF=2.62 used in estimates; "
            "metadata/header overhead is secondary."
        ),
    }
    print(
        f"  exact=YES sha=OK subprocess=OK actual_bpw={actual:.4f} <=8? {row['actual_bpw_le_8']}",
        flush=True,
    )
    del u16_map, decoded
    gc.collect()
    return row


def write_report(rows: list[dict], wall_total: float, model_dir: str) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "H95Q-container",
        "model_dir": model_dir,
        "container_version": 1,
        "magic": "H95Q",
        "exp_packing": "raw_u8",
        "honesty": HONESTY,
        "wall_seconds": round(wall_total, 2),
        "candidates": rows,
        "summary": {
            "all_exact_match": all(r["exact_match"] for r in rows),
            "all_sha_ok": all(r["sha_ok"] for r in rows),
            "s1_actual_bpw_le_8": next((r["actual_bpw_le_8"] for r in rows if r["name"] == "H95Q-S1"), None),
            "s1_actual_bpw": next((r["actual_bpw"] for r in rows if r["name"] == "H95Q-S1"), None),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# PBR-H95Q physical containerization",
        "",
        f"Model: `{model_dir}`",
        "Container: magic `H95Q` version 1; exp packing = **raw u8** (exactness).",
        f"Wall time: **{wall_total:.1f}s**",
        "",
        "## Honesty",
        "",
    ]
    for h in HONESTY:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Results",
        "",
        "| Candidate | est_bpw (prior) | actual_bpw | file_bytes | overhead | exact | sha | ≤8? |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['est_bpw']:.4f} | {r['actual_bpw']:.4f} | "
            f"{r['file_bytes']:,} | {r['overhead_bytes']:,} | "
            f"{'yes' if r['exact_match'] else 'NO'} | {'ok' if r['sha_ok'] else 'FAIL'} | "
            f"{r['actual_bpw_le_8']} |"
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
            f"- Stream BPW (payload only): {r['stream_bpw']:.4f}",
            f"- Δ(actual − est): {r['delta_actual_minus_est_bpw']:+.4f} BPW",
            f"- Prior heldout retention (stack): {r['prior_heldout_retention']:.4f} ({r['prior_proxy_gate']})",
            f"- {r['note_bpw_gap']}",
            "",
        ]
    s1 = next((r for r in rows if r["name"] == "H95Q-S1"), None)
    if s1 is not None:
        lines += ["## S1 ≤8 physical?", ""]
        if s1["actual_bpw"] > 8:
            lines += [
                f"**No** — H95Q-S1 `actual_bpw={s1['actual_bpw']:.4f}` (> 8). "
                "Expected with raw-u8 exponents; prior ≤8 was on *estimated* packed BPW "
                "(EXP_BPW_REF=2.62). Container still proves exact independent decode of Q(W).",
                "",
            ]
        else:
            lines += [f"**Yes** — H95Q-S1 `actual_bpw={s1['actual_bpw']:.4f}` ≤ 8.", ""]
    lines += [
        "## Exactness gates",
        "",
        "- Q idempotent on samples: "
        + ("PASS" if all(r["idempotence"]["all_idempotent"] for r in rows) else "FAIL"),
        "- decode(encode(Q(W))) == Q(W) uint16: "
        + ("PASS" if all(r["exact_match"] for r in rows) else "FAIL"),
        "- SHA-256(ref) == SHA-256(decoded): "
        + ("PASS" if all(r["sha_ok"] for r in rows) else "FAIL"),
        "- Subprocess decode-only entrypoint: "
        + ("PASS" if all(r["subprocess_decode"]["sha_ok"] for r in rows) else "FAIL"),
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=["H95Q-Conservative", "H95Q-Balanced", "H95Q-S1"],
    )
    args = ap.parse_args()

    t_all = time.perf_counter()
    CONTAINER_DIR.mkdir(parents=True, exist_ok=True)

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
    rows = []
    for name in args.candidates:
        rows.append(
            run_one(
                name,
                model,
                baseline_sd,
                embed_name=embed_name,
                tier_cache=tier_cache,
                model_id=model_id,
                container_dir=CONTAINER_DIR,
            )
        )

    wall = time.perf_counter() - t_all
    write_report(rows, wall, str(args.model_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
