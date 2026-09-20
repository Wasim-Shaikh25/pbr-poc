#!/usr/bin/env python3
"""Re-encode frozen H95Q-S1 quantized words with *selective* 256-node X/Y.

Does **not** introduce a new BF16→Q4 policy. Source is the S1 quantized
reference (SHA eda64747…). Quality is unchanged if that SHA matches; this
script does not re-run held-out eval.

Packed-K (S1 stream packing of true, unpadded nodes) is the product default.
A matrix-family tile is stored only when its complete physical cost is
strictly smaller than packed-K.

Prefer a frozen H95Q-S1 v2/v1 container. Fall back to rebuilding Q(W) from
the Qwen checkpoint with the S1 stack policy, then refuse if SHA drifts.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pbr_q4.const import (
    FROZEN_S1_SHA,
    PR15_ALL_TILE_BPW,
    PR15_ALL_TILE_FILE_BYTES,
    PRIOR_HELDOUT_PROXY,
    S1_N_WEIGHTS,
    S1_V2_ACTUAL_BPW,
    S1_V2_FILE_BYTES,
)
from pbr_q4.container import decode_container, encode_container, verify_decoded_against_reference
from pbr_q4.s1_source import find_s1_container, load_s1_from_container

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
OUT_JSON = Path("artifacts/pbr_q4/s1_xy_selective_qwen.json")
OUT_MD = Path("artifacts/pbr_q4/s1_xy_selective_qwen.md")
CONTAINER_DIR = Path("artifacts/pbr_q4/containers")

HONESTY = [
    "This is S1 + *selective* X/Y post-codec, not a new BF16→groupwise-Q4 line.",
    "Numerical values are the frozen H95Q-S1 quantized reference; decode must match that SHA.",
    "actual_bpw = physical file_bytes * 8 / n_weights (header, maps, mode IDs, checksums included).",
    "Packed-K of true (unpadded) nodes is the product default. X/Y is emitted only if strictly smaller including map + per-XY flag bits.",
    "Arrays are not padded to 16×16; ragged last tiles store their true shape.",
    "Always-on all-tile X/Y (PR #15, 7.453890 BPW) is not the product path.",
    "Quality: if SHA matches S1, Q(W) is identical; prior heldout proxy 0.990 is inherited, not re-evaluated.",
    "Not a production mobile runtime.",
    "Compare physical BPW to H95Q-S1 container v2 = 7.420820 (PR #14) and PR #15 all-tile = 7.453890.",
]


def _load_from_model(model_dir: Path, calib: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pbr_h95.apply_policy import bf16_tensor_to_u16, clone_cpu_state_dict
    from pbr_h95.container_h95q import sha256_state_u16
    from pbr_h95.embed_tiers import find_embed_param
    from pbr_h95.h95q_stack import apply_mlp_k3_policy, fit_embed_tiers_on_calib

    def _unique_float_state(sd):
        out = {}
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

    torch.set_num_threads(1)
    print("Loading model to rebuild frozen S1 Q(W)…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    baseline_sd = clone_cpu_state_dict(model)
    embed_name, emb_w = find_embed_param(model)
    calib_payload = json.loads(Path(calib).read_text())
    calib_texts = [t["text"] for t in calib_payload["texts"]]
    row_keeps, tier_meta = fit_embed_tiers_on_calib(
        tokenizer, calib_texts, vocab_rows=int(emb_w.shape[0]), schedule="aggressive", max_length=256
    )
    meta = apply_mlp_k3_policy(
        model,
        baseline_sd=baseline_sd,
        variant="band_8_15",
        embed_row_keeps=row_keeps,
        embed_name=embed_name,
    )
    u16_map = {n: bf16_tensor_to_u16(t) for n, t in _unique_float_state(model.state_dict()).items()}
    ref_sha = sha256_state_u16(u16_map)
    if ref_sha != FROZEN_S1_SHA:
        raise RuntimeError(f"rebuilt S1 SHA {ref_sha} != frozen {FROZEN_S1_SHA}")
    keep_map = {k: int(v) for k, v in meta["keep_map"].items()}
    precision = {
        "body": "B1",
        "embed_schedule": "aggressive",
        "mlp_k3_bands": ["mlp_band_8_15"],
        "k3_variant": meta.get("k3_variant"),
        "tier_schedule": tier_meta.get("schedule"),
        "keep_hist_rows": tier_meta.get("keep_hist_rows"),
    }
    return {
        "tensors": u16_map,
        "keep_map": keep_map,
        "embed_name": embed_name,
        "embed_row_keeps": row_keeps,
        "precision_map": precision,
        "model_id": str(getattr(getattr(model, "config", None), "_name_or_path", None) or model_dir),
        "sha256_quantized_reference": ref_sha,
        "source_path": str(model_dir),
        "source": "rebuilt_from_model",
    }


def decode_in_subprocess(container_path: Path, expect_sha: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pbr_q4.container",
        str(container_path),
        "--expect-sha",
        expect_sha,
    ]
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(Path.cwd()))
    dec_sha = ""
    for line in proc.stdout.splitlines():
        if line.startswith("sha256_decoded="):
            dec_sha = line.split("=", 1)[1].strip()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "sha256_decoded": dec_sha,
        "sha_ok_subprocess": proc.returncode == 0 and "sha_match=True" in proc.stdout,
    }


def gate_verdict(actual_bpw: float) -> str:
    return "PASS" if float(actual_bpw) <= S1_V2_ACTUAL_BPW else "FAIL"


def write_report(row: dict[str, Any], wall: float) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    actual = float(row["actual_bpw"])
    verdict = gate_verdict(actual)
    payload = {
        "phase": "H95Q-S1-XY-selective",
        "format": "H95X",
        "container_version": 2,
        "honesty": HONESTY,
        "frozen_s1_sha": FROZEN_S1_SHA,
        "s1_v2_actual_bpw": S1_V2_ACTUAL_BPW,
        "s1_v2_file_bytes": S1_V2_FILE_BYTES,
        "pr15_all_tile_actual_bpw": PR15_ALL_TILE_BPW,
        "pr15_all_tile_file_bytes": PR15_ALL_TILE_FILE_BYTES,
        "s1_n_weights": S1_N_WEIGHTS,
        "gate": "actual_bpw <= 7.420820",
        "verdict": verdict,
        "wall_seconds": round(wall, 2),
        "result": row,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    delta_s1 = actual - S1_V2_ACTUAL_BPW
    delta_pr15 = actual - PR15_ALL_TILE_BPW
    lines = [
        "# PBR H95Q-S1 + selective 256-node X/Y",
        "",
        f"**Gate vs S1 v2 7.420820: {verdict}**",
        "",
        "Post-codec of the **frozen H95Q-S1 quantized reference** "
        f"(SHA `{FROZEN_S1_SHA}`). Not a new BF16→Q4 policy. "
        "Packed-K is the product default; X/Y only when strictly smaller.",
        f"Wall time: **{wall:.1f}s**",
        "",
        "## Honesty",
        "",
    ]
    for h in HONESTY:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Physical rate",
        "",
        "| Container | file_bytes | actual_bpw | notes |",
        "| --- | ---: | ---: | --- |",
        f"| H95Q-S1 v2 (PR #14) | {S1_V2_FILE_BYTES:,} | **{S1_V2_ACTUAL_BPW:.6f}** | packed-K mantissas + adaptive exp |",
        f"| H95Q-S1 + all-tile X/Y (PR #15) | {PR15_ALL_TILE_FILE_BYTES:,} | **{PR15_ALL_TILE_BPW:.6f}** | always-on matrix family |",
        f"| H95Q-S1 + selective X/Y (this) | {row['file_bytes']:,} | **{actual:.6f}** | packed default, XY iff smaller |",
        f"| Δ (this − S1 v2) | {row['file_bytes'] - S1_V2_FILE_BYTES:+,} | **{delta_s1:+.6f}** | negative = smaller; ≤0 is PASS |",
        f"| Δ (this − PR #15) | {row['file_bytes'] - PR15_ALL_TILE_FILE_BYTES:+,} | **{delta_pr15:+.6f}** | vs always-on |",
        "",
        f"**Verdict: {verdict}** — `actual_bpw` {('≤' if verdict == 'PASS' else '>')} 7.420820.",
        "",
        "## Selective mantissa vs packed-K (true nodes, no 16×16 pad)",
        "",
        f"- Packed-K whole-tensor mantissa (S1 stream, true nodes): {row['packed_whole_bytes']:,} B "
        f"({row['packed_whole_bpw']:.6f} BPW)",
        f"- Stored mantissa blobs: {row['mant_blob_bytes']:,} B "
        f"({row['mant_blob_bpw']:.6f} BPW)",
        f"- Saved vs packed mantissa: {row['saved_vs_packed_mant_bytes']:+,} B",
        f"- XY tiles / packed tiles / all tiles: {row['n_xy_tiles']:,} / {row['n_packed_tiles']:,} / {row['n_tiles']:,}",
        f"- Tensors with at least one XY blob: {row['n_xy_sel_tensors']:,}",
        f"- Mode-map bytes: {row['xy_map_bytes']:,}",
        f"- XY flag bytes (XY tiles only): {row['xy_flag_bytes']:,}",
        f"- XY payloads: {row['xy_payload_bytes']:,}",
        f"- Product default packed: {row['product_default_packed']}",
        f"- Always-on all-tile matrix family: {row['all_tiles_matrix_family']}",
        "",
        "## Mode mix (XY tiles only; packed tiles have no predictor flags)",
        "",
        f"- Tile modes: `{row['xy_mode_histogram']}`",
        f"- Predictors: `{row['xy_pred_histogram']}`",
        f"- Traversals: `{row['xy_trav_histogram']}`",
        f"- Exp modes: complete BPW {row['exp_complete_bpw']:.6f}",
        "",
        "## Exactness",
        "",
        f"- Source: `{row['source']}` (`{row['source_path']}`)",
        f"- SHA-256 quantized ref: `{row['sha256_quantized_reference']}`",
        f"- Matches frozen S1: **{'YES' if row['sha256_quantized_reference'] == FROZEN_S1_SHA else 'NO'}**",
        f"- decode(encode(Q_S1)) == Q_S1: **{'YES' if row['exact_match'] else 'NO'}**",
        f"- Subprocess decode SHA: **{'YES' if row['subprocess_decode']['sha_ok'] else 'NO'}**",
        "",
        "## Quality",
        "",
        f"Not re-evaluated. S1 heldout proxy retention **{PRIOR_HELDOUT_PROXY:.3f}** still applies because Q(W) is bit-identical.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD} verdict={verdict}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-container", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--out-container", type=Path, default=CONTAINER_DIR / "H95Q-S1-XY-sel.h95x")
    args = ap.parse_args()

    t0 = time.perf_counter()
    src_path = find_s1_container(args.s1_container)
    if src_path is not None:
        print(f"Loading frozen S1 from {src_path}", flush=True)
        s1 = load_s1_from_container(src_path, expect_sha=FROZEN_S1_SHA)
    elif args.model_dir.is_dir():
        s1 = _load_from_model(args.model_dir, args.calib)
    else:
        raise SystemExit(
            "No H95Q-S1 container and no model dir. "
            "Pass --s1-container or --model-dir."
        )

    print(
        f"S1 source={s1['source']} sha={s1['sha256_quantized_reference'][:12]}… "
        f"n_tensors={len(s1['tensors'])}",
        flush=True,
    )
    t_enc = time.perf_counter()
    enc = encode_container(
        args.out_container,
        s1["tensors"],
        keep_map=s1["keep_map"],
        model_id=s1["model_id"],
        candidate="H95Q-S1+XY-sel",
        precision_map=s1["precision_map"],
        embed_name=s1["embed_name"],
        embed_row_keeps=s1["embed_row_keeps"],
        expect_ref_sha=FROZEN_S1_SHA,
    )
    wall_enc = time.perf_counter() - t_enc
    verdict = gate_verdict(float(enc["actual_bpw"]))
    print(
        f"encoded file_bytes={enc['file_bytes']} actual_bpw={enc['actual_bpw']:.6f} "
        f"vs S1v2 {S1_V2_ACTUAL_BPW:.6f} verdict={verdict} "
        f"tiles={enc['n_tiles']} xy={enc['n_xy_tiles']} "
        f"modes={enc['xy_mode_histogram']} wall_enc={wall_enc:.1f}s",
        flush=True,
    )

    t_dec = time.perf_counter()
    decoded = decode_container(args.out_container)
    verify = verify_decoded_against_reference(decoded["tensors"], s1["tensors"])
    wall_dec = time.perf_counter() - t_dec
    if not verify["exact_match"] or not verify["sha_ok"]:
        raise RuntimeError(f"exactness FAILED: {verify}")
    if enc["sha256_quantized_reference"] != FROZEN_S1_SHA:
        raise RuntimeError("encoded SHA != frozen S1")
    sub = decode_in_subprocess(args.out_container, FROZEN_S1_SHA)
    if not sub["sha_ok_subprocess"]:
        raise RuntimeError(f"subprocess decode failed: {sub}")

    row = {
        **enc,
        "exact_match": True,
        "sha_ok": True,
        "source": s1["source"],
        "source_path": s1["source_path"],
        "s1_v2_actual_bpw": S1_V2_ACTUAL_BPW,
        "delta_vs_s1_v2_bpw": round(float(enc["actual_bpw"]) - S1_V2_ACTUAL_BPW, 6),
        "delta_vs_pr15_bpw": round(float(enc["actual_bpw"]) - PR15_ALL_TILE_BPW, 6),
        "gate_le_s1_v2": verdict,
        "prior_heldout_retention": PRIOR_HELDOUT_PROXY,
        "quality_reevaluated": False,
        "subprocess_decode": {
            "returncode": sub["returncode"],
            "sha_ok": True,
            "sha256_decoded": sub.get("sha256_decoded") or FROZEN_S1_SHA,
        },
        "wall_encode_s": round(wall_enc, 2),
        "wall_decode_verify_s": round(wall_dec, 2),
        "note": (
            "Physical file measured after encode. Decode word SHA independently "
            "verified against frozen S1. Packed-K baseline is the true-node S1 "
            "stream (no 16×16 pad). X/Y tiles pay map+flag bits and are kept "
            "only when the tensor blob is strictly smaller than packed-K."
        ),
    }
    wall = time.perf_counter() - t0
    write_report(row, wall)
    del s1, decoded
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
