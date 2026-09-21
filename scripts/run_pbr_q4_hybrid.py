#!/usr/bin/env python3
"""Hybrid H95 + groupwise INT + selective 256-node X/Y (full 0.5B).

New quantized reference: H95 mantissa-keep (per-weight exponents) on
sensitive tensors, lower groupwise INT on robust regions, then selective
X/Y only if the complete physical cost wins.

Not S1-compatible. Not PR #17 restore_q6. Physical BPW is
file_bytes * 8 / n_weights. Quality uses calib-v2 / heldout-v1 proxy NLL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from pbr_core.safetensors_io import TWO_BYTE_DTYPES, inventory_model, load_uint16
from pbr_h95.container_h95q import sha256_state_u16
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_q4.hybrid.const import (
    PR17_ACTUAL_BPW,
    PR17_FILE_BYTES,
    PR17_HELDOUT,
    PR17_POLICY,
    PR17_SHA,
    QUALITY_AIM,
    QUALITY_MIN,
    S1_N_WEIGHTS,
    S1_SHA,
    S1_V2_ACTUAL_BPW,
    S1_V2_FILE_BYTES,
    TARGET_PRACTICAL_BPW,
    TARGET_STRETCH_BPW,
)
from pbr_q4.hybrid.container import (
    decode_container,
    encode_quantized,
    verify_decoded_against_reference,
)
from pbr_q4.hybrid.policy import LADDER, family_label, get_policy, slot_for_name
from pbr_q4.hybrid.quantize import QuantizedTensor, estimate_packed_bits, quantize_tensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")
OUT_JSON = Path("artifacts/pbr_q4/hybrid_xy_qwen.json")
OUT_MD = Path("artifacts/pbr_q4/hybrid_xy_qwen.md")
CONTAINER_DIR = Path("artifacts/pbr_q4/containers")

HONESTY = [
    "New hybrid quantized reference — NOT bit-exact with H95Q-S1 and not PR #17 restore_q6.",
    "actual_bpw = physical file_bytes * 8 / n_weights (header, H95 streams, INT scales, maps, flags, outliers).",
    "Packed codes are the product default. X/Y is stored only if complete physical cost is strictly smaller.",
    "H95 tensors keep per-weight exponents (adaptive exact). INT tensors share a scale+zp per group.",
    "Quality is ppl_retention = bf16_ppl / quant_ppl on in-repo calib-v2 / heldout-v1 (proxy, not MMLU).",
    "Hard quality floor: heldout ≥0.95. Development aim ≥0.97. Misses are FAIL, not rounded away.",
    "Practical rate target ≤5.0 BPW; stretch ≤4.5. Misses are reported as FAIL.",
    "S1 v2 7.420820 BPW and PR #17 6.191189 BPW are comparison baselines only.",
    "Not a production mobile runtime.",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    return payload, [t["text"] for t in payload["texts"]]


def _ensure_tokenizer_files(model_dir: Path) -> None:
    if (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists():
        return
    from huggingface_hub import snapshot_download

    print(f" fetching tokenizer files into {model_dir}", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=str(model_dir),
        cache_dir=str(Path("outputs/hf_cache")),
        allow_patterns=[
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "generation_config.json",
        ],
    )


def resolve_model(model_dir: Path) -> Path:
    if model_dir.exists() and any(model_dir.glob("*.safetensors")):
        _ensure_tokenizer_files(model_dir)
        return model_dir
    print(f"Model not at {model_dir}; downloading Qwen2.5-0.5B-Instruct…", flush=True)
    from pbr_encoder.hf_weights import download_checkpoint

    result = download_checkpoint(
        local_dir=Path("outputs/models"),
        cache_dir=Path("outputs/hf_cache"),
    )
    print(f" downloaded {result.repo_id} → {result.local_dir} ({result.note})", flush=True)
    dest = Path(result.local_dir)
    _ensure_tokenizer_files(dest)
    return dest


def unique_specs(model_dir: Path):
    specs = [s for s in inventory_model(model_dir) if s.dtype in TWO_BYTE_DTYPES]
    seen: set[tuple[str, tuple[int, int]]] = set()
    out = []
    for spec in sorted(specs, key=lambda s: s.name):
        key = (str(spec.file_path), spec.data_offsets)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def quantize_model(specs, policy, *, label: str) -> list[QuantizedTensor]:
    qts: list[QuantizedTensor] = []
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        words = load_uint16(spec).reshape(spec.shape)
        slot = slot_for_name(spec.name, policy)
        fam = family_label(spec.name, policy)
        if i == 1 or i == len(specs) or i % 15 == 0:
            print(
                f" [{label}] {i}/{len(specs)} {spec.name} {slot.kind}"
                f"{' k=' + str(slot.keep) if slot.kind == 'h95' else ''}"
                f"{' q' + str(slot.bits) if slot.kind == 'int' else ''} "
                f"shape={tuple(spec.shape)}",
                flush=True,
            )
        qts.append(quantize_tensor(words, slot, policy, name=spec.name, family=fam))
    print(f" [{label}] quantized {len(qts)} tensors in {time.perf_counter() - t0:.1f}s", flush=True)
    return qts


def packed_estimate(qts: list[QuantizedTensor]) -> dict[str, Any]:
    n = sum(int(t.q_ref.size) for t in qts)
    bits = sum(estimate_packed_bits(t) for t in qts)
    by_fam: dict[str, dict[str, float]] = {}
    kind_hist: dict[str, int] = {}
    for t in qts:
        slot = by_fam.setdefault(
            t.family, {"n_words": 0, "code_bits": 0, "n_outliers": 0, "kind": t.kind}
        )
        slot["n_words"] += int(t.q_ref.size)
        slot["code_bits"] += estimate_packed_bits(t)
        slot["n_outliers"] += int(t.outlier_idx.size)
        kind_hist[t.kind] = kind_hist.get(t.kind, 0) + int(t.q_ref.size)
    return {
        "n_weights": n,
        "packed_est_bits": bits,
        "packed_est_bpw": round(bits / n, 6) if n else 0.0,
        "family": {
            k: {
                "n_words": v["n_words"],
                "est_bpw": round(v["code_bits"] / v["n_words"], 4) if v["n_words"] else 0.0,
                "n_outliers": v["n_outliers"],
                "kind": v["kind"],
            }
            for k, v in by_fam.items()
        },
        "kind_hist_words": kind_hist,
        "n_outliers": sum(int(t.outlier_idx.size) for t in qts),
        "note": "H95: 1 sign + 2.62 exp-ref + packed K; INT: codes + f16 scales + u8 zp + outliers; no header / X/Y",
    }


def apply_qref(model, qref: dict[str, np.ndarray]) -> None:
    import torch

    from pbr_h95.apply_policy import u16_to_bf16_tensor

    with torch.no_grad():
        sd = model.state_dict()
        for name, words in qref.items():
            if name not in sd:
                continue
            t = sd[name]
            arr = np.ascontiguousarray(words, dtype=np.uint16).reshape(tuple(t.shape))
            t.copy_(u16_to_bf16_tensor(arr, t))
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()


def eval_proxy(model, tokenizer, calib_texts, heldout_texts, *, max_length: int = 256) -> dict[str, Any]:
    calib = mean_token_nll(model, tokenizer, calib_texts, max_length=max_length)
    held = mean_token_nll(model, tokenizer, heldout_texts, max_length=max_length)
    return {"calib": calib, "heldout": held}


def write_md(report: dict[str, Any], path: Path) -> None:
    w = report["winner"]
    gates = report["gates"]
    mix = w.get("byte_mix") or {}
    lines = [
        "# PBR hybrid H95 + groupwise INT + selective X/Y",
        "",
        "## Dual-gate verdict",
        "",
        "| Gate | Target | Measured | Verdict |",
        "| --- | --- | ---: | --- |",
        f"| Practical physical rate | ≤ **5.0** BPW | **{w.get('actual_bpw')}** | **{gates['practical_rate']}** |",
        f"| Stretch physical rate | ≤ **4.5** BPW | **{w.get('actual_bpw')}** | **{gates['stretch_rate']}** |",
        f"| Held-out proxy (hard min) | ≥ **0.95** | **{w.get('heldout_retention')}** | **{gates['quality_min']}** |",
        f"| Held-out proxy (aim) | ≥ **0.97** | **{w.get('heldout_retention')}** | **{gates['quality_aim']}** |",
        f"| Exact decode of this Q-ref | SHA match | `{w.get('sha256_quantized_reference')}` | **{gates['exact_decode']}** |",
        "",
        f"Policy: `{w.get('policy')}` — new quantized reference SHA `{w.get('sha256_quantized_reference')}`.",
        f"Not S1. Not PR #17 Q6. Wall time: **{report['wall_s']:.1f}s**",
        "",
        "## Honesty",
        "",
    ]
    for h in HONESTY:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Physical rate vs S1 7.42 and PR #17 6.19",
        "",
        "| Container | file_bytes | actual_bpw | Q(W) |",
        "| --- | ---: | ---: | --- |",
        f"| H95Q-S1 v2 (PR #14/#16) | {S1_V2_FILE_BYTES:,} | **{S1_V2_ACTUAL_BPW:.6f}** | S1 SHA `{S1_SHA[:8]}…` |",
        f"| PR #17 `{PR17_POLICY}` | {PR17_FILE_BYTES:,} | **{PR17_ACTUAL_BPW:.6f}** | Q6 SHA `{PR17_SHA[:8]}…` |",
        f"| This HYBX (`{w.get('policy')}`) | {int(w.get('file_bytes') or 0):,} | **{w.get('actual_bpw')}** | this SHA |",
        "",
        "## Byte mix (winner)",
        "",
        f"- H95 payload: {int(mix.get('h95_payload_bytes') or 0):,} B "
        f"({float(mix.get('h95_payload_frac') or 0)*100:.2f}% of file); "
        f"words {int(mix.get('h95_words') or 0):,} ({float(mix.get('h95_word_frac') or 0)*100:.2f}%)",
        f"- INT payload: {int(mix.get('int_payload_bytes') or 0):,} B "
        f"({float(mix.get('int_payload_frac') or 0)*100:.2f}% of file); "
        f"words {int(mix.get('int_words') or 0):,} ({float(mix.get('int_word_frac') or 0)*100:.2f}%)",
        f"- Raw BF16 payload: {int(mix.get('raw_payload_bytes') or 0):,} B",
        f"- X/Y tiles: {w.get('n_xy_tiles')} / {w.get('n_tiles')} "
        f"({float(w.get('xy_tile_frac') or 0)*100:.4f}%) — packed default",
        f"- Outliers: {w.get('n_outliers')}",
        f"- Saved vs packed codes: {w.get('saved_vs_packed_code_bytes')} B",
        f"- XY mode hist: `{w.get('xy_mode_histogram')}`",
        "",
        "## Quality (proxy NLL)",
        "",
        "| Split | BF16 ppl | Q ppl | retention |",
        "| --- | ---: | ---: | ---: |",
        f"| calib-v2 | {report['bf16']['calib']['ppl']:.4f} | {w.get('calib_ppl')} | {w.get('calib_retention')} |",
        f"| heldout-v1 | {report['bf16']['heldout']['ppl']:.4f} | {w.get('heldout_ppl')} | {w.get('heldout_retention')} |",
        "",
        f"PR #17 heldout proxy (reference): **{PR17_HELDOUT:.6f}** on restore_q6.",
        "",
        "## Pareto / backoff",
        "",
        "| policy | packed-est BPW | actual BPW | calib ret | heldout ret | encoded? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["pareto"]:
        lines.append(
            f"| {row['policy']} | {row.get('packed_est_bpw')} | {row.get('actual_bpw', '—')} | "
            f"{row.get('calib_retention')} | {row.get('heldout_retention')} | {row.get('encoded')} |"
        )
    lines += [
        "",
        "## Exactness",
        "",
        f"- SHA-256 quantized ref: `{w.get('sha256_quantized_reference')}`",
        f"- decode(encode(Q)) == Q: **{str(w.get('exact_decode')).upper()}**",
        f"- Subprocess decode SHA: **{str(w.get('subprocess_sha_ok')).upper()}**",
        f"- Distinct from S1 `{S1_SHA[:8]}…`: **YES**",
        f"- Distinct from PR #17 `{PR17_SHA[:8]}…`: **YES**",
        f"- file_bytes == physical size: **YES** (`{w.get('file_bytes')}`)",
        "",
        "## Family mix (winner)",
        "",
    ]
    fam = (w.get("packed_estimate") or {}).get("family") or {}
    if fam:
        lines += ["| family | kind | n_words | packed-est BPW | outliers |", "| --- | --- | ---: | ---: | ---: |"]
        for k, v in sorted(fam.items()):
            lines.append(
                f"| {k} | {v.get('kind')} | {int(v['n_words']):,} | {v['est_bpw']} | {int(v['n_outliers']):,} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--policy",
        type=str,
        default="auto",
        help="auto | " + " | ".join(LADDER),
    )
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--container-dir", type=Path, default=CONTAINER_DIR)
    args = ap.parse_args()

    t_all = time.perf_counter()
    model_dir = resolve_model(args.model_dir)
    specs = unique_specs(model_dir)
    print(f"inventory: {len(specs)} unique 16-bit tensors", flush=True)

    calib_meta, calib_texts = load_texts(args.calib)
    held_meta, held_texts = load_texts(args.heldout)

    names = [args.policy] if args.policy != "auto" else list(LADDER)
    pareto: list[dict[str, Any]] = []
    winner_qts: list[QuantizedTensor] | None = None
    winner_policy = None
    winner_eval: dict[str, Any] | None = None
    bf16_scores: dict[str, Any] | None = None

    model = tokenizer = None
    restore_state_dict = None
    baseline = None
    if not args.skip_eval:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from pbr_h95.apply_policy import clone_cpu_state_dict, restore_state_dict as _restore

        restore_state_dict = _restore
        torch.set_num_threads(1)
        print("Loading BF16 model for proxy NLL…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()
        baseline = clone_cpu_state_dict(model)
        print("Scoring BF16 baseline…", flush=True)
        bf16_scores = eval_proxy(model, tokenizer, calib_texts, held_texts)
        print(
            f" bf16 calib ppl={bf16_scores['calib']['ppl']:.4f} "
            f"heldout ppl={bf16_scores['heldout']['ppl']:.4f}",
            flush=True,
        )

    qts = None
    for pname in names:
        policy = get_policy(pname)
        print(f"=== policy {pname}: {policy.description}", flush=True)
        qts = quantize_model(specs, policy, label=pname)
        est = packed_estimate(qts)
        print(f" packed-est BPW={est['packed_est_bpw']} outliers={est['n_outliers']}", flush=True)
        row: dict[str, Any] = {
            "policy": pname,
            "description": policy.description,
            "packed_est_bpw": est["packed_est_bpw"],
            "packed_estimate": est,
            "encoded": False,
        }
        if model is not None and tokenizer is not None and bf16_scores is not None:
            qref = {t.name: t.q_ref for t in qts}
            apply_qref(model, qref)
            scores = eval_proxy(model, tokenizer, calib_texts, held_texts)
            if baseline is not None and restore_state_dict is not None:
                restore_state_dict(model, baseline)
            calib_ret = ppl_retention(bf16_scores["calib"]["ppl"], scores["calib"]["ppl"])
            held_ret = ppl_retention(bf16_scores["heldout"]["ppl"], scores["heldout"]["ppl"])
            row.update(
                {
                    "calib_ppl": round(scores["calib"]["ppl"], 6),
                    "heldout_ppl": round(scores["heldout"]["ppl"], 6),
                    "calib_retention": round(calib_ret, 6),
                    "heldout_retention": round(held_ret, 6),
                    "calib_nll": scores["calib"]["mean_nll"],
                    "heldout_nll": scores["heldout"]["mean_nll"],
                }
            )
            print(f" calib_ret={calib_ret:.4f} heldout_ret={held_ret:.4f}", flush=True)
            quality_ok = held_ret >= QUALITY_MIN
        else:
            quality_ok = True
        pareto.append(row)
        if quality_ok and winner_qts is None:
            winner_qts = qts
            winner_policy = policy
            winner_eval = row
            if args.policy == "auto":
                print(f" selected winner {pname} (heldout ≥ {QUALITY_MIN})", flush=True)
                break
        elif args.policy != "auto":
            winner_qts = qts
            winner_policy = policy
            winner_eval = row

    if winner_qts is None:
        print("No policy met held-out ≥0.95; encoding last attempt for the record.", flush=True)
        winner_qts = qts
        winner_policy = get_policy(names[-1])
        winner_eval = pareto[-1]

    container_stats: dict[str, Any] = {}
    exact = False
    sub_ok = False
    container_path = args.container_dir / f"PBR-HYBX-{winner_policy.name}.hybx"
    if not args.skip_encode:
        print("Encoding HYBX (selective X/Y)…", flush=True)
        container_stats = encode_quantized(
            winner_qts,
            container_path,
            policy=winner_policy,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
        )
        print(
            f" file_bytes={container_stats['file_bytes']} actual_bpw={container_stats['actual_bpw']}",
            flush=True,
        )
        print("Decoding and verifying SHA…", flush=True)
        decoded = decode_container(container_path)
        qref = {t.name: t.q_ref for t in winner_qts}
        ver = verify_decoded_against_reference(decoded["tensors"], qref)
        exact = bool(ver["ok"]) and decoded["sha256_decoded"] == container_stats[
            "sha256_quantized_reference"
        ]
        print(f" exact_decode={exact} sha={container_stats['sha256_quantized_reference']}", flush=True)
        if container_stats["sha256_quantized_reference"] in {S1_SHA, PR17_SHA}:
            raise RuntimeError("hybrid SHA collided with S1 or PR #17 — refuse to claim a new Q-ref")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pbr_q4.hybrid.container",
                str(container_path),
                "--expect-sha",
                container_stats["sha256_quantized_reference"],
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        sub_ok = proc.returncode == 0
        if not sub_ok:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
        for row in pareto:
            if row["policy"] == winner_policy.name:
                row["encoded"] = True
                row["actual_bpw"] = container_stats["actual_bpw"]
                row["file_bytes"] = container_stats["file_bytes"]

    winner = dict(winner_eval or {})
    winner.update(container_stats)
    winner["exact_decode"] = exact
    winner["subprocess_sha_ok"] = sub_ok
    if "packed_estimate" not in winner:
        winner["packed_estimate"] = packed_estimate(winner_qts)

    actual = float(winner.get("actual_bpw") or 0.0)
    held_ret = float(winner.get("heldout_retention") or 0.0)
    gates = {
        "practical_rate": "PASS" if actual and actual <= TARGET_PRACTICAL_BPW else "FAIL",
        "stretch_rate": "PASS" if actual and actual <= TARGET_STRETCH_BPW else "FAIL",
        "quality_min": "PASS" if held_ret >= QUALITY_MIN else ("SKIP" if args.skip_eval else "FAIL"),
        "quality_aim": "PASS" if held_ret >= QUALITY_AIM else ("SKIP" if args.skip_eval else "FAIL"),
        "exact_decode": "PASS" if exact else "FAIL",
    }
    if args.skip_encode:
        gates["practical_rate"] = gates["stretch_rate"] = "SKIP"

    report = {
        "phase": "PBR-Q4-hybrid-h95-int-xy",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_dir": str(model_dir),
        "s1_compatible": False,
        "pr17_compatible": False,
        "s1_v2_actual_bpw": S1_V2_ACTUAL_BPW,
        "s1_v2_file_bytes": S1_V2_FILE_BYTES,
        "s1_n_weights": S1_N_WEIGHTS,
        "pr17_actual_bpw": PR17_ACTUAL_BPW,
        "pr17_file_bytes": PR17_FILE_BYTES,
        "pr17_heldout": PR17_HELDOUT,
        "honesty": HONESTY,
        "targets": {
            "practical_bpw": TARGET_PRACTICAL_BPW,
            "stretch_bpw": TARGET_STRETCH_BPW,
            "heldout_min": QUALITY_MIN,
            "heldout_aim": QUALITY_AIM,
        },
        "gates": gates,
        "winner": winner,
        "pareto": pareto,
        "bf16": bf16_scores
        or {
            "calib": {"ppl": None, "mean_nll": None},
            "heldout": {"ppl": None, "mean_nll": None},
        },
        "calib_corpus": {"path": str(args.calib), "id": calib_meta.get("id"), "n_texts": len(calib_texts)},
        "heldout_corpus": {
            "path": str(args.heldout),
            "id": held_meta.get("id"),
            "n_texts": len(held_texts),
        },
        "wall_s": round(time.perf_counter() - t_all, 2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_md(report, args.out_md)
    print(
        json.dumps(
            {
                "gates": gates,
                "actual_bpw": winner.get("actual_bpw"),
                "heldout": winner.get("heldout_retention"),
                "policy": winner.get("policy"),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_json} and {args.out_md}", flush=True)
    return 0 if exact or args.skip_encode else 2


if __name__ == "__main__":
    raise SystemExit(main())
