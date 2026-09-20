"""Ablate PBR-E vs exponent-spatial/hier/cross-layer vs uint16 spatial."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.safetensors_io import load_uint16, tensor_role
from pbr_core.tiles import as_2d
from pbr_encoder.baselines import baseline_sizes
from pbr_encoder.cli import format_table, run_one
from pbr_encoder.hf_weights import (
    PRIMARY_LICENSE,
    PRIMARY_REPO,
    inventory_from_dir,
    select_weight_specs,
)
from pbr_encoder.profiles import PROFILES
from pbr_encoder.verification import ExactnessError

DISCLAIMER = (
    "Blocker-fix ablation on the Stage 1B Qwen tensor set. "
    "Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _aggregate(rows: list[dict], *, label: str) -> dict:
    orig = sum(int(r["original_bytes"]) for r in rows)
    enc = sum(int(r["encoded_bytes"]) for r in rows)
    words = sum(int(r["n_words"]) for r in rows)
    zlib_vals = [r.get("zlib_bytes") for r in rows if r.get("zlib_bytes") is not None]
    modes: Counter[str] = Counter()
    for row in rows:
        for name, info in row.get("mode_usage", {}).items():
            modes[name] += int(info["tiles"])
    exact = all(r["exact"] == "PASS" for r in rows)
    return {
        "case": "TOTAL",
        "profile": label,
        "original_bytes": orig,
        "encoded_bytes": enc,
        "n_words": words,
        "bpw": bits_per_weight(enc, words),
        "ratio_vs_raw_bf16": compression_ratio(enc, orig),
        "exact": "PASS" if exact else "FAIL",
        "winning_modes": ",".join(f"{n}:{c}" for n, c in modes.most_common(6)),
        "zlib_bytes": sum(int(v) for v in zlib_vals) if zlib_vals else None,
        "tensor_count": len(rows),
    }


def run_profile(
    *,
    specs,
    profile_name: str,
    block_sizes: list[int],
    include_baselines: bool,
) -> tuple[list[dict], dict]:
    prof = PROFILES[profile_name]
    refs: dict[tuple[str, tuple[int, int]], object] = {}
    rows: list[dict] = []
    for spec in specs:
        words = as_2d(load_uint16(spec))
        role = tensor_role(spec.name)
        key = (role, (int(words.shape[0]), int(words.shape[1])))
        ref = refs.get(key) if prof["use_refs"] else None
        print(
            f"[{profile_name}] {spec.name}  {list(spec.shape)}  ref={'yes' if ref is not None else 'no'}",
            flush=True,
        )
        row = run_one(
            words,
            case=spec.name,
            block_sizes=block_sizes,
            include_baselines=include_baselines,
            extra={"stage": "blocker-ablation", "profile": profile_name},
            disclaimer=DISCLAIMER,
            codecs=prof["codecs"],
            whole_codecs=prof["whole_codecs"],
            enable_whole=bool(prof["enable_whole"]),
            ref_words=ref,
        )
        row["profile"] = profile_name
        row.pop("blob", None)
        rows.append(row)
        print(
            f"  -> enc_B={row['encoded_bytes']}  BPW={row['bpw']:.4f}  "
            f"exact={row['exact']}  modes={row['winning_modes']}",
            flush=True,
        )
        if prof["use_refs"]:
            refs[key] = words
    total = _aggregate(rows, label=profile_name)
    return rows, total


def format_ablation(totals: list[dict]) -> str:
    lines = [
        "# Blocker-fix ablation (same 42-tensor Qwen set)",
        "",
        DISCLAIMER,
        "",
        "| profile | orig B | enc B | BPW | ratio | exact | winning modes | zlib B |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for t in totals:
        zlib = t.get("zlib_bytes")
        lines.append(
            f"| {t['profile']} | {t['original_bytes']} | {t['encoded_bytes']} | "
            f"{t['bpw']:.4f} | {t['ratio_vs_raw_bf16']:.4f} | {t['exact']} | "
            f"{t['winning_modes']} | {zlib if zlib is not None else ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablate PBR blocker fixes on Qwen.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["pbre", "new_modes", "uint16_spatial"],
        choices=list(PROFILES),
    )
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    specs = select_weight_specs(
        inventory_from_dir(args.model_dir),
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    block_sizes = [int(x) for x in cfg.get("block_sizes", [256])]
    print(DISCLAIMER)
    print(f"selected {len(specs)} tensors")
    print()
    totals: list[dict] = []
    all_rows: dict[str, list[dict]] = {}
    try:
        for name in args.profiles:
            rows, total = run_profile(
                specs=specs,
                profile_name=name,
                block_sizes=block_sizes,
                include_baselines=name == "pbre",
            )
            all_rows[name] = rows
            totals.append(total)
            # zlib row from the PBR-E pass (same tensors, same bytes).
            if name == "pbre" and total.get("zlib_bytes"):
                zlib_b = int(total["zlib_bytes"])
                totals.append(
                    {
                        "case": "TOTAL",
                        "profile": "zlib_baseline",
                        "original_bytes": total["original_bytes"],
                        "encoded_bytes": zlib_b,
                        "n_words": total["n_words"],
                        "bpw": bits_per_weight(zlib_b, total["n_words"]),
                        "ratio_vs_raw_bf16": compression_ratio(zlib_b, total["original_bytes"]),
                        "exact": "n/a (not PBR)",
                        "winning_modes": "zlib (baseline, not PBR)",
                        "zlib_bytes": zlib_b,
                        "tensor_count": total["tensor_count"],
                    }
                )
    except ExactnessError as exc:
        print(f"ABLATION FAIL: {exc}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": DISCLAIMER,
        "model": {
            "repo_id": cfg.get("repo", PRIMARY_REPO),
            "revision": cfg.get("revision", "main"),
            "license": cfg.get("license", PRIMARY_LICENSE),
            "local_dir": str(args.model_dir),
            "selected": [s.name for s in specs],
        },
        "totals": totals,
        "profiles": {
            name: [{k: v for k, v in r.items() if k not in {"blob", "container_extra", "attempts"}} for r in rows]
            for name, rows in all_rows.items()
        },
    }
    (args.output_dir / "blocker_ablation_qwen.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = format_ablation(totals)
    (args.output_dir / "blocker_ablation_qwen.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(format_table(totals))
    failed = [t for t in totals if t["exact"] not in {"PASS", "n/a (not PBR)"}]
    if failed:
        print("ABLATION FAIL: exactness")
        return 1
    print("ABLATION PASS: every PBR profile reconstructed bit-exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
