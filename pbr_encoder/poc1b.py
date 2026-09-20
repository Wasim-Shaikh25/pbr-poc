"""Stage 1B: bit-exact validation on real BF16/F16 checkpoint tensors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d
from pbr_encoder.cli import format_table, run_one, write_reports
from pbr_encoder.hf_weights import (
    FALLBACK_REPO,
    PRIMARY_LICENSE,
    PRIMARY_REPO,
    download_checkpoint,
    inventory_from_dir,
    select_weight_specs,
)
from pbr_encoder.verification import ExactnessError

DISCLAIMER = (
    "Stage 1B / Gate 1B: real checkpoint tensors, method validation only. "
    "These BPW numbers are not a full-model production compression claim "
    "and must not be scaled into a 1–2 GB / 8 GB story."
)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return loaded


def _aggregate(rows: list[dict], *, model_meta: dict) -> dict:
    orig = sum(int(r["original_bytes"]) for r in rows)
    enc = sum(int(r["encoded_bytes"]) for r in rows)
    words = sum(int(r["n_words"]) for r in rows)
    zlib_vals = [r.get("zlib_bytes") for r in rows if r.get("zlib_bytes") is not None]
    modes: Counter[str] = Counter()
    for row in rows:
        for name, info in row.get("mode_usage", {}).items():
            modes[name] += int(info["tiles"])
    exact = all(r["exact"] == "PASS" for r in rows)
    enc_s = sum(float(r.get("encode_seconds") or 0.0) for r in rows)
    dec_s = sum(float(r.get("decode_seconds") or 0.0) for r in rows)
    orig_mb = orig / 1e6
    return {
        "case": "TOTAL",
        "original_bytes": orig,
        "encoded_bytes": enc,
        "n_words": words,
        "bpw": bits_per_weight(enc, words),
        "ratio_vs_raw_bf16": compression_ratio(enc, orig),
        "exact": "PASS" if exact else "FAIL",
        "winning_modes": ",".join(f"{n}:{c}" for n, c in modes.most_common(4)),
        "zlib_bytes": sum(int(v) for v in zlib_vals) if zlib_vals else None,
        "exp_huffman_bound_bytes": sum(
            int(r["exp_huffman_bound_bytes"])
            for r in rows
            if r.get("exp_huffman_bound_bytes") is not None
        )
        or None,
        "encode_seconds": enc_s,
        "decode_seconds": dec_s,
        "encode_MB_s": (orig_mb / enc_s) if enc_s > 0 else None,
        "decode_MB_s": (orig_mb / dec_s) if dec_s > 0 else None,
        "disclaimer": DISCLAIMER,
        "model": model_meta,
        "tensor_count": len(rows),
    }


def run_poc1b(
    *,
    model_dir: Path,
    specs,
    block_sizes: list[int],
    output_dir: Path,
    include_baselines: bool,
    model_meta: dict,
    codecs=None,
    whole_codecs=None,
    enable_whole: bool = True,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in specs:
        words = as_2d(load_uint16(spec))
        print(
            f"encoding {spec.name}  shape={list(spec.shape)}  "
            f"dtype={spec.dtype}  bytes={spec.nbytes}",
            flush=True,
        )
        row = run_one(
            words,
            case=spec.name,
            block_sizes=block_sizes,
            include_baselines=include_baselines,
            extra={"stage": "1B", "source": "safetensors_uint16"},
            disclaimer=DISCLAIMER,
            codecs=codecs,
            whole_codecs=whole_codecs,
            enable_whole=enable_whole,
        )
        row["dtype"] = spec.dtype
        row["source_file"] = str(spec.file_path)
        row["disclaimer"] = DISCLAIMER
        write_reports(row, output_dir)
        rows.append(row)
        print(
            f"  -> enc_B={row['encoded_bytes']}  BPW={row['bpw']:.4f}  "
            f"exact={row['exact']}  modes={row['winning_modes']}",
            flush=True,
        )

    total = _aggregate(rows, model_meta=model_meta)
    payload = {
        "disclaimer": DISCLAIMER,
        "gate": "Gate 1B — bit-exact reconstruction on real 16-bit weights",
        "model": model_meta,
        "tensors": [{k: v for k, v in r.items() if k != "attempts"} for r in rows],
        "total": {k: v for k, v in total.items() if k != "attempts"},
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = format_table(rows + [total])
    (output_dir / "console_report.txt").write_text(DISCLAIMER + "\n\n" + table + "\n", encoding="utf-8")
    return rows + [total]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PBR Stage 1B real-tensor validation (not a production compression claim)."
    )
    parser.add_argument("--repo", default=None, help=f"HF repo id (default {PRIMARY_REPO})")
    parser.add_argument("--revision", default=None, help="HF revision or commit SHA")
    parser.add_argument("--model-dir", type=Path, default=None, help="Existing local checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--min-bytes", type=int, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument(
        "--all-16bit",
        action="store_true",
        help="Encode every 16-bit tensor (full-checkpoint, not the Stage 1B window).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Codec profile from pbr_encoder.profiles (e.g. pbre). default=STAGE1A menu.",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config if args.config.exists() else None)
    repo = args.repo or cfg.get("repo", PRIMARY_REPO)
    revision = args.revision or cfg.get("revision", "main")
    cache_dir = args.cache_dir or Path(cfg.get("cache_dir", "outputs/hf_cache"))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "outputs/reports/poc1b"))
    local_root = Path(cfg.get("local_dir", "outputs/models"))
    block_sizes = args.block_sizes or [int(x) for x in cfg.get("block_sizes", [256])]
    min_bytes = args.min_bytes if args.min_bytes is not None else int(cfg.get("min_bytes", 100 * 1024 * 1024))
    max_bytes = args.max_bytes if args.max_bytes is not None else int(cfg.get("max_bytes", 200 * 1024 * 1024))
    include_embeddings = args.include_embeddings or bool(cfg.get("include_embeddings", False))
    include_baselines = not args.no_baselines

    print(DISCLAIMER)
    print()

    download_meta: dict | None = None
    if args.model_dir is not None:
        model_dir = args.model_dir
        model_meta = {
            "repo_id": repo,
            "revision": revision,
            "local_dir": str(model_dir),
            "license": cfg.get("license", PRIMARY_LICENSE),
            "fallback_used": False,
            "note": "local --model-dir",
        }
    else:
        try:
            downloaded = download_checkpoint(
                repo_id=repo,
                revision=revision,
                local_dir=local_root,
                cache_dir=cache_dir,
                allow_fallback=not args.no_fallback,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"GATE 1B FAIL: download error: {exc}")
            print(f"Tried primary {PRIMARY_REPO}; fallback is {FALLBACK_REPO}.")
            return 2
        model_dir = downloaded.local_dir
        download_meta = {
            "repo_id": downloaded.repo_id,
            "revision": downloaded.revision,
            "commit": downloaded.commit,
            "license": downloaded.license,
            "local_dir": str(downloaded.local_dir),
            "fallback_used": downloaded.fallback_used,
            "note": downloaded.note,
        }
        model_meta = download_meta
        print(
            f"checkpoint {downloaded.repo_id}  commit={downloaded.commit}  "
            f"license={downloaded.license}  fallback={downloaded.fallback_used}"
        )

    specs_all = inventory_from_dir(model_dir)
    n16 = sum(1 for s in specs_all if s.dtype in {"BF16", "F16", "FP16"})
    print(f"inventory: {len(specs_all)} tensors, {n16} 16-bit tensors")
    try:
        if args.all_16bit:
            chosen = [s for s in specs_all if s.dtype in {"BF16", "F16", "FP16"}]
            if not chosen:
                raise ValueError("No 16-bit tensors found in the checkpoint")
        else:
            chosen = select_weight_specs(
                specs_all,
                min_bytes=min_bytes,
                max_bytes=max_bytes,
                include_embeddings=include_embeddings,
            )
    except ValueError as exc:
        print(f"GATE 1B FAIL: {exc}")
        return 2

    codecs = None
    whole_codecs = None
    enable_whole = True
    if args.profile and args.profile != "default":
        from pbr_encoder.profiles import PROFILES

        if args.profile not in PROFILES:
            print(f"GATE 1B FAIL: unknown profile {args.profile}")
            return 2
        prof = PROFILES[args.profile]
        codecs = prof["codecs"]
        whole_codecs = prof["whole_codecs"]
        enable_whole = bool(prof["enable_whole"])
        print(f"profile {args.profile}: {prof['label']}")

    selected_bytes = sum(s.nbytes for s in chosen)
    print(
        f"selected {len(chosen)} tensors, {selected_bytes} bytes "
        f"({selected_bytes / (1024 * 1024):.1f} MiB)"
    )
    for spec in chosen:
        print(f"  - {spec.name}  {list(spec.shape)}  {spec.dtype}  {spec.nbytes}")
    print()

    model_meta["selected_tensors"] = [s.name for s in chosen]
    model_meta["selected_bytes"] = selected_bytes
    if download_meta is None:
        model_meta.setdefault("attribution", "See the model card for the local checkpoint.")

    try:
        rows = run_poc1b(
            model_dir=model_dir,
            specs=chosen,
            block_sizes=block_sizes,
            output_dir=output_dir,
            include_baselines=include_baselines,
            model_meta=model_meta,
            codecs=codecs,
            whole_codecs=whole_codecs,
            enable_whole=enable_whole,
        )
    except ExactnessError as exc:
        print(f"GATE 1B FAIL: {exc}")
        return 1

    print()
    print(format_table(rows))
    print()
    failed = [r["case"] for r in rows if r["case"] != "TOTAL" and r["exact"] != "PASS"]
    if failed:
        print(f"GATE 1B FAIL: exactness failed for {', '.join(failed)}")
        return 1
    print("GATE 1B PASS: selected real tensors reconstructed bit-exactly.")
    print(f"Reports: {output_dir}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
