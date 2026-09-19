"""CLI for the Stage 2 qualification scanner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pbr_core.safetensors_io import inventory_model
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO, download_checkpoint
from pbr_qualifier.reporting import format_decision, format_scan_table
from pbr_qualifier.scanner import DISCLAIMER, scan_model


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PBR Stage 2 qualification scanner (projection, not a full-model encode)."
    )
    parser.add_argument("--model", dest="repo", default=None, help="HF repo id")
    parser.add_argument("--repo", dest="repo_alt", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/qualifier_default.yaml"))
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--sample-tiles", type=int, default=None)
    parser.add_argument("--max-full-words", type=int, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--leaderboard", type=int, default=15)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config if args.config.exists() else None)
    repo = args.repo or args.repo_alt or cfg.get("repo", PRIMARY_REPO)
    revision = args.revision or cfg.get("revision", "main")
    cache_dir = args.cache_dir or Path(cfg.get("cache_dir", "outputs/hf_cache"))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "outputs/reports/poc2"))
    local_root = Path(cfg.get("local_dir", "outputs/models"))
    block_size = args.block_size or int(cfg.get("block_size", 256))
    sample_tiles = args.sample_tiles or int(cfg.get("sample_tiles", 8))
    max_full_words = args.max_full_words or int(cfg.get("max_full_words", 16384))
    json_out = args.json_out or Path(cfg.get("json_out", output_dir / "stage2_summary.json"))

    print(DISCLAIMER)
    print()

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
            print(f"STAGE 2 FAIL: download error: {exc}")
            return 2
        model_dir = downloaded.local_dir
        model_meta = {
            "repo_id": downloaded.repo_id,
            "revision": downloaded.revision,
            "commit": downloaded.commit,
            "license": downloaded.license,
            "local_dir": str(downloaded.local_dir),
            "fallback_used": downloaded.fallback_used,
            "note": downloaded.note,
        }
        print(
            f"checkpoint {downloaded.repo_id}  commit={downloaded.commit}  "
            f"license={downloaded.license}  fallback={downloaded.fallback_used}"
        )

    specs = inventory_model(model_dir)
    print(f"inventory: {len(specs)} tensors")
    report = scan_model(
        specs,
        block_size=block_size,
        sample_tiles=sample_tiles,
        max_full_words=max_full_words,
    )
    report["model"] = model_meta
    report["scan_config"] = {
        "block_size": block_size,
        "sample_tiles": sample_tiles,
        "max_full_words": max_full_words,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if json_out.resolve() != (output_dir / "stage2_summary.json").resolve():
        (output_dir / "stage2_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    tensors = report["tensors"]
    leaderboard = report["leaderboard"][: args.leaderboard]
    table = format_scan_table(tensors)
    decision = format_decision(report["projection"])
    board = format_scan_table(leaderboard)
    text = (
        DISCLAIMER
        + "\n\n"
        + decision
        + "\n\nPer-tensor leaderboard (lowest projected BPW first):\n"
        + board
        + "\n\nAll scanned 16-bit tensors:\n"
        + table
        + "\n"
    )
    (output_dir / "console_report.txt").write_text(text, encoding="utf-8")

    print()
    print(decision)
    print()
    print("Leaderboard (lowest projected BPW first):")
    print(board)
    print()
    failed = [t["name"] for t in tensors if t.get("exact") == "FAIL"]
    if failed:
        print(f"STAGE 2 FAIL: exactness failed for {', '.join(failed)}")
        return 1
    print(f"JSON: {json_out}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
