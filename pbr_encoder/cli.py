"""Stage 1A CLI: encode controlled tensors and print a complete-cost table."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from pbr_core.hashing import sha256_bytes, sha256_words
from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.tiles import choose_tile_hw
from pbr_encoder.baselines import baseline_sizes
from pbr_encoder.controlled_data import CASES, generate_all, generate_case, load_case
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor, mode_usage
from pbr_encoder.verification import ExactnessError, assert_exact

DISCLAIMER = (
    "Stage 1A / controlled tensors only. These numbers are method-validation "
    "results, not real-model compression evidence."
)


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return loaded


def format_table(rows: list[dict]) -> str:
    headers = [
        "case",
        "orig_B",
        "enc_B",
        "BPW",
        "ratio",
        "exact",
        "winning_modes",
        "zlib_B",
    ]
    str_rows = []
    for row in rows:
        str_rows.append(
            [
                str(row["case"]),
                str(row["original_bytes"]),
                str(row["encoded_bytes"]),
                f"{row['bpw']:.4f}",
                f"{row['ratio_vs_raw_bf16']:.4f}",
                row["exact"],
                row["winning_modes"],
                str(row.get("zlib_bytes", "")),
            ]
        )
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in str_rows)
    return "\n".join(lines)


def _winning_modes(usage: dict) -> str:
    ranked = sorted(usage.items(), key=lambda kv: -kv[1]["tiles"])
    parts = []
    for name, info in ranked[:4]:
        parts.append(f"{name}:{info['tiles']}")
    return ",".join(parts) if parts else "-"


def run_one(
    words: np.ndarray,
    *,
    case: str,
    block_sizes: list[int],
    include_baselines: bool,
    extra: dict | None = None,
    disclaimer: str | None = None,
) -> dict:
    original_bytes = int(words.size * 2)
    best: dict | None = None
    attempts = []
    for block_size in block_sizes:
        t0 = time.perf_counter()
        container = encode_tensor(words, name=case, block_size=block_size, extra=extra)
        encode_s = time.perf_counter() - t0
        blob = container.dumps()
        t1 = time.perf_counter()
        restored_list = decode_container(container)
        decode_s = time.perf_counter() - t1
        restored = restored_list[0]
        verification = assert_exact(words, restored, label=case)
        usage = mode_usage(container)
        tile_hw = choose_tile_hw(block_size, int(words.shape[0]), int(words.shape[1]))
        record = {
            "case": case,
            "block_size": block_size,
            "tile_hw": list(tile_hw),
            "n_words": int(words.size),
            "shape": [int(words.shape[0]), int(words.shape[1])],
            "original_bytes": original_bytes,
            "encoded_bytes": len(blob),
            "payload_bytes": sum(t.payload_bytes for t in container.tensors[0].tiles),
            "metadata_bytes": len(blob)
            - sum(t.payload_bytes for t in container.tensors[0].tiles),
            "bpw": bits_per_weight(len(blob), int(words.size)),
            "ratio_vs_raw_bf16": compression_ratio(len(blob), original_bytes),
            "exact": "PASS" if verification.exact else "FAIL",
            "original_sha256": verification.original_sha256,
            "restored_sha256": verification.restored_sha256,
            "container_sha256": sha256_bytes(blob),
            "source_sha256": sha256_words(words),
            "winning_modes": _winning_modes(usage),
            "mode_usage": usage,
            "encode_seconds": encode_s,
            "decode_seconds": decode_s,
            "disclaimer": disclaimer or DISCLAIMER,
        }
        attempts.append(record)
        if best is None or record["encoded_bytes"] < best["encoded_bytes"]:
            best = record
            best["blob"] = blob
            best["container_extra"] = container.extra
    assert best is not None
    if include_baselines:
        bases = baseline_sizes(words)
        best["baselines"] = bases
        zlib_info = bases.get("zlib") or {}
        best["zlib_bytes"] = zlib_info.get("encoded_bytes")
        best["zlib_bpw"] = zlib_info.get("bpw")
        zstd_info = bases.get("zstd") or {}
        best["zstd_bytes"] = zstd_info.get("encoded_bytes")
        best["zstd_note"] = zstd_info.get("note")
    best["attempts"] = [
        {k: v for k, v in a.items() if k not in {"blob", "container_extra"}}
        for a in attempts
    ]
    return best


def write_reports(row: dict, output_dir: Path) -> None:
    case_dir = output_dir / row["case"]
    case_dir.mkdir(parents=True, exist_ok=True)
    blob = row.pop("blob", None)
    if blob is not None:
        (case_dir / "encoded.pbr").write_bytes(blob)
        if len(blob) != row["encoded_bytes"]:
            raise RuntimeError("Reported encoded_bytes does not match on-disk size")
    (case_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in row.items() if k != "attempts"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (case_dir / "verification.json").write_text(
        json.dumps(
            {
                "exact": row["exact"] == "PASS",
                "differing_words": 0 if row["exact"] == "PASS" else None,
                "original_sha256": row["original_sha256"],
                "restored_sha256": row["restored_sha256"],
                "disclaimer": row.get("disclaimer", DISCLAIMER),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    modes_path = case_dir / "block_modes.csv"
    with modes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "tiles", "tile_share", "bytes"])
        for name, info in row["mode_usage"].items():
            writer.writerow([name, info["tiles"], f"{info['tile_share']:.6f}", info["bytes"]])


def run_poc1(
    *,
    cases: list[str],
    n_words: int,
    cols: int,
    seed: int,
    block_sizes: list[int],
    output_dir: Path,
    input_dir: Path | None,
    include_baselines: bool,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in cases:
        if input_dir is not None:
            words, manifest = load_case(input_dir / f"{case}.u16")
            del manifest
        else:
            words, _manifest = generate_case(
                case, n_words=n_words, cols=cols, seed=seed
            )
        row = run_one(
            words,
            case=case,
            block_sizes=block_sizes,
            include_baselines=include_baselines,
        )
        write_reports(row, output_dir)
        rows.append(row)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "disclaimer": DISCLAIMER,
                "gate": "Gate 1 — codec correctness on controlled tensors",
                "cases": [{k: v for k, v in r.items() if k != "attempts"} for r in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "console_report.txt").write_text(
        DISCLAIMER + "\n\n" + format_table(rows) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PBR Stage 1A controlled-method validation (not real-model evidence)."
    )
    parser.add_argument("--self-test", action="store_true", help="Generate cases and run Gate 1")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--case", action="append", dest="cases", default=None)
    parser.add_argument("--words", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/poc_controlled.yaml"))
    parser.add_argument("--no-baselines", action="store_true")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config if args.config.exists() else None)
    n_words = args.words or int(cfg.get("words", 65536))
    cols = args.cols or int(cfg.get("cols", 256))
    seed = args.seed or int(cfg.get("seed", 7))
    block_sizes = args.block_sizes or [int(x) for x in cfg.get("block_sizes", [256])]
    output_dir = args.output_dir or Path(cfg.get("output_dir", "outputs/reports/poc1"))
    cases = tuple(args.cases) if args.cases else tuple(cfg.get("cases", list(CASES)))
    include_baselines = not args.no_baselines

    input_dir = args.input_dir
    if args.self_test and input_dir is None:
        input_dir = Path(cfg.get("controlled_dir", "outputs/controlled"))
        generate_all(input_dir, n_words=n_words, cols=cols, seed=seed, cases=cases)

    print(DISCLAIMER)
    print()
    try:
        rows = run_poc1(
            cases=list(cases),
            n_words=n_words,
            cols=cols,
            seed=seed,
            block_sizes=block_sizes,
            output_dir=output_dir,
            input_dir=input_dir,
            include_baselines=include_baselines,
        )
    except ExactnessError as exc:
        print(f"GATE 1 FAIL: {exc}")
        return 1

    print(format_table(rows))
    print()
    failed = [r["case"] for r in rows if r["exact"] != "PASS"]
    if failed:
        print(f"GATE 1 FAIL: exactness failed for {', '.join(failed)}")
        return 1
    print("GATE 1 PASS: every controlled case reconstructed bit-exactly.")
    print(f"Reports: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
