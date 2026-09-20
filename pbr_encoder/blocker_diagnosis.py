"""Diagnose why uint16 spatial / tile / dict modes lose on real BF16."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from pbr_codecs.xor_predictor import residuals_prev_row, residuals_prev_value
from pbr_core.bf16 import split_components
from pbr_core.metrics import bits_per_weight
from pbr_core.safetensors_io import load_uint16, tensor_role
from pbr_core.tiles import as_2d, choose_tile_hw, iter_tiles
from pbr_core.types import TILE_HEADER_BYTES
from pbr_encoder.hf_weights import (
    PRIMARY_LICENSE,
    PRIMARY_REPO,
    inventory_from_dir,
    select_weight_specs,
)
from pbr_qualifier.entropy import component_entropy, shannon_entropy

TILE_SIZES = (64, 256, 1024)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _exp_prev(exp: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(exp, dtype=np.uint8).ravel()
    if flat.size <= 1:
        return flat
    out = np.empty_like(flat)
    out[0] = flat[0]
    out[1:] = np.bitwise_xor(flat[1:], flat[:-1])
    return out


def _exp_prev_row(exp_2d: np.ndarray) -> np.ndarray:
    tile = np.ascontiguousarray(exp_2d, dtype=np.uint8)
    pred = np.zeros_like(tile)
    if tile.shape[0] > 1:
        pred[1:, :] = tile[:-1, :]
    return tile ^ pred


def _tile_delta(tiles: list[np.ndarray]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for tile in tiles:
        cur = np.ascontiguousarray(tile, dtype=np.uint16)
        if prev is not None and prev.shape == cur.shape:
            parts.append(cur ^ prev)
        prev = cur
    if not parts:
        return None
    return np.concatenate([p.ravel() for p in parts])


def _tile_delta_exp(tiles: list[np.ndarray]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for tile in tiles:
        _s, exp, _m = split_components(tile)
        if prev is not None and prev.shape == exp.shape:
            parts.append(exp.ravel() ^ prev.ravel())
        prev = exp
    if not parts:
        return None
    return np.concatenate(parts)


def _hash_tile(tile: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(tile, dtype="<u2").tobytes()).digest()


def _duplicate_stats(tiles: list[np.ndarray]) -> dict[str, float | int]:
    if not tiles:
        return {"n_tiles": 0, "unique": 0, "exact_duplicate_rate": 0.0}
    hashes = [_hash_tile(t) for t in tiles]
    unique = len(set(hashes))
    return {
        "n_tiles": len(tiles),
        "unique": unique,
        "exact_duplicate_rate": float(1.0 - unique / len(tiles)),
    }


def _spatial_win_fraction(tiles: list[np.ndarray], residual_fn) -> dict[str, float]:
    """Fraction of tiles whose residual entropy beats raw, ideal vs complete."""
    if not tiles:
        return {"ideal": 0.0, "complete": 0.0, "n_tiles": 0}
    ideal = 0
    complete = 0
    for tile in tiles:
        flat = np.ascontiguousarray(tile)
        n = int(flat.size)
        residual = residual_fn(tile)
        h = shannon_entropy(residual)
        unique = int(np.unique(residual).size)
        ideal_bytes = int(math.ceil(n * h / 8.0)) if h > 0 else 0
        codebook = 2 + unique * 2
        complete_bytes = TILE_HEADER_BYTES + codebook + ideal_bytes
        raw_payload = n * 2
        if ideal_bytes < raw_payload:
            ideal += 1
        if complete_bytes < TILE_HEADER_BYTES + raw_payload:
            complete += 1
    n_tiles = len(tiles)
    return {
        "ideal": ideal / n_tiles,
        "complete": complete / n_tiles,
        "n_tiles": n_tiles,
    }


def diagnose_tensor(name: str, words: np.ndarray) -> dict:
    matrix = as_2d(words)
    sign, exp, mant = split_components(matrix)
    fields = component_entropy(matrix)
    exp_2d = exp.reshape(matrix.shape)
    residuals = {
        "uint16_prev_value": shannon_entropy(residuals_prev_value(matrix)),
        "uint16_prev_row": shannon_entropy(residuals_prev_row(matrix)),
        "exp_prev_value": shannon_entropy(_exp_prev(exp)),
        "exp_prev_row": shannon_entropy(_exp_prev_row(exp_2d)),
        "exp_raw": fields["exponent_bits"],
        "uint16_raw": fields["word_bits"],
    }
    dups = {}
    tile_delta = {}
    win = {}
    for bs in TILE_SIZES:
        tiles = iter_tiles(matrix, bs)
        dups[str(bs)] = _duplicate_stats([t.words for t in tiles])
        td = _tile_delta([t.words for t in tiles])
        tde = _tile_delta_exp([t.words for t in tiles])
        tile_delta[str(bs)] = {
            "uint16": shannon_entropy(td) if td is not None else None,
            "exponent": shannon_entropy(tde) if tde is not None else None,
        }
        win[str(bs)] = {
            "prev_value_uint16": _spatial_win_fraction(
                [t.words for t in tiles], residuals_prev_value
            ),
            "prev_row_uint16": _spatial_win_fraction(
                [t.words for t in tiles], residuals_prev_row
            ),
            "prev_value_exp": _spatial_win_fraction(
                [split_components(t.words)[1] for t in tiles],
                lambda e: _exp_prev(e),
            ),
            "prev_row_exp": _spatial_win_fraction(
                [split_components(t.words)[1].reshape(t.words.shape) for t in tiles],
                _exp_prev_row,
            ),
        }
        residuals[f"uint16_tile_delta_{bs}"] = tile_delta[str(bs)]["uint16"]
        residuals[f"exp_tile_delta_{bs}"] = tile_delta[str(bs)]["exponent"]
    return {
        "name": name,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "n_words": int(matrix.size),
        "role": tensor_role(name),
        "entropy": fields,
        "residual_entropy": residuals,
        "duplicate_tiles": dups,
        "spatial_win_fraction": win,
    }


def _cross_tensor_duplicates(named: list[tuple[str, np.ndarray]]) -> dict:
    by_key: dict[tuple, list[bytes]] = defaultdict(list)
    by_role_shape: dict[tuple[str, tuple[int, int], int], list[bytes]] = defaultdict(list)
    for name, words in named:
        matrix = as_2d(words)
        role = tensor_role(name)
        for bs in TILE_SIZES:
            th, tw = choose_tile_hw(bs, int(matrix.shape[0]), int(matrix.shape[1]))
            key = (int(matrix.shape[0]), int(matrix.shape[1]), th, tw, bs)
            for tile in iter_tiles(matrix, bs):
                h = _hash_tile(tile.words)
                by_key[key].append(h)
                by_role_shape[(role, (int(matrix.shape[0]), int(matrix.shape[1])), bs)].append(h)
    out = {}
    for bs in TILE_SIZES:
        hashes = []
        role_hashes = []
        for key, hs in by_key.items():
            if key[-1] == bs:
                hashes.extend(hs)
        for key, hs in by_role_shape.items():
            if key[-1] == bs:
                role_hashes.extend(hs)
        counts = Counter(hashes)
        shared = sum(1 for h, c in counts.items() if c > 1)
        out[str(bs)] = {
            "tiles": len(hashes),
            "unique": len(counts),
            "hashes_seen_more_than_once": shared,
            "exact_duplicate_rate": float(1.0 - (len(counts) / len(hashes))) if hashes else 0.0,
            "same_role_exact_duplicate_rate": (
                float(1.0 - (len(set(role_hashes)) / len(role_hashes))) if role_hashes else 0.0
            ),
        }
    return out


def _summarize(rows: list[dict]) -> dict:
    def wavg(key_path: list[str]) -> float:
        num = 0.0
        den = 0
        for row in rows:
            cur: object = row
            for k in key_path:
                cur = cur[k]  # type: ignore[index]
            num += float(cur) * int(row["n_words"])
            den += int(row["n_words"])
        return num / den if den else 0.0

    win256 = {"ideal_uint16": 0.0, "complete_uint16": 0.0, "ideal_exp": 0.0, "complete_exp": 0.0}
    tiles = 0
    for row in rows:
        pv = row["spatial_win_fraction"]["256"]["prev_value_uint16"]
        ev = row["spatial_win_fraction"]["256"]["prev_value_exp"]
        n = int(pv["n_tiles"])
        tiles += n
        win256["ideal_uint16"] += pv["ideal"] * n
        win256["complete_uint16"] += pv["complete"] * n
        win256["ideal_exp"] += ev["ideal"] * n
        win256["complete_exp"] += ev["complete"] * n
    if tiles:
        for k in win256:
            win256[k] /= tiles
    return {
        "tensor_count": len(rows),
        "n_words": sum(r["n_words"] for r in rows),
        "weighted_H_uint16": wavg(["entropy", "word_bits"]),
        "weighted_H_exponent": wavg(["entropy", "exponent_bits"]),
        "weighted_H_mantissa": wavg(["entropy", "mantissa_bits"]),
        "weighted_H_sign": wavg(["entropy", "sign_bits"]),
        "weighted_H_uint16_prev_value": wavg(["residual_entropy", "uint16_prev_value"]),
        "weighted_H_uint16_prev_row": wavg(["residual_entropy", "uint16_prev_row"]),
        "weighted_H_exp_prev_value": wavg(["residual_entropy", "exp_prev_value"]),
        "weighted_H_exp_prev_row": wavg(["residual_entropy", "exp_prev_row"]),
        "spatial_win_fraction_tile256": win256,
        "note": (
            "Mantissa entropy stays near 7 bits; uint16 residuals stay near 16 bits. "
            "Exponent residuals can drop below H(exp). Exact duplicate tiles are rare. "
            "These numbers explain why uint16 spatial/dict modes lose to exponent Huffman."
        ),
    }


def format_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Blocker diagnosis (Qwen Stage 1B set)",
        "",
        "Lossless BF16 only. Not a 1–2 GB / 8 GB claim.",
        "",
        f"Tensors: **{s['tensor_count']}**. Words: **{s['n_words']}**.",
        "",
        "| field | weighted entropy (bits) |",
        "| --- | ---: |",
        f"| H(uint16) | {s['weighted_H_uint16']:.4f} |",
        f"| H(exponent) | {s['weighted_H_exponent']:.4f} |",
        f"| H(mantissa) | {s['weighted_H_mantissa']:.4f} |",
        f"| H(sign) | {s['weighted_H_sign']:.4f} |",
        f"| H(uint16 prev_value residual) | {s['weighted_H_uint16_prev_value']:.4f} |",
        f"| H(uint16 prev_row residual) | {s['weighted_H_uint16_prev_row']:.4f} |",
        f"| H(exponent prev_value residual) | {s['weighted_H_exp_prev_value']:.4f} |",
        f"| H(exponent prev_row residual) | {s['weighted_H_exp_prev_row']:.4f} |",
        "",
        "Tile-256 fraction where spatial residual entropy beats raw:",
        "",
        "| | ideal (ignore meta) | complete container |",
        "| --- | ---: | ---: |",
        f"| uint16 prev_value | {s['spatial_win_fraction_tile256']['ideal_uint16']:.4f} | {s['spatial_win_fraction_tile256']['complete_uint16']:.4f} |",
        f"| exponent prev_value | {s['spatial_win_fraction_tile256']['ideal_exp']:.4f} | {s['spatial_win_fraction_tile256']['complete_exp']:.4f} |",
        "",
        "Cross-tensor exact duplicate tiles:",
        "",
        "| tile | tiles | unique | dup rate | same-role dup rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for bs, info in report["cross_tensor_duplicates"].items():
        lines.append(
            f"| {bs} | {info['tiles']} | {info['unique']} | "
            f"{info['exact_duplicate_rate']:.6f} | {info['same_role_exact_duplicate_rate']:.6f} |"
        )
    lines += ["", s["note"], ""]
    return "\n".join(lines)


def run_diagnosis(
    *,
    model_dir: Path,
    specs,
    output_json: Path,
    output_md: Path,
    model_meta: dict,
) -> dict:
    named: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    for spec in specs:
        words = as_2d(load_uint16(spec))
        print(f"diagnose {spec.name}  {list(spec.shape)}  {spec.nbytes}", flush=True)
        row = diagnose_tensor(spec.name, words)
        rows.append(row)
        named.append((spec.name, words))
    report = {
        "disclaimer": (
            "Blocker diagnosis on a public checkpoint sample. "
            "Not a production ratio and not a 1–2 GB / 8 GB claim."
        ),
        "model": model_meta,
        "tensors": rows,
        "cross_tensor_duplicates": _cross_tensor_duplicates(named),
        "summary": _summarize(rows),
        "ideal_exp_huffman_bpw": bits_per_weight(
            int(
                sum(
                    r["n_words"]
                    + math.ceil(r["n_words"] * r["entropy"]["exponent_bits"] / 8.0)
                    for r in rows
                )
            ),
            sum(r["n_words"] for r in rows),
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose PBR spatial/tile blockers on Qwen.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/blocker_diagnosis_qwen.json"))
    parser.add_argument("--output-md", type=Path, default=Path("artifacts/blocker_diagnosis_qwen.md"))
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    specs_all = inventory_from_dir(args.model_dir)
    chosen = select_weight_specs(
        specs_all,
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    n16 = sum(1 for s in specs_all if s.dtype in {"BF16", "F16", "FP16"})
    meta = {
        "repo_id": cfg.get("repo", PRIMARY_REPO),
        "revision": cfg.get("revision", "main"),
        "license": cfg.get("license", PRIMARY_LICENSE),
        "local_dir": str(args.model_dir),
        "inventory_tensors": len(specs_all),
        "inventory_16bit": n16,
        "selected": [s.name for s in chosen],
        "full_model_note": (
            f"Checkpoint has {n16} 16-bit tensors; this diagnosis uses the "
            f"Stage 1B {len(chosen)}-tensor linear/attention sample."
        ),
    }
    report = run_diagnosis(
        model_dir=args.model_dir,
        specs=chosen,
        output_json=args.output_json,
        output_md=args.output_md,
        model_meta=meta,
    )
    print()
    print(format_markdown(report))
    print(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
