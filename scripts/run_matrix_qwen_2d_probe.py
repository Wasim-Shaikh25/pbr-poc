#!/usr/bin/env python3
"""Small Qwen 2-D Matrix Mantissa probe (post Gate D).

Tiles real BF16 linear weights from Qwen2.5-0.5B-Instruct into V3 16x16
blocks, picks modes via evaluate_tile, round-trips through encode_tile_blob /
decode_tile_blob.

Reports:
- candidate_payload_bpw — winner payload from evaluate_tile
- raw_candidate_bpw — RAW payload for the same tiles
- implied_total_bpw_sign1_exp8_cand — 1 + 8 + candidate (pessimistic if exp raw)

Exact field round-trip required. Probe only — not a <=4 BPW claim.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pbr_core.bf16 import split_components
from pbr_core.safetensors_io import (
    TWO_BYTE_DTYPES,
    inventory_model,
    is_linear_weight,
    load_uint16,
)
from pbr_poc import decode_tile_blob, encode_tile_blob, evaluate_tile

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
TILE = (16, 16)
VERSION = 3

PREFERRED = [
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.12.mlp.down_proj.weight",
    "layers.23.mlp.gate_proj.weight",
]


def iter_full_tiles(mant: np.ndarray, exp: np.ndarray):
    h, w = mant.shape
    th, tw = TILE
    for r0 in range(0, h - th + 1, th):
        for c0 in range(0, w - tw + 1, tw):
            yield mant[r0 : r0 + th, c0 : c0 + tw], exp[r0 : r0 + th, c0 : c0 + tw]


def mode_bucket(name: str) -> str:
    return "matrix" if name.startswith("matrix:") else name


def probe_tensor(spec, *, max_tiles: int) -> dict:
    words = load_uint16(spec)
    _sign, exp, mant = split_components(words)
    h, w = int(spec.shape[0]), int(spec.shape[1])
    mant2 = mant.reshape(h, w)
    exp2 = exp.reshape(h, w)

    modes: Counter[str] = Counter()
    cand_bytes = 0
    raw_cand_bytes = 0
    container_bytes = 0
    n_vals = 0
    tiles = 0
    exact = True

    for m, e in iter_full_tiles(mant2, exp2):
        flat_m = m.ravel()
        flat_e = e.ravel()
        (mode, payload), cands = evaluate_tile(flat_m, flat_e, VERSION, TILE)
        raw_len = next(len(p) for n, p in cands if n == "raw")
        blob = encode_tile_blob(
            flat_m, flat_e, VERSION, TILE, mode_name=mode, payload=payload
        )
        md, ed, info = decode_tile_blob(blob)
        if not (np.array_equal(md, flat_m) and np.array_equal(ed, flat_e)):
            exact = False
            break
        if info["complete_bytes"] != len(blob):
            exact = False
            break

        modes[mode_bucket(mode)] += 1
        cand_bytes += len(payload)
        raw_cand_bytes += raw_len
        container_bytes += len(blob)
        n_vals += flat_m.size
        tiles += 1
        if max_tiles and tiles >= max_tiles:
            break

    if n_vals == 0:
        return {
            "name": spec.name,
            "shape": list(spec.shape),
            "tiles": 0,
            "skipped": "no full 16x16 tiles",
        }

    cand_bpw = 8.0 * cand_bytes / n_vals
    raw_bpw = 8.0 * raw_cand_bytes / n_vals
    cont_bpw = 8.0 * container_bytes / n_vals
    return {
        "name": spec.name,
        "shape": list(spec.shape),
        "n_words_tensor": spec.n_words,
        "tiles": tiles,
        "n_vals_tiled": n_vals,
        "exact": exact,
        "mode_counts": dict(modes),
        "candidate_payload_bytes": cand_bytes,
        "candidate_payload_bpw": round(cand_bpw, 4),
        "raw_candidate_bytes": raw_cand_bytes,
        "raw_candidate_bpw": round(raw_bpw, 4),
        "vs_raw_candidate": round(cand_bpw / raw_bpw, 4) if raw_bpw else None,
        "container_bytes": container_bytes,
        "container_bpw": round(cont_bpw, 4),
        "implied_total_bpw_sign1_exp8_cand": round(1.0 + 8.0 + cand_bpw, 4),
    }


def select_tensors(model_dir: Path, limit: int):
    specs = [
        s
        for s in inventory_model(model_dir)
        if s.dtype in TWO_BYTE_DTYPES
        and s.ndim == 2
        and s.shape[0] >= 16
        and s.shape[1] >= 16
    ]
    linears = [s for s in specs if is_linear_weight(s.name)]
    pool = linears or specs
    chosen = []
    for sub in PREFERRED:
        for s in pool:
            if sub in s.name and s not in chosen:
                chosen.append(s)
                break
        if len(chosen) >= limit:
            return chosen
    for s in pool:
        if s not in chosen:
            chosen.append(s)
        if len(chosen) >= limit:
            break
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--max-tensors", type=int, default=6)
    ap.add_argument("--max-tiles-per-tensor", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("artifacts/matrix_qwen_2d_probe.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/matrix_qwen_2d_probe.md"))
    args = ap.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"model dir missing: {args.model_dir}")

    tensors = select_tensors(args.model_dir, args.max_tensors)
    print(f"Probe {len(tensors)} tensors from {args.model_dir}", flush=True)
    t0 = time.perf_counter()
    rows = []
    for spec in tensors:
        print(f"  {spec.name} {list(spec.shape)}", flush=True)
        rows.append(probe_tensor(spec, max_tiles=args.max_tiles_per_tensor))

    tot_vals = sum(r.get("n_vals_tiled", 0) for r in rows)
    tot_cand = sum(r.get("candidate_payload_bytes", 0) for r in rows)
    tot_raw = sum(r.get("raw_candidate_bytes", 0) for r in rows)
    mode_sum: Counter[str] = Counter()
    for r in rows:
        mode_sum.update(r.get("mode_counts") or {})

    cand_bpw = (8.0 * tot_cand / tot_vals) if tot_vals else None
    raw_bpw = (8.0 * tot_raw / tot_vals) if tot_vals else None
    summary = {
        "model_dir": str(args.model_dir),
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "tile": list(TILE),
        "version": VERSION,
        "max_tiles_per_tensor": args.max_tiles_per_tensor,
        "tensors": rows,
        "aggregate": {
            "n_vals_tiled": tot_vals,
            "candidate_payload_bpw": round(cand_bpw, 4) if cand_bpw is not None else None,
            "raw_candidate_bpw": round(raw_bpw, 4) if raw_bpw is not None else None,
            "implied_total_bpw_sign1_exp8_cand": (
                round(1.0 + 8.0 + cand_bpw, 4) if cand_bpw is not None else None
            ),
            "mode_counts": dict(mode_sum),
            "all_exact": all(r.get("exact", False) for r in rows if r.get("tiles")),
            "prior_adaptive_mantissa_reference": {
                "mantissa_bpw": 7.0001,
                "total_bpw": 10.6194,
                "note": "ALL_RAW adaptive mantissa on full Qwen (earlier PR)",
            },
        },
        "disclaimer": (
            "Probe only on a tiled subset of 2-D linear weights. "
            "candidate_payload_bpw is the evaluate_tile winner payload (not whole-model). "
            "implied_total assumes raw sign (1) + raw exp (8) + candidate mantissa — "
            "exp is usually ~2.6 BPW with rANS, so real totals would be lower by ~5.4 if "
            "exp coding is reused. Not a <=4 BPW claim. container_bpw includes Gate D "
            "framing + stored exp plane and is not comparable to PBR-E totals."
        ),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    agg = summary["aggregate"]
    lines = [
        "# Matrix Mantissa V3 — Qwen 2-D probe",
        "",
        f"Model: `{summary['repo']}` · tile `{TILE}` · V{VERSION}",
        "",
        "## Aggregate (tiled region only)",
        "",
        f"- Values tiled: **{agg['n_vals_tiled']}**",
        f"- Candidate payload BPW: **{agg['candidate_payload_bpw']}**",
        f"- RAW candidate BPW: **{agg['raw_candidate_bpw']}**",
        f"- Implied total (sign1+exp8+cand): **{agg['implied_total_bpw_sign1_exp8_cand']}**",
        f"- Mode counts: `{agg['mode_counts']}`",
        f"- Exact round-trip: **{agg['all_exact']}**",
        f"- Wall: {summary['wall_seconds']}s",
        "",
        "Prior full-model adaptive mantissa (reference): ~7.0001 mant / ~10.62 total (ALL_RAW).",
        "",
        "## Honesty",
        "",
        summary["disclaimer"],
        "",
        "## Per tensor",
        "",
        "| tensor | shape | tiles | cand BPW | raw BPW | vs raw | modes | exact |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        if not r.get("tiles"):
            lines.append(
                f"| `{r['name']}` | {r.get('shape')} | 0 | — | — | — | skipped | — |"
            )
            continue
        lines.append(
            f"| `{r['name']}` | {r['shape']} | {r['tiles']} | "
            f"{r['candidate_payload_bpw']} | {r['raw_candidate_bpw']} | "
            f"{r['vs_raw_candidate']} | `{r['mode_counts']}` | {r['exact']} |"
        )
    lines.append("")
    args.md.write_text("\n".join(lines) + "\n")
    print(json.dumps(agg, indent=2))
    print(f"Wrote {args.out} and {args.md}")
    return 0 if agg["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
