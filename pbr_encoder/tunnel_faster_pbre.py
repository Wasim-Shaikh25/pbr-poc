"""Even faster PBR-E decode in the disk-RAM tunnel.

Fused C rANS+join, 4-way prefetch, layer-sized decoded cache, optional warm
uint16 sidecar. Bit-exact vs BF16 source. Not an ≤8 BPW or 27B-phone claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from pbr_core.rans import rans_impl
from pbr_encoder.disk_tunnel import (
    DEFAULT_DECODED_DIR,
    DEFAULT_MODEL,
    DEFAULT_PBRE_DIR,
    DEFAULT_TOKENS,
    DISCLAIMER,
    LM_HEAD_ROWS,
    _token_ids,
    bytes_human,
    format_markdown,
    load_config_json,
    materialize_decoded_cache,
)
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO

FASTER_DISCLAIMER = (
    DISCLAIMER
    + " Faster path: fused C rANS+join, 4 decode threads, ~2-layer decoded cache, "
    "optional warm uint16 sidecar (mmap-like). decode_s is the sum of job times "
    "and can exceed wall when threads overlap."
)


def format_faster_markdown(report: dict) -> str:
    text = format_markdown(report)
    text = text.replace("# Disk–RAM tunnel PoC", "# Faster PBR-E disk–RAM tunnel (stage 2)")
    extra = [
        "",
        "## Decode path",
        "",
        f"rANS backend in this process catalog: `{report.get('rans_impl_note', rans_impl())}`.",
        "`pbre_fast` is the previous C rANS + 2-slot cache + 1 prefetch thread (~8 s).",
        "`pbre_faster` (cold): fused C tile decode (rANS + BF16 join, skip JSON header),",
        "4 worker threads, prefetch depth 16, 32-slot LRU (~two layers of linears).",
        "Single-stream rANS is not SIMD-rewritten (bitstream must stay bit-exact);",
        "parallelism is across tensors while GEMM runs.",
        "`pbre_faster_warm`: decoded uint16 sidecars materialized after SHA-checked",
        "decode; the forward copies one tensor like mmap. RAM stays a working set,",
        "not the full ~1 GB resident checkpoint. Sidecar is full uint16 (not exponents",
        "only) because join is cheap next to rANS; disk holds the extra copy.",
        "",
        "Reconstruction stays **bit-exact** (logits max_abs=0 vs full-load).",
        "Cold path SHA-256 checks every PBR-E tensor. Warm path SHA ran at materialize.",
        "",
        "Not an ≤8 BPW claim. Decode speed ≠ compression ratio.",
        "",
    ]
    pre = report.get("warm_materialize") or {}
    if pre:
        extra += [
            "## Warm sidecar",
            "",
            f"Materialize: {pre.get('n_tensors', 0)} tensors, "
            f"{bytes_human(pre.get('decoded_bytes') or 0)} decoded uint16, "
            f"{pre.get('wall_s', 0):.3f}s wall, SHA checks {pre.get('sha_checks', 0)}.",
            "",
        ]
    return text.rstrip() + "\n" + "\n".join(extra)


def _spawn(args: argparse.Namespace, mode: str, worker_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not pp else str(root) + os.pathsep + pp
    tunnel = root / "pbr_encoder" / "disk_tunnel.py"
    cmd = [
        sys.executable,
        str(tunnel),
        "--mode",
        mode,
        "--model-dir",
        str(args.model_dir),
        "--pbre-dir",
        str(args.pbre_dir),
        "--tokens",
        str(args.tokens),
        "--layers",
        str(args.layers),
        "--lm-head-chunk",
        str(args.lm_head_chunk),
        "--decoded-dir",
        str(args.decoded_dir),
        "--config",
        str(args.config),
        "--worker-dir",
        str(worker_dir),
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    subprocess.check_call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Even faster PBR-E disk-RAM tunnel microbench.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pbre-dir", type=Path, default=DEFAULT_PBRE_DIR)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--lm-head-chunk", type=int, default=LM_HEAD_ROWS)
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/tunnel_faster_pbre.json"))
    parser.add_argument("--md-out", type=Path, default=Path("artifacts/tunnel_faster_pbre.md"))
    parser.add_argument(
        "--skip-warm",
        action="store_true",
        help="Do not materialize uint16 sidecars / skip pbre_faster_warm",
    )
    args = parser.parse_args(argv)
    print(FASTER_DISCLAIMER, flush=True)
    print(f"rANS impl available: {rans_impl()}", flush=True)
    cfg = load_config_json(args.model_dir)
    yaml_meta = {}
    if args.config.is_file():
        yaml_meta = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    modes = ["full", "mmap", "pbre_fast", "pbre_faster"]

    if args.encode or not (args.pbre_dir / "index.json").is_file():
        from pbr_encoder.disk_tunnel import _maybe_encode_pbre

        _maybe_encode_pbre(args.model_dir, args.pbre_dir)

    collected: dict[str, dict] = {}
    logits_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        wdir = Path(tempfile.mkdtemp(prefix=f"pbr_faster_{mode}_"))
        print(f"=== spawn {mode} ===", flush=True)
        _spawn(args, mode, wdir)
        s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
        collected[mode] = s
        logits_by_mode[mode] = np.load(wdir / "logits.npy")
        print(
            f"  peak={bytes_human(s['rss_peak_sampled_bytes'])}  "
            f"maxrss={bytes_human(s['rss_maxrss_bytes'])}  "
            f"decode={s['decode_s']:.3f}s compute={s['compute_s']:.3f}s "
            f"wall={s['wall_s']:.3f}s rans={s.get('rans_impl')} exact={s['exact']}",
            flush=True,
        )

    warm_meta = None
    if not args.skip_warm:
        print("=== materialize decoded uint16 sidecar ===", flush=True)
        warm_meta = materialize_decoded_cache(
            args.pbre_dir,
            args.decoded_dir,
            workers=min(4, os.cpu_count() or 4),
            verify_sha=not args.no_verify,
        )
        wdir = Path(tempfile.mkdtemp(prefix="pbr_faster_warm_"))
        print("=== spawn pbre_faster_warm ===", flush=True)
        _spawn(args, "pbre_faster_warm", wdir)
        s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
        collected["pbre_faster_warm"] = s
        logits_by_mode["pbre_faster_warm"] = np.load(wdir / "logits.npy")
        print(
            f"  peak={bytes_human(s['rss_peak_sampled_bytes'])}  "
            f"maxrss={bytes_human(s['rss_maxrss_bytes'])}  "
            f"decode={s['decode_s']:.3f}s compute={s['compute_s']:.3f}s "
            f"wall={s['wall_s']:.3f}s exact={s['exact']}",
            flush=True,
        )

    match_notes: dict[str, str] = {}
    if "full" in logits_by_mode:
        ref = logits_by_mode["full"]
        for mode, arr in logits_by_mode.items():
            if mode == "full":
                continue
            delta = float(np.max(np.abs(arr.astype(np.float64) - ref.astype(np.float64))))
            ok = delta < 1e-4
            match_notes[mode] = f"{'PASS' if ok else 'FAIL'} max_abs={delta:.3e}"
            if not ok:
                collected[mode]["exact"] = "FAIL"

    ids = _token_ids(args.tokens, int(cfg.get("vocab_size") or 256), seed=1)
    report = {
        "disclaimer": FASTER_DISCLAIMER,
        "repo_id": yaml_meta.get("repo", PRIMARY_REPO),
        "revision": yaml_meta.get("revision", ""),
        "license": yaml_meta.get("license", PRIMARY_LICENSE),
        "model_dir": str(args.model_dir),
        "n_tokens": int(args.tokens),
        "n_layers": args.layers or int(cfg.get("num_hidden_layers") or 0),
        "token_ids": [int(x) for x in ids.tolist()],
        "modes": collected,
        "logits_match": match_notes,
        "subprocess_isolated": True,
        "rans_impl_note": rans_impl(),
        "warm_materialize": warm_meta,
        "before_after": {
            "pbre_fast_wall_s_prior": 7.980,
            "pbre_fast_wall_s": (collected.get("pbre_fast") or {}).get("wall_s"),
            "pbre_faster_wall_s": (collected.get("pbre_faster") or {}).get("wall_s"),
            "pbre_faster_warm_wall_s": (collected.get("pbre_faster_warm") or {}).get("wall_s"),
            "mmap_wall_s": (collected.get("mmap") or {}).get("wall_s"),
            "full_wall_s": (collected.get("full") or {}).get("wall_s"),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_faster_markdown(report)
    args.md_out.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {args.md_out} and {args.json_out}")
    fails = [m for m, s in collected.items() if s.get("exact") == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
