"""Ablate canonical Huffman vs rANS on BF16 exponents (complete bytes)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_core.bf16 import split_components
from pbr_core.metrics import bits_per_weight
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d
from pbr_core.types import TILE_HEADER_BYTES
from pbr_encoder.hf_weights import (
    PRIMARY_LICENSE,
    PRIMARY_REPO,
    inventory_from_dir,
    select_weight_specs,
)
from pbr_encoder.verification import assert_exact
from pbr_qualifier.entropy import shannon_entropy

DISCLAIMER = (
    "Exponent coder ablation (canonical Huffman vs rANS). "
    "Complete bytes include table + stream + packed sign/mantissa + one tile header. "
    "Lossless BF16 only. Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
)


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _measure(name: str, words: np.ndarray, codec) -> dict:
    enc = codec.encode(words)
    rows, cols = int(words.shape[0]), int(words.shape[1])
    if enc is None:
        return {
            "codec": name,
            "payload_bytes": None,
            "complete_bytes": None,
            "bpw": None,
            "exact": "SKIP",
        }
    enc.rows, enc.cols = rows, cols
    restored = codec.decode(enc)
    assert_exact(words, restored, label=f"{name}:{rows}x{cols}")
    complete = TILE_HEADER_BYTES + len(enc.payload)
    return {
        "codec": name,
        "mode_name": enc.mode_name,
        "payload_bytes": len(enc.payload),
        "complete_bytes": complete,
        "bpw": bits_per_weight(complete, int(words.size)),
        "exact": "PASS",
    }


def format_markdown(report: dict) -> str:
    t = report["totals"]
    lines = [
        f"# Exponent coder ablation ({report['model'].get('repo_id', 'checkpoint')})",
        "",
        DISCLAIMER,
        "",
        f"- revision: `{report['model'].get('revision', '')}`",
        f"- tensors: **{t['tensor_count']}**",
        f"- words: **{t['n_words']}**",
        f"- original bytes: **{t['original_bytes']}**",
        f"- H(exp) weighted: **{t['weighted_H_exp']:.4f}**",
        "",
        "| coder | enc B | BPW | vs bound | exact |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| entropy bound (table+stream+SM+hdr) | {t['bound_bytes']} | {t['bound_bpw']:.4f} | 0 | n/a |",
        f"| canonical Huffman | {t['huffman_bytes']} | {t['huffman_bpw']:.4f} | "
        f"+{t['huffman_bpw'] - t['bound_bpw']:.4f} | {t['huffman_exact']} |",
        f"| rANS | {t['rans_bytes']} | {t['rans_bpw']:.4f} | "
        f"+{t['rans_bpw'] - t['bound_bpw']:.4f} | {t['rans_exact']} |",
        "",
        f"Huffman − rANS: **{t['huffman_bpw'] - t['rans_bpw']:.4f} BPW** "
        f"({t['huffman_bytes'] - t['rans_bytes']} bytes).",
        "",
    ]
    delta = t["huffman_bpw"] - t["rans_bpw"]
    if delta > 0.01:
        lines.append(
            f"rANS is smaller by {delta:.4f} BPW on this sample. Still DF11-class; not ≤4 BPW."
        )
    elif delta < -0.01:
        lines.append(
            f"Huffman is smaller by {-delta:.4f} BPW (rANS table/state overhead). "
            "The 0.26 BPW PBR-E-vs-bound gap is mostly headers, not Huffman redundancy."
        )
    else:
        lines.append(
            "Huffman and rANS match within 0.01 BPW. The remaining gap to the entropy "
            "bound is container/header/table overhead, not coder choice."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Huffman vs rANS on exponents.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--tag", default="qwen")
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    specs = select_weight_specs(
        inventory_from_dir(args.model_dir),
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    huff_c = Bf16ExpHuffmanCodec()
    rans_c = Bf16ExpRansCodec()
    rows: list[dict] = []
    huff_b = rans_b = bound_b = 0
    n_words = 0
    orig = 0
    h_exp_acc = 0.0
    print(DISCLAIMER)
    print(f"selected {len(specs)} tensors", flush=True)
    for spec in specs:
        words = as_2d(load_uint16(spec))
        _s, exp, _m = split_components(words)
        h_exp = shannon_entropy(exp)
        n = int(words.size)
        unique = int(np.unique(exp).size)
        codebook = 2 + unique * 2
        bound = TILE_HEADER_BYTES + 7 + codebook + int(math.ceil(n * h_exp / 8.0)) + n
        print(f"ablate {spec.name}  {list(spec.shape)}  H(exp)={h_exp:.4f}", flush=True)
        huff = _measure("huffman", words, huff_c)
        rans = _measure("rans", words, rans_c)
        rec = {
            "name": spec.name,
            "n_words": n,
            "H_exp": h_exp,
            "bound_bytes": bound,
            "huffman": huff,
            "rans": rans,
        }
        rows.append(rec)
        orig += n * 2
        n_words += n
        h_exp_acc += h_exp * n
        bound_b += bound
        huff_b += int(huff["complete_bytes"] or 0)
        rans_b += int(rans["complete_bytes"] or 0)
        print(
            f"  -> huff={huff['complete_bytes']}  rans={rans['complete_bytes']}  "
            f"bound={bound}  exact={huff['exact']}/{rans['exact']}",
            flush=True,
        )
    totals = {
        "tensor_count": len(rows),
        "n_words": n_words,
        "original_bytes": orig,
        "weighted_H_exp": h_exp_acc / n_words if n_words else 0.0,
        "bound_bytes": bound_b,
        "bound_bpw": bits_per_weight(bound_b, n_words),
        "huffman_bytes": huff_b,
        "huffman_bpw": bits_per_weight(huff_b, n_words),
        "huffman_exact": "PASS" if all(r["huffman"]["exact"] == "PASS" for r in rows) else "FAIL",
        "rans_bytes": rans_b,
        "rans_bpw": bits_per_weight(rans_b, n_words),
        "rans_exact": "PASS" if all(r["rans"]["exact"] == "PASS" for r in rows) else "FAIL",
    }
    report = {
        "disclaimer": DISCLAIMER,
        "model": {
            "repo_id": cfg.get("repo", PRIMARY_REPO),
            "revision": cfg.get("revision", "main"),
            "license": cfg.get("license", PRIMARY_LICENSE),
            "selected": [s.name for s in specs],
        },
        "totals": totals,
        "tensors": rows,
    }
    out_json = Path(f"artifacts/exp_coder_ablation_{args.tag}.json")
    out_md = Path(f"artifacts/exp_coder_ablation_{args.tag}.md")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(report)
    out_md.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {out_json} and {out_md}")
    if totals["huffman_exact"] != "PASS" or totals["rans_exact"] != "PASS":
        print("ABLATION FAIL: exactness")
        return 1
    print("ABLATION PASS: Huffman and rANS reconstructed bit-exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
