"""CLI: adaptive 7-bit mantissa codec on synthetic cases and real BF16 checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from pbr_adaptive_mantissa.codec import (
    BLOCK_SIZE,
    DISCLAIMER,
    PBRE_REF_BPW,
    STRAT_NAMES,
    decode_bf16_bundle,
    encode_bf16_bundle,
)
from pbr_adaptive_mantissa.predict import RULE_NAMES, local_pos
from pbr_core.bf16 import join_components, special_payload_words
from pbr_core.safetensors_io import TWO_BYTE_DTYPES, inventory_model, load_uint16, tensor_role
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO
from pbr_encoder.verification import assert_exact

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")


def _bytes_human(n: int | float) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(x) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.3f} {unit}"
        x /= 1024.0
    return f"{x:.3f} GiB"


def synthetic_report() -> dict:
    rng = np.random.default_rng(0)
    n = 4096
    exp = rng.integers(110, 140, size=n, dtype=np.uint8)
    cases: dict[str, np.ndarray] = {
        "random": rng.integers(0, 128, size=n, dtype=np.uint8),
        "skewed": rng.choice(np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 17], dtype=np.uint8), size=n),
        "structured_pos": local_pos(n, BLOCK_SIZE) & np.uint8(0x7F),
    }
    out = {"disclaimer": DISCLAIMER, "n": n, "cases": {}}
    for name, mant in cases.items():
        sign = np.zeros(n, dtype=np.uint8)
        words = join_components(sign, exp, mant)
        bundle = encode_bf16_bundle(words)
        rec = decode_bf16_bundle(bundle.blob)
        assert_exact(words, rec, label=name)
        man = bundle.mantissa
        out["cases"][name] = {
            "strategy": STRAT_NAMES[man.strategy],
            "rule": RULE_NAMES.get(man.rule, str(man.rule)),
            "mantissa_bpw": man.mantissa_bpw,
            "total_bpw": bundle.total_bpw,
            "used_huffman_table": man.used_huffman_table,
            "mode_counts": man.mode_counts,
            "sha256": bundle.sha256,
            "exact": True,
        }
    # Specials: field extract without FP.
    spec = np.resize(special_payload_words(), 256)
    bundle = encode_bf16_bundle(spec)
    assert_exact(spec, decode_bf16_bundle(bundle.blob), label="specials")
    out["specials_exact"] = True
    return out


def encode_model(
    model_dir: Path,
    *,
    max_tensors: int = 0,
    block_size: int = BLOCK_SIZE,
) -> dict:
    specs = [s for s in inventory_model(model_dir) if s.dtype in TWO_BYTE_DTYPES]
    specs = sorted(specs, key=lambda s: s.name)
    if max_tensors:
        specs = specs[: int(max_tensors)]
    rows = []
    tot_words = 0
    tot_raw = 0
    tot_mant = 0
    tot_sign = 0
    tot_exp = 0
    tot_bundle = 0
    strat_counts: dict[str, int] = {}
    mode_sum = {"RAW": 0, "HUFFMAN": 0, "CONTEXT": 0}
    t0 = time.perf_counter()
    for spec in specs:
        print(f"  encode {spec.name}  {list(spec.shape)}  {spec.n_words} words", flush=True)
        words = load_uint16(spec)
        bundle = encode_bf16_bundle(words, block_size=block_size)
        rec = decode_bf16_bundle(bundle.blob)
        vr = assert_exact(words, rec, label=spec.name)
        man = bundle.mantissa
        strat_name = STRAT_NAMES[man.strategy]
        strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1
        for k, v in man.mode_counts.items():
            mode_sum[k] = mode_sum.get(k, 0) + int(v)
        rows.append(
            {
                "name": spec.name,
                "role": tensor_role(spec.name),
                "shape": list(spec.shape),
                "n_words": spec.n_words,
                "raw_bytes": spec.nbytes,
                "strategy": strat_name,
                "rule": RULE_NAMES.get(man.rule, str(man.rule)),
                "mantissa_bytes": man.mantissa_bytes,
                "mantissa_bpw": man.mantissa_bpw,
                "sign_bytes": bundle.sign_bytes,
                "exp_bytes": bundle.exp_bytes,
                "total_bytes": bundle.total_bytes,
                "total_bpw": bundle.total_bpw,
                "used_huffman_table": man.used_huffman_table,
                "codebook_bytes": man.codebook_bytes,
                "directory_bytes": man.directory_bytes,
                "mode_counts": man.mode_counts,
                "freeze_hit_rate": man.freeze_hit_rate,
                "sha256": vr.original_sha256,
                "exact": True,
            }
        )
        tot_words += spec.n_words
        tot_raw += spec.nbytes
        tot_mant += man.mantissa_bytes
        tot_sign += bundle.sign_bytes
        tot_exp += bundle.exp_bytes
        tot_bundle += bundle.total_bytes
        del words, rec, bundle
    wall = time.perf_counter() - t0
    mant_bpw = (8.0 * tot_mant / tot_words) if tot_words else 0.0
    total_bpw = (8.0 * tot_bundle / tot_words) if tot_words else 0.0
    return {
        "disclaimer": DISCLAIMER,
        "model_dir": str(model_dir),
        "n_tensors": len(rows),
        "n_words": tot_words,
        "raw_bytes": tot_raw,
        "mantissa_bytes": tot_mant,
        "sign_bytes": tot_sign,
        "exp_bytes": tot_exp,
        "bundle_bytes": tot_bundle,
        "mantissa_bpw": mant_bpw,
        "total_bpw": total_bpw,
        "raw_bpw": 16.0,
        "pbre_ref_bpw": PBRE_REF_BPW,
        "vs_raw_16": total_bpw / 16.0 if tot_words else 0.0,
        "vs_pbre_10_6": total_bpw / PBRE_REF_BPW if tot_words else 0.0,
        "strategy_counts": strat_counts,
        "mode_block_counts": mode_sum,
        "wall_s": wall,
        "tensors": rows,
    }


def format_markdown(report: dict) -> str:
    lines = [
        "# Adaptive 7-bit mantissa codec (real BF16)",
        "",
        DISCLAIMER,
        "",
        f"Model dir: `{report.get('model_dir', '')}`. "
        f"Tensors={report.get('n_tensors')}  words={report.get('n_words')}.",
        "",
        "The guide PDF was not on this agent VM; `predict()` is documented in "
        "`pbr_adaptive_mantissa/predict.py` (copy-prev, local pos, ramp, exp, xor, avg).",
        "",
        "## Aggregate",
        "",
        f"- Raw BF16: {_bytes_human(report.get('raw_bytes') or 0)} (**16.000 BPW**)",
        f"- Adaptive mantissa field: {_bytes_human(report.get('mantissa_bytes') or 0)} "
        f"(**{report.get('mantissa_bpw', 0):.4f} mantissa BPW**)",
        f"- Sign raw: {_bytes_human(report.get('sign_bytes') or 0)}",
        f"- Exp rANS: {_bytes_human(report.get('exp_bytes') or 0)}",
        f"- Complete bundle: {_bytes_human(report.get('bundle_bytes') or 0)} "
        f"(**{report.get('total_bpw', 0):.4f} total BPW**)",
        f"- vs raw 16: **{report.get('vs_raw_16', 0):.3f}×**",
        f"- vs PBR-E ~{PBRE_REF_BPW:.2f}: **{report.get('vs_pbre_10_6', 0):.3f}×** "
        f"(same complete-byte discipline; PBR-E is sign+mantissa raw + exp entropy).",
        f"- Tensor strategies: {report.get('strategy_counts')}",
        f"- Winning MIXED/ALL block modes: {report.get('mode_block_counts')}",
        f"- Wall: {report.get('wall_s', 0):.2f}s",
        "",
    ]
    mb = float(report.get("mantissa_bpw") or 0)
    if mb >= 6.9:
        lines += [
            "**Honesty:** adaptive mantissa stays ~7 BPW (mostly RAW). That matches "
            "Phase A: real Qwen mantissas are not a compact-context win. This is not "
            "≤4 or ≤8 **total** BPW.",
            "",
        ]
    elif mb < 6.5:
        lines += [
            f"Mantissa complete BPW {mb:.4f} is below the Phase A 6.5 gate on this run.",
            "",
        ]
    syn = report.get("synthetic") or {}
    if syn.get("cases"):
        lines += ["## Synthetic sanity", "", "| case | strategy | mantissa BPW | total BPW | Huffman table | exact |", "| --- | --- | ---: | ---: | --- | --- |"]
        for name, row in syn["cases"].items():
            lines.append(
                f"| `{name}` | {row['strategy']} | {row['mantissa_bpw']:.3f} | "
                f"{row['total_bpw']:.3f} | {row['used_huffman_table']} | {row['exact']} |"
            )
        lines.append("")
    lines += [
        "## Per tensor",
        "",
        "| tensor | words | strategy | mant BPW | total BPW | huff table B | exact |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("tensors") or []:
        lines.append(
            f"| `{row['name']}` | {row['n_words']} | {row['strategy']} | "
            f"{row['mantissa_bpw']:.3f} | {row['total_bpw']:.3f} | "
            f"{row['codebook_bytes']} | {row['exact']} |"
        )
    lines += [
        "",
        "SHA-256 is over restored uint16 words vs the Safetensors source. "
        "No FP32 field extract.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive 7-bit mantissa codec PoC.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tensors", type=int, default=0, help="0 = all 16-bit tensors")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/adaptive_mantissa_qwen.json"))
    parser.add_argument("--md-out", type=Path, default=Path("artifacts/adaptive_mantissa_qwen.md"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args(argv)
    print(DISCLAIMER, flush=True)
    syn = synthetic_report()
    print("synthetic:", {k: (v["strategy"], round(v["mantissa_bpw"], 3)) for k, v in syn["cases"].items()}, flush=True)
    yaml_meta = {}
    if args.config.is_file():
        yaml_meta = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    report: dict = {
        "disclaimer": DISCLAIMER,
        "repo_id": yaml_meta.get("repo", PRIMARY_REPO),
        "revision": yaml_meta.get("revision", ""),
        "license": yaml_meta.get("license", PRIMARY_LICENSE),
        "synthetic": syn,
        "predict_rules": RULE_NAMES,
        "pdf_note": (
            "User PDF was not readable on this VM; codec follows the user brief. "
            "predict() documented in pbr_adaptive_mantissa/predict.py."
        ),
    }
    if not args.skip_model:
        if not args.model_dir.is_dir():
            raise SystemExit(f"missing model dir {args.model_dir}")
        model = encode_model(args.model_dir, max_tensors=args.max_tensors, block_size=args.block_size)
        report.update(model)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(report)
    args.md_out.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {args.md_out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
