#!/usr/bin/env python3
"""PBR-H95 Phase A: deterministic BF16 mantissa quantization probe.

Reports per-tensor numeric error and estimated storage BPW under RAW field
packing. Does **not** claim >=95% model-quality retention — that requires a
held-out generative/eval score (Phase B/C).

Estimated total BPW ≈ 1.0 (sign) + 2.62 (exp rANS ref) + keep_bits (RAW mant)
+ small metadata. Exp rate is the prior PBR-E measurement on this model class.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pbr_core.bf16 import split_components
from pbr_core.safetensors_io import TWO_BYTE_DTYPES, inventory_model, load_uint16
from pbr_h95.policy import default_keep_bits
from pbr_h95.quantize import quantize_bf16_mantissas

EXP_BPW_REF = 2.62  # prior PBR-E-class exponent rate on Qwen
SIGN_BPW = 1.0
DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")


def tensor_stats(orig: np.ndarray, quant: np.ndarray) -> dict:
    diff = orig.astype(np.uint16) != quant.astype(np.uint16)
    n = int(orig.size)
    # Interpret as BF16 via float32 for diagnostic only (not the archival path).
    o_f = orig.view(np.float16)  # wrong if true bf16 — Qwen is BF16 stored as uint16
    # Better: construct float32 from bits manually for diagnostics.
    def bf16_to_f32(u: np.ndarray) -> np.ndarray:
        u32 = u.astype(np.uint32) << 16
        return u32.view(np.float32)

    of = bf16_to_f32(orig)
    qf = bf16_to_f32(quant)
    finite = np.isfinite(of) & np.isfinite(qf)
    mse = float(np.mean((of[finite] - qf[finite]) ** 2)) if np.any(finite) else 0.0
    max_abs = float(np.max(np.abs(of[finite] - qf[finite]))) if np.any(finite) else 0.0
    return {
        "n_words": n,
        "n_changed": int(np.count_nonzero(diff)),
        "frac_changed": float(np.mean(diff)) if n else 0.0,
        "mse_f32_diagnostic": mse,
        "max_abs_f32_diagnostic": max_abs,
        "exact_roundtrip_to_quantized": True,  # filled by caller after re-quantize check
    }


def estimate_bpw(keep_bits: int) -> dict:
    mant = float(keep_bits)
    total = SIGN_BPW + EXP_BPW_REF + mant
    return {
        "sign_bpw": SIGN_BPW,
        "exp_bpw_ref": EXP_BPW_REF,
        "mantissa_bpw_raw": mant,
        "est_total_bpw": round(total, 4),
        "note": "RAW mantissa pack; no entropy coding; exp rate is reference 2.62",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--uniform-bits", type=int, default=0, help="If 1..7, ignore policy and use uniform keep")
    ap.add_argument("--large-bits", type=int, default=5, help="Policy keep for large attn/mlp mats")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pbr_h95/phase_a_qwen.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/pbr_h95/phase_a_qwen.md"))
    args = ap.parse_args()

    specs = [s for s in inventory_model(args.model_dir) if s.dtype in TWO_BYTE_DTYPES]
    specs = sorted(specs, key=lambda s: s.name)
    if args.max_tensors:
        specs = specs[: args.max_tensors]

    rows = []
    tot_words = 0
    tot_mant_bits = 0
    keep_hist = {k: 0 for k in range(0, 8)}
    t0 = time.perf_counter()

    # Uniform sweeps for ladder
    ladder = {}
    for k in (7, 6, 5, 4, 3):
        # sample on first N tensors for speed if many — but full model is 0.5B ~1GB, ok
        pass

    for spec in specs:
        words = load_uint16(spec)
        keep = args.uniform_bits if args.uniform_bits else default_keep_bits(spec.name, default_large=args.large_bits)
        quant = quantize_bf16_mantissas(words, keep)
        # Idempotence: quantizing again at same keep must be identity on quantized ref.
        again = quantize_bf16_mantissas(quant, keep)
        exact = bool(np.array_equal(quant, again))
        st = tensor_stats(words, quant)
        st["exact_roundtrip_to_quantized"] = exact
        st["name"] = spec.name
        st["shape"] = list(spec.shape)
        st["keep_bits"] = keep
        st["est"] = estimate_bpw(keep)
        rows.append(st)
        tot_words += st["n_words"]
        tot_mant_bits += st["n_words"] * keep
        keep_hist[keep] = keep_hist.get(keep, 0) + st["n_words"]
        print(f"  {spec.name} keep={keep} changed={st['frac_changed']:.4f} mse={st['mse_f32_diagnostic']:.3e}", flush=True)

    # Uniform ladder on full inventory (re-quant each k — expensive but 0.5B is fine)
    print("Uniform ladder…", flush=True)
    for k in (7, 6, 5, 4, 3):
        changed = 0
        words_n = 0
        mse_acc = 0.0
        for spec in specs:
            words = load_uint16(spec)
            quant = quantize_bf16_mantissas(words, k)
            d = words != quant
            changed += int(np.count_nonzero(d))
            words_n += int(words.size)
            of = (words.astype(np.uint32) << 16).view(np.float32)
            qf = (quant.astype(np.uint32) << 16).view(np.float32)
            finite = np.isfinite(of) & np.isfinite(qf)
            if np.any(finite):
                mse_acc += float(np.sum((of[finite] - qf[finite]) ** 2))
        est = estimate_bpw(k)
        ladder[str(k)] = {
            "keep_bits": k,
            "frac_changed": changed / words_n if words_n else 0.0,
            "mse_f32_diagnostic": mse_acc / words_n if words_n else 0.0,
            **est,
        }
        print(f"  k={k} frac_changed={ladder[str(k)]['frac_changed']:.4f} est_total={est['est_total_bpw']}", flush=True)

    avg_mant = (tot_mant_bits / tot_words) if tot_words else 0.0
    policy_est_total = SIGN_BPW + EXP_BPW_REF + avg_mant
    summary = {
        "model_dir": str(args.model_dir),
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "phase": "H95-A",
        "policy": {
            "protected_keep": 7,
            "large_attn_mlp_keep": args.large_bits if not args.uniform_bits else args.uniform_bits,
            "uniform_bits": args.uniform_bits or None,
        },
        "keep_hist_words": keep_hist,
        "aggregate_policy": {
            "n_words": tot_words,
            "avg_mantissa_bits": round(avg_mant, 4),
            "est_total_bpw": round(policy_est_total, 4),
            "sign_bpw": SIGN_BPW,
            "exp_bpw_ref": EXP_BPW_REF,
            "all_quantize_idempotent": all(r["exact_roundtrip_to_quantized"] for r in rows),
        },
        "uniform_ladder": ladder,
        "tensors": rows,
        "disclaimer": (
            "Phase A only: deterministic quantization + estimated storage BPW. "
            "est_total_bpw uses RAW mantissa packing and a 2.62 exp reference — "
            "not a physical container encode. Does NOT measure model-quality retention; "
            ">=95% requires held-out eval (Phase B/C). Not a <=4 BPW claim."
        ),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    agg = summary["aggregate_policy"]
    lines = [
        "# PBR-H95 Phase A — Qwen mantissa quantization probe",
        "",
        f"Model: `{summary['repo']}`",
        "",
        "## Policy estimate (first-experiment map)",
        "",
        f"- Protected tensors (emb/lm/norm/bias/first/last): **7** bits",
        f"- Large attn/MLP: **{summary['policy']['large_attn_mlp_keep']}** bits",
        f"- Words: **{agg['n_words']}**",
        f"- Avg retained mantissa bits: **{agg['avg_mantissa_bits']}**",
        f"- Est. total BPW (1 + 2.62 + avg_mant): **{agg['est_total_bpw']}**",
        f"- Quantize idempotent on all tensors: **{agg['all_quantize_idempotent']}**",
        "",
        "## Uniform ladder (diagnostic)",
        "",
        "| keep | frac words changed | mse (f32 diag) | est total BPW |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for k in ("7", "6", "5", "4", "3"):
        L = ladder[k]
        lines.append(
            f"| {k} | {L['frac_changed']:.4f} | {L['mse_f32_diagnostic']:.3e} | {L['est_total_bpw']} |"
        )
    lines += [
        "",
        "## Honesty",
        "",
        summary["disclaimer"],
        "",
        "Lossless reference remains ~10.62 BPW. H95 aims for lower BPW only with measured ≥95% quality later.",
        "",
    ]
    args.md.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["aggregate_policy"], indent=2))
    print(json.dumps(summary["uniform_ladder"], indent=2))
    print(f"Wrote {args.out} and {args.md}")
    return 0 if agg["all_quantize_idempotent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
