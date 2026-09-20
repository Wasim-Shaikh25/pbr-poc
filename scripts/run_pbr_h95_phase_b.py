#!/usr/bin/env python3
"""PBR-H95 Phase B: layer/tensor-family sensitivity on calibration NLL/PPL.

Measures bytes-saved vs quality-loss utility on a fixed in-repo calibration
corpus (calib-v1). This is a **calibration proxy only** — not held-out eval.
Phase C owns the ≥95% held-out quality gate. Do NOT treat ppl_retention as a GO.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os

import torch

# Stable CPU calibration: multi-thread GEMM can nudge bf16 reductions enough to
# flip tiny delta_nll signs on a short calib set.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pbr_h95.apply_policy import apply_policy_to_model, clone_cpu_state_dict
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_h95.policy import (
    FAMILIES,
    default_keep_bits,
    family_ablation_keep_fn,
    policy_then_mlp_mid_keep_fn,
    tensor_family,
    uniform_mid_keep_fn,
)

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v1.json")
CALIB_VERSION = "calib-v1"

CALIB_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "Machine learning models compress patterns in data into reusable parameters.",
    "In quantum mechanics, measurement can change the state of a physical system.",
    "A library is a collection of books, manuscripts, and other sources of information.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The committee agreed to postpone the vote until next Tuesday afternoon.",
    "Software engineers write tests to catch regressions before code is deployed.",
    "Climate scientists study long-term changes in temperature, rainfall, and ice cover.",
    "Bread is typically made from flour, water, yeast, and a pinch of salt.",
    "The telescope revealed faint galaxies that formed billions of years ago.",
    "Economic forecasts depend on assumptions about interest rates and employment.",
    "A sonnet is a fourteen-line poem with a structured rhyme scheme and meter.",
]

HONESTY = (
    "CALIBRATION PROXY ONLY — not held-out evaluation. Scores use the fixed "
    "in-repo corpus calib-v1 under teacher-forcing mean token NLL / perplexity. "
    "Phase C owns the ≥95% held-out quality gate. Do NOT claim ≥95% GO from Phase B. "
    "ppl_retention = bf16_ppl / quant_ppl is reported for ranking, not acceptance."
)


def write_calibration(path: Path) -> dict:
    payload = {
        "version": CALIB_VERSION,
        "id": "pbr-h95-calibration-v1",
        "language": "en",
        "n_texts": len(CALIB_TEXTS),
        "texts": [{"id": f"c{i:02d}", "text": t} for i, t in enumerate(CALIB_TEXTS)],
        "note": (
            "Fixed deterministic calibration corpus for H95 Phase B sensitivity. "
            "Not a held-out eval set."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score_candidate(
    model,
    tokenizer,
    texts,
    keep_fn,
    *,
    baseline_sd,
    baseline_nll: float,
    baseline_ppl: float,
    max_length: int,
    name: str,
    kind: str,
) -> dict:
    t0 = time.perf_counter()
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    metrics = mean_token_nll(model, tokenizer, texts, max_length=max_length)
    delta_nll = metrics["mean_nll"] - baseline_nll
    bytes_saved = meta["bytes_saved_vs_bf16_mantissa"]
    utility = bytes_saved / max(abs(delta_nll), 1e-9)
    # also report signed utility with raw delta (positive delta = worse)
    utility_signed = bytes_saved / max(delta_nll, 1e-9)
    row = {
        "name": name,
        "kind": kind,
        "mean_nll": metrics["mean_nll"],
        "ppl": metrics["ppl"],
        "n_tokens": metrics["n_tokens"],
        "delta_nll": delta_nll,
        "ppl_retention": ppl_retention(baseline_ppl, metrics["ppl"]),
        "bytes_saved_vs_bf16_mantissa": bytes_saved,
        "utility_bytes_per_delta_nll": utility_signed,
        "utility_bytes_per_abs_delta_nll": utility,
        "keep_hist": _keep_hist(meta["keep_map"]),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    print(
        f"  [{name}] nll={row['mean_nll']:.6f} ppl={row['ppl']:.4f} "
        f"Δnll={row['delta_nll']:+.6f} ret={row['ppl_retention']:.4f} "
        f"bytes_saved={bytes_saved:.0f} util={utility_signed:.2f} "
        f"({row['wall_seconds']}s)",
        flush=True,
    )
    return row


def _keep_hist(keep_map: dict[str, int]) -> dict[str, int]:
    hist = {str(k): 0 for k in range(8)}
    for v in keep_map.values():
        hist[str(v)] = hist.get(str(v), 0) + 1
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-texts", type=int, default=0, help="If >0, truncate calib set")
    ap.add_argument("--smoke", action="store_true", help="One forward + one policy only")
    ap.add_argument("--out", type=Path, default=Path("artifacts/pbr_h95/phase_b_qwen.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/pbr_h95/phase_b_qwen.md"))
    args = ap.parse_args()

    wall0 = time.perf_counter()
    calib = write_calibration(CALIB_PATH)
    texts = [t["text"] for t in calib["texts"]]
    if args.max_texts > 0:
        texts = texts[: args.max_texts]

    print(f"Loading model from {args.model_dir} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    print("Cloning BF16 baseline state_dict …", flush=True)
    baseline_sd = clone_cpu_state_dict(model)

    # family coverage check
    fam_counts: dict[str, int] = {}
    for name in baseline_sd:
        fam = tensor_family(name) or "other"
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print("Family tensor counts:", fam_counts, flush=True)

    print("BF16 baseline NLL …", flush=True)
    t0 = time.perf_counter()
    base = mean_token_nll(model, tokenizer, texts, max_length=args.max_length)
    base_wall = round(time.perf_counter() - t0, 2)
    print(
        f"  baseline mean_nll={base['mean_nll']:.6f} ppl={base['ppl']:.4f} "
        f"tokens={base['n_tokens']} ({base_wall}s)",
        flush=True,
    )

    if args.smoke:
        print("Smoke: policy 7/5 …", flush=True)
        score_candidate(
            model,
            tokenizer,
            texts,
            lambda n: default_keep_bits(n, default_large=5),
            baseline_sd=baseline_sd,
            baseline_nll=base["mean_nll"],
            baseline_ppl=base["ppl"],
            max_length=args.max_length,
            name="policy_7_5",
            kind="policy",
        )
        print("Smoke OK", flush=True)
        return 0

    candidates: list[dict] = []

    # a) full first-experiment policy
    print("Candidate: policy 7/5 …", flush=True)
    candidates.append(
        score_candidate(
            model,
            tokenizer,
            texts,
            lambda n: default_keep_bits(n, default_large=5),
            baseline_sd=baseline_sd,
            baseline_nll=base["mean_nll"],
            baseline_ppl=base["ppl"],
            max_length=args.max_length,
            name="policy_7_5",
            kind="policy",
        )
    )

    # b) uniform mid-layer keep
    for k in (6, 5, 4):
        print(f"Candidate: uniform_mid_k{k} …", flush=True)
        candidates.append(
            score_candidate(
                model,
                tokenizer,
                texts,
                uniform_mid_keep_fn(k),
                baseline_sd=baseline_sd,
                baseline_nll=base["mean_nll"],
                baseline_ppl=base["ppl"],
                max_length=args.max_length,
                name=f"uniform_mid_k{k}",
                kind="uniform_mid",
            )
        )

    # c) family ablations
    family_rows = []
    for fam in FAMILIES:
        # lm_head_tied is alias of embed under tie_word_embeddings — still run once labeled
        print(f"Candidate: family_ablate_{fam}_k5 …", flush=True)
        row = score_candidate(
            model,
            tokenizer,
            texts,
            family_ablation_keep_fn(fam, family_keep=5),
            baseline_sd=baseline_sd,
            baseline_nll=base["mean_nll"],
            baseline_ppl=base["ppl"],
            max_length=args.max_length,
            name=f"family_{fam}_k5",
            kind="family_ablation",
        )
        row["family"] = fam
        if fam == "lm_head_tied":
            row["note"] = (
                "tie_word_embeddings=true: lm_head shares embed_tokens; "
                "ablation identical to family_embed_k5"
            )
        candidates.append(row)
        family_rows.append(row)

    # d) optional: policy then mlp_mid→4
    print("Candidate: policy_7_5_mlp_mid_k4 …", flush=True)
    candidates.append(
        score_candidate(
            model,
            tokenizer,
            texts,
            policy_then_mlp_mid_keep_fn(4, base_large=5),
            baseline_sd=baseline_sd,
            baseline_nll=base["mean_nll"],
            baseline_ppl=base["ppl"],
            max_length=args.max_length,
            name="policy_7_5_mlp_mid_k4",
            kind="policy_variant",
        )
    )

    # Rankings from family ablations (unique families; skip duplicate lm_head_tied for ranking)
    rank_src = [r for r in family_rows if r["family"] != "lm_head_tied"]
    by_delta = sorted(rank_src, key=lambda r: r["delta_nll"], reverse=True)
    by_util = sorted(rank_src, key=lambda r: r["utility_bytes_per_delta_nll"], reverse=True)
    # Prefer cost-efficiency among families that actually lose quality (delta_nll > 0).
    pos = [r for r in rank_src if r["delta_nll"] > 0]
    by_util_pos = sorted(pos, key=lambda r: r["utility_bytes_per_delta_nll"], reverse=True)

    policy_row = next(r for r in candidates if r["name"] == "policy_7_5")
    summary = {
        "phase": "H95-B",
        "model_dir": str(args.model_dir),
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "tie_word_embeddings": True,
        "calibration": {
            "version": CALIB_VERSION,
            "path": str(CALIB_PATH),
            "n_texts": len(texts),
            "max_length": args.max_length,
            "metric": "mean_token_nll_teacher_forcing",
            "ppl": "exp(mean_nll)",
            "ppl_retention": "bf16_ppl / quant_ppl",
        },
        "baseline": {
            "mean_nll": base["mean_nll"],
            "ppl": base["ppl"],
            "n_tokens": base["n_tokens"],
            "wall_seconds": base_wall,
        },
        "family_tensor_counts": fam_counts,
        "candidates": candidates,
        "rankings": {
            "families_by_delta_nll_desc": [
                {"family": r["family"], "delta_nll": r["delta_nll"], "bytes_saved": r["bytes_saved_vs_bf16_mantissa"]}
                for r in by_delta
            ],
            "families_by_utility_desc": [
                {
                    "family": r["family"],
                    "utility_bytes_per_delta_nll": r["utility_bytes_per_delta_nll"],
                    "delta_nll": r["delta_nll"],
                    "bytes_saved": r["bytes_saved_vs_bf16_mantissa"],
                }
                for r in by_util
            ],
            "families_by_utility_among_positive_delta_nll_desc": [
                {
                    "family": r["family"],
                    "utility_bytes_per_delta_nll": r["utility_bytes_per_delta_nll"],
                    "delta_nll": r["delta_nll"],
                    "bytes_saved": r["bytes_saved_vs_bf16_mantissa"],
                }
                for r in by_util_pos
            ],
            "note": (
                "utility = bytes_saved / max(delta_nll, 1e-9). "
                "delta_nll <= 0 yields huge utility (no measured calib loss); "
                "also see families_by_utility_among_positive_delta_nll_desc."
            ),
        },
        "policy_7_5_summary": {
            "ppl_retention": policy_row["ppl_retention"],
            "delta_nll": policy_row["delta_nll"],
            "bytes_saved_vs_bf16_mantissa": policy_row["bytes_saved_vs_bf16_mantissa"],
        },
        "disclaimer": HONESTY,
        "wall_seconds": round(time.perf_counter() - wall0, 2),
        "env": {
            "torch": torch.__version__,
            "device": str(device),
            "dtype": "bfloat16",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# PBR-H95 Phase B — Layer/tensor sensitivity (Qwen)",
        "",
        f"Model: `{summary['repo']}` (local BF16, CPU)",
        f"Calibration: **{CALIB_VERSION}** (`{CALIB_PATH}`), max_length={args.max_length}, n_texts={len(texts)}",
        "",
        "## Honesty",
        "",
        HONESTY,
        "",
        "## Baseline (BF16)",
        "",
        f"- mean_nll: **{base['mean_nll']:.6f}**",
        f"- ppl (=exp(mean_nll)): **{base['ppl']:.4f}**",
        f"- tokens scored: **{base['n_tokens']}**",
        "",
        "## Candidates",
        "",
        "| name | mean_nll | ppl | Δnll | ppl_retention | bytes_saved | utility (B/Δnll) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in candidates:
        lines.append(
            f"| {r['name']} | {r['mean_nll']:.6f} | {r['ppl']:.4f} | {r['delta_nll']:+.6f} | "
            f"{r['ppl_retention']:.4f} | {r['bytes_saved_vs_bf16_mantissa']:.0f} | "
            f"{r['utility_bytes_per_delta_nll']:.2f} |"
        )
    lines += [
        "",
        "### Policy 7/5",
        "",
        f"- ppl_retention: **{policy_row['ppl_retention']:.4f}** (calibration proxy — not a ≥95% GO)",
        f"- delta_nll: **{policy_row['delta_nll']:+.6f}**",
        f"- mantissa bytes saved vs BF16: **{policy_row['bytes_saved_vs_bf16_mantissa']:.0f}**",
        "",
        "### Top-3 sensitive families by Δnll (higher = more quality loss)",
        "",
    ]
    for r in by_delta[:3]:
        lines.append(f"- **{r['family']}**: Δnll={r['delta_nll']:+.6f}, bytes_saved={r['bytes_saved_vs_bf16_mantissa']:.0f}")
    lines += [
        "",
        "### Top-3 families by utility (bytes_saved / max(Δnll, 1e-9))",
        "",
        "_Note: Δnll≤0 ⇒ huge utility (no measured calib loss on this proxy)._",
        "",
    ]
    for r in by_util[:3]:
        lines.append(
            f"- **{r['family']}**: util={r['utility_bytes_per_delta_nll']:.2f}, "
            f"Δnll={r['delta_nll']:+.6f}, bytes_saved={r['bytes_saved_vs_bf16_mantissa']:.0f}"
        )
    lines += [
        "",
        "### Top-3 by utility among families with Δnll > 0",
        "",
    ]
    if not by_util_pos:
        lines.append("- _(none — no family showed positive Δnll on calib-v1)_")
    for r in by_util_pos[:3]:
        lines.append(
            f"- **{r['family']}**: util={r['utility_bytes_per_delta_nll']:.2f}, "
            f"Δnll={r['delta_nll']:+.6f}, bytes_saved={r['bytes_saved_vs_bf16_mantissa']:.0f}"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `tie_word_embeddings=true`: `lm_head_tied` ablation is identical to `embed` (same storage).",
        "- CPU eval pinned to 1 thread for stable calibration deltas.",
        "- `bytes_saved` = Σ (7−keep)×n_words/8 (mantissa bits only vs BF16).",
        "- Protected tensors (emb/norm/bias/first/last) remain at keep=7 in policy and uniform-mid maps.",
        f"- Wall time: **{summary['wall_seconds']}s**",
        "",
    ]
    args.md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out} and {args.md}", flush=True)
    print(f"Total wall: {summary['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
