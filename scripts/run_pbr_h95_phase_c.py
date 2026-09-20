#!/usr/bin/env python3
"""PBR-H95 Phase C: greedy precision allocation + held-out quality gate.

Search uses calib-v2 only. Final GO/NO-GO uses heldout-v1 ppl_retention ≥ 0.95.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pbr_h95.apply_policy import (
    apply_policy_to_model,
    clone_cpu_state_dict,
    estimate_mantissa_bytes_saved,
)
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_h95.policy import (
    BAND_SPECS,
    FAMILY_UNIT_SPECS,
    avg_keep_bits,
    default_keep_bits,
    uniform_mid_keep_fn,
    unit_keep_fn,
)

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")

HONESTY_LINES = [
    "Held-out set is small/in-repo — **not** a production LM benchmark (no MMLU/HellaSwag/etc.).",
    "ppl_retention on heldout-v1 is a **proxy** quality score for this PoC.",
    "Claim GO only if heldout ppl_retention ≥ 0.95 AND map is reported; if <0.95, say NO-GO / needs restore.",
    "Not a physical container BPW; est only.",
    "Not ≤4 BPW product claim.",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    texts = [t["text"] for t in payload["texts"]]
    return payload, texts


def keep_hist(keep_map: dict[str, int]) -> dict[str, int]:
    hist = {str(k): 0 for k in range(8)}
    for v in keep_map.values():
        hist[str(v)] = hist.get(str(v), 0) + 1
    return hist


def est_bpw(avg_keep: float) -> float:
    """RAW pack estimate: 1 sign + 2.62 exp ref + avg mantissa keep."""
    return 1.0 + 2.62 + float(avg_keep)


def eval_keep_fn(
    model,
    tokenizer,
    texts,
    keep_fn,
    *,
    baseline_sd,
    baseline_nll: float,
    baseline_ppl: float,
    max_length: int,
    label: str,
) -> dict:
    t0 = time.perf_counter()
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    metrics = mean_token_nll(model, tokenizer, texts, max_length=max_length)
    delta_nll = metrics["mean_nll"] - baseline_nll
    bytes_saved = meta["bytes_saved_vs_bf16_mantissa"]
    avg_k = avg_keep_bits(meta["keep_map"], baseline_sd)
    row = {
        "name": label,
        "mean_nll": metrics["mean_nll"],
        "ppl": metrics["ppl"],
        "n_tokens": metrics["n_tokens"],
        "delta_nll": delta_nll,
        "ppl_retention": ppl_retention(baseline_ppl, metrics["ppl"]),
        "bytes_saved_vs_bf16_mantissa": bytes_saved,
        "avg_keep_bits": avg_k,
        "est_total_bpw": est_bpw(avg_k),
        "keep_hist": keep_hist(meta["keep_map"]),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    print(
        f"  [{label}] nll={row['mean_nll']:.6f} ppl={row['ppl']:.4f} "
        f"Δnll={row['delta_nll']:+.6f} ret={row['ppl_retention']:.4f} "
        f"bytes_saved={bytes_saved:.0f} bpw≈{row['est_total_bpw']:.2f} "
        f"({row['wall_seconds']}s)",
        flush=True,
    )
    return row, meta["keep_map"]


def summarize_unit_map(unit_keeps: dict[str, int]) -> dict[str, int]:
    return dict(sorted(unit_keeps.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--units",
        choices=("bands", "family"),
        default="bands",
        help="Decision units for greedy search (fallback to family if too slow)",
    )
    ap.add_argument(
        "--time-budget-sec",
        type=float,
        default=45 * 60,
        help="If search wall exceeds this, fall back to family units (when starting from bands)",
    )
    ap.add_argument("--calib-floor", type=float, default=0.97)
    ap.add_argument("--soft-calib-floor", type=float, default=0.98)
    ap.add_argument("--heldout-gate", type=float, default=0.95)
    ap.add_argument("--candidate-keeps", type=int, nargs="+", default=[6, 5, 4, 3])
    ap.add_argument("--out", type=Path, default=Path("artifacts/pbr_h95/phase_c_qwen.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/pbr_h95/phase_c_qwen.md"))
    args = ap.parse_args()

    wall0 = time.perf_counter()
    calib_meta, calib_texts = load_texts(args.calib)
    held_meta, held_texts = load_texts(args.heldout)

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

    unit_mode = args.units
    unit_specs = BAND_SPECS if unit_mode == "bands" else FAMILY_UNIT_SPECS
    shortcuts: list[str] = []

    def run_search(unit_specs, unit_mode: str):
        unit_names = [u[0] for u in unit_specs]
        current = {u: 7 for u in unit_names}
        steps: list[dict] = []
        search_t0 = time.perf_counter()

        print("BF16 baseline on calib …", flush=True)
        t0 = time.perf_counter()
        # all-7 = identity (no quant)
        base_calib = mean_token_nll(model, tokenizer, calib_texts, max_length=args.max_length)
        base_calib_wall = round(time.perf_counter() - t0, 2)
        print(
            f"  calib baseline mean_nll={base_calib['mean_nll']:.6f} "
            f"ppl={base_calib['ppl']:.4f} tokens={base_calib['n_tokens']} "
            f"({base_calib_wall}s)",
            flush=True,
        )

        print("BF16 baseline on heldout …", flush=True)
        t0 = time.perf_counter()
        base_held = mean_token_nll(model, tokenizer, held_texts, max_length=args.max_length)
        base_held_wall = round(time.perf_counter() - t0, 2)
        print(
            f"  heldout baseline mean_nll={base_held['mean_nll']:.6f} "
            f"ppl={base_held['ppl']:.4f} tokens={base_held['n_tokens']} "
            f"({base_held_wall}s)",
            flush=True,
        )

        step_i = 0
        while True:
            if (
                unit_mode == "bands"
                and (time.perf_counter() - search_t0) > args.time_budget_sec * 0.85
                and not steps
            ):
                # early: no accept yet and already near budget → caller may restart family
                return None, {
                    "reason": "time_budget_before_first_accept",
                    "elapsed": time.perf_counter() - search_t0,
                }

            trials = []
            for uname in unit_names:
                cur_k = current[uname]
                # Stepwise: only the next lower candidate keep (6 then 5 then 4 then 3).
                # Trying all jumps in one round explodes CPU cost on band units.
                lower = [k for k in args.candidate_keeps if k < cur_k]
                if not lower:
                    continue
                next_ks = [max(lower)]  # largest k still below current (= one step down)
                for k in next_ks:
                    trial_keeps = dict(current)
                    trial_keeps[uname] = k
                    keep_fn = unit_keep_fn(trial_keeps, unit_specs)
                    label = f"trial_step{step_i}_{uname}_k{k}"
                    print(f"Trial: {uname} {cur_k}→{k} …", flush=True)
                    row, _ = eval_keep_fn(
                        model,
                        tokenizer,
                        calib_texts,
                        keep_fn,
                        baseline_sd=baseline_sd,
                        baseline_nll=base_calib["mean_nll"],
                        baseline_ppl=base_calib["ppl"],
                        max_length=args.max_length,
                        label=label,
                    )
                    row["unit"] = uname
                    row["from_keep"] = cur_k
                    row["to_keep"] = k
                    row["unit_keeps"] = summarize_unit_map(trial_keeps)
                    delta = row["delta_nll"]
                    if delta > 0:
                        utility = row["bytes_saved_vs_bf16_mantissa"] / max(delta, 1e-6)
                    else:
                        utility = None  # prefer positive-delta ranking; handle below
                    row["utility"] = utility
                    trials.append(row)

                    if (
                        unit_mode == "bands"
                        and (time.perf_counter() - wall0) > args.time_budget_sec
                    ):
                        shortcuts.append(
                            f"Exceeded {args.time_budget_sec/60:.0f} min wall during band search; "
                            "falling back to family-level units."
                        )
                        return None, {"reason": "time_budget", "elapsed": time.perf_counter() - wall0}

            # Rank trials
            passing = [t for t in trials if t["ppl_retention"] >= args.calib_floor]
            if not passing:
                print(
                    f"No trial meets calib floor {args.calib_floor}; stopping search.",
                    flush=True,
                )
                break

            pos = [t for t in passing if t["delta_nll"] > 0 and t["utility"] is not None]
            if pos:
                best = max(pos, key=lambda t: t["utility"])
            else:
                # all deltas ≤ 0 (or tiny): among soft floor, pick largest bytes_saved
                soft = [t for t in passing if t["ppl_retention"] >= args.soft_calib_floor]
                pool = soft if soft else passing
                best = max(pool, key=lambda t: t["bytes_saved_vs_bf16_mantissa"])
                print(
                    "All passing trials have Δnll≤0; picking largest bytes_saved "
                    f"among soft-floor≥{args.soft_calib_floor} (or floor).",
                    flush=True,
                )

            current[best["unit"]] = best["to_keep"]
            step_i += 1
            step_rec = {
                "step": step_i,
                "accepted": {
                    "unit": best["unit"],
                    "from_keep": best["from_keep"],
                    "to_keep": best["to_keep"],
                    "calib_ppl_retention": best["ppl_retention"],
                    "delta_nll": best["delta_nll"],
                    "bytes_saved_vs_bf16_mantissa": best["bytes_saved_vs_bf16_mantissa"],
                    "utility": best["utility"],
                },
                "unit_keeps_after": summarize_unit_map(current),
                "n_trials": len(trials),
            }
            steps.append(step_rec)
            print(
                f"ACCEPT step {step_i}: {best['unit']} → k={best['to_keep']} "
                f"ret={best['ppl_retention']:.4f} bytes={best['bytes_saved_vs_bf16_mantissa']:.0f}",
                flush=True,
            )

            # If every unit already at min candidate keep, stop
            min_k = min(args.candidate_keeps)
            if all(current[u] <= min_k for u in unit_names):
                print("All units at minimum candidate keep; stopping.", flush=True)
                break

        return {
            "unit_mode": unit_mode,
            "unit_specs": [u[0] for u in unit_specs],
            "frozen_unit_keeps": summarize_unit_map(current),
            "steps": steps,
            "baseline_calib": {
                "mean_nll": base_calib["mean_nll"],
                "ppl": base_calib["ppl"],
                "n_tokens": base_calib["n_tokens"],
                "wall_seconds": base_calib_wall,
            },
            "baseline_heldout": {
                "mean_nll": base_held["mean_nll"],
                "ppl": base_held["ppl"],
                "n_tokens": base_held["n_tokens"],
                "wall_seconds": base_held_wall,
            },
        }, None

    search, fail = run_search(unit_specs, unit_mode)
    if search is None:
        print(f"Band search aborted ({fail}); restarting with family units.", flush=True)
        shortcuts.append(
            f"Fell back to family-level units (mlp_mid, attn_mid) after bands abort: {fail}."
        )
        unit_mode = "family"
        unit_specs = FAMILY_UNIT_SPECS
        search, fail2 = run_search(unit_specs, unit_mode)
        if search is None:
            raise RuntimeError(f"Family search also failed: {fail2}")

    frozen_keeps = search["frozen_unit_keeps"]
    frozen_fn = unit_keep_fn(frozen_keeps, unit_specs)

    print("Evaluate frozen map on calib …", flush=True)
    frozen_calib, frozen_keep_map = eval_keep_fn(
        model,
        tokenizer,
        calib_texts,
        frozen_fn,
        baseline_sd=baseline_sd,
        baseline_nll=search["baseline_calib"]["mean_nll"],
        baseline_ppl=search["baseline_calib"]["ppl"],
        max_length=args.max_length,
        label="frozen_calib",
    )

    print("Evaluate frozen map on heldout …", flush=True)
    frozen_held, _ = eval_keep_fn(
        model,
        tokenizer,
        held_texts,
        frozen_fn,
        baseline_sd=baseline_sd,
        baseline_nll=search["baseline_heldout"]["mean_nll"],
        baseline_ppl=search["baseline_heldout"]["ppl"],
        max_length=args.max_length,
        label="frozen_heldout",
    )

    # Reference maps on held-out (+ calib for Pareto)
    refs = []
    ref_defs = [
        ("policy_7_5", lambda n: default_keep_bits(n, default_large=5)),
        ("uniform_mid_k6", uniform_mid_keep_fn(6)),
        ("uniform_mid_k5", uniform_mid_keep_fn(5)),
        ("uniform_mid_k4", uniform_mid_keep_fn(4)),
    ]
    pareto = []
    for name, fn in ref_defs:
        print(f"Reference {name} on calib …", flush=True)
        c_row, _ = eval_keep_fn(
            model,
            tokenizer,
            calib_texts,
            fn,
            baseline_sd=baseline_sd,
            baseline_nll=search["baseline_calib"]["mean_nll"],
            baseline_ppl=search["baseline_calib"]["ppl"],
            max_length=args.max_length,
            label=f"{name}_calib",
        )
        print(f"Reference {name} on heldout …", flush=True)
        h_row, _ = eval_keep_fn(
            model,
            tokenizer,
            held_texts,
            fn,
            baseline_sd=baseline_sd,
            baseline_nll=search["baseline_heldout"]["mean_nll"],
            baseline_ppl=search["baseline_heldout"]["ppl"],
            max_length=args.max_length,
            label=f"{name}_heldout",
        )
        refs.append({"name": name, "calib": c_row, "heldout": h_row})
        pareto.append(
            {
                "name": name,
                "calib_ppl_retention": c_row["ppl_retention"],
                "heldout_ppl_retention": h_row["ppl_retention"],
                "bytes_saved_vs_bf16_mantissa": c_row["bytes_saved_vs_bf16_mantissa"],
                "avg_keep_bits": c_row["avg_keep_bits"],
                "est_total_bpw": c_row["est_total_bpw"],
            }
        )

    pareto.insert(
        0,
        {
            "name": "frozen_greedy",
            "calib_ppl_retention": frozen_calib["ppl_retention"],
            "heldout_ppl_retention": frozen_held["ppl_retention"],
            "bytes_saved_vs_bf16_mantissa": frozen_calib["bytes_saved_vs_bf16_mantissa"],
            "avg_keep_bits": frozen_calib["avg_keep_bits"],
            "est_total_bpw": frozen_calib["est_total_bpw"],
            "unit_keeps": frozen_keeps,
        },
    )

    held_ret = frozen_held["ppl_retention"]
    go = bool(held_ret >= args.heldout_gate)
    decision = "GO" if go else "NO-GO / needs restore"

    # Compact keep map summary: count tensors per keep, plus unit keeps
    summary = {
        "phase": "H95-C",
        "model_dir": str(args.model_dir),
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "calibration": {
            "version": calib_meta.get("version"),
            "path": str(args.calib),
            "n_texts": len(calib_texts),
            "n_tokens_scored": search["baseline_calib"]["n_tokens"],
            "max_length": args.max_length,
        },
        "heldout": {
            "version": held_meta.get("version"),
            "path": str(args.heldout),
            "n_texts": len(held_texts),
            "n_tokens_scored": search["baseline_heldout"]["n_tokens"],
            "max_length": args.max_length,
            "note": held_meta.get("note"),
        },
        "search": {
            "unit_mode": search["unit_mode"],
            "units": search["unit_specs"],
            "calib_floor": args.calib_floor,
            "soft_calib_floor": args.soft_calib_floor,
            "candidate_keeps": args.candidate_keeps,
            "steps": search["steps"],
            "frozen_unit_keeps": frozen_keeps,
        },
        "baseline_calib": search["baseline_calib"],
        "baseline_heldout": search["baseline_heldout"],
        "frozen": {
            "unit_keeps": frozen_keeps,
            "keep_hist": frozen_calib["keep_hist"],
            "calib": frozen_calib,
            "heldout": frozen_held,
        },
        "references": refs,
        "pareto": pareto,
        "gate": {
            "metric": "heldout_ppl_retention",
            "threshold": args.heldout_gate,
            "heldout_ppl_retention": held_ret,
            "calib_ppl_retention": frozen_calib["ppl_retention"],
            "decision": decision,
            "go": go,
        },
        "shortcuts": shortcuts,
        "honesty": HONESTY_LINES,
        "wall_seconds": round(time.perf_counter() - wall0, 2),
        "env": {"torch": torch.__version__, "device": "cpu", "dtype": "bfloat16"},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# PBR-H95 Phase C — Greedy precision allocation (Qwen)",
        "",
        f"Model: `{summary['repo']}` (local BF16, CPU)",
        f"Calibration (search): **{calib_meta.get('version')}** (`{args.calib}`), "
        f"tokens={search['baseline_calib']['n_tokens']}, max_length={args.max_length}",
        f"Held-out (gate): **{held_meta.get('version')}** (`{args.heldout}`), "
        f"tokens={search['baseline_heldout']['n_tokens']}, max_length={args.max_length}",
        f"Decision units: **{search['unit_mode']}** ({', '.join(search['unit_specs'])})",
        "",
        "## Honesty",
        "",
    ]
    for h in HONESTY_LINES:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Gate decision",
        "",
        f"- heldout ppl_retention: **{held_ret:.4f}** (threshold {args.heldout_gate})",
        f"- calib ppl_retention (frozen): **{frozen_calib['ppl_retention']:.4f}** (search proxy only)",
        f"- **Decision: {decision}**",
        "",
        "## Frozen map (unit keeps)",
        "",
        "Protected emb/norm/bias/first/last remain at keep=7.",
        "",
        "| unit | keep |",
        "| --- | ---: |",
    ]
    for u, k in frozen_keeps.items():
        lines.append(f"| {u} | {k} |")
    lines += [
        "",
        f"- bytes_saved vs BF16 mantissa: **{frozen_calib['bytes_saved_vs_bf16_mantissa']:.0f}**",
        f"- avg keep bits: **{frozen_calib['avg_keep_bits']:.4f}**",
        f"- est_total_bpw (=1+2.62+avg_keep): **{frozen_calib['est_total_bpw']:.2f}** (est only)",
        f"- keep_hist (tensor count): `{frozen_calib['keep_hist']}`",
        "",
        "## Greedy search steps (calib-only)",
        "",
    ]
    if not search["steps"]:
        lines.append("_No lowering step accepted (all trials failed calib floor or none tried)._")
    else:
        lines += [
            "| step | unit | keep | calib ret | Δnll | bytes_saved | utility |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in search["steps"]:
            a = s["accepted"]
            util = a["utility"]
            util_s = f"{util:.2f}" if util is not None else "n/a (Δ≤0)"
            lines.append(
                f"| {s['step']} | {a['unit']} | {a['to_keep']} | "
                f"{a['calib_ppl_retention']:.4f} | {a['delta_nll']:+.6f} | "
                f"{a['bytes_saved_vs_bf16_mantissa']:.0f} | {util_s} |"
            )
    lines += [
        "",
        "## Pareto / reference table",
        "",
        "| name | calib ret | heldout ret | bytes_saved | avg_keep | est BPW |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in pareto:
        lines.append(
            f"| {r['name']} | {r['calib_ppl_retention']:.4f} | {r['heldout_ppl_retention']:.4f} | "
            f"{r['bytes_saved_vs_bf16_mantissa']:.0f} | {r['avg_keep_bits']:.4f} | "
            f"{r['est_total_bpw']:.2f} |"
        )
    lines += [
        "",
        "## Baselines (BF16)",
        "",
        f"- calib mean_nll={search['baseline_calib']['mean_nll']:.6f}, "
        f"ppl={search['baseline_calib']['ppl']:.4f}, "
        f"tokens={search['baseline_calib']['n_tokens']}",
        f"- heldout mean_nll={search['baseline_heldout']['mean_nll']:.6f}, "
        f"ppl={search['baseline_heldout']['ppl']:.4f}, "
        f"tokens={search['baseline_heldout']['n_tokens']}",
        "",
        "## Shortcuts / notes",
        "",
    ]
    if shortcuts:
        for s in shortcuts:
            lines.append(f"- {s}")
    else:
        lines.append("- None.")
    lines += [
        f"- Search floor (calib): ppl_retention ≥ {args.calib_floor} (softer than held-out 0.95).",
        f"- Candidate keeps tried when lowering: {args.candidate_keeps}.",
        f"- Wall time: **{summary['wall_seconds']}s**",
        "",
    ]
    args.md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out} and {args.md}", flush=True)
    print(f"Decision: {decision} (heldout ret={held_ret:.4f})", flush=True)
    print(f"Total wall: {summary['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
