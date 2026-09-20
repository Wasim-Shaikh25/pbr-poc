#!/usr/bin/env python3
"""PBR-H95Q follow-up stack: Track1 embed tiers, Track2 C recovery, Track3 MLP K3 + stacks.

Runtime: CPU threads=1, reuse model in memory, max_length=256.
Target wall <45–60 min; skip entropy diagnostics (rate honesty via packed K + map).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pbr_h95.apply_policy import (
    apply_policy_to_model,
    bf16_tensor_to_u16,
    clone_cpu_state_dict,
)
from pbr_h95.embed_tiers import find_embed_param, padded_row_indices, reachable_token_ids
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_h95.h95q_stack import (
    apply_b1_plus_embed_tiers,
    apply_c_k3_base_recovery,
    apply_mlp_k3_policy,
    apply_sparse_recovery_c,
    b1_body_keep_fn,
    fit_embed_tiers_on_calib,
    rate_b1_embed_tiers,
    rate_mlp_k3,
    stack_description,
)
from pbr_h95.packed_rate import rate_report_for_keep_map
from pbr_h95.policy import family_keep_breakdown
from pbr_h95.quantize import quantize_bf16_mantissas

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")

HONESTY_LINES = [
    "Proxy PPL on in-repo calib-v2 / heldout-v1 only — not a production LM benchmark.",
    "heldout ppl_retention ≥ 0.95 = proxy GO for that map; else NO-GO. Not MMLU/HellaSwag.",
    "packed_K_total_bpw = 1 sign + 2.62 exp ref + avg packed K (+ exception map_bpw for C).",
    "Candidate C map format: per-tensor bitmap (1 bit/unit) OR absolute indices; auto=min; + correction bits.",
    "Embed frequency tiers fit on calib-v2 only — risk of calib overfitting for rare tokens.",
    "Not a physical container encode; not a ≤8 BPW product claim unless packed≤8 AND heldout≥0.95.",
    "Original BF16 need not round-trip; quantized reference MUST (Q idempotent on samples).",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    texts = [t["text"] for t in payload["texts"]]
    return payload, texts


def assert_quantizer_idempotence(baseline_sd: dict, keep_map: dict[str, int], *, max_sample: int = 6) -> dict:
    checked = 0
    failures = []
    names = [n for n in keep_map if n in baseline_sd and baseline_sd[n].is_floating_point()]
    seen: set[int] = set()
    uniq = []
    for n in names:
        ptr = baseline_sd[n].data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        uniq.append(n)
    prefer = [n for n in uniq if "embed" in n.lower()]
    prefer += [n for n in uniq if "mlp" in n.lower()][:2]
    prefer += [n for n in uniq if "self_attn" in n.lower()][:1]
    sample = []
    for n in prefer + uniq:
        if n not in sample:
            sample.append(n)
        if len(sample) >= max_sample:
            break
    for name in sample:
        k = int(keep_map[name])
        if k < 0:
            k = 4  # mixed → spot-check at K4
        words = bf16_tensor_to_u16(baseline_sd[name])
        q1 = quantize_bf16_mantissas(words, k)
        q2 = quantize_bf16_mantissas(q1, k)
        checked += 1
        if not np.array_equal(q1, q2):
            failures.append(name)
    return {"checked": checked, "all_idempotent": len(failures) == 0, "failures": failures, "sampled": sample}


def _gate(packed_bpw: float, held_ret: float) -> dict:
    return {
        "heldout_threshold": 0.95,
        "heldout_ppl_retention": held_ret,
        "decision": "proxy_GO" if held_ret >= 0.95 else "proxy_NO_GO",
        "approaches_8_packed_bpw": bool(packed_bpw <= 8.0),
        "dual_gate_pass": bool(packed_bpw <= 8.0 and held_ret >= 0.95),
        "note": "proxy only — dual_gate = packed≤8 AND heldout≥0.95; not product claim",
    }


def eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length):
    calib = mean_token_nll(model, tokenizer, calib_texts, max_length=max_length)
    held = mean_token_nll(model, tokenizer, held_texts, max_length=max_length)
    calib_ret = ppl_retention(base_calib["ppl"], calib["ppl"])
    held_ret = ppl_retention(base_held["ppl"], held["ppl"])
    return {
        "calib": {
            "mean_nll": calib["mean_nll"],
            "ppl": calib["ppl"],
            "n_tokens": calib["n_tokens"],
            "ppl_retention": calib_ret,
            "delta_nll": calib["mean_nll"] - base_calib["mean_nll"],
        },
        "heldout": {
            "mean_nll": held["mean_nll"],
            "ppl": held["ppl"],
            "n_tokens": held["n_tokens"],
            "ppl_retention": held_ret,
            "delta_nll": held["mean_nll"] - base_held["mean_nll"],
        },
    }


def run_ref_b1(model, tokenizer, baseline_sd, calib_texts, held_texts, base_calib, base_held, max_length):
    t0 = time.perf_counter()
    label = "REF_B1_mlp_k4"
    meta = apply_policy_to_model(model, b1_body_keep_fn(), baseline=baseline_sd)
    keep_map = meta["keep_map"]
    idem = assert_quantizer_idempotence(baseline_sd, keep_map)
    scores = eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length)
    rate = rate_report_for_keep_map(keep_map, baseline_sd)
    packed = rate["packed_K_total_bpw"]
    row = {
        "name": label,
        "track": "REF",
        "description": stack_description(label),
        "avg_packed_K": rate["avg_packed_K"],
        "packed_K_total_bpw": packed,
        "exception_map_bpw": 0.0,
        "rate": rate,
        "family_breakdown": family_keep_breakdown(keep_map, baseline_sd),
        **scores,
        "proxy_gate": _gate(packed, scores["heldout"]["ppl_retention"]),
        "idempotence": idem,
        "c_recovery": None,
        "embed_tiers": None,
        "k3_variant": None,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    _print_row(row)
    return row


def run_track1(
    model,
    tokenizer,
    baseline_sd,
    *,
    embed_name,
    tier_cache: dict,
    schedule: str,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    max_length,
    padded_indices,
):
    t0 = time.perf_counter()
    label = f"T1_B1_embed_{schedule}"
    row_keeps, tier_meta = tier_cache[schedule]
    meta = apply_b1_plus_embed_tiers(
        model, baseline_sd=baseline_sd, row_keeps=row_keeps, embed_name=embed_name
    )
    idem = assert_quantizer_idempotence(baseline_sd, meta["keep_map"])
    scores = eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length)
    rate = rate_b1_embed_tiers(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded=False,
        padded_indices=padded_indices,
    )
    # Also report omit-padded bookkeeping (not used for gate)
    rate_omit = rate_b1_embed_tiers(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded=True,
        padded_indices=padded_indices,
    )
    packed = rate["packed_K_total_bpw"]
    row = {
        "name": label,
        "track": "T1",
        "description": stack_description(label),
        "avg_packed_K": rate["avg_packed_K"],
        "packed_K_total_bpw": packed,
        "exception_map_bpw": 0.0,
        "rate": rate,
        "rate_omit_padded_bookkeeping": rate_omit,
        "family_breakdown": None,
        **scores,
        "proxy_gate": _gate(packed, scores["heldout"]["ppl_retention"]),
        "idempotence": idem,
        "c_recovery": None,
        "embed_tiers": {
            "schedule": schedule,
            "tier_meta": {
                k: tier_meta[k]
                for k in (
                    "schedule",
                    "total_calib_tokens",
                    "n_unique_seen_ids",
                    "top_cut_n",
                    "band_row_counts",
                    "keep_hist_rows",
                    "keeps_by_band",
                )
                if k in tier_meta
            },
            "avg_embed_row_keep": meta["embed_row_meta"]["avg_row_keep"],
            "row_keep_hist": meta["embed_row_meta"]["row_keep_hist"],
        },
        "k3_variant": None,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    _print_row(row)
    return row


def run_track2(
    model,
    tokenizer,
    baseline_sd,
    *,
    label: str,
    recover_frac: float,
    mode: str,
    base_attn_keep: int,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    max_length,
    embed_row_keeps=None,
    embed_name=None,
):
    t0 = time.perf_counter()
    meta = apply_sparse_recovery_c(
        model,
        baseline_sd=baseline_sd,
        recover_frac=recover_frac,
        base_mlp_keep=4,
        base_attn_keep=base_attn_keep,
        embed_keep=7,
        recover_keep=7,
        mode=mode,
        target_families=("mlp_mid",),
        map_scheme="auto",
        embed_row_keeps=embed_row_keeps,
        embed_name=embed_name,
    )
    cr = meta["c_recovery"]
    rates = cr["rates"]
    packed = rates["packed_K_total_bpw"]
    idem = assert_quantizer_idempotence(baseline_sd, meta["keep_map"])
    scores = eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length)
    row = {
        "name": label,
        "track": "T2",
        "description": stack_description(label) if label in stack_description.__globals__.get("STACK_POLICY_DESCRIPTIONS", {}) or True else label,
        "avg_packed_K": rates["avg_mant_bpw"],
        "packed_K_total_bpw": packed,
        "exception_map_bpw": rates["exception_map_bpw"],
        "rate": rates,
        "family_breakdown": None,
        **scores,
        "proxy_gate": _gate(packed, scores["heldout"]["ppl_retention"]),
        "idempotence": idem,
        "c_recovery": {
            "recover_frac_target": cr["recover_frac_target"],
            "mode": cr["mode"],
            "base_mlp_keep": cr["base_mlp_keep"],
            "base_attn_keep": cr["base_attn_keep"],
            "recover_keep": cr["recover_keep"],
            "recovered_units": cr["recovered_units"],
            "units_total": cr["units_total"],
            "recovered_words": cr["recovered_words"],
            "extra_mantissa_bits": cr["extra_mantissa_bits"],
            "exception_map_bits": cr["exception_map_bits"],
            "map_scheme_hist": cr["map_scheme_hist"],
            "map_format": cr["map_format"],
            "n_tensors_recovered": cr["n_tensors_recovered"],
        },
        "embed_tiers": None,
        "k3_variant": None,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    row["description"] = stack_description(label) if label.startswith("T2_") else label
    _print_row(row)
    return row


def run_track3(
    model,
    tokenizer,
    baseline_sd,
    *,
    variant: str,
    label: str,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    max_length,
    embed_row_keeps=None,
    embed_name=None,
):
    t0 = time.perf_counter()
    meta = apply_mlp_k3_policy(
        model,
        baseline_sd=baseline_sd,
        variant=variant,
        attn_keep=5,
        base_mlp_keep=4,
        embed_row_keeps=embed_row_keeps,
        embed_name=embed_name,
    )
    rate = rate_mlp_k3(meta, baseline_sd, embed_name=embed_name, embed_row_keeps=embed_row_keeps)
    packed = rate["packed_K_total_bpw"]
    idem = assert_quantizer_idempotence(baseline_sd, meta["keep_map"])
    scores = eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length)
    row = {
        "name": label,
        "track": "T3" if label.startswith("T3_") else "STACK",
        "description": stack_description(label),
        "avg_packed_K": rate["avg_packed_K"],
        "packed_K_total_bpw": packed,
        "exception_map_bpw": 0.0,
        "rate": rate,
        "family_breakdown": None,
        **scores,
        "proxy_gate": _gate(packed, scores["heldout"]["ppl_retention"]),
        "idempotence": idem,
        "c_recovery": None,
        "embed_tiers": {"schedule": "aggressive"} if embed_row_keeps is not None else None,
        "k3_variant": meta.get("k3_variant"),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    _print_row(row)
    return row


def run_stack_c_k3(
    model,
    tokenizer,
    baseline_sd,
    *,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    max_length,
):
    t0 = time.perf_counter()
    label = "S2_C_k3_base_row05_k7"
    meta = apply_c_k3_base_recovery(
        model,
        baseline_sd=baseline_sd,
        recover_frac=0.05,
        base_mlp_keep=3,
        base_attn_keep=4,
        recover_keep=7,
        mode="row_magnitude",
    )
    cr = meta["c_recovery"]
    rates = cr["rates"]
    packed = rates["packed_K_total_bpw"]
    idem = assert_quantizer_idempotence(baseline_sd, meta["keep_map"])
    scores = eval_nll_pair(model, tokenizer, calib_texts, held_texts, base_calib, base_held, max_length)
    row = {
        "name": label,
        "track": "STACK",
        "description": stack_description(label),
        "avg_packed_K": rates["avg_mant_bpw"],
        "packed_K_total_bpw": packed,
        "exception_map_bpw": rates["exception_map_bpw"],
        "rate": rates,
        "family_breakdown": None,
        **scores,
        "proxy_gate": _gate(packed, scores["heldout"]["ppl_retention"]),
        "idempotence": idem,
        "c_recovery": {
            "recover_frac_target": cr["recover_frac_target"],
            "mode": cr["mode"],
            "base_mlp_keep": cr["base_mlp_keep"],
            "base_attn_keep": cr["base_attn_keep"],
            "recover_keep": cr["recover_keep"],
            "recovered_units": cr["recovered_units"],
            "units_total": cr["units_total"],
            "extra_mantissa_bits": cr["extra_mantissa_bits"],
            "exception_map_bits": cr["exception_map_bits"],
            "map_scheme_hist": cr["map_scheme_hist"],
            "map_format": cr["map_format"],
        },
        "embed_tiers": None,
        "k3_variant": {"variant": "k3_base_with_sparse_k7"},
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    _print_row(row)
    return row


def _print_row(row: dict) -> None:
    g = row["proxy_gate"]
    print(
        f"  [{row['name']}] packed≈{row['packed_K_total_bpw']:.4f} "
        f"map={row.get('exception_map_bpw', 0):.5f} "
        f"calib_ret={row['calib']['ppl_retention']:.4f} "
        f"held_ret={row['heldout']['ppl_retention']:.4f} "
        f"{g['decision']} dual={g['dual_gate_pass']} ≤8={g['approaches_8_packed_bpw']} "
        f"({row['wall_seconds']}s)",
        flush=True,
    )


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PBR-H95Q stack follow-up — Track1 / Track2 / Track3 + stacks",
        "",
        f"Model: `{payload['repo']}` (local BF16, CPU, threads=1)",
        f"Calibration: **calib-v2** tokens={payload['calib_tokens']}, max_length={payload['max_length']}",
        f"Held-out: **heldout-v1** tokens={payload['heldout_tokens']}, max_length={payload['max_length']}",
        f"Wall time: **{payload['wall_seconds']}s**",
        "",
        "## Honesty",
        "",
    ]
    for h in payload["honesty"]:
        lines.append(f"- {h}")
    lines += [
        "",
        "## BF16 baseline",
        "",
        f"- calib mean_nll={payload['bf16']['calib']['mean_nll']:.6f} ppl={payload['bf16']['calib']['ppl']:.4f}",
        f"- heldout mean_nll={payload['bf16']['heldout']['mean_nll']:.6f} ppl={payload['bf16']['heldout']['ppl']:.4f}",
        "",
        "## Results matrix",
        "",
        "| Config | track | packed BPW | map BPW | avg K | calib ret | heldout ret | proxy | ≤8? | dual-gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for c in payload["candidates"]:
        lines.append(
            f"| {c['name']} | {c['track']} | {c['packed_K_total_bpw']:.4f} | "
            f"{c.get('exception_map_bpw', 0):.5f} | {c['avg_packed_K']:.4f} | "
            f"{c['calib']['ppl_retention']:.4f} | {c['heldout']['ppl_retention']:.4f} | "
            f"{c['proxy_gate']['decision']} | {c['proxy_gate']['approaches_8_packed_bpw']} | "
            f"{c['proxy_gate']['dual_gate_pass']} |"
        )
    lines += ["", "## Policy rules", ""]
    for c in payload["candidates"]:
        lines.append(f"- **{c['name']}**: {c['description']}")
        if c.get("c_recovery"):
            cr = c["c_recovery"]
            lines.append(
                f"  - C: mode={cr['mode']} frac={cr['recover_frac_target']} "
                f"recovered={cr['recovered_units']}/{cr['units_total']} "
                f"map_bits={cr['exception_map_bits']} schemes={cr.get('map_scheme_hist')}"
            )
        if c.get("embed_tiers") and c["embed_tiers"].get("avg_embed_row_keep") is not None:
            et = c["embed_tiers"]
            lines.append(
                f"  - Embed tiers ({et.get('schedule')}): avg_row_K={et['avg_embed_row_keep']:.4f} "
                f"hist={et.get('row_keep_hist')}"
            )
        if c.get("k3_variant"):
            lines.append(f"  - K3 variant: {c['k3_variant']}")
    lines += [
        "",
        "## Candidate C map format",
        "",
        payload.get("c_map_format_doc", ""),
        "",
        "## Idempotence",
        "",
    ]
    for c in payload["candidates"]:
        idem = c["idempotence"]
        lines.append(
            f"- **{c['name']}**: checked={idem['checked']} all_ok={idem['all_idempotent']}"
        )
    s = payload["summary"]
    lines += [
        "",
        "## Summary",
        "",
        f"- Any packed ≤8: **{s['any_leq_8_packed']}**",
        f"- Any dual-gate (≤8 AND heldout≥0.95): **{s['any_dual_gate']}**",
        f"- Dual-gate names: {s['dual_gate_names']}",
        f"- Proxy-GO names: {s['proxy_go_names']}",
        f"- Best ≤8-or-closest: {s['best_leq8_or_closest']}",
        f"- Best dual-gate / proxy-GO by packed: {s['best_proxy_go']}",
        f"- Shortcuts: {s['shortcuts']}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pbr_h95/h95q_stack_qwen.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/pbr_h95/h95q_stack_qwen.md"))
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sweep: 1 schedule/track + 1 stack (still covers all tracks)",
    )
    args = ap.parse_args()

    wall0 = time.perf_counter()
    _, calib_texts = load_texts(args.calib)
    _, held_texts = load_texts(args.heldout)

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
    model.to(torch.device("cpu"))

    baseline_sd = clone_cpu_state_dict(model)
    embed_name, emb_w = find_embed_param(model)
    vocab_rows = int(emb_w.shape[0])
    reachable = reachable_token_ids(tokenizer)
    padded = set(padded_row_indices(vocab_rows, reachable))

    print("BF16 baseline NLL …", flush=True)
    base_calib = mean_token_nll(model, tokenizer, calib_texts, max_length=args.max_length)
    base_held = mean_token_nll(model, tokenizer, held_texts, max_length=args.max_length)
    print(
        f"  calib nll={base_calib['mean_nll']:.6f} ppl={base_calib['ppl']:.4f} tok={base_calib['n_tokens']}",
        flush=True,
    )
    print(
        f"  held  nll={base_held['mean_nll']:.6f} ppl={base_held['ppl']:.4f} tok={base_held['n_tokens']}",
        flush=True,
    )

    # Fit embed tiers once per schedule on calib only
    schedules = ["aggressive"] if args.quick else ["default", "aggressive", "conservative"]
    print(f"Fitting embed frequency tiers on calib-v2: {schedules} …", flush=True)
    tier_cache: dict[str, tuple] = {}
    for sch in ["default", "aggressive", "conservative"]:
        rk, meta = fit_embed_tiers_on_calib(
            tokenizer, calib_texts, vocab_rows=vocab_rows, schedule=sch, max_length=args.max_length
        )
        tier_cache[sch] = (rk, meta)
        print(
            f"  {sch}: avg_row_K={float(rk.mean()):.4f} hist={ {str(k): int((rk==k).sum()) for k in range(8)} }",
            flush=True,
        )

    results = []

    print("REF B1 …", flush=True)
    results.append(
        run_ref_b1(model, tokenizer, baseline_sd, calib_texts, held_texts, base_calib, base_held, args.max_length)
    )

    print("Track 1 — B1 + embed tiers …", flush=True)
    for sch in schedules:
        results.append(
            run_track1(
                model,
                tokenizer,
                baseline_sd,
                embed_name=embed_name,
                tier_cache=tier_cache,
                schedule=sch,
                calib_texts=calib_texts,
                held_texts=held_texts,
                base_calib=base_calib,
                base_held=base_held,
                max_length=args.max_length,
                padded_indices=padded,
            )
        )

    # Pick best Track1 by (dual gate preferred) else closest to 8 among proxy-GO else lowest packed
    t1 = [r for r in results if r["track"] == "T1"]
    best_t1 = None
    if t1:
        dual = [r for r in t1 if r["proxy_gate"]["dual_gate_pass"]]
        go = [r for r in t1 if r["proxy_gate"]["decision"] == "proxy_GO"]
        if dual:
            best_t1 = min(dual, key=lambda r: r["packed_K_total_bpw"])
        elif go:
            best_t1 = min(go, key=lambda r: abs(r["packed_K_total_bpw"] - 8.0))
        else:
            best_t1 = min(t1, key=lambda r: abs(r["packed_K_total_bpw"] - 8.0))
    best_sched = (best_t1["embed_tiers"]["schedule"] if best_t1 else "aggressive")
    print(f"Best Track1 schedule for stacks: {best_sched}", flush=True)

    print("Track 2 — proper Candidate C sparse recovery …", flush=True)
    if args.quick:
        t2_specs = [
            ("T2_C_row_mag_0.05", 0.05, "row_magnitude", 4),
            ("T2_C_channel_mag_0.05", 0.05, "channel_magnitude", 4),
        ]
    else:
        t2_specs = [
            ("T2_C_row_mag_0.02", 0.02, "row_magnitude", 4),
            ("T2_C_row_mag_0.05", 0.05, "row_magnitude", 4),
            ("T2_C_row_mag_0.10", 0.10, "row_magnitude", 4),
            ("T2_C_channel_mag_0.05", 0.05, "channel_magnitude", 4),
        ]
    for label, frac, mode, attn_k in t2_specs:
        results.append(
            run_track2(
                model,
                tokenizer,
                baseline_sd,
                label=label,
                recover_frac=frac,
                mode=mode,
                base_attn_keep=attn_k,
                calib_texts=calib_texts,
                held_texts=held_texts,
                base_calib=base_calib,
                base_held=base_held,
                max_length=args.max_length,
            )
        )

    print("Track 3 — selective MLP K3 …", flush=True)
    if args.quick:
        t3_specs = [
            ("T3_mlp_all_k3", "all_mlp_k3"),
            ("T3_mlp_band_8_15_k3", "band_8_15"),
            ("T3_mlp_robust50_k3", "robust50_rows"),
        ]
    else:
        t3_specs = [
            ("T3_mlp_all_k3", "all_mlp_k3"),
            ("T3_mlp_band_1_7_k3", "band_1_7"),
            ("T3_mlp_band_8_15_k3", "band_8_15"),
            ("T3_mlp_band_16_22_k3", "band_16_22"),
            ("T3_mlp_robust50_k3", "robust50_rows"),
        ]
    for label, variant in t3_specs:
        results.append(
            run_track3(
                model,
                tokenizer,
                baseline_sd,
                variant=variant,
                label=label,
                calib_texts=calib_texts,
                held_texts=held_texts,
                base_calib=base_calib,
                base_held=base_held,
                max_length=args.max_length,
            )
        )

    print("Stacked configs …", flush=True)
    # S1: B1 + best/aggressive embed + band 8-15 K3
    emb_keeps = tier_cache[best_sched][0]
    results.append(
        run_track3(
            model,
            tokenizer,
            baseline_sd,
            variant="band_8_15",
            label="S1_B1_aggr_embed_band815_k3",
            calib_texts=calib_texts,
            held_texts=held_texts,
            base_calib=base_calib,
            base_held=base_held,
            max_length=args.max_length,
            embed_row_keeps=emb_keeps,
            embed_name=embed_name,
        )
    )
    # Fix description if schedule wasn't aggressive
    results[-1]["description"] = (
        f"Stack: B1 + {best_sched} embed tiers + mlp band 8–15 @K3."
    )
    results[-1]["embed_tiers"] = {
        "schedule": best_sched,
        "avg_embed_row_keep": float(emb_keeps.mean()),
    }

    results.append(
        run_stack_c_k3(
            model,
            tokenizer,
            baseline_sd,
            calib_texts=calib_texts,
            held_texts=held_texts,
            base_calib=base_calib,
            base_held=base_held,
            max_length=args.max_length,
        )
    )

    proxy_go = [c for c in results if c["proxy_gate"]["decision"] == "proxy_GO"]
    dual = [c for c in results if c["proxy_gate"]["dual_gate_pass"]]
    any_leq8 = any(c["proxy_gate"]["approaches_8_packed_bpw"] for c in results)
    leq8 = [c for c in results if c["proxy_gate"]["approaches_8_packed_bpw"]]

    best_proxy = None
    if dual:
        b = min(dual, key=lambda c: c["packed_K_total_bpw"])
        best_proxy = {
            "name": b["name"],
            "packed_K_total_bpw": b["packed_K_total_bpw"],
            "heldout_ppl_retention": b["heldout"]["ppl_retention"],
            "kind": "dual_gate",
        }
    elif proxy_go:
        b = min(proxy_go, key=lambda c: c["packed_K_total_bpw"])
        best_proxy = {
            "name": b["name"],
            "packed_K_total_bpw": b["packed_K_total_bpw"],
            "heldout_ppl_retention": b["heldout"]["ppl_retention"],
            "kind": "proxy_GO_min_packed",
        }

    # Closest to ≤8: prefer dual, else leq8 with best held, else min |bpw-8| among all
    if dual:
        closest = min(dual, key=lambda c: c["packed_K_total_bpw"])
        closest_kind = "dual_gate"
    elif leq8:
        closest = max(leq8, key=lambda c: c["heldout"]["ppl_retention"])
        closest_kind = "leq8_best_held"
    else:
        closest = min(results, key=lambda c: abs(c["packed_K_total_bpw"] - 8.0))
        closest_kind = "closest_to_8"

    shortcuts = [
        "entropy diagnostics skipped (packed-K + map honesty only)",
        "C recovery salience = |w| magnitude (no per-row calib sensitivity sweep)",
        "no physical container bytes; map_bpw estimated",
        f"embed tiers fit calib-v2 only; stack used schedule={best_sched}",
    ]
    if args.quick:
        shortcuts.append("--quick: reduced schedules/fracs/bands")

    c_map_doc = (
        "Per targeted 2D mlp weight: select top-|w|-mean rows **or** channels at "
        "`recover_frac`. Exception map = cheaper of (a) bitmap 1 bit/unit or "
        "(b) absolute indices `n_recover * ceil(log2(n_units))`. Correction stream = "
        "`(K_recover - K_base) * recovered_words` extra mantissa bits, folded into "
        "avg K. Packed total = 1 + 2.62 + adj_avg_K + map_bpw. Not a serialized container."
    )

    payload = {
        "phase": "H95Q-stack",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_dir": str(args.model_dir),
        "max_length": args.max_length,
        "calib_path": str(args.calib),
        "heldout_path": str(args.heldout),
        "calib_tokens": base_calib["n_tokens"],
        "heldout_tokens": base_held["n_tokens"],
        "honesty": HONESTY_LINES,
        "c_map_format_doc": c_map_doc,
        "bf16": {"calib": {k: base_calib[k] for k in ("mean_nll", "ppl", "n_tokens")},
                 "heldout": {k: base_held[k] for k in ("mean_nll", "ppl", "n_tokens")}},
        "candidates": results,
        "summary": {
            "any_leq_8_packed": any_leq8,
            "any_dual_gate": bool(dual),
            "dual_gate_names": [c["name"] for c in dual],
            "proxy_go_names": [c["name"] for c in proxy_go],
            "best_proxy_go": best_proxy,
            "best_leq8_or_closest": {
                "name": closest["name"],
                "packed_K_total_bpw": closest["packed_K_total_bpw"],
                "heldout_ppl_retention": closest["heldout"]["ppl_retention"],
                "kind": closest_kind,
                "dual_gate_pass": closest["proxy_gate"]["dual_gate_pass"],
            },
            "best_track1_schedule": best_sched,
            "shortcuts": shortcuts,
        },
        "wall_seconds": round(time.perf_counter() - wall0, 2),
        "disclaimer": (
            "H95Q stack follow-up. Proxy GO/NO-GO on heldout-v1. "
            "Not a physical container or ≤8 product claim unless dual-gate passes."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(payload, args.md)
    print(f"Wrote {args.out} and {args.md}", flush=True)
    print(
        f"DONE wall={payload['wall_seconds']}s dual={payload['summary']['dual_gate_names']} "
        f"any≤8={any_leq8} closest={payload['summary']['best_leq8_or_closest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
