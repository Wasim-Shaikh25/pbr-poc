#!/usr/bin/env python3
"""PBR-H95E: vocabulary / frequency-aware embedding mantissa compression.

Stacks on Phase C baseline (uniform mid keep=4, protected@7) and only varies
embedding rows (E1 uniform ladder + E2 frequency tiers).
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

from pbr_h95.apply_policy import apply_policy_to_model, clone_cpu_state_dict
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_h95.policy import uniform_mid_keep_fn
from pbr_h95.embed_tiers import (
    apply_embed_row_keeps,
    assign_frequency_tiers,
    avg_keep_bits_with_embed_rows,
    compact_tier_summary,
    count_token_frequencies,
    estimate_bytes_saved_with_embed_rows,
    find_embed_param,
    inventory_embeddings,
    padded_row_indices,
    reachable_token_ids,
    uniform_row_keeps,
)

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")

HONESTY_LINES = [
    "Proxy PPL on in-repo calib/heldout only — not a multilingual/production bench.",
    "Frequency tiers fit on calib-v2 — risk of calib overfitting for rare tokens.",
    "Padded-row cleanup ≠ meaningful BPW win (padded rows are nonzero; omitting them from storage estimate is bookkeeping).",
    "est BPW is not a physical container (1 sign + 2.62 exp ref + avg mantissa keep).",
    "Not claiming ≤8 BPW product unless heldout retention ≥ 0.95 AND numbers support it — if heldout ≥ 0.95 say proxy GO for that map; else NO-GO.",
    "English-heavy calib ≠ multilingual retention proof (Qwen is multilingual).",
    "Body = Phase C frozen map (mid bands keep=4, emb/norm/bias/first/last protected@7) with only embed rows varied.",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    texts = [t["text"] for t in payload["texts"]]
    return payload, texts


def est_bpw(avg_keep: float) -> float:
    return 1.0 + 2.62 + float(avg_keep)


def phase_c_body_keep_fn(mid_keep: int = 4, protected: int = 7):
    """Phase C frozen body: mid attn/mlp @ mid_keep; protected families @ protected.

    Embed is in the protected set at *protected*; callers then overwrite embed
    rows with a separate per-row keep vector.
    """
    return uniform_mid_keep_fn(mid_keep, protected=protected)


def apply_phase_c_plus_embed_rows(
    model,
    *,
    baseline_sd,
    row_keeps: np.ndarray,
    embed_name: str,
    mid_keep: int = 4,
):
    """Restore baseline, apply Phase C body (embed left @7), then per-row embed keeps."""
    keep_fn = phase_c_body_keep_fn(mid_keep=mid_keep, protected=7)
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    # Now quantize embed rows from the *current* (body-quantized, embed still exact) weights
    # Actually embed was left at keep=7 so still BF16 from baseline — apply row keeps.
    name, weight = find_embed_param(model)
    # Use baseline embed as source of truth for row quant
    base_w = baseline_sd[embed_name]
    row_meta = apply_embed_row_keeps(weight, row_keeps, baseline_weight=base_w)
    # Re-tie if needed
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    meta["embed_row_meta"] = row_meta
    meta["embed_name"] = name
    return meta


def eval_config(
    model,
    tokenizer,
    *,
    baseline_sd,
    row_keeps: np.ndarray,
    embed_name: str,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    phase_c_calib_ppl: float,
    phase_c_held_ppl: float,
    max_length: int,
    label: str,
    padded_indices: set[int],
    omit_padded: bool = True,
) -> dict:
    t0 = time.perf_counter()
    meta = apply_phase_c_plus_embed_rows(
        model, baseline_sd=baseline_sd, row_keeps=row_keeps, embed_name=embed_name
    )
    calib = mean_token_nll(model, tokenizer, calib_texts, max_length=max_length)
    held = mean_token_nll(model, tokenizer, held_texts, max_length=max_length)

    avg_k, avg_detail = avg_keep_bits_with_embed_rows(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded_from_storage=omit_padded,
        padded_indices=padded_indices,
    )
    # Also report without omitting padded
    avg_k_full, _ = avg_keep_bits_with_embed_rows(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded_from_storage=False,
        padded_indices=padded_indices,
    )
    bytes_saved = estimate_bytes_saved_with_embed_rows(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded_from_storage=omit_padded,
        padded_indices=padded_indices,
    )
    bytes_saved_full = estimate_bytes_saved_with_embed_rows(
        meta["keep_map"],
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded_from_storage=False,
        padded_indices=padded_indices,
    )

    row = {
        "name": label,
        "calib": {
            "mean_nll": calib["mean_nll"],
            "ppl": calib["ppl"],
            "n_tokens": calib["n_tokens"],
            "delta_nll_vs_bf16": calib["mean_nll"] - base_calib["mean_nll"],
            "ppl_retention_vs_bf16": ppl_retention(base_calib["ppl"], calib["ppl"]),
            "delta_nll_vs_phase_c": calib["mean_nll"] - (base_calib["mean_nll"] + 0),  # filled below
            "ppl_retention_vs_phase_c": ppl_retention(phase_c_calib_ppl, calib["ppl"]),
        },
        "heldout": {
            "mean_nll": held["mean_nll"],
            "ppl": held["ppl"],
            "n_tokens": held["n_tokens"],
            "delta_nll_vs_bf16": held["mean_nll"] - base_held["mean_nll"],
            "ppl_retention_vs_bf16": ppl_retention(base_held["ppl"], held["ppl"]),
            "ppl_retention_vs_phase_c": ppl_retention(phase_c_held_ppl, held["ppl"]),
        },
        "avg_keep_bits": avg_k,
        "avg_keep_bits_including_padded": avg_k_full,
        "est_total_bpw": est_bpw(avg_k),
        "est_total_bpw_including_padded": est_bpw(avg_k_full),
        "bytes_saved_vs_bf16_mantissa": bytes_saved,
        "bytes_saved_including_padded_rows": bytes_saved_full,
        "avg_keep_detail": avg_detail,
        "embed_row_keep_hist": meta["embed_row_meta"]["row_keep_hist"],
        "avg_embed_row_keep": meta["embed_row_meta"]["avg_row_keep"],
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    # Fix calib delta vs phase C using phase_c nll if we stash it — use ppl path only for retention
    print(
        f"  [{label}] calib_ret={row['calib']['ppl_retention_vs_bf16']:.4f} "
        f"held_ret={row['heldout']['ppl_retention_vs_bf16']:.4f} "
        f"vsC_held={row['heldout']['ppl_retention_vs_phase_c']:.4f} "
        f"bpw≈{row['est_total_bpw']:.2f} (full_pad {row['est_total_bpw_including_padded']:.2f}) "
        f"({row['wall_seconds']}s)",
        flush=True,
    )
    return row


def write_inventory_md(inv: dict, path: Path) -> None:
    lines = [
        "# PBR-H95E E1 — Embedding inventory (Qwen2.5-0.5B-Instruct)",
        "",
        f"- Embed param: `{inv['embed_param_name']}`",
        f"- Shape: **{inv['embed_shape'][0]} × {inv['embed_shape'][1]}** ({inv['embed_dtype']})",
        f"- Embed params: **{inv['embed_params']:,}** ({inv['embed_share_of_unique']*100:.2f}% of unique storage)",
        f"- BF16 MiB (embed only): **{inv['embed_bf16_mib']:.2f}**",
        f"- tie_word_embeddings: **{inv['tie_word_embeddings']}**",
        f"- Tied aliases: `{inv['tied_aliases']}`",
        f"- Separate lm_head params: `{inv['lm_head_separate_params']}`",
        f"- Configured vocab_size: **{inv['configured_vocab_size']}**",
        f"- Tokenizer len (reachable): **{inv['reachable_ids']}** (vocab_size attr={inv['tokenizer_vocab_size_attr']})",
        f"- Padded / unreachable rows: **{inv['padded_row_count']}**",
        f"- Padded rows are zero: **{inv['padded_rows_are_zero']}** "
        f"(mean L2 norm={inv['unused_row_mean_norm']:.6f}, "
        f"max={inv['unused_row_max_norm']:.6f}, zero_rows={inv['unused_zero_rows']})",
        "",
        "## Vocab class histogram",
        "",
        "| class | count |",
        "| --- | ---: |",
    ]
    for k, v in inv["class_counts"].items():
        lines.append(f"| {k} | {v} |")
    if "softmax_mass_on_padded_rows" in inv:
        sm = inv["softmax_mass_on_padded_rows"]
        lines += [
            "",
            "## Softmax mass on padded rows (probe)",
            f"- mean={sm['mean']:.6e}, max={sm['max']:.6e}, n_probes={sm['n_probes']}",
        ]
    lines += ["", "## Notes", "", inv["note_padded_nonzero"], ""]
    path.write_text("\n".join(lines))


def write_results_md(payload: dict, path: Path) -> None:
    gate = payload["gate"]
    lines = [
        "# PBR-H95E — Embedding mantissa compression (on Phase C body)",
        "",
        f"Model: `{payload['repo']}` (local BF16, CPU)",
        f"Body: Phase C frozen = uniform mid keep=4, protected@7 (emb/norm/bias/first/last)",
        f"Calibration: **calib-v2** tokens={payload['calibration']['n_tokens_scored']}, max_length={payload['max_length']}",
        f"Held-out: **heldout-v1** tokens={payload['heldout']['n_tokens_scored']}, max_length={payload['max_length']}",
        "",
        "## Honesty",
        "",
    ]
    for h in HONESTY_LINES:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Gate (best tier / primary map)",
        "",
        f"- Primary map: **{gate['primary_map']}**",
        f"- heldout ppl_retention vs BF16: **{gate['heldout_ppl_retention']:.4f}** (threshold 0.95)",
        f"- calib ppl_retention vs BF16: **{gate['calib_ppl_retention']:.4f}**",
        f"- est_total_bpw (omit padded storage): **{gate['est_total_bpw']:.2f}** (Phase C was **8.63**)",
        f"- **Decision: {gate['decision']}**",
        "",
        "## E1 Uniform embed ladder (Phase C body fixed)",
        "",
        "| embed_keep | calib ret | heldout ret | Δnll calib | est BPW (omit pad) | est BPW (full) | wall s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["e1_uniform_ladder"]:
        lines.append(
            f"| {r['embed_keep']} | {r['calib']['ppl_retention_vs_bf16']:.4f} | "
            f"{r['heldout']['ppl_retention_vs_bf16']:.4f} | "
            f"{r['calib']['delta_nll_vs_bf16']:+.6f} | "
            f"{r['est_total_bpw']:.2f} | {r['est_total_bpw_including_padded']:.2f} | "
            f"{r['wall_seconds']} |"
        )
    lines += [
        "",
        "## E2 Frequency-tiered embed keeps",
        "",
        "| schedule | calib ret | heldout ret | avg embed row keep | est BPW (omit pad) | vs Phase C BPW | wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["e2_tiers"]:
        lines.append(
            f"| {r['name']} | {r['calib']['ppl_retention_vs_bf16']:.4f} | "
            f"{r['heldout']['ppl_retention_vs_bf16']:.4f} | "
            f"{r['avg_embed_row_keep']:.4f} | {r['est_total_bpw']:.2f} | "
            f"{r['est_total_bpw'] - 8.63:+.2f} | {r['wall_seconds']} |"
        )
    lines += [
        "",
        "### Tier band summaries",
        "",
    ]
    for ts in payload["tier_summaries"]:
        m = ts["meta"]
        lines.append(f"**{m['schedule']}** — band row counts: `{m['band_row_counts']}`")
        lines.append(f"- keeps_by_band: `{m['keeps_by_band']}`")
        lines.append(f"- keep_hist_rows: `{ts['keep_hist_rows']}`")
        lines.append(f"- top_cut_n={m['top_cut_n']}, calib_tokens={m['total_calib_tokens']}, unique_seen={m['n_unique_seen_ids']}")
        lines.append("")

    lines += [
        "## Phase C reference (embed@7)",
        "",
        f"- Phase C heldout ret: **{payload['phase_c_reference']['heldout_ppl_retention']:.4f}**",
        f"- Phase C est BPW: **{payload['phase_c_reference']['est_total_bpw']:.2f}**",
        "",
        "## BF16 baselines",
        "",
        f"- calib mean_nll={payload['baseline_calib']['mean_nll']:.6f}, ppl={payload['baseline_calib']['ppl']:.4f}",
        f"- heldout mean_nll={payload['baseline_heldout']['mean_nll']:.6f}, ppl={payload['baseline_heldout']['ppl']:.4f}",
        "",
        "## Shortcuts / notes",
        "",
    ]
    for s in payload.get("shortcuts", []):
        lines.append(f"- {s}")
    lines.append(f"- Wall time: **{payload['wall_seconds']:.2f}s**")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--embed-keeps",
        type=int,
        nargs="+",
        default=[7, 6, 5, 4, 3],
        help="E1 uniform embed keep ladder",
    )
    ap.add_argument(
        "--tier-schedules",
        nargs="+",
        default=["default", "aggressive", "conservative"],
    )
    ap.add_argument("--heldout-gate", type=float, default=0.95)
    ap.add_argument("--skip-softmax-probe", action="store_true")
    ap.add_argument(
        "--inv-out",
        type=Path,
        default=Path("artifacts/pbr_h95/h95e_inventory.json"),
    )
    ap.add_argument(
        "--inv-md",
        type=Path,
        default=Path("artifacts/pbr_h95/h95e_inventory.md"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/pbr_h95/h95e_qwen.json"),
    )
    ap.add_argument(
        "--md",
        type=Path,
        default=Path("artifacts/pbr_h95/h95e_qwen.md"),
    )
    args = ap.parse_args()

    wall0 = time.perf_counter()
    shortcuts: list[str] = []
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
    embed_name, _ = find_embed_param(model)
    vocab_rows = int(baseline_sd[embed_name].shape[0])
    reachable = reachable_token_ids(tokenizer)
    padded = set(padded_row_indices(vocab_rows, reachable))
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # --- E1 Inventory ---
    print("E1 inventory …", flush=True)
    inv = inventory_embeddings(
        model,
        tokenizer,
        check_softmax_mass=not args.skip_softmax_probe,
        device=device,
    )
    inv["model_dir"] = str(args.model_dir)
    args.inv_out.parent.mkdir(parents=True, exist_ok=True)
    args.inv_out.write_text(json.dumps(inv, indent=2) + "\n")
    write_inventory_md(inv, args.inv_md)
    print(
        f"  shape={inv['embed_shape']} share={inv['embed_share_of_unique']:.4f} "
        f"padded={inv['padded_row_count']} mean_pad_norm={inv['unused_row_mean_norm']:.4f}",
        flush=True,
    )

    # --- BF16 baselines ---
    print("BF16 baseline on calib …", flush=True)
    t0 = time.perf_counter()
    base_calib = mean_token_nll(model, tokenizer, calib_texts, max_length=args.max_length)
    base_calib["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print(
        f"  calib nll={base_calib['mean_nll']:.6f} ppl={base_calib['ppl']:.4f} "
        f"tokens={base_calib['n_tokens']} ({base_calib['wall_seconds']}s)",
        flush=True,
    )
    print("BF16 baseline on heldout …", flush=True)
    t0 = time.perf_counter()
    base_held = mean_token_nll(model, tokenizer, held_texts, max_length=args.max_length)
    base_held["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print(
        f"  heldout nll={base_held['mean_nll']:.6f} ppl={base_held['ppl']:.4f} "
        f"tokens={base_held['n_tokens']} ({base_held['wall_seconds']}s)",
        flush=True,
    )

    # Phase C reference numbers (from prior artifact / known frozen)
    phase_c_path = Path("artifacts/pbr_h95/phase_c_qwen.json")
    if phase_c_path.exists():
        pc = json.loads(phase_c_path.read_text())
        phase_c_calib_ppl = pc["frozen"]["calib"]["ppl"]
        phase_c_held_ppl = pc["frozen"]["heldout"]["ppl"]
        phase_c_held_ret = pc["frozen"]["heldout"]["ppl_retention"]
        phase_c_bpw = pc["frozen"]["heldout"]["est_total_bpw"]
    else:
        shortcuts.append("phase_c_qwen.json missing — using embed_keep=7 ladder row as Phase C proxy")
        phase_c_calib_ppl = base_calib["ppl"]
        phase_c_held_ppl = base_held["ppl"]
        phase_c_held_ret = 1.0
        phase_c_bpw = 8.63

    # --- E1 Uniform ladder ---
    print("E1 uniform embed ladder …", flush=True)
    ladder_rows = []
    for k in args.embed_keeps:
        row_keeps = uniform_row_keeps(vocab_rows, k)
        row = eval_config(
            model,
            tokenizer,
            baseline_sd=baseline_sd,
            row_keeps=row_keeps,
            embed_name=embed_name,
            calib_texts=calib_texts,
            held_texts=held_texts,
            base_calib=base_calib,
            base_held=base_held,
            phase_c_calib_ppl=phase_c_calib_ppl,
            phase_c_held_ppl=phase_c_held_ppl,
            max_length=args.max_length,
            label=f"uniform_embed_k{k}",
            padded_indices=padded,
            omit_padded=False,  # uniform: all rows stored at k
        )
        row["embed_keep"] = k
        # For uniform, omit_padded=False is honest; also compute omit for padded bookkeeping when k used on pad
        ladder_rows.append(row)

    # Recompute Phase C ppl from embed_keep=7 row if present (live measurement)
    k7 = next((r for r in ladder_rows if r.get("embed_keep") == 7), None)
    if k7 is not None:
        phase_c_calib_ppl = k7["calib"]["ppl"]
        phase_c_held_ppl = k7["heldout"]["ppl"]
        phase_c_held_ret = k7["heldout"]["ppl_retention_vs_bf16"]
        phase_c_bpw = k7["est_total_bpw"]
        # Refresh vs_phase_c for other rows
        for r in ladder_rows:
            r["calib"]["ppl_retention_vs_phase_c"] = ppl_retention(phase_c_calib_ppl, r["calib"]["ppl"])
            r["heldout"]["ppl_retention_vs_phase_c"] = ppl_retention(phase_c_held_ppl, r["heldout"]["ppl"])

    # --- E2 Frequency tiers ---
    print("E2 token frequency counts (calib-v2 only) …", flush=True)
    freq = count_token_frequencies(tokenizer, calib_texts, max_length=args.max_length)
    # Optional heldout counts for reporting only
    freq_held = count_token_frequencies(tokenizer, held_texts, max_length=args.max_length)
    freq_report = {
        "calib_total_tokens": int(sum(freq.values())),
        "calib_unique_ids": len(freq),
        "heldout_total_tokens": int(sum(freq_held.values())),
        "heldout_unique_ids": len(freq_held),
        "heldout_ids_unseen_in_calib": len(set(freq_held) - set(freq)),
        "note": "Tiers fit on calib only; heldout counts are reporting-only.",
    }
    print(f"  {freq_report}", flush=True)

    tier_rows = []
    tier_summaries = []
    for sched in args.tier_schedules:
        print(f"E2 tier schedule={sched} …", flush=True)
        row_keeps, tmeta = assign_frequency_tiers(
            vocab_rows=vocab_rows,
            reachable=reachable,
            special_ids=special_ids,
            freq=freq,
            schedule=sched,
        )
        summary = compact_tier_summary(row_keeps, tmeta)
        # reachable-only avg
        r_idx = [i for i in range(vocab_rows) if i in reachable]
        summary["avg_row_keep_reachable_only"] = float(row_keeps[r_idx].astype(np.float64).mean())
        tier_summaries.append(summary)

        row = eval_config(
            model,
            tokenizer,
            baseline_sd=baseline_sd,
            row_keeps=row_keeps,
            embed_name=embed_name,
            calib_texts=calib_texts,
            held_texts=held_texts,
            base_calib=base_calib,
            base_held=base_held,
            phase_c_calib_ppl=phase_c_calib_ppl,
            phase_c_held_ppl=phase_c_held_ppl,
            max_length=args.max_length,
            label=f"tier_{sched}",
            padded_indices=padded,
            omit_padded=True,
        )
        row["tier_meta"] = tmeta
        row["schedule"] = sched
        tier_rows.append(row)

    shortcuts.append(
        "Tier head uses top_n_cap rank floor (expanded only if mass < target); "
        "avoids 1-id head when one token exceeds 1% mass on tiny calib."
    )
    shortcuts.append(
        "Calib-v2 sees only ~1.2k unique ids; vast majority of reachable rows are "
        "never-seen → tier maps ≈ uniform embed keep for that unseen band."
    )

    # Gate: pick best tier by heldout retention among those with ret>=gate if any,
    # else best heldout retention; also consider uniform if better BPW at gate.
    candidates = []
    for r in ladder_rows + tier_rows:
        candidates.append(
            {
                "name": r["name"],
                "heldout_ppl_retention": r["heldout"]["ppl_retention_vs_bf16"],
                "calib_ppl_retention": r["calib"]["ppl_retention_vs_bf16"],
                "est_total_bpw": r["est_total_bpw"],
            }
        )
    # Prefer maps that pass gate with lowest BPW; else best retention
    passing = [c for c in candidates if c["heldout_ppl_retention"] >= args.heldout_gate]
    if passing:
        # Lowest est BPW; tie-break on higher heldout retention
        primary = min(
            passing,
            key=lambda c: (round(c["est_total_bpw"], 2), -c["heldout_ppl_retention"]),
        )
        decision = "GO"
    else:
        primary = max(candidates, key=lambda c: c["heldout_ppl_retention"])
        decision = "NO-GO"
    # Also record best tier-only map for reporting
    tier_cands = [c for c in candidates if c["name"].startswith("tier_")]
    best_tier = (
        min(
            [c for c in tier_cands if c["heldout_ppl_retention"] >= args.heldout_gate] or tier_cands,
            key=lambda c: (round(c["est_total_bpw"], 2), -c["heldout_ppl_retention"]),
        )
        if tier_cands
        else None
    )

    # Honesty: only claim proxy GO if primary passes; ≤8 BPW product only if BPW supports
    claim_notes = []
    if decision == "GO" and primary["est_total_bpw"] <= 8.0:
        claim_notes.append(
            f"Proxy GO with est BPW {primary['est_total_bpw']:.2f} ≤ 8 — still not a physical container / product claim."
        )
    elif decision == "GO":
        claim_notes.append(
            f"Proxy GO on heldout retention but est BPW {primary['est_total_bpw']:.2f} is not a ≤8 BPW product claim."
        )
    else:
        claim_notes.append("NO-GO: no map achieved heldout ppl_retention ≥ 0.95 on this proxy set.")

    wall = round(time.perf_counter() - wall0, 2)
    payload = {
        "phase": "H95E",
        "model_dir": str(args.model_dir),
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_length": args.max_length,
        "calibration": {
            "version": calib_meta.get("version", "calib-v2"),
            "path": str(args.calib),
            "n_texts": len(calib_texts),
            "n_tokens_scored": base_calib["n_tokens"],
        },
        "heldout": {
            "version": held_meta.get("version", "heldout-v1"),
            "path": str(args.heldout),
            "n_texts": len(held_texts),
            "n_tokens_scored": base_held["n_tokens"],
        },
        "inventory_path": str(args.inv_out),
        "baseline_calib": {
            "mean_nll": base_calib["mean_nll"],
            "ppl": base_calib["ppl"],
            "n_tokens": base_calib["n_tokens"],
            "wall_seconds": base_calib["wall_seconds"],
        },
        "baseline_heldout": {
            "mean_nll": base_held["mean_nll"],
            "ppl": base_held["ppl"],
            "n_tokens": base_held["n_tokens"],
            "wall_seconds": base_held["wall_seconds"],
        },
        "phase_c_reference": {
            "heldout_ppl_retention": phase_c_held_ret,
            "est_total_bpw": phase_c_bpw,
            "note": "Phase C body + embed@7 (live k=7 ladder row when available)",
        },
        "freq_report": freq_report,
        "e1_uniform_ladder": ladder_rows,
        "e2_tiers": tier_rows,
        "tier_summaries": tier_summaries,
        "gate": {
            "metric": "heldout_ppl_retention",
            "threshold": args.heldout_gate,
            "primary_map": primary["name"],
            "heldout_ppl_retention": primary["heldout_ppl_retention"],
            "calib_ppl_retention": primary["calib_ppl_retention"],
            "est_total_bpw": primary["est_total_bpw"],
            "decision": decision,
            "go": decision == "GO",
            "claim_notes": claim_notes,
            "best_tier": best_tier,
        },
        "honesty": HONESTY_LINES,
        "shortcuts": shortcuts,
        "wall_seconds": wall,
        "env": {"torch": torch.__version__, "device": "cpu", "dtype": "bfloat16", "threads": 1},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Strip huge per_text if any slipped in — our rows don't include them
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    write_results_md(payload, args.md)
    print(
        f"\nDone. decision={decision} primary={primary['name']} "
        f"held_ret={primary['heldout_ppl_retention']:.4f} bpw≈{primary['est_total_bpw']:.2f} "
        f"wall={wall}s",
        flush=True,
    )
    print(f"Wrote {args.inv_out}, {args.inv_md}, {args.out}, {args.md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
