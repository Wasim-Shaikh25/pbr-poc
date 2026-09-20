#!/usr/bin/env python3
"""PBR-H95Q first recommended run (§18): Candidates A, B1, B2 + entropy diagnostics.

Optional stretch: tiny Candidate C sketch (K4 base + magnitude row recovery).

Honesty:
  - packed_K_total_bpw vs entropy_lb_total_bpw are estimates, not physical containers.
  - heldout ppl_retention ≥ 0.95 = proxy GO for that map only (not production bench).
  - Original BF16 need not round-trip; quantized reference MUST (idempotent Q).
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
    apply_keep_map_to_state_dict,
    apply_policy_to_model,
    bf16_tensor_to_u16,
    clone_cpu_state_dict,
    u16_to_bf16_tensor,
)
from pbr_h95.entropy_diag import diagnose_keep_map
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_h95.packed_rate import rate_report_for_keep_map
from pbr_h95.policy import (
    family_keep_breakdown,
    h95q_A_7_5_keep_fn,
    h95q_B1_mlp_k4_keep_fn,
    h95q_B2_embed_k5_keep_fn,
    h95q_policy_description,
    tensor_family,
    uniform_mid_keep_fn,
)
from pbr_h95.quantize import quantize_bf16_mantissas

DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")

HONESTY_LINES = [
    "Proxy PPL on in-repo calib-v2 / heldout-v1 only — not a production LM benchmark.",
    "heldout ppl_retention ≥ 0.95 = proxy GO for that map; else NO-GO. Not MMLU/HellaSwag.",
    "packed_K_total_bpw = 1 sign + 2.62 exp ref + avg packed K — not a physical container.",
    "entropy_lb_total_bpw replaces packed K with empirical H(retained symbols); codecs need table cost.",
    "Reject rANS unless gain_vs_packed > 0 after table cost (see entropy diagnostics).",
    "Original BF16 need not round-trip; quantized reference MUST (Q idempotent).",
    "Not a ≤8 BPW product claim unless packed rate and proxy quality both support it.",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    texts = [t["text"] for t in payload["texts"]]
    return payload, texts


def assert_quantizer_idempotence(
    baseline_sd: dict,
    keep_map: dict[str, int],
    *,
    max_sample: int = 8,
) -> dict:
    """Sample tensors: Q(Q(W))=Q(W) at each tensor's keep."""
    checked = 0
    failures = []
    # Prefer large / varied families
    names = [n for n in keep_map if n in baseline_sd and baseline_sd[n].is_floating_point()]
    # unique storage
    seen: set[int] = set()
    uniq = []
    for n in names:
        ptr = baseline_sd[n].data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        uniq.append(n)
    # sample: embed + a few mid + protected
    prefer = [n for n in uniq if "embed" in n.lower()]
    prefer += [n for n in uniq if "mlp" in n.lower()][:3]
    prefer += [n for n in uniq if "self_attn" in n.lower()][:2]
    prefer += [n for n in uniq if "norm" in n.lower()][:1]
    sample = []
    for n in prefer + uniq:
        if n not in sample:
            sample.append(n)
        if len(sample) >= max_sample:
            break
    for name in sample:
        k = int(keep_map[name])
        words = bf16_tensor_to_u16(baseline_sd[name])
        q1 = quantize_bf16_mantissas(words, k)
        q2 = quantize_bf16_mantissas(q1, k)
        ok = bool(np.array_equal(q1, q2))
        checked += 1
        if not ok:
            failures.append(name)
    return {
        "checked": checked,
        "all_idempotent": len(failures) == 0,
        "failures": failures,
        "sampled": sample,
    }


def build_keep_map(names: list[str], keep_fn) -> dict[str, int]:
    return {n: int(keep_fn(n)) for n in names}


def c_sketch_row_recovery_apply(
    model,
    *,
    baseline_sd,
    recover_frac: float = 0.05,
    base_mid_keep: int = 4,
    recover_keep: int = 7,
    protected: int = 7,
) -> dict:
    """Tiny C sketch: uniform mid@K4 + protected@7, then restore top-|w| mlp_mid rows to K7.

    Exception map cost estimated as 1 bit/row for selected mlp weight matrices
    (bitmap) + (recover_keep - base) extra mantissa bits on recovered rows.
    """
    keep_fn = uniform_mid_keep_fn(base_mid_keep, protected=protected)
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    keep_map = dict(meta["keep_map"])

    extra_mant_bits = 0
    map_bits = 0
    recovered_rows = 0
    recovered_words = 0
    mlp_rows_total = 0
    details = []

    sd = model.state_dict()
    seen: set[int] = set()
    for name, tensor in list(sd.items()):
        if not tensor.is_floating_point():
            continue
        if tensor_family(name) != "mlp_mid":
            continue
        if "bias" in name.lower():
            continue
        if tensor.ndim != 2:
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        # Work from baseline rows
        base = baseline_sd[name]
        w = base.detach().float().cpu()
        row_mag = w.abs().mean(dim=1)  # [out]
        n_rows = int(row_mag.numel())
        n_recover = max(1, int(round(recover_frac * n_rows)))
        top = torch.topk(row_mag, k=n_recover).indices.tolist()
        top_set = set(top)
        mlp_rows_total += n_rows
        recovered_rows += n_recover
        cols = int(w.shape[1])
        # Re-quantize: most rows at base_mid_keep from baseline; top rows at recover_keep
        words = bf16_tensor_to_u16(base)
        words2d = words.reshape(n_rows, cols)
        out = np.empty_like(words2d)
        for r in range(n_rows):
            k = recover_keep if r in top_set else base_mid_keep
            out[r] = quantize_bf16_mantissas(words2d[r], k)
            if r in top_set:
                extra_mant_bits += (recover_keep - base_mid_keep) * cols
                recovered_words += cols
        flat = out.reshape(-1)
        new_t = u16_to_bf16_tensor(flat, tensor).reshape(tensor.shape)
        with torch.no_grad():
            tensor.copy_(new_t.to(device=tensor.device, dtype=tensor.dtype))
        # Bitmap: 1 bit per row
        map_bits += n_rows
        details.append(
            {
                "name": name,
                "n_rows": n_rows,
                "n_recover": n_recover,
                "recover_frac_actual": n_recover / n_rows,
            }
        )
        # Mark keep_map as mixed (report base for tensor-level; detail in sketch)
        keep_map[name] = base_mid_keep

    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # Rate: start from body packed rate, add exception extras amortized over all model words
    body_rate = rate_report_for_keep_map(keep_map, baseline_sd)
    n_words = body_rate["n_words"]
    # Body avg K undercounts recovered rows — add extra_mant_bits / n_words
    adj_mant = body_rate["avg_packed_K"] + (extra_mant_bits / n_words if n_words else 0.0)
    map_bpw = (map_bits / n_words) if n_words else 0.0
    packed_total = 1.0 + 2.62 + adj_mant + map_bpw
    meta["keep_map"] = keep_map
    meta["c_sketch"] = {
        "recover_frac_target": recover_frac,
        "base_mid_keep": base_mid_keep,
        "recover_keep": recover_keep,
        "recovered_rows": recovered_rows,
        "mlp_rows_total": mlp_rows_total,
        "recovered_words": recovered_words,
        "extra_mantissa_bits": extra_mant_bits,
        "exception_map_bits_bitmap_rows": map_bits,
        "adj_avg_mant_bpw": adj_mant,
        "exception_map_bpw": map_bpw,
        "packed_K_total_bpw_with_map": round(packed_total, 4),
        "details": details,
        "note": (
            "Illustrative C sketch only — row bitmap map cost + extra mant bits. "
            "Not a full sparse-recovery codec."
        ),
    }
    meta["bytes_saved_vs_bf16_mantissa"] = (7.0 - adj_mant) * n_words / 8.0
    return meta


def eval_candidate(
    model,
    tokenizer,
    *,
    baseline_sd,
    keep_fn,
    calib_texts,
    held_texts,
    base_calib,
    base_held,
    max_length: int,
    label: str,
    description: str,
    entropy_sample_cap: int,
    do_entropy: bool,
    c_sketch: bool = False,
    recover_frac: float = 0.05,
) -> dict:
    t0 = time.perf_counter()
    if c_sketch:
        meta = c_sketch_row_recovery_apply(
            model, baseline_sd=baseline_sd, recover_frac=recover_frac
        )
        keep_map = meta["keep_map"]
    else:
        meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
        keep_map = meta["keep_map"]

    idem = assert_quantizer_idempotence(baseline_sd, keep_map)

    calib = mean_token_nll(model, tokenizer, calib_texts, max_length=max_length)
    held = mean_token_nll(model, tokenizer, held_texts, max_length=max_length)

    rate = rate_report_for_keep_map(keep_map, baseline_sd)
    fam_break = family_keep_breakdown(keep_map, baseline_sd)

    entropy = None
    if do_entropy:
        entropy = diagnose_keep_map(
            keep_map,
            baseline_sd,
            bf16_to_u16=bf16_tensor_to_u16,
            sample_words_cap=entropy_sample_cap,
        )
        rate = rate_report_for_keep_map(
            keep_map,
            baseline_sd,
            entropy_mant_bpw=entropy["avg_H_retained"],
        )

    if c_sketch and "c_sketch" in meta:
        packed_bpw = meta["c_sketch"]["packed_K_total_bpw_with_map"]
        avg_k = meta["c_sketch"]["adj_avg_mant_bpw"]
    else:
        packed_bpw = rate["packed_K_total_bpw"]
        avg_k = rate["avg_packed_K"]

    calib_ret = ppl_retention(base_calib["ppl"], calib["ppl"])
    held_ret = ppl_retention(base_held["ppl"], held["ppl"])
    proxy_go = bool(held_ret >= 0.95)

    row = {
        "name": label,
        "description": description,
        "avg_packed_K": avg_k,
        "packed_K_total_bpw": packed_bpw,
        "rate": rate,
        "family_breakdown": fam_break,
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
        "proxy_gate": {
            "heldout_threshold": 0.95,
            "heldout_ppl_retention": held_ret,
            "decision": "proxy_GO" if proxy_go else "proxy_NO_GO",
            "approaches_8_packed_bpw": bool(packed_bpw <= 8.0),
            "note": "proxy only — not production bench",
        },
        "idempotence": idem,
        "entropy": (
            {
                "avg_fixed_rate_K": entropy["avg_fixed_rate_K"],
                "avg_H_retained": entropy["avg_H_retained"],
                "entropy_vs_packed_gain": entropy["entropy_vs_packed_gain"],
                "entropy_lb_total_bpw": rate.get("entropy_lb_total_bpw"),
                "families": entropy["families"],
                "n_tensors_diag": len(entropy["tensors"]),
                "n_tensors_rans_accept": sum(
                    1 for t in entropy["tensors"] if t["rans_estimate"]["accept_rans"]
                ),
                "guidance": entropy["guidance"],
            }
            if entropy
            else None
        ),
        "c_sketch": meta.get("c_sketch"),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    ent_s = ""
    if row["entropy"]:
        ent_s = (
            f" H={row['entropy']['avg_H_retained']:.4f} "
            f"ent_bpw≈{row['entropy']['entropy_lb_total_bpw']} "
            f"rans_ok={row['entropy']['n_tensors_rans_accept']}"
        )
    print(
        f"  [{label}] packed≈{packed_bpw:.4f} avgK={avg_k:.4f} "
        f"calib_ret={calib_ret:.4f} held_ret={held_ret:.4f} "
        f"{row['proxy_gate']['decision']}{ent_s} ({row['wall_seconds']}s)",
        flush=True,
    )
    return row


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PBR-H95Q — Mixed-precision first run (A / B1 / B2)",
        "",
        f"Model: `{payload['repo']}` (local BF16, CPU)",
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
        "## Candidates",
        "",
        "| Candidate | packed BPW | avg K | entropy-lb BPW | calib ret | heldout ret | proxy | ≤8 packed? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for c in payload["candidates"]:
        ent = c["entropy"]["entropy_lb_total_bpw"] if c.get("entropy") else "—"
        lines.append(
            f"| {c['name']} | {c['packed_K_total_bpw']:.4f} | {c['avg_packed_K']:.4f} | "
            f"{ent} | {c['calib']['ppl_retention']:.4f} | {c['heldout']['ppl_retention']:.4f} | "
            f"{c['proxy_gate']['decision']} | {c['proxy_gate']['approaches_8_packed_bpw']} |"
        )
    lines += ["", "## Policy rules", ""]
    for c in payload["candidates"]:
        lines.append(f"- **{c['name']}**: {c['description']}")
        if c.get("c_sketch"):
            cs = c["c_sketch"]
            lines.append(
                f"  - C sketch: recover_frac={cs['recover_frac_target']}, "
                f"recovered_rows={cs['recovered_rows']}/{cs['mlp_rows_total']}, "
                f"map_bpw={cs['exception_map_bpw']:.6f}, "
                f"packed+map={cs['packed_K_total_bpw_with_map']}"
            )
    lines += ["", "## Family keep breakdown (word-weighted)", ""]
    for c in payload["candidates"]:
        lines.append(f"### {c['name']}")
        lines.append("")
        lines.append("| family | n_words | avg_keep |")
        lines.append("| --- | ---: | ---: |")
        for fam, slot in sorted(c["family_breakdown"].items()):
            lines.append(f"| {fam} | {slot['n_words']} | {slot['avg_keep']:.4f} |")
        lines.append("")
    lines += ["## Entropy diagnostics (summary)", ""]
    for c in payload["candidates"]:
        if not c.get("entropy"):
            continue
        e = c["entropy"]
        lines.append(
            f"- **{c['name']}**: avg H={e['avg_H_retained']:.4f} vs packed K={e['avg_fixed_rate_K']:.4f} "
            f"(gain {e['entropy_vs_packed_gain']:+.4f}); "
            f"rANS-accept tensors={e['n_tensors_rans_accept']}/{e['n_tensors_diag']}"
        )
        for fam, fs in sorted(e["families"].items()):
            lines.append(
                f"  - {fam}: H={fs['avg_H_retained']:.4f} K={fs['avg_fixed_rate_K']:.4f} "
                f"gain={fs['entropy_vs_packed_gain']:+.4f} rans_ok={fs['n_tensors_rans_accept']}"
            )
    lines += [
        "",
        "## Idempotence",
        "",
    ]
    for c in payload["candidates"]:
        idem = c["idempotence"]
        lines.append(
            f"- **{c['name']}**: checked={idem['checked']} all_ok={idem['all_idempotent']}"
        )
    lines += [
        "",
        "## Summary decision",
        "",
        f"- Any candidate ≤8 packed BPW: **{payload['summary']['any_leq_8_packed']}**",
        f"- Proxy-GO candidates: {payload['summary']['proxy_go_names']}",
        f"- Best packed among proxy-GO: {payload['summary']['best_proxy_go']}",
        f"- Shortcuts: {payload['summary']['shortcuts']}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pbr_h95/h95q_qwen.json"))
    ap.add_argument("--md", type=Path, default=Path("artifacts/pbr_h95/h95q_qwen.md"))
    ap.add_argument("--skip-entropy", action="store_true")
    ap.add_argument(
        "--entropy-sample-cap",
        type=int,
        default=250_000,
        help="Max words sampled per tensor for entropy hist (0=all)",
    )
    ap.add_argument(
        "--with-c-sketch",
        action="store_true",
        help="Also run tiny Candidate C sketch (K4 base + magnitude row recover)",
    )
    ap.add_argument("--c-recover-frac", type=float, default=0.05)
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

    baseline_sd = clone_cpu_state_dict(model)
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

    candidates_spec = [
        ("h95q_A_7_5", h95q_A_7_5_keep_fn(), False),
        ("h95q_B1_mlp_k4", h95q_B1_mlp_k4_keep_fn(), False),
        ("h95q_B2_embed_k5", h95q_B2_embed_k5_keep_fn(), False),
    ]
    if args.with_c_sketch:
        candidates_spec.append(
            ("h95q_C_sketch_k4_row_recover", None, True),
        )

    results = []
    do_entropy = not args.skip_entropy
    for label, keep_fn, is_c in candidates_spec:
        print(f"Running {label} …", flush=True)
        row = eval_candidate(
            model,
            tokenizer,
            baseline_sd=baseline_sd,
            keep_fn=keep_fn,
            calib_texts=calib_texts,
            held_texts=held_texts,
            base_calib=base_calib,
            base_held=base_held,
            max_length=args.max_length,
            label=label,
            description=h95q_policy_description(label),
            entropy_sample_cap=args.entropy_sample_cap,
            do_entropy=do_entropy,
            c_sketch=is_c,
            recover_frac=args.c_recover_frac,
        )
        results.append(row)

    proxy_go = [c for c in results if c["proxy_gate"]["decision"] == "proxy_GO"]
    any_leq8 = any(c["proxy_gate"]["approaches_8_packed_bpw"] for c in results)
    best = None
    if proxy_go:
        best = min(proxy_go, key=lambda c: c["packed_K_total_bpw"])
        best = {
            "name": best["name"],
            "packed_K_total_bpw": best["packed_K_total_bpw"],
            "heldout_ppl_retention": best["heldout"]["ppl_retention"],
        }

    shortcuts = []
    if args.entropy_sample_cap:
        shortcuts.append(f"entropy subsampled to ≤{args.entropy_sample_cap} words/tensor")
    if not args.with_c_sketch:
        shortcuts.append("Candidate C/D full codecs not implemented (optional C sketch off)")
    else:
        shortcuts.append("tiny C sketch only (row-magnitude recover); no D codecs")
    shortcuts.append("rANS is table-cost estimate only — no real encode bytes")

    payload = {
        "phase": "H95Q",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_dir": str(args.model_dir),
        "max_length": args.max_length,
        "calib_path": str(args.calib),
        "heldout_path": str(args.heldout),
        "calib_tokens": base_calib["n_tokens"],
        "heldout_tokens": base_held["n_tokens"],
        "honesty": HONESTY_LINES,
        "bf16": {"calib": base_calib, "heldout": base_held},
        "candidates": results,
        "summary": {
            "any_leq_8_packed": any_leq8,
            "proxy_go_names": [c["name"] for c in proxy_go],
            "best_proxy_go": best,
            "candidate_A_packed_bpw": next(
                (c["packed_K_total_bpw"] for c in results if c["name"] == "h95q_A_7_5"), None
            ),
            "shortcuts": shortcuts,
        },
        "wall_seconds": round(time.perf_counter() - wall0, 2),
        "disclaimer": (
            "H95Q first run: packed-K candidates A/B1/B2 + entropy diagnostics. "
            "Proxy GO/NO-GO on heldout-v1 only. Not a physical container or ≤8 product claim."
        ),
    }

    # Drop huge per-text arrays from bf16 for smaller artifact
    for side in ("calib", "heldout"):
        if "per_text" in payload["bf16"][side]:
            del payload["bf16"][side]["per_text"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(payload, args.md)
    print(f"Wrote {args.out} and {args.md}", flush=True)
    print(
        f"DONE wall={payload['wall_seconds']}s A_bpw={payload['summary']['candidate_A_packed_bpw']} "
        f"proxy_GO={payload['summary']['proxy_go_names']} any≤8={any_leq8}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
