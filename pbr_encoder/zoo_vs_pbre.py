"""Explain the zoo ~10.585 vs PBR-E ~10.616 gap from bakeoff artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BAKEOFF = Path("artifacts/mantissa_multimodel_bakeoff.json")
PBRE_MEASURED = 10.6161  # Job 3 rANS complete container, 42 tensors, all words
PBRE_MEASURED_BYTES = 111253029
DISCLAIMER = (
    "Zoo vs PBR-E accounting. Held-out NLL is not a bitstream. "
    "Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
)


def analyze(report: dict) -> dict:
    s = report["summary"]
    n_all = int(s["n_words"])
    n_hold = int(s["n_hold"])
    methods = {m["name"]: m for m in s["methods"]}
    mix = methods["mixture_argmin"]
    exp_cond = methods["phase_a_H(M|exp)"]
    raw_m = methods["raw_M"]
    uncond = methods["uncond_rANS"]

    choices = s.get("per_tensor_mixture") or []
    by_mode: Counter[str] = Counter()
    hold_words: dict[str, int] = Counter()
    for row in choices:
        ch = row["choice"]
        by_mode[ch] += 1
        hold_words[ch] += int(row["n_hold"])

    sign = float(s["sign_bpw_raw"])
    exp_bpw = float(s["exp_complete_bpw"])
    hdr = float(s["header_bpw"])
    pbre_split = float(s["pbre_total_bpw"])
    zoo_total = float(mix["total_complete_bpw"])
    hold_frac = n_hold / max(n_all, 1)

    words_by_choice: dict[str, int] = {}
    bytes_by_choice: dict[str, int] = {}
    for name, n_h in hold_words.items():
        n_w = int(round(n_h / hold_frac)) if hold_frac else n_h
        words_by_choice[name] = n_w
        bytes_by_choice[name] = n_w * 2

    def _bits_to_bytes(bpw: float) -> float:
        return bpw * n_all / 8.0

    # Reconstruct holdout-charged zoo total (tables dumped on 20% only).
    zoo_hold_total = sign + exp_bpw + float(mix["holdout_charged_bpw"]) + hdr
    # vs measured full-set PBR-E rANS container
    vs_measured_codec = zoo_total - PBRE_MEASURED
    vs_measured_hold = zoo_hold_total - PBRE_MEASURED

    # Mantissa-only gap vs raw 7.
    mant_ideal_gap = 7.0 - float(mix["ideal_bpw"])
    mant_complete_gap = 7.0 - float(mix["complete_bpw"])

    # If mantissa NLL were realized as a bitstream with the same +0.003
    # rANS overhead seen on exponents (Job 3).
    rans_overhead = 10.6161 - 10.6127
    zoo_if_rans = zoo_total + rans_overhead

    orig_b = 2 * n_all
    packed_sm_bytes = n_all  # 8 raw bits: 1 sign + 7 mantissa
    return {
        "disclaimer": DISCLAIMER,
        "n_tensors": int(s["n_tensors"]),
        "n_words": n_all,
        "n_hold": n_hold,
        "hold_frac": n_hold / max(n_all, 1),
        "mixture_winner": {
            "name": mix["name"],
            "models_used": list(s.get("mixture_used") or []),
            "model_bytes": int(mix["model_bytes"]),
            "model_byte_breakdown": {
                "exp_cond_tables_256x128_u16": int(s["model_bytes"]["exp_cond"]),
                "gbdt_hist_16x2": int(s["model_bytes"]["gbdt"]),
            },
            "tensors_by_choice": dict(by_mode),
            "holdout_words_by_choice": dict(hold_words),
            "holdout_word_share_by_choice": {
                k: hold_words[k] / max(n_hold, 1) for k in hold_words
            },
            "original_words_by_choice": words_by_choice,
            "original_bytes_by_choice": bytes_by_choice,
            "tensors": choices,
        },
        "breakdown": {
            "zoo_mixture_codec_view": {
                "sign_raw_bpw": sign,
                "sign_raw_bytes": _bits_to_bytes(sign),
                "exp_complete_bpw": exp_bpw,
                "exp_complete_bytes": _bits_to_bytes(exp_bpw),
                "mantissa_complete_bpw": float(mix["complete_bpw"]),
                "mantissa_complete_bytes": _bits_to_bytes(float(mix["complete_bpw"])),
                "mantissa_ideal_nll_bpw": float(mix["ideal_bpw"]),
                "headers_bpw": hdr,
                "headers_bytes": _bits_to_bytes(hdr),
                "tables_bytes": int(mix["model_bytes"]),
                "tables_codec_view_bpw": 8.0 * int(mix["model_bytes"]) / max(n_all, 1),
                "total_bpw": zoo_total,
                "total_bytes_codec_view": _bits_to_bytes(zoo_total),
                "mantissa_nll_bits": float(mix["nll_bits"]),
                "tables_amortized_bits": 8.0 * int(mix["model_bytes"]) * hold_frac,
            },
            "pbre_split_synthetic": {
                "note": (
                    "Sign raw + exp rANS-NLL complete + mantissa stored raw 7 bits + "
                    "amortized 17-byte headers. NOT the on-disk PBR-E container."
                ),
                "sign_raw_bpw": sign,
                "sign_raw_bytes": _bits_to_bytes(sign),
                "exp_complete_bpw": exp_bpw,
                "exp_complete_bytes": _bits_to_bytes(exp_bpw),
                "mantissa_raw_bpw": 7.0,
                "mantissa_raw_bytes": _bits_to_bytes(7.0),
                "headers_bpw": hdr,
                "headers_bytes": _bits_to_bytes(hdr),
                "total_bpw": pbre_split,
                "total_bytes": _bits_to_bytes(pbre_split),
            },
            "pbre_measured_fullset": {
                "note": "Job 3 rANS complete container on the same 42 tensors, all words.",
                "encoded_bytes": PBRE_MEASURED_BYTES,
                "original_bytes": orig_b,
                "bpw": PBRE_MEASURED,
                "entropy_bound_bpw": 10.6127,
                "packed_sign_mantissa_bytes": packed_sm_bytes,
                "exp_plus_headers_bytes": PBRE_MEASURED_BYTES - packed_sm_bytes,
                "sign_raw_bpw": 1.0,
                "mantissa_raw_bpw": 7.0,
                "exp_bitstream_bpw_approx": 8.0 * (PBRE_MEASURED_BYTES - packed_sm_bytes) / max(n_all, 1),
            },
        },
        "deltas": {
            "zoo_vs_pbre_split": zoo_total - pbre_split,
            "zoo_vs_pbre_measured": vs_measured_codec,
            "zoo_holdout_charged_total_bpw": zoo_hold_total,
            "zoo_holdout_charged_vs_measured": vs_measured_hold,
            "mantissa_ideal_vs_raw7": -mant_ideal_gap,
            "mantissa_complete_vs_raw7": -mant_complete_gap,
            "exp_cond_vs_mixture": float(exp_cond["total_complete_bpw"]) - zoo_total,
            "uncond_M_vs_raw7_complete": float(uncond["complete_bpw"]) - 7.0,
            "zoo_plus_exp_rans_overhead_bpw": zoo_if_rans,
        },
        "accounting_verdict": {
            "is_0_03_real": (
                "The 0.03–0.06 BPW is a real mantissa entropy gap (H(M|exp)≈6.93 vs raw 7), "
                "not random noise. It is NOT a like-for-like container comparison: zoo totals "
                "use held-out NLL plus amortized tables; PBR-E 10.616 is an on-disk bitstream "
                "over all words. Charge the 67 KB tables to the 20% holdout only and the zoo "
                "total is ~10.611, within 0.005 of measured PBR-E. Realizing NLL as rANS would "
                "add ~0.003 BPW (the exponent overhead)."
            ),
            "headline_0_062_is": "vs synthetic split PBR-E (raw-M 7.0), codec-view amortization",
            "headline_0_031_is": "vs measured PBR-E rANS 10.616, still NLL not bitstream",
            "transferable_trick": (
                "Entropy-code the 7-bit mantissa, preferably 256 exp-conditional tables "
                "(DF11-class, same move already used on exponents). ~0.06 BPW vs raw-M, "
                "~0.03 vs current PBR-E. Not a new axis."
            ),
            "toward_8bpw": "dead-end micro-gain",
            "bits_needed_for_8bpw": 10.616 - 8.0,
            "bits_found": mant_complete_gap,
        },
        "pbre_ref_fullset": float(s["pbre_ref_fullset"]),
        "raw_m_total_split": float(raw_m["total_complete_bpw"]),
    }


def format_markdown(a: dict) -> str:
    mix = a["mixture_winner"]
    z = a["breakdown"]["zoo_mixture_codec_view"]
    split = a["breakdown"]["pbre_split_synthetic"]
    meas = a["breakdown"]["pbre_measured_fullset"]
    d = a["deltas"]
    lines = [
        "# Zoo vs PBR-E: why 10.585 vs 10.616",
        "",
        DISCLAIMER,
        "",
        "## Headline",
        "",
        "The mantissa zoo did **not** find a new ~8 BPW codec. It found the DF11 "
        "move already used on exponents, applied to the mantissa: a 256×128 "
        f"exp-conditional table plus a 1.6 KB GBDT. Codec-view total **{z['total_bpw']:.4f} BPW** "
        f"vs synthetic raw-M PBR-E **{split['total_bpw']:.4f}** (Δ **{d['zoo_vs_pbre_split']:+.4f}**) "
        f"and vs measured PBR-E rANS **{meas['bpw']:.4f}** (Δ **{d['zoo_vs_pbre_measured']:+.4f}**). "
        "Toward 8 BPW (2.6 bits missing): **dead-end micro-gain**.",
        "",
        "## Who won how many tensors",
        "",
        "Per-tensor argmin on held-out mantissa NLL. Only two models were ever selected; "
        "the mixture pays the union of their tables "
        f"({mix['model_bytes']} B = {mix['model_byte_breakdown']['exp_cond_tables_256x128_u16']} "
        f"+ {mix['model_byte_breakdown']['gbdt_hist_16x2']}).",
        "",
        "| choice | tensors | orig words | orig bytes | holdout words | holdout share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, n_t in sorted(mix["tensors_by_choice"].items(), key=lambda kv: -kv[1]):
        w = mix["holdout_words_by_choice"][name]
        sh = mix["holdout_word_share_by_choice"][name]
        ow = mix["original_words_by_choice"][name]
        ob = mix["original_bytes_by_choice"][name]
        lines.append(
            f"| `{name}` | {n_t} / {a['n_tensors']} | {ow} | {ob} | {w} | {100 * sh:.1f}% |"
        )
    gbdt_names = [t["name"] for t in mix["tensors"] if t["choice"] == "gbdt"]
    lines += [
        "",
        "GBDT wins are small attention projections (k/q/v), not the MLP walls. "
        f"Names: {', '.join('`' + n + '`' for n in gbdt_names)}.",
        "",
        "CTW, PPM, tiny AR, and IDF-lite won **0** tensors in the mixture.",
        "",
        "## Exact breakdown",
        "",
        "| piece | zoo mixture (codec-view) | PBR-E split (raw M) | PBR-E measured rANS |",
        "| --- | ---: | ---: | ---: |",
        f"| sign | {z['sign_raw_bpw']:.4f} raw ({z['sign_raw_bytes']:.0f} B) | {split['sign_raw_bpw']:.4f} raw | 1.000 in packed SM |",
        f"| exponent | {z['exp_complete_bpw']:.4f} NLL+table ({z['exp_complete_bytes']:.0f} B) | {split['exp_complete_bpw']:.4f} NLL+table | ~{meas['exp_bitstream_bpw_approx']:.4f} bitstream ({meas['exp_plus_headers_bytes']} B w/ headers) |",
        f"| mantissa | {z['mantissa_complete_bpw']:.4f} NLL+table ({z['mantissa_complete_bytes']:.0f} B) | {split['mantissa_raw_bpw']:.4f} raw ({split['mantissa_raw_bytes']:.0f} B) | 7.000 raw |",
        f"| packed SM | (sign+mant separate) | (sign+mant separate) | {meas['packed_sign_mantissa_bytes']} B |",
        f"| tables | {z['tables_bytes']} B ({z['tables_codec_view_bpw']:.4f} BPW, in mant) | 0 (M raw) | exp table in exp stream |",
        f"| headers | {z['headers_bpw']:.4f} ({z['headers_bytes']:.0f} B) | {split['headers_bpw']:.4f} | tile + JSON |",
        f"| **total** | **{z['total_bpw']:.4f}** ({z['total_bytes_codec_view']:.0f} B) | **{split['total_bpw']:.4f}** ({split['total_bytes']:.0f} B) | **{meas['bpw']:.4f}** ({meas['encoded_bytes']} B) |",
        "",
        f"Measured PBR-E rANS container: **{meas['encoded_bytes']} B** / {a['n_words']} words "
        f"= {meas['bpw']:.4f} BPW (entropy bound {meas['entropy_bound_bpw']:.4f}).",
        "",
        "## Is 0.03 BPW real or accounting?",
        "",
        a["accounting_verdict"]["is_0_03_real"],
        "",
        f"- Headline **{d['zoo_vs_pbre_split']:+.4f} BPW** is {a['accounting_verdict']['headline_0_062_is']}.",
        f"- **{d['zoo_vs_pbre_measured']:+.4f} BPW** is {a['accounting_verdict']['headline_0_031_is']}.",
        f"- Holdout-charged zoo total **{d['zoo_holdout_charged_total_bpw']:.4f}** vs measured "
        f"{d['zoo_holdout_charged_vs_measured']:+.4f} BPW.",
        f"- Mantissa ideal gap vs raw 7: **{d['mantissa_ideal_vs_raw7']:+.4f}** BPW; "
        f"after amortized tables **{d['mantissa_complete_vs_raw7']:+.4f}**.",
        f"- Mixture vs H(M\\|exp) alone: **{d['exp_cond_vs_mixture']:+.4f} BPW** (GBDT is a rounding error).",
        f"- Adding the Job-3 rANS overhead to zoo NLL: **{d['zoo_plus_exp_rans_overhead_bpw']:.4f} BPW**.",
        "",
        "Zoo scores **held-out last 20% of rows** (`n_hold="
        f"{a['n_hold']}`). PBR-E 10.616 encodes **all** `{a['n_words']}` words. "
        "Complete zoo BPW amortizes one global table over the full set "
        "(`8 × model_bytes / n_all` added to holdout NLL/n_hold). That is honest as a "
        "*codec view*, not as an on-disk file of the holdout.",
        "",
        "## Transferable trick vs 8 BPW",
        "",
        a["accounting_verdict"]["transferable_trick"],
        "",
        f"Bits needed to go from measured PBR-E {meas['bpw']:.3f} to 8.0: "
        f"**{a['accounting_verdict']['bits_needed_for_8bpw']:.3f}**. "
        f"Bits found in the zoo: **{a['accounting_verdict']['bits_found']:.4f}**. "
        "Two orders of magnitude short. CTW/AR/IDF/PBR-4 did not add any.",
        "",
        "**Verdict:** dead-end micro-gain for an 8 GB → 4 GB bit-exact target. "
        "Worth shipping as an optional DF11 mantissa rANS in the PBR-E product "
        "(~0.03–0.06 BPW), not as a new research axis.",
        "",
        "This is not a 1–2 GB / 8 GB result and not ≤4 BPW.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    report_path: Path = BAKEOFF,
    md_path: Path = Path("artifacts/zoo_vs_pbre_analysis.md"),
    json_path: Path = Path("artifacts/zoo_vs_pbre_analysis.json"),
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = analyze(report)
    json_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(analysis), encoding="utf-8")
    return analysis


def main(argv: list[str] | None = None) -> int:
    del argv
    analysis = write_artifacts()
    print(format_markdown(analysis))
    print("Wrote artifacts/zoo_vs_pbre_analysis.md and artifacts/zoo_vs_pbre_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
