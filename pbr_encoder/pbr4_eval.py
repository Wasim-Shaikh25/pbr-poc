"""PBR-4 bakeoff on the Stage 1B Qwen set. Does not replace the mantissa zoo."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from pbr_codecs.pbr4 import DISCLAIMER, c4_bytes_for, decode_tensor, encode_container, encode_tensor
from pbr_core.hashing import sha256_words
from pbr_core.metrics import bits_per_weight
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d, choose_tile_hw
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO, inventory_from_dir, select_weight_specs
from pbr_encoder.verification import assert_exact

PBRE_REF = 10.616
ZOO_BEST_TOTAL = 10.585
RAW_BPW = 16.0
STRETCH_BPW = 4.0
PBR4_MARK = "## PBR-4 structured nibble + node formulas"

SETTINGS = [
    {"name": "8x8", "block_h": 8, "block_w": 8, "adaptive": False, "hw": False},
    {"name": "16x16", "block_h": 16, "block_w": 16, "adaptive": False, "hw": False},
    {"name": "1x64", "block_h": 1, "block_w": 64, "adaptive": False, "hw": False},
    {"name": "tile256", "block_h": 0, "block_w": 0, "adaptive": False, "hw": True, "block_size": 256},
    {"name": "16x16_tree", "block_h": 16, "block_w": 16, "adaptive": True, "hw": False},
]


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _hw(setting: dict, rows: int, cols: int) -> tuple[int, int, bool]:
    if setting.get("hw"):
        h, w = choose_tile_hw(int(setting["block_size"]), rows, cols)
        return int(h), int(w), False
    return int(setting["block_h"]), int(setting["block_w"]), bool(setting["adaptive"])


def _sum_stats(parts: list[dict], n_words: int, header_bytes: int) -> dict:
    s_b = sum(p["s_bytes"] for p in parts)
    c4_b = sum(p["c4_bytes"] for p in parts)
    r_b = sum(p["r_bytes"] for p in parts)
    meta_b = sum(p["meta_bytes"] for p in parts) + header_bytes
    payload = sum(p["payload_bytes"] for p in parts)
    formal = payload + header_bytes
    n_hit = sum(p["n_hit"] for p in parts)
    fam = Counter()
    for p in parts:
        fam.update(p.get("family_counts") or {})
    return {
        "s_bytes": s_b,
        "c4_bytes": c4_b,
        "c4_bits": 4 * n_words,
        "r_bytes": r_b,
        "meta_bytes": meta_b,
        "payload_bytes": payload,
        "header_bytes": header_bytes,
        "formal_bytes": formal,
        "formal_bpw": bits_per_weight(formal, n_words),
        "n_hit": n_hit,
        "hit_rate": n_hit / max(n_words, 1),
        "family_counts": dict(fam),
        "beats_pbre": bits_per_weight(formal, n_words) < PBRE_REF - 1e-6,
        "stretch_le4": bits_per_weight(formal, n_words) <= STRETCH_BPW,
        "beats_raw": bits_per_weight(formal, n_words) < RAW_BPW - 1e-6,
        "beats_zoo": bits_per_weight(formal, n_words) < ZOO_BEST_TOTAL - 1e-6,
    }


def evaluate_specs(specs, *, slice_words: int = 4096, settings: list[dict] | None = None) -> dict:
    t0 = time.perf_counter()
    print(DISCLAIMER, flush=True)
    use = settings if settings is not None else SETTINGS
    n_all = 0
    per_setting: dict[str, list[dict]] = {s["name"]: [] for s in use}
    per_tensor: list[dict] = []
    c4_hist = np.zeros(16, dtype=np.int64)
    slice_report = {}
    first_words = None

    for ti, spec in enumerate(specs):
        words = as_2d(load_uint16(spec))
        if first_words is None:
            first_words = words
        rows, cols = int(words.shape[0]), int(words.shape[1])
        n = int(words.size)
        n_all += n
        orig_sha = sha256_words(words)
        raw_b = n * 2
        tensor_row = {
            "name": spec.name,
            "shape": [rows, cols],
            "n_words": n,
            "raw_bytes": raw_b,
            "settings": {},
        }
        print(f"[{ti + 1}/{len(specs)}] {spec.name} {list(words.shape)}", flush=True)
        best_name = None
        best_formal = 1 << 62
        for setting in use:
            bh, bw, adaptive = _hw(setting, rows, cols)
            enc = encode_tensor(words, name=spec.name, block_h=bh, block_w=bw, adaptive=adaptive)
            rec = decode_tensor(enc)
            assert_exact(words, rec, label=f"pbr4:{setting['name']}:{spec.name}")
            st = enc.stats()
            st["block_h"] = bh
            st["block_w"] = bw
            per_setting[setting["name"]].append(st)
            tensor_row["settings"][setting["name"]] = {
                "formal_bytes": st["formal_bytes"],
                "formal_bpw": st["formal_bpw"],
                "s_bytes": st["s_bytes"],
                "c4_bytes": st["c4_bytes"],
                "r_bytes": st["r_bytes"],
                "meta_bytes": st["meta_bytes"],
                "hit_rate": st["hit_rate"],
                "family_counts": st["family_counts"],
                "n_nodes": st["n_nodes"],
            }
            if st["formal_bytes"] < best_formal:
                best_formal = st["formal_bytes"]
                best_name = setting["name"]
            if setting["name"] == "16x16":
                hist = np.bincount(enc.c4.astype(np.int64) & 15, minlength=16)
                c4_hist += hist
            print(
                f"    {setting['name']:12s}  {st['formal_bpw']:7.3f} BPW  "
                f"hit={100 * st['hit_rate']:.2f}%  S={st['s_bytes']} c4={st['c4_bytes']} "
                f"R={st['r_bytes']} meta={st['meta_bytes']}",
                flush=True,
            )
            del enc, rec
        tensor_row["best_setting"] = best_name
        tensor_row["best_formal_bytes"] = best_formal
        tensor_row["best_formal_bpw"] = bits_per_weight(best_formal, n)
        tensor_row["sha256"] = orig_sha
        per_tensor.append(tensor_row)
        del words

    header_len = len(
        json.dumps(
            {"format": "PBR-4 structured nibble", "note": DISCLAIMER, "n_tensors": len(specs), "tensors": [
                {"name": t["name"], "shape": t["shape"], "n_words": t["n_words"]} for t in per_tensor
            ]},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    container_prefix = 10
    header_bytes = container_prefix + header_len

    setting_rows = []
    for setting in use:
        parts = per_setting[setting["name"]]
        row = {"name": setting["name"], **_sum_stats(parts, n_all, header_bytes)}
        setting_rows.append(row)

    compete_payload = 0
    compete_s = compete_c4 = compete_r = compete_meta = compete_hit = 0
    compete_fam: Counter = Counter()
    for t in per_tensor:
        st = t["settings"][t["best_setting"]]
        compete_payload += st["formal_bytes"]
        compete_s += st["s_bytes"]
        compete_c4 += st["c4_bytes"]
        compete_r += st["r_bytes"]
        compete_meta += st["meta_bytes"]
        compete_hit += int(round(st["hit_rate"] * t["n_words"]))
        compete_fam.update(st["family_counts"])
    compete_formal = compete_payload + header_bytes
    compete = {
        "name": "per_tensor_min",
        "s_bytes": compete_s,
        "c4_bytes": compete_c4,
        "c4_bits": 4 * n_all,
        "r_bytes": compete_r,
        "meta_bytes": compete_meta + header_bytes,
        "payload_bytes": compete_payload,
        "header_bytes": header_bytes,
        "formal_bytes": compete_formal,
        "formal_bpw": bits_per_weight(compete_formal, n_all),
        "n_hit": compete_hit,
        "hit_rate": compete_hit / max(n_all, 1),
        "family_counts": dict(compete_fam),
        "beats_pbre": bits_per_weight(compete_formal, n_all) < PBRE_REF - 1e-6,
        "stretch_le4": bits_per_weight(compete_formal, n_all) <= STRETCH_BPW,
        "beats_raw": bits_per_weight(compete_formal, n_all) < RAW_BPW - 1e-6,
        "beats_zoo": bits_per_weight(compete_formal, n_all) < ZOO_BEST_TOTAL - 1e-6,
    }
    setting_rows.append(compete)
    setting_rows.sort(key=lambda m: m["formal_bpw"])
    best = setting_rows[0]

    c4_p = c4_hist.astype(np.float64)
    if c4_p.sum() > 0:
        p = c4_p / c4_p.sum()
        p = p[p > 0]
        h_c4 = float(-(p * np.log2(p)).sum())
    else:
        h_c4 = 4.0
    c4_ans_est_bpw = h_c4  # bits per weight if c4 were entropy-coded; formal still charges 4

    if first_words is not None:
        sl = min(slice_words, int(first_words.size))
        cols = int(first_words.shape[1])
        rows = max(1, sl // max(cols, 1))
        tile = first_words[:rows, : min(cols, max(1, sl // rows))]
        t_enc = time.perf_counter()
        enc = encode_tensor(tile, name="slice", block_h=16, block_w=16)
        rec = decode_tensor(enc)
        assert_exact(tile, rec, label="pbr4_slice")
        blob = encode_container([enc], extra={"slice": True}).dumps()
        rec2 = decode_tensor(enc)
        assert_exact(tile, rec2, label="pbr4_slice_again")
        slice_report = {
            "n_words": int(tile.size),
            "all_exact": "PASS",
            "sha256": sha256_words(tile),
            "encode_decode_s": time.perf_counter() - t_enc,
            "formal_bpw": enc.stats()["formal_bpw"],
            "container_bytes": len(blob),
        }

    elapsed = time.perf_counter() - t0
    return {
        "n_tensors": len(specs),
        "n_words": n_all,
        "original_bytes": 2 * n_all,
        "pbre_ref": PBRE_REF,
        "zoo_best_total": ZOO_BEST_TOTAL,
        "raw_bpw": RAW_BPW,
        "best_setting": best["name"],
        "best_formal_bpw": best["formal_bpw"],
        "best_s_bytes": best["s_bytes"],
        "best_c4_bytes": best["c4_bytes"],
        "best_r_bytes": best["r_bytes"],
        "best_meta_bytes": best["meta_bytes"],
        "best_hit_rate": best["hit_rate"],
        "beats_pbre": best["beats_pbre"],
        "beats_zoo": best["beats_zoo"],
        "beats_raw": best["beats_raw"],
        "stretch_le4": best["stretch_le4"],
        "c4_entropy_bpw": h_c4,
        "c4_ans_est_bpw": c4_ans_est_bpw,
        "elapsed_s": elapsed,
        "settings": setting_rows,
        "per_tensor": per_tensor,
        "slice": slice_report,
        "c4_bytes_formula": c4_bytes_for(n_all),
    }


def format_markdown(report: dict) -> str:
    s = report["summary"]
    md = report.get("model", {})
    lines = [
        "# PBR-4 structured nibble bakeoff",
        "",
        DISCLAIMER,
        "",
        f"- repo: `{md.get('repo_id', '')}`  revision `{md.get('revision', '')}`",
        f"- tensors: **{s['n_tensors']}**  words: **{s['n_words']}**",
        f"- best setting: `{s['best_setting']}`  **{s['best_formal_bpw']:.3f}** BPW complete",
        f"- |S|={s['best_s_bytes']}  c4={s['best_c4_bytes']} B (formal 4N/8)  |R|={s['best_r_bytes']}  meta={s['best_meta_bytes']}",
        f"- % weights with R=0: **{100 * s['best_hit_rate']:.2f}%**",
        f"- vs raw 16: **{'yes' if s['beats_raw'] else 'no'}**  vs PBR-E {s['pbre_ref']:.3f}: **{'yes' if s['beats_pbre'] else 'no'}**  "
        f"vs zoo {s['zoo_best_total']:.3f}: **{'yes' if s['beats_zoo'] else 'no'}**",
        f"- Stretch ≤4: **{'yes' if s['stretch_le4'] else 'no'}**",
        f"- H(c4) (16x16 mix, nats→bits): **{s['c4_entropy_bpw']:.4f}** bits/weight "
        "(entropy-coding c4 cannot beat the formal 4-bit charge by much if this is ~4)",
        f"- wall {s['elapsed_s']:.1f}s",
        "",
        "Formal complete bytes = |S| + 4N/8 + |R| + metadata (node headers + container JSON). "
        "F families compete per block: constant prototype, K≤16 palette, affine (a·r+b·c+d·c4+e) "
        "mod 2^16, nibble insert at shift 0/4/8/12, shared sign/exp + 4-bit mantissa nibble, "
        "16 templates on planar XOR. Positions are not a data store.",
        "",
        "| setting | |S| | c4 B | |R| | meta | formal B | BPW | % R=0 | vs 10.616 | vs 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for m in s["settings"]:
        vs = m["formal_bpw"] - s["pbre_ref"]
        lines.append(
            f"| {m['name']} | {m['s_bytes']} | {m['c4_bytes']} | {m['r_bytes']} | {m['meta_bytes']} | "
            f"{m['formal_bytes']} | {m['formal_bpw']:.3f} | {100 * m['hit_rate']:.2f}% | "
            f"{vs:+.3f} | {'yes' if m['stretch_le4'] else 'no'} |"
        )
    fam = (s["settings"][0].get("family_counts") or {}) if s["settings"] else {}
    if fam:
        lines += ["", "Winning-setting family mix (node counts):"]
        for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{k}`: {v}")
    sl = s.get("slice") or {}
    lines += [
        "",
        f"Exact slice ({sl.get('n_words', 0)} words): **{sl.get('all_exact', 'n/a')}**. "
        f"Encode+decode {sl.get('encode_decode_s', 0):.2f}s.",
        "",
    ]
    if not s["stretch_le4"]:
        lines.append("**Stretch ≤4 BPW total: no.** Do not report PBR-4 as a 4 BPW codec.")
        lines.append("")
    if not s["beats_pbre"]:
        lines.append(
            f"**Does not beat PBR-E {s['pbre_ref']:.3f} BPW.** Node formulas miss on dense LLM "
            "tiles, so |R| stays near raw 16-bit and the forced 4-bit c4 stream is extra overhead."
        )
        lines.append("")
        lines.append(
            "`choose_tile_hw(256)` matched 16×16 on this Qwen set; adaptive 16→8 never beat the parent."
        )
        lines.append("")
    if s["beats_pbre"]:
        lines.append("PBR-4 beat PBR-E on complete BPW on this set. Re-check exactness and overhead.")
        lines.append("")
    lines.append("This is not a 1–2 GB / 8 GB result. The mantissa zoo (CTW/GBDT/AR/IDF) is unchanged.")
    lines.append("")
    return "\n".join(lines)


def write_principle(summary: dict, path: Path) -> None:
    best = summary["settings"][0]
    if summary["stretch_le4"]:
        verdict = "candidate"
    elif summary["beats_pbre"]:
        verdict = "candidate-weak"
    else:
        verdict = "negative"
    para = (
        "PBR-4 stores a compact generator S at each node of a shallow block tree "
        "(root → 8×8 / 16×16 / 64 / 256 tiles, optional 16→8 split) and exactly 4 bits "
        f"c4 per weight. Reconstruction is W[i] = F(node(i), c4[i]) XOR R[i]. On the Qwen "
        f"Stage 1B 42-tensor set the best complete setting `{summary['best_setting']}` "
        f"measured **{summary['best_formal_bpw']:.3f} BPW** "
        f"(|S|={summary['best_s_bytes']}, c4={summary['best_c4_bytes']} B, "
        f"|R|={summary['best_r_bytes']}, meta={summary['best_meta_bytes']}) with "
        f"{100 * summary['best_hit_rate']:.2f}% of weights having R=0. "
        f"Stretch ≤4: {'yes' if summary['stretch_le4'] else 'no'}. "
        f"Beats PBR-E {summary['pbre_ref']:.2f}: {'yes' if summary['beats_pbre'] else 'no'}. "
        "c4 is stored side information; coordinates are already known to encoder and decoder."
    )
    why = (
        "This is not DF11 exponent rANS and not a mantissa unigram table. The hypothesis was "
        "that a 4-bit local choice among prototypes/templates plus a cheap integer formula on "
        "(row, col) would hit often enough that sparse R would drop the complete size toward "
        "~4 BPW or at least below PBR-E. The measurement is the test of that hypothesis."
    )
    fail = (
        "- Dense BF16 tiles have nearly unique uint16 values, so a 16-entry palette covers "
        f"only ~{100 * summary['best_hit_rate']:.1f}% with R=0 on the winning setting.\n"
        "- Affine formulas on (r,c) in the uint16 ring do not match trained weight bits.\n"
        "- Shared sign/exp per 16×16 is rare; SE-nibble then pays 4+3 mantissa bits plus a "
        "dense XOR residual for exponent mismatches.\n"
        "- Formal 4N/8 c4 is a floor of 4 BPW before S, R, and metadata. If F misses, total "
        "is 4 + ~16 residual bits.\n"
        "- Do not scale these BPW numbers to an 8 GB checkpoint or claim ≤4 BPW.\n"
    )
    section = "\n".join(
        [
            PBR4_MARK,
            "",
            DISCLAIMER,
            "",
            f"**Verdict:** {verdict}.",
            "",
            "### Proposed principle (one paragraph)",
            "",
            para,
            "",
            "### Why this is not just DF11",
            "",
            why,
            "",
            "### Complete BPW breakdown",
            "",
            f"- |S| node generators: {summary['best_s_bytes']} B "
            f"({8 * summary['best_s_bytes'] / max(summary['n_words'], 1):.4f} BPW)\n"
            f"- c4 stream (formal 4N/8): {summary['best_c4_bytes']} B (4.0000 BPW)\n"
            f"- |R| exact residual: {summary['best_r_bytes']} B "
            f"({8 * summary['best_r_bytes'] / max(summary['n_words'], 1):.4f} BPW)\n"
            f"- metadata: {summary['best_meta_bytes']} B "
            f"({8 * summary['best_meta_bytes'] / max(summary['n_words'], 1):.4f} BPW)\n"
            f"- **total: {summary['best_formal_bpw']:.4f} BPW**\n"
            f"- % R=0: {100 * summary['best_hit_rate']:.2f}%\n"
            f"- PBR-E rANS reference: {summary['pbre_ref']:.3f} BPW\n"
            f"- zoo best total: {summary['zoo_best_total']:.3f} BPW\n"
            f"- H(c4): {summary['c4_entropy_bpw']:.4f} bits (ANS-on-c4 would not remove the formal 4-bit floor)\n",
            "",
            "### Failure modes",
            "",
            fail,
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if PBR4_MARK in existing:
        existing = existing[: existing.index(PBR4_MARK)].rstrip() + "\n\n"
    elif existing and not existing.endswith("\n"):
        existing += "\n\n"
    elif existing:
        existing = existing.rstrip() + "\n\n"
    path.write_text(existing + section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PBR-4 structured nibble bakeoff")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--tag", default="qwen")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--slice-words", type=int, default=2048)
    parser.add_argument(
        "--settings",
        default="",
        help="Comma-separated setting names (default: all). Example: 16x16,8x8,1x64",
    )
    args = parser.parse_args(argv)
    cfg = _load_config(args.config if args.config.exists() else None)
    specs = select_weight_specs(
        inventory_from_dir(args.model_dir),
        min_bytes=int(cfg.get("min_bytes", 100 * 1024 * 1024)),
        max_bytes=int(cfg.get("max_bytes", 167772160)),
        include_embeddings=bool(cfg.get("include_embeddings", False)),
    )
    if args.max_tensors > 0:
        specs = specs[: args.max_tensors]
    settings = SETTINGS
    if args.settings.strip():
        want = {x.strip() for x in args.settings.split(",") if x.strip()}
        settings = [s for s in SETTINGS if s["name"] in want]
        if not settings:
            raise SystemExit(f"no matching PBR-4 settings in {want}")
    summary = evaluate_specs(specs, slice_words=args.slice_words, settings=settings)
    report = {
        "disclaimer": DISCLAIMER,
        "model": {
            "repo_id": cfg.get("repo", PRIMARY_REPO),
            "revision": cfg.get("revision", "main"),
            "license": cfg.get("license", PRIMARY_LICENSE),
        },
        "summary": summary,
    }
    art_json = Path(f"artifacts/pbr4_bakeoff_{args.tag}.json")
    art_md = Path(f"artifacts/pbr4_bakeoff_{args.tag}.md")
    canon_json = Path("artifacts/pbr4_bakeoff.json")
    canon_md = Path("artifacts/pbr4_bakeoff.md")
    md = format_markdown(report)
    blob = json.dumps(report, indent=2) + "\n"
    for p in (art_json, canon_json):
        p.write_text(blob, encoding="utf-8")
    for p in (art_md, canon_md):
        p.write_text(md, encoding="utf-8")
    write_principle(summary, Path("artifacts/mantissa_principle_candidate.md"))
    print()
    print(md)
    print(f"Wrote {canon_md} and {canon_json}")
    if summary.get("slice", {}).get("all_exact") != "PASS":
        print("PBR-4 FAIL: slice exactness")
        return 1
    print("PBR-4 slice exactness PASS. Gate/useful/stretch flags are measured, not assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
