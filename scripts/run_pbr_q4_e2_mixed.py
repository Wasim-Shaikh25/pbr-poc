#!/usr/bin/env python3
"""PBR-Q4 E2 mixed groupwise Q + E3−E2 selective-tile ablation.

New quantized reference with importance-structured mixed Q4/Q5/Q6/Q8/BF16.
Not S1-compatible. Not PR #17/#18. Physical BPW is file_bytes * 8 / n_weights.
Quality uses the existing calib-v2 / heldout-v1 proxy NLL suite.

E2 = packed mixed-Q container.
E3 = same Q-ref + Phase-1 selective lossless tiles with net margin.
Δ = E2_bpw − E3_bpw (matrix net contribution).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from pbr_core.safetensors_io import TWO_BYTE_DTYPES, inventory_model, load_uint16
from pbr_h95.container_h95q import sha256_state_u16
from pbr_h95.eval_nll import mean_token_nll, ppl_retention
from pbr_q4.e2_mixed.allocate import allocate, build_units
from pbr_q4.e2_mixed.const import (
    DELTA_GOOD,
    DELTA_STRONG,
    DELTA_USEFUL,
    E2B_EPS_DEFAULT,
    PR17_Q6_BPW,
    PR17_Q6_HELDOUT,
    PR18_H95_BPW,
    PR18_H95_HELDOUT,
    QUALITY_AIM,
    QUALITY_MIN,
    S1_N_WEIGHTS,
    S1_V2_ACTUAL_BPW,
    S1_V2_FILE_BYTES,
    TARGET_PRACTICAL,
    TARGET_STRETCH,
)
from pbr_q4.e2_mixed.container import (
    decode_container,
    encode_quantized,
    verify_decoded_against_reference,
)
from pbr_q4.e2_mixed.families import family_label, projection_label
from pbr_q4.e2_mixed.quantize import QuantizedTensor, estimate_packed_bits, quantize_mixed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
CALIB_PATH = Path("docs/pbr_h95/calibration_v2.json")
HELDOUT_PATH = Path("docs/pbr_h95/heldout_v1.json")
OUT_JSON = Path("artifacts/pbr_q4/e2_mixed_qwen.json")
OUT_MD = Path("artifacts/pbr_q4/e2_mixed_qwen.md")
CONTAINER_DIR = Path("artifacts/pbr_q4/containers")

# budget_bpw, floor bits. Walk cheapest → most protected.
LADDER: tuple[tuple[str, float, int, str], ...] = (
    ("e2_stretch_4_75", 4.75, 4, "Stretch ≤4.75 packed-est; Q4 default + structured protect."),
    ("e2_budget_5_00", 5.00, 4, "Practical ≤5.0 packed-est; Q4 default + structured protect."),
    ("e2_backoff_5_50", 5.50, 4, "Quality backoff: allow 5.50 packed-est (may miss ≤5.0)."),
    ("e2_backoff_6_00", 6.00, 5, "Quality backoff: Q5 floor, budget 6.00."),
    ("e2_q6_tight", 6.25, 6, "Q6 floor with a tight budget (no leftover Q8 spend)."),
    ("e2_q6_protect", 6.75, 6, "Q6 floor + modest structured Q8 protect."),
)

HONESTY = [
    "New mixed groupwise quantized reference — NOT bit-exact with H95Q-S1 / PR #17 / PR #18.",
    "Decode is bit-exact to this quantized reference, not to original BF16.",
    "actual_bpw = physical file_bytes * 8 / n_weights (header, scales, maps, flags, specials).",
    "E2 is packed mixed-Q. E3 is the same Q-ref + Phase-1 selective tiles with net margin.",
    "Product default is packed. Matrix family is stored only if it saves ≥ max(16 bits, 0.02×raw_tile_bits) after ALL costs.",
    "Quality is ppl_retention = bf16_ppl / quant_ppl on in-repo calib-v2 / heldout-v1 (proxy, not MMLU).",
    "Hard quality floor: heldout ≥0.95. Development aim ≥0.97.",
    "Practical rate target ≤5.0 BPW; stretch ≤4.75. Misses are reported as FAIL, not rounded away.",
    "Structured channel-group protect; no sparse per-weight exceptions as a quality lever (NaN/Inf only).",
    "Do not invent more X/Y modes. Do not merge/reopen #15/#17/#18 as the product path.",
    "S1 v2 7.420820 / #17 Q6 6.191 / #18 hybrid 7.765 are comparison baselines only.",
    "Not a production mobile runtime.",
]


def load_texts(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text())
    return payload, [t["text"] for t in payload["texts"]]


def _ensure_tokenizer_files(model_dir: Path) -> None:
    if (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists():
        return
    from huggingface_hub import snapshot_download

    print(f"  fetching tokenizer files into {model_dir}", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=str(model_dir),
        cache_dir=str(Path("outputs/hf_cache")),
        allow_patterns=[
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "generation_config.json",
        ],
    )


def resolve_model(model_dir: Path) -> Path:
    if model_dir.exists() and any(model_dir.glob("*.safetensors")):
        _ensure_tokenizer_files(model_dir)
        return model_dir
    print(f"Model not at {model_dir}; downloading Qwen2.5-0.5B-Instruct…", flush=True)
    from pbr_encoder.hf_weights import download_checkpoint

    result = download_checkpoint(
        local_dir=Path("outputs/models"),
        cache_dir=Path("outputs/hf_cache"),
    )
    print(f"  downloaded {result.repo_id} → {result.local_dir} ({result.note})", flush=True)
    dest = Path(result.local_dir)
    _ensure_tokenizer_files(dest)
    return dest


def unique_specs(model_dir: Path):
    specs = [s for s in inventory_model(model_dir) if s.dtype in TWO_BYTE_DTYPES]
    seen: set[tuple[str, tuple[int, int]]] = set()
    out = []
    for spec in sorted(specs, key=lambda s: s.name):
        key = (str(spec.file_path), spec.data_offsets)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def load_state(specs) -> dict[str, np.ndarray]:
    state = {}
    for spec in specs:
        state[spec.name] = load_uint16(spec).reshape(spec.shape)
    return state


def collect_act_scale(model, tokenizer, texts, names: list[str], *, max_length: int = 256) -> dict[str, float]:
    """Per-tensor output-channel RMS from a calib forward (hook Linear/Embedding)."""
    import torch
    import torch.nn as nn

    scales: dict[str, list[float]] = {n: [] for n in names}
    handles = []

    def _hook(name):
        def fn(_mod, _inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(y):
                return
            t = y.detach().float()
            # RMS over batch/seq; mean over channels → one tensor scale.
            dims = tuple(range(t.ndim - 1)) if t.ndim >= 2 else (0,)
            rms = torch.sqrt((t * t).mean(dim=dims) + 1e-12)
            scales.setdefault(name, []).append(float(rms.mean().item()))

        return fn

    lookup = {id(p): n for n, p in model.named_parameters()}
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        wname = None
        if hasattr(mod, "weight"):
            wname = lookup.get(id(mod.weight))
        if wname and wname in scales:
            handles.append(mod.register_forward_hook(_hook(wname)))
    try:
        with torch.no_grad():
            for text in texts[:24]:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                model(**{k: v for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()
    out = {}
    for n, vals in scales.items():
        out[n] = float(np.mean(vals)) if vals else 1.0
    # Normalize so median is 1.
    finite = [v for v in out.values() if v > 0]
    med = float(np.median(finite)) if finite else 1.0
    if med > 0:
        out = {k: v / med for k, v in out.items()}
    return out


def quantize_state(
    state: dict[str, np.ndarray],
    row_bits: dict[str, np.ndarray],
    *,
    label: str,
    eps: float,
) -> list[QuantizedTensor]:
    qts: list[QuantizedTensor] = []
    names = sorted(state.keys())
    t0 = time.perf_counter()
    for i, name in enumerate(names, 1):
        words = state[name]
        arr2 = np.ascontiguousarray(words, dtype=np.uint16)
        rows = int(arr2.shape[0]) if arr2.ndim >= 1 else 1
        rb = allocation_row_bits_safe(row_bits, name, arr2)
        if i == 1 or i == len(names) or i % 15 == 0:
            uniq = {int(x) for x in np.unique(rb).tolist()}
            print(f"  [{label}] {i}/{len(names)} {name} bits={sorted(uniq)} shape={tuple(arr2.shape)}", flush=True)
        qts.append(
            quantize_mixed(
                arr2,
                rb,
                eps=eps,
                name=name,
                family=family_label(name),
                projection=projection_label(name),
            )
        )
    print(f"  [{label}] quantized {len(qts)} tensors in {time.perf_counter() - t0:.1f}s", flush=True)
    return qts


def allocation_row_bits_safe(row_bits: dict[str, np.ndarray], name: str, words: np.ndarray) -> np.ndarray:
    rows = int(words.shape[0]) if words.ndim >= 1 else 1
    if words.ndim == 1:
        rows = 1
    return allocation_row_bits_from(row_bits, name, rows)


def allocation_row_bits_from(row_bits: dict[str, np.ndarray], name: str, n_rows: int) -> np.ndarray:
    if name in row_bits:
        rb = np.asarray(row_bits[name], dtype=np.uint8)
        if int(rb.size) != int(n_rows):
            # 1-D viewed as (1, n)
            if n_rows == 1 and rb.size > 1:
                return np.array([int(rb.max())], dtype=np.uint8)
            raise ValueError(f"row_bits {rb.size} != rows {n_rows} for {name}")
        return rb
    return np.full(n_rows, 16, dtype=np.uint8)


def packed_estimate(qts: list[QuantizedTensor]) -> dict[str, Any]:
    n = sum(int(t.q_ref.size) for t in qts)
    parts = [estimate_packed_bits(t) for t in qts]
    total = sum(p["total_bits"] for p in parts)
    by_fam: dict[str, dict[str, float]] = {}
    bit_hist: dict[str, int] = {}
    for t, p in zip(qts, parts):
        slot = by_fam.setdefault(t.family, {"n_words": 0, "code_bits": 0, "n_specials": 0})
        slot["n_words"] += int(t.q_ref.size)
        slot["code_bits"] += p["total_bits"]
        slot["n_specials"] += int(t.special_idx.size)
        rb, counts = np.unique(t.row_bits, return_counts=True)
        for b, c in zip(rb.tolist(), counts.tolist()):
            key = "bf16" if int(b) >= 16 else str(int(b))
            bit_hist[key] = bit_hist.get(key, 0) + int(c) * int(t.cols)
    return {
        "n_weights": n,
        "packed_est_bits": total,
        "packed_est_bpw": round(total / n, 6) if n else 0.0,
        "family": {
            k: {
                "n_words": v["n_words"],
                "est_bpw": round(v["code_bits"] / v["n_words"], 4) if v["n_words"] else 0.0,
                "n_specials": v["n_specials"],
            }
            for k, v in by_fam.items()
        },
        "bit_hist_words": bit_hist,
        "n_specials": sum(int(t.special_idx.size) for t in qts),
        "note": "codes + f16 scales + u8 zp + 8-bit/row map + NaN/Inf specials; no container / E3",
    }


def apply_qref(model, qref: dict[str, np.ndarray]) -> None:
    import torch

    from pbr_h95.apply_policy import u16_to_bf16_tensor

    with torch.no_grad():
        sd = model.state_dict()
        for name, words in qref.items():
            if name not in sd:
                continue
            t = sd[name]
            arr = np.ascontiguousarray(words, dtype=np.uint16).reshape(tuple(t.shape))
            t.copy_(u16_to_bf16_tensor(arr, t))
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            if hasattr(model, "tie_weights"):
                model.tie_weights()


def eval_proxy(model, tokenizer, calib_texts, heldout_texts, *, max_length: int = 256) -> dict[str, Any]:
    calib = mean_token_nll(model, tokenizer, calib_texts, max_length=max_length)
    held = mean_token_nll(model, tokenizer, heldout_texts, max_length=max_length)
    return {"calib": calib, "heldout": held}


def delta_verdict(delta: float) -> str:
    if delta >= DELTA_STRONG:
        return "strong"
    if delta >= DELTA_GOOD:
        return "good"
    if delta >= DELTA_USEFUL:
        return "useful"
    return "below_useful"


def write_md(report: dict[str, Any], path: Path) -> None:
    w = report["winner"]
    gates = report["gates"]
    e2 = report.get("e2") or {}
    e3 = report.get("e3") or {}
    delta = report.get("delta_e2_minus_e3")
    lines = [
        "# PBR-Q4 E2 mixed groupwise Q + E3−E2 ablation",
        "",
        "## Dual-gate verdict",
        "",
        "| Gate | Target | Measured | Verdict |",
        "| --- | --- | ---: | --- |",
        f"| Practical physical rate | ≤ **5.0** BPW | **{e3.get('actual_bpw', w.get('actual_bpw'))}** | **{gates['practical_rate']}** |",
        f"| Stretch physical rate | ≤ **4.75** BPW | **{e3.get('actual_bpw', w.get('actual_bpw'))}** | **{gates['stretch_rate']}** |",
        f"| Held-out proxy (hard min) | ≥ **0.95** | **{w.get('heldout_retention')}** | **{gates['quality_min']}** |",
        f"| Held-out proxy (aim) | ≥ **0.97** | **{w.get('heldout_retention')}** | **{gates['quality_aim']}** |",
        f"| Exact decode of this Q-ref | SHA match | `{w.get('sha256_quantized_reference')}` | **{gates['exact_decode']}** |",
        "",
        f"Policy: `{w.get('policy')}` — new quantized reference SHA `{w.get('sha256_quantized_reference')}`.",
        f"Not S1/#17/#18 compatible. Wall time: **{report['wall_s']:.1f}s**",
        "",
        "## Critical ablation (matrix net contribution)",
        "",
        "```",
        f"E2 physical BPW  {e2.get('actual_bpw')}",
        f"E3 physical BPW  {e3.get('actual_bpw')}",
        f"Δ = E2 − E3      {delta}   ({report.get('delta_verdict')})",
        "```",
        "",
        f"Milestones: useful ≥ {DELTA_USEFUL}, good ≥ {DELTA_GOOD}, strong ≥ {DELTA_STRONG}. "
        f"**Hit: {report.get('delta_verdict')}.**",
        "",
        "## Overhead split (E3 winner)",
        "",
    ]
    split = (e3.get("overhead_split") or e2.get("overhead_split") or {})
    if split:
        lines += [
            "| part | bytes | BPW |",
            "| --- | ---: | ---: |",
            f"| weight payload | {int(split.get('weight_payload_bytes') or 0):,} | {split.get('weight_payload_bpw')} |",
            f"| scales (f16+zp) | {int(split.get('scales_bytes') or 0):,} | {split.get('scales_bpw')} |",
            f"| precision map | {int(split.get('precision_map_bytes') or 0):,} | {split.get('precision_map_bpw')} |",
            f"| codec meta (maps/flags) | {int(split.get('codec_meta_bytes') or 0):,} | {split.get('codec_meta_bpw')} |",
            f"| container | {int(split.get('container_bytes') or 0):,} | {split.get('container_bpw')} |",
            "",
        ]
    lines += ["## Honesty", ""]
    for h in HONESTY:
        lines.append(f"- {h}")
    lines += [
        "",
        "## vs baselines",
        "",
        "| Container | file_bytes | actual_bpw | Q(W) |",
        "| --- | ---: | ---: | --- |",
        f"| H95Q-S1 v2 (PR #14/#16) | {S1_V2_FILE_BYTES:,} | **{S1_V2_ACTUAL_BPW:.6f}** | S1 SHA |",
        f"| PR #17 `restore_q6` | — | **{PR17_Q6_BPW:.6f}** | Q6 (~heldout {PR17_Q6_HELDOUT}) |",
        f"| PR #18 `h_quality` | — | **{PR18_H95_BPW:.6f}** | all-H95 (~heldout {PR18_H95_HELDOUT}) |",
        f"| This E2 packed | {int(e2.get('file_bytes') or 0):,} | **{e2.get('actual_bpw')}** | this SHA |",
        f"| This E3 selective | {int(e3.get('file_bytes') or 0):,} | **{e3.get('actual_bpw')}** | same SHA |",
        "",
        "## Mode mix (E3)",
        "",
        f"- Tiles: {e3.get('n_tiles')} — X/Y {e3.get('n_xy_tiles')} / packed {e3.get('n_packed_tiles')}",
        f"- XY mode hist: `{e3.get('xy_mode_histogram')}`",
        f"- Predictors: `{e3.get('xy_pred_histogram')}`",
        f"- Traversals: `{e3.get('xy_trav_histogram')}`",
        f"- Specials (NaN/Inf only): {e3.get('n_specials', w.get('n_specials'))}",
        f"- Saved vs packed codes: {e3.get('saved_vs_packed_code_bytes')} B",
        "",
        "## Quality (proxy NLL)",
        "",
        "| Split | BF16 ppl | Q ppl | retention |",
        "| --- | ---: | ---: | ---: |",
        f"| calib-v2 | {report['bf16']['calib']['ppl']:.4f} | {w.get('calib_ppl')} | {w.get('calib_retention')} |",
        f"| heldout-v1 | {report['bf16']['heldout']['ppl']:.4f} | {w.get('heldout_ppl')} | {w.get('heldout_retention')} |",
        "",
        "## Pareto / backoff",
        "",
        "| policy | packed-est BPW | E2 BPW | E3 BPW | calib ret | heldout ret | encoded? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["pareto"]:
        lines.append(
            f"| {row['policy']} | {row.get('packed_est_bpw')} | {row.get('e2_bpw', '—')} | "
            f"{row.get('e3_bpw', '—')} | {row.get('calib_retention')} | {row.get('heldout_retention')} | {row.get('encoded')} |"
        )
    lines += [
        "",
        "## Exactness",
        "",
        f"- SHA-256 quantized ref: `{w.get('sha256_quantized_reference')}`",
        f"- E2 decode(encode(Q)) == Q: **{str((report.get('e2') or {}).get('exact_decode')).upper()}**",
        f"- E3 decode(encode(Q)) == Q: **{str((report.get('e3') or {}).get('exact_decode')).upper()}**",
        f"- Subprocess decode SHA: **{str(w.get('subprocess_sha_ok')).upper()}**",
        "",
        "## Family mix (winner packed-est)",
        "",
    ]
    fam = (w.get("packed_estimate") or {}).get("family") or {}
    if fam:
        lines += ["| family | n_words | packed-est BPW | specials |", "| --- | ---: | ---: | ---: |"]
        for k, v in sorted(fam.items()):
            lines.append(f"| {k} | {int(v['n_words']):,} | {v['est_bpw']} | {int(v['n_specials']):,} |")
    bh = (w.get("packed_estimate") or {}).get("bit_hist_words") or {}
    if bh:
        lines += ["", "Bit-width word histogram:", ""]
        for k, v in sorted(bh.items(), key=lambda kv: kv[0]):
            lines.append(f"- {k}: {int(v):,}")
    path.write_text("\n".join(lines) + "\n")


def encode_pair(
    qts: list[QuantizedTensor],
    container_dir: Path,
    policy_name: str,
) -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    e2_path = container_dir / f"PBR-Q4-{policy_name}.e2mx"
    e3_path = container_dir / f"PBR-Q4-{policy_name}.e3mx"
    print("Encoding E2 packed…", flush=True)
    e2 = encode_quantized(qts, e2_path, phase="E2", policy_name=policy_name)
    print(f"  E2 file_bytes={e2['file_bytes']} actual_bpw={e2['actual_bpw']}", flush=True)
    print("Encoding E3 selective…", flush=True)
    e3 = encode_quantized(qts, e3_path, phase="E3", policy_name=policy_name)
    print(f"  E3 file_bytes={e3['file_bytes']} actual_bpw={e3['actual_bpw']}", flush=True)
    qref = {t.name: t.q_ref for t in qts}
    print("Decoding E2/E3 and verifying SHA…", flush=True)
    d2 = decode_container(e2_path)
    d3 = decode_container(e3_path)
    v2 = verify_decoded_against_reference(d2["tensors"], qref)
    v3 = verify_decoded_against_reference(d3["tensors"], qref)
    sha = e2["sha256_quantized_reference"]
    exact = bool(v2["ok"] and v3["ok"] and d2["sha256_decoded"] == sha and d3["sha256_decoded"] == sha)
    e2["exact_decode"] = bool(v2["ok"])
    e3["exact_decode"] = bool(v3["ok"])
    proc = subprocess.run(
        [sys.executable, "-m", "pbr_q4.e2_mixed.container", str(e3_path), "--expect-sha", sha],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    sub_ok = proc.returncode == 0
    if not sub_ok:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return e2, e3, exact, sub_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--policy", type=str, default="auto", help="auto | " + " | ".join(p[0] for p in LADDER))
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--eps", type=float, default=E2B_EPS_DEFAULT)
    ap.add_argument("--calib", type=Path, default=CALIB_PATH)
    ap.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--container-dir", type=Path, default=CONTAINER_DIR)
    args = ap.parse_args()

    t_all = time.perf_counter()
    model_dir = resolve_model(args.model_dir)
    specs = unique_specs(model_dir)
    print(f"inventory: {len(specs)} unique 16-bit tensors", flush=True)
    state = load_state(specs)
    n_weights = sum(int(v.size) for v in state.values())
    print(f"n_weights={n_weights}", flush=True)

    calib_meta, calib_texts = load_texts(args.calib)
    held_meta, held_texts = load_texts(args.heldout)

    names = [args.policy] if args.policy != "auto" else [p[0] for p in LADDER]
    ladder_map = {p[0]: p for p in LADDER}

    pareto: list[dict[str, Any]] = []
    winner_qts: list[QuantizedTensor] | None = None
    winner_eval: dict[str, Any] | None = None
    winner_name = None
    bf16_scores: dict[str, Any] | None = None
    act_scale: dict[str, float] = {}
    embed_freq: dict[int, int] = {}
    special_ids: list[int] = []

    model = tokenizer = None
    baseline = None
    restore_state_dict = None
    if not args.skip_eval:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from pbr_h95.apply_policy import clone_cpu_state_dict, restore_state_dict as _restore
        from pbr_h95.embed_tiers import count_token_frequencies

        restore_state_dict = _restore
        torch.set_num_threads(1)
        print("Loading BF16 model for proxy NLL…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()
        baseline = clone_cpu_state_dict(model)
        print("Scoring BF16 baseline + collecting activation/embed stats…", flush=True)
        bf16_scores = eval_proxy(model, tokenizer, calib_texts, held_texts)
        print(
            f"  bf16 calib ppl={bf16_scores['calib']['ppl']:.4f} "
            f"heldout ppl={bf16_scores['heldout']['ppl']:.4f}",
            flush=True,
        )
        act_scale = collect_act_scale(model, tokenizer, calib_texts, list(state.keys()))
        freq = count_token_frequencies(tokenizer, calib_texts, max_length=256)
        embed_freq = {int(k): int(v) for k, v in freq.items()}
        special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

    qts = None
    for pname in names:
        _name, budget, floor, desc = ladder_map[pname]
        print(f"=== policy {pname}: budget={budget} floor={floor} — {desc}", flush=True)
        units = build_units(
            state,
            floor=floor,
            act_scale=act_scale,
            embed_freq=embed_freq,
            special_ids=special_ids,
        )
        print(f"  {len(units)} channel-group units; allocating…", flush=True)
        alloc = allocate(units, budget_bpw=budget, n_weights=n_weights)
        print(
            f"  packed-est BPW={alloc.packed_est_bpw} (budget {budget}) "
            f"protect_units={alloc.n_protected_units} hist={alloc.bit_hist_words}",
            flush=True,
        )
        qts = quantize_state(state, alloc.row_bits, label=pname, eps=float(args.eps))
        est = packed_estimate(qts)
        print(f"  post-Q packed-est BPW={est['packed_est_bpw']} specials={est['n_specials']}", flush=True)
        row: dict[str, Any] = {
            "policy": pname,
            "description": desc,
            "budget_bpw": budget,
            "floor_bits": floor,
            "alloc_packed_est_bpw": alloc.packed_est_bpw,
            "packed_est_bpw": est["packed_est_bpw"],
            "packed_estimate": est,
            "bit_hist_words": alloc.bit_hist_words,
            "encoded": False,
        }
        if model is not None and tokenizer is not None and bf16_scores is not None:
            qref = {t.name: t.q_ref for t in qts}
            apply_qref(model, qref)
            scores = eval_proxy(model, tokenizer, calib_texts, held_texts)
            if baseline is not None and restore_state_dict is not None:
                restore_state_dict(model, baseline)
            calib_ret = ppl_retention(bf16_scores["calib"]["ppl"], scores["calib"]["ppl"])
            held_ret = ppl_retention(bf16_scores["heldout"]["ppl"], scores["heldout"]["ppl"])
            row.update(
                {
                    "calib_ppl": round(scores["calib"]["ppl"], 6),
                    "heldout_ppl": round(scores["heldout"]["ppl"], 6),
                    "calib_retention": round(calib_ret, 6),
                    "heldout_retention": round(held_ret, 6),
                    "calib_nll": scores["calib"]["mean_nll"],
                    "heldout_nll": scores["heldout"]["mean_nll"],
                }
            )
            print(f"  calib_ret={calib_ret:.4f} heldout_ret={held_ret:.4f}", flush=True)
            quality_ok = held_ret >= QUALITY_MIN
        else:
            quality_ok = True
        pareto.append(row)
        if quality_ok and winner_qts is None:
            winner_qts = qts
            winner_eval = row
            winner_name = pname
            if args.policy == "auto":
                print(f"  selected winner {pname} (heldout ≥ {QUALITY_MIN})", flush=True)
                # Keep walking remaining policies only for Pareto if they are
                # cheaper; auto stops so we encode the first quality-ok point.
                break
        elif args.policy != "auto":
            winner_qts = qts
            winner_eval = row
            winner_name = pname

    if winner_qts is None:
        print("No policy met held-out ≥0.95; encoding last attempt for the record.", flush=True)
        winner_qts = qts
        winner_eval = pareto[-1]
        winner_name = names[-1]

    # Drop the BF16 numpy state and the live model before physical encode.
    del state
    e2_stats: dict[str, Any] = {}
    e3_stats: dict[str, Any] = {}
    exact = False
    sub_ok = False
    if not args.skip_encode:
        if model is not None:
            del model
            model = None
        if baseline is not None:
            del baseline
            baseline = None
        import gc

        gc.collect()
        args.container_dir.mkdir(parents=True, exist_ok=True)
        e2_stats, e3_stats, exact, sub_ok = encode_pair(winner_qts, args.container_dir, winner_name or "e2")
        for row in pareto:
            if row["policy"] == winner_name:
                row["encoded"] = True
                row["e2_bpw"] = e2_stats["actual_bpw"]
                row["e3_bpw"] = e3_stats["actual_bpw"]
                row["file_bytes_e2"] = e2_stats["file_bytes"]
                row["file_bytes_e3"] = e3_stats["file_bytes"]

    winner = dict(winner_eval or {})
    winner.update({k: v for k, v in e3_stats.items() if k not in ("phase",)})
    winner["exact_decode"] = exact
    winner["subprocess_sha_ok"] = sub_ok
    if "packed_estimate" not in winner:
        winner["packed_estimate"] = packed_estimate(winner_qts)
    winner["sha256_quantized_reference"] = (
        e2_stats.get("sha256_quantized_reference")
        or e3_stats.get("sha256_quantized_reference")
        or sha256_state_u16({t.name: t.q_ref for t in winner_qts})
    )

    e3_bpw = float(e3_stats.get("actual_bpw") or 0.0)
    e2_bpw = float(e2_stats.get("actual_bpw") or 0.0)
    delta = round(e2_bpw - e3_bpw, 6) if e2_bpw and e3_bpw else None
    held_ret = float(winner.get("heldout_retention") or 0.0)
    rate_bpw = e3_bpw or e2_bpw
    gates = {
        "practical_rate": "PASS" if rate_bpw and rate_bpw <= TARGET_PRACTICAL else "FAIL",
        "stretch_rate": "PASS" if rate_bpw and rate_bpw <= TARGET_STRETCH else "FAIL",
        "quality_min": "PASS" if held_ret >= QUALITY_MIN else ("SKIP" if args.skip_eval else "FAIL"),
        "quality_aim": "PASS" if held_ret >= QUALITY_AIM else ("SKIP" if args.skip_eval else "FAIL"),
        "exact_decode": "PASS" if exact else ("SKIP" if args.skip_encode else "FAIL"),
    }
    if args.skip_encode:
        gates["practical_rate"] = gates["stretch_rate"] = "SKIP"

    report = {
        "phase": "PBR-Q4-E2-mixed-E3-ablation",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_dir": str(model_dir),
        "s1_compatible": False,
        "pr17_compatible": False,
        "pr18_compatible": False,
        "s1_v2_actual_bpw": S1_V2_ACTUAL_BPW,
        "s1_v2_file_bytes": S1_V2_FILE_BYTES,
        "s1_n_weights": S1_N_WEIGHTS,
        "honesty": HONESTY,
        "targets": {
            "practical_bpw": TARGET_PRACTICAL,
            "stretch_bpw": TARGET_STRETCH,
            "heldout_min": QUALITY_MIN,
            "heldout_aim": QUALITY_AIM,
            "e2b_eps": float(args.eps),
        },
        "gates": gates,
        "winner": winner,
        "e2": e2_stats,
        "e3": e3_stats,
        "delta_e2_minus_e3": delta,
        "delta_verdict": delta_verdict(delta or 0.0) if delta is not None else "n/a",
        "pareto": pareto,
        "bf16": bf16_scores
        or {
            "calib": {"ppl": None, "mean_nll": None},
            "heldout": {"ppl": None, "mean_nll": None},
        },
        "calib_corpus": {"path": str(args.calib), "id": calib_meta.get("id"), "n_texts": len(calib_texts)},
        "heldout_corpus": {"path": str(args.heldout), "id": held_meta.get("id"), "n_texts": len(held_texts)},
        "wall_s": round(time.perf_counter() - t_all, 2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_md(report, args.out_md)
    print(
        json.dumps(
            {
                "gates": gates,
                "e2_bpw": e2_bpw,
                "e3_bpw": e3_bpw,
                "delta": delta,
                "delta_verdict": report["delta_verdict"],
                "heldout": winner.get("heldout_retention"),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_json} and {args.out_md}", flush=True)
    return 0 if (exact or args.skip_encode) else 2


if __name__ == "__main__":
    raise SystemExit(main())
