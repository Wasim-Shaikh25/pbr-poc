"""Hybrid lossy disk-RAM tunnel: exact sensitive tensors, int4 bulk projections.

NOT bit-exact on quantized tensors. Not an ≤8 BPW / 50% / 27B-phone claim.
Embeddings, norms, biases, and first/last layers stay BF16 (mmap or copied).
Middle-layer attention/MLP weights use per-group signed int4 + FP16 scales.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from pbr_core.hashing import sha256_words
from pbr_core.safetensors_io import TensorSpec, load_uint16
from pbr_encoder.disk_tunnel import (
    DEFAULT_MODEL,
    LM_HEAD_ROWS,
    DISCLAIMER as TUNNEL_DISCLAIMER,
    IoStats,
    WeightSource,
    bf16_to_fp32,
    bytes_human,
    format_markdown,
    load_config_json,
    malloc_trim,
    maxrss_bytes,
    qwen_forward,
    rss_bytes,
    run_mode,
    select_specs,
    _token_ids,
)
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO

DISCLAIMER = (
    "Hybrid LOSSY tunnel PoC. Sensitive tensors stay BF16; bulk MLP/attn "
    "projections are per-group int4. Not bit-exact on quantized weights. "
    "Not a phone-scale 27B runtime. Not an ≤8 BPW exact-compression claim."
)
DEFAULT_HYBRID_DIR = Path("outputs/disk_tunnel/hybrid")
GROUP = 64
MAGIC = b"HYI4"
LAYER_RE = re.compile(r"layers\.(\d+)\.")
LOSSY_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def is_sensitive_exact(name: str, n_layers: int) -> bool:
    n = name.lower()
    if "embed" in n or n.endswith("lm_head.weight"):
        return True
    if "norm" in n or n.endswith("bias"):
        return True
    match = LAYER_RE.search(name)
    if match:
        idx = int(match.group(1))
        if idx == 0 or idx == max(n_layers - 1, 0):
            return True
    return not any(name.endswith(suf) for suf in LOSSY_SUFFIXES)


def quantize_int4(fp32: np.ndarray, group_size: int = GROUP) -> tuple[np.ndarray, np.ndarray, int]:
    flat = np.ascontiguousarray(fp32, dtype=np.float32).ravel()
    n = int(flat.size)
    pad = (group_size - (n % group_size)) % group_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    groups = flat.reshape(-1, group_size)
    amax = np.max(np.abs(groups), axis=1)
    scale = np.maximum(amax / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.rint(groups / scale[:, None]), -8, 7).astype(np.int8)
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).ravel()
    return packed, scale.astype(np.float16), n


def dequant_int4(
    packed: np.ndarray,
    scales_f16: np.ndarray,
    shape: tuple[int, ...],
    group_size: int,
    n_words: int,
) -> np.ndarray:
    scales = np.ascontiguousarray(scales_f16, dtype=np.float16).astype(np.float32)
    n_groups = int(scales.size)
    u = np.ascontiguousarray(packed, dtype=np.uint8).reshape(n_groups, group_size // 2)
    q = np.empty((n_groups, group_size), dtype=np.float32)
    q[:, 0::2] = (u & np.uint8(0x0F)).astype(np.float32) - 8.0
    q[:, 1::2] = (u >> np.uint8(4)).astype(np.float32) - 8.0
    fp = (q * scales[:, None]).ravel()[: int(n_words)]
    return fp.reshape(shape)


def dump_int4(path: Path, words_u16: np.ndarray, group_size: int = GROUP) -> dict:
    fp = bf16_to_fp32(words_u16)
    packed, scales, n_words = quantize_int4(fp, group_size=group_size)
    shape = tuple(int(x) for x in words_u16.shape)
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows, cols = int(shape[0]), int(np.prod(shape[1:]))
    header = MAGIC + struct.pack(
        "<HIIII",
        1,
        rows,
        cols,
        group_size,
        n_words,
    )
    payload = header + scales.tobytes() + packed.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    recon = dequant_int4(packed, scales, words_u16.shape, group_size, n_words)
    err = float(np.max(np.abs(recon - fp)))
    return {
        "file": path.name,
        "kind": "int4",
        "shape": list(words_u16.shape),
        "n_words": n_words,
        "group_size": group_size,
        "encoded_bytes": len(payload),
        "original_bytes": int(words_u16.nbytes),
        "max_abs_dequant": err,
    }


def load_int4_file(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    blob = path.read_bytes()
    if blob[:4] != MAGIC:
        raise ValueError(f"bad hybrid magic {blob[:4]!r}")
    version, rows, cols, group_size, n_words = struct.unpack_from("<HIIII", blob, 4)
    if version != 1:
        raise ValueError(f"unsupported hybrid version {version}")
    n_groups = (int(n_words) + group_size - 1) // group_size
    off = 4 + 2 + 16
    scales = np.frombuffer(blob[off : off + n_groups * 2], dtype="<f2").copy()
    packed = np.frombuffer(blob[off + n_groups * 2 :], dtype=np.uint8).copy()
    return dequant_int4(packed, scales, shape, group_size, n_words)


def encode_hybrid_dir(
    specs: dict[str, TensorSpec],
    dest: Path,
    *,
    n_layers: int,
    group_size: int = GROUP,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    qdir = dest / "q4"
    exact_dir = dest / "exact"
    qdir.mkdir(exist_ok=True)
    exact_dir.mkdir(exist_ok=True)
    index: dict = {
        "disclaimer": DISCLAIMER,
        "group_size": group_size,
        "n_layers": n_layers,
        "tensors": {},
        "n_exact": 0,
        "n_int4": 0,
        "exact_bytes": 0,
        "int4_bytes": 0,
        "original_bytes": 0,
    }
    for key, spec in specs.items():
        words = load_uint16(spec)
        index["original_bytes"] += int(words.nbytes)
        if is_sensitive_exact(key, n_layers):
            fname = key.replace("/", ".") + ".u16"
            path = exact_dir / fname
            path.write_bytes(np.ascontiguousarray(words, dtype="<u2").tobytes())
            rec = {
                "kind": "exact_bf16",
                "file": f"exact/{fname}",
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "sha256": sha256_words(words),
                "n_words": spec.n_words,
                "original_bytes": spec.nbytes,
                "encoded_bytes": spec.nbytes,
            }
            index["n_exact"] += 1
            index["exact_bytes"] += spec.nbytes
        else:
            fname = key.replace("/", ".") + ".hyi4"
            rec = dump_int4(qdir / fname, words, group_size=group_size)
            rec["file"] = f"q4/{fname}"
            rec["dtype"] = spec.dtype
            rec["sha256"] = None
            index["n_int4"] += 1
            index["int4_bytes"] += int(rec["encoded_bytes"])
        index["tensors"][key] = rec
        del words
        gc.collect()
        print(
            f"  hybrid {rec['kind']:12s} {key}  {spec.shape}  "
            f"{rec['encoded_bytes']} B",
            flush=True,
        )
    index["pack_bytes"] = index["exact_bytes"] + index["int4_bytes"]
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(
        f"  hybrid pack exact={index['n_exact']} ({index['exact_bytes']} B)  "
        f"int4={index['n_int4']} ({index['int4_bytes']} B)  "
        f"total={index['pack_bytes']} B",
        flush=True,
    )
    return index


class HybridSource(WeightSource):
    """Disk-resident hybrid pack: exact uint16 or int4 dequant to FP32, then free."""

    def __init__(self, index: dict, hybrid_dir: Path, specs: dict[str, TensorSpec]):
        super().__init__(name="hybrid_int4")
        self.index = index
        self.hybrid_dir = Path(hybrid_dir)
        self.specs = specs

    def load(self, key: str) -> np.ndarray:
        rec = self.index["tensors"][key]
        if rec["kind"] == "exact_bf16":
            t0 = time.perf_counter()
            path = self.hybrid_dir / rec["file"]
            blob = path.read_bytes()
            self.stats.disk_read_bytes += len(blob)
            words = np.frombuffer(blob, dtype="<u2").copy().reshape(tuple(rec["shape"]))
            self.stats.decode_s += time.perf_counter() - t0
            self.stats.n_loads += 1
            if rec.get("sha256"):
                got = sha256_words(words)
                if got != rec["sha256"]:
                    self.stats.exact_fail += 1
                    raise AssertionError(f"hybrid exact SHA mismatch {key}")
                self.stats.exact_checks += 1
            self.stats.note_rss()
            return words
        # int4: return a uint16 placeholder is misleading; use load_compute.
        return self._load_int4_as_u16_roundtrip(key)

    def _load_int4_as_u16_roundtrip(self, key: str) -> np.ndarray:
        fp = self.load_compute(key)
        bits = np.ascontiguousarray(fp, dtype=np.float32).view(np.uint32)
        u16 = (bits >> np.uint32(16)).astype(np.uint16)
        return u16.reshape(fp.shape)

    def load_compute(self, key: str) -> np.ndarray:
        rec = self.index["tensors"][key]
        if rec["kind"] == "exact_bf16":
            return bf16_to_fp32(self.load(key))
        t0 = time.perf_counter()
        path = self.hybrid_dir / rec["file"]
        fp = load_int4_file(path, tuple(int(x) for x in rec["shape"]))
        self.stats.disk_read_bytes += int(path.stat().st_size)
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.n_loads += 1
        self.stats.note_rss()
        return fp

    def load_row_range(self, key: str, row0: int, row1: int) -> np.ndarray:
        rec = self.index["tensors"][key]
        if rec["kind"] != "exact_bf16":
            return self.load(key)[row0:row1]
        path = self.hybrid_dir / rec["file"]
        cols = int(rec["shape"][1]) if len(rec["shape"]) == 2 else int(rec["shape"][0])
        t0 = time.perf_counter()
        mm = np.memmap(path, dtype="<u2", mode="r")
        sl = np.array(mm[int(row0) * cols : int(row1) * cols], dtype=np.uint16, copy=True)
        del mm
        self.stats.disk_read_bytes += int(sl.nbytes)
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.n_loads += 1
        self.stats.note_rss()
        if len(rec["shape"]) == 2:
            return sl.reshape(int(row1 - row0), cols)
        return sl


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(np.clip(x, -60, 60))
    return p / np.sum(p, axis=-1, keepdims=True)


def quality_metrics(ref: np.ndarray, hyp: np.ndarray) -> dict:
    d = hyp.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.max(np.abs(d)))
    mae = float(np.mean(np.abs(d)))
    pref = softmax_rows(ref)
    phyp = softmax_rows(hyp)
    kl = pref * (np.log(pref + 1e-12) - np.log(phyp + 1e-12))
    mean_kl = float(np.mean(np.sum(kl, axis=-1)))
    argmax_match = float(np.mean(np.argmax(ref, axis=-1) == np.argmax(hyp, axis=-1)))
    return {
        "lossy": True,
        "max_abs": max_abs,
        "mae": mae,
        "mean_kl_ref_to_hyp": mean_kl,
        "argmax_match": argmax_match,
        "note": "NOT bit-exact. Lossy int4 on bulk projections.",
    }


def run_hybrid_mode(
    *,
    model_dir: Path,
    hybrid_dir: Path,
    token_ids: np.ndarray,
    n_layers: int | None,
    encode_if_missing: bool,
    lm_head_chunk: int,
) -> dict:
    gc.collect()
    malloc_trim()
    rss0 = rss_bytes()
    max0 = maxrss_bytes()
    cfg = load_config_json(model_dir)
    specs = select_specs(model_dir)
    n_layer_cfg = int(cfg.get("num_hidden_layers") or 0)
    index_path = hybrid_dir / "index.json"
    if encode_if_missing or not index_path.is_file():
        print(f"encoding hybrid pack -> {hybrid_dir}", flush=True)
        encode_hybrid_dir(specs, hybrid_dir, n_layers=n_layer_cfg)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    src = HybridSource(index, hybrid_dir, specs)
    t_wall = time.perf_counter()
    logits = qwen_forward(
        src,
        specs,
        cfg,
        token_ids,
        n_layers=n_layers,
        lm_head_chunk=lm_head_chunk,
    )
    wall = time.perf_counter() - t_wall
    src.stats.note_rss()
    n_tok = int(np.asarray(token_ids).size)
    pack_bytes = int(index.get("pack_bytes") or 0)
    summary = {
        "disclaimer": DISCLAIMER,
        "mode": "hybrid",
        "source": src.name,
        "n_tokens": n_tok,
        "n_layers": n_layers or n_layer_cfg,
        "hidden_size": int(cfg.get("hidden_size") or 0),
        "vocab_size": int(cfg.get("vocab_size") or 0),
        "n_tensors": len(specs),
        "weight_bytes": sum(s.nbytes for s in specs.values()),
        "safetensors_bytes": 0,
        "hybrid_pack_bytes": pack_bytes,
        "hybrid_exact_bytes": int(index.get("exact_bytes") or 0),
        "hybrid_int4_bytes": int(index.get("int4_bytes") or 0),
        "n_exact": int(index.get("n_exact") or 0),
        "n_int4": int(index.get("n_int4") or 0),
        "logits_shape": list(logits.shape),
        "logits_sum": float(np.sum(logits, dtype=np.float64)),
        "logits_absmax": float(np.max(np.abs(logits))),
        "rss_before_bytes": rss0,
        "rss_after_bytes": rss_bytes(),
        "rss_peak_sampled_bytes": src.stats.peak_rss,
        "rss_peak_delta_bytes": max(src.stats.peak_rss - rss0, 0),
        "rss_maxrss_bytes": maxrss_bytes(),
        "rss_maxrss_delta_bytes": maxrss_bytes() - max0,
        "disk_read_bytes": src.stats.disk_read_bytes,
        "disk_read_gb": src.stats.disk_read_bytes / 1e9,
        "decode_s": src.stats.decode_s,
        "compute_s": src.stats.compute_s,
        "wall_s": wall,
        "n_loads": src.stats.n_loads,
        "n_matmuls": src.stats.n_matmuls,
        "matmuls_per_s": src.stats.n_matmuls / wall if wall else 0.0,
        "tokens_per_s": n_tok / wall if wall else 0.0,
        "pbre_exact_checks": src.stats.exact_checks,
        "pbre_exact_fail": src.stats.exact_fail,
        "exact": "LOSSY",
        "isolated_process": True,
        "pbre_dir_bytes": pack_bytes,
    }
    return {"summary": summary, "logits": logits, "index": index}


def format_hybrid_markdown(report: dict) -> str:
    body = format_markdown(report)
    body = body.replace("# Disk–RAM tunnel PoC", "# Hybrid lossy disk–RAM tunnel")
    body = body.replace(TUNNEL_DISCLAIMER, DISCLAIMER)
    q = report.get("quality") or {}
    extra = [
        "",
        "## Lossy quality (vs full BF16, same tokens)",
        "",
        "Quantized tensors are **not** bit-exact. Exact BF16 tensors (embed, norms, "
        "biases, first/last layer) still SHA-256 match.",
        "",
    ]
    if q:
        extra += [
            f"- logit max_abs: **{q.get('max_abs', float('nan')):.4g}**",
            f"- logit MAE: {q.get('mae', float('nan')):.4g}",
            f"- mean KL(ref || hyp): **{q.get('mean_kl_ref_to_hyp', float('nan')):.4g}**",
            f"- argmax match rate: **{q.get('argmax_match', float('nan')):.3f}**",
            "",
        ]
    hy = (report.get("modes") or {}).get("hybrid") or {}
    if hy:
        extra += [
            "## Hybrid pack size",
            "",
            f"- exact BF16 copies: {hy.get('n_exact', 0)} tensors, "
            f"{bytes_human(hy.get('hybrid_exact_bytes') or 0)}",
            f"- int4 + FP16 scales: {hy.get('n_int4', 0)} tensors, "
            f"{bytes_human(hy.get('hybrid_int4_bytes') or 0)}",
            f"- pack total: **{bytes_human(hy.get('hybrid_pack_bytes') or 0)}** "
            f"(not claimed as ≤8 BPW / 50%).",
            "",
        ]
    extra.append("Working-set + lossy-prototype demonstration only. Not a 27B phone runtime.")
    extra.append("")
    return body.rstrip() + "\n" + "\n".join(extra)


def _spawn(args: argparse.Namespace, mode: str, worker_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not pp else str(root) + os.pathsep + pp
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--model-dir",
        str(args.model_dir),
        "--hybrid-dir",
        str(args.hybrid_dir),
        "--tokens",
        str(args.tokens),
        "--layers",
        str(args.layers),
        "--lm-head-chunk",
        str(args.lm_head_chunk),
        "--config",
        str(args.config),
        "--worker-dir",
        str(worker_dir),
    ]
    subprocess.check_call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid lossy int4 disk-RAM tunnel PoC.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID_DIR)
    parser.add_argument("--pbre-dir", type=Path, default=Path("outputs/disk_tunnel/pbre"))
    parser.add_argument("--mode", choices=["full", "mmap", "hybrid", "all"], default="all")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--lm-head-chunk", type=int, default=LM_HEAD_ROWS)
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/tunnel_hybrid_lossy.json"))
    parser.add_argument("--md-out", type=Path, default=Path("artifacts/tunnel_hybrid_lossy.md"))
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    print(DISCLAIMER, flush=True)
    cfg = load_config_json(args.model_dir)
    ids = _token_ids(args.tokens, int(cfg.get("vocab_size") or 256), seed=1)
    n_layers = args.layers or None
    yaml_meta = {}
    if args.config.is_file():
        yaml_meta = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    if args.worker_dir is not None:
        if args.mode == "hybrid":
            out = run_hybrid_mode(
                model_dir=args.model_dir,
                hybrid_dir=args.hybrid_dir,
                token_ids=ids,
                n_layers=n_layers,
                encode_if_missing=False,
                lm_head_chunk=args.lm_head_chunk,
            )
        else:
            out = run_mode(
                mode=args.mode,
                model_dir=args.model_dir,
                pbre_dir=args.pbre_dir,
                token_ids=ids,
                n_layers=n_layers,
                encode_if_missing=False,
                verify_pbre=False,
                lm_head_chunk=args.lm_head_chunk,
            )
        args.worker_dir.mkdir(parents=True, exist_ok=True)
        (args.worker_dir / "summary.json").write_text(
            json.dumps(out["summary"], indent=2) + "\n", encoding="utf-8"
        )
        np.save(args.worker_dir / "logits.npy", out["logits"])
        return 0

    if args.encode or not (args.hybrid_dir / "index.json").is_file():
        specs = select_specs(args.model_dir)
        encode_hybrid_dir(
            specs,
            args.hybrid_dir,
            n_layers=int(cfg.get("num_hidden_layers") or 0),
        )

    modes_to_run = ["full", "mmap", "hybrid"] if args.mode == "all" else [args.mode]
    collected: dict[str, dict] = {}
    logits_by_mode: dict[str, np.ndarray] = {}
    isolated = (not args.in_process) and len(modes_to_run) > 1
    if isolated:
        for mode in modes_to_run:
            wdir = Path(tempfile.mkdtemp(prefix=f"pbr_hy_{mode}_"))
            _spawn(args, mode, wdir)
            s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
            collected[mode] = s
            logits_by_mode[mode] = np.load(wdir / "logits.npy")
            print(
                f"=== collected {mode} === peak={bytes_human(s['rss_peak_sampled_bytes'])} "
                f"wall={s['wall_s']:.3f}s exact={s.get('exact')}",
                flush=True,
            )
    else:
        for mode in modes_to_run:
            if mode == "hybrid":
                out = run_hybrid_mode(
                    model_dir=args.model_dir,
                    hybrid_dir=args.hybrid_dir,
                    token_ids=ids,
                    n_layers=n_layers,
                    encode_if_missing=False,
                    lm_head_chunk=args.lm_head_chunk,
                )
            else:
                out = run_mode(
                    mode=mode,
                    model_dir=args.model_dir,
                    pbre_dir=args.pbre_dir,
                    token_ids=ids,
                    n_layers=n_layers,
                    encode_if_missing=False,
                    verify_pbre=False,
                    lm_head_chunk=args.lm_head_chunk,
                )
            collected[mode] = out["summary"]
            logits_by_mode[mode] = out["logits"]
            del out
            gc.collect()
            malloc_trim()

    quality = {}
    match_notes: dict[str, str] = {}
    if "full" in logits_by_mode:
        ref = logits_by_mode["full"]
        for mode, arr in logits_by_mode.items():
            if mode == "full":
                continue
            if mode == "hybrid":
                quality = quality_metrics(ref, arr)
                match_notes[mode] = (
                    f"LOSSY max_abs={quality['max_abs']:.3e} KL={quality['mean_kl_ref_to_hyp']:.3e} "
                    f"argmax={quality['argmax_match']:.3f}"
                )
            else:
                delta = float(np.max(np.abs(arr.astype(np.float64) - ref.astype(np.float64))))
                match_notes[mode] = f"{'PASS' if delta < 1e-4 else 'FAIL'} max_abs={delta:.3e}"
                if delta >= 1e-4:
                    collected[mode]["exact"] = "FAIL"

    report = {
        "disclaimer": DISCLAIMER,
        "repo_id": yaml_meta.get("repo", PRIMARY_REPO),
        "revision": yaml_meta.get("revision", ""),
        "license": yaml_meta.get("license", PRIMARY_LICENSE),
        "model_dir": str(args.model_dir),
        "n_tokens": int(args.tokens),
        "n_layers": n_layers or int(cfg.get("num_hidden_layers") or 0),
        "token_ids": [int(x) for x in ids.tolist()],
        "modes": collected,
        "logits_match": match_notes or "single-mode",
        "quality": quality,
        "subprocess_isolated": isolated,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_hybrid_markdown(report)
    args.md_out.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {args.md_out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
