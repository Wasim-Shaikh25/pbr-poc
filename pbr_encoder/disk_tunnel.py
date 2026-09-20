"""Disk-resident weight tunnel: keep the checkpoint on disk, RAM holds a working set.

Mode A: mmap/read one BF16 tensor (or embedding rows) from Safetensors, GEMM, free.
Mode B: read a per-tensor PBR-E blob, bit-exact decode to uint16, GEMM, free.
Full-load baseline: all uint16 weights resident.

Compute uses FP32 matmul (BF16 bits reinterpreted). Archive path stays uint16.
Not a phone-scale 27B claim. Not an ≤8 BPW exact-compression claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from pbr_core.bf16 import make_bf16_bits
from pbr_core.hashing import sha256_words
from pbr_core.safetensors_io import (
    TWO_BYTE_DTYPES,
    TensorSpec,
    inventory_model,
    load_uint16,
    load_uint16_words,
    write_uint16_safetensors,
)
from pbr_core.tiles import as_2d
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.hf_weights import PRIMARY_LICENSE, PRIMARY_REPO
from pbr_encoder.profiles import PROFILES
from pbr_encoder.verification import assert_exact

DISCLAIMER = (
    "Disk-RAM tunnel PoC. Weights stay on disk; RAM holds a small working set "
    "(one tensor or embedding-row chunk + activations). Not a phone-scale 27B "
    "runtime. Not an ≤8 BPW exact-compression claim."
)
DEFAULT_MODEL = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
DEFAULT_PBRE_DIR = Path("outputs/disk_tunnel/pbre")
DEFAULT_TOKENS = 8
LM_HEAD_ROWS = 4096


def rss_bytes() -> int:
    path = Path("/proc/self/status")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return maxrss_bytes()


def maxrss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def malloc_trim() -> None:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def bytes_human(n: int | float) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(x) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.3f} {unit}"
        x /= 1024.0
    return f"{x:.3f} GiB"


def bf16_to_fp32(words: np.ndarray) -> np.ndarray:
    """Reinterpret BF16 bits as FP32 (high 16 bits). Compute path only."""
    u = np.ascontiguousarray(words, dtype=np.uint16)
    bits = u.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _release() -> None:
    gc.collect()
    malloc_trim()


@dataclass
class IoStats:
    disk_read_bytes: int = 0
    decode_s: float = 0.0
    compute_s: float = 0.0
    n_loads: int = 0
    n_matmuls: int = 0
    peak_rss: int = 0
    exact_checks: int = 0
    exact_fail: int = 0

    def note_rss(self) -> None:
        cur = rss_bytes()
        if cur > self.peak_rss:
            self.peak_rss = cur


@dataclass
class WeightSource:
    name: str
    stats: IoStats = field(default_factory=IoStats)

    def load(self, key: str) -> np.ndarray:
        raise NotImplementedError

    def load_row_range(self, key: str, row0: int, row1: int) -> np.ndarray:
        return self.load(key)[row0:row1]

    def load_compute(self, key: str) -> np.ndarray:
        """FP32 weights for GEMM / RMSNorm (compute path only)."""
        return bf16_to_fp32(self.load(key))

    def prefetch(self, key: str) -> None:
        return


class FullRamSource(WeightSource):
    """Baseline: every 16-bit tensor copied into a dict and kept."""

    def __init__(self, specs: dict[str, TensorSpec]):
        super().__init__(name="full_ram")
        self._data: dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        for key, spec in specs.items():
            words = load_uint16(spec)
            self.stats.disk_read_bytes += int(words.nbytes)
            self._data[key] = np.array(words, dtype=np.uint16, copy=True)
            self.stats.n_loads += 1
            self.stats.note_rss()
        self.stats.decode_s += time.perf_counter() - t0

    def load(self, key: str) -> np.ndarray:
        self.stats.n_loads += 1
        self.stats.note_rss()
        return self._data[key]


class MmapSafetensorsSource(WeightSource):
    """Mode A: copy one tensor or row slice from Safetensors, drop the mmap."""

    def __init__(self, specs: dict[str, TensorSpec], *, max_resident: int = 1):
        super().__init__(name="mmap_safetensors")
        self.specs = specs
        self.max_resident = max_resident
        self._hold: dict[str, np.ndarray] = {}

    def _remember(self, key: str, arr: np.ndarray) -> np.ndarray:
        if self.max_resident <= 0:
            return arr
        self._hold.clear()
        self._hold[key] = arr
        return arr

    def load(self, key: str) -> np.ndarray:
        spec = self.specs[key]
        t0 = time.perf_counter()
        words = load_uint16(spec)
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.disk_read_bytes += int(words.nbytes)
        self.stats.n_loads += 1
        self.stats.note_rss()
        return self._remember(key, words)

    def load_row_range(self, key: str, row0: int, row1: int) -> np.ndarray:
        spec = self.specs[key]
        if spec.ndim != 2:
            return self.load(key)[row0:row1]
        cols = int(spec.shape[1])
        n_rows = int(row1 - row0)
        t0 = time.perf_counter()
        flat = load_uint16_words(spec, int(row0) * cols, n_rows * cols)
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.disk_read_bytes += int(flat.nbytes)
        self.stats.n_loads += 1
        self.stats.note_rss()
        return flat.reshape(n_rows, cols)


class PbrDirSource(WeightSource):
    """Mode B: one PBR-E file per tensor; decode, use, drop."""

    def __init__(
        self,
        index: dict,
        pbre_dir: Path,
        specs: dict[str, TensorSpec] | None = None,
        *,
        verify_sha: bool = True,
        mmap_fallback: dict[str, TensorSpec] | None = None,
    ):
        super().__init__(name="pbre_dir")
        self.index = index
        self.pbre_dir = Path(pbre_dir)
        self.specs = specs or {}
        self.verify_sha = verify_sha
        self.mmap_fallback = mmap_fallback or {}

    def load(self, key: str) -> np.ndarray:
        if key not in self.index["tensors"] and key in self.mmap_fallback:
            spec = self.mmap_fallback[key]
            t0 = time.perf_counter()
            words = load_uint16(spec)
            self.stats.decode_s += time.perf_counter() - t0
            self.stats.disk_read_bytes += int(words.nbytes)
            self.stats.n_loads += 1
            self.stats.note_rss()
            return words
        rec = self.index["tensors"][key]
        path = self.pbre_dir / rec["file"]
        t0 = time.perf_counter()
        blob = path.read_bytes()
        self.stats.disk_read_bytes += len(blob)
        from pbr_core.container import PBRContainer

        container = PBRContainer.loads(blob)
        decoded = decode_container(container)[0]
        logical = tuple(int(x) for x in rec["shape"])
        words = decoded.reshape(logical)
        self.stats.decode_s += time.perf_counter() - t0
        self.stats.n_loads += 1
        if self.verify_sha:
            got = sha256_words(words)
            if got != rec["sha256"]:
                self.stats.exact_fail += 1
                raise AssertionError(f"PBR-E SHA mismatch {key}: {got} != {rec['sha256']}")
            self.stats.exact_checks += 1
        del blob, container, decoded
        self.stats.note_rss()
        return words

    def load_row_range(self, key: str, row0: int, row1: int) -> np.ndarray:
        if key in self.mmap_fallback and key not in self.index["tensors"]:
            mmap = MmapSafetensorsSource(self.mmap_fallback)
            mmap.stats = self.stats
            return mmap.load_row_range(key, row0, row1)
        return self.load(key)[row0:row1]


class FastPbrDirSource(PbrDirSource):
    """PBR-E tunnel with C rANS, a 2-slot decoded cache, and one prefetch thread."""

    def __init__(
        self,
        index: dict,
        pbre_dir: Path,
        specs: dict[str, TensorSpec] | None = None,
        *,
        verify_sha: bool = True,
        mmap_fallback: dict[str, TensorSpec] | None = None,
        cache_slots: int = 2,
        prefetch: bool = True,
    ):
        super().__init__(
            index,
            pbre_dir,
            specs=specs,
            verify_sha=verify_sha,
            mmap_fallback=mmap_fallback,
        )
        self.name = "pbre_fast"
        self.cache_slots = max(1, int(cache_slots))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="pbre-prefetch")
            if prefetch
            else None
        )
        self._fut: Future | None = None
        self._fut_key: str | None = None

    def close(self) -> None:
        with self._lock:
            fut = self._fut
            self._fut = None
            self._fut_key = None
        if fut is not None:
            try:
                fut.result(timeout=120)
            except Exception:
                pass
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._cache.clear()

    def prefetch(self, key: str) -> None:
        if self._pool is None or not key:
            return
        with self._lock:
            if key in self._cache or key == self._fut_key:
                return
            if self._fut is not None and not self._fut.done():
                return
            self._fut_key = key
            self._fut = self._pool.submit(self._decode_job, key)

    def _remember_decoded(self, key: str, arr: np.ndarray) -> np.ndarray:
        self._cache[key] = arr
        while len(self._cache) > self.cache_slots:
            self._cache.popitem(last=False)
        return arr

    def _decode_job(self, key: str) -> tuple[str, np.ndarray, int, float, int, int]:
        """Thread worker: decode without touching IoStats."""
        t0 = time.perf_counter()
        disk_n = 0
        checks = 0
        fail = 0
        if key not in self.index["tensors"] and key in self.mmap_fallback:
            words = load_uint16(self.mmap_fallback[key])
            disk_n = int(words.nbytes)
            return key, words, disk_n, time.perf_counter() - t0, checks, fail
        rec = self.index["tensors"][key]
        path = self.pbre_dir / rec["file"]
        blob = path.read_bytes()
        disk_n = len(blob)
        from pbr_core.container import PBRContainer

        container = PBRContainer.loads(blob)
        decoded = decode_container(container)[0]
        words = decoded.reshape(tuple(int(x) for x in rec["shape"]))
        if self.verify_sha:
            got = sha256_words(words)
            if got != rec["sha256"]:
                fail = 1
                raise AssertionError(f"PBR-E SHA mismatch {key}: {got} != {rec['sha256']}")
            checks = 1
        del blob, container, decoded
        return key, words, disk_n, time.perf_counter() - t0, checks, fail

    def load(self, key: str) -> np.ndarray:
        with self._lock:
            if key in self._cache:
                arr = self._cache.pop(key)
                self._cache[key] = arr
                self.stats.n_loads += 1
                self.stats.note_rss()
                return arr
            fut, fkey = self._fut, self._fut_key
        if fkey == key and fut is not None:
            _k, arr, disk_n, dec_s, checks, fail = fut.result()
            with self._lock:
                if self._fut is fut:
                    self._fut = None
                    self._fut_key = None
            self.stats.disk_read_bytes += disk_n
            self.stats.decode_s += dec_s
            self.stats.n_loads += 1
            self.stats.exact_checks += checks
            self.stats.exact_fail += fail
            self.stats.note_rss()
            with self._lock:
                self._remember_decoded(key, arr)
            return arr
        _k, arr, disk_n, dec_s, checks, fail = self._decode_job(key)
        self.stats.disk_read_bytes += disk_n
        self.stats.decode_s += dec_s
        self.stats.n_loads += 1
        self.stats.exact_checks += checks
        self.stats.exact_fail += fail
        self.stats.note_rss()
        with self._lock:
            self._remember_decoded(key, arr)
        return arr


def _pbre_name(key: str) -> str:
    return key.replace("/", ".") + ".pbr"


def encode_pbre_dir(
    specs: dict[str, TensorSpec],
    dest: Path,
    *,
    skip_keys: set[str] | None = None,
    profile: str = "pbre_whole",
    block_size: int = 65535,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    prof = PROFILES[profile]
    skip = skip_keys or set()
    index: dict = {
        "disclaimer": DISCLAIMER,
        "profile": profile,
        "block_size": block_size,
        "tensors": {},
    }
    for key, spec in specs.items():
        if key in skip:
            continue
        print(f"  pbre-encode {key}  {list(spec.shape)}  {spec.nbytes} B", flush=True)
        words = load_uint16(spec)
        digest = sha256_words(words)
        container = encode_tensor(
            as_2d(words),
            name=key,
            block_size=block_size,
            codecs=prof["codecs"],
            whole_codecs=prof["whole_codecs"],
            enable_whole=bool(prof["enable_whole"]),
            extra={"original_shape": list(spec.shape), "stage": "disk-tunnel"},
        )
        blob = container.dumps()
        fname = _pbre_name(key)
        (dest / fname).write_bytes(blob)
        restored = decode_container(container)[0].reshape(spec.shape)
        assert_exact(words, restored, label=f"encode:{key}")
        index["tensors"][key] = {
            "file": fname,
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "sha256": digest,
            "n_words": spec.n_words,
            "original_bytes": spec.nbytes,
            "encoded_bytes": len(blob),
        }
        del words, restored, container, blob
        _release()
    orig = sum(int(t["original_bytes"]) for t in index["tensors"].values())
    enc = sum(int(t["encoded_bytes"]) for t in index["tensors"].values())
    index["n_tensors"] = len(index["tensors"])
    index["original_bytes_encoded"] = orig
    index["encoded_bytes_total"] = enc
    index["skipped"] = sorted(skip)
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(
        f"  encoded {index['n_tensors']} tensors  {orig} B -> {enc} B  "
        f"skipped {sorted(skip)}",
        flush=True,
    )
    return index


def _rand_bf16(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.integers(120, 130, size=n, dtype=np.uint16),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    )
    return words.reshape(shape)


def write_tiny_qwen_fixture(
    dest: Path,
    *,
    n_layers: int = 2,
    hidden: int = 32,
    n_heads: int = 4,
    n_kv: int = 2,
    intermediate: int = 64,
    vocab: int = 80,
    seed: int = 0,
) -> dict:
    """Local Qwen2-shaped BF16 toy checkpoint. Not a HuggingFace download."""
    if hidden % n_heads != 0:
        raise ValueError("hidden must divide n_heads")
    dest.mkdir(parents=True, exist_ok=True)
    head_dim = hidden // n_heads
    kv_dim = n_kv * head_dim
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": _rand_bf16((vocab, hidden), rng),
        "model.norm.weight": _rand_bf16((hidden,), rng),
    }
    for li in range(n_layers):
        p = f"model.layers.{li}"
        tensors[f"{p}.input_layernorm.weight"] = _rand_bf16((hidden,), rng)
        tensors[f"{p}.post_attention_layernorm.weight"] = _rand_bf16((hidden,), rng)
        tensors[f"{p}.self_attn.q_proj.weight"] = _rand_bf16((hidden, hidden), rng)
        tensors[f"{p}.self_attn.q_proj.bias"] = _rand_bf16((hidden,), rng)
        tensors[f"{p}.self_attn.k_proj.weight"] = _rand_bf16((kv_dim, hidden), rng)
        tensors[f"{p}.self_attn.k_proj.bias"] = _rand_bf16((kv_dim,), rng)
        tensors[f"{p}.self_attn.v_proj.weight"] = _rand_bf16((kv_dim, hidden), rng)
        tensors[f"{p}.self_attn.v_proj.bias"] = _rand_bf16((kv_dim,), rng)
        tensors[f"{p}.self_attn.o_proj.weight"] = _rand_bf16((hidden, hidden), rng)
        tensors[f"{p}.mlp.gate_proj.weight"] = _rand_bf16((intermediate, hidden), rng)
        tensors[f"{p}.mlp.up_proj.weight"] = _rand_bf16((intermediate, hidden), rng)
        tensors[f"{p}.mlp.down_proj.weight"] = _rand_bf16((hidden, intermediate), rng)
    write_uint16_safetensors(
        dest / "model.safetensors",
        tensors,
        storage_dtype="BF16",
        metadata={"note": "tiny disk-tunnel fixture, not a real checkpoint"},
    )
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def load_config_json(model_dir: Path) -> dict:
    path = model_dir / "config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _as_fp32(weight: np.ndarray) -> np.ndarray:
    if weight.dtype == np.uint16:
        return bf16_to_fp32(weight)
    return np.asarray(weight, dtype=np.float32)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None, stats: IoStats) -> np.ndarray:
    t0 = time.perf_counter()
    w = _as_fp32(weight)
    y = x @ w.T
    if bias is not None:
        y = y + _as_fp32(bias)
    stats.compute_s += time.perf_counter() - t0
    stats.n_matmuls += 1
    stats.note_rss()
    return y.astype(np.float32, copy=False)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float, stats: IoStats) -> np.ndarray:
    t0 = time.perf_counter()
    w = _as_fp32(weight)
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    y = x.astype(np.float32) * np.reciprocal(np.sqrt(var + eps)) * w
    stats.compute_s += time.perf_counter() - t0
    stats.note_rss()
    return y.astype(np.float32, copy=False)


def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return x32 * (1.0 / (1.0 + np.exp(-np.clip(x32, -60.0, 60.0))))


def _rope_cos_sin(seqlen: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seqlen, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    return np.repeat(cos, 2, axis=-1), np.repeat(sin, 2, axis=-1)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rot = np.empty_like(x)
    rot[..., 0::2] = -odd
    rot[..., 1::2] = even
    return rot


def apply_rope(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = cos[:, None, :]
    s = sin[:, None, :]
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


def attend(q: np.ndarray, k: np.ndarray, v: np.ndarray, stats: IoStats) -> np.ndarray:
    t0 = time.perf_counter()
    hq = q.shape[1]
    hkv = k.shape[1]
    tlen = q.shape[0]
    d = q.shape[2]
    k = np.repeat(k, hq // hkv, axis=1)
    v = np.repeat(v, hq // hkv, axis=1)
    scale = 1.0 / np.sqrt(d)
    scores = np.einsum("thd,shd->hts", q, k) * scale
    causal = np.triu(np.ones((tlen, tlen), dtype=np.float32), k=1) * (-1e9)
    scores = scores + causal[None, :, :]
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / np.sum(p, axis=-1, keepdims=True)
    out = np.einsum("hts,shd->thd", p, v)
    stats.compute_s += time.perf_counter() - t0
    stats.n_matmuls += 1
    return out.astype(np.float32, copy=False)


def _get(src: WeightSource, specs: dict[str, TensorSpec], name: str) -> np.ndarray | None:
    if name not in specs:
        return None
    return src.load_compute(name)


def _pf(src: WeightSource, name: str) -> None:
    src.prefetch(name)


def embed_tokens(src: WeightSource, name: str, ids: np.ndarray) -> np.ndarray:
    rows = []
    for tok in ids.tolist():
        sl = src.load_row_range(name, int(tok), int(tok) + 1)
        rows.append(bf16_to_fp32(sl)[0])
        del sl
    return np.stack(rows, axis=0).astype(np.float32)


def lm_head_chunked(
    src: WeightSource,
    name: str,
    hidden: np.ndarray,
    vocab: int,
    width: int,
    stats: IoStats,
) -> np.ndarray:
    tlen = hidden.shape[0]
    logits = np.empty((tlen, vocab), dtype=np.float32)
    for row0 in range(0, vocab, width):
        row1 = min(vocab, row0 + width)
        sl = src.load_row_range(name, row0, row1)
        t0 = time.perf_counter()
        w = bf16_to_fp32(sl)
        logits[:, row0:row1] = hidden @ w.T
        stats.compute_s += time.perf_counter() - t0
        stats.n_matmuls += 1
        del sl, w
        gc.collect()
        stats.note_rss()
    return logits


def qwen_forward(
    src: WeightSource,
    specs: dict[str, TensorSpec],
    cfg: dict,
    token_ids: np.ndarray,
    *,
    n_layers: int | None = None,
    lm_head_chunk: int = LM_HEAD_ROWS,
) -> np.ndarray:
    hidden_size = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads") or n_heads)
    n_layer = int(cfg["num_hidden_layers"])
    if n_layers is not None:
        n_layer = min(n_layer, int(n_layers))
    eps = float(cfg.get("rms_norm_eps") or 1e-6)
    theta = float(cfg.get("rope_theta") or 10000.0)
    vocab = int(cfg["vocab_size"])
    head_dim = hidden_size // n_heads
    ids = np.asarray(token_ids, dtype=np.int64).ravel()
    tlen = int(ids.size)

    embed_name = "model.embed_tokens.weight"
    x = embed_tokens(src, embed_name, ids)
    src.stats.note_rss()
    cos, sin = _rope_cos_sin(tlen, head_dim, theta)

    for li in range(n_layer):
        prefix = f"model.layers.{li}"
        _pf(src, f"{prefix}.self_attn.q_proj.weight")
        n1 = src.load_compute(f"{prefix}.input_layernorm.weight")
        h = rms_norm(x, n1, eps, src.stats)
        del n1
        _pf(src, f"{prefix}.self_attn.k_proj.weight")
        wq = src.load_compute(f"{prefix}.self_attn.q_proj.weight")
        bq = _get(src, specs, f"{prefix}.self_attn.q_proj.bias")
        q = _linear(h, wq, bq, src.stats).reshape(tlen, n_heads, head_dim)
        del wq, bq
        _pf(src, f"{prefix}.self_attn.v_proj.weight")
        wk = src.load_compute(f"{prefix}.self_attn.k_proj.weight")
        bk = _get(src, specs, f"{prefix}.self_attn.k_proj.bias")
        k = _linear(h, wk, bk, src.stats).reshape(tlen, n_kv, head_dim)
        del wk, bk
        _pf(src, f"{prefix}.self_attn.o_proj.weight")
        wv = src.load_compute(f"{prefix}.self_attn.v_proj.weight")
        bv = _get(src, specs, f"{prefix}.self_attn.v_proj.bias")
        v = _linear(h, wv, bv, src.stats).reshape(tlen, n_kv, head_dim)
        del wv, bv
        q, k = apply_rope(q, k, cos, sin)
        attn = attend(q, k, v, src.stats).reshape(tlen, hidden_size)
        del q, k, v
        _pf(src, f"{prefix}.post_attention_layernorm.weight")
        wo = src.load_compute(f"{prefix}.self_attn.o_proj.weight")
        x = x + _linear(attn, wo, None, src.stats)
        del wo, attn, h
        _release()

        _pf(src, f"{prefix}.mlp.gate_proj.weight")
        n2 = src.load_compute(f"{prefix}.post_attention_layernorm.weight")
        h = rms_norm(x, n2, eps, src.stats)
        del n2
        _pf(src, f"{prefix}.mlp.up_proj.weight")
        wg = src.load_compute(f"{prefix}.mlp.gate_proj.weight")
        g = silu(_linear(h, wg, None, src.stats))
        del wg
        _pf(src, f"{prefix}.mlp.down_proj.weight")
        wu = src.load_compute(f"{prefix}.mlp.up_proj.weight")
        u = _linear(h, wu, None, src.stats)
        del wu, h
        if li + 1 < n_layer:
            _pf(src, f"model.layers.{li + 1}.input_layernorm.weight")
        else:
            _pf(src, "model.norm.weight")
        wd = src.load_compute(f"{prefix}.mlp.down_proj.weight")
        x = x + _linear(g * u, wd, None, src.stats)
        del wd, g, u
        _release()
        src.stats.note_rss()

    nw = src.load_compute("model.norm.weight")
    x = rms_norm(x, nw, eps, src.stats)
    del nw
    logits = lm_head_chunked(src, embed_name, x, vocab, lm_head_chunk, src.stats)
    src.stats.note_rss()
    return logits


def select_specs(model_dir: Path) -> dict[str, TensorSpec]:
    specs = [s for s in inventory_model(model_dir) if s.dtype in TWO_BYTE_DTYPES]
    return {s.name: s for s in specs}


def run_mode(
    *,
    mode: str,
    model_dir: Path,
    pbre_dir: Path,
    token_ids: np.ndarray,
    n_layers: int | None,
    encode_if_missing: bool,
    verify_pbre: bool,
    lm_head_chunk: int,
) -> dict:
    gc.collect()
    malloc_trim()
    rss0 = rss_bytes()
    max0 = maxrss_bytes()
    cfg = load_config_json(model_dir)
    specs = select_specs(model_dir)
    t_wall = time.perf_counter()
    mmap_fallback = {k: v for k, v in specs.items() if k.endswith("embed_tokens.weight")}
    rans_backend = None

    if mode == "full":
        src: WeightSource = FullRamSource(specs)
    elif mode == "mmap":
        src = MmapSafetensorsSource(specs, max_resident=1)
    elif mode in {"pbre", "pbre_fast", "pbre_slow"}:
        from pbr_core.rans import rans_impl

        index_path = pbre_dir / "index.json"
        skip = set(mmap_fallback)
        if encode_if_missing and not index_path.is_file():
            print(f"encoding PBR-E per-tensor dir -> {pbre_dir} (embed stays mmap)", flush=True)
            encode_pbre_dir(specs, pbre_dir, skip_keys=skip)
        if not index_path.is_file():
            raise SystemExit(f"missing {index_path}; pass --encode")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if mode == "pbre_slow":
            os.environ["PBR_RANS_IMPL"] = "python"
            src = PbrDirSource(
                index,
                pbre_dir,
                specs=specs if verify_pbre else {},
                verify_sha=verify_pbre,
                mmap_fallback=mmap_fallback,
            )
            src.name = "pbre_slow"
        else:
            os.environ["PBR_RANS_IMPL"] = "c"
            src = FastPbrDirSource(
                index,
                pbre_dir,
                specs=specs if verify_pbre else {},
                verify_sha=verify_pbre,
                mmap_fallback=mmap_fallback,
                cache_slots=2,
                prefetch=True,
            )
            if mode == "pbre":
                src.name = "pbre_dir"
        summary_impl = rans_impl()
        rans_backend = summary_impl
    else:
        raise SystemExit(f"unknown mode {mode}")

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
    closer = getattr(src, "close", None)
    if callable(closer):
        closer()
    n_tok = int(np.asarray(token_ids).size)
    weight_files = {s.file_path.resolve() for s in specs.values()}
    safetensors_bytes = sum(p.stat().st_size for p in weight_files if p.is_file())
    pbre_bytes = 0
    if mode.startswith("pbre") and (pbre_dir / "index.json").is_file():
        pbre_bytes = sum(
            p.stat().st_size for p in pbre_dir.glob("*.pbr") if p.is_file()
        )
    summary = {
        "disclaimer": DISCLAIMER,
        "mode": mode,
        "source": src.name,
        "n_tokens": n_tok,
        "n_layers": n_layers or int(cfg.get("num_hidden_layers") or 0),
        "hidden_size": int(cfg.get("hidden_size") or 0),
        "vocab_size": int(cfg.get("vocab_size") or 0),
        "n_tensors": len(specs),
        "weight_bytes": sum(s.nbytes for s in specs.values()),
        "safetensors_bytes": int(safetensors_bytes),
        "pbre_dir_bytes": int(pbre_bytes),
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
        "exact": "FAIL" if src.stats.exact_fail else "PASS",
        "isolated_process": True,
        "rans_impl": rans_backend,
    }
    summary["logits_checksum"] = struct.pack("<dd", summary["logits_sum"], summary["logits_absmax"]).hex()
    return {"summary": summary, "logits": logits}


def format_markdown(report: dict) -> str:
    modes = report["modes"]
    lines = [
        "# Disk–RAM tunnel PoC",
        "",
        DISCLAIMER,
        "",
        f"Model: `{report.get('repo_id', PRIMARY_REPO)}`  "
        f"revision `{report.get('revision', '')}`  "
        f"tokens={report.get('n_tokens')}  layers={report.get('n_layers')}.",
        "",
        "Full BF16 checkpoint stays on disk (Safetensors). The tunnel copies **one** "
        "weight tensor (or an embedding-row chunk) into RAM, converts BF16 bits to "
        "FP32 for GEMM, then frees it. PBR-E mode decodes a per-tensor container "
        "bit-exactly before the GEMM. Embedding / lm_head uses row chunks so the "
        "large table is not required resident. Not a 27B-on-phone result.",
        "",
        "## Metrics",
        "",
        "| mode | peak sampled RSS | ru_maxrss | disk read | decode s | compute s | wall s | tok/s | matmuls/s | exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("full", "mmap", "pbre", "pbre_fast", "pbre_slow", "hybrid"):
        if name not in modes:
            continue
        s = modes[name]
        lines.append(
            f"| `{name}` | {bytes_human(s['rss_peak_sampled_bytes'])} | "
            f"{bytes_human(s['rss_maxrss_bytes'])} | "
            f"{bytes_human(s['disk_read_bytes'])} | {s['decode_s']:.3f} | "
            f"{s['compute_s']:.3f} | {s['wall_s']:.3f} | {s['tokens_per_s']:.3f} | "
            f"{s['matmuls_per_s']:.2f} | {s['exact']} |"
        )
    full = modes.get("full") or {}
    mmap = modes.get("mmap") or {}
    pbre = modes.get("pbre") or modes.get("pbre_fast") or {}
    pbre_slow = modes.get("pbre_slow") or {}
    pbre_fast = modes.get("pbre_fast") or {}
    hybrid = modes.get("hybrid") or {}
    if full and mmap and full.get("rss_maxrss_bytes"):
        ratio = mmap["rss_maxrss_bytes"] / max(full["rss_maxrss_bytes"], 1)
        samp = mmap["rss_peak_sampled_bytes"] / max(full["rss_peak_sampled_bytes"], 1)
        lines += [
            "",
            f"mmap/full ru_maxrss ratio: **{ratio:.3f}**. "
            f"Sampled peak mmap/full: **{samp:.3f}** "
            f"({bytes_human(mmap['rss_peak_sampled_bytes'])} vs "
            f"{bytes_human(full['rss_peak_sampled_bytes'])}).",
        ]
    for label, row in (("pbre", pbre), ("pbre_fast", pbre_fast), ("pbre_slow", pbre_slow), ("hybrid", hybrid)):
        if not (full and row and full.get("rss_maxrss_bytes") and row.get("rss_maxrss_bytes")):
            continue
        if label == "pbre" and pbre_fast and row is pbre_fast:
            continue
        ratio_p = row["rss_maxrss_bytes"] / max(full["rss_maxrss_bytes"], 1)
        samp_p = row["rss_peak_sampled_bytes"] / max(full["rss_peak_sampled_bytes"], 1)
        lines.append(
            f"{label}/full ru_maxrss ratio: **{ratio_p:.3f}**. "
            f"Sampled peak {label}/full: **{samp_p:.3f}**. "
            f"SHA checks: {row.get('pbre_exact_checks', 0)} "
            f"(fail {row.get('pbre_exact_fail', 0)})."
        )
    if report.get("subprocess_isolated"):
        lines += [
            "",
            "Each mode ran in its **own subprocess** so `ru_maxrss` is not a "
            "cumulative high-water mark of full-load + tunnel.",
        ]
    match = report.get("logits_match")
    if isinstance(match, dict):
        match_txt = "; ".join(f"{k}: {v}" for k, v in match.items()) or "n/a"
    else:
        match_txt = str(match)
    lines += [
        "",
        "## Correctness",
        "",
        f"Logits vs full-load: {match_txt}. Same uint16 weights and the same FP32 GEMM.",
        "",
    ]
    enc = report.get("pbre_encode") or {}
    if full:
        lines += [
            "## Disk layout",
            "",
            f"- Safetensors on disk: {bytes_human(full.get('safetensors_bytes') or 0)} "
            f"({bytes_human(full.get('weight_bytes') or 0)} weight payload, "
            f"{full.get('n_tensors', 0)} tensors).",
        ]
    if enc:
        orig = int(enc.get("original_bytes_encoded") or 0)
        nenc = int(enc.get("encoded_bytes_total") or 0)
        n_words = int(enc.get("n_words") or 0)
        bpw = (8.0 * nenc / n_words) if n_words else 0.0
        lines.append(
            f"- PBR-E per-tensor dir: {enc.get('n_tensors', 0)} files, "
            f"{bytes_human(orig)} → {bytes_human(nenc)} "
            f"({bpw:.3f} complete BPW on encoded tensors). "
            f"Skipped mmap: {enc.get('skipped', [])}."
        )
    lines += [
        "",
        "`ru_maxrss` is the kernel high-water for that isolated process. "
        "Sampled peak is max `/proc/self/status` VmRSS during the forward "
        "(the number that tracks the working set after malloc_trim).",
        "",
        "Working-set demonstration only. PBR-E is the already-measured ~10.6 BPW "
        "DF11-class codec; this tunnel does not claim ≤8 BPW.",
        "",
    ]
    return "\n".join(lines)


def _token_ids(n: int, vocab: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, max(vocab, 1), size=n, dtype=np.int64)


def _maybe_encode_pbre(model_dir: Path, pbre_dir: Path) -> None:
    specs = select_specs(model_dir)
    skip = {k for k in specs if k.endswith("embed_tokens.weight")}
    print(f"encoding PBR-E per-tensor dir -> {pbre_dir} (embed stays mmap)", flush=True)
    encode_pbre_dir(specs, pbre_dir, skip_keys=skip)


def _spawn_isolated_mode(args: argparse.Namespace, mode: str, worker_dir: Path) -> None:
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
        "--pbre-dir",
        str(args.pbre_dir),
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
    if args.no_verify:
        cmd.append("--no-verify")
    subprocess.check_call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Disk-resident BF16/PBR-E inference tunnel PoC.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pbre-dir", type=Path, default=DEFAULT_PBRE_DIR)
    parser.add_argument(
        "--mode",
        choices=["full", "mmap", "pbre", "pbre_fast", "pbre_slow", "all"],
        default="all",
    )
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--layers", type=int, default=0, help="0 = all layers")
    parser.add_argument("--encode", action="store_true", help="Build per-tensor PBR-E dir if missing")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--lm-head-chunk", type=int, default=LM_HEAD_ROWS)
    parser.add_argument("--config", type=Path, default=Path("configs/poc_real.yaml"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/disk_ram_tunnel.json"))
    parser.add_argument("--md-out", type=Path, default=Path("artifacts/disk_ram_tunnel.md"))
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run every mode in this process (ru_maxrss becomes cumulative; tests only)",
    )
    parser.add_argument("--worker-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    print(DISCLAIMER, flush=True)
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model dir {args.model_dir}")
    cfg = load_config_json(args.model_dir)
    vocab = int(cfg.get("vocab_size") or 256)
    ids = _token_ids(args.tokens, vocab, seed=1)
    n_layers = args.layers or None
    yaml_meta = {}
    if args.config.is_file():
        yaml_meta = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    def _one(mode: str, encode: bool) -> dict:
        print(f"=== mode {mode} ===", flush=True)
        out = run_mode(
            mode=mode,
            model_dir=args.model_dir,
            pbre_dir=args.pbre_dir,
            token_ids=ids,
            n_layers=n_layers,
            encode_if_missing=encode,
            verify_pbre=not args.no_verify,
            lm_head_chunk=args.lm_head_chunk,
        )
        s = out["summary"]
        print(
            f"  peak_sampled={bytes_human(s['rss_peak_sampled_bytes'])}  "
            f"maxrss={bytes_human(s['rss_maxrss_bytes'])}  "
            f"disk={bytes_human(s['disk_read_bytes'])}  decode={s['decode_s']:.2f}s  "
            f"compute={s['compute_s']:.2f}s  wall={s['wall_s']:.2f}s  exact={s['exact']}",
            flush=True,
        )
        return out

    if args.worker_dir is not None:
        if args.mode == "all":
            raise SystemExit("worker-dir requires a single --mode")
        encode = bool(args.encode) or (
            args.mode == "pbre" and not (args.pbre_dir / "index.json").is_file()
        )
        out = _one(args.mode, encode)
        args.worker_dir.mkdir(parents=True, exist_ok=True)
        (args.worker_dir / "summary.json").write_text(
            json.dumps(out["summary"], indent=2) + "\n", encoding="utf-8"
        )
        np.save(args.worker_dir / "logits.npy", out["logits"])
        return 0 if out["summary"].get("exact") != "FAIL" else 1

    modes_to_run = ["full", "mmap", "pbre"] if args.mode == "all" else [args.mode]
    collected: dict[str, dict] = {}
    logits_by_mode: dict[str, np.ndarray] = {}
    match_notes: dict[str, str] = {}
    isolated = (not args.in_process) and len(modes_to_run) > 1

    if "pbre" in modes_to_run:
        need = args.encode or not (args.pbre_dir / "index.json").is_file()
        if need:
            _maybe_encode_pbre(args.model_dir, args.pbre_dir)

    if isolated:
        for mode in modes_to_run:
            wdir = Path(tempfile.mkdtemp(prefix=f"pbr_tunnel_{mode}_"))
            _spawn_isolated_mode(args, mode, wdir)
            s = json.loads((wdir / "summary.json").read_text(encoding="utf-8"))
            s["isolated_process"] = True
            collected[mode] = s
            logits_by_mode[mode] = np.load(wdir / "logits.npy")
            print(
                f"=== collected {mode} ===  peak_sampled="
                f"{bytes_human(s['rss_peak_sampled_bytes'])}  "
                f"maxrss={bytes_human(s['rss_maxrss_bytes'])}  exact={s['exact']}",
                flush=True,
            )
    else:
        for mode in modes_to_run:
            out = _one(mode, encode=False)
            s = out["summary"]
            s["isolated_process"] = False
            collected[mode] = s
            logits_by_mode[mode] = out["logits"]
            del out
            gc.collect()
            malloc_trim()

    if "full" in logits_by_mode:
        ref = logits_by_mode["full"]
        for mode, arr in logits_by_mode.items():
            if mode == "full":
                continue
            delta = float(np.max(np.abs(arr.astype(np.float64) - ref.astype(np.float64))))
            ok = delta < 1e-4
            match_notes[mode] = f"{'PASS' if ok else 'FAIL'} max_abs={delta:.3e}"
            if not ok:
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
        "subprocess_isolated": isolated,
    }
    index_path = args.pbre_dir / "index.json"
    if index_path.is_file():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        n_words = sum(int(t.get("n_words") or 0) for t in idx.get("tensors", {}).values())
        report["pbre_encode"] = {
            "n_tensors": idx.get("n_tensors"),
            "original_bytes_encoded": idx.get("original_bytes_encoded"),
            "encoded_bytes_total": idx.get("encoded_bytes_total"),
            "n_words": n_words,
            "skipped": idx.get("skipped"),
            "profile": idx.get("profile"),
        }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(report)
    args.md_out.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {args.md_out} and {args.json_out}")
    fails = [m for m, s in collected.items() if s.get("exact") == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
