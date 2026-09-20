"""Optional general-purpose baselines. Labeled as baselines, not PBR."""

from __future__ import annotations

import zlib

import numpy as np

import math

from pbr_core.bf16 import split_components
from pbr_core.hashing import words_to_bytes
from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.types import TILE_HEADER_BYTES
from pbr_qualifier.entropy import shannon_entropy


def _try_zstd(data: bytes) -> int | None:
    try:
        import zstandard
    except ImportError:
        return None
    cctx = zstandard.ZstdCompressor(level=3)
    return len(cctx.compress(data))


def baseline_sizes(words: np.ndarray) -> dict[str, dict[str, float | int | str]]:
    raw = words_to_bytes(words)
    n = int(np.asarray(words).size)
    zlib_len = len(zlib.compress(raw, level=9))
    out: dict[str, dict[str, float | int | str]] = {
        "zlib": {
            "label": "baseline (not PBR)",
            "encoded_bytes": zlib_len,
            "bpw": bits_per_weight(zlib_len, n),
            "ratio_vs_raw_bf16": compression_ratio(zlib_len, len(raw)),
        }
    }
    zstd_len = _try_zstd(raw)
    if zstd_len is not None:
        out["zstd"] = {
            "label": "baseline (not PBR)",
            "encoded_bytes": zstd_len,
            "bpw": bits_per_weight(zstd_len, n),
            "ratio_vs_raw_bf16": compression_ratio(zstd_len, len(raw)),
        }
    else:
        out["zstd"] = {
            "label": "baseline (not PBR)",
            "encoded_bytes": None,
            "note": "zstandard package not installed",
        }
    bound = exponent_huffman_bound_bytes(words)
    out["exp_huffman_bound"] = {
        "label": "baseline DF11-style exponent-Huffman bound (not PBR)",
        "encoded_bytes": bound,
        "bpw": bits_per_weight(bound, n),
        "ratio_vs_raw_bf16": compression_ratio(bound, len(raw)),
    }
    return out


def exponent_huffman_bound_bytes(words: np.ndarray) -> int:
    """Information-theoretic exponent Huffman + raw sign/mantissa + codebook.

    This is a size *bound*, not a PBR result. It omits a full container JSON
    header but includes a tile prefix and a compact codebook estimate.
    """
    flat = np.ascontiguousarray(words)
    n = int(flat.size)
    if n == 0:
        return TILE_HEADER_BYTES
    _sign, exp, _mant = split_components(flat)
    entropy = shannon_entropy(exp)
    unique = int(np.unique(exp).size)
    codebook = 2 + unique * 2
    exp_bytes = int(math.ceil(n * entropy / 8.0)) if entropy > 0 else 0
    return TILE_HEADER_BYTES + 7 + codebook + exp_bytes + n
