"""Optional general-purpose baselines. Labeled as baselines, not PBR."""

from __future__ import annotations

import zlib

import numpy as np

from pbr_core.hashing import words_to_bytes
from pbr_core.metrics import bits_per_weight, compression_ratio


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
    return out
