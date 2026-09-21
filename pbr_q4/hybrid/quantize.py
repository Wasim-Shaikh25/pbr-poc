"""Dispatch H95 mantissa-keep vs groupwise INT vs raw BF16."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pbr_h95.bitpack import packed_bytes_for_k
from pbr_h95.quantize import quantize_bf16_mantissas
from pbr_q4.hybrid.bits import as_rank2
from pbr_q4.hybrid.const import H95_EXP_REF_BPW
from pbr_q4.hybrid.int_quant import dequantize_int, quantize_matrix
from pbr_q4.hybrid.policy import HybridPolicy, Slot


@dataclass
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    rows: int
    cols: int
    kind: str  # "h95" | "groupwise" | "bf16_raw"
    family: str
    keep: int
    bits: int | None
    group_size: int
    n_groups: int
    q_ref: np.ndarray
    scales: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float16))
    zp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    codes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint16))
    outlier_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    outlier_words: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint16))
    mse: float = 0.0
    spatial_score: float = 0.0
    n_spatial_overrides: int = 0
    n_clipped: int = 0

    @property
    def n_weights(self) -> int:
        return int(self.q_ref.size)

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "kind": self.kind,
            "family": self.family,
            "keep": self.keep,
            "bits": self.bits,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "n_weights": self.n_weights,
            "n_outliers": int(self.outlier_idx.size),
            "mse": self.mse,
            "spatial_score": self.spatial_score,
            "n_spatial_overrides": self.n_spatial_overrides,
        }


def quantize_h95(
    words: np.ndarray,
    keep: int,
    *,
    name: str = "",
    family: str = "",
) -> QuantizedTensor:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    q_ref = quantize_bf16_mantissas(w, int(keep))
    arr2, orig, rows, cols = as_rank2(q_ref)
    return QuantizedTensor(
        name=name,
        shape=orig,
        rows=rows,
        cols=cols,
        kind="h95",
        family=family,
        keep=int(keep),
        bits=None,
        group_size=0,
        n_groups=0,
        q_ref=q_ref.astype(np.uint16),
        mse=0.0,
    )


def quantize_bf16_raw(
    words: np.ndarray,
    *,
    name: str = "",
    family: str = "norm_bias",
) -> QuantizedTensor:
    w = np.ascontiguousarray(words, dtype=np.uint16)
    orig = tuple(int(x) for x in w.shape)
    rows, cols = (1, int(w.size)) if w.ndim <= 1 else (int(w.shape[0]), int(np.prod(w.shape[1:])))
    return QuantizedTensor(
        name=name,
        shape=orig,
        rows=rows,
        cols=cols,
        kind="bf16_raw",
        family=family,
        keep=7,
        bits=None,
        group_size=0,
        n_groups=0,
        q_ref=w.copy(),
        mse=0.0,
    )


def quantize_tensor(
    words: np.ndarray,
    slot: Slot,
    policy: HybridPolicy,
    *,
    name: str = "",
    family: str = "",
) -> QuantizedTensor:
    if slot.kind == "bf16":
        return quantize_bf16_raw(words, name=name, family=family or "norm_bias")
    if slot.kind == "h95":
        return quantize_h95(words, int(slot.keep), name=name, family=family)
    if slot.kind != "int":
        raise ValueError(f"unknown slot kind {slot.kind!r}")
    outlier_frac = float(policy.outlier_frac) if int(slot.bits) <= int(policy.outlier_max_bits) else 0.0
    rec = quantize_matrix(
        words,
        int(slot.bits),
        group_size=int(policy.group_size),
        mse_tie_ratio=float(policy.mse_tie_ratio),
        outlier_frac=outlier_frac,
        name=name,
        family=family,
    )
    return QuantizedTensor(
        name=rec["name"],
        shape=rec["shape"],
        rows=rec["rows"],
        cols=rec["cols"],
        kind="groupwise",
        family=rec["family"],
        keep=0,
        bits=rec["bits"],
        group_size=rec["group_size"],
        n_groups=rec["n_groups"],
        q_ref=rec["q_ref"],
        scales=rec["scales"],
        zp=rec["zp"],
        codes=rec["codes"],
        outlier_idx=rec["outlier_idx"],
        outlier_words=rec["outlier_words"],
        mse=rec["mse"],
        spatial_score=rec["spatial_score"],
        n_spatial_overrides=rec["n_spatial_overrides"],
        n_clipped=rec["n_clipped"],
    )


def estimate_packed_bits(qt: QuantizedTensor) -> int:
    """Complete packed-field bit count (no X/Y, no container header)."""
    n = int(qt.q_ref.size)
    if qt.kind == "bf16_raw":
        return n * 16
    if qt.kind == "h95":
        sign_bits = n
        exp_bits = int(round(H95_EXP_REF_BPW * n))
        mant_bits = packed_bytes_for_k(n, int(qt.keep)) * 8
        return sign_bits + exp_bits + mant_bits
    if qt.kind == "groupwise" and qt.bits is not None:
        code_bits = n * int(qt.bits)
        scale_bits = int(qt.n_groups) * 16
        zp_bits = int(qt.n_groups) * 8
        out_bits = int(qt.outlier_idx.size) * (32 + 16)
        return code_bits + scale_bits + zp_bits + out_bits
    raise ValueError(f"cannot estimate {qt.kind}")
