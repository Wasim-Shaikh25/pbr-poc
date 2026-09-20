"""Deterministic BF16 mantissa quantization (H95-A).

Retains the top *keep_bits* of the 7-bit mantissa with round-to-nearest.
Specials (exp==255 Inf/NaN) and signed zeros are preserved exactly.
Rounding overflow increments the exponent (saturating to Inf at exp 254->255).

This produces a *quantized reference* checkpoint. Storage codecs must restore
that reference exactly. Model-quality retention (>=95%) is a separate gate and
is NOT claimed by this module alone.
"""
from __future__ import annotations

import numpy as np

from pbr_core.bf16 import EXP_MASK, MANT_MASK, join_components, split_components


def quantize_bf16_mantissas(words: np.ndarray, keep_bits: int) -> np.ndarray:
    """Return quantized BF16 uint16 words with *keep_bits* in {0..7}."""
    if keep_bits not in range(0, 8):
        raise ValueError(f"keep_bits must be 0..7, got {keep_bits}")
    arr = np.ascontiguousarray(words, dtype=np.uint16)
    shape = arr.shape
    w = arr.ravel()
    if keep_bits == 7:
        return arr.copy()

    sign, exp, mant = split_components(w)
    sign = sign.astype(np.uint16)
    exp = exp.astype(np.uint16)
    mant = mant.astype(np.uint16)

    special = exp == np.uint16(255)
    # Exact keep for Inf/NaN.
    out = w.copy()

    finite = ~special
    if not np.any(finite):
        return out.reshape(shape)

    removed = 7 - keep_bits
    if keep_bits == 0:
        # Round entire mantissa away; only sign+exp remain (mant=0), with exp bump if mant>=64.
        add = np.uint16(1 << 6)
        rounded = mant + add
        overflow = (rounded >= np.uint16(128)) & finite
        new_exp = exp.copy()
        new_mant = np.zeros_like(mant)
        # bump exp on overflow; 254->255 becomes Inf (mant 0)
        bump = overflow & (exp < np.uint16(255))
        new_exp = np.where(bump, exp + np.uint16(1), exp)
        # if we hit 255 via bump from 254, keep mant 0 (Inf). NaN path already excluded.
        new_exp = np.where(special, exp, new_exp)
        recon = join_components(sign, new_exp, new_mant)
        out = np.where(finite, recon, out).astype(np.uint16)
        return out.reshape(shape)

    add = np.uint16(1 << (removed - 1))
    rounded = mant + add
    quantized = rounded >> np.uint16(removed)  # width keep_bits (+1 if overflow)
    overflow = (quantized >= np.uint16(1 << keep_bits)) & finite
    quantized = np.where(overflow, np.uint16(0), quantized)
    new_exp = exp.copy()
    bump = overflow & (exp < np.uint16(255))
    new_exp = np.where(bump, exp + np.uint16(1), new_exp)
    # If bump made Inf (255), force mant 0.
    quantized = np.where(new_exp == np.uint16(255), np.uint16(0), quantized)
    new_mant = (quantized << np.uint16(removed)) & np.uint16(MANT_MASK)
    recon = join_components(sign, new_exp, new_mant)
    out = np.where(finite, recon, out).astype(np.uint16)
    # Preserve exact signed zeros (mant already 0, exp 0).
    return out.reshape(shape)


def reconstruct_from_fields(
    sign: np.ndarray, exp: np.ndarray, mant_kept: np.ndarray, keep_bits: int
) -> np.ndarray:
    """Rebuild BF16 words from stored fields (kept mantissa left-aligned)."""
    if keep_bits not in range(0, 8):
        raise ValueError(keep_bits)
    removed = 7 - keep_bits
    mant = (np.asarray(mant_kept, dtype=np.uint16) << np.uint16(removed)) & np.uint16(MANT_MASK)
    return join_components(sign, exp, mant)


def pack_kept_mantissas(mant_full: np.ndarray, keep_bits: int) -> bytes:
    """Pack the top keep_bits of each full 7-bit mantissa into a bitstream (MSB-first).

    Delegates to ``pbr_h95.bitpack.pack_kept_mantissas`` (vectorized).
    """
    from pbr_h95.bitpack import pack_kept_mantissas as _pack
    return _pack(mant_full, keep_bits)


def mantissa_storage_bpw(n_words: int, keep_bits: int) -> float:
    """RAW packed storage BPW for kept mantissas (no entropy coding)."""
    if n_words == 0:
        return 0.0
    return keep_bits * 1.0  # exact for bit-pack without padding amortization if we ignore pad
