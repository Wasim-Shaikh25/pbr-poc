"""Entropy diagnostics on retained K-bit mantissa symbols (H95Q).

For each tensor / family:
  - fixed_rate K
  - H(retained symbols) empirical entropy
  - bit-plane bias (fraction of 1s per plane)
  - optional simple rANS table-cost estimate vs packed K

Reject entropy coding when gain after table cost is non-positive.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from pbr_core.bf16 import split_components
from pbr_h95.policy import tensor_family
from pbr_h95.quantize import quantize_bf16_mantissas


def retained_symbols(words: np.ndarray, keep_bits: int) -> np.ndarray:
    """Top *keep_bits* of each mantissa after quantization (as integer symbols)."""
    if keep_bits <= 0:
        return np.zeros(int(np.asarray(words).size), dtype=np.uint16)
    quant = quantize_bf16_mantissas(words, keep_bits)
    _, _, mant = split_components(quant)
    removed = 7 - keep_bits
    return (mant.astype(np.uint16) >> np.uint16(removed)).astype(np.uint16)


def empirical_entropy_bits(symbols: np.ndarray, *, alphabet: int | None = None) -> float:
    """Shannon entropy H in bits/symbol from empirical frequencies."""
    s = np.asarray(symbols).ravel()
    if s.size == 0:
        return 0.0
    if alphabet is None:
        # count unique via bincount if small range
        vmax = int(s.max()) if s.size else 0
        if vmax <= 4096:
            counts = np.bincount(s.astype(np.int64), minlength=vmax + 1)
            counts = counts[counts > 0]
        else:
            _, counts = np.unique(s, return_counts=True)
    else:
        counts = np.bincount(s.astype(np.int64), minlength=int(alphabet))
        counts = counts[counts > 0]
    probs = counts.astype(np.float64) / float(s.size)
    return float(-(probs * np.log2(probs)).sum())


def bit_plane_ones_frac(symbols: np.ndarray, keep_bits: int) -> list[float]:
    """Fraction of 1-bits in each plane (MSB=plane keep_bits-1 … LSB=plane 0)."""
    s = np.asarray(symbols, dtype=np.uint32).ravel()
    out = []
    n = max(int(s.size), 1)
    for plane in range(keep_bits - 1, -1, -1):
        ones = int(np.count_nonzero((s >> plane) & 1))
        out.append(ones / n)
    return out


def simple_rans_table_bits(counts: np.ndarray, *, freq_bits: int = 12) -> int:
    """Crude shared-table cost: alphabet entries × freq_bits (plus small header).

    This is intentionally conservative / simple — not a real rANS encode.
    """
    alphabet = int(np.count_nonzero(counts))
    header = 32  # mode + alphabet size + pad
    return header + alphabet * int(freq_bits)


def rans_estimate_bpw(symbols: np.ndarray, keep_bits: int, *, freq_bits: int = 12) -> dict:
    """Entropy rate + table amortized BPW; compare to packed K."""
    s = np.asarray(symbols).ravel()
    n = int(s.size)
    if n == 0 or keep_bits <= 0:
        return {
            "packed_K": float(keep_bits),
            "H_bits": 0.0,
            "rans_payload_bpw": 0.0,
            "table_bits": 0,
            "rans_complete_bpw": 0.0,
            "gain_vs_packed": 0.0,
            "accept_rans": False,
            "reason": "empty_or_K0",
        }
    alphabet = 1 << keep_bits
    counts = np.bincount(s.astype(np.int64), minlength=alphabet)
    H = empirical_entropy_bits(s, alphabet=alphabet)
    table_bits = simple_rans_table_bits(counts, freq_bits=freq_bits)
    # Ideal coded length ≈ n*H + table; ceil not needed for BPW estimate
    complete = (n * H + table_bits) / n
    packed = float(keep_bits)
    gain = packed - complete
    accept = gain > 0.01  # require tiny positive gain after table
    return {
        "packed_K": packed,
        "H_bits": round(H, 6),
        "rans_payload_bpw": round(H, 6),
        "table_bits": table_bits,
        "rans_complete_bpw": round(complete, 6),
        "gain_vs_packed": round(gain, 6),
        "accept_rans": bool(accept),
        "reason": "gain_after_table" if accept else "no_gain_after_table_cost",
    }


def diagnose_tensor_words(words: np.ndarray, keep_bits: int) -> dict:
    """Full diagnostic for one tensor's BF16 uint16 words at *keep_bits*."""
    k = int(keep_bits)
    sym = retained_symbols(words, k)
    H = empirical_entropy_bits(sym, alphabet=(1 << k) if k > 0 else 1)
    planes = bit_plane_ones_frac(sym, k) if k > 0 else []
    rans = rans_estimate_bpw(sym, k)
    return {
        "keep_bits": k,
        "n_words": int(sym.size),
        "fixed_rate_K": float(k),
        "H_retained_symbols": round(H, 6),
        "bit_plane_ones_frac_msb_first": [round(x, 6) for x in planes],
        "rans_estimate": rans,
    }


def diagnose_keep_map(
    keep_map: Mapping[str, int],
    state_dict,
    *,
    bf16_to_u16,
    num_layers: int = 24,
    max_tensors: int = 0,
    sample_words_cap: int = 0,
) -> dict:
    """Per-tensor + per-family entropy diagnostics (unique storage).

    *bf16_to_u16(tensor) -> np.uint16 words*
    If *sample_words_cap* > 0, subsample each tensor for speed (still reports n_words full).
    """
    seen: set[int] = set()
    per_tensor: list[dict] = []
    # Accumulator for family-weighted entropy: sum n_i * H_i / sum n_i
    fam_acc: dict[str, dict] = {}
    global_n = 0
    global_H_n = 0.0
    global_packed_n = 0.0

    items = list(keep_map.items())
    for name, keep in items:
        if name not in state_dict:
            continue
        t = state_dict[name]
        if not hasattr(t, "is_floating_point") or not t.is_floating_point():
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if max_tensors and len(per_tensor) >= max_tensors:
            break
        words = bf16_to_u16(t)
        n_full = int(words.size)
        if sample_words_cap and n_full > sample_words_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_full, size=sample_words_cap, replace=False)
            sample = words.ravel()[idx]
            diag = diagnose_tensor_words(sample, int(keep))
            diag["n_words"] = n_full
            diag["sampled_words"] = sample_words_cap
        else:
            diag = diagnose_tensor_words(words, int(keep))
            diag["sampled_words"] = diag["n_words"]
        diag["name"] = name
        fam = tensor_family(name, num_layers=num_layers) or "other"
        diag["family"] = fam
        per_tensor.append(diag)

        n = diag["n_words"]
        H = diag["H_retained_symbols"]
        k = float(diag["keep_bits"])
        global_n += n
        global_H_n += H * n
        global_packed_n += k * n
        slot = fam_acc.setdefault(fam, {"n_words": 0, "H_n": 0.0, "K_n": 0.0, "rans_accept": 0, "n_tensors": 0})
        slot["n_words"] += n
        slot["H_n"] += H * n
        slot["K_n"] += k * n
        slot["n_tensors"] += 1
        if diag["rans_estimate"]["accept_rans"]:
            slot["rans_accept"] += 1

    families = {}
    for fam, slot in fam_acc.items():
        nw = slot["n_words"]
        avg_H = slot["H_n"] / nw if nw else 0.0
        avg_K = slot["K_n"] / nw if nw else 0.0
        families[fam] = {
            "n_words": nw,
            "n_tensors": slot["n_tensors"],
            "avg_fixed_rate_K": round(avg_K, 6),
            "avg_H_retained": round(avg_H, 6),
            "entropy_vs_packed_gain": round(avg_K - avg_H, 6),
            "n_tensors_rans_accept": slot["rans_accept"],
        }

    avg_H = global_H_n / global_n if global_n else 0.0
    avg_K = global_packed_n / global_n if global_n else 0.0
    return {
        "n_words": global_n,
        "avg_fixed_rate_K": round(avg_K, 6),
        "avg_H_retained": round(avg_H, 6),
        "entropy_vs_packed_gain": round(avg_K - avg_H, 6),
        "families": families,
        "tensors": per_tensor,
        "guidance": (
            "Use packed K unless rans_estimate.accept_rans is True after table cost. "
            "Entropy lower-bound BPW uses avg_H_retained in place of avg packed K."
        ),
    }
