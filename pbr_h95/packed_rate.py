"""Packed-K rate accounting for H95Q (honest estimates, not physical containers).

Conventions (same as prior H95):
  sign_bpw   = 1.0
  exp_bpw    = 2.62   # PBR-E-class exponent reference on Qwen
  mant_bpw   = word-weighted average of packed keep_bits K
  total_bpw  = sign + exp + mant  (+ optional tiny metadata estimate)

Separately report entropy lower-bound BPW when diagnostics supply H(K).
Do NOT claim physical container size unless real bytes are written.
"""
from __future__ import annotations

from typing import Mapping

SIGN_BPW = 1.0
EXP_BPW_REF = 2.62
# Tiny per-tensor keep-map metadata estimate (bits per unique floating tensor).
# Charged only when include_metadata=True; default off to match Phase A ~9.29.
METADATA_BITS_PER_TENSOR = 8.0  # one byte mode/keep id — illustrative only


def _is_floating(t) -> bool:
    fn = getattr(t, "is_floating_point", None)
    if callable(fn):
        return bool(fn())
    dt = getattr(t, "dtype", None)
    if dt is None:
        return False
    name = str(dt)
    return any(x in name for x in ("float", "bfloat", "half"))


def packed_mantissa_bits(keep_map: Mapping[str, int], state_dict) -> dict:
    """sum(K_i * n_i) over unique storages; return aggregates."""
    seen: set[int] = set()
    tot_words = 0
    tot_mant_bits = 0
    keep_hist_words: dict[str, int] = {str(k): 0 for k in range(8)}
    n_tensors = 0
    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        if not hasattr(t, "data_ptr") or not _is_floating(t):
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        n = int(t.numel())
        k = int(keep)
        tot_words += n
        tot_mant_bits += k * n
        keep_hist_words[str(k)] = keep_hist_words.get(str(k), 0) + n
        n_tensors += 1
    avg_k = (tot_mant_bits / tot_words) if tot_words else 0.0
    return {
        "n_words": tot_words,
        "n_unique_tensors": n_tensors,
        "packed_mantissa_bits": tot_mant_bits,
        "avg_packed_K": avg_k,
        "keep_hist_words": keep_hist_words,
    }


def estimate_total_bpw(
    avg_packed_K: float,
    *,
    n_words: int = 0,
    n_tensors: int = 0,
    include_metadata: bool = False,
    entropy_mant_bpw: float | None = None,
) -> dict:
    """Build honest rate report separating packed vs entropy lower bound."""
    packed_mant = float(avg_packed_K)
    packed_total = SIGN_BPW + EXP_BPW_REF + packed_mant
    meta_bpw = 0.0
    if include_metadata and n_words > 0 and n_tensors > 0:
        meta_bpw = (METADATA_BITS_PER_TENSOR * n_tensors) / n_words
        packed_total += meta_bpw

    out = {
        "sign_bpw": SIGN_BPW,
        "exp_bpw_ref": EXP_BPW_REF,
        "mantissa_bpw_packed_K": round(packed_mant, 6),
        "metadata_bpw_estimate": round(meta_bpw, 6),
        "packed_K_total_bpw": round(packed_total, 4),
        "note": (
            "packed_K_total_bpw = 1 + 2.62 + avg packed K (+ optional tiny metadata). "
            "Not a physical container encode."
        ),
    }
    if entropy_mant_bpw is not None:
        ent_total = SIGN_BPW + EXP_BPW_REF + float(entropy_mant_bpw) + meta_bpw
        out["mantissa_bpw_entropy_lb"] = round(float(entropy_mant_bpw), 6)
        out["entropy_lb_total_bpw"] = round(ent_total, 4)
        out["entropy_vs_packed_mant_gain_bpw"] = round(packed_mant - float(entropy_mant_bpw), 6)
    return out


def rate_report_for_keep_map(
    keep_map: Mapping[str, int],
    state_dict,
    *,
    include_metadata: bool = False,
    entropy_mant_bpw: float | None = None,
) -> dict:
    agg = packed_mantissa_bits(keep_map, state_dict)
    rates = estimate_total_bpw(
        agg["avg_packed_K"],
        n_words=agg["n_words"],
        n_tensors=agg["n_unique_tensors"],
        include_metadata=include_metadata,
        entropy_mant_bpw=entropy_mant_bpw,
    )
    return {**agg, **rates}
