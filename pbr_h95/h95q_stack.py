"""H95Q follow-up stack: B1+embed tiers, proper Candidate C sparse recovery, selective MLP K3.

Honesty
-------
- packed BPW = 1 + 2.62 + avg packed K (+ exception map / correction bits for C).
- Not a physical container encode; map_bpw is charged but no container bytes written.
- Proxy PPL only; heldout retention ≥ 0.95 = proxy GO for that map.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal

import numpy as np
import torch

from pbr_h95.apply_policy import (
    apply_policy_to_model,
    bf16_tensor_to_u16,
    u16_to_bf16_tensor,
)
from pbr_h95.embed_tiers import (
    apply_embed_row_keeps,
    assign_frequency_tiers,
    avg_keep_bits_with_embed_rows,
    count_token_frequencies,
    find_embed_param,
    padded_row_indices,
    reachable_token_ids,
)
from pbr_h95.packed_rate import EXP_BPW_REF, SIGN_BPW, rate_report_for_keep_map
from pbr_h95.policy import (
    BAND_SPECS,
    layer_index,
    tensor_family,
    unit_keep_fn,
)
from pbr_h95.quantize import quantize_bf16_mantissas

RecoverMode = Literal["row_magnitude", "channel_magnitude"]
MapScheme = Literal["auto", "bitmap", "absolute_indices"]


# ---------------------------------------------------------------------------
# Body keep functions
# ---------------------------------------------------------------------------


def b1_body_keep_fn(
    *,
    mlp_keep: int = 4,
    attn_keep: int = 5,
    protected: int = 7,
    num_layers: int = 24,
):
    """B1-style body: mlp_mid / attn_mid / protected (embed left uniform protected)."""

    def _fn(name: str) -> int:
        n = name.lower()
        if "bias" in n:
            return protected
        fam = tensor_family(name, num_layers=num_layers)
        if fam == "mlp_mid":
            return int(mlp_keep)
        if fam == "attn_mid":
            return int(attn_keep)
        return protected

    return _fn


def aggressive_body_keep_fn(
    *,
    mlp_keep: int = 4,
    attn_keep: int = 4,
    embed_keep: int = 7,
    protected: int = 7,
    num_layers: int = 24,
):
    """Aggressive C base: mid attn/mlp low; embed/norm/bias/first/last protected or embed_keep."""

    def _fn(name: str) -> int:
        n = name.lower()
        if "bias" in n:
            return protected
        fam = tensor_family(name, num_layers=num_layers)
        if fam == "embed":
            return int(embed_keep)
        if fam == "mlp_mid":
            return int(mlp_keep)
        if fam == "attn_mid":
            return int(attn_keep)
        return protected

    return _fn


def mlp_band_k3_keep_fn(
    k3_bands: Sequence[str],
    *,
    base_mlp_keep: int = 4,
    attn_keep: int = 5,
    protected: int = 7,
    num_layers: int = 24,
):
    """B1 body with selected mlp bands dropped to K3.

    *k3_bands* members are unit names like ``mlp_band_8_15`` or ``mlp_mid`` (all mid).
    """
    k3_set = set(k3_bands)
    # Build unit keeps: default mlp bands at base_mlp_keep, attn at attn_keep
    unit_keeps: dict[str, int] = {}
    for uname, kind, _layers in BAND_SPECS:
        if kind == "mlp":
            unit_keeps[uname] = 3 if (uname in k3_set or "mlp_mid" in k3_set) else int(base_mlp_keep)
        else:
            unit_keeps[uname] = int(attn_keep)
    if "mlp_mid" in k3_set:
        for uname, kind, _ in BAND_SPECS:
            if kind == "mlp":
                unit_keeps[uname] = 3

    body = unit_keep_fn(unit_keeps, BAND_SPECS, protected=protected, num_layers=num_layers)

    def _fn(name: str) -> int:
        # unit_keep_fn already protects emb/norm/bias/first/last
        return body(name)

    return _fn


# ---------------------------------------------------------------------------
# Exception map costing
# ---------------------------------------------------------------------------


def exception_map_cost(
    n_units: int,
    n_recover: int,
    *,
    scheme: MapScheme = "auto",
) -> dict[str, Any]:
    """Honest map-bit estimate: bitmap (1 bit/unit) vs absolute indices.

    Delta-coded indices are same order as absolute for worst-case; we report
    absolute_indices = n_recover * ceil(log2(n_units)) as the index alternative.
    ``auto`` picks the cheaper of bitmap vs absolute indices.
    """
    n_units = max(int(n_units), 0)
    n_recover = max(int(n_recover), 0)
    bitmap_bits = n_units
    if n_units <= 1:
        index_bits = n_recover * 1
    else:
        index_bits = n_recover * int(math.ceil(math.log2(n_units)))
    if scheme == "bitmap":
        chosen, bits = "bitmap", bitmap_bits
    elif scheme == "absolute_indices":
        chosen, bits = "absolute_indices", index_bits
    else:
        if bitmap_bits <= index_bits:
            chosen, bits = "bitmap", bitmap_bits
        else:
            chosen, bits = "absolute_indices", index_bits
    return {
        "scheme_requested": scheme,
        "scheme_chosen": chosen,
        "map_bits": int(bits),
        "bitmap_bits": int(bitmap_bits),
        "absolute_index_bits": int(index_bits),
        "n_units": n_units,
        "n_recover": n_recover,
        "format_note": (
            "bitmap: 1 bit per row/channel unit. "
            "absolute_indices: n_recover * ceil(log2(n_units)) bits. "
            "auto picks min. Delta-coded indices same order as absolute (not cheaper here)."
        ),
    }


def packed_bpw_with_map(avg_mant: float, map_bits: int, n_words: int) -> dict[str, float]:
    map_bpw = (map_bits / n_words) if n_words else 0.0
    total = SIGN_BPW + EXP_BPW_REF + float(avg_mant) + map_bpw
    return {
        "avg_mant_bpw": float(avg_mant),
        "exception_map_bpw": float(map_bpw),
        "packed_K_total_bpw": round(total, 4),
        "sign_bpw": SIGN_BPW,
        "exp_bpw_ref": EXP_BPW_REF,
    }


# ---------------------------------------------------------------------------
# Row / channel quantization helpers
# ---------------------------------------------------------------------------


def _quantize_rows_mixed(
    baseline: torch.Tensor,
    live: torch.Tensor,
    row_keeps: np.ndarray,
) -> dict[str, int]:
    """Quantize 2D tensor rows with per-row keep; write into *live*."""
    row_keeps = np.asarray(row_keeps, dtype=np.int8)
    n_rows, cols = int(baseline.shape[0]), int(baseline.shape[1])
    words = bf16_tensor_to_u16(baseline).reshape(n_rows, cols)
    out = words.copy()
    hist: dict[str, int] = {}
    for k in sorted(set(int(x) for x in row_keeps.tolist())):
        idx = np.flatnonzero(row_keeps == k)
        hist[str(k)] = int(idx.size)
        if idx.size == 0:
            continue
        if k >= 7:
            continue
        block = words[idx].reshape(-1)
        q = quantize_bf16_mantissas(block, k)
        out[idx] = q.reshape(idx.size, cols)
    new_t = u16_to_bf16_tensor(out.reshape(-1), live).reshape(live.shape)
    with torch.no_grad():
        live.copy_(new_t.to(device=live.device, dtype=live.dtype))
    return hist


def _quantize_channels_mixed(
    baseline: torch.Tensor,
    live: torch.Tensor,
    channel_keeps: np.ndarray,
) -> dict[str, int]:
    """Quantize 2D tensor columns with per-channel keep; write into *live*."""
    channel_keeps = np.asarray(channel_keeps, dtype=np.int8)
    n_rows, cols = int(baseline.shape[0]), int(baseline.shape[1])
    words = bf16_tensor_to_u16(baseline).reshape(n_rows, cols)
    out = words.copy()
    hist: dict[str, int] = {}
    for k in sorted(set(int(x) for x in channel_keeps.tolist())):
        idx = np.flatnonzero(channel_keeps == k)
        hist[str(k)] = int(idx.size)
        if idx.size == 0:
            continue
        if k >= 7:
            continue
        block = words[:, idx].reshape(-1)
        q = quantize_bf16_mantissas(block, k)
        out[:, idx] = q.reshape(n_rows, idx.size)
    new_t = u16_to_bf16_tensor(out.reshape(-1), live).reshape(live.shape)
    with torch.no_grad():
        live.copy_(new_t.to(device=live.device, dtype=live.dtype))
    return hist


def _iter_unique_floating(sd: Mapping[str, torch.Tensor]):
    seen: set[int] = set()
    for name, tensor in sd.items():
        if not hasattr(tensor, "is_floating_point") or not tensor.is_floating_point():
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        yield name, tensor


# ---------------------------------------------------------------------------
# Track 1: B1 + embed frequency tiers
# ---------------------------------------------------------------------------


def fit_embed_tiers_on_calib(
    tokenizer,
    calib_texts: Sequence[str],
    *,
    vocab_rows: int,
    schedule: str,
    max_length: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    reachable = reachable_token_ids(tokenizer)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    freq = count_token_frequencies(tokenizer, calib_texts, max_length=max_length)
    return assign_frequency_tiers(
        vocab_rows=vocab_rows,
        reachable=reachable,
        special_ids=special,
        freq=freq,
        schedule=schedule,
    )


def apply_b1_plus_embed_tiers(
    model: torch.nn.Module,
    *,
    baseline_sd: Mapping[str, torch.Tensor],
    row_keeps: np.ndarray,
    embed_name: str,
    mlp_keep: int = 4,
    attn_keep: int = 5,
) -> dict[str, Any]:
    """Restore baseline → B1 body (embed @7) → per-row embed keeps."""
    keep_fn = b1_body_keep_fn(mlp_keep=mlp_keep, attn_keep=attn_keep, protected=7)
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    name, weight = find_embed_param(model)
    base_w = baseline_sd[embed_name]
    row_meta = apply_embed_row_keeps(weight, row_keeps, baseline_weight=base_w)
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    meta["embed_row_meta"] = row_meta
    meta["embed_name"] = name
    # Mark embed keep as mixed (-1 sentinel) for bookkeeping; rate uses row_keeps
    meta["keep_map"][embed_name] = -1
    return meta


def rate_b1_embed_tiers(
    keep_map: Mapping[str, int],
    baseline_sd: Mapping[str, torch.Tensor],
    *,
    embed_name: str,
    row_keeps: np.ndarray,
    omit_padded: bool = False,
    padded_indices: set[int] | None = None,
) -> dict[str, Any]:
    # Replace sentinel with 7 for unique-storage body pass; embed handled via rows
    km = {k: (7 if v < 0 else v) for k, v in keep_map.items()}
    avg_k, detail = avg_keep_bits_with_embed_rows(
        km,
        baseline_sd,
        embed_name=embed_name,
        row_keeps=row_keeps,
        omit_padded_from_storage=omit_padded,
        padded_indices=padded_indices,
    )
    rates = packed_bpw_with_map(avg_k, 0, detail["total_words_in_estimate"])
    return {
        **rates,
        "avg_packed_K": avg_k,
        "n_words": detail["total_words_in_estimate"],
        "embed_detail": detail,
        "omit_padded_from_storage": omit_padded,
    }


# ---------------------------------------------------------------------------
# Track 2: proper Candidate C sparse recovery
# ---------------------------------------------------------------------------


def apply_sparse_recovery_c(
    model: torch.nn.Module,
    *,
    baseline_sd: Mapping[str, torch.Tensor],
    recover_frac: float = 0.05,
    base_mlp_keep: int = 4,
    base_attn_keep: int = 4,
    embed_keep: int = 7,
    recover_keep: int = 7,
    mode: RecoverMode = "row_magnitude",
    target_families: Sequence[str] = ("mlp_mid",),
    map_scheme: MapScheme = "auto",
    protected: int = 7,
    num_layers: int = 24,
    embed_row_keeps: np.ndarray | None = None,
    embed_name: str | None = None,
) -> dict[str, Any]:
    """Low-precision body + restore top-|w| rows or channels to *recover_keep*.

    Exception map charged as bitmap or absolute indices (auto = cheaper).
    Correction bits = (recover_keep - base_keep) * recovered_words, folded into
    adj avg mantissa; map bits amortized over all model words.
    """
    keep_fn = aggressive_body_keep_fn(
        mlp_keep=base_mlp_keep,
        attn_keep=base_attn_keep,
        embed_keep=embed_keep if embed_row_keeps is None else protected,
        protected=protected,
        num_layers=num_layers,
    )
    meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
    keep_map = dict(meta["keep_map"])

    if embed_row_keeps is not None and embed_name is not None:
        name, weight = find_embed_param(model)
        apply_embed_row_keeps(weight, embed_row_keeps, baseline_weight=baseline_sd[embed_name])
        keep_map[embed_name] = -1
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            if hasattr(model, "tie_weights"):
                model.tie_weights()

    sd = model.state_dict()
    extra_mant_bits = 0
    map_bits_total = 0
    recovered_units = 0
    recovered_words = 0
    units_total = 0
    details: list[dict[str, Any]] = []
    map_scheme_hist: dict[str, int] = {}

    target_set = set(target_families)
    for name, live in list(sd.items()):
        if not live.is_floating_point() or live.ndim != 2:
            continue
        if "bias" in name.lower():
            continue
        fam = tensor_family(name, num_layers=num_layers)
        if fam not in target_set:
            continue
        # Unique storage: skip aliases
        # (state_dict may share; process via live ptr once — use baseline ptr)
        base = baseline_sd[name]
        # Skip if we already processed this storage via another name
        # Check by seeing if keep_map already marked recovered for this name only once
        if any(d["name"] == name for d in details):
            continue
        # Also skip tied duplicates: compare data_ptr of live tensors already done
        if any(sd[d["name"]].data_ptr() == live.data_ptr() for d in details if d["name"] in sd):
            continue

        w = base.detach().float().cpu()
        if mode == "row_magnitude":
            scores = w.abs().mean(dim=1)  # [out]
            n_units = int(scores.numel())
            n_recover = max(1, int(round(recover_frac * n_units))) if recover_frac > 0 else 0
            top = set(torch.topk(scores, k=max(n_recover, 1)).indices.tolist()) if n_recover else set()
            base_k = int(keep_map.get(name, base_mlp_keep if fam == "mlp_mid" else base_attn_keep))
            row_keeps = np.full(n_units, base_k, dtype=np.int8)
            for r in top:
                row_keeps[r] = recover_keep
            _quantize_rows_mixed(base, live, row_keeps)
            cols = int(w.shape[1])
            recovered_words += n_recover * cols
            extra_mant_bits += (recover_keep - base_k) * n_recover * cols
            cost = exception_map_cost(n_units, n_recover, scheme=map_scheme)
            map_bits_total += cost["map_bits"]
            map_scheme_hist[cost["scheme_chosen"]] = map_scheme_hist.get(cost["scheme_chosen"], 0) + 1
            units_total += n_units
            recovered_units += n_recover
            details.append(
                {
                    "name": name,
                    "family": fam,
                    "mode": mode,
                    "n_units": n_units,
                    "n_recover": n_recover,
                    "unit_width": cols,
                    "base_keep": base_k,
                    "recover_keep": recover_keep,
                    "map": cost,
                }
            )
            keep_map[name] = base_k
        else:  # channel_magnitude
            scores = w.abs().mean(dim=0)  # [in]
            n_units = int(scores.numel())
            n_recover = max(1, int(round(recover_frac * n_units))) if recover_frac > 0 else 0
            top = set(torch.topk(scores, k=max(n_recover, 1)).indices.tolist()) if n_recover else set()
            base_k = int(keep_map.get(name, base_mlp_keep if fam == "mlp_mid" else base_attn_keep))
            ch_keeps = np.full(n_units, base_k, dtype=np.int8)
            for c in top:
                ch_keeps[c] = recover_keep
            _quantize_channels_mixed(base, live, ch_keeps)
            rows = int(w.shape[0])
            recovered_words += n_recover * rows
            extra_mant_bits += (recover_keep - base_k) * n_recover * rows
            cost = exception_map_cost(n_units, n_recover, scheme=map_scheme)
            map_bits_total += cost["map_bits"]
            map_scheme_hist[cost["scheme_chosen"]] = map_scheme_hist.get(cost["scheme_chosen"], 0) + 1
            units_total += n_units
            recovered_units += n_recover
            details.append(
                {
                    "name": name,
                    "family": fam,
                    "mode": mode,
                    "n_units": n_units,
                    "n_recover": n_recover,
                    "unit_width": rows,
                    "base_keep": base_k,
                    "recover_keep": recover_keep,
                    "map": cost,
                }
            )
            keep_map[name] = base_k

    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # Rate: body keep_map undercounts recovered words — adjust
    if embed_row_keeps is not None and embed_name is not None:
        km_rate = {k: (7 if (isinstance(v, int) and v < 0) else v) for k, v in keep_map.items()}
        avg_body, detail = avg_keep_bits_with_embed_rows(
            km_rate,
            baseline_sd,
            embed_name=embed_name,
            row_keeps=embed_row_keeps,
            omit_padded_from_storage=False,
        )
        n_words = detail["total_words_in_estimate"]
        # body avg already uses base_k for recovered tensors; add extra correction bits
        adj_mant = avg_body + (extra_mant_bits / n_words if n_words else 0.0)
    else:
        body_rate = rate_report_for_keep_map(
            {k: v for k, v in keep_map.items() if v >= 0},
            baseline_sd,
        )
        n_words = body_rate["n_words"]
        adj_mant = body_rate["avg_packed_K"] + (extra_mant_bits / n_words if n_words else 0.0)

    rates = packed_bpw_with_map(adj_mant, map_bits_total, n_words)
    meta["keep_map"] = keep_map
    meta["c_recovery"] = {
        "recover_frac_target": recover_frac,
        "mode": mode,
        "base_mlp_keep": base_mlp_keep,
        "base_attn_keep": base_attn_keep,
        "embed_keep": embed_keep,
        "recover_keep": recover_keep,
        "target_families": list(target_families),
        "recovered_units": recovered_units,
        "units_total": units_total,
        "recovered_words": recovered_words,
        "extra_mantissa_bits": extra_mant_bits,
        "exception_map_bits": map_bits_total,
        "map_scheme_hist": map_scheme_hist,
        "map_format": (
            "Per targeted 2D weight: select top-|w| mean rows or channels. "
            "Store exception map (bitmap 1 bit/unit OR absolute indices "
            "n_recover*ceil(log2(n_units)), auto=min) + correction stream of "
            "(recover_keep-base_keep) extra mantissa bits on recovered units. "
            "No physical container bytes in this PoC; map_bpw counted in packed total."
        ),
        "adj_avg_mant_bpw": adj_mant,
        "rates": rates,
        "details_head": details[:8],
        "n_tensors_recovered": len(details),
    }
    meta["bytes_saved_vs_bf16_mantissa"] = (7.0 - adj_mant) * n_words / 8.0
    return meta


# ---------------------------------------------------------------------------
# Track 3: selective MLP K3
# ---------------------------------------------------------------------------


def apply_mlp_k3_policy(
    model: torch.nn.Module,
    *,
    baseline_sd: Mapping[str, torch.Tensor],
    variant: str,
    attn_keep: int = 5,
    base_mlp_keep: int = 4,
    protected: int = 7,
    num_layers: int = 24,
    embed_row_keeps: np.ndarray | None = None,
    embed_name: str | None = None,
) -> dict[str, Any]:
    """Apply selective MLP K3 variants on a B1-like body.

    Variants
    --------
    all_mlp_k3 : all mlp_mid → K3
    band_1_7 / band_8_15 / band_16_22 : that band → K3, other mlp → base_mlp_keep
    robust50_rows : per mlp_mid matrix, lowest-|w| 50% rows → K3, rest → base_mlp_keep
    """
    if variant == "all_mlp_k3":
        keep_fn = mlp_band_k3_keep_fn(
            ["mlp_mid"],
            base_mlp_keep=base_mlp_keep,
            attn_keep=attn_keep,
            protected=protected,
            num_layers=num_layers,
        )
        meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
        meta["k3_variant"] = {"variant": variant}
    elif variant.startswith("band_"):
        band_map = {
            "band_1_7": "mlp_band_1_7",
            "band_8_15": "mlp_band_8_15",
            "band_16_22": "mlp_band_16_22",
        }
        if variant not in band_map:
            raise KeyError(variant)
        keep_fn = mlp_band_k3_keep_fn(
            [band_map[variant]],
            base_mlp_keep=base_mlp_keep,
            attn_keep=attn_keep,
            protected=protected,
            num_layers=num_layers,
        )
        meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
        meta["k3_variant"] = {"variant": variant, "band": band_map[variant]}
    elif variant == "robust50_rows":
        keep_fn = b1_body_keep_fn(
            mlp_keep=base_mlp_keep, attn_keep=attn_keep, protected=protected, num_layers=num_layers
        )
        meta = apply_policy_to_model(model, keep_fn, baseline=baseline_sd)
        # Re-quantize mlp_mid with per-row K3/K4
        sd = model.state_dict()
        details = []
        k3_words = 0
        k4_words = 0
        done_ptrs: set[int] = set()
        for name, live in list(sd.items()):
            if tensor_family(name, num_layers=num_layers) != "mlp_mid":
                continue
            if "bias" in name.lower() or live.ndim != 2:
                continue
            ptr = live.data_ptr()
            if ptr in done_ptrs:
                continue
            done_ptrs.add(ptr)
            base = baseline_sd[name]
            w = base.detach().float().cpu()
            row_mag = w.abs().mean(dim=1)
            n_rows = int(row_mag.numel())
            n_k3 = max(1, n_rows // 2)
            # lowest magnitude → K3 (robust / less salient)
            order = torch.argsort(row_mag)  # ascending
            k3_set = set(order[:n_k3].tolist())
            row_keeps = np.full(n_rows, base_mlp_keep, dtype=np.int8)
            for r in k3_set:
                row_keeps[r] = 3
            _quantize_rows_mixed(base, live, row_keeps)
            cols = int(w.shape[1])
            k3_words += n_k3 * cols
            k4_words += (n_rows - n_k3) * cols
            details.append({"name": name, "n_rows": n_rows, "n_k3": n_k3})
            meta["keep_map"][name] = -2  # mixed row keeps
        meta["k3_variant"] = {
            "variant": variant,
            "k3_words": k3_words,
            "k4_words": k4_words,
            "n_tensors": len(details),
            "details_head": details[:6],
            "note": "Per mlp_mid matrix: lowest-|w| 50% rows @K3, rest @K4; no exception map (keeps are explicit per-row schedule, charged in avg K).",
        }
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            if hasattr(model, "tie_weights"):
                model.tie_weights()
    else:
        raise KeyError(f"Unknown mlp k3 variant {variant!r}")

    if embed_row_keeps is not None and embed_name is not None:
        _name, weight = find_embed_param(model)
        apply_embed_row_keeps(weight, embed_row_keeps, baseline_weight=baseline_sd[embed_name])
        meta["keep_map"][embed_name] = -1
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            if hasattr(model, "tie_weights"):
                model.tie_weights()

    return meta


def rate_mlp_k3(
    meta: dict[str, Any],
    baseline_sd: Mapping[str, torch.Tensor],
    *,
    embed_name: str | None = None,
    embed_row_keeps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Packed rate for K3 variants; robust50 uses explicit word counts."""
    keep_map = meta["keep_map"]
    kv = meta.get("k3_variant") or {}

    if kv.get("variant") == "robust50_rows":
        # Rebuild avg K: unique storages; mlp mixed via k3/k4 word counts
        seen: set[int] = set()
        total_w = 0
        total_k = 0.0
        mlp_ptrs_done = False
        # First account non-mlp from keep_map; mlp from k3_words/k4_words once
        k3_words = int(kv["k3_words"])
        k4_words = int(kv["k4_words"])
        for name, keep in keep_map.items():
            if name not in baseline_sd:
                continue
            t = baseline_sd[name]
            if not t.is_floating_point():
                continue
            ptr = t.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            if embed_row_keeps is not None and embed_name and (
                name == embed_name or ("embed" in name.lower() and name.endswith("weight"))
            ):
                hidden = int(t.shape[1])
                for rk in embed_row_keeps.tolist():
                    total_w += hidden
                    total_k += int(rk) * hidden
                continue
            if keep == -2 or tensor_family(name) == "mlp_mid":
                # counted once via aggregate below
                continue
            n = int(t.numel())
            total_w += n
            total_k += int(keep) * n
        total_w += k3_words + k4_words
        total_k += 3 * k3_words + 4 * k4_words
        avg_k = total_k / total_w if total_w else 0.0
        rates = packed_bpw_with_map(avg_k, 0, total_w)
        return {**rates, "avg_packed_K": avg_k, "n_words": total_w}

    if embed_row_keeps is not None and embed_name is not None:
        km = {k: (7 if (isinstance(v, int) and v < 0) else v) for k, v in keep_map.items()}
        avg_k, detail = avg_keep_bits_with_embed_rows(
            km,
            baseline_sd,
            embed_name=embed_name,
            row_keeps=embed_row_keeps,
            omit_padded_from_storage=False,
        )
        rates = packed_bpw_with_map(avg_k, 0, detail["total_words_in_estimate"])
        return {
            **rates,
            "avg_packed_K": avg_k,
            "n_words": detail["total_words_in_estimate"],
            "embed_detail": detail,
        }

    km = {k: v for k, v in keep_map.items() if isinstance(v, int) and v >= 0}
    body = rate_report_for_keep_map(km, baseline_sd)
    rates = packed_bpw_with_map(body["avg_packed_K"], 0, body["n_words"])
    return {
        **rates,
        "avg_packed_K": body["avg_packed_K"],
        "n_words": body["n_words"],
        "keep_hist_words": body.get("keep_hist_words"),
    }


# ---------------------------------------------------------------------------
# Stacked: C with K3 base + sparse K7 recovery
# ---------------------------------------------------------------------------


def apply_c_k3_base_recovery(
    model: torch.nn.Module,
    *,
    baseline_sd: Mapping[str, torch.Tensor],
    recover_frac: float = 0.05,
    base_mlp_keep: int = 3,
    base_attn_keep: int = 4,
    recover_keep: int = 7,
    mode: RecoverMode = "row_magnitude",
    map_scheme: MapScheme = "auto",
    embed_row_keeps: np.ndarray | None = None,
    embed_name: str | None = None,
) -> dict[str, Any]:
    """Candidate C with K3 mlp base + sparse recovery to K7."""
    return apply_sparse_recovery_c(
        model,
        baseline_sd=baseline_sd,
        recover_frac=recover_frac,
        base_mlp_keep=base_mlp_keep,
        base_attn_keep=base_attn_keep,
        embed_keep=7,
        recover_keep=recover_keep,
        mode=mode,
        target_families=("mlp_mid",),
        map_scheme=map_scheme,
        embed_row_keeps=embed_row_keeps,
        embed_name=embed_name,
    )


# ---------------------------------------------------------------------------
# Descriptions / registry helpers
# ---------------------------------------------------------------------------


STACK_POLICY_DESCRIPTIONS: dict[str, str] = {
    "T1_B1_embed_default": (
        "Track1: B1 body (mlp_mid@K4, attn_mid@K5, norm/bias/first/last@K7) + "
        "H95E default frequency embed row tiers (fit calib-v2 only)."
    ),
    "T1_B1_embed_aggressive": (
        "Track1: B1 body + H95E aggressive embed frequency tiers."
    ),
    "T1_B1_embed_conservative": (
        "Track1: B1 body + H95E conservative embed frequency tiers."
    ),
    "T2_C_row_mag_0.02": (
        "Track2: aggressive body mlp@K4 attn@K4 emb@K7; restore top-2% |w| mlp rows to K7; "
        "charge bitmap/index map + correction bits."
    ),
    "T2_C_row_mag_0.05": (
        "Track2: aggressive body; top-5% |w| mlp rows → K7 + honest map cost."
    ),
    "T2_C_row_mag_0.10": (
        "Track2: aggressive body; top-10% |w| mlp rows → K7 + honest map cost."
    ),
    "T2_C_channel_mag_0.05": (
        "Track2: aggressive body; top-5% |w| mlp channels → K7 + honest map cost."
    ),
    "T3_mlp_all_k3": "Track3: B1 body with all mlp_mid → K3 (attn@K5).",
    "T3_mlp_band_1_7_k3": "Track3: B1 body; mlp layers 1–7 → K3; other mlp_mid@K4.",
    "T3_mlp_band_8_15_k3": "Track3: B1 body; mlp layers 8–15 → K3; other mlp_mid@K4.",
    "T3_mlp_band_16_22_k3": "Track3: B1 body; mlp layers 16–22 → K3; other mlp_mid@K4.",
    "T3_mlp_robust50_k3": (
        "Track3: B1 body; per mlp_mid matrix lowest-|w| 50% rows@K3 rest@K4."
    ),
    "S1_B1_aggr_embed_band815_k3": (
        "Stack: B1 + aggressive embed tiers + mlp band 8–15 @K3."
    ),
    "S2_C_k3_base_row05_k7": (
        "Stack: C with mlp@K3 attn@K4 base + top-5% |w| mlp row recovery to K7 (+ map)."
    ),
    "REF_B1_mlp_k4": "Reference: plain B1 (mlp@K4 attn@K5 emb@K7) for apples-to-apples.",
}


def stack_description(name: str) -> str:
    return STACK_POLICY_DESCRIPTIONS.get(name, name)
