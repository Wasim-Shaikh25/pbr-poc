"""H95E: vocabulary / frequency-aware embedding mantissa keep tiers.

Inventory helpers, calib token-frequency counts, tier assignment, and
per-row mantissa quantization of the embedding weight matrix.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from pbr_h95.apply_policy import bf16_tensor_to_u16, u16_to_bf16_tensor
from pbr_h95.quantize import quantize_bf16_mantissas

# Lightweight Unicode-ish class labels for vocab histogram
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]"
)
_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_NUMBER_RE = re.compile(r"[0-9]")
_WHITESPACE_RE = re.compile(r"^\s+$")
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def find_embed_param(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    """Return (state_dict_name, Parameter) for the input embedding weight."""
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            # Resolve canonical state_dict name
            for name, p in model.named_parameters():
                if p is emb.weight:
                    return name, emb.weight
            return "embed_tokens.weight", emb.weight
    for name, p in model.named_parameters():
        if "embed" in name.lower() and name.endswith("weight") and p.ndim == 2:
            return name, p
    raise RuntimeError("Could not locate embedding weight on model")


def reachable_token_ids(tokenizer) -> set[int]:
    """IDs the tokenizer can produce / decode (0 .. len(tokenizer)-1 typically)."""
    n = len(tokenizer)
    return set(range(n))


def padded_row_indices(vocab_rows: int, reachable: set[int]) -> list[int]:
    return [i for i in range(vocab_rows) if i not in reachable]


def classify_token(token_str: str, *, is_special: bool, is_padded: bool) -> str:
    if is_padded:
        return "RESERVED_OR_UNUSED"
    if is_special:
        return "SPECIAL"
    if not token_str:
        return "OTHER"
    # SentencePiece / GPT2 style: leading space often Ġ or ▁
    s = token_str.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
    if _WHITESPACE_RE.match(s):
        return "WHITESPACE"
    if _CJK_RE.search(s):
        return "CJK"
    if _HANGUL_RE.search(s):
        return "HANGUL"
    if _KANA_RE.search(s):
        return "KANA"
    if _CYRILLIC_RE.search(s):
        return "CYRILLIC"
    if _ARABIC_RE.search(s):
        return "ARABIC"
    if _DEVANAGARI_RE.search(s):
        return "DEVANAGARI"
    if _ASCII_LETTER_RE.search(s) and all(ord(c) < 128 for c in s):
        return "ASCII_LETTER"
    if _NUMBER_RE.search(s) and all(ord(c) < 128 for c in s) and not _ASCII_LETTER_RE.search(s):
        return "NUMBER"
    if all(ord(c) < 128 for c in s):
        if _PUNCT_RE.match(s.strip()) or not _ASCII_LETTER_RE.search(s):
            # Distinguish punctuation-ish vs other ASCII
            if any(c.isalnum() for c in s):
                return "ASCII_OTHER"
            return "PUNCTUATION"
        return "ASCII_OTHER"
    return "OTHER_MULTILINGUAL"


def vocab_class_histogram(tokenizer, vocab_rows: int) -> dict[str, int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    reachable = reachable_token_ids(tokenizer)
    counts: Counter[str] = Counter()
    for i in range(vocab_rows):
        if i not in reachable:
            counts["RESERVED_OR_UNUSED"] += 1
            continue
        try:
            tok = tokenizer.convert_ids_to_tokens(i)
            if tok is None:
                tok = ""
            elif not isinstance(tok, str):
                tok = str(tok)
        except Exception:
            tok = ""
        cls = classify_token(tok, is_special=(i in special), is_padded=False)
        counts[cls] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def inventory_embeddings(
    model: torch.nn.Module,
    tokenizer,
    *,
    check_softmax_mass: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure embed shape, share, tie status, padded rows, class histogram."""
    name, weight = find_embed_param(model)
    w = weight.detach()
    rows, hidden = int(w.shape[0]), int(w.shape[1])
    embed_params = rows * hidden
    total_params = sum(int(p.numel()) for p in model.parameters())
    # Unique storage params (tie_word_embeddings shares embed/lm_head)
    seen: set[int] = set()
    unique_params = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        unique_params += int(p.numel())

    tie = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    lm_head_names = [
        n for n, p in model.named_parameters() if "lm_head" in n.lower() and p.data_ptr() != w.data_ptr()
    ]
    # Also check if lm_head shares storage
    tied_aliases = [
        n for n, p in model.named_parameters() if p.data_ptr() == w.data_ptr() and n != name
    ]

    reachable = reachable_token_ids(tokenizer)
    configured = int(getattr(model.config, "vocab_size", rows))
    padded = padded_row_indices(rows, reachable)
    padded_set = set(padded)

    # Padded row norms
    if padded:
        pad_w = w[padded].float()
        norms = torch.linalg.vector_norm(pad_w, dim=1)
        unused_row_mean_norm = float(norms.mean().item())
        unused_zero_rows = int((norms == 0).sum().item())
        unused_row_max_norm = float(norms.max().item())
    else:
        unused_row_mean_norm = 0.0
        unused_zero_rows = 0
        unused_row_max_norm = 0.0

    bf16_mib = embed_params * 2 / (1024 * 1024)

    out: dict[str, Any] = {
        "embed_param_name": name,
        "embed_shape": [rows, hidden],
        "embed_dtype": str(w.dtype).replace("torch.", ""),
        "embed_params": embed_params,
        "total_params_named": total_params,
        "total_params_unique_storage": unique_params,
        "embed_share_of_unique": embed_params / unique_params if unique_params else None,
        "embed_bf16_mib": bf16_mib,
        "tie_word_embeddings": tie,
        "tied_aliases": tied_aliases,
        "lm_head_separate_params": lm_head_names,
        "configured_vocab_size": configured,
        "tokenizer_len": len(tokenizer),
        "tokenizer_vocab_size_attr": int(getattr(tokenizer, "vocab_size", -1)),
        "reachable_ids": len(reachable),
        "padded_row_count": len(padded),
        "padded_ids_head": padded[:50],
        "padded_ids_tail": padded[-50:] if len(padded) > 50 else padded,
        "padded_rows_are_zero": unused_zero_rows == len(padded) and len(padded) > 0,
        "unused_row_mean_norm": unused_row_mean_norm,
        "unused_row_max_norm": unused_row_max_norm,
        "unused_zero_rows": unused_zero_rows,
        "special_ids": sorted(set(getattr(tokenizer, "all_special_ids", []) or [])),
        "class_counts": vocab_class_histogram(tokenizer, rows),
        "note_padded_nonzero": (
            "Padded/unreachable rows are NOT zero — mean L2 norm reported. "
            "Zeroing them is not a meaningful BPW win for H95E quality claims."
        ),
    }

    if check_softmax_mass and padded:
        # Optional: random hidden → logits softmax mass on padded rows
        if device is None:
            device = w.device
        with torch.no_grad():
            # Use a few random rows from reachable as fake hidden states via embed
            idx = torch.tensor(list(reachable)[:8], device=device, dtype=torch.long)
            h = w[idx].float()  # [8, H]
            logits = h @ w.float().T  # [8, V]
            probs = torch.softmax(logits, dim=-1)
            pad_idx = torch.tensor(padded, device=device, dtype=torch.long)
            mass = probs.index_select(1, pad_idx).sum(dim=1)
            out["softmax_mass_on_padded_rows"] = {
                "mean": float(mass.mean().item()),
                "max": float(mass.max().item()),
                "n_probes": int(mass.numel()),
            }
    return out


def count_token_frequencies(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 256,
) -> Counter[int]:
    """Tokenize texts and count token-id frequencies (calib fit)."""
    counts: Counter[int] = Counter()
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        for tid in enc["input_ids"]:
            counts[int(tid)] += 1
    return counts


def assign_frequency_tiers(
    *,
    vocab_rows: int,
    reachable: set[int],
    special_ids: set[int],
    freq: Mapping[int, int],
    schedule: str = "default",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build per-row keep array (int8) from calib frequencies.

    Schedules
    ---------
    default (documented in task):
      special → 7
      top ~1% mass or top ~1k → 7
      next ~9% mass → 6
      next ~30% mass → 5
      remaining seen → 4
      never-seen reachable → 4
      padded → 3 (forward); storage estimate may omit

    aggressive:
      special → 7
      top 0.5% / top 500 → 7
      next 5% → 6
      next 20% → 5
      remaining seen → 3
      never-seen → 3
      padded → 3

    conservative:
      special → 7
      top 2% / top 2k → 7
      next 15% → 6
      next 40% → 5
      remaining seen → 5
      never-seen → 4
      padded → 3
    """
    keeps = np.full(vocab_rows, 3, dtype=np.int8)  # default padded/forward
    total_mass = sum(freq.values()) or 1

    # Rank reachable non-special by frequency descending
    scored = []
    for tid in reachable:
        if tid in special_ids:
            continue
        scored.append((tid, int(freq.get(tid, 0))))
    scored.sort(key=lambda x: (-x[1], x[0]))

    if schedule == "aggressive":
        top_mass_frac, top_n_cap = 0.005, 500
        band2_frac, band3_frac = 0.05, 0.20
        k_top, k_b2, k_b3, k_seen, k_unseen, k_pad = 7, 6, 5, 3, 3, 3
    elif schedule == "conservative":
        top_mass_frac, top_n_cap = 0.02, 2000
        band2_frac, band3_frac = 0.15, 0.40
        k_top, k_b2, k_b3, k_seen, k_unseen, k_pad = 7, 6, 5, 5, 4, 3
    else:  # default
        schedule = "default"
        top_mass_frac, top_n_cap = 0.01, 1000
        band2_frac, band3_frac = 0.09, 0.30
        k_top, k_b2, k_b3, k_seen, k_unseen, k_pad = 7, 6, 5, 4, 4, 3

    # Specials
    for sid in special_ids:
        if 0 <= sid < vocab_rows:
            keeps[sid] = 7

    # Mass-cumulative bands among non-special reachable
    cum = 0
    n_top = 0
    band_counts = {"special": 0, "top": 0, "band2": 0, "band3": 0, "seen_rest": 0, "unseen": 0, "padded": 0}

    for sid in special_ids:
        if sid in reachable and 0 <= sid < vocab_rows:
            band_counts["special"] += 1

    # Head keep=7: rank floor of top_n_cap (≈1k / 500 / 2k), expanded if needed
    # so cumulative mass reaches top_mass_frac. (Avoid collapsing to 1 id when a
    # single token exceeds 1% mass on tiny calib corpora.)
    top_cut = min(top_n_cap, len(scored)) if scored else 0
    mass_acc = sum(c for _, c in scored[:top_cut])
    while top_cut < len(scored) and mass_acc / total_mass < top_mass_frac:
        mass_acc += scored[top_cut][1]
        top_cut += 1

    # Band2 / band3 by additional mass fraction of total
    def mass_cut(start: int, frac: float) -> int:
        if start >= len(scored):
            return start
        acc = 0
        j = start
        while j < len(scored) and acc / total_mass < frac:
            c = scored[j][1]
            if c <= 0:
                break  # no further positive mass (remaining are unseen)
            acc += c
            j += 1
        return j

    b2_end = mass_cut(top_cut, band2_frac)
    b3_end = mass_cut(b2_end, band3_frac)

    for i, (tid, c) in enumerate(scored):
        if i < top_cut:
            keeps[tid] = k_top
            band_counts["top"] += 1
        elif i < b2_end:
            keeps[tid] = k_b2
            band_counts["band2"] += 1
        elif i < b3_end:
            keeps[tid] = k_b3
            band_counts["band3"] += 1
        elif c > 0:
            keeps[tid] = k_seen
            band_counts["seen_rest"] += 1
        else:
            keeps[tid] = k_unseen
            band_counts["unseen"] += 1

    for i in range(vocab_rows):
        if i not in reachable:
            keeps[i] = k_pad
            band_counts["padded"] += 1

    # Keep histogram
    keep_hist = {str(k): int((keeps == k).sum()) for k in range(8)}
    seen_ids = {t for t, c in freq.items() if c > 0}
    meta = {
        "schedule": schedule,
        "total_calib_tokens": int(total_mass),
        "n_unique_seen_ids": len(seen_ids),
        "top_cut_n": top_cut,
        "top_mass_frac_target": top_mass_frac,
        "top_n_cap": top_n_cap,
        "band2_end_n": b2_end,
        "band3_end_n": b3_end,
        "band_row_counts": band_counts,
        "keep_hist_rows": keep_hist,
        "keeps_by_band": {
            "special": 7,
            "top": k_top,
            "band2": k_b2,
            "band3": k_b3,
            "seen_rest": k_seen,
            "unseen_reachable": k_unseen,
            "padded": k_pad,
        },
        "storage_note": (
            "Forward pass leaves padded rows at keep=k_pad (not zeroed). "
            "est BPW may omit padded rows from storage (count as 0 stored words)."
        ),
    }
    return keeps, meta


def apply_embed_row_keeps(
    weight: torch.Tensor,
    row_keeps: np.ndarray,
    *,
    baseline_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Quantize embedding rows in-place according to per-row keep bits.

    If *baseline_weight* is given, copy from it first (same shape).
    Groups rows by keep value for efficiency.
    """
    row_keeps = np.asarray(row_keeps, dtype=np.int8)
    rows = int(weight.shape[0])
    if row_keeps.shape[0] != rows:
        raise ValueError(f"row_keeps length {row_keeps.shape[0]} != vocab rows {rows}")

    with torch.no_grad():
        if baseline_weight is not None:
            weight.copy_(baseline_weight.to(device=weight.device, dtype=weight.dtype))

        # Work on a uint16 view of a CPU copy, then write back
        words = bf16_tensor_to_u16(weight)  # [V, H]
        for k in range(0, 8):
            idx = np.flatnonzero(row_keeps == k)
            if idx.size == 0:
                continue
            if k >= 7:
                continue  # already exact from baseline
            block = words[idx]
            words[idx] = quantize_bf16_mantissas(block, k)

        new_t = u16_to_bf16_tensor(words, weight)
        weight.copy_(new_t)

    # Stats
    hist = {str(k): int((row_keeps == k).sum()) for k in range(8)}
    avg_row_keep = float(row_keeps.astype(np.float64).mean())
    return {
        "row_keep_hist": hist,
        "avg_row_keep": avg_row_keep,
        "n_rows_quantized": int((row_keeps < 7).sum()),
    }


def avg_keep_bits_with_embed_rows(
    keep_map: Mapping[str, int],
    state_dict: Mapping[str, torch.Tensor],
    *,
    embed_name: str,
    row_keeps: np.ndarray,
    omit_padded_from_storage: bool = False,
    padded_indices: set[int] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Word-weighted avg mantissa keep; embed uses per-row keeps.

    If omit_padded_from_storage, padded embed rows contribute 0 words to the
    denominator and 0 to the numerator (storage truncation estimate).
    """
    row_keeps = np.asarray(row_keeps, dtype=np.int8)
    seen: set[int] = set()
    total_w = 0
    total_k = 0.0
    embed_words_counted = 0
    embed_words_omitted = 0
    body_words = 0

    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)

        if name == embed_name or (
            "embed" in name.lower() and name.endswith("weight") and t.ndim == 2
        ):
            hidden = int(t.shape[1])
            for i, rk in enumerate(row_keeps.tolist()):
                if omit_padded_from_storage and padded_indices and i in padded_indices:
                    embed_words_omitted += hidden
                    continue
                total_w += hidden
                total_k += int(rk) * hidden
                embed_words_counted += hidden
            continue

        n = int(t.numel())
        total_w += n
        total_k += int(keep) * n
        body_words += n

    avg = (total_k / total_w) if total_w else 7.0
    detail = {
        "avg_keep_bits": avg,
        "total_words_in_estimate": total_w,
        "embed_words_counted": embed_words_counted,
        "embed_words_omitted_padded": embed_words_omitted,
        "body_words": body_words,
        "omit_padded_from_storage": omit_padded_from_storage,
    }
    return avg, detail


def estimate_bytes_saved_with_embed_rows(
    keep_map: Mapping[str, int],
    state_dict: Mapping[str, torch.Tensor],
    *,
    embed_name: str,
    row_keeps: np.ndarray,
    omit_padded_from_storage: bool = False,
    padded_indices: set[int] | None = None,
) -> float:
    """Bytes saved vs full 7-bit mantissa (unique storages)."""
    row_keeps = np.asarray(row_keeps, dtype=np.int8)
    saved = 0.0
    seen: set[int] = set()
    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if name == embed_name or (
            "embed" in name.lower() and name.endswith("weight") and t.ndim == 2
        ):
            hidden = int(t.shape[1])
            for i, rk in enumerate(row_keeps.tolist()):
                if omit_padded_from_storage and padded_indices and i in padded_indices:
                    # Omit entirely: vs storing full 7-bit, "saving" all 7 bits
                    # (not storing the row) — count as 7/8 byte per word saved
                    saved += 7 * hidden / 8.0
                    continue
                saved += (7 - int(rk)) * hidden / 8.0
            continue
        k = int(keep)
        saved += (7 - k) * int(t.numel()) / 8.0
    return saved


def uniform_row_keeps(vocab_rows: int, keep: int) -> np.ndarray:
    if keep < 0 or keep > 7:
        raise ValueError(keep)
    return np.full(vocab_rows, int(keep), dtype=np.int8)


def compact_tier_summary(row_keeps: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    """Compact JSON-friendly summary (no full 151k list)."""
    return {
        "meta": meta,
        "keep_hist_rows": {str(k): int((row_keeps == k).sum()) for k in range(8)},
        "avg_row_keep": float(row_keeps.astype(np.float64).mean()),
        "avg_row_keep_reachable_only": None,  # filled by caller if needed
    }
