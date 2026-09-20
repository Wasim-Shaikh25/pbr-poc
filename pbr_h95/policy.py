"""First-experiment precision policy and tensor-family helpers (H95)."""

from __future__ import annotations

import re
from typing import Callable

# Qwen2.5-0.5B: num_hidden_layers=24 → last block index 23
LAST_LAYER_RE = re.compile(r"layers\.23\.")
FIRST_LAYER_RE = re.compile(r"layers\.0\.")
LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\.")

FAMILIES = (
    "embed",
    "lm_head_tied",
    "norm",
    "attn_mid",
    "mlp_mid",
    "first_block",
    "last_block",
)


def default_keep_bits(name: str, *, default_large: int = 5) -> int:
    """Recommended first map: protect emb/lm/norm/bias/first/last at 7; large mats at *default_large*."""
    n = name.lower()
    if any(x in n for x in ("embed", "lm_head", "norm", "bias")):
        return 7
    # first / last transformer block (Qwen2.5-0.5B: layers.0 / layers.23)
    if FIRST_LAYER_RE.search(n) or LAST_LAYER_RE.search(n):
        return 7
    if any(x in n for x in ("mlp.", "self_attn.", "attn.")):
        return int(default_large)
    return 7


def layer_index(name: str) -> int | None:
    m = LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def tensor_family(name: str, *, num_layers: int = 24) -> str | None:
    """Assign a primary sensitivity family for Phase-B ablations.

    With ``tie_word_embeddings=true`` there is no separate ``lm_head`` tensor;
    ``embed`` and ``lm_head_tied`` refer to the same storage (embed_tokens).
    Callers that ablate ``lm_head_tied`` should treat it as an alias of ``embed``.
    """
    n = name.lower()
    if "embed" in n or n.endswith("lm_head.weight") or "lm_head" in n:
        return "embed"  # tied lm_head shares this storage
    if "norm" in n:
        return "norm"
    li = layer_index(n)
    if li is None:
        return None
    last = num_layers - 1
    if li == 0:
        return "first_block"
    if li == last:
        return "last_block"
    if "self_attn" in n or re.search(r"\.attn\.", n):
        return "attn_mid"
    if "mlp." in n:
        return "mlp_mid"
    return None


def keep_bits_map_for_names(
    names: list[str],
    keep_fn: Callable[[str], int],
) -> dict[str, int]:
    return {name: int(keep_fn(name)) for name in names}


def uniform_mid_keep_fn(mid_keep: int, *, protected: int = 7, num_layers: int = 24):
    """Protected families stay at *protected*; mid attn/mlp (not first/last) use *mid_keep*."""

    def _fn(name: str) -> int:
        fam = tensor_family(name, num_layers=num_layers)
        if fam in ("attn_mid", "mlp_mid"):
            # still protect biases inside mid blocks at full precision
            if "bias" in name.lower():
                return protected
            return int(mid_keep)
        return protected

    return _fn


def family_ablation_keep_fn(
    family: str,
    *,
    family_keep: int = 5,
    other_keep: int = 7,
    num_layers: int = 24,
):
    """Quantize ONLY *family* to *family_keep*; everything else exact *other_keep*.

    ``lm_head_tied`` is an alias of ``embed`` (tie_word_embeddings).
    """
    target = "embed" if family == "lm_head_tied" else family

    def _fn(name: str) -> int:
        fam = tensor_family(name, num_layers=num_layers)
        if fam == target:
            return int(family_keep)
        return int(other_keep)

    return _fn


def policy_then_mlp_mid_keep_fn(
    mlp_mid_keep: int,
    *,
    base_large: int = 5,
    protected: int = 7,
    num_layers: int = 24,
):
    """Start from default_keep_bits(base_large), then drop mlp_mid further."""

    def _fn(name: str) -> int:
        base = default_keep_bits(name, default_large=base_large)
        fam = tensor_family(name, num_layers=num_layers)
        if fam == "mlp_mid" and "bias" not in name.lower():
            return int(mlp_mid_keep)
        return base

    return _fn
