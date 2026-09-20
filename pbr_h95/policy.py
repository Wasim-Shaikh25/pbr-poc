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


# --- Phase C decision units (layer bands / families) ---

BAND_SPECS = (
    ("mlp_band_1_7", "mlp", range(1, 8)),
    ("mlp_band_8_15", "mlp", range(8, 16)),
    ("mlp_band_16_22", "mlp", range(16, 23)),
    ("attn_band_1_7", "attn", range(1, 8)),
    ("attn_band_8_15", "attn", range(8, 16)),
    ("attn_band_16_22", "attn", range(16, 23)),
)

FAMILY_UNIT_SPECS = (
    ("mlp_mid", "mlp", range(1, 23)),
    ("attn_mid", "attn", range(1, 23)),
)


def _name_in_unit(name: str, kind: str, layers: range, *, num_layers: int = 24) -> bool:
    """True if *name* is a non-bias weight in the given mid-layer unit."""
    n = name.lower()
    if "bias" in n:
        return False
    li = layer_index(n)
    if li is None or li not in layers:
        return False
    # never treat first/last as mid units (bands already exclude 0 and 23)
    if li == 0 or li == num_layers - 1:
        return False
    if kind == "mlp":
        return "mlp." in n
    if kind == "attn":
        return ("self_attn" in n) or bool(re.search(r"\.attn\.", n))
    return False


def unit_keep_fn(
    unit_keeps: dict[str, int],
    unit_specs: tuple,
    *,
    protected: int = 7,
    num_layers: int = 24,
):
    """Build keep_fn from per-unit keep values; emb/norm/bias/first/last stay protected."""

    def _fn(name: str) -> int:
        # protected families always
        base = default_keep_bits(name, default_large=protected)
        if base == protected and (
            any(x in name.lower() for x in ("embed", "lm_head", "norm", "bias"))
            or FIRST_LAYER_RE.search(name)
            or LAST_LAYER_RE.search(name)
        ):
            return protected
        for uname, kind, layers in unit_specs:
            if _name_in_unit(name, kind, layers, num_layers=num_layers):
                return int(unit_keeps.get(uname, protected))
        return protected

    return _fn


def avg_keep_bits(keep_map: dict[str, int], state_dict) -> float:
    """Unique-storage average mantissa keep bits (word-weighted)."""
    seen: set[int] = set()
    total_w = 0
    total_k = 0.0
    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        if not hasattr(t, "data_ptr"):
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        n = int(t.numel())
        total_w += n
        total_k += int(keep) * n
    if total_w == 0:
        return 7.0
    return total_k / total_w


# --- H95Q named policies (guide §18 first run) ---

H95Q_POLICY_NAMES = (
    "h95q_A_7_5",
    "h95q_B1_mlp_k4",
    "h95q_B2_embed_k5",
    "h95q_C_sketch_k4_row_recover",
)


def h95q_policy_description(name: str) -> str:
    """Human-readable rule summary for artifacts / README."""
    return {
        "h95q_A_7_5": (
            "Candidate A (H95 control): emb/lm/norm/bias/first/last → K7; "
            "mid attn/mlp (non-bias) → K5. Packed retained bits. Reproduce ~9.29 total est BPW."
        ),
        "h95q_B1_mlp_k4": (
            "Candidate B1: same as A, but low-sensitivity mlp_mid (layers 1..22, non-bias) → K4; "
            "attn_mid stays K5; emb/norm/bias/first/last stay K7."
        ),
        "h95q_B2_embed_k5": (
            "Candidate B2: embedding (tied lm_head) → K5; attn_mid → K6 (sensitive); "
            "mlp_mid → K5; norm/bias/first/last → K7."
        ),
        "h95q_C_sketch_k4_row_recover": (
            "Optional C sketch: body like Phase-C mid@K4 with emb/norm/bias/first/last@K7, "
            "then restore top-magnitude fraction of mlp_mid rows to K7 (magnitude sparse recovery)."
        ),
    }.get(name, name)


def h95q_A_7_5_keep_fn(*, num_layers: int = 24):
    """Existing per-tensor 7/5 policy (Phase A first map)."""

    def _fn(name: str) -> int:
        return default_keep_bits(name, default_large=5)

    return _fn


def h95q_B1_mlp_k4_keep_fn(*, num_layers: int = 24, mlp_keep: int = 4, attn_keep: int = 5, protected: int = 7):
    """B1: drop mlp_mid K5→K4; keep attn and protected families higher."""

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


def h95q_B2_embed_k5_keep_fn(
    *,
    num_layers: int = 24,
    embed_keep: int = 5,
    attn_keep: int = 6,
    mlp_keep: int = 5,
    protected: int = 7,
):
    """B2: embedding at K5; sensitive mid-attn at K6; mlp mid at K5; rest protected."""

    def _fn(name: str) -> int:
        n = name.lower()
        if "bias" in n:
            return protected
        fam = tensor_family(name, num_layers=num_layers)
        if fam == "embed":
            return int(embed_keep)
        if fam == "attn_mid":
            return int(attn_keep)
        if fam == "mlp_mid":
            return int(mlp_keep)
        return protected

    return _fn


def h95q_named_keep_fn(name: str, *, num_layers: int = 24):
    """Dispatch named H95Q policy → keep_fn. Raises KeyError for unknown names."""
    table = {
        "h95q_A_7_5": h95q_A_7_5_keep_fn,
        "h95q_B1_mlp_k4": h95q_B1_mlp_k4_keep_fn,
        "h95q_B2_embed_k5": h95q_B2_embed_k5_keep_fn,
    }
    if name not in table:
        raise KeyError(f"Unknown H95Q policy {name!r}; known={list(table)}")
    return table[name](num_layers=num_layers)


def family_keep_breakdown(
    keep_map: dict[str, int],
    state_dict,
    *,
    num_layers: int = 24,
) -> dict[str, dict]:
    """Word-weighted keep stats per tensor family (unique storage)."""
    seen: set[int] = set()
    fams: dict[str, dict] = {}
    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        if not hasattr(t, "data_ptr"):
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        fam = tensor_family(name, num_layers=num_layers) or "other"
        n = int(t.numel())
        slot = fams.setdefault(fam, {"n_words": 0, "sum_k": 0.0, "keep_hist": {}})
        slot["n_words"] += n
        slot["sum_k"] += int(keep) * n
        kh = slot["keep_hist"]
        ks = str(int(keep))
        kh[ks] = kh.get(ks, 0) + n
    for fam, slot in fams.items():
        nw = slot["n_words"]
        slot["avg_keep"] = (slot["sum_k"] / nw) if nw else 0.0
        del slot["sum_k"]
    return fams


# --- H95Q stack follow-up (Track1/2/3) ---
# Implementation lives in pbr_h95/h95q_stack.py; names listed for discovery.

H95Q_STACK_POLICY_NAMES = (
    "REF_B1_mlp_k4",
    "T1_B1_embed_default",
    "T1_B1_embed_aggressive",
    "T1_B1_embed_conservative",
    "T2_C_row_mag_0.02",
    "T2_C_row_mag_0.05",
    "T2_C_row_mag_0.10",
    "T2_C_channel_mag_0.05",
    "T3_mlp_all_k3",
    "T3_mlp_band_1_7_k3",
    "T3_mlp_band_8_15_k3",
    "T3_mlp_band_16_22_k3",
    "T3_mlp_robust50_k3",
    "S1_B1_aggr_embed_band815_k3",
    "S2_C_k3_base_row05_k7",
)


def h95q_stack_policy_description(name: str) -> str:
    """Delegate to h95q_stack.stack_description (lazy import)."""
    from pbr_h95.h95q_stack import stack_description

    return stack_description(name)
