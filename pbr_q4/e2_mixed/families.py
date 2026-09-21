"""Tensor family / projection labels and mixed-precision candidate sets.

embed / attn / MLP body bits ∈ {Q4, Q5, Q6}.
Norms / biases ∈ {Q8, BF16}.
Protected channels (importance overlay) ∈ {Q6, Q8, BF16}.
"""

from __future__ import annotations

import re

from pbr_h95.policy import layer_index
from pbr_q4.e2_mixed.const import (
    BODY_BITS,
    LATE_MLP_FROM,
    NORM_BITS,
    NUM_LAYERS_QWEN_05B,
    PROTECT_BITS,
)

_PROJ_RE = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    re.IGNORECASE,
)

# H95 Phase B: attn_mid most sensitive under mantissa-keep; mlp_mid most robust.
# Groupwise INT is harsher than H95-K (PR #17), so embed is weighted up.
FAMILY_IMPORTANCE = {
    "embed": 1.35,
    "attn_first": 1.40,
    "attn_last": 1.40,
    "attn_mid": 1.45,
    "mlp_first": 1.20,
    "mlp_last": 1.20,
    "mlp_late": 1.25,
    "mlp_mid": 1.00,
    "norm_bias": 2.50,
    "other": 1.10,
}

# Residual-stream writers get a structured bump (protect those channels first).
PROJ_IMPORTANCE = {
    "q_proj": 1.08,
    "k_proj": 1.04,
    "v_proj": 1.10,
    "o_proj": 1.18,
    "gate_proj": 1.00,
    "up_proj": 1.00,
    "down_proj": 1.22,
    "embed": 1.00,
    "norm": 1.00,
    "bias": 1.00,
    "other": 1.00,
}


def projection_label(name: str) -> str:
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return "embed"
    if "bias" in n:
        return "bias"
    if "norm" in n:
        return "norm"
    m = _PROJ_RE.search(n)
    if m:
        return m.group(1).lower()
    return "other"


def family_label(name: str, *, num_layers: int = NUM_LAYERS_QWEN_05B, late_from: int = LATE_MLP_FROM) -> str:
    n = name.lower()
    if any(x in n for x in ("norm", "bias")):
        return "norm_bias"
    if "embed" in n or "lm_head" in n:
        return "embed"
    li = layer_index(n)
    is_mlp = "mlp." in n
    is_attn = ("self_attn" in n) or (".attn." in n)
    last = int(num_layers) - 1
    if li is None:
        return "other"
    if is_mlp:
        if li == 0:
            return "mlp_first"
        if li == last:
            return "mlp_last"
        if li >= int(late_from):
            return "mlp_late"
        return "mlp_mid"
    if is_attn:
        if li == 0:
            return "attn_first"
        if li == last:
            return "attn_last"
        return "attn_mid"
    return "other"


def candidate_bits(family: str, *, floor: int = 4, allow_protect: bool = True) -> tuple[int, ...]:
    """Allowed bit widths for a family, with optional protect overlay."""
    if family == "norm_bias":
        bits = list(NORM_BITS)
    else:
        bits = [b for b in BODY_BITS if b >= int(floor)]
        if allow_protect:
            bits.extend(b for b in PROTECT_BITS if b not in bits)
    out = sorted({int(b) for b in bits if b >= int(floor)})
    if not out:
        out = [16]
    return tuple(out)


def family_importance(family: str) -> float:
    return float(FAMILY_IMPORTANCE.get(family, 1.0))


def proj_importance(projection: str) -> float:
    return float(PROJ_IMPORTANCE.get(projection, 1.0))
