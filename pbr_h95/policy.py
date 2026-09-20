"""First-experiment precision policy from the H95 guide."""

from __future__ import annotations

import re


def default_keep_bits(name: str, *, default_large: int = 5) -> int:
    """Recommended first map: protect emb/lm/norm/bias/first/last at 7; large mats at *default_large*."""
    n = name.lower()
    if any(x in n for x in ("embed", "lm_head", "norm", "bias")):
        return 7
    # first / last transformer block
    if re.search(r"layers\.0\.", n) or re.search(r"layers\.23\.", n):
        return 7
    if any(x in n for x in ("mlp.", "self_attn.", "attn.")):
        return int(default_large)
    return 7
