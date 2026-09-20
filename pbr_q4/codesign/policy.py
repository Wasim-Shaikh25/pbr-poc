"""Sensitivity-guided mixed precision for PBR-Q4 codesign.

Reuses H95 family / layer-band evidence:

* middle MLP is robust → Q4-class
* late MLP + attention are more fragile → Q5/Q6-class
* embed previously tolerated K4/K5 mantissa (not the same as groupwise Q4)
* norms / bias / tiny vectors stay BF16 (or Q8 if forced)

These are *new* groupwise integer policies. They are not H95Q keep-bit
maps and do not claim S1 compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pbr_h95.policy import layer_index

NUM_LAYERS_QWEN_05B = 24
LATE_MLP_FROM = 16  # layers 16..last were more fragile in H95 Phase C / S1


@dataclass(frozen=True)
class CodesignPolicy:
    name: str
    group_size: int
    embed: int
    mlp_mid: int  # layers 1 .. late_from-1
    mlp_late: int  # layers late_from .. n-2
    mlp_first: int
    mlp_last: int
    attn_mid: int
    attn_first: int
    attn_last: int
    other: int | None  # None = BF16
    late_mlp_from: int = LATE_MLP_FROM
    num_layers: int = NUM_LAYERS_QWEN_05B
    mse_tie_ratio: float = 1.02
    outlier_frac: float = 0.001
    outlier_max_bits: int = 5  # protect outliers on Q4/Q5 only
    description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Stretch: aim ≤4.5 physical BPW. Quality may fail — that is a valid result.
STRETCH = CodesignPolicy(
    name="stretch",
    group_size=128,
    embed=4,
    mlp_mid=4,
    mlp_late=4,
    mlp_first=4,
    mlp_last=5,
    attn_mid=5,
    attn_first=5,
    attn_last=6,
    other=None,
    outlier_frac=0.0015,
    description=(
        "Aggressive Q4-class: mid+late MLP Q4, embed Q4, attn Q5, last-attn Q6, "
        "last-MLP Q5, norms BF16. Stretch target ≤4.5 physical BPW."
    ),
)

# Practical: aim ≤5.0 with more protection on late MLP / attn / embed.
PRACTICAL = CodesignPolicy(
    name="practical",
    group_size=128,
    embed=5,
    mlp_mid=4,
    mlp_late=5,
    mlp_first=5,
    mlp_last=6,
    attn_mid=6,
    attn_first=6,
    attn_last=6,
    other=None,
    outlier_frac=0.001,
    description=(
        "Mixed Q4/Q5/Q6: mid-MLP (1-15) Q4, late-MLP Q5, last-MLP Q6, "
        "attention Q6, embed Q5, norms BF16. Practical target ≤5.0 physical BPW."
    ),
)

# Safe backoff if practical misses the 0.95 proxy floor.
SAFE = CodesignPolicy(
    name="safe",
    group_size=128,
    embed=5,
    mlp_mid=4,
    mlp_late=6,
    mlp_first=6,
    mlp_last=6,
    attn_mid=6,
    attn_first=6,
    attn_last=6,
    other=None,
    outlier_frac=0.002,
    mse_tie_ratio=1.01,
    description=(
        "Quality backoff: mid-MLP Q4, everything late/attn/first Q6, embed Q5, "
        "norms BF16. Used only if stretch/practical miss held-out ≥0.95."
    ),
)

# Extra backoff: raise mid-MLP to Q5 (will likely exceed 5.0 — report honestly).
RESTORE = CodesignPolicy(
    name="restore",
    group_size=128,
    embed=5,
    mlp_mid=5,
    mlp_late=6,
    mlp_first=6,
    mlp_last=6,
    attn_mid=6,
    attn_first=6,
    attn_last=6,
    other=None,
    outlier_frac=0.002,
    description="Restore mid-MLP to Q5 if Q4 mid-MLP breaks the 0.95 proxy floor.",
)

# Quality-seeking points (rate likely >5.0). Measured, not claimed.
RESTORE_Q6 = CodesignPolicy(
    name="restore_q6",
    group_size=128,
    embed=6,
    mlp_mid=6,
    mlp_late=6,
    mlp_first=6,
    mlp_last=6,
    attn_mid=6,
    attn_first=6,
    attn_last=6,
    other=None,
    outlier_frac=0.002,
    outlier_max_bits=6,
    description="Uniform groupwise Q6 + BF16 norms. Quality backoff; rate expected >5.0.",
)

QUALITY = CodesignPolicy(
    name="quality",
    group_size=128,
    embed=8,
    mlp_mid=6,
    mlp_late=8,
    mlp_first=8,
    mlp_last=8,
    attn_mid=8,
    attn_first=8,
    attn_last=8,
    other=None,
    outlier_frac=0.003,
    outlier_max_bits=6,
    description="Quality-first: mid-MLP Q6, embed/attn/late Q8, norms BF16.",
)

POLICIES: dict[str, CodesignPolicy] = {
    "stretch": STRETCH,
    "practical": PRACTICAL,
    "safe": SAFE,
    "restore": RESTORE,
    "restore_q6": RESTORE_Q6,
    "quality": QUALITY,
}

POLICY_NAMES = tuple(POLICIES)
LADDER = ("stretch", "practical", "safe", "restore", "restore_q6", "quality")


def get_policy(name: str) -> CodesignPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; known={POLICY_NAMES}")
    return POLICIES[name]


def bits_for_name(name: str, policy: CodesignPolicy) -> int | None:
    """Return groupwise bit-width, or None to store BF16 raw.

    Biases, norms, and tiny non-weight vectors are BF16. Integer bit widths
    are 4/5/6/8 groupwise — not H95 mantissa keep-bits.
    """
    n = name.lower()
    if any(x in n for x in ("norm", "bias")):
        return None
    if "embed" in n or "lm_head" in n:
        return int(policy.embed)
    li = layer_index(n)
    is_mlp = "mlp." in n
    is_attn = ("self_attn" in n) or (".attn." in n)
    last = int(policy.num_layers) - 1
    if li is None:
        return policy.other
    if is_mlp:
        if li == 0:
            return int(policy.mlp_first)
        if li == last:
            return int(policy.mlp_last)
        if li >= int(policy.late_mlp_from):
            return int(policy.mlp_late)
        return int(policy.mlp_mid)
    if is_attn:
        if li == 0:
            return int(policy.attn_first)
        if li == last:
            return int(policy.attn_last)
        return int(policy.attn_mid)
    return policy.other


def family_label(name: str, policy: CodesignPolicy) -> str:
    n = name.lower()
    if any(x in n for x in ("norm", "bias")):
        return "norm_bias"
    if "embed" in n or "lm_head" in n:
        return "embed"
    li = layer_index(n)
    is_mlp = "mlp." in n
    is_attn = ("self_attn" in n) or (".attn." in n)
    last = int(policy.num_layers) - 1
    if li is None:
        return "other"
    if is_mlp:
        if li == 0:
            return "mlp_first"
        if li == last:
            return "mlp_last"
        if li >= int(policy.late_mlp_from):
            return "mlp_late"
        return "mlp_mid"
    if is_attn:
        if li == 0:
            return "attn_first"
        if li == last:
            return "attn_last"
        return "attn_mid"
    return "other"
