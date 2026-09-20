"""H95-sensitivity hybrid slots: mantissa-keep vs groupwise INT.

Evidence from H95 Phase C / H95Q stack / PR #17:

* mid-MLP (esp. bands 1–15) is robust — groupwise Q4/Q5 INT
* late MLP (16+) and last block are more fragile — H95 keep (per-weight exp)
* attention and norms need exponents — H95 (or BF16-raw for tiny vectors)
* embed tolerated H95 K3–K5 but collapsed under pure groupwise Q4 (PR #17)
* first-block MLP is moderately sensitive — H95 or milder INT

These maps are a *new* quantized reference. They are not S1 keep-maps and
are not PR #17 restore_q6.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pbr_h95.policy import layer_index

NUM_LAYERS_QWEN_05B = 24
LATE_MLP_FROM = 16


@dataclass(frozen=True)
class Slot:
    """Per-tensor storage / quantize choice."""

    kind: str  # "h95" | "int" | "bf16"
    keep: int = 7  # H95 mantissa keep-bits (ignored unless kind==h95)
    bits: int = 4  # groupwise INT width (ignored unless kind==int)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "keep": int(self.keep), "bits": int(self.bits)}


def H95(keep: int) -> Slot:
    return Slot(kind="h95", keep=int(keep), bits=0)


def INT(bits: int) -> Slot:
    return Slot(kind="int", keep=0, bits=int(bits))


def BF16() -> Slot:
    return Slot(kind="bf16", keep=7, bits=0)


@dataclass(frozen=True)
class HybridPolicy:
    name: str
    group_size: int
    embed: Slot
    mlp_mid: Slot
    mlp_late: Slot
    mlp_first: Slot
    mlp_last: Slot
    attn_mid: Slot
    attn_first: Slot
    attn_last: Slot
    other: Slot  # norms / bias / leftovers
    late_mlp_from: int = LATE_MLP_FROM
    num_layers: int = NUM_LAYERS_QWEN_05B
    mse_tie_ratio: float = 1.02
    outlier_frac: float = 0.001
    outlier_max_bits: int = 5
    description: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# Stretch: max INT coverage. Rate may pass ≤4.5; quality is the risk.
STRETCH = HybridPolicy(
    name="h_stretch",
    group_size=128,
    embed=INT(4),
    mlp_mid=INT(4),
    mlp_late=INT(4),
    mlp_first=INT(4),
    mlp_last=H95(5),
    attn_mid=H95(4),
    attn_first=H95(5),
    attn_last=H95(5),
    other=H95(7),
    outlier_frac=0.0015,
    description=(
        "Stretch ≤4.5: last-MLP + attn + norms stay H95 (per-weight exp); "
        "mid/late/first MLP and embed are groupwise Q4 INT."
    ),
)

# Practical: user-suggested protected set, INT on robust mid-MLP + embed.
PRACTICAL = HybridPolicy(
    name="h_practical",
    group_size=128,
    embed=INT(4),
    mlp_mid=INT(4),
    mlp_late=H95(4),
    mlp_first=INT(5),
    mlp_last=H95(7),
    attn_mid=H95(5),
    attn_first=H95(6),
    attn_last=H95(7),
    other=H95(7),
    outlier_frac=0.001,
    description=(
        "User-suggested split: late MLP / attn / last / norms = H95 mantissa-keep; "
        "mid MLP Q4 INT, first MLP Q5 INT, embed Q4 INT. Aim ≤5.0 + ≥0.95."
    ),
)

# Mix: late-MLP on INT Q5 (cheaper than H95 K4) if practical overshoots 5.0.
MIX = HybridPolicy(
    name="h_mix",
    group_size=128,
    embed=INT(4),
    mlp_mid=INT(4),
    mlp_late=INT(5),
    mlp_first=INT(4),
    mlp_last=H95(7),
    attn_mid=H95(5),
    attn_first=H95(6),
    attn_last=H95(7),
    other=H95(7),
    outlier_frac=0.0012,
    description=(
        "Rate-aware mix: last-MLP + attn + norms H95; late MLP groupwise Q5; "
        "mid/first MLP + embed Q4. Target ≤5.0 if H95-late overshoots."
    ),
)

# Quality backoff: raise INT widths / H95 keep if Q4 mid-MLP breaks 0.95.
RESTORE = HybridPolicy(
    name="h_restore",
    group_size=128,
    embed=INT(5),
    mlp_mid=INT(5),
    mlp_late=H95(5),
    mlp_first=H95(5),
    mlp_last=H95(7),
    attn_mid=H95(6),
    attn_first=H95(6),
    attn_last=H95(7),
    other=H95(7),
    outlier_frac=0.002,
    outlier_max_bits=5,
    mse_tie_ratio=1.01,
    description=(
        "Quality restore: mid MLP / embed Q5 INT; late/first MLP H95 K5; "
        "attn H95 K6; last/norms H95 K7. Rate expected >5.0."
    ),
)

# Stronger H95 body (near S1 quality, rate will miss ≤5).
QUALITY = HybridPolicy(
    name="h_quality",
    group_size=128,
    embed=H95(3),
    mlp_mid=H95(4),
    mlp_late=H95(5),
    mlp_first=H95(6),
    mlp_last=H95(7),
    attn_mid=H95(5),
    attn_first=H95(6),
    attn_last=H95(7),
    other=H95(7),
    outlier_frac=0.0,
    description=(
        "H95-heavy quality point: mid MLP K4 / late K5 / embed K3 / attn K5. "
        "Packed rate near S1 (~7+ BPW); used as Pareto upper quality."
    ),
)

POLICIES: dict[str, HybridPolicy] = {
    "h_stretch": STRETCH,
    "h_mix": MIX,
    "h_practical": PRACTICAL,
    "h_restore": RESTORE,
    "h_quality": QUALITY,
}

POLICY_NAMES = tuple(POLICIES)
# Search cheapest-first; restore/quality only if quality misses.
LADDER = ("h_stretch", "h_mix", "h_practical", "h_restore", "h_quality")


def get_policy(name: str) -> HybridPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown hybrid policy {name!r}; known={POLICY_NAMES}")
    return POLICIES[name]


def family_label(name: str, policy: HybridPolicy) -> str:
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


def slot_for_name(name: str, policy: HybridPolicy) -> Slot:
    fam = family_label(name, policy)
    return {
        "embed": policy.embed,
        "mlp_mid": policy.mlp_mid,
        "mlp_late": policy.mlp_late,
        "mlp_first": policy.mlp_first,
        "mlp_last": policy.mlp_last,
        "attn_mid": policy.attn_mid,
        "attn_first": policy.attn_first,
        "attn_last": policy.attn_last,
        "norm_bias": policy.other,
        "other": policy.other,
    }[fam]
