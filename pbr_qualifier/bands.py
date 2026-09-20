"""Qualification bands from the PBR-Hierarchical / PoC drafts.

Stage 2 may label a *projection*. It must not call a model one-GB qualified;
that requires a measured full container at ≤2 BPW.
"""

from __future__ import annotations

from dataclasses import dataclass


def qualification_band(bpw: float) -> str:
    """Map complete BPW to the paper's engineering band.

    Thresholds are inclusive on the lower (better) side of each named range:
    ≤1 extreme, ≤2 one-GB class, ≤4 exceptional, ≤8 strong, else not exceptional.
    """
    if bpw <= 1.0:
        return "extreme"
    if bpw <= 2.0:
        return "one_gb_class"
    if bpw <= 4.0:
        return "exceptional"
    if bpw <= 8.0:
        return "strong"
    return "not_exceptional"


BAND_HELP = {
    "extreme": "≤1 BPW — extreme (projection only; not a measured full-model result)",
    "one_gb_class": "≤2 BPW — one-GB *class* on an 8 GB BF16 reference (projection only)",
    "exceptional": "2–4 BPW — exceptional candidate band (projection only)",
    "strong": "4–8 BPW — strong candidate band (projection only)",
    "not_exceptional": ">8 BPW — not exceptional for PBR-Direct-class targets",
}


@dataclass(frozen=True)
class QualificationDecision:
    projected_bpw: float
    band: str
    high_potential: bool
    one_gb_qualified: bool
    one_gb_qualified_reason: str
    high_potential_reason: str

    def as_dict(self) -> dict:
        return {
            "projected_bpw": self.projected_bpw,
            "band": self.band,
            "high_potential": self.high_potential,
            "one_gb_qualified": self.one_gb_qualified,
            "one_gb_qualified_reason": self.one_gb_qualified_reason,
            "high_potential_reason": self.high_potential_reason,
        }


def qualification_decision(projected_bpw: float) -> QualificationDecision:
    band = qualification_band(projected_bpw)
    high = projected_bpw <= 4.0
    return QualificationDecision(
        projected_bpw=projected_bpw,
        band=band,
        high_potential=high,
        one_gb_qualified=False,
        one_gb_qualified_reason=(
            "Stage 2 cannot one-GB-qualify a model. That label requires a "
            "measured full-container encode at ≤2 BPW, not a sample projection."
        ),
        high_potential_reason=(
            "Projected complete BPW ≤ 4."
            if high
            else "Projected complete BPW > 4; papers call this not high-potential."
        ),
    )
