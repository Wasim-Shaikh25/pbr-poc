"""Phase A mantissa audit: reversible transforms + held-out conditionals."""

from __future__ import annotations

import numpy as np

from pbr_core.bf16 import make_bf16_bits
from pbr_encoder.mantissa_audit import (
    GATE_MANT_BPW,
    audit_mantissa_fields,
    format_markdown,
    gray_decode,
    gray_encode,
    haar_fwd,
    haar_inv,
)


def test_gray_is_reversible() -> None:
    m = np.arange(128, dtype=np.uint8)
    assert np.array_equal(gray_decode(gray_encode(m)), m)


def test_haar_is_reversible() -> None:
    rng = np.random.default_rng(0)
    m = rng.integers(0, 128, size=(16, 17), dtype=np.uint8)
    s, d, leftover = haar_fwd(m)
    rec = haar_inv(s, d, leftover, cols=17)
    assert np.array_equal(rec, m)


def test_uncond_random_mantissa_near_seven() -> None:
    rng = np.random.default_rng(1)
    n = 4096
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.integers(120, 132, size=n, dtype=np.uint16),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(32, 128)
    audit = audit_mantissa_fields(words)
    uncond = next(m for m in audit["methods"] if m["name"] == "H(M)")
    assert 6.5 < uncond["ideal_bpw"] < 7.1
    assert uncond["complete_bpw"] >= uncond["ideal_bpw"]
    assert audit["best_complete_bpw"] > 4.0


def test_mantissa_determined_by_exp_has_low_conditional() -> None:
    # Each row is exp 0..127 so train and hold both see every exponent.
    exp = np.tile(np.arange(128, dtype=np.uint16), 32)
    mant = exp & np.uint16(0x7F)
    words = make_bf16_bits(np.zeros_like(exp), exp, mant).reshape(32, 128)
    audit = audit_mantissa_fields(words)
    cond = next(m for m in audit["methods"] if m["name"] == "H(M|exp)")
    assert cond["ideal_bpw"] < 0.05
    # Tiny tensors may not pay for 256-count tables; complete still includes a global table.
    assert cond["ideal_bpw"] < audit["uncond_complete_bpw"]


def test_format_markdown_states_gate_and_no_false_claims() -> None:
    rng = np.random.default_rng(2)
    n = 1024
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        np.full(n, 127, dtype=np.uint16),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(16, 64)
    audit = audit_mantissa_fields(words)
    report = {
        "disclaimer": "x",
        "model": {"repo_id": "unit/test", "revision": "abc"},
        "summary": {
            "n_tensors": 1,
            "n_words": int(words.size),
            "original_bytes": int(words.nbytes),
            "weighted_H_sign": 1.0,
            "weighted_H_exp": 0.0,
            "weighted_H_mantissa": audit["H_mantissa_full"],
            "uncond_complete_bpw": audit["uncond_complete_bpw"],
            "best_method": audit["best_method"],
            "best_complete_mantissa_bpw": audit["best_complete_bpw"],
            "implied_total_bpw": audit["implied_total_bpw"],
            "beats_gate": audit["beats_gate"],
            "ranked_methods": [
                {
                    "name": m["name"],
                    "weighted_ideal_bpw": m.get("ideal_bpw") or 0.0,
                    "weighted_complete_bpw": m.get("complete_bpw") or 0.0,
                    "weighted_mi": m.get("mi_vs_uncond"),
                    "tensors_improved_vs_uncond": 1 if m.get("improves_vs_uncond") else 0,
                    "tensors_present": 1,
                    "beats_gate": bool(m.get("beats_gate")),
                    "skipped_all": bool(m.get("skipped")),
                }
                for m in audit["methods"]
                if not m.get("skipped")
            ],
        },
    }
    md = format_markdown(report)
    assert "Phase A mantissa audit" in md
    assert f"{GATE_MANT_BPW}" in md
    assert "1–2 GB" in md or "1-2 GB" in md
    assert "≤4 BPW" in md or "<=4 BPW" in md
