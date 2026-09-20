"""Mantissa zoo: exact roundtrips on toy tensors + one local real slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_codecs.mantissa_predictors import (
    ar_features,
    decode_mant_ar,
    decode_mant_gbdt,
    decode_mant_idf,
    decode_mant_markov,
    decode_mant_uncond,
    encode_mant_ar,
    encode_mant_gbdt,
    encode_mant_idf,
    encode_mant_markov,
    encode_mant_uncond,
    feature_bins,
    fit_bit_markov,
    fit_gbdt_bits,
    fit_idf_tables,
    fit_mlp,
    idf_apply,
    laplace_nll_from_counts,
    markov_nll,
    uncond_counts,
)
from pbr_core.bf16 import make_bf16_bits, special_payload_words, split_components
from pbr_core.binary_rans import binary_rans_decode, binary_rans_encode, p_to_freq1
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d
from pbr_encoder.hf_weights import inventory_from_dir, select_weight_specs
from pbr_encoder.mantissa_audit import GATE_MANT_BPW
from pbr_encoder.mantissa_zoo import DISCLAIMER, format_markdown
from pbr_encoder.verification import assert_exact

QWEN_DIR = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")


def _toy(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.choice(np.array([120, 121, 126, 127, 128], dtype=np.uint16), size=n),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(16, n // 16)


def test_binary_rans_roundtrip() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=2048, dtype=np.uint8)
    p = np.clip(rng.random(2048) * 0.8 + 0.1, 0.05, 0.95)
    blob = binary_rans_encode(bits, p_to_freq1(p))
    rec = binary_rans_decode(blob, bits.size, p_to_freq1(p))
    assert np.array_equal(rec, bits)


def test_uncond_rans_and_specials() -> None:
    words = _toy(256, 2)
    spec = np.resize(special_payload_words(), 32)
    extra = make_bf16_bits(np.zeros(32, np.uint16), np.full(32, 127, np.uint16), spec & 0x7F)
    words = np.concatenate([words.ravel(), extra]).reshape(18, 16)
    _s, _e, m = split_components(words)
    blob = encode_mant_uncond(m)
    rec = decode_mant_uncond(blob, m.size).reshape(m.shape)
    assert_exact(rec, m, label="uncond_m")


def test_idf_reversible_and_exact() -> None:
    rng = np.random.default_rng(3)
    m = rng.integers(0, 128, size=(24, 17), dtype=np.uint8)
    tables = fit_idf_tables(m, n_layers=3)
    y = idf_apply(m, tables, inverse=False)
    rec = idf_apply(y, tables, inverse=True)
    assert np.array_equal(rec, m)
    blob = encode_mant_idf(m, tables)
    rec2 = decode_mant_idf(blob, m.size, tables, m.shape)
    assert np.array_equal(rec2, m)


def test_markov_exact_on_structured_bits() -> None:
    m = np.tile(np.arange(16, dtype=np.uint8), 32).reshape(16, 32) & np.uint8(0x7F)
    counts = fit_bit_markov(m, 4)
    blob = encode_mant_markov(m, counts, 4)
    rec = decode_mant_markov(blob, m.size, counts, 4, m.shape)
    assert np.array_equal(rec, m)
    nll = markov_nll(counts, m, 4)
    assert nll / m.size < 4.0


def test_gbdt_and_ar_exact_small() -> None:
    words = _toy(128, 4)
    s, e, m = split_components(words)
    s, e, m = s.reshape(words.shape), e.reshape(words.shape), m.reshape(words.shape)
    X = feature_bins(s, e, m)
    gbdt = fit_gbdt_bits(X, m, n_trees=4, n_bins=32, lr=0.3)
    blob = encode_mant_gbdt(m, s, e, gbdt)
    rec = decode_mant_gbdt(blob, s, e, gbdt, m.shape)
    assert_exact(rec, m, label="gbdt")
    Xa = ar_features(s, e, m)
    ar = fit_mlp(Xa, m.ravel().astype(np.int64), hidden=8, epochs=2, batch=64, seed=0)
    blob = encode_mant_ar(m, s, e, ar)
    rec = decode_mant_ar(blob, s, e, ar, m.shape)
    assert_exact(rec, m, label="ar")


def test_random_mantissa_stays_near_seven() -> None:
    rng = np.random.default_rng(5)
    m = rng.integers(0, 128, size=8192, dtype=np.uint8)
    nll = laplace_nll_from_counts(uncond_counts(m[:6553]), m[6553:])
    bpw = nll / max(m.size - 6553, 1)
    assert 6.5 < bpw < 7.2
    assert bpw > 4.0
    assert GATE_MANT_BPW == 6.5


def test_format_markdown_no_false_four_bpw() -> None:
    methods = [
        {
            "name": "uncond_rANS",
            "ideal_bpw": 6.97,
            "complete_bpw": 6.98,
            "holdout_charged_bpw": 7.01,
            "total_complete_bpw": 10.62,
            "vs_pbre": 0.01,
            "model_bytes": 40,
            "beats_gate": False,
        }
    ]
    report = {
        "disclaimer": DISCLAIMER,
        "model": {"repo_id": "unit/test", "revision": "abc"},
        "summary": {
            "n_tensors": 1,
            "n_words": 100,
            "n_hold": 20,
            "weighted_H_sign": 1.0,
            "weighted_H_exp": 2.6,
            "weighted_H_mantissa": 6.97,
            "pbre_total_bpw": 10.61,
            "pbre_ref_fullset": 10.616,
            "best_method": "uncond_rANS",
            "best_mantissa_complete_bpw": 6.98,
            "best_total_complete_bpw": 10.62,
            "beats_gate": False,
            "beats_pbre": False,
            "stretch_le4": False,
            "fit_s": 0.0,
            "elapsed_s": 0.0,
            "reservoir": 10,
            "mixture_used": ["uncond"],
            "methods": methods,
            "slice": {"n_words": 8, "all_exact": "PASS", "encode_decode_s": 0.01, "mb_s": 1.0},
        },
    }
    md = format_markdown(report)
    assert "not ≤4 BPW" in md or "≤4 BPW total: no" in md
    assert "1–2 GB" in md
    assert "MISS" in md
    assert DISCLAIMER.split()[0] in md or "Mantissa" in md


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="local Qwen checkpoint not present")
def test_real_slice_uncond_markov_idf() -> None:
    specs = select_weight_specs(
        inventory_from_dir(QWEN_DIR), min_bytes=1_000_000, max_bytes=8_000_000
    )
    words = as_2d(load_uint16(specs[0]))[:16, :64]
    _s, _e, m = split_components(words)
    m = m.reshape(words.shape)
    blob = encode_mant_uncond(m)
    rec = decode_mant_uncond(blob, m.size).reshape(m.shape)
    assert_exact(rec, m, label="real_uncond")
    counts = fit_bit_markov(m, 4)
    blob = encode_mant_markov(m, counts, 4)
    rec = decode_mant_markov(blob, m.size, counts, 4, m.shape)
    assert_exact(rec, m, label="real_markov")
    tables = fit_idf_tables(m, n_layers=2)
    blob = encode_mant_idf(m, tables)
    rec = decode_mant_idf(blob, m.size, tables, m.shape)
    assert_exact(rec, m, label="real_idf")
