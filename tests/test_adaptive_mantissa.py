"""Adaptive 7-bit mantissa codec: synthetic sanity + exact round-trip."""

from __future__ import annotations

import numpy as np

from pbr_adaptive_mantissa.cli import synthetic_report
from pbr_adaptive_mantissa.codec import (
    STRAT_ALL_RAW,
    decode_bf16_bundle,
    decode_mantissa,
    encode_bf16_bundle,
    encode_mantissa,
    pack_lsb,
    unpack_lsb,
)
from pbr_adaptive_mantissa.predict import RULE_POS, predict
from pbr_core.bf16 import join_components, special_payload_words, split_components
from pbr_core.bitio import BitWriter
from pbr_core.hashing import sha256_words
from pbr_encoder.disk_tunnel import select_specs, write_tiny_qwen_fixture
from pbr_encoder.verification import assert_exact


def test_pack7_matches_bitwriter() -> None:
    rng = np.random.default_rng(2)
    m = rng.integers(0, 128, size=256, dtype=np.uint8)
    packed = pack_lsb(m, 7)
    w = BitWriter()
    w.write_array(m, 7)
    assert packed == w.finalize()
    rec = unpack_lsb(packed, 256, 7)
    assert np.array_equal(rec, m)


def test_predict_integer_only() -> None:
    assert predict(10, 200, 3, 0) == 10
    assert predict(10, 200, 3, RULE_POS) == 3
    assert predict(127, 0, 0, 2) == 0


def test_synthetic_sanity_from_guide() -> None:
    syn = synthetic_report()
    rnd = syn["cases"]["random"]
    skew = syn["cases"]["skewed"]
    st = syn["cases"]["structured_pos"]
    assert rnd["exact"] is True
    assert skew["exact"] is True
    assert st["exact"] is True
    # Random mantissas stay near 7 BPW and must not invent compression.
    assert 6.9 <= rnd["mantissa_bpw"] <= 7.4
    assert rnd["strategy"] == "ALL_RAW"
    assert rnd["used_huffman_table"] is False
    # Skewed: Huffman (or MIXED with Huffman) should beat raw 7.
    assert skew["mantissa_bpw"] < 6.5
    assert skew["strategy"] in {"ALL_HUFFMAN", "MIXED"}
    assert skew["used_huffman_table"] is True
    # Structured position: CONTEXT wins and is well under 7.
    assert st["strategy"] in {"ALL_CONTEXT", "MIXED"}
    assert st["mantissa_bpw"] < 3.0
    assert syn["specials_exact"] is True


def test_all_raw_omits_huffman_table() -> None:
    rng = np.random.default_rng(1)
    n = 512
    mant = rng.integers(0, 128, size=n, dtype=np.uint8)
    exp = rng.integers(120, 130, size=n, dtype=np.uint8)
    enc = encode_mantissa(mant, exp)
    assert enc.strategy == STRAT_ALL_RAW
    assert enc.used_huffman_table is False
    assert enc.codebook_bytes == 0
    rec = decode_mantissa(enc.blob, exp=exp)
    assert np.array_equal(rec, mant)


def test_specials_no_fp_roundtrip() -> None:
    words = np.resize(special_payload_words(), 128)
    sign, exp, mant = split_components(words)
    # Rejoin without FP.
    back = join_components(sign, exp, mant)
    assert_exact(words, back, label="components")
    bundle = encode_bf16_bundle(words)
    rec = decode_bf16_bundle(bundle.blob)
    assert_exact(words, rec, label="specials_bundle")
    assert bundle.sha256 == sha256_words(words)


def test_tiny_qwen_fixture_sha(tmp_path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    specs = select_specs(model)
    from pbr_core.safetensors_io import load_uint16

    for spec in list(specs.values())[:4]:
        words = load_uint16(spec)
        bundle = encode_bf16_bundle(words)
        rec = decode_bf16_bundle(bundle.blob)
        assert_exact(words, rec, label=spec.name)
