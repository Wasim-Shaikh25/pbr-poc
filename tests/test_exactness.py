"""Exactness tests must fail loudly. Decode(Encode(W)) == W bit-for-bit."""

from __future__ import annotations

import numpy as np
import pytest

from pbr_core.bf16 import make_bf16_bits, special_payload_words, view_uint16
from pbr_core.hashing import sha256_words
from pbr_encoder.controlled_data import CASES, generate_case
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.verification import ExactnessError, assert_exact, verify_words


def test_uint16_view_rejects_fp32_archival_path() -> None:
    with pytest.raises(TypeError, match="Refusing to convert"):
        view_uint16(np.array([1.0, 2.0], dtype=np.float32))


def test_special_bf16_payloads_survive_roundtrip() -> None:
    specials = special_payload_words()
    # Build a 8x8 tile that includes every special pattern.
    filler = np.resize(specials, 64)
    words = filler.reshape(8, 8)
    container = encode_tensor(words, name="specials", block_size=64)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="special_payloads")
    # These exact bit patterns are the ones an FP32 detour would be likely to lose.
    for value in (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 0x0001, 0x8001):
        assert value in set(int(x) for x in restored.ravel())


def test_component_join_is_bit_identity() -> None:
    signs = np.array([0, 1, 1, 0], dtype=np.uint16)
    exp = np.array([127, 0, 255, 1], dtype=np.uint16)
    mant = np.array([0, 1, 0x41, 0x7F], dtype=np.uint16)
    words = make_bf16_bits(signs, exp, mant)
    assert words.dtype == np.uint16
    assert words.tolist() == [
        (0 << 15) | (127 << 7) | 0,
        (1 << 15) | (0 << 7) | 1,
        (1 << 15) | (255 << 7) | 0x41,
        (0 << 15) | (1 << 7) | 0x7F,
    ]


def test_assert_exact_fails_loudly_on_mismatch() -> None:
    left = np.array([[0x3E00, 0x0001]], dtype=np.uint16)
    right = np.array([[0x3E00, 0x0002]], dtype=np.uint16)
    with pytest.raises(ExactnessError, match="EXACTNESS FAIL"):
        assert_exact(left, right, label="mismatch")
    result = verify_words(left, right)
    assert result.exact is False
    assert result.differing_words == 1
    assert result.original_sha256 != result.restored_sha256


@pytest.mark.parametrize("case", CASES)
def test_every_controlled_case_is_bit_exact(case: str, small_params: dict) -> None:
    words, manifest = generate_case(case, **{k: small_params[k] for k in ("n_words", "cols", "seed")})
    container = encode_tensor(words, name=case, block_size=small_params["block_size"])
    restored = decode_container(container)[0]
    result = assert_exact(words, restored, label=case)
    assert result.original_sha256 == manifest["sha256"]
    assert result.original_sha256 == sha256_words(restored)
    assert result.differing_words == 0
