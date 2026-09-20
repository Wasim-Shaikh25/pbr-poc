"""Property-style and boundary round-trips for the Stage 1A encoder."""

from __future__ import annotations

import numpy as np
import pytest

from pbr_core.container import PBRContainer
from pbr_core.hashing import sha256_words
from pbr_encoder.controlled_data import generate_case
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor, encode_words
from pbr_encoder.verification import assert_exact


@pytest.mark.parametrize("n", [1, 2, 17, 255, 256, 257, 1024])
def test_random_lengths_roundtrip(n: int) -> None:
    rng = np.random.default_rng(1000 + n)
    words = rng.integers(0, 65536, size=(1, n), dtype=np.uint16)
    container = encode_tensor(words, name=f"n{n}", block_size=64)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label=f"random_len_{n}")


def test_empty_tensor_roundtrip() -> None:
    words = np.zeros((0, 0), dtype=np.uint16)
    container = encode_tensor(words, name="empty", block_size=64)
    restored = decode_container(container)[0]
    assert restored.shape == (0, 0)
    assert restored.dtype == np.uint16


def test_one_word_tensor() -> None:
    words = np.array([[0x7FC1]], dtype=np.uint16)
    container = encode_tensor(words, name="one", block_size=256)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="one_word")


def test_non_multiple_block_length() -> None:
    rng = np.random.default_rng(3)
    words = rng.integers(0, 65536, size=(17, 13), dtype=np.uint16)
    container = encode_tensor(words, name="ragged", block_size=64)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="ragged")


def test_encode_words_1d_view() -> None:
    words = np.arange(128, dtype=np.uint16)
    container = encode_words(words, name="flat", block_size=32)
    restored = decode_container(container)[0]
    assert_exact(words.reshape(1, -1), restored, label="flat")


def test_determinism() -> None:
    words, _ = generate_case("repeated_residuals", n_words=2048, cols=64, seed=7)
    a = encode_tensor(words, name="det", block_size=256).dumps()
    b = encode_tensor(words, name="det", block_size=256).dumps()
    assert a == b
    loaded = PBRContainer.loads(a)
    assert loaded.tensors[0].sha256 == sha256_words(words)
    restored = decode_container(loaded)[0]
    assert_exact(words, restored, label="determinism")


@pytest.mark.parametrize("block_size", [64, 128, 256])
def test_multiple_block_sizes(block_size: int) -> None:
    words, _ = generate_case("previous_row", n_words=4096, cols=64, seed=7)
    container = encode_tensor(words, name="blocks", block_size=block_size)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label=f"block_{block_size}")
