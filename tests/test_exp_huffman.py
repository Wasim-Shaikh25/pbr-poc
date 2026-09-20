"""PBR-E: Huffman tables, BF16 field pack, exact exp-Huffman codec."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_core.bf16 import (
    join_components,
    make_bf16_bits,
    pack_sign_mantissa,
    special_payload_words,
    split_components,
    unpack_sign_mantissa,
)
from pbr_core.huffman import dump_table, load_table, table_from_symbols
from pbr_core.metrics import bits_per_weight
from pbr_core.types import EncodeContext
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.verification import assert_exact


def test_split_pack_is_identity() -> None:
    words = make_bf16_bits(
        [0, 1, 1, 0],
        [127, 0, 255, 120],
        [0, 1, 0x41, 0x7F],
    )
    sign, exp, mant = split_components(words)
    packed = pack_sign_mantissa(sign, mant)
    s2, m2 = unpack_sign_mantissa(packed)
    assert np.array_equal(s2, sign)
    assert np.array_equal(m2, mant)
    assert np.array_equal(join_components(s2, exp, m2), words)


@pytest.mark.parametrize("n", [1, 2, 17, 256, 1024])
def test_huffman_roundtrip_random(n: int) -> None:
    rng = np.random.default_rng(20 + n)
    symbols = rng.integers(0, 32, size=n, dtype=np.uint8)
    table = table_from_symbols(symbols)
    blob = table.encode_symbols(symbols)
    restored = table.decode_symbols(blob, n)
    assert np.array_equal(restored, symbols)
    dumped = dump_table(table)
    loaded, end = load_table(dumped, 0)
    assert end == len(dumped)
    assert np.array_equal(loaded.decode_symbols(blob, n), symbols)


def test_huffman_single_symbol_is_empty_stream() -> None:
    symbols = np.full(64, 127, dtype=np.uint8)
    table = table_from_symbols(symbols)
    assert table.encode_symbols(symbols) == b""
    assert np.array_equal(table.decode_symbols(b"", 64), symbols)


def test_exp_huffman_codec_specials() -> None:
    words = np.resize(special_payload_words(), 64).reshape(8, 8)
    codec = Bf16ExpHuffmanCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    assert_exact(words, codec.decode(enc), label="exp_huffman_specials")


def test_exp_huffman_few_exponents_near_df11() -> None:
    rng = np.random.default_rng(3)
    n = 8192
    signs = rng.integers(0, 2, size=n, dtype=np.uint16)
    exponents = rng.choice(np.array([120, 121, 126, 127, 128, 129], dtype=np.uint16), size=n)
    mantissa = rng.integers(0, 128, size=n, dtype=np.uint16)
    words = make_bf16_bits(signs, exponents, mantissa).reshape(64, 128)
    container = encode_tensor(words, name="few_exp", block_size=256)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="few_exp")
    bpw = bits_per_weight(len(container.dumps()), words.size)
    # 8 (sign+mant) + ~2.6 exp + codebook ≈ 11; allow container slack.
    assert bpw < 12.0
    names = {t.mode_name for t in container.tensors[0].tiles}
    assert any(n.startswith("bf16_exp_huffman") for n in names)


def test_exp_huffman_random_does_not_beat_raw() -> None:
    rng = np.random.default_rng(9)
    words = rng.integers(0, 65536, size=(16, 16), dtype=np.uint16)
    enc = Bf16ExpHuffmanCodec().encode(words, EncodeContext())
    assert enc is None


def test_exp_huffman_qwen_like_slice_exact() -> None:
    """Qwen-like exponent mix (~2.6 bit entropy); must be bit-exact and beat 13.6."""
    rng = np.random.default_rng(25)
    n = 4096
    exponents = rng.choice(
        np.array([119, 120, 121, 126, 127, 128, 129, 130], dtype=np.uint16),
        size=n,
        p=[0.03, 0.10, 0.14, 0.20, 0.24, 0.16, 0.10, 0.03],
    )
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        exponents,
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(32, 128)
    container = encode_tensor(words, name="qwen_like", block_size=256)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="qwen_like_slice")
    bpw = bits_per_weight(len(container.dumps()), words.size)
    assert bpw < 13.6
    names = {t.mode_name for t in container.tensors[0].tiles}
    assert any(name.startswith("bf16_exp_huffman") for name in names)


@pytest.mark.skipif(os.environ.get("PBR_LIVE_HF") != "1", reason="live Hugging Face download disabled")
def test_exp_huffman_live_qwen_slice() -> None:
    from pathlib import Path

    from pbr_core.safetensors_io import inventory_model, is_linear_weight, load_uint16_region

    model_dir = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
    if not model_dir.exists():
        pytest.skip("cached Qwen checkpoint not present")
    specs = [s for s in inventory_model(model_dir) if is_linear_weight(s.name) and s.ndim == 2]
    assert specs
    spec = max(specs, key=lambda s: s.nbytes)
    words = load_uint16_region(spec, 0, 0, min(32, spec.shape[0]), min(128, spec.shape[1]))
    container = encode_tensor(words, name="qwen_live_slice", block_size=256)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="qwen_live_slice")
