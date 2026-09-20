"""rANS exponent coder: roundtrip and size vs Huffman."""

from __future__ import annotations

import numpy as np

from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.rans import rans_decode, rans_encode, table_from_symbols
from pbr_core.types import TILE_HEADER_BYTES
from pbr_encoder.verification import assert_exact


def test_rans_roundtrip_random() -> None:
    rng = np.random.default_rng(9)
    symbols = rng.choice(np.arange(8, dtype=np.uint8), size=4096, p=[0.5, 0.2, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01])
    freq = table_from_symbols(symbols)
    blob = rans_encode(symbols, freq)
    rec = rans_decode(blob, int(symbols.size), freq)
    assert np.array_equal(rec, symbols)


def test_rans_single_symbol() -> None:
    symbols = np.full(128, 17, dtype=np.uint8)
    freq = table_from_symbols(symbols)
    blob = rans_encode(symbols, freq)
    rec = rans_decode(blob, 128, freq)
    assert np.array_equal(rec, symbols)


def test_exp_rans_codec_specials() -> None:
    words = np.resize(special_payload_words(), 64).reshape(8, 8)
    codec = Bf16ExpRansCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    assert_exact(words, codec.decode(enc), label="exp_rans_specials")


def test_exp_rans_not_worse_than_raw_on_few_exponents() -> None:
    rng = np.random.default_rng(3)
    n = 8192
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.choice(np.array([120, 121, 126, 127, 128, 129], dtype=np.uint16), size=n),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(64, 128)
    rans = Bf16ExpRansCodec().encode(words)
    huff = Bf16ExpHuffmanCodec().encode(words)
    assert rans is not None and huff is not None
    raw = TILE_HEADER_BYTES + n * 2
    assert TILE_HEADER_BYTES + len(rans.payload) < raw
    rans.rows, rans.cols = 64, 128
    assert_exact(words, Bf16ExpRansCodec().decode(rans), label="exp_rans_few")


def test_fused_c_tile_matches_python_decode() -> None:
    import os

    from pbr_core.rans import decode_exp_rans_tile_c, rans_impl

    rng = np.random.default_rng(11)
    n = 4096
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.choice(np.array([120, 126, 127, 128], dtype=np.uint16), size=n),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(32, 128)
    enc = Bf16ExpRansCodec().encode(words)
    assert enc is not None
    enc.rows, enc.cols = 32, 128
    fused = decode_exp_rans_tile_c(enc.payload, n)
    prev = os.environ.get("PBR_RANS_IMPL")
    os.environ["PBR_RANS_IMPL"] = "python"
    try:
        py = Bf16ExpRansCodec().decode(enc)
    finally:
        if prev is None:
            os.environ.pop("PBR_RANS_IMPL", None)
        else:
            os.environ["PBR_RANS_IMPL"] = prev
    if rans_impl() == "c":
        assert fused is not None
        assert_exact(words, fused.reshape(32, 128), label="fused_c")
    assert_exact(words, py, label="python_fallback")
