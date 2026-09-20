"""Bit-planes, residual grammar, transformed refs, position-value dict."""

from __future__ import annotations

import numpy as np

from pbr_codecs.bit_planes import BitPlanesCodec
from pbr_codecs.grammar import ResidualGrammarCodec
from pbr_codecs.position_value_dict import PositionValueDictCodec
from pbr_codecs.transformed_ref import TransformedRefCodec
from pbr_codecs.transforms import XF_SIGN_XOR, XF_TRANSPOSE, apply_transform
from pbr_core.bf16 import make_bf16_bits
from pbr_core.types import EncodeContext
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.verification import assert_exact


def test_bit_planes_constant_high_bits() -> None:
    # Low 8 bits vary; high 8 bits are zero — planes 8–15 are all-zero.
    words = np.arange(64, dtype=np.uint16).reshape(8, 8)
    codec = BitPlanesCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    assert_exact(words, codec.decode(enc), label="bit_planes")


def test_bit_planes_rejects_incompressible() -> None:
    rng = np.random.default_rng(3)
    words = rng.integers(0, 65536, size=(16, 16), dtype=np.uint16)
    assert BitPlanesCodec().encode(words) is None


def test_grammar_repeated_residual_phrase() -> None:
    # Repeating residual pattern 0,1,2,3 after prev_value XOR.
    phrase = np.array([0, 1, 2, 3], dtype=np.uint16)
    residuals = np.tile(phrase, 32)
    words = np.bitwise_xor.accumulate(residuals).reshape(8, 16)
    codec = ResidualGrammarCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 8, 16
    assert_exact(words, codec.decode(enc), label="grammar")


def test_transformed_ref_sign_xor_exact() -> None:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 65536, size=(8, 8), dtype=np.uint16)
    flipped = apply_transform(base, XF_SIGN_XOR)
    assert flipped is not None
    ctx = EncodeContext(tile_index=0)
    ctx.record(base)
    ctx.tile_index = 1
    enc = TransformedRefCodec().encode(flipped, ctx)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    from pbr_encoder.decoder import decode_tile

    restored = decode_tile(enc, ctx, [base])
    assert_exact(flipped, restored, label="xform_sign")


def test_transformed_ref_transpose() -> None:
    rng = np.random.default_rng(2)
    base = rng.integers(0, 65536, size=(8, 8), dtype=np.uint16)
    tr = apply_transform(base, XF_TRANSPOSE)
    assert tr is not None
    ctx = EncodeContext(tile_index=0)
    ctx.record(base)
    ctx.tile_index = 1
    enc = TransformedRefCodec().encode(tr, ctx)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    from pbr_encoder.decoder import decode_tile

    restored = decode_tile(enc, ctx, [base])
    assert_exact(tr, restored, label="xform_transpose")


def test_position_value_dict_sparse_exceptions() -> None:
    words = np.full((16, 16), 0x3C00, dtype=np.uint16)
    words[0, 0] = 0x3C01
    words[3, 4] = 0x3C02
    words[7, 7] = 0x3C01
    codec = PositionValueDictCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 16, 16
    assert_exact(words, codec.decode(enc), label="pos_value")


def test_hierarchical_modes_in_encode_tensor_roundtrip() -> None:
    rng = np.random.default_rng(9)
    exp = rng.choice(np.array([126, 127, 128], dtype=np.uint16), size=1024)
    words = make_bf16_bits(
        rng.integers(0, 2, 1024, dtype=np.uint16),
        exp,
        rng.integers(0, 128, 1024, dtype=np.uint16),
    ).reshape(32, 32)
    container = encode_tensor(words, name="hier_mix", block_size=64)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="hier_mix")
