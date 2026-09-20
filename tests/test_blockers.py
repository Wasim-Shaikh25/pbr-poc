"""Exponent-spatial / hier / cross-layer exactness and diagnosis fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pbr_codecs.cross_layer_tile_xor import CrossLayerTileXorCodec
from pbr_codecs.exp_hier_residual import ExpHierResidualCodec
from pbr_codecs.exp_spatial_huffman import ExpSpatialHuffmanCodec
from pbr_core.bf16 import make_bf16_bits
from pbr_core.metrics import bits_per_weight
from pbr_core.safetensors_io import tensor_role
from pbr_core.types import EncodeContext
from pbr_encoder.blocker_diagnosis import diagnose_tensor
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.profiles import UINT16_SPATIAL_CODECS
from pbr_encoder.verification import assert_exact


def _layer_like(n: int = 2048, *, seed: int = 4, row_exp: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if row_exp:
        rows, cols = 16, n // 16
        base = rng.choice(np.array([124, 125, 126, 127, 128], dtype=np.uint16), size=rows)
        exponents = np.repeat(base, cols)
    else:
        rows, cols = 16, n // 16
        exponents = rng.choice(
            np.array([120, 121, 126, 127, 128, 129], dtype=np.uint16), size=n
        )
    words = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        exponents,
        rng.integers(0, 128, size=n, dtype=np.uint16),
    )
    return words.reshape(rows, cols)


def test_tensor_role_strips_layer_index() -> None:
    assert tensor_role("model.layers.12.mlp.down_proj.weight") == "model.mlp.down_proj.weight"
    assert tensor_role("embed_tokens.weight") == "embed_tokens.weight"


def test_exp_spatial_row_structure_is_exact() -> None:
    words = _layer_like(row_exp=True)
    codec = ExpSpatialHuffmanCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = words.shape
    assert_exact(words, codec.decode(enc), label="exp_spatial_rows")


def test_exp_hier_block_structure_is_exact() -> None:
    rng = np.random.default_rng(8)
    rows, cols = 32, 32
    exp = np.zeros((rows, cols), dtype=np.uint16)
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            exp[i : i + 8, j : j + 8] = 120 + ((i + j) % 7)
    words = make_bf16_bits(
        rng.integers(0, 2, size=rows * cols, dtype=np.uint16),
        exp.ravel(),
        rng.integers(0, 128, size=rows * cols, dtype=np.uint16),
    ).reshape(rows, cols)
    codec = ExpHierResidualCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = rows, cols
    assert_exact(words, codec.decode(enc), label="exp_hier_blocks")
    assert enc.mode_name.startswith("exp_hier_residual")


def test_cross_layer_exp_xor_is_exact() -> None:
    rng = np.random.default_rng(2)
    n = 1024
    exp = rng.choice(np.array([126, 127, 128], dtype=np.uint16), size=n)
    a = make_bf16_bits(rng.integers(0, 2, n, dtype=np.uint16), exp, rng.integers(0, 128, n, dtype=np.uint16))
    b = make_bf16_bits(rng.integers(0, 2, n, dtype=np.uint16), exp, rng.integers(0, 128, n, dtype=np.uint16))
    a = a.reshape(16, 64)
    b = b.reshape(16, 64)
    ctx = EncodeContext(ncols=64, ref_tensor=a)
    enc = CrossLayerTileXorCodec().encode(b, ctx)
    assert enc is not None
    enc.rows, enc.cols = 16, 64
    assert_exact(b, CrossLayerTileXorCodec().decode(enc, ctx), label="cross_layer")


def test_encode_tensor_new_modes_roundtrip() -> None:
    words = _layer_like(n=4096, seed=11, row_exp=True)
    container = encode_tensor(words, name="spatial_win", block_size=256)
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="new_modes_default")
    bpw = bits_per_weight(len(container.dumps()), words.size)
    assert bpw < 13.6


def test_uint16_spatial_profile_is_exact_but_not_required_to_win() -> None:
    words = _layer_like(n=2048, seed=13)
    container = encode_tensor(
        words,
        name="uint16_only",
        block_size=256,
        codecs=UINT16_SPATIAL_CODECS,
        whole_codecs=[],
        enable_whole=False,
    )
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="uint16_spatial")


def test_diagnosis_fixture_has_expected_keys() -> None:
    words = _layer_like(n=1024, seed=1)
    row = diagnose_tensor("model.layers.0.mlp.down_proj.weight", words)
    assert row["entropy"]["mantissa_bits"] > 4.0
    assert "uint16_prev_value" in row["residual_entropy"]
    assert "256" in row["duplicate_tiles"]
    assert row["role"] == "model.mlp.down_proj.weight"


@pytest.mark.skipif(os.environ.get("PBR_LIVE_HF") != "1", reason="live Hugging Face download disabled")
def test_diagnosis_live_qwen_optional() -> None:
    from pbr_encoder.blocker_diagnosis import main

    model = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")
    if not model.exists():
        pytest.skip("cached Qwen checkpoint not present")
    assert (
        main(
            [
                "--model-dir",
                str(model),
                "--output-json",
                "outputs/reports/blocker_diag_live.json",
                "--output-md",
                "outputs/reports/blocker_diag_live.md",
            ]
        )
        == 0
    )
