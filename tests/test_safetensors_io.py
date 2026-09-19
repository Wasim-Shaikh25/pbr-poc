"""Safetensors uint16 I/O: exact views, no FP32 archival path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_core.bf16 import special_payload_words
from pbr_core.safetensors_io import (
    inventory_file,
    load_uint16,
    write_uint16_safetensors,
)
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.verification import assert_exact


def test_roundtrip_bf16_tagged_specials(tmp_path: Path) -> None:
    specials = special_payload_words().reshape(2, 4)
    path = tmp_path / "specials.safetensors"
    write_uint16_safetensors(path, {"layer.weight": specials}, storage_dtype="BF16")
    specs = inventory_file(path)
    assert len(specs) == 1
    assert specs[0].dtype == "BF16"
    assert specs[0].shape == (2, 4)
    loaded = load_uint16(specs[0])
    assert loaded.dtype == np.uint16
    assert_exact(specials, loaded, label="safetensors_specials")


def test_f16_tag_is_also_uint16_view(tmp_path: Path) -> None:
    words = np.arange(32, dtype=np.uint16).reshape(4, 8)
    path = tmp_path / "f16.safetensors"
    write_uint16_safetensors(path, {"proj.weight": words}, storage_dtype="F16")
    loaded = load_uint16(inventory_file(path)[0])
    assert_exact(words, loaded, label="f16_view")


def test_pbr_encode_tiny_real_shaped_slice(tmp_path: Path) -> None:
    # Qwen2-like skinny tile: hidden=64, intermediate=128 (scaled down).
    rng = np.random.default_rng(21)
    weight = rng.integers(0, 65536, size=(128, 64), dtype=np.uint16)
    path = tmp_path / "tiny_qwen_like.safetensors"
    write_uint16_safetensors(
        path,
        {"model.layers.0.mlp.down_proj.weight": weight},
        storage_dtype="BF16",
        metadata={"note": "fixture, not a real checkpoint"},
    )
    spec = inventory_file(path)[0]
    loaded = load_uint16(spec)
    container = encode_tensor(loaded, name=spec.name, block_size=64, extra={"stage": "1B"})
    restored = decode_container(container)[0]
    assert_exact(loaded, restored, label=spec.name)
    blob = container.dumps()
    assert len(blob) == len(blob)  # complete container is the authority
    assert container.extra.get("stage") == "1B"


def test_rejects_non_16bit_spec(tmp_path: Path) -> None:
    path = tmp_path / "ok.safetensors"
    write_uint16_safetensors(path, {"w": np.zeros((2, 2), dtype=np.uint16)})
    spec = inventory_file(path)[0]
    bad = spec.__class__(
        name=spec.name,
        dtype="F32",
        shape=spec.shape,
        data_offsets=spec.data_offsets,
        file_path=spec.file_path,
        data_start=spec.data_start,
    )
    with pytest.raises(TypeError, match="Refusing to archive"):
        load_uint16(bad)
