"""Container integrity: complete-byte accounting and parse checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_core.container import MAGIC, PBRContainer
from pbr_core.hashing import sha256_bytes
from pbr_core.metrics import bits_per_weight
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor
from pbr_encoder.verification import assert_exact


def test_reported_size_equals_on_disk(tmp_path: Path) -> None:
    words = np.full((32, 32), 0x3C00, dtype=np.uint16)
    container = encode_tensor(words, name="disk", block_size=64)
    blob = container.dumps()
    path = tmp_path / "disk.pbr"
    path.write_bytes(blob)
    assert path.stat().st_size == len(blob)
    loaded = PBRContainer.loads(path.read_bytes())
    restored = decode_container(loaded)[0]
    assert_exact(words, restored, label="on_disk")
    assert loaded.tensors[0].n_words == words.size
    bpw = bits_per_weight(len(blob), words.size)
    assert bpw < 16
    assert sha256_bytes(blob) == sha256_bytes(path.read_bytes())


def test_bad_magic_rejected() -> None:
    words = np.arange(16, dtype=np.uint16).reshape(4, 4)
    blob = bytearray(encode_tensor(words, name="magic", block_size=16).dumps())
    blob[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="Bad magic"):
        PBRContainer.loads(bytes(blob))


def test_truncated_payload_rejected() -> None:
    words = np.arange(16, dtype=np.uint16).reshape(4, 4)
    blob = encode_tensor(words, name="trunc", block_size=16).dumps()
    with pytest.raises(ValueError):
        PBRContainer.loads(blob[:-3])


def test_container_starts_with_magic() -> None:
    words = np.zeros((4, 4), dtype=np.uint16)
    blob = encode_tensor(words, name="hdr", block_size=16).dumps()
    assert blob.startswith(MAGIC)
    header = PBRContainer.loads(blob)
    assert header.extra.get("stage") == "1A"
    assert header.extra.get("selection") == "complete_encoded_bytes"
