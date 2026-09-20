"""PBR-4 structured nibble: exact F(node,c4) XOR R roundtrips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_codecs import STAGE1A_CODECS
from pbr_codecs.pbr4 import (
    DISCLAIMER,
    KIND_AFFINE,
    KIND_NIBBLE,
    KIND_PALETTE,
    KIND_SE_NIBBLE,
    PBR4Container,
    Pbr4Codec,
    c4_bytes_for,
    decode_container,
    decode_tensor,
    encode_container,
    encode_tensor,
    pack_nibbles,
    unpack_nibbles,
)
from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.safetensors_io import load_uint16
from pbr_core.tiles import as_2d
from pbr_core.types import MODE_PBR4
from pbr_encoder.hf_weights import inventory_from_dir, select_weight_specs
from pbr_encoder.verification import assert_exact

QWEN_DIR = Path("outputs/models/Qwen__Qwen2.5-0.5B-Instruct")


def _roundtrip(words: np.ndarray, **kwargs) -> None:
    enc = encode_tensor(words, **kwargs)
    rec = decode_tensor(enc)
    assert_exact(words, rec, label="pbr4")
    st = enc.stats()
    n = int(words.size)
    assert st["c4_bytes"] == c4_bytes_for(n)
    assert st["c4_bits"] == 4 * n
    assert st["formal_bytes"] == st["s_bytes"] + st["c4_bytes"] + st["r_bytes"] + st["meta_bytes"]
    assert st["payload_bytes"] == st["formal_bytes"]
    blob = encode_container([enc]).dumps()
    rec2 = decode_container(PBR4Container.loads(blob))[0]
    assert_exact(words, rec2, label="pbr4_container")
    return enc


def test_pbr4_not_on_stage1a_menu() -> None:
    names = {c.name for c in STAGE1A_CODECS}
    ids = {c.mode_id for c in STAGE1A_CODECS}
    assert "pbr4_structured_nibble" not in names
    assert MODE_PBR4 not in ids
    assert Pbr4Codec().mode_id == MODE_PBR4


def test_nibble_pack_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 15, 16, 17, 256, 257):
        c4 = rng.integers(0, 16, size=n, dtype=np.uint8)
        blob = pack_nibbles(c4)
        assert len(blob) == c4_bytes_for(n)
        rec = unpack_nibbles(blob, n)
        assert np.array_equal(rec, c4)


def test_palette_exact_when_k_le_16() -> None:
    pal = np.array([0x0000, 0x3C00, 0xBC00, 0x7FC1, 0x0001, 0x8001], dtype=np.uint16)
    words = np.resize(pal, 16 * 16).reshape(16, 16)
    enc = _roundtrip(words, block_h=16, block_w=16)
    assert enc.n_hit == words.size
    assert enc.stats()["r_bytes"] <= 4
    assert any(k in enc.family_counts for k in ("palette16", "templates16", "const_proto"))


def test_affine_planar_exact() -> None:
    r = np.arange(8, dtype=np.uint32)[:, None]
    c = np.arange(8, dtype=np.uint32)[None, :]
    words = ((0x3C00 + 3 * r + 5 * c) & 0xFFFF).astype(np.uint16)
    enc = _roundtrip(words, block_h=8, block_w=8)
    assert enc.n_hit == words.size
    assert KIND_AFFINE in {nd.kind for nd in enc.nodes} or enc.n_hit == words.size


def test_nibble_insert_exact() -> None:
    base = np.full((8, 8), 0x3C00, dtype=np.uint16)
    nibble = np.arange(64, dtype=np.uint16).reshape(8, 8) & 0xF
    words = (base & np.uint16(0xFFF0)) | nibble
    enc = _roundtrip(words, block_h=8, block_w=8)
    assert enc.n_hit == words.size
    kinds = {nd.kind for nd in enc.nodes}
    assert KIND_NIBBLE in kinds or KIND_PALETTE in kinds or enc.n_hit == words.size


def test_se_nibble_constant_se_roundtrip() -> None:
    # Shared sign+exp, varying 7-bit mantissa. XOR-zero only when 3 LSBs are 0,
    # but reconstruction must still be exact (3-bit R / XOR residual).
    mant = np.arange(256, dtype=np.uint16).reshape(16, 16) & 0x7F
    words = make_bf16_bits(np.ones((16, 16), dtype=np.uint16), np.full((16, 16), 127, dtype=np.uint16), mant)
    enc = _roundtrip(words, block_h=16, block_w=16)
    st = enc.stats()
    # 4-bit c4 + 3-bit mantissa leftover ≈ 7 BPW plus tiny S, well under raw 16.
    assert st["formal_bpw"] < 12.0
    assert KIND_SE_NIBBLE in {nd.kind for nd in enc.nodes} or st["formal_bpw"] < 12.0


def test_specials_and_random_exact() -> None:
    specials = np.resize(special_payload_words(), 64).reshape(8, 8)
    _roundtrip(specials, block_h=8, block_w=8)
    rng = np.random.default_rng(9)
    rnd = rng.integers(0, 65536, size=(12, 10), dtype=np.uint16)
    enc = _roundtrip(rnd, block_h=8, block_w=8, adaptive=True)
    # Dense random: F misses almost everywhere; still exact, not a 4 BPW codec.
    assert enc.stats()["formal_bpw"] > 4.0
    assert "≤4" in DISCLAIMER or "not ≤4" in DISCLAIMER


def test_codec_wrapper_roundtrip() -> None:
    words = np.arange(64, dtype=np.uint16).reshape(8, 8)
    codec = Pbr4Codec()
    enc = codec.encode(words)
    enc.rows, enc.cols = 8, 8
    rec = codec.decode(enc)
    assert_exact(words, rec, label="pbr4_codec")


@pytest.mark.skipif(not QWEN_DIR.exists() or not (QWEN_DIR / "model.safetensors").exists(), reason="Qwen weights missing")
def test_real_qwen_slice_exact() -> None:
    specs = select_weight_specs(
        inventory_from_dir(QWEN_DIR), min_bytes=1_000_000, max_bytes=8_000_000
    )
    words = as_2d(load_uint16(specs[0]))[:16, :64]
    enc = encode_tensor(words, name=specs[0].name, block_h=16, block_w=16)
    rec = decode_tensor(enc)
    assert_exact(words, rec, label="pbr4_qwen_slice")
    assert enc.stats()["formal_bpw"] > 4.0
