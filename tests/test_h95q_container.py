"""Smoke tests for H95Q bitpack + container round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from pbr_h95.bitpack import pack_kbit, unpack_kbit, pack_kept_mantissas, unpack_kept_mantissas
from pbr_h95.container_h95q import (
    encode_container,
    decode_container,
    verify_decoded_against_reference,
)
from pbr_h95.quantize import quantize_bf16_mantissas, pack_kept_mantissas as q_pack


@pytest.mark.parametrize("k", [1, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 64, 1000])
def test_pack_kbit_roundtrip(k, n):
    rng = np.random.default_rng(k * 1000 + n)
    v = rng.integers(0, 1 << k, size=n, dtype=np.uint16)
    b = pack_kbit(v, k)
    assert len(b) == ((n * k + 7) // 8 if k and n else 0)
    assert np.array_equal(unpack_kbit(b, n, k), v)


def test_pack_kept_matches_quantize_api():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 128, size=200, dtype=np.uint16)
    for k in range(1, 8):
        assert pack_kept_mantissas(m, k) == q_pack(m, k)
        kept = unpack_kept_mantissas(pack_kept_mantissas(m, k), 200, k)
        expect = (m >> (7 - k)) & ((1 << k) - 1)
        assert np.array_equal(kept, expect)


def test_container_uniform_and_embed_roundtrip(tmp_path):
    rng = np.random.default_rng(42)

    def make(n):
        sign = rng.integers(0, 2, size=n, dtype=np.uint16)
        exp = rng.integers(1, 254, size=n, dtype=np.uint16)
        mant = rng.integers(0, 128, size=n, dtype=np.uint16)
        return ((sign << 15) | (exp << 7) | mant).astype(np.uint16)

    w1 = quantize_bf16_mantissas(make(128), 4).reshape(8, 16)
    w2 = quantize_bf16_mantissas(make(64), 5).reshape(8, 8)
    emb_raw = make(32 * 8).reshape(32, 8)
    row_keeps = np.array([7] * 4 + [4] * 12 + [3] * 16, dtype=np.int8)
    emb = emb_raw.copy()
    for r, k in enumerate(row_keeps.tolist()):
        emb[r] = quantize_bf16_mantissas(emb_raw[r], int(k))

    tensors = {"a.weight": w1, "b.weight": w2, "model.embed_tokens.weight": emb}
    keep_map = {"a.weight": 4, "b.weight": 5, "model.embed_tokens.weight": -1}
    path = tmp_path / "toy.h95q"
    stats = encode_container(
        path,
        tensors,
        keep_map=keep_map,
        model_id="toy",
        candidate="unit",
        embed_name="model.embed_tokens.weight",
        embed_row_keeps=row_keeps,
    )
    assert stats["n_weights"] == 128 + 64 + 32 * 8
    dec = decode_container(path)
    v = verify_decoded_against_reference(dec["tensors"], tensors)
    assert v["exact_match"] and v["sha_ok"]
    assert np.array_equal(quantize_bf16_mantissas(w1.ravel(), 4), w1.ravel())


def test_embed_tiers_nan_payload_0x7fc1_roundtrip(tmp_path):
    """NaN 0x7FC1 in a k<7 embed row must survive encode/decode (not collapse to 0x7FC0)."""
    rows, cols = 8, 4
    emb = np.zeros((rows, cols), dtype=np.uint16)
    # Finite quantized-looking values in most cells
    emb[:] = 0x3F80  # 1.0 bf16-ish pattern; exact value irrelevant
    # Inject dirty-NaN into a k=3 row (row 5)
    emb[5, 1] = np.uint16(0x7FC1)
    row_keeps = np.array([7, 7, 5, 5, 4, 3, 3, 7], dtype=np.int8)
    # Simulate quantized reference: preserve specials, quantize finites per row
    from pbr_h95.quantize import quantize_bf16_mantissas

    ref = emb.copy()
    for r, k in enumerate(row_keeps.tolist()):
        ref[r] = quantize_bf16_mantissas(emb[r], int(k))
    assert int(ref[5, 1]) == 0x7FC1  # quantize must keep NaN payload

    tensors = {"model.embed_tokens.weight": ref}
    keep_map = {"model.embed_tokens.weight": -1}
    path = tmp_path / "nan.h95q"
    encode_container(
        path,
        tensors,
        keep_map=keep_map,
        model_id="toy-nan",
        candidate="nan-fix",
        embed_name="model.embed_tokens.weight",
        embed_row_keeps=row_keeps,
    )
    dec = decode_container(path)
    got = dec["tensors"]["model.embed_tokens.weight"]
    assert int(got[5, 1]) == 0x7FC1, f"got 0x{int(got[5, 1]):04X}"
    v = verify_decoded_against_reference(dec["tensors"], tensors)
    assert v["exact_match"] and v["sha_ok"]
    # Ensure an exception was actually recorded (regression guard)
    emb_spec = next(t for t in dec["header"]["tensors"] if t["name"] == "model.embed_tokens.weight")
    assert emb_spec.get("exceptions", {}).get("n", 0) >= 1
