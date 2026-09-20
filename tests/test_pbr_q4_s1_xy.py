"""S1 selective X/Y: packed default, X/Y only when strictly smaller, exact roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_h95.bitpack import pack_kbit, packed_bytes_for_k
from pbr_h95.container_h95q import sha256_state_u16
from pbr_h95.quantize import quantize_bf16_mantissas
from pbr_q4.codecs import (
    encode_tile,
    pack_kbit_batch,
    parse_flag_byte,
    reconstruct_stack,
    unpack_kbit_batch,
)
from pbr_q4.const import (
    FROZEN_S1_SHA,
    MAP_BITMAP,
    MAP_RLE,
    MAP_SPARSE,
    MATRIX_FAMILY_MODES,
    MODE_MATRIX,
    MODE_PACKED,
    PRED_AVG,
    PRED_LEFT,
    PRED_PAETH,
    PRED_PREVIOUS,
    PRED_UP,
    TRAV_COL,
    TRAV_COL_SERP,
    TRAV_ROW,
    TRAV_ROW_SERP,
)
from pbr_q4.container import decode_container, encode_container, verify_decoded_against_reference
from pbr_q4.predictors import batched_residual_maps, residual_tile, reconstruct_tile
from pbr_q4.selective import (
    decode_array_selective,
    decode_tile_map,
    encode_array_selective,
    encode_tile_map,
)

S1_V2 = Path("artifacts/pbr_h95/containers/H95Q-S1-v2.h95q")
S1_V1 = Path("artifacts/pbr_h95/containers/H95Q-S1.h95q")


def _rand_codes(h, w, nbits, rng):
    return rng.integers(0, 1 << nbits, size=(h, w), dtype=np.uint16)


def _bf16_like(n, rng):
    sign = rng.integers(0, 2, size=n, dtype=np.uint16)
    exp = rng.integers(1, 254, size=n, dtype=np.uint16)
    mant = rng.integers(0, 128, size=n, dtype=np.uint16)
    return ((sign << 15) | (exp << 7) | mant).astype(np.uint16)


@pytest.mark.parametrize("nbits", [3, 4, 5, 7])
@pytest.mark.parametrize(
    "pred", [PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH]
)
@pytest.mark.parametrize(
    "trav", [TRAV_ROW, TRAV_ROW_SERP, TRAV_COL, TRAV_COL_SERP]
)
def test_predictor_residual_invertible(nbits, pred, trav):
    rng = np.random.default_rng(nbits * 100 + pred * 10 + trav)
    tile = _rand_codes(16, 16, nbits, rng)
    res = residual_tile(tile, pred=pred, trav=trav, nbits=nbits)
    rec = reconstruct_tile(res, pred=pred, trav=trav, nbits=nbits)
    assert np.array_equal(rec, tile.astype(np.uint8))


@pytest.mark.parametrize("shape", [(1, 16), (16, 1), (7, 9), (16, 16), (3, 3)])
def test_partial_and_odd_tiles_exact(shape):
    rng = np.random.default_rng(shape[0] * 50 + shape[1])
    tile = _rand_codes(*shape, 4, rng)
    enc = encode_tile(tile, 4)
    assert enc.mode in MATRIX_FAMILY_MODES
    from pbr_q4.codecs import decode_tile

    rec, used = decode_tile(enc.flag_byte(), enc.payload, rows=shape[0], cols=shape[1], nbits=4)
    assert used == len(enc.payload)
    assert np.array_equal(rec, tile.astype(np.uint8))


@pytest.mark.parametrize("nbits", [3, 4, 5, 7])
def test_pack_kbit_batch_matches_scalar(nbits):
    rng = np.random.default_rng(nbits + 9)
    tiles = rng.integers(0, 1 << nbits, size=(4, 16, 16), dtype=np.uint16)
    batched = pack_kbit_batch(tiles, nbits)
    unpacked = unpack_kbit_batch(batched, 256, nbits)
    mask = (1 << nbits) - 1
    for i in range(4):
        assert batched[i].tobytes() == pack_kbit(tiles[i], nbits)
        expect = tiles[i].ravel() & np.uint16(mask)
        assert np.array_equal(unpacked[i], expect)


@pytest.mark.parametrize("nbits", [3, 4, 7])
@pytest.mark.parametrize(
    "pred", [PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH]
)
@pytest.mark.parametrize("trav", [TRAV_ROW, TRAV_ROW_SERP, TRAV_COL, TRAV_COL_SERP])
def test_reconstruct_stack_matches_tile(nbits, pred, trav):
    rng = np.random.default_rng(nbits * 50 + pred * 7 + trav)
    tiles = rng.integers(0, 1 << nbits, size=(3, 16, 16), dtype=np.uint16)
    res = np.stack(
        [residual_tile(tiles[i], pred=pred, trav=trav, nbits=nbits) for i in range(3)]
    )
    rec = reconstruct_stack(res, pred=pred, trav=trav, nbits=nbits)
    for i in range(3):
        assert np.array_equal(
            rec[i], reconstruct_tile(res[i], pred=pred, trav=trav, nbits=nbits)
        )
        assert np.array_equal(rec[i], tiles[i].astype(np.uint8))


def test_batched_row_col_matches_sequential():
    rng = np.random.default_rng(42)
    tiles = rng.integers(0, 16, size=(5, 16, 16), dtype=np.uint16)
    maps = {(tr, pr): res for tr, pr, res in batched_residual_maps(tiles, 4)}
    for i in range(5):
        for trav in (TRAV_ROW, TRAV_COL):
            for pred in (PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH):
                seq = residual_tile(tiles[i], pred=pred, trav=trav, nbits=4)
                assert np.array_equal(maps[(trav, pred)][i], seq), (trav, pred, i)


@pytest.mark.parametrize("n_tiles,xy", [
    (8, []),
    (8, [0]),
    (8, [0, 7]),
    (8, [1, 2, 3]),
    (64, list(range(0, 64, 5))),
    (300, [0, 10, 299]),
    (70_000, [0, 1000, 69_999]),
])
def test_tile_map_roundtrip_picks_valid_kind(n_tiles, xy):
    blob = encode_tile_map(n_tiles, xy)
    got, used = decode_tile_map(blob, n_tiles)
    assert used == len(blob)
    assert got == sorted(xy)
    assert blob[0] in (MAP_SPARSE, MAP_BITMAP, MAP_RLE)


def test_random_array_stays_packed_when_xy_does_not_win():
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 16, size=(32, 48), dtype=np.uint16)
    packed = pack_kbit(arr.ravel(), 4)
    enc = encode_array_selective(arr, 4)
    rec = decode_array_selective(enc["blob"], rows=32, cols=48, nbits=4)
    assert np.array_equal(rec, arr)
    assert enc["blob_bytes"] <= len(packed)
    assert enc["padded"] is False
    # Unstructured noise: packed should win (no XY, or hybrid strictly smaller).
    if enc["kind"] == "packed":
        assert enc["blob"] == packed
        assert enc["n_xy_tiles"] == 0
        assert enc["fallback_packed"] is True
    else:
        assert enc["n_xy_tiles"] > 0
        assert len(enc["blob"]) < len(packed)


def test_constant_array_selects_xy_and_is_strictly_smaller():
    arr = np.zeros((32, 32), dtype=np.uint16)
    packed = pack_kbit(arr.ravel(), 4)
    enc = encode_array_selective(arr, 4)
    rec = decode_array_selective(enc["blob"], rows=32, cols=32, nbits=4)
    assert np.array_equal(rec, arr)
    assert enc["kind"] == "xy_sel"
    assert enc["n_xy_tiles"] > 0
    assert enc["n_xy_tiles"] + enc["n_packed_tiles"] == enc["n_tiles"]
    assert len(enc["blob"]) < len(packed)
    assert enc["saved_vs_packed"] == len(packed) - len(enc["blob"])
    # Complete cost includes map + flags; still beats packed.
    assert enc["map_bytes"] + enc["flag_bytes"] + enc["xy_payload_bytes"] + enc["packed_stream_bytes"] == len(enc["blob"])
    assert MODE_PACKED not in (enc["mode_hist"] or {})


def test_selective_mixed_tiles_roundtrip():
    rng = np.random.default_rng(9)
    arr = rng.integers(0, 8, size=(48, 32), dtype=np.uint16)
    arr[:16, :16] = 3  # compressible block
    packed = pack_kbit(arr.ravel(), 3)
    enc = encode_array_selective(arr, 3)
    rec = decode_array_selective(enc["blob"], rows=48, cols=32, nbits=3)
    assert np.array_equal(rec, arr)
    assert len(enc["blob"]) <= len(packed)
    if enc["n_xy_tiles"]:
        assert enc["kind"] == "xy_sel"
        assert len(enc["blob"]) < len(packed)
        for name, n in enc["mode_hist"].items():
            if n:
                assert name.startswith("XY_")
                assert name != "XY_MATRIX"  # MATRIX ties packed-K, never strictly smaller


def test_ragged_last_tile_not_padded():
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 8, size=(17, 20), dtype=np.uint16)
    enc = encode_array_selective(arr, 3)
    rec = decode_array_selective(enc["blob"], rows=17, cols=20, nbits=3)
    assert np.array_equal(rec, arr)
    assert enc["padded"] is False
    assert enc["rows"] == 17 and enc["cols"] == 20
    # 17x20 → 4 tiles of true shapes, not a 32x32 pad.
    assert enc["n_tiles"] == 4
    packed = pack_kbit(arr.ravel(), 3)
    assert enc["packed_whole_bytes"] == len(packed)
    assert len(enc["blob"]) <= len(packed)


def test_selective_never_emits_xy_when_not_strictly_smaller():
    """Per-tile: XY cost (payload+flag) must beat packed of the true tile."""
    tile = _rand_codes(16, 16, 4, np.random.default_rng(0))
    packed_len = packed_bytes_for_k(256, 4)
    enc = encode_tile(tile, 4)
    # MATRIX residuals pack at K bits = packed-K, so +flag cannot win.
    if enc.mode == MODE_MATRIX:
        assert 1 + len(enc.payload) >= packed_len
    sel = encode_array_selective(tile, 4)
    rec = decode_array_selective(sel["blob"], rows=16, cols=16, nbits=4)
    assert np.array_equal(rec, tile)
    if sel["kind"] == "xy_sel":
        assert 1 + sel["xy_payload_bytes"] + sel["map_bytes"] + sel["flag_bytes"] < packed_len or sel["packed_stream_bytes"] < packed_len
        assert len(sel["blob"]) < packed_len


def test_container_roundtrip_s1_like(tmp_path):
    rng = np.random.default_rng(1)
    w4 = quantize_bf16_mantissas(_bf16_like(16 * 16, rng), 4).reshape(16, 16)
    w3 = quantize_bf16_mantissas(_bf16_like(32 * 16, rng), 3).reshape(32, 16)
    w7 = quantize_bf16_mantissas(_bf16_like(16, rng), 7).reshape(16)
    emb_raw = _bf16_like(32 * 16, rng).reshape(32, 16)
    row_keeps = np.array([7] * 4 + [4] * 12 + [3] * 16, dtype=np.int8)
    emb = emb_raw.copy()
    for r, k in enumerate(row_keeps.tolist()):
        emb[r] = quantize_bf16_mantissas(emb_raw[r], int(k))

    # Mix: one highly compressible tensor should take XY; noise stays packed.
    w3[:] = 1

    tensors = {
        "layers.8.mlp.down_proj.weight": w3,
        "layers.3.self_attn.q_proj.weight": w4,
        "model.norm.weight": w7,
        "model.embed_tokens.weight": emb,
    }
    keep_map = {
        "layers.8.mlp.down_proj.weight": 3,
        "layers.3.self_attn.q_proj.weight": 4,
        "model.norm.weight": 7,
        "model.embed_tokens.weight": -1,
    }
    path = tmp_path / "toy.h95x"
    stats = encode_container(
        path,
        tensors,
        keep_map=keep_map,
        model_id="toy",
        candidate="toy-S1+XY-sel",
        embed_name="model.embed_tokens.weight",
        embed_row_keeps=row_keeps,
        expect_ref_sha=None,
    )
    assert stats["product_default_packed"] is True
    assert stats["all_tiles_matrix_family"] is False
    assert stats["n_tiles"] > 0
    assert stats["n_xy_tiles"] >= 1
    assert stats["saved_vs_packed_mant_bytes"] >= 0
    dec = decode_container(path)
    v = verify_decoded_against_reference(dec["tensors"], tensors)
    assert v["exact_match"] and v["sha_ok"]
    assert dec["header"]["mantissa_packing"] == "kbit_or_xysel"
    assert dec["header"]["xy_default"] == "packed"
    assert dec["header"]["packed_k_is_product_default"] is True
    assert path.stat().st_size == stats["file_bytes"]
    assert dec["header"]["sha256_quantized_reference"] == sha256_state_u16(tensors)


def test_packed_only_mantissa_matches_kbit(tmp_path):
    rng = np.random.default_rng(4)
    w = quantize_bf16_mantissas(_bf16_like(256, rng), 4).reshape(16, 16)
    tensors = {"a.weight": w}
    path = tmp_path / "p.h95x"
    stats = encode_container(
        path, tensors, keep_map={"a.weight": 4}, model_id="toy", expect_ref_sha=None
    )
    # Random 16x16 rarely beats packed after map/flag; if it stays packed, blob == pack_kbit.
    from pbr_core.bf16 import split_components
    from pbr_q4.fields import kept_from_mant

    _s, _e, mant = split_components(w.ravel())
    kept = kept_from_mant(mant, 4)
    packed = pack_kbit(kept, 4)
    dec = decode_container(path)
    v = verify_decoded_against_reference(dec["tensors"], tensors)
    assert v["exact_match"]
    if stats["n_xy_tiles"] == 0:
        header = dec["header"]
        spec = header["tensors"][0]
        blob = path.read_bytes()[spec["mant_off"] : spec["mant_off"] + spec["mant_len"]]
        assert blob == packed


def test_sha_guard_rejects_drift(tmp_path):
    rng = np.random.default_rng(2)
    w = quantize_bf16_mantissas(_bf16_like(256, rng), 4).reshape(16, 16)
    tensors = {"a.weight": w}
    with pytest.raises(RuntimeError, match="SHA drift"):
        encode_container(
            tmp_path / "x.h95x",
            tensors,
            keep_map={"a.weight": 4},
            model_id="toy",
            expect_ref_sha="0" * 64,
        )


def test_deterministic_encode(tmp_path):
    rng = np.random.default_rng(3)
    w = quantize_bf16_mantissas(_bf16_like(512, rng), 4).reshape(16, 32)
    tensors = {"a.weight": w}
    keep = {"a.weight": 4}
    p1 = tmp_path / "a.h95x"
    p2 = tmp_path / "b.h95x"
    s1 = encode_container(p1, tensors, keep_map=keep, model_id="toy", expect_ref_sha=None)
    s2 = encode_container(p2, tensors, keep_map=keep, model_id="toy", expect_ref_sha=None)
    assert p1.read_bytes() == p2.read_bytes()
    assert s1["sha256_file"] == s2["sha256_file"]


def test_illegal_packed_flag_still_rejected_on_xy_wire():
    with pytest.raises(ValueError, match="illegal tile mode"):
        parse_flag_byte(0)  # mode 0 is not an XY flag byte


@pytest.mark.skipif(not (S1_V2.is_file() or S1_V1.is_file()), reason="S1 container not present")
def test_optional_s1_container_roundtrip(tmp_path):
    from pbr_q4.s1_source import find_s1_container, load_s1_from_container

    src = find_s1_container()
    assert src is not None
    s1 = load_s1_from_container(src, expect_sha=FROZEN_S1_SHA)
    assert s1["sha256_quantized_reference"] == FROZEN_S1_SHA
    out = tmp_path / "S1-XY-sel.h95x"
    stats = encode_container(
        out,
        s1["tensors"],
        keep_map=s1["keep_map"],
        model_id=s1["model_id"],
        candidate="H95Q-S1+XY-sel",
        precision_map=s1["precision_map"],
        embed_name=s1["embed_name"],
        embed_row_keeps=s1["embed_row_keeps"],
        expect_ref_sha=FROZEN_S1_SHA,
    )
    assert stats["sha256_quantized_reference"] == FROZEN_S1_SHA
    dec = decode_container(out)
    v = verify_decoded_against_reference(dec["tensors"], s1["tensors"])
    assert v["exact_match"] and v["sha_ok"]
    assert v["sha256_decoded"] == FROZEN_S1_SHA
    assert stats["all_tiles_matrix_family"] is False
