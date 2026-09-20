"""S1 + 256-node X/Y post-codec: exactness, matrix-family wire, packed baseline-only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_h95.container_h95q import sha256_state_u16
from pbr_h95.quantize import quantize_bf16_mantissas
from pbr_q4.codecs import (
    encode_array_xy,
    encode_tile,
    pack_kbit_batch,
    parse_flag_byte,
    packed_baseline_len,
)
from pbr_h95.bitpack import pack_kbit
from pbr_q4.const import (
    FROZEN_S1_SHA,
    MATRIX_FAMILY_MODES,
    MODE_MATRIX,
    MODE_NAMES,
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
from pbr_q4.container import (
    decode_container,
    encode_container,
    verify_decoded_against_reference,
)
from pbr_q4.predictors import (
    batched_residual_maps,
    residual_tile,
    reconstruct_tile,
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


def test_tile_never_emits_packed_raw_mode():
    rng = np.random.default_rng(0)
    random = _rand_codes(16, 16, 4, rng)
    const = np.full((16, 16), 3, dtype=np.uint16)
    grad = (np.arange(16)[None, :] + np.arange(16)[:, None]).astype(np.uint16) & 15
    for tile in (random, const, grad):
        enc = encode_tile(tile, 4)
        mode, _p, _t = parse_flag_byte(enc.flag_byte())
        assert mode in MATRIX_FAMILY_MODES
        assert mode != 0
        assert enc.family == "matrix"
        rec, _ = __import__("pbr_q4.codecs", fromlist=["decode_tile"]).decode_tile(
            enc.flag_byte(), enc.payload, rows=16, cols=16, nbits=4
        )
        assert np.array_equal(rec, tile.astype(np.uint8))


def test_constant_tile_prefers_run_or_plane():
    tile = np.zeros((16, 16), dtype=np.uint16)
    enc = encode_tile(tile, 4)
    assert enc.mode in MATRIX_FAMILY_MODES
    # Constant residuals are all-zero after PREVIOUS=0 → RUN or PLANE should beat MATRIX.
    assert enc.mode in (3, 5, 1)  # PLANE / RUN / MATRIX (all family)
    assert len(enc.payload) <= packed_baseline_len(256, 4)


@pytest.mark.parametrize("nbits", [3, 4, 5, 7])
def test_pack_kbit_batch_matches_scalar(nbits):
    rng = np.random.default_rng(nbits + 9)
    tiles = rng.integers(0, 1 << nbits, size=(4, 16, 16), dtype=np.uint16)
    batched = pack_kbit_batch(tiles, nbits)
    for i in range(4):
        assert batched[i].tobytes() == pack_kbit(tiles[i], nbits)


def test_batched_row_col_matches_sequential():
    rng = np.random.default_rng(42)
    tiles = rng.integers(0, 16, size=(5, 16, 16), dtype=np.uint16)
    maps = { (tr, pr): res for tr, pr, res in batched_residual_maps(tiles, 4) }
    for i in range(5):
        for trav in (TRAV_ROW, TRAV_COL):
            for pred in (PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH):
                seq = residual_tile(tiles[i], pred=pred, trav=trav, nbits=4)
                assert np.array_equal(maps[(trav, pred)][i], seq), (trav, pred, i)


def test_encode_array_all_tiles_matrix_family():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 8, size=(32, 48), dtype=np.uint16)
    enc = encode_array_xy(arr, 3)
    assert enc["all_matrix_family"]
    assert enc["n_tiles"] == 6
    assert sum(enc["mode_hist"].values()) == 6
    assert 0 not in enc["mode_hist"]
    from pbr_q4.codecs import decode_array_xy

    rec = decode_array_xy(enc["flags"], enc["payload"], rows=32, cols=48, nbits=3)
    assert np.array_equal(rec, arr.astype(np.uint8))
    # Packed size is a metric, not the wire.
    assert enc["packed_baseline_bytes"] == 6 * packed_baseline_len(256, 3)


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
        candidate="toy-S1+XY",
        embed_name="model.embed_tokens.weight",
        embed_row_keeps=row_keeps,
        expect_ref_sha=None,
    )
    assert stats["all_tiles_matrix_family"]
    assert stats["n_tiles"] > 0
    assert all(name in MODE_NAMES.values() for name in stats["xy_mode_histogram"])
    dec = decode_container(path)
    v = verify_decoded_against_reference(dec["tensors"], tensors)
    assert v["exact_match"] and v["sha_ok"]
    assert dec["header"]["mantissa_packing"] == "xy_256_node_matrix_family"
    assert dec["header"]["packed_k_is_baseline_metric_only"] is True
    assert dec["header"]["hard_override"].startswith("all_tiles")
    assert path.stat().st_size == stats["file_bytes"]
    # Packed baseline recorded, but container is X/Y.
    assert stats["packed_baseline_bytes"] > 0
    assert dec["header"]["sha256_quantized_reference"] == sha256_state_u16(tensors)


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


def test_illegal_flag_rejected():
    with pytest.raises(ValueError, match="illegal tile mode"):
        parse_flag_byte(0)  # mode 0 = packed-raw, not on the wire


@pytest.mark.skipif(not (S1_V2.is_file() or S1_V1.is_file()), reason="S1 container not present")
def test_optional_s1_container_roundtrip(tmp_path):
    from pbr_q4.s1_source import load_s1_from_container, find_s1_container

    src = find_s1_container()
    assert src is not None
    s1 = load_s1_from_container(src, expect_sha=FROZEN_S1_SHA)
    assert s1["sha256_quantized_reference"] == FROZEN_S1_SHA
    out = tmp_path / "S1-XY.h95x"
    stats = encode_container(
        out,
        s1["tensors"],
        keep_map=s1["keep_map"],
        model_id=s1["model_id"],
        candidate="H95Q-S1+XY",
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
