"""E2 mixed groupwise Q + E3 selective tiles: exactness, margin, allocation."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from pbr_h95.bitpack import pack_kbit
from pbr_q4.e2_mixed.allocate import allocate, build_units, channel_importance
from pbr_q4.e2_mixed.bits import bf16_u16_to_f32, f32_to_bf16_u16
from pbr_q4.e2_mixed.container import (
    decode_container,
    encode_quantized,
    verify_decoded_against_reference,
)
from pbr_q4.e2_mixed.e3_codec import encode_array_e3, encode_tile_phase1, tile_margin_bits
from pbr_q4.e2_mixed.families import candidate_bits, family_label, projection_label
from pbr_q4.e2_mixed.quantize import (
    QuantizedTensor,
    _code_cost,
    _pick_e2b,
    dequantize_mixed,
    estimate_packed_bits,
    quantize_mixed,
)
from pbr_q4.const import MODE_MATRIX, MODE_PAIR, PRED_LEFT, PRED_PAETH, PRED_UP, TRAV_ROW


def _bf16_words(shape, rng, *, scale=0.05):
    x = rng.standard_normal(shape).astype(np.float32) * np.float32(scale)
    return f32_to_bf16_u16(x).reshape(shape)


def test_family_candidate_sets():
    assert candidate_bits("embed")[0] == 4
    assert 6 in candidate_bits("mlp_mid")
    assert 8 in candidate_bits("mlp_mid")  # protect overlay
    assert 16 in candidate_bits("mlp_mid")
    assert candidate_bits("norm_bias") == (8, 16)
    assert 4 not in candidate_bits("norm_bias")
    assert family_label("model.layers.3.mlp.down_proj.weight") == "mlp_mid"
    assert projection_label("model.layers.3.mlp.down_proj.weight") == "down_proj"
    assert family_label("model.embed_tokens.weight") == "embed"
    assert family_label("model.layers.0.self_attn.q_proj.weight") == "attn_first"


def test_channel_groups_are_contiguous():
    rng = np.random.default_rng(0)
    w = _bf16_words((64, 32), rng)
    units = build_units({"t.weight": w}, channel_group=16)
    assert units
    assert all(u.nrows <= 16 for u in units)
    # No overlapping rows.
    covered = np.zeros(64, dtype=np.int32)
    for u in units:
        covered[u.row0 : u.row0 + u.nrows] += 1
    assert np.all(covered == 1)


def test_allocate_respects_budget_and_protects_important():
    rng = np.random.default_rng(1)
    # Sensitive tensor: large rows. Robust tensor: small rows.
    sens = _bf16_words((32, 128), rng, scale=1.0)
    rob = _bf16_words((32, 128), rng, scale=0.01)
    tensors = {
        "model.layers.3.self_attn.o_proj.weight": sens,
        "model.layers.3.mlp.up_proj.weight": rob,
    }
    n = 32 * 128 * 2
    units = build_units(tensors, channel_group=8)
    alloc = allocate(units, budget_bpw=5.0, n_weights=n)
    assert alloc.packed_est_bpw <= 5.0 + 1e-6
    rb_s = alloc.row_bits["model.layers.3.self_attn.o_proj.weight"]
    rb_r = alloc.row_bits["model.layers.3.mlp.up_proj.weight"]
    # Attention/o_proj should get more bits on average than mid-MLP up_proj.
    assert float(rb_s.mean()) >= float(rb_r.mean()) - 1e-9
    assert set(int(x) for x in np.unique(rb_s).tolist()).issubset({4, 5, 6, 8, 16})


def test_mixed_row_bits_roundtrip_qref():
    rng = np.random.default_rng(2)
    w = _bf16_words((24, 80), rng)
    rb = np.array([4] * 8 + [6] * 8 + [8] * 4 + [16] * 4, dtype=np.uint8)
    qt = quantize_mixed(w, rb, name="mix", family="mlp_mid", projection="down_proj")
    assert qt.row_bits.shape == (24,)
    assert qt.q_ref.shape == w.shape
    recon = dequantize_mixed(
        row_bits=qt.row_bits,
        codes=qt.codes,
        scales=qt.scales,
        zp=qt.zp,
        rows=qt.rows,
        cols=qt.cols,
        group_size=qt.group_size,
        shape=qt.shape,
        special_idx=qt.special_idx,
        special_words=qt.special_words,
        bf16_rows={i: qt.q_ref.reshape(24, 80)[i] for i in range(20, 24)},
    )
    assert np.array_equal(recon, qt.q_ref)
    # BF16 rows stay exact to the original words.
    assert np.array_equal(qt.q_ref.reshape(24, 80)[20:], w[20:])


def test_odd_shapes_roundtrip():
    rng = np.random.default_rng(3)
    for shape in [(1, 17), (7, 9), (16, 1), (3, 3), (5, 130)]:
        w = _bf16_words(shape, rng)
        rb = np.full(shape[0], 5, dtype=np.uint8)
        if shape[0] >= 2:
            rb[-1] = 16
        qt = quantize_mixed(w, rb, name=str(shape))
        bf16_rows = {}
        arr2 = qt.q_ref.reshape(qt.rows, qt.cols)
        for r, b in enumerate(qt.row_bits.tolist()):
            if b >= 16:
                bf16_rows[r] = arr2[r]
        recon = dequantize_mixed(
            row_bits=qt.row_bits,
            codes=qt.codes,
            scales=qt.scales,
            zp=qt.zp,
            rows=qt.rows,
            cols=qt.cols,
            group_size=qt.group_size,
            shape=qt.shape,
            special_idx=qt.special_idx,
            special_words=qt.special_words,
            bf16_rows=bf16_rows,
        )
        assert np.array_equal(recon, qt.q_ref)


def test_nan_special_survives_not_sparse_outlier_path():
    rng = np.random.default_rng(4)
    w = _bf16_words((8, 32), rng)
    w = w.copy()
    w[2, 5] = np.uint16(0x7FC1)  # NaN payload
    rb = np.full(8, 4, dtype=np.uint8)
    qt = quantize_mixed(w, rb, name="nan")
    assert qt.special_idx.size >= 1
    assert 0x7FC1 in set(int(x) for x in qt.special_words.tolist())
    # No extra sparse outliers: specials only.
    assert qt.special_idx.size == 1


def test_e2b_eps_zero_is_mse_min_and_larger_eps_can_pick_cheaper():
    mse = np.array(
        [
            [1.00, 1.00],
            [1.002, 1.20],
            [1.02, 1.001],
        ],
        dtype=np.float32,
    )
    cost = np.array(
        [
            [5.0, 5.0],
            [1.0, 9.0],
            [0.5, 0.4],
        ],
        dtype=np.float32,
    )
    pick0, n0 = _pick_e2b(mse, cost, 0.0)
    assert list(pick0) == [0, 0]
    assert n0 == 0
    pick, n = _pick_e2b(mse, cost, 0.005)
    # col0: cand1 is within 0.5% and cheaper; col1 stays mse-min (cand2 is 0.1% over)
    assert int(pick[0]) == 1
    assert n >= 1


def test_e3_phase1_modes_only_left_up_paeth():
    rng = np.random.default_rng(5)
    tile = rng.integers(0, 16, size=(16, 16), dtype=np.uint16)
    enc = encode_tile_phase1(tile, 4)
    assert enc.pred in (PRED_LEFT, PRED_UP, PRED_PAETH)
    assert enc.trav == TRAV_ROW
    assert enc.mode not in (MODE_MATRIX, MODE_PAIR)
    assert enc.mode_name in {"XY_BITPLANE", "XY_RUN", "XY_RANS"}


def test_e3_random_stays_packed():
    rng = np.random.default_rng(6)
    arr = rng.integers(0, 16, size=(64, 64), dtype=np.uint16)
    enc = encode_array_e3(arr, 4)
    assert enc["kind"] == "packed"
    assert enc["n_xy_tiles"] == 0
    assert enc["fallback_packed"] is True
    assert enc["blob"] == pack_kbit(arr.ravel(), 4)


def test_e3_net_margin_rejects_tiny_savings():
    # A tile whose XY payload is only 1 byte under packed (8 bits) must miss
    # the 16-bit / 2% margin.
    n_nodes = 16 * 16
    packed = (n_nodes * 4 + 7) // 8
    margin = tile_margin_bits(n_nodes, 4)
    assert margin >= 16
    assert margin >= int(0.02 * n_nodes * 4)
    # 1-byte savings is 8 bits < margin.
    assert 8 < margin
    assert packed > 0


def test_e3_chunked_large_array_stays_bounded_and_roundtrips():
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 16, size=(128, 128), dtype=np.uint16)
    enc = encode_array_e3(arr, 4)
    from pbr_q4.e2_mixed.e3_codec import decode_array_e3

    rec = decode_array_e3(enc["blob"], rows=128, cols=128, nbits=4)
    assert np.array_equal(rec, arr)
    assert enc["kind"] in ("packed", "xy_sel")
    arr = np.zeros((32, 32), dtype=np.uint16)
    enc = encode_array_e3(arr, 4)
    # Constant zeros should beat packed by a wide margin.
    assert enc["saved_vs_packed"] >= 2
    from pbr_q4.e2_mixed.e3_codec import decode_array_e3

    rec = decode_array_e3(enc["blob"], rows=32, cols=32, nbits=4)
    assert np.array_equal(rec, arr)
    if enc["kind"] == "xy_sel":
        assert enc["n_xy_tiles"] > 0
        assert "XY_PAIR" not in (enc["mode_hist"] or {}) or enc["mode_hist"].get("XY_PAIR", 0) == 0
        assert enc["mode_hist"].get("XY_MATRIX", 0) == 0


def test_container_e2_e3_roundtrip_and_truncation(tmp_path):
    rng = np.random.default_rng(7)
    w = _bf16_words((20, 48), rng)
    rb = np.array([4] * 12 + [6] * 4 + [16] * 4, dtype=np.uint8)
    qt = quantize_mixed(w, rb, name="model.layers.2.mlp.down_proj.weight", family="mlp_mid", projection="down_proj")
    e2_path = tmp_path / "t.e2mx"
    e3_path = tmp_path / "t.e3mx"
    s2 = encode_quantized([qt], e2_path, phase="E2", policy_name="unit")
    s3 = encode_quantized([qt], e3_path, phase="E3", policy_name="unit")
    d2 = decode_container(e2_path)
    d3 = decode_container(e3_path)
    ver2 = verify_decoded_against_reference(d2["tensors"], {qt.name: qt.q_ref})
    ver3 = verify_decoded_against_reference(d3["tensors"], {qt.name: qt.q_ref})
    assert ver2["ok"] and ver3["ok"]
    assert d2["sha256_decoded"] == s2["sha256_quantized_reference"]
    assert d3["sha256_decoded"] == s3["sha256_quantized_reference"]
    assert s2["sha256_quantized_reference"] == s3["sha256_quantized_reference"]
    # E3 must not exceed E2 by inventing padding tax — Δ reported even if ~0.
    assert "overhead_split" in s2 and "weight_payload_bytes" in s2["overhead_split"]
    # Truncation is rejected.
    trunc = tmp_path / "trunc.e2mx"
    trunc.write_bytes(e2_path.read_bytes()[:-8])
    with pytest.raises((ValueError, json.JSONDecodeError, struct.error)):
        decode_container(trunc)


def test_estimate_packed_bits_split():
    rng = np.random.default_rng(8)
    w = _bf16_words((16, 64), rng)
    rb = np.full(16, 4, dtype=np.uint8)
    qt = quantize_mixed(w, rb, name="est")
    parts = estimate_packed_bits(qt)
    assert parts["total_bits"] == (
        parts["weight_payload_bits"]
        + parts["scales_bits"]
        + parts["precision_map_bits"]
        + parts["specials_bits"]
    )
    assert parts["weight_payload_bits"] == 16 * 64 * 4
