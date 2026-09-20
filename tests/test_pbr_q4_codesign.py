"""PBR-Q4 codesign: groupwise Q-ref, compression-aware scales, selective X/Y."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from pbr_h95.bitpack import packed_bytes_for_k
from pbr_h95.container_h95q import sha256_state_u16
from pbr_q4.codesign.bits import (
    bf16_u16_to_f32,
    f32_to_bf16_u16,
    qmax_for_bits,
    signed_to_stored,
    stored_to_signed,
)
from pbr_q4.codesign.container import (
    MAGIC,
    decode_container,
    encode_quantized,
    read_header,
    verify_decoded_against_reference,
)
from pbr_q4.codesign.policy import bits_for_name, family_label, get_policy
from pbr_q4.codesign.quantize import (
    dequantize_to_bf16,
    quantize_matrix,
    quantize_tensor,
)
from pbr_q4.const import MODE_MATRIX, TILE
from pbr_q4.predictors import residual_tile, reconstruct_tile
from pbr_q4.selective import decode_array_selective, encode_array_selective


def _f32_to_words(x: np.ndarray) -> np.ndarray:
    return f32_to_bf16_u16(np.asarray(x, dtype=np.float32))


def test_bf16_roundtrip_finite():
    rng = np.random.default_rng(0)
    f = rng.normal(0, 1, size=1024).astype(np.float32)
    w = f32_to_bf16_u16(f)
    back = bf16_u16_to_f32(w)
    again = f32_to_bf16_u16(back)
    assert np.array_equal(w, again)


def test_signed_stored_roundtrip():
    for bits in (4, 5, 6, 8):
        qmax = qmax_for_bits(bits)
        q = np.arange(-qmax, qmax + 1, dtype=np.int32)
        stored = signed_to_stored(q, bits)
        assert stored.max() <= (1 << bits) - 2
        assert np.array_equal(stored_to_signed(stored, bits), q)


@pytest.mark.parametrize(
    "name,policy,expect",
    [
        ("model.layers.8.mlp.up_proj.weight", "stretch", 4),
        ("model.layers.18.mlp.down_proj.weight", "practical", 5),
        ("model.layers.18.mlp.down_proj.weight", "safe", 6),
        ("model.layers.10.self_attn.q_proj.weight", "practical", 6),
        ("model.embed_tokens.weight", "stretch", 4),
        ("model.embed_tokens.weight", "practical", 5),
        ("model.layers.3.input_layernorm.weight", "practical", None),
        ("model.layers.0.mlp.gate_proj.weight", "practical", 5),
        ("model.layers.23.mlp.down_proj.weight", "practical", 6),
    ],
)
def test_policy_assignment(name, policy, expect):
    p = get_policy(policy)
    assert bits_for_name(name, p) == expect


def test_family_labels():
    p = get_policy("practical")
    assert family_label("model.layers.8.mlp.up_proj.weight", p) == "mlp_mid"
    assert family_label("model.layers.18.mlp.up_proj.weight", p) == "mlp_late"
    assert family_label("model.layers.10.self_attn.k_proj.weight", p) == "attn_mid"
    assert family_label("model.embed_tokens.weight", p) == "embed"


def test_quantize_dequant_matches_qref():
    rng = np.random.default_rng(1)
    f = (rng.normal(0, 0.4, size=(32, 128))).astype(np.float32)
    words = _f32_to_words(f)
    qt = quantize_matrix(words, 4, group_size=32, outlier_frac=0.0, name="t")
    rec = dequantize_to_bf16(
        bits=4,
        codes=qt.codes,
        scales=qt.scales,
        rows=qt.rows,
        cols=qt.cols,
        group_size=qt.group_size,
        shape=qt.shape,
        outlier_idx=qt.outlier_idx,
        outlier_words=qt.outlier_words,
    )
    assert np.array_equal(rec, qt.q_ref)


def test_outliers_restore_original_words():
    rng = np.random.default_rng(2)
    f = rng.normal(0, 0.2, size=(16, 128)).astype(np.float32)
    f[0, 0] = 40.0  # extreme outlier
    words = _f32_to_words(f)
    qt = quantize_matrix(words, 4, group_size=32, outlier_frac=0.02, name="out")
    assert qt.outlier_idx.size >= 1
    rec = dequantize_to_bf16(
        bits=4,
        codes=qt.codes,
        scales=qt.scales,
        rows=qt.rows,
        cols=qt.cols,
        group_size=qt.group_size,
        shape=qt.shape,
        outlier_idx=qt.outlier_idx,
        outlier_words=qt.outlier_words,
    )
    assert rec.ravel()[qt.outlier_idx][0] == words.ravel()[qt.outlier_idx][0]
    assert np.array_equal(rec, qt.q_ref)


def test_scale_picker_prefers_spatial_on_mse_tie():
    from pbr_q4.codesign.quantize import _pick_scales

    mse = np.array([[1.00, 1.00], [1.01, 2.00], [1.005, 1.00]], dtype=np.float32)
    spatial = np.array([[5.0, 1.0], [0.1, 0.0], [0.2, 9.0]], dtype=np.float32)
    pick, n_ov = _pick_scales(mse, spatial, tie_ratio=1.02)
    # group 0: cand 0/1/2 are near-ties; cand 1 has best spatial
    assert int(pick[0]) == 1
    # group 1: only cand 0 and 2 near 1.00; cand 0 has better spatial
    assert int(pick[1]) == 0
    assert n_ov >= 1


def test_smooth_ramp_has_low_left_disagreement():
    x = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    tile = np.broadcast_to(x, (16, 128)).copy()
    words = _f32_to_words(tile)
    qt = quantize_matrix(words, 4, group_size=64, outlier_frac=0.0)
    # Horizontal ramp → neighbouring codes should mostly agree or step by 1.
    step = np.abs(qt.codes.astype(np.int16)[:, 1:] - qt.codes.astype(np.int16)[:, :-1])
    assert float(np.mean(step <= 1)) >= 0.9


def test_odd_and_1d_shapes():
    rng = np.random.default_rng(3)
    for shape in [(1, 17), (7, 9), (13,), (3, 128), (20, 100)]:
        f = rng.normal(0, 0.3, size=shape).astype(np.float32)
        words = _f32_to_words(f)
        p = get_policy("practical")
        qt = quantize_tensor(words, 4, p, name="odd")
        rec = dequantize_to_bf16(
            bits=4,
            codes=qt.codes,
            scales=qt.scales,
            rows=qt.rows,
            cols=qt.cols,
            group_size=qt.group_size,
            shape=qt.shape,
            outlier_idx=qt.outlier_idx,
            outlier_words=qt.outlier_words,
        )
        assert rec.shape == words.shape
        assert np.array_equal(rec, qt.q_ref)


def test_bf16_raw_policy_norms():
    p = get_policy("practical")
    words = _f32_to_words(np.ones((896,), dtype=np.float32))
    qt = quantize_tensor(words, bits_for_name("model.norm.weight", p), p, name="model.norm.weight")
    assert qt.kind == "bf16_raw"
    assert np.array_equal(qt.q_ref, words)


def test_container_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    p = get_policy("practical")
    state = {
        "model.layers.8.mlp.up_proj.weight": _f32_to_words(rng.normal(0, 0.25, size=(64, 128)).astype(np.float32)),
        "model.layers.8.input_layernorm.weight": _f32_to_words(rng.normal(1, 0.05, size=(64,)).astype(np.float32)),
        "model.embed_tokens.weight": _f32_to_words(rng.normal(0, 0.2, size=(32, 128)).astype(np.float32)),
    }
    qts = [
        quantize_tensor(w, bits_for_name(n, p), p, name=n, family=family_label(n, p))
        for n, w in state.items()
    ]
    path = tmp_path / "toy.pq4x"
    stats = encode_quantized(qts, path, policy=p, model_id="toy")
    assert path.stat().st_size == stats["file_bytes"]
    assert stats["s1_compatible"] is False
    decoded = decode_container(path)
    qref = {t.name: t.q_ref for t in qts}
    ver = verify_decoded_against_reference(decoded["tensors"], qref)
    assert ver["ok"]
    assert decoded["sha256_decoded"] == stats["sha256_quantized_reference"]
    assert decoded["sha256_decoded"] == sha256_state_u16(qref)
    header = read_header(path)
    assert header["s1_compatible"] is False
    assert header["format"] == "PQ4X"


def test_container_rejects_truncation(tmp_path):
    rng = np.random.default_rng(5)
    p = get_policy("stretch")
    w = _f32_to_words(rng.normal(0, 0.2, size=(16, 64)).astype(np.float32))
    qt = quantize_tensor(w, 4, p, name="t", family="mlp_mid")
    path = tmp_path / "trunc.pq4x"
    encode_quantized([qt], path, policy=p)
    raw = path.read_bytes()
    path.write_bytes(raw[:-8])
    with pytest.raises(ValueError):
        decode_container(path)


def test_container_rejects_bad_magic(tmp_path):
    rng = np.random.default_rng(6)
    p = get_policy("stretch")
    w = _f32_to_words(rng.normal(0, 0.2, size=(16, 32)).astype(np.float32))
    qt = quantize_tensor(w, 4, p, name="t", family="mlp_mid")
    path = tmp_path / "bad.pq4x"
    encode_quantized([qt], path, policy=p)
    blob = bytearray(path.read_bytes())
    blob[0:4] = b"XXXX"
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="magic"):
        decode_container(path)


def test_random_codes_prefer_packed():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, size=(64, 64), dtype=np.uint16)
    enc = encode_array_selective(codes, 4)
    assert enc["kind"] == "packed"
    rec = decode_array_selective(enc["blob"], rows=64, cols=64, nbits=4)
    assert np.array_equal(rec, codes)
    assert enc["blob_bytes"] == packed_bytes_for_k(64 * 64, 4)


def test_constant_tile_selects_xy_or_equals_packed():
    codes = np.full((32, 32), 3, dtype=np.uint16)
    enc = encode_array_selective(codes, 4)
    rec = decode_array_selective(enc["blob"], rows=32, cols=32, nbits=4)
    assert np.array_equal(rec, codes)
    if enc["kind"] == "xy_sel":
        assert enc["n_xy_tiles"] > 0
        assert enc["blob_bytes"] < enc["packed_whole_bytes"]
        assert MODE_MATRIX not in (
            # MATRIX ties packed and must not be selected by the product path
        )
    else:
        assert enc["blob_bytes"] == enc["packed_whole_bytes"]


def test_horizontal_gradient_roundtrip():
    codes = np.broadcast_to(np.arange(16, dtype=np.uint16), (16, 16)).copy()
    enc = encode_array_selective(codes, 4)
    rec = decode_array_selective(enc["blob"], rows=16, cols=16, nbits=4)
    assert np.array_equal(rec, codes)


def test_all_q4_codes_vs_predictors():
    from pbr_q4.const import (
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

    grid = np.arange(256, dtype=np.uint16).reshape(16, 16) & np.uint16(15)
    for pred in (PRED_PREVIOUS, PRED_LEFT, PRED_UP, PRED_AVG, PRED_PAETH):
        for trav in (TRAV_ROW, TRAV_ROW_SERP, TRAV_COL, TRAV_COL_SERP):
            res = residual_tile(grid, pred=pred, trav=trav, nbits=4)
            rec = reconstruct_tile(res, pred=pred, trav=trav, nbits=4)
            assert np.array_equal(rec, grid.astype(np.uint8))


def test_header_json_stable_and_no_s1_claim(tmp_path):
    p = get_policy("practical")
    w = _f32_to_words(np.linspace(-0.5, 0.5, 256, dtype=np.float32).reshape(16, 16))
    qt = quantize_tensor(w, 4, p, name="t", family="mlp_mid")
    path = tmp_path / "h.pq4x"
    encode_quantized([qt], path, policy=p)
    raw = path.read_bytes()
    assert raw[:4] == MAGIC
    hdr_len = struct.unpack_from("<I", raw, 12)[0]
    header = json.loads(raw[16 : 16 + hdr_len].decode("utf-8"))
    assert header["s1_compatible"] is False
    assert header["xy_default"] == "packed"
    assert header["tile"] == [TILE, TILE]


def test_deterministic_encode(tmp_path):
    rng = np.random.default_rng(8)
    p = get_policy("stretch")
    w = _f32_to_words(rng.normal(0, 0.3, size=(24, 96)).astype(np.float32))
    a = tmp_path / "a.pq4x"
    b = tmp_path / "b.pq4x"
    qt1 = quantize_tensor(w, 4, p, name="t", family="mlp_mid")
    qt2 = quantize_tensor(w, 4, p, name="t", family="mlp_mid")
    encode_quantized([qt1], a, policy=p)
    encode_quantized([qt2], b, policy=p)
    assert a.read_bytes() == b.read_bytes()
    assert np.array_equal(qt1.q_ref, qt2.q_ref)
