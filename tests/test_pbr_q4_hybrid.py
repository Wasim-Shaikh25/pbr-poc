"""Hybrid H95 + groupwise INT: Q-ref, scale ties, selective X/Y, HYBX roundtrip."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from pbr_h95.bitpack import packed_bytes_for_k
from pbr_h95.container_h95q import sha256_state_u16
from pbr_h95.quantize import quantize_bf16_mantissas
from pbr_q4.const import MODE_MATRIX, TILE
from pbr_q4.hybrid.bits import bf16_u16_to_f32, f32_to_bf16_u16, qmax_for_bits
from pbr_q4.hybrid.container import (
    MAGIC,
    decode_container,
    encode_quantized,
    read_header,
    verify_decoded_against_reference,
)
from pbr_q4.hybrid.int_quant import dequantize_int, pick_scales, quantize_matrix
from pbr_q4.hybrid.policy import family_label, get_policy, slot_for_name
from pbr_q4.hybrid.quantize import quantize_tensor
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


@pytest.mark.parametrize(
    "name,policy,kind,val",
    [
        ("model.layers.8.mlp.up_proj.weight", "h_stretch", "int", 4),
        ("model.layers.18.mlp.down_proj.weight", "h_practical", "h95", 4),
        ("model.layers.18.mlp.down_proj.weight", "h_mix", "int", 5),
        ("model.layers.10.self_attn.q_proj.weight", "h_practical", "h95", 5),
        ("model.embed_tokens.weight", "h_stretch", "int", 4),
        ("model.embed_tokens.weight", "h_quality", "h95", 3),
        ("model.layers.3.input_layernorm.weight", "h_practical", "h95", 7),
        ("model.layers.0.mlp.gate_proj.weight", "h_practical", "int", 5),
        ("model.layers.23.mlp.down_proj.weight", "h_practical", "h95", 7),
        ("model.layers.23.self_attn.o_proj.weight", "h_practical", "h95", 7),
    ],
)
def test_policy_assignment(name, policy, kind, val):
    p = get_policy(policy)
    slot = slot_for_name(name, p)
    assert slot.kind == kind
    if kind == "h95":
        assert slot.keep == val
    elif kind == "int":
        assert slot.bits == val


def test_family_labels():
    p = get_policy("h_practical")
    assert family_label("model.layers.8.mlp.up_proj.weight", p) == "mlp_mid"
    assert family_label("model.layers.18.mlp.up_proj.weight", p) == "mlp_late"
    assert family_label("model.layers.10.self_attn.k_proj.weight", p) == "attn_mid"
    assert family_label("model.embed_tokens.weight", p) == "embed"


def test_h95_qref_matches_quantize():
    rng = np.random.default_rng(1)
    words = _f32_to_words(rng.normal(0, 0.4, size=(32, 64)).astype(np.float32))
    p = get_policy("h_practical")
    from pbr_q4.hybrid.policy import H95

    qt = quantize_tensor(words, H95(4), p, name="t", family="mlp_late")
    assert qt.kind == "h95"
    assert np.array_equal(qt.q_ref, quantize_bf16_mantissas(words, 4))


def test_int_dequant_matches_qref():
    rng = np.random.default_rng(1)
    words = _f32_to_words(rng.normal(0, 0.4, size=(32, 128)).astype(np.float32))
    rec = quantize_matrix(words, 4, group_size=32, outlier_frac=0.0, name="t")
    out = dequantize_int(
        bits=4,
        codes=rec["codes"],
        scales=rec["scales"],
        zp=rec["zp"],
        rows=rec["rows"],
        cols=rec["cols"],
        group_size=rec["group_size"],
        shape=rec["shape"],
        outlier_idx=rec["outlier_idx"],
        outlier_words=rec["outlier_words"],
    )
    assert np.array_equal(out, rec["q_ref"])


def test_outliers_restore_original_words():
    rng = np.random.default_rng(2)
    f = rng.normal(0, 0.2, size=(16, 128)).astype(np.float32)
    f[0, 0] = 40.0
    words = _f32_to_words(f)
    rec = quantize_matrix(words, 4, group_size=32, outlier_frac=0.02, name="out")
    assert rec["outlier_idx"].size >= 1
    out = dequantize_int(
        bits=4,
        codes=rec["codes"],
        scales=rec["scales"],
        zp=rec["zp"],
        rows=rec["rows"],
        cols=rec["cols"],
        group_size=rec["group_size"],
        shape=rec["shape"],
        outlier_idx=rec["outlier_idx"],
        outlier_words=rec["outlier_words"],
    )
    assert out.ravel()[rec["outlier_idx"]][0] == words.ravel()[rec["outlier_idx"]][0]
    assert np.array_equal(out, rec["q_ref"])


def test_scale_picker_prefers_spatial_on_mse_tie():
    mse = np.array([[1.00, 1.00], [1.01, 2.00], [1.005, 1.00]], dtype=np.float32)
    spatial = np.array([[5.0, 1.0], [0.1, 0.0], [0.2, 9.0]], dtype=np.float32)
    pick, n_ov = pick_scales(mse, spatial, tie_ratio=1.02)
    assert int(pick[0]) == 1
    assert int(pick[1]) == 0
    assert n_ov >= 1


def test_smooth_ramp_has_low_left_disagreement():
    x = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    tile = np.broadcast_to(x, (16, 128)).copy()
    words = _f32_to_words(tile)
    rec = quantize_matrix(words, 4, group_size=64, outlier_frac=0.0)
    step = np.abs(rec["codes"].astype(np.int16)[:, 1:] - rec["codes"].astype(np.int16)[:, :-1])
    assert float(np.mean(step <= 1)) >= 0.9


def test_odd_and_1d_shapes():
    rng = np.random.default_rng(3)
    p = get_policy("h_practical")
    from pbr_q4.hybrid.policy import INT

    for shape in [(1, 17), (7, 9), (13,), (3, 128), (20, 100)]:
        words = _f32_to_words(rng.normal(0, 0.3, size=shape).astype(np.float32))
        qt = quantize_tensor(words, INT(4), p, name="odd")
        out = dequantize_int(
            bits=4,
            codes=qt.codes,
            scales=qt.scales,
            zp=qt.zp,
            rows=qt.rows,
            cols=qt.cols,
            group_size=qt.group_size,
            shape=qt.shape,
            outlier_idx=qt.outlier_idx,
            outlier_words=qt.outlier_words,
        )
        assert out.shape == words.shape
        assert np.array_equal(out, qt.q_ref)


def test_h95_k7_is_identity():
    rng = np.random.default_rng(11)
    words = _f32_to_words(rng.normal(0, 1.0, size=(8, 32)).astype(np.float32))
    p = get_policy("h_quality")
    from pbr_q4.hybrid.policy import H95

    qt = quantize_tensor(words, H95(7), p, name="id")
    assert np.array_equal(qt.q_ref, words)


def test_container_roundtrip_mixed(tmp_path):
    rng = np.random.default_rng(4)
    p = get_policy("h_practical")
    state = {
        "model.layers.8.mlp.up_proj.weight": _f32_to_words(
            rng.normal(0, 0.25, size=(64, 128)).astype(np.float32)
        ),
        "model.layers.18.mlp.down_proj.weight": _f32_to_words(
            rng.normal(0, 0.25, size=(32, 64)).astype(np.float32)
        ),
        "model.layers.8.input_layernorm.weight": _f32_to_words(
            rng.normal(1, 0.05, size=(64,)).astype(np.float32)
        ),
        "model.embed_tokens.weight": _f32_to_words(
            rng.normal(0, 0.2, size=(32, 128)).astype(np.float32)
        ),
        "model.layers.10.self_attn.q_proj.weight": _f32_to_words(
            rng.normal(0, 0.2, size=(16, 64)).astype(np.float32)
        ),
    }
    qts = [
        quantize_tensor(w, slot_for_name(n, p), p, name=n, family=family_label(n, p))
        for n, w in state.items()
    ]
    kinds = {t.name: t.kind for t in qts}
    assert kinds["model.layers.8.mlp.up_proj.weight"] == "groupwise"
    assert kinds["model.layers.18.mlp.down_proj.weight"] == "h95"
    assert kinds["model.layers.10.self_attn.q_proj.weight"] == "h95"
    path = tmp_path / "toy.hybx"
    stats = encode_quantized(qts, path, policy=p, model_id="toy")
    assert path.stat().st_size == stats["file_bytes"]
    assert stats["s1_compatible"] is False
    assert stats["pr17_compatible"] is False
    assert stats["byte_mix"]["h95_payload_bytes"] > 0
    assert stats["byte_mix"]["int_payload_bytes"] > 0
    decoded = decode_container(path)
    qref = {t.name: t.q_ref for t in qts}
    ver = verify_decoded_against_reference(decoded["tensors"], qref)
    assert ver["ok"], ver
    assert decoded["sha256_decoded"] == stats["sha256_quantized_reference"]
    assert decoded["sha256_decoded"] == sha256_state_u16(qref)
    header = read_header(path)
    assert header["s1_compatible"] is False
    assert header["format"] == "HYBX"
    assert decoded["sha256_decoded"] != (
        "eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de"
    )


def test_container_rejects_truncation(tmp_path):
    rng = np.random.default_rng(5)
    p = get_policy("h_stretch")
    from pbr_q4.hybrid.policy import INT

    w = _f32_to_words(rng.normal(0, 0.2, size=(16, 64)).astype(np.float32))
    qt = quantize_tensor(w, INT(4), p, name="t", family="mlp_mid")
    path = tmp_path / "trunc.hybx"
    encode_quantized([qt], path, policy=p)
    raw = path.read_bytes()
    path.write_bytes(raw[:-8])
    with pytest.raises(ValueError):
        decode_container(path)


def test_container_rejects_bad_magic(tmp_path):
    rng = np.random.default_rng(6)
    p = get_policy("h_stretch")
    from pbr_q4.hybrid.policy import INT

    w = _f32_to_words(rng.normal(0, 0.2, size=(16, 32)).astype(np.float32))
    qt = quantize_tensor(w, INT(4), p, name="t", family="mlp_mid")
    path = tmp_path / "bad.hybx"
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
        assert MODE_MATRIX not in (enc.get("mode_hist") or {})
    else:
        assert enc["blob_bytes"] == enc["packed_whole_bytes"]


def test_header_json_stable_and_no_s1_claim(tmp_path):
    p = get_policy("h_practical")
    from pbr_q4.hybrid.policy import INT

    w = _f32_to_words(np.linspace(-0.5, 0.5, 256, dtype=np.float32).reshape(16, 16))
    qt = quantize_tensor(w, INT(4), p, name="t", family="mlp_mid")
    path = tmp_path / "h.hybx"
    encode_quantized([qt], path, policy=p)
    raw = path.read_bytes()
    assert raw[:4] == MAGIC
    hdr_len = struct.unpack_from("<I", raw, 12)[0]
    header = json.loads(raw[16 : 16 + hdr_len].decode("utf-8"))
    assert header["s1_compatible"] is False
    assert header["pr17_compatible"] is False
    assert header["xy_default"] == "packed"
    assert header["tile"] == [TILE, TILE]


def test_deterministic_encode(tmp_path):
    rng = np.random.default_rng(8)
    p = get_policy("h_stretch")
    from pbr_q4.hybrid.policy import INT

    w = _f32_to_words(rng.normal(0, 0.3, size=(24, 96)).astype(np.float32))
    a = tmp_path / "a.hybx"
    b = tmp_path / "b.hybx"
    qt1 = quantize_tensor(w, INT(4), p, name="t", family="mlp_mid")
    qt2 = quantize_tensor(w, INT(4), p, name="t", family="mlp_mid")
    encode_quantized([qt1], a, policy=p)
    encode_quantized([qt2], b, policy=p)
    assert a.read_bytes() == b.read_bytes()
    assert np.array_equal(qt1.q_ref, qt2.q_ref)


def test_qmax_bits():
    assert qmax_for_bits(4) == 7
    assert qmax_for_bits(5) == 15
    assert qmax_for_bits(6) == 31
