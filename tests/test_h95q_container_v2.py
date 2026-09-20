"""Gates E1–E4 for H95Q container v2 adaptive exact exponent compression."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pbr_h95.container_h95q import (
    VERSION_V1,
    VERSION_V2,
    decode_container,
    encode_container,
    sha256_state_u16,
    verify_decoded_against_reference,
)
from pbr_h95.exp_codec import (
    EXP_DELTA_RANS,
    EXP_HUFFMAN,
    EXP_RANS,
    EXP_RAW8,
    EXP_RUN_RANS,
    decode_exp_blob,
    delta_u8,
    encode_delta_rans,
    encode_exp_adaptive,
    encode_huffman,
    encode_rans,
    encode_raw8,
    encode_run_rans,
    undelta_u8,
)
from pbr_h95.quantize import quantize_bf16_mantissas

S1_SHA = "eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de"
V1_S1 = Path("artifacts/pbr_h95/containers/H95Q-S1.h95q")


def _make_words(n: int, rng: np.random.Generator) -> np.ndarray:
    sign = rng.integers(0, 2, size=n, dtype=np.uint16)
    exp = rng.integers(1, 254, size=n, dtype=np.uint16)
    mant = rng.integers(0, 128, size=n, dtype=np.uint16)
    return ((sign << 15) | (exp << 7) | mant).astype(np.uint16)


# --- E1: exact exp roundtrip + edge cases ---------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 7, 63, 64, 257, 4096])
@pytest.mark.parametrize(
    "kind",
    ["empty_ok", "mono", "edge_0_255", "skew", "two_symbol"],
)
def test_e1_exp_roundtrip_edges(n, kind):
    rng = np.random.default_rng(n * 17 + hash(kind) % 10_000)
    if kind == "empty_ok" and n != 0:
        pytest.skip("empty only")
    if kind == "empty_ok":
        x = np.zeros(0, dtype=np.uint8)
    elif kind == "mono":
        x = np.full(n, 42, dtype=np.uint8)
    elif kind == "edge_0_255":
        x = np.array(([0, 255] * ((n + 1) // 2))[:n], dtype=np.uint8)
    elif kind == "two_symbol":
        x = rng.choice([100, 101], size=n).astype(np.uint8)
    else:
        x = rng.choice([110, 111, 112, 200], size=n, p=[0.6, 0.25, 0.1, 0.05]).astype(np.uint8)

    for enc_fn in (encode_raw8, encode_rans, encode_huffman, encode_delta_rans, encode_run_rans):
        if n == 0 and enc_fn is encode_run_rans:
            r = enc_fn(x)
        else:
            r = enc_fn(x)
        y, mode = decode_exp_blob(r.blob, n)
        assert mode == r.mode
        assert np.array_equal(y, x)
        assert len(r.blob) == r.complete_bytes

    d = delta_u8(x)
    assert np.array_equal(undelta_u8(d), x)

    best = encode_exp_adaptive(x)
    y2, _ = decode_exp_blob(best.blob, n)
    assert np.array_equal(y2, x)


# --- E2: selection heuristics ---------------------------------------------


def test_e2_random_prefers_raw8_skew_entropy():
    rng = np.random.default_rng(123)
    rand = rng.integers(0, 256, size=20_000, dtype=np.uint8)
    r = encode_exp_adaptive(rand)
    assert r.mode == EXP_RAW8
    assert r.complete_bytes <= 1 + 20_000  # mode + raw; no silly expansion

    skew = rng.choice([120, 121, 122], size=50_000, p=[0.85, 0.1, 0.05]).astype(np.uint8)
    s = encode_exp_adaptive(skew)
    assert s.mode in (EXP_RANS, EXP_HUFFMAN, EXP_DELTA_RANS, EXP_RUN_RANS)
    assert s.complete_bpw < 4.0

    tiny = rng.integers(0, 3, size=8, dtype=np.uint8)
    t = encode_exp_adaptive(tiny)
    # Must not expand beyond raw+small overhead absurdly
    assert t.complete_bytes <= 1 + 8 + 32


# --- E3 / E4: full S1 (optional if artifact present) ----------------------


def _rebuild_maps_from_v1(header: dict):
    keep_map = {}
    embed_name = None
    embed_row_keeps = None
    for spec in header["tensors"]:
        name = spec["name"]
        if spec["mode"] == "uniform":
            keep_map[name] = int(spec["base_keep"])
        else:
            embed_name = name
            keep_map[name] = -1
    return keep_map, embed_name


@pytest.mark.skipif(not V1_S1.is_file(), reason="H95Q-S1 v1 container missing")
def test_e3_e4_s1_v2_from_v1(tmp_path):
    from pbr_h95.container_h95q import read_header

    v1 = decode_container(V1_S1)
    header = v1["header"]
    assert header["sha256_quantized_reference"] == S1_SHA
    tensors = v1["tensors"]
    # Drop aliases for unique encode (encode_container de-dupes by pointer,
    # but alias arrays are views — rebuild unique dict from header order).
    unique = {spec["name"]: tensors[spec["name"]] for spec in header["tensors"]}
    keep_map, embed_name = _rebuild_maps_from_v1(header)
    embed_row_keeps = None
    if embed_name is not None:
        # Recover row keeps from v1 payload
        spec = next(s for s in header["tensors"] if s["name"] == embed_name)
        data = V1_S1.read_bytes()
        rk = np.frombuffer(
            data[spec["row_keeps_off"] : spec["row_keeps_off"] + spec["row_keeps_len"]],
            dtype=np.int8,
        ).copy()
        embed_row_keeps = rk

    out = tmp_path / "H95Q-S1-v2.h95q"
    enc = encode_container(
        out,
        unique,
        keep_map=keep_map,
        model_id=header.get("model_id", "qwen"),
        candidate="H95Q-S1",
        precision_map=header.get("precision_map") or {},
        embed_name=embed_name,
        embed_row_keeps=embed_row_keeps,
        version=VERSION_V2,
    )
    assert enc["sha256_quantized_reference"] == S1_SHA
    assert enc["actual_bpw"] <= 8.0
    assert enc["exp_complete_bpw"] <= 3.20

    dec = decode_container(out)
    assert int(dec["header"]["version"]) == VERSION_V2
    verify = verify_decoded_against_reference(dec["tensors"], unique)
    assert verify["exact_match"] and verify["sha_ok"]
    assert verify["sha256_decoded"] == S1_SHA

    # E4: v1 decode == v2 decode word-identical
    diffs = 0
    for name in unique:
        if not np.array_equal(v1["tensors"][name], dec["tensors"][name]):
            diffs += int(np.sum(v1["tensors"][name] != dec["tensors"][name]))
    assert diffs == 0

    # Physical section accounting
    exp_sum = sum(int(s["exp_len"]) for s in dec["header"]["tensors"])
    assert exp_sum == enc["exp_section_bytes"]
    assert out.stat().st_size == enc["file_bytes"]


def test_v1_and_v2_toy_word_identical(tmp_path):
    rng = np.random.default_rng(7)
    w1 = quantize_bf16_mantissas(_make_words(256, rng), 4).reshape(16, 16)
    w2 = quantize_bf16_mantissas(_make_words(128, rng), 5).reshape(8, 16)
    # Skew exponents for compression win
    flat = w2.ravel().copy()
    # force skewed exp field
    sign = (flat >> 15) & 1
    mant = flat & 0x7F
    exp = np.full(flat.size, 120, dtype=np.uint16)
    exp[::10] = 121
    w2 = ((sign << 15) | (exp << 7) | mant).astype(np.uint16).reshape(8, 16)
    w2 = quantize_bf16_mantissas(w2.ravel(), 5).reshape(8, 16)

    tensors = {"a.weight": w1, "b.weight": w2}
    keep_map = {"a.weight": 4, "b.weight": 5}
    p1 = tmp_path / "t-v1.h95q"
    p2 = tmp_path / "t-v2.h95q"
    encode_container(p1, tensors, keep_map=keep_map, model_id="toy", version=VERSION_V1)
    encode_container(p2, tensors, keep_map=keep_map, model_id="toy", version=VERSION_V2)
    d1 = decode_container(p1)
    d2 = decode_container(p2)
    v = verify_decoded_against_reference(d1["tensors"], tensors)
    assert v["exact_match"]
    v2 = verify_decoded_against_reference(d2["tensors"], tensors)
    assert v2["exact_match"]
    for name in tensors:
        assert np.array_equal(d1["tensors"][name], d2["tensors"][name])
