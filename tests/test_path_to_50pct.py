"""PBR-EM / optional-nibble exactness; not on STAGE1A_CODECS."""

from __future__ import annotations

import numpy as np

from pbr_codecs import STAGE1A_CODECS
from pbr_codecs.optional_nibble import decode_matrix, encode_matrix, encode_tile
from pbr_codecs.pbr_em import (
    TAG_EXPCOND_M,
    TAG_SM_EXPCOND,
    TAG_SM_UNCOND,
    TAG_UNCOND_M,
    decode_tensor_em_shared,
    dump_shared_tables,
    encode_tensor_em,
    encode_tensor_em_shared,
    roundtrip_em,
)
from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.rans import normalize_counts
from pbr_core.types import MODE_PBR4
from pbr_encoder.path_to_50pct import DISCLAIMER, TARGET_BPW, decode_escape_dict, encode_escape_dict, format_markdown
from pbr_encoder.verification import assert_exact


def _pattern(n: int = 32, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    exp = rng.choice(np.array([120, 121, 126, 127], dtype=np.uint16), size=(n, n))
    words = make_bf16_bits(
        rng.integers(0, 2, size=(n, n), dtype=np.uint16),
        exp,
        rng.integers(0, 128, size=(n, n), dtype=np.uint16),
    )
    spec = np.resize(special_payload_words(), 16)
    words[0, :16] = make_bf16_bits(np.zeros(16, np.uint16), np.full(16, 127, np.uint16), spec & 0x7F)
    return words


def test_em_not_on_stage1a_menu() -> None:
    names = {c.name for c in STAGE1A_CODECS}
    ids = {c.mode_id for c in STAGE1A_CODECS}
    assert "pbr_em" not in names
    assert MODE_PBR4 not in ids


def test_em_tags_roundtrip_specials() -> None:
    words = _pattern(24)
    for tag in (TAG_UNCOND_M, TAG_EXPCOND_M, TAG_SM_UNCOND, TAG_SM_EXPCOND):
        payload, rec = roundtrip_em(words, tag=tag)
        assert_exact(words, rec, label=f"em-{tag}")
        assert len(payload) > 0


def test_shared_expcond_roundtrip() -> None:
    a = _pattern(16, seed=1)
    b = _pattern(16, seed=2)
    from pbr_core.bf16 import split_components

    exp_c = np.zeros(256, dtype=np.int64)
    mant_c = np.zeros((256, 128), dtype=np.int64)
    for w in (a, b):
        _s, e, m = split_components(w)
        e = e.ravel()
        m = m.ravel()
        exp_c += np.bincount(e.astype(np.int64), minlength=256)
        mant_c += np.bincount(e.astype(np.int64) * 128 + m.astype(np.int64), minlength=256 * 128).reshape(256, 128)
    exp_freq = normalize_counts(exp_c if int(exp_c.sum()) else np.array([1] + [0] * 255, dtype=np.int64))
    group_freqs = []
    for row in mant_c:
        buf = np.zeros(256, dtype=np.int64)
        buf[:128] = row
        if buf.sum() <= 0:
            buf[0] = 1
        group_freqs.append(normalize_counts(buf))
    sidecar = dump_shared_tables(exp_freq, group_freqs)
    assert len(sidecar) > 0
    for w in (a, b):
        payload = encode_tensor_em_shared(w, exp_freq=exp_freq, group_freqs=group_freqs, tag=TAG_EXPCOND_M)
        rec = decode_tensor_em_shared(payload, int(w.size), exp_freq, group_freqs).reshape(w.shape)
        assert_exact(w, rec, label="shared_em")


def test_optional_nibble_const_and_random() -> None:
    const = np.full((16, 16), 0x3F80, dtype=np.uint16)
    name, payload = encode_tile(const)
    assert name == "const"
    blob, stats = encode_matrix(const, tile=16)
    rec = decode_matrix(blob)
    assert_exact(const, rec, label="nibble_const")
    assert stats["complete_bytes"] < const.size * 2

    rng = np.random.default_rng(0)
    rnd = rng.integers(0, 65536, size=(32, 32), dtype=np.uint16)
    blob, _stats = encode_matrix(rnd, tile=16)
    rec = decode_matrix(blob)
    assert_exact(rnd, rec, label="nibble_random")


def test_optional_nibble_mode_high_admits_when_shared() -> None:
    words = np.full((16, 16), np.uint16(0x3F80), dtype=np.uint16)
    words[0, :4] = np.arange(4, dtype=np.uint16)
    blob, stats = encode_matrix(words, tile=16)
    rec = decode_matrix(blob)
    assert_exact(words, rec, label="nibble_mode_high")
    assert stats["complete_bytes"] < words.nbytes


def test_escape_dict_roundtrip() -> None:
    rng = np.random.default_rng(4)
    pal = np.array([0x3F80, 0x4000, 0x0000], dtype=np.uint16)
    words = rng.choice(np.append(pal, np.uint16(0x1111)), size=(8, 8))
    payload = encode_escape_dict(words, pal)
    rec = decode_escape_dict(payload, int(words.size)).reshape(words.shape)
    assert_exact(words, rec, label="escape_dict")


def test_path_to_50pct_markdown_no_fake_fifty() -> None:
    report = {
        "disclaimer": DISCLAIMER,
        "n_tensors": 2,
        "n_words": 1000,
        "bounds": {
            "H_uint16": 10.5,
            "H_sign": 1.0,
            "H_exp": 2.61,
            "field_split_ideal_bpw": 10.6,
            "expcond_ideal_bpw": 10.55,
            "joint_sm_exp_ideal_bpw": 10.55,
            "H_sign_given_exp": 1.0,
            "H_mant_given_exp": 6.93,
            "top255_coverage": 0.01,
            "escape255_ideal_bpw": 23.8,
            "bits_missing_to_8": 2.5,
        },
        "tile_duplicates": {"dup_rate": 0.0},
        "methods": {
            "em_expcond_m": {
                "name": "em_expcond_m",
                "encoded_bytes": 1320,
                "bpw": 10.56,
                "hits_8bpw": False,
                "exact": "PASS",
            }
        },
        "best": {"name": "em_expcond_m", "encoded_bytes": 1320, "bpw": 10.56, "exact": "PASS"},
        "hits_8bpw": False,
        "elapsed_s": 1.0,
    }
    md = format_markdown(report)
    assert "Why 8.0 failed" in md
    assert TARGET_BPW == 8.0
    assert "Do not report 50%" in DISCLAIMER
    assert best_bpw_line_not_eight(md)


def best_bpw_line_not_eight(md: str) -> bool:
    return "Best measured: 10.5600 BPW" in md