"""Bit-exact base→finetune delta codecs. Tiny fixtures only; no 988 MB download."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.hashing import sha256_words
from pbr_core.safetensors_io import write_uint16_safetensors
from pbr_encoder.checkpoint_delta import (
    DELTA_METHODS,
    DISCLAIMER,
    decode_field_split,
    decode_sparse_patch,
    decode_standalone_pbre,
    decode_xor_rans,
    decode_xor_residual,
    decode_xor_zlib,
    dumps_delta,
    encode_field_split,
    encode_sparse_patch,
    encode_standalone_pbre,
    encode_xor_rans,
    encode_xor_residual,
    encode_xor_zlib,
    evaluate_pair,
    format_markdown,
    loads_delta,
    pair_tensors,
    xor_stats,
)
from pbr_encoder.hf_weights import inventory_from_dir
from pbr_encoder.verification import assert_exact


def _pair(n: int = 256, *, seed: int = 0, n_flip: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = make_bf16_bits(
        rng.integers(0, 2, size=n, dtype=np.uint16),
        rng.choice(np.array([120, 121, 126, 127, 128], dtype=np.uint16), size=n),
        rng.integers(0, 128, size=n, dtype=np.uint16),
    ).reshape(16, n // 16)
    target = base.copy()
    idx = rng.choice(n, size=min(n_flip, n), replace=False)
    flat = target.ravel()
    flat[idx] = np.bitwise_xor(flat[idx], np.uint16(0x0007))
    return base, target


def test_xor_stats_identical() -> None:
    words = np.arange(64, dtype=np.uint16).reshape(8, 8)
    stats = xor_stats(words, words)
    assert stats["unchanged_frac"] == 1.0
    assert stats["mean_xor_popcount"] == 0.0
    assert stats["H_sign_xor"] == 0.0


def test_each_delta_method_roundtrips_and_specials() -> None:
    base, target = _pair(256, seed=1, n_flip=20)
    specials = np.resize(special_payload_words(), 32)
    base = np.concatenate([base.ravel(), specials]).reshape(12, 24)
    target = np.concatenate([target.ravel(), np.bitwise_xor(specials, np.uint16(1))]).reshape(12, 24)
    codecs = {
        "xor_zlib": (encode_xor_zlib, decode_xor_zlib),
        "xor_rans": (encode_xor_rans, decode_xor_rans),
        "field_split": (encode_field_split, decode_field_split),
        "sparse_patch": (encode_sparse_patch, decode_sparse_patch),
        "xor_residual": (encode_xor_residual, decode_xor_residual),
    }
    for name, (enc, dec) in codecs.items():
        payload = enc(base, target)
        restored = dec(payload, base)
        assert_exact(target, restored, label=name)
        assert sha256_words(restored) == sha256_words(target)


def test_standalone_pbre_roundtrip() -> None:
    _base, target = _pair(512, seed=2, n_flip=40)
    payload = encode_standalone_pbre(target)
    restored = decode_standalone_pbre(payload, int(target.size), tuple(target.shape))
    assert_exact(target, restored, label="standalone")


def test_sparse_beats_raw_when_few_positions_change() -> None:
    base, target = _pair(2048, seed=3, n_flip=4)
    sparse = encode_sparse_patch(base, target)
    raw_xor = 1 + 2 * int(base.size)
    assert len(sparse) < raw_xor
    assert_exact(target, decode_sparse_patch(sparse, base), label="sparse_few")


def test_identical_delta_is_tiny() -> None:
    words = np.arange(128, dtype=np.uint16).reshape(8, 16)
    sparse = encode_sparse_patch(words, words)
    zlib_p = encode_xor_zlib(words, words)
    residual = encode_xor_residual(words, words)
    assert len(sparse) < 32
    assert len(zlib_p) < 64
    assert len(residual) < 16
    assert_exact(words, decode_sparse_patch(sparse, words), label="ident_sparse")
    assert_exact(words, decode_xor_zlib(zlib_p, words), label="ident_zlib")
    assert_exact(words, decode_xor_residual(residual, words), label="ident_res")


def test_pair_by_name_and_shape_reports_unmatched(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    shared = rng.integers(0, 65536, (16, 8), dtype=np.uint16)
    base_dir = tmp_path / "base"
    tgt_dir = tmp_path / "tgt"
    write_uint16_safetensors(base_dir / "m.safetensors", {"w": shared, "only_base": shared[:4]})
    write_uint16_safetensors(
        tgt_dir / "m.safetensors",
        {"w": shared, "only_tgt": rng.integers(0, 65536, (8, 8), dtype=np.uint16)},
    )
    paired, unmatched_base, unmatched_tgt = pair_tensors(
        inventory_from_dir(base_dir), inventory_from_dir(tgt_dir)
    )
    assert len(paired) == 1 and paired[0][0].name == "w"
    assert [s.name for s in unmatched_base] == ["only_base"]
    assert [s.name for s in unmatched_tgt] == ["only_tgt"]


def test_dumps_loads_roundtrip_container() -> None:
    base, target = _pair(64, seed=5, n_flip=6)
    payload = encode_xor_zlib(base, target)
    header = {
        "format": "PBR-Delta",
        "version": 1,
        "method": "xor_zlib",
        "tensors": [
            {
                "name": "w",
                "shape": list(target.shape),
                "n_words": int(target.size),
                "sha256": sha256_words(target),
                "payload_len": len(payload),
            }
        ],
    }
    blob = dumps_delta(header, [payload])
    h2, payloads = loads_delta(blob)
    assert h2["method"] == "xor_zlib"
    assert_exact(target, decode_xor_zlib(payloads[0], base), label="container")


def test_evaluate_pair_exact_restore(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    base_w = rng.integers(0, 65536, (32, 16), dtype=np.uint16)
    tgt_w = base_w.copy()
    tgt_w[0, :4] = np.bitwise_xor(tgt_w[0, :4], np.uint16(0x00FF))
    extra = rng.integers(0, 65536, (8, 8), dtype=np.uint16)
    base_dir = tmp_path / "base"
    tgt_dir = tmp_path / "tgt"
    write_uint16_safetensors(base_dir / "m.safetensors", {"layer.weight": base_w})
    write_uint16_safetensors(tgt_dir / "m.safetensors", {"layer.weight": tgt_w, "new.weight": extra})
    report = evaluate_pair(
        base_dir,
        tgt_dir,
        base_meta={"repo_id": "unit/base", "revision": "aaa"},
        target_meta={"repo_id": "unit/tgt", "revision": "bbb"},
    )
    assert report["totals"]["reconstruct_target"] == "PASS"
    assert report["totals"]["unmatched_target"] == ["new.weight"]
    assert 0.0 < report["totals"]["changed_frac"] < 1.0
    md = format_markdown(report)
    assert "not ≤4 BPW" in md
    assert "1–2 GB" in DISCLAIMER
    assert "Not a 1–2 GB" in md
    for method in ("standalone_pbre_rans", *DELTA_METHODS):
        assert report["totals"]["methods"][method]["exact"] == "PASS"
        assert (
            report["totals"]["methods"][method]["complete_bytes"]
            >= report["totals"]["methods"][method]["payload_bytes"]
        )
    assert (
        report["totals"]["methods"]["xor_zlib"]["bundle_bytes"]
        > report["totals"]["methods"]["xor_zlib"]["complete_bytes"]
    )
    standalone = report["totals"]["methods"]["standalone_pbre_rans"]["standalone_bpw"]
    assert standalone > 4.0
