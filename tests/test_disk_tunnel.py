"""Disk-RAM tunnel: tiny local fixture, no HuggingFace download."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pbr_core.hashing import sha256_words
from pbr_core.safetensors_io import load_uint16
from pbr_encoder.disk_tunnel import (
    DISCLAIMER,
    FullRamSource,
    MmapSafetensorsSource,
    bf16_to_fp32,
    encode_pbre_dir,
    format_markdown,
    main,
    run_mode,
    select_specs,
    write_tiny_qwen_fixture,
)
from pbr_encoder.verification import assert_exact


def _ids(n: int = 4, vocab: int = 80) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, vocab, size=n, dtype=np.int64)


def test_disclaimer_not_8bpw_or_phone() -> None:
    text = DISCLAIMER.lower()
    assert "8 bpw" in text
    assert "27b" in text
    assert "not" in text


def test_bf16_to_fp32_places_bits_in_high_half() -> None:
    words = np.array([0x3F80, 0x0000, 0x8000], dtype=np.uint16)
    fp = bf16_to_fp32(words)
    bits = fp.view(np.uint32)
    assert int(bits[0]) == 0x3F800000
    assert int(bits[1]) == 0
    assert int(bits[2]) == 0x80000000


def test_tiny_mmap_logits_match_full(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    ids = _ids()
    full = run_mode(
        mode="full",
        model_dir=model,
        pbre_dir=tmp_path / "pbre",
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=False,
        lm_head_chunk=16,
    )
    mmap = run_mode(
        mode="mmap",
        model_dir=model,
        pbre_dir=tmp_path / "pbre",
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=False,
        lm_head_chunk=16,
    )
    assert np.array_equal(full["logits"], mmap["logits"])
    assert full["summary"]["exact"] == "PASS"
    assert mmap["summary"]["exact"] == "PASS"


def test_tiny_pbre_sha_and_logits(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    specs = select_specs(model)
    pbre_dir = tmp_path / "pbre"
    index = encode_pbre_dir(specs, pbre_dir, skip_keys=set())
    assert index["n_tensors"] == len(specs)
    for key, rec in index["tensors"].items():
        src = load_uint16(specs[key])
        assert rec["sha256"] == sha256_words(src)
        assert_exact(src, src, label=key)
    ids = _ids()
    full = run_mode(
        mode="full",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=True,
        lm_head_chunk=16,
    )
    pbre = run_mode(
        mode="pbre",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=True,
        lm_head_chunk=16,
    )
    assert np.array_equal(full["logits"], pbre["logits"])
    assert pbre["summary"]["pbre_exact_fail"] == 0
    assert pbre["summary"]["pbre_exact_checks"] > 0
    assert pbre["summary"]["exact"] == "PASS"


def test_mmap_caps_resident_tensors(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    specs = select_specs(model)
    mmap = MmapSafetensorsSource(specs, max_resident=1)
    keys = [k for k in specs if k.endswith(".weight")][:4]
    for key in keys:
        mmap.load(key)
    assert len(mmap._hold) <= 1
    ram = FullRamSource(specs)
    assert len(ram._data) == len(specs)
    assert len(ram._data) > len(mmap._hold)


def test_cli_in_process_tiny(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    json_out = tmp_path / "disk_ram_tunnel.json"
    md_out = tmp_path / "disk_ram_tunnel.md"
    rc = main(
        [
            "--model-dir",
            str(model),
            "--pbre-dir",
            str(tmp_path / "pbre"),
            "--mode",
            "all",
            "--tokens",
            "3",
            "--lm-head-chunk",
            "16",
            "--in-process",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ]
    )
    assert rc == 0
    text = md_out.read_text(encoding="utf-8")
    assert "Disk–RAM tunnel" in text
    assert "27B" not in text or "Not a" in text or "not a" in text.lower()
    report = format_markdown({"modes": {}, "logits_match": "n/a", "n_tokens": 3, "n_layers": 2})
    assert "≤8 BPW" in report or "8 BPW" in report
