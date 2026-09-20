"""PBR-E product CLI: compress / decompress / verify on a tiny fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.safetensors_io import inventory_model, load_uint16, write_uint16_safetensors
from pbr_encoder.product import DISCLAIMER, compress, decompress, main, verify
from pbr_encoder.verification import assert_exact
from pbr_encoder.zoo_vs_pbre import analyze, format_markdown
from pbr_encoder.mantissa_zoo import DISCLAIMER as ZOO_DISCLAIMER


def _fixture(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 64
    exp = rng.choice(np.array([120, 121, 126, 127, 128], dtype=np.uint16), size=(n, n))
    words = make_bf16_bits(
        rng.integers(0, 2, size=(n, n), dtype=np.uint16),
        exp,
        rng.integers(0, 128, size=(n, n), dtype=np.uint16),
    )
    spec = np.resize(special_payload_words(), 16)
    words[0, :16] = make_bf16_bits(np.zeros(16, np.uint16), np.full(16, 127, np.uint16), spec & 0x7F)
    vec = make_bf16_bits(
        np.zeros(32, np.uint16),
        np.full(32, 127, np.uint16),
        rng.integers(0, 128, size=32, dtype=np.uint16),
    )
    model = tmp_path / "model"
    model.mkdir()
    write_uint16_safetensors(
        model / "model.safetensors",
        {
            "model.layers.0.mlp.down_proj.weight": words,
            "model.layers.0.input_layernorm.weight": vec,
        },
        storage_dtype="BF16",
    )
    (model / "config.json").write_text('{"hidden_size": 64}\n', encoding="utf-8")
    return model


def test_compress_decompress_verify_bundle(tmp_path: Path) -> None:
    model = _fixture(tmp_path)
    bundle = tmp_path / "out"
    summary = compress(model_dir=model, output=bundle, block_size=256)
    assert summary["exact"] == "PASS"
    assert summary["encoded_bytes"] == (bundle / "weights.pbr").stat().st_size
    assert summary["bpw"] < 16.0
    assert summary["bpw"] > 4.0
    assert "config.json" in summary["sidecars"]
    assert DISCLAIMER.split()[0] in summary["disclaimer"]

    restored = tmp_path / "restored"
    decompress(archive=bundle, output_dir=restored)
    assert (restored / "model.safetensors").exists()
    assert (restored / "config.json").exists()

    v = verify(archive=bundle, model_dir=model)
    assert v["exact"] == "PASS"
    assert v["n_tensors"] == 2
    specs = inventory_model(model)
    recs = {s.name: load_uint16(s) for s in inventory_model(restored)}
    for spec in specs:
        assert_exact(load_uint16(spec), recs[spec.name], label=spec.name)


def test_compress_single_pbr_file(tmp_path: Path) -> None:
    model = _fixture(tmp_path)
    dest = tmp_path / "weights.pbr"
    summary = compress(model_dir=model, output=dest, copy_sidecars=False, block_size=64)
    assert dest.is_file()
    assert dest.stat().st_size == summary["encoded_bytes"]
    out = tmp_path / "plain"
    decompress(archive=dest, output_dir=out)
    verify(archive=dest, model_dir=model)


def test_cli_verify_roundtrip(tmp_path: Path) -> None:
    model = _fixture(tmp_path)
    bundle = tmp_path / "cli"
    rc = main(["compress", "--model-dir", str(model), "-o", str(bundle), "--block-size", "128"])
    assert rc == 0
    rc = main(["verify", str(bundle), "--model-dir", str(model)])
    assert rc == 0
    rc = main(["decompress", str(bundle), "-o", str(tmp_path / "d")])
    assert rc == 0


def test_zoo_vs_pbre_markdown_no_false_four() -> None:
    report = {
        "summary": {
            "n_tensors": 42,
            "n_words": 1000,
            "n_hold": 200,
            "sign_bpw_raw": 1.0,
            "exp_complete_bpw": 2.65,
            "header_bpw": 0.0001,
            "pbre_total_bpw": 10.65,
            "pbre_ref_fullset": 10.616,
            "mixture_used": ["exp_cond", "gbdt"],
            "model_bytes": {"exp_cond": 65536, "gbdt": 1588},
            "methods": [
                {
                    "name": "mixture_argmin",
                    "nll_bits": 1388.0,
                    "model_bytes": 67124,
                    "ideal_bpw": 6.94,
                    "holdout_charged_bpw": 6.96,
                    "complete_bpw": 6.94,
                    "total_complete_bpw": 10.585,
                },
                {
                    "name": "phase_a_H(M|exp)",
                    "complete_bpw": 6.939,
                    "total_complete_bpw": 10.586,
                    "ideal_bpw": 6.933,
                    "holdout_charged_bpw": 6.96,
                    "nll_bits": 1386.0,
                    "model_bytes": 65536,
                },
                {
                    "name": "uncond_rANS",
                    "complete_bpw": 6.972,
                    "total_complete_bpw": 10.619,
                    "ideal_bpw": 6.972,
                    "holdout_charged_bpw": 6.972,
                    "nll_bits": 1394.0,
                    "model_bytes": 388,
                },
                {
                    "name": "raw_M",
                    "complete_bpw": 7.0,
                    "total_complete_bpw": 10.65,
                    "ideal_bpw": 7.0,
                    "holdout_charged_bpw": 7.0,
                    "nll_bits": 1400.0,
                    "model_bytes": 0,
                },
            ],
            "per_tensor_mixture": [
                {"name": "a", "choice": "exp_cond", "n_hold": 180},
                {"name": "b", "choice": "gbdt", "n_hold": 20},
            ],
        }
    }
    a = analyze(report)
    md = format_markdown(a)
    assert "dead-end" in md
    assert "≤4" in md or "not ≤4" in DISCLAIMER
    assert "10.616" in md
    assert a["accounting_verdict"]["toward_8bpw"] == "dead-end micro-gain"
    assert a["mixture_winner"]["tensors_by_choice"]["exp_cond"] == 1
    assert a["mixture_winner"]["original_bytes_by_choice"]["exp_cond"] == 1800
    assert "tables" in md.lower()
    assert ZOO_DISCLAIMER.split()[0] in ZOO_DISCLAIMER
