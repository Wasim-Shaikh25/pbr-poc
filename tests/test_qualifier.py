"""Stage 2 qualification: bands, projection math, tiny fixture scan."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pbr_core.safetensors_io import TensorSpec, inventory_model, write_uint16_safetensors
from pbr_qualifier.bands import qualification_band, qualification_decision
from pbr_qualifier.projection import combine_projections, project_from_sample, weighted_model_bpw
from pbr_qualifier.scanner import scan_model, scan_tensor


def test_band_boundaries() -> None:
    assert qualification_band(8.01) == "not_exceptional"
    assert qualification_band(8.0) == "strong"
    assert qualification_band(4.01) == "strong"
    assert qualification_band(4.0) == "exceptional"
    assert qualification_band(2.01) == "exceptional"
    assert qualification_band(2.0) == "one_gb_class"
    assert qualification_band(1.01) == "one_gb_class"
    assert qualification_band(1.0) == "extreme"
    assert qualification_band(0.5) == "extreme"


def test_stage2_never_one_gb_qualifies() -> None:
    for bpw in (0.5, 1.5, 3.0, 6.0, 13.6):
        decision = qualification_decision(bpw)
        assert decision.one_gb_qualified is False
        assert "full-container" in decision.one_gb_qualified_reason
    assert qualification_decision(3.9).high_potential is True
    assert qualification_decision(4.0).high_potential is True
    assert qualification_decision(4.01).high_potential is False
    assert qualification_decision(13.6).high_potential is False


def test_project_from_sample_scales_tiles_and_keeps_one_header() -> None:
    # 2 sample tiles / 100 bytes → 50 B/tile; 10 full tiles + 20 B header.
    assert (
        project_from_sample(
            n_words=1000,
            n_full_tiles=10,
            sample_tile_bytes=100,
            n_sample_tiles=2,
            container_header_bytes=20,
        )
        == 520
    )
    assert (
        project_from_sample(
            n_words=100,
            n_full_tiles=0,
            sample_tile_bytes=0,
            n_sample_tiles=0,
            container_header_bytes=8,
        )
        == 200
    )


def test_weighted_model_bpw_formula() -> None:
    assert weighted_model_bpw(projected_encoded_bytes=400, total_parameter_count=400) == 8.0
    assert weighted_model_bpw(projected_encoded_bytes=800, total_parameter_count=400) == 16.0
    words, encoded, bpw = combine_projections([(100, 50)], 100, unscanned_bpw=16.0)
    assert words == 200
    assert encoded == 250
    assert bpw == pytest.approx(10.0)


def test_skips_non16_without_conversion(tmp_path: Path) -> None:
    spec = TensorSpec(
        name="fp32.weight",
        dtype="F32",
        shape=(4, 4),
        data_offsets=(0, 64),
        file_path=tmp_path / "missing.safetensors",
        data_start=0,
    )
    row = scan_tensor(
        spec, block_size=16, sample_tiles=2, max_full_words=64, entropy_words=64
    )
    assert row.scanned is False
    assert row.exact == "SKIP"
    assert "not converted" in (row.skip_reason or "")


def test_fixture_scan_constant_and_random(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": np.full((32, 64), np.uint16(0x3E00)),
        "model.layers.1.mlp.down_proj.weight": rng.integers(0, 65536, (32, 64), dtype=np.uint16),
        "model.norm.weight": np.arange(64, dtype=np.uint16),
    }
    write_uint16_safetensors(tmp_path / "toy.safetensors", tensors, storage_dtype="BF16")
    specs = inventory_model(tmp_path)
    report = scan_model(specs, block_size=64, sample_tiles=4, max_full_words=4096)
    by_name = {t["name"]: t for t in report["tensors"]}
    assert all(t["exact"] == "PASS" for t in report["tensors"])
    assert report["projection"]["one_gb_qualified"] is False
    assert report["projection"]["total_16bit_parameters"] == 32 * 64 * 2 + 64
    const = by_name["model.layers.0.mlp.down_proj.weight"]
    rnd = by_name["model.layers.1.mlp.down_proj.weight"]
    # Tile headers keep even a constant 2k-word tensor above 4 BPW; it must
    # still beat unstructured data, which stays at raw-plus-overhead.
    assert const["winning_modes"].startswith("constant")
    assert const["projected_bpw"] < rnd["projected_bpw"] / 2
    assert rnd["projected_bpw"] > 16.0
    assert report["projection"]["projected_bpw"] > 4.0
    assert report["projection"]["high_potential"] is False
    assert report["projection"]["band"] == "not_exceptional"


@pytest.mark.skipif(os.environ.get("PBR_LIVE_HF") != "1", reason="live Hugging Face download disabled")
def test_live_qualifier_optional() -> None:
    pytest.importorskip("huggingface_hub")
    from pbr_qualifier.cli import main

    assert (
        main(
            [
                "--model-dir",
                "outputs/models/Qwen__Qwen2.5-0.5B-Instruct",
                "--output-dir",
                "outputs/reports/poc2_live_test",
            ]
        )
        in (0, 2)
    )
