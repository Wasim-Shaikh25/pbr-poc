"""Stage 1B selection and local-fixture runner. No multi-hundred-MB weights."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pbr_core.safetensors_io import write_uint16_safetensors
from pbr_encoder.hf_weights import inventory_from_dir, select_weight_specs
from pbr_encoder.poc1b import DISCLAIMER, run_poc1b


def _layer_tensors(layer: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        f"model.layers.{layer}.self_attn.q_proj.weight": rng.integers(0, 65536, (32, 32), dtype=np.uint16),
        f"model.layers.{layer}.mlp.down_proj.weight": rng.integers(0, 65536, (32, 64), dtype=np.uint16),
        f"model.layers.{layer}.input_layernorm.weight": rng.integers(0, 65536, (32,), dtype=np.uint16),
    }


def test_selects_stratified_linear_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    tensors: dict[str, np.ndarray] = {}
    for layer in (0, 1, 2, 3, 4):
        tensors.update(_layer_tensors(layer, rng))
    tensors["model.embed_tokens.weight"] = rng.integers(0, 65536, (128, 32), dtype=np.uint16)
    path = tmp_path / "toy.safetensors"
    write_uint16_safetensors(path, tensors, storage_dtype="BF16")
    specs = inventory_from_dir(tmp_path)
    # Each q_proj is 2048 B, down_proj 4096 B; skip 1-D norms and embeddings.
    chosen = select_weight_specs(specs, min_bytes=8_000, max_bytes=20_000)
    names = [s.name for s in chosen]
    assert any(".0." in n for n in names)
    assert any(n.endswith("q_proj.weight") or n.endswith("down_proj.weight") for n in names)
    assert all("embed" not in n and "norm" not in n for n in names)
    total = sum(s.nbytes for s in chosen)
    assert 8_000 <= total <= 20_000


def test_run_poc1b_on_local_fixture(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    tensors = {}
    for layer in (0, 3, 7):
        tensors.update(_layer_tensors(layer, rng))
    write_uint16_safetensors(tmp_path / "model.safetensors", tensors, storage_dtype="BF16")
    specs = select_weight_specs(
        inventory_from_dir(tmp_path),
        min_bytes=4_000,
        max_bytes=30_000,
    )
    out = tmp_path / "reports"
    rows = run_poc1b(
        model_dir=tmp_path,
        specs=specs,
        block_sizes=[64],
        output_dir=out,
        include_baselines=True,
        model_meta={"repo_id": "fixture", "license": "test", "note": "local fixture"},
    )
    data_rows = [r for r in rows if r["case"] != "TOTAL"]
    assert data_rows
    assert all(r["exact"] == "PASS" for r in data_rows)
    total = next(r for r in rows if r["case"] == "TOTAL")
    assert total["exact"] == "PASS"
    assert (out / "summary.json").exists()
    assert DISCLAIMER in (out / "console_report.txt").read_text(encoding="utf-8")
    # Reported encoded size matches the on-disk complete container.
    first = data_rows[0]
    encoded = out / first["case"] / "encoded.pbr"
    assert encoded.exists()
    assert encoded.stat().st_size == first["encoded_bytes"]


@pytest.mark.skipif(os.environ.get("PBR_LIVE_HF") != "1", reason="live Hugging Face download disabled")
def test_live_download_optional() -> None:
    pytest.importorskip("huggingface_hub")
    from pbr_encoder.hf_weights import download_checkpoint

    dest = Path("outputs/models")
    cache = Path("outputs/hf_cache")
    result = download_checkpoint(
        local_dir=dest,
        cache_dir=cache,
        allow_fallback=True,
    )
    assert result.local_dir.exists()
    assert list(result.local_dir.glob("*.safetensors")) or list(result.local_dir.glob("**/*.safetensors"))
