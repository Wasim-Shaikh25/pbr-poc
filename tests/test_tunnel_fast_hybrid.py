"""Fast PBR-E decode and hybrid int4 tunnel: tiny fixtures, no HF download."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pbr_core.rans import rans_decode_c, rans_decode_python, rans_encode, rans_impl, table_from_symbols
from pbr_encoder.disk_tunnel import (
    encode_pbre_dir,
    run_mode,
    select_specs,
    write_tiny_qwen_fixture,
)
from pbr_encoder.hybrid_lossy import (
    DISCLAIMER as HYBRID_DISCLAIMER,
    dequant_int4,
    encode_hybrid_dir,
    is_sensitive_exact,
    quality_metrics,
    quantize_int4,
    run_hybrid_mode,
)
from pbr_encoder.verification import assert_exact


def test_c_rans_matches_python_and_source() -> None:
    rng = np.random.default_rng(4)
    symbols = rng.choice(np.arange(10, dtype=np.uint8), size=8192)
    freq = table_from_symbols(symbols)
    blob = rans_encode(symbols, freq)
    py = rans_decode_python(blob, int(symbols.size), freq)
    c = rans_decode_c(blob, int(symbols.size), freq)
    assert np.array_equal(py, symbols)
    assert np.array_equal(c, symbols)
    assert rans_impl() in {"c", "python"}


def test_pbre_slow_env_forces_python() -> None:
    prev = os.environ.get("PBR_RANS_IMPL")
    os.environ["PBR_RANS_IMPL"] = "python"
    try:
        assert rans_impl() == "python"
    finally:
        if prev is None:
            os.environ.pop("PBR_RANS_IMPL", None)
        else:
            os.environ["PBR_RANS_IMPL"] = prev


def test_fast_pbre_tiny_logits(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    specs = select_specs(model)
    pbre_dir = tmp_path / "pbre"
    encode_pbre_dir(specs, pbre_dir, skip_keys=set())
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    full = run_mode(
        mode="full",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=False,
        lm_head_chunk=16,
    )
    fast = run_mode(
        mode="pbre_fast",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=True,
        lm_head_chunk=16,
    )
    faster = run_mode(
        mode="pbre_faster",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=True,
        lm_head_chunk=16,
    )
    assert np.array_equal(full["logits"], fast["logits"])
    assert np.array_equal(full["logits"], faster["logits"])
    assert fast["summary"]["exact"] == "PASS"
    assert faster["summary"]["exact"] == "PASS"
    assert fast["summary"]["pbre_exact_checks"] > 0
    assert faster["summary"]["pbre_exact_checks"] > 0


def test_int4_roundtrip_bounded() -> None:
    rng = np.random.default_rng(0)
    fp = rng.normal(0, 0.5, size=(32, 64)).astype(np.float32)
    packed, scales, n = quantize_int4(fp, group_size=64)
    rec = dequant_int4(packed, scales, fp.shape, 64, n)
    assert rec.shape == fp.shape
    # 4-bit with 7-level scale cannot be exact; error is finite.
    assert float(np.max(np.abs(rec - fp))) < 0.5


def test_sensitive_exact_roles() -> None:
    assert is_sensitive_exact("model.embed_tokens.weight", 24)
    assert is_sensitive_exact("model.norm.weight", 24)
    assert is_sensitive_exact("model.layers.0.mlp.down_proj.weight", 24)
    assert is_sensitive_exact("model.layers.23.mlp.up_proj.weight", 24)
    assert is_sensitive_exact("model.layers.3.input_layernorm.weight", 24)
    assert not is_sensitive_exact("model.layers.3.mlp.down_proj.weight", 24)
    assert not is_sensitive_exact("model.layers.10.self_attn.q_proj.weight", 24)


def test_hybrid_tiny_lossy_not_claimed_exact(tmp_path: Path) -> None:
    model = tmp_path / "model"
    write_tiny_qwen_fixture(model, n_layers=3, vocab=64)
    specs = select_specs(model)
    hy = tmp_path / "hybrid"
    index = encode_hybrid_dir(specs, hy, n_layers=3)
    assert index["n_int4"] > 0
    assert index["n_exact"] > 0
    ids = np.array([1, 2, 3], dtype=np.int64)
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
    hy_out = run_hybrid_mode(
        model_dir=model,
        hybrid_dir=hy,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        lm_head_chunk=16,
    )
    q = quality_metrics(full["logits"], hy_out["logits"])
    assert q["lossy"] is True
    assert np.isfinite(q["max_abs"])
    assert np.isfinite(q["mean_kl_ref_to_hyp"])
    assert hy_out["summary"]["exact"] == "LOSSY"
    assert "not bit-exact" in HYBRID_DISCLAIMER.lower() or "LOSSY" in HYBRID_DISCLAIMER
    # exact tensors in the pack round-trip
    from pbr_core.safetensors_io import load_uint16

    for key, rec in index["tensors"].items():
        if rec["kind"] != "exact_bf16":
            continue
        src = load_uint16(specs[key])
        blob = (hy / rec["file"]).read_bytes()
        got = np.frombuffer(blob, dtype="<u2").reshape(src.shape)
        assert_exact(src, got, label=key)


def test_pbre_faster_warm_tiny_logits(tmp_path: Path) -> None:
    from pbr_core.container import PBRContainer
    from pbr_encoder.decoder import decode_container, decode_pbr_blob
    from pbr_encoder.disk_tunnel import materialize_decoded_cache

    model = tmp_path / "model"
    write_tiny_qwen_fixture(model)
    specs = select_specs(model)
    pbre_dir = tmp_path / "pbre"
    encode_pbre_dir(specs, pbre_dir, skip_keys=set())
    for key, rec in json_tensors(pbre_dir).items():
        blob = (pbre_dir / rec["file"]).read_bytes()
        logical = tuple(int(x) for x in rec["shape"])
        fast = decode_pbr_blob(blob, logical)
        slow = decode_container(PBRContainer.loads(blob))[0].reshape(logical)
        assert_exact(fast, slow, label=f"blob:{key}")
    dest = tmp_path / "decoded"
    meta = materialize_decoded_cache(pbre_dir, dest, workers=2, verify_sha=True)
    assert meta["n_tensors"] == len(json_tensors(pbre_dir))
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    full = run_mode(
        mode="full",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=False,
        lm_head_chunk=16,
    )
    warm = run_mode(
        mode="pbre_faster_warm",
        model_dir=model,
        pbre_dir=pbre_dir,
        token_ids=ids,
        n_layers=None,
        encode_if_missing=False,
        verify_pbre=True,
        lm_head_chunk=16,
        decoded_dir=dest,
    )
    assert np.array_equal(full["logits"], warm["logits"])
    assert warm["summary"]["exact"] == "PASS"


def json_tensors(pbre_dir: Path) -> dict:
    import json

    return json.loads((pbre_dir / "index.json").read_text(encoding="utf-8"))["tensors"]
