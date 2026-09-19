"""Codec unit tests: intended patterns, raw fallback, cost-based selection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pbr_codecs.constant import ConstantCodec
from pbr_codecs.raw import RawCodec
from pbr_codecs.residual import best_residual, decode_residuals
from pbr_codecs.value_dictionary import ValueDictCodec
from pbr_codecs.xor_predictor import PrevRowCodec, PrevValueCodec
from pbr_core.types import EncodeContext
from pbr_encoder.controlled_data import generate_case
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor, mode_usage
from pbr_encoder.mode_search import collect_candidates, select_best
from pbr_encoder.verification import assert_exact


def test_raw_is_always_available() -> None:
    words = np.arange(32, dtype=np.uint16).reshape(4, 8)
    enc = RawCodec().encode(words)
    enc.rows, enc.cols = 4, 8
    restored = RawCodec().decode(enc)
    assert_exact(words, restored, label="raw")


def test_constant_codec_only_when_uniform() -> None:
    codec = ConstantCodec()
    same = np.full((8, 8), 0x3E00, dtype=np.uint16)
    mixed = same.copy()
    mixed[0, 0] = 1
    enc = codec.encode(same)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    assert_exact(same, codec.decode(enc), label="constant")
    assert codec.encode(mixed) is None


def test_value_dict_roundtrip() -> None:
    palette = np.array([0x0000, 0x3C00, 0xBC00], dtype=np.uint16)
    words = np.resize(palette, 64).reshape(8, 8)
    codec = ValueDictCodec()
    enc = codec.encode(words)
    assert enc is not None
    enc.rows, enc.cols = 8, 8
    assert_exact(words, codec.decode(enc), label="value_dict")


def test_prev_value_and_prev_row_roundtrip() -> None:
    rng = np.random.default_rng(11)
    words = rng.integers(0, 65536, size=(12, 10), dtype=np.uint16)
    for codec in (PrevValueCodec(), PrevRowCodec()):
        enc = codec.encode(words)
        enc.rows, enc.cols = 12, 10
        assert_exact(words, codec.decode(enc), label=codec.name)


def test_residual_default_and_dict_roundtrip() -> None:
    residuals = np.resize(np.array([0, 0, 0, 1, 2], dtype=np.uint16), 64).reshape(8, 8)
    name, payload = best_residual(residuals)
    assert name in {"constant_residual", "default_list", "default_bitmap", "residual_dict"}
    restored = decode_residuals(payload, 8, 8)
    assert_exact(residuals, restored, label=f"residual:{name}")


def test_mode_selection_uses_complete_byte_cost_not_mse() -> None:
    words = np.full((16, 16), 0x3E00, dtype=np.uint16)
    ctx = EncodeContext()
    candidates = collect_candidates(words, ctx)
    best = select_best(words, ctx)
    assert best.total_bytes == min(c.total_bytes for c in candidates)
    raw = next(c for c in candidates if c.mode_name == "raw_bf16")
    assert best.total_bytes < raw.total_bytes
    assert best.mode_name == "constant"


def test_random_control_uses_raw_or_bounded_overhead() -> None:
    words, _ = generate_case("random_uint16", n_words=4096, cols=64, seed=7)
    container = encode_tensor(words, name="random", block_size=256)
    blob = container.dumps()
    raw_bytes = int(words.size * 2)
    # Complete container includes headers; Gate 2 forbids unbounded expansion.
    assert len(blob) <= raw_bytes + raw_bytes // 10 + 4096
    usage = mode_usage(container)
    raw_tiles = usage.get("raw_bf16", {}).get("tiles", 0)
    assert raw_tiles / max(sum(v["tiles"] for v in usage.values()), 1) >= 0.8
    restored = decode_container(container)[0]
    assert_exact(words, restored, label="random_uint16")


def test_structured_cases_pick_non_raw_when_cheaper() -> None:
    for case in ("constant_block", "repeated_values", "previous_row", "repeated_residuals"):
        words, _ = generate_case(case, n_words=4096, cols=64, seed=7)
        container = encode_tensor(words, name=case, block_size=256)
        blob = container.dumps()
        assert len(blob) < words.size * 2, f"{case} should beat raw BF16"
        names = Counter(
            tile.mode_name.split("+", 1)[0] for tile in container.tensors[0].tiles
        )
        assert "raw_bf16" not in names or names["raw_bf16"] < sum(names.values())
        assert_exact(words, decode_container(container)[0], label=case)


def test_duplicate_blocks_use_references() -> None:
    words, _ = generate_case("duplicate_blocks", n_words=4096, cols=64, seed=7)
    container = encode_tensor(words, name="dups", block_size=256)
    usage = mode_usage(container)
    # Exact copies of the previous tile can win as duplicate_ref *or* as
    # ref_prev_tile + a constant-zero residual; selection is by complete bytes.
    assert any("duplicate_ref" in name or "ref_prev_tile" in name for name in usage), usage
    assert_exact(words, decode_container(container)[0], label="duplicate_blocks")
