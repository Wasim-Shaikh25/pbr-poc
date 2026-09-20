"""Matrix mantissa V1–V3: unit tests + pre-Qwen gates."""

from __future__ import annotations

import pytest
import numpy as np

from pbr_matrix_mantissa.bitio import pack_lsb, unpack_lsb
from pbr_matrix_mantissa.codec import decode_mantissa, encode_mantissa
from pbr_matrix_mantissa.errors import CodecError
from pbr_matrix_mantissa.gates import run_all_gates
from pbr_matrix_mantissa.predict import paeth, predict, PRED_LEFT, PRED_PAETH
from pbr_matrix_mantissa.residual import residual, restore


def test_pack_roundtrip() -> None:
    rng = np.random.default_rng(3)
    m = rng.integers(0, 128, size=256, dtype=np.uint8)
    packed = pack_lsb(m, 7)
    assert unpack_lsb(packed, 256, 7).tolist() == m.tolist()


def test_residual_mod128_all_pairs_sample() -> None:
    for pred in (0, 1, 64, 127):
        for actual in (0, 1, 63, 64, 127):
            assert restore(pred, residual(actual, pred)) == actual


def test_paeth_integer() -> None:
    assert paeth(10, 10, 10) == 10
    assert paeth(3, 8, 3) == 8 or paeth(3, 8, 3) in {3, 8, 3}


def test_predict_left_unavailable_is_zero() -> None:
    assert (
        predict(
            PRED_LEFT,
            left=9,
            up=4,
            up_left=1,
            prev=7,
            exp=120,
            left_exp=120,
            has_left=False,
            has_up=True,
            has_up_left=False,
        )
        == 0
    )


def test_raw_omits_directory_on_uniform() -> None:
    rng = np.random.default_rng(4)
    m = rng.integers(0, 128, size=(32, 32), dtype=np.uint8)
    enc = encode_mantissa(m, version=1)
    assert enc.strategy == "ALL_RAW"
    assert enc.n_tiles == 0
    assert enc.needs_exp is False


def test_corrupt_magic_raises() -> None:
    with pytest.raises(CodecError):
        decode_mantissa(b"nope")


def test_gates_pass() -> None:
    report = run_all_gates()
    failed = []
    for gname, gate in report["gates"].items():
        for check in gate["checks"]:
            if not check["passed"]:
                failed.append(f"{gname}:{check['name']}: {check.get('detail')}")
    assert report["passed"], "\n".join(failed)
