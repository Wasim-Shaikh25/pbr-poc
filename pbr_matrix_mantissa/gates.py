"""Pre-Qwen qualification gates A–D for the matrix mantissa codec."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pbr_matrix_mantissa.bf16 import special_payload_words
from pbr_matrix_mantissa.codec import (
    DISCLAIMER,
    FLAG_NEEDS_EXP,
    MAGIC,
    PREDS,
    SHAPES,
    TRAVS,
    VERSIONS,
    decode_bf16_matrix,
    decode_mantissa,
    encode_all_raw,
    encode_bf16_matrix,
    encode_mantissa,
    sha256_words,
    _HDR,
)
from pbr_matrix_mantissa.errors import CodecError
from pbr_matrix_mantissa.predict import (
    PRED_AVG,
    PRED_EXP_LEFT,
    PRED_LEFT,
    PRED_NAMES,
    PRED_PAETH,
    PRED_RAW,
    PRED_UP,
)
from pbr_matrix_mantissa.residual import residual, restore
from pbr_matrix_mantissa.traverse import TRAV_NAMES


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateResult:
    gate: str
    passed: bool
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))
        if not passed:
            self.passed = False


def _ok(cond: bool, msg: str = "") -> tuple[bool, str]:
    return bool(cond), msg


def gate_a() -> GateResult:
    g = GateResult("A", True)
    # 128×128 residual pairs
    bad = 0
    for pred in range(128):
        for actual in range(128):
            if restore(pred, residual(actual, pred)) != actual:
                bad += 1
    g.add(
        "residual_pairs_128x128",
        bad == 0,
        "16384 pairs (pred, actual) in 0..127; restore(pred, (actual-pred) mod 128)",
    )

    rng = np.random.default_rng(0)
    words = rng.integers(0, 65536, size=(256, 256), dtype=np.uint16)
    for ver in VERSIONS:
        bundle = encode_bf16_matrix(words, version=ver)
        rec = decode_bf16_matrix(bundle.blob)
        sha_ok = sha256_words(rec) == sha256_words(words) and np.array_equal(rec, words)
        g.add(
            f"bf16_roundtrip_65536_v{ver}",
            sha_ok,
            f"complete_bytes={bundle.complete_bytes} strategy={bundle.mantissa.strategy} "
            f"sha={bundle.sha256[:16]}",
        )

    spec = special_payload_words()
    grid = np.resize(spec, 16 * 16).reshape(16, 16)
    bun = encode_bf16_matrix(grid, version=3)
    rec = decode_bf16_matrix(bun.blob)
    g.add(
        "specials_no_fp",
        np.array_equal(rec, grid) and sha256_words(rec) == sha256_words(grid),
        "±0, ±Inf, NaN payloads, subnormals as uint16",
    )

    boundaries = [(1, 1), (1, 17), (17, 1), (15, 17), (20, 20), (33, 31), (16, 32), (32, 8)]
    b_fail = []
    for h, w in boundaries:
        mant = rng.integers(0, 128, size=(h, w), dtype=np.uint8)
        exp = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        enc = encode_mantissa(mant, version=3, exp=exp)
        got = decode_mantissa(enc.blob, exp=exp)
        if not np.array_equal(got, mant):
            b_fail.append((h, w))
    g.add("boundary_lengths", not b_fail, f"failed={b_fail}" if b_fail else f"n={len(boundaries)}")

    cand_fail = []
    n_cands = 0
    probe = rng.integers(0, 128, size=(24, 24), dtype=np.uint8)
    exp = rng.integers(90, 140, size=(24, 24), dtype=np.uint8)
    for ver in VERSIONS:
        for shape in SHAPES[ver]:
            for pred in (PRED_RAW,) + PREDS[ver]:
                for trav in TRAVS[ver]:
                    n_cands += 1
                    try:
                        enc = encode_mantissa(
                            probe,
                            version=ver,
                            exp=exp,
                            force_shape=shape,
                            force_pred=pred,
                            force_trav=trav,
                            allow_all_raw=pred == PRED_RAW,
                        )
                        got = decode_mantissa(enc.blob, exp=exp)
                        if not np.array_equal(got, probe):
                            cand_fail.append((ver, shape, PRED_NAMES[pred], TRAV_NAMES[trav], "mismatch"))
                    except Exception as exc:  # noqa: BLE001 — gate must record any candidate error
                        cand_fail.append((ver, shape, PRED_NAMES[pred], TRAV_NAMES[trav], repr(exc)))
                    if pred == PRED_RAW:
                        break  # RAW ignores traversal
    g.add(
        "every_candidate_exact",
        not cand_fail,
        f"n_cands={n_cands} failed={cand_fail[:8]}",
    )
    return g


def gate_b() -> GateResult:
    g = GateResult("B", True)
    rng = np.random.default_rng(1)
    n = 64
    exp = np.full((n, n), 120, dtype=np.uint8)

    uniform = rng.integers(0, 128, size=(n, n), dtype=np.uint8)
    enc_u = encode_mantissa(uniform, version=3, exp=exp)
    raw_u = encode_all_raw(uniform, version=3)
    raw_wins = enc_u.strategy == "ALL_RAW" or (
        enc_u.pred_counts.get("RAW", 0) == enc_u.n_tiles and enc_u.n_tiles > 0
    )
    g.add(
        "uniform_raw_wins",
        raw_wins and enc_u.complete_bytes <= raw_u.complete_bytes,
        f"strategy={enc_u.strategy} pred_counts={enc_u.pred_counts} "
        f"bytes={enc_u.complete_bytes} all_raw={raw_u.complete_bytes}",
    )

    constant = np.full((n, n), 41, dtype=np.uint8)
    enc_c = encode_mantissa(constant, version=1, exp=exp)
    struct_preds = {"LEFT", "UP", "PREVIOUS", "AVG", "PAETH", "EXP_LEFT"}
    used = set(enc_c.pred_counts)
    g.add(
        "constant_predictor_wins",
        enc_c.strategy == "TILED" and bool(used & struct_preds) and "RAW" not in used,
        f"strategy={enc_c.strategy} pred_counts={enc_c.pred_counts} bytes={enc_c.complete_bytes}",
    )

    # value constant down each column → UP
    row_ramp = np.broadcast_to(np.arange(n, dtype=np.uint8) & np.uint8(0x7F), (n, n)).copy()
    enc_up = encode_mantissa(
        row_ramp, version=1, exp=exp, force_pred=PRED_UP, force_trav=0, allow_all_raw=False, force_shape=(16, 16)
    )
    enc_left = encode_mantissa(
        row_ramp, version=1, exp=exp, force_pred=PRED_LEFT, force_trav=0, allow_all_raw=False, force_shape=(16, 16)
    )
    auto_ramp = encode_mantissa(row_ramp, version=1, exp=exp)
    g.add(
        "row_ramp_exposes_UP",
        enc_up.complete_bytes < enc_left.complete_bytes and "UP" in auto_ramp.pred_counts,
        f"UP={enc_up.complete_bytes} LEFT={enc_left.complete_bytes} auto={auto_ramp.pred_counts}",
    )

    # value constant across each row → LEFT
    col_ramp = np.broadcast_to((np.arange(n, dtype=np.uint8) & np.uint8(0x7F))[:, None], (n, n)).copy()
    enc_l2 = encode_mantissa(
        col_ramp, version=1, exp=exp, force_pred=PRED_LEFT, force_trav=0, allow_all_raw=False, force_shape=(16, 16)
    )
    enc_u2 = encode_mantissa(
        col_ramp, version=1, exp=exp, force_pred=PRED_UP, force_trav=0, allow_all_raw=False, force_shape=(16, 16)
    )
    auto_col = encode_mantissa(col_ramp, version=1, exp=exp)
    g.add(
        "col_ramp_exposes_LEFT",
        enc_l2.complete_bytes < enc_u2.complete_bytes and "LEFT" in auto_col.pred_counts,
        f"LEFT={enc_l2.complete_bytes} UP={enc_u2.complete_bytes} auto={auto_col.pred_counts}",
    )

    # V2 AVG / V3 Paeth still exact on a bilinear-ish field
    field = ((np.arange(n)[:, None] + np.arange(n)[None, :]) & 127).astype(np.uint8)
    enc_avg = encode_mantissa(
        field, version=2, exp=exp, force_pred=PRED_AVG, force_shape=(8, 8), allow_all_raw=False
    )
    enc_pa = encode_mantissa(
        field, version=3, exp=exp, force_pred=PRED_PAETH, force_shape=(32, 32), allow_all_raw=False
    )
    g.add("v2_AVG_exact", np.array_equal(decode_mantissa(enc_avg.blob, exp=exp), field), enc_avg.strategy)
    g.add("v3_PAETH_exact", np.array_equal(decode_mantissa(enc_pa.blob, exp=exp), field), enc_pa.strategy)

    # exp-conditioned LEFT: mantissa follows left only when exp matches
    el = np.zeros((32, 32), dtype=np.uint8)
    ee = np.zeros((32, 32), dtype=np.uint8)
    for r in range(32):
        for c in range(32):
            ee[r, c] = 100 if (c % 4) else 130
            el[r, c] = (el[r, c - 1] if c and ee[r, c] == ee[r, c - 1] else (3 * c + r) & 127)
    enc_el = encode_mantissa(
        el, version=3, exp=ee, force_pred=PRED_EXP_LEFT, force_shape=(16, 16), allow_all_raw=False
    )
    g.add(
        "v3_EXP_LEFT_exact",
        np.array_equal(decode_mantissa(enc_el.blob, exp=ee), el) and enc_el.needs_exp,
        f"bytes={enc_el.complete_bytes} needs_exp={enc_el.needs_exp}",
    )
    return g


def gate_c() -> GateResult:
    g = GateResult("C", True)
    rng = np.random.default_rng(2)
    uniform = rng.integers(0, 128, size=(48, 48), dtype=np.uint8)
    enc = encode_mantissa(uniform, version=3)
    magic, ver, rows, cols, n_tiles, flags = _HDR.unpack_from(enc.blob, 0)
    g.add(
        "all_raw_omits_tile_directory",
        enc.strategy == "ALL_RAW" and n_tiles == 0 and flags == 0,
        f"strategy={enc.strategy} n_tiles={n_tiles} flags={flags} bytes={enc.complete_bytes} hdr={magic!r}/{ver}",
    )
    g.add(
        "unused_exp_dep_omitted",
        not enc.needs_exp and (flags & FLAG_NEEDS_EXP) == 0,
        "EXP_LEFT not charged on uniform ALL_RAW",
    )
    rec = decode_mantissa(enc.blob)
    g.add("complete_bytes_match_blob", enc.complete_bytes == len(enc.blob) and np.array_equal(rec, uniform))

    # RAW wins ties: 7-bit residuals cannot beat RAW payload; extra residual-kind byte loses.
    noise = rng.integers(0, 128, size=(16, 16), dtype=np.uint8)
    tied = encode_mantissa(noise, version=1)
    g.add(
        "raw_wins_ties",
        tied.strategy == "ALL_RAW",
        f"strategy={tied.strategy} counts={tied.pred_counts} bytes={tied.complete_bytes}",
    )

    tail_m = rng.integers(0, 128, size=(20, 20), dtype=np.uint8)
    enc_t = encode_mantissa(tail_m, version=1, force_shape=(16, 16), allow_all_raw=False)
    got = decode_mantissa(enc_t.blob)
    g.add(
        "tails_preserved",
        np.array_equal(got, tail_m) and enc_t.n_tiles == 4,
        f"n_tiles={enc_t.n_tiles} expected 16x16 + 16x4 + 4x16 + 4x4",
    )

    # Structured constant: tiled predictors, no unused ALL_RAW plane in the blob (n_tiles>0)
    const = np.full((32, 32), 7, dtype=np.uint8)
    enc_k = encode_mantissa(const, version=1)
    g.add(
        "unused_all_raw_plane_omitted_when_tiled_wins",
        enc_k.strategy == "TILED" and enc_k.n_tiles > 0,
        f"strategy={enc_k.strategy} n_tiles={enc_k.n_tiles} bytes={enc_k.complete_bytes}",
    )
    return g


def gate_d() -> GateResult:
    g = GateResult("D", True)
    cases = {
        "empty": b"",
        "bad_magic": b"XXXX" + b"\x00" * 20,
        "truncated_header": MAGIC_HDR_ONLY(),
        "truncated_raw": encode_all_raw(np.arange(16, dtype=np.uint8).reshape(4, 4)).blob[:-2],
        "overrun_tile": _corrupt_tile_len(),
    }
    for name, blob in cases.items():
        raised = False
        try:
            decode_mantissa(blob)
        except CodecError:
            raised = True
        except Exception as exc:  # noqa: BLE001
            g.add(name, False, f"wrong exception {type(exc).__name__}: {exc}")
            continue
        g.add(name, raised, "CodecError" if raised else "did not raise")

    # EXP_LEFT blob without exp plane
    el = np.zeros((16, 16), dtype=np.uint8)
    ee = np.zeros((16, 16), dtype=np.uint8)
    for c in range(16):
        el[0, c] = c
        ee[0, c] = 120
    enc = encode_mantissa(el, version=3, exp=ee, force_pred=PRED_EXP_LEFT, force_shape=(16, 16), allow_all_raw=False)
    raised = False
    try:
        decode_mantissa(enc.blob, exp=None)
    except CodecError:
        raised = True
    g.add("exp_left_missing_exp", raised or not enc.needs_exp, f"needs_exp={enc.needs_exp}")
    return g


def MAGIC_HDR_ONLY() -> bytes:
    return MAGIC + b"\x03"


def _corrupt_tile_len() -> bytes:
    enc = encode_mantissa(
        np.full((16, 16), 3, dtype=np.uint8),
        version=1,
        force_shape=(16, 16),
        allow_all_raw=False,
    )
    if len(enc.blob) < _HDR.size + 12:
        return enc.blob[: _HDR.size]
    buf = bytearray(enc.blob)
    # payload_len is the last H of the first tile header
    buf[_HDR.size + 10 : _HDR.size + 12] = (65000).to_bytes(2, "little")
    return bytes(buf)


def run_all_gates() -> dict:
    gates = [gate_a(), gate_b(), gate_c(), gate_d()]
    passed = all(g.passed for g in gates)
    return {
        "disclaimer": DISCLAIMER,
        "passed": passed,
        "qwen_claimed": False,
        "bpw_4_claimed": False,
        "gates": {
            g.gate: {
                "passed": g.passed,
                "checks": [c.__dict__ for c in g.checks],
            }
            for g in gates
        },
    }

