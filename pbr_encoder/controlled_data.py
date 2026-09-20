"""Synthetic/controlled tensors with known structure. Not real model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pbr_core.bf16 import make_bf16_bits, special_payload_words
from pbr_core.hashing import sha256_words, words_to_bytes

CASES = (
    "repeated_values",
    "constant_block",
    "duplicate_blocks",
    "near_duplicate_blocks",
    "repeated_residuals",
    "previous_value_runs",
    "previous_row",
    "bf16_component_concentration",
    "special_payloads",
    "random_uint16",
)

EXPECTED_MODES = {
    "repeated_values": "value dictionary or constant block",
    "constant_block": "constant",
    "duplicate_blocks": "duplicate_ref after the first unique tile",
    "near_duplicate_blocks": "ref_prev_tile or residual default/list",
    "repeated_residuals": "const_pred + residual dictionary",
    "previous_value_runs": "prev_value + default residual",
    "previous_row": "prev_row + default/constant residual",
    "bf16_component_concentration": "bf16_components or value/residual codec",
    "special_payloads": "any exact mode; NaN payloads must survive",
    "random_uint16": "raw_bf16 (or no larger than raw + bounded overhead)",
}


def default_shape(n_words: int, cols: int) -> tuple[int, int]:
    if n_words % cols != 0:
        raise ValueError(f"n_words={n_words} is not divisible by cols={cols}")
    return n_words // cols, cols


def generate_case(
    case: str,
    *,
    n_words: int = 65536,
    cols: int = 256,
    block_size: int = 256,
    seed: int = 7,
) -> tuple[np.ndarray, dict]:
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}. Supported: {', '.join(CASES)}")
    rng = np.random.default_rng(seed)
    rows, cols = default_shape(n_words, cols)
    words = _build(case, rng, rows, cols, block_size)
    if words.size != n_words:
        raise RuntimeError(f"{case} produced {words.size} words, expected {n_words}")
    manifest = {
        "case": case,
        "seed": seed,
        "words": n_words,
        "shape": [int(words.shape[0]), int(words.shape[1])],
        "dtype": "uint16",
        "byte_count": int(words.size * 2),
        "sha256": sha256_words(words),
        "expected_pattern": EXPECTED_MODES[case],
        "construction": _construction(case),
        "block_size": block_size,
        "stage": "1A",
        "disclaimer": "Synthetic/controlled tensor. Not a real-model result.",
    }
    return words, manifest


def _build(case: str, rng: np.random.Generator, rows: int, cols: int, block_size: int) -> np.ndarray:
    n = rows * cols
    if case == "repeated_values":
        palette = np.array([0x3C00, 0xBC00, 0x3E00, 0x0000], dtype=np.uint16)
        return rng.choice(palette, size=(rows, cols), replace=True)
    if case == "constant_block":
        return np.full((rows, cols), np.uint16(0x3E00), dtype=np.uint16)
    if case == "duplicate_blocks":
        proto = rng.integers(0, 65536, size=block_size, dtype=np.uint16)
        repeats = n // block_size
        extra = n % block_size
        body = np.tile(proto, repeats)
        if extra:
            body = np.concatenate([body, proto[:extra]])
        return body.reshape(rows, cols)
    if case == "near_duplicate_blocks":
        proto = rng.integers(0, 65536, size=block_size, dtype=np.uint16)
        tiles = []
        n_tiles = n // block_size
        for i in range(n_tiles):
            tile = proto.copy()
            # Deterministic 4-position edits so residual structure is sparse.
            for k in range(4):
                tile[(i * 7 + k * 13) % block_size] ^= np.uint16(1 << (k % 16))
            tiles.append(tile)
        extra = n % block_size
        body = np.concatenate(tiles) if tiles else np.array([], dtype=np.uint16)
        if extra:
            body = np.concatenate([body, proto[:extra]])
        return body.reshape(rows, cols)
    if case == "repeated_residuals":
        base = np.uint16(0x3E00)
        cycle = np.array([0x0000, 0x0001, 0x0002], dtype=np.uint16)
        residuals = np.resize(cycle, n)
        return (base ^ residuals).reshape(rows, cols)
    if case == "previous_value_runs":
        run_len = 32
        n_runs = (n + run_len - 1) // run_len
        values = rng.integers(0, 65536, size=n_runs, dtype=np.uint16)
        flat = np.repeat(values, run_len)[:n]
        return flat.reshape(rows, cols)
    if case == "previous_row":
        row0 = rng.integers(0, 65536, size=cols, dtype=np.uint16)
        return np.broadcast_to(row0, (rows, cols)).copy()
    if case == "bf16_component_concentration":
        signs = rng.integers(0, 2, size=(rows, cols), dtype=np.uint16)
        exponents = rng.choice(np.array([120, 121, 127, 128], dtype=np.uint16), size=(rows, cols))
        mantissa = rng.integers(0, 128, size=(rows, cols), dtype=np.uint16)
        return make_bf16_bits(signs, exponents, mantissa)
    if case == "special_payloads":
        specials = special_payload_words()
        # Repeat specials, then fill the rest with a few ordinary BF16 patterns.
        reps = (n + specials.size - 1) // specials.size
        body = np.resize(specials, min(n, reps * specials.size))[: min(n, specials.size * 64)]
        fill = np.resize(np.array([0x3E00, 0x3E01, 0xBE00], dtype=np.uint16), n - body.size)
        return np.concatenate([body, fill]).reshape(rows, cols)
    if case == "random_uint16":
        return rng.integers(0, 65536, size=(rows, cols), dtype=np.uint16)
    raise AssertionError(case)


def _construction(case: str) -> str:
    return {
        "repeated_values": "Four exact uint16 BF16-like values sampled uniformly.",
        "constant_block": "Every word is the same exact uint16.",
        "duplicate_blocks": "One 256-word block repeated across the tensor.",
        "near_duplicate_blocks": "Repeated prototype block with four XOR bit-flips per copy.",
        "repeated_residuals": "Constant prototype XOR a cycling residual {0,1,2}.",
        "previous_value_runs": "Runs of 32 identical uint16 words.",
        "previous_row": "Every row is an exact copy of row 0.",
        "bf16_component_concentration": "Four exponents, random signs and 7-bit mantissas.",
        "special_payloads": "Signed zeros, infinities, NaN payloads, subnormals, then filler.",
        "random_uint16": "Uniform independent uint16 words (hard control).",
    }[case]


def write_case(words: np.ndarray, manifest: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest["case"]
    u16_path = output_dir / f"{stem}.u16"
    json_path = output_dir / f"{stem}.json"
    u16_path.write_bytes(words_to_bytes(words))
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return u16_path, json_path


def load_case(u16_path: Path) -> tuple[np.ndarray, dict]:
    manifest_path = u16_path.with_suffix(".json")
    data = u16_path.read_bytes()
    words_1d = np.frombuffer(data, dtype="<u2").copy()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shape = tuple(manifest["shape"])
        words = words_1d.reshape(shape)
        return words, manifest
    return words_1d.reshape(1, -1), {
        "case": u16_path.stem,
        "shape": [1, int(words_1d.size)],
        "sha256": sha256_words(words_1d),
        "byte_count": len(data),
        "disclaimer": "Synthetic/controlled tensor. Not a real-model result.",
    }


def generate_all(
    output_dir: Path,
    *,
    n_words: int = 65536,
    cols: int = 256,
    block_size: int = 256,
    seed: int = 7,
    cases: tuple[str, ...] | None = None,
) -> list[tuple[Path, Path]]:
    written = []
    for case in cases or CASES:
        words, manifest = generate_case(
            case, n_words=n_words, cols=cols, block_size=block_size, seed=seed
        )
        written.append(write_case(words, manifest, output_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1A controlled uint16 tensors (not real model weights)."
    )
    parser.add_argument("--case", default="all", help="Case name or 'all'")
    parser.add_argument("--words", type=int, default=65536)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/controlled"))
    args = parser.parse_args(argv)
    cases = CASES if args.case == "all" else (args.case,)
    written = generate_all(
        args.output_dir,
        n_words=args.words,
        cols=args.cols,
        block_size=args.block_size,
        seed=args.seed,
        cases=cases,
    )
    for u16_path, json_path in written:
        print(f"wrote {u16_path}  manifest {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
