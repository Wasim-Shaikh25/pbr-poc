"""Stage 2 qualification pipeline: inventory → entropy → sample encode → project."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pbr_core.metrics import bits_per_weight
from pbr_core.safetensors_io import TWO_BYTE_DTYPES, TensorSpec, logical_2d_shape
from pbr_core.tiles import tile_count
from pbr_encoder.decoder import decode_container
from pbr_encoder.encoder import encode_tensor, mode_usage
from pbr_encoder.verification import ExactnessError, assert_exact
from pbr_qualifier.analysis import tile_repetition
from pbr_qualifier.bands import qualification_band, qualification_decision
from pbr_qualifier.entropy import (
    component_entropy,
    is_high_entropy,
    residual_entropy,
    unique_coverage,
)
from pbr_qualifier.projection import combine_projections, project_from_sample
from pbr_qualifier.sampling import entropy_subsample, load_sample_tiles

DISCLAIMER = (
    "Stage 2 qualification scanner: projected complete-container BPW from "
    "sampled encodings. This is not a measured full-model size and cannot "
    "one-GB-qualify a checkpoint."
)

DTYPE_POLICY = (
    "Only BF16, F16, and FP16 tensors are scanned, as exact uint16 views. "
    "Other dtypes are not converted through FP32 or any other archival cast. "
    "They are inventoried, their raw bytes are reported separately, and they "
    "are excluded from the 16-bit parameter denominator. Unscanned 16-bit "
    "tensors (if any) are projected as raw BF16 at 16 BPW."
)


@dataclass
class TensorScan:
    name: str
    shape: tuple[int, ...]
    dtype: str
    n_words: int
    nbytes: int
    scanned: bool
    skip_reason: str | None = None
    entropy: dict = field(default_factory=dict)
    repetition: dict = field(default_factory=dict)
    sample_strategy: str | None = None
    sample_words: int = 0
    sample_encoded_bytes: int = 0
    sample_bpw: float = 16.0
    projected_bytes: int = 0
    projected_bpw: float = 16.0
    winning_modes: str = ""
    exact: str = "SKIP"
    high_entropy: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "n_words": self.n_words,
            "nbytes": self.nbytes,
            "scanned": self.scanned,
            "skip_reason": self.skip_reason,
            "entropy": self.entropy,
            "repetition": self.repetition,
            "sample_strategy": self.sample_strategy,
            "sample_words": self.sample_words,
            "sample_encoded_bytes": self.sample_encoded_bytes,
            "sample_bpw": self.sample_bpw,
            "projected_bytes": self.projected_bytes,
            "projected_bpw": self.projected_bpw,
            "winning_modes": self.winning_modes,
            "exact": self.exact,
            "high_entropy": self.high_entropy,
        }


def _mode_summary(usage: dict) -> str:
    ranked = sorted(usage.items(), key=lambda kv: -kv[1]["tiles"])
    return ",".join(f"{name}:{info['tiles']}" for name, info in ranked[:3]) or "-"


def scan_tensor(
    spec: TensorSpec,
    *,
    block_size: int,
    sample_tiles: int,
    max_full_words: int,
    entropy_words: int,
) -> TensorScan:
    row = TensorScan(
        name=spec.name,
        shape=spec.shape,
        dtype=spec.dtype,
        n_words=spec.n_words,
        nbytes=spec.nbytes,
        scanned=False,
        projected_bytes=spec.n_words * 2,
        projected_bpw=16.0,
    )
    if spec.dtype not in TWO_BYTE_DTYPES:
        row.skip_reason = f"non-16-bit dtype {spec.dtype}; not converted"
        return row
    if spec.n_words == 0:
        row.skip_reason = "empty tensor"
        row.exact = "SKIP"
        row.projected_bytes = 0
        row.projected_bpw = 0.0
        return row

    subsample = entropy_subsample(spec, max_words=entropy_words)
    comps = component_entropy(subsample)
    if subsample.size >= 2:
        # Reshape a 1-D subsample into a short row so residual helpers run.
        width = min(64, int(subsample.size))
        usable = (subsample.size // width) * width
        resid_src = subsample[:usable].reshape(-1, width)
        resid = residual_entropy(resid_src)
    else:
        resid = {"prev_value_bits": 0.0, "prev_row_bits": 0.0}
    cover = unique_coverage(subsample)
    row.entropy = {**comps, **resid, **cover}
    row.high_entropy = is_high_entropy(
        float(comps["word_bits"]),
        float(resid["prev_value_bits"]),
    )

    sample, tiles, strategy = load_sample_tiles(
        spec,
        block_size=block_size,
        n_samples=sample_tiles,
        max_full_words=max_full_words,
    )
    row.sample_strategy = strategy
    row.repetition = tile_repetition(tiles)
    row.sample_words = int(sample.size)
    if sample.size == 0:
        row.skip_reason = "empty sample"
        return row

    container = encode_tensor(
        sample,
        name=spec.name,
        block_size=block_size,
        extra={"stage": "2", "scan": "sample"},
    )
    blob = container.dumps()
    restored = decode_container(container)[0]
    try:
        assert_exact(sample, restored, label=f"qualifier:{spec.name}")
        row.exact = "PASS"
    except ExactnessError:
        row.exact = "FAIL"
        raise

    tile_bytes = sum(t.total_bytes for t in container.tensors[0].tiles)
    header_bytes = len(blob) - tile_bytes
    n_sample_tiles = max(len(container.tensors[0].tiles), 1)
    rows, cols = logical_2d_shape(spec.shape)
    n_full = tile_count(rows, cols, block_size)
    if strategy == "full":
        projected = len(blob)
    else:
        projected = project_from_sample(
            n_words=spec.n_words,
            n_full_tiles=n_full,
            sample_tile_bytes=tile_bytes,
            n_sample_tiles=n_sample_tiles,
            container_header_bytes=header_bytes,
        )
    row.scanned = True
    row.sample_encoded_bytes = len(blob)
    row.sample_bpw = bits_per_weight(len(blob), int(sample.size))
    row.projected_bytes = projected
    row.projected_bpw = bits_per_weight(projected, spec.n_words)
    row.winning_modes = _mode_summary(mode_usage(container))
    return row


def scan_model(
    specs: list[TensorSpec],
    *,
    block_size: int = 256,
    sample_tiles: int = 8,
    max_full_words: int = 16384,
    entropy_words: int = 65536,
) -> dict:
    sixteen = [s for s in specs if s.dtype in TWO_BYTE_DTYPES]
    other = [s for s in specs if s.dtype not in TWO_BYTE_DTYPES]
    rows: list[TensorScan] = []
    for spec in sixteen:
        print(
            f"scan {spec.name}  {list(spec.shape)}  {spec.dtype}  "
            f"{spec.nbytes} B",
            flush=True,
        )
        row = scan_tensor(
            spec,
            block_size=block_size,
            sample_tiles=sample_tiles,
            max_full_words=max_full_words,
            entropy_words=entropy_words,
        )
        print(
            f"  -> proj_BPW={row.projected_bpw:.4f}  exact={row.exact}  "
            f"modes={row.winning_modes or '-'}  strategy={row.sample_strategy}",
            flush=True,
        )
        rows.append(row)

    scanned_pairs = [(r.n_words, r.projected_bytes) for r in rows if r.scanned]
    unscanned_words = sum(r.n_words for r in rows if not r.scanned)
    total_words, total_bytes, bpw = combine_projections(scanned_pairs, unscanned_words)
    decision = qualification_decision(bpw)

    mode_tiles: Counter[str] = Counter()
    for row in rows:
        if not row.winning_modes:
            continue
        for part in row.winning_modes.split(","):
            name, _, count = part.partition(":")
            if count.isdigit():
                mode_tiles[name] += int(count)

    other_bytes = sum(s.nbytes for s in other)
    leaderboard = sorted(
        (r for r in rows if r.scanned),
        key=lambda r: (r.projected_bpw, r.name),
    )
    return {
        "disclaimer": DISCLAIMER,
        "dtype_policy": DTYPE_POLICY,
        "inventory": {
            "tensor_count": len(specs),
            "scanned_16bit": len(sixteen),
            "non16_tensor_count": len(other),
            "non16_raw_bytes": other_bytes,
            "non16_names": [s.name for s in other],
        },
        "projection": {
            "total_16bit_parameters": total_words,
            "scanned_parameters": sum(r.n_words for r in rows if r.scanned),
            "unscanned_16bit_parameters": unscanned_words,
            "unscanned_assumed_bpw": 16.0,
            "projected_encoded_bytes": total_bytes,
            "projected_bpw": bpw,
            "band": decision.band,
            "band_help": {
                "extreme": "≤1 BPW",
                "one_gb_class": "≤2 BPW (class label only; not qualified)",
                "exceptional": "2–4 BPW",
                "strong": "4–8 BPW",
                "not_exceptional": ">8 BPW",
            },
            **decision.as_dict(),
        },
        "mode_tile_counts": dict(mode_tiles),
        "tensors": [r.as_dict() for r in rows],
        "leaderboard": [r.as_dict() for r in leaderboard[:15]],
        "caveats": [
            DISCLAIMER,
            DTYPE_POLICY,
            "Sample tile bytes are scaled to the full tile count; the container "
            "header is counted once per tensor. This is conservative vs a true "
            "full encode when sample headers are a large fraction of a tiny sample.",
            "Stage 1B measured ~13.6 complete BPW on 160 MiB of Qwen linear "
            "weights with bf16_components winning. A Stage 2 projection near "
            "that range is a consistency check, not a new compression result.",
            "Hierarchical codecs (cross-layer references, grammar, adaptive "
            "regions) are not in this scanner. They could change a future projection.",
        ],
    }
