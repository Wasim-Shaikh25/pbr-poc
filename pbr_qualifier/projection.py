"""Whole-model BPW projection from sample encodes with complete overhead."""

from __future__ import annotations

from pbr_core.metrics import bits_per_weight


def project_from_sample(
    *,
    n_words: int,
    n_full_tiles: int,
    sample_tile_bytes: int,
    n_sample_tiles: int,
    container_header_bytes: int,
) -> int:
    """Estimate one tensor's complete encoded size.

    Tile on-wire bytes (mode id, geometry, payload) are scaled by the full
    tile count. The container header (magic, JSON metadata) is counted once.
    If the sample is empty, fall back to raw BF16 (2 bytes/word).
    """
    if n_words <= 0:
        return max(container_header_bytes, 0)
    if n_sample_tiles <= 0 or n_full_tiles <= 0:
        return n_words * 2
    avg_tile = sample_tile_bytes / n_sample_tiles
    return int(round(avg_tile * n_full_tiles + container_header_bytes))


def weighted_model_bpw(
    *,
    projected_encoded_bytes: int,
    total_parameter_count: int,
) -> float:
    """``Total_BPW = 8 * projected_encoded_bytes / total_parameter_count``."""
    return bits_per_weight(projected_encoded_bytes, total_parameter_count)


def combine_projections(
    scanned: list[tuple[int, int]],
    unscanned_words: int,
    *,
    unscanned_bpw: float = 16.0,
) -> tuple[int, int, float]:
    """Merge scanned (words, projected_bytes) with raw-BF16 remainder.

    Returns ``(total_words, total_projected_bytes, bpw)``.
    """
    words = sum(w for w, _ in scanned) + unscanned_words
    raw_remainder = int(round(unscanned_words * unscanned_bpw / 8.0))
    encoded = sum(b for _, b in scanned) + raw_remainder
    return words, encoded, weighted_model_bpw(
        projected_encoded_bytes=encoded,
        total_parameter_count=words,
    )
