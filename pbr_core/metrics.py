"""Complete-container size metrics. BPW always includes metadata."""

from __future__ import annotations


def bits_per_weight(encoded_bytes: int, n_words: int) -> float:
    if n_words <= 0:
        return 0.0
    return 8.0 * encoded_bytes / n_words


def compression_ratio(encoded_bytes: int, original_bytes: int) -> float:
    if original_bytes <= 0:
        return 0.0
    return encoded_bytes / original_bytes
