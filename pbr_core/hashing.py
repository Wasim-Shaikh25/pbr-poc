"""SHA-256 over exact little-endian uint16 words."""

from __future__ import annotations

import hashlib

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words_to_bytes(words: np.ndarray) -> bytes:
    return np.ascontiguousarray(words, dtype="<u2").tobytes()


def sha256_words(words: np.ndarray) -> str:
    return sha256_bytes(words_to_bytes(words))
