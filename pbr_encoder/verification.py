"""Byte-exact verification. Failures raise loudly; they never soft-pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbr_core.hashing import sha256_words


class ExactnessError(AssertionError):
    """Raised when Decode(Encode(W)) is not bit-identical to W."""


@dataclass(frozen=True)
class VerificationResult:
    exact: bool
    differing_words: int
    original_sha256: str
    restored_sha256: str
    original_shape: tuple[int, ...]
    restored_shape: tuple[int, ...]
    n_words: int

    def as_dict(self) -> dict:
        return {
            "exact": self.exact,
            "differing_words": self.differing_words,
            "original_sha256": self.original_sha256,
            "restored_sha256": self.restored_sha256,
            "original_shape": list(self.original_shape),
            "restored_shape": list(self.restored_shape),
            "n_words": self.n_words,
        }


def verify_words(original: np.ndarray, restored: np.ndarray) -> VerificationResult:
    orig = np.ascontiguousarray(original, dtype=np.uint16)
    rest = np.ascontiguousarray(restored, dtype=np.uint16)
    if orig.shape != rest.shape:
        return VerificationResult(
            exact=False,
            differing_words=int(orig.size),
            original_sha256=sha256_words(orig),
            restored_sha256=sha256_words(rest),
            original_shape=tuple(orig.shape),
            restored_shape=tuple(rest.shape),
            n_words=int(orig.size),
        )
    differing = int(np.count_nonzero(orig != rest))
    h1 = sha256_words(orig)
    h2 = sha256_words(rest)
    return VerificationResult(
        exact=differing == 0 and h1 == h2,
        differing_words=differing,
        original_sha256=h1,
        restored_sha256=h2,
        original_shape=tuple(orig.shape),
        restored_shape=tuple(rest.shape),
        n_words=int(orig.size),
    )


def assert_exact(original: np.ndarray, restored: np.ndarray, *, label: str = "tensor") -> VerificationResult:
    result = verify_words(original, restored)
    if not result.exact:
        raise ExactnessError(
            f"EXACTNESS FAIL [{label}]: {result.differing_words} differing uint16 "
            f"words; shape {result.original_shape} vs {result.restored_shape}; "
            f"sha256 {result.original_sha256} vs {result.restored_sha256}"
        )
    return result
