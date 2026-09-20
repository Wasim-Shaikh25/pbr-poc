"""Select the cheapest complete encoding. λ = 0 in Stage 1A (size only)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pbr_codecs import STAGE1A_CODECS
from pbr_codecs.raw import RawCodec
from pbr_core.types import EncodedBlock, EncodeContext

_RAW = RawCodec()


def collect_candidates(
    words: np.ndarray,
    context: EncodeContext,
    codecs: Sequence | None = None,
) -> list[EncodedBlock]:
    candidates: list[EncodedBlock] = []
    for codec in codecs or STAGE1A_CODECS:
        encoded = codec.encode(words, context)
        if encoded is None:
            continue
        candidates.append(encoded)
    if not any(c.mode_id == _RAW.mode_id for c in candidates):
        candidates.append(_RAW.encode(words, context))
    return candidates


def select_best(
    words: np.ndarray,
    context: EncodeContext,
    codecs: Sequence | None = None,
    runtime_lambda: float = 0.0,
) -> EncodedBlock:
    """Score(mode) = encoded_bytes + λ * runtime_cost. Stage 1A uses λ = 0."""
    del runtime_lambda  # reserved for Compact/Fast profiles
    candidates = collect_candidates(words, context, codecs)
    # Tie-break: smaller payload, then stable mode_id, then name.
    return min(
        candidates,
        key=lambda c: (c.total_bytes, c.mode_id, c.mode_name),
    )
