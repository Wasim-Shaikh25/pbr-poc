"""Encode a uint16/BF16 tensor by searching codecs per tile."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pbr_core.bf16 import BF16_DTYPE_TAG, view_uint16
from pbr_core.container import PBRContainer, TensorBlob, attach_geometry
from pbr_core.hashing import sha256_words
from pbr_core.tiles import as_2d, choose_tile_hw, iter_tiles
from pbr_core.types import EncodedBlock, EncodeContext
from pbr_encoder.mode_search import select_best


def encode_words(
    words: np.ndarray,
    *,
    name: str = "tensor",
    block_size: int = 256,
    codecs: Sequence | None = None,
    extra: dict | None = None,
) -> PBRContainer:
    view = view_uint16(words)
    matrix = as_2d(view, view.shape if view.ndim >= 2 else None)
    return encode_tensor(
        matrix,
        name=name,
        block_size=block_size,
        codecs=codecs,
        extra=extra,
    )


def encode_tensor(
    words_2d: np.ndarray,
    *,
    name: str = "tensor",
    block_size: int = 256,
    codecs: Sequence | None = None,
    extra: dict | None = None,
) -> PBRContainer:
    matrix = as_2d(view_uint16(words_2d))
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    tile_rows, tile_cols = choose_tile_hw(block_size, rows, cols)
    tiles = iter_tiles(matrix, block_size)
    context = EncodeContext(ncols=cols)
    encoded_tiles: list[EncodedBlock] = []
    for tile in tiles:
        context.tile_index = tile.index
        chosen = select_best(tile.words, context, codecs=codecs)
        attach_geometry(chosen, tile.rows, tile.cols, tile.row0, tile.col0)
        encoded_tiles.append(chosen)
        context.record(tile.words)

    blob = TensorBlob(
        name=name,
        shape=tuple(int(x) for x in matrix.shape),
        dtype=BF16_DTYPE_TAG,
        block_size=block_size,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        n_words=int(matrix.size),
        sha256=sha256_words(matrix),
        tiles=encoded_tiles,
    )
    meta = {
        "stage": "1A",
        "selection": "complete_encoded_bytes",
        "runtime_lambda": 0.0,
        "mode_counts": dict(Counter(t.mode_name for t in encoded_tiles)),
    }
    if extra:
        meta.update(extra)
    return PBRContainer(tensors=[blob], extra=meta)


def mode_usage(container: PBRContainer) -> dict[str, dict[str, int | float]]:
    counts: Counter[str] = Counter()
    bytes_by_mode: Counter[str] = Counter()
    for tensor in container.tensors:
        for tile in tensor.tiles:
            counts[tile.mode_name] += 1
            bytes_by_mode[tile.mode_name] += tile.total_bytes
    total_tiles = sum(counts.values()) or 1
    return {
        name: {
            "tiles": counts[name],
            "tile_share": counts[name] / total_tiles,
            "bytes": bytes_by_mode[name],
        }
        for name in counts
    }
