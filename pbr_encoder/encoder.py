"""Encode a uint16/BF16 tensor by searching codecs per tile."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pbr_core.bf16 import BF16_DTYPE_TAG, view_uint16
from pbr_core.container import PBRContainer, TensorBlob, attach_geometry
from pbr_core.hashing import sha256_words
from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_core.tiles import as_2d, choose_tile_hw, iter_tiles
from pbr_core.types import EncodedBlock, EncodeContext
from pbr_encoder.mode_search import select_best

_EXP_HUFF = Bf16ExpHuffmanCodec()


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

    tiled = _container(
        matrix,
        name=name,
        block_size=block_size,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tiles=encoded_tiles,
        extra=extra,
    )
    whole = _whole_tensor_exp_huffman(matrix, name=name, extra=extra)
    if whole is None:
        return tiled
    if len(whole.dumps()) < len(tiled.dumps()):
        return whole
    return tiled


def _container(
    matrix: np.ndarray,
    *,
    name: str,
    block_size: int,
    tile_rows: int,
    tile_cols: int,
    tiles: list[EncodedBlock],
    extra: dict | None,
) -> PBRContainer:
    blob = TensorBlob(
        name=name,
        shape=tuple(int(x) for x in matrix.shape),
        dtype=BF16_DTYPE_TAG,
        block_size=block_size,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        n_words=int(matrix.size),
        sha256=sha256_words(matrix),
        tiles=tiles,
    )
    meta = {
        "stage": "1A",
        "selection": "complete_encoded_bytes",
        "runtime_lambda": 0.0,
        "mode_counts": dict(Counter(t.mode_name for t in tiles)),
    }
    if extra:
        meta.update(extra)
    return PBRContainer(tensors=[blob], extra=meta)


def _whole_tensor_exp_huffman(
    matrix: np.ndarray,
    *,
    name: str,
    extra: dict | None,
) -> PBRContainer | None:
    """Amortize one exponent codebook across the whole tensor (PBR-E)."""
    if matrix.size == 0:
        return None
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    # Tile prefix stores rows/cols as uint16; skip whole-tensor if too wide.
    if rows > 0xFFFF or cols > 0xFFFF:
        return None
    encoded = _EXP_HUFF.encode(matrix)
    if encoded is None:
        return None
    attach_geometry(encoded, rows, cols, 0, 0)
    meta = {"whole_tensor_mode": encoded.mode_name}
    if extra:
        meta = {**extra, **meta}
    return _container(
        matrix,
        name=name,
        block_size=int(matrix.size),
        tile_rows=rows,
        tile_cols=cols,
        tiles=[encoded],
        extra=meta,
    )


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
