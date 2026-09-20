"""Encode a uint16/BF16 tensor by searching codecs per tile."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_core.bf16 import BF16_DTYPE_TAG, view_uint16
from pbr_core.container import PBRContainer, TensorBlob, attach_geometry
from pbr_core.hashing import sha256_words
from pbr_core.tiles import as_2d, choose_tile_hw, iter_tiles
from pbr_core.types import EncodedBlock, EncodeContext
from pbr_encoder.mode_search import select_best
from pbr_encoder.profiles import WHOLE_NEW

_DEFAULT_WHOLE = WHOLE_NEW


def encode_words(
    words: np.ndarray,
    *,
    name: str = "tensor",
    block_size: int = 256,
    codecs: Sequence | None = None,
    extra: dict | None = None,
    ref_words: np.ndarray | None = None,
    whole_codecs: Sequence | None = None,
    enable_whole: bool = True,
) -> PBRContainer:
    view = view_uint16(words)
    matrix = as_2d(view, view.shape if view.ndim >= 2 else None)
    return encode_tensor(
        matrix,
        name=name,
        block_size=block_size,
        codecs=codecs,
        extra=extra,
        ref_words=ref_words,
        whole_codecs=whole_codecs,
        enable_whole=enable_whole,
    )


def encode_tensor(
    words_2d: np.ndarray,
    *,
    name: str = "tensor",
    block_size: int = 256,
    codecs: Sequence | None = None,
    extra: dict | None = None,
    ref_words: np.ndarray | None = None,
    whole_codecs: Sequence | None = None,
    enable_whole: bool = True,
) -> PBRContainer:
    matrix = as_2d(view_uint16(words_2d))
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    tile_rows, tile_cols = choose_tile_hw(block_size, rows, cols)
    tiles = iter_tiles(matrix, block_size)
    ref = None if ref_words is None else as_2d(view_uint16(ref_words))
    context = EncodeContext(ncols=cols, ref_tensor=ref)
    encoded_tiles: list[EncodedBlock] = []
    for tile in tiles:
        context.tile_index = tile.index
        context.tile_row0 = tile.row0
        context.tile_col0 = tile.col0
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
    if not enable_whole:
        return tiled
    whole = _best_whole_tensor(
        matrix,
        name=name,
        extra=extra,
        ref_words=ref,
        whole_codecs=whole_codecs,
    )
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


def _best_whole_tensor(
    matrix: np.ndarray,
    *,
    name: str,
    extra: dict | None,
    ref_words: np.ndarray | None,
    whole_codecs: Sequence | None,
) -> PBRContainer | None:
    """Amortize one codebook across the whole tensor (PBR-E and successors)."""
    if matrix.size == 0:
        return None
    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    if rows > 0xFFFF or cols > 0xFFFF:
        return None
    context = EncodeContext(ncols=cols, ref_tensor=ref_words)
    best: EncodedBlock | None = None
    for codec in list(whole_codecs) if whole_codecs is not None else _DEFAULT_WHOLE:
        encoded = codec.encode(matrix, context)
        if encoded is None:
            continue
        if best is None or encoded.total_bytes < best.total_bytes:
            best = encoded
    if best is None:
        return None
    attach_geometry(best, rows, cols, 0, 0)
    meta = {"whole_tensor_mode": best.mode_name}
    if extra:
        meta = {**extra, **meta}
    return _container(
        matrix,
        name=name,
        block_size=int(matrix.size),
        tile_rows=rows,
        tile_cols=cols,
        tiles=[best],
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
