"""Exact CPU decoder. Reconstructs uint16 tiles in raster order."""

from __future__ import annotations

import numpy as np

from pbr_codecs.bf16_components import ComponentsCodec
from pbr_codecs.constant import ConstantCodec
from pbr_codecs.duplicate_blocks import RefPrevTileCodec
from pbr_codecs.raw import RawCodec
from pbr_codecs.value_dictionary import ValueDictCodec
from pbr_codecs.xor_predictor import ConstPredCodec, PrevRowCodec, PrevValueCodec
from pbr_core.container import PBRContainer, TensorBlob
from pbr_core.tiles import place_tile
from pbr_core.types import (
    MODE_COMPONENTS,
    MODE_CONST_PRED,
    MODE_CONSTANT,
    MODE_DUP_REF,
    MODE_PREV_ROW,
    MODE_PREV_VALUE,
    MODE_RAW,
    MODE_REF_PREV,
    MODE_VALUE_DICT,
    EncodedBlock,
    EncodeContext,
)

_CODECS = {
    MODE_RAW: RawCodec(),
    MODE_CONSTANT: ConstantCodec(),
    MODE_VALUE_DICT: ValueDictCodec(),
    MODE_PREV_VALUE: PrevValueCodec(),
    MODE_PREV_ROW: PrevRowCodec(),
    MODE_CONST_PRED: ConstPredCodec(),
    MODE_COMPONENTS: ComponentsCodec(),
    MODE_REF_PREV: RefPrevTileCodec(),
}


def decode_tile(
    encoded: EncodedBlock,
    context: EncodeContext,
    decoded_tiles: list[np.ndarray],
) -> np.ndarray:
    if encoded.mode_id == MODE_DUP_REF:
        if len(encoded.payload) < 4:
            raise ValueError("duplicate_ref payload missing reference index")
        ref = int.from_bytes(encoded.payload[:4], "little")
        if ref < 0 or ref >= len(decoded_tiles):
            raise ValueError(f"duplicate_ref index {ref} out of range")
        src = decoded_tiles[ref]
        if src.shape != (encoded.rows, encoded.cols):
            raise ValueError("duplicate_ref shape mismatch")
        return src.copy()
    codec = _CODECS.get(encoded.mode_id)
    if codec is None:
        raise ValueError(f"Unknown mode_id {encoded.mode_id}")
    return codec.decode(encoded, context)


def decode_tensor(tensor: TensorBlob) -> np.ndarray:
    if len(tensor.shape) != 2:
        raise ValueError("Stage 1A decoder expects a 2-D uint16 tensor")
    out = np.empty(tensor.shape, dtype=np.uint16)
    context = EncodeContext(ncols=int(tensor.shape[1]))
    decoded_tiles: list[np.ndarray] = []
    for i, tile in enumerate(tensor.tiles):
        context.tile_index = i
        words = decode_tile(tile, context, decoded_tiles)
        if words.shape != (tile.rows, tile.cols):
            raise ValueError(
                f"Decoded tile shape {words.shape} != ({tile.rows}, {tile.cols})"
            )
        place_tile(out, words, tile.row0, tile.col0)
        decoded_tiles.append(words)
        context.record(words)
    return out


def decode_container(container: PBRContainer) -> list[np.ndarray]:
    return [decode_tensor(t) for t in container.tensors]
