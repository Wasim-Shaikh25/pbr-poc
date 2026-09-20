"""Exact CPU decoder. Reconstructs uint16 tiles in raster order."""

from __future__ import annotations

import numpy as np

from pbr_codecs.bf16_components import ComponentsCodec
from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_codecs.bit_planes import BitPlanesCodec
from pbr_codecs.constant import ConstantCodec
from pbr_codecs.cross_layer_tile_xor import CrossLayerTileXorCodec
from pbr_codecs.duplicate_blocks import RefPrevTileCodec
from pbr_codecs.exp_hier_residual import ExpHierResidualCodec
from pbr_codecs.exp_spatial_huffman import ExpSpatialHuffmanCodec
from pbr_codecs.grammar import ResidualGrammarCodec
from pbr_codecs.position_value_dict import PositionValueDictCodec
from pbr_codecs.raw import RawCodec
from pbr_codecs.residual import decode_residuals
from pbr_codecs.transforms import apply_transform
from pbr_codecs.value_dictionary import ValueDictCodec
from pbr_codecs.xor_predictor import ConstPredCodec, PrevRowCodec, PrevValueCodec
from pbr_core.container import PBRContainer, TensorBlob
from pbr_core.safetensors_io import tensor_role
from pbr_core.tiles import place_tile
from pbr_core.types import (
    MODE_BITPLANES,
    MODE_COMPONENTS,
    MODE_CONST_PRED,
    MODE_CONSTANT,
    MODE_CROSS_LAYER,
    MODE_DUP_REF,
    MODE_EXP_HIER,
    MODE_EXP_HUFFMAN,
    MODE_EXP_RANS,
    MODE_EXP_SPATIAL,
    MODE_GRAMMAR,
    MODE_POS_VALUE,
    MODE_PREV_ROW,
    MODE_PREV_VALUE,
    MODE_RAW,
    MODE_REF_PREV,
    MODE_VALUE_DICT,
    MODE_XFORM_REF,
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
    MODE_EXP_HUFFMAN: Bf16ExpHuffmanCodec(),
    MODE_EXP_RANS: Bf16ExpRansCodec(),
    MODE_EXP_SPATIAL: ExpSpatialHuffmanCodec(),
    MODE_EXP_HIER: ExpHierResidualCodec(),
    MODE_CROSS_LAYER: CrossLayerTileXorCodec(),
    MODE_BITPLANES: BitPlanesCodec(),
    MODE_GRAMMAR: ResidualGrammarCodec(),
    MODE_POS_VALUE: PositionValueDictCodec(),
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
    if encoded.mode_id == MODE_XFORM_REF:
        if len(encoded.payload) < 5:
            raise ValueError("transformed_ref payload too short")
        xf_id = encoded.payload[0]
        ref = int.from_bytes(encoded.payload[1:5], "little")
        if ref < 0 or ref >= len(decoded_tiles):
            raise ValueError(f"transformed_ref index {ref} out of range")
        src = apply_transform(decoded_tiles[ref], xf_id)
        if src is None or src.shape != (encoded.rows, encoded.cols):
            raise ValueError("transformed_ref geometry mismatch")
        patch = encoded.payload[5:]
        if not patch:
            return src.copy()
        residuals = decode_residuals(patch, encoded.rows, encoded.cols)
        return residuals ^ src
    codec = _CODECS.get(encoded.mode_id)
    if codec is None:
        raise ValueError(f"Unknown mode_id {encoded.mode_id}")
    return codec.decode(encoded, context)


def decode_tensor(tensor: TensorBlob, ref_tensor: np.ndarray | None = None) -> np.ndarray:
    if len(tensor.shape) != 2:
        raise ValueError("Stage 1A decoder expects a 2-D uint16 tensor")
    out = np.empty(tensor.shape, dtype=np.uint16)
    context = EncodeContext(ncols=int(tensor.shape[1]), ref_tensor=ref_tensor)
    decoded_tiles: list[np.ndarray] = []
    for i, tile in enumerate(tensor.tiles):
        context.tile_index = i
        context.tile_row0 = tile.row0
        context.tile_col0 = tile.col0
        words = decode_tile(tile, context, decoded_tiles)
        if words.shape != (tile.rows, tile.cols):
            raise ValueError(
                f"Decoded tile shape {words.shape} != ({tile.rows}, {tile.cols})"
            )
        place_tile(out, words, tile.row0, tile.col0)
        decoded_tiles.append(words)
        context.record(words)
    return out


def decode_container(
    container: PBRContainer,
    ref_tensor: np.ndarray | None = None,
    *,
    session_refs: dict | None = None,
) -> list[np.ndarray]:
    """Decode every tensor. Cross-layer modes use a causal same-role ref."""
    refs = session_refs if session_refs is not None else {}
    out: list[np.ndarray] = []
    for i, tensor in enumerate(container.tensors):
        role = tensor_role(tensor.name)
        key = (role, tuple(tensor.shape))
        ref = refs.get(key)
        if ref is None and i == 0:
            ref = ref_tensor
        words = decode_tensor(tensor, ref_tensor=ref)
        refs[key] = words
        out.append(words)
    return out
