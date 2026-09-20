"""Stage 1A codecs. Each candidate reports complete encoded byte cost."""

from pbr_codecs.bf16_components import ComponentsCodec
from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_codecs.bf16_exp_rans import Bf16ExpRansCodec
from pbr_codecs.bit_planes import BitPlanesCodec
from pbr_codecs.constant import ConstantCodec
from pbr_codecs.cross_layer_tile_xor import CrossLayerTileXorCodec
from pbr_codecs.duplicate_blocks import DuplicateRefCodec, RefPrevTileCodec
from pbr_codecs.exp_hier_residual import ExpHierResidualCodec
from pbr_codecs.exp_spatial_huffman import ExpSpatialHuffmanCodec
from pbr_codecs.grammar import ResidualGrammarCodec
from pbr_codecs.position_value_dict import PositionValueDictCodec
from pbr_codecs.raw import RawCodec
from pbr_codecs.transformed_ref import TransformedRefCodec
from pbr_codecs.value_dictionary import ValueDictCodec
from pbr_codecs.xor_predictor import ConstPredCodec, PrevRowCodec, PrevValueCodec

STAGE1A_CODECS = [
    RawCodec(),
    ConstantCodec(),
    ValueDictCodec(),
    PrevValueCodec(),
    PrevRowCodec(),
    ConstPredCodec(),
    DuplicateRefCodec(),
    RefPrevTileCodec(),
    ComponentsCodec(),
    Bf16ExpHuffmanCodec(),
    ExpSpatialHuffmanCodec(),
    ExpHierResidualCodec(),
    CrossLayerTileXorCodec(),
    BitPlanesCodec(),
    ResidualGrammarCodec(),
    TransformedRefCodec(),
    PositionValueDictCodec(),
]

__all__ = [
    "Bf16ExpHuffmanCodec",
    "Bf16ExpRansCodec",
    "BitPlanesCodec",
    "ComponentsCodec",
    "ConstPredCodec",
    "ConstantCodec",
    "CrossLayerTileXorCodec",
    "DuplicateRefCodec",
    "ExpHierResidualCodec",
    "ExpSpatialHuffmanCodec",
    "PositionValueDictCodec",
    "PrevRowCodec",
    "PrevValueCodec",
    "RawCodec",
    "RefPrevTileCodec",
    "ResidualGrammarCodec",
    "STAGE1A_CODECS",
    "TransformedRefCodec",
    "ValueDictCodec",
]
