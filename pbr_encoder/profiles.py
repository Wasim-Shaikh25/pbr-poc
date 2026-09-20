"""Named codec menus for ablation (PBR-E vs new exp-spatial vs uint16-only)."""

from __future__ import annotations

from pbr_codecs.bf16_components import ComponentsCodec
from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
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

UINT16_SPATIAL_CODECS = [
    RawCodec(),
    ConstantCodec(),
    ValueDictCodec(),
    PrevValueCodec(),
    PrevRowCodec(),
    ConstPredCodec(),
    DuplicateRefCodec(),
    RefPrevTileCodec(),
]

PBRE_CODECS = [
    *UINT16_SPATIAL_CODECS,
    ComponentsCodec(),
    Bf16ExpHuffmanCodec(),
]

NEW_MODE_CODECS = [
    *PBRE_CODECS,
    ExpSpatialHuffmanCodec(),
    ExpHierResidualCodec(),
    CrossLayerTileXorCodec(),
]

HIER_CODECS = [
    *NEW_MODE_CODECS,
    BitPlanesCodec(),
    ResidualGrammarCodec(),
    TransformedRefCodec(),
    PositionValueDictCodec(),
]

WHOLE_PBRE = [Bf16ExpHuffmanCodec()]
WHOLE_NEW = [
    Bf16ExpHuffmanCodec(),
    ExpSpatialHuffmanCodec(),
    ExpHierResidualCodec(),
    CrossLayerTileXorCodec(),
]
WHOLE_HIER = [*WHOLE_NEW, BitPlanesCodec()]

PROFILES = {
    "pbre": {
        "codecs": PBRE_CODECS,
        "whole_codecs": WHOLE_PBRE,
        "enable_whole": True,
        "use_refs": False,
        "label": "PBR-E only (bf16_exp_huffman + existing modes)",
    },
    "new_modes": {
        "codecs": NEW_MODE_CODECS,
        "whole_codecs": WHOLE_NEW,
        "enable_whole": True,
        "use_refs": True,
        "label": "New modes enabled (exp spatial / hier / cross-layer)",
    },
    "uint16_spatial": {
        "codecs": UINT16_SPATIAL_CODECS,
        "whole_codecs": [],
        "enable_whole": False,
        "use_refs": False,
        "label": "Forced spatial-on-uint16 only (documents the blocker)",
    },
    "hierarchical": {
        "codecs": HIER_CODECS,
        "whole_codecs": WHOLE_HIER,
        "enable_whole": True,
        "use_refs": True,
        "label": "Hierarchical leftovers + PBR-E (bit-planes / grammar / xform / pos-dict)",
    },
}
