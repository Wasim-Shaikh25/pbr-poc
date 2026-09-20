"""Stage 1A codecs. Each candidate reports complete encoded byte cost."""

from pbr_codecs.bf16_components import ComponentsCodec
from pbr_codecs.bf16_exp_huffman import Bf16ExpHuffmanCodec
from pbr_codecs.constant import ConstantCodec
from pbr_codecs.duplicate_blocks import DuplicateRefCodec, RefPrevTileCodec
from pbr_codecs.raw import RawCodec
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
]

__all__ = [
    "Bf16ExpHuffmanCodec",
    "ComponentsCodec",
    "ConstPredCodec",
    "ConstantCodec",
    "DuplicateRefCodec",
    "PrevRowCodec",
    "PrevValueCodec",
    "RawCodec",
    "RefPrevTileCodec",
    "STAGE1A_CODECS",
    "ValueDictCodec",
]
