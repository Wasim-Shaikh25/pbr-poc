"""Exact uint16 primitives for PBR Stage 1A.

All archival encode/decode paths operate on little-endian uint16 views of
BF16 bit patterns. There is no FP32 conversion path.
"""

from pbr_core.bf16 import (
    BF16_DTYPE_TAG,
    join_components,
    make_bf16_bits,
    split_components,
    view_uint16,
)
from pbr_core.hashing import sha256_bytes, sha256_words
from pbr_core.metrics import bits_per_weight, compression_ratio
from pbr_core.types import (
    TILE_HEADER_BYTES,
    CostEstimate,
    EncodedBlock,
    EncodeContext,
    TileInfo,
)

__all__ = [
    "BF16_DTYPE_TAG",
    "CostEstimate",
    "EncodedBlock",
    "EncodeContext",
    "TILE_HEADER_BYTES",
    "TileInfo",
    "bits_per_weight",
    "compression_ratio",
    "join_components",
    "make_bf16_bits",
    "sha256_bytes",
    "sha256_words",
    "split_components",
    "view_uint16",
]
