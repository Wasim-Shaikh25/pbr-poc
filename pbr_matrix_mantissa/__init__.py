"""Matrix mantissa codec V1–V3 (pre-Qwen qualification).

Do not claim Qwen compression from this package. No ≤4 BPW product claim.
"""

from pbr_matrix_mantissa.codec import (
    VERSIONS,
    decode_bf16_matrix,
    decode_mantissa,
    encode_bf16_matrix,
    encode_mantissa,
)
from pbr_matrix_mantissa.errors import CodecError

__all__ = [
    "VERSIONS",
    "CodecError",
    "decode_bf16_matrix",
    "decode_mantissa",
    "encode_bf16_matrix",
    "encode_mantissa",
]
