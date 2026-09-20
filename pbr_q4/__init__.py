"""PBR-Q4 / H95Q-S1 + 256-node X/Y post-codec.

Re-encodes the *already quantized* H95Q-S1 reference (mantissa K-codes) with
an all-tile 16×16 node-matrix family. Does not introduce a new BF16→Q4 policy.

Hard rule: every eligible tile is stored as a matrix-family representation
(MATRIX / RANS / BITPLANE / PAIR / RUN). Packed-K size is a baseline metric only.
"""

from pbr_q4.container import (
    FROZEN_S1_SHA,
    MAGIC,
    VERSION,
    decode_container,
    encode_container,
    verify_decoded_against_reference,
)

__all__ = [
    "FROZEN_S1_SHA",
    "MAGIC",
    "VERSION",
    "decode_container",
    "encode_container",
    "verify_decoded_against_reference",
]
