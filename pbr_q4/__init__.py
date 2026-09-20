"""PBR-Q4 / H95Q-S1 + selective 256-node X/Y post-codec.

Re-encodes the *already quantized* H95Q-S1 reference (mantissa K-codes).
Does not introduce a new BF16→Q4 policy.

Product default: packed-K (S1 stream packing). Matrix-family X/Y is used
only when it is strictly smaller including mode/flag/map bits. Arrays are
not padded to 16×16; ragged last tiles store their true shape.
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
