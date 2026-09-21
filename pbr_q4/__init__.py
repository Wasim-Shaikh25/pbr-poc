"""PBR-Q4: S1 selective X/Y post-codec, plus groupwise codesign.

``pbr_q4`` still hosts the H95Q-S1 + selective 256-node X/Y post-codec
(not a new BF16→Q4 policy). The new groupwise quantized reference and
quantize↔X/Y co-design live in ``pbr_q4.codesign`` and are **not**
S1-compatible.
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
