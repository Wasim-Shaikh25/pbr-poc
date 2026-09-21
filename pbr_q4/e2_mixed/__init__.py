"""PBR-Q4 E2 mixed groupwise Q + E3 selective lossless tiles.

New quantized reference (not H95Q-S1, not PR #17/#18). Decode is bit-exact
to this quantized reference, not to original BF16.
"""

from pbr_q4.e2_mixed.container import (
    decode_container,
    encode_quantized,
    verify_decoded_against_reference,
)
from pbr_q4.e2_mixed.const import MAGIC_E2, MAGIC_E3, VERSION

__all__ = [
    "MAGIC_E2",
    "MAGIC_E3",
    "VERSION",
    "decode_container",
    "encode_quantized",
    "verify_decoded_against_reference",
]
