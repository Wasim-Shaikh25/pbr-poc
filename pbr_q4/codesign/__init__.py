"""PBR-Q4 quantize ↔ selective 256-node X/Y co-design.

New groupwise mixed-precision quantized reference (not H95Q-S1 / not
bit-exact with S1). Codes are laid out for 16×16 tiles; X/Y is stored
only when the complete physical cost beats packed codes.
"""

from pbr_q4.codesign.container import (
    MAGIC,
    VERSION,
    decode_container,
    encode_container,
)
from pbr_q4.codesign.policy import POLICY_NAMES, bits_for_name, get_policy
from pbr_q4.codesign.quantize import dequantize_to_bf16, quantize_tensor

__all__ = [
    "MAGIC",
    "VERSION",
    "POLICY_NAMES",
    "bits_for_name",
    "decode_container",
    "dequantize_to_bf16",
    "encode_container",
    "get_policy",
    "quantize_tensor",
]
