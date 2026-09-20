"""Hybrid H95 mantissa-keep + groupwise INT quantized reference.

New Q-ref (not S1, not PR #17 restore_q6). Sensitive tensors keep H95
per-weight exponents; robust regions use lower groupwise INT. Selective
256-node X/Y is stored only when the complete physical cost wins.
"""

from pbr_q4.hybrid.container import (
    MAGIC,
    VERSION,
    decode_container,
    encode_quantized,
    verify_decoded_against_reference,
)
from pbr_q4.hybrid.policy import POLICY_NAMES, get_policy, slot_for_name
from pbr_q4.hybrid.quantize import dequantize_int, quantize_tensor

__all__ = [
    "MAGIC",
    "VERSION",
    "POLICY_NAMES",
    "decode_container",
    "dequantize_int",
    "encode_quantized",
    "get_policy",
    "quantize_tensor",
    "slot_for_name",
    "verify_decoded_against_reference",
]
