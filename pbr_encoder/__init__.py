"""Stage 1A encoder: cost-based mode search and exact CPU decode."""

from pbr_encoder.decoder import decode_container, decode_tensor
from pbr_encoder.encoder import encode_tensor, encode_words
from pbr_encoder.verification import ExactnessError, assert_exact, verify_words

__all__ = [
    "ExactnessError",
    "assert_exact",
    "decode_container",
    "decode_tensor",
    "encode_tensor",
    "encode_words",
    "verify_words",
]
