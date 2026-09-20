"""Decoder / encoder errors for standalone corruption tests (Gate D)."""


class CodecError(ValueError):
    """Malformed or truncated matrix-mantissa blob."""
