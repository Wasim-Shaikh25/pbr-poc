#!/usr/bin/env python3
"""Phase A held-out mantissa predictability audit (lossless BF16)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.mantissa_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
