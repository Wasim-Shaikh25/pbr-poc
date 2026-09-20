#!/usr/bin/env python3
"""Adaptive 7-bit mantissa codec on synthetic cases and real BF16."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_adaptive_mantissa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
