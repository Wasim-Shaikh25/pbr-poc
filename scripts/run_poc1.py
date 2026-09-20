#!/usr/bin/env python3
"""Run PBR Stage 1A (Gate 1) on controlled tensors and print a results table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
