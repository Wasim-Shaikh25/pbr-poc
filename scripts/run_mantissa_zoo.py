#!/usr/bin/env python3
"""Mantissa multimodel bakeoff (CTW/PPM, GBDT, tiny AR, IDF, mixture)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.mantissa_zoo import main

if __name__ == "__main__":
    raise SystemExit(main())
