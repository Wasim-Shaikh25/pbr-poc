#!/usr/bin/env python3
"""Ablate PBR-E vs exponent-spatial/hier/cross-layer vs uint16 spatial."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
