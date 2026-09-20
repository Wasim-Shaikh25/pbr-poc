#!/usr/bin/env python3
"""Generate Stage 1A controlled uint16 tensors (not real model weights)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.controlled_data import main

if __name__ == "__main__":
    raise SystemExit(main())
