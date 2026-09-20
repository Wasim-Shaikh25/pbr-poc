#!/usr/bin/env python3
"""Run the PBR Stage 2 qualification scanner (projection, not a full encode)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_qualifier.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
