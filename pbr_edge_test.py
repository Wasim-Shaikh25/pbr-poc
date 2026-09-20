#!/usr/bin/env python3
"""Matrix mantissa V1–V3 pre-Qwen qualification. Exits non-zero on gate failure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_matrix_mantissa.edge_test import main

if __name__ == "__main__":
    raise SystemExit(main())
