#!/usr/bin/env python3
"""Run ≤8.0 complete-BPW experiments on a local checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.path_to_50pct import main

if __name__ == "__main__":
    raise SystemExit(main())
