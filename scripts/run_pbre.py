#!/usr/bin/env python3
"""Re-measure Qwen with PBR-E (exponent Huffman) using the Stage 1B harness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.pbre import main

if __name__ == "__main__":
    raise SystemExit(main())
