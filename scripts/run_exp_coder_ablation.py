#!/usr/bin/env python3
"""Canonical Huffman vs rANS on BF16 exponents (complete bytes)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.exp_coder_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
