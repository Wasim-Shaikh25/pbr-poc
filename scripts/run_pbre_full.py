#!/usr/bin/env python3
"""Full-checkpoint PBR-E encode (all 16-bit tensors). Bit-exact DF11-class bar."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.pbre import main as pbre_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    extra = ["--all-16bit", "--profile", "pbre", "--no-baselines"]
    if "--output-dir" not in args:
        extra.extend(["--output-dir", "outputs/reports/pbre_full"])
    print(
        "PBR-E full-checkpoint: every 16-bit tensor, profile=pbre, no zlib. "
        "Not a 1–2 GB / 8 GB claim. Success is not ≤4 BPW."
    )
    return pbre_main(extra + args)


if __name__ == "__main__":
    raise SystemExit(main())
