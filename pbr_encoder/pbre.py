"""PBR-E / Stage 2.5: re-measure the Stage 1B Qwen set with exponent Huffman."""

from __future__ import annotations

import sys
from pathlib import Path

from pbr_encoder.poc1b import main as poc1b_main

BANNER = (
    "PBR-E / Stage 2.5: exponent-Huffman on the Stage 1B Qwen tensor set. "
    "Same pinned revision. Not a 1–2 GB / 8 GB claim. Target band ~11 BPW."
)


def main(argv: list[str] | None = None) -> int:
    print(BANNER)
    print()
    args = list(argv) if argv is not None else sys.argv[1:]
    if "--output-dir" not in args:
        args.extend(["--output-dir", "outputs/reports/pbre"])
    if "--config" not in args:
        args.extend(["--config", "configs/poc_real.yaml"])
    return poc1b_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
