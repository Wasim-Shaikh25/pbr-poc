#!/usr/bin/env python3
"""Diagnosis + PBR-E + hierarchical ablation on a local checkpoint sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pbr_encoder.ablation import main as ablation_main
from pbr_encoder.blocker_diagnosis import main as diagnosis_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    # Split our flags from the shared ones.
    tag = "family"
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
    diag_args = [a for a in args if a not in {"--profiles"}]
    # diagnosis uses --output-json/--output-md; map --tag.
    if "--output-json" not in diag_args:
        diag_args.extend(["--output-json", f"artifacts/blocker_diagnosis_{tag}.json"])
    if "--output-md" not in diag_args:
        diag_args.extend(["--output-md", f"artifacts/blocker_diagnosis_{tag}.md"])
    print("=== blocker diagnosis ===", flush=True)
    rc = diagnosis_main(diag_args)
    if rc != 0:
        return rc
    print("=== PBR-E + hierarchical ablation ===", flush=True)
    ablate = list(args)
    if "--profiles" not in ablate:
        ablate.extend(["--profiles", "pbre", "hierarchical"])
    if "--tag" not in ablate:
        ablate.extend(["--tag", tag])
    return ablation_main(ablate)


if __name__ == "__main__":
    raise SystemExit(main())
