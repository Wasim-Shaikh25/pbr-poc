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


def _get(args: list[str], flag: str, default: str | None = None) -> str | None:
    if flag in args:
        return args[args.index(flag) + 1]
    return default


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    tag = _get(args, "--tag", "family") or "family"
    model_dir = _get(args, "--model-dir")
    config = _get(args, "--config")
    diag = ["--output-json", f"artifacts/blocker_diagnosis_{tag}.json", "--output-md", f"artifacts/blocker_diagnosis_{tag}.md"]
    if model_dir:
        diag.extend(["--model-dir", model_dir])
    if config:
        diag.extend(["--config", config])
    print("=== blocker diagnosis ===", flush=True)
    rc = diagnosis_main(diag)
    if rc != 0:
        return rc
    print("=== PBR-E + hierarchical ablation ===", flush=True)
    ablate: list[str] = ["--tag", tag, "--profiles", "pbre", "hierarchical"]
    if model_dir:
        ablate.extend(["--model-dir", model_dir])
    if config:
        ablate.extend(["--config", config])
    return ablation_main(ablate)


if __name__ == "__main__":
    raise SystemExit(main())
