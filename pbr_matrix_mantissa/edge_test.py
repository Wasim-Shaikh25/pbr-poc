"""CLI: run matrix-mantissa pre-Qwen gates and write artifacts. Exits 1 on fail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbr_matrix_mantissa.codec import DISCLAIMER
from pbr_matrix_mantissa.gates import run_all_gates


def format_markdown(report: dict) -> str:
    lines = [
        "# Matrix mantissa V1–V3 — pre-Qwen qualification",
        "",
        DISCLAIMER,
        "",
        "**Do not claim Qwen compression yet.** These gates are synthetic / exactness only.",
        "No ≤4 BPW product claim.",
        "",
        f"Overall: **{'PASS' if report.get('passed') else 'FAIL'}**",
        "",
        "| gate | result |",
        "| --- | --- |",
    ]
    for name, gate in (report.get("gates") or {}).items():
        lines.append(f"| {name} | {'PASS' if gate.get('passed') else 'FAIL'} |")
    lines += ["", "## Checklist", ""]
    for name, gate in (report.get("gates") or {}).items():
        lines.append(f"### Gate {name}")
        lines.append("")
        for check in gate.get("checks") or []:
            mark = "x" if check.get("passed") else " "
            detail = check.get("detail") or ""
            extra = f" — {detail}" if detail else ""
            lines.append(f"- [{mark}] `{check.get('name')}`{extra}")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "The user PDF was not on this VM; predictors and traversals follow the task brief.",
        "See `docs/Matrix_Mantissa_Codec_Guide.md`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matrix mantissa V1–V3 pre-Qwen gates.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/matrix_mantissa_edge_results.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("artifacts/matrix_mantissa_edge_results.md"),
    )
    args = parser.parse_args(argv)
    print(DISCLAIMER, flush=True)
    report = run_all_gates()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(report)
    args.md_out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.md_out} and {args.json_out}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
