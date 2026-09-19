"""Human-readable Stage 2 tables."""

from __future__ import annotations


def format_scan_table(tensors: list[dict], *, limit: int | None = None) -> str:
    rows = list(tensors)
    if limit is not None:
        rows = rows[:limit]
    headers = [
        "tensor",
        "words",
        "H_word",
        "sample_BPW",
        "proj_BPW",
        "exact",
        "modes",
    ]
    str_rows = []
    for row in rows:
        entropy = row.get("entropy") or {}
        str_rows.append(
            [
                str(row.get("name") or row.get("case") or ""),
                str(row.get("n_words", "")),
                f"{float(entropy.get('word_bits', 0.0)):.3f}",
                f"{float(row.get('sample_bpw', 0.0)):.3f}",
                f"{float(row.get('projected_bpw', 0.0)):.3f}",
                str(row.get("exact", "")),
                str(row.get("winning_modes") or "-"),
            ]
        )
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in str_rows)
    return "\n".join(lines)


def format_decision(projection: dict) -> str:
    lines = [
        f"projected BPW: {projection['projected_bpw']:.4f}",
        f"band:          {projection['band']}",
        f"high-potential (≤4 BPW projection): "
        f"{'YES' if projection['high_potential'] else 'NO'} "
        f"— {projection['high_potential_reason']}",
        f"one-GB qualified: NO — {projection['one_gb_qualified_reason']}",
        f"16-bit parameters: {projection['total_16bit_parameters']}",
        f"projected encoded bytes: {projection['projected_encoded_bytes']}",
        f"scanned parameters: {projection['scanned_parameters']}",
        f"unscanned 16-bit parameters (counted at 16 BPW): "
        f"{projection['unscanned_16bit_parameters']}",
    ]
    return "\n".join(lines)
