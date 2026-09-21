"""Sensitivity ranking and budgeted mixed-precision assignment.

Granularity: tensor → projection → contiguous output-channel groups
(default 16 rows). Structured protect, not sparse per-weight exceptions.

Start every unit at the family minimum, then greedily spend the physical-bit
budget on the upgrades with the best importance / extra-bits. Protected
channels take Q6 / Q8 / BF16 from the protect set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pbr_q4.e2_mixed.bits import as_rank2, mse_scale, n_groups_for, packed_field_bits
from pbr_q4.e2_mixed.const import CHANNEL_GROUP, GROUP_SIZE
from pbr_q4.e2_mixed.families import (
    candidate_bits,
    family_importance,
    family_label,
    proj_importance,
    projection_label,
)


@dataclass
class ChannelUnit:
    name: str
    family: str
    projection: str
    row0: int
    nrows: int
    ncols: int
    importance: float
    candidates: tuple[int, ...]
    n_weights: int = 0

    def __post_init__(self) -> None:
        self.n_weights = int(self.nrows) * int(self.ncols)


@dataclass
class Allocation:
    row_bits: dict[str, np.ndarray]  # name → uint8[rows]
    units: list[ChannelUnit]
    unit_bits: list[int]
    packed_est_bits: int
    packed_est_bpw: float
    n_weights: int
    budget_bpw: float
    bit_hist_words: dict[str, int] = field(default_factory=dict)
    family_hist: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_protected_units: int = 0
    note: str = ""


def _row_l2(mat_f32: np.ndarray) -> np.ndarray:
    sq = np.square(np.where(np.isfinite(mat_f32), mat_f32, 0.0))
    return np.sqrt(sq.sum(axis=1) + 1e-12)


def channel_importance(
    words: np.ndarray,
    *,
    family: str,
    projection: str,
    row_boost: np.ndarray | None = None,
    act_scale: float = 1.0,
) -> np.ndarray:
    """Per-row importance. Structured: later grouped into contiguous blocks."""
    arr2, _orig, rows, _cols = as_rank2(np.ascontiguousarray(words, dtype=np.uint16))
    from pbr_q4.e2_mixed.bits import bf16_u16_to_f32

    l2 = _row_l2(bf16_u16_to_f32(arr2))
    l2 = l2 / (float(np.median(l2)) + 1e-12)
    imp = l2 * family_importance(family) * proj_importance(projection) * float(act_scale)
    if row_boost is not None:
        b = np.asarray(row_boost, dtype=np.float32).reshape(rows)
        imp = imp * b
    return imp.astype(np.float64)


def embed_row_boost(n_rows: int, freq: Mapping[int, int] | None, special_ids: Sequence[int] | None = None) -> np.ndarray:
    """Boost calib-seen / special embedding rows; never-seen stay at 1.0."""
    boost = np.ones(n_rows, dtype=np.float32)
    if freq:
        for tid, c in freq.items():
            i = int(tid)
            if 0 <= i < n_rows:
                boost[i] = max(boost[i], float(1.0 + np.log1p(c)))
    if special_ids:
        for i in special_ids:
            if 0 <= int(i) < n_rows:
                boost[int(i)] = max(boost[int(i)], 4.0)
    return boost


def build_units(
    tensors: Mapping[str, np.ndarray],
    *,
    floor: int = 4,
    channel_group: int = CHANNEL_GROUP,
    act_scale: Mapping[str, float] | None = None,
    embed_freq: Mapping[int, int] | None = None,
    special_ids: Sequence[int] | None = None,
) -> list[ChannelUnit]:
    """Slice each tensor into contiguous output-channel groups."""
    units: list[ChannelUnit] = []
    act_scale = dict(act_scale or {})
    for name in sorted(tensors.keys()):
        words = np.ascontiguousarray(tensors[name], dtype=np.uint16)
        arr2, _orig, rows, cols = as_rank2(words)
        fam = family_label(name)
        proj = projection_label(name)
        cands = candidate_bits(fam, floor=floor, allow_protect=True)
        boost = None
        if fam == "embed":
            boost = embed_row_boost(rows, embed_freq, special_ids)
        imp_rows = channel_importance(
            arr2,
            family=fam,
            projection=proj,
            row_boost=boost,
            act_scale=float(act_scale.get(name, 1.0)),
        )
        cg = int(channel_group) if rows >= int(channel_group) else max(1, int(rows))
        # 1-D / tiny tensors: one unit (structured, not per-weight).
        if rows == 1 or fam == "norm_bias" or rows < 4:
            units.append(
                ChannelUnit(
                    name=name,
                    family=fam,
                    projection=proj,
                    row0=0,
                    nrows=rows,
                    ncols=cols,
                    importance=float(imp_rows.mean() * rows),
                    candidates=cands,
                )
            )
            continue
        for r0 in range(0, rows, cg):
            n = min(cg, rows - r0)
            units.append(
                ChannelUnit(
                    name=name,
                    family=fam,
                    projection=proj,
                    row0=r0,
                    nrows=n,
                    ncols=cols,
                    importance=float(imp_rows[r0 : r0 + n].sum()),
                    candidates=cands,
                )
            )
    return units


def _unit_cost(u: ChannelUnit, bits: int, group_size: int = GROUP_SIZE) -> int:
    ng = 0 if bits >= 16 else n_groups_for(u.nrows, u.ncols, group_size)
    return packed_field_bits(u.n_weights, bits, ng)


def allocate(
    units: Sequence[ChannelUnit],
    *,
    budget_bpw: float,
    n_weights: int,
    group_size: int = GROUP_SIZE,
    map_bits_per_row: int = 8,
) -> Allocation:
    """Greedy importance-structured assignment under a packed-bit budget."""
    bits = [int(min(u.candidates)) for u in units]
    row_count = {}
    for u in units:
        row_count[u.name] = row_count.get(u.name, 0) + int(u.nrows)
    map_bits = sum(row_count.values()) * int(map_bits_per_row)
    total = sum(_unit_cost(u, b, group_size) for u, b in zip(units, bits)) + map_bits
    budget_bits = float(budget_bpw) * float(n_weights)

    def next_bits(u: ChannelUnit, cur: int) -> int | None:
        higher = [b for b in u.candidates if b > cur]
        return int(higher[0]) if higher else None

    # Greedy: spend leftover budget on the best Δerror / Δbits upgrades.
    while True:
        best_i = -1
        best_nb = 0
        best_gain = 0.0
        best_dc = 0
        for i, u in enumerate(units):
            nb = next_bits(u, bits[i])
            if nb is None:
                continue
            dc = _unit_cost(u, nb, group_size) - _unit_cost(u, bits[i], group_size)
            if dc <= 0:
                continue
            if total + dc > budget_bits + 1e-6:
                continue
            dE = u.importance * (mse_scale(bits[i]) - mse_scale(nb))
            gain = dE / float(dc)
            # Tie-break: higher importance, then lower name/row for determinism.
            if gain > best_gain + 1e-18 or (
                abs(gain - best_gain) <= 1e-18 and (best_i < 0 or (u.importance, -u.row0, u.name) > (units[best_i].importance, -units[best_i].row0, units[best_i].name))
            ):
                best_gain = gain
                best_i = i
                best_nb = nb
                best_dc = dc
        if best_i < 0:
            break
        total += best_dc
        bits[best_i] = best_nb

    row_bits: dict[str, np.ndarray] = {}
    bit_hist: dict[str, int] = {}
    fam_hist: dict[str, dict[str, Any]] = {}
    n_protect = 0
    for u, b in zip(units, bits):
        slot = row_bits.get(u.name)
        if slot is None:
            n_rows = row_count[u.name]
            slot = np.zeros(n_rows, dtype=np.uint8)
            row_bits[u.name] = slot
        slot[u.row0 : u.row0 + u.nrows] = np.uint8(b)
        key = "bf16" if b >= 16 else str(b)
        bit_hist[key] = bit_hist.get(key, 0) + u.n_weights
        fh = fam_hist.setdefault(u.family, {"n_words": 0, "sum_bits": 0.0, "n_protect_words": 0})
        fh["n_words"] += u.n_weights
        fh["sum_bits"] += float(b) * u.n_weights
        if b >= 6 and u.family != "norm_bias":
            n_protect += 1
            fh["n_protect_words"] += u.n_weights
    for fh in fam_hist.values():
        nw = fh["n_words"]
        fh["avg_bits"] = (fh["sum_bits"] / nw) if nw else 0.0
        del fh["sum_bits"]

    packed_est = int(total)
    bpw = packed_est / float(n_weights) if n_weights else 0.0
    return Allocation(
        row_bits=row_bits,
        units=list(units),
        unit_bits=list(bits),
        packed_est_bits=packed_est,
        packed_est_bpw=round(bpw, 6),
        n_weights=int(n_weights),
        budget_bpw=float(budget_bpw),
        bit_hist_words=bit_hist,
        family_hist=fam_hist,
        n_protected_units=n_protect,
        note=(
            "Packed estimate = groupwise codes + f16 scales + u8 zp + 8-bit/row "
            "precision map. No container header / E3 codec. Structured channel "
            "groups, not sparse per-weight exceptions."
        ),
    )


def allocation_row_bits(alloc: Allocation, name: str, n_rows: int) -> np.ndarray:
    if name in alloc.row_bits:
        rb = np.asarray(alloc.row_bits[name], dtype=np.uint8)
        if int(rb.size) != int(n_rows):
            raise ValueError(f"row_bits length {rb.size} != rows {n_rows} for {name}")
        return rb
    return np.full(n_rows, 16, dtype=np.uint8)
