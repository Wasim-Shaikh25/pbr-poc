"""Load the frozen H95Q-S1 quantized reference (container or model rebuild)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pbr_h95.container_h95q import decode_container as decode_h95q
from pbr_q4.const import FROZEN_S1_SHA

DEFAULT_S1_V2 = Path("artifacts/pbr_h95/containers/H95Q-S1-v2.h95q")
DEFAULT_S1_V1 = Path("artifacts/pbr_h95/containers/H95Q-S1.h95q")


def keep_map_from_header(header: dict[str, Any]) -> tuple[dict[str, int], str | None, np.ndarray | None]:
    keep_map: dict[str, int] = {}
    embed_name = None
    for spec in header["tensors"]:
        if spec["mode"] == "uniform":
            keep_map[spec["name"]] = int(spec["base_keep"])
        else:
            embed_name = spec["name"]
            keep_map[spec["name"]] = -1
    return keep_map, embed_name, None


def load_row_keeps(path: Path, header: dict[str, Any], embed_name: str | None) -> np.ndarray | None:
    if not embed_name:
        return None
    spec = next(s for s in header["tensors"] if s["name"] == embed_name)
    data = path.read_bytes()
    return np.frombuffer(
        data[spec["row_keeps_off"] : spec["row_keeps_off"] + spec["row_keeps_len"]],
        dtype=np.int8,
    ).copy()


def load_s1_from_container(
    path: Path,
    *,
    expect_sha: str = FROZEN_S1_SHA,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    decoded = decode_h95q(path)
    header = decoded["header"]
    ref_sha = header["sha256_quantized_reference"]
    if expect_sha and ref_sha != expect_sha:
        raise RuntimeError(f"S1 SHA drift in {path}: {ref_sha} != {expect_sha}")
    unique = {spec["name"]: decoded["tensors"][spec["name"]] for spec in header["tensors"]}
    keep_map, embed_name, _ = keep_map_from_header(header)
    row_keeps = load_row_keeps(path, header, embed_name)
    return {
        "tensors": unique,
        "keep_map": keep_map,
        "embed_name": embed_name,
        "embed_row_keeps": row_keeps,
        "precision_map": header.get("precision_map") or {},
        "model_id": header.get("model_id") or "qwen",
        "sha256_quantized_reference": ref_sha,
        "source_path": str(path),
        "source": "h95q_container",
    }


def find_s1_container(explicit: Path | None = None) -> Path | None:
    if explicit is not None and Path(explicit).is_file():
        return Path(explicit)
    for p in (DEFAULT_S1_V2, DEFAULT_S1_V1):
        if p.is_file():
            return p
    return None
