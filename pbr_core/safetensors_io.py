"""Safetensors inventory and exact uint16 loads. No FP32 conversion."""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TWO_BYTE_DTYPES = {"BF16", "F16", "FP16"}
_HEADER_LEN = struct.Struct("<Q")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    file_path: Path
    data_start: int

    @property
    def n_words(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    @property
    def nbytes(self) -> int:
        return self.n_words * 2

    @property
    def ndim(self) -> int:
        return len(self.shape)


def list_safetensors_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    files += sorted(model_dir.glob("**/*.safetensors"))
    # Unique, skip extra index shards that are not data if both exist.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is not a Safetensors file")
        (header_len,) = _HEADER_LEN.unpack(raw_len)
        header_bytes = fh.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} truncated Safetensors header")
    header = json.loads(header_bytes.decode("utf-8"))
    return header, 8 + header_len


def inventory_file(path: Path) -> list[TensorSpec]:
    header, data_start = read_header(path)
    specs: list[TensorSpec] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            continue
        dtype = str(info["dtype"])
        shape = tuple(int(x) for x in info["shape"])
        start, end = (int(info["data_offsets"][0]), int(info["data_offsets"][1]))
        specs.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offsets=(start, end),
                file_path=path,
                data_start=data_start,
            )
        )
    return specs


def inventory_model(model_dir: Path) -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    for path in list_safetensors_files(model_dir):
        specs.extend(inventory_file(path))
    specs.sort(key=lambda s: s.name)
    return specs


def load_uint16(spec: TensorSpec) -> np.ndarray:
    """Load F16/BF16 tensor bytes as a uint16 view. Never casts through FP32."""
    if spec.dtype not in TWO_BYTE_DTYPES:
        raise TypeError(
            f"Refusing to archive {spec.name} dtype {spec.dtype}. "
            "Stage 1B only views 16-bit tensors as uint16."
        )
    start = spec.data_start + spec.data_offsets[0]
    end = spec.data_start + spec.data_offsets[1]
    expected = spec.nbytes
    if end - start != expected:
        raise ValueError(
            f"{spec.name}: data length {end - start} != expected {expected}"
        )
    mm = np.memmap(spec.file_path, dtype=np.uint8, mode="r")
    raw = np.array(mm[start:end], dtype=np.uint8, copy=True)
    del mm
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != spec.n_words:
        raise ValueError(f"{spec.name}: {words.size} words != {spec.n_words}")
    return np.array(words, dtype=np.uint16, copy=False).reshape(spec.shape)


def write_uint16_safetensors(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    storage_dtype: str = "BF16",
    metadata: dict | None = None,
) -> None:
    """Write uint16 arrays as a Safetensors file tagged BF16 or F16."""
    if storage_dtype not in TWO_BYTE_DTYPES:
        raise ValueError(f"storage_dtype must be 16-bit, got {storage_dtype}")
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<u2").tobytes()
        header[name] = {
            "dtype": storage_dtype,
            "shape": [int(x) for x in arr.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(_HEADER_LEN.pack(len(header_json)))
        fh.write(header_json)
        for blob in blobs:
            fh.write(blob)


_LAYER_RE = re.compile(r"layers[.\-](\d+)")


def layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def is_linear_weight(name: str) -> bool:
    n = name.lower()
    if "weight" not in n:
        return False
    if any(part in n for part in ("norm", "bias", "embed", "lm_head")):
        return False
    return any(part in n for part in ("attn", "mlp", "proj", "fc", "dense", "linear"))


def is_embedding_weight(name: str) -> bool:
    n = name.lower()
    return "weight" in n and ("embed" in n or n.endswith("lm_head.weight"))
