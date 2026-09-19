"""Download a public HF checkpoint and select 16-bit weight tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pbr_core.safetensors_io import (
    TensorSpec,
    inventory_model,
    is_embedding_weight,
    is_linear_weight,
    layer_index,
)

PRIMARY_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
PRIMARY_LICENSE = "Apache-2.0"
FALLBACK_LICENSE = "Apache-2.0"


@dataclass
class DownloadResult:
    repo_id: str
    revision: str
    commit: str
    license: str
    local_dir: Path
    fallback_used: bool
    note: str


def _repo_info(repo_id: str, revision: str):
    from huggingface_hub import HfApi

    return HfApi().repo_info(repo_id=repo_id, revision=revision)


def _snapshot(repo_id: str, revision: str, local_dir: Path, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
            allow_patterns=["*.safetensors", "config.json", "LICENSE", "README.md"],
        )
    )


def download_checkpoint(
    *,
    repo_id: str = PRIMARY_REPO,
    revision: str = "main",
    local_dir: Path,
    cache_dir: Path,
    allow_fallback: bool = True,
) -> DownloadResult:
    """Download weights. Prefer the primary Qwen checkpoint; fall back if needed."""
    tried = []
    candidates = [(repo_id, False, PRIMARY_LICENSE if repo_id == PRIMARY_REPO else "see model card")]
    if allow_fallback and repo_id == PRIMARY_REPO:
        candidates.append((FALLBACK_REPO, True, FALLBACK_LICENSE))

    last_error: Exception | None = None
    for name, is_fallback, license_id in candidates:
        try:
            info = _repo_info(name, revision if not is_fallback else "main")
            commit = getattr(info, "sha", None) or revision
            dest = local_dir / name.replace("/", "__")
            path = _snapshot(name, commit, dest, cache_dir)
            note = "primary checkpoint" if not is_fallback else (
                f"Fell back to {name} after {repo_id} failed: {last_error}"
            )
            if is_fallback:
                tried.append(str(last_error))
            return DownloadResult(
                repo_id=name,
                revision=commit,
                commit=commit,
                license=license_id,
                local_dir=path,
                fallback_used=is_fallback,
                note=note,
            )
        except Exception as exc:  # noqa: BLE001 — download/network, then fallback
            last_error = exc
            tried.append(f"{name}: {exc}")
            if not allow_fallback:
                raise
            continue
    raise RuntimeError("Could not download a Stage 1B checkpoint:\n" + "\n".join(tried))


def select_weight_specs(
    specs: list[TensorSpec],
    *,
    min_bytes: int,
    max_bytes: int,
    include_embeddings: bool = False,
) -> list[TensorSpec]:
    """Pick 2-D 16-bit linear/attention weights up to a complete-byte budget.

    Tensors are taken from early, middle, and late layers so the sample is not
    only the easiest prefix of the checkpoint.
    """
    eligible = [
        s
        for s in specs
        if s.dtype in {"BF16", "F16", "FP16"} and s.ndim == 2 and is_linear_weight(s.name)
    ]
    if include_embeddings:
        eligible.extend(
            s
            for s in specs
            if s.dtype in {"BF16", "F16", "FP16"} and s.ndim == 2 and is_embedding_weight(s.name)
        )

    if not eligible:
        # Last resort: any 2-D 16-bit tensor.
        eligible = [s for s in specs if s.dtype in {"BF16", "F16", "FP16"} and s.ndim == 2]
    if not eligible:
        raise ValueError("No 16-bit 2-D tensors found in the checkpoint")

    layers = sorted({layer_index(s.name) for s in eligible if layer_index(s.name) is not None})
    groups: list[list[TensorSpec]] = []
    if layers:
        first, mid, last = layers[0], layers[len(layers) // 2], layers[-1]
        want = [first, mid, last]
        for extra in layers:
            if extra not in want:
                want.append(extra)
        for idx in want:
            groups.append(sorted([s for s in eligible if layer_index(s.name) == idx], key=lambda s: s.name))
        leftovers = [s for s in eligible if layer_index(s.name) is None]
        if leftovers:
            groups.append(sorted(leftovers, key=lambda s: -s.nbytes))
    else:
        groups.append(sorted(eligible, key=lambda s: -s.nbytes))

    chosen: list[TensorSpec] = []
    total = 0
    for group in groups:
        for spec in group:
            if spec in chosen:
                continue
            if total >= max_bytes:
                break
            if total + spec.nbytes > max_bytes and total >= min_bytes:
                continue
            if total + spec.nbytes > max_bytes and total < min_bytes and spec.nbytes > max_bytes:
                # Single tensor larger than the cap: still take it if it is the
                # only way to reach a real-weight sample, but prefer not to.
                if not chosen:
                    chosen.append(spec)
                    total += spec.nbytes
                continue
            if total + spec.nbytes > max_bytes:
                continue
            chosen.append(spec)
            total += spec.nbytes
        if total >= max_bytes:
            break

    if total < min_bytes:
        for spec in sorted(eligible, key=lambda s: -s.nbytes):
            if spec in chosen:
                continue
            if total >= min_bytes:
                break
            if total + spec.nbytes > max_bytes and chosen:
                continue
            chosen.append(spec)
            total += spec.nbytes

    if not chosen:
        raise ValueError("Tensor selection produced an empty set")
    return chosen


def inventory_from_dir(model_dir: Path) -> list[TensorSpec]:
    return inventory_model(model_dir)
