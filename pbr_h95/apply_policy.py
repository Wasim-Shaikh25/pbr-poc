"""Apply per-tensor mantissa keep-bits maps onto a live model / state_dict."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from pbr_h95.quantize import quantize_bf16_mantissas


def bf16_tensor_to_u16(t: torch.Tensor) -> np.ndarray:
    """View BF16 (or float) parameter bits as uint16 words (CPU numpy copy)."""
    x = t.detach().to(dtype=torch.bfloat16).contiguous().cpu()
    return x.view(torch.uint16).numpy().copy()


def u16_to_bf16_tensor(words: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    arr = np.ascontiguousarray(words, dtype=np.uint16)
    cpu = torch.from_numpy(arr).view(torch.bfloat16)
    return cpu.to(device=like.device, dtype=torch.bfloat16)


def estimate_mantissa_bytes_saved(
    keep_map: Mapping[str, int],
    state_dict: Mapping[str, torch.Tensor],
) -> float:
    """Bytes saved vs full 7-bit mantissa: sum (7-keep)*n_words / 8 over unique storages."""
    saved = 0.0
    seen: set[int] = set()
    for name, keep in keep_map.items():
        if name not in state_dict:
            continue
        t = state_dict[name]
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        k = int(keep)
        if k < 0 or k > 7:
            raise ValueError(f"keep_bits out of range for {name}: {k}")
        saved += (7 - k) * int(t.numel()) / 8.0
    return saved


def apply_keep_map_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    keep_map: Mapping[str, int],
    *,
    inplace: bool = True,
) -> dict[str, torch.Tensor]:
    """Quantize tensors named in *keep_map* (skip keep==7). Mutates if inplace.

    Shared storages (e.g. tied embed/lm_head) are quantized once.
    """
    out = state_dict if inplace else {k: v.clone() for k, v in state_dict.items()}
    seen: set[int] = set()
    for name, keep in keep_map.items():
        if name not in out:
            continue
        k = int(keep)
        if k >= 7:
            continue
        t = out[name]
        if t.ndim == 0:
            continue
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        words = bf16_tensor_to_u16(t)
        quant = quantize_bf16_mantissas(words, k)
        new_t = u16_to_bf16_tensor(quant, t)
        # In-place copy preserves sharing with aliases that point at the same storage.
        t.copy_(new_t)
        if not inplace:
            out[name] = t
    return out


def clone_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU BF16 clone that preserves tied-parameter sharing (same data_ptr)."""
    sd = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    ptr_map: dict[int, torch.Tensor] = {}
    for name, tensor in sd.items():
        if not tensor.is_floating_point():
            out[name] = tensor.detach().cpu().clone()
            continue
        ptr = tensor.data_ptr()
        if ptr in ptr_map:
            out[name] = ptr_map[ptr]
            continue
        cloned = tensor.detach().to(dtype=torch.bfloat16).cpu().clone()
        ptr_map[ptr] = cloned
        out[name] = cloned
    return out


def restore_state_dict(model: torch.nn.Module, baseline: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if name in baseline:
                param.copy_(baseline[name].to(device=param.device, dtype=param.dtype))


def apply_policy_to_model(
    model: torch.nn.Module,
    keep_fn: Callable[[str], int],
    *,
    baseline: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Restore from *baseline* (if given), apply keep_fn, load into model.

    Returns keep_map and bytes_saved estimate for floating params in state_dict.
    """
    if baseline is not None:
        # Re-share tied storages when cloning from baseline
        sd: dict[str, torch.Tensor] = {}
        ptr_map: dict[int, torch.Tensor] = {}
        for name, tensor in baseline.items():
            ptr = tensor.data_ptr()
            if ptr in ptr_map:
                sd[name] = ptr_map[ptr]
            else:
                cloned = tensor.clone()
                ptr_map[ptr] = cloned
                sd[name] = cloned
    else:
        sd = clone_cpu_state_dict(model)

    keep_map: dict[str, int] = {}
    n_words_unique: dict[str, int] = {}
    seen: set[int] = set()
    for name, tensor in sd.items():
        if not tensor.is_floating_point():
            continue
        keep_map[name] = int(keep_fn(name))
        ptr = tensor.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            n_words_unique[name] = int(tensor.numel())

    apply_keep_map_to_state_dict(sd, keep_map, inplace=True)
    device = next(model.parameters()).device
    device_sd = {k: v.to(device=device) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(device_sd, strict=False)
    # Re-tie if config requests it (ensures forward uses one storage).
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    bytes_saved = estimate_mantissa_bytes_saved(keep_map, sd)
    return {
        "keep_map": keep_map,
        "n_words_unique": n_words_unique,
        "bytes_saved_vs_bf16_mantissa": bytes_saved,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
