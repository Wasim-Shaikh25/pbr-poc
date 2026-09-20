"""Calibration-proxy mean token NLL / perplexity for H95 Phase B.

Honesty: this is a fixed in-repo calibration corpus score under teacher forcing.
It is NOT a held-out evaluation gate. Phase C owns ≥95% held-out quality.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def mean_token_nll(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 256,
    device: torch.device | None = None,
) -> dict:
    """Return mean NLL (nats) over all non-masked next-token positions.

    Uses causal LM shift: predict token t from context <t. Batch size 1.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_text = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                per_text.append({"n_tokens": 0, "mean_nll": None, "skipped": True})
                continue
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # [1, T, V]
            # shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            if attn is not None:
                shift_mask = attn[:, 1:].contiguous().bool()
            else:
                shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            tok_nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            tok_nll = tok_nll.masked_select(shift_mask)
            n = int(tok_nll.numel())
            if n == 0:
                per_text.append({"n_tokens": 0, "mean_nll": None, "skipped": True})
                continue
            s = float(tok_nll.sum().item())
            mean = s / n
            total_nll += s
            total_tokens += n
            per_text.append({"n_tokens": n, "mean_nll": mean, "skipped": False})

    if total_tokens == 0:
        raise RuntimeError("No tokens scored; check calibration texts / max_length")
    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    return {
        "mean_nll": mean_nll,
        "ppl": ppl,
        "n_tokens": total_tokens,
        "n_texts": len(texts),
        "per_text": per_text,
        "max_length": max_length,
        "label": "calibration_proxy_only",
    }


def ppl_retention(baseline_ppl: float, quantized_ppl: float) -> float:
    """bf16_ppl / quant_ppl — values near 1.0 mean little degradation (lower ppl is better)."""
    if quantized_ppl <= 0:
        return float("nan")
    return baseline_ppl / quantized_ppl
