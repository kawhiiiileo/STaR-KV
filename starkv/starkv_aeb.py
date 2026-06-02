"""AEB: entropy-guided temperature for fixed_score KV sharpening."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch


def compute_entropy_temperature(
    attn_cache: torch.Tensor,
    min_temp: float,
    max_temp: float,
    smooth_ema: float,
    past_temperature: Optional[float],
) -> Tuple[float, dict[str, Any]]:
    """Map pooled attention entropy to temperature T (budget unchanged)."""
    eps = 1e-8
    score = attn_cache.unsqueeze(0) if attn_cache.dim() == 2 else attn_cache
    score = score.mean(dim=1)
    sum_score = score.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = score / sum_score

    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    l_past = score.shape[-1]
    max_entropy = math.log(max(1, l_past))
    normalized_entropy = entropy / (max_entropy + eps)
    temperature = min_temp + (max_temp - min_temp) * normalized_entropy

    if smooth_ema > 0 and past_temperature is not None:
        temperature = smooth_ema * past_temperature + (1 - smooth_ema) * temperature

    temp_scalar = temperature.mean().item() if isinstance(temperature, torch.Tensor) else temperature
    debug = {
        "normalized_entropy": normalized_entropy.mean().item()
        if isinstance(normalized_entropy, torch.Tensor)
        else normalized_entropy,
        "entropy": entropy.mean().item() if isinstance(entropy, torch.Tensor) else entropy,
        "max_entropy": max_entropy,
        "temperature": temp_scalar,
    }
    return temp_scalar, debug


def update_aeb_from_attn_cache(cluster, attn_cache: torch.Tensor) -> None:
    """Refresh cluster EMA temperature from attn_parent AEB settings."""
    parent = getattr(cluster, "attn_parent", None)
    if parent is None or not getattr(parent, "kv_entropy_budget_enable", False):
        cluster._aeb_temperature_ema = None
        cluster._aeb_last_debug = None
        return

    temperature, aeb_debug = compute_entropy_temperature(
        attn_cache,
        getattr(parent, "kv_entropy_budget_min_scale", 0.95),
        getattr(parent, "kv_entropy_budget_max_scale", 1.05),
        getattr(parent, "kv_entropy_budget_smooth", 0.0),
        getattr(cluster, "_aeb_temperature_ema", None),
    )
    cluster._aeb_temperature_ema = temperature
    cluster._aeb_last_debug = aeb_debug


def sharpen_scores_with_aeb(scaled: torch.Tensor, cluster) -> torch.Tensor:
    """Apply score ^= 1/T when AEB is enabled on attn_parent."""
    parent = getattr(cluster, "attn_parent", None)
    if parent is None or not getattr(parent, "kv_entropy_budget_enable", False):
        return scaled
    temp = getattr(cluster, "_aeb_temperature_ema", None)
    if temp is None or temp <= 0:
        return scaled
    return scaled.clamp(min=0.0) ** (1.0 / temp)
