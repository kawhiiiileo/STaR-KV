"""STaR-KV temporal discount (Algorithm 1 steps 8-9; no-op for single-image)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from starkv_mi import vision_spans_from_parent


def _group_stability_from_ema(
    S: torch.Tensor,
    spans: List[tuple[int, int]],
    attn_parent,
    rho: float,
) -> Optional[torch.Tensor]:
    if attn_parent is None or not spans:
        return None
    device = S.device
    s, e = spans[-1]
    s = max(0, min(s, S.shape[-1]))
    e = max(s, min(e, S.shape[-1]))
    if e <= s:
        return None
    vis_idx = torch.arange(s, e, device=device, dtype=torch.long)
    A_cur = S[:, vis_idx].clamp_min(0)
    A_cur = A_cur / A_cur.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    prev = getattr(attn_parent, "kv_group_temporal_pattern_ema", None)
    if isinstance(prev, torch.Tensor) and prev.shape == A_cur.shape:
        M = rho * prev.to(device=device, dtype=A_cur.dtype) + (1.0 - rho) * A_cur
    else:
        M = A_cur
    setattr(attn_parent, "kv_group_temporal_pattern_ema", M.detach())
    return F.cosine_similarity(A_cur, M, dim=-1, eps=1e-8).clamp(0.0, 1.0)


def _frame_distance_vector(
    spans: List[tuple[int, int]],
    past_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    delta = torch.zeros(past_seq_len, device=device, dtype=dtype)
    n = len(spans)
    for i, (s, e) in enumerate(spans):
        if e > s:
            delta[s:e] = float(max(0, n - 1 - i))
    return delta


def temporal_apply_to_scaled(
    scaled: torch.Tensor,
    S: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_groups: int,
) -> torch.Tensor:
    if attn_parent is None or not getattr(attn_parent, "kv_group_temporal_enable", False):
        return scaled
    if past_seq_len <= 0:
        return scaled

    spans = vision_spans_from_parent(attn_parent, past_seq_len)
    if len(spans) <= 1:
        return scaled

    rho = max(0.0, min(0.9999, float(getattr(attn_parent, "kv_group_temporal_rho", 0.9))))
    stab = _group_stability_from_ema(S, spans, attn_parent, rho=rho)
    if stab is None or stab.numel() != n_groups:
        return scaled

    discount_min = max(0.0, min(1.0, float(getattr(attn_parent, "kv_group_temporal_discount_min", 0.0))))
    delta_coef = float(getattr(attn_parent, "kv_group_temporal_delta", 0.1))
    eps = max(0.0, min(1.0, float(getattr(attn_parent, "kv_group_temporal_eps", 0.0))))

    delta_frame = _frame_distance_vector(spans, past_seq_len, scaled.device, scaled.dtype)
    g_star = S.argmax(dim=0).long()
    phi_t = eps + (1.0 - eps) * stab[g_star]
    raw_D = torch.exp(-delta_coef * delta_frame * phi_t)
    D = discount_min + (1.0 - discount_min) * raw_D

    warmup = int(getattr(attn_parent, "kv_group_temporal_warmup_steps", 0) or 0)
    step = int(getattr(attn_parent, "kv_group_temporal_step", 0) or 0)
    if step <= warmup:
        return scaled

    return scaled * D.detach()


__all__ = ["temporal_apply_to_scaled"]
