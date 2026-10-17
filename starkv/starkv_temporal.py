"""STaR-KV temporal discount (Algorithm 1 steps 8-9; no-op for single-image)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from starkv.starkv_utils import VisionSpan
from starkv.starkv_mi import _valid_grid_spans
from starkv.starkv_sfr import store_tem_discount


def _span_bounds(span) -> tuple[int, int]:
    if isinstance(span, VisionSpan):
        return int(span.start), int(span.end)
    return int(span[0]), int(span[1])


def _spatial_pattern(
    S: torch.Tensor,
    span: VisionSpan,
    target_hw: tuple[int, int],
) -> Optional[torch.Tensor]:
    """Normalize one frame to a common spatial grid for frame-to-frame cosine."""
    s, e = int(span.start), int(span.end)
    if s < 0 or e > S.shape[-1] or e <= s or span.validate_length() is not None:
        return None
    vis = S[:, s:e].float().clamp_min(0)
    try:
        plane = vis.reshape(
            vis.size(0), int(span.grid_t), int(span.grid_h), int(span.grid_w)
        ).mean(dim=1)
    except RuntimeError:
        return None
    if tuple(plane.shape[-2:]) != tuple(target_hw):
        plane = F.interpolate(
            plane.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=False
        ).squeeze(1)
    pattern = plane.flatten(1)
    return pattern / pattern.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _group_stability_from_ema(
    S: torch.Tensor,
    spans: Sequence,
    attn_parent,
    rho_tem: float,
) -> Optional[torch.Tensor]:
    if attn_parent is None or not spans:
        return None
    device = S.device
    s, e = _span_bounds(spans[-1])
    s = max(0, min(s, S.shape[-1]))
    e = max(s, min(e, S.shape[-1]))
    if e <= s:
        return None
    vis_idx = torch.arange(s, e, device=device, dtype=torch.long)
    A_cur = S[:, vis_idx].float().clamp_min(0)
    A_cur = A_cur / A_cur.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Multi-frame benchmarks already provide ordered history frames in the
    # current prompt. Compute stability against their EMA directly. This avoids
    # leaking state between independently sampled/interleaved episodes and also
    # makes the first multi-frame generate use real temporal evidence.
    grid_spans = [sp for sp in spans if isinstance(sp, VisionSpan)]
    if len(grid_spans) > 1:
        target_hw = (int(grid_spans[-1].grid_h), int(grid_spans[-1].grid_w))
        patterns = [_spatial_pattern(S, sp, target_hw) for sp in grid_spans]
        if all(p is not None for p in patterns):
            history = patterns[0]
            for pattern in patterns[1:-1]:
                history = rho_tem * history + (1.0 - rho_tem) * pattern
            current = patterns[-1]
            stability = F.cosine_similarity(current, history, dim=-1, eps=1e-8).clamp(0.0, 1.0)
            updated = rho_tem * history + (1.0 - rho_tem) * current
            setattr(attn_parent, "kv_group_temporal_pattern_ema", updated.detach())
            setattr(attn_parent, "kv_group_temporal_stability_source", "within_prompt_ema")
            return stability

    prev = getattr(attn_parent, "kv_group_temporal_pattern_ema", None)
    if isinstance(prev, torch.Tensor) and prev.shape == A_cur.shape:
        prev = prev.to(device=device, dtype=A_cur.dtype)
        # Equation (9): compare against history before updating the EMA.
        stability = F.cosine_similarity(A_cur, prev, dim=-1, eps=1e-8).clamp(0.0, 1.0)
        M = rho_tem * prev + (1.0 - rho_tem) * A_cur
    else:
        # The first valid frame initializes history; its discount is identity.
        stability = torch.ones(A_cur.size(0), device=device, dtype=A_cur.dtype)
        M = A_cur
    setattr(attn_parent, "kv_group_temporal_pattern_ema", M.detach())
    setattr(attn_parent, "kv_group_temporal_stability_source", "cross_generate_ema")
    return stability


def _record_stats(attn_parent, stats: dict) -> None:
    setattr(attn_parent, "kv_group_temporal_last_stats", stats)
    setattr(attn_parent, "kv_group_temporal_last_trace", stats)


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
        store_tem_discount(attn_parent, None)
        return scaled
    if past_seq_len <= 0:
        store_tem_discount(attn_parent, None)
        return scaled

    spans = _valid_grid_spans(attn_parent, past_seq_len)
    if not spans:
        _record_stats(attn_parent, {
            "mode": "identity", "applied": False,
            "reason": "no_valid_visual_spans", "valid_pair_count": 0,
        })
        store_tem_discount(attn_parent, None)
        return scaled

    rho_tem = max(0.0, min(0.9999, float(getattr(attn_parent, "kv_group_temporal_rho", 0.9))))
    stab = _group_stability_from_ema(S, spans, attn_parent, rho_tem=rho_tem)
    if stab is None or stab.numel() != n_groups:
        _record_stats(attn_parent, {
            "mode": "identity", "applied": False,
            "reason": "invalid_stability", "valid_pair_count": 0,
        })
        store_tem_discount(attn_parent, None)
        return scaled
    if len(spans) == 1:
        _record_stats(attn_parent, {
            "mode": "identity", "applied": False,
            "reason": "single_frame_identity", "valid_pair_count": 0,
            "group_stability": stab.detach(),
        })
        store_tem_discount(attn_parent, None)
        return scaled

    discount_min = max(0.0, min(1.0, float(getattr(attn_parent, "kv_group_temporal_discount_min", 0.0))))
    delta_coef = float(getattr(attn_parent, "kv_group_temporal_delta", 0.1))
    tem_stability_floor = max(0.0, min(1.0, float(getattr(attn_parent, "kv_group_temporal_eps", 0.0))))
    decay_mode = str(getattr(attn_parent, "kv_group_temporal_decay_mode", "exponential") or "exponential").lower()
    if decay_mode != "exponential":
        raise ValueError("the canonical STaR-KV path supports exponential TEM only")

    span_bounds = [_span_bounds(sp) for sp in spans]
    delta_frame = _frame_distance_vector(span_bounds, past_seq_len, scaled.device, torch.float32)
    g_star = S.argmax(dim=0).long()
    stab_f = stab.float()
    if stab_f.numel() != int(S.size(0)):
        _record_stats(attn_parent, {
            "mode": "identity", "applied": False,
            "reason": "stability_subspace_mismatch",
            "valid_pair_count": 0,
            "n_stability": int(stab_f.numel()),
            "n_subspace": int(S.size(0)),
        })
        store_tem_discount(attn_parent, None)
        return scaled
    phi_t = tem_stability_floor + (1.0 - tem_stability_floor) * stab_f[g_star]
    eta = delta_coef * delta_frame * phi_t
    raw_D = torch.exp(-eta)
    D = discount_min + (1.0 - discount_min) * raw_D

    warmup = int(getattr(attn_parent, "kv_group_temporal_warmup_steps", 0) or 0)
    step = int(getattr(attn_parent, "kv_group_temporal_step", 0) or 0)
    if step <= warmup:
        _record_stats(attn_parent, {
            "mode": "identity", "applied": False,
            "reason": "warmup", "valid_pair_count": len(spans) - 1,
            "step": step, "group_stability": stab.detach(),
        })
        store_tem_discount(attn_parent, None)
        return scaled

    _record_stats(attn_parent, {
        "mode": decay_mode,
        "applied": True,
        "reason": "applied",
        "valid_pair_count": len(spans) - 1,
        "step": step,
        "rho": rho_tem,
        "delta": delta_coef,
        "discount_min": discount_min,
        "decay_mode": decay_mode,
        "group_stability": stab.detach(),
        "stability_source": getattr(attn_parent, "kv_group_temporal_stability_source", None),
        "discount_vector": D.detach(),
    })
    store_tem_discount(attn_parent, D)
    return (scaled.float() * D.detach()).to(dtype=scaled.dtype)


__all__ = ["temporal_apply_to_scaled"]
