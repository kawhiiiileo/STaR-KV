"""STaR-KV online MI prior (Algorithm 1 steps 4-7)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


def vision_spans_from_parent(attn_parent, past_seq_len: int) -> List[Tuple[int, int]]:
    if attn_parent is None:
        return []
    cfg = getattr(attn_parent, "config", None)
    vs_list = getattr(attn_parent, "vision_start_idx", None)
    ve_list = getattr(attn_parent, "vision_end_idx", None)
    if vs_list is None and cfg is not None:
        vs_list = getattr(cfg, "vision_start_idx", None)
    if ve_list is None and cfg is not None:
        ve_list = getattr(cfg, "vision_end_idx", None)
    if not vs_list or not ve_list:
        return []
    n = min(len(vs_list), len(ve_list))
    spans: List[Tuple[int, int]] = []
    for i in range(n):
        s = int(vs_list[i])
        e = int(ve_list[i])
        s = max(0, min(s, past_seq_len))
        e = max(s, min(e, past_seq_len))
        if e > s:
            spans.append((s, e))
    return spans


def vision_span_from_parent(attn_parent, past_seq_len: int) -> Optional[Tuple[int, int]]:
    spans = vision_spans_from_parent(attn_parent, past_seq_len)
    return spans[-1] if spans else None


def _normalize_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(x.sum())
    if s <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return x / s


def strict_mi_score_2d(
    p_norm: np.ndarray,
    row_bins: np.ndarray,
    col_bins: np.ndarray,
    n_row_bins: int,
    n_col_bins: int,
    n_val_bins: int,
) -> float:
    n = int(p_norm.size)
    if n <= 1:
        return 0.0
    row_bins = np.clip(row_bins.astype(np.int64), 0, n_row_bins - 1)
    col_bins = np.clip(col_bins.astype(np.int64), 0, n_col_bins - 1)
    y_bins = row_bins * n_col_bins + col_bins
    n_y_bins = int(n_row_bins * n_col_bins)
    if np.unique(y_bins).size <= 1:
        return 0.0

    order = np.argsort(p_norm)
    rank = np.empty_like(order)
    rank[order] = np.arange(n)
    a_bins = np.floor(rank.astype(np.float64) * max(1, n_val_bins) / float(n)).astype(np.int64)
    a_bins = np.clip(a_bins, 0, n_val_bins - 1)
    if np.unique(a_bins).size <= 1:
        return 0.0

    joint = np.zeros((n_val_bins, n_y_bins), dtype=np.float64)
    for a, y in zip(a_bins, y_bins):
        joint[int(a), int(y)] += 1.0
    joint = _normalize_prob(joint)
    pa = _normalize_prob(joint.sum(axis=1, keepdims=True))
    py = _normalize_prob(joint.sum(axis=0, keepdims=True))
    denom = pa @ py
    return max(
        0.0,
        float(np.sum(joint * (np.log(joint + 1e-12) - np.log(denom + 1e-12)))),
    )


def vision_grid_shape(vision_len: int) -> Tuple[int, int]:
    vision_len = max(1, int(vision_len))
    grid_h = max(1, int(round(math.sqrt(float(vision_len)))))
    grid_w = max(1, (vision_len + grid_h - 1) // grid_h)
    return grid_h, grid_w


def mi_from_group_scores(
    S: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_row_bins: int = 12,
    n_col_bins: int = 12,
    n_val_bins: int = 16,
) -> Optional[torch.Tensor]:
    span = vision_span_from_parent(attn_parent, past_seq_len)
    if span is None:
        return None
    v_s, v_e = span
    vis = S[:, v_s:v_e]
    if vis.numel() == 0:
        return None
    v_len = vis.size(1)
    grid_h, grid_w = vision_grid_shape(v_len)
    row_bins = []
    col_bins = []
    for off in range(v_len):
        row = off // grid_w
        col = off % grid_w
        y = (row + 0.5) / float(max(grid_h, 1))
        x = (col + 0.5) / float(max(grid_w, 1))
        row_bins.append(int(max(0, min(n_row_bins - 1, math.floor(y * n_row_bins)))))
        col_bins.append(int(max(0, min(n_col_bins - 1, math.floor(x * n_col_bins)))))
    row_bins_t = torch.tensor(row_bins, device=vis.device, dtype=torch.long)
    col_bins_t = torch.tensor(col_bins, device=vis.device, dtype=torch.long)
    out = []
    for g in range(vis.size(0)):
        p = vis[g].float().clamp_min(0)
        p_sum = float(p.sum().item())
        if p_sum <= 1e-12:
            out.append(0.0)
            continue
        p = (p / p.sum()).detach().cpu().numpy()
        out.append(
            float(
                strict_mi_score_2d(
                    p_norm=p,
                    row_bins=row_bins_t.detach().cpu().numpy(),
                    col_bins=col_bins_t.detach().cpu().numpy(),
                    n_row_bins=n_row_bins,
                    n_col_bins=n_col_bins,
                    n_val_bins=n_val_bins,
                )
            )
        )
    return torch.tensor(out, device=vis.device, dtype=vis.dtype)


def mi_to_group_weights(mi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Map min-max normalized MI EMA to group weights w_g (Algorithm 1 step 6)."""
    if mi.numel() == 0:
        return mi
    mn, mx = mi.min(), mi.max()
    if float((mx - mn).item()) < 1e-12:
        s = torch.full_like(mi, 0.5)
    else:
        s = (mi - mn) / (mx - mn)
    t = max(float(tau), 1e-6)
    w = torch.softmax(s / t, dim=0)
    return w / w.sum().clamp_min(1e-8)


def online_mi_prior(
    S: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_groups: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    Online MI EMA + lambda ramp. Returns (w_g, lambda_eff).
    Steps 4-7; uniform w_g until EMA is ready.
    """
    uniform = torch.full((n_groups,), 1.0 / n_groups, device=device, dtype=dtype)
    if attn_parent is None:
        return uniform, 0.5

    n_steps = int(getattr(attn_parent, "kv_group_online_profile_steps", 0) or 0)
    target_lam = max(0.0, min(1.0, float(getattr(attn_parent, "kv_group_soft_prior_lambda", 0.5) or 0.5)))
    if n_steps <= 0:
        return uniform, target_lam

    step = int(getattr(attn_parent, "kv_group_online_profile_step", 0) or 0) + 1
    setattr(attn_parent, "kv_group_online_profile_step", step)
    decay = max(0.0, min(0.9999, float(getattr(attn_parent, "kv_group_online_profile_decay", 0.9) or 0.9)))
    tau = float(getattr(attn_parent, "kv_group_online_profile_tau", 1.0) or 1.0)

    mi_cur = mi_from_group_scores(S, attn_parent, past_seq_len)
    if mi_cur is not None:
        prev = getattr(attn_parent, "kv_group_online_profile_mi_ema", None)
        if isinstance(prev, torch.Tensor) and prev.numel() == mi_cur.numel():
            mi_ema = decay * prev.to(device=mi_cur.device, dtype=mi_cur.dtype) + (1.0 - decay) * mi_cur
        else:
            mi_ema = mi_cur
        setattr(attn_parent, "kv_group_online_profile_mi_ema", mi_ema.detach())

    ramp = max(1, int(getattr(attn_parent, "kv_group_online_profile_lambda_ramp_steps", n_steps) or n_steps))
    lam_eff = 0.0 if step <= n_steps else target_lam * min(1.0, float(step - n_steps) / float(ramp))

    mi_ema = getattr(attn_parent, "kv_group_online_profile_mi_ema", None)
    if isinstance(mi_ema, torch.Tensor) and mi_ema.numel() == n_groups:
        w_g = mi_to_group_weights(mi_ema.to(device=device, dtype=dtype), tau=tau)
    else:
        w_g = uniform
    return w_g, lam_eff


def apply_mi_prior(
    ac: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_kv: int,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (S, w_g, lambda_eff) for Algorithm 1 steps 2-7."""
    scores_g = []
    for g in range(n_kv):
        hs = g * num_key_value_groups
        he = min((g + 1) * num_key_value_groups, ac.shape[0])
        scores_g.append(ac[hs:he, :].mean(dim=0))
    S = torch.stack(scores_g, dim=0)
    w_g, lam_eff = online_mi_prior(
        S, attn_parent, past_seq_len, n_kv, ac.mean(dim=0).dtype, ac.device
    )
    return S, w_g, lam_eff


__all__ = ["strict_mi_score_2d", "vision_spans_from_parent", "apply_mi_prior"]
