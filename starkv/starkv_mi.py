"""STaR-KV online MI prior (Algorithm 1 steps 4-7)."""

from __future__ import annotations

from typing import Optional, Tuple

from starkv.starkv_utils import vision_spans_from_parent as _vision_spans_with_grid

import numpy as np
import torch


def _valid_grid_spans(attn_parent, past_seq_len: int):
    spans, err = _vision_spans_with_grid(attn_parent)
    if err is not None or not spans:
        return []
    return [sp for sp in spans if 0 <= sp.start < sp.end <= int(past_seq_len) and sp.validate_length() is None]

def vision_span_from_parent(attn_parent, past_seq_len: int) -> Optional[Tuple[int, int]]:
    spans = _valid_grid_spans(attn_parent, past_seq_len)
    return (int(spans[-1].start), int(spans[-1].end)) if spans else None


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

    order = np.argsort(p_norm, kind="stable")
    sorted_values = p_norm[order]
    # Equal values must share one rank bin. Splitting ties by raster position
    # creates artificial MI (a uniform attention map would otherwise look
    # strongly coupled to screen coordinates).
    rank = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        rank[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    a_bins = np.floor(rank * max(1, n_val_bins) / float(n)).astype(np.int64)
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



def mi_from_group_scores(
    S: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_row_bins: int = 12,
    n_col_bins: int = 12,
    n_val_bins: int = 16,
) -> Optional[torch.Tensor]:
    spans = _valid_grid_spans(attn_parent, past_seq_len)
    if not spans:
        return None
    span = spans[-1]
    vis = S[:, span.start:span.end]
    if vis.numel() == 0:
        return None
    plane = int(span.grid_h) * int(span.grid_w)
    if plane <= 0 or int(span.length) != int(span.grid_t) * plane:
        return None
    row_bins = []
    col_bins = []
    for off in range(int(span.length)):
        plane_off = off % plane
        row = plane_off // int(span.grid_w)
        col = plane_off % int(span.grid_w)
        row_bins.append(min(n_row_bins - 1, row * n_row_bins // int(span.grid_h)))
        col_bins.append(min(n_col_bins - 1, col * n_col_bins // int(span.grid_w)))
    row_np = np.asarray(row_bins, dtype=np.int64)
    col_np = np.asarray(col_bins, dtype=np.int64)
    out = []
    for g in range(vis.size(0)):
        p = vis[g].float().clamp_min(0)
        if float(p.sum().item()) <= 1e-12:
            out.append(0.0)
            continue
        p_np = (p / p.sum()).detach().cpu().numpy()
        out.append(strict_mi_score_2d(
            p_np, row_np, col_np, n_row_bins, n_col_bins, n_val_bins
        ))
    return torch.tensor(out, device=vis.device, dtype=torch.float32)


def mi_to_group_weights(mi: torch.Tensor, tau_mi: float = 1.0) -> torch.Tensor:
    """Map min-max normalized MI EMA to group weights w_g (Algorithm 1 step 6)."""
    if mi.numel() == 0:
        return mi
    mn, mx = mi.min(), mi.max()
    if float((mx - mn).item()) < 1e-12:
        s = torch.full_like(mi, 0.5)
    else:
        s = (mi - mn) / (mx - mn)
    tau_mi = max(float(tau_mi), 1e-6)
    w = torch.softmax(s / tau_mi, dim=0)
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

    target_lam_raw = getattr(attn_parent, "kv_group_soft_prior_lambda", 0.5)
    target_lam = max(0.0, min(1.0, float(0.5 if target_lam_raw is None else target_lam_raw)))
    n_steps = int(getattr(attn_parent, "kv_group_online_profile_steps", 0) or 0)

    update_step = int(getattr(attn_parent, "kv_group_online_profile_step", 0) or 0) + 1
    setattr(attn_parent, "kv_group_online_profile_step", update_step)
    decay_raw = getattr(attn_parent, "kv_group_online_profile_decay", 0.9)
    rho_mi = max(0.0, min(0.9999, float(0.9 if decay_raw is None else decay_raw)))
    tau_mi = float(getattr(attn_parent, "kv_group_online_profile_tau", 1.0) or 1.0)

    mi_cur = mi_from_group_scores(S, attn_parent, past_seq_len)
    valid_spans = _valid_grid_spans(attn_parent, past_seq_len)
    # Independent prefills (ScreenSpot / AndroidControl) reset online state before
    # every generate, so the paper warm-up/ramp never completes. Allow the current
    # MI estimate to apply immediately. ``current_frame_mi`` covers multi-image
    # prompts with history frames; ``single_image_mi`` remains the single-span alias.
    current_frame_immediate = bool(
        mi_cur is not None
        and (
            getattr(attn_parent, "kv_group_current_frame_mi_enable", False)
            or (
                getattr(attn_parent, "kv_group_single_image_mi_enable", False)
                and len(valid_spans) == 1
            )
        )
    )
    if mi_cur is not None:
        prev = getattr(attn_parent, "kv_group_online_profile_mi_ema", None)
        if isinstance(prev, torch.Tensor) and prev.numel() == mi_cur.numel():
            mi_ema = rho_mi * prev.to(device=mi_cur.device, dtype=mi_cur.dtype) + (1.0 - rho_mi) * mi_cur
        else:
            mi_ema = mi_cur
        setattr(attn_parent, "kv_group_online_profile_mi_ema", mi_ema.detach())

    ramp_raw = getattr(attn_parent, "kv_group_online_profile_lambda_ramp_steps", n_steps)
    ramp = max(1, int(n_steps if ramp_raw is None else ramp_raw))
    if current_frame_immediate:
        # A one-shot estimate must not be trusted as much as a converged EMA.
        # The confidence is the cumulative mass assigned to observations by the
        # existing EMA, c_n = 1 - rho_mi**n. This keeps the paper MI estimator,
        # group routing and multiplicative prior intact while preventing a
        # single independent prefill from applying the full target lambda.
        confidence = 1.0 - rho_mi ** max(update_step, 1)
        lam_eff = target_lam * max(0.0, min(1.0, confidence))
        profile_mode = "current_frame_confidence"
    else:
        lam_eff = 0.0 if update_step <= n_steps else target_lam * min(
            1.0, float(update_step - n_steps) / float(ramp)
        )
        profile_mode = "online_ema_ramp"

    mi_ema = getattr(attn_parent, "kv_group_online_profile_mi_ema", None)
    if isinstance(mi_ema, torch.Tensor) and mi_ema.numel() == n_groups:
        w_g = mi_to_group_weights(mi_ema.to(device=device, dtype=torch.float32), tau_mi=tau_mi)
    else:
        w_g = uniform
    setattr(attn_parent, "kv_group_online_profile_last_lam_eff", float(lam_eff))
    setattr(
        attn_parent,
        "kv_group_online_profile_last_confidence",
        float(lam_eff / target_lam) if target_lam > 0.0 else 0.0,
    )
    setattr(attn_parent, "kv_group_online_profile_last_mode", profile_mode)
    setattr(attn_parent, "kv_group_online_profile_last_group_weights", w_g.detach())
    return w_g.to(dtype=dtype), lam_eff


def apply_mi_prior(
    ac: torch.Tensor,
    attn_parent,
    past_seq_len: int,
    n_kv: int,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return canonical GQA-group scores, weights, and effective MI lambda."""
    S = gqa_group_scores(ac, n_kv, num_key_value_groups)
    w_g, lam_eff = online_mi_prior(
        S, attn_parent, past_seq_len, n_kv, ac.mean(dim=0).dtype, ac.device
    )
    return S, w_g, lam_eff


def gqa_group_scores(
    ac: torch.Tensor,
    n_kv: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Return canonical GQA-group scores, independent of MI ablations."""
    if ac.ndim != 2:
        raise ValueError(f"attention scores must have shape [heads, tokens], got {tuple(ac.shape)}")
    if int(n_kv) <= 0 or int(num_key_value_groups) <= 0:
        raise ValueError("n_kv and num_key_value_groups must be positive")
    expected_heads = int(n_kv) * int(num_key_value_groups)
    if int(ac.shape[0]) != expected_heads:
        raise ValueError(
            f"GQA layout mismatch: got {ac.shape[0]} query heads, expected "
            f"{n_kv}*{num_key_value_groups}={expected_heads}"
        )
    scores_g = []
    for g in range(n_kv):
        hs = g * num_key_value_groups
        he = min((g + 1) * num_key_value_groups, ac.shape[0])
        scores_g.append(ac[hs:he, :].mean(dim=0))
    return torch.stack(scores_g, dim=0)


__all__ = [
    "strict_mi_score_2d",
    "apply_mi_prior",
    "gqa_group_scores",
    "mi_to_group_weights",
]
