# -*- coding: utf-8 -*-
"""
STaR-KV grouped compression (soft_global): pool + spatial boost + MI + temporal + AEB + top-k.
================================================================================

Uses recent-window attention pooling over past tokens. Token scores are head-mean
attention, scaled by group priors (see starkv_mi), optional temporal (starkv_temporal)
and AEB sharpening (starkv_aeb), then global top-k gather.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starkv_aeb import sharpen_scores_with_aeb, update_aeb_from_attn_cache
from starkv_mi import apply_mi_prior, vision_span_from_parent
from starkv_temporal import temporal_apply_to_scaled
from starkv_utils import _RecentWindowKVPool


def apply_starkv_spatial_to_attn_cache(
    attn_cache: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    attn_parent,
) -> torch.Tensor:
    """Boost visual tokens in attn_cache with alpha * softmax(norm(hidden)/tau)."""
    if hidden_states is None or attn_parent is None:
        return attn_cache
    cfg = getattr(attn_parent, "config", None)
    alpha = getattr(cfg, "alpha", None) if cfg is not None else None
    if alpha is None:
        alpha = getattr(attn_parent, "alpha", None)
    if alpha is None or float(alpha) <= 0.0:
        return attn_cache
    alpha = float(alpha)
    temperature = float(getattr(cfg, "temperature", 1.0) or 1.0) if cfg is not None else 1.0
    L = attn_cache.size(-1)
    span = vision_span_from_parent(attn_parent, L)
    if span is None:
        return attn_cache
    v_s, v_e = span
    visual_hidden_states = hidden_states[:, v_s:v_e, :]
    importance_scores = torch.norm(visual_hidden_states, p=2, dim=-1)
    normalized_scores = torch.zeros_like(importance_scores)
    for batch_idx in range(importance_scores.shape[0]):
        batch_scores = importance_scores[batch_idx]
        standardized_scores = (batch_scores - batch_scores.mean()) / (batch_scores.std() + 1e-8)
        normalized_scores[batch_idx] = torch.softmax(standardized_scores / temperature, dim=0)
    hidden_state_scores = normalized_scores.unsqueeze(1) * alpha
    attn_cache[:, :, v_s:v_e] += hidden_state_scores.to(device=attn_cache.device, dtype=attn_cache.dtype)
    return attn_cache


class STARKVGroupCluster(_RecentWindowKVPool):
    """soft_global grouped KV compression (shared indices across heads)."""

    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
    ):
        super().__init__(
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
        )
        self._aeb_temperature_ema = None
        self._aeb_last_debug = None

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        num_key_value_groups,
        hidden_states: Optional[torch.Tensor] = None,
    ):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        n_kv = max(1, num_heads // max(1, num_key_value_groups))

        attn_parent = getattr(self, "attn_parent", None)
        if attn_parent is not None and getattr(attn_parent, "kv_group_temporal_enable", False):
            setattr(
                attn_parent,
                "kv_group_temporal_step",
                int(getattr(attn_parent, "kv_group_temporal_step", 0) or 0) + 1,
            )

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        attn_weights = torch.matmul(
            query_states[..., -self.window_size :, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size :, -self.window_size :] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size :, : -self.window_size].sum(dim=-2)

        if self.pooling == "avgpool":
            attn_cache = F.avg_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError("Pooling method not supported")

        attn_cache = apply_starkv_spatial_to_attn_cache(attn_cache, hidden_states, attn_parent)

        past_seq_len = key_states.size(2) - self.window_size
        base_capacity = self.max_capacity_prompt

        update_aeb_from_attn_cache(self, attn_cache)
        k_keep = int(max(1, min(base_capacity - self.window_size, past_seq_len, attn_cache.size(-1))))

        key_out_list = []
        val_out_list = []
        for b in range(bsz):
            ac = attn_cache[b]
            S, w_g, lam_eff = apply_mi_prior(
                ac, attn_parent, past_seq_len, n_kv, num_key_value_groups
            )

            base = ac.mean(dim=0)
            g_star = S.argmax(dim=0)
            w_star = w_g[g_star] * float(n_kv)
            scaled = base * ((1.0 - lam_eff) + lam_eff * w_star)

            scaled = temporal_apply_to_scaled(scaled, S, attn_parent, past_seq_len, n_kv)
            scaled = sharpen_scores_with_aeb(scaled, self)

            kk = min(k_keep, scaled.numel(), past_seq_len)
            _, idx_top = torch.topk(scaled, kk, dim=-1)
            idx_sorted = idx_top.sort().values

            idx_b = idx_sorted.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1).unsqueeze(-1).expand(
                -1, -1, -1, head_dim
            )
            ks = key_states[b : b + 1]
            vs = value_states[b : b + 1]
            k_past_compress = ks[:, :, : -self.window_size, :].gather(dim=2, index=idx_b)
            v_past_compress = vs[:, :, : -self.window_size, :].gather(dim=2, index=idx_b)
            k_cur = ks[:, :, -self.window_size :, :]
            v_cur = vs[:, :, -self.window_size :, :]
            key_out_list.append(torch.cat([k_past_compress, k_cur], dim=2))
            val_out_list.append(torch.cat([v_past_compress, v_cur], dim=2))

        return torch.cat(key_out_list, dim=0), torch.cat(val_out_list, dim=0)


__all__ = ["STARKVGroupCluster", "apply_starkv_spatial_to_attn_cache"]
