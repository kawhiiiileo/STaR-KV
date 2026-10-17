# -*- coding: utf-8 -*-
"""Canonical STaR-KV shared-index KV compressor.

The only supported path is recent-window attention pooling, spatial saliency,
GQA-group MI, exponential TEM, exact SFR, and a protected recent window.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starkv.starkv_utils import _RecentWindowKVPool
from starkv.starkv_temporal import temporal_apply_to_scaled
from starkv.starkv_mi import apply_mi_prior, vision_span_from_parent
from starkv.starkv_sfr import (
    apply_sfr,
    apply_tem_discount_to_sfr,
    assert_selection_invariants,
)


def apply_starkv_spatial_to_attn_cache(
    attn_cache: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    attn_parent,
) -> torch.Tensor:
    """Boost visual tokens with alpha * softmax(norm(hidden) / tau_res)."""
    if hidden_states is None or attn_parent is None:
        return attn_cache
    cfg = getattr(attn_parent, "config", None)
    alpha = getattr(cfg, "alpha", None) if cfg is not None else None
    if alpha is None:
        alpha = getattr(attn_parent, "alpha", None)
    if alpha is None or float(alpha) <= 0.0:
        return attn_cache
    alpha = float(alpha)
    tau_res = float(getattr(cfg, "temperature", 1.0) or 1.0) if cfg is not None else 1.0
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
        # Preserve the exact normalization used by the main-table run.
        if batch_scores.numel() <= 1:
            standardized_scores = torch.zeros_like(batch_scores)
        else:
            standardized_scores = (batch_scores - batch_scores.mean()) / (
                batch_scores.std() + 1e-8
            )
        normalized_scores[batch_idx] = torch.softmax(standardized_scores / tau_res, dim=0)
    hidden_state_scores = normalized_scores.unsqueeze(1) * alpha
    attn_cache[:, :, v_s:v_e] += hidden_state_scores.to(device=attn_cache.device, dtype=attn_cache.dtype)
    return attn_cache


def _pool1d(x: torch.Tensor, pooling: str, kernel_size: int) -> torch.Tensor:
    """Pool over last dim; accepts ``[H, T]`` or ``[B, H, T]``."""
    squeeze = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    if pooling == "avgpool":
        y = F.avg_pool1d(x, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    elif pooling == "maxpool":
        y = F.max_pool1d(x, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    else:
        raise ValueError("Pooling method not supported")
    return y.squeeze(0) if squeeze else y


class STARKVGroupCluster(_RecentWindowKVPool):
    """Shared-index Top-K KV compression with MI, TEM, and SFR."""

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

        if q_len <= self.max_capacity_prompt:
            return key_states, value_states

        # A tiny total budget may be entirely occupied by the recent window.
        # Avoid historical scoring/selection when B = 0.
        if self.max_capacity_prompt == self.window_size:
            return (
                key_states[..., -self.window_size :, :],
                value_states[..., -self.window_size :, :],
            )

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
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_window_past = attn_weights[:, :, -self.window_size :, : -self.window_size]
        attn_weights_sum = attn_window_past.sum(dim=-2)

        attn_parent = getattr(self, "attn_parent", None)
        attn_cache = _pool1d(attn_weights_sum, self.pooling, self.kernel_size)
        attn_cache = apply_starkv_spatial_to_attn_cache(attn_cache, hidden_states, attn_parent)

        past_seq_len = key_states.size(2) - self.window_size
        base_capacity = self.max_capacity_prompt
        k_keep = int(min(base_capacity - self.window_size, past_seq_len, attn_cache.size(-1)))
        tem_on = bool(getattr(attn_parent, "kv_group_temporal_enable", False)) if attn_parent else False
        if tem_on:
            attn_parent.kv_group_temporal_step = int(getattr(attn_parent, "kv_group_temporal_step", 0) or 0) + 1

        key_out_list = []
        val_out_list = []
        for b in range(bsz):
            ac = attn_cache[b]
            heads_per_group = int(num_key_value_groups)
            if heads_per_group <= 0 or ac.shape[0] % heads_per_group != 0:
                raise ValueError(f"cannot map {ac.shape[0]} heads to GQA groups of {heads_per_group}")
            n_groups = ac.shape[0] // heads_per_group
            S_mi, w_g, lam_eff = apply_mi_prior(
                ac, attn_parent, past_seq_len, n_groups, heads_per_group
            )
            base = ac.mean(dim=0)
            n_sub = int(S_mi.size(0))
            g_star = S_mi.argmax(dim=0)
            w_star = w_g[g_star] * float(n_sub)
            scaled = base * ((1.0 - lam_eff) + lam_eff * w_star)
            if tem_on:
                scaled = temporal_apply_to_scaled(
                    scaled, S_mi, attn_parent, past_seq_len, n_groups
                )
            kk = min(k_keep, past_seq_len, scaled.numel())
            idx_top = None
            sfr_on = bool(
                getattr(attn_parent, "kv_group_sfr_enable", False)
            ) if attn_parent else False
            if sfr_on and attn_parent is not None and kk > 0:
                # Use the same recent-window attention mass as the base score.
                # TEM is grafted into SFR coverage so stale frames cannot be
                # revived by the specialist fairness term.
                head_mass = attn_weights_sum[b].to(
                    device=scaled.device, dtype=torch.float32
                )
                head_mass = apply_tem_discount_to_sfr(head_mass, attn_parent)
                window = int(self.window_size)
                protected_mass = attn_weights[
                    b, :, -window:, -window:
                ].sum(dim=(-2, -1)).to(
                    device=scaled.device, dtype=torch.float32
                )
                raw_lambda = getattr(attn_parent, "kv_group_sfr_lambda", 0.3)
                sfr_lambda = 0.3 if raw_lambda is None else float(raw_lambda)
                if sfr_lambda > 0.0:
                    sfr_result = apply_sfr(
                        scaled.float(),
                        head_mass,
                        kk,
                        protected_mass=protected_mass,
                        lam=sfr_lambda,
                    )
                    idx_top = sfr_result.indices.to(
                        device=scaled.device, dtype=torch.long
                    )

            if idx_top is None:
                idx_top = torch.topk(scaled, kk, dim=-1).indices

            # Always gather in original token-position order (strictly increasing).
            # Selector order must never alter the causal KV timeline.
            idx_sorted = idx_top.sort().values

            # Fail closed if the canonical selector violates cache invariants.
            if sfr_on and attn_parent is not None and kk > 0:
                assert_selection_invariants(
                    idx_sorted,
                    k=kk,
                    n_tokens=int(past_seq_len),
                )
            idx_b = (
                idx_sorted.unsqueeze(0)
                .unsqueeze(0)
                .expand(1, num_heads, -1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, head_dim)
            )

            ks = key_states[b : b + 1]
            vs = value_states[b : b + 1]
            k_past_compress = ks[:, :, : -self.window_size, :].gather(dim=2, index=idx_b)
            v_past_compress = vs[:, :, : -self.window_size, :].gather(dim=2, index=idx_b)
            # Same indices for K and V; recent window appended after compressed past.
            k_cur = ks[:, :, -self.window_size :, :]
            v_cur = vs[:, :, -self.window_size :, :]
            key_out = torch.cat([k_past_compress, k_cur], dim=2)
            val_out = torch.cat([v_past_compress, v_cur], dim=2)
            if sfr_on and attn_parent is not None and kk > 0:
                expect_len = int(kk) + int(self.window_size)
                if int(key_out.size(2)) != expect_len or int(val_out.size(2)) != expect_len:
                    raise AssertionError(
                        f"KV length after gather {key_out.size(2)}/{val_out.size(2)} "
                        f"!= K+window={expect_len}"
                    )
            key_out_list.append(key_out)
            val_out_list.append(val_out)

        return torch.cat(key_out_list, dim=0), torch.cat(val_out_list, dim=0)


__all__ = [
    "STARKVGroupCluster",
    "apply_starkv_spatial_to_attn_cache",
]
