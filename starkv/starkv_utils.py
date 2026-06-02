"""KV pooling helpers and init_starkv for STARKVGroupCluster."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads: (batch, num_kv_heads, seqlen, dim) -> (batch, num_attn_heads, seqlen, dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _RecentWindowKVPool:
    """Internal: recent-window attention pooling + top-k gather (base for STARKVGroupCluster)."""

    def __init__(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        _bsz, _num_heads, q_len, head_dim = query_states.shape

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
        attention_mask = mask[None, None, :, :].to(attn_weights.device)
        attn_weights[:, :, -self.window_size :, -self.window_size :] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
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
        indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
        indices = torch.sort(indices, dim=-1).values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        return (
            torch.cat([k_past_compress, k_cur], dim=2),
            torch.cat([v_past_compress, v_cur], dim=2),
        )


def init_starkv(self):
    """Attach STARKVGroupCluster to an attention layer."""
    from starkv_group_cluster import STARKVGroupCluster

    if hasattr(self, "kv_cluster") and isinstance(self.kv_cluster, STARKVGroupCluster):
        self.kv_cluster.window_size = self.config.window_size
        self.kv_cluster.max_capacity_prompt = self.config.max_capacity_prompt
        self.kv_cluster.kernel_size = self.config.kernel_size
        self.kv_cluster.pooling = self.config.pooling
        return

    if not hasattr(self.config, "window_size"):
        self.config.window_size = 32
    if not hasattr(self.config, "max_capacity_prompt"):
        self.config.max_capacity_prompt = 4096
    if not hasattr(self.config, "kernel_size"):
        self.config.kernel_size = 5
    if not hasattr(self.config, "pooling"):
        self.config.pooling = "avgpool"
    self.kv_cluster = STARKVGroupCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
    )
    self.kv_cluster.attn_parent = self

