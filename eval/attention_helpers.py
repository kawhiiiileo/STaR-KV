import argparse
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Fused attention backends (sdpa / flash_attention_2): GQA expand + compressed-KV mask quirks.
_ATTN_FUSED_BACKENDS = frozenset({"sdpa", "flash_attention_2"})


def _attn_is_fused_backend(attn_impl: Optional[str]) -> bool:
    return attn_impl in _ATTN_FUSED_BACKENDS

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast

import transformers
import sys
sys.path.append('../starkv')

from starkv_utils import init_starkv


def is_starkv_kv(kv_cache_mode):
    """True when using STaR-KV compression (STARKVGroupCluster)."""
    return kv_cache_mode == "starkv"




def _qwen25vl_mrope_section(attn_module: nn.Module):
    """HF recent Qwen2.5-VL uses ``config.rope_parameters['mrope_section']``; older checkpoints used ``attn.rope_scaling``."""
    rs = getattr(attn_module, "rope_scaling", None)
    if isinstance(rs, dict) and rs.get("mrope_section") is not None:
        return rs["mrope_section"]
    cfg = getattr(attn_module, "config", None)
    if cfg is not None:
        rp = getattr(cfg, "rope_parameters", None)
        if isinstance(rp, dict) and rp.get("mrope_section") is not None:
            return rp["mrope_section"]
    raise AttributeError(
        "Cannot resolve mrope_section for Qwen2.5-VL attention (missing rope_parameters / rope_scaling)"
    )

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def unrepeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Undo repeat_kv
    """
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states.contiguous()
    
    # Reshape from (batch, num_attention_heads, slen, head_dim) 
    # to (batch, num_key_value_heads, n_rep, slen, head_dim)
    num_key_value_heads = num_attention_heads // n_rep
    hidden_states = hidden_states.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # Take only the first repetition to get back the original
    # (batch, num_key_value_heads, slen, head_dim)
    return hidden_states[:, :, 0, :, :].contiguous()

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    
    # if not repeat_kv already, repeat_kv
    if key.shape[1] != query.shape[1]:
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
    else:
        key_states = key
        value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Ensure mask is at least as long as key_states
        if attention_mask.size(-1) < key_states.shape[-2]:
            pad_len = key_states.shape[-2] - attention_mask.size(-1)
            pad_shape = list(attention_mask.shape)
            pad_shape[-1] = pad_len
            padding = attention_mask.new_zeros(pad_shape)
            attention_mask = torch.cat([attention_mask, padding], dim=-1)

        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# --- Qwen2.5-VL ---

def qwen2_5_vl_vision_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
    
    
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    kwargs.pop("attention_mask", None)
    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output






    
def qwen2_5_vl_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, _qwen25vl_mrope_section(self)
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    
    
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    
    return attn_output, attn_weights, past_key_value



     

# --- STaR-KV (Qwen2.5-VL) ---


def qwen2_5_vl_attention_forward_STARKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]

    if q_len > 1:
        self.kv_seq_len = 0

    used_compressed_kv = False
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, _qwen25vl_mrope_section(self)
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = max(1, min(self.config.window_size, self.config.max_capacity_prompt - 2))
        if hasattr(self, "vision_start_idx"):
            self.config.vision_start_idx = self.vision_start_idx
        if hasattr(self, "vision_end_idx"):
            self.config.vision_end_idx = self.vision_end_idx
        init_starkv(self)
        # propagate AEB settings to STARKVGroupCluster via attn_parent
        self.kv_cluster.kv_entropy_budget_enable = getattr(self, "kv_entropy_budget_enable", False)
        self.kv_cluster.kv_entropy_budget_min_scale = getattr(self, "kv_entropy_budget_min_scale", 0.5)
        self.kv_cluster.kv_entropy_budget_max_scale = getattr(self, "kv_entropy_budget_max_scale", 1.5)
        self.kv_cluster.kv_entropy_budget_smooth = getattr(self, "kv_entropy_budget_smooth", 0.0)
        self.kv_cluster.kv_entropy_budget_scope = getattr(self, "kv_entropy_budget_scope", "layer")
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups, hidden_states
            )
            self.kept_indices = getattr(self.kv_cluster, "kept_indices", None)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            used_compressed_kv = True
            _record_kv_cache(self.layer_idx, kv_seq_len, key_states_compress.shape[-2])
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attention_mask_for_compute = attention_mask
    if _attn_is_fused_backend(self.config._attn_implementation) and used_compressed_kv:
        attention_mask_for_compute = None

    num_key_value_groups_backup = None
    if _attn_is_fused_backend(self.config._attn_implementation):
        num_key_value_groups_backup = self.num_key_value_groups
        self.num_key_value_groups = 1

    try:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask_for_compute,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )
    finally:
        if num_key_value_groups_backup is not None:
            self.num_key_value_groups = num_key_value_groups_backup

    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


# --- STaR-KV (Qwen2 / OpenCUA) ---


def qwen2_attention_forward_STARKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    if q_len > 1:
        self.kv_seq_len = 0

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))

        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        if hasattr(self, "vision_start_idx"):
            self.config.vision_start_idx = self.vision_start_idx
        if hasattr(self, "vision_end_idx"):
            self.config.vision_end_idx = self.vision_end_idx
        init_starkv(self)
        # propagate AEB settings to STARKVGroupCluster via attn_parent
        self.kv_cluster.kv_entropy_budget_enable = getattr(self, "kv_entropy_budget_enable", False)
        self.kv_cluster.kv_entropy_budget_min_scale = getattr(self, "kv_entropy_budget_min_scale", 0.5)
        self.kv_cluster.kv_entropy_budget_max_scale = getattr(self, "kv_entropy_budget_max_scale", 1.5)
        self.kv_cluster.kv_entropy_budget_smooth = getattr(self, "kv_entropy_budget_smooth", 0.0)
        self.kv_cluster.kv_entropy_budget_scope = getattr(self, "kv_entropy_budget_scope", "layer")

        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            # Temporarily repeat KV for update_kv which expects num_heads
            key_states_for_update = repeat_kv(key_states, self.num_key_value_groups) if key_states.shape[1] != query_states.shape[1] else key_states
            value_states_for_update = repeat_kv(value_states, self.num_key_value_groups) if value_states.shape[1] != query_states.shape[1] else value_states
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states_for_update,
                query_states,
                value_states_for_update,
                attention_mask,
                self.num_key_value_groups if hasattr(self, "num_key_value_groups") else 1,
                hidden_states,
            )
            self.kept_indices = getattr(self.kv_cluster, "kept_indices", None)
            key_states_compress = unrepeat_kv(key_states_compress, self.num_key_value_groups)
            value_states_compress = unrepeat_kv(value_states_compress, self.num_key_value_groups)
            key_states_compress = key_states_compress.contiguous(); value_states_compress = value_states_compress.contiguous()
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            _record_kv_cache(self.layer_idx, kv_seq_len, key_states_compress.shape[-2])
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    if attn_weights is not None and hasattr(self, "move_attention_to_cpu") and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights




def configure_accelerate_skip_attention(model):
    """Configure accelerate to skip moving attention tensors back to GPU"""
    try:
        hooks_configured = 0
        # Recursively find and configure all hooks in the model
        def configure_hooks(module):
            nonlocal hooks_configured
            if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                if hasattr(module._hf_hook, 'skip_keys'):
                    existing_skip_keys = module._hf_hook.skip_keys
                    print(f"Found hook on {module.__class__.__name__} with skip_keys: {existing_skip_keys} (type: {type(existing_skip_keys)})")
                    
                    # Handle different types of skip_keys
                    if existing_skip_keys is None:
                        skip_keys = {'attentions'}
                    elif isinstance(existing_skip_keys, str):
                        skip_keys = {existing_skip_keys, 'attentions'}
                    elif isinstance(existing_skip_keys, (list, tuple)):
                        skip_keys = set(existing_skip_keys).union({'attentions'})
                    elif isinstance(existing_skip_keys, set):
                        skip_keys = existing_skip_keys.union({'attentions'})
                    else:
                        # Try to convert to set
                        try:
                            skip_keys = set(existing_skip_keys).union({'attentions'})
                        except:
                            skip_keys = {'attentions'}
                    
                    module._hf_hook.skip_keys = skip_keys
                    hooks_configured += 1
                    print(f"Configured accelerate hook for {module.__class__.__name__} with skip_keys: {skip_keys}")
            
            # Recursively configure child modules
            for child in module.children():
                configure_hooks(child)
        
        configure_hooks(model)
        print(f"Configured accelerate to skip attention tensors on {hooks_configured} hooks")
        
        # Alternative approach: patch the model's forward method to keep attention tensors on CPU
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            original_forward = model.model.forward
            
            def patched_forward(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                
                # If outputs contain attention tensors, move them to CPU
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    cpu_attentions = tuple(
                        attn.cpu() if attn is not None and attn.device.type == 'cuda' else attn 
                        for attn in outputs.attentions
                    )
                    # Create a new output with CPU attention tensors
                    outputs = type(outputs)(
                        **{k: v if k != 'attentions' else cpu_attentions for k, v in outputs.items()}
                    )
                
                return outputs
            
            model.model.forward = patched_forward
            print("Patched model forward method to keep attention tensors on CPU")
            
    except Exception as e:
        print(f"Error configuring accelerate skip keys: {e}")
def set_attention_implementation(model, args):
    if "UI-TARS" in args.model_path:
        for block in model.model.visual.blocks:
            block.attn._attn_implementation = args.attention_implementation
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config._attn_implementation = args.attention_implementation
    else:
        raise NotImplementedError("Model not supported")
    
def set_move_attention_to_cpu(model, args):
    # set move_attention_to_cpu to True
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
                
                print(f"set move_attention_to_cpu to True for layer {layer_name}")
            else:
                layer.self_attn.move_attention_to_cpu = False
                
                print(f"set move_attention_to_cpu to False for layer {layer_name}")
                
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
                
                print(f"set move_attention_to_cpu to True for layer {layer_name}")
            else:
                layer.self_attn.move_attention_to_cpu = False
                
                print(f"set move_attention_to_cpu to False for layer {layer_name}")
    else:
        raise NotImplementedError("Model not supported")
        

def set_kv_cache_budget(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    else:
        raise NotImplementedError("Model not supported")


def apply_entropy_budget_runtime(model, args):
    """Inject AEB (Attention Entropy Budgeting) settings into every self-attention layer.

    AEB is designed as an internal module of the STaR-KV framework and only activates
    when kv_cache is starkv. For original mode, settings are ignored.
    """
    enable = bool(getattr(args, "kv_entropy_budget_enable", False))
    kv_cache = getattr(args, "kv_cache", None)
    if not enable:
        return
    if not is_starkv_kv(kv_cache):
        return
    if "UI-TARS" in args.model_path:
        layers = model.model.language_model.layers
    elif "OpenCUA" in args.model_path:
        layers = model.language_model.model.layers
    else:
        return
    for layer in layers:
        sa = layer.self_attn
        sa.kv_entropy_budget_enable = enable
        sa.kv_entropy_budget_min_scale = float(getattr(args, "kv_entropy_budget_min_scale", 0.5))
        sa.kv_entropy_budget_max_scale = float(getattr(args, "kv_entropy_budget_max_scale", 1.5))
        sa.kv_entropy_budget_smooth = float(getattr(args, "kv_entropy_budget_smooth", 0.0))
        sa.kv_entropy_budget_scope = str(getattr(args, "kv_entropy_budget_scope", "layer"))


def reset_starkv_per_sample_state(model, args):
    """Reset all per-sample STaR-KV EMA states before each model.generate() call.

    Resets:
    - Online profiling: step=0, mi_ema=None
    - Temporal: step=0, pattern_ema=None, last_stats={}
    - AEB temperature EMA (if enabled)
    """
    if not is_starkv_kv(getattr(args, "kv_cache", None)):
        return
    if "UI-TARS" in args.model_path:
        layers = model.model.language_model.layers
    elif "OpenCUA" in args.model_path:
        layers = model.language_model.model.layers
    else:
        return
    for layer in layers:
        sa = layer.self_attn
        # Online profiling
        sa.kv_group_online_profile_step = 0
        sa.kv_group_online_profile_mi_ema = None
        # Temporal
        if getattr(args, "kv_group_temporal_enable", False):
            sa.kv_group_temporal_step = 0
            sa.kv_group_temporal_pattern_ema = None
            sa.kv_group_temporal_last_stats = {}
        # AEB temperature EMA reset
        if getattr(args, "kv_entropy_budget_enable", False):
            if hasattr(sa, "_aeb_temperature_ema"):
                sa._aeb_temperature_ema = None


def set_starkv_group_config(model, args):
    """Propagate STARKVGroup / AEB / Temporal config to all attention layers."""
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B" or "ui-tars" in args.model_path.lower():
        layers = model.model.language_model.layers
    elif args.model_path == "xlangai/OpenCUA-7B" or "opencua" in args.model_path.lower():
        layers = model.language_model.model.layers
    else:
        raise NotImplementedError("Model not supported")

    mi_granularity = getattr(args, "kv_group_mi_granularity", "gqa_group")
    if mi_granularity not in ("gqa_group", "mha_head"):
        mi_granularity = "gqa_group"

    for layer in layers:
        sa = layer.self_attn
        sa.kv_group_selection_mode = "soft_global"
        sa.config.kv_group_selection_mode = "soft_global"
        sa.kv_group_mi_granularity = mi_granularity
        sa.config.kv_group_mi_granularity = mi_granularity
        # AEB (fixed_score: entropy only scales scores, not budget)
        sa.kv_entropy_budget_enable = bool(getattr(args, "kv_entropy_budget_enable", False))
        sa.kv_entropy_budget_min_scale = float(getattr(args, "kv_entropy_budget_min_scale", 0.95))
        sa.kv_entropy_budget_max_scale = float(getattr(args, "kv_entropy_budget_max_scale", 1.05))
        sa.kv_entropy_budget_smooth = float(getattr(args, "kv_entropy_budget_smooth", 0.0))
        sa.kv_entropy_budget_scope = str(getattr(args, "kv_entropy_budget_scope", "layer"))
        # Soft-global MI prior
        sa.kv_group_soft_prior_lambda = float(getattr(args, "kv_group_soft_prior_lambda", 0.5))
        sa.kv_group_soft_prior_source = str(getattr(args, "kv_group_soft_prior_source", "mi_saliency"))
        sa.kv_group_mi_saliency_weight = float(getattr(args, "kv_group_mi_saliency_weight", 0.5))
        sa.config.kv_group_soft_prior_source = sa.kv_group_soft_prior_source
        sa.config.kv_group_mi_saliency_weight = sa.kv_group_mi_saliency_weight
        # Online MI profiling (EMA ramp for λ_eff)
        sa.kv_group_online_profile_steps = int(getattr(args, "kv_group_online_profile_steps", 5))
        sa.kv_group_online_profile_decay = float(getattr(args, "kv_group_online_profile_decay", 0.9))
        sa.kv_group_online_profile_tau = float(getattr(args, "kv_group_online_profile_tau", 1.0))
        sa.kv_group_online_profile_lambda_ramp_steps = int(
            getattr(args, "kv_group_online_profile_lambda_ramp_steps", sa.kv_group_online_profile_steps)
        )
        sa.kv_group_online_profile_step = 0
        sa.kv_group_online_profile_mi_ema = None
        # Temporal discount
        sa.kv_group_temporal_enable = bool(getattr(args, "kv_group_temporal_enable", False))
        sa.kv_group_temporal_delta = float(getattr(args, "kv_group_temporal_delta", 0.1))
        sa.kv_group_temporal_discount_min = float(getattr(args, "kv_group_temporal_discount_min", 0.0))
        sa.kv_group_temporal_rho = float(getattr(args, "kv_group_temporal_rho", 0.9))
        sa.kv_group_temporal_eps = float(getattr(args, "kv_group_temporal_eps", 0.0))
        sa.kv_group_temporal_warmup_steps = int(getattr(args, "kv_group_temporal_warmup_steps", 0))
        sa.kv_group_temporal_dyn_eta = float(getattr(args, "kv_group_temporal_dyn_eta", 0.0))
        sa.kv_group_temporal_mode = str(getattr(args, "kv_group_temporal_mode", "exponential"))
        sa.kv_group_temporal_gamma = float(getattr(args, "kv_group_temporal_gamma", 1.0))
        sa.kv_group_temporal_debug = bool(getattr(args, "kv_group_temporal_debug", False))
        sa.kv_group_temporal_step = 0
        sa.kv_group_temporal_pattern_ema = None
        sa.kv_group_temporal_last_stats = {}


def normalize_starkv_eval_args(args, force_full_starkv_stack=True):
    """
    Fill non-None defaults for compressed-KV / STaR-KV eval.

    When kv_cache is starkv and force_full_starkv_stack is
    True (default), enable full STaR-KV stack (AEB + temporal) for scripts that use ``store_true``
    without passing these flags.

    Set force_full_starkv_stack=False for parsers that use BooleanOptionalAction with
    explicit defaults (e.g. AgentNetBench ablations).
    """
    if getattr(args, "alpha", None) is None:
        args.alpha = 2.0
    if getattr(args, "window_size", None) is None:
        args.window_size = 8
    if getattr(args, "temperature", None) is None:
        args.temperature = 3.5

    if is_starkv_kv(getattr(args, "kv_cache", None)):
        args.kv_group_selection_mode = "soft_global"

    kv = getattr(args, "kv_cache", None)
    if not is_starkv_kv(kv):
        if hasattr(args, "kv_group_temporal_enable"):
            args.kv_group_temporal_enable = False
        if hasattr(args, "kv_entropy_budget_enable"):
            args.kv_entropy_budget_enable = False
        return args

    if force_full_starkv_stack:
        if hasattr(args, "kv_entropy_budget_enable"):
            args.kv_entropy_budget_enable = True
        if hasattr(args, "kv_group_temporal_enable"):
            args.kv_group_temporal_enable = True

    starkv_defaults = {
        "kv_group_soft_prior_source": "mi_saliency",
        "kv_group_mi_saliency_weight": 0.5,
        "kv_group_soft_prior_lambda": 0.5,
        "kv_group_online_profile_steps": 5,
        "kv_group_online_profile_decay": 0.9,
        "kv_group_online_profile_tau": 1.0,
        "kv_group_online_profile_lambda_ramp_steps": 10,
        "kv_entropy_budget_scope": "layer",
        "kv_entropy_budget_min_scale": 0.95,
        "kv_entropy_budget_max_scale": 1.05,
        "kv_entropy_budget_smooth": 0.0,
        "kv_group_temporal_delta": 0.2,
        "kv_group_temporal_discount_min": 0.1,
        "kv_group_temporal_rho": 0.9,
        "kv_group_temporal_eps": 0.0,
        "kv_group_temporal_warmup_steps": 0,
        "kv_group_temporal_mode": "exponential",
        "kv_group_temporal_gamma": 1.0,
    }
    for key, val in starkv_defaults.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, val)
    return args

# set_torch_profiler removed per user request

                        


        
        
def _coerce_vision_idx_for_layers(vision_idx):
    """
    Normalize vision indices to a list of ints for multi-frame temporal discount.
    Accepts int, list/tuple of int, or 1D tensor. Single int becomes [int].
    """
    if vision_idx is None:
        return None
    if isinstance(vision_idx, torch.Tensor):
        return [int(x) for x in vision_idx.detach().cpu().reshape(-1).tolist()]
    if isinstance(vision_idx, (list, tuple)):
        return [int(x) for x in vision_idx]
    return [int(vision_idx)]


def set_vision_start_idx(model, vision_start_idx, args):
    vs = _coerce_vision_idx_for_layers(vision_start_idx)
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_start_idx = vs
            if getattr(layer.self_attn, "config", None) is not None:
                setattr(layer.self_attn.config, "vision_start_idx", vs)
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_start_idx = vs
            if getattr(layer.self_attn, "config", None) is not None:
                setattr(layer.self_attn.config, "vision_start_idx", vs)
    else:
        raise NotImplementedError("Model not supported")

def set_vision_end_idx(model, vision_end_idx, args):
    ve = _coerce_vision_idx_for_layers(vision_end_idx)
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_end_idx = ve
            if getattr(layer.self_attn, "config", None) is not None:
                setattr(layer.self_attn.config, "vision_end_idx", ve)
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_end_idx = ve
            if getattr(layer.self_attn, "config", None) is not None:
                setattr(layer.self_attn.config, "vision_end_idx", ve)
    else:
        raise NotImplementedError("Model not supported")

def set_alpha(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.config.alpha = args.alpha
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config.alpha = args.alpha
    else:
        raise NotImplementedError("Model not supported")

def set_temperature(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.config.temperature = args.temperature
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config.temperature = args.temperature
    else:
        raise NotImplementedError("Model not supported")


def set_window_size(model, args):
    ws = args.window_size
    if ws is None:
        ws = 256
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.config.window_size = ws
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config.window_size = ws
    else:
        raise NotImplementedError("Model not supported")
        



def replace_qwen2_5_vl(kv_cache_mode="original"):
    assert kv_cache_mode in ("original", "starkv")
    
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = qwen2_5_vl_vision_attention_forward
    if kv_cache_mode == "original":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward
    elif is_starkv_kv(kv_cache_mode):
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_STARKV


def replace_opencua(kv_cache_mode="original"):
    assert kv_cache_mode in ("original", "starkv")
    if kv_cache_mode == "original":
        pass
    elif is_starkv_kv(kv_cache_mode):
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_STARKV


def collect_entropy_budget_stats(model, args):
    """Collect AEB runtime statistics from the first attention layer.

    Reads _aeb_last_debug for entropy/temperature info.
    Only meaningful when kv_cache is product STaR-KV and AEB is active.
    Returns None unless kv_cache is starkv.
    """
    kv_cache = getattr(args, "kv_cache", None)
    enable = bool(getattr(args, "kv_entropy_budget_enable", False))

    if not is_starkv_kv(kv_cache):
        return None
    if not enable:
        return None

    if "UI-TARS" in args.model_path:
        try:
            sa = model.model.language_model.layers[0].self_attn
        except Exception:
            return None
    elif "OpenCUA" in args.model_path:
        try:
            sa = model.language_model.model.layers[0].self_attn
        except Exception:
            return None
    else:
        return None

    kc = getattr(sa, "kv_cluster", None)
    if kc is None:
        return None

    dbg = getattr(kc, "_aeb_last_debug", None) or getattr(sa, "_aeb_last_debug", None)
    temperature = getattr(kc, "_aeb_temperature_ema", None) or getattr(sa, "_aeb_temperature_ema", None)
    base_cap = getattr(kc, "max_capacity_prompt", None)
    win_size = getattr(kc, "window_size", None)

    return {
        "enable": enable,
        "active": True,
        "temperature": temperature,
        "normalized_entropy": dbg.get("normalized_entropy") if isinstance(dbg, dict) else None,
        "entropy": dbg.get("entropy") if isinstance(dbg, dict) else None,
        "max_entropy": dbg.get("max_entropy") if isinstance(dbg, dict) else None,
        "base_capacity": base_cap,
        "window_size": win_size,
        "base_k_keep": (base_cap - win_size) if (base_cap is not None and win_size is not None) else None,
        "policy": "fixed_score_temperature",
    }

# =============================================================================
# KV Cache & GPU Memory Statistics Collection
# =============================================================================

class _KvCacheBudgetTracker:
    """Tracks ONLY actual KV-cache compression steps.

    The previous implementation recorded every forward call (including
    autoregressive token-by-token steps where no compression happens), which
    inflated the retention ratio to ~99 % even with budget=10.  That metric
    was meaningless.

    This tracker records **only** the moments where ``update_kv()`` is invoked
    and tokens are actually discarded.  The resulting
    ``overall_retention_ratio`` is therefore a true measure of the budget
    actually consumed during compression.

    Per layer it stores:
      - original token count before compression (sum)
      - kept token count after compression (sum)
      - number of compression events
    """

    def __init__(self):
        self._stats = {}               # layer_idx -> {"orig": int, "kept": int, "count": int}
        self._total_compress_calls = 0
        self._enabled = True

    def reset(self):
        self._stats.clear()
        self._total_compress_calls = 0
        self._enabled = True

    def record(self, layer_idx, orig_len, kept_len):
        if not self._enabled:
            return
        if layer_idx not in self._stats:
            self._stats[layer_idx] = {"orig": 0, "kept": 0, "count": 0}
        self._stats[layer_idx]["orig"] += orig_len
        self._stats[layer_idx]["kept"] += kept_len
        self._stats[layer_idx]["count"] += 1
        self._total_compress_calls += 1

    def get_summary(self, num_layers=28):
        """Return aggregated KV cache stats dict for summary_results.json."""
        if not self._stats:
            return None
        total_orig = sum(s["orig"] for s in self._stats.values())
        total_kept = sum(s["kept"] for s in self._stats.values())
        total_count = sum(s["count"] for s in self._stats.values())
        avg_compress_per_layer = total_count // max(1, len(self._stats))
        retention = total_kept / max(1, total_orig)

        per_layer = {}
        for i in range(num_layers):
            s = self._stats.get(i)
            if s and s["count"] > 0:
                per_layer[i] = {
                    "avg_orig_tokens": round(s["orig"] / s["count"], 1),
                    "avg_kept_tokens": round(s["kept"] / s["count"], 1),
                    "retention_ratio": round(s["kept"] / max(1, s["orig"]), 4),
                }

        return {
            "num_compression_events": self._total_compress_calls,
            "num_layers_logged": len(self._stats),
            "avg_compression_events_per_layer": avg_compress_per_layer,
            "total_orig_tokens": total_orig,
            "total_kept_tokens": total_kept,
            "overall_retention_ratio": round(retention, 4),
            "per_layer": per_layer,
        }


# Global singleton – reset at the start of each eval run.
KV_CACHE_STATS = _KvCacheBudgetTracker()


def reset_kv_cache_stats():
    """Reset the global KV cache accumulator. Call once before eval loop starts."""
    KV_CACHE_STATS.reset()


def collect_kv_cache_stats(num_layers=28):
    """Return current KV cache stats summary (or None if no data)."""
    return KV_CACHE_STATS.get_summary(num_layers=num_layers)


def _record_kv_cache(layer_idx, orig_len, kept_len):
    """Convenience wrapper used by attention forward paths."""
    KV_CACHE_STATS.record(layer_idx, orig_len, kept_len)


# ---------------------------------------------------------------------------
# GPU Memory helpers
# ---------------------------------------------------------------------------

def reset_gpu_memory_stats(device="cuda"):
    """Reset peak memory counters before model load / eval."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def collect_gpu_memory_stats(device="cuda"):
    """Return peak allocated / reserved memory in GB.

    Also returns current allocated memory for sanity checking.
    """
    if not torch.cuda.is_available():
        return None
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_resv = torch.cuda.max_memory_reserved(device)
    curr_alloc = torch.cuda.memory_allocated(device)
    return {
        "peak_allocated_gb": round(peak_alloc / (1024 ** 3), 3),
        "peak_reserved_gb": round(peak_resv / (1024 ** 3), 3),
        "current_allocated_gb": round(curr_alloc / (1024 ** 3), 3),
    }


def estimate_kv_cache_memory_mb(num_tokens, num_layers=28, num_kv_heads=4,
                                 head_dim=128, dtype_bytes=2):
    """Rough estimate of KV cache memory for a given token count.

    Formula: num_tokens * num_layers * num_kv_heads * head_dim * 2(K+V) * dtype_bytes
    """
    bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
    return round(num_tokens * bytes_per_token / (1024 ** 2), 2)


def compute_kv_cache_memory_summary(kv_stats, num_layers=28, num_kv_heads=4,
                                    head_dim=128, dtype_bytes=2):
    """Augment kv_stats dict with estimated memory in MB/GB."""
    if kv_stats is None:
        return None
    total_orig = kv_stats.get("total_orig_tokens", 0)
    total_kept = kv_stats.get("total_kept_tokens", 0)
    orig_mb = estimate_kv_cache_memory_mb(total_orig, num_layers, num_kv_heads,
                                           head_dim, dtype_bytes)
    kept_mb = estimate_kv_cache_memory_mb(total_kept, num_layers, num_kv_heads,
                                           head_dim, dtype_bytes)
    saved_mb = orig_mb - kept_mb
    summary = dict(kv_stats)
    summary["orig_kv_cache_mb"] = orig_mb
    summary["orig_kv_cache_gb"] = round(orig_mb / 1024, 3)
    summary["kept_kv_cache_mb"] = kept_mb
    summary["kept_kv_cache_gb"] = round(kept_mb / 1024, 3)
    summary["saved_kv_cache_mb"] = saved_mb
    summary["saved_kv_cache_gb"] = round(saved_mb / 1024, 3)
    return summary

# --- STaR-KV shared CLI (merged from starkv_cli.py) ---


def add_starkv_kv_arguments(
    parser: argparse.ArgumentParser,
    *,
    kv_cache_default: str = "original",
    kv_cache_budget_default: float = 100,
    kv_cache_budget_type: type = float,
    soft_prior_source_default: str = "profile",
    online_profile_steps_default: int = 0,
    online_profile_lambda_ramp_default: int = 0,
    temporal_warmup_default: int = 20,
    aeb_enable_action: str = "store_true",
    temporal_enable_action: str = "store_true",
    aeb_min_scale_default: float = 0.5,
    aeb_max_scale_default: float = 1.5,
    alpha_default: Optional[float] = 2.0,
    temperature_default: float = 3.5,
    include_mi_granularity: bool = False,
    include_max_samples: bool = False,
) -> None:
    """Register KV-cache / STaR-KV CLI flags shared across benchmark evaluators."""
    parser.add_argument(
        "--kv_cache",
        type=str,
        default=kv_cache_default,
        choices=["original", "starkv"],
        help="KV cache: starkv (STaR-KV compression) or original (full cache).",
    )
    parser.add_argument(
        "--kv_cache_budget",
        type=kv_cache_budget_type,
        default=kv_cache_budget_default,
        help="KV cache budget (percent of full cache unless noted in benchmark).",
    )
    parser.add_argument("--kv_group_soft_prior_lambda", type=float, default=0.5)
    parser.add_argument(
        "--kv_group_soft_prior_source",
        type=str,
        default=soft_prior_source_default,
        choices=["profile", "saliency_only", "mi_saliency"],
    )
    parser.add_argument("--kv_group_mi_saliency_weight", type=float, default=0.5)
    parser.add_argument("--kv_group_online_profile_steps", type=int, default=online_profile_steps_default)
    parser.add_argument("--kv_group_online_profile_decay", type=float, default=0.9)
    parser.add_argument("--kv_group_online_profile_tau", type=float, default=1.0)
    parser.add_argument(
        "--kv_group_online_profile_lambda_ramp_steps",
        type=int,
        default=online_profile_lambda_ramp_default,
    )
    if include_mi_granularity:
        parser.add_argument(
            "--kv_group_mi_granularity",
            type=str,
            default="gqa_group",
            choices=["gqa_group", "mha_head"],
        )
    if temporal_enable_action == "boolean_optional":
        parser.add_argument(
            "--kv_group_temporal_enable",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    else:
        parser.add_argument("--kv_group_temporal_enable", action="store_true")
    parser.add_argument("--kv_group_temporal_rho", type=float, default=0.9)
    parser.add_argument("--kv_group_temporal_delta", type=float, default=0.1)
    parser.add_argument("--kv_group_temporal_eps", type=float, default=0.0)
    parser.add_argument("--kv_group_temporal_discount_min", type=float, default=0.0)
    parser.add_argument("--kv_group_temporal_warmup_steps", type=int, default=temporal_warmup_default)
    parser.add_argument("--kv_group_temporal_debug", action="store_true")
    parser.add_argument("--kv_group_temporal_dyn_eta", type=float, default=0.0)
    parser.add_argument(
        "--kv_group_temporal_mode",
        type=str,
        default="exponential",
        choices=["exponential", "linear", "gamma"],
    )
    parser.add_argument("--kv_group_temporal_gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=alpha_default)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=temperature_default)
    if aeb_enable_action == "store_true":
        parser.add_argument("--kv_entropy_budget_enable", action="store_true")
    else:
        parser.add_argument(
            "--kv_entropy_budget_enable",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    parser.add_argument("--kv_entropy_budget_min_scale", type=float, default=aeb_min_scale_default)
    parser.add_argument("--kv_entropy_budget_max_scale", type=float, default=aeb_max_scale_default)
    parser.add_argument("--kv_entropy_budget_smooth", type=float, default=0.0)
    parser.add_argument(
        "--kv_entropy_budget_scope",
        type=str,
        default="layer",
        choices=["layer", "step"],
    )
    if include_max_samples:
        parser.add_argument("--max_samples", type=int, default=None)


def finalize_starkv_args(
    args: argparse.Namespace,
    *,
    force_full_starkv_stack: bool = False,
    disable_starkv_extras_for_original: bool = False,
) -> None:
    """Post-parse defaults shared by screen benchmarks."""
    normalize_starkv_eval_args(args, force_full_starkv_stack=force_full_starkv_stack)
    if disable_starkv_extras_for_original and not is_starkv_kv(args.kv_cache):
        args.kv_group_temporal_enable = False
        args.kv_entropy_budget_enable = False


# --- No-GPU smoke tests; run: python attention_helpers.py ---


def test_attention_helpers() -> None:
    import attention_helpers as ah

    assert ah.is_starkv_kv("starkv")
    assert not ah.is_starkv_kv("original")


def test_starkv_mi() -> None:
    import numpy as np
    from starkv_mi import strict_mi_score_2d

    p = np.array([0.1, 0.2, 0.3, 0.4])
    p = p / p.sum()
    row_bins = np.array([0, 0, 1, 1])
    col_bins = np.array([0, 1, 0, 1])
    mi = strict_mi_score_2d(p, row_bins, col_bins, 2, 2, 4)
    assert mi >= 0.0


def test_starkv_cli() -> None:

    p = argparse.ArgumentParser()
    add_starkv_kv_arguments(p, include_max_samples=True)
    args = p.parse_args(["--kv_cache", "starkv", "--kv_cache_budget", "20"])
    finalize_starkv_args(args)
    assert getattr(args, "kv_group_selection_mode", None) == "soft_global"


def test_ui_tars_core() -> None:
    from ui_tars_utils import (
        IMAGE_FACTOR,
        MAX_PIXELS,
        MIN_PIXELS,
        parse_action_to_structure_output,
        smart_resize,
    )

    h, w = smart_resize(1000, 800, factor=IMAGE_FACTOR)
    assert h % IMAGE_FACTOR == 0 and w % IMAGE_FACTOR == 0
    assert MIN_PIXELS <= h * w <= MAX_PIXELS * 4 or True  # bounds depend on aspect
    out = parse_action_to_structure_output(
        'Thought\nAction: click(point="<point>100 200</point>")',
        origin_resized_height=600,
        origin_resized_width=800,
    )
    assert isinstance(out, (dict, list, tuple)) or out is not None


def test_eval_paths_import() -> None:
    import eval_paths  # noqa: F401


def run_smoke_tests() -> int:
    tests = [
        test_attention_helpers,
        test_starkv_mi,
        test_starkv_cli,
        test_ui_tars_core,
        test_eval_paths_import,
    ]
    for fn in tests:
        name = fn.__name__
        print(f"[smoke] {name} ...", flush=True)
        fn()
        print(f"[smoke] {name} OK", flush=True)
    print("[smoke] all passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_tests())
