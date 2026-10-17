import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Fused attention backends (sdpa / flash_attention_2): GQA expand + compressed-KV mask quirks.
_ATTN_FUSED_BACKENDS = frozenset({"sdpa", "flash_attention_2"})
FLASH_ATTENTION_2 = "flash_attention_2"


def add_flash_attention_2_argument(parser: argparse.ArgumentParser) -> None:
    """Register the only attention backend allowed by official evaluations."""
    parser.add_argument(
        "--attention_implementation",
        type=str,
        default=FLASH_ATTENTION_2,
        choices=[FLASH_ATTENTION_2],
        help="Official evaluations require FlashAttention2; fallback backends are forbidden.",
    )


def require_flash_attention_2(args, device) -> None:
    """Fail closed instead of silently mixing eager/SDPA with FA2 results."""
    impl = getattr(args, "attention_implementation", None)
    if impl != FLASH_ATTENTION_2:
        raise SystemExit(
            f"Official evaluation requires {FLASH_ATTENTION_2!r}; got {impl!r}. "
            "eager/SDPA results are not comparable."
        )
    device_type = str(device).split(":", 1)[0]
    if device_type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "flash_attention_2 requires an available CUDA device; "
            f"selected device={device!r}, cuda_available={torch.cuda.is_available()}."
        )
    if importlib.util.find_spec("flash_attn") is None:
        raise SystemExit(
            "flash_attention_2 was requested but the flash_attn package is not installed. "
            "Fallback to eager/SDPA is forbidden."
        )
    print("[attention backend] flash_attention_2 (required, fallback disabled)", flush=True)


def _fit_fa2_mask_to_compressed_kv(
    attention_mask: Optional[torch.Tensor], key_states: torch.Tensor
) -> Optional[torch.Tensor]:
    """Match a 2-D FA2 padding mask to the physical compressed-cache length.

    Hugging Face generation keeps the logical, uncompressed sequence length in
    its padding mask. After KV compression, FA2's unpadding code indexes the
    physical K/V tensor with that mask, so a longer mask causes a CUDA index
    OOB. Compression only removes historical tokens; retaining the rightmost
    ``kv_len`` mask entries preserves the recent/decode suffix and is exact for
    the batch-size-one, unpadded official evaluations.
    """
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            "flash_attention_2 compressed KV requires a 2-D padding mask; "
            f"got shape={tuple(attention_mask.shape)}"
        )
    kv_len = int(key_states.shape[-2])
    mask_len = int(attention_mask.shape[-1])
    if mask_len < kv_len:
        raise ValueError(
            "flash_attention_2 padding mask is shorter than compressed KV: "
            f"mask_len={mask_len}, kv_len={kv_len}"
        )
    if mask_len > kv_len:
        attention_mask = attention_mask[:, -kv_len:].contiguous()
    return attention_mask


def _attn_is_fused_backend(attn_impl: Optional[str]) -> bool:
    return attn_impl in _ATTN_FUSED_BACKENDS

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast

import transformers
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from starkv.starkv_utils import init_starkv
from runtime_metrics import (
    RUNTIME_METRICS,
    CudaTimer,
    collect_runtime_metrics,
    reset_runtime_metrics,
    timed_generate,
)


def call_update_kv(cluster, key_states, query_states, value_states,
                   attention_mask, num_key_value_groups, hidden_states=None):
    """Call the sole compressor, optionally timing it for evaluation."""
    parent = getattr(cluster, "attn_parent", None)
    args = (key_states, query_states, value_states, attention_mask,
            num_key_value_groups, hidden_states)
    if not getattr(parent, "kv_runtime_metrics_enable", False):
        return cluster.update_kv(*args)
    with CudaTimer(use_cuda_events=bool(getattr(parent, "kv_runtime_metrics_cuda_events", False))) as timer:
        result = cluster.update_kv(*args)
    RUNTIME_METRICS.record_compress_timing(timer.wall_ms)
    return result


def is_starkv_kv(kv_cache_mode):
    """True when using STaR-KV compression (STARKVGroupCluster + MI/TEM/SFR)."""
    return kv_cache_mode == "starkv"


def is_compressed_kv(kv_cache_mode):
    """True only for the canonical STaR-KV compressor."""
    return is_starkv_kv(kv_cache_mode)


def needs_vision_spans(kv_cache_mode):
    """True when STaR-KV requires vision token span indices."""
    return is_starkv_kv(kv_cache_mode)




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


def apply_paper_kv_budget(config, kv_seq_len: int, budget_pct: float, window_size: int):
    """Allocate a total prefill KV capacity that includes the recent window.

    C = max(1, floor(sequence_length * budget_pct / 100)), capped at the
    sequence length. The effective window is min(requested_window, C),
    and SFR selects B = C - window historical tokens (possibly zero).
    """
    seq = int(kv_seq_len)
    pct = float(budget_pct)
    win = int(window_size)
    if seq < 1 or not 0.0 < pct <= 100.0 or win < 1:
        raise ValueError("KV budget requires length >= 1, 0 < percent <= 100, window >= 1")
    capacity = min(seq, max(1, int(seq * (pct / 100.0))))
    win = min(win, capacity)
    hist_b = capacity - win
    config.kv_historical_budget = hist_b
    config.window_size = win
    config.max_capacity_prompt = capacity
    return hist_b, win


def unrepeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Collapse expanded GQA replicas back to native ``num_kv_heads``.

    ``repeat_kv`` duplicates each KV head ``n_rep`` times. Prefer compressors
    that already return native ``num_kv_heads`` (STaR-KV shared indices).
    This helper is a safety net when outputs are still expanded: it keeps the first
    replica per group (valid when all replicas share the same keep set).
    """
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states.contiguous()
    if num_attention_heads % n_rep != 0:
        raise ValueError(
            f"Cannot unrepeat_kv: num_attention_heads={num_attention_heads} not divisible by n_rep={n_rep}"
        )
    num_key_value_heads = num_attention_heads // n_rep
    hidden_states = hidden_states.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states[:, :, 0].contiguous()

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
    # Keep native GQA KV-head layout for cache writes (same as HF full-cache).
    # Only expand temporarily when the compressor expects query-head K/V.
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        apply_paper_kv_budget(
            self.config,
            kv_seq_len,
            self.kv_cache_budget,
            getattr(self, "kv_window_size", getattr(self.config, "window_size", 32)),
        )
        if hasattr(self, "vision_start_idx"):
            self.config.vision_start_idx = self.vision_start_idx
        if hasattr(self, "vision_end_idx"):
            self.config.vision_end_idx = self.vision_end_idx
        if hasattr(self, "vision_token_grids"):
            self.config.vision_token_grids = self.vision_token_grids
        if hasattr(self, "vision_spans"):
            self.config.vision_spans = self.vision_spans
        init_starkv(self)
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_for_update = (
                repeat_kv(key_states, self.num_key_value_groups)
                if key_states.shape[1] != query_states.shape[1]
                else key_states
            )
            value_states_for_update = (
                repeat_kv(value_states, self.num_key_value_groups)
                if value_states.shape[1] != query_states.shape[1]
                else value_states
            )
            key_states_compress, value_states_compress = call_update_kv(
                self.kv_cluster,
                key_states_for_update,
                query_states,
                value_states_for_update,
                attention_mask,
                self.num_key_value_groups,
                hidden_states,
            )
            self.kept_indices = getattr(self.kv_cluster, "kept_indices", None)
            # Collapse back to native KV heads before writing cache.
            if key_states_compress.shape[1] != key_states.shape[1]:
                key_states_compress = unrepeat_kv(key_states_compress, self.num_key_value_groups)
                value_states_compress = unrepeat_kv(value_states_compress, self.num_key_value_groups)
            key_states_compress = key_states_compress.contiguous()
            value_states_compress = value_states_compress.contiguous()
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            _record_kv_cache(
                self.layer_idx,
                kv_seq_len,
                key_states_compress.shape[-2],
                key_tensor=key_states_compress,
                value_tensor=value_states_compress,
            )
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
        sliding_window=None,
        **kwargs,
    )

    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


# --- STaR-KV compressed KV (Qwen2 / OpenCUA) ---


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

        apply_paper_kv_budget(
            self.config,
            kv_seq_len,
            self.kv_cache_budget,
            getattr(self, "kv_window_size", getattr(self.config, "window_size", 32)),
        )
        if hasattr(self, "vision_start_idx"):
            self.config.vision_start_idx = self.vision_start_idx
        if hasattr(self, "vision_end_idx"):
            self.config.vision_end_idx = self.vision_end_idx
        if hasattr(self, "vision_token_grids"):
            self.config.vision_token_grids = self.vision_token_grids
        if hasattr(self, "vision_spans"):
            self.config.vision_spans = self.vision_spans
        init_starkv(self)
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            # Temporarily repeat KV for update_kv which expects num_heads
            key_states_for_update = repeat_kv(key_states, self.num_key_value_groups) if key_states.shape[1] != query_states.shape[1] else key_states
            value_states_for_update = repeat_kv(value_states, self.num_key_value_groups) if value_states.shape[1] != query_states.shape[1] else value_states
            key_states_compress, value_states_compress = call_update_kv(
                self.kv_cluster,
                key_states_for_update,
                query_states,
                value_states_for_update,
                attention_mask,
                self.num_key_value_groups if hasattr(self, "num_key_value_groups") else 1,
                hidden_states,
            )
            self.kept_indices = getattr(self.kv_cluster, "kept_indices", None)
            # Collapse only if compressor still returned expanded query-head layout.
            if key_states_compress.shape[1] != key_states.shape[1]:
                key_states_compress = unrepeat_kv(key_states_compress, self.num_key_value_groups)
                value_states_compress = unrepeat_kv(value_states_compress, self.num_key_value_groups)
            key_states_compress = key_states_compress.contiguous()
            value_states_compress = value_states_compress.contiguous()
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            _record_kv_cache(
                self.layer_idx,
                kv_seq_len,
                key_states_compress.shape[-2],
                key_tensor=key_states_compress,
                value_tensor=value_states_compress,
            )
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation == FLASH_ATTENTION_2:
        attention_mask = _fit_fa2_mask_to_compressed_kv(attention_mask, key_states)

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


def set_kv_cache_mode(model, args):
    """Stamp --kv_cache mode onto every attention layer for cluster dispatch."""
    mode = getattr(args, "kv_cache", "original")
    if "UI-TARS" in args.model_path:
        layers = model.model.language_model.layers
    elif "OpenCUA" in args.model_path:
        layers = model.language_model.model.layers
    else:
        raise NotImplementedError("Model not supported")
    for layer in layers:
        layer.self_attn.kv_cache_mode = mode
        layer.self_attn.baseline_gqa_mode = getattr(
            args, "baseline_gqa_mode", "group_mean"
        )
        layer.self_attn.kv_budget_semantics = getattr(
            args, "kv_budget_semantics", "total_including_window"
        )

def reset_starkv_per_sample_state(model, args):
    """Reset canonical MI, TEM, and SFR state before an independent sample."""
    if not is_starkv_kv(getattr(args, "kv_cache", None)):
        return
    layers = iter_language_model_layers(model, getattr(args, "model_path", ""))
    if not layers:
        return
    for layer in layers:
        sa = layer.self_attn
        sa.kv_group_online_profile_step = 0
        sa.kv_group_online_profile_mi_ema = None
        sa.kv_group_temporal_step = 0
        sa.kv_group_temporal_pattern_ema = None
        sa.kv_group_temporal_last_discount = None
        sa.kv_group_temporal_last_trace = None
        sa.kv_group_temporal_last_stats = {}


def set_starkv_group_config(model, args):
    """Propagate the canonical MI/TEM/SFR configuration to attention layers."""
    layers = iter_language_model_layers(model, getattr(args, "model_path", ""))
    if not layers:
        raise NotImplementedError("Model not supported")
    enable_metrics = bool(getattr(args, "runtime_metrics", False))
    tem_on = bool(getattr(args, "kv_group_temporal_enable", False))
    tem_delta_raw = getattr(args, "kv_group_temporal_delta", 0.20)
    tem_delta = 0.20 if tem_delta_raw is None else float(tem_delta_raw)
    sfr_on = bool(getattr(args, "kv_group_sfr_enable", False))
    _sfr_lam = getattr(args, "kv_group_sfr_lambda", 0.3)
    sfr_lam = 0.3 if _sfr_lam is None else float(_sfr_lam)
    single_image_mi_on = bool(getattr(args, "kv_group_single_image_mi_enable", False))
    current_frame_mi_on = bool(getattr(args, "kv_group_current_frame_mi_enable", False))
    tem_decay_mode = str(getattr(args, "kv_group_temporal_decay_mode", "exponential") or "exponential")
    mi_granularity = str(getattr(args, "kv_group_mi_granularity", "group") or "group")
    mi_prior_mode = str(getattr(args, "kv_group_mi_prior_mode", "online") or "online")
    for layer in layers:
        sa = layer.self_attn
        sa.kv_runtime_metrics_enable = enable_metrics
        sa.kv_group_temporal_enable = tem_on
        sa.kv_group_temporal_delta = tem_delta
        sa.kv_group_sfr_enable = sfr_on
        sa.kv_group_sfr_lambda = sfr_lam
        if getattr(args, "kv_cache", None) == "starkv":
            sa.kv_group_soft_prior_lambda = float(args.kv_group_soft_prior_lambda)
            sa.kv_group_online_profile_steps = int(args.kv_group_online_profile_steps)
            sa.kv_group_online_profile_decay = float(args.kv_group_online_profile_decay)
            sa.kv_group_online_profile_tau = float(args.kv_group_online_profile_tau)
            sa.kv_group_online_profile_lambda_ramp_steps = int(args.kv_group_online_profile_lambda_ramp_steps)
            sa.kv_group_single_image_mi_enable = single_image_mi_on
            sa.kv_group_current_frame_mi_enable = current_frame_mi_on
            sa.kv_group_temporal_rho = float(args.kv_group_temporal_rho)
            sa.kv_group_temporal_discount_min = float(args.kv_group_temporal_discount_min)
            sa.kv_group_temporal_eps = float(args.kv_group_temporal_eps)
            sa.kv_group_temporal_warmup_steps = int(args.kv_group_temporal_warmup_steps)
            sa.kv_group_temporal_decay_mode = tem_decay_mode
            sa.kv_group_mi_granularity = mi_granularity
            sa.kv_group_mi_prior_mode = mi_prior_mode
        sa.kv_group_temporal_last_trace = None
        sa.kv_group_temporal_last_stats = {}
        if getattr(sa, "config", None) is not None:
            sa.config.kv_group_temporal_enable = tem_on
            sa.config.kv_group_temporal_delta = tem_delta
            sa.config.kv_group_sfr_enable = sfr_on
            sa.config.kv_group_sfr_lambda = sfr_lam
            sa.config.kv_group_single_image_mi_enable = single_image_mi_on
            sa.config.kv_group_current_frame_mi_enable = current_frame_mi_on
            sa.config.kv_group_temporal_decay_mode = tem_decay_mode
            sa.config.kv_group_mi_granularity = mi_granularity
            sa.config.kv_group_mi_prior_mode = mi_prior_mode


def normalize_starkv_eval_args(args):
    """Fill defaults for Top-K + spatial (+ TEM delta / SFR)."""
    kv = getattr(args, "kv_cache", None)
    if getattr(args, "alpha", None) is None:
        args.alpha = 2.0
    if getattr(args, "window_size", None) is None:
        args.window_size = 8
    if getattr(args, "temperature", None) is None:
        args.temperature = 3.5
    if not hasattr(args, "kv_group_temporal_enable"):
        args.kv_group_temporal_enable = False
    if getattr(args, "kv_group_temporal_delta", None) is None:
        args.kv_group_temporal_delta = 0.20
    if not hasattr(args, "kv_group_sfr_enable"):
        args.kv_group_sfr_enable = False
    if getattr(args, "kv_group_sfr_lambda", None) is None:
        args.kv_group_sfr_lambda = 0.3
    if kv == "starkv":
        if not 0.0 <= float(args.kv_group_soft_prior_lambda) <= 1.0:
            raise ValueError("kv_group_soft_prior_lambda must be in [0,1]")
        if int(args.kv_group_online_profile_steps) < 0 or int(args.kv_group_online_profile_lambda_ramp_steps) <= 0:
            raise ValueError("MI profile/ramp steps must be non-negative/positive")
        if not 0.0 <= float(args.kv_group_online_profile_decay) < 1.0 or float(args.kv_group_online_profile_tau) <= 0.0:
            raise ValueError("MI decay must be in [0,1) and tau > 0")
        if not 0.0 <= float(args.kv_group_temporal_rho) < 1.0:
            raise ValueError("TEM rho must be in [0,1)")
        if float(args.kv_group_temporal_delta) < 0.0 or not 0.0 <= float(args.kv_group_temporal_discount_min) <= 1.0:
            raise ValueError("TEM delta must be >=0 and discount_min in [0,1]")
        if not 0.0 <= float(args.kv_group_temporal_eps) <= 1.0 or int(args.kv_group_temporal_warmup_steps) < 0:
            raise ValueError("TEM eps must be in [0,1] and warmup >= 0")
    if not is_starkv_kv(kv):
        args.kv_group_temporal_enable = False
        args.kv_group_sfr_enable = False
    return args


def _coerce_vision_idx_for_layers(vision_idx):
    if vision_idx is None:
        return None
    if isinstance(vision_idx, torch.Tensor):
        return [int(x) for x in vision_idx.detach().cpu().reshape(-1).tolist()]
    if isinstance(vision_idx, (list, tuple)):
        return [int(x) for x in vision_idx]
    return [int(vision_idx)]


def set_vision_start_idx(model, vision_start_idx, args):
    vs = _coerce_vision_idx_for_layers(vision_start_idx)
    for layer in iter_language_model_layers(model, getattr(args, "model_path", "")):
        layer.self_attn.vision_start_idx = vs
        if getattr(layer.self_attn, "config", None) is not None:
            setattr(layer.self_attn.config, "vision_start_idx", vs)


def set_vision_end_idx(model, vision_end_idx, args):
    ve = _coerce_vision_idx_for_layers(vision_end_idx)
    for layer in iter_language_model_layers(model, getattr(args, "model_path", "")):
        layer.self_attn.vision_end_idx = ve
        if getattr(layer.self_attn, "config", None) is not None:
            setattr(layer.self_attn.config, "vision_end_idx", ve)


def resolve_spatial_merge_size(model, args):
    """Read model.visual.spatial_merge_size (UI-TARS / OpenCUA)."""
    try:
        path = str(getattr(args, "model_path", "") or "")
        if "UI-TARS" in path or "ui-tars" in path.lower():
            vis = getattr(model, "visual", None)
            if vis is None and hasattr(model, "model"):
                vis = getattr(model.model, "visual", None)
            if vis is not None and hasattr(vis, "spatial_merge_size"):
                return int(vis.spatial_merge_size)
        if "OpenCUA" in path or "opencua" in path.lower():
            vis = getattr(model, "visual", None)
            if vis is None:
                lm = getattr(model, "language_model", None)
                vis = getattr(lm, "visual", None) if lm is not None else None
            if vis is not None and hasattr(vis, "spatial_merge_size"):
                return int(vis.spatial_merge_size)
            return 2
    except Exception:
        return None
    return None


def set_vision_spans(model, vision_spans, args):
    """Attach validated VisionSpan list (+ legacy start/end) to every attention layer."""
    layers = iter_language_model_layers(model, getattr(args, "model_path", ""))
    if not layers:
        raise NotImplementedError("Model not supported")
    for layer in layers:
        sa = layer.self_attn
        sa.vision_spans = vision_spans
        if getattr(sa, "config", None) is not None:
            setattr(sa.config, "vision_spans", vision_spans)
        if vision_spans:
            sa.vision_start_idx = [int(s.start) for s in vision_spans]
            sa.vision_end_idx = [int(s.end) for s in vision_spans]
            if getattr(sa, "config", None) is not None:
                setattr(sa.config, "vision_start_idx", sa.vision_start_idx)
                setattr(sa.config, "vision_end_idx", sa.vision_end_idx)


def set_vision_spans_from_grids(
    model,
    vision_start_idx,
    vision_end_idx,
    image_grid_thw,
    args,
    spatial_merge_size=None,
):
    """Build real-grid VisionSpan metadata; never sqrt-guess."""
    from starkv.starkv_utils import build_vision_spans

    ms = spatial_merge_size if spatial_merge_size is not None else resolve_spatial_merge_size(model, args)
    starts = _coerce_vision_idx_for_layers(vision_start_idx) or []
    ends = _coerce_vision_idx_for_layers(vision_end_idx) or []
    spans, err = build_vision_spans(starts, ends, image_grid_thw, ms)
    set_vision_start_idx(model, starts, args)
    set_vision_end_idx(model, ends, args)
    layers = iter_language_model_layers(model, getattr(args, "model_path", ""))
    if err is not None:
        set_vision_spans(model, [], args)
        for layer in layers:
            layer.self_attn.vision_spans_error = err
        return [], err
    set_vision_spans(model, spans, args)
    for layer in layers:
        layer.self_attn.vision_spans_error = None
    return spans, None


def apply_vision_analysis_to_model(model, vision_analysis, args):
    """Stamp vision start/end (+ grids / VisionSpan) from analyze_vision_tokens_*."""
    if not vision_analysis:
        return None
    starts = vision_analysis.get("vision_start_idx")
    ends = vision_analysis.get("vision_end_idx")
    grids = vision_analysis.get("vision_token_grids")
    thw = vision_analysis.get("image_grid_thw")
    merge = vision_analysis.get("spatial_merge_size", vision_analysis.get("merge_size"))
    set_vision_start_idx(model, starts, args)
    set_vision_end_idx(model, ends, args)
    layers = iter_language_model_layers(model, getattr(args, "model_path", ""))
    if grids is not None:
        for layer in layers:
            layer.self_attn.vision_token_grids = grids
            if getattr(layer.self_attn, "config", None) is not None:
                setattr(layer.self_attn.config, "vision_token_grids", grids)
    if thw is not None and starts is not None and ends is not None:
        return set_vision_spans_from_grids(
            model, starts, ends, thw, args, spatial_merge_size=merge
        )
    # Fallback: build VisionSpan from post-merge (H, W) grids when thw omitted.
    if grids is not None and starts is not None and ends is not None:
        from starkv.starkv_utils import VisionSpan

        start_list = _coerce_vision_idx_for_layers(starts) or []
        end_list = _coerce_vision_idx_for_layers(ends) or []
        spans = []
        for i, (s, e) in enumerate(zip(start_list, end_list)):
            if i >= len(grids):
                break
            g = grids[i]
            if isinstance(g, (list, tuple)) and len(g) >= 2:
                gh, gw = int(g[-2]), int(g[-1])
            else:
                continue
            spans.append(VisionSpan(start=int(s), end=int(e), grid_t=1, grid_h=gh, grid_w=gw))
        if spans:
            set_vision_spans(model, spans, args)
            return spans, None
    return None


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
            layer.self_attn.kv_window_size = ws
            layer.self_attn.config.window_size = ws
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.kv_window_size = ws
            layer.self_attn.config.window_size = ws
    else:
        raise NotImplementedError("Model not supported")
        



def replace_qwen2_5_vl(kv_cache_mode="original"):
    assert kv_cache_mode in ("original", "starkv")
    
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = qwen2_5_vl_vision_attention_forward
    if kv_cache_mode == "original":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward
    elif is_compressed_kv(kv_cache_mode):
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_STARKV


def replace_opencua(kv_cache_mode="original"):
    assert kv_cache_mode in ("original", "starkv")
    if kv_cache_mode == "original":
        pass
    elif is_compressed_kv(kv_cache_mode):
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_STARKV


def iter_language_model_layers(model, model_path: str):
    """Yield language-model transformer layers for UI-TARS / OpenCUA layouts."""
    path = str(model_path or "")
    if "UI-TARS" in path:
        return list(model.model.language_model.layers)
    if "OpenCUA" in path or "opencua" in path.lower():
        return list(model.language_model.model.layers)
    # Best-effort fallbacks
    for attr_path in (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
    ):
        cur = model
        ok = True
        for a in attr_path:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok:
            return list(cur)
    return []


def collect_mi_diagnostics_from_model(model, model_path: str) -> Dict[str, Any]:
    """Stub: MI removed."""
    return {"mi_stats_by_layer": {}, "mi_stats_aggregate": {}, "mi_stats": None}



def collect_cvi_diagnostics_from_model(model, model_path: str) -> Dict[str, Any]:
    """Stub: CVI removed."""
    return {"cvi_stats_by_layer": {}, "cvi_stats": None, "cvi_recon_by_layer": {}}



def collect_group_specialization_from_model(model, model_path: str) -> Dict[str, Any]:
    """Stub: per-group specialization removed."""
    return {
        "group_specialization_by_layer": {},
        "group_specialization": None,
        "per_group_by_layer": {},
        "per_group": None,
    }


def collect_temporal_diagnostics_from_model(model, model_path: str) -> Dict[str, Any]:
    """Aggregate last TEM traces across layers (after one generate)."""
    layers = iter_language_model_layers(model, model_path)
    by_layer = {}
    applied = 0
    reasons = {}
    for layer in layers:
        sa = layer.self_attn
        tr = getattr(sa, "kv_group_temporal_last_trace", None) or getattr(
            sa, "kv_group_temporal_last_stats", None
        )
        if not tr:
            continue
        idx = getattr(sa, "layer_idx", None)
        key = str(idx if idx is not None else id(sa))
        tr_applied = bool(tr.get("applied", tr.get("mode") == "exponential"))
        tr_reason = tr.get("reason") or ("applied" if tr_applied else "unknown")
        stability = tr.get("group_stability")
        stability_mean = None
        stability_p90 = None
        if isinstance(stability, torch.Tensor) and stability.numel():
            stability_f = stability.detach().float()
            stability_mean = float(stability_f.mean().item())
            stability_p90 = float(torch.quantile(stability_f, 0.9).item())
        by_layer[key] = {
            "applied": tr_applied,
            "reason": tr_reason,
            "valid_pair_count": tr.get("valid_pair_count"),
            "redundancy_p90": tr.get("redundancy_p90", stability_p90),
            "redundancy_mean": tr.get("redundancy_mean", stability_mean),
        }
        if tr_applied:
            applied += 1
        reasons[tr_reason] = reasons.get(tr_reason, 0) + 1
    return {
        "tem_layers_applied": applied,
        "tem_reason_counts": reasons,
        "tem_by_layer": by_layer,
    }


def collect_original_mi_diagnostics_from_model(model, model_path: str) -> Dict[str, Any]:
    """Verify that the paper MI prior materially reweights at the last prefill."""
    by_layer = {}
    active = 0
    modes = {}
    for layer in iter_language_model_layers(model, model_path):
        sa = layer.self_attn
        lam = float(getattr(sa, "kv_group_online_profile_last_lam_eff", 0.0) or 0.0)
        mode = str(getattr(sa, "kv_group_online_profile_last_mode", "not_run"))
        weights = getattr(sa, "kv_group_online_profile_last_group_weights", None)
        spread = None
        if isinstance(weights, torch.Tensor) and weights.numel():
            wf = weights.detach().float()
            spread = float((wf.max() - wf.min()).item())
        is_active = bool(lam > 0.0 and spread is not None and spread > 1e-8)
        topk_changed = bool(
            getattr(sa, "kv_group_online_profile_last_topk_changed", False)
        )
        topk_jaccard = getattr(
            sa, "kv_group_online_profile_last_topk_jaccard", None
        )
        idx = getattr(sa, "layer_idx", None)
        key = str(idx if idx is not None else id(sa))
        by_layer[key] = {
            "active": is_active,
            "lambda_eff": lam,
            "mode": mode,
            "group_weight_spread": spread,
            "pre_sfr_topk_changed": topk_changed,
            "pre_sfr_topk_jaccard": topk_jaccard,
        }
        active += int(is_active)
        modes[mode] = modes.get(mode, 0) + 1
    return {
        "mi_layers_active": active,
        "mi_mode_counts": modes,
        "mi_layers_pre_sfr_topk_changed": sum(
            int(row["pre_sfr_topk_changed"]) for row in by_layer.values()
        ),
        "mi_by_layer": by_layer,
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
      - actual cache tensor shapes / bytes when provided
    """

    def __init__(self):
        self._stats = {}               # layer_idx -> stats dict
        self._total_compress_calls = 0
        self._enabled = True

    def reset(self):
        self._stats.clear()
        self._total_compress_calls = 0
        self._enabled = True

    def record(self, layer_idx, orig_len, kept_len, key_tensor=None, value_tensor=None):
        if not self._enabled:
            return
        if layer_idx not in self._stats:
            self._stats[layer_idx] = {
                "orig": 0,
                "kept": 0,
                "count": 0,
                "kept_bytes_sum": 0,
                "key_shape": None,
                "value_shape": None,
                "dtype": None,
            }
        s = self._stats[layer_idx]
        s["orig"] += orig_len
        s["kept"] += kept_len
        s["count"] += 1
        self._total_compress_calls += 1
        if key_tensor is not None and value_tensor is not None:
            s["key_shape"] = list(key_tensor.shape)
            s["value_shape"] = list(value_tensor.shape)
            s["dtype"] = str(key_tensor.dtype).replace("torch.", "")
            s["kept_bytes_sum"] += int(key_tensor.numel() * key_tensor.element_size()) + int(
                value_tensor.numel() * value_tensor.element_size()
            )

    def get_summary(self, num_layers=28):
        """Return aggregated KV cache stats dict for summary_results.json."""
        if not self._stats:
            return None
        total_orig = sum(s["orig"] for s in self._stats.values())
        total_kept = sum(s["kept"] for s in self._stats.values())
        total_count = sum(s["count"] for s in self._stats.values())
        avg_compress_per_layer = total_count // max(1, len(self._stats))
        retention = total_kept / max(1, total_orig)
        total_kept_bytes = sum(s.get("kept_bytes_sum", 0) for s in self._stats.values())

        per_layer = {}
        for i in range(num_layers):
            s = self._stats.get(i)
            if s and s["count"] > 0:
                entry = {
                    "avg_orig_tokens": round(s["orig"] / s["count"], 1),
                    "avg_kept_tokens": round(s["kept"] / s["count"], 1),
                    "retention_ratio": round(s["kept"] / max(1, s["orig"]), 4),
                }
                if s.get("key_shape") is not None:
                    entry["cache_key_shape"] = s["key_shape"]
                    entry["cache_value_shape"] = s["value_shape"]
                    entry["cache_dtype"] = s["dtype"]
                    entry["avg_kept_kv_bytes"] = round(s["kept_bytes_sum"] / s["count"], 1)
                per_layer[i] = entry

        # Infer native head layout from a recorded compress tensor when available.
        sample_shape = None
        sample_dtype = None
        for s in self._stats.values():
            if s.get("key_shape") is not None:
                sample_shape = s["key_shape"]
                sample_dtype = s.get("dtype")
                break

        out = {
            "num_compression_events": self._total_compress_calls,
            "num_layers_logged": len(self._stats),
            "avg_compression_events_per_layer": avg_compress_per_layer,
            "total_orig_tokens": total_orig,
            "total_kept_tokens": total_kept,
            "overall_retention_ratio": round(retention, 4),
            "total_kept_kv_bytes": int(total_kept_bytes),
            "total_kept_kv_mb": round(total_kept_bytes / (1024 ** 2), 3),
            "per_layer": per_layer,
        }
        if sample_shape is not None and len(sample_shape) >= 4:
            # (batch, num_kv_heads, seq, head_dim)
            out["reported_cache_layout"] = {
                "batch": sample_shape[0],
                "num_kv_heads": sample_shape[1],
                "seq_len_example": sample_shape[2],
                "head_dim": sample_shape[3],
                "dtype": sample_dtype,
                "note": "Shapes/bytes are measured from tensors written to past_key_value, not theoretical estimates.",
            }
        return out


# Global singleton – reset at the start of each eval run.
KV_CACHE_STATS = _KvCacheBudgetTracker()


def reset_kv_cache_stats():
    """Reset the global KV cache accumulator. Call once before eval loop starts."""
    KV_CACHE_STATS.reset()


def collect_kv_cache_stats(num_layers=28):
    """Return current KV cache stats summary (or None if no data)."""
    return KV_CACHE_STATS.get_summary(num_layers=num_layers)


def _record_kv_cache(layer_idx, orig_len, kept_len, key_tensor=None, value_tensor=None):
    """Convenience wrapper used by attention forward paths."""
    KV_CACHE_STATS.record(layer_idx, orig_len, kept_len, key_tensor=key_tensor, value_tensor=value_tensor)


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
    """Theoretical KV-cache size assuming native GQA ``num_kv_heads`` layout.

    Prefer measured ``total_kept_kv_bytes`` from compression tracking when
    comparing methods; this formula is only a fallback for token-count scaling.
    """
    bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
    return round(num_tokens * bytes_per_token / (1024 ** 2), 2)


def compute_kv_cache_memory_summary(kv_stats, num_layers=28, num_kv_heads=4,
                                    head_dim=128, dtype_bytes=2):
    """Augment kv_stats with measured bytes when available, else theoretical MB."""
    if kv_stats is None:
        return None
    summary = dict(kv_stats)
    total_orig = kv_stats.get("total_orig_tokens", 0)
    total_kept = kv_stats.get("total_kept_tokens", 0)

    # Prefer actual tensor bytes written at compression time.
    measured_kept_bytes = kv_stats.get("total_kept_kv_bytes")
    layout = kv_stats.get("reported_cache_layout") or {}
    measured_heads = layout.get("num_kv_heads")
    measured_dim = layout.get("head_dim")
    if measured_heads:
        num_kv_heads = int(measured_heads)
    if measured_dim:
        head_dim = int(measured_dim)

    orig_mb = estimate_kv_cache_memory_mb(total_orig, num_layers, num_kv_heads,
                                           head_dim, dtype_bytes)
    if measured_kept_bytes is not None and measured_kept_bytes > 0:
        kept_mb = round(measured_kept_bytes / (1024 ** 2), 2)
        summary["kept_memory_source"] = "measured_tensor_nbytes"
    else:
        kept_mb = estimate_kv_cache_memory_mb(total_kept, num_layers, num_kv_heads,
                                               head_dim, dtype_bytes)
        summary["kept_memory_source"] = "token_count_x_native_kv_heads"
    saved_mb = orig_mb - kept_mb
    summary["memory_estimate_assumes_num_kv_heads"] = num_kv_heads
    summary["memory_estimate_assumes_head_dim"] = head_dim
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
    temporal_enable_action: str = "store_true",
    alpha_default: Optional[float] = 2.0,
    temperature_default: float = 3.5,
    include_max_samples: bool = False,
) -> None:
    """Register KV-cache CLI flags: shared Top-K + spatial + optional TEM."""
    parser.add_argument(
        "--kv_cache",
        type=str,
        default=kv_cache_default,
        choices=["original", "starkv"],
        help=(
            "KV cache method: original (full cache) or starkv (MI + TEM + SFR)."
        ),
    )
    parser.add_argument(
        "--kv_cache_budget",
        type=kv_cache_budget_type,
        default=kv_cache_budget_default,
        help=(
            "Total retained prefill KV percent, including the recent window. "
            "SFR selects the remaining capacity after reserving the window."
        ),
    )
    parser.add_argument(
        "--baseline_gqa_mode",
        choices=("group_mean",),
        default="group_mean",
        help="Canonical native-GQA projection; no query-head expansion is supported.",
    )
    parser.add_argument(
        "--kv_budget_semantics",
        choices=("total_including_window",),
        default="total_including_window",
        help="Total prefill KV capacity, including the protected recent window.",
    )
    parser.add_argument(
        "--runtime_metrics",
        action="store_true",
        help="Record wall-clock TTFT/ITL/e2e generate latency into summary JSON.",
    )
    parser.add_argument(
        "--runtime_metrics_sync_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA around generate latency probes (default: true).",
    )
    if temporal_enable_action == "boolean_optional":
        parser.add_argument(
            "--kv_group_temporal_enable",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable STaR-KV temporal stability reweighting.",
        )
    else:
        parser.add_argument(
            "--kv_group_temporal_enable",
            action="store_true",
            help="Enable STaR-KV temporal stability reweighting.",
        )
    parser.add_argument(
        "--kv_group_temporal_delta",
        type=float,
        default=0.20,
        help="Fixed TEM redundancy strength (default 0.20).",
    )
    parser.add_argument("--kv_group_soft_prior_lambda", type=float, default=0.5)
    parser.add_argument("--kv_group_online_profile_steps", type=int, default=5)
    parser.add_argument("--kv_group_online_profile_decay", type=float, default=0.9)
    parser.add_argument("--kv_group_online_profile_tau", type=float, default=1.0)
    parser.add_argument("--kv_group_online_profile_lambda_ramp_steps", type=int, default=10)
    parser.add_argument(
        "--kv_group_single_image_mi_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the current paper MI estimate immediately for independent single-image "
            "samples; multi-frame evaluation keeps the online EMA/ramp."
        ),
    )
    parser.add_argument(
        "--kv_group_current_frame_mi_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the current-frame paper MI estimate immediately for independent "
            "prefills, including prompts that contain history images (AndroidControl)."
        ),
    )
    parser.add_argument("--kv_group_temporal_rho", type=float, default=0.9)
    parser.add_argument("--kv_group_temporal_discount_min", type=float, default=0.1)
    parser.add_argument("--kv_group_temporal_eps", type=float, default=0.0)
    parser.add_argument("--kv_group_temporal_warmup_steps", type=int, default=0)
    parser.add_argument(
        "--kv_group_temporal_decay_mode",
        type=str,
        default="exponential",
        choices=("exponential",),
        help="Canonical paper exponential TEM discount.",
    )
    parser.add_argument(
        "--kv_group_mi_granularity",
        type=str,
        default="group",
        choices=("group",),
        help="Canonical native-GQA group/subspace MI granularity.",
    )
    parser.add_argument(
        "--kv_group_mi_prior_mode",
        type=str,
        default="online",
        choices=("online",),
        help="Canonical online EMA MI prior.",
    )
    parser.add_argument(
        "--kv_group_sfr_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable STaR-KV Specialist Fair Retention.",
    )
    parser.add_argument(
        "--kv_group_sfr_lambda",
        type=float,
        default=0.3,
        help="SFR fairness weight λ in [0,1] (default 0.3).",
    )
    parser.add_argument("--alpha", type=float, default=alpha_default,
                        help="Spatial boost strength (0 disables). Default 2.0 for starkv.")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=temperature_default,
                        help="Spatial softmax temperature (default 3.5 for starkv).")
    if include_max_samples:
        parser.add_argument("--max_samples", type=int, default=None)


def finalize_starkv_args(
    args: argparse.Namespace,
    *,
    disable_starkv_extras_for_original: bool = False,
) -> None:
    """Post-parse defaults for Top-K + spatial + TEM/SFR."""
    normalize_starkv_eval_args(args)
    if disable_starkv_extras_for_original and not is_starkv_kv(
        getattr(args, "kv_cache", None)
    ):
        args.kv_group_temporal_enable = False
        if hasattr(args, "kv_group_sfr_enable"):
            args.kv_group_sfr_enable = False


# --- No-GPU smoke tests; run: python attention_helpers.py ---


def test_attention_helpers() -> None:
    import attention_helpers as ah

    assert ah.is_starkv_kv("starkv")
    assert not ah.is_starkv_kv("original")
    assert ah.needs_vision_spans("starkv")
    assert not ah.is_compressed_kv("original")



if __name__ == "__main__":
    test_attention_helpers()
    print("attention_helpers ok")
