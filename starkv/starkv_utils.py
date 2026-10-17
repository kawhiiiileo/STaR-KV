"""Construction helpers for the canonical STaR-KV compressor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

TensorLike = Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]]


@dataclass(frozen=True)
class VisionSpan:
    """One contiguous visual-token span bound to its true LLM grid."""

    start: int
    end: int
    grid_t: int
    grid_h: int
    grid_w: int

    @property
    def length(self) -> int:
        return int(self.end) - int(self.start)

    @property
    def n_tokens(self) -> int:
        return int(self.grid_t) * int(self.grid_h) * int(self.grid_w)

    def validate_length(self) -> Optional[str]:
        if self.start < 0 or self.end <= self.start:
            return "invalid_span_bounds"
        if self.grid_t <= 0 or self.grid_h <= 0 or self.grid_w <= 0:
            return "invalid_grid"
        if self.length != self.n_tokens:
            return "span_grid_length_mismatch"
        return None



def llm_thw_from_patch_thw(
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int,
) -> Tuple[int, int, int]:
    """Convert vit patch grid (T,H,W) to LLM visual-token grid after spatial merge."""
    ms = int(spatial_merge_size)
    if ms <= 0:
        raise ValueError(f"invalid spatial_merge_size={spatial_merge_size}")
    t_i, h_i, w_i = int(t), int(h), int(w)
    if h_i % ms != 0 or w_i % ms != 0:
        raise ValueError(
            f"patch grid H,W=({h_i},{w_i}) not divisible by spatial_merge_size={ms}"
        )
    return t_i, h_i // ms, w_i // ms


def _as_thw_list(image_grid_thw: TensorLike) -> List[Tuple[int, int, int]]:
    if isinstance(image_grid_thw, torch.Tensor):
        t = image_grid_thw.detach().cpu()
        if t.ndim == 1:
            t = t.view(1, 3)
        elif t.ndim == 3:
            t = t.reshape(-1, 3)
        elif t.ndim != 2 or t.shape[-1] != 3:
            raise ValueError(f"image_grid_thw shape {tuple(t.shape)} invalid")
        return [(int(r[0]), int(r[1]), int(r[2])) for r in t.tolist()]
    # OpenCUA's image processor returns numpy.ndarray. Normalize numpy-like
    # containers before the generic sequence path so an [N, 3] array is not
    # mistaken for one flat THW row when N != 3.
    if hasattr(image_grid_thw, "tolist"):
        image_grid_thw = image_grid_thw.tolist()
    # sequence
    rows = list(image_grid_thw)
    if rows and not isinstance(rows[0], (list, tuple)):
        if len(rows) != 3:
            raise ValueError("flat image_grid_thw must have length 3")
        return [(int(rows[0]), int(rows[1]), int(rows[2]))]
    out: List[Tuple[int, int, int]] = []
    for r in rows:
        if len(r) != 3:
            raise ValueError(f"bad thw row {r}")
        out.append((int(r[0]), int(r[1]), int(r[2])))
    return out


def build_vision_spans(
    starts: Sequence[int],
    ends: Sequence[int],
    image_grid_thw: Optional[TensorLike],
    spatial_merge_size: Optional[int],
) -> Tuple[List[VisionSpan], Optional[str]]:
    """
    Bind each vision span to its real LLM grid.

    Returns (spans, error_reason). On any metadata failure returns ([], reason).
    Never guesses a square grid via sqrt.
    """
    if image_grid_thw is None:
        return [], "missing_image_grid_thw"
    if spatial_merge_size is None:
        return [], "missing_spatial_merge_size"
    try:
        ms = int(spatial_merge_size)
    except Exception:
        return [], "invalid_spatial_merge_size"
    if ms <= 0:
        return [], "invalid_spatial_merge_size"

    if starts is None or ends is None:
        return [], "missing_vision_span_indices"
    if len(starts) != len(ends):
        return [], "vision_span_start_end_length_mismatch"
    if len(starts) == 0:
        return [], "empty_vision_spans"

    try:
        thw_list = _as_thw_list(image_grid_thw)
    except Exception as e:
        return [], f"bad_image_grid_thw:{type(e).__name__}"

    if len(thw_list) != len(starts):
        return [], "num_grids_ne_num_spans"

    spans: List[VisionSpan] = []
    prev_end = -1
    for i, (s, e) in enumerate(zip(starts, ends)):
        s_i, e_i = int(s), int(e)
        if e_i <= s_i:
            return [], "invalid_span_bounds"
        if s_i < prev_end:
            return [], "overlapping_or_non_monotonic_spans"
        # Allow text between spans (s_i >= prev_end); contiguous vision blocks
        # themselves must be continuous token ranges (end-start == n_tokens).
        try:
            gt, gh, gw = llm_thw_from_patch_thw(*thw_list[i], spatial_merge_size=ms)
        except ValueError as err:
            return [], f"merge_rule_error:{err}"
        span = VisionSpan(start=s_i, end=e_i, grid_t=gt, grid_h=gh, grid_w=gw)
        reason = span.validate_length()
        if reason is not None:
            return [], reason
        spans.append(span)
        prev_end = e_i
    return spans, None


def vision_spans_from_parent(attn_parent) -> Tuple[List[VisionSpan], Optional[str]]:
    """Read pre-validated VisionSpan list from attention module / config."""
    if attn_parent is None:
        return [], "missing_attn_parent"
    spans = getattr(attn_parent, "vision_spans", None)
    if spans is None:
        cfg = getattr(attn_parent, "config", None)
        if cfg is not None:
            spans = getattr(cfg, "vision_spans", None)
    if not spans:
        return [], "missing_vision_spans"
    out: List[VisionSpan] = []
    for s in spans:
        if isinstance(s, VisionSpan):
            out.append(s)
        elif isinstance(s, dict):
            out.append(VisionSpan(**{k: int(s[k]) for k in ("start", "end", "grid_t", "grid_h", "grid_w")}))
        else:
            return [], "bad_vision_span_type"
    for sp in out:
        reason = sp.validate_length()
        if reason is not None:
            return [], reason
    return out, None



class _RecentWindowKVPool:
    """Store the four cache-selection parameters shared by STaR-KV calls."""

    def __init__(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
    ):
        self.reset(window_size, max_capacity_prompt, kernel_size, pooling)

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
    ):
        window_size = int(window_size)
        max_capacity_prompt = int(max_capacity_prompt)
        kernel_size = int(kernel_size)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if max_capacity_prompt < window_size:
            raise ValueError("max_capacity_prompt must be at least window_size")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if pooling not in {"avgpool", "maxpool"}:
            raise ValueError(f"unsupported pooling mode: {pooling!r}")
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling


def init_starkv(self):
    """Attach STARKVGroupCluster to an attention layer."""
    from starkv.starkv_group_cluster import STARKVGroupCluster

    if hasattr(self, "kv_cluster") and isinstance(self.kv_cluster, STARKVGroupCluster):
        self.kv_cluster.reset(
            self.config.window_size,
            self.config.max_capacity_prompt,
            self.config.kernel_size,
            self.config.pooling,
        )
        self.kv_cluster.attn_parent = self
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
    # Paper-aligned defaults (Appendix A.1).
    self.kv_group_soft_prior_lambda = float(getattr(self, "kv_group_soft_prior_lambda", 0.5))
    self.kv_group_online_profile_steps = int(getattr(self, "kv_group_online_profile_steps", 5))
    self.kv_group_online_profile_decay = float(getattr(self, "kv_group_online_profile_decay", 0.9))
    self.kv_group_online_profile_tau = float(getattr(self, "kv_group_online_profile_tau", 1.0))
    self.kv_group_online_profile_lambda_ramp_steps = int(getattr(self, "kv_group_online_profile_lambda_ramp_steps", 10))
    self.kv_group_online_profile_step = 0
    self.kv_group_online_profile_mi_ema = None
    self.kv_group_temporal_rho = float(getattr(self, "kv_group_temporal_rho", 0.9))
    self.kv_group_temporal_delta = float(getattr(self, "kv_group_temporal_delta", 0.2))
    self.kv_group_temporal_discount_min = float(getattr(self, "kv_group_temporal_discount_min", 0.1))
    self.kv_group_temporal_eps = float(getattr(self, "kv_group_temporal_eps", 0.0))
    self.kv_group_temporal_warmup_steps = int(getattr(self, "kv_group_temporal_warmup_steps", 0))
    self.kv_group_temporal_step = 0
    self.kv_group_temporal_pattern_ema = None
