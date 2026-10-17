"""Specialist Fair Retention (SFR) for shared-index KV compression.

SFR keeps one shared historical token set while protecting query-head
specialists.  It maximizes the paper objective with exact greedy updates over
the union of global Top-4K tokens and per-head Top-R tokens::

    F(S) = (1 - lambda) * sum(u_i)
         + lambda * sum_h w_h * log((c_h + sum(a_hi)) / c_h)

``u`` is the MI/TEM-adjusted saliency, ``a`` is per-head attention mass, and
``c`` is attention mass already protected by the recent window.  TEM's token
discount is also applied to ``a`` so stale frames cannot be reintroduced by
the fairness term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


DEFAULT_LAMBDA = 0.3
DEFAULT_EPS = 1e-2
DEFAULT_SHORTLIST_MULTIPLIER = 4


@dataclass(frozen=True)
class SfrResult:
    """Indices selected by the canonical exact SFR objective."""

    indices: torch.Tensor
    lambda_used: float
    shortlist_size: int


def _as_vector(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().reshape(-1).to(device=device, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _as_matrix(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().to(device=device, dtype=torch.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape [heads, tokens], got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _build_shortlist(
    saliency: torch.Tensor,
    head_mass: torch.Tensor,
    k: int,
    multiplier: int,
) -> torch.Tensor:
    """Return sorted global Top-mK union per-head Top-ceil(K/H)."""
    n_tokens = int(saliency.numel())
    n_heads = int(head_mass.shape[0])
    keep = torch.zeros(n_tokens, dtype=torch.bool, device=saliency.device)

    n_global = min(n_tokens, max(k, int(multiplier) * k))
    if n_global:
        global_idx = torch.topk(saliency, n_global, largest=True, sorted=False).indices
        keep[global_idx] = True

    per_head = min(n_tokens, max(1, (k + n_heads - 1) // n_heads))
    if per_head:
        head_idx = torch.topk(
            head_mass,
            per_head,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        keep[head_idx.reshape(-1)] = True

    return keep.nonzero(as_tuple=False).reshape(-1)


def _exact_greedy(
    saliency: torch.Tensor,
    head_mass: torch.Tensor,
    protected_mass: torch.Tensor,
    head_weight: torch.Tensor,
    candidates: torch.Tensor,
    k: int,
    lam: float,
    eps: float,
) -> torch.Tensor:
    """Run exact per-token greedy SFR on a fixed candidate set."""
    candidate_mass = head_mass.index_select(1, candidates).contiguous()
    candidate_saliency = saliency.index_select(0, candidates).contiguous()
    selected = torch.zeros(
        int(candidates.numel()),
        dtype=torch.bool,
        device=saliency.device,
    )
    covered = protected_mass + eps
    importance = (1.0 - lam) * candidate_saliency

    for _ in range(k):
        fair_gain = torch.log1p(candidate_mass / covered[:, None])
        gain = importance + lam * (head_weight[:, None] * fair_gain).sum(dim=0)
        gain.masked_fill_(selected, float("-inf"))
        winner = torch.argmax(gain)
        selected[winner] = True
        covered = covered + candidate_mass[:, winner]

    return candidates[selected].sort().values


def apply_sfr(
    base_score: torch.Tensor,
    head_mass: torch.Tensor,
    k: int,
    *,
    protected_mass: Optional[torch.Tensor] = None,
    head_weight: Optional[torch.Tensor] = None,
    lam: float = DEFAULT_LAMBDA,
    eps: float = DEFAULT_EPS,
    shortlist_multiplier: int = DEFAULT_SHORTLIST_MULTIPLIER,
) -> SfrResult:
    """Select exactly ``k`` historical tokens with canonical exact SFR."""
    if not isinstance(base_score, torch.Tensor):
        raise TypeError("base_score must be a torch.Tensor")
    device = base_score.device
    saliency = _as_vector(base_score, "base_score", device).clamp_min(0)
    mass = _as_matrix(head_mass, "head_mass", device).clamp_min(0)
    n_heads, n_tokens = mass.shape
    if n_heads == 0:
        raise ValueError("head_mass must contain at least one head")
    if n_tokens != int(saliency.numel()):
        raise ValueError(
            f"token mismatch: base_score has {saliency.numel()}, head_mass has {n_tokens}"
        )
    if not (0.0 <= float(lam) <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if not (float(eps) > 0.0):
        raise ValueError(f"eps must be positive, got {eps}")

    k = min(max(int(k), 0), n_tokens)
    if k == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return SfrResult(empty, float(lam), 0)

    if protected_mass is None:
        protected = torch.zeros(n_heads, device=device, dtype=torch.float32)
    else:
        protected = _as_vector(protected_mass, "protected_mass", device).clamp_min(0)
        if int(protected.numel()) != n_heads:
            raise ValueError(
                f"head mismatch: protected_mass has {protected.numel()}, expected {n_heads}"
            )

    if head_weight is None:
        weight = torch.full(
            (n_heads,),
            1.0 / float(n_heads),
            device=device,
            dtype=torch.float32,
        )
    else:
        weight = _as_vector(head_weight, "head_weight", device).clamp_min(0)
        if int(weight.numel()) != n_heads:
            raise ValueError(
                f"head mismatch: head_weight has {weight.numel()}, expected {n_heads}"
            )
        weight_sum = weight.sum()
        weight = (
            torch.full_like(weight, 1.0 / float(n_heads))
            if float(weight_sum.item()) <= 0.0
            else weight / weight_sum
        )

    saliency = saliency / saliency.sum().clamp_min(1e-8)
    if float(lam) == 0.0:
        indices = torch.argsort(saliency, descending=True, stable=True)[:k].sort().values
        return SfrResult(indices, 0.0, n_tokens)

    candidates = _build_shortlist(
        saliency,
        mass,
        k,
        max(1, int(shortlist_multiplier)),
    )
    if int(candidates.numel()) < k:
        raise AssertionError(
            f"SFR shortlist has {candidates.numel()} candidates for k={k}"
        )
    indices = _exact_greedy(
        saliency,
        mass,
        protected,
        weight,
        candidates,
        k,
        float(lam),
        float(eps),
    )
    assert_selection_invariants(indices, k=k, n_tokens=n_tokens)
    return SfrResult(indices, float(lam), int(candidates.numel()))


def assert_selection_invariants(
    indices: torch.Tensor,
    *,
    k: int,
    n_tokens: int,
) -> None:
    """Fail closed if SFR does not produce a valid chronological Exact-K set."""
    selected = indices.detach().reshape(-1).to(dtype=torch.long)
    if int(selected.numel()) != int(k):
        raise AssertionError(f"selected {selected.numel()} tokens, expected {k}")
    if selected.numel() == 0:
        return
    if bool((selected < 0).any().item()) or bool((selected >= int(n_tokens)).any().item()):
        raise AssertionError(f"selected token outside [0, {n_tokens})")
    if int(torch.unique(selected).numel()) != int(k):
        raise AssertionError("selected token indices are not unique")
    if not bool(torch.all(selected[1:] > selected[:-1]).item()):
        raise AssertionError("selected token indices are not strictly increasing")


def store_tem_discount(attn_parent, discount: Optional[torch.Tensor]) -> None:
    """Store TEM's latest token discount for the immediately following SFR call."""
    if attn_parent is None:
        return
    value = None if discount is None else discount.detach()
    setattr(attn_parent, "kv_group_temporal_last_discount", value)


def apply_tem_discount_to_sfr(
    head_mass: torch.Tensor,
    attn_parent,
) -> torch.Tensor:
    """Apply TEM discount to SFR coverage and suppress floor-stale tokens."""
    mass = head_mass.to(dtype=torch.float32)
    if mass.ndim != 2 or attn_parent is None:
        return mass
    discount = getattr(attn_parent, "kv_group_temporal_last_discount", None)
    if not isinstance(discount, torch.Tensor) or discount.numel() != mass.shape[-1]:
        return mass

    discount = discount.detach().to(device=mass.device, dtype=mass.dtype).clamp(0.0, 1.0)
    if bool(torch.all(discount >= 1.0 - 1e-6).item()):
        return mass

    grafted = mass * discount.unsqueeze(0)
    floor_raw = getattr(attn_parent, "kv_group_temporal_discount_min", 0.1)
    floor = 0.1 if floor_raw is None else float(floor_raw)
    floor = min(max(floor, 0.0), 1.0)
    stale = discount <= floor + 1e-5
    if bool(stale.any().item()):
        grafted = grafted.clone()
        grafted[:, stale] = 0.0
    return grafted


__all__ = [
    "SfrResult",
    "apply_sfr",
    "apply_tem_discount_to_sfr",
    "assert_selection_invariants",
    "store_tem_discount",
]
