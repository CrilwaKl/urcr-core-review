"""Independent action-mean PPO objective for URCR v2 fixed support rewards."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class URCRLocalLossTerms:
    query_loss: torch.Tensor
    think_loss: torch.Tensor
    query_numerator: torch.Tensor
    think_numerator: torch.Tensor
    query_eligible_count: torch.Tensor
    think_eligible_count: torch.Tensor
    query_action_value_sum: torch.Tensor
    think_action_value_sum: torch.Tensor


def effective_local_alpha(
    *,
    local_max: float,
    global_step: int,
    warmup_steps: int,
) -> float:
    """Scale the existing linear warm-up by an explicit asymptotic cap."""
    if not 0.0 < float(local_max) <= 1.0:
        raise ValueError("URCR local_max must be in (0, 1]")
    if int(global_step) < 1:
        raise ValueError("URCR global_step must be positive")
    if int(warmup_steps) < 0:
        raise ValueError("URCR warmup_steps must be nonnegative")
    warmup_fraction = (
        1.0
        if int(warmup_steps) == 0
        else min(1.0, int(global_step) / int(warmup_steps))
    )
    return float(local_max) * warmup_fraction


def positive_ppo_multiplier(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> torch.Tensor:
    """Return the exact positive-advantage PPO multiplier used by vanilla PPO."""
    ratio = torch.exp(log_prob - old_log_prob)
    clipped = torch.clamp(
        ratio,
        1.0 - float(clip_ratio_low),
        1.0 + float(clip_ratio_high),
    )
    return torch.minimum(ratio, clipped)


def _span_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("URCR local span mask must match log-prob shape")
    numeric_mask = mask.to(dtype=values.dtype)
    counts = numeric_mask.sum(dim=-1)
    return (values * numeric_mask).sum(dim=-1) / counts.clamp_min(1.0)


def compute_urcr_local_policy_losses(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    query_mask: torch.Tensor,
    think_mask: torch.Tensor,
    query_eligible: torch.Tensor,
    think_eligible: torch.Tensor,
    query_reward: torch.Tensor,
    think_reward: torch.Tensor,
    selected_think_length: torch.Tensor,
    think_length_ref: float,
    global_query_eligible_count: torch.Tensor | float | int,
    global_think_eligible_count: torch.Tensor | float | int,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> URCRLocalLossTerms:
    """Compute local numerators and global-action-mean losses for one shard."""
    if old_log_prob.shape != log_prob.shape:
        raise ValueError("URCR local old/new log-prob shapes must match")
    batch_size = log_prob.shape[0]
    one_dimensional = {
        "query_eligible": query_eligible,
        "think_eligible": think_eligible,
        "query_reward": query_reward,
        "think_reward": think_reward,
        "selected_think_length": selected_think_length,
    }
    for name, value in one_dimensional.items():
        if value.shape != (batch_size,):
            raise ValueError(f"URCR local {name} must have shape ({batch_size},)")
    if not float(think_length_ref) > 0:
        raise ValueError("URCR think_length_ref must be positive")

    multiplier = positive_ppo_multiplier(
        old_log_prob,
        log_prob,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
    )
    dtype = multiplier.dtype
    device = multiplier.device
    query_eligible_f = query_eligible.to(dtype=dtype)
    think_eligible_f = think_eligible.to(dtype=dtype)
    query_values = _span_mean(multiplier, query_mask)
    think_values = _span_mean(multiplier, think_mask)
    n1_scale = torch.sqrt(
        selected_think_length.to(dtype=dtype).clamp_min(0.0)
        / float(think_length_ref)
    )
    query_action_values = (
        query_reward.to(dtype=dtype) * query_values * query_eligible_f
    )
    think_action_values = (
        think_reward.to(dtype=dtype) * n1_scale * think_values * think_eligible_f
    )
    query_numerator = query_action_values.sum()
    think_numerator = think_action_values.sum()
    global_query_count = torch.as_tensor(
        global_query_eligible_count, dtype=dtype, device=device
    )
    global_think_count = torch.as_tensor(
        global_think_eligible_count, dtype=dtype, device=device
    )
    graph_zero = 0.0 * log_prob.sum()
    query_loss = torch.where(
        global_query_count > 0,
        -query_numerator / global_query_count.clamp_min(1.0),
        graph_zero,
    )
    think_loss = torch.where(
        global_think_count > 0,
        -think_numerator / global_think_count.clamp_min(1.0),
        graph_zero,
    )
    return URCRLocalLossTerms(
        query_loss=query_loss,
        think_loss=think_loss,
        query_numerator=query_numerator,
        think_numerator=think_numerator,
        query_eligible_count=query_eligible_f.sum(),
        think_eligible_count=think_eligible_f.sum(),
        query_action_value_sum=query_action_values.detach().sum(),
        think_action_value_sum=think_action_values.detach().sum(),
    )
