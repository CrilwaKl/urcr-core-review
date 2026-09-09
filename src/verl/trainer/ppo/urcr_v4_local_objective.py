"""Signed span-local PPO and trajectory-mean reduction for URCR-V4."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SignedSpanPPOTerms:
    per_span_loss: torch.Tensor
    per_row_numerator: torch.Tensor
    active_token_count: torch.Tensor


def effective_v4_lambda(
    *,
    lambda_max: float,
    global_step: int,
    warmup_steps: int = 30,
) -> float:
    lambda_max = float(lambda_max)
    if not math.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be finite and nonnegative")
    if int(global_step) < 1:
        raise ValueError("global_step must be positive")
    if int(warmup_steps) < 0:
        raise ValueError("warmup_steps must be nonnegative")
    fraction = (
        1.0
        if int(warmup_steps) == 0
        else min(int(global_step) / int(warmup_steps), 1.0)
    )
    return lambda_max * fraction


def signed_span_ppo(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    span_masks: torch.Tensor,
    coefficients: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> SignedSpanPPOTerms:
    """Evaluate independently clipped signed PPO losses for arbitrary spans.

    ``span_masks`` has shape ``[row, span, token]`` and ``coefficients`` has
    shape ``[row, span]``.  Empty spans produce an exact graph-connected zero.
    """
    if old_log_prob.shape != log_prob.shape or old_log_prob.ndim != 2:
        raise ValueError("old/new log-probs must share shape [row, token]")
    if span_masks.ndim == 2:
        span_masks = span_masks.unsqueeze(1)
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(1)
    expected_masks = (log_prob.shape[0], coefficients.shape[1], log_prob.shape[1])
    if span_masks.shape != expected_masks:
        raise ValueError(
            f"span_masks must have shape {expected_masks}, got {tuple(span_masks.shape)}"
        )
    if coefficients.shape[0] != log_prob.shape[0]:
        raise ValueError("coefficient row count must match log-probs")
    if clip_ratio_low < 0 or clip_ratio_high < 0:
        raise ValueError("clip ratios must be nonnegative")

    ratio = torch.exp(log_prob - old_log_prob)
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - float(clip_ratio_low),
        1.0 + float(clip_ratio_high),
    )
    coefficient = coefficients.to(dtype=ratio.dtype, device=ratio.device).unsqueeze(-1)
    mask = span_masks.to(dtype=ratio.dtype, device=ratio.device)
    unclipped = ratio.unsqueeze(1) * coefficient
    clipped = clipped_ratio.unsqueeze(1) * coefficient
    token_surrogate = torch.minimum(unclipped, clipped)
    counts = mask.sum(dim=-1)
    graph_zero = 0.0 * log_prob.sum()
    per_span = torch.where(
        counts > 0,
        -(token_surrogate * mask).sum(dim=-1) / counts.clamp_min(1.0),
        graph_zero,
    )
    return SignedSpanPPOTerms(
        per_span_loss=per_span,
        per_row_numerator=per_span.sum(dim=-1),
        active_token_count=counts,
    )


def local_minibatch_estimator(
    row_numerators: torch.Tensor,
    *,
    total_sampling_row_count: int,
    original_trajectory_count: int,
    global_minibatch_row_count: int,
    dp_world_size: int = 1,
) -> torch.Tensor:
    """Preserve the original trajectory-mean objective under row minibatches.

    The returned rank-local scalar includes ``dp_world_size`` because FSDP/DDP
    averages rank gradients. ``global_minibatch_row_count`` includes zero-local
    and all-rank non-padding sampling rows.
    """
    if row_numerators.ndim != 1:
        raise ValueError("row_numerators must be one-dimensional")
    if total_sampling_row_count <= 0 or original_trajectory_count <= 0:
        raise ValueError("population counts must be positive")
    if global_minibatch_row_count <= 0 or dp_world_size <= 0:
        raise ValueError("minibatch/DP counts must be positive")
    if global_minibatch_row_count > total_sampling_row_count:
        raise ValueError("minibatch rows cannot exceed total sampling rows")
    scale = (
        int(dp_world_size)
        * int(total_sampling_row_count)
        / (int(original_trajectory_count) * int(global_minibatch_row_count))
    )
    return row_numerators.sum() * scale


def frozen_trajectory_mean_objective(
    row_numerators: torch.Tensor,
    *,
    original_trajectory_count: int,
) -> torch.Tensor:
    if row_numerators.ndim != 1:
        raise ValueError("row_numerators must be one-dimensional")
    if original_trajectory_count <= 0:
        raise ValueError("original_trajectory_count must be positive")
    return row_numerators.sum() / int(original_trajectory_count)
