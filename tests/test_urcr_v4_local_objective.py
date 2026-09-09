from __future__ import annotations

import pytest
import torch

from verl.trainer.ppo.urcr_v4_local_objective import (
    effective_v4_lambda,
    frozen_trajectory_mean_objective,
    local_minibatch_estimator,
    signed_span_ppo,
)


def test_signed_ppo_clips_positive_and_negative_coefficients_correctly() -> None:
    ratios = torch.tensor([1.5, 1.5, 0.5, 0.5])
    terms = signed_span_ppo(
        old_log_prob=torch.zeros((4, 1)),
        log_prob=torch.log(ratios).unsqueeze(1),
        span_masks=torch.ones((4, 1, 1), dtype=torch.bool),
        coefficients=torch.tensor([[2.0], [-2.0], [2.0], [-2.0]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert terms.per_span_loss[:, 0].tolist() == pytest.approx(
        [-2.4, 3.0, -1.0, 1.6]
    )


def test_empty_spans_are_zero_and_tokens_outside_local_masks_have_zero_gradient() -> None:
    log_prob = torch.zeros((1, 4), requires_grad=True)
    masks = torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool)
    terms = signed_span_ppo(
        old_log_prob=torch.zeros_like(log_prob),
        log_prob=log_prob,
        span_masks=masks,
        coefficients=torch.tensor([[1.0, 4.0]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert terms.per_span_loss[0, 1] == 0
    assert torch.isfinite(terms.per_span_loss).all()
    terms.per_row_numerator.sum().backward()
    assert log_prob.grad is not None
    assert log_prob.grad[0, 1] != 0
    assert log_prob.grad[0, [0, 2, 3]].abs().sum() == 0


def test_minibatch_estimator_preserves_frozen_trajectory_mean_for_unequal_batches() -> None:
    rows = torch.arange(1.0, 8.0)
    total_rows = len(rows)
    trajectories = 4
    batches = (rows[:2], rows[2:5], rows[5:])
    losses = [
        local_minibatch_estimator(
            batch,
            total_sampling_row_count=total_rows,
            original_trajectory_count=trajectories,
            global_minibatch_row_count=len(batch),
        )
        for batch in batches
    ]
    reconstructed = sum(
        len(batch) / total_rows * loss for batch, loss in zip(batches, losses)
    )
    reference = frozen_trajectory_mean_objective(
        rows, original_trajectory_count=trajectories
    )
    assert reconstructed == pytest.approx(reference)


def test_dp_gradient_average_handles_unequal_activity_and_all_zero_rank() -> None:
    active_rank = torch.tensor([1.0, 2.0])
    zero_rank = torch.tensor([0.0, 0.0])
    active_loss = local_minibatch_estimator(
        active_rank,
        total_sampling_row_count=4,
        original_trajectory_count=2,
        global_minibatch_row_count=4,
        dp_world_size=2,
    )
    zero_loss = local_minibatch_estimator(
        zero_rank,
        total_sampling_row_count=4,
        original_trajectory_count=2,
        global_minibatch_row_count=4,
        dp_world_size=2,
    )
    averaged_rank_loss = (active_loss + zero_loss) / 2
    global_loss = local_minibatch_estimator(
        torch.cat([active_rank, zero_rank]),
        total_sampling_row_count=4,
        original_trajectory_count=2,
        global_minibatch_row_count=4,
    )
    assert averaged_rank_loss == pytest.approx(global_loss)


def test_v4_lambda_warms_to_full_strength_without_final_decay() -> None:
    assert effective_v4_lambda(lambda_max=0.12, global_step=1) == pytest.approx(0.004)
    assert effective_v4_lambda(lambda_max=0.12, global_step=30) == 0.12
    assert effective_v4_lambda(lambda_max=0.12, global_step=300) == 0.12
    with pytest.raises(ValueError, match="nonnegative"):
        effective_v4_lambda(lambda_max=-0.1, global_step=1)
