from __future__ import annotations

import numpy as np
import pytest
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def _turn(trajectory_id: str, token: int) -> dict:
    return {
        "traj_uid": trajectory_id,
        "active_masks": True,
        "input_ids": torch.tensor([token]),
        "responses": torch.tensor([token + 1]),
    }


def _gather(*, terminal_binary_em):
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    trajectories = [[_turn("t0", 1)], [_turn("t1", 3)]]
    output = collector.gather_rollout_data(
        total_batch_list=trajectories,
        episode_rewards=np.asarray([0.25, 0.0], dtype=np.float32),
        episode_lengths=np.asarray([1, 1], dtype=np.float32),
        success={"success_rate": np.asarray([1.0, 0.0])},
        traj_uid=np.asarray(["t0", "t1"], dtype=object),
        tool_callings=np.asarray([0, 0], dtype=np.float32),
        terminal_binary_em=terminal_binary_em,
    )
    return trajectories, output


def test_v4_rollout_keeps_raw_binary_terminal_outcome_separate_from_reward() -> None:
    trajectories, output = _gather(terminal_binary_em=np.asarray([1.0, 0.0]))
    assert output.non_tensor_batch["terminal_binary_em"].tolist() == [1, 0]
    assert output.non_tensor_batch["episode_rewards"].tolist() == [0.25, 0.0]
    assert trajectories[0][0]["terminal_binary_em"] == 1


def test_v4_off_does_not_add_terminal_outcome_field() -> None:
    _trajectories, output = _gather(terminal_binary_em=None)
    assert "terminal_binary_em" not in output.non_tensor_batch


def test_v4_rollout_rejects_nonbinary_terminal_outcome() -> None:
    with pytest.raises(ValueError, match="only binary"):
        _gather(terminal_binary_em=np.asarray([0.25, 0.0]))
