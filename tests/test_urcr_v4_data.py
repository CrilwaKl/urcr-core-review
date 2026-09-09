from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.urcr_v4_data import (
    build_observation_pool,
    build_v4_turn_records,
    original_trajectory_count,
    record_is_visible_exact_repeat,
    visible_history_passages,
)
from verl.trainer.ppo.urcr_v4_method import forbidden_online_metadata_paths


class PieceTokenizer:
    pieces = {
        1: "prompt",
        2: "<think>",
        3: "reason",
        4: "</think>",
        5: "<search>",
        6: "query",
        7: "</search>",
        8: "<answer>",
        9: "Paris",
        10: "</answer>",
        11: "No</answer>",
        0: "",
    }

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(value)] for value in ids)

    def batch_decode(self, batches, **_kwargs):
        return [self.decode(values) for values in batches]

    def encode(self, text, **_kwargs):
        return list(range(len(str(text).split())))


def _batch(*, adjustment_copy: bool = False):
    # Two trajectories from one rollout group, with a search then answer row.
    responses = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 8, 9, 10],
            [2, 3, 4, 8, 9, 10],
        ]
    )
    response_mask = torch.ones_like(responses)
    prompts = torch.tensor([[0, 1], [0, 1], [0, 1]])
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    reward = {"ground_truth": {"target": ["Paris"]}, "style": "rule"}
    extra = {"index": 7, "split": "train", "question": "Where?"}
    env = {"ground_truth": {"target": ["Paris"]}, "question": "Where?"}
    return SimpleNamespace(
        batch={
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": torch.arange(18, dtype=torch.float32).reshape(3, 6),
        },
        non_tensor_batch={
            "data_source": np.asarray(["nq", "nq", "nq"], dtype=object),
            "extra_info": np.asarray([extra, extra, extra], dtype=object),
            "env_kwargs": np.asarray([env, env, env], dtype=object),
            "reward_model": np.asarray([reward, reward, reward], dtype=object),
            # This field must never be read or copied by the V4 projection.
            "metadata": np.asarray(
                [{"supporting_facts": ["secret"]}] * 3, dtype=object
            ),
            "uid": np.asarray(["group-random", "group-random", "group-random"]),
            "traj_uid": np.asarray(["traj-a", "traj-a", "traj-b"]),
            "turn_step": np.asarray([0, 1, 0]),
            "episode_lengths": np.asarray([2.0, 2.0, 1.0]),
            "episode_rewards": np.asarray([1.0, 1.0, 0.0]),
            "terminal_binary_em": np.asarray([1, 1, 0], dtype=object),
            "is_action_valid": np.asarray([True, True, True]),
            "search_feedback": np.asarray(
                ["<documents>Doc 1: Found\nEvidence</documents>", "", ""],
                dtype=object,
            ),
            "turn_context_text": np.asarray(["H0", "H1", "H0b"], dtype=object),
            "urcr_is_adjustment_copy": np.asarray(
                [False, adjustment_copy, False]
            ),
        },
        __len__=lambda self: 3,
    )


class FakeBatch(SimpleNamespace):
    def __len__(self):
        return len(self.batch["responses"])


def _as_fake_batch(**kwargs):
    value = _batch(**kwargs)
    return FakeBatch(batch=value.batch, non_tensor_batch=value.non_tensor_batch)


def test_v4_projection_uses_stable_ids_and_drops_dataset_metadata() -> None:
    records = build_v4_turn_records(
        batch=_as_fake_batch(),
        tokenizer=PieceTokenizer(),
        global_step=4,
        expected_rollouts_per_group=2,
    )
    assert len(records) == 3
    assert records[0].rollout_group_id == records[2].rollout_group_id
    assert records[0].trajectory_id == records[1].trajectory_id
    assert records[0].turn_id != records[1].turn_id
    assert records[0].rollout_index == 0
    assert records[2].rollout_index == 1
    assert records[0].observation_consumable
    assert records[0].action_type == "search"
    assert records[1].action_type == "answer"
    assert records[1].terminal_binary_em == 1
    assert records[0].think_content_positions == (1,)
    assert records[0].action_target_positions == (4,)
    assert records[0].action_content_positions == (4,)
    assert records[0].action_boundary_positions == ()
    assert records[0].old_action_target_log_probs == (4.0,)
    assert all(not forbidden_online_metadata_paths(record.__dict__) for record in records)
    assert original_trajectory_count(records) == 2


def test_dataset_evidence_metadata_cannot_change_v4_turn_records() -> None:
    first = _as_fake_batch()
    second = _as_fake_batch()
    first.non_tensor_batch["metadata"] = np.asarray(
        [{"supporting_facts": ["first"]}] * 3,
        dtype=object,
    )
    second.non_tensor_batch["metadata"] = np.asarray(
        [{"supporting_facts": ["different"], "context": ["hidden"]}] * 3,
        dtype=object,
    )
    first_records = build_v4_turn_records(
        batch=first,
        tokenizer=PieceTokenizer(),
        global_step=4,
    )
    second_records = build_v4_turn_records(
        batch=second,
        tokenizer=PieceTokenizer(),
        global_step=4,
    )
    assert first_records == second_records


def test_adjustment_copy_has_no_effect_on_original_trajectory_population() -> None:
    records = build_v4_turn_records(
        batch=_as_fake_batch(adjustment_copy=True),
        tokenizer=PieceTokenizer(),
        global_step=4,
    )
    assert records[1].is_adjustment_copy
    assert original_trajectory_count(records) == 2


def test_observation_pool_and_repeat_check_use_only_visible_documents() -> None:
    records = build_v4_turn_records(
        batch=_as_fake_batch(),
        tokenizer=PieceTokenizer(),
        global_step=4,
    )
    pool = build_observation_pool(records, tokenizer=PieceTokenizer())
    assert len(pool) == 1
    assert pool[0].question_id == records[0].question_id
    assert pool[0].call_bucket == "1"
    assert pool[0].valid

    repeated = replace(
        records[0],
        visible_context_text=(
            "History: <documents>Doc 1: Same\nBody text</documents>"
        ),
        observation_text="<documents>Doc 1: Same\nBody text</documents>",
    )
    assert visible_history_passages(repeated.visible_context_text) == (
        "Same\nBody text",
    )
    assert record_is_visible_exact_repeat(repeated)
    assert not record_is_visible_exact_repeat(
        replace(repeated, observation_text="<documents>Doc 1: New\nBody</documents>")
    )


def test_v4_projection_uses_explicit_binary_outcome_not_shaped_episode_reward() -> None:
    batch = _as_fake_batch()
    batch.non_tensor_batch["episode_rewards"][0] = 0.5
    records = build_v4_turn_records(batch=batch, tokenizer=PieceTokenizer(), global_step=1)
    assert records[0].terminal_binary_em == 1


def test_v4_projection_rejects_nonbinary_terminal_outcome() -> None:
    batch = _as_fake_batch()
    batch.non_tensor_batch["terminal_binary_em"][0] = 0.5
    with pytest.raises(ValueError, match="raw binary evaluator outcome"):
        build_v4_turn_records(batch=batch, tokenizer=PieceTokenizer(), global_step=1)


def test_v4_projection_requires_consistent_trajectory_outcome() -> None:
    batch = _as_fake_batch()
    batch.non_tensor_batch["terminal_binary_em"][1] = 0
    with pytest.raises(ValueError, match="inconsistent terminal outcomes"):
        build_v4_turn_records(batch=batch, tokenizer=PieceTokenizer(), global_step=1)


def test_boundary_merged_short_answer_remains_valid_and_keeps_scoring_target() -> None:
    batch = _as_fake_batch()
    short_answer = torch.tensor([2, 3, 4, 8, 11, 0])
    batch.batch["responses"][2] = short_answer
    batch.batch["input_ids"][2, -6:] = short_answer
    batch.batch["response_mask"][2, -1] = 0
    batch.batch["attention_mask"][2, -1] = 0
    records = build_v4_turn_records(
        batch=batch,
        tokenizer=PieceTokenizer(),
        global_step=1,
    )
    answer = records[2]
    assert answer.valid_action
    assert answer.action_text == "No"
    assert answer.action_target_positions == (4,)
    assert answer.action_content_positions == ()
    assert answer.action_boundary_positions == (4,)
    assert answer.old_action_target_log_probs == (16.0,)
