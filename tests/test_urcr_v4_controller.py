from __future__ import annotations

from dataclasses import replace

from verl.trainer.ppo.urcr_v4_controller import (
    build_calibration_responsibility_request_plan,
    build_responsibility_request_plan,
    build_search_request_plan,
    resolve_responsibilities,
    resolve_search_utilities,
)
from verl.trainer.ppo.urcr_v4_data import V4TurnRecord


class Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text, **_kwargs):
        return [ord(character) + 1 for character in str(text)]

    def __call__(self, text, **_kwargs):
        value = str(text)
        return {
            "input_ids": self.encode(value),
            "offset_mapping": [(index, index + 1) for index in range(len(value))],
        }


def _record(action_type: str = "answer") -> V4TurnRecord:
    return V4TurnRecord(
        global_step=1,
        batch_row_index=0,
        is_adjustment_copy=False,
        question_id="nq:train:row:1",
        rollout_group_id="group",
        trajectory_id="trajectory",
        turn_id="trajectory:turn:0",
        rollout_index=0,
        turn_index=0,
        data_source="nq",
        question="Where?",
        answer_aliases=("Paris",),
        visible_context_text="Question",
        visible_prompt_token_ids=(1, 2),
        sampled_response_token_ids=(3, 4, 5, 6),
        action_type=action_type,
        action_text="Paris" if action_type == "answer" else "capital France",
        observation_text=(
            "<documents>Doc 1: France\nParis is the capital.</documents>"
            if action_type == "search"
            else ""
        ),
        valid_action=True,
        observation_consumable=action_type == "search",
        terminal_binary_em=1,
        think_content_positions=(0,),
        action_target_positions=(2,),
        action_content_positions=(2,),
        action_boundary_positions=(),
        action_tag_positions=(3,),
        think_chunks=((0,),),
        old_action_target_log_probs=(-1.0,),
    )


def test_missing_other_question_control_zeros_search_without_dropping_turn() -> None:
    record = _record("search")
    plan = build_search_request_plan(
        [record],
        tokenizer=Tokenizer(),
        actor_snapshot_id="step-1",
        max_total_tokens=64,
    )
    assert not plan.items
    utilities = resolve_search_utilities(
        plan,
        {},
        [record],
        delta=0.1,
        scale=1.0,
    )
    assert utilities[record.turn_id].utility == 0.0
    assert utilities[record.turn_id].zero_reason == "no_other_question_control"


def test_search_diagnostic_subset_does_not_score_other_trajectories() -> None:
    first = _record("search")
    second = replace(
        first,
        batch_row_index=1,
        question_id="nq:train:row:2",
        rollout_group_id="group-2",
        trajectory_id="trajectory-2",
        turn_id="trajectory-2:turn:0",
        observation_text="<documents>Doc 1: Other\nDifferent evidence.</documents>",
    )
    plan = build_search_request_plan(
        [first, second],
        tokenizer=Tokenizer(),
        actor_snapshot_id="step-1",
        max_total_tokens=64,
        anchor_turn_ids={first.turn_id},
    )
    assert all(bundle.turn_id == first.turn_id for bundle in plan.bundles)
    assert plan.zero_by_turn[second.turn_id].zero_reason == (
        "diagnostic_trajectory_subset"
    )


def test_negative_utility_and_boundary_only_action_keep_responsibility_request() -> None:
    record = replace(
        _record(),
        action_content_positions=(),
        action_boundary_positions=(2,),
    )
    utilities = {record.turn_id: -0.2}
    plan = build_responsibility_request_plan(
        [record],
        utilities,
        actor_snapshot_id="step-1",
        max_total_tokens=64,
    )
    assert len(plan.bundles) == 1
    assert plan.bundles[0].full_target_ids == (5,)


def test_diagnostic_responsibility_cap_zeros_think_routing_but_keeps_action() -> None:
    record = _record()
    utilities = {record.turn_id: -0.2}
    plan = build_responsibility_request_plan(
        [record],
        utilities,
        actor_snapshot_id="step-1",
        max_total_tokens=64,
        max_answer_anchors=0,
    )
    resolved = resolve_responsibilities(
        plan,
        {},
        [record],
        utilities,
        query_delta=0.0,
        query_scale=1.0,
        answer_delta=0.0,
        answer_scale=1.0,
        query_chunk_delta=0.0,
        answer_chunk_delta=0.0,
    )
    assert resolved[record.turn_id].rho == 0.0
    assert resolved[record.turn_id].chunk_routing.weights == (0.0,)
    assert utilities[record.turn_id] == -0.2


def test_noncontiguous_target_zeros_only_think_routing_with_explicit_reason() -> None:
    record = replace(
        _record(),
        action_target_positions=(1, 3),
        action_content_positions=(1, 3),
        old_action_target_log_probs=(-1.0, -2.0),
    )
    utilities = {record.turn_id: -0.2}

    plan = build_responsibility_request_plan(
        [record],
        utilities,
        actor_snapshot_id="step-1",
        max_total_tokens=64,
    )
    resolved = resolve_responsibilities(
        plan,
        {},
        [record],
        utilities,
        query_delta=0.0,
        query_scale=1.0,
        answer_delta=0.0,
        answer_scale=1.0,
        query_chunk_delta=0.0,
        answer_chunk_delta=0.0,
    )

    assert plan.zero_reasons == {record.turn_id: "noncontiguous_action_target"}
    assert not plan.bundles
    assert resolved[record.turn_id].rho == 0.0
    assert utilities[record.turn_id] == -0.2


def test_calibration_responsibility_prefers_active_then_adds_shadow() -> None:
    records = []
    for index in range(3):
        record = replace(
            _record("search"),
            batch_row_index=index,
            trajectory_id=f"trajectory-{index}",
            turn_id=f"turn-{index}",
        )
        records.append(record)
    utilities = {"turn-0": -0.2, "turn-1": 0.0, "turn-2": 0.0}
    plan = build_calibration_responsibility_request_plan(
        records,
        utilities,
        actor_snapshot_id="step-1",
        max_total_tokens=64,
        max_search_anchors=2,
        max_answer_anchors=1,
    )
    assert {bundle.turn_id for bundle in plan.bundles} >= {"turn-0"}
    assert len(plan.bundles) == 2
    assert len(plan.shadow_turn_ids) == 1
    assert all(bundle.plain_item in plan.items for bundle in plan.bundles)


def test_calibration_skips_unrepresentable_target_before_applying_anchor_cap() -> None:
    good = replace(_record(), trajectory_id="good-trajectory", turn_id="good")
    bad = replace(
        _record(),
        trajectory_id="bad-trajectory",
        turn_id="bad",
        action_target_positions=(1, 3),
        action_content_positions=(1, 3),
        old_action_target_log_probs=(-1.0, -2.0),
    )

    plan = build_calibration_responsibility_request_plan(
        [bad, good],
        {bad.turn_id: 0.2, good.turn_id: 0.2},
        actor_snapshot_id="step-1",
        max_total_tokens=64,
        max_search_anchors=1,
        max_answer_anchors=1,
    )

    assert [bundle.turn_id for bundle in plan.bundles] == [good.turn_id]
    assert plan.zero_reasons[bad.turn_id] == "noncontiguous_action_target"
