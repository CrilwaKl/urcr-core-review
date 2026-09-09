from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from verl.trainer.ppo.urcr_v4_data import V4TurnRecord
from verl.trainer.ppo.urcr_v4_method import ChunkRouting, SearchUtilityResult
from verl.trainer.ppo.urcr_v4_pipeline import (
    ResponsibilityAggregation,
    aggregate_responsibility_scores,
    aggregate_search_utility_scores,
    assemble_local_credits,
    interleaved_gradient_batch_layout,
    local_credit_tensors,
)
from verl.trainer.ppo.urcr_v4_requests import (
    ResponsibilityRequestBundle,
    SearchUtilityRequestBundle,
)
from verl.trainer.ppo.urcr_v4_scorer import build_teacher_forced_item


def _item(identity: str, view: str, *, blocked=(1,)):
    return build_teacher_forced_item(
        request_id=identity,
        view=view,
        actor_snapshot_id="snapshot",
        context_ids=[1, 2],
        target_prefix_ids=[3],
        target_content_ids=[4],
        blocked_key_positions=blocked,
    )


def test_search_score_aggregation_averages_aliases_per_view_before_contrasts() -> None:
    views = ["real", "empty", "control_0", "control_1", "control_2"]
    items = tuple(
        _item(f"{view}:{alias}", view)
        for view in views
        for alias in range(2)
    )
    bundle = SearchUtilityRequestBundle(
        turn_id="turn",
        probe_aliases=("a", "b"),
        control_observation_ids=("c0", "c1", "c2"),
        items=items,
    )
    values = {
        "real:0": 1.0,
        "real:1": 0.8,
        "empty:0": 0.0,
        "empty:1": 0.2,
        "control_0:0": 0.2,
        "control_0:1": 0.4,
        "control_1:0": 0.2,
        "control_1:1": 0.4,
        "control_2:0": 0.2,
        "control_2:1": 0.4,
    }
    result = aggregate_search_utility_scores(
        bundle,
        values,
        delta=0.1,
        scale=0.5,
        is_visible_exact_repeat=False,
    )
    assert result.view_scores["real"] == pytest.approx(0.9)
    assert result.utility.g_incremental == pytest.approx(0.8)
    assert result.utility.g_control == pytest.approx(0.6)
    assert result.utility.g_directed == pytest.approx(0.6)
    assert result.utility.utility == pytest.approx(math.tanh(1.0))


def test_gradient_layout_spreads_real_rows_across_every_microbatch() -> None:
    layout, original = interleaved_gradient_batch_layout(
        list(range(10)),
        target_row_count=32,
        micro_batch_size=8,
    )
    assert len(layout) == len(original) == 32
    assert sum(original) == 10
    assert {row for row, keep in zip(layout, original, strict=True) if keep} == set(
        range(10)
    )
    for start in range(0, 32, 8):
        assert any(original[start : start + 8])


def test_responsibility_aggregation_uses_old_full_score_and_masked_requests() -> None:
    whole = _item("whole", "whole_mask")
    chunks = (_item("chunk0", "chunk_mask_0"), _item("chunk1", "chunk_mask_1"))
    bundle = ResponsibilityRequestBundle(
        turn_id="turn",
        action_type="search",
        full_score_from_old_log_probs=-1.0,
        full_target_ids=(4,),
        plain_item=_item("plain", "plain", blocked=()),
        whole_mask_item=whole,
        chunk_mask_items=chunks,
    )
    result = aggregate_responsibility_scores(
        bundle,
        {"whole": -1.5, "chunk0": -1.4, "chunk1": -1.1},
        delta=0.1,
        scale=0.4,
        chunk_delta=0.05,
    )
    assert result.whole_dependency == pytest.approx(0.5)
    assert result.rho == pytest.approx(1 - math.exp(-1.0))
    assert sum(result.chunk_routing.weights) == pytest.approx(1.0)
    assert result.chunk_routing.weights[0] > result.chunk_routing.weights[1]


def _record(
    row: int,
    trajectory: str,
    turn: int,
    action: str,
    *,
    answer: str = "",
    adjustment: bool = False,
) -> V4TurnRecord:
    group = "group"
    return V4TurnRecord(
        global_step=1,
        batch_row_index=row,
        is_adjustment_copy=adjustment,
        question_id="nq:train:row:1",
        rollout_group_id=group,
        trajectory_id=trajectory,
        turn_id=f"{trajectory}:turn:{turn}",
        rollout_index=0 if trajectory == "t0" else 1,
        turn_index=turn,
        data_source="nq",
        question="Where?",
        answer_aliases=("Paris",),
        visible_context_text="H",
        visible_prompt_token_ids=(1,),
        sampled_response_token_ids=(2, 3, 4),
        action_type=action,
        action_text=answer if action == "answer" else "query",
        observation_text="docs" if action == "search" else "",
        valid_action=True,
        observation_consumable=action == "search",
        terminal_binary_em=0,
        think_content_positions=(0,),
        action_target_positions=(1,),
        action_content_positions=(1,),
        action_boundary_positions=(),
        action_tag_positions=(2,),
        think_chunks=((0,),),
        old_action_target_log_probs=(-1.0,),
    )


def _responsibility(turn_id: str) -> ResponsibilityAggregation:
    del turn_id
    return ResponsibilityAggregation(
        full_score=-1.0,
        whole_masked_score=-2.0,
        whole_dependency=1.0,
        rho=1.0,
        chunk_routing=ChunkRouting((1.0,), (1.0,), 1.0),
    )


def test_credit_assembly_uses_full_group_answer_residual_and_one_trajectory_cap() -> None:
    records = [
        _record(0, "t0", 0, "search"),
        _record(1, "t0", 1, "search"),
        _record(2, "t0", 2, "answer", answer="Paris city"),
        _record(3, "t1", 0, "answer", answer="London"),
        _record(4, "t0", 0, "search", adjustment=True),
    ]
    search = {
        record.turn_id: SearchUtilityResult(1.0, 1.0, 1.0, 1.0, False, True, None)
        for record in records[:2]
    }
    responsibility = {
        record.turn_id: _responsibility(record.turn_id)
        for record in records[:4]
    }
    assembled = assemble_local_credits(
        records,
        search_utilities=search,
        responsibilities=responsibility,
    )
    by_row = {row.batch_row_index: row for row in assembled.rows}
    assert assembled.original_trajectory_count == 2
    assert assembled.search_raw_absolute_mass_by_trajectory["t0"] == 4.0
    assert assembled.search_capped_absolute_mass_by_trajectory["t0"] == pytest.approx(2.0)
    assert by_row[0].search_cap_scale == pytest.approx(0.5)
    assert by_row[0].action_coefficient == pytest.approx(0.5)
    assert by_row[0].think_chunk_coefficients == pytest.approx((0.5,))
    assert by_row[1].action_coefficient == pytest.approx(0.5)
    # Same-R answer residuals are equal and opposite before the shared 0.25 factor.
    assert by_row[2].action_coefficient > 0
    assert by_row[3].action_coefficient < 0
    assert by_row[2].think_chunk_coefficients[0] > 0
    assert by_row[3].think_chunk_coefficients[0] < 0
    assert by_row[2].action_coefficient + by_row[3].action_coefficient == pytest.approx(0)
    assert by_row[4].zero_reason == "adjustment_copy"
    assert by_row[4].absolute_mass == 0.0
    original_rows = local_credit_tensors(
        assembled,
        response_mask=torch.ones((5, 3), dtype=torch.long),
    )["urcr_v4_original_row_mask"]
    assert original_rows.tolist() == [True, True, True, True, False]
    assert all(value <= 2.5 + 1e-8 for value in assembled.total_absolute_mass_by_trajectory.values())


def test_action_credit_survives_missing_think_and_keeps_both_signs() -> None:
    records = [
        replace(
            _record(0, "t0", 0, "search"),
            think_content_positions=(),
            think_chunks=(),
        ),
        replace(
            _record(1, "t0", 1, "answer", answer="Paris city"),
            think_content_positions=(),
            think_chunks=(),
        ),
        replace(
            _record(2, "t1", 0, "answer", answer="London"),
            think_content_positions=(),
            think_chunks=(),
        ),
    ]
    search = {
        records[0].turn_id: SearchUtilityResult(
            -0.1,
            -1.0,
            -1.0,
            -1.0,
            False,
            True,
            None,
        )
    }
    assembled = assemble_local_credits(
        records,
        search_utilities=search,
        responsibilities={},
    )
    by_row = {row.batch_row_index: row for row in assembled.rows}
    assert by_row[0].action_coefficient == pytest.approx(-0.1)
    assert by_row[0].think_chunk_coefficients == ()
    assert by_row[1].action_coefficient > 0
    assert by_row[2].action_coefficient < 0
    assert by_row[1].think_chunk_coefficients == ()
    assert by_row[2].think_chunk_coefficients == ()


def test_diagnostic_trajectory_subset_changes_the_local_population_only() -> None:
    records = [
        _record(0, "t0", 0, "answer", answer="Paris city"),
        _record(1, "t1", 0, "answer", answer="London"),
    ]
    assembled = assemble_local_credits(
        records,
        search_utilities={},
        responsibilities={record.turn_id: _responsibility(record.turn_id) for record in records},
        active_trajectory_ids={"t0"},
    )
    assert assembled.original_trajectory_count == 1
    assert assembled.local_population_trajectory_ids == ("t0",)
    by_row = {row.batch_row_index: row for row in assembled.rows}
    assert by_row[1].zero_reason == "diagnostic_trajectory_subset"
    tensors = local_credit_tensors(
        assembled,
        response_mask=torch.ones((2, 3), dtype=torch.long),
    )
    assert tensors["urcr_v4_original_row_mask"].tolist() == [True, False]


def test_local_credit_tensors_touch_only_declared_content_spans() -> None:
    records = [
        _record(0, "t0", 0, "search"),
        _record(1, "t0", 1, "answer", answer="Paris city"),
        _record(2, "t1", 0, "answer", answer="London"),
    ]
    search = {
        records[0].turn_id: SearchUtilityResult(
            0.5,
            1.0,
            1.0,
            1.0,
            False,
            True,
            None,
        )
    }
    responsibility = {
        record.turn_id: _responsibility(record.turn_id) for record in records
    }
    assembled = assemble_local_credits(
        records,
        search_utilities=search,
        responsibilities=responsibility,
    )
    response_mask = torch.tensor(
        [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]
    )
    tensors = local_credit_tensors(assembled, response_mask=response_mask)
    masks = tensors["urcr_v4_span_masks"]
    coefficients = tensors["urcr_v4_span_coefficients"]
    channels = tensors["urcr_v4_span_channels"]
    original_rows = tensors["urcr_v4_original_row_mask"]
    assert masks.shape == (3, 7, 5)
    assert masks[0, 0].nonzero().flatten().tolist() == [2]
    assert masks[0, 1].nonzero().flatten().tolist() == [1]
    assert coefficients[0, 0] == pytest.approx(0.5)
    assert coefficients[0, 1] == pytest.approx(0.5)
    assert channels[0, :2].tolist() == [0, 1]
    assert channels[1, :2].tolist() == [2, 3]
    assert original_rows.tolist() == [True, True, True]
    assert not masks[..., 0].any()
    assert not masks[..., 3].any()
    assert not masks[..., 4].any()


def test_boundary_only_short_answer_keeps_think_credit_without_updating_tag_token() -> None:
    boundary_answer = replace(
        _record(0, "t0", 0, "answer", answer="Paris city"),
        action_content_positions=(),
        action_boundary_positions=(1,),
        action_tag_positions=(1, 2),
    )
    other_answer = _record(1, "t1", 0, "answer", answer="London")
    records = [boundary_answer, other_answer]
    responsibility = {
        record.turn_id: _responsibility(record.turn_id) for record in records
    }
    assembled = assemble_local_credits(
        records,
        search_utilities={},
        responsibilities=responsibility,
    )
    by_row = {row.batch_row_index: row for row in assembled.rows}
    assert by_row[0].action_coefficient == 0.0
    assert by_row[0].think_chunk_coefficients[0] > 0
    assert by_row[0].zero_reason == "action_boundary_only"
    tensors = local_credit_tensors(
        assembled,
        response_mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
    )
    assert not tensors["urcr_v4_span_masks"][0, 0].any()
    assert tensors["urcr_v4_span_masks"][0, 1, 0]
    assert not tensors["urcr_v4_span_masks"][0, :, 1].any()
