from __future__ import annotations

import pytest

from verl.trainer.ppo.urcr_v4_calibration import (
    answer_residual_diagnostics,
    calibrate_responsibility_scores,
    calibrate_search_scores,
)
from verl.trainer.ppo.urcr_v4_controller import (
    ResponsibilityRequestPlan,
    SearchRequestPlan,
)
from verl.trainer.ppo.urcr_v4_data import V4TurnRecord
from verl.trainer.ppo.urcr_v4_requests import (
    ResponsibilityRequestBundle,
    SearchUtilityRequestBundle,
)
from verl.trainer.ppo.urcr_v4_scorer import build_teacher_forced_item


def _item(identity: str, view: str, group: str, *, blocked=()):
    return build_teacher_forced_item(
        request_id=identity,
        comparison_group_id=group,
        view=view,
        actor_snapshot_id="snapshot",
        context_ids=[1, 2, 3],
        target_prefix_ids=[],
        target_content_ids=[4],
        blocked_key_positions=blocked,
    )


def _record(
    row: int,
    trajectory: str,
    action_type: str,
    *,
    answer: str = "",
    binary_em: int = 0,
) -> V4TurnRecord:
    return V4TurnRecord(
        global_step=1,
        batch_row_index=row,
        is_adjustment_copy=False,
        question_id="nq:train:row:1",
        rollout_group_id="group",
        trajectory_id=trajectory,
        turn_id=f"{trajectory}:turn:0",
        rollout_index=row,
        turn_index=0,
        data_source="nq",
        question="Where?",
        answer_aliases=("Paris",),
        visible_context_text="",
        visible_prompt_token_ids=(1,),
        sampled_response_token_ids=(2, 3, 4),
        action_type=action_type,
        action_text=answer if action_type == "answer" else "query",
        observation_text="<documents>Doc 1: France\nParis</documents>",
        valid_action=True,
        observation_consumable=action_type == "search",
        terminal_binary_em=binary_em,
        think_content_positions=(0,),
        action_target_positions=(1,),
        action_content_positions=(1,),
        action_boundary_positions=(),
        action_tag_positions=(2,),
        think_chunks=((0,),),
        old_action_target_log_probs=(-1.0,),
    )


def test_search_calibration_freezes_shared_dead_zone_and_keeps_raw_contrasts() -> None:
    record = _record(0, "t0", "search")
    views = ["real", "empty", "control_0", "control_1", "control_2"]
    items = tuple(_item(view, view, record.turn_id) for view in views)
    bundle = SearchUtilityRequestBundle(
        turn_id=record.turn_id,
        probe_aliases=("Paris",),
        control_observation_ids=("a", "b", "c"),
        items=items,
        control_unique_count=3,
        control_used_replacement=False,
    )
    plan = SearchRequestPlan((bundle,), items, {})
    scores = {
        "real": 1.0,
        "empty": 0.0,
        "control_0": 0.2,
        "control_1": 0.3,
        "control_2": 0.4,
    }
    result = calibrate_search_scores(
        plan,
        scores,
        [record],
        numeric_absolute_differences=[0.0],
    )
    assert result.calibration.delta == pytest.approx(0.1)
    assert result.calibration.scale == 1.0
    assert result.calibration.limited
    assert result.rows[0]["g_incremental"] == pytest.approx(1.0)
    assert result.rows[0]["g_control"] == pytest.approx(0.7)
    assert result.utilities[record.turn_id].utility > 0


def test_answer_diagnostics_preserve_within_stratum_zero_sum() -> None:
    records = [
        _record(0, "t0", "answer", answer="Paris nearby", binary_em=0),
        _record(1, "t1", "answer", answer="London", binary_em=0),
    ]
    result = answer_residual_diagnostics(records)
    values = [row["residual"] for row in result["rows"]]
    assert values[0] > 0
    assert values[1] < 0
    assert sum(values) == pytest.approx(0.0)
    assert result["summary"]["maximum_absolute_group_residual_sum"] < 1e-12


def test_responsibility_calibration_uses_same_forward_plain_score() -> None:
    bundles = []
    scores = {}
    for action_type in ("search", "answer"):
        plain = _item(f"{action_type}:plain", "plain", action_type)
        whole = _item(
            f"{action_type}:whole", "whole_mask", action_type, blocked=(1,)
        )
        chunk = _item(
            f"{action_type}:chunk", "chunk_mask_0", action_type, blocked=(1,)
        )
        bundles.append(
            ResponsibilityRequestBundle(
                turn_id=action_type,
                action_type=action_type,
                full_score_from_old_log_probs=-0.8,
                full_target_ids=(4,),
                plain_item=plain,
                whole_mask_item=whole,
                chunk_mask_items=(chunk,),
            )
        )
        scores[plain.request_id] = -1.0
        scores[whole.request_id] = -2.0
        scores[chunk.request_id] = -1.5
    plan = ResponsibilityRequestPlan(
        bundles=tuple(bundles),
        items=tuple(
            item
            for bundle in bundles
            for item in (bundle.plain_item, bundle.whole_mask_item, *bundle.chunk_mask_items)
        ),
        zero_reasons={},
        shadow_turn_ids=("answer",),
    )
    result = calibrate_responsibility_scores(
        plan,
        scores,
        numeric_absolute_differences={"search": [0.0], "answer": [0.0]},
    )
    assert result.search.delta == pytest.approx(0.6)
    assert result.answer.delta == pytest.approx(0.6)
    assert all(row["rho"] > 0 for row in result.rows)
    assert result.summary["zero_utility_shadow_count"] == 1
    assert result.summary["old_vs_plain_absolute_difference"]["p95"] == pytest.approx(0.2)
    assert result.summary["backend_alignment_error_used_for_delta"]
