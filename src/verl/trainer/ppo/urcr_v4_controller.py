"""CPU request planning around the URCR-V4 scorer RPC."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

from verl.trainer.ppo.urcr_v4_data import (
    V4TurnRecord,
    build_observation_pool,
    record_is_visible_exact_repeat,
)
from verl.trainer.ppo.urcr_v4_method import (
    AnswerTrajectory,
    ChunkRouting,
    METHOD_REVISION,
    SearchUtilityResult,
    compute_answer_residuals,
    control_rng_for_anchor,
    select_observation_controls,
)
from verl.trainer.ppo.urcr_v4_pipeline import (
    ResponsibilityAggregation,
    aggregate_responsibility_scores,
    aggregate_search_utility_scores,
)
from verl.trainer.ppo.urcr_v4_requests import (
    ResponsibilityRequestBundle,
    SearchUtilityRequestBundle,
    build_responsibility_requests,
    build_search_utility_requests,
    responsibility_request_unavailable_reason,
    should_score_responsibility,
)
from verl.trainer.ppo.urcr_v4_scorer import TeacherForcedItem


def _stable_subset(records: Sequence[V4TurnRecord], limit: int | None):
    if limit is None or len(records) <= limit:
        return tuple(records)
    if limit <= 0:
        return ()
    return tuple(
        sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{METHOD_REVISION}|diagnostic-anchor|{record.turn_id}".encode()
            ).digest(),
        )[:limit]
    )


def _zero_search(reason: str) -> SearchUtilityResult:
    return SearchUtilityResult(0.0, None, None, None, False, False, reason)


@dataclass(frozen=True)
class SearchRequestPlan:
    bundles: tuple[SearchUtilityRequestBundle, ...]
    items: tuple[TeacherForcedItem, ...]
    zero_by_turn: dict[str, SearchUtilityResult]


def build_search_request_plan(
    records: Sequence[V4TurnRecord],
    *,
    tokenizer,
    actor_snapshot_id: str,
    max_total_tokens: int,
    max_anchors: int | None = None,
    anchor_turn_ids: set[str] | None = None,
) -> SearchRequestPlan:
    pool = build_observation_pool(list(records), tokenizer=tokenizer)
    anchor_by_turn = {candidate.observation_id: candidate for candidate in pool}
    candidates = [
        record
        for record in records
        if not record.is_adjustment_copy
        and record.action_type == "search"
        and record.valid_action
        and record.observation_consumable
        and (anchor_turn_ids is None or record.turn_id in anchor_turn_ids)
    ]
    selected = {record.turn_id for record in _stable_subset(candidates, max_anchors)}
    bundles = []
    zero_by_turn: dict[str, SearchUtilityResult] = {}
    for record in records:
        if record.action_type != "search" or record.is_adjustment_copy:
            continue
        if not record.valid_action:
            zero_by_turn[record.turn_id] = _zero_search("invalid_search")
            continue
        if not record.observation_consumable:
            zero_by_turn[record.turn_id] = _zero_search(
                "unconsumable_observation"
            )
            continue
        if anchor_turn_ids is not None and record.turn_id not in anchor_turn_ids:
            zero_by_turn[record.turn_id] = _zero_search(
                "diagnostic_trajectory_subset"
            )
            continue
        if record.turn_id not in selected:
            zero_by_turn[record.turn_id] = _zero_search("diagnostic_anchor_cap")
            continue
        anchor = anchor_by_turn.get(record.turn_id)
        if anchor is None or not anchor.valid:
            zero_by_turn[record.turn_id] = _zero_search("invalid_observation")
            continue
        controls = select_observation_controls(
            anchor,
            pool,
            rng=control_rng_for_anchor(
                global_step=record.global_step,
                turn_id=record.turn_id,
            ),
        )
        if controls.failure_reason is not None:
            zero_by_turn[record.turn_id] = _zero_search(controls.failure_reason)
            continue
        try:
            bundles.append(
                build_search_utility_requests(
                    record,
                    controls,
                    tokenizer=tokenizer,
                    actor_snapshot_id=actor_snapshot_id,
                    max_total_tokens=max_total_tokens,
                )
            )
        except ValueError as exc:
            if str(exc) != "score_context_overflow":
                raise
            zero_by_turn[record.turn_id] = _zero_search("score_context_overflow")
    return SearchRequestPlan(
        bundles=tuple(bundles),
        items=tuple(item for bundle in bundles for item in bundle.items),
        zero_by_turn=zero_by_turn,
    )


def resolve_search_utilities(
    plan: SearchRequestPlan,
    scores: Mapping[str, float],
    records: Sequence[V4TurnRecord],
    *,
    delta: float,
    scale: float,
) -> dict[str, SearchUtilityResult]:
    output = dict(plan.zero_by_turn)
    record_by_turn = {record.turn_id: record for record in records}
    for bundle in plan.bundles:
        record = record_by_turn[bundle.turn_id]
        output[bundle.turn_id] = aggregate_search_utility_scores(
            bundle,
            scores,
            delta=delta,
            scale=scale,
            is_visible_exact_repeat=record_is_visible_exact_repeat(record),
        ).utility
    return output


def action_utilities(
    records: Sequence[V4TurnRecord],
    search_utilities: Mapping[str, SearchUtilityResult],
) -> dict[str, float]:
    answers = [
        AnswerTrajectory(
            trajectory_id=record.trajectory_id,
            rollout_group_id=record.rollout_group_id,
            binary_em=record.terminal_binary_em,
            valid_terminal_answer=bool(
                record.action_type == "answer" and record.valid_action
            ),
            answer=record.action_text if record.action_type == "answer" else None,
            aliases=record.answer_aliases,
        )
        for record in records
        if not record.is_adjustment_copy and record.action_type == "answer"
    ]
    answer_residuals = compute_answer_residuals(answers)
    output = {}
    for record in records:
        if record.is_adjustment_copy:
            continue
        if record.action_type == "search":
            result = search_utilities.get(record.turn_id)
            output[record.turn_id] = result.utility if result is not None else 0.0
        elif record.action_type == "answer":
            residual = answer_residuals.get(record.trajectory_id)
            output[record.turn_id] = residual.residual if residual is not None else 0.0
    return output


@dataclass(frozen=True)
class ResponsibilityRequestPlan:
    bundles: tuple[ResponsibilityRequestBundle, ...]
    items: tuple[TeacherForcedItem, ...]
    zero_reasons: dict[str, str]
    shadow_turn_ids: tuple[str, ...] = ()


def build_responsibility_request_plan(
    records: Sequence[V4TurnRecord],
    utilities: Mapping[str, float],
    *,
    actor_snapshot_id: str,
    max_total_tokens: int,
    max_search_anchors: int | None = None,
    max_answer_anchors: int | None = None,
) -> ResponsibilityRequestPlan:
    eligible = [
        record
        for record in records
        if should_score_responsibility(
            record,
            action_utility=float(utilities.get(record.turn_id, 0.0)),
        )
    ]
    unavailable = {
        record.turn_id: reason
        for record in eligible
        if (reason := responsibility_request_unavailable_reason(record)) is not None
    }
    scoreable = [record for record in eligible if record.turn_id not in unavailable]
    selected_search = {
        record.turn_id
        for record in _stable_subset(
            [record for record in scoreable if record.action_type == "search"],
            max_search_anchors,
        )
    }
    selected_answer = {
        record.turn_id
        for record in _stable_subset(
            [record for record in scoreable if record.action_type == "answer"],
            max_answer_anchors,
        )
    }
    selected = selected_search | selected_answer
    bundles = []
    zero_reasons = dict(unavailable)
    for record in scoreable:
        if record.turn_id not in selected:
            zero_reasons[record.turn_id] = "diagnostic_anchor_cap"
            continue
        try:
            bundles.append(
                build_responsibility_requests(
                    record,
                    actor_snapshot_id=actor_snapshot_id,
                    max_total_tokens=max_total_tokens,
                )
            )
        except ValueError as exc:
            if str(exc) != "score_context_overflow":
                raise
            zero_reasons[record.turn_id] = "score_context_overflow"
    return ResponsibilityRequestPlan(
        bundles=tuple(bundles),
        items=tuple(
            item
            for bundle in bundles
            for item in (bundle.whole_mask_item, *bundle.chunk_mask_items)
        ),
        zero_reasons=zero_reasons,
    )


def build_calibration_responsibility_request_plan(
    records: Sequence[V4TurnRecord],
    utilities: Mapping[str, float],
    *,
    actor_snapshot_id: str,
    max_total_tokens: int,
    max_search_anchors: int = 64,
    max_answer_anchors: int = 64,
) -> ResponsibilityRequestPlan:
    """Prefer nonzero-credit anchors, then add stable zero-utility shadows."""
    limits = {"search": int(max_search_anchors), "answer": int(max_answer_anchors)}
    if any(value <= 0 for value in limits.values()):
        raise ValueError("calibration responsibility limits must be positive")

    selected: list[V4TurnRecord] = []
    shadow_turn_ids: set[str] = set()
    zero_reasons: dict[str, str] = {}
    for action_type, limit in limits.items():
        candidates = [
            record
            for record in records
            if not record.is_adjustment_copy
            and record.action_type == action_type
            and record.valid_action
            and record.think_content_positions
            and record.think_chunks
        ]
        for record in candidates:
            unavailable_reason = responsibility_request_unavailable_reason(record)
            if unavailable_reason is not None:
                zero_reasons[record.turn_id] = unavailable_reason
        candidates = [
            record for record in candidates if record.turn_id not in zero_reasons
        ]
        active = [
            record
            for record in candidates
            if float(utilities.get(record.turn_id, 0.0)) != 0.0
        ]
        chosen_active = list(_stable_subset(active, limit))
        chosen_ids = {record.turn_id for record in chosen_active}
        remaining = max(0, limit - len(chosen_active))
        shadows = [
            record
            for record in candidates
            if record.turn_id not in chosen_ids
            and float(utilities.get(record.turn_id, 0.0)) == 0.0
        ]
        chosen_shadows = list(_stable_subset(shadows, remaining))
        selected.extend(chosen_active)
        selected.extend(chosen_shadows)
        shadow_turn_ids.update(record.turn_id for record in chosen_shadows)
        for record in active:
            if record.turn_id not in chosen_ids:
                zero_reasons[record.turn_id] = "calibration_anchor_cap"

    bundles = []
    for record in sorted(selected, key=lambda value: value.turn_id):
        try:
            bundles.append(
                build_responsibility_requests(
                    record,
                    actor_snapshot_id=actor_snapshot_id,
                    max_total_tokens=max_total_tokens,
                )
            )
        except ValueError as exc:
            if str(exc) != "score_context_overflow":
                raise
            zero_reasons[record.turn_id] = "score_context_overflow"
            shadow_turn_ids.discard(record.turn_id)
    return ResponsibilityRequestPlan(
        bundles=tuple(bundles),
        items=tuple(
            item
            for bundle in bundles
            for item in (
                bundle.plain_item,
                bundle.whole_mask_item,
                *bundle.chunk_mask_items,
            )
        ),
        zero_reasons=zero_reasons,
        shadow_turn_ids=tuple(sorted(shadow_turn_ids)),
    )


def _zero_responsibility(
    record: V4TurnRecord,
) -> ResponsibilityAggregation:
    count = len(record.think_chunks)
    return ResponsibilityAggregation(
        full_score=0.0,
        whole_masked_score=0.0,
        whole_dependency=0.0,
        rho=0.0,
        chunk_routing=ChunkRouting(
            dependencies=(0.0,) * count,
            weights=(0.0,) * count,
            positive_mass=0.0,
        ),
    )


def resolve_responsibilities(
    plan: ResponsibilityRequestPlan,
    scores: Mapping[str, float],
    records: Sequence[V4TurnRecord],
    utilities: Mapping[str, float],
    *,
    query_delta: float,
    query_scale: float,
    answer_delta: float,
    answer_scale: float,
    query_chunk_delta: float,
    answer_chunk_delta: float,
) -> dict[str, ResponsibilityAggregation]:
    record_by_turn = {record.turn_id: record for record in records}
    output = {
        turn_id: _zero_responsibility(record_by_turn[turn_id])
        for turn_id in plan.zero_reasons
    }
    for bundle in plan.bundles:
        is_search = bundle.action_type == "search"
        output[bundle.turn_id] = aggregate_responsibility_scores(
            bundle,
            scores,
            delta=query_delta if is_search else answer_delta,
            scale=query_scale if is_search else answer_scale,
            chunk_delta=(
                query_chunk_delta if is_search else answer_chunk_delta
            ),
        )
    for record in records:
        if (
            float(utilities.get(record.turn_id, 0.0)) != 0.0
            and record.think_chunks
            and record.turn_id not in output
        ):
            output[record.turn_id] = _zero_responsibility(record)
    return output
