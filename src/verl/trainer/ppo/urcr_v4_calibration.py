"""Frozen S150 calibration summaries for the URCR-V4 method."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from verl.trainer.ppo.urcr_v4_controller import (
    ResponsibilityRequestPlan,
    SearchRequestPlan,
)
from verl.trainer.ppo.urcr_v4_data import (
    V4TurnRecord,
    record_is_visible_exact_repeat,
)
from verl.trainer.ppo.urcr_v4_method import (
    AnswerTrajectory,
    ResponsibilityCalibration,
    SearchUtilityCalibration,
    SearchUtilityResult,
    calibrate_responsibility,
    calibrate_search_utility,
    compute_answer_residuals,
    responsibility_strength,
    soft_sparse_chunk_weights,
)
from verl.trainer.ppo.urcr_v4_pipeline import aggregate_search_utility_scores


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if numeric.size == 0:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    if not np.isfinite(numeric).all():
        raise ValueError("calibration distribution contains a nonfinite value")
    return {
        "count": int(numeric.size),
        "min": float(numeric.min()),
        "p25": float(np.quantile(numeric, 0.25)),
        "median": float(np.quantile(numeric, 0.5)),
        "p75": float(np.quantile(numeric, 0.75)),
        "p95": float(np.quantile(numeric, 0.95)),
        "max": float(numeric.max()),
    }


@dataclass(frozen=True)
class FrozenSearchCalibration:
    calibration: SearchUtilityCalibration
    utilities: dict[str, SearchUtilityResult]
    rows: tuple[dict, ...]
    summary: dict


def calibrate_search_scores(
    plan: SearchRequestPlan,
    scores: Mapping[str, float],
    records: Sequence[V4TurnRecord],
    *,
    numeric_absolute_differences: Sequence[float],
) -> FrozenSearchCalibration:
    """Calibrate one shared search dead zone and scale from frozen scores."""
    record_by_turn = {record.turn_id: record for record in records}
    raw = {}
    control_differences = []
    directed_gaps = []
    for bundle in plan.bundles:
        record = record_by_turn[bundle.turn_id]
        aggregation = aggregate_search_utility_scores(
            bundle,
            scores,
            delta=0.0,
            scale=1.0,
            is_visible_exact_repeat=record_is_visible_exact_repeat(record),
        )
        raw[bundle.turn_id] = aggregation
        if not bundle.control_used_replacement:
            control_differences.append(
                abs(
                    aggregation.view_scores["control_0"]
                    - aggregation.view_scores["control_1"]
                )
            )
        directed_gaps.append(float(aggregation.utility.g_directed))
    if not control_differences:
        raise ValueError("search calibration has no nondegenerate control pair")
    calibration = calibrate_search_utility(
        control_pair_absolute_differences=control_differences,
        numeric_absolute_differences=numeric_absolute_differences,
        directed_gaps=directed_gaps,
    )

    utilities = dict(plan.zero_by_turn)
    rows = []
    for bundle in plan.bundles:
        record = record_by_turn[bundle.turn_id]
        aggregation = aggregate_search_utility_scores(
            bundle,
            scores,
            delta=calibration.delta,
            scale=calibration.scale,
            is_visible_exact_repeat=record_is_visible_exact_repeat(record),
        )
        utility = aggregation.utility
        utilities[bundle.turn_id] = utility
        rows.append(
            {
                "turn_id": bundle.turn_id,
                "question_id": record.question_id,
                "data_source": record.data_source,
                "turn_index": record.turn_index,
                "probe_alias_count": len(bundle.probe_aliases),
                "control_observation_ids": list(bundle.control_observation_ids),
                "control_unique_count": bundle.control_unique_count,
                "control_used_replacement": bundle.control_used_replacement,
                "control_tiers": list(bundle.control_tiers),
                "view_scores": aggregation.view_scores,
                "g_incremental": utility.g_incremental,
                "g_control": utility.g_control,
                "g_directed": utility.g_directed,
                "contrast_sign_conflict": utility.contrast_sign_conflict,
                "visible_exact_repeat": record_is_visible_exact_repeat(record),
                "utility": utility.utility,
                "zero_reason": utility.zero_reason,
            }
        )

    scored_utilities = [row["utility"] for row in rows]
    summary = {
        "calibration": asdict(calibration),
        "scored_anchor_count": len(rows),
        "unscored_reason_counts": dict(
            sorted(
                Counter(
                    value.zero_reason or "unspecified"
                    for value in plan.zero_by_turn.values()
                ).items()
            )
        ),
        "control_pair_absolute_difference": distribution(control_differences),
        "g_incremental": distribution([row["g_incremental"] for row in rows]),
        "g_control": distribution([row["g_control"] for row in rows]),
        "g_directed": distribution([row["g_directed"] for row in rows]),
        "utility": distribution(scored_utilities),
        "utility_positive_count": sum(value > 0 for value in scored_utilities),
        "utility_negative_count": sum(value < 0 for value in scored_utilities),
        "utility_zero_count": sum(value == 0 for value in scored_utilities),
        "sign_conflict_count": sum(
            bool(row["contrast_sign_conflict"]) for row in rows
        ),
        "visible_exact_repeat_count": sum(
            bool(row["visible_exact_repeat"]) for row in rows
        ),
    }
    return FrozenSearchCalibration(
        calibration=calibration,
        utilities=utilities,
        rows=tuple(rows),
        summary=summary,
    )


def answer_residual_diagnostics(records: Sequence[V4TurnRecord]) -> dict:
    original_answers = [
        record
        for record in records
        if not record.is_adjustment_copy and record.action_type == "answer"
    ]
    answer_rows = [
        AnswerTrajectory(
            trajectory_id=record.trajectory_id,
            rollout_group_id=record.rollout_group_id,
            binary_em=record.terminal_binary_em,
            valid_terminal_answer=record.valid_action,
            answer=record.action_text if record.valid_action else None,
            aliases=record.answer_aliases,
        )
        for record in original_answers
    ]
    residuals = compute_answer_residuals(answer_rows)
    rows = []
    group_sums: dict[tuple[str, int], float] = defaultdict(float)
    for record in original_answers:
        residual = residuals[record.trajectory_id]
        rows.append(
            {
                "turn_id": record.turn_id,
                "trajectory_id": record.trajectory_id,
                "rollout_group_id": record.rollout_group_id,
                "question_id": record.question_id,
                "data_source": record.data_source,
                "binary_em": record.terminal_binary_em,
                "valid_terminal_answer": record.valid_action,
                "quality": residual.quality,
                "residual": residual.residual,
                "stratum_size": residual.stratum_size,
            }
        )
        if residual.eligible:
            group_sums[(record.rollout_group_id, record.terminal_binary_em)] += (
                residual.residual
            )
    eligible = [row for row in rows if row["valid_terminal_answer"]]
    values = [row["residual"] for row in eligible]
    return {
        "rows": rows,
        "summary": {
            "answer_turn_count": len(rows),
            "eligible_answer_count": len(eligible),
            "invalid_answer_count": len(rows) - len(eligible),
            "residual": distribution(values),
            "residual_positive_count": sum(value > 0 for value in values),
            "residual_negative_count": sum(value < 0 for value in values),
            "residual_zero_count": sum(value == 0 for value in values),
            "nonzero_residual_group_count": len(
                {
                    (row["rollout_group_id"], row["binary_em"])
                    for row in eligible
                    if row["residual"] != 0
                }
            ),
            "maximum_absolute_group_residual_sum": max(
                (abs(value) for value in group_sums.values()),
                default=0.0,
            ),
        },
        "residual_by_turn": {
            record.turn_id: residuals[record.trajectory_id].residual
            for record in original_answers
        },
    }


@dataclass(frozen=True)
class FrozenResponsibilityCalibration:
    search: ResponsibilityCalibration
    answer: ResponsibilityCalibration
    rows: tuple[dict, ...]
    summary: dict


def calibrate_responsibility_scores(
    plan: ResponsibilityRequestPlan,
    scores: Mapping[str, float],
    *,
    numeric_absolute_differences: Mapping[str, Sequence[float]],
) -> FrozenResponsibilityCalibration:
    """Calibrate the production old-logprob versus masked-score routing."""
    raw_rows = []
    dependencies: dict[str, list[float]] = {"search": [], "answer": []}
    backend_alignment_errors: dict[str, list[float]] = {
        "search": [],
        "answer": [],
    }
    for bundle in plan.bundles:
        plain = float(scores[bundle.plain_item.request_id])
        whole_masked = float(scores[bundle.whole_mask_item.request_id])
        if not math.isfinite(plain) or not math.isfinite(whole_masked):
            raise ValueError("responsibility calibration contains nonfinite scores")
        old_full = float(bundle.full_score_from_old_log_probs)
        dependency = old_full - whole_masked
        dependencies[bundle.action_type].append(dependency)
        backend_alignment_errors[bundle.action_type].append(abs(old_full - plain))
        chunk_masked = [
            float(scores[item.request_id]) for item in bundle.chunk_mask_items
        ]
        raw_rows.append(
            {
                "turn_id": bundle.turn_id,
                "action_type": bundle.action_type,
                "is_zero_utility_shadow": bundle.turn_id in plan.shadow_turn_ids,
                "old_full_score": old_full,
                "plain_scorer_score": plain,
                "old_vs_plain_absolute_difference": abs(old_full - plain),
                "whole_masked_score": whole_masked,
                "whole_dependency": dependency,
                "plain_whole_dependency": plain - whole_masked,
                "chunk_masked_scores": chunk_masked,
                "chunk_dependencies": [old_full - value for value in chunk_masked],
                "plain_chunk_dependencies": [plain - value for value in chunk_masked],
            }
        )

    return calibrate_responsibility_rows(
        raw_rows,
        numeric_absolute_differences=numeric_absolute_differences,
    )


def calibrate_responsibility_rows(
    raw_rows: Sequence[Mapping],
    *,
    numeric_absolute_differences: Mapping[str, Sequence[float]],
) -> FrozenResponsibilityCalibration:
    """Finalize saved frozen rows without requiring another GPU rollout."""
    normalized_rows = []
    dependencies: dict[str, list[float]] = {"search": [], "answer": []}
    backend_alignment_errors: dict[str, list[float]] = {
        "search": [],
        "answer": [],
    }
    for source in raw_rows:
        row = dict(source)
        action_type = str(row["action_type"])
        if action_type not in dependencies:
            raise ValueError(f"invalid responsibility action type: {action_type}")
        old_full = float(row["old_full_score"])
        plain = float(row["plain_scorer_score"])
        whole_masked = float(row["whole_masked_score"])
        chunk_masked = [float(value) for value in row["chunk_masked_scores"]]
        numeric = [old_full, plain, whole_masked, *chunk_masked]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("responsibility calibration contains nonfinite scores")
        dependency = old_full - whole_masked
        alignment_error = abs(old_full - plain)
        dependencies[action_type].append(dependency)
        backend_alignment_errors[action_type].append(alignment_error)
        normalized_rows.append(
            {
                **row,
                "action_type": action_type,
                "old_full_score": old_full,
                "plain_scorer_score": plain,
                "old_vs_plain_absolute_difference": alignment_error,
                "whole_masked_score": whole_masked,
                "whole_dependency": dependency,
                "plain_whole_dependency": plain - whole_masked,
                "chunk_masked_scores": chunk_masked,
                "chunk_dependencies": [old_full - value for value in chunk_masked],
                "plain_chunk_dependencies": [plain - value for value in chunk_masked],
            }
        )

    same_shape_numeric_p95 = {}
    backend_alignment_p95 = {}
    calibrated = {}
    for action_type in ("search", "answer"):
        same_shape = distribution(
            numeric_absolute_differences.get(action_type, ())
        )["p95"]
        backend = distribution(backend_alignment_errors[action_type])["p95"]
        same_shape_numeric_p95[action_type] = float(same_shape or 0.0)
        backend_alignment_p95[action_type] = float(backend or 0.0)
        effective_numeric_error = max(
            same_shape_numeric_p95[action_type],
            backend_alignment_p95[action_type],
        )
        calibrated[action_type] = calibrate_responsibility(
            action_type=action_type,
            numeric_absolute_differences=[effective_numeric_error],
            whole_dependencies=dependencies[action_type],
        )
    rows = []
    for row in normalized_rows:
        params = calibrated[row["action_type"]]
        rho = responsibility_strength(
            row["old_full_score"],
            row["whole_masked_score"],
            delta=params.delta,
            scale=params.scale,
        )
        routing = soft_sparse_chunk_weights(
            row["old_full_score"],
            row["chunk_masked_scores"],
            delta=params.delta,
        )
        rows.append(
            {
                **row,
                "rho": rho,
                "chunk_weights": list(routing.weights),
                "positive_chunk_mass": routing.positive_mass,
            }
        )

    identity_differences = [
        row["old_vs_plain_absolute_difference"] for row in rows
    ]
    summary = {
        "search_calibration": asdict(calibrated["search"]),
        "answer_calibration": asdict(calibrated["answer"]),
        "same_shape_numeric_error_p95": same_shape_numeric_p95,
        "old_vs_plain_backend_alignment_p95": backend_alignment_p95,
        "backend_alignment_error_used_for_delta": True,
        "anchor_count": len(rows),
        "search_anchor_count": sum(row["action_type"] == "search" for row in rows),
        "answer_anchor_count": sum(row["action_type"] == "answer" for row in rows),
        "zero_utility_shadow_count": sum(
            bool(row["is_zero_utility_shadow"]) for row in rows
        ),
        "old_vs_plain_absolute_difference": distribution(identity_differences),
        "search_whole_dependency": distribution(dependencies["search"]),
        "answer_whole_dependency": distribution(dependencies["answer"]),
        "search_positive_rho_count": sum(
            row["action_type"] == "search" and row["rho"] > 0 for row in rows
        ),
        "answer_positive_rho_count": sum(
            row["action_type"] == "answer" and row["rho"] > 0 for row in rows
        ),
        "positive_chunk_mass_count": sum(
            row["positive_chunk_mass"] > 0 for row in rows
        ),
    }
    return FrozenResponsibilityCalibration(
        search=calibrated["search"],
        answer=calibrated["answer"],
        rows=tuple(rows),
        summary=summary,
    )
