"""CPU aggregation and credit assembly for the URCR-V4 online pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from verl.trainer.ppo.urcr_v4_data import V4TurnRecord
from verl.trainer.ppo.urcr_v4_method import (
    ANSWER_ALPHA,
    MAX_THINK_CHUNKS,
    AnswerTrajectory,
    ChunkRouting,
    RoutedActionCredit,
    SearchUtilityResult,
    cap_search_trajectory_credits,
    compute_answer_residuals,
    compute_search_utility,
    responsibility_strength,
    route_action_credit,
    soft_sparse_chunk_weights,
)
from verl.trainer.ppo.urcr_v4_requests import (
    ResponsibilityRequestBundle,
    SearchUtilityRequestBundle,
)


def interleaved_gradient_batch_layout(
    selected_row_indices: Sequence[int],
    *,
    target_row_count: int,
    micro_batch_size: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Spread real diagnostic rows across fixed-size actor microbatches."""
    selected = [int(value) for value in selected_row_indices]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("gradient rows must be nonempty and unique")
    if min(selected) < 0:
        raise ValueError("gradient row indices must be nonnegative")
    if target_row_count <= 0 or micro_batch_size <= 0:
        raise ValueError("gradient target and microbatch sizes must be positive")
    if target_row_count % micro_batch_size:
        raise ValueError("gradient target rows must divide into microbatches")
    if len(selected) > target_row_count:
        raise ValueError("gradient rows exceed one actor minibatch")
    block_count = target_row_count // micro_batch_size
    if len(selected) < block_count:
        raise ValueError("each gradient microbatch requires at least one real row")
    blocks: list[list[tuple[int, bool]]] = [[] for _ in range(block_count)]
    for offset, row_index in enumerate(selected):
        blocks[offset % block_count].append((row_index, True))
    for block_index, block in enumerate(blocks):
        padding_source = block[0][0]
        while len(block) < micro_batch_size:
            block.append((padding_source, False))
        if len(block) != micro_batch_size:
            raise RuntimeError(f"gradient microbatch {block_index} overflowed")
    layout = tuple(row for block in blocks for row, _ in block)
    original = tuple(flag for block in blocks for _, flag in block)
    if len(layout) != target_row_count or sum(original) != len(selected):
        raise RuntimeError("gradient batch layout lost real rows")
    return layout, original


def _score(scores: Mapping[str, float], request_id: str) -> float:
    if request_id not in scores:
        raise ValueError(f"missing scorer result: {request_id}")
    value = float(scores[request_id])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite scorer result: {request_id}")
    return value


@dataclass(frozen=True)
class SearchScoreAggregation:
    utility: SearchUtilityResult
    view_scores: dict[str, float]


def aggregate_search_utility_scores(
    bundle: SearchUtilityRequestBundle,
    scores: Mapping[str, float],
    *,
    delta: float,
    scale: float,
    is_visible_exact_repeat: bool,
) -> SearchScoreAggregation:
    """Average fixed alias targets into Phi for each counterfactual view."""
    by_view: dict[str, list[float]] = defaultdict(list)
    for item in bundle.items:
        by_view[item.view].append(_score(scores, item.request_id))
    expected_views = {"real", "empty", "control_0", "control_1", "control_2"}
    if set(by_view) != expected_views:
        raise ValueError("search utility scorer views are incomplete")
    alias_count = len(bundle.probe_aliases)
    if any(len(values) != alias_count for values in by_view.values()):
        raise ValueError("search utility views do not share the fixed alias set")
    view_scores = {
        view: math.fsum(values) / len(values) for view, values in by_view.items()
    }
    utility = compute_search_utility(
        real_score=view_scores["real"],
        empty_score=view_scores["empty"],
        control_scores=[view_scores[f"control_{index}"] for index in range(3)],
        delta=delta,
        scale=scale,
        is_visible_exact_repeat=is_visible_exact_repeat,
    )
    return SearchScoreAggregation(utility=utility, view_scores=view_scores)


@dataclass(frozen=True)
class ResponsibilityAggregation:
    full_score: float
    whole_masked_score: float
    whole_dependency: float
    rho: float
    chunk_routing: ChunkRouting


def aggregate_responsibility_scores(
    bundle: ResponsibilityRequestBundle,
    scores: Mapping[str, float],
    *,
    delta: float,
    scale: float,
    chunk_delta: float,
) -> ResponsibilityAggregation:
    full = float(bundle.full_score_from_old_log_probs)
    whole = _score(scores, bundle.whole_mask_item.request_id)
    rho = responsibility_strength(full, whole, delta=delta, scale=scale)
    chunk_masked = [
        _score(scores, item.request_id) for item in bundle.chunk_mask_items
    ]
    routing = soft_sparse_chunk_weights(full, chunk_masked, delta=chunk_delta)
    return ResponsibilityAggregation(
        full_score=full,
        whole_masked_score=whole,
        whole_dependency=full - whole,
        rho=rho,
        chunk_routing=routing,
    )


@dataclass(frozen=True)
class TurnLocalCredit:
    batch_row_index: int
    turn_id: str
    trajectory_id: str
    action_type: str
    action_positions: tuple[int, ...]
    action_coefficient: float
    think_chunk_positions: tuple[tuple[int, ...], ...]
    think_chunk_coefficients: tuple[float, ...]
    raw_utility: float
    responsibility: float
    search_cap_scale: float
    zero_reason: str | None

    @property
    def absolute_mass(self) -> float:
        return abs(self.action_coefficient) + math.fsum(
            abs(value) for value in self.think_chunk_coefficients
        )


@dataclass(frozen=True)
class CreditAssembly:
    rows: tuple[TurnLocalCredit, ...]
    original_trajectory_count: int
    search_raw_absolute_mass_by_trajectory: dict[str, float]
    search_capped_absolute_mass_by_trajectory: dict[str, float]
    total_absolute_mass_by_trajectory: dict[str, float]
    local_population_trajectory_ids: tuple[str, ...]


V4_LOCAL_CHANNELS = {
    "search_action": 0,
    "search_think": 1,
    "answer_action": 2,
    "answer_think": 3,
}


def _zero_credit(record: V4TurnRecord, reason: str) -> TurnLocalCredit:
    return TurnLocalCredit(
        batch_row_index=record.batch_row_index,
        turn_id=record.turn_id,
        trajectory_id=record.trajectory_id,
        action_type=record.action_type,
        action_positions=record.action_content_positions,
        action_coefficient=0.0,
        think_chunk_positions=record.think_chunks,
        think_chunk_coefficients=(0.0,) * len(record.think_chunks),
        raw_utility=0.0,
        responsibility=0.0,
        search_cap_scale=1.0,
        zero_reason=reason,
    )


def _exclude_boundary_only_action_credit(
    record: V4TurnRecord,
    routed: RoutedActionCredit,
) -> tuple[RoutedActionCredit, str | None]:
    """Keep tag-crossing sampled tokens global while preserving think routing."""
    if record.action_content_positions:
        return routed, None
    if not record.action_boundary_positions:
        raise ValueError("valid V4 action has no localizable or boundary target token")
    return RoutedActionCredit(0.0, routed.think_chunk_credits), "action_boundary_only"


def assemble_local_credits(
    records: list[V4TurnRecord],
    *,
    search_utilities: Mapping[str, SearchUtilityResult],
    responsibilities: Mapping[str, ResponsibilityAggregation],
    active_trajectory_ids: set[str] | None = None,
) -> CreditAssembly:
    """Freeze all V4 local coefficients before PPO shuffle/minibatching."""
    original_records = [record for record in records if not record.is_adjustment_copy]
    original_trajectories = {record.trajectory_id for record in original_records}
    local_population = (
        original_trajectories
        if active_trajectory_ids is None
        else set(active_trajectory_ids)
    )
    if not local_population or not local_population <= original_trajectories:
        raise ValueError("V4 local trajectory population is empty or unknown")
    answer_rows = [
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
        for record in original_records
        if record.action_type == "answer"
    ]
    answer_residuals = compute_answer_residuals(answer_rows)

    raw_by_row: dict[int, tuple[RoutedActionCredit, float, float, str | None]] = {}
    search_rows_by_trajectory: dict[str, list[V4TurnRecord]] = defaultdict(list)
    for record in original_records:
        if record.trajectory_id not in local_population:
            raw_by_row[record.batch_row_index] = (
                RoutedActionCredit(0.0, (0.0,) * len(record.think_chunks)),
                0.0,
                0.0,
                "diagnostic_trajectory_subset",
            )
            continue
        if record.action_type == "search":
            search_rows_by_trajectory[record.trajectory_id].append(record)
            utility_result = search_utilities.get(record.turn_id)
            if utility_result is None:
                if record.valid_action and record.observation_consumable:
                    raise ValueError(f"missing search utility for {record.turn_id}")
                raw_by_row[record.batch_row_index] = (
                    RoutedActionCredit(0.0, (0.0,) * len(record.think_chunks)),
                    0.0,
                    0.0,
                    "ineligible_search",
                )
                continue
            utility = float(utility_result.utility)
            responsibility = responsibilities.get(record.turn_id)
            if utility != 0.0 and record.think_chunks and responsibility is None:
                raise ValueError(f"missing responsibility for active {record.turn_id}")
            rho = responsibility.rho if responsibility is not None else 0.0
            weights = (
                responsibility.chunk_routing.weights
                if responsibility is not None
                else (0.0,) * len(record.think_chunks)
            )
            if len(weights) != len(record.think_chunks):
                raise ValueError("responsibility chunk count does not match turn chunks")
            routed = route_action_credit(utility, rho=rho, chunk_weights=weights)
            routed, boundary_reason = _exclude_boundary_only_action_credit(
                record,
                routed,
            )
            raw_by_row[record.batch_row_index] = (
                routed,
                utility,
                rho,
                utility_result.zero_reason or boundary_reason,
            )
        elif record.action_type == "answer":
            if not record.valid_action:
                raw_by_row[record.batch_row_index] = (
                    RoutedActionCredit(0.0, (0.0,) * len(record.think_chunks)),
                    0.0,
                    0.0,
                    "ineligible_answer",
                )
                continue
            residual = answer_residuals.get(record.trajectory_id)
            utility = residual.residual if residual is not None else 0.0
            responsibility = responsibilities.get(record.turn_id)
            if utility != 0.0 and record.think_chunks and responsibility is None:
                raise ValueError(f"missing responsibility for active {record.turn_id}")
            rho = responsibility.rho if responsibility is not None else 0.0
            weights = (
                responsibility.chunk_routing.weights
                if responsibility is not None
                else (0.0,) * len(record.think_chunks)
            )
            if len(weights) != len(record.think_chunks):
                raise ValueError("responsibility chunk count does not match turn chunks")
            routed = route_action_credit(
                utility,
                rho=rho,
                chunk_weights=weights,
                alpha=ANSWER_ALPHA,
            )
            routed, boundary_reason = _exclude_boundary_only_action_credit(
                record,
                routed,
            )
            raw_by_row[record.batch_row_index] = (
                routed,
                utility,
                rho,
                boundary_reason,
            )
        else:
            raw_by_row[record.batch_row_index] = (
                RoutedActionCredit(0.0, (0.0,) * len(record.think_chunks)),
                0.0,
                0.0,
                "invalid_or_other_action",
            )

    search_caps: dict[
        str, tuple[float, float, float, dict[int, RoutedActionCredit]]
    ] = {}
    for trajectory_id, trajectory_rows in search_rows_by_trajectory.items():
        ordered = sorted(trajectory_rows, key=lambda value: value.turn_index)
        raw = [raw_by_row[record.batch_row_index][0] for record in ordered]
        capped = cap_search_trajectory_credits(raw)
        search_caps[trajectory_id] = (
            capped.raw_absolute_mass,
            capped.capped_absolute_mass,
            capped.applied_scale,
            {
                record.batch_row_index: credit
                for record, credit in zip(ordered, capped.credits)
            },
        )

    output: list[TurnLocalCredit] = []
    for record in records:
        if record.is_adjustment_copy:
            output.append(_zero_credit(record, "adjustment_copy"))
            continue
        routed, utility, rho, reason = raw_by_row[record.batch_row_index]
        cap_scale = 1.0
        if record.action_type == "search" and record.trajectory_id in search_caps:
            _raw_mass, _capped_mass, cap_scale, by_row = search_caps[
                record.trajectory_id
            ]
            capped = by_row[record.batch_row_index]
            routed = capped
        output.append(
            TurnLocalCredit(
                batch_row_index=record.batch_row_index,
                turn_id=record.turn_id,
                trajectory_id=record.trajectory_id,
                action_type=record.action_type,
                action_positions=record.action_content_positions,
                action_coefficient=routed.action_credit,
                think_chunk_positions=record.think_chunks,
                think_chunk_coefficients=routed.think_chunk_credits,
                raw_utility=utility,
                responsibility=rho,
                search_cap_scale=cap_scale,
                zero_reason=reason,
            )
        )

    search_raw = {
        trajectory_id: values[0] for trajectory_id, values in search_caps.items()
    }
    search_capped = {
        trajectory_id: values[1] for trajectory_id, values in search_caps.items()
    }
    total_mass: dict[str, float] = defaultdict(float)
    for row in output:
        if row.zero_reason != "adjustment_copy":
            total_mass[row.trajectory_id] += row.absolute_mass
    if any(value > 2.5 + 1e-6 for value in total_mass.values()):
        raise RuntimeError("V4 trajectory local coefficient budget exceeded")
    return CreditAssembly(
        rows=tuple(output),
        original_trajectory_count=len(local_population),
        search_raw_absolute_mass_by_trajectory=search_raw,
        search_capped_absolute_mass_by_trajectory=search_capped,
        total_absolute_mass_by_trajectory=dict(total_mass),
        local_population_trajectory_ids=tuple(sorted(local_population)),
    )


def local_credit_tensors(
    assembly: CreditAssembly,
    *,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map frozen V4 spans back to the original padded response rows."""
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [row, token]")
    row_count, response_width = response_mask.shape
    if len(assembly.rows) != row_count:
        raise ValueError("credit rows must match response rows")
    by_index = {row.batch_row_index: row for row in assembly.rows}
    if set(by_index) != set(range(row_count)):
        raise ValueError("credit assembly must contain each batch row exactly once")

    span_count = 1 + MAX_THINK_CHUNKS
    masks = torch.zeros(
        (row_count, span_count, response_width),
        dtype=torch.bool,
        device=response_mask.device,
    )
    coefficients = torch.zeros(
        (row_count, span_count),
        dtype=torch.float32,
        device=response_mask.device,
    )
    channels = torch.full(
        (row_count, span_count),
        -1,
        dtype=torch.long,
        device=response_mask.device,
    )
    original_rows = torch.zeros(
        row_count,
        dtype=torch.bool,
        device=response_mask.device,
    )
    valid_response = response_mask.bool()
    for row_index in range(row_count):
        credit = by_index[row_index]
        original_rows[row_index] = bool(
            credit.zero_reason != "adjustment_copy"
            and credit.trajectory_id in assembly.local_population_trajectory_ids
        )
        valid_positions = torch.nonzero(
            valid_response[row_index], as_tuple=False
        ).flatten()

        def assign_span(
            slot: int,
            local_positions: tuple[int, ...],
            coefficient: float,
            channel: int,
        ) -> None:
            if not local_positions:
                return
            if min(local_positions) < 0 or max(local_positions) >= len(valid_positions):
                raise ValueError("V4 local span leaves the valid response")
            selected = valid_positions[
                torch.tensor(
                    local_positions,
                    dtype=torch.long,
                    device=valid_positions.device,
                )
            ]
            masks[row_index, slot, selected] = True
            coefficients[row_index, slot] = float(coefficient)
            channels[row_index, slot] = channel

        action_channel = (
            V4_LOCAL_CHANNELS["search_action"]
            if credit.action_type == "search"
            else V4_LOCAL_CHANNELS["answer_action"]
            if credit.action_type == "answer"
            else -1
        )
        if action_channel >= 0:
            assign_span(
                0,
                credit.action_positions,
                credit.action_coefficient,
                action_channel,
            )
        think_channel = (
            V4_LOCAL_CHANNELS["search_think"]
            if credit.action_type == "search"
            else V4_LOCAL_CHANNELS["answer_think"]
            if credit.action_type == "answer"
            else -1
        )
        if len(credit.think_chunk_positions) > span_count - 1:
            raise ValueError("V4 local routing exceeds the six-chunk limit")
        if len(credit.think_chunk_coefficients) != len(
            credit.think_chunk_positions
        ):
            raise ValueError("V4 think spans and coefficients are misaligned")
        if think_channel >= 0:
            for offset, (positions, coefficient) in enumerate(
                zip(
                    credit.think_chunk_positions,
                    credit.think_chunk_coefficients,
                ),
                start=1,
            ):
                assign_span(offset, positions, coefficient, think_channel)

    overlap = masks.sum(dim=1)
    if (overlap > 1).any():
        raise ValueError("V4 local content spans overlap")
    return {
        "urcr_v4_span_masks": masks,
        "urcr_v4_span_coefficients": coefficients,
        "urcr_v4_span_channels": channels,
        "urcr_v4_original_row_mask": original_rows,
    }
