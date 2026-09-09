"""Pure method primitives for URCR-V4.

This module deliberately contains no rollout, model, Ray, or dataset-metadata
dependency.  It is the CPU reference for the frozen
``urcr_v4_unified_typed_r1_final`` method contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer


METHOD_REVISION = "urcr_v4_unified_typed_r1_final"
CONTROL_COUNT = 3
ANSWER_ALPHA = 0.25
SEARCH_TRAJECTORY_ABSOLUTE_CAP = 2.0
SEARCH_NEGATIVE_SCALE = 0.1
MAX_THINK_CHUNKS = 6
CONTROL_RNG_VERSION = "urcr_v4_sha256_counter_r1"

_FORBIDDEN_ONLINE_METADATA_FIELDS = frozenset(
    {
        "metadata",
        "metadata_json",
        "context",
        "supporting_facts",
        "supporting_fact",
        "support_hit",
        "support_hits",
        "new_fact",
        "new_facts",
        "new_doc",
        "new_docs",
        "doc_only",
        "privileged_prefix",
        "gold_evidence",
    }
)
_FORBIDDEN_ONLINE_METADATA_PREFIXES = (
    "new_fact_",
    "new_doc_",
    "new_support_",
    "supporting_fact_",
    "support_hit_",
    "doc_only_",
)


def _require_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_nonempty(value: Any) -> tuple[str, str] | None:
    original = str(value)
    normalized = normalize_answer(original)
    if not normalized:
        return None
    return original, normalized


def select_probe_aliases(
    aliases: Sequence[Any],
    *,
    primary: Any | None = None,
    max_aliases: int = 3,
) -> tuple[str, ...]:
    """Choose the fixed, normalized-deduplicated probe target set.

    The original spelling is retained for tokenization.  An explicit primary
    answer is kept first; all remaining aliases use a stable lexical order.
    """
    if max_aliases <= 0:
        raise ValueError("max_aliases must be positive")

    primary_value = _normalized_nonempty(primary) if primary is not None else None
    by_normalized: dict[str, str] = {}
    for alias in aliases:
        value = _normalized_nonempty(alias)
        if value is None:
            continue
        original, normalized = value
        current = by_normalized.get(normalized)
        if current is None or original < current:
            by_normalized[normalized] = original

    selected: list[str] = []
    primary_normalized = None
    if primary_value is not None:
        primary_original, primary_normalized = primary_value
        selected.append(primary_original)
        by_normalized.pop(primary_normalized, None)

    for _normalized, original in sorted(by_normalized.items()):
        if len(selected) >= max_aliases:
            break
        selected.append(original)
    return tuple(selected[:max_aliases])


def word_fbeta(
    prediction: Any,
    target: Any,
    *,
    beta: float = 0.5,
) -> float:
    """Evaluator-normalized word-multiset F-beta."""
    beta = _require_finite(beta, "beta")
    if beta <= 0:
        raise ValueError("beta must be positive")
    prediction_tokens = normalize_answer(str(prediction)).split()
    target_tokens = normalize_answer(str(target)).split()
    if not prediction_tokens or not target_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


def answer_quality(
    answer: Any,
    aliases: Sequence[Any],
    *,
    binary_em: int | bool,
) -> float:
    """Return q_i from the V4 answer utility definition."""
    if int(binary_em) not in (0, 1):
        raise ValueError("binary_em must be 0 or 1")
    if bool(binary_em):
        return 1.0
    return max(
        (word_fbeta(answer, alias, beta=0.5) for alias in aliases),
        default=0.0,
    )


@dataclass(frozen=True)
class AnswerTrajectory:
    trajectory_id: str
    rollout_group_id: str
    binary_em: int
    valid_terminal_answer: bool
    answer: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class AnswerResidual:
    eligible: bool
    quality: float
    residual: float
    stratum_size: int


def compute_answer_residuals(
    rows: Sequence[AnswerTrajectory],
) -> dict[str, AnswerResidual]:
    """Compute within-``rollout_group_id + binary_em`` answer residuals."""
    row_by_id: dict[str, AnswerTrajectory] = {}
    quality: dict[str, float] = {}
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)

    for row in rows:
        if not row.trajectory_id or not row.rollout_group_id:
            raise ValueError("trajectory_id and rollout_group_id must be nonempty")
        if row.trajectory_id in row_by_id:
            raise ValueError(f"duplicate trajectory_id: {row.trajectory_id}")
        if int(row.binary_em) not in (0, 1):
            raise ValueError("binary_em must be 0 or 1")
        row_by_id[row.trajectory_id] = row
        if row.valid_terminal_answer:
            if row.answer is None:
                raise ValueError("valid terminal answer must include answer text")
            value = answer_quality(row.answer, row.aliases, binary_em=row.binary_em)
            quality[row.trajectory_id] = value
            groups[(row.rollout_group_id, int(row.binary_em))].append(
                row.trajectory_id
            )

    output: dict[str, AnswerResidual] = {}
    for trajectory_id, row in row_by_id.items():
        if trajectory_id not in quality:
            output[trajectory_id] = AnswerResidual(False, 0.0, 0.0, 0)
            continue
        members = groups[(row.rollout_group_id, int(row.binary_em))]
        if len(members) < 2:
            output[trajectory_id] = AnswerResidual(
                True, quality[trajectory_id], 0.0, len(members)
            )
            continue
        mean_quality = math.fsum(quality[member] for member in members) / len(members)
        output[trajectory_id] = AnswerResidual(
            True,
            quality[trajectory_id],
            quality[trajectory_id] - mean_quality,
            len(members),
        )
    return output


@dataclass(frozen=True)
class ObservationCandidate:
    observation_id: str
    question_id: str
    data_source: str
    call_bucket: str
    serialized_observation: str
    token_length: int
    document_count: int
    document_signature: tuple[str, ...] = ()
    valid: bool = True


@dataclass(frozen=True)
class ControlSelection:
    controls: tuple[ObservationCandidate, ...]
    unique_control_count: int
    used_replacement: bool
    selected_tiers: tuple[int, ...]
    failure_reason: str | None


def _is_near_length(anchor: ObservationCandidate, candidate: ObservationCandidate) -> bool:
    if anchor.token_length <= 0 or candidate.token_length <= 0:
        return False
    ratio = candidate.token_length / anchor.token_length
    return 0.75 <= ratio <= 1.25


def _control_tier(
    anchor: ObservationCandidate,
    candidate: ObservationCandidate,
) -> int | None:
    if not candidate.valid or not candidate.serialized_observation.strip():
        return None
    if candidate.question_id == anchor.question_id:
        return None
    if candidate.document_count != anchor.document_count:
        return None
    if candidate.serialized_observation == anchor.serialized_observation:
        return None
    if (
        anchor.document_signature
        and candidate.document_signature
        and candidate.document_signature == anchor.document_signature
    ):
        return None

    near = _is_near_length(anchor, candidate)
    same_source = candidate.data_source == anchor.data_source
    same_bucket = candidate.call_bucket == anchor.call_bucket
    if same_source and same_bucket and near:
        return 0
    if same_source and near:
        return 1
    if near:
        return 2
    return 3


def control_rng_for_anchor(
    *,
    global_step: int,
    turn_id: str,
) -> np.random.Generator:
    """Return a counter-style control RNG without touching training RNG state."""
    if int(global_step) < 1 or not turn_id:
        raise ValueError("control RNG requires a positive step and stable turn ID")
    payload = (
        f"{CONTROL_RNG_VERSION}|{METHOD_REVISION}|{int(global_step)}|{turn_id}"
    ).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return np.random.default_rng(seed)


def select_observation_controls(
    anchor: ObservationCandidate,
    pool: Sequence[ObservationCandidate],
    *,
    rng: np.random.Generator,
    count: int = CONTROL_COUNT,
) -> ControlSelection:
    """Select other-question controls with a caller-owned, resumable RNG."""
    if count <= 0:
        raise ValueError("control count must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")

    by_fingerprint: dict[
        tuple[str, ...] | tuple[str, str], tuple[int, ObservationCandidate]
    ] = {}
    for candidate in pool:
        if not candidate.observation_id:
            raise ValueError("observation_id must be nonempty")
        tier = _control_tier(anchor, candidate)
        if tier is None:
            continue
        fingerprint: tuple[str, ...] | tuple[str, str]
        if candidate.document_signature:
            fingerprint = candidate.document_signature
        else:
            fingerprint = ("serialized", candidate.serialized_observation)
        current = by_fingerprint.get(fingerprint)
        if current is None or (tier, candidate.observation_id) < (
            current[0],
            current[1].observation_id,
        ):
            by_fingerprint[fingerprint] = (tier, candidate)

    tiered: dict[int, list[ObservationCandidate]] = defaultdict(list)
    for tier, candidate in by_fingerprint.values():
        tiered[tier].append(candidate)
    ordered: list[tuple[int, ObservationCandidate]] = []
    for tier in range(4):
        candidates = sorted(tiered.get(tier, ()), key=lambda item: item.observation_id)
        if candidates:
            permutation = rng.permutation(len(candidates)).tolist()
            ordered.extend((tier, candidates[index]) for index in permutation)

    if not ordered:
        return ControlSelection((), 0, False, (), "no_other_question_control")

    selected = ordered[:count]
    unique_count = len(selected)
    used_replacement = False
    if len(selected) < count:
        used_replacement = True
        source = tuple(selected)
        while len(selected) < count:
            selected.append(source[int(rng.integers(0, len(source)))])

    return ControlSelection(
        tuple(item[1] for item in selected),
        unique_count,
        used_replacement,
        tuple(item[0] for item in selected),
        None,
    )


def visible_exact_repeat(
    observation_passages: Sequence[str],
    visible_history_passages: Sequence[str],
) -> bool:
    """Return true only when every nonempty returned passage is visibly present."""
    current = [" ".join(str(value).split()) for value in observation_passages]
    current = [value for value in current if value]
    if not current:
        return False
    visible = {" ".join(str(value).split()) for value in visible_history_passages}
    visible.discard("")
    return all(value in visible for value in current)


def directed_agreement(g_incremental: float, g_control: float) -> tuple[float, bool]:
    """Combine the two V4 contrasts without turning sign conflict positive."""
    g_incremental = _require_finite(g_incremental, "g_incremental")
    g_control = _require_finite(g_control, "g_control")
    sign_conflict = g_incremental * g_control < 0
    if g_incremental > 0 and g_control > 0:
        return min(g_incremental, g_control), False
    if g_incremental < 0 and g_control < 0:
        return -min(abs(g_incremental), abs(g_control)), False
    return 0.0, sign_conflict


@dataclass(frozen=True)
class SearchUtilityResult:
    utility: float
    g_incremental: float | None
    g_control: float | None
    g_directed: float | None
    contrast_sign_conflict: bool
    scored: bool
    zero_reason: str | None


def compute_search_utility(
    *,
    real_score: float | None,
    empty_score: float | None,
    control_scores: Sequence[float],
    delta: float,
    scale: float,
    negative_scale: float = SEARCH_NEGATIVE_SCALE,
    is_valid_search: bool = True,
    observation_consumable: bool = True,
    is_visible_exact_repeat: bool = False,
) -> SearchUtilityResult:
    """Compute dynamic search utility from frozen teacher-forced scores."""
    delta = _require_finite(delta, "delta")
    scale = _require_finite(scale, "scale")
    negative_scale = _require_finite(negative_scale, "negative_scale")
    if delta < 0 or scale <= 0 or not 0 <= negative_scale <= 1:
        raise ValueError("invalid search utility scale")
    if not is_valid_search:
        return SearchUtilityResult(0.0, None, None, None, False, False, "invalid_search")
    if not observation_consumable:
        return SearchUtilityResult(
            0.0, None, None, None, False, False, "unconsumable_observation"
        )
    if real_score is None or empty_score is None or len(control_scores) != CONTROL_COUNT:
        return SearchUtilityResult(0.0, None, None, None, False, False, "score_unavailable")
    scores = [real_score, empty_score, *control_scores]
    if any(not math.isfinite(float(value)) for value in scores):
        return SearchUtilityResult(0.0, None, None, None, False, False, "score_unavailable")

    real = float(real_score)
    incremental = real - float(empty_score)
    control = real - math.fsum(float(value) for value in control_scores) / CONTROL_COUNT
    directed, conflict = directed_agreement(incremental, control)
    excess = max(abs(directed) - delta, 0.0) / scale
    if directed > delta:
        utility = math.tanh(excess)
    elif directed < -delta:
        utility = -negative_scale * math.tanh(excess)
    else:
        utility = 0.0
    zero_reason = None
    if utility == 0.0:
        zero_reason = "contrast_sign_conflict" if conflict else "dead_zone"
    if is_visible_exact_repeat and utility > 0:
        utility = 0.0
        zero_reason = "visible_exact_repeat"
    return SearchUtilityResult(
        utility,
        incremental,
        control,
        directed,
        conflict,
        True,
        zero_reason,
    )


def _quantile(values: Sequence[float], q: float, name: str) -> float:
    numeric = np.asarray([_require_finite(value, name) for value in values], dtype=np.float64)
    if numeric.size == 0:
        raise ValueError(f"{name} must not be empty")
    return float(np.quantile(numeric, q))


@dataclass(frozen=True)
class SearchUtilityCalibration:
    delta: float
    scale: float
    numeric_error_p95: float
    control_difference_p75: float
    active_excess_count: int
    limited: bool


def calibrate_search_utility(
    *,
    control_pair_absolute_differences: Sequence[float],
    numeric_absolute_differences: Sequence[float],
    directed_gaps: Sequence[float],
) -> SearchUtilityCalibration:
    control_p75 = _quantile(
        control_pair_absolute_differences, 0.75, "control differences"
    )
    if any(float(value) < 0 for value in control_pair_absolute_differences):
        raise ValueError("control differences must be nonnegative")
    if any(float(value) < 0 for value in numeric_absolute_differences):
        raise ValueError("numeric differences must be nonnegative")
    numeric_p95 = (
        _quantile(numeric_absolute_differences, 0.95, "numeric differences")
        if len(numeric_absolute_differences) > 0
        else 0.0
    )
    delta = max(3.0 * numeric_p95, float(np.clip(control_p75, 0.05, 0.5)))
    excess = [
        abs(_require_finite(value, "directed gap")) - delta
        for value in directed_gaps
        if abs(_require_finite(value, "directed gap")) > delta
    ]
    limited = len(excess) < 8
    scale = 1.0 if limited else float(np.clip(np.quantile(excess, 0.75), 0.25, 2.0))
    return SearchUtilityCalibration(
        delta, scale, numeric_p95, control_p75, len(excess), limited
    )


@dataclass(frozen=True)
class ResponsibilityCalibration:
    delta: float
    scale: float
    numeric_error_p95: float
    positive_excess_count: int
    limited: bool


@dataclass(frozen=True)
class GradientScaleCalibration:
    global_over_local: tuple[float, ...]
    lambda_0: float
    lambda_max: float


def calibrate_gradient_scale(
    *,
    global_gradient_norms: Sequence[float],
    local_gradient_norms: Sequence[float],
) -> GradientScaleCalibration:
    """Apply the frozen full-strength gradient-ratio rule from Plan section 14."""
    if len(global_gradient_norms) != len(local_gradient_norms):
        raise ValueError("global/local gradient norm counts must match")
    if len(global_gradient_norms) < 2:
        raise ValueError("gradient calibration requires two nonzero subbatches")
    ratios = []
    for global_norm, local_norm in zip(
        global_gradient_norms, local_gradient_norms, strict=True
    ):
        global_value = _require_finite(global_norm, "global gradient norm")
        local_value = _require_finite(local_norm, "local gradient norm")
        if global_value <= 0 or local_value <= 0:
            raise ValueError("gradient calibration norms must be positive")
        ratios.append(global_value / local_value)
    lambda_0 = float(
        np.clip(0.1 * np.median(np.asarray(ratios)), 1e-4, 0.2)
    )
    lambda_max = min(lambda_0, 0.3 * min(ratios))
    return GradientScaleCalibration(tuple(ratios), lambda_0, lambda_max)


def apply_base_shadow_gradient_ceiling(
    *,
    lambda_max: float,
    global_gradient_norm: float,
    local_gradient_norm: float,
) -> tuple[float, float]:
    """Only lower lambda when the Base full-strength local/global ratio exceeds .3."""
    value = _require_finite(lambda_max, "lambda_max")
    global_norm = _require_finite(global_gradient_norm, "global gradient norm")
    local_norm = _require_finite(local_gradient_norm, "local gradient norm")
    if not 0 <= value <= 0.2 or global_norm <= 0 or local_norm < 0:
        raise ValueError("invalid Base shadow gradient inputs")
    ratio = value * local_norm / global_norm
    if local_norm > 0 and ratio > 0.3:
        value = min(value, 0.3 * global_norm / local_norm)
        ratio = value * local_norm / global_norm
    return value, ratio


def calibrate_responsibility(
    *,
    action_type: str,
    numeric_absolute_differences: Sequence[float],
    whole_dependencies: Sequence[float],
) -> ResponsibilityCalibration:
    if action_type not in {"search", "answer"}:
        raise ValueError("action_type must be search or answer")
    if any(float(value) < 0 for value in numeric_absolute_differences):
        raise ValueError("numeric differences must be nonnegative")
    numeric_p95 = (
        _quantile(numeric_absolute_differences, 0.95, "numeric differences")
        if len(numeric_absolute_differences) > 0
        else 0.0
    )
    delta = max(0.01, 3.0 * numeric_p95)
    excess = [
        _require_finite(value, "whole dependency") - delta
        for value in whole_dependencies
        if _require_finite(value, "whole dependency") > delta
    ]
    limited = len(excess) < 8
    fallback = 0.6201 if action_type == "search" else 0.2
    scale = (
        fallback
        if limited
        else float(np.clip(np.median(np.asarray(excess, dtype=np.float64)), 0.1, 2.0))
    )
    return ResponsibilityCalibration(delta, scale, numeric_p95, len(excess), limited)


def responsibility_strength(
    full_score: float,
    masked_score: float,
    *,
    delta: float,
    scale: float,
) -> float:
    full_score = _require_finite(full_score, "full_score")
    masked_score = _require_finite(masked_score, "masked_score")
    delta = _require_finite(delta, "delta")
    scale = _require_finite(scale, "scale")
    if delta < 0 or scale <= 0:
        raise ValueError("invalid responsibility scale")
    excess = max((full_score - masked_score) - delta, 0.0)
    return 1.0 - math.exp(-excess / scale)


@dataclass(frozen=True)
class ChunkRouting:
    dependencies: tuple[float, ...]
    weights: tuple[float, ...]
    positive_mass: float


def soft_sparse_chunk_weights(
    full_score: float,
    masked_scores: Sequence[float],
    *,
    delta: float,
    power: float = 2.0,
) -> ChunkRouting:
    full_score = _require_finite(full_score, "full_score")
    delta = _require_finite(delta, "delta")
    power = _require_finite(power, "power")
    if delta < 0 or power <= 0:
        raise ValueError("invalid chunk routing parameters")
    if len(masked_scores) > MAX_THINK_CHUNKS:
        raise ValueError(f"at most {MAX_THINK_CHUNKS} think chunks are supported")
    dependencies = np.asarray(
        [full_score - _require_finite(value, "masked score") for value in masked_scores],
        dtype=np.float32,
    )
    mass = np.power(np.maximum(dependencies - np.float32(delta), 0.0), power).astype(
        np.float32
    )
    total = float(mass.sum(dtype=np.float32))
    if total > 0:
        weights = (mass / np.float32(total)).astype(np.float32)
    else:
        weights = np.zeros_like(mass, dtype=np.float32)
    return ChunkRouting(
        tuple(float(value) for value in dependencies),
        tuple(float(value) for value in weights),
        total,
    )


@dataclass(frozen=True)
class RoutedActionCredit:
    action_credit: float
    think_chunk_credits: tuple[float, ...]

    @property
    def absolute_mass(self) -> float:
        return abs(self.action_credit) + math.fsum(
            abs(value) for value in self.think_chunk_credits
        )


def route_action_credit(
    utility: float,
    *,
    rho: float,
    chunk_weights: Sequence[float],
    alpha: float = 1.0,
) -> RoutedActionCredit:
    utility = _require_finite(utility, "utility")
    rho = _require_finite(rho, "rho")
    alpha = _require_finite(alpha, "alpha")
    if not 0 <= rho <= 1 or alpha < 0:
        raise ValueError("invalid routed-credit parameters")
    weights = [_require_finite(value, "chunk weight") for value in chunk_weights]
    if any(value < 0 for value in weights):
        raise ValueError("chunk weights must be nonnegative")
    total_weight = math.fsum(weights)
    if weights and not (
        math.isclose(total_weight, 0.0, abs_tol=1e-7)
        or math.isclose(total_weight, 1.0, abs_tol=1e-6)
    ):
        raise ValueError("chunk weights must sum to zero or one")
    action = alpha * utility
    think = tuple(alpha * rho * weight * utility for weight in weights)
    return RoutedActionCredit(action, think)


@dataclass(frozen=True)
class CappedSearchCredits:
    credits: tuple[RoutedActionCredit, ...]
    raw_absolute_mass: float
    applied_scale: float
    capped_absolute_mass: float


def cap_search_trajectory_credits(
    credits: Sequence[RoutedActionCredit],
    *,
    cap: float = SEARCH_TRAJECTORY_ABSOLUTE_CAP,
) -> CappedSearchCredits:
    cap = _require_finite(cap, "cap")
    if cap <= 0:
        raise ValueError("cap must be positive")
    raw_mass = math.fsum(value.absolute_mass for value in credits)
    scale = min(1.0, cap / raw_mass) if raw_mass > 0 else 1.0
    capped = tuple(
        RoutedActionCredit(
            value.action_credit * scale,
            tuple(item * scale for item in value.think_chunk_credits),
        )
        for value in credits
    )
    capped_mass = math.fsum(value.absolute_mass for value in capped)
    return CappedSearchCredits(capped, raw_mass, scale, capped_mass)


def forbidden_online_metadata_paths(value: Any) -> tuple[str, ...]:
    """Find evidence-metadata fields that must not enter the V4 online path."""
    found: list[str] = []

    def visit(current: Any, path: tuple[str, ...]) -> None:
        if isinstance(current, Mapping):
            for raw_key, child in current.items():
                key = str(raw_key)
                lowered = key.lower()
                child_path = (*path, key)
                if lowered in _FORBIDDEN_ONLINE_METADATA_FIELDS or lowered.startswith(
                    _FORBIDDEN_ONLINE_METADATA_PREFIXES
                ):
                    found.append(".".join(child_path))
                visit(child, child_path)
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                visit(child, (*path, str(index)))

    visit(value, ())
    return tuple(sorted(set(found)))


def require_no_online_evidence_metadata(value: Any) -> None:
    paths = forbidden_online_metadata_paths(value)
    if paths:
        raise ValueError(
            "V4 online data contains forbidden evidence metadata: " + ", ".join(paths)
        )
