"""Frozen hard-localized think-credit allocation for Plan07 fast-track."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from verl.trainer.ppo.evisd_teacher import token_char_offsets_from_ids
from verl.trainer.ppo.urcr_pi_builders import parse_think_chunks


EPS = 1e-12
LOCALIZED_MODES = (
    "whole_old",
    "whole_fix",
    "loo_mass50",
    "random_matched",
    "q_zero",
)
TRAINING_THINK_MODES = LOCALIZED_MODES[:4]
WHOLE_FIX_SCALE = 0.21586636230106632
LOCAL_SCALE = 0.2096083311877244
POSITIVE_MASS_FRACTION = 0.5
LOCALIZER_SEED = 20260828
MAX_CHUNKS = 6
_ACTION_TAG_RE = re.compile(r"</?(?:think|search|answer)>", re.IGNORECASE)


@dataclass(frozen=True)
class SupportSelection:
    mode: str
    response_positions: tuple[int, ...]
    chunk_indices: tuple[int, ...]
    fallback_reason: str | None
    positive_mass: float
    random_identity_unavoidable: bool = False

    @property
    def token_count(self) -> int:
        return len(self.response_positions)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_indices)


def prepare_think_support(
    tokenizer,
    *,
    response_token_ids: Sequence[int],
    think_mask: Sequence[int],
    include_chunks: bool,
) -> dict[str, Any]:
    """Reconstruct content-only think support with the frozen Stage2 parser."""
    response_ids = [int(value) for value in response_token_ids]
    stored_think = [int(bool(value)) for value in think_mask]
    if len(response_ids) != len(stored_think):
        raise ValueError("think mask and response token IDs must have equal length")
    response_text, offsets = token_char_offsets_from_ids(tokenizer, response_ids)
    tag_spans = [
        (match.start(), match.end()) for match in _ACTION_TAG_RE.finditer(response_text)
    ]
    tag_mask = [
        int(
            token_end > token_start
            and any(
                token_start < span_end and token_end > span_start
                for span_start, span_end in tag_spans
            )
        )
        for token_start, token_end in offsets
    ]
    think_tag_mask = [
        int(selected and tag)
        for selected, tag in zip(stored_think, tag_mask)
    ]
    content_mask = [
        int(selected and not tag)
        for selected, tag in zip(stored_think, tag_mask)
    ]
    content_positions = [
        index for index, selected in enumerate(content_mask) if selected
    ]
    result = {
        "think_content_mask": content_mask,
        "think_tag_mask": think_tag_mask,
        "think_chunks": [],
        "localizer_prepare_fallback_reason": None,
    }
    if not content_positions:
        result["localizer_prepare_fallback_reason"] = "empty_think_content"
        return result
    if not include_chunks:
        return result
    try:
        chunks = parse_think_chunks(
            tokenizer,
            response_ids,
            content_mask,
            max_chunk_tokens=64,
            min_chunk_tokens=8,
            max_chunks=MAX_CHUNKS,
        )
    except (TypeError, ValueError, RuntimeError):
        result["localizer_prepare_fallback_reason"] = "chunk_boundary_failure"
        return result
    flattened = [
        int(position)
        for chunk in chunks
        for position in chunk["token_positions"]
    ]
    if (
        len(flattened) != len(set(flattened))
        or sorted(flattened) != content_positions
        or any(
            set(map(int, chunk["token_positions"]))
            & set(index for index, selected in enumerate(think_tag_mask) if selected)
            for chunk in chunks
        )
    ):
        result["localizer_prepare_fallback_reason"] = "chunk_boundary_failure"
        return result
    result["think_chunks"] = chunks
    return result


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def select_positive_mass(
    scores: Sequence[float], *, fraction: float = 0.5, eps: float = EPS
) -> tuple[int, ...]:
    """Select the smallest descending-score set covering positive mass."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("positive-mass fraction must be in (0, 1]")
    positive = [
        (index, float(score))
        for index, score in enumerate(scores)
        if math.isfinite(float(score)) and float(score) > eps
    ]
    positive.sort(key=lambda item: (-item[1], item[0]))
    if not positive:
        return ()
    target = fraction * sum(value for _, value in positive)
    selected: list[int] = []
    accumulated = 0.0
    for index, value in positive:
        selected.append(index)
        accumulated += value
        if accumulated + eps >= target:
            break
    return tuple(sorted(selected))


def _positions(
    chunks: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(position)
                for index in indices
                for position in chunks[int(index)]["token_positions"]
            }
        )
    )


def loo_mass50_support(
    *,
    chunks: Sequence[Mapping[str, Any]],
    loo_scores: Sequence[float],
    content_positions: Sequence[int],
    fraction: float = POSITIVE_MASS_FRACTION,
    eps: float = EPS,
) -> SupportSelection:
    """Return hard mass50 support or the frozen whole-content fallback."""
    content = tuple(sorted(map(int, content_positions)))
    if not content:
        return SupportSelection("loo_mass50", (), (), "empty_think_content", 0.0)
    if len(chunks) <= 1:
        return SupportSelection(
            "loo_mass50",
            content,
            tuple(range(len(chunks))),
            "single_chunk",
            sum(max(float(value), 0.0) for value in loo_scores),
        )
    if len(chunks) != len(loo_scores):
        return SupportSelection(
            "loo_mass50",
            content,
            tuple(range(len(chunks))),
            "score_shape_failure",
            0.0,
        )
    positive_mass = sum(
        max(float(value), 0.0)
        for value in loo_scores
        if math.isfinite(float(value))
    )
    selected = select_positive_mass(loo_scores, fraction=fraction, eps=eps)
    if not selected or positive_mass <= eps:
        return SupportSelection(
            "loo_mass50",
            content,
            tuple(range(len(chunks))),
            "no_positive_mass",
            positive_mass,
        )
    positions = _positions(chunks, selected)
    if not positions or not set(positions) <= set(content):
        return SupportSelection(
            "loo_mass50",
            content,
            tuple(range(len(chunks))),
            "invalid_chunk_boundary",
            positive_mass,
        )
    return SupportSelection(
        "loo_mass50", positions, selected, None, positive_mass
    )


def _combo_positions(
    *,
    chunks: Sequence[Mapping[str, Any]],
    combo: tuple[int, ...],
    content_positions: Sequence[int],
    target_positions: Sequence[int],
    seed_key: str,
) -> tuple[int, ...]:
    """Select exact token mass while touching every requested chunk."""
    content = tuple(sorted(map(int, content_positions)))
    target = tuple(sorted(map(int, target_positions)))
    rank = {position: index for index, position in enumerate(content)}
    denominator = max(1, len(content) - 1)
    target_norm = np.asarray([rank[value] / denominator for value in target])
    pools = {
        index: tuple(sorted(map(int, chunks[index]["token_positions"])))
        for index in combo
    }
    candidate = tuple(sorted({position for values in pools.values() for position in values}))
    if len(candidate) < len(target):
        raise ValueError("matched-random chunk combination lacks token capacity")

    rng = np.random.default_rng(int(_digest(20260828, seed_key, combo)[:16], 16))
    centers = np.asarray(
        [np.mean([rank[value] / denominator for value in pools[index]]) for index in combo]
    )
    anchor_cost = np.abs(centers[:, None] - target_norm[None, :])
    anchor_cost += rng.uniform(0.0, 1e-4, size=anchor_cost.shape)
    chunk_rows, target_columns = linear_sum_assignment(anchor_cost)
    anchors: list[int] = []
    used_targets: set[int] = set()
    target_set = set(target)
    for chunk_row, target_column in zip(chunk_rows, target_columns):
        chunk_index = combo[int(chunk_row)]
        desired = float(target_norm[int(target_column)])
        available = [value for value in pools[chunk_index] if value not in anchors]
        chosen = min(
            available,
            key=lambda value: (
                int(value in target_set),
                abs(rank[value] / denominator - desired),
                int(_digest(seed_key, combo, chunk_index, target_column, value)[:16], 16),
            ),
        )
        anchors.append(chosen)
        used_targets.add(int(target_column))

    remaining_targets = [
        index for index in range(len(target)) if index not in used_targets
    ]
    remaining_candidates = [value for value in candidate if value not in anchors]
    if remaining_targets:
        cost = np.empty(
            (len(remaining_targets), len(remaining_candidates)), dtype=np.float64
        )
        for row_index, target_index in enumerate(remaining_targets):
            for column_index, value in enumerate(remaining_candidates):
                cost[row_index, column_index] = (
                    10.0 * int(value in target_set)
                    + abs(rank[value] / denominator - target_norm[target_index])
                    + rng.uniform(0.0, 0.20)
                )
        rows, columns = linear_sum_assignment(cost)
        if len(rows) != len(remaining_targets):
            raise RuntimeError("matched-random assignment is incomplete")
        anchors.extend(remaining_candidates[int(index)] for index in columns)

    selected = tuple(sorted(anchors))
    if len(selected) != len(target) or len(set(selected)) != len(target):
        raise RuntimeError("matched-random token mass mismatch")
    touched = {
        index
        for index in combo
        if set(selected) & set(map(int, chunks[index]["token_positions"]))
    }
    if touched != set(combo):
        raise RuntimeError("matched-random chunk-count constraint failed")
    return selected


def matched_random_support(
    *,
    chunks: Sequence[Mapping[str, Any]],
    content_positions: Sequence[int],
    true_support: SupportSelection,
    seed_key: str,
) -> SupportSelection:
    """Match token/chunk mass exactly and minimize overlap/position drift.

    If the true chunk combination is the only combination with enough capacity,
    the control is explicitly marked ``identity_unavoidable``.  This preserves
    both matching constraints instead of silently changing the estimand.
    """
    if true_support.fallback_reason is not None:
        return SupportSelection(
            "random_matched",
            true_support.response_positions,
            true_support.chunk_indices,
            f"localized_{true_support.fallback_reason}",
            true_support.positive_mass,
            True,
        )
    token_count = true_support.token_count
    chunk_count = true_support.chunk_count
    content = tuple(sorted(map(int, content_positions)))
    content_rank = {position: index for index, position in enumerate(content)}
    denominator = max(1, len(content) - 1)
    true_set = set(true_support.response_positions)
    target_quantiles = np.asarray(
        sorted(content_rank[value] / denominator for value in true_set)
    )
    candidates: list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]] = []
    for combo in itertools.combinations(range(len(chunks)), chunk_count):
        capacity = sum(len(chunks[index]["token_positions"]) for index in combo)
        if capacity < token_count:
            continue
        positions = _combo_positions(
            chunks=chunks,
            combo=combo,
            content_positions=content,
            target_positions=true_support.response_positions,
            seed_key=seed_key,
        )
        quantiles = np.asarray(
            sorted(content_rank[value] / denominator for value in positions)
        )
        overlap = len(set(positions) & true_set)
        rmse = float(np.sqrt(np.mean((quantiles - target_quantiles) ** 2)))
        jitter = int(_digest(20260828, seed_key, combo, positions)[:16], 16)
        candidates.append(((overlap, rmse, jitter, combo), combo, positions))
    if not candidates:
        raise RuntimeError("no exact-mass/chunk-count random control exists")
    _, combo, positions = min(candidates, key=lambda item: item[0])
    identity = positions == true_support.response_positions
    return SupportSelection(
        "random_matched",
        positions,
        combo,
        None,
        true_support.positive_mass,
        identity,
    )


def residual_coefficients(
    *,
    response_length: int,
    total_credit: float,
    positions: Sequence[int],
    normalization: str,
    scale: float = 1.0,
) -> np.ndarray:
    """Build detached per-token residual coefficients."""
    output = np.zeros(int(response_length), dtype=np.float64)
    selected = tuple(sorted(set(map(int, positions))))
    if not selected or abs(float(total_credit)) <= EPS:
        return output
    if selected[0] < 0 or selected[-1] >= response_length:
        raise ValueError("localized support leaves the response")
    if normalization == "N0":
        denominator = float(len(selected))
    elif normalization == "N1":
        denominator = math.sqrt(float(len(selected)))
    else:
        raise ValueError(f"unsupported localized normalization: {normalization}")
    output[list(selected)] = float(scale) * float(total_credit) / denominator
    if not np.isfinite(output).all():
        raise ValueError("localized residual contains non-finite values")
    return output
