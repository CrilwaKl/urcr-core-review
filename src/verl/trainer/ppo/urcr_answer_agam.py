"""Gold-grounded, answer-only modulation of an existing GRPO advantage.

AGAM-Core constructs deterministic token labels on CPU, then applies one
batched tensor correction.  It does not add a separate mixed-group gate:
when a GRPO group is tied, its outcome advantage is zero and the residual
vanishes through ``abs(outcome_advantage)``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import string
from typing import Any, Mapping, Sequence

import torch

from verl.trainer.ppo.urcr_diagnostics import parse_generated_action_spans
from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


@dataclass(frozen=True)
class AnswerAgamConfig:
    enabled: bool
    lambda_start: float


@dataclass(frozen=True)
class _NormalizedWord:
    text: str
    source_positions: tuple[int, ...]


@dataclass(frozen=True)
class AnswerTokenLabels:
    eligible: bool
    abstain_reason: str | None
    z_by_local_token: tuple[int, ...]
    answer_token_count: int
    aligned_token_count: int
    unmatched_token_count: int
    neutral_token_count: int
    normalized_answer: str
    best_alias: str | None
    best_alias_token_f1: float
    evaluator_correct: bool


@dataclass(frozen=True)
class AnswerAgamApplication:
    advantages: torch.Tensor
    residual: torch.Tensor
    answer_content_mask: torch.Tensor
    token_quality: torch.Tensor
    metrics: dict[str, float]


def validate_answer_agam_config(value: Any) -> AnswerAgamConfig:
    """Accept the minimal AGAM-Core interface; lambda is the initial amplitude."""
    if value is None:
        value = {}
    if not hasattr(value, "get"):
        raise ValueError("algorithm.urcr.answer_agam must be a mapping")
    unknown = set(value.keys()) - {"enable", "lambda"}
    if unknown:
        raise ValueError(
            "algorithm.urcr.answer_agam has unsupported keys: "
            f"{sorted(unknown)}"
        )
    enabled = bool(value.get("enable", False))
    if enabled and "lambda" not in value:
        raise ValueError(
            "algorithm.urcr.answer_agam.lambda must be explicit when AGAM is enabled"
        )
    configured_lambda = float(value.get("lambda", 0.0))
    if not math.isfinite(configured_lambda) or not 0.0 <= configured_lambda < 1.0:
        raise ValueError("algorithm.urcr.answer_agam.lambda must be finite and in [0, 1)")
    return AnswerAgamConfig(
        enabled=enabled,
        lambda_start=configured_lambda if enabled else 0.0,
    )


def linear_annealed_answer_lambda(
    lambda_start: float,
    *,
    global_step: int,
    total_training_steps: int,
) -> float:
    """Linearly remove direct gold-answer modulation over optimizer updates.

    ``global_step`` is one-based for the update about to run.  The first update
    receives ``lambda_start`` and the final configured update receives exactly
    zero.  A resumed run therefore continues the same global schedule.
    """
    if not math.isfinite(lambda_start) or not 0.0 <= lambda_start < 1.0:
        raise ValueError("AGAM lambda_start must be finite and in [0, 1)")
    step = int(global_step)
    total = int(total_training_steps)
    if step < 1:
        raise ValueError("AGAM global_step must be at least 1")
    if total < 2:
        raise ValueError("AGAM linear annealing requires at least 2 training steps")
    clamped_step = min(step, total)
    return float(lambda_start) * float(total - clamped_step) / float(total - 1)


def _normalized_words_with_source_positions(text: str) -> list[_NormalizedWord]:
    """Reproduce the SearchQA evaluator normalization while retaining offsets."""
    collapsed_chars: list[str] = []
    source_positions: list[int] = []
    punctuation = set(string.punctuation)
    for source_index, character in enumerate(str(text)):
        for lowered in character.lower():
            if lowered in punctuation:
                continue
            collapsed_chars.append(lowered)
            source_positions.append(source_index)

    article_stripped = list(collapsed_chars)
    collapsed_text = "".join(collapsed_chars)
    for match in _ARTICLE_RE.finditer(collapsed_text):
        for index in range(match.start(), match.end()):
            article_stripped[index] = " "

    normalized_text = "".join(article_stripped)
    words: list[_NormalizedWord] = []
    for match in re.finditer(r"\S+", normalized_text):
        positions = tuple(
            sorted(set(source_positions[index] for index in range(match.start(), match.end())))
        )
        if positions:
            words.append(_NormalizedWord(match.group(0), positions))

    reconstructed = " ".join(word.text for word in words)
    evaluator_value = normalize_answer(str(text))
    if reconstructed != evaluator_value:
        raise ValueError(
            "AGAM normalization diverged from the active SearchQA evaluator: "
            f"{reconstructed!r} != {evaluator_value!r}"
        )
    return words


def _token_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    common = sum((Counter(left) & Counter(right)).values())
    if not common:
        return 0.0
    precision = common / len(left)
    recall = common / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_pairs(left: Sequence[str], right: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return one deterministic maximum monotonic word alignment."""
    n_left = len(left)
    n_right = len(right)
    lengths = [[0] * (n_right + 1) for _ in range(n_left + 1)]
    for left_index in range(n_left - 1, -1, -1):
        for right_index in range(n_right - 1, -1, -1):
            if left[left_index] == right[right_index]:
                lengths[left_index][right_index] = (
                    1 + lengths[left_index + 1][right_index + 1]
                )
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index + 1][right_index],
                    lengths[left_index][right_index + 1],
                )

    pairs: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < n_left and right_index < n_right:
        if (
            left[left_index] == right[right_index]
            and lengths[left_index][right_index]
            == 1 + lengths[left_index + 1][right_index + 1]
        ):
            pairs.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif lengths[left_index + 1][right_index] >= lengths[left_index][right_index + 1]:
            left_index += 1
        else:
            right_index += 1
    return tuple(pairs)


def _best_alias(
    sampled_words: Sequence[_NormalizedWord],
    aliases: Sequence[str],
) -> tuple[str, list[_NormalizedWord], tuple[tuple[int, int], ...], float] | None:
    sampled_tokens = [word.text for word in sampled_words]
    sampled_normalized = " ".join(sampled_tokens)
    candidates = []
    for raw_alias in aliases:
        alias_words = _normalized_words_with_source_positions(str(raw_alias))
        if not alias_words:
            continue
        alias_tokens = [word.text for word in alias_words]
        alias_normalized = " ".join(alias_tokens)
        pairs = _lcs_pairs(sampled_tokens, alias_tokens)
        f1 = _token_f1(sampled_tokens, alias_tokens)
        evaluator_match = int(sampled_normalized == alias_normalized)
        candidates.append(
            (
                (
                    -evaluator_match,
                    -f1,
                    -len(pairs),
                    abs(len(sampled_tokens) - len(alias_tokens)),
                    alias_normalized,
                    str(raw_alias),
                ),
                str(raw_alias),
                alias_words,
                pairs,
                f1,
            )
        )
    if not candidates:
        return None
    _, raw_alias, alias_words, pairs, f1 = min(candidates, key=lambda item: item[0])
    return raw_alias, alias_words, pairs, float(f1)


def _abstained_labels(
    response_length: int,
    answer_token_count: int,
    reason: str,
) -> AnswerTokenLabels:
    return AnswerTokenLabels(
        eligible=False,
        abstain_reason=reason,
        z_by_local_token=(0,) * response_length,
        answer_token_count=answer_token_count,
        aligned_token_count=0,
        unmatched_token_count=0,
        neutral_token_count=answer_token_count,
        normalized_answer="",
        best_alias=None,
        best_alias_token_f1=0.0,
        evaluator_correct=False,
    )


def label_answer_tokens(
    frozen_row: Mapping[str, Any],
    tokenizer,
) -> AnswerTokenLabels:
    """Label original generated-token positions without decode-retokenize indices."""
    response_token_ids = [int(value) for value in frozen_row.get("response_token_ids", [])]
    response_length = len(response_token_ids)
    answer_mask = [int(value) for value in frozen_row.get("answer_content_mask", [])]
    if len(answer_mask) != response_length:
        raise ValueError("AGAM answer mask length does not match original response IDs")
    answer_token_count = int(sum(bool(value) for value in answer_mask))
    if str(frozen_row.get("action_type", "other")) != "answer":
        return _abstained_labels(response_length, answer_token_count, "not_answer_action")
    if bool(frozen_row.get("invalid_action", False)):
        return _abstained_labels(response_length, answer_token_count, "invalid_answer_action")
    if bool(frozen_row.get("empty_action", False)) or not answer_token_count:
        return _abstained_labels(response_length, answer_token_count, "empty_answer")
    if bool(frozen_row.get("unclosed_action", False)):
        return _abstained_labels(response_length, answer_token_count, "unclosed_answer")

    aliases = [str(value) for value in frozen_row.get("ground_truth_aliases", [])]
    if not aliases:
        return _abstained_labels(response_length, answer_token_count, "missing_gold_alias")

    offsets = frozen_row.get("_token_char_offsets")
    response_text = str(frozen_row.get("response_text", ""))
    char_span = frozen_row.get("answer_content_char_span")
    if offsets is None or char_span is None:
        parsed = parse_generated_action_spans(tokenizer, response_token_ids)
        offsets = parsed["token_char_offsets"]
        char_span = parsed["answer_content_char_span"]
        if response_text and parsed["response_text"] != response_text:
            return _abstained_labels(response_length, answer_token_count, "response_decode_mismatch")
        response_text = parsed["response_text"]
    offsets = [(int(start), int(end)) for start, end in offsets]
    if len(offsets) != response_length:
        raise ValueError("AGAM token offsets do not match original response IDs")
    if char_span is None or len(char_span) != 2:
        return _abstained_labels(response_length, answer_token_count, "missing_answer_char_span")
    span_start, span_end = (int(char_span[0]), int(char_span[1]))
    if not 0 <= span_start < span_end <= len(response_text):
        return _abstained_labels(response_length, answer_token_count, "invalid_answer_char_span")

    answer_text = response_text[span_start:span_end]
    sampled_words = _normalized_words_with_source_positions(answer_text)
    if not sampled_words:
        return _abstained_labels(response_length, answer_token_count, "empty_normalized_answer")
    selected = _best_alias(sampled_words, aliases)
    if selected is None:
        return _abstained_labels(response_length, answer_token_count, "empty_normalized_aliases")
    best_alias, alias_words, pairs, best_f1 = selected
    aligned_word_indices = {left_index for left_index, _ in pairs}

    char_labels = [0] * len(answer_text)
    semantic_source_positions: set[int] = set()
    for word_index, word in enumerate(sampled_words):
        label = 1 if word_index in aligned_word_indices else -1
        for position in word.source_positions:
            if not 0 <= position < len(char_labels):
                raise ValueError("AGAM normalized word points outside answer text")
            char_labels[position] = label
            semantic_source_positions.add(position)

    local_z = [0] * response_length
    covered_source_positions: set[int] = set()
    for token_index, ((token_start, token_end), selected_by_answer_mask) in enumerate(
        zip(offsets, answer_mask)
    ):
        if not selected_by_answer_mask:
            continue
        local_start = max(token_start, span_start) - span_start
        local_end = min(token_end, span_end) - span_start
        if local_end <= local_start:
            continue
        nonzero_labels = {
            char_labels[position]
            for position in range(local_start, local_end)
            if char_labels[position]
        }
        if len(nonzero_labels) > 1:
            return _abstained_labels(
                response_length,
                answer_token_count,
                "ambiguous_cross_word_token",
            )
        if nonzero_labels:
            label = nonzero_labels.pop()
            local_z[token_index] = label
            covered_source_positions.update(
                position
                for position in range(local_start, local_end)
                if char_labels[position] == label
            )
    if covered_source_positions != semantic_source_positions:
        return _abstained_labels(
            response_length,
            answer_token_count,
            "semantic_token_mapping_incomplete",
        )

    aligned_count = sum(value == 1 for value in local_z)
    unmatched_count = sum(value == -1 for value in local_z)
    neutral_count = answer_token_count - aligned_count - unmatched_count
    if neutral_count < 0:
        raise RuntimeError("AGAM token labels are not mutually exclusive")
    normalized_sample = " ".join(word.text for word in sampled_words)
    normalized_alias = " ".join(word.text for word in alias_words)
    return AnswerTokenLabels(
        eligible=True,
        abstain_reason=None,
        z_by_local_token=tuple(local_z),
        answer_token_count=answer_token_count,
        aligned_token_count=aligned_count,
        unmatched_token_count=unmatched_count,
        neutral_token_count=neutral_count,
        normalized_answer=normalized_sample,
        best_alias=best_alias,
        best_alias_token_f1=best_f1,
        evaluator_correct=normalized_sample == normalized_alias,
    )


def apply_answer_agam(
    advantages: torch.Tensor,
    outcome_anchor: torch.Tensor,
    response_mask: torch.Tensor,
    frozen_rows: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    lambda_effective: float,
    atol: float = 1e-6,
) -> AnswerAgamApplication:
    """Apply AGAM-Core with CPU label construction and one batched GPU op."""
    if advantages.shape != outcome_anchor.shape or advantages.shape != response_mask.shape:
        raise ValueError("AGAM advantages, outcome anchor, and response mask must match")
    if len(frozen_rows) != len(advantages):
        raise ValueError("AGAM frozen-row count must equal actor batch size")
    if not math.isfinite(lambda_effective) or not 0.0 <= lambda_effective < 1.0:
        raise ValueError("AGAM effective lambda must be finite and in [0, 1)")

    response_mask_cpu = response_mask.detach().cpu().bool()
    z_cpu = torch.zeros(response_mask.shape, dtype=torch.int8)
    answer_mask_cpu = torch.zeros(response_mask.shape, dtype=torch.bool)
    counters: defaultdict[str, int] = defaultdict(int)
    token_f1_values: list[float] = []

    for row_index, frozen_row in enumerate(frozen_rows):
        valid_positions = torch.nonzero(
            response_mask_cpu[row_index], as_tuple=False
        ).flatten()
        response_token_ids = list(frozen_row.get("response_token_ids", []))
        if len(valid_positions) != len(response_token_ids):
            raise ValueError(
                "AGAM frozen response length does not match valid response-mask positions"
            )
        local_answer_mask = [
            bool(value) for value in frozen_row.get("answer_content_mask", [])
        ]
        if len(local_answer_mask) != len(response_token_ids):
            raise ValueError("AGAM frozen answer mask has the wrong length")
        answer_local_positions = [
            index for index, selected in enumerate(local_answer_mask) if selected
        ]
        if answer_local_positions:
            answer_mask_cpu[row_index, valid_positions[answer_local_positions]] = True

        labels = label_answer_tokens(frozen_row, tokenizer)
        if str(frozen_row.get("action_type", "other")) == "answer":
            counters["answer_row_count"] += 1
        counters["answer_content_token_count"] += labels.answer_token_count
        if not labels.eligible:
            if str(frozen_row.get("action_type", "other")) == "answer":
                counters["abstained_answer_row_count"] += 1
                counters[f"abstain_{labels.abstain_reason}_count"] += 1
            continue

        counters["labelled_answer_row_count"] += 1
        counters["aligned_token_count"] += labels.aligned_token_count
        counters["unmatched_token_count"] += labels.unmatched_token_count
        counters["neutral_token_count"] += labels.neutral_token_count
        counters["evaluator_correct_answer_row_count"] += int(labels.evaluator_correct)
        token_f1_values.append(labels.best_alias_token_f1)
        local_z = torch.tensor(labels.z_by_local_token, dtype=torch.int8)
        z_cpu[row_index, valid_positions] = local_z

    device = advantages.device
    dtype = advantages.dtype
    answer_content_mask = answer_mask_cpu.to(device=device, dtype=dtype)
    token_quality = z_cpu.to(device=device, dtype=dtype)
    if not torch.isfinite(advantages).all() or not torch.isfinite(outcome_anchor).all():
        raise RuntimeError("AGAM received non-finite input advantages")
    detached_advantages = advantages.detach().clone()
    detached_anchor = outcome_anchor.detach().clone()
    anchor_error = (detached_advantages - detached_anchor).abs() * answer_content_mask
    anchor_error_max = float(anchor_error.max().item()) if anchor_error.numel() else 0.0
    if anchor_error_max > atol:
        raise RuntimeError(
            "AGAM answer content no longer carries the unmodified GRPO anchor: "
            f"max_error={anchor_error_max}"
        )

    residual = (
        float(lambda_effective)
        * detached_anchor.abs()
        * token_quality
        * answer_content_mask
    )
    output = detached_advantages + residual
    if not torch.isfinite(output).all():
        raise RuntimeError("AGAM produced non-finite advantages")
    outside_error = (residual * (1.0 - answer_content_mask)).abs()
    outside_error_max = float(outside_error.max().item()) if outside_error.numel() else 0.0
    if outside_error_max != 0.0:
        raise RuntimeError("AGAM residual escaped the canonical answer content mask")

    semantic_active = token_quality.ne(0) & detached_anchor.ne(0)
    if semantic_active.any():
        active_output = output[semantic_active]
        active_anchor = detached_anchor[semantic_active]
        sign_flip_count = int(
            torch.count_nonzero(torch.sign(active_output) != torch.sign(active_anchor)).item()
        )
        multipliers = active_output / active_anchor
        multiplier_min = float(multipliers.min().item())
        multiplier_max = float(multipliers.max().item())
    else:
        sign_flip_count = 0
        multiplier_min = 1.0
        multiplier_max = 1.0
    if sign_flip_count:
        raise RuntimeError("AGAM changed the sign of an outcome advantage")
    dtype_tolerance = (
        4.0 * torch.finfo(dtype).eps if dtype.is_floating_point else atol
    )
    bound_tolerance = max(atol, dtype_tolerance)
    if (
        multiplier_min < 1.0 - lambda_effective - bound_tolerance
        or multiplier_max > 1.0 + lambda_effective + bound_tolerance
    ):
        raise RuntimeError("AGAM multiplier escaped its configured bound")

    correction_active_rows = int(torch.count_nonzero(residual.abs().sum(dim=-1)).item())
    counters["correction_active_row_count"] = correction_active_rows
    answer_token_count = max(1, counters["answer_content_token_count"])
    active_token_count = int(torch.count_nonzero(residual).item())
    answer_mask_sum = answer_content_mask.sum().clamp(min=1)
    metrics = {
        "agam/answer_row_count": float(counters["answer_row_count"]),
        "agam/labelled_answer_row_count": float(counters["labelled_answer_row_count"]),
        "agam/abstained_answer_row_count": float(counters["abstained_answer_row_count"]),
        "agam/correction_active_row_count": float(correction_active_rows),
        "agam/answer_content_token_count": float(counters["answer_content_token_count"]),
        "agam/aligned_token_count": float(counters["aligned_token_count"]),
        "agam/unmatched_token_count": float(counters["unmatched_token_count"]),
        "agam/neutral_token_count": float(counters["neutral_token_count"]),
        "agam/evaluator_correct_answer_row_count": float(
            counters["evaluator_correct_answer_row_count"]
        ),
        "agam/best_alias_token_f1_mean": float(
            sum(token_f1_values) / len(token_f1_values) if token_f1_values else 0.0
        ),
        "agam/correction_active_token_count": float(active_token_count),
        "agam/correction_active_token_rate": float(active_token_count / answer_token_count),
        "agam/residual_abs_mean_answer": float(
            (residual.abs() * answer_content_mask).sum().item() / answer_mask_sum.item()
        ),
        "agam/residual_abs_max": float(residual.abs().max().item()),
        "agam/multiplier_min": multiplier_min,
        "agam/multiplier_max": multiplier_max,
        "agam/sign_flip_count": float(sign_flip_count),
        "agam/base_answer_anchor_max_error": anchor_error_max,
        "agam/residual_outside_answer_max_abs": outside_error_max,
    }
    for key, value in counters.items():
        if key.startswith("abstain_"):
            metrics[f"agam/{key}"] = float(value)
    return AnswerAgamApplication(
        advantages=output,
        residual=residual,
        answer_content_mask=answer_content_mask,
        token_quality=token_quality,
        metrics=metrics,
    )
