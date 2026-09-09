"""Metadata-free CPU request construction for URCR-V4 teacher forcing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from verl.trainer.ppo.urcr_v4_data import V4TurnRecord
from verl.trainer.ppo.urcr_v4_method import (
    CONTROL_COUNT,
    ControlSelection,
    require_no_online_evidence_metadata,
    select_probe_aliases,
)
from verl.trainer.ppo.urcr_v4_scorer import (
    PROBE_TEMPLATE_VERSION,
    TeacherForcedItem,
    build_teacher_forced_item,
)


EMPTY_OBSERVATION_SLOT = "<documents></documents>"
# The newline is a structural boundary. Qwen2's tokenizer can otherwise merge
# the closing ``>`` with the first alias token, making content-only target
# alignment ambiguous.
ANSWER_TARGET_PREFIX = "<answer>\n"


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))


def _encode_prefix_and_target(
    tokenizer,
    *,
    prefix: str,
    target: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tokenize one concatenated string and split only at a true token boundary."""
    combined = prefix + target
    try:
        encoded = tokenizer(
            combined,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
        boundary = len(prefix)
        prefix_ids: list[int] = []
        target_ids: list[int] = []
        for token_id, (start, end) in zip(ids, offsets):
            if start == end == 0:
                raise ValueError("offset-free special token at target boundary")
            if end <= boundary:
                prefix_ids.append(token_id)
            elif start >= boundary:
                target_ids.append(token_id)
            else:
                raise ValueError("probe prefix/target boundary crosses one token")
        if not target_ids:
            raise ValueError("probe alias tokenizes to an empty target")
        return tuple(prefix_ids), tuple(target_ids)
    except (KeyError, TypeError, NotImplementedError):
        prefix_ids = _encode(tokenizer, prefix)
        target_ids = _encode(tokenizer, target)
        if _encode(tokenizer, combined) != (*prefix_ids, *target_ids):
            raise ValueError("tokenizer cannot prove the probe target boundary")
        if not target_ids:
            raise ValueError("probe alias tokenizes to an empty target")
        return prefix_ids, target_ids


def _ensure_observation_slot(observation: str) -> str:
    value = str(observation).strip()
    if not value:
        return EMPTY_OBSERVATION_SLOT
    if value.lower().startswith("<documents>") and value.lower().endswith(
        "</documents>"
    ):
        return value
    return f"<documents>{value}</documents>"


def _search_probe_context_ids(
    record: V4TurnRecord,
    observation: str,
    *,
    tokenizer,
) -> tuple[int, ...]:
    eos_token = str(getattr(tokenizer, "eos_token", "") or "")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not eos_token or eos_token_id is None:
        raise ValueError("native probe rendering requires tokenizer EOS identity")
    response = record.sampled_response_token_ids
    if response and int(response[-1]) == int(eos_token_id):
        next_user = "\n<|im_start|>user\n"
    else:
        next_user = f"{eos_token}\n<|im_start|>user\n"
    after_observation = f"{eos_token}\n<|im_start|>assistant\n"
    return (
        *record.visible_prompt_token_ids,
        *response,
        *_encode(tokenizer, next_user),
        *_encode(tokenizer, _ensure_observation_slot(observation)),
        *_encode(tokenizer, after_observation),
    )


@dataclass(frozen=True)
class SearchUtilityRequestBundle:
    turn_id: str
    probe_aliases: tuple[str, ...]
    control_observation_ids: tuple[str, ...]
    items: tuple[TeacherForcedItem, ...]
    control_unique_count: int = CONTROL_COUNT
    control_used_replacement: bool = False
    control_tiers: tuple[int, ...] = ()


def build_search_utility_requests(
    record: V4TurnRecord,
    controls: ControlSelection,
    *,
    tokenizer,
    actor_snapshot_id: str,
    max_total_tokens: int = 8192,
) -> SearchUtilityRequestBundle:
    """Build real/empty/three-control requests without terminal-outcome input."""
    if record.is_adjustment_copy:
        raise ValueError("adjustment copies cannot create V4 scorer requests")
    if record.action_type != "search" or not record.valid_action:
        raise ValueError("search utility requires a valid sampled search action")
    if not record.observation_consumable:
        raise ValueError("unconsumable observation has zero utility and is not scored")
    if len(controls.controls) != CONTROL_COUNT:
        raise ValueError(f"search utility requires exactly {CONTROL_COUNT} controls")
    aliases = select_probe_aliases(
        record.answer_aliases,
        primary=record.answer_aliases[0] if record.answer_aliases else None,
        max_aliases=3,
    )
    if not aliases:
        raise ValueError("search utility requires at least one nonempty probe alias")
    views = [
        ("real", record.observation_text),
        ("empty", EMPTY_OBSERVATION_SLOT),
        *[
            (f"control_{index}", control.serialized_observation)
            for index, control in enumerate(controls.controls)
        ],
    ]
    items: list[TeacherForcedItem] = []
    for view, observation in views:
        context_ids = _search_probe_context_ids(
            record, observation, tokenizer=tokenizer
        )
        for alias_index, alias in enumerate(aliases):
            prefix_ids, target_ids = _encode_prefix_and_target(
                tokenizer,
                prefix=ANSWER_TARGET_PREFIX,
                target=alias,
            )
            items.append(
                build_teacher_forced_item(
                    request_id=f"{record.turn_id}:{view}:alias:{alias_index}",
                    comparison_group_id=record.turn_id,
                    view=view,
                    actor_snapshot_id=actor_snapshot_id,
                    context_ids=context_ids,
                    target_prefix_ids=prefix_ids,
                    target_content_ids=target_ids,
                    max_total_tokens=max_total_tokens,
                    probe_version=PROBE_TEMPLATE_VERSION,
                )
            )
    bundle = SearchUtilityRequestBundle(
        turn_id=record.turn_id,
        probe_aliases=aliases,
        control_observation_ids=tuple(
            value.observation_id for value in controls.controls
        ),
        items=tuple(items),
        control_unique_count=int(controls.unique_control_count),
        control_used_replacement=bool(controls.used_replacement),
        control_tiers=tuple(int(value) for value in controls.selected_tiers),
    )
    require_no_online_evidence_metadata(bundle.__dict__)
    return bundle


@dataclass(frozen=True)
class ResponsibilityRequestBundle:
    turn_id: str
    action_type: str
    full_score_from_old_log_probs: float
    full_target_ids: tuple[int, ...]
    plain_item: TeacherForcedItem
    whole_mask_item: TeacherForcedItem
    chunk_mask_items: tuple[TeacherForcedItem, ...]


def should_score_responsibility(
    record: V4TurnRecord,
    *,
    action_utility: float,
) -> bool:
    """Select think scoring without gating either action sign or action credit."""
    utility = float(action_utility)
    if not math.isfinite(utility):
        raise ValueError("action utility must be finite")
    return bool(
        not record.is_adjustment_copy
        and record.valid_action
        and record.action_type in {"search", "answer"}
        and utility != 0.0
        and record.think_content_positions
        and record.think_chunks
    )


def _contiguous(values: Sequence[int]) -> bool:
    return bool(values) and list(values) == list(range(values[0], values[-1] + 1))


def responsibility_request_unavailable_reason(
    record: V4TurnRecord,
) -> str | None:
    """Return why an otherwise eligible sampled action cannot be scored exactly."""
    action_positions = record.action_target_positions
    if not _contiguous(action_positions):
        return "noncontiguous_action_target"
    if len(record.old_action_target_log_probs) != len(action_positions):
        return "old_action_log_probs_misaligned"
    if not any(
        position < action_positions[0]
        for position in record.think_content_positions
    ):
        return "no_preceding_think_target"
    return None


def build_responsibility_requests(
    record: V4TurnRecord,
    *,
    actor_snapshot_id: str,
    max_total_tokens: int = 8192,
) -> ResponsibilityRequestBundle:
    """Build masked same-action requests; the full score reuses old log-probs."""
    if record.is_adjustment_copy:
        raise ValueError("adjustment copies cannot create V4 scorer requests")
    if record.action_type not in {"search", "answer"} or not record.valid_action:
        raise ValueError("responsibility requires a valid sampled action")
    action_positions = record.action_target_positions
    unavailable_reason = responsibility_request_unavailable_reason(record)
    if unavailable_reason is not None:
        raise ValueError(unavailable_reason)
    think_positions = tuple(
        position
        for position in record.think_content_positions
        if position < action_positions[0]
    )
    if not think_positions:
        raise ValueError("responsibility requires a preceding think content span")
    prompt_length = len(record.visible_prompt_token_ids)
    context_ids = (
        *record.visible_prompt_token_ids,
        *record.sampled_response_token_ids[: action_positions[0]],
    )
    target_ids = tuple(
        record.sampled_response_token_ids[position] for position in action_positions
    )
    shared = {
        "comparison_group_id": record.turn_id,
        "actor_snapshot_id": actor_snapshot_id,
        "context_ids": context_ids,
        "target_prefix_ids": (),
        "target_content_ids": target_ids,
        "max_total_tokens": max_total_tokens,
    }
    plain = build_teacher_forced_item(
        request_id=f"{record.turn_id}:plain",
        view="plain",
        **shared,
    )
    whole = build_teacher_forced_item(
        request_id=f"{record.turn_id}:whole_mask",
        view="whole_mask",
        blocked_key_positions=tuple(prompt_length + value for value in think_positions),
        **shared,
    )
    chunks: list[TeacherForcedItem] = []
    for chunk_index, chunk in enumerate(record.think_chunks):
        selected = tuple(value for value in chunk if value in think_positions)
        if not selected:
            continue
        chunks.append(
            build_teacher_forced_item(
                request_id=f"{record.turn_id}:chunk_mask:{chunk_index}",
                view=f"chunk_mask_{chunk_index}",
                blocked_key_positions=tuple(
                    prompt_length + value for value in selected
                ),
                **shared,
            )
        )
    bundle = ResponsibilityRequestBundle(
        turn_id=record.turn_id,
        action_type=record.action_type,
        full_score_from_old_log_probs=math.fsum(record.old_action_target_log_probs)
        / len(record.old_action_target_log_probs),
        full_target_ids=target_ids,
        plain_item=plain,
        whole_mask_item=whole,
        chunk_mask_items=tuple(chunks),
    )
    require_no_online_evidence_metadata(bundle.__dict__)
    return bundle


def probe_template_manifest(tokenizer) -> dict[str, Any]:
    """Return the fixed renderer contract to persist beside calibration."""
    eos_token = str(getattr(tokenizer, "eos_token", "") or "")
    if not eos_token:
        raise ValueError("tokenizer has no EOS token")
    example_prefix, example_target = _encode_prefix_and_target(
        tokenizer,
        prefix=ANSWER_TARGET_PREFIX,
        target="example",
    )
    return {
        "version": PROBE_TEMPLATE_VERSION,
        "assistant_close_and_user_open": f"{eos_token}\n<|im_start|>user\n",
        "empty_observation_slot": EMPTY_OBSERVATION_SLOT,
        "user_close_and_assistant_open": f"{eos_token}\n<|im_start|>assistant\n",
        "answer_target_prefix": ANSWER_TARGET_PREFIX,
        "answer_target_prefix_ids_example": list(example_prefix),
        "answer_content_ids_example": list(example_target),
        "score_tokens": "answer_content_only",
        "first_target_predictor": "target_position_minus_one",
    }
