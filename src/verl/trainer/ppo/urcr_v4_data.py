"""Metadata-free online turn records and stable identities for URCR-V4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from verl.trainer.ppo.urcr_diagnostics import parse_generated_action_spans
from verl.trainer.ppo.urcr_localized import prepare_think_support
from verl.trainer.ppo.urcr_text import normalize_match_text, parse_retrieved_documents
from verl.trainer.ppo.urcr_v4_method import (
    ObservationCandidate,
    require_no_online_evidence_metadata,
    visible_exact_repeat,
)


_DOCUMENTS_BLOCK_RE = re.compile(
    r"<documents>.*?</documents>", re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True)
class V4TurnRecord:
    global_step: int
    batch_row_index: int
    is_adjustment_copy: bool
    question_id: str
    rollout_group_id: str
    trajectory_id: str
    turn_id: str
    rollout_index: int
    turn_index: int
    data_source: str
    question: str
    answer_aliases: tuple[str, ...]
    visible_context_text: str
    visible_prompt_token_ids: tuple[int, ...]
    sampled_response_token_ids: tuple[int, ...]
    action_type: str
    action_text: str
    observation_text: str
    valid_action: bool
    observation_consumable: bool
    terminal_binary_em: int
    think_content_positions: tuple[int, ...]
    action_target_positions: tuple[int, ...]
    action_content_positions: tuple[int, ...]
    action_boundary_positions: tuple[int, ...]
    action_tag_positions: tuple[int, ...]
    think_chunks: tuple[tuple[int, ...], ...]
    old_action_target_log_probs: tuple[float, ...]


def _row_value(batch, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def _answer_aliases(env_kwargs: Any, reward_model: Any) -> tuple[str, ...]:
    ground_truth = None
    if isinstance(env_kwargs, dict):
        ground_truth = env_kwargs.get("ground_truth")
    if ground_truth is None and isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth)
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if ground_truth is None:
        return ()
    if isinstance(ground_truth, (list, tuple)):
        return tuple(str(value) for value in ground_truth)
    return (str(ground_truth),)


def _positions(mask: list[int]) -> tuple[int, ...]:
    return tuple(index for index, selected in enumerate(mask) if selected)


def _question_identity(batch, index: int) -> tuple[str, str, str]:
    data_source = str(_row_value(batch, "data_source", index, "unknown"))
    extra_info = _row_value(batch, "extra_info", index, {})
    env_kwargs = _row_value(batch, "env_kwargs", index, {})
    if not isinstance(extra_info, dict) or "index" not in extra_info:
        raise ValueError("V4 requires the stable dataset row index in extra_info")
    dataset_index = int(extra_info["index"])
    split = str(extra_info.get("split", "train"))
    question = ""
    for source in (env_kwargs, extra_info):
        if isinstance(source, dict) and source.get("question"):
            question = str(source["question"])
            break
    return f"{data_source}:{split}:row:{dataset_index}", data_source, question


def _binary_terminal_outcome(value: Any) -> int:
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "V4 requires terminal_binary_em from the environment's per-trajectory won flag"
        ) from exc
    if not np.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(
            "V4 terminal_binary_em must be the raw binary evaluator outcome before penalties"
        )
    return int(reward)


def build_v4_turn_records(
    *,
    batch,
    tokenizer,
    global_step: int,
    expected_rollouts_per_group: int | None = None,
) -> list[V4TurnRecord]:
    """Project rollout rows into the V4 online allowlist.

    Dataset evidence metadata is read nowhere and copied nowhere. Stable IDs
    derive from dataset identity, outer step, and existing row order; this does
    not consume or alter dataloader, rollout, model, or NumPy RNG state.
    """
    if int(global_step) < 1:
        raise ValueError("global_step must be positive")
    if expected_rollouts_per_group is not None and expected_rollouts_per_group <= 0:
        raise ValueError("expected_rollouts_per_group must be positive")

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"].bool()
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must share shape [row, token]")
    response_width = responses.shape[1]
    prompt_width = input_ids.shape[1] - response_width
    if prompt_width <= 0 or attention_mask.shape != input_ids.shape:
        raise ValueError("invalid prompt/response tensor layout")

    identities: list[dict[str, Any]] = []
    raw_group_question: dict[str, str] = {}
    group_occurrences: dict[str, int] = defaultdict(int)
    group_id_by_raw: dict[str, str] = {}
    trajectory_index_by_group: dict[str, dict[str, int]] = defaultdict(dict)

    for index in range(len(batch)):
        question_id, data_source, question = _question_identity(batch, index)
        raw_group = str(_row_value(batch, "uid", index, ""))
        raw_trajectory = str(_row_value(batch, "traj_uid", index, ""))
        if not raw_group or not raw_trajectory:
            raise ValueError("V4 requires nonempty rollout group and trajectory IDs")
        prior_question = raw_group_question.setdefault(raw_group, question_id)
        if prior_question != question_id:
            raise ValueError("one rollout group contains multiple question identities")
        if raw_group not in group_id_by_raw:
            occurrence = group_occurrences[question_id]
            group_occurrences[question_id] += 1
            group_id_by_raw[raw_group] = (
                f"step:{int(global_step)}:{question_id}:group:{occurrence}"
            )
        group_id = group_id_by_raw[raw_group]
        group_trajectories = trajectory_index_by_group[group_id]
        if raw_trajectory not in group_trajectories:
            group_trajectories[raw_trajectory] = len(group_trajectories)
        identities.append(
            {
                "question_id": question_id,
                "data_source": data_source,
                "question": question,
                "group_id": group_id,
                "raw_trajectory": raw_trajectory,
                "rollout_index": group_trajectories[raw_trajectory],
            }
        )

    records: list[V4TurnRecord] = []
    for index, identity in enumerate(identities):
        valid_response = response_mask[index]
        response_ids = tuple(
            int(value) for value in responses[index][valid_response].detach().cpu().tolist()
        )
        prompt_mask = attention_mask[index, :prompt_width]
        prompt_ids = tuple(
            int(value)
            for value in input_ids[index, :prompt_width][prompt_mask]
            .detach()
            .cpu()
            .tolist()
        )
        parsed = parse_generated_action_spans(tokenizer, list(response_ids))
        prepared_think = prepare_think_support(
            tokenizer,
            response_token_ids=response_ids,
            think_mask=parsed["think_mask"],
            include_chunks=True,
        )
        action_type = str(parsed["action_type"])
        action_mask = (
            parsed["search_content_mask"]
            if action_type == "search"
            else parsed["answer_content_mask"]
            if action_type == "answer"
            else [0] * len(response_ids)
        )
        action_target_positions = _positions(action_mask)
        action_tag_positions = _positions(parsed["action_tag_mask"])
        action_boundary_positions = tuple(
            sorted(set(action_target_positions) & set(action_tag_positions))
        )
        action_positions = tuple(
            value
            for value in action_target_positions
            if value not in set(action_boundary_positions)
        )
        think_content_positions = _positions(
            prepared_think["think_content_mask"]
        )
        if set(action_positions) & set(think_content_positions):
            raise ValueError("V4 action content overlaps think content")
        environment_valid = bool(_row_value(batch, "is_action_valid", index, True))
        valid_action = bool(
            action_type in {"search", "answer"}
            and action_target_positions
            and environment_valid
            and not parsed["invalid_action"]
            and not parsed["empty_action"]
            and not parsed["unclosed_action"]
        )
        turn_index = int(_row_value(batch, "turn_step", index, -1))
        if turn_index < 0:
            raise ValueError("V4 requires a nonnegative turn_step")
        episode_length = int(float(_row_value(batch, "episode_lengths", index, 0)))
        observation_consumable = bool(
            action_type == "search" and turn_index + 1 < episode_length
        )
        reward_model = _row_value(batch, "reward_model", index, {})
        env_kwargs = _row_value(batch, "env_kwargs", index, {})
        aliases = _answer_aliases(env_kwargs, reward_model)
        if not aliases:
            raise ValueError("V4 gold probe requires accepted answer aliases")
        trajectory_id = (
            f"{identity['group_id']}:rollout:{identity['rollout_index']}"
        )
        turn_id = f"{trajectory_id}:turn:{turn_index}"
        old_log_probs = batch.batch.get("old_log_probs")
        old_action_target_log_probs: tuple[float, ...] = ()
        if old_log_probs is not None and action_target_positions:
            valid_old = old_log_probs[index][valid_response].detach().float().cpu()
            old_action_target_log_probs = tuple(
                float(valid_old[position]) for position in action_target_positions
            )
        record = V4TurnRecord(
            global_step=int(global_step),
            batch_row_index=index,
            is_adjustment_copy=bool(
                _row_value(batch, "urcr_is_adjustment_copy", index, False)
            ),
            question_id=identity["question_id"],
            rollout_group_id=identity["group_id"],
            trajectory_id=trajectory_id,
            turn_id=turn_id,
            rollout_index=int(identity["rollout_index"]),
            turn_index=turn_index,
            data_source=identity["data_source"],
            question=identity["question"],
            answer_aliases=aliases,
            visible_context_text=str(
                _row_value(batch, "turn_context_text", index, "")
            ),
            visible_prompt_token_ids=prompt_ids,
            sampled_response_token_ids=response_ids,
            action_type=action_type,
            action_text=str(parsed["action_text"]),
            observation_text=str(_row_value(batch, "search_feedback", index, "")),
            valid_action=valid_action,
            observation_consumable=observation_consumable,
            terminal_binary_em=_binary_terminal_outcome(
                _row_value(batch, "terminal_binary_em", index)
            ),
            think_content_positions=think_content_positions,
            action_target_positions=action_target_positions,
            action_content_positions=action_positions,
            action_boundary_positions=action_boundary_positions,
            action_tag_positions=action_tag_positions,
            think_chunks=tuple(
                tuple(int(value) for value in chunk["token_positions"])
                for chunk in prepared_think["think_chunks"]
            ),
            old_action_target_log_probs=old_action_target_log_probs,
        )
        require_no_online_evidence_metadata(record.__dict__)
        records.append(record)

    originals_by_group: dict[str, set[str]] = defaultdict(set)
    outcomes_by_trajectory: dict[str, int] = {}
    seen_original_turns: set[str] = set()
    for record in records:
        if record.is_adjustment_copy:
            continue
        if record.turn_id in seen_original_turns:
            raise ValueError(f"duplicate original turn identity: {record.turn_id}")
        seen_original_turns.add(record.turn_id)
        originals_by_group[record.rollout_group_id].add(record.trajectory_id)
        prior_outcome = outcomes_by_trajectory.setdefault(
            record.trajectory_id, record.terminal_binary_em
        )
        if prior_outcome != record.terminal_binary_em:
            raise ValueError(
                f"one trajectory contains inconsistent terminal outcomes: {record.trajectory_id}"
            )
    if expected_rollouts_per_group is not None:
        bad = {
            group: len(trajectories)
            for group, trajectories in originals_by_group.items()
            if len(trajectories) != expected_rollouts_per_group
        }
        if bad:
            raise ValueError(f"V4 rollout group cardinality mismatch: {bad}")
    return records


def original_trajectory_count(records: list[V4TurnRecord]) -> int:
    return len(
        {
            record.trajectory_id
            for record in records
            if not record.is_adjustment_copy
        }
    )


def _document_passages(serialized_observation: str) -> tuple[str, ...]:
    return tuple(
        f"{document.title}\n{document.body}".strip()
        for document in parse_retrieved_documents(serialized_observation)
    )


def visible_history_passages(visible_context_text: str) -> tuple[str, ...]:
    passages: list[str] = []
    for match in _DOCUMENTS_BLOCK_RE.finditer(str(visible_context_text)):
        passages.extend(_document_passages(match.group(0)))
    return tuple(passages)


def record_is_visible_exact_repeat(record: V4TurnRecord) -> bool:
    return visible_exact_repeat(
        _document_passages(record.observation_text),
        visible_history_passages(record.visible_context_text),
    )


def build_observation_pool(
    records: list[V4TurnRecord],
    *,
    tokenizer,
) -> tuple[ObservationCandidate, ...]:
    """Build the complete valid other-question control pool on the controller."""
    output: list[ObservationCandidate] = []
    for record in sorted(records, key=lambda value: value.batch_row_index):
        if record.action_type != "search":
            continue
        documents = parse_retrieved_documents(record.observation_text)
        valid = bool(record.valid_action and documents and record.observation_text.strip())
        if record.is_adjustment_copy:
            valid = False
        serialized = str(record.observation_text)
        token_length = len(tokenizer.encode(serialized, add_special_tokens=False))
        signature = tuple(
            sorted(
                normalize_match_text(f"{document.title}\n{document.body}")
                for document in documents
            )
        )
        output.append(
            ObservationCandidate(
                observation_id=record.turn_id,
                question_id=record.question_id,
                data_source=record.data_source,
                call_bucket=("1" if record.turn_index == 0 else "2+"),
                serialized_observation=serialized,
                token_length=token_length,
                document_count=len(documents),
                document_signature=signature,
                valid=valid,
            )
        )
    return tuple(output)
