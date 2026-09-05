"""Frozen-turn export for Plan 02 utility-responsibility diagnostics.

This module is deliberately limited to serialization and token-aligned span
parsing. It does not generate rollouts, score new objectives, or update model
parameters; those remain in the existing EviSD execution path.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from verl.trainer.ppo.evisd_teacher import token_char_offsets_from_ids


_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_OPEN_RE = re.compile(r"<search>", re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _decode(tokenizer, token_ids: Iterable[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _mask_for_char_spans(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[int]:
    mask = [0] * len(offsets)
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in spans):
            mask[token_index] = 1
    return mask


def _span_from_mask(mask: list[int]) -> list[int] | None:
    positions = [index for index, selected in enumerate(mask) if selected]
    return [positions[0], positions[-1] + 1] if positions else None


def parse_generated_action_spans(tokenizer, token_ids: list[int]) -> dict[str, Any]:
    """Parse the first executable action using the Search environment's rules."""
    response_text, offsets = token_char_offsets_from_ids(tokenizer, token_ids)
    search_count = len(_SEARCH_OPEN_RE.findall(response_text))
    answer_count = len(_ANSWER_OPEN_RE.findall(response_text))

    # Match search_projection._postprocess_action: a search close tag takes
    # precedence; otherwise an answer close tag truncates trailing text.
    if re.search(r"</search>", response_text, re.IGNORECASE):
        close = re.search(r"</search>", response_text, re.IGNORECASE)
        trimmed_text = response_text[: close.end()]
    elif re.search(r"</answer>", response_text, re.IGNORECASE):
        close = re.search(r"</answer>", response_text, re.IGNORECASE)
        trimmed_text = response_text[: close.end()]
    else:
        trimmed_text = response_text

    match = _SEARCH_RE.search(trimmed_text)
    action_type = "search" if match else "other"
    if match is None:
        match = _ANSWER_RE.search(trimmed_text)
        action_type = "answer" if match else "other"

    invalid_action = (
        action_type == "other"
        or (search_count > 0 and answer_count > 0)
        or search_count > 1
        or answer_count > 1
    )
    unclosed_action = match is None and bool(search_count or answer_count)

    think_spans: list[tuple[int, int]] = []
    search_spans: list[tuple[int, int]] = []
    answer_spans: list[tuple[int, int]] = []
    tag_spans: list[tuple[int, int]] = []
    action_text = ""
    search_content_char_span = None
    answer_content_char_span = None
    if match is not None:
        think_start, think_end = _trim_span(response_text, 0, match.start())
        if think_start < think_end:
            think_spans.append((think_start, think_end))
        content_start, content_end = _trim_span(
            response_text,
            match.start(1),
            match.end(1),
        )
        if content_start < content_end:
            target = search_spans if action_type == "search" else answer_spans
            target.append((content_start, content_end))
            action_text = response_text[content_start:content_end]
            if action_type == "search":
                search_content_char_span = [content_start, content_end]
            else:
                answer_content_char_span = [content_start, content_end]
        tag_spans.extend(
            [
                (match.start(), match.start(1)),
                (match.end(1), match.end()),
            ]
        )

    think_mask = _mask_for_char_spans(offsets, think_spans)
    search_content_mask = _mask_for_char_spans(offsets, search_spans)
    answer_content_mask = _mask_for_char_spans(offsets, answer_spans)
    action_tag_mask = _mask_for_char_spans(offsets, tag_spans)
    return {
        "response_text": response_text,
        "action_type": action_type,
        "action_text": action_text,
        "invalid_action": bool(invalid_action),
        "empty_action": bool(match is not None and not action_text),
        "unclosed_action": bool(unclosed_action),
        "think_mask": think_mask,
        "search_content_mask": search_content_mask,
        "answer_content_mask": answer_content_mask,
        "action_tag_mask": action_tag_mask,
        "think_span": _span_from_mask(think_mask),
        "search_content_span": _span_from_mask(search_content_mask),
        "answer_content_span": _span_from_mask(answer_content_mask),
        "search_content_char_span": search_content_char_span,
        "answer_content_char_span": answer_content_char_span,
        # Kept only when explicitly requested by an online consumer.  Reusing
        # the offsets avoids decoding every response a second time.
        "token_char_offsets": offsets,
    }


def _row_value(batch, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def _tensor_values(tensor: torch.Tensor | None, index: int, mask: torch.Tensor) -> list[float]:
    if tensor is None:
        return []
    return tensor[index][mask].detach().float().cpu().tolist()


def _ground_truth_aliases(env_kwargs: Any, reward_model: Any) -> list[str]:
    ground_truth = None
    if isinstance(env_kwargs, dict):
        ground_truth = env_kwargs.get("ground_truth")
    if ground_truth is None and isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth)
    if ground_truth is None:
        return []
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, (list, tuple)):
        return [str(item) for item in ground_truth]
    return [str(ground_truth)]


def build_frozen_turn_rows(
    *,
    batch,
    tokenizer,
    teacher_info: list[dict[str, Any]],
    evisd_intermediates: dict[str, torch.Tensor],
    modulation_mask: torch.Tensor,
    global_step: int,
    include_search_modulation: bool = True,
    include_answer_modulation: bool = True,
    include_token_char_offsets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build token-aligned frozen rows shared by offline and online URCR paths."""
    frozen_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    response_width = responses.shape[1]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"].bool()
    prompt_width = input_ids.shape[1] - response_width
    prompts = input_ids[:, :prompt_width]
    prompt_masks = attention_mask[:, :prompt_width]

    for index in range(len(batch)):
        valid_response = response_mask[index]
        response_ids = responses[index][valid_response].detach().cpu().tolist()
        prompt_ids = prompts[index][prompt_masks[index]].detach().cpu().tolist()
        parsed = parse_generated_action_spans(tokenizer, response_ids)
        info = teacher_info[index] if index < len(teacher_info) else {}
        env_kwargs = _row_value(batch, "env_kwargs", index, {})
        reward_model = _row_value(batch, "reward_model", index, {})
        extra_info = _row_value(batch, "extra_info", index, {})
        metadata = _row_value(batch, "metadata", index)
        aliases = _ground_truth_aliases(env_kwargs, reward_model)
        question = ""
        for source in (env_kwargs, extra_info):
            if isinstance(source, dict) and source.get("question"):
                question = str(source["question"])
                break

        student_lp = _tensor_values(batch.batch.get("old_log_probs"), index, valid_response)
        teacher_lp = _tensor_values(batch.batch.get("teacher_log_probs"), index, valid_response)
        delta = _tensor_values(evisd_intermediates.get("delta"), index, valid_response)
        score = _tensor_values(evisd_intermediates.get("evisd_score"), index, valid_response)
        bonus = _tensor_values(evisd_intermediates.get("evisd_bonus"), index, valid_response)
        advantage = _tensor_values(evisd_intermediates.get("advantages_before"), index, valid_response)
        official_action_mask = _tensor_values(modulation_mask, index, valid_response)
        diagnostic_action_mask = [
            int(
                (include_search_modulation and search)
                or (include_answer_modulation and answer)
            )
            for search, answer in zip(
                parsed["search_content_mask"],
                parsed["answer_content_mask"],
            )
        ]
        mask_match = official_action_mask == [float(value) for value in diagnostic_action_mask]

        data_source = str(_row_value(batch, "data_source", index, "unknown"))
        uid = str(_row_value(batch, "uid", index, ""))
        traj_uid = str(_row_value(batch, "traj_uid", index, ""))
        turn_step = int(_row_value(batch, "turn_step", index, -1))
        episode_reward = float(_row_value(batch, "episode_rewards", index, 0.0))
        privileged_prefix = str(info.get("privileged_prefix", ""))
        teacher_scored = bool(info.get("teacher_scored", True))
        pi_available = bool(
            parsed["action_type"] == "search"
            and privileged_prefix
            and teacher_scored
        )
        dataset_index = int(extra_info.get("index", -1)) if isinstance(extra_info, dict) else -1
        question_uid = (
            f"{data_source}:{dataset_index}"
            if dataset_index >= 0
            else f"{data_source}:uid:{uid}"
        )
        row = {
            "global_step": int(global_step),
            "batch_row_index": int(index),
            "is_adjustment_copy": bool(
                _row_value(batch, "urcr_is_adjustment_copy", index, False)
            ),
            "question_uid": question_uid,
            "uid": uid,
            "traj_uid": traj_uid,
            "turn_uid": f"{traj_uid}:{turn_step}",
            "turn_step": turn_step,
            "dataset_index": dataset_index,
            "data_source": data_source,
            "question": question,
            "ground_truth_aliases": aliases,
            "metadata_json": _json_text(metadata),
            "env_kwargs_json": _json_text(env_kwargs),
            "prompt_token_ids": [int(value) for value in prompt_ids],
            "response_token_ids": [int(value) for value in response_ids],
            "prompt_text": _decode(tokenizer, prompt_ids),
            "turn_context_text": str(_row_value(batch, "turn_context_text", index, "")),
            "response_text": parsed["response_text"],
            "action_type": parsed["action_type"],
            "query_text": parsed["action_text"] if parsed["action_type"] == "search" else "",
            "observation_text": str(_row_value(batch, "search_feedback", index, "")),
            "invalid_action": parsed["invalid_action"],
            "empty_action": parsed["empty_action"],
            "unclosed_action": parsed["unclosed_action"],
            "think_mask": parsed["think_mask"],
            "search_content_mask": parsed["search_content_mask"],
            "answer_content_mask": parsed["answer_content_mask"],
            "action_tag_mask": parsed["action_tag_mask"],
            "think_span": parsed["think_span"],
            "search_content_span": parsed["search_content_span"],
            "answer_content_span": parsed["answer_content_span"],
            "search_content_char_span": parsed["search_content_char_span"],
            "answer_content_char_span": parsed["answer_content_char_span"],
            "episode_reward": episode_reward,
            "episode_length": float(_row_value(batch, "episode_lengths", index, 0.0)),
            "old_student_log_probs": student_lp,
            "teacher_log_probs": teacher_lp,
            "evisd_delta_token": delta,
            "evisd_score_token": score,
            "evisd_bonus_token": bonus,
            "outcome_advantage_token": advantage,
            "evisd_action_mask": official_action_mask,
            "token_action_mask_matches_evisd": bool(mask_match),
            "privileged_prefix": privileged_prefix,
            "teacher_scored": teacher_scored,
            "pi_available": pi_available,
        }
        if include_token_char_offsets:
            row["_token_char_offsets"] = parsed["token_char_offsets"]
        frozen_rows.append(row)

        search_positions = [
            position for position, selected in enumerate(parsed["search_content_mask"]) if selected
        ]
        answer_positions = [
            position for position, selected in enumerate(parsed["answer_content_mask"]) if selected
        ]
        if search_positions:
            delta_q = [delta[position] for position in search_positions]
            bonus_q = [bonus[position] for position in search_positions]
            score_q = [score[position] for position in search_positions]
            mean_bonus = float(np.mean(bonus_q))
            query_rows.append(
                {
                    "uid": uid,
                    "traj_uid": traj_uid,
                    "turn_step": turn_step,
                    "dataset_index": row["dataset_index"],
                    "data_source": data_source,
                    "episode_reward": episode_reward,
                    "pi_available": pi_available,
                    "query_text": row["query_text"],
                    "observation_text": row["observation_text"],
                    "delta_q_token": delta_q,
                    "search_score_token": score_q,
                    "search_bonus_token": bonus_q,
                    "mean_delta_q": float(np.mean(delta_q)),
                    "mean_search_bonus": mean_bonus,
                    "abs_search_bonus": float(np.mean(np.abs(bonus_q))),
                    "bonus_sign": int(np.sign(mean_bonus)),
                    "mean_outcome_advantage": float(np.mean(advantage)) if advantage else 0.0,
                    "answer_bonus_token": [bonus[position] for position in answer_positions],
                }
            )

    return frozen_rows, query_rows


def write_frozen_turn_batch(
    *,
    batch,
    tokenizer,
    teacher_info: list[dict[str, Any]],
    evisd_intermediates: dict[str, torch.Tensor],
    modulation_mask: torch.Tensor,
    output_dir: str | Path,
    global_step: int,
    artifact_prefix: str = "",
) -> tuple[Path, Path]:
    """Write one no-update diagnostic batch as compressed Parquet parts."""
    output_dir = Path(output_dir)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    frozen_rows, query_rows = build_frozen_turn_rows(
        batch=batch,
        tokenizer=tokenizer,
        teacher_info=teacher_info,
        evisd_intermediates=evisd_intermediates,
        modulation_mask=modulation_mask,
        global_step=global_step,
    )

    frozen_path = parts_dir / f"{artifact_prefix}frozen_turns_step_{global_step:06d}.parquet"
    query_path = parts_dir / f"{artifact_prefix}evisd_query_scores_step_{global_step:06d}.parquet"
    pq.write_table(pa.Table.from_pylist(frozen_rows), frozen_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(query_rows), query_path, compression="zstd")
    return frozen_path, query_path


def _consolidate_parts(paths: list[Path], output_path: Path, key_fields: tuple[str, ...]) -> int:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            key = tuple(row[field] for field in key_fields)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No diagnostic rows found for {output_path}")
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    return len(rows)


def consolidate_frozen_turn_parts(
    output_dir: str | Path,
    artifact_prefix: str = "",
) -> dict[str, int]:
    """Create the two canonical Plan 02 files and remove padding duplicates."""
    output_dir = Path(output_dir)
    parts_dir = output_dir / "parts"
    frozen_count = _consolidate_parts(
        sorted(parts_dir.glob(f"{artifact_prefix}frozen_turns_step_*.parquet")),
        output_dir / f"{artifact_prefix}frozen_turns.parquet",
        ("traj_uid", "turn_step"),
    )
    query_count = _consolidate_parts(
        sorted(parts_dir.glob(f"{artifact_prefix}evisd_query_scores_step_*.parquet")),
        output_dir / f"{artifact_prefix}evisd_query_scores.parquet",
        ("traj_uid", "turn_step"),
    )
    summary = {"frozen_turn_count": frozen_count, "search_turn_count": query_count}
    (output_dir / f"{artifact_prefix}collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
