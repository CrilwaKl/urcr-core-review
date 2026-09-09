#!/usr/bin/env python3
"""Run the smallest real-Qwen GPU exactness check for the URCR-V4 scorer."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.urcr_v4_scorer import (
    TeacherForcedItem,
    build_teacher_forced_item,
    pad_teacher_forced_batch,
    score_padded_teacher_forced_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _item(tokenizer, identity: str, question: str) -> TeacherForcedItem:
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    think_open = _encode(tokenizer, "<think>\n")
    think_content = _encode(tokenizer, "Paris is supported by the visible evidence.")
    action_prefix = _encode(tokenizer, "\n</think>\n<answer>\n")
    target = _encode(tokenizer, "Paris")
    context = (*chat, *think_open, *think_content, *action_prefix)
    blocked_start = len(chat) + len(think_open)
    blocked = tuple(range(blocked_start, blocked_start + len(think_content)))
    return build_teacher_forced_item(
        request_id=identity,
        view="whole_mask",
        actor_snapshot_id="base_grpo_s150",
        context_ids=context,
        target_prefix_ids=(),
        target_content_ids=target,
        blocked_key_positions=blocked,
        max_total_tokens=8192,
    )


def _score(model, tokenizer, items: list[TeacherForcedItem]) -> torch.Tensor:
    batch = pad_teacher_forced_batch(items, pad_token_id=tokenizer.pad_token_id)
    device = next(model.parameters()).device
    batch = replace(
        batch,
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        position_ids=batch.position_ids.to(device),
        target_ids=batch.target_ids.to(device),
        target_mask=batch.target_mask.to(device),
        predictor_positions=batch.predictor_positions.to(device),
        blocked_key_mask=batch.blocked_key_mask.to(device),
    )
    return score_padded_teacher_forced_batch(
        model,
        batch,
        temperature=1.0,
        autocast_dtype=torch.bfloat16,
    ).cpu()


def _score_queue(
    model,
    tokenizer,
    items: list[TeacherForcedItem],
    *,
    micro_batch_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            _score(model, tokenizer, items[start : start + micro_batch_size])
            for start in range(0, len(items), micro_batch_size)
        ]
    )


def _mutate_prefix(item: TeacherForcedItem, *, delta: int, vocab_size: int):
    values = list(item.input_ids)
    for position in range(item.target_positions[0]):
        values[position] = (values[position] + int(delta)) % int(vocab_size)
    return replace(item, input_ids=tuple(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer-micro-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactness check requires exactly one visible CUDA GPU")
    if args.scorer_micro_batch_size not in (1, 2):
        raise ValueError("this two-item check supports scorer micro-batch size 1 or 2")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()

    masked = [
        _item(tokenizer, "short", "What is the capital of France?"),
        _item(
            tokenizer,
            "long",
            "Using the visible evidence only, identify the capital city of France.",
        ),
    ]
    plain = [replace(item, blocked_key_positions=(), view="plain") for item in masked]
    plain_single = torch.cat([_score(model, tokenizer, [item]) for item in plain])
    plain_batch_two = _score(model, tokenizer, plain)
    masked_single = torch.cat([_score(model, tokenizer, [item]) for item in masked])
    masked_batch_two = _score(model, tokenizer, masked)
    plain_configured = _score_queue(
        model,
        tokenizer,
        plain,
        micro_batch_size=args.scorer_micro_batch_size,
    )
    masked_configured = _score_queue(
        model,
        tokenizer,
        masked,
        micro_batch_size=args.scorer_micro_batch_size,
    )

    other_prefix_mutation = _mutate_prefix(
        plain[1], delta=31, vocab_size=model.config.vocab_size
    )
    plain_cross_row_mutation = abs(
        float(_score(model, tokenizer, plain)[0])
        - float(_score(model, tokenizer, [plain[0], other_prefix_mutation])[0])
    )
    masked_other_prefix_mutation = replace(
        _mutate_prefix(masked[1], delta=31, vocab_size=model.config.vocab_size),
        blocked_key_positions=masked[1].blocked_key_positions,
    )
    masked_cross_row_mutation = abs(
        float(_score(model, tokenizer, masked)[0])
        - float(
            _score(model, tokenizer, [masked[0], masked_other_prefix_mutation])[0]
        )
    )

    mutated_ids = list(masked[0].input_ids)
    for position in masked[0].blocked_key_positions:
        mutated_ids[position] = (mutated_ids[position] + 17) % model.config.vocab_size
    mutated_masked = replace(
        masked[0],
        request_id="short_mutated_masked",
        input_ids=tuple(mutated_ids),
    )
    mutated_plain = replace(
        mutated_masked,
        request_id="short_mutated_plain",
        view="plain",
        blocked_key_positions=(),
    )
    masked_mutation = abs(
        float(_score(model, tokenizer, [masked[0]])[0])
        - float(_score(model, tokenizer, [mutated_masked])[0])
    )
    plain_mutation = abs(
        float(_score(model, tokenizer, [plain[0]])[0])
        - float(_score(model, tokenizer, [mutated_plain])[0])
    )
    torch.cuda.synchronize()

    plain_batch_two_error = float((plain_single - plain_batch_two).abs().max())
    masked_batch_two_error = float((masked_single - masked_batch_two).abs().max())
    plain_configured_error = float((plain_single - plain_configured).abs().max())
    masked_configured_error = float((masked_single - masked_configured).abs().max())
    batch_two_target_met = bool(
        plain_batch_two_error <= 0.03 and masked_batch_two_error <= 0.03
    )
    passed = bool(
        plain_configured_error <= 0.03
        and masked_configured_error <= 0.03
        and plain_cross_row_mutation <= 1e-7
        and masked_cross_row_mutation <= 1e-7
        and masked_mutation <= 0.03
        and plain_mutation > 1e-7
    )
    result = {
        "schema_version": 1,
        "artifact": "urcr_v4_s150_gpu_scorer_exactness",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "model": str(args.model.resolve()),
        "model_config_sha256": _sha256(args.model / "config.json"),
        "tokenizer_json_sha256": _sha256(args.model / "tokenizer.json"),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "temperature": 1.0,
        "diagnostic_only": True,
        "formal_scorer_micro_batch_size_per_gpu": None,
        "diagnostic_local_scorer_micro_batch_size": args.scorer_micro_batch_size,
        "configured_plain_vs_single_max_abs": plain_configured_error,
        "configured_masked_vs_single_max_abs": masked_configured_error,
        "batch_size_two_plain_vs_single_max_abs": plain_batch_two_error,
        "batch_size_two_masked_vs_single_max_abs": masked_batch_two_error,
        "batch_size_two_equivalence_target_met": batch_two_target_met,
        "plain_cross_row_mutation_abs_diff": plain_cross_row_mutation,
        "masked_cross_row_mutation_abs_diff": masked_cross_row_mutation,
        "masked_future_mutation_abs_diff": masked_mutation,
        "plain_future_mutation_abs_diff": plain_mutation,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.perf_counter() - started,
        "status": "pass" if batch_two_target_met else "pass_with_limitation",
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("URCR-V4 scorer exactness check failed; result was saved")


if __name__ == "__main__":
    main()
