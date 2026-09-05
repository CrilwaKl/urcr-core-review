"""Detached old-policy responsibility utilities for Plan 05."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Mapping

import torch

from verl.trainer.ppo.urcr_sources import RESPONSIBILITY_SCALE, URCR_EPS, outcome_advantage


def set_model_attention_backend(
    model,
    backend: str,
    *,
    extra_configs: tuple[Any, ...] = (),
) -> list[tuple[Any, Any]]:
    """Set every distinct module config, not only the worker's config copy."""
    configs = []
    seen = set()
    for candidate in (*extra_configs, *(getattr(module, "config", None) for module in model.modules())):
        if candidate is None or not hasattr(candidate, "_attn_implementation"):
            continue
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        configs.append((candidate, candidate._attn_implementation))
        candidate._attn_implementation = backend
    if not configs:
        raise RuntimeError("Plan 05 responsibility scorer found no model attention config")
    return configs


def restore_model_attention_backends(states: list[tuple[Any, Any]]) -> None:
    for config, backend in states:
        config._attn_implementation = backend


def model_attention_backends_restored(states: list[tuple[Any, Any]]) -> bool:
    return all(config._attn_implementation == backend for config, backend in states)


def build_query_to_think_block_mask(
    sequence_length: int,
    think_token_positions: list[int],
    query_token_positions: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the causal mask blocking query predictors from current think."""
    min_value = torch.finfo(dtype).min
    mask = torch.full(
        (sequence_length, sequence_length),
        min_value,
        dtype=dtype,
        device=device,
    )
    causal = torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=device,
        )
    )
    mask.masked_fill_(causal, 0.0)
    predictor_rows = sorted(
        {position - 1 for position in query_token_positions if position > 0}
    )
    if predictor_rows and think_token_positions:
        row_index = torch.tensor(predictor_rows, dtype=torch.long, device=device)
        col_index = torch.tensor(think_token_positions, dtype=torch.long, device=device)
        mask[row_index[:, None], col_index[None, :]] = min_value
    return mask.unsqueeze(0).unsqueeze(0)


def mask_positions(mask: list[int]) -> list[int]:
    return [index for index, selected in enumerate(mask) if selected]


def needs_responsibility_score(
    frozen_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    eps: float = URCR_EPS,
) -> bool:
    if source_row.get("support_reward_version") == "v2_fixed_local":
        return bool(
            source_row.get("rho_score_required", False)
            and mask_positions(
                list(
                    frozen_row.get("think_content_mask")
                    or frozen_row.get("think_mask", [])
                )
            )
            and mask_positions(list(frozen_row.get("search_content_mask", [])))
        )
    return bool(
        source_row.get("valid_search", False)
        and float(source_row.get("g2_applied", 0.0)) > 0.0
        and abs(outcome_advantage(frozen_row)) > eps
        and mask_positions(list(frozen_row.get("think_mask", [])))
        and mask_positions(list(frozen_row.get("search_content_mask", [])))
    )


def rho_from_d_mask_tensor(
    d_mask: torch.Tensor,
    *,
    scale: float = RESPONSIBILITY_SCALE,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("Responsibility scale must be positive")
    detached = d_mask.detach().float()
    return 1.0 - torch.exp(-torch.clamp_min(detached, 0.0) / (scale + URCR_EPS))


def rho_from_d_mask(d_mask: float, *, scale: float = RESPONSIBILITY_SCALE) -> float:
    if scale <= 0:
        raise ValueError("Responsibility scale must be positive")
    return 1.0 - math.exp(-max(float(d_mask), 0.0) / (scale + URCR_EPS))


def build_response_span_tensor(
    response_mask: torch.Tensor,
    selected_rows: list[int],
    frozen_rows: list[dict[str, Any]],
    field: str,
) -> torch.Tensor:
    """Map masks over valid generated IDs back to the padded response width."""
    output = torch.zeros(
        (len(selected_rows), response_mask.shape[1]),
        dtype=torch.bool,
        device=response_mask.device,
    )
    for output_index, batch_index in enumerate(selected_rows):
        valid_positions = torch.nonzero(response_mask[batch_index].bool(), as_tuple=False).flatten()
        local_mask = list(frozen_rows[batch_index].get(field, []))
        if len(local_mask) != len(valid_positions):
            raise ValueError(
                f"{field} length mismatch for {frozen_rows[batch_index].get('turn_uid')}: "
                f"{len(local_mask)} != {len(valid_positions)}"
            )
        selected_local = [index for index, value in enumerate(local_mask) if value]
        if selected_local:
            output[output_index, valid_positions[selected_local]] = True
    return output


def build_response_chunk_tensor(
    response_mask: torch.Tensor,
    selected_rows: list[int],
    frozen_rows: list[dict[str, Any]],
    *,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map variable content chunks onto padded response tensors."""
    output = torch.zeros(
        (len(selected_rows), max_chunks, response_mask.shape[1]),
        dtype=torch.bool,
        device=response_mask.device,
    )
    counts = torch.zeros(
        len(selected_rows), dtype=torch.long, device=response_mask.device
    )
    for output_index, batch_index in enumerate(selected_rows):
        valid_positions = torch.nonzero(
            response_mask[batch_index].bool(), as_tuple=False
        ).flatten()
        chunks = list(frozen_rows[batch_index].get("think_chunks", []))
        if len(chunks) > max_chunks:
            raise ValueError("Online localizer exceeds its frozen maximum chunk count")
        # Single/empty think is a whole-content fallback and needs no LOO forward.
        if len(chunks) <= 1:
            continue
        for chunk_index, chunk in enumerate(chunks):
            local_positions = list(map(int, chunk["token_positions"]))
            if not local_positions or max(local_positions) >= len(valid_positions):
                raise ValueError("Online localizer chunk leaves the valid response")
            output[output_index, chunk_index, valid_positions[local_positions]] = True
        counts[output_index] = len(chunks)
    return output, counts


def _gather_log_probs(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)


def score_query_to_think_batch(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    think_mask: torch.Tensor,
    query_mask: torch.Tensor,
    chunk_mask: torch.Tensor | None = None,
    chunk_count: torch.Tensor | None = None,
    temperature: float,
    micro_batch_size: int,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Score full/whole/chunk masks in base-style fixed micro-batches.

    The caller already pads the selected turn rows equally across data-parallel
    ranks. Chunk expansion is variable, so this function additionally pads the
    expanded chunk variants to the cross-rank maximum before issuing model
    forwards. This keeps every FSDP rank on the same collective schedule while
    discarding padding outputs.
    """
    if temperature <= 0:
        raise ValueError("Responsibility scorer temperature must be positive")
    if micro_batch_size <= 0:
        raise ValueError("Responsibility scorer micro_batch_size must be positive")
    if autocast_dtype not in (None, torch.bfloat16):
        raise ValueError("Responsibility scorer autocast_dtype must be bf16 or None")
    if position_ids.ndim != 2:
        raise ValueError("Plan 05 responsibility scorer currently supports text position_ids only")
    if response_mask.shape != think_mask.shape or response_mask.shape != query_mask.shape:
        raise ValueError("Responsibility response/span mask shape mismatch")
    localizer_enabled = chunk_mask is not None or chunk_count is not None
    if localizer_enabled:
        if chunk_mask is None or chunk_count is None:
            raise ValueError("Localized scorer requires both chunk_mask and chunk_count")
        if chunk_mask.ndim != 3 or chunk_mask.shape[0] != len(response_mask):
            raise ValueError("Localized scorer chunk mask has invalid shape")
        if chunk_mask.shape[2] != response_mask.shape[1]:
            raise ValueError("Localized scorer response width mismatch")
        if chunk_count.shape != (len(response_mask),):
            raise ValueError("Localized scorer chunk counts have invalid shape")

    batch_size, response_width = response_mask.shape
    full_result = torch.full(
        (batch_size, response_width),
        float("nan"),
        dtype=torch.float32,
        device=input_ids.device,
    )
    masked_result = torch.full_like(full_result, float("nan"))
    d_mask = torch.empty(batch_size, dtype=torch.float32, device=input_ids.device)
    query_token_count = torch.empty(batch_size, dtype=torch.long, device=input_ids.device)
    if localizer_enabled:
        chunk_loo = torch.full(
            (batch_size, chunk_mask.shape[1]),
            float("nan"),
            dtype=torch.float32,
            device=input_ids.device,
        )
        localizer_forward_count = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )

    device_type = input_ids.device.type

    def autocast_context():
        return (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device_type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )

    try:
        mask_dtype = next(model.parameters()).dtype
    except StopIteration:
        mask_dtype = torch.float32
    if mask_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        mask_dtype = torch.float32
    if device_type == "cuda" and autocast_dtype is not None:
        mask_dtype = autocast_dtype

    prepared: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        active = attention_mask[row_index].bool()
        sequence = input_ids[row_index][active]
        positions = position_ids[row_index][active]
        valid_response = torch.nonzero(
            response_mask[row_index].bool(), as_tuple=False
        ).flatten()
        if not len(valid_response):
            raise ValueError("Responsibility scorer received an empty response")
        response_values = responses[row_index][valid_response]
        prompt_length = len(sequence) - len(response_values)
        if prompt_length < 0 or not torch.equal(
            sequence[prompt_length:], response_values
        ):
            raise ValueError("Responsibility scorer input_ids/response alignment failed")

        query_response_indices = torch.nonzero(
            query_mask[row_index].bool() & response_mask[row_index].bool(),
            as_tuple=False,
        ).flatten()
        think_response_indices = torch.nonzero(
            think_mask[row_index].bool() & response_mask[row_index].bool(),
            as_tuple=False,
        ).flatten()
        rank_by_response_index = {
            int(response_index): rank
            for rank, response_index in enumerate(valid_response.tolist())
        }
        query_local = [
            rank_by_response_index[int(value)] for value in query_response_indices
        ]
        think_local = [
            rank_by_response_index[int(value)] for value in think_response_indices
        ]
        if not query_local:
            raise ValueError("Responsibility scorer received a row without query content")
        query_absolute = [prompt_length + value for value in query_local]
        think_absolute = [prompt_length + value for value in think_local]
        predictors = [value - 1 for value in query_absolute]
        if min(predictors) < 0:
            raise ValueError("Query token cannot be the first sequence token")

        chunk_absolute: list[list[int]] = []
        if localizer_enabled:
            count = int(chunk_count[row_index])
            if count < 0 or count > chunk_mask.shape[1]:
                raise ValueError("Localized scorer chunk count is out of range")
            for chunk_index in range(count):
                chunk_response_indices = torch.nonzero(
                    chunk_mask[row_index, chunk_index].bool()
                    & response_mask[row_index].bool(),
                    as_tuple=False,
                ).flatten()
                if not len(chunk_response_indices):
                    raise ValueError("Localized scorer received an empty chunk")
                chunk_local = [
                    rank_by_response_index[int(value)]
                    for value in chunk_response_indices
                ]
                chunk_absolute.append(
                    [prompt_length + value for value in chunk_local]
                )

        prepared.append(
            {
                "row_index": row_index,
                "sequence": sequence,
                "position_ids": positions,
                "query_response_indices": query_response_indices,
                "query_absolute": query_absolute,
                "think_absolute": think_absolute,
                "chunk_absolute": chunk_absolute,
                "predictors": predictors,
                "padding": False,
            }
        )
        query_token_count[row_index] = len(query_response_indices)

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        local_rows = torch.tensor(batch_size, dtype=torch.long, device=input_ids.device)
        min_rows = local_rows.clone()
        max_rows = local_rows.clone()
        torch.distributed.all_reduce(min_rows, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(max_rows, op=torch.distributed.ReduceOp.MAX)
        if int(min_rows) != int(max_rows):
            raise RuntimeError(
                "Responsibility scorer requires equal padded row counts across FSDP ranks"
            )

    full_items = [{**item, "kind": "full", "block_absolute": None} for item in prepared]
    whole_items = [
        {**item, "kind": "whole", "block_absolute": item["think_absolute"]}
        for item in prepared
    ]
    chunk_items: list[dict[str, Any]] = []
    if localizer_enabled:
        for item in prepared:
            for chunk_index, positions in enumerate(item["chunk_absolute"]):
                chunk_items.append(
                    {
                        **item,
                        "kind": "chunk",
                        "chunk_index": chunk_index,
                        "block_absolute": positions,
                    }
                )

    local_chunk_variants = len(chunk_items)
    padded_chunk_variants = local_chunk_variants
    if localizer_enabled and distributed:
        max_chunk_variants = torch.tensor(
            local_chunk_variants, dtype=torch.long, device=input_ids.device
        )
        torch.distributed.all_reduce(
            max_chunk_variants, op=torch.distributed.ReduceOp.MAX
        )
        padded_chunk_variants = int(max_chunk_variants)
    chunk_padding_count = padded_chunk_variants - local_chunk_variants
    if chunk_padding_count:
        if not prepared:
            raise RuntimeError("Cannot pad localized scorer without a real turn")
        template = min(prepared, key=lambda item: len(item["sequence"]))
        chunk_items.extend(
            {
                **template,
                "kind": "padding",
                "chunk_index": -1,
                "block_absolute": [],
                "padding": True,
            }
            for _ in range(chunk_padding_count)
        )

    expected_forward_batches = 2 * math.ceil(batch_size / micro_batch_size)
    if padded_chunk_variants:
        expected_forward_batches += math.ceil(
            padded_chunk_variants / micro_batch_size
        )
    if distributed:
        local_expected = torch.tensor(
            expected_forward_batches, dtype=torch.long, device=input_ids.device
        )
        min_expected = local_expected.clone()
        max_expected = local_expected.clone()
        torch.distributed.all_reduce(min_expected, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(max_expected, op=torch.distributed.ReduceOp.MAX)
        if int(min_expected) != int(max_expected):
            raise RuntimeError(
                "Responsibility scorer would issue unequal FSDP forward batch counts"
            )

    forward_batch_count = 0

    def score_stream(items: list[dict[str, Any]], *, blocked: bool) -> None:
        nonlocal forward_batch_count
        for start in range(0, len(items), micro_batch_size):
            current = items[start : start + micro_batch_size]
            max_length = max(len(item["sequence"]) for item in current)
            predictor_union = sorted(
                {position for item in current for position in item["predictors"]}
            )
            union_lookup = {
                value: index for index, value in enumerate(predictor_union)
            }
            padded_ids = []
            padded_attention = []
            padded_positions = []
            block_masks = []
            for item in current:
                length = len(item["sequence"])
                pad = max_length - length
                padded_ids.append(
                    torch.cat(
                        [
                            item["sequence"],
                            torch.zeros(
                                pad, dtype=torch.long, device=input_ids.device
                            ),
                        ]
                    )
                )
                padded_attention.append(
                    torch.cat(
                        [
                            torch.ones(
                                length,
                                dtype=attention_mask.dtype,
                                device=input_ids.device,
                            ),
                            torch.zeros(
                                pad,
                                dtype=attention_mask.dtype,
                                device=input_ids.device,
                            ),
                        ]
                    )
                )
                padded_positions.append(
                    torch.cat(
                        [
                            item["position_ids"],
                            torch.zeros(
                                pad,
                                dtype=position_ids.dtype,
                                device=input_ids.device,
                            ),
                        ]
                    )
                )
                if blocked:
                    block_masks.append(
                        build_query_to_think_block_mask(
                            max_length,
                            item["block_absolute"],
                            item["query_absolute"],
                            dtype=mask_dtype,
                            device=input_ids.device,
                        )
                    )
            model_input_ids = torch.stack(padded_ids)
            model_positions = torch.stack(padded_positions)
            model_attention = (
                torch.cat(block_masks, dim=0)
                if blocked
                else torch.stack(padded_attention)
            )
            predictor_tensor = torch.tensor(
                predictor_union, dtype=torch.long, device=input_ids.device
            )
            with torch.inference_mode(), autocast_context():
                output = model(
                    input_ids=model_input_ids,
                    attention_mask=model_attention,
                    position_ids=model_positions,
                    use_cache=False,
                    logits_to_keep=predictor_tensor,
                )
            forward_batch_count += 1

            for local_index, item in enumerate(current):
                if item["padding"]:
                    continue
                target_indices = torch.tensor(
                    [union_lookup[value] for value in item["predictors"]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                targets = model_input_ids[
                    local_index,
                    torch.tensor(
                        item["query_absolute"],
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                ]
                log_probs = _gather_log_probs(
                    output.logits[local_index, target_indices],
                    targets,
                    temperature,
                )
                row_index = item["row_index"]
                response_indices = item["query_response_indices"]
                if item["kind"] == "full":
                    full_result[row_index, response_indices] = log_probs
                elif item["kind"] == "whole":
                    masked_result[row_index, response_indices] = log_probs
                elif item["kind"] == "chunk":
                    full_values = full_result[row_index, response_indices]
                    chunk_loo[row_index, item["chunk_index"]] = (
                        full_values.mean() - log_probs.mean()
                    )
                    localizer_forward_count[row_index] += 1
                else:
                    raise RuntimeError(
                        f"Unexpected responsibility scorer item kind: {item['kind']}"
                    )
            del output

    score_stream(full_items, blocked=False)
    score_stream(whole_items, blocked=True)
    if chunk_items:
        score_stream(chunk_items, blocked=True)

    d_mask.copy_(
        torch.stack(
            [
                full_result[index][query_mask[index].bool()].mean()
                - masked_result[index][query_mask[index].bool()].mean()
                for index in range(batch_size)
            ]
        )
    )

    if forward_batch_count != expected_forward_batches:
        raise RuntimeError(
            "Responsibility scorer forward batch count diverged from its preflight"
        )

    output = {
        "urcr_full_query_log_probs": full_result.detach(),
        "urcr_masked_query_log_probs": masked_result.detach(),
        "urcr_d_mask": d_mask.detach(),
        "urcr_rho": rho_from_d_mask_tensor(d_mask).detach(),
        "urcr_query_token_count": query_token_count.detach(),
    }
    if localizer_enabled:
        output.update(
            {
                "urcr_chunk_loo_scores": chunk_loo.detach(),
                "urcr_chunk_count": chunk_count.detach(),
                "urcr_localizer_forward_count": localizer_forward_count.detach(),
                "urcr_localizer_padding_variant_count": torch.full(
                    (batch_size,),
                    chunk_padding_count,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "urcr_scorer_forward_batch_count": torch.full(
                    (batch_size,),
                    forward_batch_count,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            }
        )
    return output
