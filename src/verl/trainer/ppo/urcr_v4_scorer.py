"""Token alignment, masking, caching, and CPU scheduling for URCR-V4 scoring."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import torch


PROBE_TEMPLATE_VERSION = "urcr_v4_native_neutral_answer_probe_r1"


@dataclass(frozen=True)
class TeacherForcedItem:
    request_id: str
    comparison_group_id: str
    view: str
    actor_snapshot_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    predictor_positions: tuple[int, ...]
    blocked_key_positions: tuple[int, ...] = ()
    probe_version: str = PROBE_TEMPLATE_VERSION

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def build_teacher_forced_item(
    *,
    request_id: str,
    comparison_group_id: str | None = None,
    view: str,
    actor_snapshot_id: str,
    context_ids: Sequence[int],
    target_prefix_ids: Sequence[int],
    target_content_ids: Sequence[int],
    blocked_key_positions: Sequence[int] = (),
    max_total_tokens: int = 8192,
    probe_version: str = PROBE_TEMPLATE_VERSION,
) -> TeacherForcedItem:
    """Append a target and record its true causal predictor positions.

    ``target_prefix_ids`` contains structural tokens such as ``<answer>`` and
    is not scored.  Only ``target_content_ids`` contributes to the mean score.
    """
    group_id = request_id if comparison_group_id is None else str(comparison_group_id)
    if not request_id or not group_id or not view or not actor_snapshot_id or not probe_version:
        raise ValueError("scorer identities must be nonempty")
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
    context = tuple(int(value) for value in context_ids)
    prefix = tuple(int(value) for value in target_prefix_ids)
    target = tuple(int(value) for value in target_content_ids)
    if not target:
        raise ValueError("target content must not be empty")
    input_ids = (*context, *prefix, *target)
    if len(input_ids) > max_total_tokens:
        raise ValueError("score_context_overflow")
    target_start = len(context) + len(prefix)
    target_positions = tuple(range(target_start, target_start + len(target)))
    predictor_positions = tuple(position - 1 for position in target_positions)
    if predictor_positions[0] < 0:
        raise ValueError("the first target token has no causal predecessor")
    blocked = tuple(sorted(set(int(value) for value in blocked_key_positions)))
    if any(value < 0 or value >= target_start for value in blocked):
        raise ValueError("blocked keys must be in the scored prefix")
    return TeacherForcedItem(
        request_id=request_id,
        comparison_group_id=group_id,
        view=view,
        actor_snapshot_id=actor_snapshot_id,
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        position_ids=tuple(range(len(input_ids))),
        target_ids=target,
        target_positions=target_positions,
        predictor_positions=predictor_positions,
        blocked_key_positions=blocked,
        probe_version=probe_version,
    )


def scorer_cache_key(item: TeacherForcedItem) -> str:
    """Hash every value that changes a frozen-snapshot likelihood."""
    payload = {
        "actor_snapshot_id": item.actor_snapshot_id,
        # Equal mathematical requests in different contrast groups must stay in
        # their own padded forward. BF16 kernels can depend on batch shape, and
        # cross-group deduplication would then inject that drift into a contrast.
        "comparison_group_id": item.comparison_group_id,
        "prefix_ids": item.input_ids[: item.target_positions[0]],
        "attention_mask": item.attention_mask,
        "position_ids": item.position_ids,
        "target_ids": item.target_ids,
        "target_positions": item.target_positions,
        "predictor_positions": item.predictor_positions,
        "blocked_key_positions": item.blocked_key_positions,
        "probe_version": item.probe_version,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ScorerDeduplication:
    unique_items: tuple[TeacherForcedItem, ...]
    representative_request_by_request: dict[str, str]


def deduplicate_scorer_items(
    items: Sequence[TeacherForcedItem],
) -> ScorerDeduplication:
    """Deduplicate equal frozen-snapshot forwards while retaining every view ID."""
    request_ids = [item.request_id for item in items]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id must be unique within a scorer queue")
    representative_by_key: dict[str, TeacherForcedItem] = {}
    representative_by_request: dict[str, str] = {}
    for item in items:
        key = scorer_cache_key(item)
        representative = representative_by_key.setdefault(key, item)
        representative_by_request[item.request_id] = representative.request_id
    return ScorerDeduplication(
        tuple(representative_by_key.values()),
        representative_by_request,
    )


def expand_deduplicated_scores(
    deduplication: ScorerDeduplication,
    representative_scores: dict[str, float],
) -> dict[str, float]:
    missing = {
        request_id
        for request_id in set(deduplication.representative_request_by_request.values())
        if request_id not in representative_scores
    }
    if missing:
        raise ValueError(f"missing representative scorer results: {sorted(missing)}")
    return {
        request_id: float(representative_scores[representative])
        for request_id, representative in deduplication.representative_request_by_request.items()
    }


def build_outgoing_information_barrier(
    sequence_length: int,
    blocked_key_positions: Sequence[int],
    *,
    valid_token_mask: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build the V4 position-preserving causal attention allow-mask.

    Masked think tokens remain at their original token/position IDs.  Their
    keys are hidden from every later valid, unmasked query row, which includes
    structural/action-prefix rows and the true predictors of action content.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    blocked = tuple(sorted(set(int(value) for value in blocked_key_positions)))
    if any(value < 0 or value >= sequence_length for value in blocked):
        raise ValueError("blocked key position is outside the sequence")
    if valid_token_mask is None:
        valid = torch.ones(sequence_length, dtype=torch.bool)
    else:
        if len(valid_token_mask) != sequence_length:
            raise ValueError("valid token mask has the wrong length")
        valid = torch.tensor([bool(value) for value in valid_token_mask])

    allow = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool))
    allow &= valid[:, None] & valid[None, :]
    blocked_set = set(blocked)
    for key in blocked:
        for query in range(key + 1, sequence_length):
            if valid[query] and query not in blocked_set:
                allow[query, key] = False
    return allow


def build_batched_outgoing_information_barrier(
    attention_mask: torch.Tensor,
    blocked_key_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorize the outgoing barrier for one already-padded microbatch."""
    if attention_mask.ndim != 2 or blocked_key_mask.shape != attention_mask.shape:
        raise ValueError("attention and blocked-key masks must share shape [batch, sequence]")
    valid = attention_mask.bool()
    blocked = blocked_key_mask.bool() & valid
    batch_size, sequence_length = valid.shape
    positions = torch.arange(sequence_length, device=valid.device)
    causal = positions[:, None] >= positions[None, :]
    allow = (
        causal.unsqueeze(0)
        & valid[:, :, None]
        & valid[:, None, :]
    )
    later = positions[:, None] > positions[None, :]
    outgoing_block = (
        later.unsqueeze(0)
        & (~blocked)[:, :, None]
        & blocked[:, None, :]
    )
    return allow & ~outgoing_block


def additive_attention_mask(
    allow_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a boolean allow-mask to a 4-D additive attention mask."""
    if allow_mask.ndim not in (2, 3):
        raise ValueError("allow_mask must have shape [L,L] or [B,L,L]")
    if allow_mask.shape[-1] != allow_mask.shape[-2]:
        raise ValueError("allow_mask must be square")
    target_device = device if device is not None else allow_mask.device
    numeric = torch.full(
        allow_mask.shape,
        torch.finfo(dtype).min,
        dtype=dtype,
        device=target_device,
    )
    numeric.masked_fill_(allow_mask.to(device=target_device), 0.0)
    if numeric.ndim == 2:
        numeric = numeric.unsqueeze(0)
    return numeric.unsqueeze(1)


@dataclass(frozen=True)
class ScorerMicrobatch:
    items: tuple[TeacherForcedItem, ...]
    padded_token_count: int


def pack_scorer_items(
    items: Sequence[TeacherForcedItem],
    *,
    token_budget: int,
    max_items: int | None = None,
    batch_divisor: int = 1,
) -> tuple[ScorerMicrobatch, ...]:
    """Length-bucket complete comparison groups into batched RPC stages."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive")
    if batch_divisor <= 0:
        raise ValueError("batch_divisor must be positive")
    request_ids = [item.request_id for item in items]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id must be unique within a scorer queue")
    def padded_count(count: int) -> int:
        return ((count + batch_divisor - 1) // batch_divisor) * batch_divisor

    grouped: dict[str, list[TeacherForcedItem]] = {}
    for item in items:
        grouped.setdefault(item.comparison_group_id, []).append(item)
    ordered_groups = sorted(
        (
            (
                max(item.sequence_length for item in group),
                group_id,
                sorted(group, key=lambda item: item.request_id),
            )
            for group_id, group in grouped.items()
        ),
        key=lambda value: (-value[0], value[1]),
    )
    batches: list[ScorerMicrobatch] = []
    current: list[TeacherForcedItem] = []
    current_width = 0
    for group_width, group_id, group in ordered_groups:
        group_count = len(group)
        if max_items is not None and group_count > max_items:
            raise ValueError(
                f"comparison group {group_id!r} exceeds the scorer item limit"
            )
        if group_width * padded_count(group_count) > token_budget:
            raise ValueError(
                f"comparison group {group_id!r} exceeds the scorer token budget"
            )
        proposed_width = max(current_width, group_width)
        proposed_count = len(current) + group_count
        exceeds_items = max_items is not None and proposed_count > max_items
        exceeds_tokens = proposed_width * padded_count(proposed_count) > token_budget
        if current and (exceeds_items or exceeds_tokens):
            batches.append(
                ScorerMicrobatch(
                    tuple(current),
                    current_width * padded_count(len(current)),
                )
            )
            current = []
            current_width = 0
        current.extend(group)
        current_width = max(current_width, group_width)
    if current:
        batches.append(
            ScorerMicrobatch(
                tuple(current),
                current_width * padded_count(len(current)),
            )
        )
    return tuple(batches)


@dataclass(frozen=True)
class PaddedTeacherForcedBatch:
    request_ids: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    predictor_positions: torch.Tensor
    blocked_key_mask: torch.Tensor


@dataclass(frozen=True)
class ScoredTeacherForcedBatch:
    mean_target_log_probs: torch.Tensor
    forward_microbatch_count: int


def pad_teacher_forced_batch(
    items: Sequence[TeacherForcedItem],
    *,
    pad_token_id: int,
) -> PaddedTeacherForcedBatch:
    """Materialize a variable-prefix/variable-target independent-view batch.

    Complete sequences are right padded, preserving every sampled prefix and
    target position exactly. Targets carry a separate score mask and a per-row
    predictor-position table. Each item remains a separate batch row, so no
    view can attend to another.
    """
    if not items:
        raise ValueError("teacher-forced batch must not be empty")
    target_lengths = [len(item.target_ids) for item in items]
    target_width = max(target_lengths)
    sequence_width = max(item.sequence_length for item in items)

    batch_ids: list[list[int]] = []
    batch_attention: list[list[int]] = []
    batch_positions: list[list[int]] = []
    batch_targets: list[list[int]] = []
    batch_target_masks: list[list[bool]] = []
    batch_predictors: list[list[int]] = []
    blocked_masks: list[list[bool]] = []
    for item, target_length in zip(items, target_lengths):
        sequence_pad = sequence_width - item.sequence_length
        target_pad = target_width - target_length
        if len(item.attention_mask) != item.sequence_length:
            raise ValueError("item attention mask does not match its sequence")
        if len(item.position_ids) != item.sequence_length:
            raise ValueError("item position IDs do not match its sequence")
        if len(item.predictor_positions) != target_length:
            raise ValueError("item predictor positions do not match its target")
        ids = list(item.input_ids) + [int(pad_token_id)] * sequence_pad
        attention = list(item.attention_mask) + [0] * sequence_pad
        pad_position = int(item.position_ids[-1]) if item.position_ids else 0
        positions = list(item.position_ids) + [pad_position] * sequence_pad
        targets = list(item.target_ids) + [int(pad_token_id)] * target_pad
        target_mask = [True] * target_length + [False] * target_pad
        predictors = list(item.predictor_positions) + [0] * target_pad
        blocked = set(item.blocked_key_positions)
        blocked_mask = [index in blocked for index in range(sequence_width)]
        batch_ids.append(ids)
        batch_attention.append(attention)
        batch_positions.append(positions)
        batch_targets.append(targets)
        batch_target_masks.append(target_mask)
        batch_predictors.append(predictors)
        blocked_masks.append(blocked_mask)

    return PaddedTeacherForcedBatch(
        request_ids=tuple(item.request_id for item in items),
        input_ids=torch.tensor(batch_ids, dtype=torch.long),
        attention_mask=torch.tensor(batch_attention, dtype=torch.long),
        position_ids=torch.tensor(batch_positions, dtype=torch.long),
        target_ids=torch.tensor(batch_targets, dtype=torch.long),
        target_mask=torch.tensor(batch_target_masks, dtype=torch.bool),
        predictor_positions=torch.tensor(batch_predictors, dtype=torch.long),
        blocked_key_mask=torch.tensor(blocked_masks, dtype=torch.bool),
    )


def scorer_cost_summary(
    items: Sequence[TeacherForcedItem],
    microbatches: Sequence[ScorerMicrobatch],
) -> dict[str, int | float]:
    item_ids = [item.request_id for item in items]
    batched_ids = [item.request_id for batch in microbatches for item in batch.items]
    if sorted(item_ids) != sorted(batched_ids):
        raise ValueError("microbatches must contain every scorer item exactly once")
    true_tokens = sum(item.sequence_length for item in items)
    padded_tokens = sum(batch.padded_token_count for batch in microbatches)
    return {
        "item_count": len(items),
        "forward_microbatch_count": len(microbatches),
        "true_token_count": true_tokens,
        "padded_token_count": padded_tokens,
        "padding_token_count": padded_tokens - true_tokens,
        "padding_fraction": (
            (padded_tokens - true_tokens) / padded_tokens if padded_tokens else 0.0
        ),
    }


def _slice_padded_batch(
    batch: PaddedTeacherForcedBatch,
    start: int,
    end: int,
) -> PaddedTeacherForcedBatch:
    return PaddedTeacherForcedBatch(
        request_ids=batch.request_ids[start:end],
        input_ids=batch.input_ids[start:end],
        attention_mask=batch.attention_mask[start:end],
        position_ids=batch.position_ids[start:end],
        target_ids=batch.target_ids[start:end],
        target_mask=batch.target_mask[start:end],
        predictor_positions=batch.predictor_positions[start:end],
        blocked_key_mask=batch.blocked_key_mask[start:end],
    )


def score_teacher_forced_microbatches(
    model,
    batch: PaddedTeacherForcedBatch,
    *,
    micro_batch_size: int,
    temperature: float = 1.0,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> ScoredTeacherForcedBatch:
    """Score one staged request queue using bounded model forwards."""
    row_count = len(batch.request_ids)
    if row_count <= 0:
        raise ValueError("scorer request queue must not be empty")
    if micro_batch_size <= 0:
        raise ValueError("scorer micro_batch_size must be positive")
    scores: list[torch.Tensor] = []
    for start in range(0, row_count, micro_batch_size):
        scores.append(
            score_padded_teacher_forced_batch(
                model,
                _slice_padded_batch(
                    batch,
                    start,
                    min(start + micro_batch_size, row_count),
                ),
                temperature=temperature,
                autocast_dtype=autocast_dtype,
            )
        )
    return ScoredTeacherForcedBatch(
        mean_target_log_probs=torch.cat(scores),
        forward_microbatch_count=len(scores),
    )


def gather_mean_target_log_probs(
    logits: torch.Tensor,
    *,
    target_ids: torch.Tensor,
    predictor_positions: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather content-token log-probs using the explicit causal shift."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0).expand(logits.shape[0], -1)
    if predictor_positions.ndim != 1:
        raise ValueError("predictor_positions must be one-dimensional")
    if target_ids.shape != (logits.shape[0], len(predictor_positions)):
        raise ValueError("target IDs do not match batch/predictor shape")
    if target_mask is None:
        target_mask = torch.ones_like(target_ids, dtype=torch.bool)
    if target_mask.shape != target_ids.shape:
        raise ValueError("target mask must match target IDs")
    if not target_mask.any(dim=-1).all():
        raise ValueError("every scorer row must contain target content")
    selected = logits.index_select(1, predictor_positions.to(device=logits.device))
    log_probs = torch.log_softmax(selected.float(), dim=-1)
    gathered = log_probs.gather(
        -1, target_ids.to(device=logits.device).unsqueeze(-1)
    ).squeeze(-1)
    numeric_mask = target_mask.to(dtype=gathered.dtype, device=gathered.device)
    return (gathered * numeric_mask).sum(dim=-1) / numeric_mask.sum(dim=-1)


def score_padded_teacher_forced_batch(
    model,
    batch: PaddedTeacherForcedBatch,
    *,
    temperature: float = 1.0,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """Run one batched, no-grad teacher-forced scorer microbatch.

    The caller controls microbatch packing. Only this function materializes a
    square outgoing barrier, and only when the current microbatch contains a
    masked responsibility view.
    """
    if temperature <= 0:
        raise ValueError("scorer temperature must be positive")
    if autocast_dtype not in (None, torch.bfloat16):
        raise ValueError("scorer autocast dtype must be bf16 or None")
    input_ids = batch.input_ids
    if input_ids.ndim != 2:
        raise ValueError("scorer input IDs must have shape [batch, sequence]")
    if batch.attention_mask.shape != input_ids.shape:
        raise ValueError("scorer attention mask must match input IDs")
    if batch.position_ids.shape != input_ids.shape:
        raise ValueError("scorer position IDs must match input IDs")
    if batch.blocked_key_mask.shape != input_ids.shape:
        raise ValueError("scorer blocked-key mask must match input IDs")

    if batch.blocked_key_mask.any():
        allow = build_batched_outgoing_information_barrier(
            batch.attention_mask,
            batch.blocked_key_mask,
        )
        try:
            mask_dtype = next(model.parameters()).dtype
        except StopIteration:
            mask_dtype = torch.float32
        if mask_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            mask_dtype = torch.float32
        if input_ids.device.type == "cuda" and autocast_dtype is not None:
            mask_dtype = autocast_dtype
        attention = additive_attention_mask(
            allow,
            dtype=mask_dtype,
            device=input_ids.device,
        )
        del allow
    else:
        attention = batch.attention_mask

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if input_ids.device.type == "cuda" and autocast_dtype is not None
        else nullcontext()
    )
    predictors = batch.predictor_positions.to(device=input_ids.device)
    target_mask = batch.target_mask.to(device=input_ids.device)
    if predictors.shape != batch.target_ids.shape or target_mask.shape != predictors.shape:
        raise ValueError("scorer predictor positions must match padded targets")
    active_predictors = torch.unique(predictors[target_mask], sorted=True)
    if not len(active_predictors):
        raise ValueError("scorer batch has no active target predictors")
    if active_predictors[0] < 0 or active_predictors[-1] >= input_ids.shape[1]:
        raise ValueError("scorer predictor position leaves the sequence")
    with torch.inference_mode(), autocast_context:
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=batch.position_ids,
            use_cache=False,
            logits_to_keep=active_predictors,
        )
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != input_ids.shape[0]:
        raise RuntimeError("scorer model returned an invalid logits tensor")
    if logits.shape[1] == len(active_predictors):
        predictor_logits = logits
    elif logits.shape[1] == input_ids.shape[1]:
        predictor_logits = logits.index_select(1, active_predictors)
    else:
        raise RuntimeError("scorer backend did not honor the requested predictor rows")
    safe_predictors = torch.where(target_mask, predictors, active_predictors[0])
    predictor_indices = torch.searchsorted(active_predictors, safe_predictors)
    if not torch.equal(
        active_predictors[predictor_indices[target_mask]], predictors[target_mask]
    ):
        raise RuntimeError("scorer predictor remapping failed")
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1)
    selected = predictor_logits[batch_indices, predictor_indices]
    log_probs = torch.log_softmax(selected.float() / float(temperature), dim=-1)
    targets = batch.target_ids.to(device=log_probs.device)
    target_mask = target_mask.to(device=log_probs.device)
    if targets.shape != log_probs.shape[:2] or target_mask.shape != targets.shape:
        raise ValueError("scorer targets do not match the selected predictor rows")
    if not target_mask.any(dim=-1).all():
        raise ValueError("every scorer row must contain target content")
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    numeric_mask = target_mask.to(dtype=gathered.dtype)
    return ((gathered * numeric_mask).sum(-1) / numeric_mask.sum(-1)).detach()
