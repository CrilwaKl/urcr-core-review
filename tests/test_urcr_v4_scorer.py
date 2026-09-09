from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.urcr_v4_scorer import (
    additive_attention_mask,
    build_batched_outgoing_information_barrier,
    build_outgoing_information_barrier,
    build_teacher_forced_item,
    deduplicate_scorer_items,
    expand_deduplicated_scores,
    gather_mean_target_log_probs,
    pad_teacher_forced_batch,
    pack_scorer_items,
    scorer_cache_key,
    scorer_cost_summary,
    score_padded_teacher_forced_batch,
    score_teacher_forced_microbatches,
)


def _item(identity: str, context_length: int):
    return build_teacher_forced_item(
        request_id=identity,
        view="real",
        actor_snapshot_id="outer-step-1-pre-update",
        context_ids=list(range(context_length)),
        target_prefix_ids=[90],
        target_content_ids=[91, 92],
    )


def test_teacher_forcing_scores_content_with_the_true_causal_shift() -> None:
    item = build_teacher_forced_item(
        request_id="q:0:real:alias0",
        view="real",
        actor_snapshot_id="snapshot-1",
        context_ids=[10, 11],
        target_prefix_ids=[20],
        target_content_ids=[30, 31],
        blocked_key_positions=[1],
    )
    assert item.input_ids == (10, 11, 20, 30, 31)
    assert item.target_positions == (3, 4)
    assert item.predictor_positions == (2, 3)
    assert item.target_ids == (30, 31)
    assert item.blocked_key_positions == (1,)


def test_scorer_cache_key_binds_snapshot_prefix_mask_position_target_and_version() -> None:
    item = _item("one", 4)
    key = scorer_cache_key(item)
    variants = [
        replace(item, actor_snapshot_id="other"),
        replace(item, comparison_group_id="other"),
        replace(item, input_ids=(99, *item.input_ids[1:])),
        replace(item, attention_mask=(0, *item.attention_mask[1:])),
        replace(item, position_ids=(7, *item.position_ids[1:])),
        replace(item, target_ids=(88, item.target_ids[1])),
        replace(item, blocked_key_positions=(0,)),
        replace(item, probe_version="other"),
    ]
    assert all(scorer_cache_key(variant) != key for variant in variants)


def test_scorer_deduplication_reuses_identical_forward_across_view_ids() -> None:
    first = _item("control_0", 4)
    duplicate = replace(
        first,
        request_id="control_2",
        comparison_group_id=first.comparison_group_id,
        view="control_2",
    )
    distinct = replace(first, request_id="real", input_ids=(99, *first.input_ids[1:]))
    deduplicated = deduplicate_scorer_items([first, duplicate, distinct])
    assert [item.request_id for item in deduplicated.unique_items] == [
        "control_0",
        "real",
    ]
    scores = expand_deduplicated_scores(
        deduplicated,
        {"control_0": -1.2, "real": -0.7},
    )
    assert scores == {"control_0": -1.2, "control_2": -1.2, "real": -0.7}


def test_scorer_deduplication_keeps_equal_requests_in_distinct_comparisons() -> None:
    first = _item("first", 4)
    second = replace(
        first,
        request_id="second",
        comparison_group_id="second-comparison",
    )
    deduplicated = deduplicate_scorer_items([first, second])
    assert [item.request_id for item in deduplicated.unique_items] == [
        "first",
        "second",
    ]


def test_outgoing_barrier_blocks_all_later_unmasked_consumers() -> None:
    allow = build_outgoing_information_barrier(6, [1, 2])
    assert allow[1, 1] and allow[2, 1]
    assert not allow[3, 1] and not allow[3, 2]
    assert not allow[4, 1] and not allow[4, 2]
    assert allow[4, 0] and allow[4, 3] and allow[4, 4]
    assert not allow[0, 1]

    padded = build_outgoing_information_barrier(
        6, [1], valid_token_mask=[1, 1, 1, 1, 0, 0]
    )
    assert not padded[4].any() and not padded[:, 4].any()
    additive = additive_attention_mask(allow, dtype=torch.float32)
    assert additive.shape == (1, 1, 6, 6)
    assert additive[0, 0, 4, 3] == 0
    assert additive[0, 0, 4, 1] == torch.finfo(torch.float32).min


def test_single_chunk_and_whole_think_build_identical_barriers() -> None:
    whole = build_outgoing_information_barrier(7, [1, 2, 3])
    single_chunk = build_outgoing_information_barrier(7, [1, 2, 3])
    assert torch.equal(whole, single_chunk)


def test_batched_barrier_matches_row_reference_without_queue_sized_square_masks() -> None:
    attention = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
    blocked = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    actual = build_batched_outgoing_information_barrier(attention, blocked)
    expected = torch.stack(
        [
            build_outgoing_information_barrier(
                5,
                [1],
                valid_token_mask=attention[0].tolist(),
            ),
            build_outgoing_information_barrier(
                5,
                [2],
                valid_token_mask=attention[1].tolist(),
            ),
        ]
    )
    assert torch.equal(actual, expected)


def test_token_budget_packing_is_batched_complete_and_deterministic() -> None:
    items = [_item("short", 2), _item("long", 7), _item("mid", 5), _item("tiny", 1)]
    batches = pack_scorer_items(items, token_budget=20, max_items=3)
    assert all(batch.padded_token_count <= 20 for batch in batches)
    assert [item.request_id for batch in batches for item in batch.items] == [
        "long",
        "mid",
        "short",
        "tiny",
    ]
    summary = scorer_cost_summary(items, batches)
    assert summary["item_count"] == 4
    assert summary["forward_microbatch_count"] < summary["item_count"]
    assert summary["padded_token_count"] >= summary["true_token_count"]


def test_token_budget_accounts_for_equal_fsdp_rank_padding() -> None:
    items = [_item(str(index), length) for index, length in enumerate([7, 7, 7, 3, 3])]
    batches = pack_scorer_items(
        items,
        token_budget=40,
        max_items=8,
        batch_divisor=4,
    )
    assert all(batch.padded_token_count <= 40 for batch in batches)
    assert all(
        batch.padded_token_count
        == max(item.sequence_length for item in batch.items)
        * (((len(batch.items) + 3) // 4) * 4)
        for batch in batches
    )


def test_token_budget_never_splits_one_comparison_group() -> None:
    first = replace(_item("g1-long", 7), comparison_group_id="g1")
    second = replace(_item("g1-short", 2), comparison_group_id="g1")
    third = replace(_item("g2", 6), comparison_group_id="g2")
    batches = pack_scorer_items(
        [first, third, second],
        token_budget=28,
        max_items=3,
        batch_divisor=1,
    )
    locations = {
        item.request_id: batch_index
        for batch_index, batch in enumerate(batches)
        for item in batch.items
    }
    assert locations["g1-long"] == locations["g1-short"]


def test_variable_prefix_and_target_lengths_align_in_one_independent_view_batch() -> None:
    first = build_teacher_forced_item(
        request_id="first",
        view="real",
        actor_snapshot_id="snapshot",
        context_ids=[10, 11, 12],
        target_prefix_ids=[20],
        target_content_ids=[30],
        blocked_key_positions=[1],
    )
    second = build_teacher_forced_item(
        request_id="second",
        view="control",
        actor_snapshot_id="snapshot",
        context_ids=[13],
        target_prefix_ids=[20],
        target_content_ids=[31, 32, 33],
        blocked_key_positions=[0],
    )
    batch = pad_teacher_forced_batch([first, second], pad_token_id=0)
    assert batch.input_ids.tolist() == [
        [10, 11, 12, 20, 30],
        [13, 20, 31, 32, 33],
    ]
    assert batch.target_ids.tolist() == [[30, 0, 0], [31, 32, 33]]
    assert batch.target_mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]
    assert batch.predictor_positions.tolist() == [[3, 0, 0], [1, 2, 3]]
    assert batch.position_ids.tolist() == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]
    assert batch.blocked_key_mask.tolist() == [
        [False, True, False, False, False],
        [True, False, False, False, False],
    ]
    barrier = build_batched_outgoing_information_barrier(
        batch.attention_mask,
        batch.blocked_key_mask,
    )
    # The first target predictor cannot read the masked think key after padding.
    assert not barrier[0, 3, 1]
    assert not barrier[1, 1, 0]


def test_target_logprob_gather_uses_explicit_predictor_rows() -> None:
    logits = torch.zeros((1, 5, 10))
    logits[0, 1, 7] = 8.0
    logits[0, 3, 8] = 8.0
    score = gather_mean_target_log_probs(
        logits,
        target_ids=torch.tensor([7, 8]),
        predictor_positions=torch.tensor([1, 3]),
    )
    wrong = gather_mean_target_log_probs(
        logits,
        target_ids=torch.tensor([7, 8]),
        predictor_positions=torch.tensor([2, 4]),
    )
    assert score.shape == (1,)
    assert score.item() > wrong.item()


def test_scorer_rejects_overflow_and_duplicate_request_ids() -> None:
    with pytest.raises(ValueError, match="score_context_overflow"):
        build_teacher_forced_item(
            request_id="overflow",
            view="real",
            actor_snapshot_id="snapshot",
            context_ids=[1, 2],
            target_prefix_ids=[3],
            target_content_ids=[4],
            max_total_tokens=3,
        )
    item = _item("duplicate", 2)
    with pytest.raises(ValueError, match="request_id must be unique"):
        pack_scorer_items([item, item], token_budget=32)


class _RecordingScorer(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 128) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocabulary_size = vocabulary_size
        self.last_attention_mask = None
        self.forward_calls = 0

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        logits_to_keep,
    ):
        del position_ids, use_cache
        self.forward_calls += 1
        self.last_attention_mask = attention_mask.detach().clone()
        logits = torch.zeros(
            (*input_ids.shape, self.vocabulary_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits.index_select(1, logits_to_keep))


def test_scorer_forward_materializes_square_barrier_only_for_masked_microbatch() -> None:
    plain = pad_teacher_forced_batch([_item("plain", 3)], pad_token_id=0)
    model = _RecordingScorer()
    scores = score_padded_teacher_forced_batch(model, plain, autocast_dtype=None)
    assert scores.shape == (1,)
    assert model.last_attention_mask.ndim == 2
    assert not any(parameter.grad is not None for parameter in model.parameters())

    masked_item = replace(_item("masked", 3), blocked_key_positions=(1,))
    masked = pad_teacher_forced_batch([masked_item], pad_token_id=0)
    score_padded_teacher_forced_batch(model, masked, autocast_dtype=None)
    assert model.last_attention_mask.shape == (1, 1, 6, 6)
    first_predictor = int(masked.predictor_positions[0, 0])
    assert model.last_attention_mask[0, 0, first_predictor, 1] < 0


def test_staged_scorer_queue_uses_bounded_microbatches() -> None:
    batch = pad_teacher_forced_batch(
        [_item("one", 2), _item("two", 4), _item("three", 6)],
        pad_token_id=0,
    )
    model = _RecordingScorer()
    result = score_teacher_forced_microbatches(
        model,
        batch,
        micro_batch_size=2,
        autocast_dtype=None,
    )
    assert result.mean_target_log_probs.shape == (3,)
    assert result.forward_microbatch_count == 2
    assert model.forward_calls == 2


def test_tiny_qwen_variable_prefix_batch_matches_unpadded_single_scores() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(7)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_sliding_window=False,
            sliding_window=128,
            max_window_layers=2,
        )
        model = Qwen2ForCausalLM(config).eval()
        items = [_item("short", 2), _item("long", 7)]
        masked_items = [
            replace(item, blocked_key_positions=(1,)) for item in items
        ]
        for views in (items, masked_items):
            singles = torch.cat(
                [
                    score_padded_teacher_forced_batch(
                        model,
                        pad_teacher_forced_batch([item], pad_token_id=0),
                        autocast_dtype=None,
                    )
                    for item in views
                ]
            )
            batched = score_padded_teacher_forced_batch(
                model,
                pad_teacher_forced_batch(views, pad_token_id=0),
                autocast_dtype=None,
            )
            assert torch.allclose(singles, batched, atol=1e-6, rtol=1e-6)
