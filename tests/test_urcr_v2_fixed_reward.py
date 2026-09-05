"""Exactness tests for URCR v2 fixed support rewards and local action means."""

from __future__ import annotations

import json

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_matrix
from verl.trainer.ppo.urcr_local_objective import (
    compute_urcr_local_policy_losses,
    effective_local_alpha,
)
from verl.trainer.ppo.urcr_responsibility import needs_responsibility_score
from verl.trainer.ppo.urcr_routing import build_v2_local_routing
from verl.trainer.ppo.urcr_sources import (
    SupportRewardConfig,
    build_online_g2_rows,
    compute_v2_support_utility,
    validate_plan05_config,
)


def _support_config(base: float = 0.5) -> SupportRewardConfig:
    return SupportRewardConfig(
        enabled=True,
        version="v2_fixed_local",
        utility_mode="binary_hierarchical",
        base_query_reward=base,
        fact_utility=1.0,
        doc_only_utility=0.5,
        multihit_bonus=0.0,
        multihit_cap=1.25,
    )


def _v2_config() -> dict:
    return {
        "enable": True,
        "method": "g2_real",
        "source": {
            "enable": True,
            "name": "g2_hierarchical_acquisition",
            "eta_doc_only": 0.25,
            "lambda_a": 0.25,
            "negative_no_hit": False,
            "normalize": False,
        },
        "support_reward": {
            "enable": True,
            "version": "v2_fixed_local",
            "utility_mode": "binary_hierarchical",
            "base_query_reward": 0.5,
            "fact_utility": 1.0,
            "doc_only_utility": 0.5,
            "multihit_bonus": 0.0,
            "multihit_cap": 1.25,
            "repeat_reward": 0.0,
            "miss_reward": 0.0,
            "invalid_reward": 0.0,
            "no_metadata_reward": 0.0,
            "detach": True,
            "insert_into_terminal_reward": False,
            "grpo_group_normalize": False,
            "outcome_interaction": "additive",
        },
        "local_objective": {
            "mode": "separate_eligible_action_mean",
            "query_reduction": "span_mean",
            "think_reduction": "n1_reference",
            "think_length_ref_source": "frozen_median_selected_content",
            "think_length_ref": 16,
            "think_content_only": True,
            "query_loss_weight": 1.0,
            "think_loss_weight": 1.0,
            "lambda_r": 1.0,
            "local_max": 0.2,
            "warmup_steps": 30,
            "use_existing_ppo_clip": True,
            "extra_actor_forward": False,
            "old_s_local_compatible": False,
        },
        "responsibility": {
            "name": "query_to_current_think_attention_block",
            "mapping": "exp_positive",
            "scale": 0.6201,
            "no_grad": True,
            "lazy_on_effective_source": True,
            "score_eligibility": "source_active_independent_of_a_out",
        },
        "routing": {
            "mode": "real",
            "lambda_r": 1.0,
            "query_allocation": "uniform_action_total",
            "think_allocation": "uniform_action_total",
            "residual_only": True,
            "normalize": False,
            "think_credit_mode": "loo_mass50",
        },
        "protection": {
            "protect_verified_support_query": False,
            "protect_routed_think": False,
        },
        "audit": {
            "compute_shadow_modes": False,
            "compute_q0_shadow": False,
            "save_surrogate_coefficients": False,
            "save_turn_components": True,
            "output_dir": "artifacts/urcr_v2",
        },
    }


def _state(**updates):
    row = {
        "evidence_available": True,
        "valid_search": True,
        "new_support_doc_count": 0,
        "new_support_fact_count": 0,
        "redundant_support_doc_count": 0,
        "redundant_support_fact_count": 0,
    }
    row.update(updates)
    return row


def _metadata() -> dict:
    return {
        "context": {
            "title": ["Bridge", "Answer"],
            "sentences": [["Bridge fact."], ["The answer is Paris."]],
        },
        "supporting_facts": {"title": ["Bridge", "Answer"], "sent_id": [0, 0]},
    }


def _frozen(a_out: float, observation: str) -> dict:
    return {
        "turn_uid": "traj:0",
        "traj_uid": "traj",
        "turn_step": 0,
        "dataset_index": 3,
        "data_source": "hotpotqa",
        "question": "Where?",
        "ground_truth_aliases": ["Paris"],
        "metadata_json": json.dumps(_metadata()),
        "action_type": "search",
        "search_content_mask": [0, 0, 0, 1, 1, 0],
        "think_mask": [1, 1, 1, 0, 0, 0],
        "think_content_mask": [0, 1, 0, 0, 0, 0],
        "think_chunks": [{"chunk_index": 0, "token_positions": [1]}],
        "localizer_prepare_fallback_reason": "single_chunk",
        "invalid_action": False,
        "empty_action": False,
        "unclosed_action": False,
        "observation_text": observation,
        "episode_reward": 0.0,
        "outcome_advantage_token": [a_out] * 6,
    }


def test_v2_config_is_explicit_and_rejects_legacy_local_scale():
    parsed = validate_plan05_config(_v2_config())
    assert parsed.uses_v2_fixed_support_reward
    assert parsed.support_reward.base_query_reward == 0.5
    assert parsed.local_objective.think_length_ref == 16
    assert parsed.local_objective.local_max == 0.2
    cfg = _v2_config()
    cfg["local_objective"]["old_s_local_compatible"] = True
    with pytest.raises(ValueError, match="old_s_local_compatible"):
        validate_plan05_config(cfg)


@pytest.mark.parametrize("local_max", (0.0, -0.1, 1.1, float("inf")))
def test_v2_local_max_must_be_a_finite_positive_cap(local_max):
    cfg = _v2_config()
    cfg["local_objective"]["local_max"] = local_max
    with pytest.raises(ValueError, match="local_max"):
        validate_plan05_config(cfg)


def test_v2_local_max_must_be_explicit():
    cfg = _v2_config()
    del cfg["local_objective"]["local_max"]
    with pytest.raises(ValueError, match="local_max must be explicit"):
        validate_plan05_config(cfg)


def test_v2_local_max_multiplies_the_existing_full_warmup_schedule():
    values = [
        effective_local_alpha(local_max=0.2, global_step=step, warmup_steps=30)
        for step in (1, 15, 30, 100)
    ]
    assert values == pytest.approx([0.2 / 30.0, 0.1, 0.2, 0.2])


def test_v2_utility_is_categorical_not_cardinality_fractional():
    fact = compute_v2_support_utility(
        _state(new_support_fact_count=1, new_support_doc_count=0)
    )
    assert fact.utility == 1.0
    assert fact.hit_type == "fact"
    doc = compute_v2_support_utility(_state(new_support_doc_count=1))
    assert doc.utility == 0.5
    assert doc.hit_type == "doc_only"
    fact_only_doc = compute_v2_support_utility(
        _state(new_support_doc_count=1), utility_mode="binary_fact_only"
    )
    assert fact_only_doc.utility == 0.0
    assert fact_only_doc.hit_type == "doc_only"
    multi = compute_v2_support_utility(
        _state(new_support_doc_count=2, new_support_fact_count=4)
    )
    assert multi.utility == 1.0
    repeat = compute_v2_support_utility(
        _state(redundant_support_doc_count=1, redundant_support_fact_count=2)
    )
    assert repeat.utility == 0.0 and repeat.hit_type == "repeat"
    assert compute_v2_support_utility(_state()).hit_type == "no_hit"
    assert compute_v2_support_utility(_state(valid_search=False)).hit_type == "invalid"
    no_meta = compute_v2_support_utility(_state(evidence_available=False))
    assert no_meta.utility == 0.0 and no_meta.hit_type == "no_metadata"


@pytest.mark.parametrize("a_out", (-1.0, 0.0, 1.0))
def test_v2_source_and_routing_are_outcome_independent(a_out):
    row = _frozen(a_out, "<documents>Doc 1: Bridge\nBridge fact.</documents>")
    source = build_online_g2_rows([row], support_reward=_support_config())
    assert source[0]["support_utility_v2"] == 1.0
    assert source[0]["rho_score_required"]
    assert needs_responsibility_score(row, source[0])
    tensors, components = build_v2_local_routing(
        torch.ones(1, 6),
        [row],
        source,
        {"traj:0": {"rho": 0.4, "rho_scored": True, "d_mask": 0.2}},
        base_query_reward=0.5,
        lambda_r=1.0,
        mode="real",
        shuffle_seed=7,
        think_length_ref=16,
    )
    assert tensors["urcr_v2_query_reward"].item() == 0.5
    assert torch.isclose(tensors["urcr_v2_think_reward"], torch.tensor([0.2])).all()
    assert components[0]["a_out"] == a_out


def test_v2_history_turns_new_hit_into_repeat_without_future_leak():
    observation = "<documents>Doc 1: Bridge\nBridge fact.</documents>"
    first = _frozen(0.0, observation)
    second = dict(first, turn_uid="traj:1", turn_step=1)
    states = build_online_g2_rows([first, second], support_reward=_support_config())
    assert states[0]["support_utility_v2"] == 1.0
    assert states[1]["support_utility_v2"] == 0.0
    assert states[1]["support_hit_type"] == "repeat"
    assert "future_new_support_fact_count" not in states[0]

    changed_future = dict(second, observation_text="unrelated future observation")
    changed = build_online_g2_rows(
        [first, changed_future], support_reward=_support_config()
    )
    assert changed[0]["support_utility_v2"] == states[0]["support_utility_v2"]


def test_no_metadata_is_zero_and_excluded_from_local_denominators():
    row = _frozen(0.0, "<documents>anything</documents>")
    row.update(
        {
            "data_source": "nq",
            "metadata_json": "null",
            "turn_uid": "nq:0",
            "traj_uid": "nq",
        }
    )
    source = build_online_g2_rows([row], support_reward=_support_config())[0]
    assert source["support_hit_type"] == "no_metadata"
    assert source["support_utility_v2"] == 0.0
    assert not source["query_local_eligible"]
    assert not source["rho_score_required"]


@pytest.mark.parametrize("query_length", (1, 2, 4, 8, 16, 32, 64))
def test_query_action_mean_is_length_invariant_and_mask_scoped(query_length):
    width = query_length + 3
    log_prob = torch.zeros(1, width, requires_grad=True)
    old = torch.zeros_like(log_prob)
    query_mask = torch.zeros_like(log_prob, dtype=torch.bool)
    query_mask[:, 1 : 1 + query_length] = True
    terms = compute_urcr_local_policy_losses(
        old_log_prob=old,
        log_prob=log_prob,
        query_mask=query_mask,
        think_mask=torch.zeros_like(query_mask),
        query_eligible=torch.ones(1),
        think_eligible=torch.zeros(1),
        query_reward=torch.tensor([0.5]),
        think_reward=torch.zeros(1),
        selected_think_length=torch.zeros(1),
        think_length_ref=16,
        global_query_eligible_count=1,
        global_think_eligible_count=0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert torch.isclose(terms.query_loss, torch.tensor(-0.5))
    terms.query_loss.backward()
    assert torch.count_nonzero(log_prob.grad[~query_mask]) == 0
    assert torch.isclose(log_prob.grad[query_mask].sum(), torch.tensor(-0.5))


@pytest.mark.parametrize("length", (1, 2, 4, 8, 16, 32, 64, 128))
def test_think_n1_reference_matches_closed_form_and_is_mask_scoped(length):
    width = length + 2
    log_prob = torch.zeros(1, width, requires_grad=True)
    old = torch.zeros_like(log_prob)
    think_mask = torch.zeros_like(log_prob, dtype=torch.bool)
    think_mask[:, 1 : 1 + length] = True
    terms = compute_urcr_local_policy_losses(
        old_log_prob=old,
        log_prob=log_prob,
        query_mask=torch.zeros_like(think_mask),
        think_mask=think_mask,
        query_eligible=torch.zeros(1),
        think_eligible=torch.ones(1),
        query_reward=torch.zeros(1),
        think_reward=torch.tensor([0.2]),
        selected_think_length=torch.tensor([float(length)]),
        think_length_ref=16,
        global_query_eligible_count=0,
        global_think_eligible_count=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    expected = -0.2 * (length / 16) ** 0.5
    assert terms.think_loss.item() == pytest.approx(expected, abs=1e-7)
    terms.think_loss.backward()
    assert torch.count_nonzero(log_prob.grad[~think_mask]) == 0


def test_zero_eligible_returns_graph_connected_zero():
    log_prob = torch.zeros(2, 4, requires_grad=True)
    zeros = torch.zeros(2)
    terms = compute_urcr_local_policy_losses(
        old_log_prob=torch.zeros_like(log_prob),
        log_prob=log_prob,
        query_mask=torch.zeros_like(log_prob, dtype=torch.bool),
        think_mask=torch.zeros_like(log_prob, dtype=torch.bool),
        query_eligible=zeros,
        think_eligible=zeros,
        query_reward=zeros,
        think_reward=zeros,
        selected_think_length=zeros,
        think_length_ref=16,
        global_query_eligible_count=0,
        global_think_eligible_count=0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    (terms.query_loss + terms.think_loss).backward()
    assert torch.equal(log_prob.grad, torch.zeros_like(log_prob))


def test_positive_multiplier_matches_existing_vanilla_ppo_primitive():
    old = torch.tensor([[0.0, 0.0, 0.0]])
    new = torch.log(torch.tensor([[0.7, 1.1, 1.4]])).requires_grad_()
    mask = torch.ones_like(old, dtype=torch.bool)
    terms = compute_urcr_local_policy_losses(
        old_log_prob=old,
        log_prob=new,
        query_mask=mask,
        think_mask=torch.zeros_like(mask),
        query_eligible=torch.ones(1),
        think_eligible=torch.zeros(1),
        query_reward=torch.ones(1),
        think_reward=torch.zeros(1),
        selected_think_length=torch.zeros(1),
        think_length_ref=16,
        global_query_eligible_count=1,
        global_think_eligible_count=0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    ratio = torch.exp(new - old)
    matrix, _, _ = compute_policy_loss_matrix(
        ratio=ratio,
        advantages=torch.ones_like(ratio),
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    assert torch.equal(terms.query_loss, matrix.mean())


def _local_loss_for_slice(parameter, features, payload, row_slice, global_counts):
    log_prob = parameter * features[row_slice]
    old = torch.zeros_like(log_prob)
    terms = compute_urcr_local_policy_losses(
        old_log_prob=old,
        log_prob=log_prob,
        query_mask=payload["query_mask"][row_slice],
        think_mask=payload["think_mask"][row_slice],
        query_eligible=payload["query_eligible"][row_slice],
        think_eligible=payload["think_eligible"][row_slice],
        query_reward=payload["query_reward"][row_slice],
        think_reward=payload["think_reward"][row_slice],
        selected_think_length=payload["selected_length"][row_slice],
        think_length_ref=4,
        global_query_eligible_count=global_counts[0],
        global_think_eligible_count=global_counts[1],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    return terms.query_loss + terms.think_loss


def test_ddp_world_scaling_and_accumulation_match_global_action_mean():
    features = torch.tensor(
        [
            [0.1, 0.2, 0.0, 0.3],
            [0.2, 0.0, 0.4, 0.1],
            [0.0, 0.3, 0.2, 0.2],
            [0.4, 0.1, 0.3, 0.0],
            [0.2, 0.2, 0.1, 0.4],
            [0.3, 0.0, 0.2, 0.1],
        ]
    )
    payload = {
        "query_mask": torch.tensor(
            [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
            dtype=torch.bool,
        ),
        "think_mask": torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 1, 0]],
            dtype=torch.bool,
        ),
        # Rank 1 (rows 3:6) has only one eligible query and no positive reward.
        "query_eligible": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        "think_eligible": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        "query_reward": torch.tensor([0.5, 0.25, 0.5, 0.0, 0.0, 0.0]),
        "think_reward": torch.tensor([0.2, 0.1, 0.3, 0.0, 0.0, 0.0]),
        "selected_length": torch.tensor([2.0, 2.0, 2.0, 0.0, 0.0, 2.0]),
    }
    global_counts = (
        payload["query_eligible"].sum(),
        payload["think_eligible"].sum(),
    )
    reference_parameter = torch.tensor(0.2, requires_grad=True)
    reference_loss = _local_loss_for_slice(
        reference_parameter, features, payload, slice(None), global_counts
    )
    reference_loss.backward()
    reference_grad = reference_parameter.grad.detach().clone()

    rank_grads = []
    rank_losses = []
    for rank_slice in (slice(0, 3), slice(3, 6)):
        parameter = torch.tensor(0.2, requires_grad=True)
        # Actor implementation multiplies each rank-local numerator by world size;
        # DDP subsequently averages shared-parameter gradients across ranks.
        rank_loss = 2.0 * _local_loss_for_slice(
            parameter, features, payload, rank_slice, global_counts
        )
        rank_loss.backward()
        rank_losses.append(rank_loss.detach())
        rank_grads.append(parameter.grad.detach())
    assert torch.allclose(torch.stack(rank_grads).mean(), reference_grad, atol=1e-7)
    assert torch.allclose(torch.stack(rank_losses).mean(), reference_loss, atol=1e-7)

    accumulated_rank_grads = []
    for micro_slices in (
        (slice(0, 1), slice(1, 3)),
        (slice(3, 5), slice(5, 6)),
    ):
        parameter = torch.tensor(0.2, requires_grad=True)
        loss = sum(
            2.0
            * _local_loss_for_slice(
                parameter, features, payload, micro_slice, global_counts
            )
            for micro_slice in micro_slices
        )
        loss.backward()
        accumulated_rank_grads.append(parameter.grad.detach())
    assert torch.allclose(
        torch.stack(accumulated_rank_grads).mean(), reference_grad, atol=1e-7
    )
