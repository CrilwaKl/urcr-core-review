from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.urcr_evidence_state import (
    build_evidence_state_rows,
    semantic_turn_bucket,
)
from verl.trainer.ppo.urcr_pi_builders import (
    NegativeCandidate,
    R1_PREFIX,
    build_query_prefixes,
    build_think_prefixes,
    choose_matched_negative,
    parse_think_chunks,
    serialize_evidence,
)
from verl.trainer.ppo.urcr_shadow_objectives import (
    acquisition_totals,
    shuffled_rho,
    softmax_chunk_weights,
    typed_token_allocation,
    uniform_token_allocation,
)
from verl.trainer.ppo.urcr_pi_scorers import score_replaced_query_dependencies


def _metadata():
    return {
        "context": {
            "title": ["Bridge Article", "Answer Article", "Distractor"],
            "sentences": [
                ["Alpha connects to Beta."],
                ["The final city is Paris."],
                ["Unrelated text about Rome."],
            ],
        },
        "supporting_facts": {
            "title": ["Bridge Article", "Answer Article"],
            "sent_id": [0, 0],
        },
    }


def _row(turn_step: int, observation: str, *, source: str = "hotpotqa"):
    metadata = _metadata() if source == "hotpotqa" else None
    return {
        "uid": "rollout-question",
        "traj_uid": "trajectory",
        "turn_step": turn_step,
        "dataset_index": 17,
        "data_source": source,
        "question": "Where is the final city?",
        "ground_truth_aliases": ["Paris"],
        "metadata_json": json.dumps(metadata),
        "action_type": "search",
        "search_content_mask": [0, 1, 0],
        "invalid_action": False,
        "empty_action": False,
        "unclosed_action": False,
        "observation_text": observation,
        "episode_reward": 1.0,
        "privileged_prefix": (
            "[Privileged Hint]\nGolden Evidence:\n"
            "- Bridge Article: Alpha connects to Beta.\n"
            "- Answer Article: The final city is Paris.\n\n"
        ) if source == "hotpotqa" else "",
    }


def test_evidence_state_uses_only_prior_observations_and_counts_redundancy():
    bridge = (
        "<documents>Doc 1: Bridge Article\n"
        "Alpha connects to Beta.</documents>"
    )
    states = build_evidence_state_rows([_row(0, bridge), _row(1, bridge)])

    assert states[0]["prior_evidence_state"] == "none"
    assert states[0]["remaining_fact_count_before"] == 2
    assert states[0]["new_support_fact_count"] == 1
    assert states[0]["new_bridge_fact_count"] == 1
    assert states[1]["prior_evidence_state"] == "partial"
    assert states[1]["remaining_fact_count_before"] == 1
    assert states[1]["new_support_fact_count"] == 0
    assert states[1]["redundant_support_fact_count"] == 1
    assert states[0]["turn_uid"] != states[1]["turn_uid"]


def test_zero_based_turns_use_one_two_three_plus_strata():
    assert [semantic_turn_bucket(step) for step in range(4)] == [
        "1",
        "2",
        "3+",
        "3+",
    ]


def test_nq_has_no_query_or_think_pi_and_exactly_degrades():
    frozen = _row(0, "", source="nq")
    state = build_evidence_state_rows([frozen])[0]
    query = build_query_prefixes(frozen, state, negative=None)
    think = build_think_prefixes(frozen, state)

    assert not state["evidence_available"]
    assert not query["query_pi_available"]
    assert query["q0_prefix"] == query["q1_positive_prefix"] == ""
    assert think["r2a_prefix"] == think["r2b_prefix"] == R1_PREFIX
    assert think["r3_prefix"] == ""


def test_official_and_remaining_serialization_are_deterministic():
    text, count = serialize_evidence(
        ["Bridge Article", "Answer Article"],
        ["Alpha connects to Beta.", "The final city is Paris."],
    )
    assert count == 2
    assert text == (
        "- Bridge Article: Alpha connects to Beta.\n"
        "- Answer Article: The final city is Paris."
    )


class WordTokenizer:
    pad_token_id = 0

    def encode(self, text, **_kwargs):
        return list(range(len(str(text).split())))


def test_matched_negative_selection_is_seeded_and_prefers_local_priority():
    frozen = _row(0, "")
    state = build_evidence_state_rows([frozen])[0]
    local = NegativeCandidate(
        "local",
        "- Distractor: Unrelated text about Rome.",
        1,
        2,
        state["question_uid"],
        None,
        "same_question_distractor",
        "hotpotqa",
        "0",
    )
    donor = NegativeCandidate(
        "donor",
        "- Other: Another unrelated passage.",
        1,
        3,
        "hotpotqa:99",
        "other:0",
        "matched_other_question",
        "hotpotqa",
        "0",
    )
    first = choose_matched_negative(
        frozen_row=frozen,
        state=state,
        local_candidates=[local],
        donor_candidates=[donor],
        tokenizer=WordTokenizer(),
        seed=20260818,
    )
    second = choose_matched_negative(
        frozen_row=frozen,
        state=state,
        local_candidates=[local],
        donor_candidates=[donor],
        tokenizer=WordTokenizer(),
        seed=20260818,
    )
    assert first == second == local
    prefixes = build_query_prefixes(frozen, state, first)
    assert prefixes["q1_positive_prefix"].splitlines()[:2] == prefixes[
        "q2_negative_prefix"
    ].splitlines()[:2]


def test_r2b_masks_answer_aliases_in_titles_and_fact_text():
    frozen = _row(0, "")
    state = build_evidence_state_rows([frozen])[0]
    think = build_think_prefixes(frozen, state)

    assert "Paris" in think["r2b_unmasked_text"]
    assert "Paris" not in think["r2b_prefix"]
    assert "[ANSWER]" in think["r2b_prefix"]
    assert think["r2b_alias_mask_count"] >= 1


class PieceTokenizer:
    pieces = ["first", " step.", " next", " query", " now?"]

    def batch_decode(self, batches, **_kwargs):
        return ["".join(self.pieces[index] for index in batch) for batch in batches]

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[index] for index in ids)


def test_think_chunks_keep_original_token_positions():
    chunks = parse_think_chunks(
        PieceTokenizer(),
        response_token_ids=list(range(5)),
        think_mask=[1, 1, 1, 1, 1],
        min_chunk_tokens=1,
    )
    assert [chunk["token_positions"] for chunk in chunks] == [[0, 1], [2, 3, 4]]
    assert [chunk["token_ids"] for chunk in chunks] == [[0, 1], [2, 3, 4]]


class ReplacementScoreModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def parameters(self):
        yield self.weight

    def __call__(self, *, input_ids, logits_to_keep, **_kwargs):
        batch = input_ids.shape[0]
        logits = torch.zeros((batch, len(logits_to_keep), 32), device=input_ids.device)
        # Make the unchanged query target (token 7) depend on the replaced prefix.
        logits[:, :, 7] = input_ids.float().sum(dim=1, keepdim=True) * 0.01
        return SimpleNamespace(logits=logits)


def test_replacement_scorer_preserves_query_and_batches_variants():
    scored, forwards = score_replaced_query_dependencies(
        ReplacementScoreModel(),
        prompt_token_ids=[1, 2],
        response_token_ids=[3, 4, 7],
        query_response_positions=[2],
        replacements=[([0], [8]), ([0, 1], [8, 9])],
        device=torch.device("cpu"),
        replacement_microbatch=1,
    )
    assert len(scored) == 2
    assert forwards == 2
    assert scored[1][0] > scored[0][0]

    with pytest.raises(ValueError, match="overlap query"):
        score_replaced_query_dependencies(
            ReplacementScoreModel(),
            prompt_token_ids=[1, 2],
            response_token_ids=[3, 4, 7],
            query_response_positions=[2],
            replacements=[([2], [8])],
            device=torch.device("cpu"),
            replacement_microbatch=1,
        )


def test_shadow_allocations_conserve_credit_and_zero_degrade():
    weights = softmax_chunk_weights([0.0, 1.0], robust_scale=1.0)
    typed = typed_token_allocation(2.0, weights, [2, 3])
    uniform = uniform_token_allocation(2.0, 5)
    query_total, think_total = acquisition_totals(
        outcome_advantage=-2.0,
        acquisition_credit=0.5,
        rho=0.4,
        lambda_a=0.0,
        lambda_r=1.0,
    )

    assert np.isclose(typed.sum(), 2.0)
    assert np.isclose(uniform.sum(), 2.0)
    assert query_total == think_total == 0.0


def test_stratified_shuffle_preserves_each_stratum_marginal():
    values = np.asarray([0.1, 0.2, 0.7, 0.8])
    strata = [("a",), ("a",), ("b",), ("b",)]
    shuffled = shuffled_rho(values, strata, seed=0)
    assert sorted(shuffled[:2]) == sorted(values[:2])
    assert sorted(shuffled[2:]) == sorted(values[2:])
