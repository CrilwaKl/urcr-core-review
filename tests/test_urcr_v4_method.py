from __future__ import annotations

import math

import numpy as np
import pytest

from verl.trainer.ppo.urcr_v4_method import (
    ANSWER_ALPHA,
    AnswerTrajectory,
    ObservationCandidate,
    apply_base_shadow_gradient_ceiling,
    answer_quality,
    calibrate_gradient_scale,
    calibrate_responsibility,
    calibrate_search_utility,
    cap_search_trajectory_credits,
    compute_answer_residuals,
    compute_search_utility,
    control_rng_for_anchor,
    directed_agreement,
    forbidden_online_metadata_paths,
    require_no_online_evidence_metadata,
    responsibility_strength,
    route_action_credit,
    select_observation_controls,
    select_probe_aliases,
    soft_sparse_chunk_weights,
    visible_exact_repeat,
    word_fbeta,
)


def test_gradient_scale_uses_only_the_frozen_two_batch_rule() -> None:
    result = calibrate_gradient_scale(
        global_gradient_norms=[2.0, 1.0],
        local_gradient_norms=[10.0, 10.0],
    )
    assert result.global_over_local == pytest.approx((0.2, 0.1))
    assert result.lambda_0 == pytest.approx(0.015)
    assert result.lambda_max == pytest.approx(0.015)


def test_base_shadow_ceiling_only_lowers_lambda_above_thirty_percent() -> None:
    unchanged, ratio = apply_base_shadow_gradient_ceiling(
        lambda_max=0.02,
        global_gradient_norm=1.0,
        local_gradient_norm=10.0,
    )
    assert unchanged == pytest.approx(0.02)
    assert ratio == pytest.approx(0.2)
    lowered, ratio = apply_base_shadow_gradient_ceiling(
        lambda_max=0.05,
        global_gradient_norm=1.0,
        local_gradient_norm=10.0,
    )
    assert lowered == pytest.approx(0.03)
    assert ratio == pytest.approx(0.3)


def test_probe_aliases_are_fixed_deduplicated_and_primary_first() -> None:
    assert select_probe_aliases(
        ["the New York", "New York", "NYC", "", "Big Apple"],
        primary="NYC",
    ) == ("NYC", "Big Apple", "New York")
    assert select_probe_aliases(["beta", "Alpha", "gamma"]) == (
        "Alpha",
        "beta",
        "gamma",
    )


def test_answer_quality_uses_word_multisets_and_binary_em_override() -> None:
    assert math.isclose(word_fbeta("red red red", "red"), 5 / 13)
    assert answer_quality("wrong", ["right"], binary_em=1) == 1.0
    assert answer_quality("red red red", ["red", "blue"], binary_em=0) == pytest.approx(
        5 / 13
    )


def test_answer_residuals_use_complete_group_and_binary_em_stratum() -> None:
    rows = [
        AnswerTrajectory("success-a", "group-a", 1, True, "Paris", ("Paris",)),
        AnswerTrajectory("success-b", "group-a", 1, True, "Paris", ("Paris",)),
        AnswerTrajectory("fail-close", "group-a", 0, True, "Paris city", ("Paris",)),
        AnswerTrajectory("fail-far", "group-a", 0, True, "London", ("Paris",)),
        AnswerTrajectory("fail-invalid", "group-a", 0, False, None, ("Paris",)),
        AnswerTrajectory("singleton", "group-b", 0, True, "Paris city", ("Paris",)),
    ]
    result = compute_answer_residuals(rows)

    assert result["success-a"].residual == result["success-b"].residual == 0.0
    assert result["fail-close"].residual > 0
    assert result["fail-far"].residual < 0
    assert result["fail-close"].residual + result["fail-far"].residual == pytest.approx(0)
    assert not result["fail-invalid"].eligible
    assert result["singleton"].eligible and result["singleton"].residual == 0.0


def test_answer_residuals_reject_duplicate_trajectory_identity() -> None:
    row = AnswerTrajectory("same", "group", 0, True, "x", ("y",))
    with pytest.raises(ValueError, match="duplicate trajectory_id"):
        compute_answer_residuals([row, row])


def _observation(
    identity: str,
    question: str,
    *,
    source: str = "nq",
    bucket: str = "1",
    length: int = 100,
    body: str | None = None,
    documents: int = 3,
) -> ObservationCandidate:
    text = body if body is not None else f"documents for {identity}"
    return ObservationCandidate(
        observation_id=identity,
        question_id=question,
        data_source=source,
        call_bucket=bucket,
        serialized_observation=text,
        token_length=length,
        document_count=documents,
        document_signature=(identity,),
    )


def test_control_selection_uses_global_priority_and_explicit_rng() -> None:
    anchor = _observation("anchor", "question-a")
    pool = [
        _observation("same-question", "question-a"),
        _observation("tier-0", "question-b"),
        _observation("tier-1", "question-c", bucket="2+"),
        _observation("tier-2", "question-d", source="hotpotqa"),
        _observation("tier-3", "question-e", source="hotpotqa", length=300),
    ]
    first = select_observation_controls(
        anchor, pool, rng=np.random.default_rng(17), count=3
    )
    second = select_observation_controls(
        anchor, pool, rng=np.random.default_rng(17), count=3
    )
    assert first == second
    assert [item.observation_id for item in first.controls] == [
        "tier-0",
        "tier-1",
        "tier-2",
    ]
    assert first.selected_tiers == (0, 1, 2)
    assert first.unique_control_count == 3
    assert not first.used_replacement


def test_control_rng_is_counter_based_and_does_not_advance_numpy_global_rng() -> None:
    np.random.seed(11)
    expected = np.random.random(3)
    np.random.seed(11)
    first = control_rng_for_anchor(global_step=7, turn_id="stable-turn").random(4)
    observed = np.random.random(3)
    second = control_rng_for_anchor(global_step=7, turn_id="stable-turn").random(4)
    different = control_rng_for_anchor(global_step=8, turn_id="stable-turn").random(4)
    assert np.array_equal(observed, expected)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_control_selection_repeats_only_after_exhausting_unique_pool() -> None:
    anchor = _observation("anchor", "question-a")
    only = _observation("only", "question-b")
    result = select_observation_controls(
        anchor, [only], rng=np.random.default_rng(2), count=3
    )
    assert result.unique_control_count == 1
    assert result.used_replacement
    assert result.controls == (only, only, only)


def test_control_selection_deduplicates_equal_document_sets() -> None:
    anchor = _observation("anchor", "question-a")
    first = _observation("first", "question-b")
    duplicate = ObservationCandidate(
        observation_id="duplicate",
        question_id="question-c",
        data_source="nq",
        call_bucket="1",
        serialized_observation="different wrapper text",
        token_length=100,
        document_count=3,
        document_signature=first.document_signature,
    )
    result = select_observation_controls(
        anchor, [first, duplicate], rng=np.random.default_rng(3), count=3
    )
    assert result.unique_control_count == 1
    assert {item.observation_id for item in result.controls} == {"duplicate"}


def test_visible_repeat_requires_every_returned_passage_in_visible_history() -> None:
    assert visible_exact_repeat(["alpha  text", "beta"], ["alpha text", "beta", "old"])
    assert not visible_exact_repeat(["alpha", "new"], ["alpha", "old"])
    assert not visible_exact_repeat([], ["alpha"])


def test_directed_agreement_and_search_utility_cover_all_sign_cases() -> None:
    assert directed_agreement(0.7, 0.4) == (0.4, False)
    assert directed_agreement(-0.7, -0.4) == (-0.4, False)
    assert directed_agreement(0.7, -0.4) == (0.0, True)
    assert directed_agreement(0.0, 0.4) == (0.0, False)

    positive = compute_search_utility(
        real_score=1.0,
        empty_score=0.0,
        control_scores=[0.2, 0.2, 0.2],
        delta=0.1,
        scale=0.5,
    )
    assert positive.utility == pytest.approx(math.tanh(1.4))
    negative = compute_search_utility(
        real_score=0.0,
        empty_score=1.0,
        control_scores=[0.5, 0.5, 0.5],
        delta=0.1,
        scale=0.5,
    )
    assert negative.utility == pytest.approx(-0.1 * math.tanh(0.8))
    conflict = compute_search_utility(
        real_score=0.5,
        empty_score=0.0,
        control_scores=[1.0, 1.0, 1.0],
        delta=0.1,
        scale=0.5,
    )
    assert conflict.utility == 0.0
    assert conflict.contrast_sign_conflict
    assert conflict.zero_reason == "contrast_sign_conflict"


def test_search_utility_repeat_budget_and_score_failures_are_zero() -> None:
    repeated = compute_search_utility(
        real_score=1.0,
        empty_score=0.0,
        control_scores=[0.0, 0.0, 0.0],
        delta=0.1,
        scale=1.0,
        is_visible_exact_repeat=True,
    )
    assert repeated.utility == 0.0 and repeated.zero_reason == "visible_exact_repeat"
    unconsumable = compute_search_utility(
        real_score=1.0,
        empty_score=0.0,
        control_scores=[0.0, 0.0, 0.0],
        delta=0.1,
        scale=1.0,
        observation_consumable=False,
    )
    assert not unconsumable.scored
    assert unconsumable.zero_reason == "unconsumable_observation"
    missing = compute_search_utility(
        real_score=None,
        empty_score=0.0,
        control_scores=[],
        delta=0.1,
        scale=1.0,
    )
    assert missing.utility == 0.0 and missing.zero_reason == "score_unavailable"


def test_frozen_calibration_formulas_and_limited_fallbacks() -> None:
    utility = calibrate_search_utility(
        control_pair_absolute_differences=[0.1] * 10,
        numeric_absolute_differences=[0.01] * 10,
        directed_gaps=[0.5] * 8,
    )
    assert utility.delta == pytest.approx(0.1)
    assert utility.scale == pytest.approx(0.4)
    assert not utility.limited

    query = calibrate_responsibility(
        action_type="search",
        numeric_absolute_differences=[0.001] * 8,
        whole_dependencies=[0.31] * 8,
    )
    assert query.delta == pytest.approx(0.01)
    assert query.scale == pytest.approx(0.3)
    assert not query.limited
    answer = calibrate_responsibility(
        action_type="answer",
        numeric_absolute_differences=[],
        whole_dependencies=[0.5],
    )
    assert answer.limited and answer.scale == 0.2


def test_responsibility_chunk_routing_and_caps_are_bounded() -> None:
    rho = responsibility_strength(0.6, 0.2, delta=0.1, scale=0.3)
    assert rho == pytest.approx(1 - math.exp(-1.0))
    assert responsibility_strength(0.2, 0.6, delta=0.1, scale=0.3) == 0.0

    routing = soft_sparse_chunk_weights(
        1.0, [0.5, 0.9, 1.1], delta=0.05
    )
    assert sum(routing.weights) == pytest.approx(1.0)
    assert routing.weights[0] > routing.weights[1] > routing.weights[2]
    no_mass = soft_sparse_chunk_weights(1.0, [1.0, 1.2], delta=0.01)
    assert no_mass.weights == (0.0, 0.0)

    one = route_action_credit(1.0, rho=1.0, chunk_weights=[1.0])
    two = route_action_credit(1.0, rho=1.0, chunk_weights=[1.0])
    capped = cap_search_trajectory_credits([one, two])
    assert capped.raw_absolute_mass == 4.0
    assert capped.applied_scale == 0.5
    assert capped.capped_absolute_mass == pytest.approx(2.0)
    assert capped.credits[0].action_credit == 0.5
    assert capped.credits[0].think_chunk_credits == (0.5,)

    answer = route_action_credit(
        -1.0, rho=1.0, chunk_weights=[0.25, 0.75], alpha=ANSWER_ALPHA
    )
    assert answer.absolute_mass == pytest.approx(0.5)


def test_metadata_boundary_rejects_evidence_but_allows_answer_labels() -> None:
    safe = {
        "question_id": "q",
        "ground_truth_aliases": ["Paris"],
        "visible_observation": "retrieved text",
    }
    require_no_online_evidence_metadata(safe)
    unsafe = {
        **safe,
        "extra": {
            "metadata": {"supporting_facts": ["hidden"]},
            "new_doc_count": 1,
        },
    }
    assert forbidden_online_metadata_paths(unsafe) == (
        "extra.metadata",
        "extra.metadata.supporting_facts",
        "extra.new_doc_count",
    )
    with pytest.raises(ValueError, match="forbidden evidence metadata"):
        require_no_online_evidence_metadata(unsafe)
