"""Credit-conserving online routing for the repaired Plan05-MIX objective."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from verl.trainer.ppo.urcr_evidence_state import semantic_turn_bucket
from verl.trainer.ppo.urcr_localized import (
    LOCAL_SCALE,
    LOCALIZER_SEED,
    POSITIVE_MASS_FRACTION,
    TRAINING_THINK_MODES,
    WHOLE_FIX_SCALE,
    SupportSelection,
    loo_mass50_support,
    matched_random_support,
    residual_coefficients,
)
from verl.trainer.ppo.urcr_shadow_objectives import shuffled_rho
from verl.trainer.ppo.urcr_sources import URCR_EPS, outcome_advantage


ROUTING_MODES = ("zero", "full", "shuffled", "real")


def build_outcome_anchor(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return detached GRPO credit with no privileged-teacher correction."""
    if advantages.shape != response_mask.shape:
        raise ValueError("URCR outcome advantages and response mask must match")
    outcome = torch.nan_to_num(
        advantages.detach(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return outcome * response_mask.detach().to(dtype=outcome.dtype)


def _positions(mask: list[int]) -> list[int]:
    return [index for index, selected in enumerate(mask) if selected]


def _actual_positions(
    response_mask: torch.Tensor,
    batch_index: int,
    local_positions: list[int],
) -> torch.Tensor:
    valid = torch.nonzero(response_mask[batch_index].bool(), as_tuple=False).flatten()
    if local_positions and max(local_positions) >= len(valid):
        raise ValueError("URCR span points outside valid generated response tokens")
    return valid[local_positions] if local_positions else valid[:0]


def _routing_values(
    frozen_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    responsibility: Mapping[str, Mapping[str, float]],
    *,
    mode: str,
    shuffle_seed: int,
    eps: float,
    require_responsibility: bool,
    outcome_independent: bool = False,
) -> dict[str, float]:
    if mode not in ROUTING_MODES:
        raise ValueError(f"Unknown routing mode: {mode}")
    eligible: list[tuple[str, float, tuple[Any, ...]]] = []
    eligible_by_turn: dict[str, tuple[float, tuple[Any, ...]]] = {}
    for frozen, source in zip(frozen_rows, source_rows):
        turn_uid = str(frozen["turn_uid"])
        effective = bool(
            source.get("source_active")
            and (outcome_independent or abs(outcome_advantage(frozen)) > eps)
            and _positions(
                list(
                    frozen.get("think_content_mask")
                    or frozen.get("think_mask", [])
                )
            )
        )
        if not effective:
            continue
        scored = responsibility.get(turn_uid, {})
        if (
            require_responsibility
            and mode in {"real", "shuffled"}
            and not scored.get("rho_scored", False)
        ):
            raise ValueError(f"Missing detached responsibility score for effective turn {turn_uid}")
        real_rho = float(scored.get("rho", 0.0))
        g2_type = "fact-active" if float(source.get("g1_fact_credit", 0.0)) > 0 else "doc-only"
        stratum = (
            str(source.get("data_source", "unknown")),
            semantic_turn_bucket(int(source["turn_step"])),
            "success" if float(source.get("episode_reward", 0.0)) > 0 else "failure",
            g2_type,
        )
        previous = eligible_by_turn.get(turn_uid)
        if previous is not None:
            previous_rho, previous_stratum = previous
            if previous_stratum != stratum or not math.isclose(
                previous_rho,
                real_rho,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Inconsistent duplicate effective turn in URCR routing: "
                    f"{turn_uid}"
                )
            continue
        eligible_by_turn[turn_uid] = (real_rho, stratum)
        eligible.append((turn_uid, real_rho, stratum))

    if mode == "zero":
        return {turn_uid: 0.0 for turn_uid, _, _ in eligible}
    if mode == "full":
        return {turn_uid: 1.0 for turn_uid, _, _ in eligible}
    if mode == "real":
        return {turn_uid: value for turn_uid, value, _ in eligible}
    values = shuffled_rho(
        [item[1] for item in eligible],
        [item[2] for item in eligible],
        seed=shuffle_seed,
    )
    return {item[0]: float(value) for item, value in zip(eligible, values)}


def apply_urcr_routing(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    frozen_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    responsibility: Mapping[str, Mapping[str, float]],
    *,
    lambda_a: float,
    lambda_r: float,
    mode: str,
    shuffle_seed: int,
    think_credit_mode: str = "whole_old",
    whole_fix_scale: float = WHOLE_FIX_SCALE,
    local_scale: float = LOCAL_SCALE,
    positive_mass_fraction: float = POSITIVE_MASS_FRACTION,
    localizer_seed: int = LOCALIZER_SEED,
    eps: float = URCR_EPS,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Add detached uniform action-total residuals to a GRPO outcome anchor."""
    if len(frozen_rows) != len(advantages) or len(source_rows) != len(advantages):
        raise ValueError("URCR row count must equal actor batch size")
    if min(lambda_a, lambda_r) < 0:
        raise ValueError("URCR lambdas must be nonnegative")
    if think_credit_mode not in TRAINING_THINK_MODES:
        raise ValueError(f"Unknown think-credit mode: {think_credit_mode}")
    routed_rho = _routing_values(
        frozen_rows,
        source_rows,
        responsibility,
        mode=mode,
        shuffle_seed=shuffle_seed,
        eps=eps,
        require_responsibility=lambda_a > 0 and lambda_r > 0,
    )
    output = advantages.detach().clone()
    components: list[dict[str, Any]] = []

    for index, (frozen, source) in enumerate(zip(frozen_rows, source_rows)):
        turn_uid = str(frozen["turn_uid"])
        a_out = outcome_advantage(frozen)
        g2 = float(source.get("g2_applied", 0.0))
        query_local = _positions(list(frozen.get("search_content_mask", [])))
        think_local = _positions(list(frozen.get("think_mask", [])))
        content_mask = frozen.get("think_content_mask")
        content_local = (
            _positions(list(content_mask))
            if content_mask is not None
            else list(think_local)
        )
        query_actual = _actual_positions(response_mask, index, query_local)
        think_actual = _actual_positions(response_mask, index, think_local)
        effective_query = bool(
            source.get("valid_search")
            and g2 > 0.0
            and abs(a_out) > eps
            and len(query_actual) > 0
        )
        c_query = lambda_a * abs(a_out) * g2 if effective_query else 0.0
        query_each = c_query / len(query_actual) if len(query_actual) else 0.0
        if c_query:
            output[index, query_actual] += torch.as_tensor(
                query_each, dtype=output.dtype, device=output.device
            )

        scored = responsibility.get(turn_uid, {})
        rho_real = float(scored.get("rho", 0.0))
        d_mask = float(scored.get("d_mask", 0.0))
        rho_used = float(routed_rho.get(turn_uid, 0.0))
        c_think = lambda_r * rho_used * c_query if len(think_actual) else 0.0
        selected_local = tuple(think_local)
        selected_chunks: tuple[int, ...] = ()
        loo_selected_chunks: tuple[int, ...] = ()
        fallback_reason = None
        random_identity_unavoidable = False
        normalization = "N0"
        think_scale = 1.0
        loo_scores = [float(value) for value in scored.get("chunk_loo_scores", [])]
        chunks = list(frozen.get("think_chunks", []))
        prepare_failure = frozen.get("localizer_prepare_fallback_reason")
        if think_credit_mode != "whole_old":
            normalization = "N1"
            selected_local = tuple(content_local)
            think_scale = whole_fix_scale
        if think_credit_mode in {"loo_mass50", "random_matched"}:
            if prepare_failure:
                true_support = SupportSelection(
                    "loo_mass50",
                    tuple(content_local),
                    tuple(range(len(chunks))),
                    str(prepare_failure),
                    0.0,
                )
            else:
                true_support = loo_mass50_support(
                    chunks=chunks,
                    loo_scores=loo_scores,
                    content_positions=content_local,
                    fraction=positive_mass_fraction,
                )
            loo_selected_chunks = true_support.chunk_indices
            support = true_support
            if think_credit_mode == "random_matched":
                support = matched_random_support(
                    chunks=chunks,
                    content_positions=content_local,
                    true_support=true_support,
                    seed_key=f"{localizer_seed}:{turn_uid}",
                )
                random_identity_unavoidable = support.random_identity_unavoidable
            selected_local = support.response_positions
            selected_chunks = support.chunk_indices
            fallback_reason = support.fallback_reason
            think_scale = local_scale

        valid_response = torch.nonzero(
            response_mask[index].bool(), as_tuple=False
        ).flatten()
        if think_credit_mode == "whole_old":
            think_each = c_think / len(selected_local) if selected_local else 0.0
            expected_allocated = c_think
            if c_think and selected_local:
                selected_actual = valid_response[list(selected_local)]
                output[index, selected_actual] += torch.as_tensor(
                    think_each, dtype=output.dtype, device=output.device
                )
        else:
            coefficients = residual_coefficients(
                response_length=len(valid_response),
                total_credit=c_think,
                positions=selected_local,
                normalization=normalization,
                scale=think_scale,
            )
            think_each = (
                float(coefficients[selected_local[0]]) if selected_local else 0.0
            )
            expected_allocated = float(coefficients.sum())
            if selected_local and c_think:
                output[index, valid_response] += torch.as_tensor(
                    coefficients, dtype=output.dtype, device=output.device
                )

        query_allocated = query_each * len(query_actual)
        think_allocated = think_each * len(selected_local)
        finite_loo = [value for value in loo_scores if math.isfinite(value)]
        positive_loo = [value for value in finite_loo if value > 0]
        components.append(
            {
                "turn_uid": turn_uid,
                "batch_row_index": int(frozen.get("batch_row_index", index)),
                "is_adjustment_copy": bool(frozen.get("is_adjustment_copy", False)),
                "question_uid": str(frozen.get("question_uid", frozen.get("uid", ""))),
                "uid": str(frozen.get("uid", "")),
                "traj_uid": str(frozen["traj_uid"]),
                "turn_step": int(frozen["turn_step"]),
                "data_source": str(frozen.get("data_source", "unknown")),
                "episode_reward": float(frozen.get("episode_reward", 0.0)),
                "valid_search": bool(source.get("valid_search", False)),
                "action_type": str(frozen.get("action_type", "other")),
                "invalid_action": bool(frozen.get("invalid_action", False)),
                "pi_available": bool(source.get("pi_available", False)),
                "a_out": a_out,
                "g_fact": float(source.get("g_fact", 0.0)),
                "g_doc_only": float(source.get("g_doc_only", 0.0)),
                "u_g2": g2,
                "covered_doc_ids_before": list(source.get("covered_doc_ids_before", [])),
                "covered_fact_ids_before": list(source.get("covered_fact_ids_before", [])),
                "new_supporting_doc_ids": list(source.get("new_supporting_doc_ids", [])),
                "new_supporting_fact_ids": list(source.get("new_supporting_fact_ids", [])),
                "doc_only_new_support_ids": list(source.get("doc_only_new_support_ids", [])),
                "redundant_support_doc_ids": list(source.get("redundant_support_doc_ids", [])),
                "redundant_support_fact_ids": list(source.get("redundant_support_fact_ids", [])),
                "source_active": bool(source.get("source_active", False)),
                "effective_query": effective_query,
                "rho_scored": bool(scored.get("rho_scored", False)),
                "d_mask": d_mask,
                "rho_real": rho_real,
                "rho_used": rho_used,
                "routing_mode": mode,
                "query_token_count": len(query_actual),
                "think_token_count": len(selected_local),
                "whole_think_token_count": len(think_actual),
                "think_content_token_count": len(content_local),
                "think_tag_token_count": max(0, len(think_actual) - len(content_local)),
                "c_query_total": c_query,
                "c_query_per_token": query_each,
                "c_think_total": c_think,
                "c_think_per_token": think_each,
                "query_conservation_error": float(query_allocated - c_query),
                "think_conservation_error": float(
                    think_allocated - expected_allocated
                ),
                "think_credit_mode": think_credit_mode,
                "think_normalization": normalization,
                "think_scale": think_scale,
                "think_allocated_coefficient_sum": think_allocated,
                "selected_token_fraction": (
                    len(selected_local) / len(content_local) if content_local else 0.0
                ),
                "selected_chunk_count": len(selected_chunks),
                "selected_chunk_indices": list(selected_chunks),
                "loo_selected_chunk_indices": list(loo_selected_chunks),
                "localizer_active": bool(
                    think_credit_mode in {"loo_mass50", "random_matched"}
                    and fallback_reason is None
                ),
                "fallback_reason": fallback_reason,
                "random_identity_unavoidable": random_identity_unavoidable,
                "loo_score_count": len(loo_scores),
                "loo_score_mean": (
                    float(sum(finite_loo) / len(finite_loo)) if finite_loo else 0.0
                ),
                "loo_score_nonfinite_count": len(loo_scores) - len(finite_loo),
                "loo_positive_score_mass": float(sum(positive_loo)),
                "effective_think": bool(c_think > 0.0 and selected_local),
            }
        )

    trajectory_outcomes: dict[str, dict[str, bool]] = {}
    for row in components:
        trajectory_outcomes.setdefault(row["question_uid"], {})[row["traj_uid"]] = bool(
            row["episode_reward"] > 0
        )
    group_labels = {}
    for question_uid, outcomes in trajectory_outcomes.items():
        values = list(outcomes.values())
        group_labels[question_uid] = (
            "all-success"
            if all(values)
            else "all-fail"
            if not any(values)
            else "mixed-outcome"
        )
    for row in components:
        row["outcome_group"] = group_labels[row["question_uid"]]

    return output * response_mask, components


def build_v2_local_routing(
    response_mask: torch.Tensor,
    frozen_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    responsibility: Mapping[str, Mapping[str, float]],
    *,
    base_query_reward: float,
    lambda_r: float,
    mode: str,
    shuffle_seed: int,
    think_length_ref: float,
    positive_mass_fraction: float = POSITIVE_MASS_FRACTION,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Build detached v2 action masks/rewards without modifying GRPO advantages."""
    if len(frozen_rows) != len(response_mask) or len(source_rows) != len(response_mask):
        raise ValueError("URCR v2 row count must equal actor batch size")
    if base_query_reward < 0 or lambda_r < 0 or think_length_ref <= 0:
        raise ValueError("URCR v2 reward scales and think_length_ref must be valid")
    routed_rho = _routing_values(
        frozen_rows,
        source_rows,
        responsibility,
        mode=mode,
        shuffle_seed=shuffle_seed,
        eps=URCR_EPS,
        require_responsibility=base_query_reward > 0 and lambda_r > 0,
        outcome_independent=True,
    )
    shape = response_mask.shape
    device = response_mask.device
    query_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    think_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    query_eligible = torch.zeros(shape[0], dtype=torch.float32, device=device)
    think_eligible = torch.zeros(shape[0], dtype=torch.float32, device=device)
    query_reward = torch.zeros(shape[0], dtype=torch.float32, device=device)
    think_reward = torch.zeros(shape[0], dtype=torch.float32, device=device)
    selected_length = torch.zeros(shape[0], dtype=torch.float32, device=device)
    rho_tensor = torch.zeros(shape[0], dtype=torch.float32, device=device)
    components: list[dict[str, Any]] = []

    for index, (frozen, source) in enumerate(zip(frozen_rows, source_rows)):
        if source.get("support_reward_version") != "v2_fixed_local":
            raise ValueError("build_v2_local_routing received a non-v2 source row")
        turn_uid = str(frozen["turn_uid"])
        valid_response = torch.nonzero(response_mask[index].bool(), as_tuple=False).flatten()
        query_local = _positions(list(frozen.get("search_content_mask", [])))
        content_local = _positions(
            list(
                frozen.get("think_content_mask")
                or frozen.get("think_mask", [])
            )
        )
        query_actual = _actual_positions(response_mask, index, query_local)
        is_query_eligible = bool(
            source.get("query_local_eligible", False) and len(query_actual) > 0
        )
        is_think_eligible = bool(is_query_eligible and len(content_local) > 0)
        utility = float(source.get("support_utility_v2", 0.0))
        r_query = base_query_reward * utility if is_query_eligible else 0.0
        scored = responsibility.get(turn_uid, {})
        rho_real = float(scored.get("rho", 0.0))
        rho_used = float(routed_rho.get(turn_uid, 0.0))
        r_think = lambda_r * rho_used * r_query if is_think_eligible else 0.0
        if r_think > 0.0 and not scored.get("rho_scored", False):
            raise ValueError(
                f"Missing detached chunk responsibility score for v2 turn {turn_uid}"
            )

        selected_local = tuple(content_local)
        selected_chunks: tuple[int, ...] = ()
        loo_selected_chunks: tuple[int, ...] = ()
        fallback_reason = None
        loo_scores = [float(value) for value in scored.get("chunk_loo_scores", [])]
        chunks = list(frozen.get("think_chunks", []))
        prepare_failure = frozen.get("localizer_prepare_fallback_reason")
        if r_think > 0.0:
            if prepare_failure:
                true_support = SupportSelection(
                    "loo_mass50",
                    tuple(content_local),
                    tuple(range(len(chunks))),
                    str(prepare_failure),
                    0.0,
                )
            else:
                true_support = loo_mass50_support(
                    chunks=chunks,
                    loo_scores=loo_scores,
                    content_positions=content_local,
                    fraction=positive_mass_fraction,
                )
            loo_selected_chunks = true_support.chunk_indices
            support = true_support
            selected_local = support.response_positions
            selected_chunks = support.chunk_indices
            fallback_reason = support.fallback_reason

        selected_actual = (
            valid_response[list(selected_local)] if selected_local else valid_response[:0]
        )
        if is_query_eligible:
            query_mask[index, query_actual] = True
            query_eligible[index] = 1.0
        if is_think_eligible:
            think_eligible[index] = 1.0
        if selected_actual.numel():
            think_mask[index, selected_actual] = True
        query_reward[index] = r_query
        think_reward[index] = r_think
        selected_length[index] = float(len(selected_local))
        rho_tensor[index] = rho_used
        components.append(
            {
                "turn_uid": turn_uid,
                "batch_row_index": int(frozen.get("batch_row_index", index)),
                "is_adjustment_copy": bool(frozen.get("is_adjustment_copy", False)),
                "question_uid": str(frozen.get("question_uid", frozen.get("uid", ""))),
                "uid": str(frozen.get("uid", "")),
                "traj_uid": str(frozen["traj_uid"]),
                "turn_step": int(frozen["turn_step"]),
                "data_source": str(frozen.get("data_source", "unknown")),
                "episode_reward": float(frozen.get("episode_reward", 0.0)),
                "a_out": outcome_advantage(frozen),
                "valid_search": bool(source.get("valid_search", False)),
                "action_type": str(frozen.get("action_type", "other")),
                "invalid_action": bool(frozen.get("invalid_action", False)),
                "pi_available": bool(source.get("pi_available", False)),
                "support_reward_version": "v2_fixed_local",
                "support_utility_v2": utility,
                "support_hit_type": str(source.get("support_hit_type", "unknown")),
                "support_new_doc_count": int(source.get("support_new_doc_count", 0)),
                "support_new_fact_count": int(source.get("support_new_fact_count", 0)),
                "support_reward_query": r_query,
                "support_reward_think": r_think,
                "query_local_eligible": is_query_eligible,
                "think_local_eligible": is_think_eligible,
                "source_active": bool(source.get("source_active", False)),
                "effective_query": bool(r_query > 0.0),
                "effective_think": bool(r_think > 0.0 and selected_local),
                "u_g2": utility,
                "c_query_total": r_query,
                "c_think_total": r_think,
                "query_conservation_error": 0.0,
                "think_conservation_error": 0.0,
                "rho_score_required": bool(source.get("rho_score_required", False)),
                "rho_scored": bool(scored.get("rho_scored", False)),
                "d_mask": float(scored.get("d_mask", 0.0)),
                "rho_real": rho_real,
                "rho_used": rho_used,
                "routing_mode": mode,
                "query_token_count": int(len(query_actual)),
                "selected_think_content_len": int(len(selected_local)),
                "think_content_token_count": int(len(content_local)),
                "think_token_count": int(len(selected_local)),
                "think_tag_token_count": max(
                    0,
                    len(_positions(list(frozen.get("think_mask", []))))
                    - len(content_local),
                ),
                "think_length_ref": float(think_length_ref),
                "think_credit_mode": "loo_mass50",
                "think_normalization": "N1-reference",
                "think_scale": float(
                    math.sqrt(len(selected_local) / think_length_ref)
                    if selected_local
                    else 0.0
                ),
                "think_allocated_coefficient_sum": r_think,
                "selected_token_fraction": (
                    len(selected_local) / len(content_local) if content_local else 0.0
                ),
                "selected_chunk_count": len(selected_chunks),
                "selected_chunk_indices": list(selected_chunks),
                "loo_selected_chunk_indices": list(loo_selected_chunks),
                "localizer_active": bool(
                    r_think > 0.0 and fallback_reason is None and len(chunks) > 1
                ),
                "fallback_reason": fallback_reason,
                "random_identity_unavoidable": False,
                "loo_score_count": len(loo_scores),
                "loo_score_mean": float(
                    sum(value for value in loo_scores if math.isfinite(value))
                    / max(1, sum(math.isfinite(value) for value in loo_scores))
                ),
                "loo_score_nonfinite_count": int(
                    sum(not math.isfinite(value) for value in loo_scores)
                ),
                "loo_positive_score_mass": float(
                    sum(value for value in loo_scores if math.isfinite(value) and value > 0)
                ),
            }
        )

    return (
        {
            "urcr_v2_query_mask": query_mask,
            "urcr_v2_think_mask": think_mask,
            "urcr_v2_query_eligible": query_eligible,
            "urcr_v2_think_eligible": think_eligible,
            "urcr_v2_query_reward": query_reward,
            "urcr_v2_think_reward": think_reward,
            "urcr_v2_selected_think_length": selected_length,
            "urcr_v2_rho": rho_tensor,
        },
        components,
    )
