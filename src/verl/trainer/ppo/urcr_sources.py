"""Frozen Plan 05 configuration and online G2 evidence-acquisition source."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from verl.trainer.ppo.urcr_evidence_state import ETA_DOC_ONLY, build_evidence_state_rows
from verl.trainer.ppo.urcr_localized import (
    LOCAL_SCALE,
    LOCALIZER_SEED,
    POSITIVE_MASS_FRACTION,
    TRAINING_THINK_MODES,
    WHOLE_FIX_SCALE,
)


LAMBDA_A = 0.25
RESPONSIBILITY_SCALE = 0.6201
LAMBDA_R = 1.0
URCR_EPS = 1e-8
SHUFFLE_SEED = 20260820
METHOD_ROUTING_MODE = {
    "g2_query_only": "zero",
    "g2_full": "full",
    "g2_shuffled": "shuffled",
    "g2_real": "real",
}
LEGACY_SUPPORT_REWARD_VERSION = "legacy_outcome_scaled_fractional"
V2_SUPPORT_REWARD_VERSION = "v2_fixed_local"
V2_UTILITY_MODES = (
    "binary_hierarchical",
    "binary_doc_primary",
    "binary_fact_only",
)


@dataclass(frozen=True)
class SupportRewardConfig:
    enabled: bool
    version: str
    utility_mode: str
    base_query_reward: float
    fact_utility: float
    doc_only_utility: float
    multihit_bonus: float
    multihit_cap: float


@dataclass(frozen=True)
class LocalObjectiveConfig:
    enabled: bool
    mode: str
    query_reduction: str
    think_reduction: str
    think_length_ref: float
    query_loss_weight: float
    think_loss_weight: float
    lambda_r: float
    local_max: float
    warmup_steps: int


@dataclass(frozen=True)
class SupportUtility:
    utility: float
    hit_type: str
    new_doc_count: int
    new_fact_count: int
    is_repeat_only: bool
    has_metadata: bool
    is_valid_search: bool


@dataclass(frozen=True)
class Plan05Config:
    enabled: bool
    method: str
    source_enabled: bool
    lambda_a: float
    responsibility_scale: float
    lazy_responsibility: bool
    routing_mode: str
    lambda_r: float
    shuffle_seed: int
    think_credit_mode: str
    whole_fix_scale: float
    local_scale: float
    positive_mass_fraction: float
    localizer_seed: int
    save_turn_components: bool
    turn_component_steps: frozenset[int] | None
    compute_shadow_modes: bool
    compute_q0_shadow: bool
    save_surrogate_coefficients: bool
    audit_output_dir: str | None
    capture_update_summary: bool
    capture_update_vectors: bool
    save_update_reference: bool
    compare_update_reference: bool
    update_reference_dir: str | None
    support_reward: SupportRewardConfig
    local_objective: LocalObjectiveConfig

    @property
    def uses_v2_fixed_support_reward(self) -> bool:
        return bool(
            self.enabled
            and self.support_reward.enabled
            and self.support_reward.version == V2_SUPPORT_REWARD_VERSION
        )


def _disabled_support_reward() -> SupportRewardConfig:
    return SupportRewardConfig(
        enabled=False,
        version=LEGACY_SUPPORT_REWARD_VERSION,
        utility_mode="legacy_fractional",
        base_query_reward=0.0,
        fact_utility=1.0,
        doc_only_utility=ETA_DOC_ONLY,
        multihit_bonus=0.0,
        multihit_cap=1.0,
    )


def _disabled_local_objective() -> LocalObjectiveConfig:
    return LocalObjectiveConfig(
        enabled=False,
        mode="disabled",
        query_reduction="span_mean",
        think_reduction="n1_reference",
        think_length_ref=1.0,
        query_loss_weight=0.0,
        think_loss_weight=0.0,
        lambda_r=0.0,
        local_max=0.0,
        warmup_steps=0,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None or not hasattr(value, "get"):
        raise ValueError(f"algorithm.urcr.{name} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, float):
        matches = isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"{path} must be {expected!r}, got {actual!r}")


def validate_plan05_config(value: Any) -> Plan05Config:
    """Validate implemented Plan 05 options and reject semantic drift."""
    cfg = _mapping(value, "")
    enabled = bool(cfg.get("enable", False))
    audit = _mapping(cfg.get("audit", {}), "audit")
    capture_update_vectors = bool(audit.get("capture_update_vectors", False))
    capture_update_summary = bool(audit.get("capture_update_summary", False))
    save_update_reference = bool(audit.get("save_update_reference", False))
    compare_update_reference = bool(audit.get("compare_update_reference", False))
    update_reference_dir = (
        str(audit.get("update_reference_dir")) if audit.get("update_reference_dir") else None
    )
    audit_output_dir = str(audit.get("output_dir")) if audit.get("output_dir") else None
    if capture_update_vectors:
        if capture_update_summary:
            raise ValueError(
                "Plan05-MIX update audit cannot enable summary and vector capture together"
            )
        if save_update_reference == compare_update_reference:
            raise ValueError(
                "Plan 05 update audit requires exactly one of "
                "save_update_reference/compare_update_reference"
            )
        if not update_reference_dir:
            raise ValueError("Plan 05 update audit requires audit.update_reference_dir")
        if not audit_output_dir:
            raise ValueError("Plan 05 update audit requires audit.output_dir")
    elif save_update_reference or compare_update_reference:
        raise ValueError("Plan 05 update reference flags require capture_update_vectors=true")

    if not enabled:
        return Plan05Config(
            enabled=False,
            method="g2_query_only",
            source_enabled=False,
            lambda_a=0.0,
            responsibility_scale=RESPONSIBILITY_SCALE,
            lazy_responsibility=True,
            routing_mode="zero",
            lambda_r=0.0,
            shuffle_seed=SHUFFLE_SEED,
            think_credit_mode="whole_old",
            whole_fix_scale=WHOLE_FIX_SCALE,
            local_scale=LOCAL_SCALE,
            positive_mass_fraction=POSITIVE_MASS_FRACTION,
            localizer_seed=LOCALIZER_SEED,
            save_turn_components=False,
            turn_component_steps=None,
            compute_shadow_modes=False,
            compute_q0_shadow=False,
            save_surrogate_coefficients=False,
            audit_output_dir=audit_output_dir,
            capture_update_summary=capture_update_summary,
            capture_update_vectors=capture_update_vectors,
            save_update_reference=save_update_reference,
            compare_update_reference=compare_update_reference,
            update_reference_dir=update_reference_dir,
            support_reward=_disabled_support_reward(),
            local_objective=_disabled_local_objective(),
        )

    method = str(cfg.get("method"))
    if method not in METHOD_ROUTING_MODE:
        raise ValueError(
            "algorithm.urcr.method must be one of "
            f"{sorted(METHOD_ROUTING_MODE)}, got {method!r}"
        )
    source = _mapping(cfg.get("source"), "source")
    responsibility = _mapping(cfg.get("responsibility"), "responsibility")
    routing = _mapping(cfg.get("routing"), "routing")

    support_reward_value = cfg.get("support_reward")
    if support_reward_value is None:
        support_reward = SupportRewardConfig(
            enabled=True,
            version=LEGACY_SUPPORT_REWARD_VERSION,
            utility_mode="legacy_fractional",
            base_query_reward=0.0,
            fact_utility=1.0,
            doc_only_utility=ETA_DOC_ONLY,
            multihit_bonus=0.0,
            multihit_cap=1.0,
        )
        local_objective = _disabled_local_objective()
    else:
        support_cfg = _mapping(support_reward_value, "support_reward")
        support_enabled = bool(support_cfg.get("enable", True))
        support_version = str(
            support_cfg.get("version", LEGACY_SUPPORT_REWARD_VERSION)
        )
        if support_version not in {
            LEGACY_SUPPORT_REWARD_VERSION,
            V2_SUPPORT_REWARD_VERSION,
        }:
            raise ValueError(
                "algorithm.urcr.support_reward.version must be "
                f"{LEGACY_SUPPORT_REWARD_VERSION!r} or {V2_SUPPORT_REWARD_VERSION!r}"
            )
        if support_version == V2_SUPPORT_REWARD_VERSION:
            utility_mode = str(support_cfg.get("utility_mode"))
            if utility_mode not in V2_UTILITY_MODES:
                raise ValueError(
                    "algorithm.urcr.support_reward.utility_mode must be one of "
                    f"{list(V2_UTILITY_MODES)}, got {utility_mode!r}"
                )
            for key in (
                "repeat_reward",
                "miss_reward",
                "invalid_reward",
                "no_metadata_reward",
                "multihit_bonus",
            ):
                _require_equal(
                    support_cfg.get(key, 0.0),
                    0.0,
                    f"algorithm.urcr.support_reward.{key}",
                )
            _require_equal(
                support_cfg.get("detach", True),
                True,
                "algorithm.urcr.support_reward.detach",
            )
            _require_equal(
                support_cfg.get("insert_into_terminal_reward", False),
                False,
                "algorithm.urcr.support_reward.insert_into_terminal_reward",
            )
            _require_equal(
                support_cfg.get("grpo_group_normalize", False),
                False,
                "algorithm.urcr.support_reward.grpo_group_normalize",
            )
            _require_equal(
                support_cfg.get("outcome_interaction", "additive"),
                "additive",
                "algorithm.urcr.support_reward.outcome_interaction",
            )
            base_query_reward = float(support_cfg.get("base_query_reward"))
            if base_query_reward not in (0.125, 0.25, 0.5):
                raise ValueError(
                    "algorithm.urcr.support_reward.base_query_reward must be 0.125, 0.25, or 0.5"
                )
            fact_utility = float(support_cfg.get("fact_utility", 1.0))
            _require_equal(
                fact_utility,
                1.0,
                "algorithm.urcr.support_reward.fact_utility",
            )
            doc_only_utility = float(support_cfg.get("doc_only_utility", 0.5))
            if doc_only_utility not in (0.0, 0.5, 1.0):
                raise ValueError(
                    "algorithm.urcr.support_reward.doc_only_utility must be 0, 0.5, or 1"
                )
            multihit_cap = float(support_cfg.get("multihit_cap", 1.25))
            if multihit_cap < 1.0:
                raise ValueError(
                    "algorithm.urcr.support_reward.multihit_cap must be at least 1"
                )
            support_reward = SupportRewardConfig(
                enabled=support_enabled,
                version=support_version,
                utility_mode=utility_mode,
                base_query_reward=base_query_reward,
                fact_utility=fact_utility,
                doc_only_utility=doc_only_utility,
                multihit_bonus=0.0,
                multihit_cap=multihit_cap,
            )

            local_cfg = _mapping(cfg.get("local_objective"), "local_objective")
            _require_equal(
                local_cfg.get("mode"),
                "separate_eligible_action_mean",
                "algorithm.urcr.local_objective.mode",
            )
            _require_equal(
                local_cfg.get("query_reduction"),
                "span_mean",
                "algorithm.urcr.local_objective.query_reduction",
            )
            _require_equal(
                local_cfg.get("think_reduction"),
                "n1_reference",
                "algorithm.urcr.local_objective.think_reduction",
            )
            _require_equal(
                local_cfg.get("think_length_ref_source"),
                "frozen_median_selected_content",
                "algorithm.urcr.local_objective.think_length_ref_source",
            )
            _require_equal(
                local_cfg.get("think_content_only", True),
                True,
                "algorithm.urcr.local_objective.think_content_only",
            )
            _require_equal(
                local_cfg.get("use_existing_ppo_clip", True),
                True,
                "algorithm.urcr.local_objective.use_existing_ppo_clip",
            )
            _require_equal(
                local_cfg.get("extra_actor_forward", False),
                False,
                "algorithm.urcr.local_objective.extra_actor_forward",
            )
            _require_equal(
                local_cfg.get("old_s_local_compatible", False),
                False,
                "algorithm.urcr.local_objective.old_s_local_compatible",
            )
            think_length_ref = float(local_cfg.get("think_length_ref"))
            if not math.isfinite(think_length_ref) or think_length_ref <= 0:
                raise ValueError(
                    "algorithm.urcr.local_objective.think_length_ref must be positive"
                )
            warmup_steps = int(local_cfg.get("warmup_steps", 30))
            if warmup_steps < 0:
                raise ValueError(
                    "algorithm.urcr.local_objective.warmup_steps must be nonnegative"
                )
            if "local_max" not in local_cfg:
                raise ValueError(
                    "algorithm.urcr.local_objective.local_max must be explicit"
                )
            local_max = float(local_cfg.get("local_max"))
            if not math.isfinite(local_max) or not 0.0 < local_max <= 1.0:
                raise ValueError(
                    "algorithm.urcr.local_objective.local_max must be in (0, 1]"
                )
            local_objective = LocalObjectiveConfig(
                enabled=support_enabled,
                mode="separate_eligible_action_mean",
                query_reduction="span_mean",
                think_reduction="n1_reference",
                think_length_ref=think_length_ref,
                query_loss_weight=float(local_cfg.get("query_loss_weight", 1.0)),
                think_loss_weight=float(local_cfg.get("think_loss_weight", 1.0)),
                lambda_r=float(local_cfg.get("lambda_r", 1.0)),
                local_max=local_max,
                warmup_steps=warmup_steps,
            )
            _require_equal(
                local_objective.query_loss_weight,
                1.0,
                "algorithm.urcr.local_objective.query_loss_weight",
            )
            _require_equal(
                local_objective.think_loss_weight,
                1.0,
                "algorithm.urcr.local_objective.think_loss_weight",
            )
            _require_equal(
                local_objective.lambda_r,
                1.0,
                "algorithm.urcr.local_objective.lambda_r",
            )
        else:
            support_reward = SupportRewardConfig(
                enabled=support_enabled,
                version=support_version,
                utility_mode="legacy_fractional",
                base_query_reward=0.0,
                fact_utility=1.0,
                doc_only_utility=ETA_DOC_ONLY,
                multihit_bonus=0.0,
                multihit_cap=1.0,
            )
            local_objective = _disabled_local_objective()

    _require_equal(source.get("name"), "g2_hierarchical_acquisition", "algorithm.urcr.source.name")
    _require_equal(source.get("eta_doc_only"), ETA_DOC_ONLY, "algorithm.urcr.source.eta_doc_only")
    _require_equal(source.get("negative_no_hit"), False, "algorithm.urcr.source.negative_no_hit")
    _require_equal(source.get("normalize"), False, "algorithm.urcr.source.normalize")
    lambda_a = float(source.get("lambda_a"))
    if lambda_a not in (0.0, LAMBDA_A):
        raise ValueError(f"algorithm.urcr.source.lambda_a must be 0 or {LAMBDA_A}, got {lambda_a}")

    _require_equal(
        responsibility.get("name"),
        "query_to_current_think_attention_block",
        "algorithm.urcr.responsibility.name",
    )
    _require_equal(responsibility.get("mapping"), "exp_positive", "algorithm.urcr.responsibility.mapping")
    _require_equal(responsibility.get("scale"), RESPONSIBILITY_SCALE, "algorithm.urcr.responsibility.scale")
    _require_equal(responsibility.get("no_grad"), True, "algorithm.urcr.responsibility.no_grad")
    _require_equal(
        responsibility.get("lazy_on_effective_source"),
        True,
        "algorithm.urcr.responsibility.lazy_on_effective_source",
    )
    if support_reward.version == V2_SUPPORT_REWARD_VERSION:
        _require_equal(
            responsibility.get("score_eligibility"),
            "source_active_independent_of_a_out",
            "algorithm.urcr.responsibility.score_eligibility",
        )
    if "micro_batch_size" in responsibility:
        raise ValueError(
            "algorithm.urcr.responsibility.micro_batch_size is obsolete; "
            "the scorer inherits actor_rollout_ref.rollout."
            "log_prob_micro_batch_size_per_gpu"
        )

    mode = str(routing.get("mode"))
    if mode not in {"zero", "full", "shuffled", "real"}:
        raise ValueError(f"algorithm.urcr.routing.mode is not implemented: {mode!r}")
    _require_equal(
        mode,
        METHOD_ROUTING_MODE[method],
        f"algorithm.urcr.routing.mode for method={method}",
    )
    lambda_r = float(routing.get("lambda_r"))
    if lambda_r not in (0.0, LAMBDA_R):
        raise ValueError(f"algorithm.urcr.routing.lambda_r must be 0 or {LAMBDA_R}, got {lambda_r}")
    _require_equal(routing.get("query_allocation"), "uniform_action_total", "algorithm.urcr.routing.query_allocation")
    _require_equal(routing.get("think_allocation"), "uniform_action_total", "algorithm.urcr.routing.think_allocation")
    _require_equal(routing.get("residual_only"), True, "algorithm.urcr.routing.residual_only")
    _require_equal(routing.get("normalize"), False, "algorithm.urcr.routing.normalize")
    think_credit_mode = str(routing.get("think_credit_mode", "whole_old"))
    if think_credit_mode not in TRAINING_THINK_MODES:
        raise ValueError(
            "algorithm.urcr.routing.think_credit_mode must be one of "
            f"{list(TRAINING_THINK_MODES)}, got {think_credit_mode!r}"
        )
    if support_reward.version == V2_SUPPORT_REWARD_VERSION:
        _require_equal(
            think_credit_mode,
            "loo_mass50",
            "algorithm.urcr.routing.think_credit_mode",
        )
        protection = _mapping(cfg.get("protection"), "protection")
        _require_equal(
            protection.get("protect_verified_support_query"),
            False,
            "algorithm.urcr.protection.protect_verified_support_query",
        )
        _require_equal(
            protection.get("protect_routed_think"),
            False,
            "algorithm.urcr.protection.protect_routed_think",
        )
    if support_reward.version == V2_SUPPORT_REWARD_VERSION:
        if routing.get("local_scale") is not None:
            raise ValueError(
                "algorithm.urcr.routing.local_scale is forbidden for v2_fixed_local"
            )
        whole_fix_scale = WHOLE_FIX_SCALE
        local_scale = LOCAL_SCALE
    else:
        whole_fix_scale = float(routing.get("whole_fix_scale", WHOLE_FIX_SCALE))
        local_scale = float(routing.get("local_scale", LOCAL_SCALE))
    positive_mass_fraction = float(
        routing.get("positive_mass_fraction", POSITIVE_MASS_FRACTION)
    )
    localizer_seed = int(routing.get("localizer_seed", LOCALIZER_SEED))
    if support_reward.version != V2_SUPPORT_REWARD_VERSION:
        _require_equal(
            whole_fix_scale,
            WHOLE_FIX_SCALE,
            "algorithm.urcr.routing.whole_fix_scale",
        )
        _require_equal(local_scale, LOCAL_SCALE, "algorithm.urcr.routing.local_scale")
    _require_equal(
        positive_mass_fraction,
        POSITIVE_MASS_FRACTION,
        "algorithm.urcr.routing.positive_mass_fraction",
    )
    _require_equal(
        localizer_seed,
        LOCALIZER_SEED,
        "algorithm.urcr.routing.localizer_seed",
    )
    raw_turn_component_steps = audit.get("turn_component_steps")
    turn_component_steps = None
    if raw_turn_component_steps is not None:
        turn_component_steps = frozenset(int(step) for step in raw_turn_component_steps)
        if any(step <= 0 for step in turn_component_steps):
            raise ValueError("algorithm.urcr.audit.turn_component_steps must be positive")

    return Plan05Config(
        enabled=True,
        method=method,
        source_enabled=bool(source.get("enable", True)),
        lambda_a=lambda_a,
        responsibility_scale=RESPONSIBILITY_SCALE,
        lazy_responsibility=True,
        routing_mode=mode,
        lambda_r=lambda_r,
        shuffle_seed=int(routing.get("shuffle_seed", SHUFFLE_SEED)),
        think_credit_mode=think_credit_mode,
        whole_fix_scale=whole_fix_scale,
        local_scale=local_scale,
        positive_mass_fraction=positive_mass_fraction,
        localizer_seed=localizer_seed,
        save_turn_components=bool(audit.get("save_turn_components", True)),
        turn_component_steps=turn_component_steps,
        compute_shadow_modes=bool(audit.get("compute_shadow_modes", True)),
        compute_q0_shadow=bool(audit.get("compute_q0_shadow", True)),
        save_surrogate_coefficients=bool(audit.get("save_surrogate_coefficients", True)),
        audit_output_dir=audit_output_dir,
        capture_update_summary=capture_update_summary,
        capture_update_vectors=capture_update_vectors,
        save_update_reference=save_update_reference,
        compare_update_reference=compare_update_reference,
        update_reference_dir=update_reference_dir,
        support_reward=support_reward,
        local_objective=local_objective,
    )


def compute_v2_support_utility(
    turn: Mapping[str, Any],
    *,
    utility_mode: str = "binary_hierarchical",
    kappa_doc: float = 0.5,
) -> SupportUtility:
    """Map one already-causal evidence-state row to the fixed v2 utility."""
    if utility_mode not in V2_UTILITY_MODES:
        raise ValueError(f"Unknown v2 support utility mode: {utility_mode!r}")
    has_metadata = bool(turn.get("evidence_available", False))
    is_valid_search = bool(turn.get("valid_search", False))
    new_doc_count = int(turn.get("new_support_doc_count", 0))
    new_fact_count = int(turn.get("new_support_fact_count", 0))
    repeated = bool(
        int(turn.get("redundant_support_doc_count", 0)) > 0
        or int(turn.get("redundant_support_fact_count", 0)) > 0
    )

    def zero(hit_type: str) -> SupportUtility:
        return SupportUtility(
            utility=0.0,
            hit_type=hit_type,
            new_doc_count=new_doc_count,
            new_fact_count=new_fact_count,
            is_repeat_only=bool(repeated and not new_doc_count and not new_fact_count),
            has_metadata=has_metadata,
            is_valid_search=is_valid_search,
        )

    if not has_metadata:
        return zero("no_metadata")
    if not is_valid_search:
        return zero("invalid")
    if new_fact_count > 0:
        return SupportUtility(
            utility=1.0,
            hit_type="fact",
            new_doc_count=new_doc_count,
            new_fact_count=new_fact_count,
            is_repeat_only=False,
            has_metadata=True,
            is_valid_search=True,
        )
    if new_doc_count > 0:
        utility = (
            0.0
            if utility_mode == "binary_fact_only"
            else 1.0
            if utility_mode == "binary_doc_primary"
            else float(kappa_doc)
        )
        return SupportUtility(
            utility=utility,
            hit_type="doc_only",
            new_doc_count=new_doc_count,
            new_fact_count=0,
            is_repeat_only=False,
            has_metadata=True,
            is_valid_search=True,
        )
    return zero("repeat" if repeated else "no_hit")


def build_online_g2_rows(
    frozen_rows: list[dict[str, Any]],
    *,
    source_enabled: bool = True,
    support_reward: SupportRewardConfig | None = None,
) -> list[dict[str, Any]]:
    """Return Plan04-identical evidence state in original batch row order."""
    unique_rows: list[dict[str, Any]] = []
    canonical_by_turn: dict[str, dict[str, Any]] = {}
    identity_fields = (
        "traj_uid",
        "turn_step",
        "question_uid",
        "response_token_ids",
        "observation_text",
        "metadata_json",
        "episode_reward",
        "outcome_advantage_token",
    )
    for row in frozen_rows:
        turn_uid = str(row["turn_uid"])
        canonical = canonical_by_turn.get(turn_uid)
        if canonical is None:
            canonical_by_turn[turn_uid] = row
            unique_rows.append(row)
            continue
        if any(canonical.get(field) != row.get(field) for field in identity_fields):
            raise ValueError(f"Non-identical rows collide on Plan 05 turn_uid {turn_uid}")

    states = build_evidence_state_rows(unique_rows, include_future_audit=False)
    by_turn = {row["turn_uid"]: row for row in states}
    if len(by_turn) != len(unique_rows):
        raise ValueError("Online G2 requires one unique evidence-state row per frozen turn")

    output = []
    for frozen in frozen_rows:
        turn_uid = str(frozen["turn_uid"])
        if turn_uid not in by_turn:
            raise ValueError(f"Missing evidence state for {turn_uid}")
        row = dict(by_turn[turn_uid])
        raw_g2 = float(row["g2_hierarchical_credit"])
        row["g_fact"] = float(row["g1_fact_credit"])
        row["g_doc_only"] = (
            len(row["doc_only_new_support_ids"])
            / max(1, len(row["gold_supporting_doc_ids"]))
            if row["evidence_available"]
            else 0.0
        )
        row["g2_raw"] = raw_g2
        row["g2_applied"] = raw_g2 if source_enabled else 0.0
        if (
            support_reward is not None
            and support_reward.enabled
            and support_reward.version == V2_SUPPORT_REWARD_VERSION
        ):
            fixed = compute_v2_support_utility(
                row,
                utility_mode=support_reward.utility_mode,
                kappa_doc=support_reward.doc_only_utility,
            )
            utility = float(fixed.utility) if source_enabled else 0.0
            query_local_eligible = bool(
                source_enabled
                and fixed.has_metadata
                and fixed.is_valid_search
                and not frozen.get("is_adjustment_copy", False)
            )
            row.update(
                {
                    "support_reward_version": V2_SUPPORT_REWARD_VERSION,
                    "support_utility_v2": utility,
                    "support_hit_type": fixed.hit_type,
                    "support_new_doc_count": fixed.new_doc_count,
                    "support_new_fact_count": fixed.new_fact_count,
                    "support_repeat_only": fixed.is_repeat_only,
                    "support_has_metadata": fixed.has_metadata,
                    "query_local_eligible": query_local_eligible,
                    "rho_score_required": bool(
                        query_local_eligible
                        and utility > 0.0
                        and any(frozen.get("think_content_mask") or frozen.get("think_mask") or [])
                    ),
                }
            )
            row["source_active"] = bool(
                row["query_local_eligible"] and utility > 0.0
            )
        else:
            row["support_reward_version"] = LEGACY_SUPPORT_REWARD_VERSION
            row["source_active"] = bool(
                row["valid_search"] and row["g2_applied"] > 0.0
            )
        output.append(row)
    return output


def outcome_advantage(row: Mapping[str, Any], *, atol: float = 1e-6) -> float:
    """Extract the scalar GRPO outcome anchor preserved in a frozen row."""
    values = np.asarray(row.get("outcome_advantage_token", []), dtype=np.float64)
    if not len(values):
        return 0.0
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite outcome advantage in {row.get('turn_uid')}")
    if float(values.max() - values.min()) > atol:
        raise ValueError(
            f"Outcome advantage is not token-constant for {row.get('turn_uid')}: "
            f"range={float(values.max() - values.min())}"
        )
    return float(values[0])
