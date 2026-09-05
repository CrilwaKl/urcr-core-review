"""Compact online metrics for Plan 05 source and routing coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_matrix


SAME_FORWARD_PROFILES = (
    "official_evisd",
    "answer_only",
    "g2_query_only",
    "g2_full",
    "g2_shuffled",
    "g2_real",
)
SAME_FORWARD_PAIRS = {
    "query_minus_answer": ("g2_query_only", "answer_only", "query"),
    "real_minus_query": ("g2_real", "g2_query_only", "think"),
    "full_minus_query": ("g2_full", "g2_query_only", "think"),
    "shuffled_minus_query": ("g2_shuffled", "g2_query_only", "think"),
    "real_minus_shuffled": ("g2_real", "g2_shuffled", "think"),
}


def assert_plan06_training_step_health(
    metrics: dict[str, Any],
    *,
    global_step: int,
) -> None:
    """Fail before checkpointing if the frozen Plan 06 update is invalid."""

    def scalar(key: str) -> float:
        if key not in metrics:
            raise RuntimeError(f"Plan 06 health gate is missing metric: {key}")
        value = metrics[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            values = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(values) != 1:
                raise RuntimeError(
                    f"Plan 06 health metric is not scalar: {key}={value}"
                )
            value = values[0]
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(
                    f"Plan 06 health metric is not scalar: {key}={value}"
                )
            value = value.detach().cpu().item()
        result = float(value)
        if not np.isfinite(result):
            raise RuntimeError(f"Plan 06 health metric is non-finite: {key}={result}")
        return result

    for key in (
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/lr",
        "actor/entropy_loss",
        "actor/kl_loss",
        "actor/audit_update_delta_l2",
        "actor/audit_gradient_sum_l2",
        "actor/audit_optimizer_steps",
        "actor/audit_update_nonfinite_count",
        "actor/audit_gradient_nonfinite_count",
        "urcr/privileged_teacher_scored_row_count",
        "urcr/query_conservation_error_max",
        "urcr/think_conservation_error_max",
        "urcr/selected_profile_advantage_max_error",
    ):
        scalar(key)

    if scalar("actor/audit_update_nonfinite_count") != 0:
        raise RuntimeError("Plan 06 actor update contains non-finite values")
    if scalar("actor/audit_gradient_nonfinite_count") != 0:
        raise RuntimeError("Plan 06 actor gradient contains non-finite values")
    if scalar("actor/audit_optimizer_steps") <= 0:
        raise RuntimeError("Plan 06 actor performed no optimizer step")
    if scalar("urcr/privileged_teacher_scored_row_count") != 0:
        raise RuntimeError("Plan 06 URCR-only path scored a privileged teacher row")
    if scalar("actor/audit_gradient_sum_l2") <= 0:
        raise RuntimeError("Plan 06 actor gradient is identically zero")
    if int(global_step) > 1 and scalar("actor/audit_update_delta_l2") <= 0:
        raise RuntimeError("Plan 06 actor update is identically zero after warmup step 1")
    if abs(scalar("urcr/query_conservation_error_max")) > 1e-6:
        raise RuntimeError("Plan 06 query-credit conservation gate failed")
    if abs(scalar("urcr/think_conservation_error_max")) > 1e-6:
        raise RuntimeError("Plan 06 think-credit conservation gate failed")
    if scalar("urcr/selected_profile_advantage_max_error") != 0:
        raise RuntimeError("Plan 06 selected g2_real advantage exactness gate failed")
    if "urcr/localized_tag_residual_nonzero_count" in metrics:
        if scalar("urcr/localized_tag_residual_nonzero_count") != 0:
            raise RuntimeError("Localized think credit reached an action tag")
    if "urcr/localizer_nonfinite_count" in metrics:
        if scalar("urcr/localizer_nonfinite_count") != 0:
            raise RuntimeError("Localized think scorer produced a non-finite value")


def build_training_batch_record(
    batch_dict: dict[str, Any],
    *,
    global_step: int,
    epoch: int,
    shuffle_seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Capture the scientific question batch before rollout expansion."""
    sources = [str(value) for value in batch_dict["data_source"]]
    row_indices = [int(value) for value in batch_dict["dataset_row_index"]]
    extra_infos = list(batch_dict.get("extra_info", [{}] * len(sources)))
    if not (len(sources) == len(row_indices) == len(extra_infos)):
        raise ValueError("training batch identity fields have inconsistent lengths")

    rows = []
    for position, (source, row_index, extra_info) in enumerate(
        zip(sources, row_indices, extra_infos)
    ):
        question = str((extra_info or {}).get("question", "")).strip()
        source_index = (extra_info or {}).get("index", row_index)
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        question_uid = f"{source}:{source_index}:{question_hash}"
        rows.append(
            {
                "within_batch_position": position,
                "dataset_row_index": row_index,
                "question_uid": question_uid,
                "data_source": source,
                "rollout_group_identity": question_uid,
            }
        )
    identity_sha256 = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_counts = {
        source: int(sum(value == source for value in sources))
        for source in sorted(set(sources))
    }
    record = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "shuffle_seed": int(shuffle_seed),
        "sampler_state_id": identity_sha256,
        "question_count": len(rows),
        "source_counts": source_counts,
        "rows": rows,
    }
    denominator = max(1, len(rows))
    metrics = {
        "batch/hotpotqa_question_count": float(source_counts.get("hotpotqa", 0)),
        "batch/nq_question_count": float(source_counts.get("nq", 0)),
        "batch/other_question_count": float(
            len(rows)
            - source_counts.get("hotpotqa", 0)
            - source_counts.get("nq", 0)
        ),
        "batch/hotpotqa_fraction": source_counts.get("hotpotqa", 0) / denominator,
        "batch/nq_fraction": source_counts.get("nq", 0) / denominator,
    }
    return record, metrics


def append_training_batch_record(
    record: dict[str, Any], *, output_dir: str | Path
) -> Path:
    path = Path(output_dir) / "training_batch_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def assert_selected_profile_advantages(
    selected: torch.Tensor,
    profile: torch.Tensor,
    *,
    method: str,
) -> float:
    """Fail before backward if the selected actor signal diverges from its audit profile."""
    if selected.shape != profile.shape:
        raise RuntimeError(
            f"Plan05-MIX selected/profile shape mismatch for {method}: "
            f"{tuple(selected.shape)} != {tuple(profile.shape)}"
        )
    if torch.equal(selected, profile):
        return 0.0
    difference = (selected.detach().float() - profile.detach().float()).abs()
    finite = torch.isfinite(difference)
    max_error = float(difference[finite].max()) if finite.any() else float("inf")
    raise RuntimeError(
        f"Plan05-MIX selected actor advantages diverge from profile {method}: "
        f"max_error={max_error}"
    )


def summarize_urcr_components(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}

    scientific_rows = [row for row in rows if not row.get("is_adjustment_copy", False)]
    valid_search = [row for row in scientific_rows if row.get("valid_search")]
    search_fragments = [row for row in scientific_rows if row.get("action_type") == "search"]
    scored = [row for row in scientific_rows if row.get("rho_scored")]
    source = [row for row in scientific_rows if row.get("source_active")]
    effective_query = [row for row in scientific_rows if row.get("effective_query")]
    effective_think = [row for row in scientific_rows if row.get("effective_think")]

    metrics = {
        "urcr/turn_count": float(len(scientific_rows)),
        "urcr/adjustment_copy_count": float(len(rows) - len(scientific_rows)),
        "urcr/trajectory_count": float(len({row["traj_uid"] for row in scientific_rows})),
        "urcr/search_fragment_count": float(len(search_fragments)),
        "urcr/valid_search_count": float(len(valid_search)),
        "urcr/pi_active_valid_search_count": float(sum(row.get("pi_available", False) for row in valid_search)),
        "urcr/source_active_count": float(len(source)),
        "urcr/effective_query_count": float(len(effective_query)),
        "urcr/rho_scored_count": float(len(scored)),
        "urcr/effective_think_count": float(len(effective_think)),
        "urcr/a_out_zero_count": float(
            sum(abs(float(row.get("a_out", 0.0))) <= 1e-8 for row in scientific_rows)
        ),
        "urcr/source_active_a_out_zero_count": float(sum(abs(float(row.get("a_out", 0.0))) <= 1e-8 for row in source)),
        "urcr/effective_think_per_search_fragment": float(
            len(effective_think) / max(1, len(search_fragments))
        ),
        "urcr/query_conservation_error_max": max(abs(float(row["query_conservation_error"])) for row in rows),
        "urcr/think_conservation_error_max": max(abs(float(row["think_conservation_error"])) for row in rows),
    }
    if any(row.get("support_reward_version") == "v2_fixed_local" for row in rows):
        eligible_v2 = [
            row for row in scientific_rows if row.get("query_local_eligible", False)
        ]
        positive_v2 = [
            row for row in eligible_v2 if float(row.get("support_utility_v2", 0.0)) > 0
        ]
        fact_rows = [row for row in eligible_v2 if row.get("support_hit_type") == "fact"]
        doc_rows = [
            row for row in eligible_v2 if row.get("support_hit_type") == "doc_only"
        ]
        repeat_rows = [row for row in eligible_v2 if row.get("support_hit_type") == "repeat"]
        no_hit_rows = [row for row in eligible_v2 if row.get("support_hit_type") == "no_hit"]
        selected_lengths = [
            float(row.get("selected_think_content_len", 0.0))
            for row in scientific_rows
            if row.get("think_local_eligible", False)
        ]
        denominator = max(1, len(eligible_v2))
        metrics.update(
            {
                "urcr_v2/metadata_eligible_search_count": float(len(eligible_v2)),
                "urcr_v2/fact_hit_count": float(len(fact_rows)),
                "urcr_v2/fact_hit_rate": float(len(fact_rows) / denominator),
                "urcr_v2/doc_only_count": float(len(doc_rows)),
                "urcr_v2/doc_only_rate": float(len(doc_rows) / denominator),
                "urcr_v2/repeat_count": float(len(repeat_rows)),
                "urcr_v2/repeat_rate": float(len(repeat_rows) / denominator),
                "urcr_v2/no_hit_count": float(len(no_hit_rows)),
                "urcr_v2/no_hit_rate": float(len(no_hit_rows) / denominator),
                "urcr_v2/support_positive_a_out_zero_count": float(
                    sum(abs(float(row.get("a_out", 0.0))) <= 1e-8 for row in positive_v2)
                ),
                "urcr_v2/support_positive_a_out_negative_count": float(
                    sum(float(row.get("a_out", 0.0)) < -1e-8 for row in positive_v2)
                ),
                "urcr_v2/support_positive_a_out_positive_count": float(
                    sum(float(row.get("a_out", 0.0)) > 1e-8 for row in positive_v2)
                ),
                "urcr_v2/multi_doc_hit_rate": float(
                    np.mean(
                        [row.get("support_new_doc_count", 0) > 1 for row in eligible_v2]
                    )
                    if eligible_v2
                    else 0.0
                ),
                "urcr_v2/multi_fact_hit_rate": float(
                    np.mean(
                        [row.get("support_new_fact_count", 0) > 1 for row in eligible_v2]
                    )
                    if eligible_v2
                    else 0.0
                ),
                "urcr_v2/selected_think_length_mean": float(
                    np.mean(selected_lengths) if selected_lengths else 0.0
                ),
                "urcr_v2/support_reward_query_mean_positive": float(
                    np.mean([row.get("support_reward_query", 0.0) for row in positive_v2])
                    if positive_v2
                    else 0.0
                ),
                "urcr_v2/support_reward_think_mean_positive": float(
                    np.mean([row.get("support_reward_think", 0.0) for row in positive_v2])
                    if positive_v2
                    else 0.0
                ),
            }
        )
    if source:
        metrics["urcr/u_g2_mean_source_active"] = float(np.mean([row["u_g2"] for row in source]))
    if scored:
        metrics["urcr/d_mask_mean"] = float(np.mean([row["d_mask"] for row in scored]))
        metrics["urcr/d_mask_negative_rate"] = float(np.mean([row["d_mask"] < 0 for row in scored]))
        metrics["urcr/rho_mean"] = float(np.mean([row["rho_real"] for row in scored]))
        metrics["urcr/rho_zero_rate"] = float(np.mean([row["rho_real"] <= 0 for row in scored]))
        metrics["urcr/rho_saturation_rate"] = float(np.mean([row["rho_real"] >= 0.95 for row in scored]))
    localized_rows = [
        row
        for row in scientific_rows
        if row.get("think_credit_mode") in {"loo_mass50", "random_matched"}
    ]
    if localized_rows:
        eligible_localizer = [
            row
            for row in localized_rows
            if row.get("rho_scored") and row.get("c_query_total", 0.0) > 0.0
        ]
        active_localizer = [
            row for row in eligible_localizer if row.get("localizer_active")
        ]
        fallback_localizer = [
            row for row in eligible_localizer if row.get("fallback_reason")
        ]
        finite_scores = [
            float(value)
            for row in localized_rows
            for value in row.get("chunk_loo_scores", [])
            if np.isfinite(float(value))
        ]
        metrics.update(
            {
                "urcr/localizer_eligible_count": float(len(eligible_localizer)),
                "urcr/localizer_active_count": float(len(active_localizer)),
                "urcr/localizer_active_rate": float(
                    len(active_localizer) / max(1, len(eligible_localizer))
                ),
                "urcr/localizer_fallback_count": float(len(fallback_localizer)),
                "urcr/localizer_fallback_rate": float(
                    len(fallback_localizer) / max(1, len(eligible_localizer))
                ),
                "urcr/localizer_selected_token_fraction_mean": float(
                    np.mean(
                        [row["selected_token_fraction"] for row in eligible_localizer]
                    )
                    if eligible_localizer
                    else 0.0
                ),
                "urcr/localizer_selected_chunk_count_mean": float(
                    np.mean(
                        [row["selected_chunk_count"] for row in eligible_localizer]
                    )
                    if eligible_localizer
                    else 0.0
                ),
                "urcr/localizer_loo_score_mean": float(
                    np.mean(finite_scores) if finite_scores else 0.0
                ),
                "urcr/localizer_loo_score_p95": float(
                    np.quantile(finite_scores, 0.95) if finite_scores else 0.0
                ),
                "urcr/localizer_nonfinite_count": float(
                    sum(
                        int(row.get("loo_score_nonfinite_count", 0))
                        for row in localized_rows
                    )
                ),
                "urcr/localizer_forward_count": float(
                    sum(
                        int(row.get("localizer_forward_count", 0))
                        for row in localized_rows
                    )
                ),
                "urcr/localizer_random_identity_unavoidable_count": float(
                    sum(
                        bool(row.get("random_identity_unavoidable"))
                        for row in eligible_localizer
                    )
                ),
            }
        )
    for group in ("all-fail", "mixed-outcome", "all-success"):
        group_rows = [row for row in scientific_rows if row.get("outcome_group") == group]
        prefix = group.replace("-", "_")
        metrics[f"urcr_group/{prefix}_turn_count"] = float(len(group_rows))
        metrics[f"urcr_group/{prefix}_source_active_count"] = float(
            sum(row.get("source_active", False) for row in group_rows)
        )
        metrics[f"urcr_group/{prefix}_effective_query_count"] = float(
            sum(row.get("effective_query", False) for row in group_rows)
        )
        metrics[f"urcr_group/{prefix}_effective_think_count"] = float(
            sum(row.get("effective_think", False) for row in group_rows)
        )
    return metrics


def _local_values(
    tensor: torch.Tensor,
    response_mask: torch.Tensor,
    batch_index: int,
    local_mask: list[int],
) -> list[float]:
    valid = torch.nonzero(response_mask[batch_index].bool(), as_tuple=False).flatten()
    if len(local_mask) != len(valid):
        raise ValueError("URCR component span/response length mismatch")
    selected = [index for index, value in enumerate(local_mask) if value]
    if not selected:
        return []
    actual = valid[selected]
    return tensor[batch_index, actual].detach().float().cpu().tolist()


def enrich_urcr_components(
    rows: list[dict[str, Any]],
    *,
    frozen_rows: list[dict[str, Any]],
    response_mask: torch.Tensor,
    final_advantages: torch.Tensor,
    evisd_bonus: torch.Tensor,
    responsibility: dict[str, dict[str, Any]],
    global_step: int,
) -> list[dict[str, Any]]:
    if len(rows) != len(frozen_rows):
        raise ValueError("URCR component/frozen row count mismatch")
    output = []
    for index, (row, frozen) in enumerate(zip(rows, frozen_rows)):
        item = dict(row)
        query_mask = list(frozen.get("search_content_mask", []))
        think_mask = list(frozen.get("think_mask", []))
        search_bonus = _local_values(evisd_bonus, response_mask, index, query_mask)
        final_query = _local_values(final_advantages, response_mask, index, query_mask)
        final_think = _local_values(final_advantages, response_mask, index, think_mask)
        scorer = responsibility.get(str(row["turn_uid"]), {})
        fingerprint_payload = {
            "data_source": frozen.get("data_source"),
            "dataset_index": frozen.get("dataset_index"),
            "turn_step": frozen.get("turn_step"),
            "response_token_ids": frozen.get("response_token_ids", []),
            "observation_text": frozen.get("observation_text", ""),
        }
        rollout_content_sha256 = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        item.update(
            {
                "global_step": int(global_step),
                "rollout_content_sha256": rollout_content_sha256,
                "question": str(frozen.get("question", "")),
                "ground_truth_aliases": list(frozen.get("ground_truth_aliases", [])),
                "turn_context_text": str(frozen.get("turn_context_text", "")),
                "response_text": str(frozen.get("response_text", "")),
                "query_text": str(frozen.get("query_text", "")),
                "observation_text": str(frozen.get("observation_text", "")),
                "think_mask": list(frozen.get("think_mask", [])),
                "think_content_mask": list(frozen.get("think_content_mask", [])),
                "think_tag_mask": list(frozen.get("think_tag_mask", [])),
                "search_content_mask": list(frozen.get("search_content_mask", [])),
                "evisd_search_bonus_token": search_bonus,
                "evisd_search_bonus_sum": float(sum(search_bonus)),
                "evisd_search_bonus_mean": float(np.mean(search_bonus)) if search_bonus else 0.0,
                "final_query_advantage_token": final_query,
                "final_think_advantage_token": final_think,
                "full_query_logp_token": list(scorer.get("full_query_logp_token", [])),
                "masked_query_logp_token": list(scorer.get("masked_query_logp_token", [])),
                "old_query_logp_token": list(scorer.get("old_query_logp_token", [])),
                "full_old_mean_abs_error": scorer.get("full_old_mean_abs_error"),
                "full_old_max_token_error": scorer.get("full_old_max_token_error"),
                "chunk_loo_scores": list(scorer.get("chunk_loo_scores", [])),
                "localizer_forward_count": int(
                    scorer.get("localizer_forward_count", 0)
                ),
                "query_sign_preserved": (
                    None
                    if row.get("support_reward_version") == "v2_fixed_local"
                    else bool(
                        not final_query
                        or abs(float(row["a_out"])) <= 1e-8
                        or all(
                            np.sign(value) == np.sign(row["a_out"])
                            for value in final_query
                        )
                    )
                ),
                "think_sign_preserved": (
                    None
                    if row.get("support_reward_version") == "v2_fixed_local"
                    else bool(
                        not final_think
                        or abs(float(row["a_out"])) <= 1e-8
                        or all(
                            np.sign(value) == np.sign(row["a_out"])
                            for value in final_think
                        )
                    )
                ),
                "separate_local_objective": bool(
                    row.get("support_reward_version") == "v2_fixed_local"
                ),
            }
        )
        output.append(item)
    return output


def write_turn_component_part(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    global_step: int,
) -> Path:
    path = Path(output_dir) / "turn_components_parts" / f"step_{global_step:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Plan 05 turn components: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return path


def write_step_metrics(
    metrics: dict[str, Any],
    *,
    output_dir: str | Path,
    global_step: int,
) -> Path:
    path = Path(output_dir) / "step_metrics" / f"step_{global_step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Plan 05 step metrics: {path}")
    serializable = {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in metrics.items()
    }
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_resolved_config(config: dict[str, Any], *, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "resolved_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Plan 05 resolved config: {path}")
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def urcr_ppo_reduction_terms(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    query_span_mask: torch.Tensor,
    think_span_mask: torch.Tensor,
    query_residual: torch.Tensor,
    think_residual: torch.Tensor,
    clip_ratio: float,
    clip_ratio_low: float | None,
    clip_ratio_high: float | None,
    clip_ratio_c: float,
    loss_agg_mode: str,
    gradient_accumulation: int,
) -> dict[str, torch.Tensor]:
    """Audit residual totals under the exact vanilla PPO token reduction."""
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    with torch.no_grad():
        ratio = torch.exp(log_prob.detach() - old_log_prob.detach())
        detached_advantages = advantages.detach()
        actual_mask = response_mask.detach().bool()
        full_loss, clip_indicator, lower_clip_indicator = compute_policy_loss_matrix(
            ratio=ratio,
            advantages=detached_advantages,
            cliprange=clip_ratio,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
        )
        output = {
            "denominator_token_sum": actual_mask.sum().double(),
        }
        for role, span_mask, residual in (
            ("query", query_span_mask, query_residual),
            ("think", think_span_mask, think_residual),
        ):
            role_mask = actual_mask & span_mask.detach().bool()
            detached_residual = residual.detach()
            without_role_loss, _, _ = compute_policy_loss_matrix(
                ratio=ratio,
                advantages=detached_advantages - detached_residual,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
            )
            role_float = role_mask.to(detached_advantages.dtype)
            output[f"{role}_token_count"] = role_mask.sum().double()
            output[f"{role}_preclip_residual_total"] = (
                detached_residual * role_float
            ).sum().double()
            output[f"{role}_postratio_residual_total"] = (
                detached_residual * ratio * role_float
            ).sum().double()
            output[f"{role}_postclip_surrogate_total"] = (
                -(full_loss - without_role_loss) * role_float
            ).sum().double()
            clipped = (clip_indicator.bool() | lower_clip_indicator.bool()) & role_mask
            output[f"{role}_clipped_token_count"] = clipped.sum().double()
            output[f"{role}_ratio_sum"] = (ratio * role_float).sum().double()
            output[f"{role}_actual_reduced_pg_loss_delta"] = (
                agg_loss(
                    loss_mat=full_loss,
                    loss_mask=actual_mask,
                    loss_agg_mode=loss_agg_mode,
                )
                - agg_loss(
                    loss_mat=without_role_loss,
                    loss_mask=actual_mask,
                    loss_agg_mode=loss_agg_mode,
                )
            ).double() / gradient_accumulation
        return output


def urcr_same_forward_terms(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    profile_advantages: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    query_span_mask: torch.Tensor,
    think_span_mask: torch.Tensor,
    clip_ratio: float,
    clip_ratio_low: float | None,
    clip_ratio_high: float | None,
    clip_ratio_c: float,
    loss_agg_mode: str,
    gradient_accumulation: int,
) -> dict[str, torch.Tensor]:
    """Compare all repaired methods on one current-log-prob forward graph."""
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if not log_prob.requires_grad:
        raise ValueError("Same-forward audit requires differentiable current log probabilities")
    if set(profile_advantages) != set(SAME_FORWARD_PROFILES):
        raise ValueError(
            "Same-forward profiles must be exactly "
            f"{SAME_FORWARD_PROFILES}, got {sorted(profile_advantages)}"
        )

    active = response_mask.detach().bool()
    query = query_span_mask.detach().bool() & active
    think = think_span_mask.detach().bool() & active
    ratio = torch.exp(log_prob - old_log_prob.detach())
    output: dict[str, torch.Tensor] = {
        "denominator_token_count": active.sum().double(),
        "query_token_count": query.sum().double(),
        "think_token_count": think.sum().double(),
        "query_think_overlap_count": (query & think).sum().double(),
    }
    raw_matrices: dict[str, torch.Tensor] = {}
    clipped_matrices: dict[str, torch.Tensor] = {}
    raw_coefficients: dict[str, torch.Tensor] = {}
    clipped_coefficients: dict[str, torch.Tensor] = {}

    active_float = active.to(dtype=log_prob.dtype)
    for profile in SAME_FORWARD_PROFILES:
        advantages = profile_advantages[profile].detach()
        if advantages.shape != log_prob.shape:
            raise ValueError(f"Same-forward advantage shape mismatch for {profile}")
        raw_matrix = -advantages * ratio
        clipped_matrix, clip_indicator, lower_clip_indicator = compute_policy_loss_matrix(
            ratio=ratio,
            advantages=advantages,
            cliprange=clip_ratio,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
        )
        raw_loss = agg_loss(
            loss_mat=raw_matrix,
            loss_mask=active,
            loss_agg_mode=loss_agg_mode,
        )
        clipped_loss = agg_loss(
            loss_mat=clipped_matrix,
            loss_mask=active,
            loss_agg_mode=loss_agg_mode,
        )
        raw_coefficient = torch.autograd.grad(
            raw_loss / gradient_accumulation,
            log_prob,
            retain_graph=True,
        )[0]
        clipped_coefficient = torch.autograd.grad(
            clipped_loss / gradient_accumulation,
            log_prob,
            retain_graph=True,
        )[0]
        raw_matrices[profile] = raw_matrix
        clipped_matrices[profile] = clipped_matrix
        raw_coefficients[profile] = raw_coefficient
        clipped_coefficients[profile] = clipped_coefficient

        prefix = f"profile/{profile}"
        output[f"{prefix}/raw_surrogate_total"] = (
            raw_matrix.detach() * active_float
        ).sum().double()
        output[f"{prefix}/clipped_surrogate_total"] = (
            clipped_matrix.detach() * active_float
        ).sum().double()
        output[f"{prefix}/actual_raw_pg_loss"] = (
            raw_loss.detach().double() / gradient_accumulation
        )
        output[f"{prefix}/actual_clipped_pg_loss"] = (
            clipped_loss.detach().double() / gradient_accumulation
        )
        output[f"{prefix}/clip_count"] = (
            (clip_indicator.bool() | lower_clip_indicator.bool()) & active
        ).sum().double()
        for role, role_mask in (("query", query), ("think", think)):
            role_float = role_mask.to(dtype=log_prob.dtype)
            output[f"{prefix}/{role}_raw_total"] = (
                raw_matrix.detach() * role_float
            ).sum().double()
            output[f"{prefix}/{role}_clipped_total"] = (
                clipped_matrix.detach() * role_float
            ).sum().double()
        for stage, coefficient in (
            ("preclip", raw_coefficient),
            ("postclip", clipped_coefficient),
        ):
            finite = torch.isfinite(coefficient)
            selected = torch.where(active, coefficient, torch.zeros_like(coefficient))
            output[f"{prefix}/{stage}_coefficient_nonfinite_count"] = (
                (~finite & active).sum().double()
            )
            output[f"{prefix}/{stage}_coefficient_l2_sq"] = (
                selected.detach().double().square().sum()
            )
            output[f"{prefix}/{stage}_coefficient_abs_sum"] = (
                selected.detach().double().abs().sum()
            )

    for pair, (left, right, role) in SAME_FORWARD_PAIRS.items():
        allowed = query if role == "query" else think
        outside = active & ~allowed
        prefix = f"pair/{pair}"
        raw_delta_matrix = raw_matrices[left] - raw_matrices[right]
        clipped_delta_matrix = clipped_matrices[left] - clipped_matrices[right]
        output[f"{prefix}/raw_surrogate_delta_total"] = (
            raw_delta_matrix.detach() * active_float
        ).sum().double()
        output[f"{prefix}/clipped_surrogate_delta_total"] = (
            clipped_delta_matrix.detach() * active_float
        ).sum().double()
        output[f"{prefix}/actual_raw_loss_delta"] = (
            agg_loss(raw_delta_matrix, active, loss_agg_mode).detach().double()
            / gradient_accumulation
        )
        output[f"{prefix}/actual_clipped_loss_delta"] = (
            agg_loss(clipped_delta_matrix, active, loss_agg_mode).detach().double()
            / gradient_accumulation
        )
        for stage, coefficients in (
            ("preclip", raw_coefficients),
            ("postclip", clipped_coefficients),
        ):
            delta = (coefficients[left] - coefficients[right]).detach()
            finite = torch.isfinite(delta)
            safe_delta = torch.where(finite, delta, torch.zeros_like(delta))
            output[f"{prefix}/{stage}_nonfinite_count"] = (
                (~finite & active).sum().double()
            )
            output[f"{prefix}/{stage}_delta_l2_sq"] = (
                (safe_delta.double().square() * active).sum()
            )
            output[f"{prefix}/{stage}_delta_abs_sum"] = (
                (safe_delta.double().abs() * active).sum()
            )
            output[f"{prefix}/{stage}_nonzero_count"] = (
                ((safe_delta != 0) & active).sum().double()
            )
            output[f"{prefix}/{stage}_outside_role_nonzero_count"] = (
                ((safe_delta != 0) & outside).sum().double()
            )
            output[f"{prefix}/{stage}_outside_role_abs_sum"] = (
                (safe_delta.double().abs() * outside).sum()
            )
    return output
