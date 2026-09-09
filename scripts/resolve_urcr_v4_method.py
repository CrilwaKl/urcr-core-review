#!/usr/bin/env python3
"""Freeze the calibrated URCR-V4 method and its executable provenance."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from verl.trainer.ppo.urcr_v4_method import (
    ANSWER_ALPHA,
    CONTROL_COUNT,
    MAX_THINK_CHUNKS,
    METHOD_REVISION,
    SEARCH_NEGATIVE_SCALE,
    SEARCH_TRAJECTORY_ABSOLUTE_CAP,
)
from verl.trainer.ppo.urcr_v4_resolution import (
    sha256_file,
    validate_resolved_method_document,
)
from verl.trainer.ppo.urcr_v4_scorer import PROBE_TEMPLATE_VERSION


REPO_ROOT = Path("/data2/kongmu/agentic-RL/EviSD-URCR")
WORKSPACE_ROOT = REPO_ROOT.parent
PLAN_PATH = WORKSPACE_ROOT / "research_design/urcr_evisd_staged_plans/plan_URCR-V4.md"
BASE_MODEL = Path("/data0/kongmu/agentic-RL/models/Qwen2.5-3B-Instruct")
S150_MODEL = Path("/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf")
TRAIN_DATA = Path("/data0/kongmu/agentic-RL/data/searchR1_processed_direct/train.parquet")
IMPLEMENTATION_FILES = (
    "agent_system/multi_turn_rollout/rollout_loop.py",
    "agent_system/multi_turn_rollout/utils.py",
    "examples/urcr_online/run_urcr_v4_training.sh",
    "scripts/pool_urcr_v4_calibration_population.py",
    "scripts/resolve_urcr_v4_method.py",
    "scripts/run_urcr_v4_four_gpu_smoke.sh",
    "verl/trainer/config/ppo_trainer.yaml",
    "verl/trainer/main_evisd.py",
    "verl/trainer/ppo/evisd_ray_trainer.py",
    "verl/trainer/ppo/evisd_teacher.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/ppo/urcr_v4_calibration.py",
    "verl/trainer/ppo/urcr_v4_controller.py",
    "verl/trainer/ppo/urcr_v4_data.py",
    "verl/trainer/ppo/urcr_v4_local_objective.py",
    "verl/trainer/ppo/urcr_v4_method.py",
    "verl/trainer/ppo/urcr_v4_pipeline.py",
    "verl/trainer/ppo/urcr_v4_requests.py",
    "verl/trainer/ppo/urcr_v4_resolution.py",
    "verl/trainer/ppo/urcr_v4_scorer.py",
    "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/fsdp_workers.py",
)


def _load(path: Path, artifact: str) -> tuple[dict, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact") != artifact or not data.get("pass", False):
        raise ValueError(f"invalid prerequisite artifact: {path}")
    return data, sha256_file(path)


def _git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(WORKSPACE_ROOT), *args],
        stderr=subprocess.DEVNULL,
    )


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _implementation_manifest() -> tuple[list[dict], str]:
    rows = []
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return rows, _hash_bytes(encoded)


def _file_manifest_identity(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled", type=Path, required=True)
    parser.add_argument("--s150-gradient", type=Path, required=True)
    parser.add_argument("--base-shadow", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--base-model-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to overwrite resolved method: {args.output}")

    pooled, pooled_sha = _load(args.pooled.resolve(), "urcr_v4_s150_pooled_calibration")
    s150_gradient, s150_gradient_sha = _load(
        args.s150_gradient.resolve(), "urcr_v4_s150_gradient_calibration"
    )
    base_shadow, base_shadow_sha = _load(
        args.base_shadow.resolve(), "urcr_v4_base_shadow_gradient_calibration"
    )
    if pooled.get("resolved_method_frozen") is not False:
        raise ValueError("pooled artifact is not pre-resolution evidence")
    if pooled.get("optimizer_update_executed") is not False:
        raise ValueError("pooled calibration performed an optimizer update")
    if s150_gradient["cost"]["optimizer_update_executed"] is not False:
        raise ValueError("S150 gradient calibration performed an optimizer update")
    if base_shadow["cost"]["optimizer_update_executed"] is not False:
        raise ValueError("Base shadow performed an optimizer update")
    if s150_gradient["pooled_calibration_sha256"] != pooled_sha:
        raise ValueError("S150 gradient calibration references another pooled artifact")
    if base_shadow["pooled_calibration_sha256"] != pooled_sha:
        raise ValueError("Base shadow references another pooled artifact")
    if base_shadow["s150_gradient_calibration_sha256"] != s150_gradient_sha:
        raise ValueError("Base shadow references another S150 gradient artifact")
    if any(
        value.get("method_revision") != METHOD_REVISION
        for value in (pooled, s150_gradient, base_shadow)
    ):
        raise ValueError("calibration artifacts disagree on method revision")

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    if (
        probe.get("method_revision") != METHOD_REVISION
        or probe.get("version") != PROBE_TEMPLATE_VERSION
    ):
        raise ValueError("probe template identity changed")
    if str(pooled["model"]) != str(S150_MODEL):
        raise ValueError("pooled calibration does not use the approved S150 model")
    if str(base_shadow["model"]["path"]) != str(BASE_MODEL):
        raise ValueError("Base shadow does not use the original Base model")
    if int(pooled["batch_count"]) != 10 or int(pooled["trajectory_count"]) != 10240:
        raise ValueError("pooled calibration population is incomplete")
    if base_shadow["base_shadow_lowered_lambda"] not in (True, False):
        raise ValueError("Base shadow ceiling decision is missing")

    search = pooled["pooled"]["search"]["calibration"]
    responsibility = pooled["pooled"]["responsibility"]["summary"]
    lambda_max = float(base_shadow["lambda_max"])
    population_path = Path(str(pooled["population_manifest"])).resolve()
    if sha256_file(population_path) != str(pooled["population_manifest_sha256"]):
        raise ValueError("population manifest changed after pooling")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    implementation_files, implementation_sha = _implementation_manifest()
    status = _git("status", "--short", "--untracked-files=all")
    diff = _git("diff", "--no-ext-diff", "--binary")

    document = {
        "schema_version": 1,
        "artifact": "urcr_v4_resolved_method",
        "method_revision": METHOD_REVISION,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "frozen": True,
        "code": {
            "repo_root": str(REPO_ROOT),
            "workspace_git_head": _git("rev-parse", "HEAD").decode().strip(),
            "workspace_git_status_sha256": _hash_bytes(status),
            "workspace_tracked_diff_sha256": _hash_bytes(diff),
            "implementation_manifest_sha256": implementation_sha,
            "implementation_files": implementation_files,
        },
        "protocol_authority": _file_manifest_identity(PLAN_PATH),
        "data": {
            "train_path": str(TRAIN_DATA),
            "train_sha256": str(population["input_sha256"]),
            "sources": ["hotpotqa", "nq"],
            "metadata_online_pipeline": False,
        },
        "models": {
            "formal_initialization": {
                "path": str(BASE_MODEL),
                "hash_manifest": _file_manifest_identity(
                    args.base_model_hashes.resolve()
                ),
            },
            "diagnostic_s150": {
                "path": str(S150_MODEL),
                "hash_manifest": _file_manifest_identity(
                    S150_MODEL / "SHA256SUMS"
                ),
            },
        },
        "retriever": {
            "url": "http://127.0.0.1:18268/retrieve",
            "topk": 3,
            "launcher": _file_manifest_identity(
                REPO_ROOT / "examples/urcr_online/start_retriever.sh"
            ),
            "index_path": "/data2/kongmu/pretrain/wiki/e5_Flat.index",
            "corpus_path": "/data2/kongmu/pretrain/wiki/wiki-18.jsonl",
            "retriever_model": "/data2/kongmu/pretrain/e5-base-v2",
        },
        "probe": {
            **probe,
            "artifact_path": str(args.probe.resolve()),
            "artifact_sha256": sha256_file(args.probe.resolve()),
            "alias_selection": "primary_then_stable_normalized_dedup_max_3",
        },
        "search_utility": {
            "control_count": CONTROL_COUNT,
            "control_selection": "same_source_call_bucket_then_stable_fallback",
            "combine": "same_sign_min_magnitude",
            "delta": float(search["delta"]),
            "scale": float(search["scale"]),
            "squash": "continuous_deadzone_tanh",
            "negative_scale": SEARCH_NEGATIVE_SCALE,
            "visible_exact_repeat_positive": False,
            "unconsumable_observation_credit": 0.0,
            "group_whitening": False,
        },
        "metadata": {
            "online_pipeline": False,
            "credit_enabled": False,
            "diagnostics": "offline_sidecar_only",
        },
        "answer_utility": {
            "quality": "evaluator_consistent_word_fbeta",
            "beta": 0.5,
            "residual_group": "rollout_group_id_and_binary_em",
            "baseline": "within_stratum_mean_including_self",
            "std_normalization": False,
            "singleton_credit": 0.0,
            "all_equal_credit": 0.0,
            "dependence_on_a_out": False,
        },
        "responsibility": {
            "targets": "actual_sampled_action_content",
            "intervention": "position_preserving_outgoing_information_barrier",
            "mapping": "positive_exponential",
            "probe_template_version": PROBE_TEMPLATE_VERSION,
            "search": {
                "delta": float(responsibility["search_calibration"]["delta"]),
                "scale": float(responsibility["search_calibration"]["scale"]),
            },
            "answer": {
                "delta": float(responsibility["answer_calibration"]["delta"]),
                "scale": float(responsibility["answer_calibration"]["scale"]),
            },
        },
        "chunks": {
            "power": 2.0,
            "max_chunks": MAX_THINK_CHUNKS,
            "no_mass": "zero",
            "whole_think_fallback": False,
        },
        "credit": {
            "alpha_query": 1.0,
            "alpha_answer": ANSWER_ALPHA,
            "lambda_think_query": 1.0,
            "lambda_think_answer": 1.0,
            "query_think_split": False,
            "search_trajectory_absolute_cap": SEARCH_TRAJECTORY_ABSOLUTE_CAP,
            "terminal_broadcast_to_old_thinks": False,
        },
        "local_objective": {
            "span_reduction": "mean_once",
            "surrogate": "signed_tokenwise_ppo",
            "population_denominator": "original_trajectories",
            "minibatch_estimator": "preserve_outer_trajectory_mean",
            "lambda_max": lambda_max,
            "warmup_outer_steps": 30,
            "final_decay": False,
            "same_actor_forward_for_training": True,
        },
        "scorer": {
            "backend": "fsdp_actor_sdpa_position_preserving_barrier",
            "dtype": "bfloat16",
            "micro_batch_size_per_gpu": 32,
            "token_budget_per_gpu": 262144,
            "max_total_tokens": 8192,
            "rollout_temperature": 1.0,
        },
        "calibration": {
            "population": {
                "artifact": str(args.pooled.resolve()),
                "sha256": pooled_sha,
                "batch_count": int(pooled["batch_count"]),
                "question_count": int(pooled["question_count"]),
                "trajectory_count": int(pooled["trajectory_count"]),
                "status": str(pooled["status"]),
                "limitations": list(pooled["limitations"]),
                "last_three_prefix_ranges": pooled["last_three_prefix_ranges"],
            },
            "s150_gradient": {
                "artifact": str(args.s150_gradient.resolve()),
                "sha256": s150_gradient_sha,
                "global_over_local": s150_gradient["global_over_local"],
                "lambda_0": s150_gradient["lambda_0"],
                "lambda_max": s150_gradient["lambda_max"],
                "full_strength_local_over_global_max": s150_gradient[
                    "full_strength_local_over_global_max"
                ],
                "warmup_applied": False,
                "optimizer_update_executed": False,
            },
            "base_shadow": {
                "artifact": str(args.base_shadow.resolve()),
                "sha256": base_shadow_sha,
                "global_over_local": base_shadow["global_over_local"],
                "lambda_max_before": base_shadow[
                    "lambda_max_before_base_shadow"
                ],
                "lambda_max_after": lambda_max,
                "full_strength_local_over_global": base_shadow[
                    "full_strength_local_over_global_max"
                ],
                "lowered_lambda": base_shadow["base_shadow_lowered_lambda"],
                "optimizer_update_executed": False,
            },
        },
        "formal_protocol": {
            "initialization": "original_base",
            "total_outer_steps": 300,
            "global_question_batch": 128,
            "rollouts_per_question": 8,
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 16,
            "ppo_epochs": 1,
            "checkpoint_frequency": 25,
            "evaluation_steps": [150, 300],
        },
        "legacy": {
            "fixed_support_reward": False,
            "legacy_v1_residual": False,
            "agam": False,
            "visible_focus_kl": False,
            "evisd_teacher": False,
            "evisd_search_pi": False,
            "evisd_answer_pi": False,
            "old_s_local": None,
            "n1_reference": False,
        },
    }
    validate_resolved_method_document(document)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".writing")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    output_sha = sha256_file(args.output)
    sidecar = args.output.with_suffix(".sha256")
    sidecar.write_text(f"{output_sha}  {args.output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": output_sha,
                "implementation_manifest_sha256": implementation_sha,
                "delta_U": document["search_utility"]["delta"],
                "s_U": document["search_utility"]["scale"],
                "delta_R_Q": document["responsibility"]["search"]["delta"],
                "s_R_Q": document["responsibility"]["search"]["scale"],
                "delta_R_A": document["responsibility"]["answer"]["delta"],
                "s_R_A": document["responsibility"]["answer"]["scale"],
                "lambda_max": lambda_max,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
