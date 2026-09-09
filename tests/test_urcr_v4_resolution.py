from __future__ import annotations

import hashlib
import json

import pytest

from verl.trainer.ppo.urcr_v4_method import METHOD_REVISION
from verl.trainer.ppo.urcr_v4_resolution import (
    assert_urcr_v4_training_step_health,
    load_resolved_method,
    validate_resolved_method_document,
)
from verl.trainer.ppo.urcr_v4_scorer import PROBE_TEMPLATE_VERSION


def _document(tmp_path):
    implementation = tmp_path / "core.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    implementation_sha = hashlib.sha256(implementation.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "artifact": "urcr_v4_resolved_method",
        "method_revision": METHOD_REVISION,
        "frozen": True,
        "code": {
            "repo_root": str(tmp_path),
            "implementation_manifest_sha256": "manifest",
            "implementation_files": [
                {"path": "core.py", "sha256": implementation_sha}
            ],
        },
        "search_utility": {
            "control_count": 3,
            "negative_scale": 0.1,
            "group_whitening": False,
            "delta": 0.5,
            "scale": 2.0,
        },
        "answer_utility": {
            "beta": 0.5,
            "std_normalization": False,
            "residual_group": "rollout_group_id_and_binary_em",
        },
        "responsibility": {
            "probe_template_version": PROBE_TEMPLATE_VERSION,
            "targets": "actual_sampled_action_content",
            "intervention": "position_preserving_outgoing_information_barrier",
            "mapping": "positive_exponential",
            "search": {"delta": 0.08, "scale": 0.43},
            "answer": {"delta": 0.15, "scale": 0.84},
        },
        "chunks": {
            "power": 2.0,
            "max_chunks": 6,
            "no_mass": "zero",
            "whole_think_fallback": False,
        },
        "credit": {
            "alpha_query": 1.0,
            "alpha_answer": 0.25,
            "lambda_think_query": 1.0,
            "lambda_think_answer": 1.0,
            "query_think_split": False,
            "search_trajectory_absolute_cap": 2.0,
        },
        "metadata": {
            "online_pipeline": False,
            "credit_enabled": False,
            "diagnostics": "offline_sidecar_only",
        },
        "local_objective": {
            "surrogate": "signed_tokenwise_ppo",
            "span_reduction": "mean_once",
            "population_denominator": "original_trajectories",
            "same_actor_forward_for_training": True,
            "lambda_max": 0.04,
            "warmup_outer_steps": 30,
        },
        "scorer": {
            "micro_batch_size_per_gpu": 32,
            "rollout_temperature": 1.0,
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


def test_resolved_method_loader_checks_artifact_and_implementation_hashes(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    runtime, identity = load_resolved_method(path, expected_sha256=expected)

    assert runtime["local_loss"] == {"lambda_max": 0.04, "warmup_steps": 30}
    assert identity["resolved_method_sha256"] == expected

    (tmp_path / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation changed"):
        load_resolved_method(path, expected_sha256=expected)


def test_resolved_method_rejects_scientific_drift(tmp_path):
    document = _document(tmp_path)
    document["credit"]["alpha_answer"] = 1.0
    with pytest.raises(ValueError, match="credit constant changed"):
        validate_resolved_method_document(document)


def _healthy_step_metrics():
    return {
        "actor/pg_loss": 0.01,
        "actor/grad_norm": 2.0,
        "actor/lr": 1e-7,
        "actor/entropy_loss": 0.3,
        "actor/kl_loss": 0.0,
        "urcr_v4/effective_local_loss": -0.01,
        "urcr_v4/lambda_effective": 0.001,
        "urcr_v4/original_trajectory_count": 1024.0,
        "urcr_v4/pipeline_seconds": 10.0,
    }


def test_v4_health_checks_optimizer_plumbing_without_credit_sign_gate():
    metrics = _healthy_step_metrics()
    assert_urcr_v4_training_step_health(
        metrics,
        global_step=2,
        expected_trajectory_count=1024,
    )

    metrics["urcr_v4/effective_local_loss"] = 0.0
    assert_urcr_v4_training_step_health(
        metrics,
        global_step=2,
        expected_trajectory_count=1024,
    )

    metrics["actor/lr"] = 0.0
    assert_urcr_v4_training_step_health(
        metrics,
        global_step=1,
        expected_trajectory_count=1024,
    )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("actor/grad_norm", float("nan"), "non-finite"),
        ("actor/grad_norm", 0.0, "identically zero"),
        ("actor/lr", 0.0, "learning rate"),
        ("urcr_v4/lambda_effective", 0.0, "not active"),
        ("urcr_v4/original_trajectory_count", 1023.0, "population changed"),
    ],
)
def test_v4_health_rejects_broken_runtime_plumbing(key, value, match):
    metrics = _healthy_step_metrics()
    metrics[key] = value
    with pytest.raises(RuntimeError, match=match):
        assert_urcr_v4_training_step_health(
            metrics,
            global_step=2,
            expected_trajectory_count=1024,
        )
