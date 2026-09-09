"""Load and verify the frozen URCR-V4 method artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from verl.trainer.ppo.urcr_v4_method import (
    ANSWER_ALPHA,
    CONTROL_COUNT,
    MAX_THINK_CHUNKS,
    METHOD_REVISION,
    SEARCH_NEGATIVE_SCALE,
    SEARCH_TRAJECTORY_ABSOLUTE_CAP,
)
from verl.trainer.ppo.urcr_v4_scorer import PROBE_TEMPLATE_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"resolved {name} must be {qualifier}")
    return result


def validate_resolved_method_document(document: Mapping[str, Any]) -> dict:
    """Validate frozen scientific choices and return the runtime numeric view."""
    if document.get("artifact") != "urcr_v4_resolved_method":
        raise ValueError("unexpected URCR-V4 resolved method identity")
    if int(document.get("schema_version", -1)) != 1:
        raise ValueError("unsupported URCR-V4 resolved method schema")
    if document.get("method_revision") != METHOD_REVISION:
        raise ValueError("URCR-V4 method revision changed")
    if document.get("frozen") is not True:
        raise ValueError("URCR-V4 method artifact is not frozen")

    search = document["search_utility"]
    if int(search["control_count"]) != CONTROL_COUNT:
        raise ValueError("URCR-V4 control count changed")
    if float(search["negative_scale"]) != SEARCH_NEGATIVE_SCALE:
        raise ValueError("URCR-V4 negative utility scale changed")
    if search["group_whitening"] is not False:
        raise ValueError("URCR-V4 search utility enabled group whitening")

    answer = document["answer_utility"]
    if float(answer["beta"]) != 0.5 or answer["std_normalization"] is not False:
        raise ValueError("URCR-V4 answer residual definition changed")
    if answer["residual_group"] != "rollout_group_id_and_binary_em":
        raise ValueError("URCR-V4 answer residual stratum changed")

    responsibility = document["responsibility"]
    if responsibility["probe_template_version"] != PROBE_TEMPLATE_VERSION:
        raise ValueError("URCR-V4 responsibility probe template changed")
    if responsibility["targets"] != "actual_sampled_action_content":
        raise ValueError("URCR-V4 responsibility target changed")
    if responsibility["intervention"] != (
        "position_preserving_outgoing_information_barrier"
    ):
        raise ValueError("URCR-V4 responsibility intervention changed")
    if responsibility["mapping"] != "positive_exponential":
        raise ValueError("URCR-V4 responsibility mapping changed")

    chunks = document["chunks"]
    if float(chunks["power"]) != 2.0:
        raise ValueError("URCR-V4 chunk power changed")
    if int(chunks["max_chunks"]) != MAX_THINK_CHUNKS:
        raise ValueError("URCR-V4 chunk count changed")
    if chunks["no_mass"] != "zero" or chunks["whole_think_fallback"] is not False:
        raise ValueError("URCR-V4 no-mass routing changed")

    credit = document["credit"]
    expected_credit = {
        "alpha_query": 1.0,
        "alpha_answer": ANSWER_ALPHA,
        "lambda_think_query": 1.0,
        "lambda_think_answer": 1.0,
        "search_trajectory_absolute_cap": SEARCH_TRAJECTORY_ABSOLUTE_CAP,
    }
    for key, expected in expected_credit.items():
        if float(credit[key]) != float(expected):
            raise ValueError(f"URCR-V4 credit constant changed: {key}")
    if credit["query_think_split"] is not False:
        raise ValueError("URCR-V4 enabled query/think split")

    metadata = document["metadata"]
    if metadata != {
        "online_pipeline": False,
        "credit_enabled": False,
        "diagnostics": "offline_sidecar_only",
    }:
        raise ValueError("URCR-V4 metadata boundary changed")

    local = document["local_objective"]
    if local["surrogate"] != "signed_tokenwise_ppo":
        raise ValueError("URCR-V4 local PPO surrogate changed")
    if local["span_reduction"] != "mean_once":
        raise ValueError("URCR-V4 local span reduction changed")
    if local["population_denominator"] != "original_trajectories":
        raise ValueError("URCR-V4 local denominator changed")
    if local["same_actor_forward_for_training"] is not True:
        raise ValueError("URCR-V4 local loss no longer shares the actor forward")
    warmup_steps = int(local["warmup_outer_steps"])
    if warmup_steps != 30:
        raise ValueError("URCR-V4 warmup changed")
    lambda_max = _finite(local["lambda_max"], "lambda_max", positive=True)
    if lambda_max > 0.2:
        raise ValueError("URCR-V4 lambda_max exceeds the frozen ceiling")

    legacy = document["legacy"]
    if any(value not in (False, None) for value in legacy.values()):
        raise ValueError("URCR-V4 resolved method enables a legacy path")

    scorer = document["scorer"]
    if int(scorer["micro_batch_size_per_gpu"]) != 32:
        raise ValueError("URCR-V4 scorer microbatch changed after calibration")
    if float(scorer["rollout_temperature"]) != 1.0:
        raise ValueError("URCR-V4 rollout temperature changed")

    return {
        "search_utility": {
            "delta": _finite(search["delta"], "search delta"),
            "scale": _finite(search["scale"], "search scale", positive=True),
        },
        "responsibility": {
            "search": {
                "delta": _finite(
                    responsibility["search"]["delta"],
                    "search responsibility delta",
                ),
                "scale": _finite(
                    responsibility["search"]["scale"],
                    "search responsibility scale",
                    positive=True,
                ),
            },
            "answer": {
                "delta": _finite(
                    responsibility["answer"]["delta"],
                    "answer responsibility delta",
                ),
                "scale": _finite(
                    responsibility["answer"]["scale"],
                    "answer responsibility scale",
                    positive=True,
                ),
            },
        },
        "local_loss": {
            "lambda_max": lambda_max,
            "warmup_steps": warmup_steps,
        },
    }


def load_resolved_method(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict, dict]:
    """Load one immutable method file and verify its implementation manifest."""
    artifact_path = Path(path).resolve()
    actual_sha256 = sha256_file(artifact_path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError("URCR-V4 resolved method SHA256 mismatch")
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    runtime = validate_resolved_method_document(document)

    code = document["code"]
    repo_root = Path(str(code["repo_root"])).resolve()
    for entry in code["implementation_files"]:
        relative = Path(str(entry["path"]))
        candidate = (repo_root / relative).resolve()
        if repo_root not in candidate.parents:
            raise ValueError("URCR-V4 implementation manifest escapes repo root")
        if sha256_file(candidate) != str(entry["sha256"]):
            raise ValueError(f"URCR-V4 implementation changed: {relative}")
    return runtime, {
        "resolved_method_artifact": str(artifact_path),
        "resolved_method_sha256": actual_sha256,
        "implementation_manifest_sha256": str(
            code["implementation_manifest_sha256"]
        ),
        "method_revision": METHOD_REVISION,
        "scorer": dict(document["scorer"]),
    }


def assert_urcr_v4_training_step_health(
    metrics: Mapping[str, Any],
    *,
    global_step: int,
    expected_trajectory_count: int,
) -> None:
    """Reject broken optimizer plumbing without gating valid V4 credit patterns."""

    def scalar(key: str) -> float:
        if key not in metrics:
            raise RuntimeError(f"URCR-V4 health check is missing metric: {key}")
        value = metrics[key]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RuntimeError(
                    f"URCR-V4 health metric is not scalar: {key}={value}"
                )
            value = value[0]
        if hasattr(value, "numel"):
            if int(value.numel()) != 1:
                raise RuntimeError(
                    f"URCR-V4 health metric is not scalar: {key}={value}"
                )
            value = value.detach().cpu().item()
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError(f"URCR-V4 health metric is non-finite: {key}={result}")
        return result

    for key in (
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/lr",
        "actor/entropy_loss",
        "actor/kl_loss",
        "urcr_v4/effective_local_loss",
        "urcr_v4/lambda_effective",
        "urcr_v4/original_trajectory_count",
        "urcr_v4/pipeline_seconds",
    ):
        scalar(key)
    if scalar("actor/grad_norm") <= 0:
        raise RuntimeError("URCR-V4 actor gradient is identically zero")
    learning_rate = scalar("actor/lr")
    if learning_rate < 0 or (int(global_step) > 1 and learning_rate <= 0):
        raise RuntimeError("URCR-V4 actor learning rate is invalid after step 1")
    if scalar("urcr_v4/lambda_effective") <= 0:
        raise RuntimeError("URCR-V4 local objective is not active")
    actual_trajectories = int(scalar("urcr_v4/original_trajectory_count"))
    if actual_trajectories != int(expected_trajectory_count):
        raise RuntimeError(
            "URCR-V4 trajectory population changed: "
            f"expected={expected_trajectory_count}, actual={actual_trajectories}"
        )
