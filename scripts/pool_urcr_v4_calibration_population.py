#!/usr/bin/env python3
"""Pool frozen URCR-V4 calibration batches without changing their formulas."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from verl.trainer.ppo.urcr_v4_calibration import (
    calibrate_responsibility_rows,
    distribution,
)
from verl.trainer.ppo.urcr_v4_method import (
    calibrate_search_utility,
    compute_search_utility,
)


EXPECTED_MODEL = "/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf"
EXPECTED_QUESTIONS_PER_BATCH = 128
EXPECTED_TRAJECTORIES_PER_BATCH = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pooled_search(documents: Sequence[dict]) -> dict:
    rows = [row for document in documents for row in document["search"]["rows"]]
    control_differences = [
        abs(
            float(row["view_scores"]["control_0"])
            - float(row["view_scores"]["control_1"])
        )
        for row in rows
        if not bool(row["control_used_replacement"])
    ]
    directed_gaps = [float(row["g_directed"]) for row in rows]
    numeric_differences = [
        float(
            document["numeric_reference"][
                "same_shape_repeat_max_abs_difference"
            ]
        )
        for document in documents
    ]
    calibration = calibrate_search_utility(
        control_pair_absolute_differences=control_differences,
        numeric_absolute_differences=numeric_differences,
        directed_gaps=directed_gaps,
    )
    utilities = []
    zero_reasons = Counter()
    for row in rows:
        view = row["view_scores"]
        result = compute_search_utility(
            real_score=float(view["real"]),
            empty_score=float(view["empty"]),
            control_scores=[
                float(view[f"control_{index}"]) for index in range(3)
            ],
            delta=calibration.delta,
            scale=calibration.scale,
            is_visible_exact_repeat=bool(row["visible_exact_repeat"]),
        )
        utilities.append(float(result.utility))
        if result.zero_reason:
            zero_reasons[result.zero_reason] += 1
    unscored = Counter()
    for document in documents:
        unscored.update(document["search"]["summary"]["unscored_reason_counts"])
    return {
        "calibration": asdict(calibration),
        "scored_anchor_count": len(rows),
        "control_pair_absolute_difference": distribution(control_differences),
        "g_incremental": distribution(
            [float(row["g_incremental"]) for row in rows]
        ),
        "g_control": distribution([float(row["g_control"]) for row in rows]),
        "g_directed": distribution(directed_gaps),
        "utility_recomputed_with_pooled_values": distribution(utilities),
        "utility_positive_count": sum(value > 0 for value in utilities),
        "utility_negative_count": sum(value < 0 for value in utilities),
        "utility_zero_count": sum(value == 0 for value in utilities),
        "recomputed_zero_reason_counts": dict(sorted(zero_reasons.items())),
        "unscored_reason_counts": dict(sorted(unscored.items())),
        "sign_conflict_count": sum(
            bool(row["contrast_sign_conflict"]) for row in rows
        ),
        "visible_exact_repeat_count": sum(
            bool(row["visible_exact_repeat"]) for row in rows
        ),
    }


def _pooled_responsibility(documents: Sequence[dict]) -> dict:
    rows = [
        row
        for document in documents
        for row in document["responsibility"]["rows"]
    ]
    repeats = [
        float(
            document["numeric_reference"][
                "same_shape_repeat_max_abs_difference"
            ]
        )
        for document in documents
    ]
    frozen = calibrate_responsibility_rows(
        rows,
        numeric_absolute_differences={"search": repeats, "answer": repeats},
    )
    return {
        "summary": frozen.summary,
        "full_score_used_for_calibration": "old_policy_log_probs",
        "plain_scorer_role": "backend_alignment_diagnostic",
        "backend_alignment_error_used_for_delta": True,
    }


def _pooled_answer(documents: Sequence[dict]) -> dict:
    rows = [row for document in documents for row in document["answer"]["rows"]]
    turn_ids = [str(row["turn_id"]) for row in rows]
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("answer rows overlap across calibration batches")
    group_sums: dict[tuple[str, int], float] = defaultdict(float)
    eligible = []
    for row in rows:
        if bool(row["valid_terminal_answer"]):
            eligible.append(row)
            group_sums[
                (str(row["rollout_group_id"]), int(row["binary_em"]))
            ] += float(row["residual"])
    values = [float(row["residual"]) for row in eligible]
    maximum_error = max(
        (abs(value) for value in group_sums.values()), default=0.0
    )
    if maximum_error > 1e-6:
        raise ValueError("pooled answer residuals violate within-stratum zero sum")
    return {
        "answer_turn_count": len(rows),
        "eligible_answer_count": len(eligible),
        "invalid_answer_count": len(rows) - len(eligible),
        "residual": distribution(values),
        "residual_positive_count": sum(value > 0 for value in values),
        "residual_negative_count": sum(value < 0 for value in values),
        "residual_zero_count": sum(value == 0 for value in values),
        "nonzero_residual_group_count": len(
            {
                (str(row["rollout_group_id"]), int(row["binary_em"]))
                for row in eligible
                if float(row["residual"]) != 0
            }
        ),
        "maximum_absolute_group_residual_sum": maximum_error,
    }


def pool_calibration_documents(documents: Sequence[dict]) -> dict:
    if not documents:
        raise ValueError("no calibration documents to pool")
    search = _pooled_search(documents)
    responsibility = _pooled_responsibility(documents)
    answer = _pooled_answer(documents)
    return {
        "search": search,
        "responsibility": responsibility,
        "answer": answer,
    }


def _prefix_stability(documents: Sequence[dict]) -> list[dict]:
    output = []
    for count in range(1, len(documents) + 1):
        pooled = pool_calibration_documents(documents[:count])
        output.append(
            {
                "batch_count": count,
                "question_count": count * EXPECTED_QUESTIONS_PER_BATCH,
                "trajectory_count": count * EXPECTED_TRAJECTORIES_PER_BATCH,
                "delta_U": pooled["search"]["calibration"]["delta"],
                "s_U": pooled["search"]["calibration"]["scale"],
                "delta_R_Q": pooled["responsibility"]["summary"][
                    "search_calibration"
                ]["delta"],
                "s_R_Q": pooled["responsibility"]["summary"][
                    "search_calibration"
                ]["scale"],
                "delta_R_A": pooled["responsibility"]["summary"][
                    "answer_calibration"
                ]["delta"],
                "s_R_A": pooled["responsibility"]["summary"][
                    "answer_calibration"
                ]["scale"],
            }
        )
    return output


def _last_three_ranges(prefixes: Sequence[dict]) -> dict:
    tail = prefixes[-min(3, len(prefixes)) :]
    keys = ("delta_U", "s_U", "delta_R_Q", "s_R_Q", "delta_R_A", "s_R_A")
    output = {}
    for key in keys:
        values = [float(row[key]) for row in tail]
        absolute_range = max(values) - min(values)
        denominator = max(abs(values[-1]), 1e-12)
        output[key] = {
            "absolute_range": absolute_range,
            "range_over_final_absolute_value": absolute_range / denominator,
        }
    return output


def _validate_and_load(population: dict) -> tuple[list[dict], list[dict]]:
    if population.get("artifact") != "urcr_v4_s150_calibration_population":
        raise ValueError("unexpected population manifest identity")
    entries = sorted(population["batches"], key=lambda row: int(row["batch_index"]))
    if [int(entry["batch_index"]) for entry in entries] != list(
        range(1, int(population["batch_count"]) + 1)
    ):
        raise ValueError("population batch indices are incomplete")
    selected_indices = []
    question_ids = []
    documents = []
    method_revision = None
    numeric_reference_sha = None
    for entry in entries:
        batch_index = int(entry["batch_index"])
        selected_indices.extend(int(value) for value in entry["selected_input_row_indices"])
        question_ids.extend(map(str, entry["question_ids"]))
        data_path = Path(str(entry["data_path"]))
        if not data_path.is_file() or _sha256(data_path) != str(entry["data_sha256"]):
            raise ValueError(f"batch {batch_index:02d} data is missing or changed")
        data_manifest_path = Path(str(entry["data_manifest"]))
        if not data_manifest_path.is_file():
            raise FileNotFoundError(data_manifest_path)
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        if list(map(int, data_manifest["selected_input_row_indices"])) != list(
            map(int, entry["selected_input_row_indices"])
        ):
            raise ValueError(f"batch {batch_index:02d} data manifest disagrees")

        artifact_path = Path(str(entry["calibration_artifact"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        if document.get("artifact") != "urcr_v4_s150_frozen_calibration":
            raise ValueError(f"batch {batch_index:02d} artifact identity is wrong")
        if not document.get("pass", False):
            raise ValueError(f"batch {batch_index:02d} did not pass")
        if str(document.get("model")) != EXPECTED_MODEL:
            raise ValueError(f"batch {batch_index:02d} uses a different model")
        if float(document.get("temperature", float("nan"))) != 1.0:
            raise ValueError(f"batch {batch_index:02d} temperature is not 1")
        if int(document.get("world_size", -1)) != 4:
            raise ValueError(f"batch {batch_index:02d} did not use four GPUs")
        if int(document.get("scorer_micro_batch_size_per_gpu", -1)) != 32:
            raise ValueError(f"batch {batch_index:02d} scorer microbatch changed")
        if int(document.get("question_group_count", -1)) != EXPECTED_QUESTIONS_PER_BATCH:
            raise ValueError(f"batch {batch_index:02d} question count changed")
        if int(document.get("original_trajectory_count", -1)) != EXPECTED_TRAJECTORIES_PER_BATCH:
            raise ValueError(f"batch {batch_index:02d} trajectory count changed")
        if bool(document["cost"].get("optimizer_update_executed", True)):
            raise ValueError(f"batch {batch_index:02d} executed an optimizer update")
        if batch_index > 1 and int(document.get("calibration_batch_index", -1)) != batch_index:
            raise ValueError(f"batch {batch_index:02d} runtime identity is missing")
        allowed_questions = set(map(str, entry["question_ids"]))
        observed_questions = {
            str(row["question_id"])
            for section in ("search", "answer")
            for row in document[section]["rows"]
        }
        if not observed_questions.issubset(allowed_questions):
            raise ValueError(f"batch {batch_index:02d} contains an unexpected question")
        if method_revision is None:
            method_revision = str(document["method_revision"])
            numeric_reference_sha = str(document["numeric_reference"]["sha256"])
        if str(document["method_revision"]) != method_revision:
            raise ValueError("calibration method revision changed across batches")
        if str(document["numeric_reference"]["sha256"]) != numeric_reference_sha:
            raise ValueError("numeric scorer reference changed across batches")
        document["_artifact_path"] = str(artifact_path.resolve())
        document["_batch_index"] = batch_index
        documents.append(document)
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("population input rows overlap")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("population question identities overlap")
    return entries, documents


def _cost_summary(documents: Sequence[dict]) -> dict:
    keys = sorted(
        {
            key
            for document in documents
            for key, value in document["cost"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: {
            "sum": math.fsum(float(document["cost"].get(key, 0.0)) for document in documents),
            "per_batch": distribution(
                [float(document["cost"].get(key, 0.0)) for document in documents]
            ),
        }
        for key in keys
    }


def _atomic_write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".writing")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pooled calibration: {args.output}")
    manifest_bytes = args.population_manifest.read_bytes()
    population = json.loads(manifest_bytes)
    entries, documents = _validate_and_load(population)
    pooled = pool_calibration_documents(documents)
    prefixes = _prefix_stability(documents)

    limitations = sorted(
        {
            str(value)
            for document in documents
            for value in document.get("limitations", [])
        }
    )
    if pooled["search"]["calibration"]["limited"]:
        limitations.append("pooled_search_scale_has_fewer_than_8_active_excesses")
    for action_type, label in (("search", "query"), ("answer", "answer")):
        if pooled["responsibility"]["summary"][f"{action_type}_calibration"]["limited"]:
            limitations.append(f"pooled_{label}_responsibility_uses_scale_fallback")
    if pooled["answer"]["nonzero_residual_group_count"] == 0:
        limitations.append("pooled_answer_residual_has_no_nonzero_group")

    result = {
        "schema_version": 1,
        "artifact": "urcr_v4_s150_pooled_calibration",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "population_manifest": str(args.population_manifest.resolve()),
        "population_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "protocol_authority": population["protocol_authority"],
        "method_revision": documents[0]["method_revision"],
        "model": EXPECTED_MODEL,
        "temperature": 1.0,
        "rollouts_per_question": 8,
        "batch_count": len(documents),
        "question_count": len(documents) * EXPECTED_QUESTIONS_PER_BATCH,
        "trajectory_count": len(documents) * EXPECTED_TRAJECTORIES_PER_BATCH,
        "pairwise_disjoint_questions": True,
        "batch_01_preserved_and_included": True,
        "optimizer_update_executed": False,
        "resolved_method_frozen": False,
        "pooling_rule": "concatenate_raw_empirical_rows_then_reapply_unchanged_formulas",
        "batch_artifacts": [
            {
                "batch_index": int(document["_batch_index"]),
                "path": document["_artifact_path"],
                "status": document["status"],
                "limitations": document.get("limitations", []),
                "search_calibration": document["search"]["summary"]["calibration"],
                "query_responsibility_calibration": document["responsibility"]["summary"]["search_calibration"],
                "answer_responsibility_calibration": document["responsibility"]["summary"]["answer_calibration"],
            }
            for document in documents
        ],
        "pooled": pooled,
        "cumulative_prefix_calibrations": prefixes,
        "last_three_prefix_ranges": _last_three_ranges(prefixes),
        "cost": _cost_summary(documents),
        "limitations": sorted(set(limitations)),
        "status": "pass_with_limitation" if limitations else "pass",
        "pass": True,
    }
    _atomic_write_json(result, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "batch_count": result["batch_count"],
                "question_count": result["question_count"],
                "trajectory_count": result["trajectory_count"],
                "search_calibration": pooled["search"]["calibration"],
                "query_responsibility_calibration": pooled["responsibility"]["summary"]["search_calibration"],
                "answer_responsibility_calibration": pooled["responsibility"]["summary"]["answer_calibration"],
                "last_three_prefix_ranges": result["last_three_prefix_ranges"],
                "resolved_method_frozen": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
