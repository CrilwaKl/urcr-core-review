#!/usr/bin/env python3
"""Extend the frozen URCR-V4 calibration set with disjoint random batches."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

POPULATION_SEED = 20260909
SOURCE_COUNTS = {"hotpotqa": 64, "nq": 64}
MINIMUM_TOTAL_BATCHES = 5
MAXIMUM_TOTAL_BATCHES = 10
QUESTIONS_PER_BATCH = sum(SOURCE_COUNTS.values())
ROLLOUTS_PER_QUESTION = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_table(table: pa.Table) -> tuple[np.ndarray, list[dict]]:
    required = {"data_source", "extra_info"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"training table is missing columns: {sorted(missing)}")
    sources = np.asarray(table["data_source"].to_pylist(), dtype=object)
    extra_info = table["extra_info"].to_pylist()
    if len(extra_info) != len(table) or not all(
        isinstance(value, dict) for value in extra_info
    ):
        raise ValueError("training table has invalid extra_info identities")
    if {str(value.get("split", "")) for value in extra_info} != {"train"}:
        raise ValueError("calibration population must come only from the train split")
    return sources, extra_info


def question_identity(source: str, extra_info: dict) -> str:
    return (
        f"{source}:{extra_info['split']}:row:{int(extra_info['index'])}"
    )


def select_population_batches(
    table: pa.Table,
    *,
    excluded_indices: list[int] | tuple[int, ...],
    additional_batch_count: int,
    seed: int = POPULATION_SEED,
) -> tuple[tuple[pa.Table, tuple[int, ...]], ...]:
    """Draw balanced, pairwise-disjoint batches after fixed exclusions."""
    if additional_batch_count <= 0:
        raise ValueError("additional_batch_count must be positive")
    sources, _ = _validated_table(table)
    excluded = {int(value) for value in excluded_indices}
    if len(excluded) != len(excluded_indices):
        raise ValueError("excluded row indices contain duplicates")
    if any(value < 0 or value >= len(table) for value in excluded):
        raise ValueError("excluded row index is out of bounds")

    rng = np.random.default_rng(int(seed))
    draws_by_source: dict[str, np.ndarray] = {}
    for source, count in SOURCE_COUNTS.items():
        candidates = np.asarray(
            [
                int(value)
                for value in np.flatnonzero(sources == source)
                if int(value) not in excluded
            ],
            dtype=np.int64,
        )
        required = additional_batch_count * count
        if len(candidates) < required:
            raise ValueError(
                f"training table has {len(candidates)} unused {source} rows, "
                f"requires {required}"
            )
        draws_by_source[source] = rng.choice(
            candidates, size=required, replace=False
        )

    output = []
    all_selected = set(excluded)
    for batch_offset in range(additional_batch_count):
        row_indices: list[int] = []
        for source, count in SOURCE_COUNTS.items():
            start = batch_offset * count
            row_indices.extend(
                int(value)
                for value in draws_by_source[source][start : start + count]
            )
        row_indices = [
            row_indices[index] for index in rng.permutation(len(row_indices))
        ]
        if all_selected.intersection(row_indices):
            raise RuntimeError("calibration population selection overlapped")
        all_selected.update(row_indices)
        batch = table.take(pa.array(row_indices, type=pa.int64()))
        if "metadata" in batch.column_names:
            batch = batch.drop(["metadata"])
        if "metadata" in batch.column_names:
            raise RuntimeError("evidence metadata remained in calibration batch")
        if Counter(batch["data_source"].to_pylist()) != Counter(SOURCE_COUNTS):
            raise RuntimeError("calibration batch is not source balanced")
        output.append((batch, tuple(row_indices)))
    return tuple(output)


def _atomic_write_parquet(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".writing", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


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


def _batch_entry(
    *,
    batch_index: int,
    data_path: Path,
    data_manifest: Path,
    calibration_artifact: Path,
    selected_indices: list[int] | tuple[int, ...],
    table: pa.Table,
) -> dict:
    sources, extra_info = _validated_table(table)
    question_ids = [
        question_identity(str(sources[index]), extra_info[index])
        for index in selected_indices
    ]
    if len(set(question_ids)) != QUESTIONS_PER_BATCH:
        raise ValueError(f"batch {batch_index:02d} has duplicate question identities")
    return {
        "batch_index": batch_index,
        "question_count": QUESTIONS_PER_BATCH,
        "trajectory_count": QUESTIONS_PER_BATCH * ROLLOUTS_PER_QUESTION,
        "source_counts": SOURCE_COUNTS,
        "selected_input_row_indices": list(map(int, selected_indices)),
        "question_ids": question_ids,
        "data_path": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "data_manifest": str(data_manifest.resolve()),
        "calibration_artifact": str(calibration_artifact.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-01-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--calibration-artifact-dir", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--total-batches", type=int, default=MAXIMUM_TOTAL_BATCHES)
    parser.add_argument("--seed", type=int, default=POPULATION_SEED)
    args = parser.parse_args()

    if not MINIMUM_TOTAL_BATCHES <= args.total_batches <= MAXIMUM_TOTAL_BATCHES:
        raise ValueError(
            f"total batches must be in [{MINIMUM_TOTAL_BATCHES}, "
            f"{MAXIMUM_TOTAL_BATCHES}]"
        )
    if not args.input.is_file() or not args.batch_01_manifest.is_file():
        raise FileNotFoundError("missing source data or batch_01 manifest")
    if args.population_manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite population manifest: {args.population_manifest}"
        )

    batch_01_manifest = json.loads(
        args.batch_01_manifest.read_text(encoding="utf-8")
    )
    expected_input = str(args.input.resolve())
    input_sha256 = _sha256(args.input)
    if batch_01_manifest.get("input") != expected_input:
        raise ValueError("batch_01 manifest uses a different source parquet")
    if batch_01_manifest.get("input_sha256") != input_sha256:
        raise ValueError("batch_01 source parquet hash changed")
    if int(batch_01_manifest.get("row_count", -1)) != QUESTIONS_PER_BATCH:
        raise ValueError("batch_01 does not contain 128 questions")
    if Counter(batch_01_manifest.get("source_counts", {})) != Counter(
        SOURCE_COUNTS
    ):
        raise ValueError("batch_01 is not 64 HotpotQA + 64 NQ")
    batch_01_data = Path(str(batch_01_manifest["output"]))
    if not batch_01_data.is_file() or _sha256(batch_01_data) != str(
        batch_01_manifest["output_sha256"]
    ):
        raise ValueError("batch_01 parquet is missing or changed")

    expected_outputs = []
    for batch_index in range(2, args.total_batches + 1):
        expected_outputs.extend(
            [
                args.output_dir / f"calibration_batch_{batch_index:02d}.parquet",
                args.manifest_dir / f"data_batch_{batch_index:02d}_manifest.json",
            ]
        )
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite calibration population files: "
            + ", ".join(existing)
        )

    table = pq.read_table(args.input)
    excluded = [int(value) for value in batch_01_manifest["selected_input_row_indices"]]
    selections = select_population_batches(
        table,
        excluded_indices=excluded,
        additional_batch_count=args.total_batches - 1,
        seed=args.seed,
    )

    batch_entries = []
    batch_01_artifact = args.calibration_artifact_dir.parent / "s150_calibration.json"
    batch_entries.append(
        _batch_entry(
            batch_index=1,
            data_path=batch_01_data,
            data_manifest=args.batch_01_manifest,
            calibration_artifact=batch_01_artifact,
            selected_indices=excluded,
            table=table,
        )
    )
    for batch_index, (selected, indices) in enumerate(selections, start=2):
        data_path = args.output_dir / f"calibration_batch_{batch_index:02d}.parquet"
        manifest_path = (
            args.manifest_dir / f"data_batch_{batch_index:02d}_manifest.json"
        )
        calibration_artifact = (
            args.calibration_artifact_dir
            / f"calibration_batch_{batch_index:02d}.json"
        )
        _atomic_write_parquet(selected, data_path)
        entry = _batch_entry(
            batch_index=batch_index,
            data_path=data_path,
            data_manifest=manifest_path,
            calibration_artifact=calibration_artifact,
            selected_indices=indices,
            table=table,
        )
        batch_manifest = {
            "schema_version": 2,
            "artifact": "urcr_v4_s150_calibration_questions",
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                timespec="seconds"
            ),
            "input": expected_input,
            "input_sha256": input_sha256,
            "population_seed": int(args.seed),
            "population_batch_index": batch_index,
            "sampling_rule": "balanced_random_without_replacement_after_batch_01",
            "temperature": 1.0,
            "rollouts_per_question": ROLLOUTS_PER_QUESTION,
            "metadata_online_pipeline": False,
            "removed_columns": ["metadata"],
            **entry,
            "output": entry["data_path"],
            "output_sha256": entry["data_sha256"],
            "row_count": entry["question_count"],
        }
        _atomic_write_json(batch_manifest, manifest_path)
        batch_entries.append(entry)

    selected_sets = [
        set(entry["selected_input_row_indices"]) for entry in batch_entries
    ]
    if any(
        selected_sets[left].intersection(selected_sets[right])
        for left in range(len(selected_sets))
        for right in range(left + 1, len(selected_sets))
    ):
        raise RuntimeError("population batches are not pairwise disjoint")

    population = {
        "schema_version": 1,
        "artifact": "urcr_v4_s150_calibration_population",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "protocol_authority": "user_instruction_2026-09-09",
        "protocol_note": (
            "This population-size override supersedes only the original two-batch "
            "diagnostic cap; all frozen scientific definitions are unchanged."
        ),
        "input": expected_input,
        "input_sha256": input_sha256,
        "diagnostic_model": "/data1/kongmu/model_checkpoints/base_grpo_3b_s150_hf",
        "population_seed": int(args.seed),
        "sampling_rule": "balanced_random_without_replacement",
        "pairwise_disjoint": True,
        "temperature": 1.0,
        "rollouts_per_question": ROLLOUTS_PER_QUESTION,
        "optimizer_updates": False,
        "batch_count": args.total_batches,
        "question_count": args.total_batches * QUESTIONS_PER_BATCH,
        "trajectory_count": (
            args.total_batches * QUESTIONS_PER_BATCH * ROLLOUTS_PER_QUESTION
        ),
        "batch_01_preserved": True,
        "batches": batch_entries,
    }
    _atomic_write_json(population, args.population_manifest)
    print(
        json.dumps(
            {
                "population_manifest": str(args.population_manifest.resolve()),
                "batch_count": population["batch_count"],
                "question_count": population["question_count"],
                "trajectory_count": population["trajectory_count"],
                "pairwise_disjoint": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
