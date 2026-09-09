#!/usr/bin/env python3
"""Select the frozen URCR-V4 S150 diagnostic questions without evidence metadata."""

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


DIAGNOSTIC_SEED = 20260908
SOURCE_COUNTS = {"hotpotqa": 64, "nq": 64}


def select_calibration_rows(
    table: pa.Table,
    *,
    seed: int = DIAGNOSTIC_SEED,
) -> tuple[pa.Table, list[int]]:
    if "data_source" not in table.column_names:
        raise ValueError("training table has no data_source column")
    if "extra_info" not in table.column_names:
        raise ValueError("training table has no stable extra_info identity")
    sources = np.asarray(table["data_source"].to_pylist(), dtype=object)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for source, count in SOURCE_COUNTS.items():
        candidates = np.flatnonzero(sources == source)
        if len(candidates) < count:
            raise ValueError(
                f"training table has {len(candidates)} {source} rows, requires {count}"
            )
        selected.extend(
            int(value) for value in rng.choice(candidates, size=count, replace=False)
        )
    selected = [selected[index] for index in rng.permutation(len(selected))]
    output = table.take(pa.array(selected, type=pa.int64()))
    if "metadata" in output.column_names:
        output = output.drop(["metadata"])
    if "metadata" in output.column_names:
        raise RuntimeError("evidence metadata remained in calibration output")
    splits = {
        str(value.get("split", ""))
        for value in output["extra_info"].to_pylist()
        if isinstance(value, dict)
    }
    if splits != {"train"}:
        raise ValueError(f"calibration input must contain only train rows, got {splits}")
    counts = Counter(output["data_source"].to_pylist())
    if dict(counts) != SOURCE_COUNTS:
        raise RuntimeError(f"unexpected selected source counts: {dict(counts)}")
    return output, selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_parquet(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite calibration data or manifest")
    table = pq.read_table(args.input)
    selected, row_indices = select_calibration_rows(table)
    _write_atomic_parquet(selected, args.output)
    manifest = {
        "schema_version": 1,
        "artifact": "urcr_v4_s150_calibration_questions",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "seed": DIAGNOSTIC_SEED,
        "source_counts": SOURCE_COUNTS,
        "selected_input_row_indices": row_indices,
        "row_count": len(selected),
        "removed_columns": ["metadata"],
        "metadata_online_pipeline": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
