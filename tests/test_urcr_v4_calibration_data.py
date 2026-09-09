from __future__ import annotations

from collections import Counter

import pyarrow as pa

from scripts.prepare_urcr_v4_calibration_data import (
    DIAGNOSTIC_SEED,
    select_calibration_rows,
)


def test_calibration_selection_is_fixed_balanced_and_metadata_free() -> None:
    rows = []
    for source in ("nq", "hotpotqa"):
        for index in range(80):
            rows.append(
                {
                    "data_source": source,
                    "extra_info": {"index": index, "split": "train"},
                    "reward_model": {"ground_truth": {"target": [str(index)]}},
                    "metadata": {"supporting_facts": ["must not leave input"]},
                }
            )
    table = pa.Table.from_pylist(rows)
    first, first_indices = select_calibration_rows(table, seed=DIAGNOSTIC_SEED)
    second, second_indices = select_calibration_rows(table, seed=DIAGNOSTIC_SEED)
    assert first_indices == second_indices
    assert first.equals(second)
    assert len(first) == 128
    assert Counter(first["data_source"].to_pylist()) == {
        "nq": 64,
        "hotpotqa": 64,
    }
    assert "metadata" not in first.column_names
    assert "metadata" in table.column_names
    assert "reward_model" in first.column_names
