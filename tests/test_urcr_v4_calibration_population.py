from __future__ import annotations

from collections import Counter

import pyarrow as pa
import pytest

from scripts.prepare_urcr_v4_calibration_population import (
    POPULATION_SEED,
    select_population_batches,
)
from scripts.pool_urcr_v4_calibration_population import (
    pool_calibration_documents,
)


def _table(rows_per_source: int = 256) -> pa.Table:
    rows = []
    for source in ("nq", "hotpotqa"):
        offset = 0 if source == "nq" else rows_per_source
        for local_index in range(rows_per_source):
            index = offset + local_index
            rows.append(
                {
                    "data_source": source,
                    "extra_info": {"index": index, "split": "train"},
                    "metadata": {"must_leave_online_pipeline": True},
                }
            )
    return pa.Table.from_pylist(rows)


def test_population_selection_is_fixed_balanced_disjoint_and_metadata_free() -> None:
    table = _table()
    excluded = list(range(64)) + list(range(256, 320))
    first = select_population_batches(
        table,
        excluded_indices=excluded,
        additional_batch_count=3,
        seed=POPULATION_SEED,
    )
    second = select_population_batches(
        table,
        excluded_indices=excluded,
        additional_batch_count=3,
        seed=POPULATION_SEED,
    )
    assert [indices for _, indices in first] == [indices for _, indices in second]
    seen = set(excluded)
    for (batch, indices), (repeat, _) in zip(first, second, strict=True):
        assert batch.equals(repeat)
        assert len(batch) == 128
        assert Counter(batch["data_source"].to_pylist()) == {
            "nq": 64,
            "hotpotqa": 64,
        }
        assert "metadata" not in batch.column_names
        assert not seen.intersection(indices)
        seen.update(indices)


def test_population_selection_rejects_duplicate_exclusions() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        select_population_batches(
            _table(),
            excluded_indices=[0, 0],
            additional_batch_count=1,
        )


def _calibration_document(index: int) -> dict:
    question_id = f"nq:train:row:{index}"
    group = f"step:1:{question_id}:group:0"
    return {
        "numeric_reference": {"same_shape_repeat_max_abs_difference": 0.0},
        "search": {
            "rows": [
                {
                    "question_id": question_id,
                    "turn_id": f"{group}:rollout:0:turn:0",
                    "control_used_replacement": False,
                    "view_scores": {
                        "real": 3.0,
                        "empty": 0.0,
                        "control_0": 1.0,
                        "control_1": 1.2,
                        "control_2": 1.4,
                    },
                    "g_incremental": 3.0,
                    "g_control": 1.8,
                    "g_directed": 1.8,
                    "visible_exact_repeat": False,
                    "contrast_sign_conflict": False,
                }
            ],
            "summary": {"unscored_reason_counts": {}},
        },
        "responsibility": {
            "rows": [
                {
                    "turn_id": f"{group}:search",
                    "action_type": "search",
                    "is_zero_utility_shadow": False,
                    "old_full_score": -1.0,
                    "plain_scorer_score": -1.0,
                    "whole_masked_score": -2.0,
                    "chunk_masked_scores": [-1.5],
                },
                {
                    "turn_id": f"{group}:answer",
                    "action_type": "answer",
                    "is_zero_utility_shadow": False,
                    "old_full_score": -1.0,
                    "plain_scorer_score": -1.0,
                    "whole_masked_score": -2.0,
                    "chunk_masked_scores": [-1.5],
                },
            ]
        },
        "answer": {
            "rows": [
                {
                    "turn_id": f"{group}:rollout:0:answer",
                    "rollout_group_id": group,
                    "binary_em": 0,
                    "valid_terminal_answer": True,
                    "residual": 0.1,
                },
                {
                    "turn_id": f"{group}:rollout:1:answer",
                    "rollout_group_id": group,
                    "binary_em": 0,
                    "valid_terminal_answer": True,
                    "residual": -0.1,
                },
            ]
        },
    }


def test_pool_reapplies_formulas_to_concatenated_raw_rows() -> None:
    pooled = pool_calibration_documents(
        [_calibration_document(1), _calibration_document(2)]
    )
    assert pooled["search"]["scored_anchor_count"] == 2
    assert pooled["search"]["calibration"]["delta"] == pytest.approx(0.2)
    assert pooled["search"]["calibration"]["scale"] == 1.0
    assert pooled["responsibility"]["summary"]["anchor_count"] == 4
    assert pooled["answer"]["eligible_answer_count"] == 4
    assert pooled["answer"]["maximum_absolute_group_residual_sum"] < 1e-12
