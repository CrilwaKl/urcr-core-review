#!/usr/bin/env python3
"""Re-freeze responsibility scales from a saved V4 calibration artifact."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from verl.trainer.ppo.urcr_v4_calibration import calibrate_responsibility_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    if data.get("artifact") != "urcr_v4_s150_frozen_calibration":
        raise ValueError("unexpected calibration artifact identity")
    repeat = float(
        data["numeric_reference"]["same_shape_repeat_max_abs_difference"]
    )
    frozen = calibrate_responsibility_rows(
        data["responsibility"]["rows"],
        numeric_absolute_differences={
            "search": [repeat],
            "answer": [repeat],
        },
    )
    data["responsibility"]["summary"] = frozen.summary
    data["responsibility"]["rows"] = list(frozen.rows)
    data["responsibility"]["full_score_used_for_calibration"] = (
        "old_policy_log_probs"
    )
    data["responsibility"]["plain_scorer_role"] = (
        "backend_alignment_diagnostic"
    )
    data["responsibility"]["backend_alignment_error_used_for_delta"] = True
    data["responsibility"].pop("old_logprob_identity_requires_review", None)
    data["limitations"] = [
        value
        for value in data.get("limitations", [])
        if value != "old_logprob_vs_sdpa_plain_numeric_drift"
    ]
    data["status"] = (
        "pass_with_limitation" if data["limitations"] else "pass"
    )
    data["schema_version"] = 2
    data["responsibility_finalized_at"] = datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).isoformat(timespec="seconds")
    temporary = artifact.with_suffix(artifact.suffix + ".writing")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, artifact)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "search_calibration": frozen.summary["search_calibration"],
                "answer_calibration": frozen.summary["answer_calibration"],
                "backend_alignment_error_used_for_delta": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
