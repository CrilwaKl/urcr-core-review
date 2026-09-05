"""Deterministic online evidence-state construction for URCR.

The state at turn ``t`` is built only from observations before ``t``.  The
current observation is then aligned separately so acquisition can be audited
without leaking it into the remaining-evidence PI.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable

from verl.trainer.ppo.urcr_text import (
    align_supporting_evidence,
    extract_supporting_facts,
    normalize_match_text,
    normalized_alias_hit,
    parse_json_object,
)


ETA_DOC_ONLY = 0.25


def semantic_turn_bucket(turn_step: int) -> str:
    """Map zero-based environment turns to the frozen 1/2/3+ strata."""
    turn_step = int(turn_step)
    if turn_step < 0:
        raise ValueError(f"turn_step must be nonnegative, got {turn_step}")
    return "3+" if turn_step >= 2 else str(turn_step + 1)


def question_uid(row: dict[str, Any]) -> str:
    """Return a stable source-row identity, independent of rollout UUIDs."""
    source = str(row.get("data_source", "unknown"))
    dataset_index = int(row.get("dataset_index", -1))
    if dataset_index >= 0:
        return f"{source}:{dataset_index}"
    question = str(row.get("question", ""))
    digest = hashlib.sha256(f"{source}|{question}".encode()).hexdigest()[:20]
    return f"{source}:sha256:{digest}"


def turn_uid(row: dict[str, Any]) -> str:
    return f"{row['traj_uid']}:{int(row['turn_step'])}"


def _fact_records(metadata: Any, aliases: Iterable[Any]) -> list[dict[str, Any]]:
    records = []
    for fact in extract_supporting_facts(metadata):
        records.append(
            {
                "fact_id": fact.fact_id,
                "title": fact.title,
                "normalized_title": normalize_match_text(fact.title),
                "sent_id": int(fact.sent_id),
                "sentence": fact.sentence,
                "normalized_sentence": normalize_match_text(fact.sentence),
                "role": (
                    "answer_bearing"
                    if normalized_alias_hit(fact.sentence, aliases)
                    else "bridge"
                ),
            }
        )
    return records


def _state_label(covered_facts: set[str], all_facts: set[str]) -> str:
    if not covered_facts:
        return "none"
    if all_facts and all_facts.issubset(covered_facts):
        return "sufficient"
    return "partial"


def _valid_search(row: dict[str, Any]) -> bool:
    return bool(
        row.get("action_type") == "search"
        and any(row.get("search_content_mask") or [])
        and not row.get("invalid_action", False)
        and not row.get("empty_action", False)
        and not row.get("unclosed_action", False)
    )


def build_evidence_state_rows(
    frozen_rows: list[dict[str, Any]],
    *,
    include_future_audit: bool = True,
) -> list[dict[str, Any]]:
    """Build turn-before state and current-turn acquisition for frozen rows."""
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        by_trajectory[str(row["traj_uid"])].append(row)

    output: list[dict[str, Any]] = []
    for trajectory_rows in by_trajectory.values():
        trajectory_rows.sort(key=lambda item: int(item["turn_step"]))
        covered_docs: set[str] = set()
        covered_facts: set[str] = set()
        trajectory_output: list[dict[str, Any]] = []
        for row in trajectory_rows:
            aliases = [str(value) for value in row.get("ground_truth_aliases", [])]
            metadata = parse_json_object(row.get("metadata_json"))
            facts = _fact_records(metadata, aliases)
            fact_by_id = {fact["fact_id"]: fact for fact in facts}
            all_fact_ids = set(fact_by_id)
            all_doc_ids = {fact["normalized_title"] for fact in facts}
            remaining_fact_ids = all_fact_ids - covered_facts
            remaining_doc_ids = all_doc_ids - covered_docs
            valid_search = _valid_search(row)

            # Alignment of O_t is deliberately performed only after the
            # turn-before remaining sets have been frozen above.
            aligned = align_supporting_evidence(
                row.get("observation_text", ""), metadata, aliases
            )
            hit_docs = set(aligned["hit_supporting_doc_ids"]) if valid_search else set()
            hit_facts = set(aligned["hit_supporting_fact_ids"]) if valid_search else set()
            new_docs = hit_docs - covered_docs
            new_facts = hit_facts - covered_facts
            redundant_docs = hit_docs & covered_docs
            redundant_facts = hit_facts & covered_facts
            new_fact_docs = {
                fact_by_id[fact_id]["normalized_title"]
                for fact_id in new_facts
                if fact_id in fact_by_id
            }
            doc_only_new = new_docs - new_fact_docs
            new_bridge = {
                fact_id
                for fact_id in new_facts
                if fact_by_id.get(fact_id, {}).get("role") == "bridge"
            }
            new_answer = {
                fact_id
                for fact_id in new_facts
                if fact_by_id.get(fact_id, {}).get("role") == "answer_bearing"
            }
            fact_denominator = max(1, len(all_fact_ids))
            doc_denominator = max(1, len(all_doc_ids))
            g1 = min(1.0, len(new_facts) / fact_denominator) if facts else 0.0
            g2 = min(
                1.0,
                g1 + ETA_DOC_ONLY * len(doc_only_new) / doc_denominator,
            ) if facts else 0.0

            # Preserve the annotated order used by the official EviSD builder;
            # sorted ID lists below are only for set-valued audit fields.
            sorted_facts = list(facts)
            remaining_facts = [fact for fact in sorted_facts if fact["fact_id"] in remaining_fact_ids]
            record = {
                "question_uid": question_uid(row),
                "traj_uid": str(row["traj_uid"]),
                "turn_uid": turn_uid(row),
                "turn_step": int(row["turn_step"]),
                "dataset_index": int(row.get("dataset_index", -1)),
                "data_source": str(row.get("data_source", "unknown")),
                "question": str(row.get("question", "")),
                "ground_truth_aliases": aliases,
                "episode_reward": float(row.get("episode_reward", 0.0)),
                "valid_search": valid_search,
                "pi_available": bool(facts and valid_search),
                "evidence_available": bool(facts),
                "prior_evidence_state": _state_label(covered_facts, all_fact_ids),
                "gold_supporting_doc_ids": sorted(all_doc_ids),
                "gold_supporting_fact_ids": sorted(all_fact_ids),
                "gold_supporting_titles": [fact["title"] for fact in sorted_facts],
                "gold_supporting_sentences": [fact["sentence"] for fact in sorted_facts],
                "gold_supporting_fact_roles": [fact["role"] for fact in sorted_facts],
                "covered_doc_ids_before": sorted(covered_docs & all_doc_ids),
                "covered_fact_ids_before": sorted(covered_facts & all_fact_ids),
                "remaining_doc_ids_before": sorted(remaining_doc_ids),
                "remaining_fact_ids_before": sorted(remaining_fact_ids),
                "remaining_titles_before": [fact["title"] for fact in remaining_facts],
                "remaining_sentences_before": [fact["sentence"] for fact in remaining_facts],
                "remaining_roles_before": [fact["role"] for fact in remaining_facts],
                "hit_supporting_doc_ids": sorted(hit_docs),
                "hit_supporting_fact_ids": sorted(hit_facts),
                "new_supporting_doc_ids": sorted(new_docs),
                "new_supporting_fact_ids": sorted(new_facts),
                "new_bridge_fact_ids": sorted(new_bridge),
                "new_answer_bearing_fact_ids": sorted(new_answer),
                "doc_only_new_support_ids": sorted(doc_only_new),
                "redundant_support_doc_ids": sorted(redundant_docs),
                "redundant_support_fact_ids": sorted(redundant_facts),
                "new_support_doc_count": len(new_docs),
                "new_support_fact_count": len(new_facts),
                "new_bridge_fact_count": len(new_bridge),
                "new_answer_bearing_fact_count": len(new_answer),
                "redundant_support_doc_count": len(redundant_docs),
                "redundant_support_fact_count": len(redundant_facts),
                "no_annotated_support_hit": not bool(hit_docs or hit_facts),
                "remaining_doc_count_before": len(remaining_doc_ids),
                "remaining_fact_count_before": len(remaining_fact_ids),
                "remaining_doc_count_after": len(all_doc_ids - (covered_docs | hit_docs)),
                "remaining_fact_count_after": len(all_fact_ids - (covered_facts | hit_facts)),
                "supporting_doc_hit": bool(hit_docs),
                "supporting_fact_hit": bool(hit_facts),
                "bridge_only_supporting_hit": bool(new_bridge),
                "answer_bearing_supporting_hit": bool(new_answer),
                "observation_alias_hit": normalized_alias_hit(
                    row.get("observation_text", ""), aliases
                ),
                "retrieved_document_count": int(aligned["retrieved_document_count"]),
                "retrieved_titles": list(aligned["retrieved_titles"]),
                "g1_fact_credit": float(g1),
                "g2_hierarchical_credit": float(g2),
                "eta_doc_only": ETA_DOC_ONLY,
                "match_method": aligned["match_method"],
            }
            trajectory_output.append(record)
            covered_docs.update(hit_docs)
            covered_facts.update(hit_facts)

        if include_future_audit:
            # Deterministic offline audits only; the online source path disables
            # this block so future-derived fields are never constructed there.
            for index, record in enumerate(trajectory_output):
                future = trajectory_output[index + 1 :]
                record["future_new_support_fact_count"] = sum(
                    int(item["new_support_fact_count"]) for item in future
                )
                record["future_new_support_doc_count"] = sum(
                    int(item["new_support_doc_count"]) for item in future
                )
                record["remaining_search_count"] = sum(
                    bool(item["valid_search"]) for item in future
                )
        output.extend(trajectory_output)

    output.sort(
        key=lambda row: (
            row["dataset_index"], row["traj_uid"], row["turn_step"]
        )
    )
    keys = [row["turn_uid"] for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate turn_uid in evidence-state output")
    return output
