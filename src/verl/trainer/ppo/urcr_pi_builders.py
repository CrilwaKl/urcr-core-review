"""Role-conditioned privileged-information builders for URCR audits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable

from verl.trainer.ppo.urcr_text import (
    exact_alias_spans,
    lexical_jaccard,
    longest_common_word_substring,
    normalize_match_text,
    normalized_alias_hit,
    parse_json_object,
    parse_retrieved_documents,
)


QUERY_STATE_HEADER = (
    "[Privileged Search State]\n"
    "Supporting evidence still missing at this turn:\n"
)
QUERY_SCAFFOLD_PREFIX = QUERY_STATE_HEADER + "[CONTENT OMITTED]\n\n"
NO_REMAINING_EVIDENCE = "No annotated supporting evidence remains."
NO_MATCHED_NEGATIVE = "[MATCHED NEGATIVE UNAVAILABLE]"

R1_PREFIX = (
    "[Privileged Reasoning Role]\n"
    "Assess whether the sampled reasoning identifies the current\n"
    "information need and prepares the next search action.\n"
    "Do not solve the question or reveal an answer.\n\n"
)


@dataclass(frozen=True)
class NegativeCandidate:
    candidate_id: str
    text: str
    entry_count: int
    priority: int
    donor_question_uid: str | None
    donor_turn_uid: str | None
    source: str
    data_source: str
    turn_bucket: str


def _truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def serialize_evidence(
    titles: Iterable[Any],
    sentences: Iterable[Any],
    *,
    max_docs: int = 2,
    max_sentences_per_doc: int = 2,
    max_chars_per_doc: int = 400,
    max_total_chars: int = 1000,
) -> tuple[str, int]:
    """Serialize evidence with the official EviSD document budgets."""
    grouped: list[tuple[str, list[str]]] = []
    title_to_position: dict[str, int] = {}
    for raw_title, raw_sentence in zip(titles, sentences):
        title = str(raw_title)
        sentence = str(raw_sentence).strip()
        if title not in title_to_position:
            if len(grouped) >= max_docs:
                continue
            title_to_position[title] = len(grouped)
            grouped.append((title, []))
        position = title_to_position.get(title)
        if position is None:
            continue
        values = grouped[position][1]
        if sentence and len(values) < max_sentences_per_doc:
            values.append(sentence)

    evidence: list[str] = []
    total_chars = 0
    for title, values in grouped:
        body = _truncate_text(" ".join(values), max_chars_per_doc) if values else ""
        item = f"- {title}: {body}" if body else f"- {title}"
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        item = _truncate_text(item, remaining)
        evidence.append(item)
        total_chars += len(item)
    return ("\n".join(evidence) if evidence else "None provided", len(evidence))


def build_remaining_evidence(state: dict[str, Any]) -> tuple[str, int]:
    if not state.get("evidence_available", False):
        return "None provided", 0
    if not state.get("remaining_fact_ids_before"):
        return NO_REMAINING_EVIDENCE, 0
    return serialize_evidence(
        state.get("remaining_titles_before", []),
        state.get("remaining_sentences_before", []),
    )


def build_query_prefixes(
    frozen_row: dict[str, Any],
    state: dict[str, Any],
    negative: NegativeCandidate | None,
) -> dict[str, Any]:
    """Build Q0/Q1/Q2 prefixes without changing the sampled query."""
    pi_available = bool(state.get("pi_available", False))
    if not pi_available:
        return {
            "query_pi_available": False,
            "q0_prefix": "",
            "q1_positive_prefix": "",
            "q1_scaffold_prefix": "",
            "q2_negative_prefix": "",
            "q1_evidence_text": "",
            "q1_entry_count": 0,
            "q2_negative_available": False,
            "q2_negative_text": "",
            "q2_negative_id": None,
            "q2_negative_source": None,
            "q2_negative_donor_question_uid": None,
            "q2_negative_donor_turn_uid": None,
        }
    remaining, entry_count = build_remaining_evidence(state)
    q0_prefix = str(frozen_row.get("privileged_prefix", ""))
    q1_positive = QUERY_STATE_HEADER + remaining + "\n\n"
    negative_available = negative is not None
    negative_text = negative.text if negative else NO_MATCHED_NEGATIVE
    return {
        "query_pi_available": True,
        "q0_prefix": q0_prefix,
        "q1_positive_prefix": q1_positive,
        "q1_scaffold_prefix": QUERY_SCAFFOLD_PREFIX,
        "q2_negative_prefix": QUERY_STATE_HEADER + negative_text + "\n\n",
        "q1_evidence_text": remaining,
        "q1_entry_count": int(entry_count),
        "q2_negative_available": negative_available,
        "q2_negative_text": negative_text,
        "q2_negative_id": negative.candidate_id if negative else None,
        "q2_negative_source": negative.source if negative else None,
        "q2_negative_donor_question_uid": negative.donor_question_uid if negative else None,
        "q2_negative_donor_turn_uid": negative.donor_turn_uid if negative else None,
    }


def _answer_mask(text: str, aliases: Iterable[Any]) -> tuple[str, int]:
    spans = exact_alias_spans(text, aliases)
    if not spans:
        return text, 0
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        pieces.extend([text[cursor:start], "[ANSWER]"])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(spans)


def build_think_prefixes(
    frozen_row: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    aliases = [str(value) for value in frozen_row.get("ground_truth_aliases", [])]
    evidence_available = bool(state.get("evidence_available", False))
    if not evidence_available:
        return {
            "think_pi_available": False,
            "r0_prefix": "",
            "r1_prefix": R1_PREFIX,
            "r2a_prefix": R1_PREFIX,
            "r2b_prefix": R1_PREFIX,
            "r3_prefix": "",
            "r2b_unmasked_text": "",
            "r2b_masked_text": "",
            "r2b_alias_mask_count": 0,
        }

    r2a = (
        "[Privileged Reasoning State]\n"
        f"Evidence status: {state['prior_evidence_state']}\n"
        f"Observed supporting-document count: {len(state['covered_doc_ids_before'])}\n"
        f"Remaining supporting-document count: {state['remaining_doc_count_before']}\n"
        f"Remaining supporting-fact count: {state['remaining_fact_count_before']}\n"
        "The sampled reasoning should identify the current information\n"
        "need and prepare an appropriate next action. Do not reveal the answer.\n\n"
    )
    observed_titles = ", ".join(state.get("covered_doc_ids_before", [])) or "None"
    missing_entries = [
        f"- {title}: {sentence}"
        for title, sentence in zip(
            state.get("remaining_titles_before", []),
            state.get("remaining_sentences_before", []),
        )
    ]
    missing_text = "\n".join(missing_entries) or NO_REMAINING_EVIDENCE
    r2b_content = (
        r2a.rstrip()
        + "\n"
        + f"Observed supporting titles: {observed_titles}\n"
        + "Still-missing supporting titles / fact snippets:\n"
        + missing_text
        + "\n\n"
    )
    r2b_masked, mask_count = _answer_mask(r2b_content, aliases)
    return {
        "think_pi_available": True,
        "r0_prefix": "",
        "r1_prefix": R1_PREFIX,
        "r2a_prefix": r2a,
        "r2b_prefix": r2b_masked,
        "r3_prefix": str(frozen_row.get("privileged_prefix", "")),
        "r2b_unmasked_text": r2b_content,
        "r2b_masked_text": r2b_masked,
        "r2b_alias_mask_count": int(mask_count),
    }


def _safe_negative(text: str, state: dict[str, Any], aliases: Iterable[Any]) -> bool:
    normalized = normalize_match_text(text)
    if not normalized or normalized_alias_hit(text, aliases):
        return False
    forbidden = [
        *state.get("gold_supporting_doc_ids", []),
        *(normalize_match_text(value) for value in state.get("gold_supporting_sentences", [])),
    ]
    return not any(value and value in normalized for value in forbidden)


def local_negative_candidates(
    frozen_row: dict[str, Any], state: dict[str, Any]
) -> list[NegativeCandidate]:
    """Return priority-1 retrieval and priority-2 same-question distractors."""
    aliases = [str(value) for value in frozen_row.get("ground_truth_aliases", [])]
    gold_titles = set(state.get("gold_supporting_doc_ids", []))
    candidates: list[NegativeCandidate] = []
    for document in parse_retrieved_documents(frozen_row.get("observation_text", "")):
        if normalize_match_text(document.title) in gold_titles:
            continue
        text, count = serialize_evidence([document.title], [document.body])
        if _safe_negative(text, state, aliases):
            digest = hashlib.sha256(text.encode()).hexdigest()[:20]
            candidates.append(
                NegativeCandidate(
                    candidate_id=f"retrieved:{state['turn_uid']}:{document.rank}:{digest}",
                    text=text,
                    entry_count=count,
                    priority=1,
                    donor_question_uid=state["question_uid"],
                    donor_turn_uid=state["turn_uid"],
                    source="retrieved_non_supporting",
                    data_source=str(state["data_source"]),
                    turn_bucket="3+" if int(state["turn_step"]) >= 3 else str(int(state["turn_step"])),
                )
            )

    metadata = parse_json_object(frozen_row.get("metadata_json"))
    context = metadata.get("context") if isinstance(metadata, dict) else None
    if isinstance(context, dict):
        titles = list(context.get("title") or [])
        sentences = list(context.get("sentences") or [])
        for index, title in enumerate(titles):
            if normalize_match_text(title) in gold_titles:
                continue
            raw_sentences = sentences[index] if index < len(sentences) else []
            if hasattr(raw_sentences, "tolist"):
                raw_sentences = raw_sentences.tolist()
            body = " ".join(str(value) for value in list(raw_sentences)[:2])
            text, count = serialize_evidence([title], [body])
            if _safe_negative(text, state, aliases):
                candidates.append(
                    NegativeCandidate(
                        candidate_id=f"context:{state['question_uid']}:{index}",
                        text=text,
                        entry_count=count,
                        priority=2,
                        donor_question_uid=state["question_uid"],
                        donor_turn_uid=None,
                        source="same_question_distractor",
                        data_source=str(state["data_source"]),
                        turn_bucket="3+" if int(state["turn_step"]) >= 3 else str(int(state["turn_step"])),
                    )
                )
    return candidates


def donor_candidate(
    frozen_row: dict[str, Any], state: dict[str, Any]
) -> NegativeCandidate | None:
    evidence, count = build_remaining_evidence(state)
    if not state.get("pi_available") or count == 0:
        return None
    return NegativeCandidate(
        candidate_id=f"donor:{state['turn_uid']}",
        text=evidence,
        entry_count=count,
        priority=3,
        donor_question_uid=state["question_uid"],
        donor_turn_uid=state["turn_uid"],
        source="matched_other_question",
        data_source=str(state["data_source"]),
        turn_bucket="3+" if int(state["turn_step"]) >= 3 else str(int(state["turn_step"])),
    )


def choose_matched_negative(
    *,
    frozen_row: dict[str, Any],
    state: dict[str, Any],
    local_candidates: list[NegativeCandidate],
    donor_candidates: list[NegativeCandidate],
    tokenizer,
    seed: int,
    token_length_cache: dict[str, int] | None = None,
) -> NegativeCandidate | None:
    aliases = [str(value) for value in frozen_row.get("ground_truth_aliases", [])]
    positive, target_entries = build_remaining_evidence(state)
    target_tokens = len(tokenizer.encode(positive, add_special_tokens=False))
    turn_bucket = "3+" if int(state["turn_step"]) >= 3 else str(int(state["turn_step"]))

    eligible = list(local_candidates)
    for candidate in donor_candidates:
        if candidate.donor_question_uid == state["question_uid"]:
            continue
        if not _safe_negative(candidate.text, state, aliases):
            continue
        eligible.append(candidate)
    if not eligible:
        return None

    token_length_cache = token_length_cache if token_length_cache is not None else {}

    def rank(candidate: NegativeCandidate) -> tuple[Any, ...]:
        if candidate.candidate_id not in token_length_cache:
            token_length_cache[candidate.candidate_id] = len(
                tokenizer.encode(candidate.text, add_special_tokens=False)
            )
        token_length = token_length_cache[candidate.candidate_id]
        stable = hashlib.sha256(
            f"{seed}|{state['turn_uid']}|{turn_bucket}|{candidate.candidate_id}".encode()
        ).hexdigest()
        return (
            candidate.priority,
            candidate.data_source != str(state["data_source"]),
            candidate.turn_bucket != turn_bucket,
            abs(candidate.entry_count - target_entries),
            abs(token_length - target_tokens),
            stable,
        )

    return min(eligible, key=rank)


def parse_think_chunks(
    tokenizer,
    response_token_ids: list[int],
    think_mask: list[int],
    *,
    max_chunk_tokens: int = 64,
    min_chunk_tokens: int = 8,
    max_chunks: int = 6,
) -> list[dict[str, Any]]:
    positions = [index for index, selected in enumerate(think_mask) if selected]
    if not positions:
        return []
    pieces = tokenizer.batch_decode(
        [[int(response_token_ids[position])] for position in positions],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    chunks: list[list[int]] = []
    current: list[int] = []
    boundary = re.compile(r"[.!?;\n。！？；]")
    for position, piece in zip(positions, pieces):
        current.append(position)
        if boundary.search(piece):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    split_chunks: list[list[int]] = []
    for chunk in chunks:
        split_chunks.extend(
            chunk[start : start + max_chunk_tokens]
            for start in range(0, len(chunk), max_chunk_tokens)
        )
    chunks = split_chunks
    index = 0
    while len(chunks) > 1 and index < len(chunks):
        if len(chunks[index]) >= min_chunk_tokens:
            index += 1
            continue
        if index == 0:
            chunks[1] = chunks[0] + chunks[1]
            chunks.pop(0)
        else:
            chunks[index - 1].extend(chunks[index])
            chunks.pop(index)
            index -= 1
    while len(chunks) > max_chunks:
        pair = min(
            range(len(chunks) - 1),
            key=lambda left: (len(chunks[left]) + len(chunks[left + 1]), left),
        )
        chunks[pair] = chunks[pair] + chunks[pair + 1]
        chunks.pop(pair + 1)

    output = []
    for chunk_index, chunk_positions in enumerate(chunks):
        token_ids = [int(response_token_ids[position]) for position in chunk_positions]
        output.append(
            {
                "chunk_index": int(chunk_index),
                "token_positions": chunk_positions,
                "token_start": int(chunk_positions[0]),
                "token_end": int(chunk_positions[-1] + 1),
                "token_ids": token_ids,
                "chunk_text": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "chunk_token_count": len(token_ids),
                "post_cap_over_64": len(token_ids) > max_chunk_tokens,
            }
        )
    return output


def chunk_audit_fields(
    chunk_text: str,
    *,
    query_text: str,
    aliases: Iterable[Any],
    supporting_evidence_text: str,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    copied_length, copied_text = longest_common_word_substring(
        chunk_text, supporting_evidence_text
    )
    return {
        "chunk_position": chunk_index / max(1, chunk_count - 1),
        "query_lexical_overlap": lexical_jaccard(chunk_text, query_text),
        "answer_alias_overlap": normalized_alias_hit(chunk_text, aliases),
        "supporting_evidence_overlap": lexical_jaccard(
            chunk_text, supporting_evidence_text
        ),
        "longest_copied_substring_tokens": int(copied_length),
        "longest_copied_substring": copied_text,
    }
