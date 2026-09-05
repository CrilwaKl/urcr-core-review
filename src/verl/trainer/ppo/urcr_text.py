"""Deterministic text and evidence-alignment helpers for URCR."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Iterable


_DOC_HEADER_RE = re.compile(r"(?m)^Doc\s+(\d+):\s*")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

@dataclass(frozen=True)
class RetrievedDocument:
    rank: int
    title: str
    body: str


@dataclass(frozen=True)
class SupportingFact:
    fact_id: str
    title: str
    sent_id: int
    sentence: str


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_match_text(text: Any) -> str:
    """Normalize for audited exact title/sentence matching, without fuzzy search."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    characters = [character if character.isalnum() else " " for character in normalized]
    return " ".join("".join(characters).split())


def word_tokens(text: Any) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return [token for token in _WORD_RE.findall(normalized) if token]


def extract_supporting_facts(metadata: Any) -> list[SupportingFact]:
    metadata = parse_json_object(metadata)
    supporting = metadata.get("supporting_facts")
    context = metadata.get("context")
    if not isinstance(supporting, dict) or not isinstance(context, dict):
        return []

    supporting_titles = [str(value) for value in _plain_list(supporting.get("title"))]
    supporting_ids = _plain_list(supporting.get("sent_id"))
    context_titles = [str(value) for value in _plain_list(context.get("title"))]
    context_sentences = _plain_list(context.get("sentences"))
    title_to_index = {title: index for index, title in enumerate(context_titles)}

    facts: list[SupportingFact] = []
    for title, raw_sent_id in zip(supporting_titles, supporting_ids):
        try:
            sent_id = int(raw_sent_id)
        except (TypeError, ValueError):
            continue
        sentence = ""
        context_index = title_to_index.get(title)
        if context_index is not None and context_index < len(context_sentences):
            sentences = _plain_list(context_sentences[context_index])
            if 0 <= sent_id < len(sentences):
                sentence = str(sentences[sent_id])
        normalized_title = normalize_match_text(title)
        fact_id = f"{normalized_title}#{sent_id}"
        facts.append(
            SupportingFact(
                fact_id=fact_id,
                title=title,
                sent_id=sent_id,
                sentence=sentence,
            )
        )
    return facts


def parse_retrieved_documents(observation: Any) -> list[RetrievedDocument]:
    text = str(observation).strip()
    text = re.sub(r"^<documents>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</documents>$", "", text, flags=re.IGNORECASE)
    matches = list(_DOC_HEADER_RE.finditer(text))
    documents: list[RetrievedDocument] = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : stop].strip()
        if not block:
            continue
        title, separator, body = block.partition("\n")
        documents.append(
            RetrievedDocument(
                rank=int(match.group(1)),
                title=title.strip(),
                body=body.strip() if separator else "",
            )
        )
    return documents


def normalized_alias_hit(text: Any, aliases: Iterable[Any]) -> bool:
    normalized_text = normalize_match_text(text)
    return any(
        normalized_alias and normalized_alias in normalized_text
        for normalized_alias in (normalize_match_text(alias) for alias in aliases)
    )


def exact_alias_spans(text: str, aliases: Iterable[Any]) -> list[dict[str, Any]]:
    """Find non-overlapping case-insensitive literal alias spans.

    Longer aliases win when aliases overlap. Positions always refer to the
    original observation string.
    """
    candidates: list[tuple[int, int, str]] = []
    unique_aliases = sorted(
        {str(alias) for alias in aliases if str(alias)},
        key=lambda value: (-len(value), value.casefold()),
    )
    for alias in unique_aliases:
        for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
            candidates.append((match.start(), match.end(), alias))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].casefold()))

    selected: list[tuple[int, int, str]] = []
    for start, end, alias in candidates:
        if any(start < kept_end and end > kept_start for kept_start, kept_end, _ in selected):
            continue
        selected.append((start, end, alias))
    selected.sort()
    return [
        {
            "alias": alias,
            "start": start,
            "end": end,
            "matched_text": text[start:end],
        }
        for start, end, alias in selected
    ]


def align_supporting_evidence(
    observation: Any,
    metadata: Any,
    aliases: Iterable[Any],
) -> dict[str, Any]:
    """Align retrieved documents to HotpotQA support by audited exact rules."""
    aliases = [str(alias) for alias in aliases]
    facts = extract_supporting_facts(metadata)
    documents = parse_retrieved_documents(observation)
    gold_titles = {normalize_match_text(fact.title): fact.title for fact in facts}
    retrieved_by_title: dict[str, list[RetrievedDocument]] = {}
    for document in documents:
        retrieved_by_title.setdefault(normalize_match_text(document.title), []).append(document)

    hit_doc_ids = sorted(set(gold_titles) & set(retrieved_by_title))
    hit_fact_ids: list[str] = []
    bridge_fact_ids: list[str] = []
    answer_fact_ids: list[str] = []
    for fact in facts:
        normalized_title = normalize_match_text(fact.title)
        normalized_sentence = normalize_match_text(fact.sentence)
        if not normalized_sentence or normalized_title not in retrieved_by_title:
            continue
        matched = any(
            normalized_sentence in normalize_match_text(document.body)
            for document in retrieved_by_title[normalized_title]
        )
        if not matched:
            continue
        hit_fact_ids.append(fact.fact_id)
        if normalized_alias_hit(fact.sentence, aliases):
            answer_fact_ids.append(fact.fact_id)
        else:
            bridge_fact_ids.append(fact.fact_id)

    supporting_doc_text = "\n".join(
        f"{document.title}\n{document.body}"
        for title in hit_doc_ids
        for document in retrieved_by_title[title]
    )
    return {
        "retrieved_document_count": len(documents),
        "retrieved_titles": [document.title for document in documents],
        "gold_supporting_doc_count": len(gold_titles),
        "gold_supporting_fact_count": len(facts),
        "hit_supporting_doc_ids": hit_doc_ids,
        "hit_supporting_fact_ids": sorted(set(hit_fact_ids)),
        "hit_bridge_fact_ids": sorted(set(bridge_fact_ids)),
        "hit_answer_bearing_fact_ids": sorted(set(answer_fact_ids)),
        "supporting_doc_hit": bool(hit_doc_ids),
        "supporting_fact_hit": bool(hit_fact_ids),
        "bridge_only_supporting_hit": bool(bridge_fact_ids),
        "answer_bearing_supporting_hit": bool(answer_fact_ids),
        "supporting_with_direct_answer_alias": bool(hit_doc_ids)
        and normalized_alias_hit(supporting_doc_text, aliases),
        "observation_alias_hit_normalized": normalized_alias_hit(observation, aliases),
        "non_supporting_retrieval": not bool(hit_doc_ids or hit_fact_ids),
        "match_method": "normalized_title_exact+normalized_supporting_sentence_substring",
    }


def longest_common_word_substring(left: Any, right: Any) -> tuple[int, str]:
    left_tokens = word_tokens(left)
    right_tokens = word_tokens(right)
    if not left_tokens or not right_tokens:
        return 0, ""
    previous = [0] * (len(right_tokens) + 1)
    best_length = 0
    best_end = 0
    for left_index, left_token in enumerate(left_tokens, start=1):
        current = [0] * (len(right_tokens) + 1)
        for right_index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current[right_index] = previous[right_index - 1] + 1
                if current[right_index] > best_length:
                    best_length = current[right_index]
                    best_end = left_index
        previous = current
    return best_length, " ".join(left_tokens[best_end - best_length : best_end])


def lexical_jaccard(left: Any, right: Any) -> float:
    left_tokens = set(word_tokens(left))
    right_tokens = set(word_tokens(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0
