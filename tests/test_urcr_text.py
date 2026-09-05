from verl.trainer.ppo.urcr_text import (
    align_supporting_evidence,
    longest_common_word_substring,
    parse_retrieved_documents,
)


def _metadata():
    return {
        "context": {
            "title": ["Bridge Article", "Answer Article"],
            "sentences": [
                ["The bridge connects Alpha to Beta."],
                ["The final city is Paris."],
            ],
        },
        "supporting_facts": {
            "title": ["Bridge Article", "Answer Article"],
            "sent_id": [0, 0],
        },
    }


def test_document_and_support_alignment_stays_exact():
    observation = (
        '<documents>Doc 1: "Bridge Article"\n'
        "The bridge connects Alpha to Beta.\n"
        "Doc 2: Distractor\nThe final city is Paris.\n</documents>"
    )
    documents = parse_retrieved_documents(observation)
    aligned = align_supporting_evidence(observation, _metadata(), ["Paris"])

    assert [document.title for document in documents] == [
        '"Bridge Article"',
        "Distractor",
    ]
    assert aligned["supporting_doc_hit"]
    assert aligned["supporting_fact_hit"]
    assert aligned["bridge_only_supporting_hit"]
    assert not aligned["answer_bearing_supporting_hit"]
    assert aligned["observation_alias_hit_normalized"]
    assert aligned["hit_supporting_fact_ids"] == ["bridge article#0"]


def test_longest_common_word_substring_is_contiguous():
    length, text = longest_common_word_substring(
        "first find the bridge article then inspect it",
        "bridge article then answer",
    )

    assert length == 3
    assert text == "bridge article then"
