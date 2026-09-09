from __future__ import annotations

from dataclasses import fields, replace

from verl.trainer.ppo.urcr_v4_data import V4TurnRecord
from verl.trainer.ppo.urcr_v4_method import (
    ControlSelection,
    ObservationCandidate,
    forbidden_online_metadata_paths,
)
from verl.trainer.ppo.urcr_v4_requests import (
    EMPTY_OBSERVATION_SLOT,
    build_responsibility_requests,
    build_search_utility_requests,
    probe_template_manifest,
    should_score_responsibility,
)


class CharTokenizer:
    eos_token = "<E>"
    eos_token_id = 250
    pad_token_id = 0

    def _tokens(self, text):
        text = str(text)
        output = []
        index = 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                output.append((self.eos_token_id, index, index + len(self.eos_token)))
                index += len(self.eos_token)
            else:
                output.append((ord(text[index]) + 1, index, index + 1))
                index += 1
        return output

    def encode(self, text, **_kwargs):
        return [item[0] for item in self._tokens(text)]

    def __call__(self, text, **_kwargs):
        values = self._tokens(text)
        return {
            "input_ids": [item[0] for item in values],
            "offset_mapping": [(item[1], item[2]) for item in values],
        }

    def decode(self, ids):
        output = []
        for value in ids:
            if int(value) == self.eos_token_id:
                output.append(self.eos_token)
            elif int(value) != self.pad_token_id:
                output.append(chr(int(value) - 1))
        return "".join(output)


def _record(*, action_type: str = "search") -> V4TurnRecord:
    tokenizer = CharTokenizer()
    if action_type == "search":
        response_text = "<think>why</think><search>term</search>"
        action_text = "term"
    else:
        response_text = "<think>why</think><answer>Paris</answer>"
        action_text = "Paris"
    response = tuple(tokenizer.encode(response_text))
    think_start = response_text.index("why")
    think_positions = tuple(range(think_start, think_start + 3))
    action_start = response_text.index(action_text, response_text.index(f"<{action_type}>"))
    action_positions = tuple(range(action_start, action_start + len(action_text)))
    return V4TurnRecord(
        global_step=1,
        batch_row_index=0,
        is_adjustment_copy=False,
        question_id="nq:train:row:1",
        rollout_group_id="group",
        trajectory_id="trajectory",
        turn_id="turn",
        rollout_index=0,
        turn_index=0,
        data_source="nq",
        question="Where?",
        answer_aliases=("Paris", "City of Paris"),
        visible_context_text="visible H only",
        visible_prompt_token_ids=tuple(tokenizer.encode("PROMPT")),
        sampled_response_token_ids=response,
        action_type=action_type,
        action_text=action_text,
        observation_text="<documents>Doc 1: Real\nUseful</documents>",
        valid_action=True,
        observation_consumable=action_type == "search",
        terminal_binary_em=1,
        think_content_positions=think_positions,
        action_target_positions=action_positions,
        action_content_positions=action_positions,
        action_boundary_positions=(),
        action_tag_positions=(),
        think_chunks=(think_positions[:1], think_positions[1:]),
        old_action_target_log_probs=tuple(
            -float(index + 1) for index in range(len(action_text))
        ),
    )


def _control(identity: str, body: str) -> ObservationCandidate:
    return ObservationCandidate(
        observation_id=identity,
        question_id=identity,
        data_source="nq",
        call_bucket="1",
        serialized_observation=f"<documents>Doc 1: {identity}\n{body}</documents>",
        token_length=20,
        document_count=1,
        document_signature=(identity,),
    )


def test_search_request_queue_has_five_views_per_fixed_alias_without_outcome() -> None:
    tokenizer = CharTokenizer()
    controls = ControlSelection(
        controls=(
            _control("c1", "one"),
            _control("c2", "two"),
            _control("c3", "three"),
        ),
        unique_control_count=3,
        used_replacement=False,
        selected_tiers=(0, 0, 0),
        failure_reason=None,
    )
    bundle = build_search_utility_requests(
        _record(),
        controls,
        tokenizer=tokenizer,
        actor_snapshot_id="step-1-pre-update",
    )
    assert bundle.probe_aliases == ("Paris", "City of Paris")
    assert len(bundle.items) == 5 * len(bundle.probe_aliases)
    assert bundle.control_unique_count == 3
    assert not bundle.control_used_replacement
    assert {item.comparison_group_id for item in bundle.items} == {bundle.turn_id}
    assert {item.view for item in bundle.items} == {
        "real",
        "empty",
        "control_0",
        "control_1",
        "control_2",
    }
    for alias_index in range(len(bundle.probe_aliases)):
        targets = {
            item.target_ids
            for item in bundle.items
            if item.request_id.endswith(f"alias:{alias_index}")
        }
        assert len(targets) == 1
    empty = next(item for item in bundle.items if item.view == "empty")
    assert EMPTY_OBSERVATION_SLOT in tokenizer.decode(empty.input_ids)
    assert "terminal_binary_em" not in {field.name for field in fields(bundle)}
    assert not forbidden_online_metadata_paths(bundle.__dict__)


def test_responsibility_reuses_old_full_score_and_preserves_sampled_target() -> None:
    record = _record(action_type="answer")
    bundle = build_responsibility_requests(
        record,
        actor_snapshot_id="step-1-pre-update",
    )
    assert bundle.action_type == "answer"
    assert bundle.full_score_from_old_log_probs == -3.0
    expected = tuple(
        record.sampled_response_token_ids[position]
        for position in record.action_target_positions
    )
    assert bundle.full_target_ids == expected
    assert bundle.plain_item.target_ids == expected
    assert not bundle.plain_item.blocked_key_positions
    assert bundle.whole_mask_item.target_ids == expected
    assert {
        bundle.plain_item.comparison_group_id,
        bundle.whole_mask_item.comparison_group_id,
        *(item.comparison_group_id for item in bundle.chunk_mask_items),
    } == {record.turn_id}
    assert len(bundle.chunk_mask_items) == 2
    prompt_length = len(record.visible_prompt_token_ids)
    assert bundle.whole_mask_item.blocked_key_positions == tuple(
        prompt_length + value for value in record.think_content_positions
    )
    assert all(
        item.target_ids == expected for item in bundle.chunk_mask_items
    )


def test_responsibility_selection_keeps_negative_utility_and_does_not_gate_action() -> None:
    record = _record(action_type="answer")
    assert should_score_responsibility(record, action_utility=0.2)
    assert should_score_responsibility(record, action_utility=-0.2)
    assert not should_score_responsibility(record, action_utility=0.0)
    no_think = replace(
        record,
        think_content_positions=(),
        think_chunks=(),
    )
    assert not should_score_responsibility(no_think, action_utility=-0.2)
    boundary_only = replace(
        record,
        action_content_positions=(),
        action_boundary_positions=record.action_target_positions,
        action_tag_positions=record.action_target_positions,
    )
    assert should_score_responsibility(boundary_only, action_utility=-0.2)
    assert build_responsibility_requests(
        boundary_only,
        actor_snapshot_id="step-1-pre-update",
    ).full_target_ids


def test_probe_template_manifest_records_neutral_native_boundaries() -> None:
    manifest = probe_template_manifest(CharTokenizer())
    assert manifest["empty_observation_slot"] == "<documents></documents>"
    assert "enough" not in str(manifest).lower()
    assert manifest["score_tokens"] == "answer_content_only"
    assert manifest["first_target_predictor"] == "target_position_minus_one"
