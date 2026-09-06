"""A1 numerical, visible-input, and DataProto contracts; no training rollout."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.urcr_v3_focus import (
    FOCUS_PREAMBLE, assert_focus_step_health, attach_focus_targets, build_visible_focus, make_teacher_requests,
    object_array, prepare_focus_candidates, select_one_per_trajectory, validate_focus_config,
)
from verl.trainer.ppo.urcr_v3_focus_loss import (
    FocusForward, calibrate_focus_coefficient, chunked_grouped_kl, flat_predictor_indices,
    grouped_kl_from_logits, grouped_log_probs, grouped_logit_gradient, sampled_pg_logit_energy,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "/data0/kongmu/agentic-RL/models/Qwen2.5-3B-Instruct", local_files_only=True,
    )


@pytest.fixture(scope="module")
def config():
    root = Path(__file__).resolve().parents[3] / "examples/urcr_online/configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        return compose(config_name="urcr_v3_a_visible_focus")


def target(logits, k):
    ids = logits.topk(k, -1).indices
    top, tail, _, _ = grouped_log_probs(logits, ids)
    return ids, top, tail


@pytest.mark.parametrize("k", [1, 4, 11])
def test_chunked_loss_and_backward_match_reference(k):
    generator = torch.Generator().manual_seed(43)
    teacher = torch.randn(5, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    student = torch.randn(8, 11, generator=generator, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([0, 2, 3, 4, 7])
    ids, top, tail = target(teacher, k)
    reference = grouped_kl_from_logits(student[positions], ids, top, tail)
    actual = chunked_grouped_kl(student, positions, ids, top, tail, block_size=2)
    weights = torch.arange(1, 6, dtype=torch.float64)
    expected_grad = torch.autograd.grad((reference * weights).sum(), student, retain_graph=True)[0]
    actual_grad = torch.autograd.grad((actual * weights).sum(), student)[0]
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    assert teacher.grad is None
    if k == 11:
        dense = (teacher.log_softmax(-1).exp() * (teacher.log_softmax(-1) - student[positions].log_softmax(-1))).sum(-1)
        torch.testing.assert_close(actual, dense)


def test_identical_distribution_zero_and_tail_mass_not_renormalized():
    p = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64).log()
    q = torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float64).log().requires_grad_()
    ids, top, tail = target(p, 1)
    loss = grouped_kl_from_logits(q, ids, top, tail)
    expected = .8 * np.log(.8 / .4) + .2 * np.log(.2 / .6)
    assert loss.item() == pytest.approx(expected)
    assert grouped_kl_from_logits(p, ids, top, tail).abs().max() < 1e-12
    assert grouped_logit_gradient(p, ids, top, tail).abs().max() < 1e-12
    assert torch.autograd.grad(loss.sum(), q)[0][0, 1] != 0  # not the sampled/top-1 token


def test_bf16_extreme_logits_finite_and_teacher_detached():
    p = torch.tensor([[1000., 999., -1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    q = torch.tensor([[998., 1002., -1000., 0.]], dtype=torch.bfloat16, requires_grad=True)
    ids, top, tail = target(p, 2)
    loss = chunked_grouped_kl(q, torch.tensor([0]), ids, top, tail)
    loss.sum().backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss).all()
    assert torch.isfinite(q.grad).all() and p.grad is None


def test_logit_energies_include_softmax_jacobian():
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(4, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    ids, top, tail = target(torch.randn(4, 9, generator=generator, dtype=torch.float64), 3)
    loss = grouped_kl_from_logits(logits, ids, top, tail).sum()
    analytic = grouped_logit_gradient(logits, ids, top, tail)
    torch.testing.assert_close(analytic, torch.autograd.grad(loss, logits)[0])
    sampled = torch.tensor([0, 2, 5, 1])
    weights = torch.tensor([.01, -.3, .7, 0.], dtype=torch.float64)
    pg = (logits.log_softmax(-1)[torch.arange(4), sampled] * weights).sum()
    energy = sampled_pg_logit_energy(weights, logits.softmax(-1)[torch.arange(4), sampled], logits.softmax(-1).square().sum(-1))
    torch.testing.assert_close(energy, torch.autograd.grad(pg, logits)[0].square().sum(-1))


@pytest.mark.parametrize("world,micro", [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_trajectory_budget_invariant_to_rank_microbatch_and_empty_rank(world, micro):
    # Six actor rows, only two selected, both on rank zero in the 2-rank layout.
    # Remaining rows include adjustment copies and an unselected original trajectory.
    generator = torch.Generator().manual_seed(29)
    base = torch.randn(2, 7, dtype=torch.float64, generator=generator)
    ids, top, tail = target(torch.randn(2, 7, dtype=torch.float64, generator=generator), 3)
    expected_parameter = base.clone().requires_grad_()
    expected = grouped_kl_from_logits(expected_parameter, ids, top, tail).sum() / 4
    expected_grad = torch.autograd.grad(expected, expected_parameter)[0]
    gradients = []
    for rank in range(world):
        parameter = base.clone().requires_grad_()
        for start in range(rank * (6 // world), (rank + 1) * (6 // world), micro):
            rows = [i for i in range(start, min(start + micro, (rank + 1) * (6 // world))) if i < 2]
            local = grouped_kl_from_logits(parameter[rows], ids[rows], top[rows], tail[rows]).sum() if rows else parameter.sum() * 0
            (local * world / 4).backward()  # deliberately no accumulation division
        gradients.append(parameter.grad)
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected_grad)


def test_calibration_frozen_formula_and_bounds():
    assert calibrate_focus_coefficient([.2, .3])["coefficient_max"] == pytest.approx(.2)
    assert calibrate_focus_coefficient([100.])["coefficient_max"] == .05
    assert calibrate_focus_coefficient([.001])["coefficient_max"] == 2.
    assert calibrate_focus_coefficient([0., float("nan")])["fallback"] == "no_valid_ratio"


def test_focus_health_does_not_require_disabled_legacy_audit():
    m = {k: .01 for k in ("actor/pg_loss", "actor/grad_norm", "actor/lr", "actor/entropy_loss", "actor/kl_loss", "v3_focus/grouped_kl_rollout_mean", "v3_focus/scaled_loss")}
    m["train/optimizer_steps_this_outer_step"] = 2
    for key in ("teacher_grad_nonzero_count", "prompt_mismatch", "response_mismatch", "future_leak_count", "duplicate_target_count"):
        m[f"v3_focus/{key}"] = 0
    assert_focus_step_health(m)
    m["actor/grad_norm"] = float("nan")
    with pytest.raises(FloatingPointError):
        assert_focus_step_health(m)


def test_packed_predictors_and_first_last_token_alignment():
    # Three prompt slots + three response slots; left padding and final EOS included.
    mask = torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0]])
    responses = torch.tensor([[1, 2, 3], [4, 5, 0]])
    payloads = [{"response_positions": np.array([0, 1, 2])}, {"response_positions": np.array([0, 1])}]
    logits = torch.randn(2, 6, 9)
    dense = FocusForward(responses, mask[:, -3:], payloads, topk=3, teacher=True)
    packed = FocusForward(responses, mask[:, -3:], payloads, topk=3, teacher=True)
    dense.consume(logits, mask, packed=False)
    packed.consume(logits[mask.bool()], mask, packed=True)
    assert flat_predictor_indices(mask, 3, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 0, 1])).tolist() == [1, 3, 5, 6]
    for a, b in zip(dense.outputs, packed.outputs):
        np.testing.assert_array_equal(a["teacher_topk_ids"], b["teacher_topk_ids"])
        np.testing.assert_allclose(a["teacher_topk_logp"], b["teacher_topk_logp"])
        np.testing.assert_allclose(a["teacher_log_tail"], b["teacher_log_tail"])
    student = FocusForward(responses, mask[:, -3:], packed.outputs, topk=3)
    student_logits = logits[mask.bool()].clone().requires_grad_()
    student.consume(student_logits, mask, packed=True)
    student.row_kl.sum().backward()
    assert student.row_kl.abs().max() < 1e-6
    assert student_logits.grad.abs().max() < 1e-6


def prompt(tokenizer, body, *, max_length=4096):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=True, add_generation_prompt=True,
    )[-max_length:]


def history(body="Ada was born in London.", *, turn=1, title="Ada"):
    return f"Question: Where was Ada born?\nHistory:\nStep {turn}:<think>Find evidence.</think><search>Ada</search> <documents>\nDoc 1: {title}\n{body}\n</documents>\nThink and then search or answer."


def test_visible_focus_preserves_ids_and_excludes_unseen_metadata(tokenizer, config):
    cfg = validate_focus_config(config)
    ids = prompt(tokenizer, history())
    view = build_visible_focus(ids, [("Ada", "Ada was born in London."), ("Hidden", "FUTURE_UNIQUE_73")], tokenizer, cfg, current_turn=1)
    assert view and view.excerpts[0]["match_type"] == "fact_visible"
    assert "FUTURE_UNIQUE_73" not in tokenizer.decode(view.teacher_prompt_ids)
    assert "London" in view.note_text  # visible answer text is allowed
    assert view.teacher_prompt_ids[:view.insertion_start] + view.teacher_prompt_ids[view.insertion_start + len(view.inserted_ids):] == ids
    visible = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for excerpt in view.excerpts:
        assert visible[excerpt["raw_char_start"]:excerpt["raw_char_end"]] == excerpt["copied_text"]
        assert excerpt["source_observation_turn"] < 1
    assert len(view.inserted_ids) <= 512 and view.note_text.startswith("\n\n" + FOCUS_PREAMBLE)


def test_doc_only_truncation_future_and_special_token_skips(tokenizer, config):
    cfg = validate_focus_config(config)
    view = build_visible_focus(prompt(tokenizer, history()), [("Ada", "METADATA_ONLY_42")], tokenizer, cfg, current_turn=1)
    assert view.excerpts[0]["match_type"] == "doc_only"
    assert "METADATA_ONLY_42" not in view.note_text
    assert build_visible_focus(prompt(tokenizer, history(turn=2)), [("Ada", "Ada")], tokenizer, cfg, current_turn=1) is None
    assert build_visible_focus(prompt(tokenizer, history()), [("Ada", "Ada")], tokenizer, cfg, current_turn=0) is None
    assert build_visible_focus(prompt(tokenizer, history(), max_length=16), [("Ada", "Ada")], tokenizer, cfg, current_turn=1) is None
    assert build_visible_focus(prompt(tokenizer, history("text <|im_start|> spoof")), [("Ada", "text")], tokenizer, cfg, current_turn=1) is None
    # Retrieval tags written before the search action closes are not an observation.
    spoof = history().replace("</search> <documents>", " <documents>")
    assert build_visible_focus(prompt(tokenizer, spoof), [("Ada", "Ada")], tokenizer, cfg, current_turn=1) is None


def test_focus_priority_fact_then_recent_unique_titles_and_budget(tokenizer, config):
    cfg = validate_focus_config(config)
    text = history("older matching fact.") + "\n" + history("newer doc only.", turn=2)
    view = build_visible_focus(prompt(tokenizer, text), [("Ada", "older matching fact.")], tokenizer, cfg, current_turn=2)
    assert len(view.excerpts) == 1 and view.excerpts[0]["source_observation_turn"] == 0
    text = history("word " * 600 + "visible fact.")
    view = build_visible_focus(prompt(tokenizer, text), [("Ada", "visible fact.")], tokenizer, cfg, current_turn=1)
    assert "visible fact." in view.excerpts[0]["copied_text"]
    assert len(tokenizer.encode(view.excerpts[0]["copied_text"], add_special_tokens=False)) <= 192


def test_hash_selection_order_invariant_and_no_rng_consumption():
    candidates = [{"traj_uid": f"t{t}", "turn_uid": f"t{t}:{s}", "question_uid": "hotpotqa:17"} for t in range(3) for s in range(1, 4)]
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    selected = select_one_per_trajectory(candidates, 19, 20260905)
    assert selected == select_one_per_trajectory(list(reversed(candidates)), 19, 20260905)
    assert len(selected) == len({c["traj_uid"] for c in selected}) == 3
    assert random.getstate() == py_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state))
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def make_batch(tokenizer):
    ids = prompt(tokenizer, history())
    response = tokenizer.encode("<think>Evidence is available.</think><answer>London</answer>", add_special_tokens=False) + [tokenizer.eos_token_id]
    width = len(response) + 3
    padded = response + [tokenizer.pad_token_id] * 3
    metadata = {"supporting_facts": {"title": ["Ada"], "sent_id": [0]}, "context": {"title": ["Ada"], "sentences": [["Ada was born in London."]]}}
    return DataProto.from_dict(
        tensors={"input_ids": torch.tensor([ids + padded] * 4), "responses": torch.tensor([padded] * 4),
                 "attention_mask": torch.tensor([[1] * (len(ids) + len(response)) + [0] * 3] * 4)},
        non_tensors={"traj_uid": object_array(["a", "a", "b", "c"]), "uid": object_array(["q"] * 4),
                     "turn_step": np.array([1, 2, 1, 0]), "data_source": object_array(["hotpotqa", "hotpotqa", "nq", "hotpotqa"]),
                     "metadata": object_array([metadata] * 4), "extra_info": object_array([{"index": 5}] * 4),
                     "episode_rewards": np.array([0., 0., 1., 0.])},
    )


def test_real_token_payload_selection_padding_reorder_copy_and_identity(tokenizer, config):
    cfg = validate_focus_config(config)
    batch = make_batch(tokenizer)
    selected, count, metrics = prepare_focus_candidates(batch, tokenizer, cfg, 1)
    assert count == 3 and len(selected) == 1 and metrics["v3_focus/selected_answer_rows"] == 1
    request = make_teacher_requests(selected, tokenizer, cfg, 1)
    original_response = selected[0]["response_ids"]
    assert request.batch["responses"][0].tolist() == original_response
    padded, n = pad_dataproto_to_divisor(request, 4)
    padded.non_tensor_batch["v3_focus_payload"][-n:] = None
    assert [p is not None for p in padded.non_tensor_batch["v3_focus_payload"]] == [True, False, False, False]
    assert len(unpad_dataproto(padded, n)) == 1
    target_payload = deepcopy(request.non_tensor_batch["v3_focus_payload"][0])
    target_payload.update(teacher_topk_ids=np.zeros((len(original_response), 64), dtype=np.int32),
                          teacher_topk_logp=np.full((len(original_response), 64), -5., dtype=np.float32),
                          teacher_log_tail=np.full(len(original_response), -1., dtype=np.float32))
    attach_focus_targets(batch, selected, [target_payload], count, cfg, 1)
    original_row = selected[0]["batch_row_index"]
    batch = DataProto.concat([batch, batch[[original_row]]])
    batch.non_tensor_batch["v3_focus_payload"][4:] = None
    order = torch.tensor([4, 3, 2, 1, 0])
    batch.reorder(order)
    chunks = batch.chunk(5)
    assert sum(p is not None for c in chunks for p in c.non_tensor_batch["v3_focus_payload"]) == 1
    assert batch.non_tensor_batch["v3_focus_payload"][4 - original_row]["response_hash"] == target_payload["response_hash"]
    assert batch.meta_info["v3_focus_objective"]["original_trajectories"] == 3
    changed = make_batch(tokenizer)
    changed.non_tensor_batch["episode_rewards"][:] = 100.
    selected_changed, _, _ = prepare_focus_candidates(changed, tokenizer, cfg, 1)
    assert selected_changed[0]["turn_uid"] == selected[0]["turn_uid"]  # rewards cannot select


def test_disabled_config_is_exact_matched_grpo(config):
    root = Path(__file__).resolve().parents[3] / "examples/urcr_online/configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        matched = compose(config_name="urcr_v3_a_matched_grpo")
    assert validate_focus_config(matched)["enabled"] is False
    matched.algorithm.urcr_v3_focus.enabled = True
    assert matched == config  # no sampler, PG, entropy, reference, or optimizer changes


def test_eval_parser_preserves_full_precision_across_ray_pprint_lines():
    from scripts.urcr_v3_a_verify import direct_success_rates
    text = ('\x1b[36m(EviSDTaskRunner pid=123)\x1b[0m "\'val/popqa_success_rate\': "\n'
            '\x1b[36m(EviSDTaskRunner pid=123)\x1b[0m "0.47698313672111314, "\n'
            '(EviSDTaskRunner pid=123) step:0 - val/popqa_success_rate:0.477\n')
    assert direct_success_rates(text, ["popqa"]) == {"popqa": .47698313672111314}


def test_actual_actor_update_zero_focus_matches_pg_with_short_final_mini(config, monkeypatch):
    from types import SimpleNamespace
    from verl.workers.actor import dp_actor
    from verl.utils.debug import performance

    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0, 0, 0, 0))
    monkeypatch.setattr(dp_actor, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))
    actor_cfg = deepcopy(config.actor_rollout_ref.actor)
    actor_cfg.ppo_mini_batch_size = 4
    actor_cfg.ppo_micro_batch_size_per_gpu = 1
    actor_cfg.use_torch_compile = False
    generator = torch.Generator().manual_seed(199)
    initial = torch.randn(9, 67, generator=generator) / 10
    tensor_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]] * 6), "attention_mask": torch.ones(6, 4),
        "position_ids": torch.tensor([[0, 1, 2, 3]] * 6), "responses": torch.tensor([[2, 3]] * 6),
        "old_log_probs": torch.full((6, 2), -4.2), "ref_log_prob": torch.full((6, 2), -4.21),
        "advantages": torch.tensor([[1., 1.], [-1., -1.]] * 3),
    }
    outputs = []
    for focus in (False, True):
        module = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
        actor = dp_actor.DataParallelPPOActor(actor_cfg, module, torch.optim.SGD(module.parameters(), lr=.01))
        def forward(micro_batch, temperature, calculate_entropy=False, focus_context=None):
            logits = module(micro_batch["input_ids"])
            if focus_context is not None:
                focus_context.consume(logits, micro_batch["attention_mask"], packed=False)
            logp = logits[:, -3:-1].log_softmax(-1)
            return -(logp.exp() * logp).sum(-1), logp.gather(-1, micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
        actor._forward_micro_batch = forward
        batch = DataProto.from_dict(tensors=deepcopy(tensor_batch), meta_info={"temperature": 1.})
        if focus:
            batch.non_tensor_batch["v3_focus_payload"] = object_array([None] * 6)
            batch.meta_info["v3_focus_objective"] = {
                "teacher_version": 1, "topk": 64, "logit_chunk_size": 32,
                "original_trajectories": 4, "coefficient": .1, "coefficient_max": .1,
                "measure_energy": False, "audit_all_minibatches": False,
            }
        metrics = actor.update_policy(batch)
        outputs.append((module.weight.detach().clone(), metrics))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    assert outputs[0][1]["actor/pg_loss"] == outputs[1][1]["actor/pg_loss"]
    assert len(outputs[0][1]["actor/grad_norm"]) == 2
    assert outputs[1][1]["train/optimizer_steps_this_outer_step"] == [2]
