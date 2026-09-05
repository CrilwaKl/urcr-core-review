from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.summarize_evisd_answer_multipliers import distribution
from verl.trainer.ppo.urcr_answer_agam import (
    apply_answer_agam,
    label_answer_tokens,
    linear_annealed_answer_lambda,
    validate_answer_agam_config,
)
from verl.trainer.ppo.urcr_diagnostics import parse_generated_action_spans


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def batch_decode(self, batches, **_kwargs):
        return [self.decode(token_ids) for token_ids in batches]


def _answer_row():
    tokenizer = PieceTokenizer(
        [
            "reasoning ",
            "<",
            "answer",
            ">",
            "Paris",
            ",",
            " ",
            "Tex",
            "as",
            "</",
            "answer",
            ">",
        ]
    )
    token_ids = list(range(len(tokenizer.pieces)))
    parsed = parse_generated_action_spans(tokenizer, token_ids)
    row = {
        **parsed,
        "response_token_ids": token_ids,
        "ground_truth_aliases": ["Paris, France"],
        "_token_char_offsets": parsed["token_char_offsets"],
    }
    return tokenizer, row


def test_answer_token_labels_preserve_original_subwords_and_neutral_punctuation():
    tokenizer, row = _answer_row()
    labels = label_answer_tokens(row, tokenizer)

    assert labels.eligible
    assert labels.best_alias == "Paris, France"
    assert labels.best_alias_token_f1 == pytest.approx(0.5)
    assert labels.z_by_local_token[4] == 1
    assert labels.z_by_local_token[5] == 0  # punctuation
    assert labels.z_by_local_token[6] == 0  # whitespace
    assert labels.z_by_local_token[7] == -1
    assert labels.z_by_local_token[8] == -1  # same normalized word, second subword
    assert labels.aligned_token_count == 1
    assert labels.unmatched_token_count == 2
    assert labels.neutral_token_count == 2


@pytest.mark.parametrize(
    ("anchor_value", "aligned_value", "unmatched_value"),
    [(2.0, 2.4, 1.6), (-2.0, -1.6, -2.4)],
)
def test_agam_is_bounded_sign_preserving_and_answer_only(
    anchor_value,
    aligned_value,
    unmatched_value,
):
    tokenizer, row = _answer_row()
    response_length = len(row["response_token_ids"])
    response_mask = torch.ones((1, response_length))
    anchor = torch.full((1, response_length), anchor_value)
    routed = anchor.clone()
    routed[0, 0] += 0.25  # existing think-side URCR credit

    result = apply_answer_agam(
        routed,
        anchor,
        response_mask,
        [row],
        tokenizer,
        lambda_effective=0.2,
    )

    assert result.advantages[0, 0].item() == pytest.approx(anchor_value + 0.25)
    assert result.advantages[0, 1:4].tolist() == pytest.approx(
        [anchor_value] * 3
    )
    assert result.advantages[0, 4].item() == pytest.approx(aligned_value)
    assert result.advantages[0, 5:7].tolist() == pytest.approx(
        [anchor_value] * 2
    )
    assert result.advantages[0, 7:9].tolist() == pytest.approx(
        [unmatched_value] * 2
    )
    assert result.advantages[0, 9:].tolist() == pytest.approx(
        [anchor_value] * 3
    )
    assert result.metrics["agam/residual_outside_answer_max_abs"] == 0.0
    assert result.metrics["agam/base_answer_anchor_max_error"] == 0.0
    assert result.metrics["agam/sign_flip_count"] == 0.0
    assert result.metrics["agam/multiplier_min"] == pytest.approx(0.8)
    assert result.metrics["agam/multiplier_max"] == pytest.approx(1.2)


def test_zero_grpo_advantage_naturally_disables_agam_without_group_gate():
    tokenizer, row = _answer_row()
    response_length = len(row["response_token_ids"])
    response_mask = torch.ones((1, response_length))
    anchor = torch.zeros((1, response_length))
    routed = anchor.clone()
    routed[0, 0] = 0.25

    result = apply_answer_agam(
        routed,
        anchor,
        response_mask,
        [row],
        tokenizer,
        lambda_effective=0.1,
    )

    assert torch.equal(result.advantages, routed)
    assert torch.count_nonzero(result.residual).item() == 0
    assert result.metrics["agam/correction_active_row_count"] == 0.0


def test_lambda_zero_is_exact_current_objective():
    tokenizer, row = _answer_row()
    response_length = len(row["response_token_ids"])
    response_mask = torch.ones((1, response_length))
    anchor = torch.full((1, response_length), 1.5)
    routed = anchor.clone()
    routed[0, 0] += 0.5

    result = apply_answer_agam(
        routed,
        anchor,
        response_mask,
        [row],
        tokenizer,
        lambda_effective=0.0,
    )

    assert torch.equal(result.advantages, routed)
    assert torch.count_nonzero(result.residual).item() == 0


def test_agam_rejects_answer_positions_modified_by_an_upstream_objective():
    tokenizer, row = _answer_row()
    response_length = len(row["response_token_ids"])
    response_mask = torch.ones((1, response_length))
    anchor = torch.ones((1, response_length))
    routed = anchor.clone()
    routed[0, 4] += 0.01

    with pytest.raises(RuntimeError, match="unmodified GRPO anchor"):
        apply_answer_agam(
            routed,
            anchor,
            response_mask,
            [row],
            tokenizer,
            lambda_effective=0.1,
        )


def test_answer_agam_config_is_minimal_and_frozen():
    assert not validate_answer_agam_config({}).enabled
    with pytest.raises(ValueError, match="must be explicit"):
        validate_answer_agam_config({"enable": True})
    configured = validate_answer_agam_config({"enable": True, "lambda": 0.1})
    assert configured.enabled and configured.lambda_start == 0.1
    weaker = validate_answer_agam_config({"enable": True, "lambda": 0.05})
    assert weaker.enabled and weaker.lambda_start == 0.05
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        validate_answer_agam_config({"enable": True, "lambda": 1.0})
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_answer_agam_config(
            {"enable": True, "lambda": 0.1, "mixed_group_gate": True}
        )


def test_answer_agam_linear_anneal_uses_global_optimizer_step():
    start = 0.1
    assert linear_annealed_answer_lambda(
        start, global_step=1, total_training_steps=300
    ) == pytest.approx(0.1)
    assert linear_annealed_answer_lambda(
        start, global_step=150, total_training_steps=300
    ) == pytest.approx(0.1 * 150 / 299)
    assert linear_annealed_answer_lambda(
        start, global_step=300, total_training_steps=300
    ) == 0.0
    assert linear_annealed_answer_lambda(
        start, global_step=301, total_training_steps=300
    ) == 0.0
    assert linear_annealed_answer_lambda(
        start, global_step=101, total_training_steps=300
    ) == pytest.approx(0.1 * 199 / 299)
    with pytest.raises(ValueError, match="global_step"):
        linear_annealed_answer_lambda(
            start, global_step=0, total_training_steps=300
        )
    with pytest.raises(ValueError, match="at least 2"):
        linear_annealed_answer_lambda(
            start, global_step=1, total_training_steps=1
        )


def test_answer_agam_formal_config_preserves_loo_protocol():
    repo = Path(__file__).resolve().parents[3]
    config = OmegaConf.to_container(
        OmegaConf.load(
            repo / "examples/urcr_online/configs/plan07_answer_agam_core.yaml"
        ),
        resolve=True,
    )
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    urcr = config["algorithm"]["urcr"]

    assert config["data"]["train_batch_size"] == 128
    assert config["data"]["shuffle"] is True
    assert config["data"]["seed"] == 1
    assert config["env"]["rollout"]["n"] == 8
    assert actor["optim"]["lr"] == 1e-6
    assert actor["optim"]["lr_warmup_steps_ratio"] == 0.1
    assert actor["ppo_mini_batch_size"] == 256
    assert actor["ppo_micro_batch_size_per_gpu"] == 16
    assert rollout["log_prob_micro_batch_size_per_gpu"] == 32
    assert urcr["method"] == "g2_real"
    assert urcr["routing"]["think_credit_mode"] == "loo_mass50"
    assert urcr["answer_agam"] == {"enable": True, "lambda": 0.1}
    assert config["algorithm"]["evisd"]["enable"] is False
    assert config["algorithm"]["evisd"]["answer_enable"] is False
    assert config["trainer"]["n_gpus_per_node"] == 4
    assert config["trainer"]["total_training_steps"] == 300
    assert config["trainer"]["save_freq"] == 50


def test_evisd_multiplier_histogram_uses_requested_half_open_bins():
    summary = distribution([0.8, 0.849, 0.85, 0.90, 1.0, 1.05, 1.199, 1.2])

    assert summary["count"] == 8
    assert summary["bins"]["[0.80,0.85)"]["count"] == 2
    assert summary["bins"]["[0.85,0.90)"]["count"] == 1
    assert summary["bins"]["[0.90,0.95)"]["count"] == 1
    assert summary["bins"]["[1.00,1.05)"]["count"] == 1
    assert summary["bins"]["[1.05,1.10)"]["count"] == 1
    assert summary["bins"]["[1.15,1.20]"]["count"] == 2
