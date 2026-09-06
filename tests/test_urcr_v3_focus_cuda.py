"""Actual Qwen2/FlashAttention actor path, explicitly opt-in on one selected GPU."""

from copy import deepcopy
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(os.environ.get("URCR_V3_CUDA_TEST") != "1", reason="explicit GPU test opt-in required")


def test_actual_remove_padding_teacher_and_student_backward():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.trainer.ppo.urcr_v3_focus_loss import FocusForward
    from verl.utils.model import compute_position_id_with_mask
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.manual_seed(20260905)
    model_config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        attention_dropout=0., pad_token_id=0, eos_token_id=3,
    )
    model_config._attn_implementation = "flash_attention_2"
    model = Qwen2ForCausalLM(model_config).to(device="cuda", dtype=torch.bfloat16).eval()
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False)
    cfg = OmegaConf.create({"use_remove_padding": False, "ulysses_sequence_parallel_size": 1, "use_torch_compile": False})
    dense_actor = DataParallelPPOActor(cfg, model)
    packed_cfg = deepcopy(cfg)
    packed_cfg.use_remove_padding = True
    packed_actor = DataParallelPPOActor(packed_cfg, model)
    ids = torch.tensor([[0, 0, 21, 22, 23, 31, 32, 3], [0, 41, 42, 43, 44, 51, 3, 0]], device="cuda")
    mask = ids.ne(0).long()
    batch = {"input_ids": ids, "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask), "responses": ids[:, -3:]}
    payloads = [{"response_positions": np.array([0, 1, 2])}, {"response_positions": np.array([0, 1])}]
    contexts, log_probs = [], []
    with torch.no_grad():
        for actor in (dense_actor, packed_actor):
            context = FocusForward(batch["responses"], mask[:, -3:], payloads, topk=64, teacher=True)
            actor._forward_micro_batch(batch, 1., focus_context=context)
            contexts.append(context)
            log_probs.append(actor._forward_micro_batch(batch, 1.)[1])
    torch.testing.assert_close(log_probs[0][mask[:, -3:].bool()], log_probs[1][mask[:, -3:].bool()], atol=.015, rtol=.003)
    for row in range(2):
        # Top-K order can swap at BF16 ties, so compare teacher distributions by ID.
        a, b = contexts[0].outputs[row], contexts[1].outputs[row]
        for position in (0, len(payloads[row]["response_positions"]) - 1):
            pa = dict(zip(a["teacher_topk_ids"][position], a["teacher_topk_logp"][position]))
            pb = dict(zip(b["teacher_topk_ids"][position], b["teacher_topk_logp"][position]))
            common = pa.keys() & pb.keys()
            assert len(common) >= 60
            assert max(abs(pa[i] - pb[i]) for i in common) < .02
        np.testing.assert_allclose(a["teacher_log_tail"], b["teacher_log_tail"], atol=.01)
    assert all(p.grad is None for p in model.parameters())
    # Same logits feed sampled PG, entropy, and grouped KL; no in-place overwrite.
    context = FocusForward(batch["responses"], mask[:, -3:], contexts[1].outputs, topk=64, measure_energy=True)
    entropy, logp = packed_actor._forward_micro_batch(batch, 1., calculate_entropy=True, focus_context=context)
    assert context.row_kl.abs().max() < 1e-5
    loss = -(logp * mask[:, -3:]).sum() - .001 * (entropy * mask[:, -3:]).sum() + .1 * context.row_kl.sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.float().square().sum().item() for p in model.parameters() if p.grad is not None) > 0

