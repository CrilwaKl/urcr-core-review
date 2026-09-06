"""Top-K plus tail forward KL for the URCR V3-A trajectory budget.

The custom backward recomputes only 32 selected positions at a time. It keeps
the actor's existing BF16 logits, never a full-vocabulary FP32 target/cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


def grouped_log_probs(logits: torch.Tensor, topk_ids: torch.Tensor):
    """Normalize against the complete vocabulary, including an explicit tail."""
    values = logits.double() if logits.dtype == torch.float64 else logits.float()
    ids = topk_ids.long()
    normalizer = values.logsumexp(-1)
    top = values.gather(-1, ids) - normalizer.unsqueeze(-1)
    remaining = values.scatter(-1, ids, -torch.inf)
    tail_normalizer = remaining.logsumexp(-1)
    return top, tail_normalizer - normalizer, values, remaining


def grouped_kl_from_logits(logits, topk_ids, teacher_topk_logp, teacher_log_tail):
    """Differentiable small-tensor reference, with no top-K renormalization."""
    top, tail, _, _ = grouped_log_probs(logits, topk_ids)
    p_log = teacher_topk_logp.detach().to(top)
    p_tail_log = teacher_log_tail.detach().to(tail)
    result = (p_log.exp() * (p_log - top)).sum(-1)
    finite = torch.isfinite(p_tail_log)
    # Do not evaluate 0 * (-inf - -inf), including the K == vocabulary case.
    safe_p = torch.where(finite, p_tail_log, torch.zeros_like(p_tail_log))
    safe_q = torch.where(finite, tail, torch.zeros_like(tail))
    return result + torch.where(finite, safe_p.exp() * (safe_p - safe_q), 0.0)


def grouped_logit_gradient(logits, topk_ids, teacher_topk_logp, teacher_log_tail):
    """Analytic d grouped-KL / d logits, also used for smoke calibration."""
    top, tail, values, remaining = grouped_log_probs(logits, topk_ids)
    q = torch.softmax(values, dim=-1)
    p_top = teacher_topk_logp.detach().to(top).exp()
    p_tail = teacher_log_tail.detach().to(tail).exp()
    # Account for tiny rounding error in the cached teacher probability mass.
    mass = p_top.sum(-1) + p_tail
    gradient = q * mass.unsqueeze(-1)
    if topk_ids.shape[-1] < logits.shape[-1]:
        gradient = gradient - torch.softmax(remaining, -1) * p_tail.unsqueeze(-1)
    gradient.scatter_add_(-1, topk_ids.long(), -p_top)
    return gradient


def sampled_pg_logit_energy(dlogp, sampled_probability, probability_square_sum):
    return dlogp.double().square() * (
        1.0 - 2.0 * sampled_probability.double() + probability_square_sum.double()
    )


def calibrate_focus_coefficient(ratios):
    values = sorted(float(x) for x in ratios if math.isfinite(float(x)) and float(x) > 0)
    if not values:
        return {"coefficient_max": 0.10, "median_ratio": None, "fallback": "no_valid_ratio"}
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    raw = 0.05 / median
    return {
        "coefficient_max": min(2.0, max(0.05, raw)),
        "median_ratio": median,
        "unclipped_coefficient": raw,
        "bound_hit": raw < 0.05 or raw > 2.0,
        "fallback": None,
    }


def flat_predictor_indices(attention_mask, response_width, row_indices, response_positions):
    """Map padded response positions to causal predictors in a packed forward."""
    batch, width = attention_mask.shape
    padded = row_indices.long() * width + width - response_width + response_positions.long() - 1
    live = attention_mask.reshape(-1).bool()
    if (padded < 0).any() or not live[padded].all():
        raise ValueError("V3 response predictor is outside its visible sequence")
    inverse = live.long().cumsum(0) - 1
    return inverse[padded]


class _ChunkedGroupedKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, positions, topk_ids, p_log, p_tail, block_size):
        ctx.save_for_backward(logits, positions, topk_ids, p_log, p_tail)
        ctx.block_size = int(block_size)
        losses = torch.empty(len(positions), dtype=torch.float32, device=logits.device)
        if logits.dtype == torch.float64:
            losses = losses.double()
        for start in range(0, len(positions), ctx.block_size):
            sl = slice(start, start + ctx.block_size)
            losses[sl] = grouped_kl_from_logits(
                logits[positions[sl]], topk_ids[sl], p_log[sl], p_tail[sl]
            )
        return losses

    @staticmethod
    def backward(ctx, upstream):
        logits, positions, ids, p_log, p_tail = ctx.saved_tensors
        gradient = torch.zeros_like(logits)
        for start in range(0, len(positions), ctx.block_size):
            sl = slice(start, start + ctx.block_size)
            local = grouped_logit_gradient(logits[positions[sl]], ids[sl], p_log[sl], p_tail[sl])
            local = local * upstream[sl].unsqueeze(-1)
            gradient.index_add_(0, positions[sl], local.to(logits.dtype))
        return gradient, None, None, None, None, None


def chunked_grouped_kl(logits, positions, topk_ids, p_log, p_tail, block_size=32):
    return _ChunkedGroupedKL.apply(
        logits, positions.long(), topk_ids.long(), p_log.detach(), p_tail.detach(), block_size
    )


@dataclass
class FocusForward:
    """One real actor forward's sparse target consumer (teacher or student)."""

    responses: torch.Tensor
    response_mask: torch.Tensor
    payloads: list[Any]
    topk: int = 64
    block_size: int = 32
    teacher: bool = False
    measure_energy: bool = False
    original_trajectories: int = 1024
    input_ids: torch.Tensor | None = None

    def consume(self, logits, attention_mask, *, packed):
        bsz, response_width = self.responses.shape
        width = attention_mask.shape[-1]
        flat = logits if packed else logits.reshape(-1, logits.shape[-1])
        if self.topk > flat.shape[-1]:
            raise ValueError("focus top-K exceeds vocabulary")
        device = flat.device
        row_indices, response_positions, target_ids, target_logp, target_tail = [], [], [], [], []
        lengths, selected_rows = [], []
        self.outputs = [None] * bsz
        for row, payload in enumerate(self.payloads):
            if payload is None:
                continue
            positions = list(map(int, payload["response_positions"]))
            actual = torch.nonzero(self.response_mask[row].bool(), as_tuple=False).flatten().tolist()
            if positions != actual:
                raise ValueError("V3 payload/response mask mismatch after actor reorder")
            if not positions:
                raise ValueError("V3 selected row has no response tokens")
            if not self.teacher and self.input_ids is not None:
                from verl.trainer.ppo.urcr_v3_focus import token_ids_hash
                prompt = self.input_ids[row, :width - response_width][attention_mask[row, :width - response_width].bool()]
                response = self.responses[row][self.response_mask[row].bool()]
                if token_ids_hash(prompt.tolist()) != payload["prompt_hash"] or token_ids_hash(response.tolist()) != payload["response_hash"]:
                    raise ValueError("V3 prompt/response identity mismatch after dispatch")
            selected_rows.append(row)
            lengths.append(len(positions))
            row_indices.extend([row] * len(positions))
            response_positions.extend(positions)
            if not self.teacher:
                target_ids.append(torch.as_tensor(payload["teacher_topk_ids"], device=device, dtype=torch.long))
                target_logp.append(torch.as_tensor(payload["teacher_topk_logp"], device=device, dtype=torch.float32))
                target_tail.append(torch.as_tensor(payload["teacher_log_tail"], device=device, dtype=torch.float32))
        ri = torch.tensor(row_indices, device=device, dtype=torch.long)
        rp = torch.tensor(response_positions, device=device, dtype=torch.long)
        def predictor(rows, positions):
            if packed:
                return flat_predictor_indices(attention_mask, response_width, rows, positions)
            return rows * width + width - response_width + positions - 1
        positions = predictor(ri, rp)
        self.row_kl = torch.zeros(bsz, device=device, dtype=torch.float32)
        self.focus_energy = torch.zeros((), device=device, dtype=torch.float64)
        self.negative_kl_count = 0
        self.selected_tokens = len(positions)
        if len(positions) and self.teacher:
            ids_parts, logp_parts, tail_parts = [], [], []
            for start in range(0, len(positions), self.block_size):
                sl = slice(start, start + self.block_size)
                values = flat[positions[sl]].float()
                ids = values.topk(self.topk, dim=-1).indices
                top, tail, _, _ = grouped_log_probs(values, ids)
                ids_parts.append(ids.to(dtype=torch.int32, device="cpu"))
                logp_parts.append(top.detach().cpu())
                tail_parts.append(tail.detach().cpu())
            ids, p_log, p_tail = map(torch.cat, (ids_parts, logp_parts, tail_parts))
            cursor = 0
            for row, length in zip(selected_rows, lengths):
                payload = dict(self.payloads[row])
                payload.update(
                    teacher_topk_ids=ids[cursor:cursor + length].numpy(),
                    teacher_topk_logp=p_log[cursor:cursor + length].numpy(),
                    teacher_log_tail=p_tail[cursor:cursor + length].numpy(),
                )
                self.outputs[row] = payload
                cursor += length
        elif len(positions):
            ids, p_log, p_tail = map(torch.cat, (target_ids, target_logp, target_tail))
            token_kl = chunked_grouped_kl(flat, positions, ids, p_log, p_tail, self.block_size)
            if not torch.isfinite(token_kl).all() or token_kl.detach().min() < -1e-4:
                raise FloatingPointError("V3 grouped KL is nonfinite or substantially negative")
            self.negative_kl_count = int((token_kl.detach() < 0).sum())
            row_lengths = torch.zeros(bsz, device=device)
            row_lengths[selected_rows] = torch.tensor(lengths, device=device, dtype=torch.float32)
            self.row_kl = self.row_kl.index_add(0, ri, token_kl / row_lengths[ri])
            if self.measure_energy:
                with torch.no_grad():
                    for start in range(0, len(positions), self.block_size):
                        sl = slice(start, start + self.block_size)
                        gradient = grouped_logit_gradient(flat[positions[sl]], ids[sl], p_log[sl], p_tail[sl])
                        scale = 1.0 / (self.original_trajectories * row_lengths[ri[sl]])
                        self.focus_energy += (
                            gradient.double().square().sum(-1) * scale.double().square()
                        ).sum()
        else:
            # A scalar read retains graph connectivity without reducing all logits.
            self.row_kl = self.row_kl + 0.0 * flat[0, 0]
        self.sampled_probability = self.probability_square_sum = None
        if self.measure_energy and not self.teacher:
            self.sampled_probability = torch.zeros_like(self.response_mask, dtype=torch.float32)
            self.probability_square_sum = torch.zeros_like(self.response_mask, dtype=torch.float32)
            live = torch.nonzero(self.response_mask.bool(), as_tuple=False)
            pred = predictor(live[:, 0], live[:, 1])
            with torch.no_grad():
                for start in range(0, len(live), self.block_size):
                    sl = slice(start, start + self.block_size)
                    rows, resp = live[sl, 0], live[sl, 1]
                    q = torch.softmax(flat[pred[sl]].float(), -1)
                    self.sampled_probability[rows, resp] = q.gather(
                        -1, self.responses[rows, resp].unsqueeze(-1)
                    ).squeeze(-1)
                    self.probability_square_sum[rows, resp] = q.square().sum(-1)


def payload_cache_bytes(payloads):
    return sum(
        int(value.nbytes)
        for payload in payloads if payload is not None
        for value in payload.values() if isinstance(value, np.ndarray)
    )
