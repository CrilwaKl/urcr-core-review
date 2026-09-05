"""Credit-conserving shadow objectives for URCR Plan 04."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def softmax_chunk_weights(scores: Iterable[float], *, robust_scale: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if not len(values):
        return values
    if not math.isfinite(robust_scale) or robust_scale <= 0:
        raise ValueError("robust_scale must be finite and positive")
    logits = np.clip(values / robust_scale, -5.0, 5.0)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def uniform_token_allocation(total: float, token_count: int) -> np.ndarray:
    if token_count <= 0:
        return np.asarray([], dtype=np.float64)
    return np.full(token_count, float(total) / token_count, dtype=np.float64)


def typed_token_allocation(
    total: float,
    chunk_weights: Iterable[float],
    chunk_token_counts: Iterable[int],
) -> np.ndarray:
    weights = np.asarray(list(chunk_weights), dtype=np.float64)
    counts = np.asarray(list(chunk_token_counts), dtype=np.int64)
    if len(weights) != len(counts):
        raise ValueError("chunk weights/counts length mismatch")
    if not len(weights):
        return np.asarray([], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("chunk token counts must be positive")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("chunk weights must sum to one")
    allocations = [
        np.full(int(count), float(total) * weight / int(count), dtype=np.float64)
        for weight, count in zip(weights, counts)
    ]
    result = np.concatenate(allocations)
    if not np.isclose(result.sum(), total, atol=1e-9):
        raise AssertionError("typed allocation does not conserve total credit")
    return result


def allocation_entropy(weights: Iterable[float]) -> tuple[float | None, float | None]:
    values = np.asarray(list(weights), dtype=np.float64)
    if not len(values):
        return None, None
    positive = values[values > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return entropy, float(math.exp(entropy))


def acquisition_totals(
    *,
    outcome_advantage: float,
    acquisition_credit: float,
    rho: float,
    lambda_a: float,
    lambda_r: float,
) -> tuple[float, float]:
    if min(acquisition_credit, rho, lambda_a, lambda_r) < 0:
        raise ValueError("Plan04 conservative regime requires nonnegative inputs")
    query_total = lambda_a * abs(float(outcome_advantage)) * acquisition_credit
    think_total = lambda_r * rho * query_total
    return float(query_total), float(think_total)


def route_a_think_total(
    *, query_signed_source: float, rho: float, lambda_r: float
) -> float:
    if rho < 0 or lambda_r < 0:
        raise ValueError("rho and lambda_r must be nonnegative")
    return float(lambda_r * rho * query_signed_source)


def shuffled_rho(
    values: Iterable[float],
    strata: Iterable[tuple],
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    strata = list(strata)
    if len(values) != len(strata):
        raise ValueError("rho/strata length mismatch")
    output = values.copy()
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for index, key in enumerate(strata):
        groups.setdefault(tuple(key), []).append(index)
    for indices in groups.values():
        output[indices] = values[rng.permutation(indices)]
    return output
