"""Low-overhead update-vector audits for Plan 05 full-batch comparisons.

The audit is opt-in and keeps the training path unchanged when disabled.  It
stores only rank-local float32 gradient/update vectors for one reference run;
comparison runs stream against those vectors and do not create checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch


_CHUNK_ELEMENTS = 4 * 1024 * 1024


def _row_value(batch, key: str, index: int, default=None):
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    return values[index]


def _stable_update_audit_row_key(batch, index: int, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Identify one rollout turn without using run-local UUIDs or arrival order."""
    extra_info = _row_value(batch, "extra_info", index, {})
    env_kwargs = _row_value(batch, "env_kwargs", index, {})
    dataset_index = extra_info.get("index", -1) if isinstance(extra_info, dict) else -1
    question = ""
    for value in (extra_info, env_kwargs):
        if isinstance(value, dict) and value.get("question"):
            question = str(value["question"])
            break
    descriptor = {
        "data_source": str(_row_value(batch, "data_source", index, "")),
        "dataset_index": int(dataset_index),
        "question": question,
        "turn_step": int(_row_value(batch, "turn_step", index, -1)),
        "episode_reward": float(_row_value(batch, "episode_rewards", index, 0.0)),
        "episode_length": int(_row_value(batch, "episode_lengths", index, -1)),
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    valid_ids = input_ids[index][attention_mask[index].bool()].contiguous().numpy()
    digest.update(str(valid_ids.dtype).encode("ascii"))
    digest.update(np.asarray(valid_ids.shape, dtype=np.int64).tobytes())
    digest.update(valid_ids.tobytes())
    return digest.digest()


def canonicalize_update_audit_batch(batch):
    """Return a semantically ordered batch for repeatable full-update audits.

    Multi-turn rollout collection may return the same trajectories in a different
    arrival order, while ``uid`` and ``traj_uid`` are intentionally run-local.
    Update audits therefore sort by stable question/turn metadata and generated
    token content before divisibility copies and sequence-length balancing.
    """
    if len(batch) <= 1:
        return batch
    if batch.batch is None or "input_ids" not in batch.batch or "attention_mask" not in batch.batch:
        raise ValueError("Plan 05 update audit requires input_ids and attention_mask")
    input_ids = batch.batch["input_ids"].detach().cpu()
    attention_mask = batch.batch["attention_mask"].detach().cpu()
    keys = [
        _stable_update_audit_row_key(batch, index, input_ids, attention_mask)
        for index in range(len(batch))
    ]
    order = np.asarray(sorted(range(len(batch)), key=keys.__getitem__), dtype=np.int64)
    return batch.select_idxs(order)


def _parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters()]


def _cosine(dot: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    return float(dot / denominator) if denominator > 0 else float("nan")


class DistributedUpdateAudit:
    """Capture one distributed actor update without retaining a checkpoint."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        reference_dir: str | Path | None,
        rank: int,
        world_size: int,
        save_reference: bool,
        compare_reference: bool,
        summary_only: bool = False,
    ) -> None:
        if summary_only:
            if save_reference or compare_reference:
                raise ValueError("Summary-only update audit cannot save or compare vectors")
        elif save_reference == compare_reference:
            raise ValueError("Exactly one of save_reference/compare_reference must be true")
        if not summary_only and reference_dir is None:
            raise ValueError("Vector update audit requires a reference directory")
        self.reference_dir = Path(reference_dir) if reference_dir is not None else None
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.save_reference = bool(save_reference)
        self.compare_reference = bool(compare_reference)
        self.summary_only = bool(summary_only)
        self._params = _parameters(model)
        self._base = [parameter.detach().cpu().float().clone() for parameter in self._params]
        self._gradient_sum = [torch.zeros_like(value) for value in self._base]
        self.optimizer_steps = 0

    @property
    def _update_path(self) -> Path:
        assert self.reference_dir is not None
        return self.reference_dir / f"update_world_size_{self.world_size}_rank_{self.rank}.f32"

    @property
    def _gradient_path(self) -> Path:
        assert self.reference_dir is not None
        return self.reference_dir / f"gradient_world_size_{self.world_size}_rank_{self.rank}.f32"

    @property
    def _manifest_path(self) -> Path:
        assert self.reference_dir is not None
        return self.reference_dir / f"manifest_world_size_{self.world_size}_rank_{self.rank}.json"

    def capture_gradient_step(self) -> None:
        """Accumulate the pre-clipping gradient for each optimizer step."""
        for parameter, accumulator in zip(self._params, self._gradient_sum):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().reshape(-1)
            target = accumulator.reshape(-1)
            if gradient.numel() != target.numel():
                raise RuntimeError("FSDP gradient shard changed shape during Plan 05 update audit")
            for start in range(0, gradient.numel(), _CHUNK_ELEMENTS):
                stop = min(start + _CHUNK_ELEMENTS, gradient.numel())
                target[start:stop].add_(gradient[start:stop].float().cpu())
        self.optimizer_steps += 1

    def _update_chunks(self) -> Iterable[torch.Tensor]:
        for parameter, base in zip(self._params, self._base):
            current = parameter.detach().reshape(-1)
            base_flat = base.reshape(-1)
            if current.numel() != base_flat.numel():
                raise RuntimeError("FSDP parameter shard changed shape during Plan 05 update audit")
            for start in range(0, current.numel(), _CHUNK_ELEMENTS):
                stop = min(start + _CHUNK_ELEMENTS, current.numel())
                yield current[start:stop].float().cpu() - base_flat[start:stop]

    def _gradient_chunks(self) -> Iterable[torch.Tensor]:
        for value in self._gradient_sum:
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), _CHUNK_ELEMENTS):
                yield flat[start : start + _CHUNK_ELEMENTS]

    def _group_local_stats(self) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        update_groups = []
        gradient_groups = []
        for parameter, base, gradient_sum in zip(
            self._params,
            self._base,
            self._gradient_sum,
        ):
            update_groups.append(
                self._local_stats(
                    (
                        parameter.detach().reshape(-1)[start:stop].float().cpu()
                        - base.reshape(-1)[start:stop]
                        for start in range(0, parameter.numel(), _CHUNK_ELEMENTS)
                        for stop in (min(start + _CHUNK_ELEMENTS, parameter.numel()),)
                    )
                )
            )
            gradient_groups.append(
                self._local_stats(
                    (
                        gradient_sum.reshape(-1)[start:stop]
                        for start in range(0, gradient_sum.numel(), _CHUNK_ELEMENTS)
                        for stop in (min(start + _CHUNK_ELEMENTS, gradient_sum.numel()),)
                    )
                )
            )
        return update_groups, gradient_groups

    @staticmethod
    def _reduce_group_stats(
        update_groups: list[dict[str, float]],
        gradient_groups: list[dict[str, float]],
        device: torch.device,
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        stat_keys = ("current_sq", "nonfinite", "elements")
        values = []
        for update, gradient in zip(update_groups, gradient_groups):
            values.extend(update[key] for key in stat_keys)
            values.extend(gradient[key] for key in stat_keys)
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        reduced_update = []
        reduced_gradient = []
        values = tensor.cpu().tolist()
        stride = 2 * len(stat_keys)
        for offset in range(0, len(values), stride):
            reduced_update.append(
                dict(zip(stat_keys, values[offset : offset + len(stat_keys)]))
            )
            reduced_gradient.append(
                dict(
                    zip(
                        stat_keys,
                        values[offset + len(stat_keys) : offset + stride],
                    )
                )
            )
        return reduced_update, reduced_gradient

    @staticmethod
    def _local_stats(
        chunks: Iterable[torch.Tensor],
        *,
        reference: np.memmap | None = None,
        output_path: Path | None = None,
    ) -> dict[str, float]:
        offset = 0
        sums = {
            "current_sq": 0.0,
            "reference_sq": 0.0,
            "dot": 0.0,
            "difference_sq": 0.0,
            "nonfinite": 0.0,
            "elements": 0.0,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        context = output_path.open("xb") if output_path is not None else nullcontext(None)
        with context as handle:
            for chunk in chunks:
                value = chunk.detach().float().cpu().contiguous()
                if handle is not None:
                    value.numpy().tofile(handle)
                finite = torch.isfinite(value)
                sums["nonfinite"] += float((~finite).sum())
                safe = torch.where(finite, value, torch.zeros_like(value)).double()
                sums["current_sq"] += float(torch.dot(safe, safe))
                sums["elements"] += float(value.numel())
                if reference is not None:
                    ref_np = np.asarray(reference[offset : offset + value.numel()])
                    if len(ref_np) != value.numel():
                        raise RuntimeError("Plan 05 reference vector is shorter than the actor shard")
                    ref = torch.from_numpy(np.array(ref_np, copy=True)).double()
                    sums["reference_sq"] += float(torch.dot(ref, ref))
                    sums["dot"] += float(torch.dot(safe, ref))
                    difference = safe - ref
                    sums["difference_sq"] += float(torch.dot(difference, difference))
                    offset += value.numel()
        if reference is not None and offset != len(reference):
            raise RuntimeError("Plan 05 reference vector length does not match the actor shard")
        return sums

    @staticmethod
    def _reduce(sums: dict[str, float], device: torch.device) -> dict[str, float]:
        keys = tuple(sums)
        values = torch.tensor([sums[key] for key in keys], dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}

    def _reference(self, path: Path, expected_elements: int) -> np.memmap:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Plan 05 reference vector: {path}")
        reference = np.memmap(path, dtype=np.float32, mode="r")
        if len(reference) != expected_elements:
            raise RuntimeError(
                f"Plan 05 reference size mismatch for {path}: {len(reference)} != {expected_elements}"
            )
        return reference

    def finalize(self, model: torch.nn.Module) -> dict[str, float]:
        if [id(value) for value in _parameters(model)] != [id(value) for value in self._params]:
            raise RuntimeError("Actor parameter ordering changed during Plan 05 update audit")
        expected_elements = sum(value.numel() for value in self._base)

        update_reference = None
        gradient_reference = None
        if self.compare_reference:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if manifest["parameter_numels"] != [value.numel() for value in self._base]:
                raise RuntimeError("Plan 05 reference parameter layout mismatch")
            if int(manifest["optimizer_steps"]) != self.optimizer_steps:
                raise RuntimeError("Plan 05 reference optimizer-step count mismatch")
            update_reference = self._reference(self._update_path, expected_elements)
            gradient_reference = self._reference(self._gradient_path, expected_elements)

        if self.save_reference:
            if self._manifest_path.exists() or self._update_path.exists() or self._gradient_path.exists():
                raise FileExistsError(f"Refusing to overwrite Plan 05 update reference in {self.reference_dir}")

        update_local = self._local_stats(
            self._update_chunks(),
            reference=update_reference,
            output_path=self._update_path if self.save_reference else None,
        )
        gradient_local = self._local_stats(
            self._gradient_chunks(),
            reference=gradient_reference,
            output_path=self._gradient_path if self.save_reference else None,
        )

        if self.save_reference:
            self._manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "parameter_numels": [value.numel() for value in self._base],
                        "optimizer_steps": self.optimizer_steps,
                        "dtype": "float32",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        device = next(model.parameters()).device
        update = self._reduce(update_local, device)
        gradient = self._reduce(gradient_local, device)
        metrics = {
            "actor/audit_update_delta_l2": math.sqrt(max(update["current_sq"], 0.0)),
            "actor/audit_gradient_sum_l2": math.sqrt(max(gradient["current_sq"], 0.0)),
            "actor/audit_update_nonfinite_count": update["nonfinite"],
            "actor/audit_gradient_nonfinite_count": gradient["nonfinite"],
            "actor/audit_optimizer_steps": float(self.optimizer_steps),
        }
        update_groups, gradient_groups = self._group_local_stats()
        update_groups, gradient_groups = self._reduce_group_stats(
            update_groups,
            gradient_groups,
            device,
        )
        metrics["actor/audit_parameter_group_count"] = float(len(update_groups))
        for index, (update_group, gradient_group) in enumerate(
            zip(update_groups, gradient_groups)
        ):
            prefix = f"actor/audit_group_{index:03d}"
            metrics[f"{prefix}_elements"] = update_group["elements"]
            metrics[f"{prefix}_update_l2"] = math.sqrt(
                max(update_group["current_sq"], 0.0)
            )
            metrics[f"{prefix}_gradient_l2"] = math.sqrt(
                max(gradient_group["current_sq"], 0.0)
            )
            metrics[f"{prefix}_update_nonfinite_count"] = update_group["nonfinite"]
            metrics[f"{prefix}_gradient_nonfinite_count"] = gradient_group["nonfinite"]
        if self.compare_reference:
            metrics.update(
                {
                    "actor/audit_update_delta_cosine_vs_reference": _cosine(
                        update["dot"], update["current_sq"], update["reference_sq"]
                    ),
                    "actor/audit_update_delta_relative_l2_vs_reference": math.sqrt(
                        max(update["difference_sq"], 0.0) / max(update["reference_sq"], 1e-300)
                    ),
                    "actor/audit_gradient_sum_cosine_vs_reference": _cosine(
                        gradient["dot"], gradient["current_sq"], gradient["reference_sq"]
                    ),
                    "actor/audit_gradient_sum_relative_l2_vs_reference": math.sqrt(
                        max(gradient["difference_sq"], 0.0) / max(gradient["reference_sq"], 1e-300)
                    ),
                }
            )
        return metrics
