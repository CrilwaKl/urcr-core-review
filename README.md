# URCR-V4 Core Review

This repository is a focused external-review snapshot of the implemented
URCR-V4 method. It lets a reviewer inspect the frozen design, the exact
reward-to-loss path, the formal training configuration, and the evidence used
to accept the implementation without access to the private training machine.

The principal review document is
[`REPORT_URCR_V4_VALIDATION.md`](REPORT_URCR_V4_VALIDATION.md). It records the
ten-batch on-policy calibration, scorer and masking checks, gradient-scale
calibration, Base shadow check, three optimizer smoke steps, measured costs,
limitations, and concrete questions for an independent reviewer.

## Review order

1. [`PLAN_URCR_V4.md`](PLAN_URCR_V4.md) — frozen scientific specification.
2. [`REPORT_URCR_V4_VALIDATION.md`](REPORT_URCR_V4_VALIDATION.md) — measured
   validation evidence and unresolved review questions.
3. [`configs/resolved_method.json`](configs/resolved_method.json) — the frozen
   empirical scales and all method invariants consumed by training.
4. [`src/verl/trainer/ppo/urcr_v4_method.py`](src/verl/trainer/ppo/urcr_v4_method.py)
   — answer residual, search utility, responsibility, routing, cap, and metadata
   boundary.
5. [`src/verl/trainer/ppo/urcr_v4_scorer.py`](src/verl/trainer/ppo/urcr_v4_scorer.py)
   and [`urcr_v4_requests.py`](src/verl/trainer/ppo/urcr_v4_requests.py) —
   teacher-forced request construction and position-preserving information
   barrier.
6. [`src/verl/trainer/ppo/urcr_v4_pipeline.py`](src/verl/trainer/ppo/urcr_v4_pipeline.py),
   [`urcr_v4_data.py`](src/verl/trainer/ppo/urcr_v4_data.py), and
   [`urcr_v4_controller.py`](src/verl/trainer/ppo/urcr_v4_controller.py) —
   rollout records, scorer orchestration, and frozen token credits.
7. [`src/verl/trainer/ppo/urcr_v4_local_objective.py`](src/verl/trainer/ppo/urcr_v4_local_objective.py)
   — signed span-mean PPO and the trajectory-mean minibatch estimator.
8. [`patches/evisd_ray_trainer.patch`](patches/evisd_ray_trainer.patch),
   [`patches/dp_actor.patch`](patches/dp_actor.patch), and
   [`patches/fsdp_workers.patch`](patches/fsdp_workers.patch) — train-loop,
   actor-loss, and distributed scorer integration relative to the frozen V2
   source snapshot.
9. [`launchers/run_urcr_v4_training.sh`](launchers/run_urcr_v4_training.sh) and
   [`configs/formal_resolved_config.yaml`](configs/formal_resolved_config.yaml)
   — exact launcher and resolved configuration for the 300-step formal run.
10. [`artifacts/`](artifacts/) and [`tests/`](tests/) — compact evidence,
    including the final-hash formal S1–S5 prefix, and focused contracts.

## Implemented data flow

The ordinary terminal-reward GRPO path remains the global objective. URCR-V4
adds a separately normalized local PPO term. Search actions receive a signed
utility only when real-vs-null and real-vs-control answer-probe contrasts agree
in direction. Final answers receive a within-question, within-binary-EM-stratum
F0.5 residual. A no-grad scorer estimates how much the sampled think promotes
the actual sampled query or answer; positive dependency routes part of the
same signed action utility to at most six think chunks. Search local mass is
capped per trajectory. All coefficients are frozen before PPO shuffling and
the actor uses one forward pass for the global and local terms.

Training-time evidence metadata is rejected by the V4 data path. The formal
configuration disables V1/V2 fixed support rewards, AGAM, V3 focus KL, EviSD
teacher modulation, SDAR/SDL actor losses, and the old URCR path. The inherited
data and environment seeds remain, while no extra rollout-controller seed or
rollout duplication override is configured.

## Scope

This is a code-review bundle, not a standalone training checkout. V4-owned
modules and scripts are copied exactly. Integration changes are patches against
the V2 umbrella revision named in [`SOURCE_STATE.md`](SOURCE_STATE.md); the
previous commit of this review repository preserves that V2 review snapshot.
Models, checkpoints, datasets, retrieval indexes, raw trajectories, full batch
artifacts, runtime logs, caches, and credentials are excluded.

[`configs/plan07_answer_agam_core.yaml`](configs/plan07_answer_agam_core.yaml)
is included only because the four-GPU diagnostic script composes that inherited
Hydra base before applying V4 overrides. Its historical filename is not an
enabled Plan-07/AGAM mechanism; the resolved formal config is authoritative.

The 300-step formal run was launched from the original Qwen2.5-3B-Instruct
Base. Its final training and held-out evaluation results are intentionally not
claimed in this pre-result validation snapshot.
