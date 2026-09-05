# URCR Core Review Snapshot

This private repository is a compact, code-level review bundle for the active
URCR implementation. It is intended to let a research collaborator inspect the
reward, responsibility, routing, and PPO-loss data flow without loading the
full EviSD/veRL tree or any runtime artifacts.

It is **not** a standalone training repository. The integration patches are
defined relative to the pinned EviSD source revision documented in
[`UPSTREAM.md`](UPSTREAM.md).

## Recommended review order

1. `configs/urcr_v2_fixed_support_agam.yaml` — active V2 method settings.
2. `src/verl/trainer/ppo/urcr_sources.py` — metadata-derived support reward.
3. `src/verl/trainer/ppo/urcr_responsibility.py` — masked-query responsibility
   scoring.
4. `src/verl/trainer/ppo/urcr_localized.py` and `urcr_routing.py` — content-only
   chunk selection and credit routing.
5. `src/verl/trainer/ppo/urcr_local_objective.py` — local PPO surrogate terms,
   normalization, warm-up, and `local_max` scaling.
6. `src/verl/trainer/ppo/urcr_answer_agam.py` — answer-side modulation and
   annealing.
7. `patches/evisd_ray_trainer.patch` — construction and attachment of the URCR
   tensors in the training loop.
8. `patches/fsdp_workers.patch` — no-grad responsibility scorer execution.
9. `patches/dp_actor.patch` — the exact point where local losses are combined
   with the ordinary PPO actor loss and back-propagated.
10. `patches/core_algos.patch` — token-level PPO loss helpers used by the local
    objective and audits.

The files under `tests/` cover the current V2 reward, routing, local-objective,
AGAM, and token-alignment contracts. The launcher and its parent config are
included so that batch, rollout, optimizer, sequence, FSDP, and method settings
can be checked together. Local filesystem paths are operational details, not
portable dependencies of this review bundle.

## Scope intentionally excluded

- the complete EviSD, SDAR, or veRL source trees;
- models, datasets, retrieval indexes, checkpoints, and trajectories;
- training/evaluation logs and result reports;
- tmux/watchdog scripts and historical Plan 02–07 diagnostics;
- caches, generated outputs, credentials, and local environment files.

## Snapshot identity

The exported files themselves, and the commit that contains them, are the
authoritative review snapshot. See [`SOURCE_STATE.md`](SOURCE_STATE.md) for the
local source revision from which it was produced.

