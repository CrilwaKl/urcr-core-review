# URCR V2 Core Review

This repository is a compact code-review bundle for the formal URCR V2
implementation. It is intended for an external research collaborator or a
long-context model to inspect the complete V2 reward-to-loss path without
loading the full EviSD/veRL workspace, runtime artifacts, or the later V3
experiments.

The selected source is frozen at umbrella revision
`b4f1c0be3351166a0f76223f3b15ccae39ffdab9`. This is a mechanism review
subset, not a standalone training repository or a full archive of that commit.
The integration patches are relative to the EviSD revision pinned in
[`UPSTREAM.md`](UPSTREAM.md).

## V2 path represented here

The formal config uses GRPO for the ordinary outcome-reward path while adding
three V2-specific operations:

1. `urcr_sources.py` derives a detached, metadata-based support utility for
   each search action. With the formal settings, a new supporting-fact hit has
   utility `1.0`, a new supporting-document-only hit has utility `0.5`, and a
   repeated, missed, invalid, or metadata-free search has utility `0.0`. The
   base query reward is `0.5`.
2. `urcr_responsibility.py`, `urcr_localized.py`, and `urcr_routing.py` score
   the preceding think content under `no_grad`, select the LOO positive-mass
   50% support, and route the local credit to the query and selected think
   tokens.
3. `urcr_local_objective.py` computes a separate clipped PPO surrogate over
   eligible local actions. It is combined with the ordinary actor objective in
   `dp_actor.patch`, with `local_max=0.01` and a 30-update warm-up.

`urcr_answer_agam.py` contains the V2 answer-side AGAM used by the formal
configuration: lambda starts at `0.1` and is linearly annealed to zero at the
final optimizer update. The EviSD teacher/search modulation and the SDAR/SDL
actor losses are disabled by the supplied configuration.

## Recommended review order

1. `configs/urcr_v2_fixed_support_agam.yaml` — the V2 overrides.
2. `configs/plan07_answer_agam_core.yaml` — the inherited full training
   configuration.
3. `src/verl/trainer/ppo/urcr_sources.py` — config validation and fixed support
   reward construction.
4. `src/verl/trainer/ppo/urcr_responsibility.py` — masked-query responsibility
   scoring.
5. `src/verl/trainer/ppo/urcr_localized.py` and `urcr_routing.py` — content-only
   selection and token routing.
6. `src/verl/trainer/ppo/urcr_local_objective.py` — local PPO terms,
   normalization, warm-up, and scaling.
7. `src/verl/trainer/ppo/urcr_answer_agam.py` — answer-side modulation and
   annealing.
8. `patches/evisd_ray_trainer.patch` — construction and attachment of the V2
   tensors in the training loop.
9. `patches/fsdp_workers.patch` — no-grad responsibility scorer execution.
10. `patches/dp_actor.patch` — combination with the ordinary PPO actor loss.
11. `patches/core_algos.patch` — token-level PPO loss helpers.
12. `tests/` — reward, routing, objective, AGAM, and token-alignment contracts.

The most useful review questions are whether the support utility remains
strictly outside terminal GRPO reward normalization, whether the query-to-think
alignment can misassign credit, whether the local loss scale has the intended
meaning after its reductions, and whether AGAM's global-advantage modulation
remains cleanly separated from the local objective.

## Scope

Included are the V2 method modules, the exact formal configs and launcher, five
narrow integration patches, and focused CPU tests. V3-A focus selection,
generic environment/rollout copies, update-audit code, controller RNG seeding,
models, datasets, retrieval indexes, checkpoints, logs, results, caches, and
credentials are outside this review bundle.

The tests retain their original full-tree imports and relative path checks. Run
them from the pinned full EviSD-URCR layout; their presence here is for review
and does not make this compact repository independently executable.

See [`SOURCE_STATE.md`](SOURCE_STATE.md) for the exact source and verification
record.
