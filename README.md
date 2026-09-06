# URCR Core Review — V3-A / A1

This public repository is a compact code-review snapshot, **not a standalone
training repository**. The active subject is V3-A A1: visible-evidence focus for
the next decision, using a frozen pre-update actor teacher and top-64-plus-tail
forward KL. Legacy URCR/AGAM objectives are disabled in A1.

The earlier V2 review remains pinned at
[`c7d6290`](https://github.com/CrilwaKl/urcr-core-review/tree/c7d629047ed1ae40b2b6ad219d15c1b6241789ff).
Existing V2 modules/configs remain useful for dependency and comparison review;
they are not the active A1 objective. This update does not claim to fix A1.

## Recommended review order

1. [Formal A1 settings](configs/urcr_v3_a_formal_review.json): scientific values
   from the real resolved configuration; machine paths and the retriever address
   are redacted. In particular, the formal coefficient is **1.589512505866416**,
   actor microbatch 16, teacher microbatch 32, and the schedule is 300 steps with
   rolling saves every 25 steps, without an S200 milestone.
2. [Focus selection and view builder](src/verl/trainer/ppo/urcr_v3_focus.py):
   `prepare_focus_candidates`, visible-only copying, stable one-per-trajectory
   selection, payload attachment, selected-only audit, and health guards.
3. [Action/span parser](src/verl/trainer/ppo/urcr_diagnostics.py),
   [environment action projection](src/agent_system/environments/env_package/search/projection.py),
   and [Search prompt](src/agent_system/environments/prompts/search.py): compare
   environment-valid actions with focus-response eligibility. They are distinct
   predicates; environment validity does not require a think block.
4. [Rollout serialization/collection](src/agent_system/multi_turn_rollout/rollout_loop.py),
   [Search environment manager](src/agent_system/environments/env_manager.py),
   and [SearchMemory](src/agent_system/memory/memory.py): trace real prompt IDs,
   response decoding, action validity, and visible history. The included files
   retain generic classes alongside the Search path; this is not a full agent tree.
5. [Grouped KL and sparse consumer](src/verl/trainer/ppo/urcr_v3_focus_loss.py):
   full-vocabulary normalization, explicit tail bucket, causal predictors,
   chunked backward, and the rollout-trajectory denominator.
6. [Actor integration](patches/dp_actor.patch): same-forward PG plus focus,
   disabled destructive inplace log-prob backward, short final mini-batches,
   and `lambda * DP_world / N_s` scaling without another accumulation divisor.
7. [Teacher RPC](patches/fsdp_workers.patch): no-grad scoring on the pre-update
   FSDP actor, frozen targets, version checks, and equal dummy forward counts.
8. [Trainer integration](patches/evisd_ray_trainer.patch): selection before
   adjustment/balance, target consumption checks, and skipped zero-target scoring.
   Also inspect [update-audit ordering](src/verl/trainer/ppo/urcr_update_audit.py)
   and [batch adjustment helpers](src/agent_system/multi_turn_rollout/utils.py).
   Disabling the old update audit can change canonical row ordering and the
   divisibility-copy RNG path; matched question batches do not prove identical
   actor update plumbing versus V2.
9. [CPU tests](tests/test_urcr_v3_focus.py) and
   [opt-in CUDA test](tests/test_urcr_v3_focus_cuda.py): check the actual boundaries
   of mathematical, payload, predictor, and zero-target tests.
10. [Preparation/calibration verifier](scripts/urcr_v3_a_verify.py),
    [launcher](launchers/run_urcr_v3_a_visible_focus.sh),
    [A1 template](configs/urcr_v3_a_visible_focus.yaml), and
    [matched-GRPO template](configs/urcr_v3_a_matched_grpo.yaml).
    The template coefficient 0.10 is a pre-calibration setting, not the formal
    coefficient. A matched-control config is not a completed control experiment.

The main-entry change is in [main_evisd.patch](patches/main_evisd.patch).
The retained [core_algos.patch](patches/core_algos.patch) and
[evisd_teacher.patch](patches/evisd_teacher.patch) expose earlier shared helpers.
Integration patches are relative to the frozen local EviSD baseline identified
in [UPSTREAM.md](UPSTREAM.md), not instructions to patch this compact repository.

## Important review boundary

The action-format gate is a high-priority failure candidate, not a demonstrated
historical root cause. For example, a complete answer-only action is valid for
the environment but fails A1's response gate. The gate checks text before the
action and a lowercase `</think>` substring; it is not a complete validator of
nonempty think content. Reconstructing a later policy's rejected outputs still
requires actual saved response IDs, not synthetic examples.

The health guard accepts zero targets and does not enforce sustained coverage.
`write_focus_views` saves only up to 32 selected rows on configured audit steps;
it cannot reveal rejected outputs, and a zero-target step produces an empty file.
An empty-target KL metric of zero does not establish teacher/student agreement.

## Validation and portability

- At this export, the 21 V3 CPU tests passed in the real project environment,
  with GPU visibility disabled. The CUDA test was **not rerun** for publication.
- All six integration patches passed read-only `git apply --check` against the
  frozen local EviSD baseline.
- New source/test/template/launcher files are byte-identical copies of their
  active-project counterparts. The formal-review JSON differs only in operational
  paths/address. See [SOURCE_STATE.md](SOURCE_STATE.md) for identity and limits.
- Tests import dependencies from the full project. Their presence here does not
  make this bundle installable or independently runnable. Exported launchers retain
  original operational paths for source fidelity; do not execute them as portable
  commands. No training settings were changed to make a review export.

## Intentionally excluded

- Complete EviSD/SDAR/veRL source trees and unrelated upstream packages.
- Models, datasets, retrieval indexes, checkpoints, response dumps, and runtime logs.
- Training-result reports, which are shared separately by the user.
- Credentials, local environment files, authentication helpers, and caches.

The repository commit pins the reviewable source, not model weights or an
evaluation result. A1's correctness or usefulness is not established by publication.
