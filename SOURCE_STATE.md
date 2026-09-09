# Source state

- Export date: 2026-09-09 (Asia/Shanghai)
- Method revision: `urcr_v4_unified_typed_r1_final`
- Frozen resolved-method SHA-256:
  `ffbd0c355ed0a9c6334675ac3bcef8bb151233bac3906124c40d64cb698e35d8`
- Umbrella Git HEAD underlying the working tree:
  `b4f1c0be3351166a0f76223f3b15ccae39ffdab9`
- HEAD subject: `Snapshot URCR V2 results and evaluation workflow before V3-A`
- Prior external-review snapshot: `6c2bb19bdd4581e2ce4ac43d8f2c27e9a054d1f9`
- EviSD upstream revision:
  `e72922f891f7e66d773eeb53fa84435f08f8e495`

URCR-V4 was implemented in the active `EviSD-URCR` working tree after the
umbrella V2 snapshot. This review commit is therefore a selected snapshot of
that working tree rather than a claim that V4 already exists in the umbrella
Git HEAD. The V4-owned files are direct byte-for-byte copies. Files under
`patches/` are the exact tracked diffs from the umbrella V2 revision to the
current working-tree integration files, with only the leading
`EviSD-URCR/` path component removed so the patches are readable against the
base tree.

The frozen [`resolved_method.json`](configs/resolved_method.json) records 23
implementation files and their hashes. At export time, every recorded file
matched its stored SHA-256 and byte size. The copied formal resolved config has
SHA-256
`31c6c69ff4d60074027973d5a54c98f638e7cf58ffe9730272d4af5dd7d726bb`.

Verification performed immediately before export:

- all twelve focused V4 test modules passed in the project evaluation
  environment: `82 passed`;
- the resolved-method document validated and all 23 implementation-file
  identities matched;
- the calibration population contains ten pairwise-disjoint batches, the
  original batch 01 is retained, and every batch records zero optimizer
  updates;
- the exact formal config selects the original Base, 300 outer steps, 128
  questions by eight environment rollouts, temperature 1, scorer microbatch 32,
  PPO minibatch 256, PPO microbatch 16 per GPU, and rolling checkpoint cadence
  25;
- all legacy URCR, AGAM, EviSD teacher/modulation, V3 focus, SDAR, and SDL
  switches are disabled in that config.

The inherited `plan07_answer_agam_core.yaml` is present solely to make the
four-GPU diagnostic script reviewable. V4 launch overrides and the resolved
formal config disable its historical AGAM path.

The selected evidence omits raw generated responses and the ten roughly
1.1 MB per-batch calibration files. Their exact source hashes and deterministic
aggregate counts are preserved in
[`artifacts/calibration_batch_summary.json`](artifacts/calibration_batch_summary.json).
The original population manifest and pooled recomputation are included. No
model, checkpoint, dataset, retrieval index, runtime log, secret, or credential
file is included.

The formal 300-step run was active when this snapshot was prepared. Its first
five completed optimizer steps all identify the final resolved-method hash and
are summarized in `artifacts/formal_start_steps_1_5.json`. This is an execution
identity/health prefix, not a frozen performance result; the run continued
after capture.
