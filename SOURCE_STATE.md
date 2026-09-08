# Source state

- Export date: 2026-09-08 (Asia/Shanghai)
- V2 umbrella revision: `b4f1c0be3351166a0f76223f3b15ccae39ffdab9`
- Revision subject: `Snapshot URCR V2 results and evaluation workflow before V3-A`
- EviSD upstream base: `e72922f891f7e66d773eeb53fa84435f08f8e495`
- Active source directory at that revision: `EviSD-URCR`

This repository is a selected V2 mechanism snapshot rather than a full source
export. Every included config, launcher, test, and direct Python source file was
compared byte-for-byte with its object at the V2 umbrella revision. Each of the
five integration patches was also checked by blob identity: its old side
matches the pinned local EviSD base and its new side matches the corresponding
file at the V2 revision.

The subset intentionally omits non-mechanism controller setup, including the
extra controller RNG seed, along with V3-A focus code and generic copies of
environment/rollout infrastructure. Those omissions make the bundle easier to
review, but mean it should not be treated as a bit-for-bit runnable checkout.
The inherited data seed in the supplied V2 config remains visible because it is
part of the upstream-style experiment configuration.

Validation for this exported tree:

- the four included V2 test files passed from an isolated checkout of the pinned
  revision with GPU visibility disabled: `53 passed`;
- all five integration patches passed `git apply --check` against the pinned
  EviSD base;
- the final review tree contains 28 files (399,584 bytes), with no model,
  checkpoint, dataset, log, symlink, credential file, or recognized secret
  pattern.
