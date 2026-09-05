# Source state

- Export date: 2026-09-05 (Asia/Shanghai)
- Umbrella repository HEAD: `975194d9b2e9076900d36f19129246beef8fadc3`
- EviSD upstream base: `e72922f891f7e66d773eeb53fa84435f08f8e495`
- Export source: the current `EviSD-URCR` working tree, including the active
  uncommitted V2 edits present at export time.

The source umbrella worktree was intentionally not modified or committed while
creating this review snapshot. The review repository commit fixes the exact
contents supplied to the external reviewer.

Validation at export: the four included test files passed under the local
`evaluation` environment (`53 passed`). All five integration patches passed
`git apply --check` against the pinned local EviSD base.
