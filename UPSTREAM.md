# Upstream provenance

- Upstream project: EviSD — Evidence-Conditioned Self-Distillation for
  Search-Augmented Agents
- Repository: <https://github.com/JiananXie/EviSD>
- Pinned revision: `e72922f891f7e66d773eeb53fa84435f08f8e495`
- Local active implementation: `EviSD-URCR`

The pinned EviSD root did not contain a root-level `LICENSE`, `COPYING`, or
`NOTICE` file when imported locally. This review repository therefore does not
republish the full upstream tree and grants no license to upstream code. It
contains URCR-focused source files plus narrow integration patches for research
review. Public availability of this review bundle does not grant a license to
the upstream EviSD source.

Selected Search action, prompt, environment-manager, memory, and rollout utility
files carry explicit Apache-2.0 copyright/license headers from the original
NTU / verl-agent (GiGPO) authors. Those headers are preserved, and a copy of
the applicable license is supplied at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
Their inclusion is scoped to reviewing A1's action-to-trajectory data flow; it
does not republish the complete upstream environment framework or grant a new
blanket license to the repository's other contents.
