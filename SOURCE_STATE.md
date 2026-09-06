# Source state — V3-A A1

- Export date: 2026-09-06 (Asia/Shanghai).
- Active implementation: `EviSD-URCR`, V3-A A1 visible-focus next-decision KL.
- Umbrella source HEAD: `b4f1c0be3351166a0f76223f3b15ccae39ffdab9`.
  This is the pre-V3 V2 snapshot, not a claim that the V3 implementation was
  committed there. V3 comes from the actual working tree on top of this commit.
- EviSD upstream pin: `e72922f891f7e66d773eeb53fa84435f08f8e495`.
- Previous public V2 review: `c7d629047ed1ae40b2b6ad219d15c1b6241789ff`.
- Formal A1 launch's implementation-diff SHA256:
  `80a81f8398797fa5fe9d44cded0cc87eb3ab965a88a753c7253c1d6a3c055591`.
- Formal launch-config SHA256 recorded at launch:
  `92e97164cf6064677ee85e5d56f1e2a3ccaf6a91a540da773da4202c940dc85c`.
  The redacted review JSON is not this original file and must not be expected
  to have the same hash.

The eight training files registered by the formal launch manifest were checked
against the live implementation during this review and matched. Core identity:

| Source | SHA256 |
|---|---|
| `urcr_v3_focus.py` | `98f4ccc52f0b174b9061f8fe4215ad427248c21b7643bb116424c0ae624632d8` |
| `urcr_v3_focus_loss.py` | `a9d8f35497f264482d83b73a1a3e68f44d965e977e32c2545a458adc5f4e0222` |
| Full actor source represented by its patch | `3791f1ee72a544688e7d5274c83e7676a8e36edec283bcc5d1fa12d913a75f36` |
| Full FSDP worker source represented by its patch | `5d4e0b4524e65fe7b12ff5ee15dc5f7aad75f2388d100f2b7eb828016e7e76e1` |
| Full trainer source represented by its patch | `0b97f94d2a5629fddef824b53a5f2df470b38fa948515e35a028bcadc33d1dad` |

New direct exports preserve source bytes, including copyright headers and
original operational paths. The formal-review JSON preserves scientific values
while replacing absolute machine paths and the retriever address with explicit
placeholders. It is an inspection artifact, not an approved runnable config.

At publication, 21 V3 CPU tests passed in 13.96 seconds (GPU visibility disabled),
and all six integration patches passed `git apply --check` against the frozen
local EviSD baseline. The source/test copies and patch reconstruction are also
checked against the active project before push. GPU tests and training were not
rerun to publish this bundle. Historical smoke validation does not establish
sustained focus eligibility or held-out method effectiveness.

Only this separate review repository is committed/pushed. The publication does
not commit or modify the active project's training implementation or experiment
parameters. The source report/index is maintained separately and is not uploaded.
