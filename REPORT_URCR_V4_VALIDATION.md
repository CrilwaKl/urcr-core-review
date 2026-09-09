# URCR-V4 Implementation and Validation Report

## 1. Purpose and claim boundary

This report accompanies the implemented URCR-V4 code for independent method
acceptance. It answers two questions: whether the current code follows the
frozen [`PLAN_URCR_V4.md`](PLAN_URCR_V4.md), and whether the measured numerical
scales and execution checks are sufficient to start the formal run.

The evidence supports an implementation-level PASS. It does not establish that
URCR-V4 improves held-out accuracy, tool use, or sample efficiency. The formal
300-step run has started from the original Base model, but its outcome is not
part of this report. Any `episode/success_rate` below is an on-policy training
diagnostic and must not be read as seven-dataset evaluation accuracy.

The method revision is `urcr_v4_unified_typed_r1_final`; the frozen method file
is [`configs/resolved_method.json`](configs/resolved_method.json), SHA-256
`ffbd0c355ed0a9c6334675ac3bcef8bb151233bac3906124c40d64cb698e35d8`.

## 2. What was implemented

Global GRPO continues to use the original terminal environment reward and the
original group-normalized advantage. V4 never writes its utility into terminal
reward and does not regroup or renormalize the global advantage. It adds a
local actor loss after the ordinary PPO loss:

\[
L = L_{\mathrm{GRPO}} + \lambda(k)L_{\mathrm{local}},
\qquad
\lambda(k)=\lambda_{\max}\min(k/30,1).
\]

For a search action, the frozen model scores a fixed set of answer aliases
under the real observation, an empty observation, and three controls from
other questions. Let

\[
g_{inc}=\Phi(real)-\Phi(empty),\qquad
g_{ctl}=\Phi(real)-\frac{1}{3}\sum_j\Phi(control_j).
\]

The directed gap is the smaller magnitude when both contrasts are positive,
the negative smaller magnitude when both are negative, and zero when their
signs conflict. A continuous dead zone and tanh map produce

\[
U_Q=\begin{cases}
\tanh((g-\delta_U)/s_U), & g>\delta_U,\\
-0.1\tanh((-g-\delta_U)/s_U), & g<-\delta_U,\\
0, & \text{otherwise}.
\end{cases}
\]

A positive utility is also zeroed when all returned passages are already
visible in the actual history. Negative repeated evidence is retained.

For a terminal answer, quality is 1 for binary-EM-correct trajectories and the
best evaluator-consistent word F0.5 over aliases otherwise. The local answer
utility is the residual from the mean inside the same question's eight-rollout
group and the same binary-EM stratum. Singleton and all-equal strata produce
zero. Consequently, the correct stratum has exactly zero answer-local residual;
correctness remains the global GRPO signal.

Responsibility targets the actual sampled query or answer, rather than a gold
answer surrogate. The scorer uses the old-policy full-action log probability
and a position-preserving outgoing-information barrier over the preceding
think. With dependency $D=\ell_{full}-\ell_{masked}$,

\[
\rho=1-\exp\left(-\frac{\max(D-\delta_R,0)}{s_R}\right).
\]

Positive per-chunk dependencies above the same dead zone are squared and
normalized across at most six chunks. For action utility $U$, the action
coefficient is $\alpha U$ and think-chunk coefficient is
$\alpha\rho w_cU$, with `alpha_Q=1` and `alpha_A=0.25`. Query and think
credits are additive. Search action-plus-think absolute mass is capped at 2 per
trajectory; including the answer budget, the checked total bound is 2.5.

Each nonempty local span uses one mean over its tokens and the standard signed
clipped PPO surrogate. Span contributions are summed and divided by the number
of original trajectories. The minibatch estimator explicitly restores this
outer trajectory mean under row minibatches and FSDP's averaged gradients.
Credits are frozen before PPO shuffling, and the local and global losses share
the same actor forward and optimizer step.

The online V4 path accepts visible history, sampled think/action, returned
observation, answer aliases, and terminal binary outcome. Evidence/supporting
fact metadata is neither copied into V4 turn records nor scored, and recursive
guards reject it if present. The formal config disables all old method paths.

## 3. Frozen empirical parameters

| Parameter | Frozen value | Estimation rule |
|---|---:|---|
| `delta_U` | 0.500000 | `max(3 × numeric P95, clip(control-pair P75, .05, .5))` |
| `s_U` | 2.000000 | P75 active excess, clipped to `[.25, 2]` |
| `delta_R^Q` | 0.0810656814 | `max(.01, 3 × backend-alignment P95)` |
| `s_R^Q` | 0.4315986055 | median positive excess, clipped to `[.1, 2]` |
| `delta_R^A` | 0.1551302329 | `max(.01, 3 × backend-alignment P95)` |
| `s_R^A` | 0.8357414350 | median positive excess, clipped to `[.1, 2]` |
| `lambda_max` | 0.0467186302 | S150 gradient rule, retained by Base shadow |

Other fixed design values are three controls, negative utility scale 0.1,
answer F-beta 0.5, chunk power 2, at most six chunks, answer alpha 0.25, search
trajectory cap 2, and 30 outer steps of linear local-loss warm-up. There was no
hyperparameter search against held-out test results.

## 4. Ten-batch frozen calibration population

The diagnostic model was the identity-checked pure-GRPO S150 export. The
population consists of ten pairwise-disjoint batches. Each batch has 64
HotpotQA plus 64 NQ training questions, eight independent rollouts per question,
temperature 1, the same prompt/retriever/action schema, and no optimizer
update. Batch 01 is the original 128-question diagnostic and was preserved and
included. Totals are 1,280 unique questions and 10,240 trajectories.

The expensive calibration scorer used a fixed deterministic subset per batch:
up to 256 valid search anchors, 64 query-responsibility anchors, and 64
answer-responsibility anchors. Across all batches this produced 2,526 scored
search anchors and 1,280 responsibility anchors. All eligible terminal answers
were still used for answer-residual calibration. Thus 10,240 is the rollout
population size, while the scorer-based empirical-scale sample sizes are the
anchor counts just stated.

The final values were recomputed by concatenating all raw empirical rows and
reapplying the unchanged formulas. They are not averages of independently
calibrated batch values. The exact pooled output and population identity are in
[`pooled_calibration_10_batches.json`](artifacts/pooled_calibration_10_batches.json)
and [`population_manifest.json`](artifacts/population_manifest.json).

### 4.1 Search contrast signs and resulting utility

Across 2,526 scored search anchors, the raw two-contrast signs were:

| Raw relation | Count | Directed-agreement result |
|---|---:|---|
| both positive | 1,246 | positive smaller magnitude |
| both negative | 851 | negative smaller magnitude |
| sign conflict | 426 | exact zero |
| at least one exact zero | 3 | exact zero |

After pooled dead-zone/tanh recomputation and the positive-repeat rule, utility
counts were 804 positive, 648 negative, and 1,074 zero. Recomputed zero reasons
were 426 sign conflicts, 642 within the dead zone, and 6 positive visible exact
repeats. The artifact records 139 visible-repeat anchors in total; only the six
that would otherwise have positive utility are zeroed. Thus the raw `++` and
`--` counts should not equal the final positive and negative utility counts.
The smaller magnitude allowed on the negative side is intentional because its
fixed multiplier is 0.1.

Per-batch values below use each batch's provisional scales. They demonstrate
coverage; final responsibility counts are recomputed under the pooled scales.

|Batch|Turns|Anchors|++|--|Conflict|Edge|U+|U-|U0|A+|A-|A0|rhoQ+|rhoA+|Chunks+|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|2536|252|142|74|35|1|84|59|109|59|33|909|37|28|64|
|2|2551|256|125|89|42|0|81|66|109|42|41|911|42|37|60|
|3|2527|254|119|93|42|0|73|71|110|88|84|819|35|29|64|
|4|2500|253|128|69|56|0|76|53|124|53|72|871|36|40|66|
|5|2495|253|111|108|34|0|73|90|90|39|83|885|37|38|52|
|6|2458|245|86|112|47|0|58|78|109|47|38|909|38|25|56|
|7|2527|254|142|79|33|0|95|65|94|58|50|889|39|34|66|
|8|2523|253|129|72|52|0|88|53|112|60|59|875|39|31|61|
|9|2479|250|133|75|40|2|92|55|103|47|31|908|36|39|59|
|10|2546|256|131|80|45|0|84|58|114|54|62|888|35|36|58|

The machine-readable table, source hashes, per-batch costs, provisional scales,
and pooled recomputation are in
[`calibration_batch_summary.json`](artifacts/calibration_batch_summary.json).

### 4.2 Answer residual behavior

There were 9,964 eligible terminal answers: 547 positive residuals, 553
negative residuals, and 8,864 zeros across 161 nonzero strata. The maximum
absolute sum of residuals within any stratum was
`6.106226635438361e-16`, consistent with floating-point zero. The nearly
balanced positive/negative counts and the zero-sum check are expected from
within-stratum mean subtraction; they are an algebraic check, not evidence of
better answers. The high zero count follows from correct strata having constant
quality 1 and from many incorrect answers having equal zero token overlap.

### 4.3 Responsibility behavior

The pooled calibration used 1,280 anchors: 640 search and 640 answer. Under the
final scales, 366 search anchors and 331 answer anchors had positive rho; 588
anchors had positive chunk-routing mass. Same-shape repeat numerical P95 was
zero for both action types.

The old-policy full score and the plain scorer are mathematically intended to
represent the same full view but differ slightly under their actual BF16
execution paths. Their pooled absolute-difference P95 was 0.0270218938 for
queries and 0.0517100776 for answers. The frozen rule includes this backend
alignment error as numerical error, so three times those values determines the
two responsibility dead zones. Whether treating backend mismatch as part of a
scientific dead zone is preferable to enforcing a single scoring backend is an
explicit review question below.

### 4.4 Prefix stability

|Batches|`delta_U`|`s_U`|`delta_R^Q`|`s_R^Q`|`delta_R^A`|`s_R^A`|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.500000|2.000000|0.067861|0.395223|0.098880|0.640943|
|2|0.500000|2.000000|0.080681|0.439483|0.126451|0.831381|
|3|0.500000|2.000000|0.080463|0.444916|0.137209|0.887708|
|4|0.500000|2.000000|0.080245|0.444628|0.152128|0.889527|
|5|0.500000|2.000000|0.083005|0.433563|0.164481|1.064624|
|6|0.500000|2.000000|0.085351|0.413818|0.162838|0.950458|
|7|0.500000|2.000000|0.080681|0.418489|0.166857|0.900362|
|8|0.500000|2.000000|0.080463|0.431893|0.169115|0.856751|
|9|0.500000|2.000000|0.080245|0.413858|0.168728|0.822319|
|10|0.500000|2.000000|0.081066|0.431599|0.155130|0.835741|

Over prefixes 8–10, range divided by the final absolute value was 0% for
`delta_U` and `s_U`, 1.01% for `delta_R^Q`, 4.18% for `s_R^Q`, 9.01% for
`delta_R^A`, and 4.12% for `s_R^A`. No monotonic drift is visible, but the
answer responsibility dead zone is the least stable of the six values. This
9% residual variation is retained as a limitation for external judgment rather
than being converted into a new calibration rule.

The apparent 0% stability of `delta_U` and `s_U` should not be overinterpreted.
The pooled control-pair P75 was 1.209458, so `delta_U` reached its configured
upper clip of 0.5; `s_U` likewise reached its upper clip of 2. Their constant
prefix values partly reflect saturation of the predeclared bounds, rather than
an unconstrained estimator that independently converged to the same number.

### 4.5 Measured calibration cost

Median per 1,024-trajectory batch was 121.293 s for rollout, 28.067 s for old
log probabilities, and 53.835 s for the V4 calibration stage. Within the V4
stage, search scoring was 27.573 s and responsibility scoring 13.516 s. Across
ten batches, the corresponding sums were 1,210.632 s, 280.693 s, 541.342 s,
279.825 s, and 136.059 s. These are measured phase timers rather than an
independent end-to-end wall-clock benchmark.

## 5. Scorer, information-barrier, and span checks

The following checks address the highest-risk implementation details:

| Check | Measured result | Interpretation |
|---|---|---|
| CPU tiny-Qwen SDPA barrier | masked mutation difference `0`; unmasked difference `0.0190101` | the outgoing-information barrier blocks the mutated think while the plain model remains sensitive |
| S150 GPU fixed-shape repeat | configured masked and plain repeat differences `0`; cross-row mutation differences `0` | repeated scoring in the configured shape is deterministic and rows do not leak into one another |
| S150 GPU future mutation | masked difference `0`; plain difference `2.5598952` | future information is blocked only in the masked view |
| BF16 shape change | pair-vs-single max differences `0.101874` masked and `0.134913` plain | scores are not bit-equivalent across batch shapes |
| Four-GPU FSDP scorer | 128 items; microbatches 8/16/32; fixed-32 repeat difference `0`; one RPC and `0.9348 s` at 32 | formal scoring fixes microbatch 32; cross-shape values are not mixed |
| Four-GPU pipeline | 4 questions × 8 = 32 trajectories, 128 turns; 59 valid searches, 32 valid answers; one search RPC and one responsibility RPC; `4.171 s` | end-to-end record/request/score/route/tensor path executes on four GPUs without an optimizer update |

The four-GPU pipeline smoke produced 4 positive, 8 negative, and 79 zero
action utilities with 27 nonzero local spans; `metadata_present=false`. The
cross-shape FSDP comparison differed by up to about 0.625 between microbatch 8
and 32, so the PASS is deliberately limited to fixed-shape reproducibility and
mask semantics. It does not claim BF16 batch-shape invariance.

A real Qwen tokenizer audit over 1,588 historical turns found 1,079 valid
searches and 507 valid answers. All remained valid responsibility targets and
all had routable think. Boundary overlap occurred for 177 searches and 199
answers; 26 answers were boundary-only for the local action span. Boundary
tokens remain global-GRPO-only while responsibility can still target all sampled
action-overlap tokens, so a boundary condition does not discard the think or
the action. Pure local action content remained available for all 1,079 searches
and 481 answers.

The implementation also handles the tokenizer's rare Unicode/noncontiguous
offset case conservatively. Fast-tokenizer offsets are accepted only when
re-encoding exactly reproduces the sampled IDs. If a contiguous action target
cannot be represented, only think responsibility is zeroed; action validity,
action utility, and global credit remain. This avoids the whole-action gates
that previously discarded otherwise valid think/action paths.

## 6. Gradient-scale calibration

No optimizer step was executed during gradient calibration. Two nonzero S150
subbatches were normalized to one 256-row actor minibatch, used 32 original
trajectories each, and used actor microbatch 16 per GPU across four GPUs.

|Subbatch|Global L2|Local L2|Global/local|Cosine|Nonfinite global/local|Peak allocated GiB|Wall s|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|3.038684|5.249981|0.578799|0.059079|0 / 0|43.964|54.297|
|2|3.103689|8.728681|0.355574|0.096580|0 / 0|45.750|53.193|

The frozen rule uses
`clip(0.1 × median(global/local), 1e-4, 0.2)` and then the
`0.3 × min(global/local)` ceiling. It yielded
`lambda_0=lambda_max=0.0467186302`. At that coefficient, the maximum scaled
local/global norm ratio was 0.131389, below the 0.3 ceiling.

The calibration artifact's pipeline coverage field named
`urcr_v4/lambda_effective=0.033333...` belongs to the diagnostic method object.
The gradient-stat routine explicitly backpropagates the full unscaled local
objective and does not multiply by that field; the resolved artifact records
`warmup_applied=false` for this calibration. This distinction is included to
prevent an external reviewer from mistaking a coverage metric for the gradient
scale used in the formula.

The Base shadow then used the original Qwen2.5-3B-Instruct initialization, a
full 1,024-trajectory local pipeline, and no optimizer update. Coverage was
2,514 turns; action utility counts were 267 positive, 213 negative, and 2,033
zero; responsibility had 480 bundles and 347 positive values; 920 local spans
were active. On a 32-trajectory gradient subbatch, global L2 was 3.284125,
local L2 3.664941, global/local 0.896092, cosine 0.071854, and both nonfinite
counts were zero. The final scaled local/global ratio was 0.052136, so the Base
ceiling did not lower lambda.

## 7. Optimizer smoke, final-hash prefix, and formal configuration

Three optimizer steps completed with 1,024 original trajectories each. Steps 1
and 2 started from the original Base. Step 3 resumed the S2 smoke checkpoint
only to validate the resume path, then the smoke was stopped. The formal run
does not inherit this scientific state; it starts again from the original Base.

|Step|Turns|Action U + / - / 0|rho + Q / A|Local spans|Abs. coefficient sum|Grad norm|V4 pipeline s|Actor update s|Total s|Train success|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|2520|274 / 233 / 2012|178 / 203|985|163.535|1.712|122.945|146.451|470.465|0.293|
|2|2582|257 / 253 / 2066|199 / 192|998|144.782|1.664|127.622|147.597|510.634|0.259|
|3|2409|236 / 211 / 1960|143 / 199|876|126.889|1.622|107.123|137.725|544.011|0.272|

These three smoke components identify resolved-method SHA-256
`e137caf418f6dd39b23ad02b7e63dcccdcf3697c97e967cdb4ead763acfec022`,
which was superseded before formal launch. The earlier resolved JSON was
overwritten, so an exact old-vs-final document diff cannot be reconstructed.
The retained S1/S2 smoke config and final formal config differ only in the
resolved hash, run/output/checkpoint paths, save cadence, and stop boundary;
all visible scientific configuration fields are identical. The smoke is
therefore supporting execution evidence, not a claim of byte-identical final
method identity.

The step-2 total includes 23.139 s for the one-off full checkpoint written to
test resume. The old EviSD teacher scored zero rows at all three steps. The
health checks required 1,024 original trajectories, finite core metrics, a
positive actor gradient norm, active lambda, and positive learning rate after
the upstream first-step warm-up behavior. Nonzero local spans and signed
utilities were observed separately at every step. No S4/S5 smoke was run.
The compact source-linked values are in
[`training_smoke_steps_1_3.json`](artifacts/training_smoke_steps_1_3.json).

The formal run supplies the byte-identity check that the smoke cannot: its
first five saved component files all identify the final `ffbd0c...` resolved
method and the original Base initialization. Each completed optimizer step had
1,024 trajectories, finite actor gradients, active signed utility and local
spans, and zero rows scored by the legacy EviSD teacher.

|Formal step|Turns|Action U + / - / 0|rho + Q / A|Local spans|Abs. coefficient sum|Grad norm|V4 pipeline s|Total s|Train success|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|2520|272 / 238 / 2009|178 / 203|982|160.027|1.706|123.075|469.957|0.293|
|2|2582|253 / 246 / 2077|190 / 192|978|142.870|1.658|126.418|473.232|0.259|
|3|2476|262 / 253 / 1957|158 / 234|975|131.952|1.707|116.302|455.762|0.260|
|4|2380|272 / 225 / 1880|156 / 215|980|142.881|1.616|106.882|427.502|0.317|
|5|2443|263 / 280 / 1895|186 / 238|1075|148.121|1.769|115.506|449.785|0.245|

This finite prefix is recorded in
[`formal_start_steps_1_5.json`](artifacts/formal_start_steps_1_5.json). The run
continued after capture, and the success column remains a training diagnostic.

The exact formal config selects four GPUs, 128 questions × 8 environment
rollouts, temperature 1, four environment steps, top-3 retrieval, GRPO with
standard-deviation normalization, one PPO epoch, minibatch 256, actor
microbatch 16 per GPU, scorer microbatch 32 per GPU, AdamW LR `1e-6`, 30-step
LR warm-up, 300 outer steps, and checkpoint cadence 25. It contains the
inherited `data.seed=1` and `env.seed=0`; the rollout config contains no extra
seed. Formal `resume_mode=disable` and model path is the original Base. V1/V2,
AGAM, EviSD, V3, SDAR, and SDL switches are false.

## 8. Verification status and limitations

All twelve focused V4 test modules passed immediately before export:
`82 passed`. The frozen document's 23 file hashes all matched the live source.
Structural and semantic checks passed, the ten calibration batches passed, both
gradient calibrations passed, and the three-step optimizer path remained
finite with nonzero V4 coverage. The final resolved hash additionally completed
the captured formal S1–S5 prefix with finite gradients and nonzero coverage.

Material limitations retained for review are:

- responsibility is a promoting-dependency estimate for the sampled action,
  not a causal effect over resampled actions and not an unbiased Q-value;
- the answer-local proxy is lexical F0.5 within the incorrect stratum and can
  reward a relatively better but still wrong answer;
- the negative search utility is intentionally capped to one tenth the positive
  magnitude, so sign counts do not imply symmetric gradient mass;
- scorer values depend on BF16 batch shape; calibration and formal execution
  control this by fixing microbatch 32 rather than by proving shape invariance;
- `delta_R^A` retained 9.01% last-three-prefix range;
- the full online V4 scorer added roughly 107–128 s per smoke step in the
  observed three-step sample;
- this report validates implementation and numerical viability only; it has no
  completed formal-run accuracy or efficiency result.

## 9. Questions for independent acceptance

1. Does the implemented data flow preserve the Plan's separation between
   terminal GRPO and additive local credit, with no route for evidence metadata
   into rollout, scoring, routing, or update?
2. Is the directed-agreement implementation correct for the observed 1,246
   both-positive, 851 both-negative, 426 conflict, and 3 zero-edge anchors, and
   is the 0.1 negative-side multiplier defensible?
3. Are 1,280 disjoint questions and 10,240 trajectories sufficient to freeze
   these empirical scales, especially given the 9.01% prefix range of
   `delta_R^A`?
4. Should old-policy-vs-plain backend alignment error determine the
   responsibility dead zone, or should the implementation instead force the
   full and masked views through one identical backend?
5. Is the span-mean, original-trajectory denominator and FSDP minibatch scaling
   in `urcr_v4_local_objective.py` and `dp_actor.patch` mathematically correct?
6. Is the S150 lambda derivation and the Base shadow ceiling applied to the
   intended full-strength local gradient without accidental warm-up scaling?
7. Is fixing scorer microbatch 32 an adequate response to the measured BF16
   batch-shape dependence, or does the scientific definition require a stronger
   invariance guarantee?
8. Do the boundary-token and rare noncontiguous-offset fallbacks preserve the
   intended distinction between action-local credit, think responsibility, and
   global GRPO credit?
9. Is the measured scoring overhead an acceptable consequence of the specified
   counterfactual views, or is there a semantics-preserving batching opportunity
   visible in the current implementation?
10. Can any legacy V1/V2/V3/EviSD path still affect the formal run despite the
    explicit disabled switches and V4 health checks?
