# Design: WallBallDepthCurriculum — test serve-landing alignment

Status: experimental implementation, revised 2026-07-24 after artifact and
log audit. The branch preserves the fixed-origin production recipe, adds a
separately named aligned treatment, and contains paired validation
instrumentation. The treatment is not yet certified for a full 6M retrain.
This document reviews the run-2 sliding-ladder pilot
(`WallBallDepthCurriculum`, cold-start SAC, 6M-step budget) and defines a
staged test of serve-origin alignment. It does **not** yet establish alignment
as the root cause of the run's stage-1 stall or authorize a full 6M retrain.
The design being reviewed is `design_wall_ball_depth_curriculum.md`; the prior
run's review is `wall_ball_depth_curriculum_run1_review.md`.

Evidence base:

- 13 Drive milestone checkpoints at every 250k steps from 250k to 3.25M,
  each with a model, `vecnormalize.pkl`, rollout video, and per-step rollout
  CSV;
- reproduced rollouts of the 1.75M, 2.75M, and 3.25M models;
- fixed-geometry serve-speed and matched-stage sweeps;
- parked-paddle serve-landing measurements and zero-action/controller
  prototypes; and
- the separately recovered curriculum history, which records promotion from
  stage 0 to stage 1 at 1.425M and no promotion to stage 2 through roughly
  3.4M.

The review-log archive omitted the curriculum history and matched gate-eval
artifacts. They must be included with the next experiment so the stage
timeline is auditable from the same bundle.

## Decision summary

- **Confirmed geometry:** with `serve_start_x=1.0` at every stage, the mean
  first-bounce position moves much less than the paddle start. The mean
  landing-minus-start gap is about +0.30 m at stage 1 and grows to +1.34 m at
  stage 4.
- **Candidate mechanism, not root cause:** the run actually trained at stage 1.
  Therefore the stage-1 gap is the only alignment mismatch that can plausibly
  contribute to the observed promotion stall. The larger stage-2-to-stage-4
  gaps were never encountered in training and cannot have caused failure to
  leave stage 1; they matter only to future stages and final-stage transfer.
- **The first prototype was not a passed pre-flight:** it aligned the *mean*
  landing and preserved the zero-action score, but the unchanged oracle fell
  below the existing 90% feasibility threshold at stage 3 (88%) and stage 4
  (66%). The crude controller was mixed and is coupled to serve placement.
- **Implemented for testing:** `WallBallDepthCurriculum` remains the
  fixed-origin control and `WallBallDepthCurriculumAligned` is the separately
  named treatment. Stage 0 remains at `serve_start_x=1.0`; candidate origins
  are wired into treatment stages 1–4 and its final evaluation. The sweep now
  selects either ladder, records exact per-seed controller and landing
  outcomes, and applies a blocking distributional landing contract to the
  candidate.
- **Held-out certification remains blocked:** on seeds 1000–1199, every
  aligned landing-distribution check passed, but its oracle reached the
  existing ≥90% feasibility bar only at stages 0–2. Stages 3 and 4 scored
  84.5% and 60%. The paired fixed-origin baseline also scored only 89% at
  stage 4, one point below the bar.
- **Before a 6M pilot:** run a paired stage-1 baseline-versus-aligned training
  experiment. Promotion past stage 1 is the primary outcome. Advance to a
  full aligned ladder only if that focused experiment supports the mechanism.

## What the milestone artifacts measure

The per-checkpoint
`wall_ball_depth_curriculum_sac_<steps>.csv` / `.mp4` artifacts are
three-episode rollouts at the fixed stage-4 geometry: fence
`(-4.7, -3.0)`, paddle start `-3.9`, and serve speed `7.0`. They are written
through the recipe's final-evaluation overrides, not the current
performance-gate stage. The stream therefore measures final-geometry
**transfer**, not matched-stage gate performance.

The callback reused the same three milestone seeds at every checkpoint.
Consequently, the pooled set is 13 policies evaluated on three repeated seed
slots, not 39 independent samples from one policy. A single checkpoint mean
is quantized in thirds and the pooled percentages are descriptive summaries,
not confidence-calibrated estimates of one policy's level.

At 3.25M, the milestone episodes contain return counts `[1, 0, 0]`, for a mean
of 0.33. A separate 30-episode reroll of the same checkpoint at stage 4 gives
0.63 completed returns and a 63% rate of at least one return. That difference
is compatible with ordinary n=3 sampling noise; it should not be described as
proof that either estimate is the policy's true performance. Future milestone
rollouts must use more episodes and rotate or explicitly record seeds.

Across the 39 milestone rows:

- completed returns total 14, for a descriptive mean of 0.359;
- 31 episodes terminate on double bounce and 8 on out-of-bounds, with no
  timeouts;
- 36 legal hits occur across 33 of 39 episodes;
- all 36 legal hits are post-bounce and there are no opening volleys; and
- the legal-hit-weighted contact position is about `-3.43`.

Thus “100% post-bounce” applies to the 36 observed legal hits. It must not be
read as a 100% episode hit rate; six episodes have no legal hit. The most
common failure bucket is hit-but-no-completed-return (20/39), followed by an
episode with at least one completed return (13/39), then a whiff (6/39).
Those counts show both first-return conversion and later recovery problems;
they do not by themselves identify post-return recovery as the sole binding
failure.

## Observed policy sweeps

### Serve speed at fixed stage-4 geometry

Holding the stage-4 fence and paddle start fixed while varying serve speed
for the 3.25M model gives:

| serve | completed returns | ≥1 return | hit rate | contact x |
|---:|---:|---:|---:|---:|
| 5.2 | 0.00 | 0% | 100% | −3.02 |
| 5.5 | 0.00 | 0% | 100% | −3.04 |
| 6.0 | 0.10 | 10% | 100% | −3.11 |
| 6.5 | 0.40 | 40% | 100% | −3.23 |
| 7.0 | 0.63 | 63% | 100% | −3.31 |

The same monotone pattern appears at the 1.75M and 2.75M checkpoints. This
supports the narrow statement that the tested policies perform better with
the faster serve under the fixed deep geometry. Because those policies were
trained under the existing serve schedule, it does not prove that `7.0` is
universally required, that another policy could not learn a slower serve, or
that serve speed is free of distribution shift.

The stage-1 alignment experiment therefore keeps serve speed unchanged. It
tests origin placement without co-varying pace.

### Matched full-stage configurations

Completed returns from 30 episodes per cell:

| stage (start / serve) | 1.75M | 2.75M | 3.25M |
|:---|---:|---:|---:|
| 0 (−1.6 / 5.2) | 1.53 | 2.30 | 2.33 |
| 1 (−2.1 / 5.5) | 1.50 | 2.03 | 2.20 |
| 2 (−2.7 / 6.0) | 1.27 | 1.30 | 1.20 |
| 3 (−3.3 / 6.5) | 0.77 | 0.77 | 0.97 |
| 4 (−3.9 / 7.0) | 0.63 | 0.60 | 0.63 |

Performance degrades monotonically with the matched-stage index, and
improvement across checkpoints is concentrated at stages 0–1. This is
consistent with the recovered timeline showing training at stage 1. It is
not a causal isolation of serve alignment: fence, paddle start, serve speed,
and training exposure all change together.

The matched-stage means also vary by held-out seed batch. For the same 3.25M
model, stage-0 means of 2.33, 2.80, and 2.90 were observed in separate
25–30-episode batches; stage-1 means were 2.20, 2.36, and 2.83. A single
2.33 estimate below the 3.0 gate cannot establish the promotion history.
The recovered timeline, rather than this inference, is the source of truth
for the stage-1 stall.

## Candidate mechanism: landing relative to paddle start

With the paddle parked away from the serve, mean ball x at the first floor
bounce is:

| stage | paddle start | serve speed | current mean landing | mean gap (landing − start) |
|---:|---:|---:|---:|---:|
| 0 | −1.60 | 5.2 | −1.65 | −0.05 |
| 1 | −2.10 | 5.5 | −1.80 | +0.30 |
| 2 | −2.70 | 6.0 | −2.06 | +0.64 |
| 3 | −3.30 | 6.5 | −2.31 | +0.99 |
| 4 | −3.90 | 7.0 | −2.56 | +1.34 |

This is a policy-independent geometry measurement. The paddle start recedes
2.3 m over the ladder while the mean landing recedes only about 0.9 m because
`serve_start_x` remains fixed at `1.0`.

The measurement establishes the mismatch, but the causal interpretation
needs two corrections:

1. The current stall is at stage 1, where the measured mean mismatch is about
   0.30 m. Stage-2-to-stage-4 gaps cannot explain a stage-1 promotion failure
   because the run never trained there.
2. The stage-4 landing gap is not paddle travel. The ball continues toward the
   paddle after bouncing. With start `-3.9`, observed contact around
   `-3.31` to `-3.38` implies approximately 0.5–0.6 m of forward movement,
   not 1.35 m.

A plausible mechanism remains: changing the bounce location may alter
interception timing, pose, and hit-to-return conversion at stage 1, and the
effect could grow at later stages. The existing data show association, not
that this mechanism is necessary or sufficient. In particular, stage 0 is
already approximately aligned and still does not reliably clear the 3.0
promotion gate.

Applying aligned origins to a policy trained on the current ladder produced
mixed or worse transfer results. That is compatible with controller/policy
coupling to the old serve, but it is not affirmative evidence that the policy
learned a specific forward-charge strategy or that a newly trained policy
will benefit. Only a training A/B can answer that question.

## Candidate serve origins

Use stage 0 as an unchanged control. Its current mean gap is small and it is
inside the proposed tolerance, so changing `1.0` to `1.04` would introduce an
unnecessary difference before the causal test.

| stage | start | serve speed | current origin | candidate origin | candidate mean landing |
|---:|---:|---:|---:|---:|---:|
| 0 | −1.60 | 5.2 | 1.00 | **1.00** | −1.65 |
| 1 | −2.10 | 5.5 | 1.00 | **0.69** | −2.11 |
| 2 | −2.70 | 6.0 | 1.00 | **0.34** | −2.72 |
| 3 | −3.30 | 6.5 | 1.00 | **−0.01** | −3.32 |
| 4 | −3.90 | 7.0 | 1.00 | **−0.35** | −3.91 |

These are calibration candidates, not final certified values. The reported
candidate landings are sample means. “Within ±0.02 m” describes the distance
between each reported mean and the corresponding start; it does **not** mean
that every serve under `serve_speed_jitter=0.5` lands within ±0.02 m.

## What the 2026-07-24 prototype did and did not pass

The prototype used candidate origins at stages 1–4 and `1.04` at stage 0. It
ran mean landing, zero-action, crude-controller, and existing-oracle checks:

| stage | mean landing (current → prototype) | zero-action mean | crude ≥2 (current → prototype) | oracle ≥2 (current → prototype) |
|---:|---:|---:|---:|---:|
| 0 | −1.65 → −1.61 | 0.00 | 85% → 74% | 99% → 98% |
| 1 | −1.80 → −2.11 | 0.00 | 74% → 95% | 99% → 100% |
| 2 | −2.06 → −2.72 | 0.00 | 65% → 90% | 95% → 94% |
| 3 | −2.31 → −3.32 | 0.00 | 52% → 51% | 95% → 88% |
| 4 | −2.56 → −3.91 | 0.00 | 11% → 22% | 98% → 66% |

The correct reading is:

- **Mean landing check passed.** The measured mean moved to within 0.02 m of
  the start at each prototype stage. Landing standard deviations, quantiles,
  and tail failures were not retained in the log, so distributional alignment
  remains unverified.
- **Zero-action check passed.** A zero-action policy scored 0 on both ladders.
- **Crude-controller evidence is mixed.** It improves at stages 1, 2, and 4
  but declines at stages 0 and 3. The observed stage-4 rate doubles from
  11% to 22%, but this is 80 episodes with no retained paired per-episode
  outcomes and should be treated as suggestive. The controller sweeps across
  the fence and is coupled to serve placement.
- **Full feasibility failed.** The existing contract requires the oracle to
  complete at least two returns on at least 90% of serves. The prototype
  scores 88% at stage 3 and 66% at stage 4, so it did not pass the full
  pre-flight.
- **Why the oracle failed is not yet established.** Stale charge timing is a
  reasonable hypothesis, but it must be tested by deriving a placement-fair
  oracle on calibration seeds and evaluating it on held-out seeds. It should
  not be declared a calibration artifact in advance.

No production ladder or 6M pilot should proceed from this partial prototype
alone.

## Implemented held-out sweep

The revised sweep measures the exact first-floor-contact substep rather than
the ball position on the following environment frame, holds the paddle at its
configured start, accepts explicit ladder and held-out seed selections, and
persists the complete per-seed controller and landing records as JSON. The
report also records the invocation, UTC timestamp, Git/source fingerprints,
and any declared calibration seed range.

The paired runs used 200 episodes per cell and common seeds 1000–1199. The
stage-0 configuration, controller rows, and landing samples are exactly equal
across the two artifacts.

| stage | mean landing offset (baseline → aligned) | aligned std | aligned within ±0.30 m | crude ≥2 (baseline → aligned) | oracle ≥2 (baseline → aligned) |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.001 → −0.001 | 0.155 | 100.0% | 82.0% → 82.0% | 95.5% → 95.5% |
| 1 | +0.349 → +0.039 | 0.155 | 100.0% | 72.0% → 94.0% | 95.5% → 96.0% |
| 2 | +0.698 → +0.038 | 0.156 | 100.0% | 68.5% → 89.5% | 94.0% → 94.0% |
| 3 | +1.048 → +0.038 | 0.156 | 100.0% | 56.0% → 46.5% | 92.5% → **84.5%** |
| 4 | +1.397 → +0.047 | 0.156 | 99.5% | 16.0% → 12.0% | **89.0%** → **60.0%** |

The aligned geometry portion passes with raw distribution evidence, not only
a close sample mean. Full certification still fails: the aligned oracle
misses the existing feasibility contract at stages 3–4, with Wilson 95%
intervals of 78.8–88.9% and 53.1–66.5%. The baseline also misses at stage 4
by one percentage point (178/200; Wilson 83.9–92.6%), showing that the
controller threshold is seed-sensitive near the boundary. The paired results
are mixed: the aligned crude point estimates are higher at stages 1–2 and
lower at stages 3–4. Those marginal percentages are descriptive; the
artifacts do not yet report paired-contrast uncertainty. The aligned oracle
declines at stages 3–4 are much larger, but they remain controller-specific
mechanical evidence rather than evidence about SAC learning.

An exploratory `oracle_charge_gap` timing search on separate seeds 0–199 did
not restore the 90% bar, especially at stage 4. It produced no standalone raw
calibration artifact; that declared range appears in the aligned report. The
baseline's older oracle has no retained calibration-range provenance. These
limitations are now explicit rather than inferred. The remaining issue should
be treated as controller redesign or task recalibration rather than presumed
to be a one-parameter timing fix.

Seeds 1000–1199 are now consumed diagnostic data. Any controller redesign
that responds to these outcomes must use them as calibration evidence, not
reuse them for certification. Pre-reserve a fresh final set (2000–2199) before
the next controller iteration and do not inspect it until parameters are
frozen.

Both reports were produced from a dirty working tree. They fingerprint the
script and tracked diff, so they are useful diagnostics, but the hashes do not
contain the patch needed to reconstruct that state independently. A final
certification artifact must run from a clean committed revision (or archive
the exact patch and relevant dependency environment).

## 2026-07-25 replication on calibration seeds 0–199

A second paired sweep was run at 200 episodes per cell on seeds **0–199**,
from a **clean committed revision** (`89dbe61`, `git_dirty: false`, empty
tracked-diff hash) — the first sweep in this campaign to satisfy that part
of the reproducibility requirement stated below. The reports regenerate
from the recorded revision, command, and `seed_range` field, so the 2.5 MB
raw JSONs are deliberately not vendored. Item 5 of the contract is still
unmet, however: both artifacts record `calibration: null`, because the
tool refuses a calibration range that overlaps the held-out range and
here the two coincide.

**These are calibration seeds, not certification.** Range 0–199 is the same
range the exploratory `oracle_charge_gap` timing search used. Results here
are therefore biased *in the oracle's favour* and must not be quoted as
held-out feasibility.

Oracle ≥2-return rate, with Wilson 95% intervals (successes/200):

| stage | aligned (0–199) | baseline (0–199) | aligned (1000–1199) | baseline (1000–1199) |
|---:|---:|---:|---:|---:|
| 0 | 94.0% [89.8, 96.5] | 94.0% [89.8, 96.5] | 95.5% | 95.5% |
| 1 | 98.5% [95.7, 99.5] | 96.0% [92.3, 98.0] | 96.0% | 95.5% |
| 2 | 93.5% [89.2, 96.2] | 92.0% [87.4, 95.0] | 94.0% | 94.0% |
| 3 | **85.5%** [80.0, 89.7] | 94.0% [89.8, 96.5] | **84.5%** | 92.5% |
| 4 | **63.5%** [56.6, 69.9] | 95.0% [91.0, 97.3] | **60.0%** | **89.0%** |

Stage 0 is identical across ladders (188/200 both), confirming the negative
control holds. The aligned landing contract passes again: mean offsets
0.001–0.049 m against the 0.10 m bar, with 100% of individual landings
inside ±0.30 m at stages 0–3 and 99.5% at stage 4. Baseline offsets grow
0.001 → 1.399 m as before.

Two conclusions follow that the single 1000–1199 sweep could not support:

1. **The aligned deep-stage deficit is robust, not seed noise.** Stage 4
   scores 60.0% and 63.5% on two disjoint 200-episode seed sets — 27 to 30
   points below the bar, with intervals excluding any plausible 90% target.
   Stage 3 replicates at 84.5% and 85.5%.
2. **The baseline's stage-4 near-miss was seed noise.** It scored 89.0%
   (178/200) on 1000–1199 and 95.0% (190/200) on 0–199. The bar sits inside
   that spread, so the earlier one-point miss should not be read as a
   baseline defect. On these seeds the fixed-origin ladder passes every
   criterion with no failures recorded.

The failure mode is a clean split. At stage 4 the aligned oracle's 73
failures are overwhelmingly missed returns (65 double bounce, 8 out of
bounds), while the baseline fails only 10 times at all; its 182
out-of-bounds terminations are mostly completed rallies (173) that ran out
of court *after* the required returns. Aligned out-of-bounds terminations
are likewise mostly successes (72 of 80), so the raw cause counts must not
be read as failure counts.

**The attribution is confounded, and in the baseline's favour.** The
oracle's per-stage timing (`oracle_run_up=1.1` at stage 0, and
`oracle_charge_gap` 1.0/1.0/1.8/1.7 at stages 1–4) arrived at commit
`24223f4`, when every stage still served from `serve_start_x=1.0`; `3fd9fb3`
added the aligned ladder by changing **only** `serve_start_x` and left
every probe value untouched. `BASELINE_STAGES` then derives from
`ALIGNED_STAGES` by overwriting the origin, so both arms today run a
controller calibrated for a ball landing 1.0–1.4 m in front of the paddle.
Under alignment the ball lands at the paddle's feet — a different
interception problem the charge timing was never re-derived for.

**That confound has since been measured, and it is small.** A probe grid
over 19 settings per stage (11 `charge_gap` values, 8 `run_up` values,
200 episodes each, calibration seeds 0–199) recovers only part of the
deficit:

| aligned stage | shipped probe | best over the grid | gap to the 90% bar |
|---:|---:|---:|---:|
| 3 | 85.5% (`charge_gap` 1.8) | **88.5%** [83.3, 92.2] (`charge_gap` 2.4) | −1.5 |
| 4 | 63.5% (`charge_gap` 1.7) | **69.0%** [62.3, 75.0] (`charge_gap` 2.4) | −21.0 |

Zero of 19 settings clear 90% at either stage. Retuning buys +3.0 points
at stage 3 and +5.5 at stage 4 — about a fifth of each deficit. The same
grid on the fixed-origin ladder clears the bar comfortably (4 of 11
settings at stage 3, best 94.0%; 2 of 11 at stage 4, best 94.0%), so the
controller family is capable and it is the aligned geometry it cannot
handle.

Two mechanism details fall out. The aligned ladder's optimum sits at a
*larger* charge gap (2.4 against the shipped 1.8/1.7), which is what a
ball landing at the paddle's feet should require — commit earlier. And
`run_up` mode is useless at these depths (best 5.5% at stage 3, 1.0% at
stage 4), confirming the docstring's rationale that narrow deep windows
need a timed charge rather than a landing-point run-up.

The attribution therefore stands, with the confound priced in: the
deep-stage loss is dominated by the serve-origin geometry, and stale
controller timing accounts for roughly a fifth of it. Stage 4 is
decisively infeasible for this controller family; stage 3 is marginal,
its interval still straddling the bar.

**But full alignment is not the only option, and the hypothesis
survives.** A follow-on blend study (`plan_wall_ball_aligned_deep_stages.md`,
Phase B) interpolated the serve origins between the fixed origin and the
aligned ladder. Everything up to λ = 0.85 clears 90% at every stage;
λ = 0.90 fails stage 4 at 82%. The collapse is a sharp threshold in
landing distance rather than a gradual decay — 0.251 m in front of the
deep paddle passes, 0.184 m does not — so the paddle needs roughly a
quarter-metre of approach room that full alignment removes. Feasibility
is also non-monotonic: λ = 0.75 beats the fixed origin at stage 3 (99.0%
against 94.0%) while cutting the stage-4 landing gap from 1.399 m to
0.386 m. The design's premise — that moving the serve origin with depth
is worth testing — is intact; only its most extreme setting is not.

The crude controller remains mixed and placement-coupled (aligned higher
at stages 1, 2, and 4 — 95.5/90.0/19.5% against 73.0/69.0/14.0% — and
lower at stage 3, 46.0% against 58.5%), so it still carries no weight in
either direction.

## Strengthened calibration contract

The existing sweep certifies feasibility (oracle ≥2 returns on at least 90%
of episodes) and a minimal crude-controller learnability signal. Add a
distributional landing contract and make the certification reproducible.

For each stage and ladder variant:

1. Run 200 held-out episodes with `serve_speed_jitter` enabled and common
   paired seeds across baseline and candidate. The tool uses two explicit
   invocations (`--ladder baseline` and `--ladder aligned`) with the same
   `--seed-start` and episode count; the JSON pairing key makes the
   relationship auditable.
2. Hold the paddle at the configured start, make any physical contact before
   the first bounce a blocking failure, save every first-bounce x, and report
   mean, standard deviation, p05, median, p95, maximum absolute error, and the
   fraction within ±0.30 m of `paddle_start_x`.
3. For the aligned candidate, require at least 95% of individual serve
   landings to be within ±0.30 m. A close mean is insufficient if the
   jittered tails miss the band. The fixed-origin baseline is measured on the
   same seeds as a comparison condition; its known deeper-stage misalignment
   is not itself a baseline certification failure.
4. Run the true hold-start parked, crude, and oracle policies on the same
   held-out seed set and save per-episode returns, contacts, terminations, and
   telemetry identities rather than only rounded aggregates. Treat any
   hold-start contact or completed return as a blocking invariant failure.
5. Tune controller parameters on a separate calibration seed set, freeze
   them, then evaluate the held-out 200. Do not tune on the certification
   episodes. Record the calibration range and artifact identifier together
   with the code revision, source-state fingerprint, command, and timestamp.
   Once a held-out result has informed a redesign, retire that seed set and
   certify on a fresh pre-reserved range.
6. Require every existing static, telemetry, learnability, feasibility, and
   monotonicity criterion to pass. In particular, the held-out oracle point
   estimate must meet the existing ≥90% threshold at every stage; report a
   binomial confidence interval alongside it.

Stage 0 must be byte-for-byte equivalent across the baseline and candidate
ladder in this sweep, including `serve_start_x=1.0`. That preserves a negative
control and prevents a nominal alignment correction from changing the task
before stage 1.

## Paired stage-1 training A/B

Stage 1 already passes its candidate landing, hold-start, crude, oracle, and
telemetry checks even though the full ladder fails at later stages. That is
enough mechanical evidence to test the only mismatch that could have
contributed to the observed stage-1 stall; it is not certification of stages
2–4 and does not show that SAC learns better.

- **Baseline arm:** `WallBallDepthCurriculum`, stage-1
  `serve_start_x=1.0`.
- **Candidate arm:** `WallBallDepthCurriculumAligned`, stage-1
  `serve_start_x=0.69`.
- Keep stage 0 at `1.0` and keep every other environment, reward, gate,
  observation, logging, and training parameter identical.
- For each independent pair, train a separate baseline seed to stage-1 entry,
  save its complete state (model, normalizer, replay buffer, optimizer,
  counters, and RNG state), then clone that state once into the baseline and
  aligned arms. Use a fixed horizon of 1.5M post-fork training steps. Cloning
  one shared checkpoint into many RNG continuations is only a conditional
  pilot, not independent training-seed replication; it cannot authorize the
  6M decision without independent-state confirmation.
- Before the four-pair variance pilot, pre-register an ordered
  checkpoint-generation list of 48 seeds (twice the maximum supported pair
  count) and a 1.75M stage-0 entry horizon. Train seeds in that order until
  enough entry states exist, retain and report every non-entry, and use the
  first `n_required` entrants in the pre-registered order. Never replace a
  seed selectively after inspecting its trajectory. If the list yields fewer
  than `n_required` entrants, the confirmatory A/B is infeasible/no-go rather
  than an invitation to extend or cherry-pick.
- If complete resumable states are unavailable, pre-register a separate
  common-seed cold-start design with independently trained seed pairs and a
  3.5M total horizon. Do not mix continuation and cold-start pairs in one
  estimate.
- Define the primary paired outcome before launch. For continuations, use
  `min(T_baseline, H) - min(T_aligned, H)`, where `T` is steps from the fork
  to stage-2 promotion and `H=1.5M`; positive values favor alignment. For the
  cold-start fallback, use the same restricted-time difference from training
  start with `H=3.5M`.
- Use 250k steps as the minimum worthwhile mean gain. Begin with four blinded
  pairs solely to estimate the paired-outcome variance, then set the final
  sample size for 80% power at one-sided α=0.05. Use at least eight total
  pairs; if the calculated requirement is 9–24, use that requirement. If it
  exceeds 24, the confirmatory A/B is infeasible/no-go under the available
  budget before unblinding—do not run 24 and describe it as 80%-powered. Do
  not inspect arm labels, select seeds, or extend only one arm during
  sample-size re-estimation.
- The go rule is conjunctive: mean paired gain at least 250k steps, a
  one-sided 95% percentile-bootstrap lower bound above zero (100,000 resamples
  of whole pairs with a fixed analysis seed), zero telemetry-identity
  violations, and a still-passing scripted hold-start invariant. Hitting 24
  pairs without meeting every condition is inconclusive/no-go for a full
  aligned pilot, not permission to keep sampling. Because `rally_style` is
  deliberately `open`, contact style is descriptive rather than a hidden
  pass/fail condition.
- Save the complete curriculum history, gate-eval rows, progress data, raw
  evaluation episodes, and configuration snapshot for both arms.
- Cross-evaluate the resulting baseline and aligned policies on both the
  fixed-origin and aligned serve geometries using common episode seeds. This
  2×2 policy-by-geometry matrix separates training effects from the changed
  final-evaluation distribution.

Primary estimand:

- the mean paired restricted-time gain defined above.

Secondary endpoints at matched stage 1 (reported with paired intervals but
not substituted for the primary go rule):

- promotion proportion by the fixed horizon;
- gate `bounce_count_ep_mean`;
- hit-to-completed-return conversion;
- double-bounce and out-of-bounds shares;
- contact position and post-bounce style; and
- sensitivity across held-out seed batches.

The candidate mechanism is supported only if the aligned arm passes the
pre-registered restricted-time go rule without violating hold-start or
telemetry contracts. If the arms are indistinguishable or the aligned arm is
worse, stop: the stage-1 stall is more likely dominated by recovery,
interception, gate noise, or another mechanism. Do not expose stages 2–4 in a
6M run merely because their landing means look better.

## What deliberately does not change

The calibration and stage-1 A/B isolate serve origin:

- reward coefficients and components;
- the `bounce_count_ep_mean ≥ 3.0` promotion gate, its window, pause, and
  replay-clear behavior;
- rally style, fence geometry, paddle start, and serve-speed schedule;
- the 23-dimensional observation space; and
- recovery reset rules, evaluation pipeline, metric definitions, and cadence.

The final-task **distribution does change** in the aligned recipe because its
final evaluator uses `serve_start_x=-0.35` instead of the baseline's `1.0`.
Historical and aligned final scores or videos are therefore not directly
comparable. Use the 2×2 cross-evaluation above for attribution; “unchanged
evaluation” here refers only to machinery and metrics, not serve geometry.

The landing-point observation feature and gate/ladder redesign remain separate
levers. If the stage-1 alignment A/B fails, evaluate those mechanisms in their
own experiments rather than combining them here.

## Serve-speed alternative remains open

The stage-4 crude-controller probe measured:

| approach | serve | mean landing | ball vx at bounce | crude ≥2 | OOB |
|---|---:|---:|---:|---:|---:|
| baseline origin 1.0 | 7.0 | −2.49 | 6.14 | 11% | 4% |
| origin 1.0 | 8.0 | −2.99 | 7.03 | 86% | 55% |
| origin 1.0 | 9.0 | −3.50 | 7.93 | 80% | 39% |
| origin 1.0 | 10.0 | −4.00 | 8.83 | 11% | 6% |
| move origin to −0.35 | 7.0 | −3.84 | 6.14 | 22% | 6% |

Moving the origin remains the preferred next experiment because it changes
landing position without simultaneously increasing horizontal pace. However,
the data do not categorically reject a speed adjustment: speeds 8–9 improve
the crude controller's ≥2-return metric substantially while also increasing
OOB terminations and ball velocity. Because the controller is placement
coupled and OOB can occur after completed returns, this table describes a
tradeoff rather than selecting a winner.

Do not change speed in the stage-1 origin A/B. If origin alignment fails,
serve speed or a joint landing/energy calibration can be tested later under a
separate pre-registered comparison.

Serve loft likewise remains a separate alternative. Prior experiments found
that lofted serves can pass over the fixed-height paddle face, but that does
not need to be resolved for the origin-only A/B.

## Experimental implementation scope

The branch now contains the minimum isolated implementation needed for
certification and the A/B:

1. the existing `WallBallDepthCurriculum` recipe and starter remain the
   fixed-origin control;
2. `WallBallDepthCurriculumAligned` is a separate treatment with effective
   stage origins `1.0`, `0.69`, `0.34`, `-0.01`, and `-0.35`;
3. stage 0 is identical across recipes and remains at origin `1.0`;
4. the treatment's final evaluation is explicitly pinned to
   `serve_start_x=-0.35`, with cross-geometry evaluation required for
   historical comparison;
5. the sweep selects baseline or aligned stage tables, enforces the aligned
   landing contract, and records raw paired outcomes plus run provenance;
6. separate packaged starter configurations and tests prevent control/
   treatment drift.

These changes make the branch runnable as the treatment configuration. They
do not change the production default: the failed stage-3/4 feasibility cells
and the unrun stage-1 training A/B remain blockers for using the aligned
recipe in a full pilot.

This remains a configuration and calibration change; no environment, reward,
or observation implementation change is intended.

## Method and evidence limitations

- This is one training run. Checkpoint comparisons do not provide independent
  training seeds.
- Milestone checkpoints repeat the same three evaluation seeds.
- The 30-episode policy sweeps and 80-episode controller checks were retained
  only as rounded aggregates, so paired uncertainty cannot be reconstructed.
- The prototype landing log retained means but discarded standard deviations
  and individual landing positions.
- The exploratory stage-3/4 timing search used calibration seeds 0–199 but
  produced no standalone raw calibration artifact. New sweep reports record
  that declared range for the aligned search; the older baseline oracle has
  no retained calibration-range provenance. Repeat any future tuning into a
  separately retained artifact.
- Seeds 1000–1199 have been inspected and must not be reused as held-out data
  after controller changes.
- The current diagnostic reports identify a dirty source state by hashes but
  do not embed its patch. Certification requires a clean committed revision
  or an archived exact source patch and dependency environment.
- Matched-stage comparisons co-vary fence, start, speed, and training
  exposure.
- A trained-policy transfer failure under a new origin does not predict
  from-scratch learnability in either direction.
- All future claims should distinguish individual legal-hit rate,
  episode-hit rate, completed-return count, and gate bounce count.

## Final recommendation

Treat serve alignment as a plausible, low-dimensional candidate mechanism.
Begin the experimental configuration and instrumentation work, but do not
ship the production ladder or start a 6M aligned pilot yet.

Proceed in this order:

1. preserve stage 0 at origin `1.0`;
2. pre-register and run the powered paired stage-1
   baseline-versus-`0.69` training A/B now; stage 1 already passes the
   mechanical landing and oracle checks relevant to the observed stall;
3. only if that A/B passes its go rule, redesign or recalibrate the stage-3/4
   feasibility controller on calibration data, freeze it, and certify on the
   untouched 2000–2199 range without weakening the existing criteria;
4. advance to a full aligned-ladder 6M pilot only after both the A/B and fresh
   late-stage certification pass; and
5. if the A/B fails, stop and test recovery/interception, observation, or gate-noise
   hypotheses as separate levers.

This sequence tests the observed stall directly, keeps the intervention
falsifiable, and avoids attributing the unencountered stage-2-to-stage-4 gaps
to a stage-1 failure.
