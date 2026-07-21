# Design: WallBallDepthCurriculum — earn your way back to the baseline

Status: implemented in 0.15.0 (calibration sweep passed 2026-07-20; see
the outcome addendum at the end for the final stage table and sweep
numbers — the candidate table below is kept as proposed for the record).
Awaiting its first training run.

## Motivation

Ten training runs of the service-line rally (`wall_ball_baseline_review.md`,
`lessons_learned.md`) bracket the same ceiling — ~3.2–3.4 eval bounces,
~2.7–3.4 long-horizon returns — and the three cheap levers are exhausted:
budget (lesson 8), incentives (lesson 19), and network capacity (lesson 19
addendum). The failure taxonomy is pure rally precision: from the pinned
mid-court lane, a return must be struck so its rebound lands somewhere
recoverable, and that placement problem is what every model plateaus on.

Meanwhile the *goal* of the baseline environment was never mid-court play:
it is to rally from the baseline. The tennis-court overlay (0.13.0) made
the gap visible — today's lane straddles the ITF service line, 5.5–7.1 m
from the wall, while the paddle's physical workspace ends at x = −4.7
(8.6 m) and the true baseline sits at −7.985 (11.885 m).

This design curricularizes **position**: start where striking is easy
(near the wall, volley range), and walk the paddle backward each time the
policy demonstrates a sustained 3-exchange rally. Decisions fixed by the
review discussion: observations stay 23-dim; no early-touch fine; volley
play at the start is explicitly acceptable.

## Core idea: open scoring, emergent style

The curriculum runs under the env's existing **`open` rally style**, not
`one_bounce`:

- Any paddle hit opens the return gate; wall contact pays +1. Volleys are
  legal, paid returns — there is no "early touch" concept and no fine.
- Out-of-bounds, double-bounce, and stall terminals keep play honest
  (unchanged, symmetric — lesson 1).
- **The one-bounce pattern emerges from geometry instead of rules.** Near
  the wall the natural game is volleying; from deep positions every ball
  has bounced long before it arrives, so post-bounce baseline play is what
  the deep stages *are*, without a fault taxonomy to price.

Why not `one_bounce` with the fine set to zero: the fine is not what
blocks volleys — the bounce-gate is. A volleyed ball would reach the wall
unpaid, while pre-bounce serve-taming (the trick every trained model
already uses in 20–45% of episodes) would become free. Open scoring
removes both problems at once.

Not carried over from the one-bounce presets: `first_hit_bonus`,
`weak_return_penalty` (no floor-before-wall fault exists in open),
recovery fragments and `recoverable_bounce_*` (one-bounce machinery; the
curriculum's early stages are the easy practice those existed to
synthesize), and `wall_reward_increment` (stays dark, lesson 19).

## Stage ladder (candidate — final values come from the sweep)

The action mapping stays pinned to the full physical range (−4.7, 0.3)
for the whole run so action semantics never drift; each stage confines
position with `paddle_x_fence` and moves `paddle_start_x` /
`paddle_home_x`. Serve energy co-moves so the ball reliably reaches the
deeper paddle. All values are `set_wrapper_attr`-settable between
episodes — the same plumbing the bootstrap gate uses.

| Stage | Fence (x) | Home / start | Serve (speed, lob) | Character |
|---|---|---|---|---|
| 0 | (−2.2, 0.3) | −1.0 | 6.0, 2.0 | Volley range: block the serve back on the fly |
| 1 | (−2.7, −0.3) | −1.5 | 5.8, 1.5 | Short court: mix of volleys and first-bounce play |
| 2 | (−3.2, −0.9) | −2.1 | 5.5, 1.0 | Approaching today's lane |
| 3 | (−3.7, −1.6) | −2.7 | 5.5, 0.5 | The proven service-line regime, wider |
| 4 | (−4.2, −2.3) | −3.3 | 5.5, 0.25 | Deep court |
| 5 | (−4.7, −3.1) | −3.9 | 5.5, 0.0 | The workspace baseline: 7.0–8.6 m from the wall |

Fence width stays ≥ ~1.4 m at every stage (the damping×lane sweep showed
narrow-and-deep fails). `paddle_joint_damping` is per-stage-sweepable;
the candidate keeps 8.0 throughout, but stage 0 may want the volley
preset's 5.0 — the sweep decides.

### Pre-flight calibration (blocking, lessons 2/3/9)

Before any training, every candidate stage is swept with the scripted
ladder — parked / blind tracker / oracle (the open-style
`wall_ball_oracle_action`, fence-projected like `stage_sweep.py` did):

1. **Within-stage monotonicity**: parked < tracker < oracle, strictly.
2. **Feasibility**: oracle completes ≥2 returns from ≥90% of serves at
   every stage.
3. **Learnability**: the blind tracker completes a second exchange in a
   non-trivial fraction of episodes at every stage (the 0.10.0
   calibration bar).
4. **No difficulty inversion across stages**: tracker/oracle scores may
   decline with depth but must not spike down then up (the U-shape that
   killed the one-bounce depth ladder). Stages that invert get re-tuned
   (serve energy, fence width, damping) or dropped.

The sweep's output — the final stage table — ships in the recipe with
the sweep numbers recorded in this doc's outcome addendum.

## Gate and evaluation

- **Gate**: `PerformanceGatedEnvStagesCallback`, `metric_key =
  bounce_count_ep_mean`, `threshold = 3.0` (the review's "3 successful
  rallies"), `sustain_evals = 2`, evaluated at the **current stage's
  geometry** (matched-stage info-eval, stages applied to the eval env,
  selection state reset on every advance — all existing behavior).
- **Selection**: `best_metric_keys = (bounce_count_ep_mean,
  bounce_count_ep_ge_5_rate)` with the standard min-delta, confirmation
  batch, and degenerate guard.
- **Final scoring**: `final_info_eval = True` at the deepest reached
  stage, plus the 50-seed long-horizon audit at that stage's geometry.
  `curriculum/stage_index` in TensorBoard is the campaign's real
  headline: how far back did it earn its way?
- **Era break**: open scoring + moving geometry means these numbers are
  a new metric era; the one-bounce baseline era stays frozen with run
  `20260718_023737` as its reference. No baseline recipe or metric
  changes.

## Recipe and packaging

New recipe `WallBallDepthCurriculum` (new name on purpose — see the
Bootstrap note): `rally_style="open"`, stage-0 kwargs as env defaults,
`performance_gate` carrying the swept ladder, SAC with auto-entropy
(recipe `model_kwargs` empty), `n_envs = 8`, `early_stop_patience = 20`,
budget 3M (six stages need room; early stop still applies within the
final plateau), `eval_freq` 25k. Packaged starter TOML follows via the
drift test; the notebook catalog picks the recipe up automatically.

`WallBallBootstrap` is marked **historical/superseded** in its recipe
description and the README: its cold-start problem was solved by
auto-entropy before it ever ran (lesson 5), and its reward package
contains since-falsified components. The recipe and its tests remain
for the record.

## Out of scope (explicitly deferred)

- **True-baseline stages (x < −4.7)**: need the XML workspace extension
  (slide range / paddle base) plus a serve-and-rebound energy
  recalibration so balls reach 8–12 m. That is campaign phase 2, gated
  on the policy earning stage 5.
- **Landing-point observation feature**: observations stay 23-dim per
  the review decision. If deep stages stall on interception ballistics
  (visible as stage-gate stalls with tracker-like failure taxonomy),
  this is the queued follow-up, measured against the stage it stalled
  on.
- Multi-seed runs, PPO arm, human-in-the-loop stage overrides.

## Testing and rollout

- Recipe pin tests (open style, no fine kwargs, gate structure,
  threshold-backed selection keys) + starter drift lockstep + the
  standard loud-failure checks on stage dicts (attrs must be
  `set_wrapper_attr`-reachable; the gate already raises on typos).
- The calibration sweep runs first and gates implementation of the
  final stage table; its script lands in the repo (not scratchpad) so
  future stage edits re-run it.
- One training run; review against the gate trajectory (stages reached,
  time-in-stage) rather than raw bounce means — reaching stage 3+ with
  sustained 3-rallies already exceeds everything the fixed-lane era
  achieved at equivalent depth.

## Outcome addendum: calibration sweep (2026-07-20)

The blocking sweep (`tools/depth_stage_sweep.py`) passed after seven
iterations. The shipped ladder differs from the candidate table above
in three sweep-forced ways:

1. **All serves are flat** (`serve_lob = 0`). Lofted serves arc over
   the fixed-height paddle face — the candidate's lob taper was
   unplayable at every stage (iteration 1: oracle near 0%).
2. **Serve speed rises with depth** (5.2 → 7.0) instead of falling.
   Slow serves die mid-court and never reach a deep paddle; faster
   serves land deeper (5.5 lands ≈ −1.5..−2.0; 7.0 ≈ −2.6) and carry
   the energy a deep return needs. The pure-volley stage 0 was dropped
   for the same reason — with flat serves, volley-range play emerges at
   the shallow fence without a dedicated stage.
3. **Five stages, generous fence fronts** (width 2.6–3.5 m). Narrow
   deep fences strand the paddle behind unreachable mid-court rebounds
   (the 0.10.0 lesson, re-confirmed here).

Final ladder (each stage: fence, start, serve speed): (−2.7, 0.3) /
−1.6 / 5.2 → (−3.2, −0.6) / −2.1 / 5.5 → (−3.7, −1.0) / −2.7 / 6.0 →
(−4.2, −1.2) / −3.3 / 6.5 → (−4.7, −1.2) / −3.9 / 7.0. Damping 8.0 and
`serve_start_x` 1.0 throughout.

Probe notes: the fence-projected `wall_ball_baseline_oracle_action`
fails at wide fences (its ≤0.9 m commit gate starts the charge too
late; double-bounce dominated) — the sweep ships its own
*charge-and-lead* oracle (commit the full charge at the bounce,
ballistic y/z lead at the closing-speed intercept, ready position a
calibrated run-up behind the predicted landing point: 1.1 m shallow,
1.4 m deep). A timed pre-bounce charge variant was tried and rejected
(it launches at serve time and arrives stationary — weak volley
blocks). The crude full-swing rung is the learnability bar, per lesson
9; a pure y-tracker recovers 0% here (fixed-depth face sits in the flat
serve's dead zone) and was the wrong rung, as it was for the 0.10.0
baseline calibration.

Sweep verdict at 200 episodes/cell (per stage 0→4):

| Measure | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| parked reward | −1.00 | −1.00 | −1.00 | −1.00 | −1.00 |
| crude reward | 5.66 | 5.87 | 5.98 | 6.38 | 7.30 |
| oracle reward | 8.23 | 8.77 | 8.51 | 9.54 | 11.57 |
| oracle ≥2 returns | 94% | 93% | 94% | 94% | 97% |
| crude ≥2 returns | 83% | 73% | 70% | 66% | 74% |

All four blocking criteria hold: strict parked < crude < oracle at
every stage, oracle feasibility ≥ 90%, crude learnability far above
zero, and no cross-stage difficulty inversion (oracle bounce means
2.57 / 2.54 / 2.35 / 2.42 / 2.68). Failure taxonomies stay committed
(OOB-heavy, not double-bounce-heavy) for both probe policies at depth.
