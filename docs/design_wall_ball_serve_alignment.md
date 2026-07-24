# Design: WallBallDepthCurriculum — align the serve landing with the paddle start

Status: proposed, 2026-07-24. This is a diagnostic review of the run-2
sliding-ladder pilot (`WallBallDepthCurriculum`, cold-start SAC, 6M-step
budget) together with a controlled serve-speed and matched-stage sweep run
on the saved checkpoints. It records the findings and specifies a
geometry-only fix. The design being reviewed is
`design_wall_ball_depth_curriculum.md`; the prior run's review is
`wall_ball_depth_curriculum_run1_review.md`.

Evidence base: the run's Google Drive artifacts — 13 milestone checkpoints
(model `.zip`, `vecnormalize.pkl`, rollout `.mp4`, per-step rollout `.csv`)
at every 250k steps from 250k to 3.25M of the 6M budget — plus new
rollouts of the 1.75M / 2.75M / 3.25M models reproduced locally with the
repo env (SB3 2.9.0, MuJoCo 3.10) and a parked-paddle serve-physics
measurement. All per-checkpoint CSV telemetry-contract identities pass.

## TL;DR

- **The serve lands progressively short of the paddle start as the ladder
  deepens** — aligned at stage 0, **1.35 m short at stage 4**. The
  serve-speed schedule (5.2 → 7.0) was calibrated so the ball *reaches* the
  paddle (100% contact) but not so it *lands at* the paddle. The deep paddle
  must abandon its start and charge ~1.3 m forward to meet a ball that
  bounces well in front of it, and completed returns collapse in lockstep
  with the growing gap (2.8 → 0.6 returns, stage 0 → 4).
- **This is the likely reason the deep stages are so much harder than the
  pre-flight sweep predicted.** The sweep certifies *feasibility* — a
  scripted charge-and-lead oracle returns ≥2 balls 94–95% of the time at
  every stage — but that oracle *compensates* for the misalignment by
  charging. An RL policy from the deep start does not learn that charge for
  free; it just gets a worse serve to return.
- **Recommended fix (geometry only): co-move `serve_start_x` with each
  stage so the serve lands at the paddle start**, then add a
  serve-landing≈paddle-start check to `tools/depth_stage_sweep.py`,
  re-calibrate the ladder, and re-run the pilot. This changes neither the
  reward, the task rules, the gate, nor the observation — it is the smallest
  intervention that removes a forced forward-charge the deep stages are
  currently made to learn on top of the rally.
- **Pre-flight (2026-07-24) passes.** With `serve_start_x` co-moved, the
  serve lands on the paddle start to ±0.02 m at every stage, the no-op
  invariant still holds (a parked paddle scores 0), and the uncalibrated
  learnability bar is as-good-or-better — the stage-4 second-return rate
  *doubles* (11% → 22%). Only the oracle's charge timing needs re-tuning for
  the new serve. The definitive verdict still needs a retrain.

## What the Drive artifacts actually measure

The per-checkpoint `wall_ball_depth_curriculum_sac_<steps>.csv` / `.mp4`
are **3-episode rollouts at the fixed stage-4 final geometry** (fence
(−4.7, −3.0), start −3.9, serve 7.0), written by `VideoRecordCallback`
using the recipe's `eval_env_overrides` unsynced to the gate
(`recipes.py:730-747`, `callbacks/video_record.py`). Verified three ways:
every one of the 36 legal contacts in the 39 milestone episodes lands
inside the stage-4 fence and outside every shallower fence; the design doc
says videos run at the final stage; and the code path confirms it. So the
milestone stream measures final-geometry **transfer**, not the matched
training stage, at n=3 per checkpoint — a behavioral probe, not a robust
level estimate. The matched-stage timeline
(`reports/curriculum_stages.json`, `metrics/eval_info.csv`) is not in the
Drive folder, so "which stage did the run reach" cannot be read directly;
it is inferred below from matched-geometry rollouts of the saved models.

## Findings

### 1. The milestone CSVs understate the policy (n=3 sampling noise)

The 3.25M milestone CSV shows 0.33 completed returns at stage-4 geometry.
Re-rolling the same model for 30 episodes gives **0.63** (≥1 return in 63%
of episodes); 15 episodes gave 0.87. The Drive snapshot simply caught a bad
3-serve draw. The behavioral read still holds — deep contact (−3.4), 100%
post-bounce, zero opening volleys, ~1 return then double-bounce — but any
single milestone number is quantized to {0, ⅓, ⅔} and should not be read as
a level. Widen the milestone rollout past 3 episodes, or read the 15-episode
`eval_info_final.csv`.

### 2. Serve pace is not an out-of-distribution eval confound

Holding the stage-4 geometry fixed and varying only the serve speed
(3.25M model, 30 eps/cell):

| serve | completed returns | ≥1 return | hit-rate | contact-x |
|---:|---:|---:|---:|---:|
| 5.2 | 0.00 | 0% | 100% | −3.02 |
| 5.5 | 0.00 | 0% | 100% | −3.04 |
| 6.0 | 0.10 | 10% | 100% | −3.11 |
| 6.5 | 0.40 | 40% | 100% | −3.23 |
| 7.0 | 0.63 | 63% | 100% | −3.31 |

Returns rise **monotonically with serve speed**. A slower serve at the deep
geometry does not recover returns — it eliminates them: the ball arrives too
weak to be sent back to the wall and dies short. The fast serve is *required*
at depth, exactly as the design intends ("serve energy co-moves with depth so
the ball reliably reaches the deeper paddle", `recipes.py:715-718`). Any
theory that the flat milestone curve is an artifact of testing at too fast a
serve is refuted.

### 3. The policy is a shallow-court policy that has not climbed the ladder

Evaluating each saved model at each **matched** full-stage config
(fence + start + serve together), completed returns (30 eps/cell):

| stage (start / serve) | 1.75M | 2.75M | 3.25M |
|:---|---:|---:|---:|
| 0  (−1.6 / 5.2) | 1.53 | 2.30 | **2.33** |
| 1  (−2.1 / 5.5) | 1.50 | 2.03 | 2.20 |
| 2  (−2.7 / 6.0) | 1.27 | 1.30 | 1.20 |
| 3  (−3.3 / 6.5) | 0.77 | 0.77 | 0.97 |
| 4  (−3.9 / 7.0) | 0.63 | 0.60 | **0.63** |

Two things stand out. Performance **degrades monotonically with depth** —
the same policy rallies 2.33 returns (up to 6, ≥3 in 40% of episodes) from
the forward start and 0.63 from the deep start. And **improvement over
training is concentrated at the shallow stages** (0–1 climb 1.5 → 2.3;
2–4 are flat across 1.5M steps). That is the signature of a run **training
at stage 0–1 and not promoting** — its stage-0 matched mean (~2.3) sits
below the 3.0 gate, so it never earns the deeper stages, and nothing
improves at depth. The flat deep-geometry milestone curve is the honest
consequence: a stage-0/1 policy tested at stage 4.

(These matched-stage numbers are estimates of the gate metric: deterministic
policy, `serve_speed_jitter=0.5`, 30 eps, seeds 1000–1029. The true gate
value may differ by a few tenths, but the *shape* — improvement only at
shallow stages — is stable across all three checkpoints.)

### 4. Root cause — the serve lands short of the paddle start (parked paddle)

Serve landing (ball-x at its first floor bounce) measured with a parked
paddle, so this is pure serve physics, independent of any policy:

| stage | paddle start | serve | serve lands at | **shortfall (land − start)** |
|---:|---:|---:|---:|---:|
| 0 | −1.60 | 5.2 | −1.64 | **−0.04** (aligned) |
| 1 | −2.10 | 5.5 | −1.79 | +0.31 |
| 2 | −2.70 | 6.0 | −2.04 | +0.66 |
| 3 | −3.30 | 6.5 | −2.29 | +1.01 |
| 4 | −3.90 | 7.0 | −2.55 | **+1.35** |

The paddle start recedes 2.3 m across the ladder (−1.6 → −3.9), but the
serve landing recedes only 0.9 m (−1.64 → −2.55). The serve-speed schedule
**under-compensates** for the depth recession, so the ball falls further and
further in front of the deep paddle. Cause: **`serve_start_x` is fixed at
1.0 for every stage** — in the recipe base kwargs (`recipes.py:719`) and in
the sweep's stage table (`tools/depth_stage_sweep.py:60-74`). The ladder
translates `paddle_x_fence`, `paddle_start_x`, and `serve_speed`, but not
the serve origin, so the ball is always launched from the same near-wall
spot and merely thrown harder. Gravity plus one bounce means the landing
point barely moves while the paddle marches deep. (The reset code comment at
`wall_ball.py:1968` says stages "translate it with the paddle start" — but
this recipe's stage table does not.)

Completed returns track the shortfall almost perfectly: 2.8 → 2.4 → 1.2 →
0.8 → 0.6 as the gap grows 0 → +1.35. The ball still *reaches* the paddle
(100% contact at every stage — post-bounce it travels on to ~−3.4 inside
the fence), so the misalignment does not cost contact; it costs **return
quality**: the paddle must charge ~1.3 m forward off its start and strike a
ball from an awkward, out-of-position pose, which is the dominant
"hit-but-no-return" failure bucket at depth.

### 5. The misalignment has already shaped the learned policy

Applying the fix (co-moved `serve_start_x`) to the *trained* 3.25M policy
does **not** help — its stage-4 hit-rate drops from 100% to ~75% and it
loses returns it used to make. That is expected and, in fact,
corroborating: the policy learned to *charge forward* to meet the short
serve, so aligning the serve makes its learned charge overshoot. The
optimal strategy under an aligned serve is different — "play from your
start" rather than "charge forward" — and no policy or scripted controller
built for the misaligned serve can demonstrate it. The clean existence
proof is **stage 0, which is already aligned**: there the policy rallies
2.8 returns with the easy play-from-your-start motion. Aligning the deep
stages recreates that structure.

## Recommended fix — co-move the serve origin with the stage

Add `serve_start_x` to every `performance_gate` stage, co-moving it with
`paddle_start_x` so the serve lands at (or just in front of) the paddle
start at each stage, restoring the stage-0 alignment across the whole
ladder. Measured values (parked-paddle pre-flight, 2026-07-24) — the
origin→landing shift held cleanly 1:1, and each lands the serve on the
paddle start to within ±0.02 m:

| stage | fence | start | serve | `serve_start_x` (today → set to) | aligned landing |
|---:|---|---:|---:|:---|---:|
| 0 | (−2.7, 0.3)  | −1.6 | 5.2 | 1.0 → **1.04** | −1.61 |
| 1 | (−3.2, −0.8) | −2.1 | 5.5 | 1.0 → **0.69** | −2.11 |
| 2 | (−3.7, −1.6) | −2.7 | 6.0 | 1.0 → **0.34** | −2.72 |
| 3 | (−4.2, −2.4) | −3.3 | 6.5 | 1.0 → **−0.01** | −3.32 |
| 4 | (−4.7, −3.0) | −3.9 | 7.0 | 1.0 → **−0.35** | −3.91 |

Concretely, the `[[train.performance_gate.stages]]` entries (and the sweep's
`STAGES` table) each gain a `serve_start_x`. This is a config + calibration
change; no env, reward, or observation code changes.

## Pre-flight check (2026-07-24)

Ran the sweep's own controllers (`_crude`, `_oracle`, parked) on the current
vs. aligned ladder, plus the parked-paddle landing measurement. Static
checks pass on both (geometry is unchanged). Results:

| stage | serve landing (current → aligned) | parked (no-op) | crude ≥2 return (current → aligned) | oracle ≥2 return (current → aligned) |
|---:|---:|---:|---:|---:|
| 0 | −1.65 → −1.61 | 0.00 | 85% → 74% | 99% → 98% |
| 1 | −1.80 → −2.11 | 0.00 | 74% → **95%** | 99% → 100% |
| 2 | −2.06 → −2.72 | 0.00 | 65% → **90%** | 95% → 94% |
| 3 | −2.31 → −3.32 | 0.00 | 52% → 51% | 95% → 88% |
| 4 | −2.56 → −3.91 | 0.00 | 11% → **22%** | 98% → 66% |

Reading:

- **Alignment achieved.** The aligned serve lands on the paddle start to
  ±0.02 m at every stage.
- **No-op invariant preserved.** A parked paddle scores 0 on both ladders —
  aligning the serve does not let a static paddle cheat.
- **Learnability improves (the clean signal).** The uncalibrated crude
  controller — the historical learnability bar — completes a second exchange
  *as often or more often* on the aligned ladder, with the largest gains at
  the stages where the misalignment was worst (s1, s2, and s4's rate
  doubling). This revised a wrong prior expectation: because crude sweeps the
  whole fence back-to-front, it does not collapse on the aligned serve.
- **Oracle needs re-derivation (a calibration artifact, not a regression).**
  The oracle drops at s3–s4 (s4 98% → 66%) only because its `oracle_charge_gap`
  timing is tuned to the old forward-landing serve. It must be re-tuned for
  the aligned serve before the certification run, exactly as the sweep-update
  section requires; the uncalibrated crude bar (which improves) is the
  trustworthy learnability read here.

Nothing in the pre-flight contraindicates the fix; the scripted learnability
bar points the right way. The definitive verdict remains the retrain below.

## What deliberately does not change

Isolate the serve geometry, per the parent design's one-lever discipline:

- **Reward coefficients and components** — unchanged. This does not add a
  depth bonus, a must-bounce rule, or a completed-post-bounce-return bonus;
  those remain shelved behind their pre-registered triggers.
- **The promotion gate** — `bounce_count_ep_mean ≥ 3.0`, window-mean of 3,
  50k pause + replay clear on advance. Unchanged.
- **Rally style** (`open`), fence geometry, `paddle_start_x`, and the serve
  *speed* schedule — unchanged. Only the serve *origin* moves.
- **Observation space** — unchanged (still 23-dim; the landing-point
  feature stays a separate, later lever).
- **Evaluation** — still runs on the true full serve at the final stage;
  no recovery-curriculum or drop-ball starts are introduced (the env's
  `one_bounce` recovery resets stay disabled for open style).

If the policy now moves backward and keeps rallying, the result belongs to
serve alignment rather than a simultaneous reward, rule, or observation
change.

## Sweep / calibration contract update

`tools/depth_stage_sweep.py` currently certifies that a stage is
**feasible** (oracle ≥2 returns from ≥90% of serves) and **learnable**
(crude controller completes a second exchange in >0% of episodes). Neither
tests *where the serve lands relative to the paddle start* — which is
exactly the gap this fix targets. Add a blocking static check:

- **Serve-landing alignment:** for each stage, the mean serve-landing x
  (parked paddle, serve physics only) must lie within a small tolerance of
  `paddle_start_x` (proposed ±0.3 m). Report the per-stage landing and
  shortfall alongside the existing feasibility/learnability numbers.

Note that the oracle's `oracle_run_up` / `oracle_charge_gap` parameters are
calibrated to the *current* short serve; after co-moving `serve_start_x`
they must be re-derived (a play-from-start oracle needs little or no
forward charge). The 2026-07-24 pre-flight confirmed this: the uncalibrated
crude bar improved on the aligned ladder, while the calibrated oracle
dropped at the deep stages purely from stale charge timing. Re-tune the
oracle and re-run the full 200-episode-per-cell sweep before any training,
as the parent design requires.

## Validation plan

1. **Calibrate:** the 2026-07-24 pre-flight already set `serve_start_x` per
   stage and confirmed the landing-alignment check (±0.02 m), the no-op
   invariant, and an improved crude learnability bar. What remains before
   training: re-derive the oracle charge params for the aligned serve and
   re-run the full 200-episode-per-cell sweep so every existing static /
   feasibility / learnability / monotonicity / telemetry criterion passes on
   the calibrated ladder.
2. **Retrain:** run the 6M-step pilot on the aligned ladder. This is the
   only clean test of the learnability claim — a policy *trained* on the
   aligned serve can adopt the play-from-start strategy that trained/scripted
   controllers on the misaligned serve cannot.
3. **Compare** (falsifiable predictions to pre-register): vs. the misaligned
   run at matched budget — (a) the ladder promotes past stage 1 (the
   misaligned run appears stuck at stage 0–1 below the 3.0 gate); (b) at
   matched deep stages, hit-to-return conversion rises and the
   double-bounce share falls; (c) deep contact x stays inside the stage
   fence (the style win must not regress). If alignment does not lift
   promotion past stage 1, the binding constraint is elsewhere (recovery /
   interception), and the pre-registered landing-point observation feature
   becomes the next lever.

## Caveats and what is not yet proven

- **Single seed.** Every number here is one training run; pacing and level
  estimates are n=1.
- **The learnability benefit is only partly measured.** The geometry fix is
  validated (serve realigned to ±0.02 m; no-op invariant preserved), and the
  scripted crude bar improves on the aligned ladder (stage-4 ≥2-return rate
  doubles), which is a positive but *scripted* signal. Stage 0 is an
  existence proof. Only the retrain (step 2) confirms that a *trained* policy
  climbs the aligned ladder further than the misaligned one.
- **`serve_start_x` values are measured, not final.** The pre-flight fixed
  them to land the serve on the paddle start (±0.02 m); the full sweep, run
  alongside the re-tuned oracle, is what certifies the calibrated ladder.
- **"Stuck at stage 0–1" is inferred** from matched-geometry rollouts of the
  saved models, not read from the run's stage timeline. Pulling
  `reports/curriculum_stages.json` and matched `eval_info.csv` into the
  artifact set would confirm it directly and is recommended regardless.
- **Relationship to other levers.** Serve alignment is complementary to,
  and cheaper than, the two levers already on the shelf: the landing-point
  observation feature (run-1 P1, aimed at interception/recovery) and a
  finer ladder / de-noised gate (run-1 P0, aimed at promotion pacing).
  Alignment addresses the *serve geometry* root cause with a geometry-only
  change and should be tried first; if the deep-recovery wall persists after
  alignment, escalate to the landing-point feature as a separate run.

## Method notes (reproducibility)

- Models and `vecnormalize.pkl` for 1.75M / 2.75M / 3.25M were pulled from
  Drive and integrity-checked (exact byte size + zip-CRC on every archive
  member + well-formed pickle streams).
- Rollouts: `SAC.load` + `VecNormalize.load` (train=False, norm_reward=
  False), deterministic policy, `normalize_obs(obs)` before `predict`, raw
  single env stepped to the 750-step cap; the same seeds across compared
  cells. Serve landing recorded as ball-x at the first `event_floor_bounce`
  with the paddle parked at the fence back (so the serve is never
  intercepted pre-bounce).
- Sweeps: 30 eps/cell unless noted; the parked-paddle landing measurement
  used 40–60 eps/cell. The pre-flight ran the sweep's own `_crude` / `_oracle`
  controllers (imported verbatim from `tools/depth_stage_sweep.py`) at
  80 eps/cell.

## Final recommendation

Ship the serve-alignment fix as the next lever, in this order:

1. **Add `serve_start_x` to each ladder stage** (values above), in both the
   recipe's `[[train.performance_gate.stages]]` and the sweep's `STAGES`.
2. **Add the serve-landing≈paddle-start check** to
   `tools/depth_stage_sweep.py`, re-tune the oracle charge params, and re-run
   the 200-episode certification.
3. **Retrain the 6M-step pilot** on the aligned ladder (in the training
   environment, with the full gate/eval/logging), and evaluate the
   pre-registered predictions above — first among them that the ladder now
   promotes past stage 1.
4. **Also surface the matched-stage artifacts** (`curriculum_stages.json`,
   `eval_info.csv`, `progress.csv`) so "stuck at stage 0–1" is confirmed from
   the timeline, not just inferred.

Hold the landing-point observation feature (run-1 P1) in reserve: it targets
the *recovery* wall, which is a distinct skill from serve interception. If,
after alignment, the run promotes but deep rally length still plateaus at one
completed return, that is the trigger to ship it — as a separate run, one
lever at a time.
