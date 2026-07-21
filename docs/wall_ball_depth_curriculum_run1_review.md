# WallBallDepthCurriculum run 1 review — budget-bound at stage 2, transfer confirmed

Status: review of run `WallBallDepthCurriculum/sac/20260721_004722`
(v0.15.0, git `1330509`, seed 0, 3M steps, 10h24m, completed without
early stop). **First-round fixes shipped in 0.15.1** (see CHANGELOG):
P0 items 1–3 (SAC warm start, window-mean gate, promotion warm-up
package — the recipe adopts pause 50k + buffer clear as the
continuation run's single training-dynamics lever), P1 item 5
(`legal_paddle_hit_x_*` positional instrumentation), and P2 items 6–8
(stage-index stamping into eval artifacts, multi-episode milestone
videos, `reward_eval_episodes = 5`). Still open: the landing-point
observation feature (item 4, trigger-gated on a de-noised stage-2+
stall) and the second seed (item 9). Evidence: `stage_summary.txt`, `config.json`,
`progress.csv` (stage/gate/eval time series), `eval_info.csv` and
`eval_info_final.csv` (matched vs final-geometry eval streams),
`best_model_long_horizon_eval.json` (50-seed audit), and
`media/best_model.mp4` frames. Numbers below are from those artifacts;
the design being reviewed is `design_wall_ball_depth_curriculum.md`.

## TL;DR

- **The curriculum works.** Final-geometry (stage 4) performance rose
  monotonically with every stage earned — `bounce_count_ep_mean` at the
  fixed goal geometry averaged **0.30 → 0.98 → 1.76** across the run's
  three stage residencies, without the goal task ever being trained on.
  Positional transfer is real, exactly what the design bet on.
- **The run is budget-truncated, not plateaued.** It spent 1.43M steps
  (48% of budget) earning stage 0, 0.97M on stage 1, and ran out of
  budget mid-climb in **stage 2 of 4** — matched-stage best 3.80 at
  2.95M steps, still setting new bests at the end, early stop never
  fired. Lesson 8 ("budget cures nothing after a plateau") does not
  apply: there is no plateau here.
- **Two mechanical frictions burned most of the budget**: promotion
  shock (each stage advance halved matched-eval performance — 3.37 →
  1.43, 3.10 → 1.70 — costing ~0.5–1M steps of recovery each), and gate
  noise (stage 2 cleared the 3.0 bar in four separate 30-episode evals —
  3.07, 3.30, 3.20, 3.80 — but never twice consecutively, so it was
  never promoted).
- **The binding skill is post-return recovery, not striking.** At the
  goal geometry the median long-horizon episode is exactly one completed
  return; 86% of deaths are double-bounce; only 46% of episodes ever
  recover a post-bounce ball; the best-model video shows the paddle
  making its return mid-court and then drifting while the rebound dies.
  This is the tracker-like, interception-limited taxonomy the design
  doc pre-registered as the trigger for the landing-point observation
  feature.
- Recommended sequence: **(A)** continue the campaign from this run's
  final model at stage 2 (extend warm start to SAC; truncated stage
  table via TOML) with the gate de-noised, **(B)** if stage 2 stalls
  again, ship the landing-point observation feature measured against
  stage 2, **(C)** independent lever runs (finer ladder / smaller
  buffer) only as needed, one per run.

## What the run did

### Stage timeline (from `curriculum/stage_index` in progress.csv)

| Stage | Fence (x) | Start | Serve | Entered | Exited | Steps in stage | Share |
|---|---|---|---|---|---|---|---|
| 0 | (−2.7, 0.3) | −1.6 | 5.2 | 0 | 1,425,680 | 1.43M | 48% |
| 1 | (−3.2, −0.6) | −2.1 | 5.5 | 1,425,680 | 2,400,360 | 0.97M | 32% |
| 2 | (−3.7, −1.0) | −2.7 | 6.0 | 2,400,360 | 3,000,000 (end) | 0.60M | 20% |
| 3 | (−4.2, −1.2) | −3.3 | 6.5 | — | — | never reached | |
| 4 | (−4.7, −1.2) | −3.9 | 7.0 | — | — | never reached | |

Promotion 0→1 fired on consecutive matched evals of 3.13 / 3.37 (after
a near-miss at 1.30M: 3.27 then 2.90). Promotion 1→2 fired on 3.33 /
3.10. Both promotions immediately crashed the matched eval (3.37 → 1.43;
3.10 → 1.70) followed by ~1M / ~0.5M steps of recovery — the
`sustain_evals` dip the gate docstring predicts, but at a much larger
scale than "transient".

### Performance by stage (means over eval batches in each residency)

| Geometry | Stage | bounce mean | ge3 rate | ep len | double-bounce | OOB | timeout |
|---|---|---|---|---|---|---|---|
| matched | 0 | 1.69 | 0.19 | 273 | 78% | 20% | 1.5% |
| matched | 1 | 2.02 | 0.30 | 312 | 73% | 22% | 3.8% |
| matched | 2 | 2.48 | 0.34 | 338 | 79% | 14% | 6.4% |
| final (stage 4) | 0 | 0.30 | 0.01 | 150 | 68% | 32% | 0% |
| final (stage 4) | 1 | 0.98 | 0.07 | 196 | 91% | 8% | 0% |
| final (stage 4) | 2 | 1.76 | 0.22 | 276 | 85% | 11% | 3.3% |

Two healthy signs: the final-geometry column climbs with every earned
stage (the transfer the campaign exists to buy), and timeouts — rallies
still alive at the 750-step cap — grow steadily (censored *successes*,
per lesson 13).

### The goal-task verdict (50-seed long-horizon audit, best model, stage-4 geometry)

Completed returns mean **2.12** (median 1, p90 5, max 6); survival ≥2
38%, ≥3 34%, ≥5 14%; terminations 86% double-bounce / 14% OOB; mean
episode 310 of 5000 steps; post-bounce recovery in 46% of episodes.
Reward remains ~80% tracking shaping (6.75 of 8.40) — diagnostic only
(lesson 11), but worth remembering when reading `episode_reward`.

For scale, the frozen fixed-lane era reference (`20260718_023737`,
one-bounce style, service-line lane) audits at 3.42 returns / ≥5 at
22% on its own — easier, incomparable — task. This run's 2.12 at a
deeper, never-trained geometry after earning only 2 of 4 stages is a
defensible first campaign data point, not a verdict.

### Best-model video

`best_model.mp4` is 104 frames (~one short episode) despite
`video_length = 10000` — the recorder truncation from the baseline
review (Finding 8) is still present, so the behavioral record per
milestone remains n=1. The one episode it does show is diagnostic:
paddle starts deep, charges, returns mid-court, then wanders laterally
while the rebound double-bounces — a recovery failure, not a striking
failure.

## Diagnosis

1. **Pacing, not capability, consumed the run.** Stage 0 alone took
   1.43M steps of genuine learning (the near-random policy needed
   ~1.3M steps to first touch 3.0). Every subsequent run that cold
   starts will re-pay that bill before doing any new science.
2. **Promotion shock is the second tax.** Three things change at once
   per stage (fence, start, serve speed); the 1M-transition replay
   buffer is 100% previous-stage data on promotion day; and the fence
   is not observable, so an MLP policy can only discover the new clamp
   by acting (same observation, different action consequence — the one
   place the env's own "no hidden state" doctrine is violated, though
   only across stage boundaries).
3. **The gate is noisier than the thing it measures.** With 30-episode
   batches and per-episode std ≈ 1.7–2.5, the batch SE is ≈ 0.35–0.45.
   A policy whose true mean is ~3.0 clears "≥3.0 twice consecutively"
   more or less by coin flip — stage 2 demonstrated exactly this
   failure mode four times.
4. **The skill wall, when it comes, is interception after the return.**
   Double-bounce dominates every stage and geometry; the design doc's
   pre-registered trigger for the landing-point observation feature
   ("stage-gate stalls with tracker-like failure taxonomy") has now
   half-fired — stage 2's stall is at least partly measurement noise,
   so continue first, and treat a *de-noised* stall as the real
   trigger.

## Recommendations

### P0 — continue the campaign; make progression cheap and honest

1. **Continue from this run instead of cold-starting.** The final
   model (3.0M) is the most capable stage-2 policy the project has:
   matched 2.37–3.80 band, final-geometry ~1.9. Concretely:
   - Extend `warm_start` to SAC (`train.py` currently raises for
     anything but PPO): load `final_model.zip` weights (policy +
     critics) and `vec_normalize.pkl`, fresh replay buffer, and raise
     `learning_starts` to ~25k for the continuation so updates resume
     on frontier-stage data rather than 100 stale transitions.
   - Start the ladder at stage 2 via the packaged TOML — the
     `[train.performance_gate]` table replaces wholesale, so a
     continuation config lists only stages 2–4:

     ```toml
     [train.performance_gate]
     metric_key = "bounce_count_ep_mean"
     threshold = 3.0
     sustain_evals = 2
     [[train.performance_gate.stages]]
     paddle_x_fence = [-3.7, -1.0]
     paddle_start_x = -2.7
     serve_speed = 6.0
     [[train.performance_gate.stages]]
     paddle_x_fence = [-4.2, -1.2]
     paddle_start_x = -3.3
     serve_speed = 6.5
     [[train.performance_gate.stages]]
     paddle_x_fence = [-4.7, -1.2]
     paddle_start_x = -3.9
     serve_speed = 7.0
     ```
   - Budget 3M. Given measured pacing (~0.6–1M per remaining stage
     including shock recovery), that is tight but plausible; 4–5M if
     wall-clock allows. This is not a lesson-8 violation — the run was
     climbing when it stopped.

2. **De-noise the gate without lowering the bar.** Keep threshold 3.0
   (it is the campaign's definition of mastery, and matched mastery
   demonstrably bought final-geometry transfer). Change what is
   measured, not the bar: gate on the **mean of the last two eval
   batches ≥ 3.0** (equivalently a 60-episode batch), instead of two
   independent ≥3.0 events. Under the observed variance this promotes
   a true-3.2 policy in weeks-of-evals fewer and still blocks a
   true-2.5 one. A one-line change in
   `PerformanceGatedEnvStagesCallback._on_step` (track the previous
   metric value and compare the pair mean), plus its unit test.

3. **Blunt the promotion shock — one lever, one run** (lesson 6), in
   preference order:
   - **Finer ladder** (config-only): split each remaining stage delta
     in two — first move serve speed, then move fence/start — and
     re-run `tools/depth_stage_sweep.py` on the new table before
     training (the sweep is the contract; its blocking criteria and
     the ≥1.4 m fence-width floor still apply). Smaller deltas mean
     the sustain dip stays "transient" as the gate docstring intends.
   - **Buffer sized to a stage's lifetime** (one `model_kwargs` line,
     `buffer_size = 500_000`): on promotion day the 1M SB3-default
     buffer is entirely previous-stage physics; 500k evicts it within
     half a stage residency. This is the same argument the (never-run)
     Bootstrap package recorded. It deviates from the "SB3 defaults
     are what every learning run used" stance, so it ships alone and
     gets its own verdict.

### P1 — the capability lever, pre-registered and now half-triggered

4. **Landing-point observation feature, measured against stage 2.** If
   the continuation run (with the de-noised gate) still cannot sustain
   3.0 at stage 2+ after ~1M steps, the design doc's queued follow-up
   is earned: append the ball's current-arc ballistic touchdown
   (`land_x`, `land_y`, `t_land` — gravity-only projection of the
   current free flight, a deterministic function of existing obs
   dims) to the observation (23 → 26, a declared metric-era break).
   Two facts argue it will bite: the sweep's charge-and-lead oracle
   needed exactly this computation (its `t_land`/`land_x` code is the
   feature, hand-evaluated) to beat wide fences, and the failure
   taxonomy is uniformly "arrived late/wrong to the bounce". Falsifiable
   prediction to record with the run: stage-2 dwell time drops and the
   double-bounce share falls at matched geometry.

5. **Make "plays from the baseline" measurable before it matters.**
   Stage 4's fence front is −1.2: a policy could in principle master
   it by sprinting to the front and volleying, satisfying the gate
   while defeating the campaign's purpose. Today nothing measures
   where contact happens. Add cheap info keys (e.g.
   `paddle_hit_x_last`, per-episode mean paddle x at legal hits) and
   surface `*_ep_mean` in eval aggregation. This also arms the phase-2
   decision (true-baseline XML extension) with data instead of video
   anecdotes.

### P2 — hygiene, in the order they hurt this review

6. **Log the stage everywhere results land.** `eval_info.csv` /
   `eval_info_final.csv` rows, `best_model_meta.json`, and
   `stage_summary.txt` ("Headline best: 3.80") carry no stage index —
   reconstructing "3.80 *at stage 2*" required joining progress.csv
   serve-speed columns by timestep. One extra long-format metric row
   (`curriculum_stage_index`) per eval batch and one line in the
   summary fix this permanently.
7. **Fix the milestone video recorder** (`video_length` truncation,
   flagged in the baseline review, still shipping 104-frame n=1
   records) and record ≥3 episodes per milestone. At depth, behavior
   review is how front-camping (item 5) gets caught early.
8. **Cut the redundant reward-eval stream.** Three 30-episode streams
   every 25k steps (~10,800 eval episodes) cost roughly as many env
   steps as the 3M training steps themselves at ~300 steps/episode.
   The `EvalCallback` reward stream is selection-inert (review item
   11, still open); 5 episodes keep the `evaluations.npz` artifact
   alive and return ~15% wall clock — worth ~0.4M extra training steps
   per day of L4 time.
9. **Second seed on the winning configuration** (lesson 7). Every
   pacing number above is n=1; the continuation run doubles as a
   second sample of stage-2 dynamics only if its config is held still.
10. **Watch items, no action yet:** auto-entropy sits at ~0.0007–0.0009
    from 100k onward and does not re-inflate after promotions — if a
    de-noised continuation still recovers slowly from stage shocks,
    consider resetting the entropy coefficient state on advance (one
    lever, own run). VecNormalize still normalizes the five flag/counter
    dims (`normalize_obs_excluded_indices = []`) whose base rates now
    drift per stage — mechanism exists, was never adopted; candidate
    for a bundled-era change only.

### Explicitly not recommended

- **Reward-magnitude tuning to fix double-bounce** (escalator,
  penalties, shaping scale): lesson 19 — it redistributes failure
  style without moving the ceiling. The open-style placement bonus
  (arming `recoverable_bounce_bonus` on gated wall contacts outside
  `one_bounce`) stays on the shelf unless the landing-point feature
  moves the ceiling and placement becomes the *new* binding skill.
- **Lowering the gate threshold.** 3.0-mastery is what produced the
  transfer column; promoting weaker policies buys deeper stages the
  policy then can't survive (the sustain docstring's own argument).
  De-noise the estimator instead (item 2).
- **Capacity or budget-only reruns from scratch** — both falsified
  (lesson 19 addendum, lesson 8) except where budget extends an
  actively climbing run, which is what item 1 is.

## Environment-review notes (repo, current HEAD)

Reviewed `envs/wall_ball.py`, `wall_ball_baseline.toml` /
`wall_ball_depth_curriculum.toml`, the `WallBallDepthCurriculum` /
`WallBallBaseline` recipes, `performance_gate.py`, and
`depth_stage_sweep.py` against the design docs:

- The env's event machinery (substep-ordered contacts, refundable
  advances, nonfinite guards, no-op invariant) and the gate/selection
  harness behaved exactly as documented in this run — no misapplied
  stages, no selection-on-noise, terminations partition cleanly. The
  0.15.0 recipe matches its packaged TOML and the design doc's
  outcome addendum, including the final-stage `eval_env_overrides`
  pin.
- One doctrine gap, curriculum-specific: `paddle_x_fence` participates
  in dynamics (target clamping) but is not observable, so across stage
  boundaries the policy faces identical observations with different
  action consequences. Within the fixed-lane era this never mattered;
  under a moving fence it is a real (if bounded) POMDP-ness that the
  finer ladder mitigates and an obs feature would eliminate — fold
  that decision into item 4's era break rather than spending a
  separate one.
- `recoverable_bounce_*` is hard-wired to `one_bounce` style
  (`accept_wall_contact` arms eligibility only there), so open-style
  runs have zero placement gradient by construction — intentional per
  the design doc, noted here so nobody expects that channel to appear
  in curriculum telemetry.
- The baseline (`WallBallBaseline`, one-bounce era) remains frozen and
  correct as the reference task; nothing in this review proposes
  touching it, its metrics era, or run `20260718_023737` as reference.
