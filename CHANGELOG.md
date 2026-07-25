# Changelog

Per-release notes for Courtside Dynamics. Entries describe physics,
observation, action, and recipe changes that determine which saved policies,
`VecNormalize` statistics, and learning curves remain comparable across
versions. Newest releases first.

## 0.20.0

Curriculum-harness changes aimed at the depth campaign's dominant cost:
per-stage promotion price, not total budget. Runs
`20260721_004722` (3M steps, reached stage 2 of 4) and
`20260724_152530` (6M budget, still on stage 1 at 2.55M) place the same
ladder position at twice the compute, so what binds is the cost of each
promotion.

One new `performance_gate` key, **default off**, so existing runs are
bit-identical:

- `reset_entropy_on_advance` restores SAC's auto-tuned entropy
  temperature to its initial value on every stage advance and clears the
  temperature optimizer's moment buffers. Run `20260721_004722` ended at
  `train/ent_coef` 9.2e-4 and never re-inflated after a promotion, so the
  policy met each new geometry very nearly deterministic — and
  `advance_update_pause_steps` sets `gradient_steps = 0`, freezing
  `log_ent_coef` too, so the tuner could not recover during the pause
  either. Restores the *pressure* to re-expand entropy, not the action
  noise itself: sampling spread lives in the policy's learned `log_std`
  and re-expands over the following gradient steps. Raises at training
  start when `ent_coef` is a fixed float, rather than being a silent
  no-op. The restore target is read from the **configured** `ent_coef`
  string rather than the live tensor: a warm-started continuation
  deliberately inherits the source run's collapsed `log_ent_coef`, so
  sampling the tensor would restore precisely the collapse the reset
  exists to undo. `entropy_reset_value` overrides that target when
  `"auto"`'s 1.0 is more exploration than a mid-campaign continuation
  wants.

`InfoDictEvalCallback` now publishes `last_confirmation_metrics` (cleared
per evaluation) so `confirm_best`'s second batch is observable instead of
vanishing except for the copy in `best_model_meta.json`.

A `pool_confirmation_samples` gate switch that folded that batch into the
promotion window was **built and then rejected** — see `docs/DECISIONS.md`.
The batch is conditionally sampled (it only runs when the primary batch
beat the running best), so averaging it in regresses exactly the high
draws toward the mean while leaving low draws alone. That biases the
window downward and silently raises the bar, the one thing run 1's review
said not to do. Simulated at the campaign's own numbers it cost **+27
evaluations ≈ +666k env steps per promotion** in the slow-climb regime
Run A is actually in.

Evaluation streams are consolidated. Under headline-metric selection the
reward `EvalCallback` and the final-config info-eval stream roll the
*same* distribution (the recipe's `eval_env_overrides`; the gate re-syncs
only the matched evaluator), and the reward stream is reporting-only
there. It is now retired and the final-config stream owns
`evaluations.npz`, emitting SB3's exact schema and mirroring
`eval/mean_reward` / `eval/mean_ep_length`, so TensorBoard dashboards,
`notebook_utils` plots, and `stage_summary.txt` read unchanged. One env
and one rollout pass fewer per evaluation; for run `20260724_152530`'s
config the three streams cost 50 episodes per 25k training steps.

New `final_eval_episodes` sizes that stream. It defaults to the **full
`n_eval_episodes`** when the merge happens — 30 for the depth recipe,
against the 5 + 15 the two split streams used — because the survivor is
then the only stream scoring the goal task during training. At 5 episodes
that estimate had a standard error of ~0.8–1.1 bounces, too noisy to
resolve the 0.30 → 0.98 → 1.76 transfer curve the campaign exists to buy.
Runs that keep both streams are unchanged (`n_eval_episodes // 2`).

Net eval cost per 25k training steps for the depth recipe: 50 episodes
across three rollout passes becomes 60 across two. Ten more episodes buys
a 6× larger sample on the campaign's target metric; set
`final_eval_episodes` explicitly to trade it back for wall clock.

Retiring the reward `EvalCallback` also removed the only periodic
progress a run printed, leaving a multi-hour job silent apart from stage
advances and the run-ending line. New `eval_verbose` restores it on the
info-eval streams alone, **default off** (`None` follows `verbose`):
`eval_verbose = 1` prints one line per evaluation per stream — step,
episodes, mean reward, episode length, curriculum stage, and the
selection metric — prefixed with the stream's `log_prefix` so the
matched and final-config lines stay apart. It is deliberately separate
from `verbose`, which drives SB3's per-rollout table: SAC emits that
every `log_interval` (4) episodes, thousands of tables over a
multi-million-step run, so it is not a usable substitute for progress at
`eval_freq` cadence.

Selection, the gate, `best_model.zip`, `best_model_meta.json`, and the
per-stage `stage_bests/` archive are all driven by the matched stream and
are unchanged.

Reporting and provenance:

- `curriculum_stages.json` records `reset_entropy_on_advance` (additive:
  existing readers are unaffected, but the file is not byte-identical to
  a 0.19.0 run's) and `curriculum/entropy_resets` is a new TensorBoard
  series.
- `config.json` now records `reward_eval_episodes` and
  `final_eval_episodes`. Both were absent from the hand-maintained
  `train_config` block, so a run's artifacts could not say whether its
  reward stream rolled 5 episodes or 30. A new test pins that the block
  covers every `TrainConfig` field except the four code-valued ones and
  the two recorded at top level, so the next field cannot drift the same
  way.
- `InfoDictEvalCallback` now rejects a multi-worker `eval_env`. Its
  rollout loop reads `infos[0]`/`rewards[0]` and counts episodes from
  worker 0 only, so extra workers were stepped but never measured — a
  silent mismeasurement, and the reason vectorizing evaluation is a
  rewrite of that loop rather than a bigger env.

Metrics era: unchanged for the matched stream and for `eval_info.csv`.
`evaluations.npz` keeps its schema but its rows come from a different
(larger) episode count, so absolute reward curves are comparable in
meaning and less noisy, not bit-identical.

## 0.19.0

`WallBallDepthCurriculum` now slides the paddle's entire movement
window toward the baseline instead of only extending its rear edge.
Run `20260722_124613` exposed the old ladder's loophole: all five
stages shared a front-court interval, so the policy could reach the
final stage while continuing to contact the ball at a shallow
position. The replacement ladder keeps adjacent stages
overlapping but removes any position common to every stage:

- paddle fences are now `[-2.7, 0.3]`, `[-3.2, -0.8]`,
  `[-3.7, -1.6]`, `[-4.2, -2.4]`, and `[-4.7, -3.0]`;
- paddle starts, serve speeds, open scoring, rewards, the
  three-evaluation promotion gate, and its replay-clear / 50k-update
  pause are unchanged;
- the default recipe and starter-config budget increase from 3M to 6M
  steps so the run has time to reach and train at the final stage; and
- fixed final evaluation now uses the new final-stage fence.

WallBall `info` now partitions legal paddle hits into pre- and
post-bounce counts, records whether the episode opened with a volley,
and counts completed returns initiated after a bounce. Matching
one-step event flags make contact ordering auditable without changing
the reward or observation. The long-horizon report advances to schema
3 and includes pooled contact-sequence rates alongside its existing
floor-bounce proxy diagnostics. The calibration sweep validates the
counter identities, reports the new behavior metrics, and adds a
non-blocking pre-bounce interception probe.

The replacement geometry passed the 200-episode-per-cell scripted sweep
on 2026-07-23. Oracle ≥2-return rates were 92–96% and crude-controller
rates remained nonzero at every stage; the diagnostic opening-volley
probe fell from 100% in stages 0–2 to 38% in stage 3 and 0% in stage 4.
There is no environment policy-space or reward change, so 0.18.0
checkpoints remain loadable; however, the new stage geometry and budget
make curriculum progression and aggregate learning curves a new
comparison era.

## 0.18.0

Performance-gated curriculum runs now keep their per-stage history.
Previously each stage advance called `reset_selection_state()` and the
next stage's first evaluation overwrote `best_model.zip` — destroying
the departing stage's champion. Runs `20260721_142121` and
`20260722_002913` each ended wanting exactly that checkpoint (the
stage-entry policy was the best either run measured at the next
stage's geometry), and only luck preserved one of them as the
run-level best.

- **`model/stage_bests/stage_NN/`**: on every advance the gate
  archives the departing stage's `best_model.zip` /
  `best_vec_normalize.pkl` / `best_model_meta.json` (immediately
  before the selection reset), and the final stage's best is
  duplicated there at training end — skipped when that stage recorded
  no evaluation, since the on-disk triple would belong to the previous
  stage. The run's `config.json` is copied alongside each triple, so a
  `stage_NN/` directory is a valid (legacy-flat-layout)
  `WarmStartConfig.source_run_dir`: a continuation really can
  warm-start from any archived stage's champion.
- **`reports/curriculum_stages.json`**: refreshed atomically on every
  stage close and finalized at training end, with each stage's
  entry/exit timesteps, evaluation count, promotion window values
  (kept under both promotion rules), streak, and archived best
  selection values — the table every campaign review previously
  reconstructed by hand from `eval_info.csv`. Interrupted runs keep
  it too: train()'s KeyboardInterrupt salvage path finalizes the gate
  (SB3 skips `on_training_end` when an exception escapes the training
  loop), and completed-stage rows are already durable on disk even
  against a hard runtime death.
- **`config.json` gate provenance**: the structured
  `performance_gate` block now records `promotion_rule`,
  `advance_update_pause_steps`, and `clear_replay_buffer_on_advance`
  (mirroring `train()`'s resolution defaults) — run
  `20260721_142121`'s warm-up package was active but provably absent
  from its artifact.

Both new artifacts are registered in `RUN_LAYOUT` as conditional
(gate-only) entries, so non-gated runs audit clean. No physics,
reward, or training-behavior change; all 0.17.0 artifacts and curves
remain comparable.

## 0.17.0

`WallBallDepthCurriculum` widens the promotion window: `sustain_evals`
2 → 3 (`window_mean`, so promotion now needs the mean of the last
three 30-episode evals ≥ 3.0). Run `20260721_142121` — the first
campaign run under `window_mean` — promoted out of stage 0 on a 2-eval
mean of 3.08, only ~0.3 standard errors above the bar (per-eval SE
near 3.0 is ~0.4 bounces), i.e. a variance-driven promotion; its
stage-1 exit (3.37) would have cleared any reasonable window. The
threshold itself stays 3.0: observed eval maxima at the easy stages
(3.27/3.47) leave no headroom to raise it without gating the whole
run, and the run's stage-2 stall was a post-advance recovery failure,
not premature promotion. Cost is ~50k extra steps per stage. Promotion
timing — and therefore `curriculum/stage_index` trajectories — are not
comparable with 2-eval runs; per-stage eval metrics remain comparable.
`WallBallBootstrap` (historical) keeps its 2-eval gate.

## 0.16.1

Version bump only; no physics, observation, action, or recipe changes.
All 0.16.0 artifacts and learning curves remain comparable.

## 0.16.0

Humanoid-tennis stage overhaul: the curriculum's racket forgiveness now
physically works, and the three PPO stage recipes adopt the open
remediations from `docs/humanoid_env_review.md` / `docs/DECISIONS.md`.
A new metric era for the tennis stages — contact behavior and rewards
changed, so prior tennis-stage artifacts and curves are not comparable.
WallBall and ball recipes are unchanged.

- **Stringbed collision bounds fixed.** The `racket_contact_scale`
  enlargement wrote `geom_size` and called `mj_setConst`, which never
  refreshes `geom_rbound`/`geom_aabb`; MuJoCo's broadphase culled every
  contact beyond the nominal extent, leaving the forgiveness inert.
  Both bounds are now recomputed from the scaled semi-axes, and a
  regression test pins that a ball in the enlarged-only annulus
  generates a contact (and does not against the nominal stringbed).
  The stage-1 feasibility oracle was recalibrated for the corrected
  physics (swing at control call 56, pitch 0.65; both mirrored sides
  land the target return in 129 steps).
- **Task-metric selection for Stages 0–2.** `headline_key =
  "stage_success"` (plus `confirm_best_eval` and 5-episode
  reporting-only reward evals) moves `best_model.zip` and early stop
  off the shaped eval reward — the run-20260712_190054 failure mode the
  WallBall recipes already guard against. `legal_hit_count_ep_mean`
  sits between the success keys and the reward tie-break so
  pre-success contact progress still resets patience and updates the
  best model.
- **Exploration and shaping unblockers.** Stock iid Gaussian
  exploration measured zero contacts in 264 random episodes while a
  constant held action succeeds ~15% — the recipes now set
  `use_sde = true` with a small `ent_coef = 0.01` floor, and Stages 1–2
  enable the (previously inert) escrowed `valid_hit_shaping = 0.25` so
  a hit that becomes a target return keeps a contact-time reward while
  episode-total farming stays impossible.
- **Reward normalization off for the tennis stages.** The PPO default
  (normalize returns) would amplify the first sparse success toward the
  clip ceiling on this hand-scaled terminal-only reward
  (`normalize_reward = false` in recipes and starters).
- **Early stop reachable per stage.** Patience scaled to each stage's
  eval budget (8 / 12 / 20 for 20 / 40 / 80 evals); the shared 20 was
  mathematically inert for Stages 0–1.
- **Eager config validation.** `build_train_config` now resolves the
  algo name and validates merged `model_kwargs` against the resolved
  algorithm's constructor before any environment or artifact work
  (cross-algorithm keys previously failed only inside `train()` after
  the env fleet was built, and a string `ent_coef` on PPO survived
  until the first gradient update).

## 0.15.1

First-round fixes from the depth-curriculum run-1 review
(`docs/wall_ball_depth_curriculum_run1_review.md`, run `20260721_004722`:
budget-bound at stage 2 of 4, monotone final-geometry transfer confirmed).
The gate gains an opt-in `promotion_rule = "window_mean"` (promote on the
mean of the last `sustain_evals` evaluations — stage 2 cleared 3.0 in four
separate 30-episode evals without ever managing two in a row) and a
promotion warm-up package (`advance_update_pause_steps`,
`clear_replay_buffer_on_advance`): each advance drops the stale-stage
replay buffer and pauses gradient updates until fresh frontier-stage data
exists, targeting the measured promotion shocks (matched eval 3.37→1.43
and 3.10→1.70, ~0.5–1M steps of recovery each). `WallBallDepthCurriculum`
adopts all three (window mean, 50k pause, clear) plus
`reward_eval_episodes = 5` (the reporting-only reward eval stream cost
~as many env steps as training itself). Policy-only `warm_start` now
supports SAC (actor/critics/targets via the policy state dict plus the
auto-tuned entropy temperature), enabling stage-2 continuation runs with
a truncated `[train.performance_gate]` stage table. The gate stamps
`curriculum_stage_index` into the matched eval stream (eval_info.csv,
TensorBoard, `best_model_meta.json`, stage summary) so scores are never
again ambiguous about their geometry. WallBall info dicts add
`legal_paddle_hit_x_sum` / `legal_paddle_hit_x_mean` (where legal hits
happen — the fence-front-camping tripwire for deep stages).
`VideoRecordCallback` no longer stops at the first episode end: milestone
videos/CSVs record up to `max_episodes` (default 3) within the
`video_length` cap. No physics, observation-space, or reward changes;
existing eval metrics remain comparable within the 0.15 era.

## 0.15.0

Adds `WallBallDepthCurriculum` — the campaign's answer to the falsified
cheap levers (budget, incentives, capacity; see `docs/lessons_learned.md`
lesson 19): curricularize *position* instead. The recipe runs the env's
`open` rally style (any paddle hit opens the gate, volleys are legal paid
returns, no early-touch fine) with a performance-gated five-stage depth
ladder that walks `paddle_x_fence` / `paddle_start_x` from volley range
(−2.7, 0.3) back to the workspace baseline (−4.7, −1.2) while serve speed
co-moves 5.2 → 7.0, each stage earned by sustaining ≥3 eval exchanges.
The action mapping stays pinned to the full workspace so action semantics
never drift. Every stage was calibrated by the new
`tools/depth_stage_sweep.py` scripted ladder (parked < crude full-swing <
charge-and-lead oracle strictly monotone; oracle ≥2 returns from 93–97%
of serves; crude second exchanges 66–83%; 200 episodes/cell).
`WallBallBootstrap` is marked historical/superseded (its cold-start
problem was solved by auto-entropy before it ever ran, and its reward
package bundles the falsified weak-return retry). Depth-ladder metrics
are a new era — not comparable to fixed-lane baseline runs, whose
reference remains `20260718_023737`. No physics or observation changes;
existing recipes untouched.

## 0.14.0

Reverts the falsified 0.13.0 `WallBallBaseline` recalibration: run
`20260718_213222` showed the asymmetric weak-return retry teaches soft,
unchainable returns (1.33 vs 3.23 eval bounces; 82% double-bounce), so
terminal weak returns and SB3-default gamma are restored, keeping only
the ≥5-rate selection tiebreak. Adds `wall_reward_increment` (the n-th
completed return banks `1 + (n−1)×increment`; default 0.0 is
bit-identical and it ships dark, ladder-calibrated at 0.5 — run
`20260719_223139` later showed it redistributes failure style without
moving the ceiling). Run directories are reorganized into `model/`,
`metrics/`, `reports/`, `media/`, `checkpoints/` via a shared registry
with legacy-flat fallback for existing runs, and the unreadable
`eval_info.png` grid is split into four themed, size-bounded pages.
Training-side reward semantics return to 0.12 behavior; eval metrics
remain comparable with 0.10+.

## 0.13.0

Recalibrates `WallBallBaseline` from the first two runs that learned the full
task (`20260717_165358`, `20260718_023737`): training fines weak returns
(`weak_return_penalty = 0.1` retries) while evaluation re-asserts the strict
terminal rule so scores stay comparable, the critic's horizon doubles
(`gamma = 0.995`, auto-entropy untouched), and best-model selection tie-breaks
on the ≥5-bounce rate the `023737` run actually improved instead of the
saturated ≥2 rate. WallBall also gains a render-only `court_style` kwarg —
`"tennis"` draws a to-size ITF half-court (baseline 11.885 m from the
wall-as-net, service line at 6.40 m, singles/doubles sidelines) for replay
footage, which the notebook now records by default — and the notebook resolves
the recipe's run config automatically (`CONFIG_FILE = "auto"` creates your
editable Drive copy from the packaged starter on first use). Training-side
reward semantics changed, so 0.12 baseline *learning curves* are not
comparable; eval metrics remain comparable with 0.10+.

## 0.12.0

Makes run configuration Colab-native: the starter TOMLs ship inside the
package and the SAC recipes carry their calibrated `n_envs = 8`, so the
training notebook no longer pins `MODEL_KWARGS` (the 20260717 A/B showed the
old pinned entropy bundle prevents WallBall from learning at all) and only
passes variables you explicitly set — a TOML's `[train]` table is not silently
overridden (`seed` stays notebook-owned: the loader rejects it in files,
loudly). Recipes now also carry the calibrated `early_stop_patience = 20`.
Physics unchanged; 0.11 artifacts remain comparable.

## 0.11.1

Adds render-only court markings to WallBall so videos show where on the court
the action is: bold white lines at the wall base (x = 3.9) and the deepest
paddle reach ("baseline", x = −4.7), faint 1 m coordinate ticks, a cyan strip
marking the preset's paddle lane with a yellow paddle-home line, orange fence
lines (visible only when a `paddle_x_fence` is set), and a warm line at the
serve drop x. The markers are MuJoCo sites — they cannot collide — and the
preset-dependent ones are repositioned every reset, so curriculum changes made
between episodes stay visible. No physics, observation, or reward change.

## 0.11.0

Targets the WallBall bootstrap failure (three runs in which SAC never touched
the ball while a scripted tracker contacts 100% of serves). New
`WallBallBootstrap` recipe: a `first_hit_bonus` paid outright once per episode
makes touching the ball strictly out-earn passivity for the first time;
`weak_return_penalty` softens the terminal `floor_before_wall` fault into a
fined retry so feeble early swings stop scoring identically to doing nothing; a
performance-gated ladder (new `PerformanceGatedEnvStagesCallback`) widens the
serve's lateral corridor from `serve_vy_max` 1.1 to the full 2.0 as competence
is demonstrated, with matched-stage evaluation driving selection and
`eval_info_final.csv` tracking the canonical serve. Depth and serve-pace
ladders were swept and rejected (close-court rebounds fly out; slow serves
underpower returns). WallBallEnv also gains `serve_start_x`, `paddle_start_x`,
`paddle_x_fence`, and a schedulable `paddle_joint_damping` for future
curricula, and run provenance now records the installed git commit on Colab
(pip VCS installs) and honors a `COURTSIDE_DYNAMICS_GIT_SHA` override. Existing
recipes are unchanged and 0.10 artifacts remain comparable.

## 0.10.0

Recalibrates `WallBallBaseline`'s geometry so the second rally exchange is
learnable, not just scriptable. The baseline recipe raises its paddle slide
damping from 5 to 8 (new `paddle_joint_damping` env kwarg), capping saturated
swings at 12.5 m/s and slowing returns ~15% so rebounds land shallower; it
widens its lane front from x=-2.1 to x=-1.6; and it softens the pre-bounce
paddle touch from a terminal style violation to a non-terminal
`early_touch_penalty` fine (new env kwarg; the default `None` keeps the
terminal rule). In the calibration sweep a placement-blind full-swing tracker
recovers a second exchange in 70% of episodes under the new geometry versus 0%
under the old, while the scripted oracle still returns 500/500 calibrated
serves and completes two or more returns from 92% of them. Observation and
action shapes are unchanged, but the physics and lane changes make 0.9
`WallBallBaseline` policies, `VecNormalize` statistics, and learning curves
non-comparable; start fresh baseline runs. The open `WallBall` and
`WallBallVolley` presets keep the shared XML's damping 5 and their existing
calibration. The recoverable-bounce placement score now projects to the new
baseline lane front (x=-1.6).

## 0.9.0

Adds recovery-focused training to the strict baseline style. WallBall supports
fixed per-run rally styles while keeping the same 3-action interface.
`WallBallVolley` forbids floor contacts. `WallBallBaseline` requires exactly
one bounce before each paddle return, starts the paddle farther back at world
x=-2.7, restricts it to the [-3.2, -2.1] baseline lane, and uses a calibrated
lower serve. Its training factory mixes normal serves with incoming-wall and
post-bounce recovery fragments, then tapers that practice using global training
steps. Checkpoint evaluation, videos, and post-training endurance scoring
always start from a normal serve. A one-bit recoverability flag expands
WallBall observations to 23 values, so all 0.8 `WallBall`, `WallBallVolley`,
and `WallBallBaseline` policies and `VecNormalize` statistics require a fresh
run. The original `WallBall` recipe remains the permissive `open` setup; train
separate policies for the strict volley and baseline recipes.

## 0.8.0

Simplifies WallBall to the paddle face alone at a fixed 10° upward pitch. Its
three `[-1, 1]` actions are absolute x/y/z position targets tracked by
force-limited servos, and its observation shrinks from 26 to 22 values after
removing yaw/pitch state. Previous WallBall models, `VecNormalize` statistics,
replay buffers, and raw MuJoCo states are incompatible; start a fresh run
rather than resuming a pre-0.8 artifact.

## 0.7.0

Corrects BallBounce's rotation units and actuator authority and replaces
control-boundary touch rewards with substep-resolved, top-face rebound events.
Earlier BallBounce policies and learning curves are not comparable and should
be retrained; the observation grows from 18 to 30 values to include ball spin
and the event detector's Markov state. Its training recipe reports success
after ten deliberate rebounds in one episode; passive paddle contacts do not
increment that metric.

## 0.6.0

Registered environment IDs become unversioned. Callers using the previous
version suffixes should switch to the unversioned IDs listed in the README
environment table.
