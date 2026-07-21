# Changelog

Per-release notes for Courtside Dynamics. Entries describe physics,
observation, action, and recipe changes that determine which saved policies,
`VecNormalize` statistics, and learning curves remain comparable across
versions. Newest releases first.

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
