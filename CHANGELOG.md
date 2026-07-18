# Changelog

Per-release notes for Courtside Dynamics. Entries describe physics,
observation, action, and recipe changes that determine which saved policies,
`VecNormalize` statistics, and learning curves remain comparable across
versions. Newest releases first.

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
