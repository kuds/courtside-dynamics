# Changelog

Per-release notes for Courtside Dynamics. Entries describe physics,
observation, action, and recipe changes that determine which saved policies,
`VecNormalize` statistics, and learning curves remain comparable across
versions. Newest releases first.

## Unreleased

The PaddleTennis environment — the two-sided rung between wall-ball
and humanoid tennis — plus the infrastructure hardening between the
campaigns. Existing environments other than the new PaddleTennis are
unchanged on any healthy trajectory (the BallBalance and WallBall
changes below are confined to already-pathological nonfinite states
and to rendering), so their saved policies, `VecNormalize`
statistics, and learning curves all remain comparable with 0.25.0.
Within the unreleased PaddleTennis itself there are two eras: the
initial volley-rules freeze and the ground-rules amendment that
supersedes it (see the ground-rules bullet).

- **PaddleTennis environment and recipe.** *(Partially superseded by
  the ground-rules bullet below: the default opponent, rally-rule
  profile, reference band, and certification described here are the
  volley era's, reproducible via `volley_rule="legal"`.)*
  `CourtsideDynamics/PaddleTennis` (`PaddleTennisEnv`): 1v1
  cooperative rally on the probe-frozen paddle court — half-length
  6.5 m, regulation 0.914 m net, singles width ±4.115 m, the wall-ball
  ball and paddles verbatim — with every task number frozen by the
  pre-committed probe battery before the env existed
  (`docs/paddle_tennis_probes_20260802.md`,
  `docs/paddle_tennis_probes_p3_p4_20260802.md`). The policy plays
  side A through the familiar 3-action target interface; side B is an
  injectable `opponent_controller` (default: the frozen `lead_charge`
  scripted controller) reading the exact side-B mirror of the
  48-value side-relative observation (24 physical + 12 rally-state +
  12 contact-memory values; probe P4 pinned the mirror bit-for-bit).
  One point per episode from the P3-measured serve band (origin
  3.25 m, 9 m/s, 21°; 100% legal, 100% returnable), serve side
  alternating every reset; +1 per rules-confirmed return by either
  side, −1 on the ending fault, −2 on unsafe/nonfinite physics. The
  `PaddleTennis` SAC recipe selects and succeeds on `crossings` (the
  P3 reference band: 3.15–3.42 for the scripted pair) and normalizes
  only the physical block; a `court_style` kwarg
  (`diagnostic`/`tennis`/`none`) mirrors the WallBall render-only
  presentation styles, with visibility lists derived from the
  compiled model. The frozen definition passed held-out
  certification (`tools/paddle_tennis_probes.py --certify`, reserved
  seeds 3100–3199: mean crossings 3.22 against the pre-registered
  2.6 floor, zero unsafe terminations;
  `docs/paddle_tennis_env_20260802.md`).
- **PaddleTennis exploration package (training change).** The
  ground-era diagnosis's ranked remedy, shipped recipe-level:
  `model_kwargs` gains `use_sde=True`, `ent_coef="auto_0.02"`,
  `target_entropy=-1.5`, and `train_freq=(64, "step")`
  (docs/paddle_tennis_exploration_20260808.md). The raised target is
  the mechanism fix — the stock run's 5e-5 coefficient was SB3's
  tuner resting at the too-low default target (−3.0), so the target
  is what moves, not the coefficient — and the multi-step
  `train_freq` is what makes gSDE real under SAC at all (the
  off-policy collector resets the noise matrix every rollout;
  at `train_freq=1` that is every step — iid noise). iid per-step
  noise cannot produce coherent ball-reaching runs (83/100 points
  ended `policy_never_reached`). Task definition, reference band,
  and certification unchanged; learning curves start a new
  training-configuration regime. Pilot criteria pre-registered in
  the doc before the pilot ran.
- **Checkpoint behavioral diagnosis.** The exchange/positioning
  instrument that diagnosed the ground-era pilot (one memorized
  serve-return macro, no general ball-reaching;
  docs/paddle_tennis_diagnosis_20260808.md) now lives in the package
  (`training/paddle_diagnosis.py`; `tools/paddle_tennis_diagnosis_probe.py`
  stays as the CLI) and runs automatically at every checkpoint save
  via `TrainConfig.checkpoint_diagnosis` (enabled in the
  `PaddleTennis` recipe): per-checkpoint reports plus a cached oracle
  reference row under `reports/diagnosis/`, exception-isolated so a
  probe failure can never kill a run.
- **PaddleTennis ground rules (behavior change).** The first learned
  GPU run maximized cooperative return rate with a close-net volley
  loop (a net crossing every ~14 control steps, crossings 37.6 and
  climbing at the 2M budget). Pre-bounce returns are now
  `VOLLEY_RETURN` faults via `RallyRules.require_bounce_before_return`
  (off for wall-ball/humanoid consumers) and the new `PaddleTennisEnv`
  kwarg `volley_rule` — **default `"fault"`**, with `"legal"`
  reproducing the superseded volley era exactly. The scripted
  reference is the recalibrated ground oracle (run-up wait behind the
  predicted landing, soft `GROUND_SWING` stroke, hold-low recovery;
  the frozen P1 port's returns were measured to be largely volleys),
  giving a new reference band of 7.78 crossings at a 126-step cadence
  (probed: both volley witnesses collapse to 100% volley faults;
  ground play is rule-neutral; a volley fault confirms nothing, so
  touching a doomed ball earns nothing) and a new held-out
  certification (PASS 7.68 on reserved block 4200–4299, floors
  5.9/0.90 pre-registered from the pre-fix calibration band). `term_volley`
  joins the env's terminal flags and the recipe's eval keys.
  Volley-era artifacts (run `20260803_004559`, the local pilot, the
  3100-block certification at 3.22) are not comparable with
  ground-era numbers; docs/paddle_tennis_ground_rules_20260803.md is
  the amendment record.
- **P5 transfer instrument.** *(Recalibrated under ground rules —
  see the ground-rules snapshot §6; the numbers below are the
  volley-era stub calibration, and the baseline row now runs the
  ground oracle.)* `tools/paddle_tennis_p5_transfer.py`
  renders the paddle court as the wall-ball world (23-value
  observation, true-baseline action mapping) so the wall-ball
  champions can play `PaddleTennisEnv` unmodified. Scripted
  calibration (`docs/paddle_tennis_p5_transfer_20260802.md`): the
  `scaled` identification plus a serve-yield overlay is the only
  viable configuration; champion measurements and the pre-registered
  pool-admission rule run on Colab.
- **BallBalance nonfinite guard.** BallBalance now carries the same
  guard as every sibling env: actions are shape/finiteness-validated,
  the physics state is checked before stepping, and a nonfinite
  action/state ends the episode with reward 0 on the echoed last
  finite observation. Previously a NaN action reached MuJoCo, which
  warns and silently resets the simulation mid-episode — so on
  solver-blow-up seeds, episode lengths and termination counts differ
  from a stock 0.25.0 tree. Healthy episodes are bit-identical.
- **WallBall court-style rendering fix.** The 0.25.0 coordinate ticks
  (`court_tick_xm5..xm8`) escaped the hand-maintained style-visibility
  list and stayed visible in `court_style="tennis"`/`"none"` footage;
  the visibility lists are now derived from the compiled model, so the
  ticks hide correctly. Render-only (sites cannot collide); pinned by
  the cross-style observation-equality tests.
- **Run-summary stop reasons.** `stage_summary.txt` gains a
  `Stop reason` line naming which guard ended a stopped-early run
  (early-stop patience, degenerate signal, or stage budget). Artifact
  addition only; existing readers scan line-prefixes and are
  unaffected.

## 0.25.0

The true-baseline extension
(`docs/design_wall_ball_true_baseline.md`). The paddle workspace, serve
energy, and in-play volume extend to the ITF baseline — **verified
bit-for-bit neutral for every pre-existing configuration**, so all
saved policies, `VecNormalize` statistics, and learning curves remain
comparable with 0.22.0–0.24.0. Every geometry and mechanics number was
measured by scripted probes before the change landed, and the frozen
era task passed a held-out certification (seeds 4000–4099) before its
first run.

- **`wall_ball.xml` workspace extension.** `paddle_slide_x` joint
  range and `paddle_target_x` ctrlrange widen (−3, 2) → (−6.5, 2),
  taking the physical world-space workspace from (−4.7, 0.3) to
  (−8.2, 0.3). The bold baseline marker moves to −8.2 (just behind
  the ITF baseline the tennis style draws at −7.985) and gains −5..−8
  ticks (render-only sites). **The default x action mapping does NOT
  follow**: it is now frozen at the historical workspace via
  `_DEFAULT_PADDLE_X_TARGET_RANGE = (-4.7, 0.3)` — previously it was
  derived from the XML ctrlrange, and widening the XML would have
  silently rescaled action semantics for every config without an
  explicit `paddle_x_target_range`. The extended workspace is strictly
  opt-in. Verified by hashing full obs/reward/termination streams,
  HEAD vs extended tree, on four legacy configs: all identical.
- **`WallBallEnv` gains `ball_in_play_min_x`** (default −6.0 =
  bit-for-bit historical behavior; validated finite and negative).
  The deep out-of-bounds edge is task geometry: an 11 m/s serve dies
  OOB at −6.0 before a paddle at −7.9 can touch it (measured 60/60).
  Only the new recipe widens it (−10.0).
- **`ladder_certification` gains `feasibility_ge2_floor`** (spec knob,
  default the stock 0.90, validated in (0, 1]; the resolved floor is
  recorded in every report's provenance). No scripted reference
  dominates the 11.8 m task (calibrated oracle band: 1.98 mean, 67%
  ≥2 rate), so the true-baseline recipe declares 0.50 — the band
  minus two 30-episode sampling standard deviations. Every other
  recipe keeps the stock floor.
- **`WallBallTrueBaseline` recipe + packaged starter**
  (`run_configs/wall_ball_true_baseline.toml`): the goal-rally
  structure (direct task, single-stage informational gate at the
  campaign's 3.0 mark, no curriculum) at the era task — fence
  (−8.2, −2.6), start −7.9, home −5.4 (fence midpoint), serve 11.0
  from origin 1.0, mapping (−8.2, 0.3), in-play bound −10.0,
  patience 60 (the replication's record seed banked its best at 3.2M
  under patience 60). Certification probe `lead_charge` 2.6 (gap
  sweep winner), 30 episodes, seeds 30000+. Held-out certification
  passes all blocking criteria: oracle 1.81 mean / 61% ≥2 (Wilson CI
  [0.51, 0.70]), crude 9% > 0, monotone references, clean landing
  probe. `WallBallGoalRally` and every other recipe are untouched.

## 0.24.0

The campaign diagnosis release
(`docs/wall_ball_rally_diagnosis_20260728_review.md`). No physics,
observation, action, or reward changes: **saved policies and
`VecNormalize` statistics remain comparable with 0.22.0/0.23.0.** The
existing recipes are untouched; a new recipe replaces the depth ladder
as the campaign's production configuration.

**`WallBallGoalRally` — the structural replacement for the
sliding-fence depth ladder is no curriculum at all.** Three ladder
generations stalled below a flat 3.0 promotion bar that no scripted
reference reaches at any rung (~70 controller configurations, best
2.7) and that four multi-million-step runs plateaued 0.2–0.9 under
everywhere past stage 0, while each promotion bought a measured
+0.35 m serve-landing jump and (in 0.22.0) an advance package whose
one exercised instance cost ~4.3M of 6M steps. Meanwhile the
constant-width 0.22.0 fence work had already made the goal task itself
learnable (crude placement-blind two-return rate 80–84% vs 9.5–16% on
the retired fence) — and no run ever trained there, because the
ladder was in the way. The pre-registered paired local SAC structure
A/B/C (direct vs aligned-serve vs serve-origin mixture, common seeds;
review doc §4) settled the rest: direct goal-task training scores
2.03/2.71 completed returns on 100 held-out seeds at 500k local steps
— versus 1.14 all-time from any 6M ladder run — with the mixture no
better (2.27) and aligned-only worse (1.09–1.51). The recipe
therefore trains the depth ladder's final stage AS the whole task
(fence (−4.7, −2.6), start −3.9, serve origin 1.0, speed 7.0;
evaluation equals training equals the historical goal task, so every
prior goal number stays comparable). A single-stage gate that can
never promote keeps stage stamping, `curriculum_stages.json`, and
startup certification (training geometry certified held-out twice:
96% / 91% oracle two-return rates on seed blocks 3000-3099 /
3100-3199); its informational 3.0 bar makes
`curriculum/gate_window_mean` crossing 3.0 the campaign-goal marker.

**`stage_eval_budget` promotion-staleness guard** on
`PerformanceGatedEnvStagesCallback` (and the gate spec): a non-final
stage that records N evaluations without promoting either stops the
run (`"stop"`, default — run `20260727_233859` burned ~4.5M steps in
exactly that state with nothing able to notice) or force-advances
(`"advance"`, recorded as `promoted: false` with
`advance_reason: "stage_eval_budget"` in `curriculum_stages.json`, so
an unearned rung is auditable). History rows gain `advance_reason` and
`stage_eval_budget_exhausted` fields.

**Startup-certification probes recalibrated for the constant-width
geometry.** New `lead_charge` oracle probe mode (charge-gap trigger +
ballistic y/z lead at the intercept while charging): the stock
y/z-tracking charge failed the 90% feasibility floor at stages 0 and 3
of the 0.22.0 ladder (36%/70% two-return rates). Frozen per-stage gaps
(3.0/0.8/1.0/1.2/3.0, chosen at n=200 on calibration seeds 0-199 by
rate-with-margin rather than raw argmax) certify held-out at
95–100% per stage (seeds 3000-3099) — `tools/depth_stage_sweep.py
--ladder release` and every gated run's startup certification now pass
their blocking criteria. Startup episodes 10 → 30 (the 0.90 floor sits
close to the probes' true rates; 10-episode verdicts flip on one
seed). The remaining advisory warnings on the depth ladder — no
scripted reference reaches its 3.0 bar; every adjacent stage jumps the
serve landing by ~0.35 m — are true findings about that ladder.

**`wall_ball_oracle_action` mapping parameterized.** It hardcoded the
retired fixed −1.7 pivot and old spans, silently scoring ~0.5 bounces
on any re-pivoted geometry (review 20260727_233859). New
`paddle_home_x` / `paddle_x_target_range` kwargs derive the inversion;
defaults preserve the legacy `WallBall` recipe contract byte-for-byte.

## 0.23.0

Startup ladder certification. No physics, observation, action, reward,
or recipe-geometry changes: **learning curves and saved policies remain
comparable with 0.22.0.**

Review of run `20260727_233859` (the first 0.22.0 run;
`docs/wall_ball_depth_curriculum_20260727_233859_review.md`) found the
0.22.0 ladder had shipped without a calibration sweep:
`tools/depth_stage_sweep.py` still hardcoded the retired 0.21 stage
tables and fixed pivot, nothing forced it to track the recipe, and the
mismatch surfaced only after a 17-hour run stalled at stage 1. The
sweep is no longer ad hoc:

**Training runs self-certify.** A new `TrainConfig.ladder_certification`
spec (set by both depth recipes) makes `train()` sweep every
`performance_gate` stage with the scripted reference cells — parked /
crude / calibrated oracle plus the policy-independent serve-landing
probe — *before* the worker fleet is built, on fresh `env_fn()`
instances with each stage applied the way the gate applies it. The
verdict lands in `reports/ladder_certification.json` (new run-layout
artifact, recorded in `config.json`) and is printed at startup.
Certification is derived entirely from the live gate spec and env
factory, so there is no second stage table to go stale.

**Stage-application integrity is a blocking criterion.** Every stage
key must already exist on the constructed env and must read back with
the written value — the exact class of the pre-0.22.0 `paddle_home_x`
silent no-op, now caught in seconds instead of by post-mortem.
Certification also warns when no scripted reference reaches the
promotion bar and when the serve-landing offset jumps more than 0.25 m
between adjacent stages (the measured stage-0→1 receive discontinuity).

**Advisory by default.** The stock probes are per-index calibrations
from the 0.21 geometry; with them the current ladder is *known* to fail
feasibility at stages 0 and 3 and the inversion detector at stage 1, so
failures print loudly and are recorded but do not abort unless the spec
sets `enforce`. Certification probes draw from a reserved seed block
(30000+), disjoint from every burned experiment range.

**`tools/depth_stage_sweep.py --ladder release` (new default)** carries
no stage table: it certifies the `WallBallDepthCurriculum` recipe's
live gate stages through the same package certifier. The
`aligned`/`baseline` tables remain pinned to the 0.21-era serve-origin
candidates for reproducing the alignment campaign's paired comparisons.
Also fixed: the recipe's usable-action-share comment now carries the
measured values (0.49/0.43/0.42/0.47/0.63); the previous figures
(0.52/0.48/0.49/0.54/0.66) matched neither the code nor the changelog.

## 0.22.0

Four changes to both depth recipes, all aimed at defects the
`20260727_004014` post-mortem measured directly. That run promoted three
times (a first — the previous best reached stage 1) but plateaued on the
goal task at **1.14 completed returns**: its 50-episode long-horizon
audit landed the serve in 100% of episodes, a second ball in 10%, a
third in 2%, and ended every single episode in a double bounce.
**Neither learning curves nor saved policies are comparable with any
earlier depth run.**

**Constant 2.1 m fence, ≥1.3 m of runway at every stage.** Return pace is
set by the paddle's forward speed at contact, and that speed needs
runway. A swept probe on the old goal fence scored 0 completed returns
from 0.0–0.4 m of forward travel, 1 from 0.6–0.9 m, and 2–3 from
1.2–1.6 m. The old ladder narrowed the fence 3.0 → 1.7 m as it receded
and left exactly 0.9 m at the goal — enough for one good shot and no
re-load, which is precisely the 100%-then-fail signature the audit
found. Fences now hold a constant 2.1 m and every stage clears 1.3 m.
2.1 m is the widest constant width that keeps the ladder free of an
all-stage refuge: the fence travels 2.4 m back (−2.3 → −4.7), so any
width ≥ 2.4 would hand every stage a shared front-court interval again —
the flaw run `20260722_124613` exposed. The 0.3 m separation matches the
ladder it replaces, and `paddle_start_x` is unchanged at every stage.

**`paddle_home_x` is a live property, re-centred per stage.** It is the
pivot of the normalized x action map, and it was a plain attribute — so a
stage that moved it was a silent no-op (`_control_home` is derived once
during setup, and `set_wrapper_attr(..., force=False)` reports success
for any existing attribute). With the pivot pinned at −1.7 against the
old goal fence, 71.7% of the x action range — all of action 0 and the
entire positive half — clamped onto the fence's front edge. Each stage
now sets the pivot to its fence midpoint, holding the usable action
share roughly flat (0.49, 0.43, 0.42, 0.48, 0.63) instead of collapsing
0.67 → 0.28. `paddle_x_target_range` is untouched, so action *scale*
still never drifts across stages.

**`return_shaping_scale` = 0.15 (new env kwarg, default 0.0).** The
outgoing leg had no dense reward at all: `accept_paddle_hit` closes the
tracking window and only wall contact reopens it, so "how hard did I hit
it" rode on one sparse +1 about 50 steps later. Measured consequence — a
1.5 m/s pop-up (which the fixed 10° face pitch sends mostly *upward*)
paid exactly what a 17 m/s drive paid, and 6 of 63 legal hits never
reached the wall. The new term is potential-based on the ball's
remaining gap to the wall and joins `_pending_shaping`, so it is
refunded unless the return actually completes. Deliberately far below
`track_shaping_scale` (0.5): the incoming term already crowds out the
flat +1 wall payment (57% → 73% of per-cycle reward as the fence
recedes), and 0.15 adds ~1.0 per completed return rather than dominating.

**`reset_entropy_on_advance` = True.** The run finished with
`train/ent_coef` at 0.0011 — effectively deterministic — after sitting on
stage 3 for 97 evaluations. Every advance had been dropping a policy
with no exploration budget into new geometry. The lever was added in
0.20.0 for exactly this failure and had never been exercised; it is
legal here because `model_kwargs` pins only `gamma`, leaving SB3's
`ent_coef` on `"auto"`.

Not changed: `serve_speed` still co-moves with depth (5.2 → 7.0), so
depth and ball speed remain confounded. `wall_reward_increment` stays at
0.0. Both are the next levers if the goal-task number stays flat.

## 0.21.0

Two changes to both depth recipes, aimed at the two defects the
2026-07-26 measurement pass identified. **Neither learning curves nor
saved policies are comparable with any earlier depth run.**

`gamma` 0.99 → 0.995, pinned in `model_kwargs`. Entropy stays on SB3
auto — this is a one-lever change. The exchange cadence in this task
measures 117–135 env steps on the calibrated oracle, so the SB3 default
prices the next completed return at `0.99^130 ≈ 0.27` of the one already
banked and the third at `≈ 0.07`. A policy that stops trying after
exchange two is behaving optimally under that discount, which is a
mechanical candidate for the ~2.0–2.4 plateau every depth run has hit.
The only run in this project that ever produced a long rally — WallBall
`20260713_192636`, `bounce_count_ep_mean` **12.30**, 13 exchanges with
zero floor bounces, every episode timing out rather than dying — used
gamma 0.995. That run appears nowhere in this changelog or in
`docs/`, and `lessons_learned.md`'s "five healthy runs bracket the same
~3.2–3.4 ceiling … the binding constraint is capability" omits it.
gamma was tested once before, in 0.13.0, bundled with the falsified
weak-return retry and reverted with it, so it has never been evaluated
alone.

`n_eval_episodes` 30 → 60 on the matched/selection stream, with
`final_eval_episodes` pinned at 30 and `best_metric_min_delta`
0.5/30 → 0.5/60. Promotion is a threshold crossing on a noisy statistic:
per-episode `bounce_count` sd is ~0.87, so a 30-episode batch has sd
~0.16 and a 3-eval window ~0.092 — and runs `20260724_152530` /
`20260725_171747` cleared the 3.0 bar at stage 1 by **0.011**, about a
tenth of a standard error. 60 episodes cut the window sd to ~0.065.

The threshold itself is deliberately unchanged. An earlier reading of
this evidence proposed lowering it, on the grounds that the sweep's
`ds._oracle` averages only 2.25–2.57 at every stage. That oracle is not
a feasibility ceiling: a predictive controller driving the *same* three
slide actuators scores **4.39** at stage 0 (max 8, ≥5 in 40% of
episodes, 49/200 episodes still alive at the 750-step cap). 3.0 is
reachable; the defect was the measurement, not the bar. The flat
~2.4-across-stages curve is likewise an artifact — each stage's oracle
probe is tuned individually and criterion 7 rejects any stage scoring
more than 1.5× its predecessor, so under one uniform controller the
stages read 4.39 / 2.85 / 1.52 at s0 / s2 / s4.

Cost: eval episodes per cycle go 60 → 90 (+50%). The three eval streams
already cost about as many env steps as training itself, so expect
throughput to fall roughly 15–20%; budget wall clock accordingly.

Metrics era: `bounce_count_ep_mean` keeps its meaning and gets less
noisy, but promotion *timing* is not comparable with 30-episode runs,
and no reward-bearing quantity is comparable across the gamma change.

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
