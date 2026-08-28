# Decisions & Lessons Learned

A running journal of the non-obvious engineering decisions, bugs, and dead ends
behind Courtside Dynamics — the institutional memory that would otherwise be
scattered across review docs, commit messages, and version notes. When a design
choice, a calibration value, or a "we tried X and it failed" finding is worth
remembering, it goes here.

**How to use this file**

- Newest entries first, within each themed section.
- Each entry names the **decision or lesson**, the **evidence** behind it (with
  the concrete numbers that make it actionable), and a **Status**:
  *implemented* (shipped, with the version), *open* (recommended but not done),
  or *characteristic* (a property to be aware of, not a bug).
- The deep source material lives in the review snapshots under `docs/` (see
  [`docs/README.md`](README.md)); this file distills their durable conclusions.
  Per-release migration notes live in the root [`CHANGELOG.md`](../CHANGELOG.md).
- **Status caveat:** WallBall entries were tracked through implementation
  (0.9.0 → 0.13.0) and their statuses are current. The humanoid-tennis entries
  are from the v0.7.0 review; spot-checks against the current tree are noted
  inline, but they have **not** been fully re-audited — treat "open" there as
  "open as of 0.7.0, not since disproven."

---

## Cardinal rules

The meta-lessons that keep recurring across the project. Most specific entries
below are an instance of one of these.

1. **Fail loud, never silent.** This repo's history is a catalog of silent
   no-ops: a `VecEnv.set_attr` that wrote a shadow attribute instead of the real
   one (a whole training run at the wrong curriculum value), silent normalizer
   fallbacks that record garbage videos of a competent policy, an inert
   `clip_reward`, `git_sha` quietly `null`. Every config loader, callback, and
   schedule should raise or log its applied value, not swallow the failure. The
   run-config system ([spec](run_config_file_spec.md)) was built on this
   principle.

2. **Keep the reward ladder strictly monotone at the episode level.** The single
   most expensive failure mode here: every pre-scoring behavior — do nothing,
   touch and miss, weak return — netting *exactly* `-1.0` after claw-backs, so
   SAC saw a dead-flat valley with no episode-return gradient toward contact.
   Before launching a run, verify `parked < weak-swing < full-swing < oracle` as
   distinct returns (the 0.11.0 bootstrap package measures exactly this ladder).

3. **Prove solvability with a scripted oracle before scoring against a bar.**
   A `bounce_count >= 2` success threshold was treated as a skill gap for
   thousands of steps when part of it was geometry. Gate any new success bar on
   an oracle demonstrating it is reachable *from the evaluation distribution*
   (standard serve, `recovery_reset_probability = 0`).

4. **Make model-selection and early-stopping noise-proof.** A "best" checkpoint
   was once crowned on ~1e-8 of floating-point telescoping residue, which reset
   the patience counter and kept a provably-dead run alive for 500k extra steps.
   Quantize selection keys to `1/n_eval_episodes`, require improvement `> ε`,
   drop `episode_reward_mean` from the selection keys, and add a degenerate-signal
   guard (zero variance + zero paddle contacts ⇒ stop).

5. **Don't carry hyperparameters across an action-space or physics change.** The
   fixed `ent_coef` that "worked" was tuned on the legacy 5-action WallBall; on
   the 3-action env it disabled auto-entropy and helped block learning entirely.
   Re-validate exploration/entropy config whenever the action space changes.

6. **Match curriculum fragments to the distribution the policy actually
   creates/faces.** 60% of one run trained on synthetic "recovery" balls
   (bounce x ≈ −0.6, centered y) the policy almost never faces, while the balls
   it *does* create (bounce x ≈ 2.1, |y| up to 3.7) were barely practiced.
   Fragment skill did not transfer to serve receipt at all.

7. **Isolate side-effectful callbacks so they cannot kill a run.** A missing
   headless-GL backend made the first scheduled video recording raise
   `mujoco.FatalError` out of `model.learn()`, losing `final_model.zip`, the
   final eval, and `stage_summary.txt` ~45 min in. Recording (and any optional
   diagnostic) must be try/except log-and-continue.

8. **Change one variable at a time, and run ≥3 seeds before calling something a
   baseline.** One regression changed five things at once (success threshold,
   obs dim, reward channel, reset distribution, lane front) on a single seed —
   nothing about it was attributable.

---

## PaddleTennis — the rally campaign

Source: the dated `paddle_tennis_*` docs and `design_paddle_tennis_*`
verdict sections indexed in [`docs/README.md`](README.md); this
section distills the campaign's era verdicts (2026-08-15 → 08-23;
the earlier exploration/diagnosis-era verdicts remain in their
source docs' status lines).
The P0–P2 court-scaling entry predates the campaign and stays in the
WallBall section below.

### The review snapshot corrects the closing record; verdicts stand — *booked (2026-08-23)*
Independent re-derivation of every closing number
([`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md)):
RK1/KH1 FAIL and the hold-line closure all reproduce, but §4c's
extension observation had missed that k=2 receiving read **5% at the
2.5M probe** — the era's strongest reading, at the PASS-bar values,
booked as noise floor. Fresh-seed re-measurement: 1.6% [0.3, 4.5]
(real but small, seed-conditioned at its peak, not via stillness —
thrash unchanged at 92% saturation, hold dose +0.211 paid and all
clawed back in the audited trace). Also measured: **MuJoCo
3.11 ≡ 3.12 on this task** (digit-exact deterministic replays), so
LH1c's record eval belongs to the run, not the physics; and
task-metric selection does not track k=2 — the crowned best (2.325M)
is not the k=2-richest checkpoint (2.5M). Lesson 13a bit again in
draft (caught by the review's own verification pass, commit
`ec0606a`): the first draft's pooled estimate blended the seeds that
nominated the checkpoint with the fresh ones; only the fresh arm
scores.

### The post-swing wander is commanded thrash, not drift — *diagnosis (PT1, 2026-08-22)*
`data.ctrl` replay of the best checkpoints: commanded XY path 3–4×
the paddle's actual travel (31.0 vs 7.8 m; oracle ratio 1.0), 88–91%
of post-swing steps saturated (|action| > 0.9; the 2026-08-23
review's 2.5M row extends the band to 93%), 0.26–0.38 m/step
command saccades, no attractor (≈4 m from hit point and home, 5–6 m
from the ball). The dynamics-side hypothesis is dead; the action head is
emitting bang-bang under a dead temperature (ent_coef ~1.7e-4, std
~0.014). Era law refined: **paid windows organize only where
exploration can find the paying behavior** — the hold escrow needed
~80 consecutive near-zero commands from a saturated head with no
noise left to discover stillness. Routes: temperature-skip warm
start first, an interface-side command-rate/low-pass treatment
second (a new comparability era), no further escrow scale.

### The hold-escrow line closes without adoption — and a shaping term must pay where the policy already is — *closed (2026-08-22)*
Three pilots (`design_paddle_tennis_postswing_hold.md` §4–§4c): LH1
(0.25, 4.0 m) and LH1b (0.5, 4.0 m) delivered **exactly zero
gradient** — the 4.0 m travel budget was a cliff entirely outside
the policy's 6–10 m band, proven by LH1b training byte-identical to
LH1 (doubling a zero); the design error is recorded plainly: choosing
the budget so the measured wander "pays zero" is precisely what makes
it unlearnable. LH1c (0.5, 12.0 m — ramp moved onto the occupied
band) delivered the dose and the policy kept the wander (registered
1M window: KH1 FAIL, hold travel back to 8.1–8.3 m). Honest
counterpoint, unregistered extension: campaign-record eval **+2.483
at 2.425M**, the first warm-started run whose best came from deep
training, k=1 receiving to 90–98%, and (per the 2026-08-23 erratum)
a small real late k=2 rise by a non-hold route. Kwargs stay in the
env, default off, certified farm-proof.

### The registered n-point run: from-scratch bootstrapping proven, k=2 is the standing blocker — *adjudicated (2026-08-22, run 20260816_235141)*
LS1 PASS (the adopted task trains from scratch — the campaign no
longer depends on the transfer lineage), RE3 PASS (k=1 receiving to
95%), M PASS (77.5% of evals), RE1/RS2 middle, **RK1 FAIL** (k=2
≤ 1% at every checkpoint — 1% on eleven, 0% on nineteen) → the
stop/amend branch fired into the hold-escrow line. R2 watch read
7.5–9.1 m all run: the post-swing window was the one unpaid segment
of the k=2 chain. Held-out block 4100–4199 not opened; stays sealed
for the first registered-result branch.

### Reach shaping fires the ADOPT branch; the recipe enters the n-point era — *implemented + adopted (2026-08-16)*
LR1 (escrowed reach at 0.25, radius 3.0, warm-started): first
positive learned final eval (+0.14), k=2 above zero at five
checkpoints, touch 41%, ready error 1.08 m. Recipe defaults since
the adoption commit: `points_per_episode=None` (continuous n-point),
`contact_shaping=0.25`, `reach_shaping=0.25`, side-A-only
success/guards (`legal_hit_count_a`, `best_metric_min_delta=0.25`,
`confirm_best_eval`, 5-eval degenerate stop). Env defaults stay
frozen (everything default-off at env level).

### The n-point L2 verdict: continuous play un-samples contact from scratch; transfer collects the credit — *stop/pivot (2026-08-15)*
Both from-scratch n-point pilots collapsed to zero contact (3.1M
cumulative steps, one shot in bounds, ever). The validation battery
localized it: the reward pays competence (oracle +11.6 vs statue
−4.4 in-mode) but contact is **undiscoverable by exploration noise
at any std** (only full-range held-64 uniform touches, ~1/10k
steps), noise is locally punished, so entropy collapse was the
correct local gradient; the volley trap runs 3–10× (ball-chasing
toucher −52/episode). L2W then proved the design's inter-point
credit mechanism by transfer: the k=1-mastered checkpoint scored
+1.92 with **zero gradient updates** — and training regressed it
(k=1 100% → ~30% equilibrium). Chronic α-collapse is on the record
here too: 0.02 → 9.8e-4 inside 12k steps; appendix D.6's
tanh-saturation mechanism (saturated squashed mean inflates latent
−log π, annealing α to zero) empirically falsifies the target −1.5
rationale. At the verdict there was no warm-start temperature-skip
flag in code; it shipped 2026-08-23 as
`WarmStartConfig.transfer_log_ent_coef`, with LT1
(`paddle_tennis_lt1_prereg_20260823.md`) as its pre-registered
pilot (frozen 2026-08-28).

## WallBall — reward, curriculum & geometry

Source: [`wall_ball_baseline_review.md`](wall_ball_baseline_review.md) (runs from
2026-07-14, reviewed at `cdb17d4`/v0.9.0, with addenda through the 0.11.0
package). See also `CHANGELOG.md` 0.9.0 → 0.13.0. Depth-campaign entries
below distill [`wall_ball_rally_diagnosis_20260728_review.md`](wall_ball_rally_diagnosis_20260728_review.md).

### PaddleTennis P0–P2: premise measured, court scaled to paddle physics, loft is the frontier — *probe result (2026-08-02)*
The pivot's premise is now a measurement, not an argument: on a
two-paddle prototype (wall-ball calibrations verbatim, net at x=0),
in-rally strokes land **mean 3.3 m past the net, 50% deeper than
3 m** — the depth distribution the wall never produced. Two design
facts came out non-obvious: (1) the **full-scale ITF court is
infeasible** at the paddles' ~12.5 m/s ceiling (nothing clears from
11.9 m at any net height) — the era's court freezes at half-length
6.5 m with the regulation 0.914 m net, placing baseline strokes
exactly at the power ceiling; (2) with a fixed-pitch face, **depth
control spans a knife's edge** (hard ↔ long, soft ↔ net), and a
strike-height offset (~0.12 m below ball center) measurably widens
it — the first quantitative sign that paddle-pitch actuation may
graduate from parked consistency lever to era prerequisite, a
question for the first learned runs. Scripted reference band:
2.0 crossings/point, max 10, ≥4 in 33% (bimodal — half of points die
on serve-return, the wall-ball pattern repeating). P3–P5 still gate
env code. Details:
[`paddle_tennis_probes_20260802.md`](paddle_tennis_probes_20260802.md).

### The replication splits — reliability 2/2, era skill 1/2 — and the volley loophole closes the wall-ball chapter — *result (run 20260801_144043); chapter closed*
Seed 1 ran its full 6M budget, out-scored seed 0 everywhere (best
window 3.072 — the first ≥3.0 window in campaign history, essentially
uncompressed; uncapped audit 3.08; champion banked at literally step
6,000,000 and confirmed at 3.417, the project's highest), and **failed
the deep-receive criterion completely**: measured first contact mean
x = −4.49 (bar ≤ −6.0), 0% of episodes deep. Mechanism: in 48% of
episodes it *volleys the serve out of the air* at −3.4 before the
ball bounces; the rest intercept the post-bounce leg early at −5.5.
Two seeds, two coherent opposite styles — and the style that refuses
the era's skill scores higher. Training reliability is settled at 2/2
(no basin, no collapse, 384 evals). The durable lesson: **an
environment's dominant strategy is set by its physics, not by reward
or rule decree** — the wall only hits short (rebounds mean −1.0,
never deeper than −6.1), so depth can never be instrumentally useful
against it, and every decree that forces it (fences, serve energy,
style bans) invites the next exploit. Decision: the 0.26.0
loophole-closing and episode-cap changes are **rejected as moot**, the
wall-ball chapter closes with its deliverables banked (calibrated
physics, the probes-first methodology, 2/2 unattended training
safety, two stylistically opposite champions), and the campaign moves
to opponent play, where deep balls exist because someone hits them:
[`design_paddle_tennis.md`](design_paddle_tennis.md). Details:
[`wall_ball_true_baseline_replication_20260801_review.md`](wall_ball_true_baseline_replication_20260801_review.md).

### The true-baseline era opens on the first run — and the episode cap becomes the measured ceiling — *result (run 20260731_132322)*
The first `WallBallTrueBaseline` GPU run passed every pre-registered
primary criterion: best window 2.856 (bar 2.0), uncapped audit **3.02
mean completed returns** (100% of episodes ≥1, 98% ≥2, max 9), and a
measured mean first contact of **x = −6.49** (bar ≤ −6.0; 98% of
episodes deeper) — 2.8 m deeper than the goal era, with zero opening
volleys. Band-relative it is the project's strongest policy (1.53×
its oracle band vs 1.31–1.39× for the goal-era seeds). The champion
banked at 2.1M; early stop at 3.6M with the final policy off-peak
(1.98) — best-model banking again saved the result. The stretch (a
≥3.0 window) was NOT met, and three independent observations pin the
cause on the 750-step episode cap, not the policy: eval max rallies
sat at *exactly* 5 for ~120 consecutive evals (750 ÷ 156-step cadence
= 4.8 exchanges), timeout terminations tracked the ≥5-rate
one-for-one on strong evals, and 14% of uncapped audit episodes
outran the cap (longest 1,487 steps, 9 returns). Lesson for future
eras: when cadence changes, the episode cap silently rescales the
headline metric's ceiling — re-derive `episode_len` from cadence
(target ~6 exchanges) whenever geometry moves. `episode_len` 750 →
~1,100 is the pre-registered 0.26.0 candidate, to land between runs;
the seed-1 replication runs first, unchanged. Details:
[`wall_ball_true_baseline_20260731_132322_review.md`](wall_ball_true_baseline_20260731_132322_review.md).

### The true-baseline era task is probe-frozen and certified before any run — *implemented (0.25.0)*
The extension to the ITF baseline (workspace (-8.2, 0.3), fence
(-8.2, -2.6), start -7.9, home -5.4, serve 11.0, in-play bound -10.0 —
`WallBallTrueBaseline`) was measured before it was built, and three of
the measurements overturned the "obvious" design:
- **Serve speed is the depth lever, and it has a floor.** Landing
  scales ~0.5 m per +1 m/s from origin 1.0; below 10 m/s the ball dies
  short of a paddle at -7.9, so the goal-era serve (7.0) is
  unreturnable there because it *never arrives*. At 11.0 the
  post-bounce leg crosses -7.9 at 0.54 m and 9.4 m/s in 100% of
  episodes — reachable, and (from -7.9) impossible to intercept
  pre-bounce, so the deep receive is forced without any fence trick.
- **A baseline-only fence is geometrically unsustainable.** Rebounds
  off legal deep returns land at mean x = -1.0 and are *never* deeper
  than -6.1 — the rally always comes forward. The era fence adds the
  deep band to the goal fence instead of relocating play.
- **The in-play bound is task geometry, not a constant.** The
  historical OOB edge (-6.0) sits 1.9 m in front of the new paddle
  start: the serve terminates out-of-bounds before the receiver can
  touch it (measured 60/60 dead serves). `ball_in_play_min_x` is now a
  per-task kwarg (default -6.0, bit-for-bit historical), -10.0 here.
- The certification feasibility floor became a spec knob
  (`feasibility_ge2_floor`, default 0.90): no scripted reference
  dominates an 11.8 m court (oracle band mean 1.98, >=2 rate 67%,
  cadence 156 steps), so the recipe declares 0.50 = band minus two
  30-episode sampling sd, stamped into every report. Held-out
  certification on fresh seeds 4000-4099 passes every blocking
  criterion (oracle 61%, CI [0.51, 0.70]). Probe-design lesson en
  route: score the serve by *crossing-state kinematics* at the receive
  depth, not by arrival at a parked face — the post-bounce hop
  (apex ~0.6 m) passes under a parked face (bottom 0.95 m) and reads
  as a false 0% reachable.
  Details: [`design_wall_ball_true_baseline.md`](design_wall_ball_true_baseline.md).

### The default action mapping must never be derived from the XML — *implemented (0.25.0)*
Widening `paddle_slide_x`/ctrlrange for the workspace extension
silently rescaled the **default** x action mapping (action -1 went
from world -4.7 to -8.2) for any env built without an explicit
`paddle_x_target_range` — caught by the bit-for-bit default-config
guard test, which exists for exactly this. The default mapping is now
a frozen module constant (`_DEFAULT_PADDLE_X_TARGET_RANGE =
(-4.7, 0.3)`); the extended workspace is opt-in per recipe. Verified
by hashing full obs/reward/termination streams across HEAD vs the
extended tree on four legacy configs — all bitwise identical. The
general rule: physical capability (XML ranges) and action semantics
(the mapping) are separate contracts; growing the first must never
move the second implicitly.

### The campaign goal is met: a sustained rally from the workspace baseline — *result (run 20260728_225217); replicated 2026-07-30*
**Replication update (runs 20260729_140112 / 20260730_005134):** 2 of 3
seeds pass, and the third success beats the first everywhere (best window
3.750 vs 3.311; audit 3.76 mean / median 4 / max 11 / 32% >=5-survival;
84% of audit episodes now end with the ball running out AFTER a rally).
Seed 1 exposed a new failure mode — one-and-done basin capture for 1.4M
steps, unaided escape to 1.45, then full collapse to zero paddle contact
at ~2.5M, ended by the degenerate-signal guard at 3.175M with the
pre-collapse champion banked. All measured policies play DEEP (contact-x
-3.4 to -3.95; nobody camps the fence front), settling that the
fixed-pitch face suffices for deep rally play. Reliability (~2/3), not
capability, is the recipe's open item. Details:
[`wall_ball_goal_rally_replication_20260730_review.md`](wall_ball_goal_rally_replication_20260730_review.md).

Original seed-0 entry follows.
The first `WallBallGoalRally` run resolved the campaign: goal-task eval
sustained `bounce_count_ep_mean ≥ 3.0` over 3×60-episode windows from 1.25M
steps (best window 3.311; best model 3.333 confirmed 3.267 at 1.325M), and
the 50-seed long-horizon audit reads **3.54 mean completed returns** (median
3, p90 6, max 8; ≥5-survival 26%; 100% of episodes complete ≥1 return; zero
opening volleys, 95% post-bounce play) — versus 1.14 and
every-episode-double-bounce for the best ladder policy on the identical
task. Every pre-registered success criterion passed including the stretch
bar; early stop ended the run at 1.825M of 6M (~6.6 h) on a settled plateau.
The bar that five curriculum runs treated as an unreachable promotion gate
was a reachable performance level once training happened on the task itself.
Caveats: seed 0, n=1 (second seed is the standing next step); plateau
~3.0–3.3 with the ≥5 tail at 26% — consistency, not competence, is the
remaining headroom on this geometry. Snapshot:
[`wall_ball_goal_rally_20260728_225217_review.md`](wall_ball_goal_rally_20260728_225217_review.md).

### The sliding-fence depth ladder is retired; the goal task is trained directly — *implemented (0.24.0); vindicated by run 20260728_225217*
Three ladder generations (0.15/0.19-0.21/0.22) stalled the same way: a flat
`bounce_count_ep_mean >= 3.0` bar that **no scripted reference reaches at any
rung** (~70 controller configurations swept in the diagnosis, best 2.70;
held-out oracle means 2.47-2.89) and that four multi-million-step learned
runs plateaued 0.2-0.9 below at every rung past stage 0 (0.22.0 stage 1:
201 evals, best 2.82, still climbing at 6M; 0.19-aligned stage 2: 0 of 128
evals >= 2.9). Meanwhile each promotion moved the serve landing +0.35 m —
above the measured ~0.25 m approach-room threshold — so every rung swapped
the receive task at the exact moment the 0.22.0 advance package deleted the
buffer and re-randomized the policy (measured cost: train reward below
untrained levels for ~250k steps and not back to pre-advance level until
~5.3M of a 6M run). The replacement (`WallBallGoalRally`) trains the
goal task directly — the constant-width 0.22.0 fence had already restored
crude learnability there (80-84% placement-blind two-return rate vs
9.5-16% on the retired narrow fence), which no run ever saw because none
trained past stage 1. A serve-origin curriculum (the one axis measured to
control receive difficulty, with rungs that change only the reset
distribution) was designed, certified, and then *tested against direct
training* in the pre-registered paired local SAC A/B/C: direct goal-task
training won — 2.03/2.71 completed returns on 100 held-out seeds at 500k
local steps (all-time ladder best: 1.14 from 6M Colab steps), with a
serve-origin mixture no better (2.27) and an aligned-serve entry rung
strictly worse on goal skill (1.09-1.51, and near-zero zero-shot transfer
for most of training). **Two lessons:** (1) when a curriculum's guard
rails outlive the hazard they guarded (the 0.22.0 geometry fix removed
the goal-task learnability collapse), the curriculum itself becomes the
obstacle — re-test the direct task after any geometry change; (2) if a
curriculum axis is ever needed again, pick one that leaves replayed
experience valid across rungs (reset-distribution changes, not dynamics
changes) — the fence axis taxed every promotion with off-policy
staleness by construction.

### Serve alignment is falsified as a learning aid — easier receives train weaker policies — *closed (0.24.0)*
The serve-alignment hypothesis (move the serve landing to the paddle so
receive is easy; `design_wall_ball_serve_alignment.md`) was tested at both
of its pre-registered sites in the diagnosis campaign's paired local SAC
battery. Stage-1 A/B: pooled paired Δ −0.34 against alignment, both pairs
negative, and the 2×2 cross-eval shows baseline-trained policies beating
aligned-trained ones on BOTH serve geometries — including the aligned
serve itself. Goal-fence A/B/C: the aligned-entry arm was the worst of
three conditions on goal-task skill (held-out 1.09-1.51 vs 2.03-2.71 for
direct training). Mechanism: approach-to-receive training generalizes
down to easy receives; easy-receive training does not generalize up — the
alignment removes exactly the skill the campaign needs. **Do not
re-propose serve-origin alignment (full or partial) as a training aid;**
the λ-blend feasibility maps remain valid as probe-calibration reference.

### A promotion bar must be calibrated against references on the rung it gates — *implemented (0.24.0)*
The 3.0 bar was calibrated once, on the retired 0.21 stage 0 (a predictive
controller scored 4.39 there), then inherited by geometries where the
scripted ceiling is ~2.5-2.9 and SAC's own multi-million-step asymptotes
are 2.1-2.8. The standing "do not lower the bar on scripted-oracle evidence
alone" rule was honored: no bar was lowered — the gated ladder was retired
outright, and `WallBallGoalRally`'s single-stage gate keeps 3.0 as a purely
informational campaign-goal marker (`curriculum/gate_window_mean` crossing
it in TensorBoard) that can never gate anything. Note the arithmetic that
made 3.0 unreachable-in-practice as a *promotion* bar: at the goal cadence
(~125 steps/exchange, measured) `episode_len = 750` caps an episode at ~5
returns, so 3.0 demands ~60% of the physical cap sustained across a
3x60-episode window.

### Promotion staleness needs its own clock — *implemented (0.24.0)*
Run `20260727_233859` sat on stage 1 for 201 evaluations (~5M steps) with
no mechanism able to notice: early stopping watches best-model improvement
(and noise kept resetting it via a `best_metric_min_delta` ~1/13 of a batch
SE), while promotion staleness was unmonitored. `stage_eval_budget` on the
gate now bounds a non-final stage's residency: `"stop"` ends the run (the
right default when the bar is load-bearing), `"advance"` force-promotes
with `advance_reason: "stage_eval_budget"` recorded in
`curriculum_stages.json` (the right choice when rungs are benign
reset-distribution shifts and the anneal must complete inside the budget).

### Scripted probes must lead the intercept, and their gaps are seed-sensitive — *implemented (0.24.0)*
The 0.21-era certification probes tracked the ball's current y/z while
charging; on the constant-width ladder that scores 36%/70% two-return rates
at stages 0/3 — probe defects indistinguishable from geometry defects until
recalibrated. The `lead_charge` mode (same commit trigger, ballistic y/z
lead at the closing-speed intercept) clears 90% everywhere; gaps were
frozen at n=200 by rate-with-margin rather than raw argmax (calibration
argmaxes shed 3-5 points held-out here, twice measured), then certified
once on held-out seeds 3000-3099 at 95-100%. Startup certification runs 30
episodes/cell, not 10: a 0.90 floor against true rates of 92-98% flips on
one unlucky seed at n=10.

### The 0.11.0 bootstrap package — a strictly monotone competence ladder — *implemented (0.11.0)*
The fix for the flat `-1` valley. On the stage-0 serve (n=120) the ladder now
reads **parked −1.00 < weak-swing tracker −0.85 < placement-blind full swing
+7.63 < oracle +12.07**. Mechanisms: a once-per-episode outright
`first_hit_bonus` (0.25); a `weak_return_penalty` (0.1) fined retry replacing the
terminal weak-return fault; early-touch softening; a performance-gated serve
ladder (`serve_vy_max` 1.1 → 2.0, gate `bounce_count_ep_mean ≥ 1.3` sustained
2 evals) with matched-stage selection; and the previously-untried exploration
config (`ent_coef="auto_0.02"`, `target_entropy=-1.5`, `learning_starts=10k`,
buffer 500k). Adversarial review closed two high-severity exploits before
commit: repeat paddle taps no longer reset the stall clock (touch-then-deaden
possession was a risk-free +0.25 ride to truncation), and a stage advance now
resets best-model selection state (an easy-stage score otherwise permanently
barred better final-stage policies, ~0.6–0.7 metric inflation). The no-op
invariant holds: a parked paddle still scores 0 contacts and −1.

### Geometry: the old lane was unforgiving, not unsolvable — *implemented (0.10.0)*
The chosen bundle: baseline lane `(-3.2, -1.6)` + paddle slide damping `8`
(new `paddle_joint_damping`, baseline recipe only) + `early_touch_penalty 0.25`
(non-terminal fine replacing the terminal `paddle_before_bounce` fault). The two
geometry changes only work *together* — either alone collapses oracle second
returns to ~50%. Calibration sweep (config → oracle ≥2 / crude-tracker ≥2):
old `(-2.1, damping 5)` → **0.95 / 0.00**; chosen `(-1.6, damping 8)` →
**0.92 / 0.70**. `(-1.4, 8)` scored marginally better for crude play but put the
whole serve-bounce footprint inside the lane (94% pre-bounce faults for a naive
front-camper vs 39% at −1.6). The earlier "~73% physically unsaveable" framing
was a *correction target*: that figure is for untouched rebounds from fast/flat
returns; a retreating oracle always recovered ≥2 from 95% of standard serves.
**Lesson:** distinguish "unsolvable" from "unforgiving of mediocre play" — the
crude-tracker column, not the oracle column, is what RL has to climb.

### Difficulty ladders that were swept and rejected — *rejected*
- **Depth ladder:** difficulty vs distance-to-wall is U-shaped, not monotone.
  Close-court returns hit the wall hot and rebound out (oracle ≥2 collapses to
  ~50%); mid-court to baseline is flat. Moving the paddle closer does not ease
  the task.
- **Serve-pace ladder:** return power rides on incoming momentum, so *slower*
  serves underpower returns (oracle first-returns fall from 100% at serve speed
  5.5 to 12% at 3.5). Slowing the serve makes it harder, not easier.

### The recovery-schedule bug: forward attr writes through the wrapper stack — *implemented (P0)*
`LinearEnvAttrScheduleCallback` called `training_env.set_attr(...)`, which under
SB3 2.9.0 / gymnasium 1.3.0 `setattr`s the outer `Monitor` wrapper — a shadow
attribute the `WallBallEnv` property setter never sees. One run trained at
`recovery_reset_probability = 0.6` for all 750k steps instead of tapering to
0.15, and `get_attr` read the shadow copy so probes were masked too. Fix:
`env_method("set_wrapper_attr", name, value)` (walks the stack), log the applied
value each eval, and integration-test *through* `make_vec_env` + `Monitor`, not a
fake recording VecEnv. **General rule:** never `set_attr` a semantic env
attribute on a VecEnv; go through `set_wrapper_attr`/`unwrapped`.

### Reward channels must actually pay out on the real distribution — *open (P1)*
`recoverable_bounce_bonus` graded the bottleneck skill (return placement) with a
hard `clip(1 − |proj_y|/2.0, 0, 1)` — but real rebounds run to |y| = 3.7 against
the 2.0 clip, so it paid **exactly zero in 76%** of completed returns (~0.2% of
per-cycle reward, zero slope where it mattered). Recommended: smooth kernel
(`exp(−|proj_y|/limit)`), raise the lateral limit 2.0 → 3.0, raise the bonus
0.25 → ~0.75 to be commensurate with the +1.0 wall bonus.

### Keep `ent_coef` adaptive; exclude binary/counter obs dims from normalization — *implemented (0.11.0)*
Fixed `ent_coef = 0.02` disables SAC auto-entropy (the recorded
`target_entropy = -3.0` is inert for a float coefficient) and was carried over
from the legacy 5-action env. `learning_starts` was never raised off the SB3
default of 100 (batch-256 updates began after ~104 transitions, VecNormalize
fit on <2 episodes), and WallBall normalized binary flags/counters (obs dims
12, 13, 20, 21, 22) whose base rates drift with the curriculum, even though
`SelectiveVecNormalize` already supports `normalize_obs_excluded_indices`.
Shipped in the bootstrap package as `auto_0.02` / `target_entropy=-1.5` /
`learning_starts=10k` / buffer 500k.

### Make each additional exchange worth strictly more than the last — *open (P1)*
In the "one-and-done" run, kept shaping was +1.74 = **87.9%** of net return; the
−1.0 double-bounce penalty was simply priced in against ~+3.0 of one-cycle
income, so a second exchange was never worth visiting (max training return in
1,152 episodes: 2.04). Recommended: cut `track_shaping_scale` 0.5 → 0.2 and make
the wall reward escalate per exchange (`1.0 + 0.5*(bounce_count − 1)`, capped
~3.0).

### The "one-and-done" degenerate policy — *diagnosed; remediated by the bundle above*
Signature to recognize: converges by ~225k steps to exactly one legal hit, one
wall return, then lets the ball double-bounce; rollout CSVs at 250k/500k/750k/1M
are behaviorally identical (froze at 250k); 50/50 long-horizon episodes show
exactly one return. It is not one bug — it is the confluence of rules 2, 3, and
"each exchange must pay more" above.

---

## Training harness & instrumentation

### Certification artifacts must be derived from the live spec, never maintained in parallel — *implemented (0.23.0)*

The 0.22.0 ladder shipped without a calibration sweep because
`tools/depth_stage_sweep.py` carried its own copy of the stage tables and
nothing forced that copy to track the recipe; the drift surfaced only when
review `20260727_233859` re-swept the live geometry after a 17-hour run
stalled at stage 1 (oracle clears the 3.0 bar at **no** stage; feasibility
fails at stages 0 and 3). Evidence that parallel tables rot: the tool's
pivot (−1.7) and fences were two releases stale, `wall_ball_oracle_action`
still inverts the retired fixed-pivot map, and the recipe's own comment
carried share numbers matching nothing. Since 0.23.0, gated runs
self-certify at startup — stages read from `performance_gate`, envs built
by the run's `env_fn`, verdict written to
`reports/ladder_certification.json` — and stage-application integrity
(attribute exists + round-trips) is a blocking criterion, closing the
`paddle_home_x` silent-no-op class for good. Certification is advisory
until the probes are recalibrated for the constant-width geometry:
enforcing criteria that the stock probes are known to fail would brick
legitimate runs.

### An argmax over a logged eval stream is a hypothesis, not a result — *characteristic*
Second occurrence in this campaign, so it is a pattern rather than a
mishap. Run `20260727_004014`'s unsynced goal-geometry stream made the
late-run region look better than the selected checkpoint (8-eval trailing
means: 1.163 at the selected 4,650,000 vs 1.283 at 5,750,000 and 1.254 at
6,000,000), and the run was called wrong for having selected on
matched-stage geometry. Re-scored on **200 fresh paired seeds**
(`design_wall_ball_checkpoint_selection_audit.md`), `final_model` came
back **worse** (−0.080, t = −1.80) and the best candidate was not
significant (+0.085, t = +1.62, p ≈ 0.11 before any multiplicity
correction). The trailing means had been computed from a 30-episode
estimator over the same 240 evaluations the maximum was drawn from. The
first occurrence is in `plan_wall_ball_aligned_deep_stages.md`: a control
arm's calibration argmax (94.0%) fell to 89.0% held-out and failed while
the stale shipped value passed at 94.0%. *Rule: re-score the candidate
region on seeds no selection has touched, pair the arms on identical
seeds, report the paired statistic, and size the re-score against the
effect claimed — 3–5 point calibration-to-held-out swings at n = 200 are
routine here, which is the same order as every margin that has mattered.*
Two useful side effects of that audit: `best_model.zip` **stands** (no
re-selection), and four checkpoints spanning 1.35M steps all landing at
1.07–1.30 returns is direct evidence the depth plateau is structural, not
a choice-of-snapshot artifact.

### Per-stage promotion price, not total budget, is what binds the depth campaign — *characteristic*
Run `20260721_004722` reached stage 2 of 4 on 3M steps; run
`20260724_152530` was still on stage 1 at 2.55M of a 6M budget, with
stage 0 costing 1,425,000 steps in *both* (within 0.02%). Doubling the
budget did not move the ladder position, so the lever is the cost of each
promotion, not the total. Two of the three components are self-inflicted:
promotion-shock recovery (~0.5–1M steps per advance) and gate latency.
Note also that `20260724_152530` cold-started (`warm_start: null`)
despite run 1's review making warm continuation its P0 #1 and 0.15.1
shipping SAC warm start — so it re-paid stage 0's 1.43M-step bill for
nothing. **Lesson:** for a gated ladder, measure steps-per-promotion and
project it against the budget *before* launching; a cold start on an
already-mastered stage 0 is ~24% of a 6M run spent re-learning a solved
problem.

### Resetting the entropy temperature on advance — *implemented (0.20.0, default off)*
`train/ent_coef` sat at 0.0007–0.0009 from ~100k steps onward in run
`20260721_004722` and never re-inflated after a promotion, so the policy
met each new geometry very nearly deterministic. The warm-up package made
that worse: `advance_update_pause_steps` sets `gradient_steps = 0`, which
freezes `log_ent_coef` along with everything else, so the tuner could not
recover *during* the pause either. Run 1's review filed this as a watch
item gated on "if a de-noised continuation still recovers slowly from
stage shocks"; `20260724_152530` fired it (stage 1 >1.4M steps, vs 0.97M
for the same stage in run 1). Shipped as the `reset_entropy_on_advance`
gate key. Two implementation notes worth keeping: clearing the Adam moment
buffers matters as much as restoring the value (stale moments push the
restored coefficient straight back down), and this restores *pressure*
to re-expand entropy — sampling spread lives in the policy's learned
`log_std` and re-expands over subsequent gradient steps, so expect a ramp,
not a step.

**Default-on for both depth recipes from 0.22.0.** The lever shipped in
0.20.0 and was then never exercised: run `20260727_004014` finished with
`train/ent_coef` at **0.0011** after 97 evaluations on stage 3, so every
one of its three advances still handed new geometry a policy with no
exploration budget. Legal there because `model_kwargs` pins only `gamma`,
leaving SB3's `ent_coef` on `"auto"` — a recipe that pins a float
coefficient cannot use this key.

### A sliding fence turns a growing slab of the action range into a zero-gradient plateau — *characteristic; remediated (0.22.0)*
`_action_to_controls` clamps the *target*, not the mapping
(`wall_ball.py:1097-1105`) — correct, and it keeps action semantics fixed
across stages. The consequence, unrecorded until now: with
`paddle_x_target_range = (-4.7, 0.3)` and `paddle_home_x = -1.7` the map
is piecewise linear about `a = 0`, so the fraction of the x-action range
that saturates against the fence grows **33.3% → 52.5% → 64.2% → 70.0% →
71.7%** across stages 0–4 (live band shrinking 66.7% → 28.3%). Inside a
saturated region `∂target/∂a = 0`, so Q is flat and the actor's
reparameterized gradient vanishes. The policy a fence slide clamps is
exactly the front-camper the ladder exists to break, and once clamped it
has no gradient pointing deeper — it can only escape on action noise,
which by then is nearly gone (see the entropy entry above). This is a
**third** promotion-shock mechanism alongside the two already recorded
(stale replay buffer, unobservable fence).

**Remediated in 0.22.0**, after run `20260727_004014` paid the predicted
price (three promotions, then 97 evaluations stalled on stage 3). Two
independent re-derivations reproduced the band above to the digit, which
is worth noting only because the mechanism was already on record here and
was rediscovered from the run data rather than read off this entry —
*check this journal before re-deriving.* The fix has two halves.
`paddle_home_x` became a property whose setter recomputes
`_control_home[0]`: it had been a plain attribute, so a stage that moved
it was a **silent no-op** (`set_wrapper_attr(..., force=False)` reports
success for any existing attribute) — cardinal rule 1, again. Each stage
now pivots the map on its own fence midpoint, holding the live band
roughly flat (0.49 / 0.43 / 0.42 / 0.48 / 0.63) instead of collapsing
0.67 → 0.28. `paddle_x_target_range` is untouched, so action *scale*
still never drifts across stages — the property this entry was written to
protect survives.

### Don't pool the `confirm_best` batch into the promotion window — *built, then rejected (0.20.0)*
Tempting and wrong. `confirm_best` re-rolls a full `n_eval_episodes` batch
on the current stage before dethroning a best, and the gate never saw it,
so folding it into the promotion window looks like free evidence — the
switch was written, tested, and reverted before shipping. The flaw: that
batch is **conditionally sampled**. It only runs when the primary batch
beat the running best, so pooling averages a deliberately-selected *high*
draw with an independent one (regressing it toward the mean) while leaving
low draws untouched. The window mean is therefore biased downward, which
silently *raises* the bar by an unquantified amount — precisely what run
1's review forbade ("do not lower the bar; de-noise the estimator
instead", and raising it invisibly is worse). Simulated at the campaign's
own numbers (true mean climbing toward a 3.0 bar, per-episode std 2.0,
n=30, 3-eval window mean, `min_delta` 0.5/30): mean window entry
**3.005 → 2.954** (bias −0.046), and evaluations-to-promotion **41.8 →
47.6** on a fast climb and **132 → 159** on a slow one — **+27 evals ≈
+666k env steps per promotion** in the regime Run A is actually in, on a
campaign whose binding constraint *is* steps-per-promotion. **Lesson:**
extra episodes only reduce variance when whether you collect them is
independent of what the first sample said. The honest way to buy gate
evidence is unconditional — raise `n_eval_episodes` or `sustain_evals`,
or pair the episode set (next entry). `last_confirmation_metrics` is still
published as diagnostic surface; nothing consumes it for decisions.

### Unpaired evaluation is the root of the gate noise — *open (P1)*
`InfoDictEvalCallback` calls `self.eval_env.reset()` with no seed and
never re-seeds (`info_dict_eval.py`), so **every evaluation draws a fresh
30 episodes**. Consecutive evals are therefore unpaired and the full
~0.4-bounce batch SE lands on every promotion decision and every
best-model comparison. The project's response was to widen the promotion
window (2 → 3 evals), which buys reliability at 75k steps of latency per
decision — treating the variance rather than removing it. Re-seeding the
*matched* stream to a fixed episode set each evaluation makes the
comparison paired (common random numbers): the variance of eval-to-eval
*differences* collapses, and `sustain_evals` could go back to 2. Keep the
final-config stream fresh-random, where an unbiased estimate is what is
wanted. Recommended alongside 0.20.0's `pool_confirmation_samples`, which
attacks the same SE from the sample-size side.

### Three eval streams, two of them on the same distribution — *implemented (0.20.0)*
Under headline selection a gated run stood up three periodic evaluators;
the reward `EvalCallback` and the final-config info-eval both rolled the
recipe's `eval_env_overrides` distribution (the gate re-syncs only the
matched evaluator). For run `20260724_152530`: 5 + 30 + 15 = 50 episodes
per 25k training steps, ~17,500 env steps of evaluation per 25,000 of
training at ~350-step episodes — and up to ~112% overhead on evals that
trigger a confirmation. `InfoDictEvalCallback` was already a strict
superset of the reward evaluator (it collects per-episode returns for its
own reward tie-break), so the duplicate is retired and the final-config
stream owns `evaluations.npz`. 0.20.0 also makes the single-worker
requirement explicit: the rollout loop reads `infos[0]`/`rewards[0]` and
counted worker 0 only, so a multi-worker eval env was stepped in full and
three quarters of it went unmeasured — now rejected at construction.
**Still open:** all eval envs are built `n_envs=1` (`train.py`), so
evaluation runs batch-1 and serial while training runs 8-wide. That is the
real wall-clock lever (~21,000 serial eval steps per 25,000 training steps
at the 0.20.0 episode counts), and claiming it means rewriting the rollout
loop to aggregate every worker — not passing a bigger env.

### `config.json`'s `train_config` is a hand-maintained allowlist — *partially implemented (0.20.0)*
`artifacts.py` enumerates the fields to serialize by hand, so a new
`TrainConfig` field is silently absent from every run's provenance
snapshot until someone remembers to add it. `reward_eval_episodes` — set
to 5 by two recipes and directly changing how much evidence a run's
reward stream collects — was missing for its whole life. 0.20.0 records
it and `final_eval_episodes`, and a test now pins that the block covers
every field except the four code-valued ones (`env_fn`, `eval_env_fn`,
`extra_callbacks`, `info_row_fn`) and the two recorded at top level
(`recipe_name`, `run_config_file`) — in both directions, so a derived
value cannot be smuggled into the block either. Adding a field without
recording it is now a test failure rather than a discovery made months
later while reading a run.

### `final_stage_index` reports the departing stage mid-run — *open (P3)*
`_write_stage_history` reads `self._stage_index` but is called from
`_close_stage_record`, which runs *before* the increment on an advance, so
a live or hard-killed run's `curriculum_stages.json` names the stage that
just ended. Run `20260724_152530` showed `final_stage_index: 0` while
`best_model_meta.json` recorded `curriculum_stage_index: 1.0`.
Self-corrects at `finalize()`; only misleads while a run is in flight.

### Instrument for post-hoc attribution — *mostly open (P2)*
Recurring pain: Monitor CSVs persisted only `r,l,t` (no `info_keywords`), so
training-time terminations couldn't be attributed after the fact; `*_ep_mean`
eval aggregates are means of *final-step* values (instantaneous flags read as
nonsense); `one_bounce_recovery_count` incremented on every first paddle hit
(reads ~1.0 with zero rallies); rollout/video used a single fixed seed
(n = 1 behavioral record per checkpoint). Recommended: pass
`info_keywords=("term_*","rew_*","bounce_count","reset_mode_id")` to Monitor,
add per-episode *sum* aggregates for `rew_*`, and vary the video/rollout seed.

### Provenance: record the git SHA correctly and bump the version on behavior changes — *partially implemented*
`artifacts.py` ran `git rev-parse HEAD` without `-C <repo>`, so `git_sha` was
`null` unless launched from the repo CWD (the SHA-capture fix landed on the
post-review branch). Worse for reproducibility: one run's `config.json` reported
`v0.8.0` while running features merged *after* the 0.8.0 bump (commit `1eec0d4`)
— so its "0.8.0 vs 0.9.0" comparison framing is wrong. **Rule:** bump the
package version on any behavior-changing merge, and never trust a version string
that isn't backed by a recorded SHA.

### Cut the redundant eval pass; add a curriculum-matched eval stream — *open (P1/P2)*
When headline selection is active, `EvalCallback` is reporting-only yet still
runs 30 deterministic episodes every 25k steps (~15% of run env-steps).
Recommended: cut it to ~5, raise the *selection* stream toward 50, and add an
eval stream on the *current training reset distribution* — the gap to the
canonical from-serve eval is the transfer deficit, visible within 1–2 evals. A
`deterministic=False` arm would also separate "mean policy degenerate" from
"competence lives in action noise."

---

## Humanoid tennis environment

Source: [`humanoid_env_review.md`](humanoid_env_review.md) (main @ `0d294f2`,
v0.7.0, 2026-07-13). Environment correctness, determinism, and physics rated
strong (trajectories bitwise-reproducible; Stages 0–1 proven solvable by
scripted oracles); **learning feasibility was blocked as configured**. Statuses
below were last updated at 0.16.0; items still marked open were spot-checked as
still present in the current tree.

### Update `geom_rbound`/AABB when resizing a MuJoCo geom at runtime — *implemented (0.16.0)*
The stringbed enlargement for "early contact forgiveness" writes `geom_size` and
calls `mj_setConst`, which does **not** recompute the broadphase bounding sphere
(`geom_rbound`) or midphase AABB. MuJoCo culls contact pairs on the stale
bounds, so the enlarged outer band never generates contacts — the forgiveness is
essentially inert at the racket tip and the recorded `racket_contact_scale`
metadata is misleading. Videos show the ball passing through the visibly-enlarged
racket. Fix: recompute `geom_rbound`/AABB after scaling, or compile the scale
per-stage via `mujoco.MjSpec`; add a regression test firing a ball at the
enlarged-only zone. *(0.16.0 recomputes both bounds from the scaled semi-axes
when scaling — `mj_setConst` does not refresh them;
`test_racket_forgiveness_band_generates_contacts` pins a ball contact in the
enlarged-only annulus, and the stage-1 oracle was recalibrated for the
corrected contact timing.)*

### Sparse reward + iid per-step Gaussian exploration at 100 Hz = zero gradient — *partially implemented (0.16.0)*
0 valid hits / 0 nonzero rewards in 264 random episodes; the end-to-end 25k run
scored 1 success in 177 episodes, all evals `0.000 ± 0.000`. The tell: a
*constant* random action held for a whole episode succeeds ~15% of the time —
iid noise averages to a near-still arm, so the failure is exploration structure,
not task difficulty. Recommended (several together): `use_sde=True` or action
repeat; enable escrowed `valid_hit_shaping` by default; add distance-to-ball
shaping; extend `episode_len` past the miss-fault so a miss pays −1, not an
indistinguishable 0. *(0.16.0 ships `use_sde=True` + `ent_coef=0.01` and
Stage 1–2 recipe-level `valid_hit_shaping=0.25`; distance shaping, action
repeat, and the `episode_len` extension remain open, and the env-level
shaping default stays 0.0.)*

### A flat −1 for all Stage 1–2 outcomes gives no aim gradient — *partially implemented (0.16.0)*
Under `VALID_TARGET_RETURN`, a whiff, a hit landing out, and a net fault all pay
exactly −1, and valid hits pay 0 (`valid_hit_shaping` defaults 0.0). The Stage 1
oracle under Stage 2 randomization got 12/20 *hits* but 0/20 *target returns* —
hit-vs-aim is the entire Stage 1→2 task and it is unrewarded. (Same shape as
WallBall's flat-valley problem, Cardinal Rule 2.) *(0.16.0 sets the escrowed
`valid_hit_shaping=0.25` in the Stage 1–2 recipes, separating hit-then-miss
from never-hitting at contact time; hit-landing-out vs whiff still terminate
at the same −1, so a graded aim signal remains open.)*

### The zero-action "PD standing hold" only holds welded robots — *open (major)*
The documented "zero action = two-player standing-reference PD hold" is
anchored-only. Free-standing (Stage 6), zero actions collapse both G1s from
pelvis z = 0.78 m to ~0.11 m within ~1.2 s, and no fall fault fires. Free-standing
training begins every episode with an unrewarded, unterminated balance problem —
the Stage 2 → 6 gap is larger than the missing Stages 3–5 imply. Fix: fall
detection + termination and/or an upright-alive reward; treat a standing
controller as the true Stage 3 prerequisite.

### Clamp Wilson-interval bounds into `[0, rate]` / `[rate, 1]` — *partially addressed*
`_wilson_interval(0, 50)` returned a lower bound of ~6.9e-18, violating
`0 <= low <= success_rate` and crashing the promotion gate exactly at the
0%/100%-on-a-side extremes an early curriculum run produces most. The current
function clamps to `[0, 1]` (`max(0.0, …), min(1.0, …)`) but not to the
per-side rate — add gate tests for 0-success and all-success at
n ∈ {25, 30, 50, 100}.

### Isolate the video callback (and fix its render defaults) — *partially implemented*
At 0.7.0 the callback had no error isolation, so a missing GL backend lost the
run's artifacts (Cardinal Rule 7). *(2026-08-02: the isolation actually landed —
an earlier revision of this entry claimed try/except around the rollout that the
code did not have. The whole record pass is now wrapped in
except-Exception/log-and-continue with a regression test, the normalizer-sync
fallback warns loudly instead of passing silently, and the duplicate
`rec_env.render()` — `VecVideoRecorder.step_wait` already captures the frame —
is gone.)* Still worth doing: encode at `env.metadata["render_fps"]` (100), not
60 (replays were 0.6× slow-mo), and pass `camera_name="sideline"` (the default
free camera renders the ball invisible).

### Surface the task metric in `stage_summary`; un-pin `n_envs`; observe world-frame spin — *open (major)*
- `stage_summary.txt` renders `<key>_final` (the last eval *episode's* terminal
  value) and humanoid recipes set no `headline_key`, so the configured task
  metric (`stage_success`) never appears — the first file a human reads shows
  single-episode noise. Render `success_rate`/`*_ep_mean` and set `headline_key`.
  *(0.16.0 sets `headline_key = "stage_success"` on the three stage recipes, so
  selection, early stop, and `best_model_meta.json` follow the task metric, and
  the existing headline block now surfaces `stage_success_ep_mean` in
  `stage_summary.txt`; still open: rendering `success_rate` and the
  non-headline `*_ep_mean` keys, which remain single-episode `*_final`
  values.)*
- Fixed-stage recipes pin `n_envs = 1`; Stage 0 trains 2 live action dims among
  56 inactive (96.6% dead weight), and the 0.5M/1M/2M budgets are ≈
  3.6h/7.3h/14.6h on 4-core CPU (undocumented). Un-pin `n_envs` for PPO
  (callback cadences are already `n_envs`-independent) for ~4–8× wall-clock.
- Observed ball "spin" is free-joint `qvel[3:6]`, a body-*local* frame rotated by
  the unobserved orientation quaternion — topspin and sidespin can read
  identically. Only magnitude is reliable today (bounded because all launches use
  zero spin). Expose world-frame angular velocity under the existing obs names.

### Don't re-litigate these (verified correct) — *characteristic*
Shaping escrow has no double-pay path and is net-zero on Stage 0 success;
selective normalization (indices 0–192 normalized, bounded rally/contact/mask
tail 193–298 raw) is the right design for cross-stage normalizer transfer; the
warm-start path correctly transfers policy + obs_rms while resetting reward
normalization and the optimizer; SIGINT salvage works; replay-from-info is
bit-exact. The many remaining minor findings (rules-engine edge cases, logging
hygiene, CSV column gaps) live in the review doc's §2.3.

### Environment gotcha: a broken CUDA `triton` wheel segfaults the suite on CPU boxes — *characteristic*
A suite segfault at ~90% traced to the container's `triton` wheel (pulled in by
CUDA torch on a CPU-only box), not repo code; uninstalling `triton` gave a clean
run (422 passed, 81 s). Don't chase phantom repo bugs on CPU-only environments.
`MUJOCO_GL=osmesa` works headless (`egl` works but emits cosmetic teardown errors
without a GPU).

---

## Physics reference values

Measured characteristics of the ball/court model (not bugs — reference points for
anyone tuning spin or bounce play). Source: `humanoid_env_review.md` §3.

| Quantity | Measured | Real-world reference |
|---|---|---|
| Court restitution (6 m/s drop) | 0.763 | hard court 0.73–0.76 |
| Vertical COR (fast oblique) | 0.795 | decreases with speed |
| Horizontal speed retained (10 m/s skid) | 83% | hard court ~60–80% |
| Topspin from a 10 m/s skid | 70.6 rad/s (rolling ≈ 242) | — |
| Drag deceleration @ 16 m/s | 5.45 m/s² (env Cd 0.55 → 5.27) | — |

Oblique-bounce friction is low, so spin-based play is only weakly supported today;
raising ball/court friction (including rolling/torsional) is the lever if spin
tactics matter later.

---

## Run-configuration system

Source: [`run_config_file_spec.md`](run_config_file_spec.md) (v1.1, implemented
in 0.13.0) and [`design_court_and_config_updates.md`](design_court_and_config_updates.md).

### One editable TOML per experiment, deep-merged, loud on error — *implemented (0.13.0)*
Motivated directly by Cardinal Rule 1: the `WallBallBootstrap` recipe made the
notebook's `model_kwargs=MODEL_KWARGS` cell a live footgun — an explicit kwarg
*replaces* the recipe's whole exploration package (auto-entropy, `learning_starts`,
buffer) silently. The TOML layer sits between recipe defaults and `quick_test`
(precedence: `recipe < file < quick_test < explicit kwargs`) and **deep-merges**
mapping-valued fields so "tweak one hyperparameter" cannot discard a calibrated
bundle. `performance_gate` replaces wholesale (ordered stage ladders can't merge
element-wise). The loader fails loudly on every class of mistake (unknown key
with a `difflib` suggestion, rejected builder-owned field, typo'd env kwarg
caught by an eager probe env), and `config.json` records the file path, sha256,
and full content.

### Guard against invisible configuration; TOML has no `None` — *implemented (0.13.0, spec v1.1)*
v1 required the config path to always be explicit — an implicitly-discovered
config that silently changes a run is exactly the failure mode this repo keeps
paying for. v1.1 softens this to *assisted-explicit* for the notebook only
(`CONFIG_FILE = "auto"` materializes the packaged starter into Drive on first
use and prints path + sha256 every run); `build_train_config` itself still never
discovers anything. TOML gotcha worth remembering: there is no `None`, so a kwarg
that needs `None` (e.g. `early_touch_penalty` for the legacy terminal rule) uses
the sentinel string `"none"`, converted recursively — and `"true"`/`"false"` are
*rejected* because both are truthy Python strings that would silently enable what
the file tried to disable.

### `weak_return_penalty` in training, strict rule in eval — *implemented (0.13.0)*
So training gets practice reps out of near-misses (12–16% of best-model episodes)
while evaluation keeps the exact task every prior run was measured on
(`bounce_count_ep_mean` stays comparable with the 165358/023737 runs). Paired
with `gamma = 0.995` (credit horizon ~200 steps > one ~130-step exchange,
targeting the now-dominant 54% double-bounce failure), and best-model tie-break
moved from the saturated `ge_2_rate` to the `ge_5_rate` the long run actually
improved (8% → 22%). This is why *learning curves* across 0.12 → 0.13 baseline
runs are not comparable but *eval metrics* still are.

### Court markings are render-only MuJoCo sites — *implemented (0.11.1 / 0.13.0)*
The `court_style` kwarg (`diagnostic` / `tennis` / `none`) and all lane/serve
markers are MuJoCo *sites* — they cannot collide, so they have provably zero
physics/observation/reward impact and are safe to reposition every reset. The
`tennis` style fits a to-size ITF half-court onto the existing 16 m × 12 m floor
with no scaling (wall face = net at x = 3.9, baseline at x = −7.985); the service
line lands at x = −2.50, *inside* the paddle lane — a happy accident worth
keeping. Metrics-producing paths always keep the default style; only the recorded
video differs.

### PaddleTennis freezes as one cooperative alternating-serve point per episode — *implemented (unreleased)*
The phase-P1 env (`CourtsideDynamics/PaddleTennis`) shipped only after P0–P4
measured every number it froze (`paddle_tennis_env_20260802.md`). The
non-obvious calls: **either side's fault pays the shared −1** (a cooperative
rally has no useful "whose fault" asymmetry — an opponent fault usually
punishes a poor incoming ball; asymmetry waits for the scoring phase), **one
point per episode with strict serve alternation** (keeps `crossings` an
un-averaged per-point tail and the side statistics provably 50/50 — the P4
mirror makes alternation exactly fair), **episode_len 1500** (the wall-ball
750-cap lesson: never truncate a healthy rally; the scripted tail ends in
~270 steps), **the bounded rally/contact observation tail stays unnormalized**
(the humanoid stage-boundary variance lesson), and **no shaping at the
freeze** (every pre-evidence shaping term this repo added was later falsified
or exploited). The policy physically trains side A only; P4's bit-for-bit
mirror is the reason that loses nothing, and `serve_side_is_policy` records
the alternation in every info stream. `ladder_certification` stays
WallBall-only — PaddleTennis certifies through the probes harness on the
reserved held-out seed blocks instead.
