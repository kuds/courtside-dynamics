# Design: the true-baseline extension (`WallBallTrueBaseline`, 0.25.0)

Status: implemented design, 2026-07-31, written against v0.24.0
(`1d8d4ee`) and shipping with 0.25.0 on branch
`claude/wall-ball-curriculum-diagnosis-0j8pjz`. Every geometry,
serve-physics, and mechanics number below was measured by scripted
probes on a sandboxed prototype **before** the corresponding change
landed in the tree, per the campaign doctrine (probes first, direct
task before curriculum, certification before launch). Local SAC
numbers use the 1:2 update ratio and are never compared to GPU runs.

## 1. Goal

The goal-rally era ended with its bar met twice: policies sustaining
3.3–3.8 completed returns while receiving at x = −3.4..−3.7, the
deepest the 0.13.0-era workspace allows
(`wall_ball_goal_rally_replication_20260730_review.md`). The next
campaign era plays from the **true baseline**: the ITF half-court the
tennis style draws puts the baseline at x = −7.985, ~3.3 m behind the
old workspace edge at −4.7. This design extends the paddle workspace,
the serve energy, and the in-play volume so that a rally can genuinely
be received near the baseline — as a delta on the proven direct-task
recipe, not a new design.

## 2. The frozen era task

| Parameter | Value | Origin |
|---|---|---|
| `paddle_x_target_range` | (−8.2, 0.3) | extended workspace, explicit opt-in (§3) |
| `paddle_x_fence` | (−8.2, −2.6) | wide fence; baseline-only refuted by rebound geometry (§5) |
| `paddle_start_x` | −7.9 | behind the ITF baseline; forces the deep receive (§4) |
| `paddle_home_x` | −5.4 | fence midpoint (0.22.0 usable-share rule) |
| `serve_start_x` | 1.0 | unchanged; alignment falsified in the diagnosis campaign |
| `serve_speed` | 11.0 | minimum speed whose post-bounce leg reliably reaches −7.9 (§4) |
| `ball_in_play_min_x` | −10.0 | new per-task kwarg; task impossible at the old −6.0 (§5) |
| `episode_len` | 750 | unchanged (≈4.8 exchanges at the measured 156-step cadence) |
| everything else | goal-rally era values | reward, gamma 0.995, damping 8.0, obs/action interface untouched |

## 3. Paddle location: the workspace extension and its neutrality

**Change.** `wall_ball.xml`: `paddle_slide_x` joint range and
`paddle_target_x` ctrlrange widen from (−3, 2) to (−6.5, 2), moving
the physical world-space workspace from (−4.7, 0.3) to (−8.2, 0.3).
The bold baseline marker moves from −4.7 to −8.2 and picks up −5..−8
coordinate ticks (render-only sites).

**Hazard found and closed.** The env derived the *default* x action
mapping from the XML ctrlrange, so widening the XML silently rescaled
action semantics for any config that never passed an explicit
`paddle_x_target_range` (action −1 remapped from −4.7 to −8.2 in
world x). The pre-existing bit-for-bit guard test caught this. Fix:
the default mapping is now frozen at the historical workspace via a
module constant (`_DEFAULT_PADDLE_X_TARGET_RANGE = (-4.7, 0.3)`); the
extended workspace is **strictly opt-in** by passing the range
explicitly, as this recipe does.

**Verification.**
- T6 (prototype phase): the XML edit alone, replayed under every
  current-era recipe config, produced bitwise-identical
  observation/reward streams.
- T7 (shipped tree): deterministic 6-episode random-action rollouts
  hashed over the full (obs, reward, termination) stream, HEAD
  worktree vs working tree, four configs — bare default env, default
  `one_bounce`, the goal-rally task, the baseline lane. **All four
  hashes match**: the extension is invisible to every pre-existing
  configuration, so all saved policies and `VecNormalize` statistics
  remain comparable with 0.22.0–0.24.0.
- Validation semantics are pinned by tests: `paddle_home_x = −5.0`
  still raises on a default-mapped env (outside the frozen default
  range) while ranges beyond −8.2 raise against the physical bound.

## 4. Serve physics: 11 m/s is a measurement, not a guess

The serve keeps its origin (1.0), lob (0.0), jitter (±0.5), and lane
spread; only `serve_speed` changes, 7.0 → 11.0. Probe T1/T1b swept
the origin × speed grid with a parked paddle and measured the serve's
post-bounce crossing state at the receive depths (60 episodes/cell,
calibration seeds):

| speed | first bounce | legal-receive rate at −7.9 | crossing height | crossing speed |
|---|---|---|---|---|
| 8.0 | −3.02 | 7% | 0.24 m | 7.9 m/s |
| 9.0 | −3.52 | 73% | 0.27 m | 8.1 m/s |
| 10.0 | −4.03 | **100%** | 0.44 m | 8.7 m/s |
| 11.0 | −4.53 | **100%** | 0.54 m | 9.4 m/s |

Findings the task freeze rests on:

- **Landing scales ~0.5 m per +1 m/s** at origin 1.0 — smooth,
  no cliffs.
- **Below 10 m/s the deep task does not exist**: the ball dies short
  of the paddle. The goal-era serve (7.0) is unreturnable from −7.9
  because it never arrives — a serve-physics fact, not a paddle
  limitation.
- At 11 m/s the ball crosses −7.9 at 0.54 m — inside the face's
  addressable band (centre 0.05–2.0 m) — so the receive is
  *possible*; and the crossing arrives ~1.7 s after serve, far outside
  the paddle's reach envelope for any pre-bounce interception from
  −7.9 (≈12.5 m/s terminal speed), so the deep post-bounce receive is
  *forced*. 11.0 was chosen over 10.0 for crossing-height margin
  (0.54 vs 0.44 m) against the serve jitter.
- Serve-origin blends were **not** revisited: alignment was falsified
  at both pre-registered sites in the diagnosis campaign and is on
  the dead-levers list.

An earlier probe iteration (T1) scored "arrival at the parked face"
instead of crossing-state kinematics and read 0% arrivals everywhere —
an artifact: the post-bounce hop apex (~0.6 m) passes *under* a parked
face whose bottom sits at 0.95 m. The face must drop to address the
ball (it can: bottom edge reaches 0.05 m). Probe design lesson
recorded in the journal.

## 5. Game mechanics: fence, home, and the in-play bound

**Wide fence, not a baseline box.** Probe T3b measured where rebounds
land after legal deep returns (oracle play, 100+ returns/cell): mean
x = −1.0, p10 −3.2, **0% deeper than −6.1**. A baseline-only fence
(e.g. (−8.2, −6.0)) would strand the paddle where the ball almost
never comes back — unsustainable by geometry, in either direction of
play. The era fence therefore spans (−8.2, −2.6): it *adds* the deep
band to the goal era's fence rather than relocating play, the deep
receive is forced by the serve (§4), and forward recovery stays
legal. Oracle play on the frozen task measures first contact at mean
−7.01 (p90 −6.99) — the deep receive happens, every episode.
`paddle_home_x` −5.4 is the fence midpoint, per the 0.22.0
usable-share rule.

**The in-play bound had to move — the task is impossible without
it.** The historical OOB edge x = −6.0 sits 1.9 m *in front of* the
new paddle start: an 11 m/s serve terminates out-of-bounds mid-flight
before the receiver can ever touch it. (Directly demonstrated: the
frozen task under the −6.0 bound scores 60/60 untouched-serve OOB
episodes, zero returns.) Untouched serves legitimately run past −9.4
on their second hop, and deep pass-balls are a real fault class, so
the bound becomes a **per-task constructor kwarg**
(`ball_in_play_min_x`, default −6.0 = bit-for-bit historical
behavior, validated finite and negative) and this era sets −10.0 —
1.8 m behind the workspace edge, comparable to the 1.3 m the −6.0
bound left behind −4.7, scaled for the hotter serve.

**OOB taxonomy on the frozen task** (oracle band, 60 episodes,
seeds 0–59, shipped code): 41 of 60 episodes end OOB — 24 long balls
*over the wall* (x ≥ 4.9: deep receives arrive hot and the return
sails), 14 deep pass-balls (x ≤ −9.4: missed or whiffed receives), 3
lateral. The rest end double-bounce (14) or timeout. Mean completed
returns at OOB exit 1.71 — OOB exits are mostly *rallies that end*,
not dead serves. The long-ball fault is expected to be the era's
main learnable error mode.

## 6. Reference band and startup certification

The recalibrated `lead_charge` oracle family from 0.24.0 transfers;
the gap sweep on the frozen task picks **2.6** (mean 1.98 at gap 2.6
vs 1.75/1.32/1.33 at 3.2/4.0/5.0). Reference band (calibration seeds
0–59): **mean 1.98 completed returns, ≥2 rate 67%, ≥3 rate 22%,
exchange cadence 156 steps** (~25% longer than the goal task's ~125 —
gamma 0.995 still sees across an exchange: 0.995^156 ≈ 0.46).

Certification: the stock feasibility criterion (oracle ≥2 rate
≥ 0.90) encodes "a scripted reference completes two exchanges almost
every serve", which is true of every shallow era and false here — no
scripted controller dominates an 11.8 m court. The floor becomes a
spec knob (`feasibility_ge2_floor`, default 0.90 untouched for every
other recipe), and this recipe declares **0.50** = the measured band
minus two 30-episode sampling standard deviations. The resolved floor
is stamped into every certification report, so a lowered bar is
always visible in run artifacts. This is not a promotion-bar change:
the single-stage gate has nothing to promote to, and the 3.0
informational threshold is untouched.

**Held-out certification** (fresh seeds 4000–4099, n=100, the full
startup-certification code path on the built recipe): **PASS** — all
blocking criteria. Oracle 1.81 mean, ≥2 rate 61% (Wilson 95% CI
[0.51, 0.70], lower bound above the floor); crude 1.09 mean, ≥2 rate
9% (> 0); parked exactly zero contact; monotone parked < crude <
oracle rewards (−1.0 < 6.9 < 13.3); serve landing observed 100/100
with no pre-bounce paddle contact; landing offset +3.38 m from paddle
start (consistent with §4's first bounce at −4.53). The only warning
is the expected informational-bar shortfall. Calibration pre-check on
burned seeds 1000–1029 agreed (oracle 60%, crude 10%).

## 7. Local SAC learnability pilot (S4, pre-registered)

Three cold-start local arms (training seeds 401–403), 500k steps,
1:2 update ratio, on the frozen task — pre-registered in the
experiment ledger before any arm ran, with the read-out: "learnable"
= ≥2 of 3 seeds end with last-5-eval bounce mean ≥ 1.0 and paddle
contact ≥ 80%; "marginal" = 1 of 3 or means 0.5–1.0 rising; "not
learnable at this budget" otherwise. For scale: equivalent local arms
on the goal task reached ~1.9, and this task is harder (oracle band
1.98 vs ~2.7) with ~25% longer cadence.

**Result: [PENDING — the pilot is in flight; this section is filled
with the measured read-out before this document ships. Do not merge
with this placeholder in place.]** In-flight health at 125k steps:
all three arms contact the serve and complete returns (eval bounce
means 0.57–1.00 and rising) — no arm is in the zero-contact basin.

## 8. Pre-registered next run (GPU, Colab)

Config: the packaged starter
(`run_configs/wall_ball_true_baseline.toml`) — recipe
`WallBallTrueBaseline`, 6M steps, n_envs 8, eval 60 episodes / 25k
steps, patience 60 (the replication's record seed banked its best at
3.2M under patience 60; this era's cadence is longer still),
degenerate-signal guard live, startup certification enforced by the
launch checklist (seeds 30000+, floor 0.50).

Success criteria, registered before the run:

1. **Primary ("the era opens")**: best 3-eval window
   `bounce_count_ep_mean` ≥ **2.0** on the matched stream, AND the
   post-training long-horizon audit of the banked best confirms mean
   ≥ 2.0 completed returns with ≥80% of episodes completing ≥1
   return.
2. **Deep-receive integrity**: audited mean first-contact x ≤ −6.0
   (vs −3.7 at the goal era; the serve forces the oracle to −7.0). A
   policy that meets (1) by somehow refusing the deep receive does
   not open the era.
3. **Stretch (campaign bar)**: a ≥ 3.0 window — this is the
   gate_window_mean mark and is NOT required; the oracle band (1.98)
   sits ~1.0 below the goal task's (~2.7), and goal-era policies beat
   their band by ~1.4×, which extrapolates to ~2.7 here.
4. **Reliability expectation** (not a criterion): ~1 in 3 seeds may
   need a restart (goal-era replication rate); the degenerate guard
   is the safety net.

Anything that misses (1) triggers a run review before any lever is
touched; the dead-levers list stands.

## 9. Seed ledger

- **Burned this phase: 4000–4099** (held-out certification of the
  frozen task + probe + floor; the 4100–4199 remainder of the block
  stays clean). Registered here.
- Calibration reused already-burned blocks 0–199 and 1000–1199
  (probes T1–T4c, gap sweep, certification pre-check).
- Local training seeds 401–403 (training-seed namespace, S4 pilot).
- Startup certification keeps its reserved 30000+ block; 3100–3199
  and 4100–4199 remain clean for future held-out checks.

## 10. What did not change

Serve origin/lob/jitter, rally rules and scoring, every reward term
and scale, gamma, the 23-dim observation layout, the 3-action
normalized interface, paddle damping, episode length, the wall, court
width, floor friction — and every pre-existing recipe's behavior,
bit-for-bit (§3). `WallBallGoalRally` remains available unchanged for
goal-task replication runs.
