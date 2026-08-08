# PaddleTennis ground rules — the volley loop is measured, killed by design, and the era re-certified

Status: review snapshot + task-definition amendment, 2026-08-03. The
first learned GPU run found a legal cooperative exploit in the frozen
phase-P1 task; this snapshot records the run's evidence, the rules
fix (`volley_rule="fault"`, now the registered default), the probe
battery that validated it, the recalibrated scripted reference, and
the new held-out certification. This document amends
[`paddle_tennis_env_20260802.md`](paddle_tennis_env_20260802.md) §4–§5;
everything else in that freeze stands.

## 1. What the first GPU run found (the volley-era record)

Run `20260803_004559` (SAC, 2M steps, L4, stock recipe, seed 0 — not
the pre-registered seed 1, so formally an unregistered run; its
evidence is treated as diagnosis, and reserved block 4100–4199 was
NOT spent on it): `crossings_ep_mean` **37.60** at the budget
ceiling, still climbing — versus the pre-registered ≥ 6.0.

The curve is two regimes. Through ~1M steps, honest rally learning
(0.4 → 5.4 crossings, pilot-like). Then a runaway loop: 6.2 at
1.08M, 10.0 at 1.38M, 25.9 at 1.83M, 37.6 at 2M, with the signature
of **close-net cooperative volleying**: a net crossing every ~14
control steps (63–70 crossings in ~900-step episodes — only possible
with both paddles patting at point-blank range), `return_in_flight`
time rising from ~71% to 91.6%, and the transition's fault
fingerprints (`ball_net` spiking to 0.77 at 1.65M, then
`term_illegal_hit` to 0.77 at 1.88M) before the loop stabilized.
Everything else stayed healthy: zero nonfinite, exact 50/50 serve
alternation at all 80 evals, task-metric selection working.

The wall-ball verdict applies verbatim: *physics sets the dominant
strategy*. Volleys were deliberately legal in the freeze (the
humanoid-rules convention); reward counts confirmed returns; nothing
values depth — so the highest-rate legal return loop wins, and the
scripted opponent cooperates with it happily. The pivot's premise
("deep play becomes useful because of what the opponent does") holds
only against an opponent that punishes net play; the phase-P1
scripted opponent doesn't.

## 2. The fix: ground rules

`RallyRules.require_bounce_before_return` (default off, so wall-ball
and humanoid consumers are untouched): striking the incoming ball
before it has bounced on the hitter's side is a **`VOLLEY_RETURN`
fault** that confirms nothing — not even the incoming shot. (The
first draft credited the incoming shot, real-tennis style; the
pre-push adversarial review measured the consequence in the
cooperative reward: touching a doomed out-bound ball banked its +1
against the same fault penalty an untouched landing would cost, so
lunge-volleying predicted-out balls was strictly reward-optimal. A
volley fault now ends the point with no credit to anyone.)
`PaddleTennisEnv` grows
`volley_rule="fault" | "legal"` with **"fault" the registered
default**; `"legal"` reproduces the superseded era exactly (its
artifacts — the GPU run, the pilot, the 3100-block certification —
remain reproducible under it).

Why this fix and not the alternatives considered:

- a **no-volley zone** adds a geometric parameter and still permits
  fast patting from behind the line;
- **depth-credited crossings** rewires the metric itself and is
  shaping-flavored (lessons 4/19);
- the bounce rule is expressible in the existing rules vocabulary,
  observable in the existing observation (bounce count and phase are
  already Markov inputs), and puts a *physical* floor under exchange
  cadence: every legal crossing now contains a full flight-bounce-
  strike cycle (~126 steps measured, vs the loop's 14).

## 3. What the fix revealed about the frozen oracle

The bring-up measured something the P3 era missed: **the frozen P1
`lead_charge` port's returns were largely volleys.** Its
pre-positioning tracks the ball's y/z at the home column — inside the
incoming serve's descent path (the serve lands 4.55 m deep, behind
the column it gets intercepted at). Under ground rules that touch is
an instant fault, so the era needed a recalibrated reference
controller, built in three measured steps
(`ground_lead_charge_local_action`):

1. **Run-up wait**: ballistically predict the landing point and wait
   0.9 m behind it, face low (the wall-ball `run_up` instrument) —
   fixes the descent-path intercept.
2. **Soft ground stroke** (`GROUND_SWING` 0.1): the frozen 0.4
   swing-through slams post-bounce balls flat and long — measured
   landings mean **11.0 m** from the net, every stroke out. The
   6.5 m court needs the rebound-dominant touch; contact low on the
   rise lets the fixed +10° face supply the loft (the P0–P2 strike-
   height channel).
3. **Hold-low recovery**: the frozen post-hit re-homing chases
   straight through the soft return's path (double_hit 9–12/16); the
   ground oracle holds position with the face dropped until the ball
   crosses.

## 4. The probe (pre-registered criteria; seeds 5100–5199)

`tools/paddle_tennis_volley_probe.py`, 100 episodes/cell, each
player on both sides through the P4 mirror:

| player | rule | crossings | ≥1 | returns | steps/x | volley faults | truncated | top terminations |
|---|---|---|---|---|---|---|---|---|
| ground | fault | 7.78 | 97% | 7.59 | 126 | 0% | 30 | ball_net 56, out_of_bounds 12 |
| ground | legal | 7.78 | 97% | 7.59 | 126 | 0% | 30 | ball_net 56, out_of_bounds 12 |
| volley (frozen P1) | fault | 0.00 | 0% | 0.00 | – | 100% | 0 | volley_return 100 |
| volley (frozen P1) | legal | 3.14 | 100% | 2.42 | 90 | 0% | 0 | out_of_bounds 89 |
| patting witness | fault | 0.00 | 0% | 0.00 | – | 100% | 0 | volley_return 100 |
| patting witness | legal | 1.57 | 100% | 1.53 | 89 | 0% | 0 | failed_to_cross 48, ball_net 39 |

All four pre-registered criteria PASS → **adopt `volley_rule="fault"`
as the era default**:

- **A. Kill**: both volley-style players collapse to 0.00 crossings,
  100% `volley_return` — the rule leaves no volley loop representable.
- **B. Feasibility**: the ground pair sustains 7.78 (old floor 2.6).
- **C. Cadence**: 126 steps/crossing (floor 60).
- **D. Rule-neutrality**: ground play is identical under both
  profiles (Δ = 0.00) — the rule touches only volleys.

Honest notes: the patting witness volleys every return it makes but
does NOT reproduce the learned loop's cadence/persistence (the GPU
run remains the loop's documentation); and 30% of ground episodes
rally to the 1500-step cap — the cap now binds on healthy scripted
play, which the next pre-registration must treat as a reported
metric, not a failure signal. The first probe run measured a 1/100
residual ground-oracle self-volley; the review traced it not to
landing-prediction error but to **workspace-margin collapse** (seed
5117: a legal ball landing 6.32 m deep against the paddle's 6.4 m
reach leaves no room to wait behind it, and the y-tracking low face
sat in the descent path). The fix — a lateral dodge that fires only
when the landing is within 0.3 m of the reach limit — eliminated the
residual; a broader dodge trigger was itself measured and rejected
(it degraded deep-serve returns, ≥1 rate 97% → 82%, and failed the
first held-out certification attempt at 89% vs the 90% floor). The
matrix above is the post-fix re-run on the same seeds.

## 5. New reference band and held-out certification

The ground-era scripted band replaces the P3-era 3.15–3.42:
**7.78 mean crossings** (probe block, post-fix), taxonomy ball_net-
dominant, ~1.3 s of simulated time per crossing (126 control steps
at the 0.01 s control step). Held-out certification (`--certify`,
updated contract) ran on newly designated reserved block
**4200–4299** against floors pre-registered from the PRE-fix
calibration band (7.04: mean ≥ 5.9; ≥1 rate ≥ 0.90; zero unsafe) —
the review-driven controller fixes landed between registration and
the final run, so the floors are conservative relative to the final
band, and they were not re-tuned: **PASS** — mean crossings **7.68**
(std 4.51), ≥1 99%, valid returns 7.53, exact 50/50 serve split,
zero unsafe, 27 cap-truncations.

## 6. Ripples

- **P5 transfer** (matrix re-run on seeds 5000–5099 under the era
  default): the shimmed wall-ball stub via `scaled + yield` still
  crosses (1.42 mean, 97% ≥1 — the pre-committed admission floors
  1.0 / 0.60 stand unconsumed) but its **confirmed returns collapse
  0.69 → 0.00**: wall-ball's flat-hard stroke calibration lands out
  under ground confirmation (native ground oracle on the same block:
  7.02 crossings, 3.44 returns/side). Champion measurements on Colab
  should read the returns column beside the binary verdict — a
  wall-ball champion is likely a serve-returner, not a rally
  partner, on this court.
- **The volley-era pre-registration**
  ([`paddle_tennis_pilot_and_first_run_20260802.md`](paddle_tennis_pilot_and_first_run_20260802.md)
  §2) is **superseded unconsumed** — its criteria were calibrated on
  the volley-era pilot. The ground-era first run should be
  pre-registered afresh once a ground-era pilot exists; reserved
  block 4100–4199 remains clean for that run's held-out gate.
- The recipe is unchanged except its documentation: same metric
  (`crossings`), same selection, same normalization; `term_volley`
  joins the terminal eval keys.

## 7. Seed ledger

Burned here: **5100–5199** (probe matrix, all cells share it) and
reserved block **4200–4299** (the ground-era certification's single
use). Re-burned: 5000–5099 (P5 recalibration, same block as the P5
snapshot). **4100–4199 remains the only clean reserved block**, held
for the ground-era first learned run's held-out gate.
