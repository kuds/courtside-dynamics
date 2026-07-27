# Plan: resolve the aligned ladder's deep-stage feasibility

Status: proposed, 2026-07-25. Scope is the stage-3/4 feasibility failure of
`WallBallDepthCurriculumAligned`. The design being executed is
`design_wall_ball_serve_alignment.md`; this document sequences the work that
its "Final recommendation" leaves open and adds the parametric study the
replication sweep now justifies.

This plan does **not** authorize a full 6M aligned pilot. That remains
blocked on the stage-1 A/B and fresh late-stage certification.

## What is settled

Two paired 200-episode sweeps on disjoint seed sets agree:

- Aligned stages 0–2 clear the ≥90% oracle feasibility bar (93.5–98.5% on
  seeds 0–199). Stage 3 is 84.5%/85.5% and stage 4 is 60.0%/63.5%.
- The fixed-origin baseline clears every stage on seeds 0–199, including
  stage 4 at 95.0%. Its earlier 89.0% was seed noise, not a defect.
- Aligned landing alignment works: mean offset ≤0.049 m against a 0.10 m
  bar, ≥99.5% of individual landings inside ±0.30 m.
- Stage 0 is byte-identical across ladders, so the negative control holds.

The deep-stage loss is associated with the serve-origin change rather than
the sliding-window ladder both arms share — but see the confound below
before treating that as causal.

## What is open

1. **Mechanism, and a known confound.** The oracle's per-stage timing was
   introduced at commit `24223f4`, when every stage served from
   `serve_start_x=1.0`, and commit `3fd9fb3` added the aligned ladder by
   changing only `serve_start_x`. Both arms therefore run a controller
   calibrated for a ball landing 1.0–1.4 m ahead of the paddle, which
   **flatters the baseline**: under alignment the ball lands at the
   paddle's feet, an interception problem the charge timing was never
   re-derived for. The failure split is consistent with insufficient
   approach room — 65 of the aligned oracle's 73 stage-4 failures are
   double bounces, while the baseline fails only 10 times — but an unknown
   share of the deficit is stale timing rather than infeasible geometry.
   Phase A exists to separate the two, and no causal claim should be made
   until it has.
2. **Whether partial alignment is viable.** Alignment and feasibility are
   currently all-or-nothing. Nothing has measured the interior.
3. **Whether alignment helps SAC at all.** Only the stage-1 training A/B
   answers this, and it is unrun.
4. **Whether the gate's bar and the sweep's bar describe the same task.**
   The sweep certifies feasibility as "oracle completes ≥2 returns on
   ≥90% of serves"; the gate promotes on "mean `bounce_count_ep_mean`
   ≥ 3.0". Run `20260725_171747` showed these can disagree sharply: it
   spent 3.2M steps and 128 evaluations at stage 2 — which the sweep
   certifies at 93.5% — without a single evaluation reaching 2.9, while
   the oracle's own mean at that stage is 2.43. A rung can be feasible
   by the sweep's definition and unpromotable by the gate's. Any ladder
   change from Phase B must be checked against **both** bars, not just
   the sweep's. See `wall_ball_aligned_patience_review.md`.

## Seed ledger

Certification depends on never tuning against the seeds used to certify.

| Range | State | Use |
|---|---|---|
| 0–199 | **burned** | oracle charge-gap search; 2026-07-25 replication |
| 1000–1199 | **burned** | 2026-07-24 held-out sweep, now diagnostic |
| 2000–2199 | **reserved** | first certification after controller/ladder freeze |
| 3000–3199 | **reserved** | second certification, if 2000–2199 is inspected |

Do not open a reserved range until the parameters it will certify are
frozen and committed. Every sweep runs from a clean committed revision;
the 2026-07-25 artifacts already meet this, so their raw JSONs need not be
vendored — the recorded revision, command, and `seed_range` field
regenerate them. Note the command alone is not sufficient: `--seed-start`
defaults to 0 and was omitted, so the range lives only in `seed_range`.
Pass it explicitly on future runs.

## Implementation prerequisites

Phase B is not runnable from the current CLI. It needs a small, contained
change to `tools/depth_stage_sweep.py`:

- ~~**Probe parameters are hardcoded per stage.**~~ **Done.**
  `--oracle-probe STAGE=run_up:VALUE|charge_gap:VALUE` now replaces a
  stage's probe key outright (repeatable, at most once per stage), which
  is what Phase A needed: `_oracle` raises unless exactly one of the two
  is set, so a value-only flag could not have expressed "try `run_up` at
  stage 3". Resolved values land in the JSON report's `stages` block and
  the invocation in `provenance.command`, so a grid point is
  self-describing.
- **Ladders are a fixed registry.** `--ladder` accepts only the two keys
  in `LADDERS`, and `BASELINE_STAGES` is derived from `ALIGNED_STAGES` by
  overwriting `serve_start_x`. A blended ladder needs either a third
  registry entry per blend or a `--serve-origin-blend LAMBDA` flag
  computing the same override.
- **The landing contract assumes full alignment.** `require_alignment` is
  derived solely from `ladder_name == "aligned"`, and the offset is always
  measured against `paddle_start_x` using the module constants
  0.10/0.30/0.95. A blended candidate has no way to declare its own target,
  so Phase C could not gate one: a third ladder key gets
  `require_alignment=False` automatically, while a blend flag under
  `--ladder aligned` keeps the full-alignment thresholds and fails by
  construction. The contract needs a per-stage declared target offset, and
  `require_alignment` must become settable independently of ladder name.

Do **not** try to record the calibration range for a rerun over seeds
0–199. The tool exits with "calibration and held-out seed ranges must be
disjoint" whenever the declared calibration window overlaps the held-out
window, and here they coincide. The `calibration: null` in the 2026-07-25
artifacts is the guard working as designed, not an omission; a free-text
`--calibration-note` would be the way to record the relationship if that
provenance is wanted.

## Phase A — RESULT (2026-07-26): geometry, not the oracle

**Run and answered.** `--oracle-probe` landed in the sweep tool, and a
grid of 19 settings per stage (11 `charge_gap`, 8 `run_up`, 200 episodes
each, calibration seeds 0–199) was evaluated on both ladders.

| stage | shipped | best over grid | settings clearing 90% |
|---:|---:|---:|---:|
| aligned 3 | 85.5% | **88.5%** [83.3, 92.2] @ `charge_gap` 2.4 | **0 / 19** |
| aligned 4 | 63.5% | **69.0%** [62.3, 75.0] @ `charge_gap` 2.4 | **0 / 19** |
| baseline 3 | 94.0% | 94.0% @ `charge_gap` 1.8 | 4 / 11 |
| baseline 4 | ~89.5% | 94.0% @ `charge_gap` 1.4 | 2 / 11 |

**Gate A verdict: stage 4 fails decisively** — best achievable 69.0%
with the interval entirely below the bar, comfortably inside the "≤80%
at stage 4 is decisive" rule set below. **Phase B is therefore
mandatory.** Stage 3 is a marginal fail: no setting clears the point
estimate, but its interval still straddles 90%, so it is not
independently condemned.

Retuning buys +3.0 points at stage 3 and +5.5 at stage 4 — roughly a
fifth of each deficit. The confound was real and is now priced: geometry
dominates. The fixed-origin ladder clears the bar with several settings,
so the controller family is capable and it is the aligned geometry it
cannot play.

Two mechanism notes for whoever runs Phase B: the aligned ladder's
optimum is a *larger* charge gap (2.4 against the shipped 1.8/1.7),
exactly what a ball landing at the paddle's feet should demand, and
`run_up` mode is dead at these depths (best 5.5% / 1.0%), so the probe
mode itself was never the issue. The grid reproduces the shipped-setting
figure exactly at aligned stage 3 (85.5%), which validates the driver
against the committed sweep path.

Ceiling recorded, per the note below: **69.0% is the upper bound any
stage-4 policy can be measured against on this geometry.**

## Phase A — method (as specified before the run)

The decisive question, and cheap to answer. A 200-episode sweep costs about
two minutes of CPU on four cores per ladder — roughly five for a paired run
— so a parameter grid is affordable.

- Grid-search the oracle's per-stage timing and run-up (`oracle_run_up` and
  `oracle_charge_gap`) at stages 3 and 4 on **calibration seeds 0–199**,
  which are already burned and correct for this purpose.
- Report the best achievable oracle ≥2 rate per stage, with Wilson
  intervals, alongside the parameter that achieved it.

The design note records that an earlier charge-gap search on this range
"did not restore the 90% bar, especially at stage 4," but produced no
standalone artifact. Phase A repeats it as a retained, reproducible grid
rather than trusting that recollection.

**Gate A.** If some setting clears 90% at both stages, the failure is
controller calibration: freeze that setting and proceed to Phase C. If the
best achievable stays materially below 90% — treat ≤80% at stage 4 as
decisive — the aligned stage-4 geometry is infeasible by construction and
Phase B becomes mandatory before any ladder work.

Recording the *ceiling* matters even when the gate passes: it is the upper
bound any learned policy could reach, and the campaign has repeatedly
mistaken a task ceiling for a training problem.

## Phase B — RESULT (2026-07-26): partial alignment works

**Run and answered, and the answer is positive.** `--serve-origin-blend`
landed alongside a landing contract that measures against the target a
blend declares. Nine λ values were swept on calibration seeds 0–199, 200
episodes per cell, retuning the oracle's charge gap at each point (six
values per deep stage) because Phase A showed the optimum shifts with
geometry.

| λ | stage 3 best | stage 4 best | stage-4 landing in front of paddle | verdict |
|---:|---:|---:|---:|:--|
| 0.00 (fixed origin) | 94.0% @2.8 | 94.0% @1.4 | 1.399 m | pass |
| 0.25 | 95.5% @1.2 | 91.0% @1.4 | 1.061 m | pass |
| 0.50 | 96.5% @1.2 | 96.5% @2.4 | 0.724 m | pass |
| **0.75** | **99.0% @2.8** | **96.0% @2.4** | **0.386 m** | **pass** |
| 0.80 | 97.5% @2.4 | 91.0% @2.4 | 0.319 m | pass |
| 0.85 | 98.0% @2.8 | 91.0% @2.4 | 0.251 m | pass |
| 0.90 | 95.5% @2.8 | **82.0%** @2.4 | 0.184 m | fail |
| 0.95 | 93.0% @2.4 | **81.0%** @2.4 | 0.116 m | fail |
| 1.00 (fully aligned) | 88.5% @2.4 | **69.0%** @2.4 | 0.049 m | fail |

Stages 0–2 pass at every λ (94.0% / 93.5–98.5% / 92.0–98.0%); stage 0 is
λ-invariant by construction and holds at 94.0% throughout, so the
negative control survives the new lever.

**Gate B verdict: the largest λ clearing 90% at every stage is 0.85.**
Partial alignment is viable, and Phase B's pessimistic framing was
wrong — the ladder does not need redesigning.

The failure is a **sharp threshold in landing distance, not a gradual
decay**: everything down to 0.251 m in front of the deep paddle passes,
and 0.184 m fails by 8 points. The paddle needs roughly a quarter of a
metre of approach room, and full alignment removes it.

Feasibility looks non-monotonic in λ on calibration — the interior scores
above both endpoints at the deep stages — but note these are
argmax-selected cells at n = 200, and Phase C later showed 3–5 point
calibration-to-held-out swings are routine. Held-out, λ = 0.75 and the
fixed origin are statistically indistinguishable on oracle feasibility.
The interior's advantage over the endpoints should be read as "does not
cost feasibility", not as "improves it".

**Recommended freeze for Phase C: λ = 0.75, not the maximum-passing
0.85.** Two reasons. λ = 0.75 carries real margin (96.0% at stage 4
against 91.0%), and the charge gap at each point was chosen as the best
of six on the *same* calibration seeds, so every rate in the table above
is optimistically biased by that selection. A configuration sitting one
point above the bar on calibration data is not a safe thing to certify.
Frozen probe values at λ = 0.75: stage 0 `run_up` 1.1, stages 1–2
`charge_gap` 1.0, stage 3 `charge_gap` 2.8, stage 4 `charge_gap` 2.4.

## Phase B — method (as specified before the run)

Only alignment's *target* is in question, not alignment itself. The current
ladder aims the first bounce at `paddle_start_x`; the baseline leaves it up
to 1.4 m in front. Measure the interior rather than guessing.

Parameterize the origins as a blend of the two known endpoints:

```
serve_start_x(lambda, stage) = 1.0 + lambda * (aligned[stage] - 1.0)
aligned = (1.0, 0.69, 0.34, -0.01, -0.35)
```

`lambda = 0` reproduces the baseline, `lambda = 1` the current aligned
ladder. Sweep `lambda` in {0.25, 0.5, 0.75} — at 0.5 the origins are
(1.0, 0.845, 0.67, 0.495, 0.325) — on calibration seeds, reporting for each
stage the oracle ≥2 rate, the mean landing offset, and the within-±0.30 m
fraction.

~~The landing contract must be **non-blocking** during this study.~~
**Superseded by the implementation.** Rather than switching the contract
off, `_landing_statistics` now takes the target a blend declares —
`(1 − λ) × (1.0 − aligned)` metres in front of the paddle start — so the
same ±0.10 m mean tolerance and ±0.30 m / 95% window gate an interior
blend meaningfully. Every λ in the study passed it (worst mean deviation
from target 0.049 m, minimum within-window 99.5%), which also confirms
the landing really does move one-for-one with the serve origin.

**Gate B.** Identify the largest `lambda` whose oracle clears 90% at every
stage. If `lambda = 0.25` already fails, alignment is incompatible with the
current deep geometry and the ladder itself needs redesign — shallower
depth steps, a lower deep-stage serve speed, or a wider deep fence — which
is a separate design change, not a parameter choice.

An honest possible outcome is that no `lambda > 0` clears the bar. That
result is worth having: it closes the serve-origin lever cleanly instead of
leaving it as a permanent open question.

*(Outcome: λ = 0.85 was the largest passing value and λ = 0.75 is the
recommended freeze. The ladder-redesign branch is not needed.)*

## Phase C — RESULT (2026-07-26): certified, but a null against the control

**λ = 0.75 certifies.** Frozen probes (stage 0 `run_up` 1.1, stages 1–2
`charge_gap` 1.0, stage 3 2.8, stage 4 2.4), held-out seeds 2000–2199,
200 episodes per cell, clean revision `9b960ea`, calibration range
declared. All eight blocking criteria pass.

| stage | candidate λ=0.75 | control (shipped probes) | control (own argmax probes) |
|---:|---:|---:|---:|
| 0 | 94.5% [90.4, 96.9] | 94.5% | 94.5% |
| 1 | 96.5% [93.0, 98.3] | 94.0% | 94.0% |
| 2 | 96.5% [93.0, 98.3] | 93.5% | 93.5% |
| 3 | 96.5% [93.0, 98.3] | 92.5% | 92.5% |
| 4 | 94.5% [90.4, 96.9] | 94.0% | **89.0% — fails** |

**The candidate is NOT demonstrably better than the control.** The arms
share seeds, so the comparison is paired, and exact McNemar per stage
gives p = 1.00, 0.302, 0.146, 0.115, 1.00. Every difference is
consistent with zero; the eye-catching 96.5% vs 92.5% at stage 3 is a
14-vs-6 discordant split. Stage 0 is bit-identical between arms (zero
discordant pairs) and carries no information at all. The defensible
statement is **"no regression detected, and λ = 0.75 clears the bar with
Wilson lower bounds of 90.4–93.0%"** — not superiority.

**The winner's curse is directly visible here, and it is large.** Giving
the control an equivalent tuning pass makes it *worse*: its own
calibration argmax at stage 4 (`charge_gap` 1.4, 94.0% on calibration)
drops to **89.0% held-out and fails**, while the stale shipped 1.7
passes at 94.0%. Calibration-to-held-out swings of 3–5 points are
routine at n = 200, which is the same order as every margin discussed in
Phases A–C. Treat all calibration rankings accordingly.

**What genuinely separates the arms is the crude controller, not the
oracle.** Paired McNemar on the placement-blind reflex: stage 1 p =
2.0e-4, stage 2 p = 5.9e-12, stage 3 p = 4.4e-4, stage 4 p = 1.8e-22
(113 vs 11 discordant, +51 points). `_crude` takes no probe arguments,
so this is unconfounded by tuning. But it is one hand-written reflex,
not a learner: the honest reading is **"reward is reachable without
placement skill under the blend, because the ball lands ~1.0 m closer to
where the paddle already is"** — a proxy for exploration density, not
evidence of learnability. Only a training run can establish the latter.

**The blend changes the task's option set — but the new option is a dead
end.** The non-blocking volley probe goes from 0% contact / 0% opening
volley at control stage 4 and 62% / 38% at control stage 3 to **100% /
100% at every candidate stage**, so under λ = 0.75 a pre-bounce opening
volley becomes geometrically reachable at all depths.

It is also *scoreable* in principle: `wall_ball.py:1339-1348` gates a
completed return on `_paddle_hit_since_last_wall` and, under
`rally_style = "open"`, accepts a pre-bounce hit. So the scoring does not
exclude volleys by definition.

It excludes them empirically. A stateful volley-then-rally controller —
intercept the serve pre-bounce, then fall back to the calibrated oracle —
scores **zero completed returns in 200/200 episodes** at λ = 0.75 stages
3 and 4, and at λ = 1.00 stage 4, against 2.29–2.35 mean returns for the
same oracle without the volley. Every one of those 600 episodes ends in a
double bounce, and since `bounce_count` increments only on a wall contact
that follows a paddle hit, a score of exactly zero means **the volleyed
ball never reaches the wall at all**.

The sweep's own `_volley_probe` could not have shown this: it parks after
the volley by design ("do not intentionally mount a post-bounce rally"),
so its 0% was an artifact of the probe, not a measurement of the
strategy.

**Verdict: not a shortcut.** An agent that learns to volley scores
nothing and eats the −1.0 double-bounce penalty every episode, so the
reward gradient points away from it. This does not block a recipe change.
Caveat: one hand-written volley controller, 200 seeds per cell; a
cleverer volley aim might convert where this one does not, though
0-for-600 with the ball never reaching the wall is not a marginal
result.

**Gate C: pass, with the scope stated.** What is certified is that a
stage-calibrated scripted oracle clears the 90% two-return bar at every
stage under λ = 0.75 on 200 unseen seeds, and that serve landings fall
where the blend declares. It says nothing about SAC's sample efficiency
or curriculum transfer. Caveats to carry forward:

- The gate is a point-estimate floor, not a confidence statement. Under
  Bonferroni across five stages the lower bounds at stages 0 and 4 fall
  to 89.5%, so *simultaneous* 95% coverage above 90% is not established.
- The five stages reuse one 200-seed block; they are correlated draws,
  not five independent replications.
- The held-out run discharges selection bias only for λ = 0.75's own
  rates. The *ranking* of 0.75 above 0.5, 0.8 and 0.85 remains
  contaminated — roughly 187 n=200 oracle cells were evaluated across
  Phases A and B before it was chosen.
- λ = 0.85 was never run held-out. It was not frozen because its
  stage-4 calibration point estimate of 91.0% carried a Wilson lower
  bound of 86.2%, i.e. it never cleared with margin — not because of any
  extrapolated calibration-to-held-out drop.
- The alignment contract is a self-consistency check, not an alignment
  discriminator: it passes identically at blend 0.0 and 1.0.
- The calibration artifacts (Phases A and B) carry no `seed_range` or
  git hash and were produced from an uncommitted tree; the certification
  artifact declares only `phase_b_grid.json`. Re-emit them with
  provenance before publishing anything that leans on them.

## Phase C — method (as specified before the run)

Once controller parameters and `lambda` are frozen and committed:

- Run the full paired sweep at 200 episodes per cell on **seeds 2000–2199**
  from a clean revision, both ladders, same seeds.
- Require every existing criterion: static geometry, serve alignment,
  hold-start invariance, within-stage monotonicity, ≥90% oracle
  feasibility at every stage with a reported binomial interval, >0% crude
  learnability, the 1.5× U-shape bound, and telemetry integrity.
- Serve alignment must be blocking for the candidate **against the target
  its `lambda` actually declares**, not against `paddle_start_x`. This
  requires the contract change listed in the prerequisites; without it a
  blended candidate is either ungated or fails by construction.
- Since Phase C's seeds are disjoint from the Phase A/B calibration range,
  `--calibration-seed-start` / `--calibration-episodes` are usable here and
  should be passed, alongside the code revision, command, and timestamp.

**Gate C.** Any blocking failure returns to Phase A or B. Do not weaken a
criterion to pass, and do not re-certify on 2000–2199 after inspecting it —
move to 3000–3199.

## Phase D — stage-1 training A/B

Unchanged from `design_wall_ball_serve_alignment.md`, which specifies it in
full: paired baseline-versus-aligned at stage 1, pre-registered restricted
-time primary outcome, variance pilot then powered sample size, conjunctive
go rule. Stage 1 already passes its mechanical checks on both seed sets, so
this phase does not depend on Phases A–C and **can run in parallel** with
them.

**This reverses the design note's ordering, deliberately.** Its Final
recommendation step 3 gates controller recalibration and certification on
the stage-1 A/B passing first. Phases A–C cost CPU-minutes while the A/B
costs GPU-days, and Phase A's result changes how the existing sweep
evidence should be read regardless of how the A/B turns out — so they run
concurrently rather than in series. If this plan is adopted, amend the
design note's step 3 to match rather than leaving the two documents
prescribing different orders.

It is the only phase that tests whether alignment helps SAC learn. Phases
A–C establish only that a scripted controller *can* play the geometry.

## Phase E — full aligned pilot

Blocked until Phase C certifies every stage and Phase D passes its go rule.
When both hold, size the pilot against the depth campaign's measured costs
rather than the historical 6M default:

- Stage 0 has cost 1.425M steps in both campaign runs. Stage 1 cost 0.97M
  in `20260721_004722` and 1.65M in `20260724_152530`. Five stages at that
  spread projects to roughly **6–8M steps**, so the historical 6M default
  is the optimistic end of the range rather than a safe ceiling.
- That projection is now known to be optimistic in a second way: stage 2
  is not merely expensive, it did not complete at all. Run
  `20260725_171747` spent 3.2M steps and 128 evaluations there without
  promoting, so no stage-2 cost figure exists — only a lower bound that
  already exceeds stages 0 and 1 combined. Do not size a pilot on the
  assumption that later rungs cost what the early ones did.
- At the measured ~90 FPS on one L4 that is about 18.5–24.7 h of wall
  clock, and no single session has yet demonstrated the upper end. Budget
  against a confirmed session ceiling or implement checkpoint-resume first.
- `early_stop_patience` must exceed the documented 0.5–1M-step
  post-advance recovery. The calibrated 20 (500k steps at `eval_freq`
  25k) ended `20260724_152530` at 3.65M with stage 2 having run only 575k
  steps since its advance.

Note on provenance: `CHANGELOG.md` and `docs/DECISIONS.md` describe
`20260724_152530` as "still on stage 1 at 2.55M", which was accurate when
written mid-flight. The run's final artifacts
(`reports/curriculum_stages.json`) record stage 0 exiting at 1.425M, stage
1 at 3.075M, and stage 2 entered but never promoted before the run stopped
at 3.65M. The figures above come from those final artifacts; the two
in-repo descriptions are stale and should be refreshed.

## No-go conditions

Stop and report rather than continuing if:

- Phase A shows a ceiling below the bar and Phase B finds no viable
  `lambda`. The serve-origin lever is closed; test recovery/interception,
  observation, or gate-noise hypotheses instead.
- Phase D's A/B is indistinguishable or favours the baseline. Alignment
  does not help learning, and deep-stage feasibility work is moot.
- Certification requires inspecting a reserved seed range twice. Retire it
  and use the next rather than reusing.

## Effort

| Phase | Cost | Blocking |
|---|---|---|
| prerequisites | one contained change to the sweep tool | yes |
| A — oracle grid | CPU-minutes per cell; hours total | yes |
| B — offset study | ~5 min per paired sweep; 3 blends | on A |
| C — certification | ~5 min; needs a clean commit | on A, B |
| D — stage-1 A/B | GPU-days; see design note for sizing | no |
| E — full pilot | ~19–25 GPU-hours, session ceiling permitting | on C, D |

Phases A–C are cheap enough to complete before committing any GPU time,
which is the point. The aligned ladder has already consumed two runs
totalling **31 GPU-hours** — `20260724_152530` (11h15m) and
`20260725_171747` (20h16m), both recording `serve_start_x`
1.0 / 0.69 / 0.34 in `curriculum_stages.json`, i.e. the aligned origins —
started despite the starter config's own header warning that the
stage-3/4 oracle fails the ≥90% bar. Both stalled at stage 2 and neither
reached the unplayable rungs, so deep-stage infeasibility does not
explain either outcome. The cost was incurred without the cheap checks
having been run first, and the second run's 20 hours went to answering a
question — "is patience the constraint?" — that a distribution of the
first run's stage-2 evaluations would have made much cheaper. Minutes of
scripted sweep, and a look at an existing eval distribution, can price
what GPU-hours cannot.
