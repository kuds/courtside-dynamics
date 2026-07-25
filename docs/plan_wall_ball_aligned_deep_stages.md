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

Neither Phase A nor Phase B is runnable from the current CLI. Both need a
small, contained change to `tools/depth_stage_sweep.py`:

- **Probe parameters are hardcoded per stage.** `oracle_run_up` and
  `oracle_charge_gap` live in the `ALIGNED_STAGES` table (stage 0 uses
  `run_up=1.1`; stages 1–4 use charge gaps 1.0, 1.0, 1.8, 1.7) and
  `_oracle` raises when both or neither are set — exactly one must
  configure it. A grid search therefore needs a **probe-mode** override,
  not merely a value override: something like a repeatable
  `--oracle-probe STAGE=run_up:VALUE|charge_gap:VALUE` that replaces the
  stage's probe key outright. A value-only flag cannot express "try
  `run_up` at stage 3", because that requires clearing `charge_gap`.
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

## Phase A — diagnose: oracle or geometry? (blocking)

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

## Phase B — parametric offset study

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

The landing contract must be **non-blocking** during this study: interior
blends will not satisfy a ±0.30 m band that assumes full alignment, and
that is the point. Report it; do not gate on it.

**Gate B.** Identify the largest `lambda` whose oracle clears 90% at every
stage. If `lambda = 0.25` already fails, alignment is incompatible with the
current deep geometry and the ladder itself needs redesign — shallower
depth steps, a lower deep-stage serve speed, or a wider deep fence — which
is a separate design change, not a parameter choice.

An honest possible outcome is that no `lambda > 0` clears the bar. That
result is worth having: it closes the serve-origin lever cleanly instead of
leaving it as a permanent open question.

## Phase C — freeze and certify

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
which is the point. The aligned ladder has already consumed one 11h15m
run — `20260724_152530`, whose `curriculum_stages.json` records
`serve_start_x` 1.0 / 0.69 / 0.34, i.e. the aligned origins — started
despite the starter config's own header warning that the stage-3/4 oracle
fails the ≥90% bar. That run stalled at stage 2 and never reached the
unplayable rungs, so deep-stage infeasibility does not explain its
outcome; the cost was incurred without the cheap check having been run
first. Minutes of scripted sweep can price a ladder that GPU-hours
cannot.
