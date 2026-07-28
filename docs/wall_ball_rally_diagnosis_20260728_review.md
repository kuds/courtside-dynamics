# Wall-ball campaign diagnosis — why no run rallies from the baseline, and what replaces the depth ladder

Status: review snapshot, 2026-07-28, written against `main` @ `52438b0`
(v0.23.0) plus the changes shipped in this PR (v0.24.0). Scope is the
whole wall-ball campaign with the depth-curriculum runs as the primary
corpus: `WallBallDepthCurriculum/sac/{20260721_004722, 20260727_004014,
20260727_233859}`, the aligned-arm runs `{20260724_152530,
20260725_171747}` through their review docs and final artifacts, the
`WallBallBaseline`/`WallBallBootstrap` era through
`wall_ball_baseline_review.md` and final artifacts, and `WallBall
20260713_192636` (the only 12-bounce rally on record). New evidence:
scripted probe batteries on the live geometry (~70 controller
configurations, n=100-200 per cell), a held-out ladder certification on
untouched seeds, and a 13-arm paired local SAC battery (Phase D stage-1
alignment A/B + a goal-fence structure A/B/C). Every load-bearing claim
inherited from earlier docs was re-measured here or is cited to its
run artifact.

The campaign goal, restated: a sustained rally at the workspace
baseline — fence (−4.7, −2.6), paddle start −3.9, serve origin 1.0,
serve speed 7.0. Best ever recorded on that task:
`bounce_count_ep_mean` **1.14** (run `20260727_004014`). The latest run
(`20260727_233859`, 0.22.0) scored **exactly −1.0 goal-stream reward
(zero paddle contact) from ~3.4M steps to its 6M-step end** — verified
here from the run's final `evaluations.npz`: 161 of 240 evaluations at
exactly −1.0, everything after 3.4M at or below −0.688.

## TL;DR

1. **The diagnosis, ranked:** the ladder's flat 3.0 promotion bar sits
   at or above the practical per-stage ceiling everywhere past stage 0
   (P1, highest confidence); each promotion buys a serve-receipt
   discontinuity plus (in 0.22.0) an advance package whose measured
   cost exceeded its benefit by millions of steps (P2); matched-stage
   mastery transfers weakly to the goal task and stage training can
   move *away* from it (P3); and the gate's estimator/staleness
   machinery lets a stalled run burn its whole budget invisibly (P4).
   The binding constraint is **the structure, not the task**: the goal
   geometry is feasible (scripted 95% ≥2-return rate, best mean 2.7)
   and learnable (crude placement-blind play completes ≥2 returns in
   80-84% of serves on the current constant-width goal fence; local
   cold SAC gets goal-task contact and returns within 100k steps).
2. **Structural verdict: replace the sliding-fence depth ladder.** The
   fence dimension is the wrong curriculum axis: moving it changes the
   *dynamics* (target clamping), which is what makes every promotion
   pay the replay/entropy/action-map taxes, and its difficulty enters
   through a serve-receipt geometry the fence itself never fixes. The
   replacement trains **at the goal fence from step one** and anneals
   the only variable the campaign ever measured to control receive
   difficulty: the serve origin (landing point). No dynamics change
   between rungs ⇒ no replay wipe, no entropy reset, no update pause,
   no action-map drift, and the final rung *is* the goal task.
3. **What the experiments showed** (pre-registered; see appendix):
   [S1/S2 RESULTS]
4. **Shipped in this PR:** recalibrated startup-certification probes
   (lead-charge oracle; held-out 95-100% ≥2 at every stage of the
   0.22.0 ladder — `--ladder release` now passes), a
   promotion-staleness guard (`stage_eval_budget`), the
   `WallBallGoalServeCurriculum` recipe implementing the replacement,
   a parameterized `wall_ball_oracle_action`, and the pre-registered
   6M-step Colab run config.

## 1. What the final artifacts say (verified)

### 1.1 Run 20260727_233859 (0.22.0, the primary corpus) to its actual end

The mid-flight review (`wall_ball_depth_curriculum_20260727_233859_review.md`,
written at ~5.5M) predicted completion would not change its
conclusions. The final artifacts bear that out and sharpen two numbers:

- Goal stream: peak +2.265 at **250k** (an untrained-ish, high-entropy
  policy touches and sometimes returns the goal serve), decay to the
  −1.0 zero-contact floor, flat to 6M. The best model the run has for
  the campaign goal was recorded in its first 4% of steps.
- Train-env reward by 250k bins (monitor CSVs, 8 workers): stage-0
  climb 1.46 → 8.75 by 1M; the 975k advance package craters it to
  **−0.65 mean / −1.00 median** in the 1.25-1.5M bin (below the 1.41
  of the untrained first bin); recovery is then a multi-million-step
  grind that only re-reaches the pre-advance level (8.66-9.70) in the
  **5.25-6M bins** — the advance package plus stage-1's harder receive
  cost the run ~4.3M steps of net regression, and it was still
  climbing when the budget ended (stage-1 best 2.82 vs the 3.0 bar).
- One promotion in 6M steps; 181+ evaluations on stage 1 with no
  mechanism able to notice the staleness.

### 1.2 The rest of the corpus, one line each

- `20260727_004014` (0.21.0): three promotions, 97-eval stage-3 stall,
  goal plateau 1.14; four checkpoints spanning 1.35M steps all score
  1.07-1.30 at the goal — the plateau is structural, not a
  checkpoint-selection artifact (`design_wall_ball_checkpoint_selection_audit.md`).
- `20260724_152530` / `20260725_171747` (aligned arm, 0.19/0.20
  fences): a clean single-variable patience experiment; stage 2's bar
  is unreachable — 0 of 128 evaluations ≥ 2.9, best 3-eval window
  2.567 vs the 3.0 bar, plateau slope −0.004/M
  (`wall_ball_aligned_patience_review.md`).
- `20260721_004722` (0.15.0): the ladder's best showing — transfer
  column 0.30 → 0.98 → 1.76 rising with each earned stage — and the
  origin of the promotion-shock and gate-noise price list.
- `WallBallBaseline` era: the fixed mid-depth lane (−3.2, −1.6)
  plateaued at 3.2-3.4 eval bounces across five healthy runs; budget,
  capacity, and reward magnitude all falsified as levers
  (`lessons_learned.md` 8, 19).
- `WallBall 20260713_192636`: the only 12-bounce rally ever recorded
  in this project — gamma 0.995, 1.5M steps, eval reward rising to ~54
  at 1.375M and still high at end. Long rallies are inside SAC's reach
  when the geometry cooperates. [VERIFY config when Drive fetch lands]

## 2. Fresh measurements on the live geometry

All scripted probes on burned calibration ranges (0-199, 1000-1199)
except the held-out certification (3000-3099, registered below).
Controllers invert the env's live action map (`ladder_certification`
module); the stale fixed-pivot `wall_ball_oracle_action` was not used.

### 2.1 The serve-receipt geometry is the ladder's real difficulty axis

Parked-paddle first-bounce offsets from `paddle_start_x` (100
eps/stage, seeds 1000-1099; std 0.15 m everywhere):

| stage | 0 | 1 | 2 | 3 | 4 (goal) |
|---|---|---|---|---|---|
| landing offset | −0.01 | +0.34 | +0.69 | +1.04 | +1.39 |

Every adjacent rung jumps the landing by ~0.35 m — above the ~0.25 m
approach-room threshold the serve-alignment campaign measured — so
*every* promotion, not just 0→1, changes the receive task by more than
the step that made Phase B's blend fail. The held-out certification
now warns on all four transitions. Meanwhile the fence slide itself
adds nothing to receive difficulty: it changes where the paddle may
stand, while the serve (origin 1.0, speed co-moving) determines where
the ball arrives.

### 2.2 No scripted reference clears the 3.0 bar anywhere

Across ~70 controller configurations (stock charge/run-up family, the
new lead-charge family, wait-position and wait-lead variants; n=100-200
per cell): best per-stage means **2.26 / 2.46 / 2.63 / 2.58 / 2.70**
(stages 0-4), ≥3-return rates 21-44%. Held-out (seeds 3000-3099,
n=100): oracle means 2.47-2.89, ≥2 rates 95-100%. The task review's
one 3.30 measurement at the goal did not reproduce with any variant
tried here; even taking it at face value, it is the only scripted
number ever recorded at or above the bar on any current stage.

Learned-policy evidence agrees: across four multi-million-step
exposures at stages ≥1 (two ladders, two serve arms), SAC's plateaus
were 2.1-2.8 — every one below the bar, every one above or near the
scripted ceiling. SAC beat 3.0 only at stage-0-class geometry
(stationary receive: 3.25-3.37). This is no longer "oracle evidence
alone" (the standing 0.21.0 objection): it is the joint pattern of
every learned run plus every reference controller.

At the measured exchange cadence (~124-127 steps at the goal fence,
verified), `episode_len = 750` caps an episode at ~5 completed
returns, so the 3.0 bar demands ~60% of the theoretical cap sustained
over a 3×60-episode window — a mastery definition, applied to rungs
whose only purpose is to be passed through.

### 2.3 The constant-width goal fence already fixed goal learnability —
and no run ever trained there

The 0.22.0 geometry work (constant 2.1 m fences, ≥1.3 m runway,
per-stage pivot) was motivated by the 0.21.0 goal-runway audit. It
worked: on the current goal fence the placement-blind crude controller
completes ≥2 returns in **80-84%** of serves (seeds 0-199 and
1000-1099) vs **9.5-16%** on the retired (−4.7, −3.0) fence. The
exploration-density collapse that justified easing into depth is gone
at the goal itself. But 0.22.0's run never trained past stage 1, so
this fact was invisible: the ladder's guard rails were guarding
against a hazard its own geometry fix had already removed.

Serve-origin blend at the goal fence (λ interpolating origin 1.0 →
−0.35; n=200, seeds 0-199), lead-charge oracle gap 3.0 and crude:

| λ | landing offset | oracle mean (≥2) | crude ≥2 |
|---|---|---|---|
| 0.75 | +0.38 | 2.67 (97.5%) | 77.5% |
| 0.60 | +0.58 | 2.56 (98.5%) | 92.5% |
| 0.45 | +0.78 | 2.47 (95.5%) | 94.5% |
| 0.30 | +0.99 | 2.42 (93.5%) | 87.5% |
| 0.15 | +1.19 | 2.41 (91.5%) | 84.0% |
| 0.00 | +1.39 | 2.56 (95.0%) | 80.0% |

Every rung of a goal-fence serve-origin ladder is feasible with margin
and crude-learnable; landing steps of ~0.20 m stay under the 0.25 m
approach-room threshold. (Full alignment λ=1.0 remains dead: crude
25%, oracle 69% — the Phase B approach-room threshold replicates on
the new fence.)

## 3. Diagnosis: ranked causes with confidence

**P1 — The flat 3.0 bar exceeds the practical per-stage ceiling for
every rung past stage 0.** (High confidence.) Evidence: §2.2's joint
scripted+learned pattern; the aligned patience experiment's 0-in-128
at stage 2; 0.22.0's 6M-step stage-1 residency ending at 2.82 and
still climbing; the episode-cap arithmetic. Consequence: the ladder's
expected behavior is exactly what was observed three times — one or
two early promotions, then a terminal stall at whichever rung the
policy's asymptote first drops below 3.0.

**P2 — Promotion is priced so that even earned promotions lose.**
(High confidence for 0.22.0's configuration; the components are
separately measured.) The 0.22.0 advance package (replay wipe + α=1.0
entropy reset + 50k pause) turned the run's only promotion into a
~4.3M-step net regression (§1.1). Under it sits the policy-independent
serve-receipt discontinuity (+0.35 m per rung, §2.1) — each promotion
hands the policy a materially different receive task at the exact
moment its buffer is deleted and its policy re-randomized. The
0.15-0.21 ladders paid smaller versions of the same tax (0.5-1M steps
per advance).

**P3 — Matched-stage mastery is a weak and sometimes negative proxy
for goal progress.** (Medium-high confidence.) The best transfer ever
bought was 1.76 goal bounces after two earned stages (run 1); 0.21.0's
three promotions bought 1.14 with a structurally flat plateau; the
aligned patience run measured stage-2 training *reducing* goal-task
skill (≥2 rate 6% → 0%) as it specialized; 0.22.0's disjoint stage-0
fence bought exactly nothing (−1.0 from 500k). The ladder spends
~100% of its budget optimizing proxies whose payoff at the goal has
never exceeded 38% of the target level.

**P4 — The gate cannot notice its own failure.** (Certain; mechanism
verified in code and in run histories.) Promotion staleness is
unmonitored (181 evals at stage 1); `best_metric_min_delta` (0.0083)
sits ~1/13 of a batch SE so noise resets early-stop patience
indefinitely; evaluation is unpaired, putting the full ~0.4-bounce SE
on every gate decision. A stalled gated run therefore runs to budget
exhaustion by construction.

What the data does NOT support: task infeasibility (§2.2-2.3 refute),
reward-design regression (the 0.22.0 audit verified the reward
machinery bit-exact; the flat-valley and one-and-done failures of the
baseline era remain fixed — local arms get contact within 25-75k
steps), capacity/budget limits (falsified twice in the baseline era,
and 0.22.0's budget was consumed by P2/P4, not by learning).

## 4. The pre-registered local experiments

Pre-registration (verbatim) in the appendix; protocol amendments were
all made before the affected arms launched. All arms: SAC, n_envs=8,
`gradient_steps=4` (1:2 update ratio — NEVER comparable to the 1:1
Colab runs; arms are only compared to their common-seed siblings),
gamma 0.995, 23-dim obs, no videos, no certification pass.

### S1 — Phase D stage-1 alignment A/B (2 pairs, 400k steps/arm)

[S1 RESULTS]

### S2 — goal-fence structure A/B/C (3 seeds × 3 conditions, 500k steps/arm)

[S2 RESULTS]

## 5. Structural verdict

[VERDICT — to be finalized from S1/S2; drafted per §2-3: replace the
sliding-fence ladder with the goal-fence serve-origin curriculum]

## 6. What ships in this PR

1. **Recalibrated startup certification** (`lead_charge` probe mode +
   frozen per-stage gaps 3.0/0.8/1.0/1.2/3.0): held-out ≥2 rates
   95/100/98/98/96% on seeds 3000-3099; `tools/depth_stage_sweep.py
   --ladder release` and every gated run's startup certification now
   pass their blocking criteria on the constant-width ladder. The
   remaining advisory warnings (no reference reaches 3.0; +0.35 m
   landing jumps at every transition) are true findings, not probe
   defects, and are part of §3's evidence.
2. **`stage_eval_budget` staleness guard** on the performance gate
   (`"stop"` or `"advance"` action) — closes P4's unbounded-stall
   failure mode.
3. **`WallBallGoalServeCurriculum` recipe** [DETAILS AFTER VERDICT].
4. **`wall_ball_oracle_action` mapping parameterized** (was hardcoded
   to the retired −1.7 pivot; defaults preserve the legacy `WallBall`
   contract).
5. Review doc + seed-ledger update + the next-run config below.

## 7. Pre-registered next Colab run (6M steps, GPU)

**Launch** (`notebooks/sb3_training.ipynb`, one L4):

```
ENV         = "WallBallGoalServeCurriculum"
ALGO        = None            # recipe default: SAC
SEED        = 0               # campaign convention; n=1 caveat stands
QUICK_TEST  = False
CONFIG_FILE = "auto"          # materializes the packaged starter
                              # wall_ball_goal_serve_curriculum.toml
```

Everything else is the recipe: 6M steps, n_envs 8, gamma 0.995 (only
pinned model kwarg; SB3 auto-entropy untouched), eval_freq 25k with
n_eval_episodes 60 matched + 30 goal-stream episodes, patience 20,
gate = serve-origin rungs (0.2 → 1.0) at threshold 2.5 window-mean-3,
`stage_eval_budget` 40 with forced advance, startup certification at
seeds 30000+ (expected verdict: pass — held-out precedent above). At
the measured ~85-90 FPS this is ~18.5-19.5h of wall clock — inside the
longest demonstrated session (20h16m), but checkpoint cadence 250k
means a session death loses at most ~45 min.

**Pre-registered success criteria** (decided before launch; evaluate
at run end against the run's own artifacts):

1. **Primary:** goal-task stream (`eval_info_final.csv`,
   `bounce_count_ep_mean` at the true serve) reaches a 3-eval window
   mean **≥ 2.0** at any point — ~1.75× the all-time best (1.14) on
   the identical eval task. Stretch: ≥ 3.0 (the campaign's rally
   definition).
2. **Structural health:** no rung residency exceeds 40 evaluations
   (guard-enforced by construction — verify `curriculum_stages.json`
   shows ≤ 2 forced advances (`advance_reason:
   "stage_eval_budget"`); more than 2 means the 2.5 scheduler bar is
   still miscalibrated and the review's gate section needs revisiting,
   whatever the primary shows.
3. **No zero-contact collapse:** goal-stream `paddle_hit_count_ep_mean`
   > 0 at every evaluation after 250k (the 0.22.0 failure signature).
4. **Long-horizon audit** (50 seeds, 5000-step cap, true goal task):
   mean completed returns ≥ 3.0 **or** ≥5-return survival ≥ 20%
   (vs 2.12 / 14% for the best ladder policy at goal-adjacent
   geometry).
5. **Comparability guard:** the goal eval task is byte-identical to
   `WallBallDepthCurriculum`'s (`eval_env_overrides` produce the same
   env; pinned by test), so 1-4 compare directly against the ladder
   history.

**Pre-registered escalation if the primary fails** (goal window mean
plateaus < 2.0 with the staleness guard keeping rungs moving): the
landing-point observation feature (obs 23 → 26; its "de-noised
stage-2+ stall" trigger has now fired twice) measured against this
run's plateau — NOT further reward or gate tuning (lesson 19), and
NOT a return to fence ladders. If instead the run stalls at rung 0-1
with goal-stream ≥ 2.0 never approached, treat the serve-origin axis
as falsified for SAC-learnability and test the observation feature
plus `episode_len` economics in separate pre-registered arms.

## 8. Seed ledger update

| Range | State | Use |
|---|---|---|
| 0-199 | burned | reused as calibration only: lead-charge gap grids (stages and goal rungs, n=100-200), controller-variant battery |
| 1000-1199 | burned | reused as calibration only: landing probes, stock-probe verification, goal-fence λ sweep (n=100) |
| 2000-2199 | burned | (unchanged; 2026-07-26 certification) |
| **3000-3099** | **burned 2026-07-28** | held-out certification of the frozen lead-charge probes on the 0.22.0 ladder (this PR); inspected |
| 3100-3199 | clean | reserved |
| 5000-5029, 6000-6039, 10000-10049, 20000-20199 | burned | (unchanged) |
| 30000+ | reserved | startup certification only (runtime) |
| **50000-50199** | **burned 2026-07-28** | S1/S2 final-model cross-evaluations on common seeds |

## Appendix A: pre-registration (verbatim)

[PREREGISTRATION]

## Appendix B: artifacts and reproduction

[ARTIFACTS]
