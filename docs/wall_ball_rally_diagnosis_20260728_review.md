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
- One promotion in 6M steps; the final `curriculum_stages.json`
  records stage 1 at entry 975k → exit 6,000,000 with **201
  evaluations, promoted: false**, best selection 2.817 (confirmation
  2.617) banked at literally the run's last evaluation — a run ended
  by its budget while still improving against a bar it never
  approached, with no mechanism able to notice the staleness.

### 1.2 The rest of the corpus, one line each

- `20260727_004014` (0.21.0): stage exits 800k / 1.7M / 3.0M, then
  stage 3 for its remaining 3.0M steps — final `curriculum_stages.json`
  records **120 evaluations, promoted: false, final window
  [2.37, 2.55, 2.45]** against the 3.0 bar. Goal plateau 1.14; four
  checkpoints spanning 1.35M steps all score 1.07-1.30 at the goal —
  the plateau is structural, not a checkpoint-selection artifact
  (`design_wall_ball_checkpoint_selection_audit.md`).
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
  (`lessons_learned.md` 8, 19). Corpus note: **`WallBallBootstrap`
  has no runs in Drive at all** — a title search over the training
  archive returns nothing, so the bootstrap recipe (already marked
  HISTORICAL) contributed calibration ideas, never training evidence.
- `WallBall 20260713_192636`: the only 12-bounce rally ever recorded
  in this project. Config verified from the run's `config.json`: the
  *legacy* WallBall task (v0.8.0, 22-dim obs, no fence, receive at the
  fixed −1.7 home column), SAC with gamma 0.995, fixed `ent_coef`
  0.02, buffer 2M, 1.5M steps; eval reward rose to ~54 by 1.375M and
  held. Two readings matter for this diagnosis: gamma 0.995 was
  present in the one long-rally run (the depth recipes' 0.21.0 choice
  is corroborated), and rally-chaining is demonstrably inside SAC's
  reach — on a task whose serve lands at the paddle's ready column.
  The campaign's missing skill has always been the deep *receive*, not
  the rally.

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
   defects, and are part of §3's evidence. Note the
   `WallBallDepthCurriculumAligned` recipe inherits these probes and
   its startup certification is *expected to fail* feasibility at its
   deep stages (~80-87% measured) — that is the certifier correctly
   reporting the aligned deep geometry Phase A already falsified, and
   any aligned pilot remains blocked on certification regardless.
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
| **3000-3099** | **burned 2026-07-28** | held-out certification of the frozen lead-charge probes on the 0.22.0 ladder (this PR). Inspected twice — once by an intermediate probe candidate (gaps 1.0/1.4 at stages 1/3), once by the frozen set (0.8/1.2); the frozen gaps were chosen from n=200 calibration (seeds 0-199) before any 3000-3099 inspection, so the held-out property holds, but the range must not be reused |
| **3100-3199** | **burned 2026-07-28** | held-out certification of the `WallBallGoalServeCurriculum` rungs (100 eps/cell; all blocking criteria pass, zero warnings) |
| 4000-4199 | clean | reserved — the next untouched block; do not open until a frozen configuration needs certifying |
| 5000-5029, 6000-6039, 10000-10049, 20000-20199 | burned | (unchanged) |
| 30000+ | reserved | startup certification only (runtime) |
| **50000-50199** | **burned 2026-07-28** | S1/S2 final-model cross-evaluations on common seeds |

## Appendix A: pre-registration (verbatim)

```
# Pre-registration: local experiment battery, 2026-07-28

Written BEFORE any experiment below was run. Environment: 4-core CPU,
SB3 SAC, repo v0.23.0 @ 52438b0, branch
claude/wall-ball-curriculum-diagnosis-0j8pjz. Measured throughput:
env-only 6486 steps/s single-thread; SAC n_envs=8, 1 torch thread,
533 env-steps/s (24k-step probe, learning_starts=1000).

Local SAC arms use n_envs=8. AMENDED BEFORE LAUNCH (after a 30k-step
smoke on throwaway seed 999, before any registered arm ran): the repo's
train() sets gradient_steps=-1 (true 1:1 update ratio — the ratio every
Colab run used), which measures at ~71 env-steps/s locally; a 1:1
13-arm battery does not fit the budget. All local arms therefore use
model_kwargs gradient_steps=4 (a 1:2 update ratio), identical across
every arm. CPU arms are NEVER compared to Colab runs — only to their
paired local siblings (common seed, identical config except the
treatment variable). Horizons resized to fit: S1 400k steps/arm,
2 pairs (seeds 101, 102); S2 500k steps/arm, 3 seeds. Decision rules
adjusted for 2 S1 pairs: "alignment helps" requires pooled mean
delta-AUC >= +0.25 with BOTH pairs positive (mirrored for "hurts");
S1 AUC window becomes evals 125k..400k; S2 primary window becomes
evals 300k..500k.

## Seed ledger plan

- Scripted probe CALIBRATION reuses burned ranges 0-199 and 1000-1199
  (burned = unusable as held-out; fine for calibration/diagnostics).
- Held-out certification of any frozen probe/ladder configuration:
  seeds 3000-3099 (from the clean 3000-3199 block; 3100-3199 stays
  clean). This burn will be registered in the review doc.
- Post-hoc cross-evaluation of trained A/B arms on common seeds:
  50000-50199 (previously unused namespace; registered as burned).
- Startup certification keeps its reserved 30000+ block; not touched
  by hand here.
- SAC training seeds (a different namespace from env eval seeds):
  S1 pairs use 101, 102, 103; S2 triples use 201, 202, 203.

## P: scripted probe battery (calibration seeds, exploratory)

P1. Serve-landing offset per current-ladder stage (parked paddle,
    100 eps/stage, seeds 1000-1099): verify review 20260727_233859's
    -0.01/+0.34/+0.69/+1.07/+1.42 m ladder.
P2. Certification-module controllers (parked/crude/oracle with the
    recipe's stock probes) on the live 5-stage table, 100 eps/cell,
    seeds 1000-1099: verify oracle means ~1.51/2.47/2.40/1.86/2.20 and
    the stage-0/3 feasibility failures.
P3. Improved oracle (ballistic lead while charging) calibrated per
    stage on seeds 0-199; report best per-stage mean bounce + >=2 rate.
    Purpose: recalibrate startup-certification probes so the ladder the
    next run trains is certified by a credible reference, and measure
    whether ANY scripted reference clears the 3.0 promotion bar
    per stage.
P4. Goal-fence serve-origin blend sweep: fence (-4.7,-2.6), start -3.9,
    home -3.65, serve 7.0, serve_start_x = 1.0 + lambda*(-1.35) for
    lambda in {0, 0.25, 0.5, 0.75, 0.9, 1.0}; oracle (stock + improved)
    and crude cells, 100 eps, seeds 1000-1099 + 0-99 replication.
    Purpose: feasibility/learnability map for a fixed-goal-fence
    serve-origin curriculum on the CURRENT (2.1 m) fence — all prior
    blend numbers are from the retired 0.21 fences.
P5. Exchange cadence at the goal fence (steps between completed
    returns, improved oracle): verify ~130.

No pass/fail bars for P1/P2/P5 (verification). P3/P4 feed design; the
frozen configuration that ships gets a held-out certification on
3000-3099 with the standard blocking criteria (oracle >=2 on >=90% of
serves per rung; crude >0%; monotone parked<crude<oracle; landing
contract where declared).

## S1: Phase D stage-1 alignment A/B (local adaptation)

The pre-registered Phase D design (design_wall_ball_serve_alignment.md)
specifies checkpoint-forked continuations with 1.5M-step horizons and
8-24 pairs on GPU. That is out of local budget. Local adaptation =
its pre-registered cold-start fallback, shrunk: independently seeded
cold-start pairs with common seeds, fixed horizon, restricted-time
outcomes. This cannot authorize a full aligned-ladder pilot by itself
(the design note's go rule is not evaluable at this scale); it CAN
answer directionally whether stage-1 serve alignment helps early SAC
learning, which is what the structural verdict needs.

- Arms: fixed single-stage gate at the 0.22.0 stage-1 geometry
  (fence (-2.9,-0.8), start -2.1, home -1.85, serve 5.5).
  Baseline serve_start_x=1.0; aligned serve_start_x=0.69.
- Config: WallBallDepthCurriculum recipe, performance_gate replaced by
  the single stage; ladder_certification=None; record_video=False;
  n_eval_episodes=30; final_info_eval=True with final_eval_episodes=10
  (goal-task transfer diagnostic); eval_freq=25k; gamma 0.995 (recipe);
  early_stop_patience=None (fixed horizon); 500,000 steps/arm;
  3 pairs, seeds 101/102/103 shared within pair.
- PRIMARY outcome: paired difference (aligned - baseline) in
  matched-stream AUC = mean of bounce_count_ep_mean over evals at
  125k..500k inclusive.
  Decision rule: "alignment helps stage 1" if pooled mean delta >=
  +0.25 bounces AND >=2 of 3 pairs positive. "Alignment hurts" if
  <= -0.25 AND >=2 of 3 negative. Else inconclusive.
- SECONDARY (reported with the primary, no substitution): first
  timestep with 3-eval window mean >= 2.0 on the matched stream;
  final-model 2x2 cross-eval (both arms x both stage-1 serve origins)
  on common seeds 50000-50099, mean bounce_count; goal-task transfer
  stream level.
- Caveat pre-declared: matched streams score different serve origins,
  so the primary conflates task ease with learning speed — that is the
  campaign-relevant quantity (each arm gates on its own geometry), and
  the 2x2 cross-eval carries the mechanism reading.

## S2: goal-fence structure A/B/C (the structural-verdict experiment)

Question: at the campaign goal geometry (fence (-4.7,-2.6), start -3.9,
home -3.65, serve 7.0), which training distribution gives SAC a
learnable task: the true serve, an aligned-blend serve, or a mixture?

- Common: single-stage gate at goal fence; 600,000 steps/arm; 3 seeds
  201/202/203 x 3 conditions (paired via common seed); eval_freq 25k.
  AMENDED BEFORE LAUNCH (no arm had run): instead of a matched
  training-distribution stream + separate goal stream, every S2 arm
  carries ONE identical evaluation stream — the true goal task
  (serve_start_x=1.0), n_eval_episodes=30 — by omitting serve_start_x
  from the single gate stage so the matched evaluator keeps the
  recipe's eval override. This makes selection and the outcome stream
  identical across conditions; arm B/C skill on their own training
  serves is recovered post-hoc in the cross-eval instead.
- Conditions (differ ONLY in training-serve origin):
  A "direct": serve_start_x = 1.0 (the true goal task).
  B "aligned75": serve_start_x = -0.0125 (lambda=0.75; landing ~0.39 m
    ahead of paddle start).
  C "mixture": serve_start_x drawn per episode uniformly from
    {1.0, 0.6625, 0.325, -0.0125} (lambda in {0,.25,.5,.75}).
- PRIMARY outcome: per-condition goal-task learning signal = mean of
  the goal-stream bounce_count_ep_mean over evals at 350k..600k, paired
  across common seeds.
  Decision rules (pre-registered):
  1. "Goal task directly learnable with current shaping" if condition A
     reaches pooled primary >= 1.0 OR shows a monotone rise with final
     3-eval window >= 1.0 in >=2 of 3 seeds.
  2. "Serve-origin curriculum dominates" if (B or C) beats A on the
     pooled primary by >= +0.3 bounces with >=2 of 3 pairs agreeing
     in sign.
  3. If all conditions are flat at ~0 goal-task contact by 600k, the
     serve-origin lever is insufficient at the goal fence and the
     structural verdict must consider observation/geometry levers
     instead (landing-point feature; episode economics).
- SECONDARY: matched-stream levels (B's matched stream reads skill at
  lambda=0.75); contact rate (paddle_hit_count_ep_mean) on the final
  stream — separates "never reaches the ball" from "reaches but cannot
  rally"; final-model cross-eval at the true goal on common seeds
  50100-50199.

## Explicitly out of scope locally

Full-length (6M) runs; any comparison of local arms to Colab runs; any
gate-threshold change justified only by scripted-oracle evidence
(standing 0.21.0 decision — bar changes require either learned-policy
evidence or a redesigned gate whose bars are certified per rung by
references AND validated against learned behavior).
```


## Appendix B: artifacts and reproduction

[ARTIFACTS]
