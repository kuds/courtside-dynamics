# WallBallDepthCurriculumAligned patience experiment — the stage-2 bar is unreachable

Status: review of the matched pair
`WallBallDepthCurriculumAligned/sac/20260724_152530` (v0.19.0, git
`3fd9fb3`, seed 0, 3,650,000 of 6M steps, 11h15m43s, 90 FPS) and
`WallBallDepthCurriculumAligned/sac/20260725_171747` (v0.20.0, git
`89dbe61`, seed 0, 6,275,000 of 7M steps, 20h16m14s, 85 FPS). Both
cold-start SAC on one L4, `n_envs = 8`, `eval_freq` 25,000,
`n_eval_episodes` 30. The second run changed exactly two values:
`early_stop_patience` 20 → 60 and `total_timesteps` 6M → 7M.

Evidence: `stage_summary.txt`, `reports/curriculum_stages.json`, and
`reports/best_model_long_horizon_eval.json` for both runs (read
directly); `metrics/eval_info.csv`, `metrics/progress.csv`, and
`reports/best_model_long_horizon_episodes.csv` (analysis pass — see
*Method and evidence limitations*); and the 200-episode scripted sweep
recorded in `design_wall_ball_serve_alignment.md`. The design being
tested is `design_wall_ball_serve_alignment.md`; the follow-on work is
`plan_wall_ball_aligned_deep_stages.md`.

## TL;DR

- **The experiment succeeded and falsified its hypothesis.** Patience
  was not the binding constraint at stage 2. Given 3.2M steps and 128
  evaluations — 5.6× the first run's 575k and 23 — the policy still
  never promoted.
- **The 3.0 promotion bar is not reachable at stage 2.** Across 128
  evaluations not one reached 2.9; the stage maximum was a single
  2.833 and the best 3-eval promotion window was **2.567**, a 0.433
  shortfall. Both promoting stages cleared on windows of exactly
  3.0111.
- **Stage 2 plateaued, it did not run out of time.** It climbed for
  ~1.9M steps, peaked around 5.0M, then held flat for the final 50
  evaluations (mean 2.141, sd 0.254, slope −0.004 per 1M steps).
- **Promotion in this curriculum is a tail event on a plateau.** Stage
  0 and stage 1 each promoted via a single upper-tail spike off
  plateaus roughly 0.4 higher than stage 2's. Stage 2's plateau is too
  low for its tail to reach 3.0, not too short-lived.
- **This falsifies the standing "post-advance recovery failure"
  diagnosis** (CHANGELOG 0.15.1). Recovery happened — the stage-2 best
  rose 2.00 → 2.63 — and the wall remained.
- **The extra 3.2M steps traded goal-task skill for stage-2 skill.**
  Matched stage-2 performance improved while stage-4 transfer fell:
  every two-return episode was lost (≥2 rate 6% → 0%) and maximum legal
  paddle hits fell 2 → 1.
- **Stages 0 and 1 are bit-identical across the two runs**, which makes
  this an unusually clean single-variable comparison.

## What the runs did

Identical through the first 3.075M steps — same exit timesteps, same
evaluation counts, same promotion windows, same `best_vec_normalize.pkl`
SHA-256s (`71a9852b…` at stage 0, `5d38d287…` at stage 1). The patience
change cannot act before a stall, and neither stage 0 nor stage 1
stalled.

| | 20260724_152530 | 20260725_171747 |
|---|---|---|
| Stage 0 | 0 → 1,425,000, 57 evals, promoted | identical |
| Stage 1 | 1,425,000 → 3,075,000, 66 evals, promoted | identical |
| Stage 0 / 1 promotion window | [2.533, 3.133, 3.367] / [3.3, 2.8, 2.933] | identical (both mean 3.0111) |
| Stage 2 residency | 3,075,000 → 3,650,000, **23 evals** | 3,075,000 → 6,275,000, **128 evals** |
| Stage 2 best (selection / confirm) | 2.00 / 2.20 @ 3,150,000 | **2.633 / 2.70** @ 4,775,000 |
| Stage 2 exit window | [1.933, 1.600, 1.667] | [1.867, 2.667, 2.333] |
| Promoted to stage 3 | no | **no** |
| Headline final | 1.67 | 2.33 |
| Stop cause | patience 20 | patience 60 |

Early stopping fired exactly as configured in both runs: the second
stopped at `4,775,000 + 60 × 25,000 = 6,275,000`, leaving 725,000 steps
of budget unspent.

## Diagnosis: the stage-2 ceiling is below the bar

`bounce_count_ep_mean` over the 128 stage-2 evaluations: max 2.833,
p95 2.52, p90 2.40, median 2.033, mean 2.032, sd 0.316. Evaluations at
or above 3.0: **zero**. At or above 2.9: **zero**. The best 3-eval
rolling window — the gate's actual decision statistic — peaked at
**2.567**, and the 3.0 bar sits about 3.8 standard deviations above the
window-mean distribution.

The trajectory in five equal blocks: 1.665, 1.960, 2.260 (peak), 2.151,
2.131. The post-recovery slope is +0.091 per 1M steps, and over the
final 50 evaluations it is −0.004 per 1M — flat. Extrapolating even the
generous slope needs several million further steps to lift the window to
3.0.

For contrast, the two stages that promoted did so from higher plateaus:

| | plateau mean | max eval | evals ≥ 3.0 |
|---|---:|---:|---:|
| Stage 0 (57 evals) | — | 3.367 | 3 |
| Stage 1 (66 evals) | 2.534 | 3.300 | 1 |
| Stage 2 (128 evals) | **2.141** | **2.833** | **0** |

Stage 1 promoted on a single spike to 3.30 off a plateau of 2.534 after
66 attempts. Stage 2 had 128 attempts off a plateau 0.39 lower and never
came within 0.43 of the bar. Dispersion is nearly the same in both
(sd 0.264 vs 0.254); only the level differs.

The scripted sweep points the same way: the calibrated oracle averages
**2.43** bounces at aligned stage 2, and its mean declines across the
ladder (2.57, 2.58, 2.43, 2.17, 1.86) while the promotion bar stays flat
at 3.0. The trained policy's 2.633 is already *above* that probe. Two
cautions before leaning on this: the oracle is a feasibility probe
rather than a proven performance ceiling — a learned policy exceeded it
at every stage — and its timing was calibrated for the fixed-origin
geometry and inherited unchanged by the aligned ladder, so it may
understate what stage 2 permits (see the confound in
`design_wall_ball_serve_alignment.md`).

**Counter-argument, recorded rather than dismissed.** The final block
contained the stage maximum (2.833 at 6,100,000), two further evals
above 2.6, and the widest dispersion of any post-recovery block; its
best window (2.544) statistically ties the stage-wide best set 1.3M
steps earlier. Since stages 0 and 1 promoted from exactly this pattern —
flat mean, widening tail, eventual spike — a re-expanding tail is the
one signature consistent with "still reachable", and patience cut the
run off with budget remaining. The level argument still bites: that tail
would have to lift the window 0.433 off a plateau that had not moved in
50 evaluations.

## Diagnosis: stage-2 training moved away from the goal task

Goal-task transfer of the best model, 50 held-out seeds (10000–10049) at
stage-4 aligned geometry:

| | 20260724 | 20260725 |
|---|---:|---:|
| Completed returns (mean) | 0.62 | 0.50 |
| ≥1 completed return | 56% | 50% |
| **≥2 completed returns** | 6% | **0%** |
| legal paddle hits (max) | 2 | **1** |
| Episode length (mean / max) | 195.9 / 468 | 164.0 / 291 |
| Terminations | 90% double-bounce | 52% / 48% out-of-bounds |

Three readings that the raw table invites and the episode data does not
support:

1. **The ≥1 rate change is not significant.** Only four seeds flipped
   (McNemar exact p = 0.375). The unambiguous losses are all three
   two-return episodes and the maximum legal-hit count.
2. **The out-of-bounds surge is a whiff, not an overhit.** All 24
   out-of-bounds episodes never reached the wall, and 17 of 50 episodes
   recorded zero paddle contact — the earlier run had none at all. The
   serve now crosses untouched and leaves the court before a second
   floor bounce, relabelling the same failure.
3. **The reward-component shift carries no independent information.**
   `rew_oob` and `rew_double_bounce` sum to exactly −1.00 per episode in
   both runs; `rew_oob_mean` is the out-of-bounds fraction by
   construction.

Conditional on making contact the newer policy is *better*: hit-to-
completed-return conversion rose 50.0% → 75.8%. The return skill
improved; ball acquisition degraded. With stages 0 and 1 bit-identical
there is no earlier competence that could have been overwritten, so this
is geometry specialisation, not catastrophic forgetting — a policy tuned
to stage-2 launch timing (paddle start −2.7, serve origin 0.34, serve
speed 6.0) arriving late at stage-4 geometry (−3.9, −0.35, 7.0).

The `opening_volley` collapse (26% → 2%) should not be reported as a
regression: opening volleys are 0-for-14 across both runs on this
evaluation.

## Training health

No optimiser pathology anywhere in the 3.2M-step stage. `critic_loss`
drifts up about 46% across stage 2 without spikes; `actor_loss` dips
negative immediately after each advance — the expected signature of a
freshly cleared replay buffer — then recovers monotonically. The buffer
wipe and 50,000-step update pause are visible exactly as configured
(`n_updates` advances 256 over 49,368 environment steps at the
boundary).

`ent_coef` collapsed to ~8×10⁻⁴ within the first 200,000 steps and never
recovered, sitting in the same band through stage 1 (which promoted) and
stage 2 (which did not) — so it does not discriminate between them. Each
advance produces only a transient ~2× re-inflation from the tuner;
`reset_entropy_on_advance` was **false** for this run, so 0.20.0's new
lever was not exercised. The `ent_coef_loss` sign difference against the
previous run (+1.049 vs −1.695) corresponds to a policy entropy about
0.15 nats either side of the −3.0 target and is within the per-dump
noise of that channel; it should not be read as a regime change.

The headline train-versus-eval gap (recent train 7.983 against an eval
headline of 2.33) is **not** overfitting to the training reset
distribution: against the stage-matched evaluator the gap is +0.06. The
entire gap is that the reported evaluation sits two curriculum rungs
ahead of where the policy trains.

## Recommendations

1. **Do not re-run this configuration.** Two runs totalling 31 GPU-hours
   now agree that stage 2 ends the aligned ladder, for a reason more
   time does not address.
2. **Shrink the stage 1→2 transition, or decouple its variables.** Each
   rung moves fence, paddle start, and serve speed together. The
   transfer evidence — 17 zero-contact episodes at the faster serve —
   points at serve timing specifically. Either interpolate a rung
   between stages 1 and 2, or ramp `serve_speed` on its own schedule so
   the binding variable is identifiable.
3. **A flat 3.0 bar is wrong for a ladder whose ceiling falls.** The
   oracle's achievable mean declines monotonically across stages while
   the threshold does not. Stage-scaling it is defensible, but lowering
   it alone only promotes a weaker policy into stage 3, whose oracle
   feasibility already fails at 85.5%.
4. **Run Phase A of `plan_wall_ball_aligned_deep_stages.md` first.** It
   is CPU-only and it determines whether the deep-stage numbers reflect
   the geometry or a stale controller — which changes whether
   recommendation 2 or 3 is the real fix.
5. **Stop treating patience as the lever.** Both runs stopped exactly
   when `early_stop_patience` said they would, and in the second case it
   stopped a plateau rather than a climb. Patience 60 behaved correctly;
   the question it was raised to answer is now settled.

## Method and evidence limitations

- One seed (0) per configuration, as with every run in this campaign.
  The stage-2 comparison is a genuine single-variable contrast because
  stages 0 and 1 are bit-identical, but it remains n = 1.
- The curriculum history, stage summary, and long-horizon evaluation
  were read directly from the run artifacts. The finer statistics — the
  0-of-128 counts, the 2.567 best window, the block slopes, the plateau
  bootstrap, and the per-episode contact analysis — come from an
  analysis pass over `eval_info.csv`, `progress.csv`, and the episodes
  CSV, and have not been re-derived independently line by line.
- The 50-seed goal-task evaluation cannot resolve the ≥1 return rate;
  a follow-up needs substantially more seeds.
- The episodes CSV exports no ball velocity, landing position, or
  contact-x column, so "arrives late" is inferred from contact counts
  and `wall_contact_count` rather than measured. Adding a ball exit
  position or exit speed would make the whiff-versus-overhit
  distinction direct.
- `reset_entropy_on_advance` was off, so this run says nothing about
  whether 0.20.0's entropy reset would change the stage-2 outcome.
- The scripted-oracle comparison inherits the calibration confound
  documented in `design_wall_ball_serve_alignment.md`.
