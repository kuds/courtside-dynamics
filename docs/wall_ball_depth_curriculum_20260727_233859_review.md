# WallBallDepthCurriculum 20260727_233859 review — the 0.22.0 ladder stalls at stage 1

Status: review of run `WallBallDepthCurriculum/sac/20260727_233859`
(v0.22.0, git `e312486`, seed 0, 6M-step budget). Written while the run
was still in flight at ~5.5M steps (2026-07-28 ~18:00 UTC); the last
2M steps are flat on every stream, so completion is not expected to
change the conclusions. This is the first run of the 0.22.0 recipe
(constant 2.1 m fences, live `paddle_home_x` pivot,
`return_shaping_scale = 0.15`, entropy reset + replay clear on
advance, gamma 0.995 carried from 0.21.0).

Evidence: `reports/curriculum_stages.json`, `model/best_model_meta.json`,
`metrics/evaluations.npz` (goal stream, 220 evals), the 21 milestone
rollout CSVs under `media/videos` (250k–5.25M), `metrics/monitor/0.monitor.csv`,
`config.json` / `run_config.toml`, the previous run `20260727_004014`'s
stage history and eval curve, a line-level audit of the 0.21.0/0.22.0
changes at `e312486`/`db2280d`, and fresh physical experiments run on
this exact geometry (scripted-ladder sweep re-run on the live stage
table at 15 and 100 episodes/cell; 50-episode oracle probes at stages
0–2 and the goal stage; serve-landing and action-map probes; a
step-level shaping/refund audit). Experiment conclusions were
independently replicated on disjoint seeds before being cited here.

## TL;DR

- **One promotion in 5.5M steps.** Stage 0 cleared at 975k (window
  [2.70, 3.25, 3.22]). The run then sat on stage 1 for **181
  consecutive evals** (the previous run cleared its stage 1 in 36).
  Best confirmed stage-1 selection value after all that:
  `bounce_count_ep_mean` **2.283** (confirmation 2.483) at 5.475M —
  ~0.5–0.7 below the 3.0 bar, never close after 1M.
- **The 0.22.0 advance package is expensive.** The 975k promotion
  (replay-buffer wipe + entropy reset to α=1.0 + 50k update pause)
  cratered train-env reward from 8.92 to **−0.59 mean / −1.00 median**
  — worse than an untrained policy's first bin (+1.41) — and took
  ~525k steps to recover a positive median. Four more promotions at
  that price is ~2M steps of a 6M budget spent on shock recovery.
- **Goal-geometry transfer is zero, and died before the promotion.**
  The goal stream (`evaluations.npz`, `eval_info_final.csv`, and every
  milestone video) shows the policy never touching the ball from
  ~500k onward: episode reward exactly −1.0 (all shaping refunded,
  OOB/double-bounce at ~110 steps) for essentially every eval after
  3.4M. Part of this is by design — the 0.22.0 no-refuge ladder makes
  stage 0's fence ([−2.3,−0.2]) **disjoint** from the goal fence
  ([−4.7,−2.6]), so goal transfer must now be earned through
  promotions that never came. But it means the paddle-orientation
  design note's gate (a) — beat `20260727_004014`'s goal ceiling —
  is currently failing outright, not marginally.
- **New policy-independent finding: the serve-receipt discontinuity at
  the stage-0→1 boundary.** With `serve_start_x` fixed at 1.0, the
  stage-0 serve lands **exactly at the parked paddle** (offset −0.01 m,
  100-episode probe), while stage 1 lands it **+0.34 m in front**
  (stage 2: +0.69, goal: +1.42). Stage 0 teaches "stand still and the
  ball arrives"; stage 1 is the first time the policy must approach to
  receive — with its replay buffer freshly wiped and its entropy
  randomized. This is the largest discontinuity we measured anywhere
  in the ladder.
- **The 0.22.0 ladder shipped sweep-uncertified, and it does not pass
  its own sweep.** `tools/depth_stage_sweep.py` still hardcodes the
  retired narrowing ladder (and pivot −1.7), so the new geometry was
  never calibration-swept. Re-running the sweep on the live stage
  table (100 eps/cell): oracle bounce means **1.51 / 2.47 / 2.40 /
  1.86 / 2.20** across stages 0–4 — feasibility (≥2 returns in ≥90%
  of serves) fails at stages 0 and 3, the difficulty-inversion
  detector fires at stage 1 (stage 0 is anomalously *hard* for the
  scripted probes), and no scripted reference clears the 3.0
  promotion bar at any stage. Scripted references are demonstrably
  not a ceiling for SAC (the live policy hit 3.25 at stage 0 where
  the oracle managed 1.5), so this does not prove the bar is
  unreachable — but the run's 2.3–2.5 stage-1 plateau sits exactly at
  the scripted ceiling, and nobody currently has evidence the bar is
  reachable on this geometry either.
- **The 0.22.0 code itself is sound.** Every changelog claim checked
  out at line level: the fence table, the live pivot (usable action
  shares reproduce to four decimals: 0.491/0.428/0.424/0.475/0.633),
  and the outgoing return shaping is genuinely potential-based —
  failed outgoing legs net exactly 0.0 shaping across 120 audited
  refund events, and per-step reward components sum to the scalar
  reward within 9e−16. The defects found are in the periphery: a
  stale sweep tool, a stale scripted oracle, a wrong comment, a stale
  Drive config copy.

## 1. What the run did

Stage history (`reports/curriculum_stages.json`, rewritten atomically
on every stage close — its 02:49 UTC timestamp *is* the promotion
record; `performance_gate.py:458–530`):

| stage | fence | serve | entry → exit | evals | outcome |
|---|---|---|---|---|---|
| 0 | [−2.3, −0.2] | 5.2 | 0 → 975k | 39 | promoted, window [2.70, 3.25, 3.22] |
| 1 | [−2.9, −0.8] | 5.5 | 975k → still there at 5.5M | 181+ | best 2.283/2.483 @ 5.475M |

Train-env reward (monitor, 250k bins, approximate global-step
mapping): healthy stage-0 climb 1.41 → 8.92 over 0–1M; **crater to
−0.59 mean / −1.00 median in the 1.25–1.5M bin** after the advance;
recovery from ~1.5M; then a slow, real grind — +1.37 reward/M over
3–5M, with the episode mode flipping from one completed return
(median ≈ 2.45) to two (median ≈ 7.7) in the 4.25M+ bins. The run is
not frozen; it is converging far too slowly toward a bar it has
never approached.

Goal stream (220 evals, 25k cadence): peak **+2.265 at 250k**, decay
to the −1.0 zero-contact floor, transient partial recoveries around
0.9M, 2.05–2.3M (+1.43 max) and 2.6–3.2M, then flat −1.0 (std ~5e−8)
from ~3.4M. All 60 milestone-rollout episodes from 500k to 5.25M end
with zero paddle contact. Note for anyone eyeballing the Drive
videos: **milestone videos always roll the goal geometry**
(`train.py:1106–1118` passes the resolved eval env to
`VideoRecordCallback`, which builds a fresh env; the gate never syncs
it) — mid-run video quality says nothing about current-stage skill.

Comparison at equal timesteps, previous run `20260727_004014`
(0.21.0, old ladder): stage exits at 800k / 1.7M / 3.0M, goal-stream
reward +1.62 / +2.32 / +1.62 at 975k / 1.7M / 3.0M vs this run's
−1.00 / −1.00 / +0.96. On outcomes, 0.22.0 is decisively behind —
with the honest caveat that this is not a controlled A/B (fence
widths, pivot, outgoing shaping, and the advance package all changed
at once, and the old ladder's shared front-court interval gave its
early stages free positional overlap with the goal).

Why early stop never fired across 181 stale evals: patience counts
consecutive matched-stream evals without an *accepted* new best,
and the acceptance threshold (`best_metric_min_delta` = 0.0083) is
~1/13 of a 60-episode batch's standard error (~0.11), while
confirm-best banks the **weaker** of the two batches
(`info_dict_eval.py:794–801`) — so the stored best sits low in the
noise band and ordinary noise keeps resetting the clock. A
tie-on-mean can even reset patience via `ge_5_rate` alone
(`info_dict_eval.py:758–763`). Promotion staleness is not monitored
at all.

## 2. Code audit of the 0.21.0/0.22.0 changes

All eight changelog claims verified at line level (details in the
audit notes; key cites): constant-2.1 m stage table
(`recipes.py:897–933`), live `paddle_home_x` property + setter
(`wall_ball.py:827–866`) with the old silent no-op confirmed in
history, outgoing shaping pays 0.15 × gap-closed per step and is
refunded on illegal wall contact, weak-return floor drops, and any
episode end (`wall_ball.py:1646–1667, 1427–1442, 1757–1762`),
entropy reset fills `log_ent_coef` to log(1.0) and clears the Adam
moments while leaving `target_entropy` untouched
(`performance_gate.py:577–602`), the replay wipe drops the whole
buffer while collection continues and gradient updates pause for 50k
steps, gamma is the only pinned model kwarg, and promotion now reads
unpooled 60-episode matched batches with a 3-eval window mean.

Empirical confirmations from the physical experiments: usable
action-x shares **0.4909 / 0.4284 / 0.4242 / 0.4749 / 0.6329**
(analytic and env-stepped, matching the changelog's 0.49/0.43/0.42/
0.48/0.63 within rounding) — the re-pivot works, action-map collapse
is *not* the stage-1 bottleneck; exchange cadence on the goal
geometry 128.6 steps mean (p10–p90 118–139), so gamma 0.995 prices
the next return at 0.52 vs 0.27 at the old 0.99 — the 0.21.0
motivation holds on the new geometry; shaping refunds are bit-exact.

Defects found around the change (none in it):

1. **`tools/depth_stage_sweep.py` still encodes the retired ladder**
   (old fences, fixed pivot −1.7). The 0.22.0 geometry shipped
   without sweep certification, and when swept it fails its own
   blocking criteria (see §3).
2. **`scripted_policies.wall_ball_oracle_action` is stale** — it
   inverts the action map with hardcoded home −1.7 / old spans
   (`scripted_policies.py:132–138`), scoring ~0.5 bounces flat on
   0.22.0 geometry. Any future calibration that uses it silently
   measures the mismatch.
3. **`recipes.py:709–710` comment states wrong shares**
   (0.52/0.48/0.49/0.54/0.66 — matches neither code, changelog, nor
   tests).
4. **The Drive run-config copy is stale**: it still says
   `total_timesteps = 3_000_000`, `early_stop_patience = 20` (0.15.0
   vintage; the repo TOML says 6M). The run got 6M/60 only because
   the notebook passed explicit kwargs, which beat the file.
   `resolve_run_config_file` reuses an existing Drive copy untouched
   (`notebook_utils.py:154–157`), so the drift persists until
   someone passes `overwrite=True`.

## 3. Physical experiments on the live geometry

Scripted-ladder sweep re-run with the exact live stage table
(driver reusing the tool's functions; 15 eps/cell primary, 100
eps/cell confirmation):

| stage | parked | crude | oracle (100 ep) | oracle ≥2 rate | vs 3.0 bar |
|---|---|---|---|---|---|
| 0 | 0.00 | 1.93 | 1.51 | 49% | −1.49 |
| 1 | 0.00 | 2.04 | 2.47 | 92% | −0.53 |
| 2 | 0.00 | 1.94 | 2.40 | 95% | −0.60 |
| 3 | 0.00 | 1.86 | 1.86 | 75% | −1.14 |
| 4 | 0.00 | 2.00 | 2.20 | 93% | −0.80 |

Blocking criteria: feasibility fails at stages 0 and 3; monotonicity
fails at stage 0 (crude beats oracle); the difficulty-inversion
detector fires at stage 1 (oracle 1.64× easier than stage 0 —
i.e. stage 0 is anomalously hard for scripted play). Caveat both
ways: probe parameters were calibrated on the old geometry, so these
are a lower bound on the scripted ceiling; and SAC demonstrably
exceeds scripted references here (live policy 3.25 at stage 0). Per
the 0.21.0 decision, none of this justifies lowering the bar by
itself — but the 4.39-predictive-controller measurement that
justified 3.0 was taken on the *old* stage 0 and does not transfer;
on the current ladder the bar is uncalibrated in both directions.

Stage probes (50 oracle episodes/stage, replicated on disjoint
seeds): stage-aware baseline oracle 2.14 / 2.30 / 2.18 at stages
0–2 (replication: 2.53 ± 0.94 at stage 1); ≥3 rates 0.22–0.37; no
scripted reference clears 3.0 anywhere. The run's stage-1 plateau
(2.28–2.48) sits exactly in the scripted-reference band.

Serve-receipt geometry (policy-independent, parked paddle, 100
eps/stage): first floor bounce lands at offset **−0.007 / +0.343 /
+0.692 m** from `paddle_start_x` at stages 0/1/2 (sweep concurs:
+0.37/+0.72/+1.07/+1.42 at stages 1–4). The 0→1 boundary flips the
receive task from "stationary" to "approach", at the same moment the
advance package deletes the buffer and randomizes the policy.

Goal-stage feasibility: the stock probe family caps at 2.16 mean
bounces (94% OOB — returns spray laterally because the probe doesn't
lead the ball while charging). A variant adding a ballistic lead
during the charge reaches **3.30 mean, 74% ≥3** on held-out seeds —
the goal task is attainable; the binding defect in the scripted
family is control, not geometry. Structural note: at ~130-step
cadence, `episode_len = 750` caps an episode at ~5 completed returns.

## 4. Interpretation

Ranked causes for the stage-1 stall, by strength of evidence:

1. **The advance package converted a promotion into a catastrophe**
   (measured: train reward below untrained levels for ~250k steps,
   ~525k to recover). The 0.20.0 design expected the wipe+reset to
   help escape the "no exploration budget in new geometry" failure;
   the first exercised instance cost half a stage-0's worth of
   learning. This alone plausibly explains most of the 181-eval
   residency relative to the previous run's 36.
2. **The serve-receipt discontinuity** (measured, policy-independent)
   made stage 1 the first stage requiring approach-to-receive, so
   little of the stage-0 competence transferred — precisely when the
   buffer that encoded it was deleted.
3. **The bar may sit above the stage's practical ceiling** (open):
   no reference — scripted or learned — has produced ≥3.0 on this
   stage-1 geometry; the policy's own asymptote (2.3–2.5 and still
   creeping at +1.37 train-reward/M) is consistent with either "needs
   more time" or "bar unreachable". The ladder being sweep-uncertified
   means we cannot currently distinguish these.

## 5. Next steps

Ordered; each tied to a measured defect or a pre-registered lever.

1. **Close out the run as a negative result for the advance package
   as configured** (it completes its 6M budget ~18:40 UTC today).
   Archive normally; do not extend. The paddle pitch/yaw gate (a)
   ("0.22.0 must beat 20260727_004014's goal ceiling") is failing at
   −1.0 goal reward vs +3.33, so that proposal stays gated.
2. **Soften the advance package before the next run** — two levers,
   both already in the recipe surface:
   (a) set `entropy_reset_value` to a calibrated value (e.g. ~10×
   the pre-advance α) instead of the α=1.0 refill — the measured
   α=1.0 reset threw away a working policy's behavior for 500k
   steps; (b) run the pre-registered **clear-vs-keep replay A/B**
   (docs list it as escalation-gated; a 525k-step regression is the
   escalation trigger firing). Keep the 50k update pause — it wasn't
   implicated.
3. **Smooth the stage-0→1 serve-receipt boundary.** Give stage 0 a
   landing offset comparable to stage 1's (+0.3–0.4 m in front of the
   paddle, via stage-0 `serve_speed`/`serve_start_x`), so
   approach-to-receive is learned before the first promotion instead
   of after it. This is adjacent to the unrun **Phase D stage-1
   alignment A/B**, which our data now independently motivates; note
   partial alignment already cleared every sweep stage (Phase B), and
   full alignment stays dead (Phase B falsified it).
4. **Re-certify the ladder**: update `tools/depth_stage_sweep.py` to
   the 0.22.0 stage table and per-stage pivot, fix
   `wall_ball_oracle_action`'s hardcoded mapping, and fold in the
   ballistic-lead controller (goal ceiling 2.16 → 3.30) so the sweep's
   oracle is a credible reference again. Then re-derive per-stage
   evidence about the 3.0 bar from the improved controller — do not
   move the bar before that, per the standing 0.21.0 decision.
5. **Add a promotion-staleness guard**: stop (or at least alert) when
   a stage has gone N evals (e.g. 60) without a promotion, and/or
   raise `best_metric_min_delta` toward ~1 SE (0.1) so patience can
   actually expire on a plateau. This run burned ~4.5M steps after
   its last promotion with no mechanism able to notice.
6. **Hygiene** (small): refresh the stale Drive TOML copy
   (`overwrite=True` once), fix the `recipes.py:709` share comment,
   and document that milestone videos/`evaluations.npz` roll the goal
   geometry so mid-run footage isn't misread as stage skill.
7. **If the goal number stays flat after 2–4**: the changelog's own
   named next levers remain (decouple `serve_speed` from depth;
   `wall_reward_increment`), and the landing-point observation
   feature's pre-registered trigger ("a de-noised stage-2+ stall")
   has now arguably fired twice — this run is a de-noised *stage-1*
   stall on cleaner geometry.

## Appendix: artifacts and reproduction

- Run: `training_runs/WallBallDepthCurriculum/sac/20260727_233859`
  (Drive). Best model = stage-1 champion at 5.475M
  (2.283/2.483); `model/stage_bests/stage_00/` holds the stage-0
  champion (3.25/3.27 at 950k).
- Experiments in this review: sweep driver re-using
  `tools/depth_stage_sweep.py` functions with the live stage table
  (15 and 100 eps/cell, seeds 0..N−1); stage probes 50 eps/stage
  (seeds 1000–1049), replicated at 30 eps (seeds 5000–5029);
  serve-landing probes 100 eps/stage (seeds 1000–1099); goal-stage
  probes 50 eps (calibration 1000–1011, evaluation 2000–2049);
  shaping audit reconstructed refunds bit-exactly over 53,339 steps.
  Seeds 1000–1099/2000–2049/5000–5029/6000–6039 are now burned for
  scripted probes on this geometry.
