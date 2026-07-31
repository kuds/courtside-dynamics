# WallBallGoalRally replication — 2 of 3 seeds pass, one guarded collapse, deep play everywhere

Status: review snapshot, 2026-07-30, covering the three seed runs of
`WallBallGoalRally` (v0.24.0, git `5f38c82`): `20260728_225217`
(seed 0, reviewed in `wall_ball_goal_rally_20260728_225217_review.md`),
`20260729_140112` (seed 1, patience 60), `20260730_005134` (seed 2,
patience 60). Evidence: each run's `curriculum_stages.json`,
`eval_info.csv`, `best_model_meta.json`, and (seeds 0/2)
`best_model_long_horizon_eval.json`. Seed 1/2 differ from seed 0 only
in `seed` and `early_stop_patience` (20 → 60).

## TL;DR

- **The campaign result replicates: 2 of 3 seeds meet the goal, and
  the second success beats the first everywhere.** Seed 2's best
  window hit **3.750** (vs seed 0's 3.311), its audit reads **3.76
  mean returns, median 4, ≥5-survival 32%, longest rally 11**, and
  its failure mode inverted — 84% of audit episodes end with the ball
  running out *after* a rally rather than a missed recovery.
- **Seed 1 exposes the recipe's one open defect: training stability.**
  It was captured by the one-and-done basin for ~1.4M steps, escaped
  to 1.45, then **collapsed to zero paddle contact at ~2.5M** and
  never recovered; the degenerate-signal guard ended the run at
  3.175M with the 2.2M champion banked. Every safety mechanism
  shipped in 0.23/0.24 behaved exactly as designed.
- **All successful policies play deep.** Measured contact-x: seed 0
  −3.4 → −3.6, seed 2 ≈ −3.7, (and seed 1's blocked returns ≈ −3.95).
  Nobody camps the fence front: the serve lands at ≈ −2.46 but keeps
  travelling, and the learned strategy is to receive it deep and
  recover. This settles the "can it play deep with the fixed-pitch
  face?" question affirmatively — no paddle-orientation work was
  needed for deep rally play. (An earlier in-flight reading of seed 0
  as "front-edge style" was an unmeasured inference and is corrected
  here: no seed plays the front edge.)

## 1. The three seeds side by side

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| outcome | **goal met** | collapse (guarded) | **goal met, record** |
| first ≥2.0 window | 650k | never | 500k |
| first ≥3.0 window | 1.25M | never | 1.40M |
| best window | 3.311 @ 1.375M | 1.394 @ 2.15M | **3.750 @ 3.2M** |
| best model (sel/conf) | 3.333 / 3.267 @ 1.325M | 1.450 / 1.367 @ 2.2M | **3.833 (40% ≥5) / 3.717 @ 3.15M** |
| audit: mean (median, max) | 3.54 (3, 8) | n/a (champion audit in run dir) | **3.76 (4, 11)** |
| audit: ≥5-survival | 26% | — | **32%** |
| audit terminations | 48% dbl / 48% OOB | — | 16% dbl / **84% OOB** |
| contact-x (late) | −3.6 | −3.95 (blocks) | −3.7 |
| stop | patience-20 @ 1.825M | **degenerate guard @ 3.175M** | patience-60 @ 4.65M |

Seed 2's audit details worth keeping: 92% of episodes complete ≥1
return (4 zero-return episodes — a small dud tail seed 0 did not
have), 100% of hits and returns post-bounce, zero opening volleys,
20% of episodes outlast the 750-step training cap (vs 16% for
seed 0), one episode ran 1,649 steps with 11 returns.

## 2. The seed-1 collapse, as a case study

Timeline from its eval stream (60-episode batches):

- **0 → ~1.5M: one-and-done capture.** `bounce_count_ep_mean` pinned
  at 1.00 ± 0.05, ≥2-rate 0%, contact-x −3.95, reward ~3.6 — the
  paddle parks at its start, blocks exactly one return, and banks the
  single-cycle income. The classic attractor from the baseline era
  (`lessons_learned` "one-and-done"), in a deep-parked variant.
- **~1.5M → 2.2M: slow escape.** ≥2-rate 0% → 47%, mean to 1.47
  (banked best 1.45/1.367 @ 2.2M). The escape was unaided — evidence
  the basin is not absorbing, just slow.
- **~2.3M → 2.5M: destabilization; then collapse.** Mean 1.15 → 0.82
  → **0.02**, paddle contact 2.7 → 0.07 per episode, the rare
  touches vanish entirely, terminations flip to untouched-serve OOB.
  From 2.5M to 3.175M: reward −1.0 flat, zero contact — a full
  policy collapse *on its own training distribution*, shaped like
  SAC value-divergence/catastrophic forgetting, 700k steps with no
  recovery.
- **3.175M: the degenerate-signal stop fires** (5 consecutive
  zero-contact, flat-score evals), saving the remaining 2.8M steps.
  Best-model confirmation had already banked the pre-collapse
  champion.

This is a *new* failure mode for the campaign record: every prior
failure was "never learns" or "stalls under a bar"; this is "learns,
then falls off a cliff." It did not recur in seed 2 (which trained
1.5M steps past the equivalent point while setting records), so at
n=3 it reads as a low-frequency instability rather than a recipe
defect — but 2/3 is the honest reliability number until more seeds
exist.

**The pre-registered local 8-seed sweep (S3) has since completed**
(500k steps/seed, 1:2 update ratio — local-only numbers, never
comparable to GPU runs). Classification from the last 5 evaluations:
**2 of 8 captured** (seeds 302, 305: flat at ~1.0 with ≥2-rate below
10% — the seed-1 signature), 5 escaping (means 1.47-1.81, ≥2-rates
43-68%), 1 climbing (2.03). Every seed spends its first ~200-300k
steps in or near the basin; most exit unaided. The pre-registered
lever-A/B trigger (≥3 of 8 captured) did **not** fire, so no
`wall_reward_increment`/stability experiment is authorized by this
data — the recorded conclusion is that the basin is a *delay* with a
~25% chance (at this horizon and ratio) of still holding a seed at
500k, while the catastrophic late collapse remains a separate,
so-far-once-observed phenomenon this sweep cannot reach. Seed 305
was trending upward at its final evaluations (last eval 1.20), so
the 2/8 figure is conservative.

## 3. What the replication changes

1. **The structural verdict is confirmed at n=3**: direct goal-task
   training reached the campaign bar twice, with different seeds, in
   1.25-1.4M steps each time — a bar no ladder configuration touched
   in five runs. The diagnosis's ranking of causes needs no revision.
2. **The paddle-orientation project loses its strongest motivation.**
   Seed 2 rallies at 3.8 mean from −3.7 contact depth with the fixed
   face, and its OOB terminations are predominantly *successful*
   rallies running out, not sprayed returns (84% OOB with median 4
   returns). Pitch/yaw remains a plausible *consistency* lever, not a
   prerequisite for anything currently planned.
3. **Reliability, not capability, is the recipe's open item.** Next
   era inherits: expect ~1-in-3 runs to need a restart, watch the
   degenerate guard (it works), and treat the S3 sweep's
   basin-capture rate as the trigger for a `wall_reward_increment`
   or stability-lever A/B (separate pre-registration; one lever, one
   run).
4. **Reference policy for the era: seed 2's champion**
   (`20260730_005134/model/best_model.zip`, sha256 `0430c693…`, with
   its paired normalizer) — the strongest wall-ball policy the
   project has produced.

## 4. Recommended next phase

Unchanged from the seed-0 review, now with the replication in hand:
close this era and open the **true-baseline extension** (workspace
beyond x = −4.7 toward the ITF baseline at −7.985, serve-energy
rework) — starting, per doctrine, with scripted feasibility probes
and a certified task definition before any XML lands, and trying the
direct task before any curriculum. The deep-receive skill this era
proved is exactly the skill that phase stresses further.

## Seed ledger

No new scripted-probe burns. Long-horizon audits reused the standing
10000-10049 block. Local S3 sweep uses training seeds 301-308
(training-seed namespace) and no held-out env-seed evaluation.
