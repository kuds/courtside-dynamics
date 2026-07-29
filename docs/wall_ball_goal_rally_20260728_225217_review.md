# WallBallGoalRally 20260728_225217 review — the campaign goal is met

Status: review snapshot of run `WallBallGoalRally/sac/20260728_225217`
(v0.24.0, git `5f38c82`, seed 0, one L4, launched 2026-07-28 22:52 UTC,
early-stopped cleanly at 1,825,000 of 6,000,000 steps after ~6.6 h,
~77 FPS). First run of the recipe that
`wall_ball_rally_diagnosis_20260728_review.md` shipped, scored here
against that document's §7 pre-registered success criteria. Evidence:
`config.json`, `reports/ladder_certification.json`,
`reports/curriculum_stages.json`, `metrics/eval_info.csv` (73
evaluations), `model/best_model_meta.json`, and
`reports/best_model_long_horizon_eval.json` (+ episodes CSV). The
config was checked against the pre-registration at launch (one defect
caught and fixed — see provenance note).

## TL;DR

**A sustained rally from the workspace baseline — the campaign's goal
since the depth ladder was conceived — was achieved in 1.25M steps,
and every pre-registered success criterion passed, including the
stretch bar.** The goal-task evaluation stream sustained
`bounce_count_ep_mean ≥ 3.0` over 3×60-episode windows (16 window
crossings from 1.25M; best window 3.311) — the exact bar that five
curriculum runs and ~25M cumulative training steps never passed
beyond a stage-0 geometry. The 50-seed long-horizon audit of the best
model reads **3.54 mean completed returns** (median 3, p90 6, max 8),
≥5-return survival 26%, 100% of episodes completing at least one
return — versus 1.14 / ~0% / every-episode-double-bounce for the best
ladder policy on this same task. The run then stopped itself on a
settled plateau with 70% of its budget unspent.

## 1. Scorecard against the pre-registration

| # | Criterion (review doc §7) | Bar | Result | Verdict |
|---|---|---|---|---|
| 1 | Goal-stream 3-eval window | ≥ 2.0; stretch ≥ 3.0 | ≥2.0 at 650k (2.094); ≥3.0 first at 1.25M, 16 crossings, best 3.311 at 1.375M | **pass + stretch** |
| 2 | No zero-contact collapse | contact > 0 every eval after 250k | contact rose monotonically 0.88 → ~2.9; never near zero | **pass** |
| 3 | Long-horizon audit (50 seeds, 5000-step cap) | mean ≥ 3.0 **or** ≥5-survival ≥ 20% | mean **3.54** and ≥5-survival **26%** — both clauses | **pass** |
| 4 | Comparability | eval task identical to the ladder's goal task | pinned by test; best selection 3.333 vs prior all-time 1.14 | **pass** |

## 2. What the run did

- **Startup certification passed** (30 eps, seeds 30000+): oracle 2.53
  mean / 96.7% ≥2, crude 2.07 / 83.3%, parked 0.0 with zero contact,
  landing +1.44 m; the only warning was the expected informational one
  (no scripted reference reaches the 3.0 marker).
- **Learning curve** (60-episode goal-task evals every 25k): 0.25 at
  25k → 1.20 at 250k (already past the 1.14 all-time record, 4% of
  budget in) → 2.70 at 800k → first 3.0-window at 1.25M → oscillation
  in the 2.9–3.3 band thereafter. The ≥2-return episode rate went 0%
  → 93–98%; ≥3 to 60–77%; the first ≥5 episode appeared at 475k.
- **Best model** at 1,325,000 steps: selection 3.333 mean / 21.7%
  ge5-rate, independently confirmed at 3.267 / 18.3% before being
  crowned (`confirm_best_eval`). Final promotion-window read
  [3.18, 3.27, 2.90].
- **Early stop fired exactly as designed**: 20 evaluations after the
  last accepted best (1.325M + 20×25k = 1.825M), on a plateau, with
  the champion banked and the full artifact suite (including the
  long-horizon audit) written. Wall clock ~6.6 h of a budgeted ~19.5.
- The policy the audit describes: all play post-bounce from inside the
  baseline fence (zero opening volleys in 50 episodes; 94.9% of
  completed returns post-bounce — the emergent style the 0.19 design
  doc wanted from geometry rather than fault rules), episodes ending
  48% double-bounce / 48% out-of-bounds / 4% stall, and rallies ending
  by fault rather than by clock (longest audit episode 1183 of 5000
  steps; only 16% of episodes outlast the 750-step training cap).

## 3. What this confirms, and what it does not

Confirmed, now with GPU-scale 1:1-ratio evidence on top of the local
battery: the diagnosis chain held end to end. The goal task was
learnable all along on the 0.22.0 constant-width geometry; the ladder
was the obstacle, not the path; and the campaign bar that had been
functioning as an unreachable promotion gate was an entirely
reachable *performance level* once training happened on the task
itself. SAC again finished above every scripted reference (3.3+ vs
the lead-oracle's 2.5–2.7), repeating the stage-0 pattern at the
goal.

Not yet established:

- **Replication.** Every number here is seed 0, n=1 — the standing
  lesson-7 caveat. A second seed of the unchanged recipe is the
  cheapest high-value follow-up (~7 h of L4 given early stop).
- **The ceiling.** The plateau sits at ~3.0–3.3 with a 26% ≥5 tail.
  Whether more budget, more patience, or a capability lever (the
  landing-point observation stays shelved — its trigger condition, a
  stall *below* target, did not fire) moves the tail is unknown; the
  plateau mirrors the old fixed-task era's 3.2–3.4 on a much harder
  task, so diminishing returns are plausible.
- **Episode economics barely bind** (16% of audit episodes reach the
  750-step cap), so `episode_len` changes are not currently
  motivated.

## 4. Recommended next steps

1. **Second seed, unchanged config** (lesson 7): same recipe, same
   TOML, `SEED = 1`. Success reading: window ≥ 3.0 reached by ~2M.
2. **Close the campaign phase in the docs** (done in this PR:
   DECISIONS outcome entry + this snapshot) and treat further
   wall-ball work as the *next* phase: extending play beyond the
   x = −4.7 workspace toward the true ITF baseline (−7.985), which
   requires the XML workspace + serve-energy work the depth design
   explicitly deferred. That phase should begin the way this one
   ended: scripted feasibility probes and a certified task before any
   training run.
3. If rally *consistency* (the ≥5 tail) becomes the next target on
   this geometry, the pre-registered candidates are the landing-point
   observation feature and — untested but never-disfavored —
   `wall_reward_increment`; per lesson 6, one lever, one run.

## Provenance notes

- A first launch (`20260728_224706`) was stopped at ~5 minutes: its
  notebook `TOTAL_TIMESTEPS` variable silently overrode the TOML's 6M
  with 3M — caught by the pre-flight config check this review's
  parent doc prescribed. The stub folder in Drive holds only startup
  artifacts and should not be read as a run; its `config.json`
  (total_timesteps 3,000,000) distinguishes it. The mirror-image of
  the 233859-era stale-TOML incident: the explicit-kwargs-win
  precedence bit in the opposite direction this time.
- The long-horizon audit rolled its standing seed block 10000-10049
  (already burned for this purpose in the ledger; no new burns).
- `evaluations.npz` carries the 5-episode reward stream only (the
  goal-task info stream owns selection in this recipe); anyone
  plotting it should use `eval_info.csv` for the headline metric.
