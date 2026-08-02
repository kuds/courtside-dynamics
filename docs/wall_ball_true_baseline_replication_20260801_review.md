# True-baseline replication — the split verdict: reliability 2/2, era skill 1/2, and the volley loophole that closes the wall-ball chapter

Status: review snapshot, 2026-08-02, of run `20260801_144043` — the
seed-1 replication of `WallBallTrueBaseline` (v0.25.0, git `f8886e2`,
NVIDIA L4, starter TOML sha256 `57ea74df…` byte-identical to seed 0's).
Scored against the same pre-registered criteria as seed 0
([`design_wall_ball_true_baseline.md`](design_wall_ball_true_baseline.md)
§8; seed-0 review:
[`wall_ball_true_baseline_20260731_132322_review.md`](wall_ball_true_baseline_20260731_132322_review.md)).
Evidence: the run's artifacts plus a checksum-verified local replay of
the banked champion for the first-contact criterion.

## TL;DR

- **Seed 1 out-scores seed 0 and fails the era.** It ran its full 6M
  budget (~20 h, no early stop — it was still setting records at step
  5,999,000), met the primary bars with margin (best window **3.072**,
  uncapped audit **3.08**), and became the first policy to post a
  **≥3.0 window on the capped stream** — the campaign's stretch mark.
  It then failed the deep-receive integrity criterion **completely**:
  measured mean first contact **x = −4.49** (bar ≤ −6.0), with **zero**
  of 50 audit episodes receiving deep.
- **How: it learned to volley.** In 48% of audit episodes the first
  contact is a pre-bounce volley at x ≈ −3.4 — the paddle sprints
  forward off its −7.9 start and takes the 11 m/s serve out of the air
  just behind the service line. The remaining episodes intercept the
  post-bounce leg early at ≈ −5.5. 78 of its 167 legal hits are
  pre-bounce (seed 0: zero). The pre-registration anticipated exactly
  this case: a policy that scores by refusing the deep receive does
  not open the era.
- **Training reliability is settled at 2/2.** No basin capture, no
  collapse, across 12.6M combined steps and 384 evaluations — the
  goal era's ~1-in-3 failure rate did not recur under the 0.24/0.25
  recipe structure.
- **The loophole is a finding about physics, not a bug to fix.** The
  wall cannot hit deep (measured rebounds: mean x = −1.0, never
  deeper than −6.1), so the only deep ball in any episode is the
  serve, once — and seed 1 proved the rational strategy is to refuse
  even that. Every available counter (deeper fence, pre-bounce bans,
  serve tricks) legislates against the environment's physics rather
  than changing them. **The wall-ball chapter closes here**; the
  0.26.0 loophole-closing and episode-cap changes are rejected as
  moot, and the next era moves to opponent play
  ([`design_paddle_tennis.md`](design_paddle_tennis.md)).

## 1. Scorecard

| # | Criterion | Bar | Seed 1 | Seed 0 |
|---|---|---|---|---|
| 1a | Best 3-eval window | ≥ 2.0 | **3.072** | 2.856 |
| 1b | Uncapped audit mean | ≥ 2.0 | **3.08** | 3.02 |
| 1c | Audit episodes ≥1 return | ≥ 80% | 100% | 100% |
| 2 | Audit mean first-contact x | ≤ −6.0 | **−4.49 (0% deep) — FAIL** | −6.49 (98% deep) — pass |
| 3 | Stretch: ≥3.0 window | ≥ 3.0 | **3.072 — pass** | 2.856 — not met (cap-compressed) |

Era verdict: **seed 0 opened the era; seed 1 did not** — a passing
score with a failed integrity check, exactly the case criterion 2 was
pre-registered to catch.

## 2. The run

Provenance: an exact replication — same starter TOML byte-for-byte,
same 6M budget/patience 60/eval cadence, `seed = 1` the only change
(plus the docs-only #65 merge in the trained commit). Startup
certification passed with a report *identical to seed 0's down to
every bounce count* — expected, since certification runs scripted
policies on its own reserved seed block, and a pleasant determinism
check of the full stack across two fresh containers.

Trajectory highlights (60-episode evals):

- **Faster start, slower middle.** Returning serves by 50k (seed 0:
  150k), level with seed 0 at 500k (1.70 both), then a slow stretch —
  at 1.1M it trailed seed 0 by ~400–500k steps of progress.
- **First ≥2.0 window at 1.175M**; oracle-band (1.98) crossing
  ~1.15M; long climb through a 2.0–2.6 plateau.
- **A genuine late game**: first ≥3.0 window at 5.325M, best window
  3.072 at 5.35M, and the **champion banked at step 6,000,000** —
  selection 3.35 / 23.3% ≥5, confirmation **3.417 / 25% ≥5**, the
  highest confirmed evaluation in the project's history on any task.
- **Essentially uncompressed**: mean timeout rate 1.3% (max 6.7%) —
  its faster forward game fits inside the 750-step cap that clipped
  10–20% of seed 0's episodes, so its stream numbers are close to its
  true level.

## 3. The audit (uncapped, 50 episodes, seeds 10000–10049)

- Completed returns mean **3.08 ± 1.45** (min 1, p10 1.9, median 3,
  p90 5, max 7). Survival ≥1 **100%**, ≥2 90%, ≥3 58%, ≥4 30%,
  ≥5 22%, ≥7 2%.
- **Style diagnostics — the loophole in the run's own artifacts**:
  `opening_volley_episode_rate` **0.48**, pre-bounce legal hits
  **78/167**, post-bounce completed-return rate 0.55. (Seed 0: zero
  opening volleys, zero pre-bounce hits.)
- Episode length mean 406, max 859; only 2% exceed the training cap.
- Terminations **inverted vs seed 0**: 62% double bounce / 34% OOB /
  4% stall (seed 0: 64% OOB / 36% double bounce). The volleyer's
  rallies die failing to recover; the deep receiver's die hitting
  long.

## 4. First-contact measurement (criterion 2)

Method identical to seed 0's: local deterministic replay of the
banked champion on the audit seed block, artifacts checksum-verified
against `best_model_meta.json` (`942a2deb…`, `3ad415cc…`); replay
return mean 3.00 corroborates the audit's 3.08.

- First contact, all episodes: **mean −4.486 ± 1.05**, p10 −5.59,
  median −5.19, p90 −3.36, shallowest −3.25. **0% ≤ −6.0.**
- Split by type: **volley first contacts 24/50 (48%) at mean −3.40**
  (range −3.54..−3.25 — just behind the ITF service line at −2.5);
  post-bounce first contacts 26/50 at mean **−5.49** (range
  −5.71..−5.18 — intercepting the rising post-bounce leg mid-court,
  never near the baseline).
- The replay's 48% volley rate matches the run's audit diagnostics
  exactly.

Geometry of the exploit: the serve's pre-bounce flight crosses the
fence-reachable zone (front edge −2.6) for ~0.4 s at comfortable
height before bouncing at −4.5, and nothing in the open rally style
prices a pre-bounce hit. From a −7.9 start the paddle cannot reach
the *serve* corridor in time — but it can reach the volley window,
and seed 1 found it.

## 5. What the replication settles

1. **Training reliability: 2/2.** Across both seeds: no one-and-done
   capture, no value collapse, every dip recovered within 1–2 evals,
   both champions banked and confirmed. The recipe structure's open
   stability question from the goal era is, at n=2 on the harder
   task, resolved favorably.
2. **Era skill: 1/2 — and the miss is informative, not noise.** Seed 1
   did not fail to learn the deep receive; it learned something
   *better under the rules as written*. Two seeds, two coherent,
   opposite styles (deep receiver at −6.5; volleyer at −3.4), with
   the volleyer scoring higher — the forward game is at least as
   strong an attractor as the deep game.
3. **The wall cannot host a baseline era.** Probe T3b measured wall
   rebounds landing at mean x = −1.0, never deeper than −6.1: the
   wall only hits short, so depth can never be *instrumentally*
   useful against it — it must be decreed (fences, serve energy,
   style bans), and each decree invites the next exploit. Deep play
   will be learnable as a strategy only against an opponent whose
   returns actually land deep.

## 6. Decision: the chapter closes

Recorded as a deliberate stop, not an abandonment:

- The **0.26.0 loophole-closing changes are rejected** (fence
  front-edge retreat, pre-bounce restrictions, serve rework): each
  fights measured physics for an aesthetic outcome and teaches the
  campaign nothing transferable. Journal lesson: *an environment's
  dominant strategy is set by its physics; a reward or rule decree
  that opposes the physics buys compliance at best, and an exploit
  at worst.*
- The **episode-cap change (750 → ~1,100) is moot** with no further
  wall-ball runs planned; the analysis stands in the seed-0 review
  if the env is ever revisited.
- Wall-ball's deliverables stand: calibrated physics, the
  probes-first methodology, the training-safety machinery (now 2/2
  unattended), and two confirmed champions — a deep receiver
  (`20260731_132322`, first contact −6.49) and a volleyer
  (`20260801_144043`, confirmed 3.417) — which become a
  stylistically diverse opponent pool for the next era.
- Next era: **paddle 1v1 on the full court**, where an opponent's
  returns make depth matter naturally. Proposed design and probe
  plan: [`design_paddle_tennis.md`](design_paddle_tennis.md).

## Seed ledger

No new burns. The audit and first-contact replay reused the standing
10000–10049 audit block; startup certification used its reserved
30000+ block. Clean blocks 3100–3199 and 4100–4199 carry forward to
the next era.
