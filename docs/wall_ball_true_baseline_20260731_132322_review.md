# WallBallTrueBaseline run 1 — the era opens: every primary criterion passes, and the episode cap is now the binding container

Status: review snapshot, 2026-08-01, of run `20260731_132322` — the
first GPU run of `WallBallTrueBaseline` (v0.25.0, git `877ca53`,
seed 0, NVIDIA L4, starter TOML sha256 `57ea74df…` verbatim). Scored
against the criteria pre-registered in
[`design_wall_ball_true_baseline.md`](design_wall_ball_true_baseline.md)
§8 *before* the run existed. Evidence: the run's `config.json`,
`ladder_certification.json`, `eval_info.csv`, `stage_summary.txt`,
`best_model_meta.json`, `best_model_long_horizon_eval.json` +
episodes CSV, and a local checksum-verified replay of the banked
champion for the one criterion the standard artifacts do not record.

## TL;DR

- **The era opens on the first attempt.** Best 3-eval window **2.856**
  (primary bar 2.0); the uncapped 50-episode audit of the banked
  champion reads **3.02 mean completed returns** (median 3, p90 5,
  max 9), with **100%** of episodes completing ≥1 return and 98%
  completing ≥2. Every number beats the scripted oracle reference band
  (1.98 mean / 67% ≥2) — relative to its band, this is the strongest
  wall-ball policy the project has produced (1.53× band, vs 1.31×/1.39×
  for the goal-era seeds).
- **The deep receive is real.** Measured first contact of every audit
  episode: **mean x = −6.49 ± 0.25** (bar ≤ −6.0; 98% of episodes
  deeper; shallowest −5.96) — 2.8 m deeper than the goal era's −3.7.
  Zero opening volleys, zero pre-bounce hits in the audit: all 163
  legal hits are genuine post-bounce receives.
- **The stretch (a ≥3.0 window) was not met, and the evidence says the
  container, not the policy, is why.** The matched eval stream's
  episode cap (750 steps ≈ 4.8 exchanges at this era's 156-step
  cadence) clipped 10–20% of episodes at exactly 5 returns on every
  strong eval; the uncapped audit crosses 3.0, and 14% of audit
  episodes ran past 750 steps (longest 1,487 — a 9-return rally, the
  era record). Raising `episode_len` is the next pre-registered
  change, between runs.
- **Every safety/selection mechanism worked.** Startup certification
  passed live at the 0.50 floor (stamped in provenance; oracle 73% ≥2
  — third independent confirmation of the band). The champion was
  banked at 2.1M and confirmed (2.95 / 2.92); the run early-stopped at
  3.6M of 6M on patience 60 with the final policy off-peak (1.98) —
  exactly the case best-model banking exists for.
- Reliability is the open question n=1 cannot answer: the goal era ran
  ~2 of 3. **Next step: replicate with seed 1, same starter TOML.**

## 1. Scorecard against the pre-registered criteria

| # | Criterion (design doc §8) | Bar | Measured | Verdict |
|---|---|---|---|---|
| 1a | Best 3-eval window `bounce_count_ep_mean` | ≥ 2.0 | **2.856** (ending 1.65M) | **pass** |
| 1b | Uncapped audit mean of banked best | ≥ 2.0 | **3.02** | **pass** |
| 1c | Audit episodes completing ≥1 return | ≥ 80% | **100%** | **pass** |
| 2 | Audit mean first-contact x | ≤ −6.0 | **−6.49** (98% deeper) | **pass** |
| 3 | Stretch: any ≥3.0 window | ≥ 3.0 | 2.856 | not met (cap-compressed; §4) |
| 4 | Reliability expectation | — | n=1 | replication pending |

## 2. The run

Provenance is clean end to end: the packaged starter TOML ran
verbatim (6M budget, n_envs 8, eval 60 episodes / 25k, patience 60),
the recipe's env kwargs match the frozen era task exactly, and
startup certification passed on the live geometry (30 episodes/cell,
seeds 30000+): oracle 1.93 mean / 73% ≥2, crude 1.03 / 3.3%, parked
zero-contact, monotone rewards, landing 30/30 with offset
+3.43 ± 0.15 m — consistent with both the calibration (67%) and
held-out (61%) certifications. `feasibility_ge2_floor: 0.5` is
stamped in the report provenance as designed.

Trajectory (60-episode evals):

| phase | steps | what happened |
|---|---|---|
| cold start | 0–50k | zero → first contacts (untouched serves, standard) |
| climb to one return | 50k–150k | bounce 0.08 → 1.00, receive depth ≈ −6.2 |
| one-and-done shoulder | 150k–275k | pinned ~1.0, ≥2 rate 0→13% |
| liftoff | 275k–625k | ≥2 rate 13% → 72%; **crosses the oracle band (1.98) at 625k** — 10% of budget |
| climb | 625k–1.65M | best window 2.856; single evals to 2.95; ≥2 rate up to 95% |
| plateau | 1.65M–3.6M | band 2.1–2.95, dips recover within one eval, no new window record |
| stop | 3.6M | patience 60 fires; champion long banked |

The champion was banked at **2.1M** (selection 2.95 mean / 13.3% ≥5;
confirmation 2.92 / 6.7%) and the final policy at stop read 1.98 with
`ent_coef` at 0.00107 — the same late-run near-determinism drift the
corpus has seen before, without the seed-1 collapse. The run ended
12h 43m after launch at 78 FPS. No degenerate-guard events, no
stalls, zero nonfinite terminations across 144 evals.

## 3. The audit (uncapped, 50 episodes, seeds 10000–10049)

- **Completed returns: mean 3.02 ± 1.38**, min 1, p10 2, median 3,
  p90 5, max 9. Survival: ≥1 **100%**, ≥2 **98%**, ≥3 54%, ≥4 28%,
  ≥5 12%, ≥6 4%, ≥9 2%.
- **Episode length: mean 537, max 1,487** — 14% of episodes exceed
  the 750-step training cap, direct proof the cap binds (§4).
- **Style: 100% post-bounce play.** 0 opening volleys, 0 pre-bounce
  hits; 151 completed returns off 163 legal hits (93% conversion).
- **Terminations: 64% out-of-bounds, 36% double bounce**, zero
  stalls/timeouts — the probe-predicted error profile (long balls
  from hot deep receives dominate).
- **First-contact depth** (not in the standard artifacts): measured
  by local deterministic replay of the banked champion on the same
  seed block, both artifacts checksum-verified against
  `best_model_meta.json` (`e92ab95d…`, `2ed0c170…`): mean **−6.49**,
  std 0.25, p10 −6.82, median −6.51, p90 −6.15, shallowest −5.96;
  98% of episodes ≤ −6.0. The replay's return mean (3.32) sits
  slightly above the audit harness's 3.02 — a seeding-path
  difference between harnesses; both clear every bar, and the audit
  number is the one scored.

Goal-era comparison, kept honest (different task, so band-relative):
seed-0 goal audit 3.54 on a ~2.7 band (1.31×); seed-2 3.76 (1.39×);
**this run 3.02 on a 1.98 band (1.53×)** — while receiving 2.8 m
deeper off a 57% faster serve.

## 4. The cap is now the measured ceiling

Three independent observations converge:

1. From ~625k on, `bounce_count_ep_max` pinned at **exactly 5** in
   every one of ~120 evals — 750 steps ÷ 156-step cadence = 4.8
   exchanges; a fifth exchange finishes the clock.
2. Timeout terminations on strong evals ran 10–20% and tracked the
   ≥5-rate almost one-for-one: the best episodes end by clock, not by
   error.
3. Uncapped, the same policy plays 6-, 7-, and 9-return rallies, and
   14% of audit episodes outrun the cap.

The goal era's identical 750-step cap held ~6 exchanges at its faster
cadence, so this era's stretch bar (3.0, kept for campaign
continuity) is materially harder *relative to the container* than it
was there. The recommended change — pre-registered separately as
0.26.0, applied between runs so the seed-1 replication stays
comparable to this run — is `episode_len` 750 → ~1,100, restoring
~6-exchange headroom at cadence 156. The stretch verdict for this
era should be read against the uncapped audit until that lands.

## 5. What this run settles

1. **The probe-first task freeze worked.** Every load-bearing number
   the design doc measured in scripted probes reproduced in the live
   run: serve landing offset (+3.43 vs probe +3.37), oracle band
   (1.93–1.98 across four independent measurements), termination
   taxonomy (long-ball dominant), exchange cadence (~156), and the
   forced deep receive (−6.5 first contact).
2. **Direct training transfers to the harder era unchanged.** The
   goal-rally recipe structure — no curriculum, single informational
   gate, patience 60 — opened a task 3.5 m deeper with a 57% hotter
   serve on the first attempt, in 2.1M steps to champion.
3. **The 0.25.0 machinery additions all fired correctly in
   production**: the certification floor knob (stamped, passed), the
   extended in-play bound (zero spurious deep OOBs; untouched-serve
   pass-balls terminate as designed), and the frozen default mapping
   (nothing about earlier eras changed).

## 6. Next steps

1. **Replicate: seed 1, same starter TOML, no other change.** The
   only open question is reliability (~2/3 in the goal era). A third
   seed if the first two disagree.
2. **Pre-register 0.26.0: `episode_len` 750 → ~1,100** (with the
   matched change to `video_length`/audit expectations), between
   runs. Cheap, evidence-backed, and it un-compresses the metric the
   campaign scores.
3. Longer-horizon levers stay parked: paddle pitch/yaw only if the
   long-ball error rate resists further training; network size only
   on underfitting evidence (none yet — see the run's clean climb).

## Seed ledger

No new burns. The audit and the first-contact replay reused the
standing long-horizon audit block 10000–10049; startup certification
used its reserved 30000+ block. Blocks 3100–3199 and 4100–4199 remain
clean.
