# PaddleTennis local pilot — the task learns fast, and the first GPU run is pre-registered

> **SUPERSEDED (volley era), 2026-08-03**
> ([`paddle_tennis_ground_rules_20260803.md`](paddle_tennis_ground_rules_20260803.md)):
> both halves of this document describe the superseded volley-rules
> profile. The pilot record stands as that era's evidence
> (reproducible via `volley_rule="legal"`), but the §2
> pre-registration is **void and was never consumed** — do NOT run
> it, and do NOT touch reserved block 4100–4199 on its authority.
> The ground-era first run gets a fresh pre-registration once a
> ground-era pilot exists; 4100–4199 is held for that run's gate.

Status: review snapshot + pre-registration, 2026-08-02. Part 1
records the local SAC calibration pilot of the frozen `PaddleTennis`
recipe (the era's first learned run anywhere). Part 2 pre-registers
the first GPU run against the pilot's evidence, per the doctrine that
every campaign run commits its criteria before it starts.

## 1. The pilot (calibration, not a campaign run)

One local CPU run of the stock recipe: `build_train_config
("PaddleTennis", seed=0, total_timesteps=500_000, n_envs=4,
record_video=False)` — the only deviations from the recipe are
throughput knobs (workers, budget, no video: the runner has no GL
context), never task definition. 2h04m at 67 FPS. Provenance: the
run's launch snapshot (`config.json`) records the tree at the env
freeze push (`39b5ecd`); a tools/tests/docs-only commit landed
mid-run (zero `src/` changes between the two SHAs, so the trained
package is byte-identical at both — `stage_summary.txt` stamps the
later SHA because it snapshots HEAD at completion). Artifacts
(including per-eval CSVs and the selected checkpoint) live outside
the repo; the numbers below are the committed record.

### The rally tail vs the scripted reference

`crossings_ep_mean`, the complete series (30-episode evals, every
25k steps):

| steps | 25k | 50k | 75k | 100k | 125k | 150k | 175k | 200k | 225k | 250k |
|---|---|---|---|---|---|---|---|---|---|---|
| crossings | 0.63 | 0.47 | 2.43 | 2.13 | 2.80 | 3.07 | 3.27 | 3.77 | 3.50 | 4.03 |

| steps | 275k | 300k | 325k | 350k | 375k | 400k | 425k | 450k | 475k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|
| crossings | 3.63 | 3.90 | 4.40 | 4.63 | 5.17 | 5.17 | 6.13 | 5.63 | **6.40** | 6.13 |

- **The learned policy passes the scripted pair's certified band
  (3.22) at the 175k eval and nearly doubles it by 475k** (best
  6.40, task-metric selection; final 6.13). The curve is noisy eval
  to eval (dips at 50k, 100k, 225k, 275k, 450k) but trends up
  through the budget; the final quarter oscillates 5.17–6.40 (last
  four evals 6.13/5.63/6.40/6.13, mean ≈ 6.1). Whether that is a
  slow climb or a plateau near 6 is exactly what the GPU budget
  answers.
- Distribution at the best window: p50 6, p90 9–10, max 11.
- `success_rate` (serve returned) is ≥ 0.90 from 75k on, and 1.00 at
  every eval from 400k (with 0.97 dips at 325k and 375k).
- Failure taxonomy stays task-shaped: out_of_bounds ~0.70,
  ball_net ~0.07, **zero timeouts** (the 1500-step cap never binds;
  mean episode 311 steps at the best window, 327 at the final eval —
  the wall-ball 750-cap lesson held) and **zero nonfinite/unsafe
  terminations**.
- `serve_side_is_policy` = 0.50 at every single eval: alternation
  exact under vectorized training.
- Pipeline proof: `SelectiveVecNormalize` exclusions (tail 24–47
  raw), info-dict eval, task-metric best-model selection
  (475k beat the reward-selected window — selection provenance in
  `best_model_meta.json`), checkpointing, and `stage_summary.txt`
  all ran unmodified on the new recipe.

Reading: the cooperative rally task is *directly learnable* from a
cold start at the frozen definition — no shaping, no curriculum —
and the sparse +1-per-return signal is dense enough in practice
because the P3 serve band makes the first return nearly free. The
open P0–P2 question (fixed-pitch loft as the eventual ceiling) is
untouched: 6+ crossings happen well inside the paddle's fixed-pitch
envelope. Seeds: training seed 0 and its derived helper seeds only;
no ledgered evaluation block was touched.

## 2. Pre-registered first GPU run

Committed before the run starts; changes after start void the run as
campaign evidence.

- **Recipe**: `PaddleTennis`, SAC, stock starter TOML
  (2M steps, n_envs 8, patience 20, eval every 25k). No overrides
  beyond `seed`.
- **Seed**: 1 (the pilot documents seed 0; the GPU run doubles as
  the first cross-seed replication).
- **Primary criteria** (all required):
  1. best `crossings_ep_mean` window ≥ **6.0** (match the pilot on a
     fresh seed at GPU budget);
  2. `success_rate` ≥ 0.95 at the best window;
  3. zero `term_nonfinite` across all evals;
  4. `term_timeout_ep_mean` ≤ 0.05 at the best window (the cap must
     stay non-binding).
- **Stretch** (reported, not pass/fail): best window ≥ **10.0** —
  whether GPU budget finds the next regime the pilot's p90 (10)
  already brushes.
- **Held-out gate**: after the run selects its best model, evaluate
  that champion once on the reserved block **4100–4199**
  (100 episodes, deterministic policy, frozen normalizer): held-out
  mean crossings must be ≥ **85% of the selected best window** (the
  eval-selection-overfit guard). This is the block's single
  pre-registered use; it burns on that run.
- **Comparison era**: these are the first learned-run numbers of the
  PaddleTennis era; nothing before this doc is comparable except the
  scripted bands (P3 3.15–3.42; certification 3.22).
- **Failure rule**: if primary criterion 1 fails, the follow-up is a
  diagnosis snapshot before any recipe change (lesson 18: documented
  falsification over quiet iteration) — with special attention to
  whether the fixed-pitch loft ceiling (P0–P2's open question) is
  the binding constraint.

## 3. What waits on this run

- The phase-P2 opponent-pool decision (P5's champion rows on Colab +
  this run's own graduate).
- The paddle-pitch actuation question, if the stretch regime stalls
  in a loft-limited failure taxonomy.
