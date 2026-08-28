# Pre-registration: LT1 — the temperature-skip warm start

Status: **frozen 2026-08-28, before any run step** (maintainer
decision, recorded via the review session; drafted 2026-08-23 as
step 3 of
[`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md)
§4, implementing the §4a shelf item of
[`design_paddle_tennis_postswing_hold.md`](design_paddle_tennis_postswing_hold.md)
and routing item 1 of
[`paddle_tennis_postswing_targets_20260822.md`](paddle_tennis_postswing_targets_20260822.md)
§6). Every §3 bar is binding as written, with its anchors shown.
The standing doctrine applies: changes after launch void the run as
campaign evidence.

## 1. What this run asks

PT1's F4 named the mechanism that starved every reward-side fix:
**dead exploration under a saturated action head.** The warm-start
loader transfers the source's collapsed entropy temperature
(LH1c inherited α = 1.589e-4; every campaign run ends collapsed), so
no warm-started pilot has ever trained with a live entropy bonus.
The stack now has the flag the record called for
(`WarmStartConfig.transfer_log_ent_coef=False`, implemented with the
2026-08-23 review): the target keeps `auto_0.02`'s fresh init
(log α = log 0.02 ≈ −3.91 — a ~125× stronger entropy bonus than the
transferred temperature), and SAC's entropy gradient then pushes
directly against tanh-rail saturation, which is exactly the
F2-measured pathology (88–91% of post-swing steps saturated; the
review's 2.5M row extends the band to 93%).

LT1 asks: **does a live temperature unsaturate the post-swing action
head — and if it does, does k=2 follow?**

**Declared risk, from the record:** α-collapse is chronic. From
scratch, `auto_0.02` annealed to 9.8e-4 inside 12k steps (npoint
appendix C.1), and appendix D.6's tanh-saturation mechanism (a
saturated squashed-Gaussian mean inflates latent −log π, annealing α
toward zero) applies with full force to a warm start that *begins*
~88% saturated. Restoring α buys a window, not a regime. The §3
mechanism observables are chosen so that even a fast re-collapse is
an informative outcome, not a wasted run.

## 2. Shape (frozen)

- **Source: the registered run's protected best** —
  `training_runs/PaddleTennis/sac/20260816_235141`, best checkpoint
  step 2,400,000 (`best_model.zip` sha `838997fb…`,
  `best_vec_normalize.pkl` sha `d0502c14…`; full digests in the
  run's `best_model_meta.json`). This is the §4a shelf item's own
  pairing and keeps the flag a **one-lever change**: LH1 warm-started
  the identical artifacts with the temperature transferred (and, per
  §4b, an effectively zero hold dose), so LH1's 1M trajectory is the
  standing same-source control. The LH1c-lineage candidates (crowned
  2.325M; the 2.5M checkpoint) were considered by the review and
  set aside — each is a second lever with no
  temperature-transferred control, and the measurements give 2.5M no
  established edge (fresh-seed k=2 1.6% [0.3, 4.5]; the crowned best
  reads 1.1% on the full calibration block). A lineage change, if
  ever wanted, is its own pre-registered decision.
- **Artifact pinning enforced in code**: the launch passes
  `expected_artifact_sha256` with the two digests above, so a moved
  artifact aborts the launch instead of voiding the pairing
  silently.
- **Seed 0** (pilot convention), **1,000,000 steps**, n_envs 4,
  eval 25k, checkpoint/diagnosis 100k — the LH1 cadence, for
  row-for-row comparability with the control.
- **Task: recipe defaults at the launch commit** (the adopted
  n-point era; contact 0.25, reach 0.25, hold **off**). No `[env]`
  table.
- Run config, frozen verbatim (Drive-side copy as
  `paddle_tennis_lt1_tempskip.toml`, sha recorded by provenance):

  ```toml
  # LT1 — temperature-skip warm start
  # (docs/paddle_tennis_lt1_prereg_20260823.md §2). Code-side
  # pairing: seed 0, total_timesteps 1_000_000, warm start from
  # training_runs/PaddleTennis/sac/20260816_235141 with
  # transfer_log_ent_coef = false and both artifact shas pinned.

  [train]
  n_envs = 4
  eval_freq = 25_000
  checkpoint_freq = 100_000

  [train.model_kwargs]
  learning_starts = 25_000
  ```

  (Warm start is code-side by design — `run_config.py` rejects it
  from TOML — via `build_train_config(..., warm_start=
  WarmStartConfig(source_run_dir=..., transfer_log_ent_coef=False,
  expected_artifact_sha256={...}))`, the same notebook cell every
  prior pilot used plus the two new fields.)

## 3. Criteria (frozen 2026-08-28; anchors shown)

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| KT1 (headline) | k=2 exchange survival, either parity, 100k-cadence diagnosis | ≤ 1% at every checkpoint | one checkpoint at 2% | ≥ 3% at some checkpoint |
| **T1 (mechanism)** | post-swing action saturation, PT1 instrument, strike-ended windows, on the 100k checkpoints | never < 85% | [70%, 85%) everywhere | ≤ 70% at some checkpoint |
| R1 (retention) | k=1 receiving survival | < 60% at every checkpoint | [60%, 80%) everywhere | ≥ 80% at some checkpoint |
| guards | recipe defaults | an abort books KT1 FAIL directly | — | — |

Anchors: KT1 is the KH1 bar unchanged (the source's own band on
these seeds is 0–1%). T1's anchor is the source checkpoint's
measured PT1 row — 87.6% saturation, 0.381 m/step saccade (the
registered 2.4M subject in `paddle_tennis_postswing_targets_20260822.md`
§4); ≤ 70% demands a real, not cosmetic, unsaturation. R1 is LH1's
retention bar verbatim.

**T2 — mechanism observables, recorded but non-verdict:**

- the `train/ent_coef` trajectory from progress.csv, specifically
  **steps until α < 1e-3** (precedent predicts ~10–20k; a
  materially longer live window is itself a finding);
- PT1 command saccade (m/step) alongside T1's saturation at each
  scored checkpoint (source band 0.38);
- the standing R2/hold-travel watch from the automated diagnosis.

**Instrument note (honest about the plumbing):** the in-run
automated diagnosis does *not* compute PT1 metrics. T1/T2's
saturation and saccade are scored post-run by replaying the saved
100k checkpoints through
`tools/paddle_tennis_postswing_target_probe.py` (deterministic,
calibration seeds 5200+, the diagnosis convention) — the
checkpoint cadence retains everything needed. Scoring this way was
validated by the review's digit-exact replications.

## 4. Decision rule (frozen)

- **KT1 PASS ∧ T1 PASS** → the exploration mechanism is confirmed
  and paying: pre-register the era's RK1 retry at registered scale
  (3M, seed 1) with the flag as the warm-start convention.
- **T1 PASS, KT1 FAIL** — the head unsaturated and k=2 still did
  not move → the blocker is past the action head; re-diagnose (PT1 +
  the shot ledger) before any further change. The command-rate
  design does **not** automatically fire on this branch.
- **T1 FAIL** — α re-collapsed with no saturation movement (the
  declared risk) → the optimizer-side lever is spent: route to the
  interface-side treatment
  ([`design_paddle_tennis_command_rate.md`](design_paddle_tennis_command_rate.md))
  with no further temperature retry.
- **Middles** → maintainer's call, documented with the post-hoc
  label.

No re-pair branch: the flag is boolean. A different entropy target
or a fixed α would be a new design with its own document, not a
re-pair.

## 5. Seed ledger

No new blocks. Training seed 0 per the pilot convention; diagnosis
and PT1 scoring stay on calibration 5200+ per convention.
**4100–4199 remains sealed** (only a registered-result branch opens
it; a pilot never does).

## 6. Launch checklist (notebook deltas)

`REPO_REF` = the flag's implementation commit on main;
`ENV = "PaddleTennis"`, `SEED = 0`, `TOTAL_TIMESTEPS = 1_000_000`,
`CONFIG_FILE` = the §2 TOML; the config-build cell adds the §2
`WarmStartConfig` with both sha pins. Before the first checkpoint,
validate the run's `config.json` against §2 — it is written at run
start, and `notebook_utils.validate_run_config_against_plan` performs
the full check (the campaign notebook automates it post-run as the
enforced backstop; the before-first-checkpoint reading is a manual
step at launch) — including the new
provenance: `initialization.transfer_log_ent_coef == false`,
`"log_ent_coef"` present in `initialization.reset` and absent from
`initialization.transferred`, and `initialization.source_artifacts`
digests equal to the §2 pins.
