# Pre-registration: the n-point era's registered run

Status: **frozen 2026-08-16, before any run step; the run has not
launched.** The ADOPT branch of
[`design_paddle_tennis_reach_shaping.md`](design_paddle_tennis_reach_shaping.md)
§4a commits two follow-ups: the recipe adoption (shipped with this
document) and this pre-registration. Everything numeric below is
frozen from the LR1 band before the run; changes after launch void
the run as campaign evidence (the standing doctrine).

## 1. What this run is

The era's registered result: the adopted task (continuous n-point
play, contact escrow 0.25, reach escrow 0.25, hardened guards — all
recipe defaults as of the adoption commit) at triple the pilot
budget, warm-started from the same lineage as both pilots, with the
long-reserved held-out gate finally consumed. LR1 proved the
mechanism (k=2 above zero at five checkpoints, every engagement bar
passing); the registered run asks whether the k≥2 channel *grows*
with budget now that it exists, and certifies the result held-out.

## 2. Run shape (frozen)

- Recipe `PaddleTennis` at the adoption commit; run config frozen
  here verbatim (the Drive-side convention all three recent runs
  used — copy to the configs folder as
  `paddle_tennis_registered_npoint.toml`; the run's provenance
  records its sha256 as usual):

  ```toml
  # Registered n-point era run — frozen shape
  # (docs/paddle_tennis_registered_run_prereg_20260816.md §2; the
  # recipe carries the adopted task. Launch pairing, code-side:
  # seed 1, total_timesteps 3_000_000, warm start from
  # training_runs/PaddleTennis/sac/20260809_211147.)

  [train]
  n_envs = 4
  eval_freq = 25_000
  checkpoint_freq = 100_000

  [train.model_kwargs]
  learning_starts = 25_000
  ```
- **Seed 1** (the registered-run convention; the pilots hold seed 0).
- **3,000,000 steps** — ≈19 h at the measured 44 FPS, inside a
  single session; chosen over 6M deliberately (every productive run
  in this campaign delivered its verdict-relevant signal by ~2M, and
  the extend-once lean below covers the long tail).
- Warm start from `20260809_211147` best (policy + critics +
  `obs_rms` + temperature, the loader's contract — identical source
  artifacts, by sha256, to both pilots), passed code-side via
  `build_train_config(..., warm_start=...)`.
- `record_video` on (recipe default cadence): this is the run whose
  replays should finally contain rallies.

## 3. Criteria (frozen; each metric at its own best checkpoint from
the automated diagnosis rows / eval CSV)

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| **RK1** (headline) | k=2 exchange survival, either parity | ≤ 1% at every checkpoint (no advance over LR1's band) | (1%, 5%) | ≥ 5% at some checkpoint |
| RK2 (record, non-binding) | k=3 survival | — | — | > 0% at any checkpoint is recorded as the era's first triple exchange; not a verdict input |
| RE1 | touched-after-bounce | ≤ 41% everywhere (the LR1 peak) | (41%, 50%) | ≥ 50% at some checkpoint |
| RE3 | k=1 receiving survival | < 68% at every checkpoint after 500k | one checkpoint ≥ 80% | ≥ 80% at two consecutive checkpoints (LR1: 68/83/68) |
| RS2 | serving-side k=1 survival | 0% at every checkpoint | (0%, 25%) | ≥ 25% at some checkpoint (LR1: 12% at 1M, the channel's first life) |
| M | behavioral mechanism | `legal_hit_count_a_ep_mean` = 0 at ≥ half the evals | — | > 0 at ≥ half the evals |

**R2 (inter-point recovery ≤ 6.0 m) is a named watch metric, not a
verdict input**: it is the window the reward does not pay, its
failure is expected, and its trajectory routes the *next* design
(the post-swing-hold escrow) rather than this verdict.

## 4. Held-out gate (consumes reserved block 4100–4199)

After the run, the selected best checkpoint is re-evaluated once on
seeds **4100–4199** (100 episodes — the block's single sanctioned
opening, reserved since the volley era). Floors, frozen now:

- `crossings_ep_mean` ≥ **85%** of its value over the selection
  window (the prior certifications' convention);
- **k=2 > 0%** reproduced in the held-out batch (the era's headline
  must survive fresh seeds);
- zero unsafe terminations.

## 5. Decision rule (three branches)

- **Registered result stands**: RK1 PASS, M intact, ≥ 2 of
  RE1/RE3/RS2 PASS, and the held-out gate passes → the era's record
  is booked; the next design phase is the opponent side (the
  original sketch's P3, adversarial scoring), with the
  post-swing-hold escrow as the parallel reward-side candidate,
  prioritized by R2's watch trajectory.
- **Extend once** (non-forcing): RK1 in the middle with diagnosis
  rows still improving at 3M → a single 3M extension of the
  identical configuration; its verdict re-applies this rule without
  this branch. The "once" is binding.
- **Stop/amend**: RK1 FAIL or M broken → no budget extension; the
  next probed change is the post-swing-hold escrow (pay position at
  the opponent's strike — the same event-escrow family, its own
  RS-style battery), targeting the measured unpaid window before any
  further scale.

## 6. Launch procedure (notebook deltas)

`REPO_REF` = the adoption commit on main; `CONFIG_FILE` = the
committed TOML's path (in-tree copy or its Drive mirror — the sha
must match); `ENV = "PaddleTennis"`, `SEED = 1`,
`TOTAL_TIMESTEPS = 3_000_000`, `N_ENVS = 4`; the §5 config-build
cell adds the same `warm_start` override both pilots used. Validate
the run's `config.json` against §2 before letting it proceed past
the first checkpoint.

## 7. Seed ledger

No new blocks. Diagnosis stays on calibration 5200+. **4100–4199
remains sealed until §4 opens it** (that opening consumes it).
All prior burns stand — 4300–4399 (NP3 certification, consumed
2026-08-10) and the review-workpaper blocks 5500–6199 included.
