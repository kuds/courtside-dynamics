# Design: recalibration revert, escalating wall reward, run-directory layout

Status: proposed (awaiting review). Motivated by run `20260718_213222`,
which falsified the 0.13.0 recalibration bundle (best
`bounce_count_ep_mean` 1.33 vs 3.23 for both references at identical
budget/seed/eval; long-horizon completed returns 1.26 vs 2.98/3.42;
failure taxonomy inverted to 82% double bounce / 8% OOB — the
soft-return signature of the asymmetric weak-return retry).

## Phase 1 — Revert the recalibration (immediate)

Restore the empirically-proven configuration; keep only the harmless
selection improvement:

| Setting | 0.13.0 (falsified) | 0.13.1 (revert) |
|---|---|---|
| `env_kwargs.weak_return_penalty` | `0.1` retry | **removed** (terminal rule, as in the reference runs) |
| `eval_env_overrides.weak_return_penalty` | `None` | removed (moot without the training retry) |
| `extra_cfg.model_kwargs` | `{"gamma": 0.995}` | `{}` (SB3 default 0.99, as in the reference runs) |
| `extra_cfg.best_metric_keys` | `(mean, ge_5_rate)` | **kept** — selection-only, still threshold-backed, and the tail is where headroom lives |

Mechanism note recorded in the recipe comments: terminal weak returns
and terminal OOB were *symmetric* (−1 each), pushing the policy toward
committed swings; the 0.1 retry made undershooting 10× cheaper than
overshooting and kept the episode alive, so SAC rationally converged to
soft, unchainable returns (`recoverable_bounce_score` peaked at 0.44 vs
~0.99). Any future retry experiment must keep the fine near-symmetric
with the OOB penalty.

Deliverables: recipe + starter TOML (drift test keeps them in
lockstep), updated pin tests asserting the revert, design-doc outcome
addendum in `design_court_and_config_updates.md`, README paragraph,
version 0.13.1. The best model on record remains run `20260718_023737`.

## Phase 2 — Escalating wall reward (the next training lever)

The double-bounce ceiling (~54% of failures in the best reference run)
means the policy is not paid enough for *chaining* returns: every
return is worth +1 whether or not it sets up the next one. Proposal:

- New env kwarg `wall_reward_increment: float = 0.0` — the n-th
  completed return in an episode banks `1 + (n − 1) × increment`
  (n=1 → 1.0, n=2 → 1.5, n=3 → 2.0 at increment 0.5). Refundable
  through the existing pending-advance machinery like the base wall
  reward; `0.0` is today's behavior bit-for-bit.
- Training-only: the recipe sets it in `env_kwargs`, and
  `eval_env_overrides` re-asserts `0.0` so `episode_reward` stays
  comparable (bounce metrics are counts and unaffected either way; the
  drift test's new eval-env coverage enforces this).
- Calibration before shipping (same bar as every reward change): the
  scripted ladder (parked / tracker / crude swing / oracle) must stay
  strictly monotone, and the oracle's per-episode return total must not
  create a farmable loop with recovery fragments (increment resets per
  episode; fragments already claw back pending advances).
- Ships dark (recipe increment TBD from the sweep, plausibly 0.5) —
  not bundled with Phase 1; it needs its own calibrated commit and its
  own training run, one variable at a time this time.

## Phase 3 — Run-directory layout + splitting the mega-plot

### Problem

A finished run currently drops ~19 files/dirs flat in the run root, and
`eval_info.png` is a single grid of ~170 metric panels (1.6–2.4 MB) —
too big to read, too big to casually download.

### Proposed layout

Written by a single shared registry (`artifacts.py` owns it) so writers,
`check_run_artifacts`, and the notebook stop hard-coding paths:

```
<run>/
  config.json              # identity & provenance (stays at root)
  stage_summary.txt        # human summary (stays at root)
  run_config.toml          # the TOML that ran (stays at root)
  model/
    best_model.zip          best_vec_normalize.pkl   best_model_meta.json
    final_model.zip         vec_normalize.pkl
  metrics/
    eval_info.csv           evaluations.npz
    progress.csv            monitor/                 tensorboard/
  reports/
    learning_curve.png      training_health.png
    eval_headline.png       eval_terminations.png
    eval_rewards.png        eval_diagnostics.png
    best_model_long_horizon_eval.json
    best_model_long_horizon_episodes.csv
  media/
    best_model.mp4          videos/
  checkpoints/
```

- `progress.csv` moves out of `tensorboard/` (it is pandas-readable
  metrics, not TB event data) into `metrics/`.
- Root keeps exactly the three files a human opens first; everything
  else is one folder deep by audience (model artifacts, machine
  metrics, human reports, media).

### The mega-plot split

`plot_eval_info` gains metric grouping and writes four themed,
bounded-size pages instead of one grid:

| Page | Contents |
|---|---|
| `eval_headline.png` | bounce mean/rates/percentiles, episode reward/length, paddle & return counts |
| `eval_terminations.png` | every `term_*` rate + phase fractions |
| `eval_rewards.png` | every `rew_*` component |
| `eval_diagnostics.png` | everything else (sensor stats, recovery scores, …) |

Each page capped at a fixed panel grid (spillover flows to a numbered
continuation page), DPI tuned so a page stays well under ~400 KB.

### Compatibility

- **Writers** switch to the registry in one release (0.14.0).
- **Readers** (`evaluate_best_wall_ball`, `record_best_model_video`,
  `plot_*`, `check_run_artifacts`, warm-start loading) resolve through
  a `locate_artifact(run_dir, name)` helper that checks the new path
  first and falls back to the legacy flat root, so every existing run
  in Drive keeps working unmodified.
- `check_run_artifacts` reports against the registry, so the audit cell
  keeps a single source of truth.
- No metric, format, or filename content changes — only placement and
  the plot pagination — so nothing about cross-run comparability moves.

## Sequencing

Phase 1 lands now (it unblocks the next training run). Phase 3 is
mechanical and can land in the same release train. Phase 2 waits for
its calibration sweep and ships separately — one variable at a time.
