# Design: baseline recalibration, auto-resolved run configs, tennis-court replay

Status: implemented in 0.13.0 (design approved 2026-07-18). Three changes motivated by runs
`20260717_165358` and `20260718_023737` (two independent samples of the
same config agreeing on a ~3.2–3.3 eval-bounce ceiling, with the
long-horizon tail — not the mean — showing the real remaining headroom).

## A. WallBallBaseline recalibration (recommended settings)

### What changes

The recommended settings become the **recipe's** calibrated values, and
the packaged starter TOML follows automatically (the
`TestStarterConfigs` drift test forces starter == recipe, so "update the
TOML" and "update the recipe" are deliberately the same act):

| Setting | Today | Proposed | Why |
|---|---|---|---|
| `env_kwargs.weak_return_penalty` | `None` (terminal `floor_before_wall`) | `0.1` (fined retry) | 12–16% of best-model episodes still die as weak returns; the retry converts them into practice reps (v0.11.0 mechanism, calibrated in the bootstrap work) |
| `eval_env_overrides.weak_return_penalty` | — | `None` (re-assert strict) | Evaluation keeps the exact task every prior run was measured on; `bounce_count_ep_mean` stays comparable with 165358/023737 |
| `extra_cfg.model_kwargs` | `{}` (SB3 defaults) | `{"gamma": 0.995}` | Credit horizon ~200 steps (> one full exchange at ~130 steps); targets the now-dominant failure (54% double bounce = failing to commit to the *next* exchange). Auto-entropy is untouched — the 20260717 A/B poison was fixed `ent_coef`, not `gamma` |
| `extra_cfg.best_metric_keys` | `(bounce_count_ep_mean, ge_2_rate)` | `(bounce_count_ep_mean, ge_5_rate)` | `ge_2` saturated at ~100% and is a dead tiebreaker; `ge_5` is exactly the tail the 023737 run improved (8%→22% long-horizon) while eval's 750-step cap compresses the mean |

Not changed: eval episode length (750) and the headline metric — full
cross-run comparability is retained; the long-horizon audit remains the
sensitive instrument for tail effects.

### Alternative considered

A divergent "experiment" TOML not matching the recipe. Rejected: it
creates a second source of truth and defeats the drift test; per-run
divergence belongs in the user's *copied* TOML in Drive, and once
evidence promotes settings to "recommended", the recipe is their home.

### Compatibility

Training-side reward semantics change (weak returns fined, γ 0.995), so
0.12 WallBallBaseline *learning curves* are not directly comparable;
eval metrics remain comparable (strict eval unchanged). Version bumps
to **0.13.0** with README migration paragraph.

## B. Notebook auto-resolves the environment's run config

### What changes

New helper in `notebook_utils` (it owns Drive-path conventions;
`run_config.copy_starter_config` stays the primitive):

```python
def resolve_run_config_file(
    env_name: str,
    *,
    use_drive: bool = False,
    drive_subdir: str = "courtside-dynamics",
    local_root: str = "./configs",
) -> Path | None:
    """Return the experiment TOML for ``env_name``, creating it from the
    packaged starter on first use.

    Root: ``<MyDrive>/Finding Theta/<drive_subdir>/configs/`` when Drive
    is available (same root convention as ``resolve_run_dir``), else
    ``local_root``. If ``<root>/<name_prefix>.toml`` exists it is
    returned as-is (the user's edits win); otherwise the packaged
    starter is copied there (idempotent via ``copy_starter_config``).
    Prints the resolved path and sha256 so the choice is always visible.
    Returns None, printing why, for recipes without a starter.
    """
```

Notebook rewiring (`sb3_training.ipynb`):

- Section 2 variable becomes a **mode switch** instead of a raw path:
  ```python
  # "auto": use <configs root>/<recipe>.toml, created from the packaged
  #         starter on first use -- edit it in the Drive UI between runs.
  # None:   no config file (recipe defaults only).
  # "/path/to/file.toml": explicit file, used as-is.
  CONFIG_FILE = "auto"
  ```
- New small cell in section 3 (after the Drive mount, which the
  resolution needs):
  ```python
  if CONFIG_FILE == "auto":
      CONFIG_FILE = resolve_run_config_file(ENV, use_drive=USE_DRIVE)
  ```
- Section 2b's discovery cell keeps the catalog print; its commented
  manual `copy_starter_config` snippet is dropped (the helper subsumes
  it).
- Build cell unchanged (`config_file=CONFIG_FILE`; `None` still means
  "no file").

### Spec decision revisited (documented in the spec as v1.1)

v1 said "the config path is always explicit (never auto-discovered)".
That guarded against *invisible* configuration. The helper keeps the
resolution visible and opt-in: the notebook cell shows the call, the
helper prints path + sha256 + created/reused, the build cell prints the
same, and `config.json` / `run_config.toml` record it. `CONFIG_FILE =
None` remains one edit away. `build_train_config` itself still never
discovers anything.

### Failure modes considered

- Drive not mounted with `CONFIG_FILE="auto"` and `USE_DRIVE=True` →
  helper falls back to `./configs/` (ephemeral) and says so loudly.
- Recipe renamed/no starter → helper returns None with a printed
  explanation rather than raising (a missing starter should not block a
  default-config run).
- A stale Drive copy from an older package (e.g. missing new keys) is
  *by design* left untouched — the loader's validation plus the printed
  sha keep it diagnosable; `copy_starter_config(..., overwrite=True)`
  remains the manual reset.

## C. Tennis-court mode for WallBall replay video

### What changes

New WallBall env kwarg `court_style: str = "diagnostic"`:

- `"diagnostic"` — today's 0.11.1 markings (wall base, baseline line,
  metre ticks, lane strip/lines, home, fence, serve lines). Default;
  training-time periodic videos keep it.
- `"tennis"` — the ground reads as a real tennis half-court; diagnostic
  overlays hidden for clean presentation footage.
- `"none"` — bare checker floor (both marker sets hidden).

All marks remain **MuJoCo sites** (never collide → provably zero
physics/observation/reward impact, same argument and test pattern as
0.11.1). The renderer draws them with correct 3D perspective in every
camera, so replay, training videos, and live viewers all agree.

### Court geometry (to size)

The wall face plane (x = 3.9) plays the role of the net; the court
unrolls backward into the paddle's side. Real ITF dimensions fit the
existing 16 m × 12 m floor with no scaling:

| Feature | Real dimension | World placement |
|---|---|---|
| Net line (wall base) | — | x = 3.9 |
| Baseline | 11.885 m from net | x = −7.985 (floor edge is −8.0) |
| Service line | 6.40 m from net | x = −2.50 — lands *inside* the paddle lane (−3.2, −1.6), a happy accident worth keeping |
| Singles sidelines | ±4.115 m | y = ±4.115 |
| Doubles sidelines | ±5.485 m | y = ±5.485 (floor edge ±6) |
| Center service line | y = 0 | from x = 3.9 to x = −2.50 |
| Center mark | 0.10 m stub at baseline | x = −7.985, y = 0 |
| Line width | 0.05 m (0.10 m baseline) | site box half-widths 0.025 / 0.05 |

Plus an opaque hard-court blue surface quad spanning net→baseline ×
doubles width at z = 0.002 (lines staggered above at z = 0.006–0.012,
reusing the 0.11.1 anti-z-fighting scheme), and a muted green apron
quad covering the rest of the floor on the paddle side. Sites are
authored statically in the XML (`court_tennis_*`); `court_style` only
toggles visibility (alpha), handled by the existing
`_refresh_court_markers` (renamed `_refresh_court_style`), so runtime
switching via `set_wrapper_attr` + reset works like every other
curriculum attribute.

### Replay wiring

- `record_best_model_video` already takes `env_fn`; the notebook replay
  cell gains one variable:
  ```python
  REPLAY_COURT_STYLE = "tennis"   # "diagnostic" | "tennis" | "none"
  ```
  and builds the replay env via
  `make_eval_env_fn(ENV, env_overrides={"court_style": REPLAY_COURT_STYLE})`
  for wall-ball recipes (same `issubclass(..., WallBallEnv)` guard the
  long-horizon cell uses). Metrics-producing paths (eval callbacks,
  long-horizon audit) keep the default — visuals only differ, but zero
  gratuitous divergence between measured and recorded envs.
- Being an env kwarg, `court_style` is also settable per-run via the
  TOML's `[env]`/`[eval_env]` tables for whole-run tennis footage.

## Testing plan

- A: recipe/starter drift auto-enforced by `TestStarterConfigs`; update
  recipe tests pinning WallBallBaseline values (lane pin test, oracle
  defaults) and add one asserting eval strictness (`eval_env_overrides`
  re-asserts `weak_return_penalty=None`); calibrate nothing new (the
  weak-return mechanics were calibrated in v0.11.0).
- B: helper unit tests with tmp roots (creates-on-first-use, reuses
  edited copy, explicit-path passthrough, no-Drive fallback, missing
  starter → None); notebook contract test updated for the mode switch
  and resolution cell.
- C: marker tests extended — per-style visibility matrix, tennis line
  positions pinned to the table above, runtime style switch after
  reset, and the existing sites-only/no-geom physics invariance test
  covering the new sites.
- Full suite + ruff + mypy + adversarial review before commit, as
  usual.

## Versioning & rollout

One release, **0.13.0**: A is the behavior change (recipe
recalibration), B and C are additive. README migration paragraph, spec
v1.1 note for the resolution decision, version-history comment.
Implementation order: A → C → B (B's notebook cell wants C's
`REPLAY_COURT_STYLE` landing in the same notebook edit pass).
