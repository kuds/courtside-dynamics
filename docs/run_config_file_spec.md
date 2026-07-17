# Spec: TOML run-configuration files

Status: proposed (not yet implemented)

## Motivation

Run hyperparameters currently live in three places with sharp edges:

1. **Recipe defaults** (`src/courtside_dynamics/recipes.py`) — versioned and
   calibrated, but editing them means editing package code.
2. **Notebook cell arguments** to `build_train_config(...)` — convenient, but
   explicit arguments *replace* recipe values wholesale. The
   `WallBallBootstrap` recipe made this a live footgun: a stale
   `model_kwargs=MODEL_KWARGS` cell silently replaces the recipe's entire
   exploration package (auto-entropy, `learning_starts`, buffer size) with
   old values, and nothing warns.
3. **`config.json`** — records what ran, but only after the fact.

A user-editable configuration file gives experiments a durable, diffable
home that is neither package code nor notebook state, with merge semantics
designed so "tweak one hyperparameter" cannot silently discard a calibrated
bundle.

## Format

TOML, read with the standard library's `tomllib` (Python ≥ 3.11, which the
package already requires — **no new dependency**). One file describes one
run configuration. Three optional top-level tables:

```toml
# wall_ball_bootstrap.toml — everything is optional; omit what the
# recipe already gets right.

[train]                      # TrainConfig fields
total_timesteps = 1_500_000
n_envs = 8
early_stop_patience = 20

[train.model_kwargs]         # DEEP-MERGED onto the recipe's model_kwargs
ent_coef = "auto_0.05"       # tweak one key; the recipe's learning_starts,
                             # buffer_size, gamma etc. survive untouched

[env]                        # merged into the recipe's env_kwargs
serve_vy_max = 1.1

[eval_env]                   # merged into recipe env_kwargs + the recipe's
                             # eval_env_overrides (this layer wins)
```

- `[train]` keys must be `TrainConfig` field names. Callable-valued and
  runtime-only fields are **rejected**: `env_fn`, `eval_env_fn`,
  `extra_callbacks`, `info_row_fn`, `warm_start` (v1; see Open questions).
- `[env]` / `[eval_env]` keys are environment constructor kwargs. TOML has
  no `None`: a kwarg whose meaning requires `None` (e.g.
  `early_touch_penalty = None` for the legacy terminal rule) uses the
  sentinel string `"none"`, which the loader converts. TOML arrays map to
  Python lists; env constructors already accept sequences where tuples are
  documented.
- Nested tables under `[train]` (e.g. `performance_gate`) follow the merge
  rules below.

## Precedence

From weakest to strongest, later layers override earlier ones:

```
recipe defaults  <  TOML file  <  quick_test presets  <  explicit
                                                         build_train_config
                                                         keyword arguments
```

Rationale: `quick_test=True` is an explicit in-code declaration of "smoke
test" and must shrink whatever the file asked for; explicit keyword
arguments remain the strongest layer because they are the most deliberate
(and today's behavior — no existing call site changes meaning). The
existing rule that an explicit `total_timesteps=` beats `quick_test` is
unchanged.

## Merge semantics

The layer-vs-layer footgun is *replacement*; the file layer fixes it:

| Value kind | TOML-over-recipe behavior |
|---|---|
| Scalars, strings, arrays | Replace |
| Mapping-valued `TrainConfig` fields (`model_kwargs`, `phase_labels`, `info_eval_survival_thresholds`) | **Deep-merge, one level**: file keys override recipe keys, unmentioned recipe keys survive |
| `performance_gate` | Replace **wholesale** — stage ladders are ordered lists whose element-wise merging would be ambiguous; a file that touches the gate must state the whole gate |
| `[env]` / `[eval_env]` tables | Key-wise merge into the recipe's kwargs (same as today's `env_overrides`) |

Explicit keyword arguments keep today's replace-wholesale semantics (no
behavior change for existing code); the file is the recommended layer for
partial tweaks precisely because it merges.

## Validation and failure behavior

This repo's run history is a catalog of silent no-ops (the `set_attr`
curriculum, shadow attributes, inert `clip_reward`). The config loader
therefore fails loudly on everything:

- Missing file → `FileNotFoundError` (never silently skipped).
- Malformed TOML → the `tomllib` error, wrapped with the file path.
- Unknown `[train]` key → `ValueError` naming the key and closest valid
  field names (`difflib.get_close_matches`).
- A rejected field (`env_fn`, ...) → `ValueError` explaining why.
- Unknown top-level table → `ValueError` (only `train`, `env`, `eval_env`).
- `[env]`/`[eval_env]` keys are validated by the environment constructor:
  when a config file is supplied, `build_train_config` eagerly constructs
  and closes one probe env (and one eval env) so a typo'd env kwarg fails
  at config-build time in seconds, not at `train()` after callbacks and
  loggers spin up.

## Provenance

`config.json` gains one block:

```json
"run_config_file": {
  "path": "/content/drive/.../wall_ball_bootstrap.toml",
  "sha256": "…",
  "content": { "train": { … }, "env": { … } }
}
```

(`null` when no file was used.) The resolved winning values continue to be
recorded where they always were (`train_config`, `env.constructor_kwargs`),
so an audit can answer both "what did the file say" and "what actually
won". `stage_summary.txt` adds a one-line `Run config: <basename>
(<sha256[:12]>)`.

## API

```python
cfg = build_train_config(
    "WallBallBootstrap",
    log_dir=LOG_DIR,
    seed=SEED,
    config_file=CONFIG_FILE,   # str | Path | None (default None)
)
```

New module `courtside_dynamics/run_config.py`:

```python
@dataclass(frozen=True)
class RunFileConfig:
    path: str
    sha256: str
    train: dict[str, Any]
    env: dict[str, Any]
    eval_env: dict[str, Any]
    raw: dict[str, Any]

def load_run_config(path: str | Path) -> RunFileConfig: ...
```

`build_train_config` applies it between the recipe layer and `quick_test`,
routes `[env]`/`[eval_env]` through the existing
`make_env_fn(env_overrides=...)` / `make_eval_env_fn(env_overrides=...)`
factories (never mutating the recipe registry), and stashes the
`RunFileConfig` on the returned `TrainConfig` for `artifacts.py` to record.

### Colab usage

```python
CONFIG_FILE = os.path.join(DRIVE_ROOT, "configs", "wall_ball_bootstrap.toml")
```

The file lives in Drive next to the runs: editable from the Drive UI
without touching a notebook cell, survives runtime restarts, and each run
records which file (and which content hash) produced it. The notebook's
`MODEL_KWARGS` cell is deleted for recipes with calibrated bundles; ad-hoc
sweeps edit the TOML.

## Testing plan

- Precedence: recipe < file < `quick_test` < explicit kwargs, one test per
  boundary, plus `total_timesteps`-beats-`quick_test` preserved.
- Deep-merge: file overriding one `model_kwargs` key preserves the
  recipe's remaining keys; `performance_gate` replaces wholesale.
- Loud failure: unknown train key (with suggestion text), rejected field,
  unknown table, missing file, malformed TOML, typo'd env kwarg caught by
  the eager probe.
- `"none"` sentinel round-trips to `None` for env kwargs.
- Provenance: `config.json` records path + sha256 + content; recipe
  registry is not mutated (build twice with different files → independent
  configs).

## Out of scope (v1)

- Multi-run sweep files, file inheritance/includes, a CLI entry point,
  YAML support, schema export. All are additive later.
- Changing the replace semantics of explicit keyword arguments.

## Open questions

1. Should `warm_start` be file-configurable (it is data — a run-dir path
   and index tuple)? Leaning yes in v2 once its dataclass gets a
   from-mapping constructor.
2. Should the notebook default to a conventional path
   (`<drive_root>/configs/<recipe>.toml`) when it exists, or must the file
   always be named explicitly? Leaning explicit-only: an implicitly
   discovered config that silently changes runs is the exact failure mode
   this repo keeps paying for.
3. Per-recipe sections in one shared file (`[recipes.WallBallBootstrap]`)
   vs one file per experiment. Leaning one file per experiment for v1 —
   simpler mental model, better diffs.
