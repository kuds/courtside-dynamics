# Run-configuration starter files

One TOML per registered recipe (`courtside_dynamics.recipes.RECIPES`),
shipped as package data so pip installs (Colab included) carry them.
Discover and use them via:

```python
from courtside_dynamics.run_config import (
    available_run_configs,   # {recipe name: packaged Path}
    copy_starter_config,     # copy one next to your runs for editing
)

config_file = copy_starter_config("WallBallBootstrap", DRIVE_CONFIG_DIR)
cfg = build_train_config("WallBallBootstrap", log_dir=LOG_DIR, seed=SEED,
                         config_file=config_file)
```

Full format, precedence, and merge semantics:
`docs/run_config_file_spec.md`.

Conventions for these starters:

- **Every value matches the recipe today.** Running with the file is
  equivalent to running without it; the file exists to give the run a
  diffable home and to document which knobs are worth turning (and in
  which direction — see the comments in each file). Edit a value to
  diverge; `config.json` and `log_dir/run_config.toml` record exactly
  what ran. `tests/test_run_config.py` keeps the starters loadable and
  drift-free against the recipes.
- **One file per experiment.** For a new experiment, copy the starter
  (`copy_starter_config` puts it e.g. in your Drive `configs/` for
  Colab — rerun-safe while unedited, refusing to clobber an edited
  copy), rename it, and edit the copy — don't accumulate variants in
  one file.
- `[train.model_kwargs]` deep-merges onto the recipe's bundle, so
  tweaking one key never discards the rest. `performance_gate`
  replaces wholesale. `[env]` reaches both the training and eval envs.
