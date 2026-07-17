# Run-configuration starter files

One TOML per registered recipe (`courtside_dynamics.recipes.RECIPES`),
consumed via:

```python
cfg = build_train_config("WallBallBootstrap", log_dir=LOG_DIR, seed=SEED,
                         config_file="configs/wall_ball_bootstrap.toml")
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
  (e.g. into your Drive `configs/` for Colab), rename it, and edit the
  copy — don't accumulate variants in one file.
- `[train.model_kwargs]` deep-merges onto the recipe's bundle, so
  tweaking one key never discards the rest. `performance_gate`
  replaces wholesale. `[env]` reaches both the training and eval envs.
