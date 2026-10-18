# Run-index fixtures

`training_runs/` holds five run directories from the Drive training-run
archive, for `tests/test_run_index.py`:

| Run | What it exercises |
|---|---|
| `PaddleTennis/sac/20260810_211754` | n-point L2 pilot: an early stop with a stop reason; the latest checkpoint (1.1M) is below the steps trained (1.125M) |
| `PaddleTennis/sac/20260815_180815` | L2W: a warm start, and a stop reason on a full-budget run (the train-3 label) |
| `PaddleTennis/sac/20260828_114038` | an LT1 CPU false start: `config.json` and a first-evaluation best model, no summary, no final model (a stub) |
| `WallBallBaseline/sac/20260714_050506` | the first WallBall run: pre-0.14 flat layout, no git sha, no recipe name, an early stop whose summary names no reason |
| `WallBallGoalRally/sac/20260728_224706` | GoalRally's first launch, stopped at about 5 minutes: a final model saved, no summary (aborted, not a stub) |

Text files (`config.json`, `stage_summary.txt`, `best_model_meta.json`,
`run_config.toml`) are copied byte for byte except one substitution:
the Drive mount prefix of every absolute path is replaced by
`/scrubbed/courtside-dynamics`. Model and checkpoint files are empty
placeholders with the archived names, so presence checks work; the
checkpoint folders hold only the newest checkpoint's name.
