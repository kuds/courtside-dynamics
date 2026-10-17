# Run records, manifests and the run index

Living reference for `courtside_dynamics.experiments` (design L1 in
[`design_campaign_structure_20261016.md`](design_campaign_structure_20261016.md)).
Nothing in the package imports Stable-Baselines3 or torch, so all of it
runs without the `train` extra.

**The rule:** each run directory's own files are the source of truth.
The index is derived from them and regenerated whole, never appended to
(two Colab sessions co-appending one Drive file is how an index goes
wrong).

## What a run directory says about itself

| File | Written | Says |
|---|---|---|
| `run_record.json` | By `train()` at start (`running`), rewritten on every terminal path | What ran and how it ended (`courtside-run-record/1`): status `completed`, `early_stop`, `interrupted` or `crashed` with a `stop_reason`; budget; algorithm; task fingerprints; comparability key; seeds; git; versions; warm start; selection; wall clock |
| `sync_manifest.json` | Last, after the final record | Path, size and sha256 of every other file (`courtside-sync-manifest/1`) |

A directory is **complete** when its manifest verifies and the record it
lists is terminal. Complete means "exactly what `train()` left", a
crashed run included; it does not mean the run succeeded. Without a
manifest (still training, killed, a dead runtime, or older than run
records) a directory is incomplete by definition.

```bash
python -m courtside_dynamics.experiments.verify <run_dir> [<run_dir> ...]
python -m courtside_dynamics.experiments.verify --strict --json <run_dir>
```

Exit 0 only when every directory is complete. Files written after the
manifest (the notebooks' plots and replay video) are listed as unlisted
and fail only `--strict`.

With `TrainConfig.sync_dir` (the notebooks' `LOCAL_THEN_SYNC`), a run
trains on local disk and mirrors itself to Drive at every checkpoint and
at the end; the mirror gets its own manifest only after every copied
file re-hashes, so a partial mirror never verifies.

## The run index

```bash
python -m courtside_dynamics.experiments.index --root <training_runs>
python -m courtside_dynamics.experiments.index --root <training_runs> \
    --out runs.jsonl --legacy-levers my_map.toml --skip-verify
```

Writes `<root>/index/runs.jsonl` (or `--out`): one JSON object per run
directory, sorted by its path under the root, keys sorted. Running it
twice over unchanged directories gives the same bytes. It prints how
many rows came from run records and how many were backfilled, how many
are aborted and stubs, how many joined the campaign map, every skipped
directory with its reason, and a `WARNING` per row problem.

Every row has the run record's keys plus two:

- `index`: `path` (relative to the root), `source` (`run_record` or
  `legacy`), `version_inferred`, `inferred` (the fields derived rather
  than copied), `stub`, `verification` (the manifest verdict of a
  run-record directory), `files` (which model files exist, the
  checkpoint count and latest step, the video count), `run_config` (the
  TOML's basename and sha256), `summary` (the parsed head of
  `stage_summary.txt`) and `problems`.
- `campaign`: the curated map's entry for this run id, or `null`.

**What becomes a row.**

- A directory with `run_record.json`: the record as written, plus its
  manifest verdict. `--skip-verify` leaves the verdict `not_checked`
  (verifying re-reads every checkpoint, slow on a Drive mount).
- A directory with a courtside `config.json` and no record (every run
  before run records): backfilled into the record's shape. Algorithm,
  recipe, task fingerprints, seeds, git, versions, warm start and budget
  come from `config.json` through the same code `train()` uses; status,
  stop reason and steps trained from `stage_summary.txt`; the selection
  from `best_model_meta.json`. The summary says `completed` for an early
  stop too: its `X of Y budget (stopped early)` line makes the row
  `early_stop`. A stop reason on a run that trained its whole budget
  (the train-3 label) becomes `completed` with a `full_budget: ...`
  reason, as `train()` records it now.
- No `stage_summary.txt`: the run never reached a terminal path.
  Status `aborted`; a **stub** when no final model was saved either.
- An empty directory named like a run (`YYYYMMDD_HHMMSS`): an aborted
  stub, a launch that died before `config.json`.

Anything else is never parsed as a run: a directory holding files but
no run marker, a `config.json` without a courtside `train_config`, or a
run-named directory without `config.json` is listed as skipped.

A backfilled row holds no machine path. The warm start is named by its
source directory's basename, the run config by its basename.

**The campaign map.** `experiments/legacy_levers.toml` (package data)
maps each pre-spec run id to `campaign`, `lever`, `verdict`, `doc` and
an optional `note`. Every PaddleTennis run directory maps to one lever,
`baseline` or `infra`. WallBall runs map to their recipe era
(`era/<name>`). A run named by no doc has a `note` saying so. Runs
written by the spec system will carry their lever in the record's own
`experiment` block.

## On Colab

```python
from courtside_dynamics.notebook_utils import mount_drive
mount_drive()
RUNS = "/content/drive/MyDrive/<your folder>/courtside-dynamics/training_runs"
!python -m courtside_dynamics.experiments.verify "{RUNS}/PaddleTennis/sac/<run_id>"
!python -m courtside_dynamics.experiments.index --root "{RUNS}" --skip-verify

import json
import pandas as pd
with open(f"{RUNS}/index/runs.jsonl") as rows:
    runs = pd.json_normalize([json.loads(row) for row in rows])
runs[["run_id", "status", "campaign.lever", "budget.trained_timesteps"]]
```

Run the index from one session at a time: it rewrites the file
atomically, but two sessions writing it concurrently would race.
