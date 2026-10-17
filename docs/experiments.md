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
| `run_record.json` | By `train()` at start (`running`), rewritten on every terminal path | What ran and how it ended (`courtside-run-record/1`): status `completed`, `early_stop`, `interrupted` or `crashed` with a `stop_reason`; budget; algorithm; task fingerprints; comparability key; seeds; git; versions; warm start; selection, checked against `best_model_meta.json`'s sha256 bindings; wall clock |
| `sync_manifest.json` | Last, after the final record | Path, size and sha256 of every other file (`courtside-sync-manifest/1`) |

A directory is **complete** when its manifest verifies, the record it
lists is terminal, and every file the manifest does not list is a
post-training file (below). Complete means "exactly what `train()`
left", a crashed run included; it does not mean the run succeeded.
Without a manifest (still training, killed, a dead runtime, or older
than run records) a directory is incomplete by definition. A failure
while the run closes after it ended (an env or the TensorBoard flush
raising after a completed run) makes `train()` raise and the record
read `crashed`, its stop reason quoting the outcome the run had
reached.

On both selection paths the best-model files (`best_model.zip`,
`best_vec_normalize.pkl` and, under headline selection,
`best_model_meta.json`) are written under temporary names and renamed
into place, the meta last: an interrupt (a stopped Colab cell) inside a
write leaves the previous best whole. One that lands between two
renames leaves a meta whose sha256s no longer bind the files beside it.
The record then says so instead of copying the meta:
`selection.bindings` is `unbound` with an `unbound_reason`, its
timestep and values are `null` (never the previous best's), and the
triple is left out of `artifacts`; `stage_summary.txt` says the same
and `train()` prints a warning. Otherwise `bindings` is `verified`
(`not_checked` in rows the index backfills). A reward-selected run
writes no meta, so nothing binds its pair.

```bash
python -m courtside_dynamics.experiments.verify <run_dir> [<run_dir> ...]
python -m courtside_dynamics.experiments.verify --strict --json <run_dir>
```

Exit 0 only when every directory is complete. Every file the manifest
does not list is reported. Those at the post-training locations (the
notebooks' plots, audits and gate reports under `reports/`, and the
replay video `media/best_model.mp4`) fail only `--strict`; any other
fails the directory. A writer's temporary leftover (`.x.tmp-<pid>`,
`.x.sync-tmp`, `x.tmp.npz`, `x.tmp`) anywhere fails it too, whether
or not the manifest lists it: one listed was already there when the run
was sealed. The writers remove their temporaries when a write fails or
is interrupted. A directory whose record is terminal but which has no
manifest was never sealed: a mirror pass that did not finish (re-run
`sync_run_dir`) or a run that died while sealing itself.

With `TrainConfig.sync_dir` (the notebooks' `LOCAL_THEN_SYNC`), a run
trains on local disk and mirrors itself to Drive at every checkpoint and
at the end; the mirror gets its own manifest only after every copied
file re-hashes, so a partial mirror never verifies. `sync_dir` is
resolved once (`~` expanded) before any output, and that directory is
the one mirrored and recorded. The final pass must reproduce the local
manifest's files exactly, or it is redone without the change cache.

`experiments.sync.sync_run_dir(local_run_dir, mirror_dir)` re-runs a
mirror by hand (the notebooks' last step). It never deletes or
overwrites a file it cannot prove is this run's (one it copied, one the
mirror's manifest for this run id lists, or one byte-identical to the
local file): a destination holding anything else, or another run's
mirror, is refused before anything is written, so a slip in the path
(the Drive root) costs nothing.

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
are aborted, how many run records still say `running`, how many rows
are stubs, how many joined the campaign map, every skipped directory
with its reason, and a `WARNING` per row problem.

Every row has the run record's keys plus two:

- `index`: `path` (relative to the root), `source` (`run_record` or
  `legacy`), `version_inferred`, `inferred` (the fields derived rather
  than copied), `stub`, `verification` (the manifest verdict of a
  run-record directory), `files` (which model files exist, the
  checkpoint count and latest step, the periodic video count, and
  `best_model_video` for the replay, at the root of a pre-0.14 run),
  `run_config` (the TOML's basename and sha256), `summary` (the parsed
  head of `stage_summary.txt`) and `problems`.
- `campaign`: the curated map's entry for this run id, or `null`.

**What becomes a row.**

- A directory with `run_record.json`: the record as written, plus its
  manifest verdict. `--skip-verify` leaves the verdict `not_checked`
  (verifying re-reads every checkpoint, slow on a Drive mount). A
  record that still says `running` gets a problem either way: without
  a manifest the run is still training or died (a killed VM), with one
  it is a mid-run mirror snapshot. It is a stub when no final model
  exists, like a backfilled launch.
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
  Status `aborted`; a **stub** when no final model was saved either. A
  summary that exists but cannot be read leaves the status `null`,
  with a problem: the run may have finished.
- A run-named directory (`YYYYMMDD_HHMMSS`) whose `config.json` cannot
  be read or is not a courtside config (a zero-byte file from a partial
  sync): a row without the config's fields, the problem recorded.
  A UTF-8 byte-order mark and an integral float budget (`1e6`) are
  read as written.
- A directory named like a run with no `config.json`: an aborted stub
  when it is empty or holds a courtside launch file, `run_config.toml`
  (written a fraction of a second before `config.json`) or the 0.14
  layout's `metrics/`: a launch that died before writing `config.json`.
- A field of `config.json` or `best_model_meta.json` with an
  unexpected type (a string where an object belongs) is read as
  missing and named in the row's problems; it never fails the index.
  A run record whose selection is `unbound` gets a problem too.

Anything else is never parsed as a run: a directory holding files but
no run marker, a `config.json` without a courtside `train_config`
outside a run-named directory, or a run-named directory of another
layout (TennisWall's `run_config.json` runs) is listed as skipped. A
run (or skipped) directory is searched further only when a
subdirectory holds a run marker; then the nested runs get rows of
their own and the outer row or skip names them, so a stray
`config.json` in a recipe folder hides nothing.

A backfilled row holds no machine path. The warm start is named by its
source directory's basename, the run config by its basename, and every
problem names files relative to the run directory.

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
