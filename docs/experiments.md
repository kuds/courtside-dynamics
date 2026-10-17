# Run records, manifests, the run index and offline re-scoring

Living reference for `courtside_dynamics.experiments` (design L1 to L3
in [`design_campaign_structure_20261016.md`](design_campaign_structure_20261016.md),
M3 in [`design_env_ladder_and_structure_20261016.md`](design_env_ladder_and_structure_20261016.md)).
Records, manifests, verification, the index, snapshot indexes, score
files and `select()` need neither Stable-Baselines3 nor torch, so they
run without the `train` extra; scoring a model (the harness, `rescore`,
`probe`) loads the training stack.

**The rule:** each run directory's own files are the source of truth.
The index is derived from them and regenerated whole, never appended to
(two Colab sessions co-appending one Drive file is how an index goes
wrong).

## What a run directory says about itself

| File | Written | Says |
|---|---|---|
| `run_record.json` | By `train()` at start (`running`), rewritten on every terminal path | What ran and how it ended (`courtside-run-record/1`): status `completed`, `early_stop`, `interrupted` or `crashed` with a `stop_reason`; budget; algorithm; task fingerprints; comparability key; seeds; git; versions; warm start; selection, checked against `best_model_meta.json`'s sha256 bindings and the saves `train()` committed; the evaluation `snapshots`, unindexed files named; wall clock |
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
renames leaves files of two saves: a meta whose sha256s no longer bind
the files beside it, or, where no meta binds the pair (a reward-selected
run's, a first headline save interrupted before its meta's rename), a
model beside another save's normalizer or none. `train()` keeps the
sha256s of every best-model save it committed
(`records.BestSaveLedger`) and checks the files on disk against them
when it seals the record, as well as the meta's bindings. Either check
failing makes the record say so: `selection.bindings` is `unbound` with
an `unbound_reason` naming which save each file is from, its timestep
and values are `null` (never the previous best's; a reward-selected
run, which has no selection otherwise, gets this block too), and the
best-model files are left out of `artifacts`. `stage_summary.txt` says
the same and suggests no deploy, `train()` prints a warning, `verify`
warns, the index flags the row, and the loaders
(`verify_best_checkpoint_pair`) and the harness refuse the pair.
Otherwise `bindings` is `verified` (`not_checked` in rows the index
backfills) under headline selection, and a reward-selected run's
selection stays `null`.

```bash
python -m courtside_dynamics.experiments.verify <run_dir> [<run_dir> ...]
python -m courtside_dynamics.experiments.verify --strict --json <run_dir>
```

Exit 0 only when every directory is complete. Every file the manifest
does not list is reported. Those at the post-training locations (the
notebooks' plots, audits and gate reports under `reports/`, the
harness's `scores/`, and the replay video `media/best_model.mp4`) fail
only `--strict`; any other fails the directory. A writer's temporary leftover (`.x.tmp-<pid>`,
`.x.sync-tmp`, `x.tmp.npz`, `x.tmp`) anywhere fails it too, whether
or not the manifest lists it: one listed was already there when the run
was sealed. The writers remove their temporaries when a write fails or
is interrupted. What the sealed record reports wrong inside a complete
directory (an unbound best model, snapshot files the index does not
bind) prints as a `WARNING` line (`warnings` in `--json`): the
directory is still exactly what `train()` left, and `--strict` fails
it. A directory whose record is terminal but which has no
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
  What a run record reports wrong (an `unbound` selection, unindexed
  snapshot files: `verify`'s warnings) is a problem too.

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

## Snapshots, protocols and re-scoring

**Snapshots.** With `TrainConfig.snapshot_evals` (on in the
PaddleTennis recipe), `train()` saves the policy and its
`VecNormalize` statistics at every evaluation, before any evaluator
runs at that step: `snapshots/snapshot_<N>_steps.zip` and
`snapshot_vecnormalize_<N>_steps.pkl`, the format of `best_model.zip`
and `best_vec_normalize.pkl` (no replay buffer), each pair bound by
sha256 in `snapshots/index.json` (`courtside-snapshots/1`) and written
like the best model (temporaries, then renames, the index last). The
run record's `snapshots` block lists their steps and any that failed
(a disk error is listed and training continues), and names in
`unindexed` the files the index does not bind: an interrupt between a
snapshot's renames leaves complete files outside the index (at the
first snapshot, no index at all, which the block reports as a
`problem`). `verify` warns about them and the harness refuses to score
them, never skipping them silently. The manifest hashes every file.
Saving reads the model only: the trainer digest's training components
are unchanged. A PaddleTennis SAC snapshot is about 3.5 MB.

**Protocols.** A protocol (`experiments.protocols.EvalProtocol`) fixes
everything an evaluation batch measures besides the policy: the eval
env (class and every constructor argument), episodes, the paired seed
block and reset options, the info keys and aggregates, the selection
keys and per-key deltas, the confirmation block, and the device (CPU).
Its fingerprint is the sha256 of that content. Registered protocols
score on the run's own seed block (its recorded paired block, else its
`eval_seed`, else `seed + 1,000,000`); a score row records the block it
used. `inloop` stands for each run's own in-loop evaluation, rebuilt
from its `config.json`.

**A performance-gated run has no protocol.** The gate writes each
stage's env attributes into its in-loop evaluator's env (stage 0 at
training start, the next at every advance), stamps the stage index into
every `eval_info.csv` row, and resets the selection at each advance.
No protocol reproduces that, so the five gated recipes
(`protocols.GATED_RECIPES`: WallBallBootstrap, WallBallDepthCurriculum,
WallBallDepthCurriculumAligned, WallBallGoalRally,
WallBallTrueBaseline) register none, `inloop` refuses a gated run, and
a gated run records `{"id": null, "fingerprint": null, "reason":
"performance_gate"}`. Its artifacts can still be scored under an
explicit protocol (a probe derived with a stage's env overrides, say),
which is not its in-loop evaluation.

| Protocol | What it is |
|---|---|
| `paddle-npoint@v3` | The PaddleTennis recipe's selection evaluation since 2026-10-17 (design L3): 30 paired 1,500-step episodes with alternating serves on the run's block, selected on the k≥2 headline (`episode_rally_returns_a_ep_mean`), `success_rate` and reward (deltas 1/60, 1/60, 0.25), confirmed on the block 100,000 on. The eval env has `contact_shaping`, `reach_shaping` and `hold_shaping` at 0, so the reward is task reward only |
| `paddle-npoint@v2` | The same with the contact and reach escrows on in the eval env: PaddleTennis selection from the 2026-10-05 instruments until 2026-10-17. Declared literally, so it stays scorable |
| `<recipe>-inloop@v1` | Every other ungated recipe's in-loop info-dict evaluator, derived from the recipe and pinned to its registration fingerprint (a recipe change makes it refuse to load: declare the old version literally, never re-pin) |

**@v2 and @v3 read the same policy the same way, except for reward.**
PaddleTennis observations hold no escrow state (no balance, no hold
window), so with the escrows on or off a policy sees bit-identical
observations and acts identically; every non-reward metric, the k≥2
headline and `success_rate` included, is identical under both, and
`episode_reward_mean` differs by the shaping the episodes collected
(`tests/test_eval_harness.py` pins both on the oracle and a trained
run). A selection under @v3 can differ from @v2's only through the
reward tie-break; that is the era boundary. Comparing evaluation
rewards across it needs the older runs re-scored under @v3 (below);
headline, success and k=1 compare as they are.

**What a run records.** `config.json`'s `comparability_key.eval_protocol`
(and the run record's `task.eval_protocol`, copied from it) is
`{"id", "fingerprint"}`: the fingerprint of the run's in-loop selection
protocol, the one `--protocol inloop` rebuilds from `config.json`, and
the id of the registered protocol of the run's recipe with that
fingerprint. A stock PaddleTennis run records `paddle-npoint@v3`; one
whose `[eval_env]` table restores the escrows records
`paddle-npoint@v2`; a quick test, or a run file that changes the
evaluation episodes or env, records `id: null` with its fingerprint,
so two runs compare only when both fields match. A performance-gated
run records both as `null`, with `"reason": "performance_gate"`.
`eval_protocol` is `null` for a run without an info-dict evaluator and
for runs recorded before 2026-10-17. A run file's `[env]` contact,
reach or hold shaping stays out of a PaddleTennis evaluation (the
recipe's @v3 overrides sit above `[env]`); an `[eval_env]` table, which
sits above them, restores it, and the run then records the protocol it
evaluates under.

**Re-scoring.**

```bash
python -m courtside_dynamics.experiments.rescore --protocol paddle-npoint@v3 <run_dir> ...
python -m courtside_dynamics.experiments.rescore --protocol inloop --which snapshots <run_dir>
python -m courtside_dynamics.experiments.rescore --protocol paddle-npoint@v3 --out <dir> <run_dir> ...
python -m courtside_dynamics.experiments.rescore --protocol paddle-npoint@v3 \
    --which best,checkpoints --steps 100000,1000000,2000000 <run_dir>
```

Every best model, snapshot and periodic checkpoint is scored (the
August runs' `checkpoints/paddle_tennis_sac_<N>_steps.zip` with
`paddle_tennis_sac_vecnormalize_<N>_steps.pkl` included), loaded through
the algorithm the run recorded. Before anything is scored, the best and
snapshot pairs are checked against the sha256s that bind them (the best
model against its run record's verdict too), every model's zip against
the run's recorded algorithm family, and an unindexed snapshot file is
named; checkpoints are hashed and recorded, since nothing binds them
(SB3's `CheckpointCallback` writes them in place, so a checkpoint torn
by an interrupt seals and fails only when it is loaded). One problem
anywhere refuses the whole rescore before anything is scored. Rows go to `<run_dir>/scores/<protocol>.jsonl`
(`courtside-score/1`: artifact, step, sha256s, protocol fingerprint,
seed block, `metrics` and `confirm_metrics`), or one combined file under
`--out`. `--steps` scores only the snapshots and checkpoints at those
env steps (besides the best model); a step that names none is refused.
The scoring code is the in-loop evaluator's own
(`collect_episode_metrics`), so on the same device a snapshot's row
under `inloop` equals its `eval_info.csv` row bit for bit for every
evaluation of a paired run, and for the first evaluation only of an
unpaired run (the BallBalance, WallBall and humanoid recipes' default,
and the August runs'): its in-loop stream continues across
evaluations, and its confirmation block is a fresh block with no
in-loop counterpart. The CLI's summary notes an unpaired run; a gated
run is refused (above). A model outside a run directory needs its
normalizer, or `raw_observations=True` (`probe --raw-observations`)
for one trained without `VecNormalize`: nothing else says which. A full
PaddleTennis score costs about 2.6 CPU-minutes per artifact (30
episodes of 1,500 steps on each of the two blocks).

**Re-scoring a Drive runs folder: `notebooks/rescore_runs.ipynb`.** The
Colab driver for design L3's re-scoring (a CPU runtime is enough). It
indexes `training_runs/PaddleTennis/sac` (the run index above), picks
the runs (by default every 2026-08 run that is not a stub and has a
model, which must include the registered run `20260816_235141`), and
scores each run's best model and every checkpoint under
`paddle-npoint@v3` into
`<Drive>/courtside-dynamics/rescores/PaddleTennis/sac/paddle-npoint@v3/<run id>/<series>/scores/`,
beside `training_runs/`, never inside it: the run directories stay
exactly what `train()` left, and the index of `training_runs/` never
meets a score file. Each run is scored whole or refused with its
reasons (a placeholder or unbound file, a missing normalizer) while the
others go on; its file is written when the run is done, and re-running
skips a run already scored for the same files (every model's and
normalizer's sha256). It then prints the per-checkpoint k≥2 headline
and k=1 (each point's first confirmed return:
`episode_valid_return_count_a` minus the headline) on both blocks,
plots them (`curves_<series>.png`), and writes a `select()` verdict per
run and series (`verdicts.json`): the checkpoint the protocol's rule
picks, next to the run's own `best_model` scored under the same
protocol. The plot and the verdicts cover every run scored into the
folder, and a session merges its verdicts and refusals into
`verdicts.json` rather than replacing what earlier sessions wrote. `QUICK =
True` scores the best model and three checkpoints per run (first,
middle, last) into a `-quick` folder; `SERIES = "snapshots"` scores a
newer run's evaluation snapshots instead of its checkpoints. The August
runs kept checkpoints every 100k against evaluations every 25k, so
their offline pick can differ from the in-loop one for that reason
alone.

**Selecting.** `experiments.select.select(rows, protocol)` is the
in-loop lexicographic rule as a pure function, except that confirmation
is compared within block: the first best is also scored on the
confirmation block, and a candidate must beat the best on each block
separately. The in-loop callback still compares a candidate's
confirmation with the first best's standard-block score until a later
best is confirmed (`repo_review_20261016.md` section 2.2);
`in_loop_select()` replays that rule over the same rows, so the two can
be compared run by run. It replays the paired, ungated rule only: given
an unpaired protocol that confirms (whose callback banks the weaker of
a candidate's two samples) it refuses, and gated runs have no protocol
(their selection resets at every advance). Changing the in-loop rule
waits on the maintainer's decision.

**Probing.** `python -m courtside_dynamics.experiments.probe --recipe
PaddleTennis --policy zero|random|oracle|camper|<model.zip>
[--normalizer ...] --protocol <id> [--episodes N] [--seed-start S]
[--env KEY=VALUE ...]` prints a policy's metrics under a protocol. The
witnesses are the golden digests' definitions; `--env` and `--episodes`
derive an unregistered protocol whose fingerprint is printed with the
result. Example, the RS1 witnesses of
`design_paddle_tennis_reach_shaping.md` section 3a (one-point play,
reach only, seeds 5500-5599):

```bash
python -m courtside_dynamics.experiments.probe --recipe PaddleTennis --policy oracle \
  --protocol paddle-npoint@v2 --env points_per_episode=1 --env contact_shaping=0.0 \
  --episodes 100 --seed-start 5500
```

The same witness under `--protocol paddle-npoint@v3` scores the bare
task. The lever "audit mode" (a reward term logged but not paid)
arrives with the levers (design L6).

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
