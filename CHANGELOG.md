# Changelog

Per-release notes for Courtside Dynamics. Entries describe physics,
observation, action, and recipe changes that determine which saved policies,
`VecNormalize` statistics, and learning curves remain comparable across
versions. Newest releases first.

## Unreleased

The PaddleTennis environment — the two-sided rung between wall-ball
and humanoid tennis — plus the infrastructure hardening between the
campaigns. Existing environments other than the new PaddleTennis are
unchanged on any healthy trajectory (the BallBalance and WallBall
changes below are confined to already-pathological nonfinite states
and to rendering), so their saved policies, `VecNormalize`
statistics, and learning curves all remain comparable with 0.25.0.
Within the unreleased PaddleTennis itself there are two eras: the
initial volley-rules freeze and the ground-rules amendment that
supersedes it (see the ground-rules bullet).

- **Offline evaluation fails loud where it cannot be faithful (phase-4
  verifier findings, 2026-10-17).** No env stream or training change:
  golden digests 22/22, recipe snapshot unchanged (96), and the trainer
  digest against `9f54ae5` differs only in the three components the
  @v3 step declared.
  - **Performance-gated runs have no in-loop protocol.** The gate
    rescales its evaluator's env per stage (stage 0 from training
    start), stamps the stage into every row and resets the selection at
    each advance, so the protocol rebuilt from the recipe's eval env
    scored an evaluation the run never ran (a tiny gated run's rows
    disagreed with `eval_info.csv` silently). `from_train_config`,
    `from_run_config`, `protocol_for_run` and `rescore(..., "inloop")`
    now refuse a gated run (`GatedEvaluationError`); the five gated
    WallBall recipes no longer register `<recipe>-inloop@v1` (registered
    since `0be8d77`, unreleased), and a gated run's
    `eval_protocol` is `{"id": null, "fingerprint": null, "reason":
    "performance_gate"}`.
  - **Every best pair is bound.** `train()` now keeps the sha256s of
    every best-model save it commits (`records.BestSaveLedger`) and
    checks the files on disk against them when it seals the record. A
    reward-path pair split between its two renames (a new model beside
    the previous normalizer) and a first headline save interrupted
    before its meta's rename (a model with no meta) were sealed as
    healthy, listed in the record and scored by the harness; now the
    record's selection is `unbound` (a reward-selected run gets the
    block too), the files leave `artifacts`, the summary suggests no
    deploy, and the loaders and the harness refuse the pair.
  - **`verify` reports what the record says is wrong.** A complete
    directory whose record marks the best model unbound, or lists
    snapshot files its index does not bind, prints a `WARNING` line
    (`warnings` in `--json`); `--strict` fails it. The index flags both.
  - **Unindexed snapshot files are named.** A snapshot interrupted
    between its renames left files the index did not list (or, at the
    first snapshot, no index); the record's `snapshots` block now lists
    them in `unindexed` (additive), and the harness refuses to score
    them instead of ignoring them or failing the run as unreadable.
  - **Harness and CLI refusals.** A model outside a run is refused
    without a normalizer unless `raw_observations=True`
    (`probe --raw-observations`); a run whose `config.json` records the
    wrong algorithm is refused before scoring (it raised a `TypeError`
    inside SB3 that the CLI and the notebook did not catch);
    `in_loop_select()` refuses an unpaired confirming protocol, whose
    in-loop rule it does not replay; `rescore --protocol inloop` notes
    when only an unpaired run's first evaluation can match
    `eval_info.csv`.
  - **`notebooks/rescore_runs.ipynb`** keeps each series' scores apart
    (`<run id>/<series>/scores/`), re-scores a run whose normalizer
    changed, plots `curves_<series>.png` and merges `verdicts.json` (keyed
    by run and series) across sessions instead of rewriting it from one
    session's picks.
  - **Fixed:** `test_l2_toml_spelling_reaches_both_envs` asserted the
    pre-@v3 contract (an `[env]` contact escrow reaching evaluation);
    it now pins @v3's, and an `[eval_env]` table restoring both escrows
    reaches the evaluation env and records `paddle-npoint@v2`.
  - **Open:** periodic checkpoints are hashed and recorded, never bound
    (SB3's `CheckpointCallback` writes them in place, so a torn one
    fails only when loaded); the docs now say so instead of claiming
    every scored pair is checked against a binding.
- **PaddleTennis selection moves to `paddle-npoint@v3`, the
  evaluation env without training aids (design L3, 2026-10-17): a
  selection-protocol era.** The recipe's `eval_env_overrides` set
  `contact_shaping`, `reach_shaping` and `hold_shaping` to 0, so the
  reward tie-break in selection and every PaddleTennis evaluation
  reward (`eval_info.csv`, `evaluations.npz`, the closing evaluation,
  the summary's eval lines) are task reward only. Training keeps the
  escrows: golden digests 22/22; the trainer digest against `9f54ae5`
  matches every training, `evaluations.npz`, best-model weight and
  diagnosis component of the four jobs, and differs only in the
  PaddleTennis job's `eval_info.csv` and `best_model_meta.json`
  metrics (at 768 steps @v2's `episode_reward_mean` was a 3.7e-9 float32
  residue of a reach escrow paid and clawed back; @v3 reads 0.0, the same
  model selected) and its `run_layout_artifacts` (`snapshots_dir`, from
  the previous step). The recipe snapshot was re-recorded: the 12
  PaddleTennis resolutions' evaluation kwargs read 0.0 for the contact
  and reach scales (were 0.25) and gain `hold_shaping` 0.0. The
  overrides sit above a run file's `[env]` table, so an `[env]`
  contact, reach or hold shaping no longer reaches evaluation; an
  `[eval_env]` table restores it (and the run records the protocol it
  then evaluates under).
  - **Comparability.** The observations hold no escrow state (tested),
    so a policy acts identically under @v2 and @v3: the k≥2 headline,
    `success_rate`, k=1 and every other non-reward metric are identical
    (tested on the oracle and a trained run), and `episode_reward_mean`
    differs by the shaping collected. Evaluation rewards and reward
    tie-break picks compare across 2026-10-17 only after re-scoring the
    older runs under @v3.
  - **`paddle-npoint@v3`** is registered beside `paddle-npoint@v2`
    (still scorable); `RECIPE_PROTOCOLS["PaddleTennis"]` names @v3.
  - **Runs record their protocol.** `config.json`'s
    `comparability_key.eval_protocol` and the run record's
    `task.eval_protocol` (both `null` since L1) are now
    `{"id", "fingerprint"}`: the in-loop selection protocol's
    fingerprint and the registered id it equals (`paddle-npoint@v3` for
    a stock PaddleTennis run, `ball-balance-inloop@v1` for BallBalance,
    `null` for a quick test or a run file that changes the evaluation).
    `null` for a run without an info-dict evaluator.
  - **`notebooks/rescore_runs.ipynb`** (new): re-scores a Drive runs
    folder (default: the 2026-08 PaddleTennis runs, the registered run
    `20260816_235141` among them) under @v3 on CPU, best model and every
    checkpoint, into `courtside-dynamics/rescores/` beside
    `training_runs/`, never inside a run; prints and plots the
    per-checkpoint k≥2 and k=1 curves and a `select()` verdict per run;
    resumes, and `QUICK` scores the best model and 3 checkpoints. Its
    `REPO_REF` is a placeholder until the week-1 wrap-up bumps it.
  - **Harness:** `rescore(steps=...)` and the CLI's `--steps` score only
    the snapshots and checkpoints at those steps (a missing step is
    refused); `harness.series_steps()` lists them without hashing.
- **Offline evaluation: snapshots, protocols, the re-scoring harness,
  `select()` and a probe CLI (design M3 + L2, 2026-10-17).** No env
  stream or training change: golden digests 22/22; the recipe snapshot
  differs only by the new `snapshot_evals` field (re-recorded: `false`
  in 80 resolutions, `true` in the 12 PaddleTennis ones); the trainer
  digest against `9f54ae5` matches every policy, normalizer, monitor,
  RNG, `resolved_model`, evaluation, `eval_info.csv` and best-model
  component, and the PaddleTennis job's `run_layout_artifacts` gains
  `snapshots_dir` as declared.
  - **Snapshots.** `TrainConfig.snapshot_evals` (default off; on in the
    PaddleTennis recipe) saves the policy and its normalizer at every
    evaluation, before any evaluator runs at that step:
    `snapshots/snapshot_<N>_steps.zip` and
    `snapshot_vecnormalize_<N>_steps.pkl`, the best model's format,
    bound by sha256 in `snapshots/index.json`, written atomically,
    listed in the run record's new `snapshots` block and in the
    manifest. A disk error is listed and training continues. About 3.5
    MB per PaddleTennis SAC snapshot (critics and optimizer states
    included), so about 140 MB per 1M steps.
  - **Protocols** (`experiments.protocols`): frozen `EvalProtocol`s with
    a content fingerprint. `paddle-npoint@v2` was the PaddleTennis
    in-loop selection evaluation until @v3 (above), declared literally;
    every other recipe got `<recipe>-inloop@v1`, derived and pinned
    (the gated ones since withdrawn, above). Scores use the run's own
    seed block and record it.
  - **Harness** (`experiments.harness`): `evaluate()` and `rescore()`
    run the in-loop evaluator's own collection core, which moved out of
    `InfoDictEvalCallback._collect_metrics` into
    `collect_episode_metrics` (verbatim; the in-loop path is
    bit-identical). A snapshot's offline row equals its `eval_info.csv`
    row bit for bit on CPU (SAC and PPO, paired; an unpaired run's first
    evaluation only). `python -m
    courtside_dynamics.experiments.rescore --protocol <id> <run_dir>...`
    scores best models, snapshots and checkpoints (the August
    `checkpoints/paddle_tennis_sac_<N>_steps.zip` layout included) into
    `scores/<protocol>.jsonl`, now a post-training location for
    `verify`.
  - **`select()`** (`experiments.select`): the lexicographic per-key
    delta rule as a pure function over score rows, with confirmation
    compared within block (`repo_review_20261016` section 2.2's
    semantics). The in-loop `confirm_best` rule is unchanged pending
    the maintainer's decision; `in_loop_select()` replays it for
    comparison. The callback's comparison is now `select.improves`.
  - **Probe CLI** (`python -m courtside_dynamics.experiments.probe`):
    the zero, random, oracle and camper witnesses (the golden digests'
    definitions, now `digests.scripted_policy`) or a model under a
    protocol, with `--env`/`--episodes` deriving a fingerprinted probe
    protocol. It reproduces the RS1 reach-shaping witness table.
  - Not yet: the lever audit mode (L6). (`eval_protocol` in the
    comparability key and run record was `null` until L3, above.)
- **The best model is written atomically and the record checks what it
  seals (L1 verifier round 2, 2026-10-17).** No env stream or training
  change: golden digests 22/22, recipe snapshot unchanged (96), and the
  trainer digest against `9f54ae5` matches every component of all four
  jobs, the best-model ones included.
  - **Best-model writes are atomic on both selection paths.**
    `best_model.zip`, `best_vec_normalize.pkl` and
    `best_model_meta.json` go to `.<name>.tmp-<pid>` files beside their
    targets and are renamed into place together, the meta last
    (`experiments.records.ReplaceTogether`). A `KeyboardInterrupt` inside
    a write (a stopped Colab cell, the commonest interrupted run) left a
    torn `best_model.zip` that the record listed and `verify --strict`
    certified; now the previous best stands whole and nothing is left
    behind. Under reward selection the save moved from SB3's
    `EvalCallback` (an in-place write) into its new-best callback
    (`_SaveBestOnNewBest`, which replaces `_SaveVecNormalizeOnNewBest`):
    same model at the same moment, so the same content (the trainer
    digest's best-model components match).
  - **The record's selection is checked against the meta's sha256
    bindings.** An interrupt between two renames leaves a meta that no
    longer binds the files beside it; the record copied its selection,
    describing the previous best beside the new model. `selection` now
    carries `bindings` (`verified`, `unbound`, or `not_checked` for
    backfilled index rows) and `unbound_reason`; an unbound selection has
    `null` timestep, values and updates, and the best triple is left out
    of `artifacts`. `stage_summary.txt` reports the best model as
    unbound, `train()` prints a warning, and the index flags the row.
  - **A temporary leftover fails verification whether or not the
    manifest lists it.** The rule applied only to unlisted files, so a
    leftover present when the run sealed itself verified complete. The
    merged-stream `evaluations.npz` writer (which left
    `evaluations.npz.tmp.npz` on any non-`OSError`, an interrupt
    included), the gate's `curriculum_stages.json` and the ladder
    certification report now remove their temporaries when a write fails
    or is interrupted.
  - **Index robustness.** A `config.json` or `best_model_meta.json` field
    of the wrong type (a string `warm_start`, a list `env`, an integer
    `selection_keys` ...) is read as missing and named in the row's
    problems (`records.shape_problems`) instead of failing the whole
    build with a bare `AttributeError`; so is a record whose status is
    missing or not a string. A run-named directory holding
    `run_config.toml` or `metrics/` but no `config.json` (a launch that
    died in the 0.1-0.2 s between the two files) is an aborted stub row,
    not a skip; TennisWall's `run_config.json` layout is still skipped.
  - **Messages.** A directory whose record is terminal but which has no
    manifest is "never sealed" (an unfinished mirror pass, re-run
    `sync_run_dir`), not "never reached a terminal path". An interrupted
    final mirror prints the `sync_run_dir` re-run before re-raising. An
    on-policy guard that held at the final evaluation reads `full_budget:
    collected all N steps ... the last rollout's update was skipped (SB3
    n_updates K)`: SB3 stops before that rollout's update.
  - The campaign design's S2 row credits the 3M override to LT1's
    unregistered companion (`20260828_113136`), not the registered LT1
    run, which trained its frozen 1M.
- **Run-record, mirror and index fixes from the L1 verifiers, plus the
  PPO round-2 leftovers (2026-10-17).** No env stream or training
  change: golden digests 22/22, recipe snapshot unchanged, and the
  trainer digest against `09f202a` matches every training component of
  the SAC, DemoSAC, PaddleTennis-SAC and PPO jobs, `run_layout_artifacts`
  differing only by `run_record` and `sync_manifest` as declared. The
  PPO changes are summary and message text.
  - **A run that raises while closing never reads `completed`.** An
    exception after the run reported its outcome (an env close or the
    SB3 logger's TensorBoard flush failing after a completed run) left
    the record `completed` with no stop reason while `train()` raised.
    The record now says `crashed` (`interrupted` for a
    `KeyboardInterrupt`) with `crashed after the run ended as completed:
    <error>`. The exception is the final mirror: the local directory is
    sealed first, so a mirror that keeps failing (or is interrupted)
    raises with the local record `completed`, which the record cannot
    describe; the error names the `sync_run_dir` re-run.
  - **`reuse_log_dir=True` rotates every run-layout artifact but
    `config.json`** (the model files, `metrics/`, `reports/`, `media/`,
    `checkpoints/`, summary, record and manifest), not just the logs:
    an attempt that crashed before its first evaluation recorded the
    previous attempt's selection and models as its own.
  - **Mirrors.** `train()` mirrors to and records the `sync_dir` it
    validated (`~` expanded, resolved once); a `~/Drive/run` used to land
    in a literal `./~/Drive/run`. `RunDirMirror` and `sync_run_dir`
    expand and resolve both paths, refuse one inside the other, and
    never delete or overwrite a file they cannot prove is this run's
    (copied by the mirror, listed by its manifest for this run id, or
    byte-identical to the source): a destination holding anything else,
    or another run's mirror, raises `SyncError` before anything is
    written, instead of silently deleting every unrelated file under a
    mistyped path. The final pass must reproduce the local manifest's
    files, or it is redone without the change cache.
  - **`verify_run_dir` / `experiments.verify` fail stray files by
    default.** An unlisted file is tolerated only at the post-training
    locations (`reports/`, `media/best_model.mp4`) and only without
    `--strict`; any other unlisted file, and a writer's temporary
    leftover (`.x.tmp-<pid>`, `.x.sync-tmp`, `*.tmp`) anywhere, makes
    the directory incomplete. Problem messages name files relative to
    the run directory; `read_json_object` takes a `label`, reads a
    UTF-8 byte-order mark, and no longer puts a file's bytes into its
    error.
  - **Run index.** A run record that still says `running` is flagged in
    `index.problems` with or without verification (training or dead
    without a manifest; a mid-run mirror snapshot with one), is a stub
    without a final model, and is counted on the CLI line. A summary
    that exists but cannot be decoded leaves the status `null` with a
    short problem instead of `aborted`. A stray `config.json` in a
    non-run folder no longer hides the runs beneath it; a run-named
    directory with a zero-byte or foreign `config.json` is an aborted
    row carrying the problem; a BOM and an integral float budget are
    read as written; `files.best_model_video` reports the replay video,
    at the root of a pre-0.14 run; an on-policy summary's `Best eval`
    step (`(at N steps, SB3 n_updates K)`) is parsed instead of read as
    missing. The full 60-directory archive
    indexes to 59 rows, 9 aborted, 7 stubs, 47 of 47 map entries joined,
    no warnings.
  - **`stage_summary.txt` is written as UTF-8** whatever the locale.
  - **PPO round 2.** A reward-selected on-policy run with
    `info_dict_eval=False` gets its best's 0 updates from
    `resolved_training.rollout_env_steps` (the zero-update warning used
    to be skipped); a warm-started run's zero-update notes name the
    warm-start source's policy instead of an untrained one, and a
    `log_std` carried from the source is reported as differing from
    `log_std_init`, not as drift; the on-policy cadence refusal states
    its real reason; the README no longer promises a nightly probe.
  - `legacy_levers.toml`: LH1b's note records its undeclared 2M /
    patience-60 override (now counted in the campaign design's S2), and
    the L2 replication's its patience change.
- **The regenerated run index and the legacy backfill (design L1,
  second half; 2026-10-17).** No env stream, training or run-directory
  change: golden digests 22/22, recipe snapshot unchanged, and the
  trainer digest against `09f202a` differs only where the first half
  declared (`run_layout_artifacts` gains `run_record` and
  `sync_manifest`).
  - **`python -m courtside_dynamics.experiments.index --root <runs
    root> [--out <file>] [--legacy-levers <toml>] [--skip-verify]`**
    (`experiments.index.build_index`, no SB3 or torch) writes
    `<root>/index/runs.jsonl`: one row per run directory, sorted by path,
    keys sorted, byte-identical when regenerated over unchanged
    directories. A row is the run record's keys plus `index` (path,
    source, `version_inferred`, the inferred fields, stub, manifest
    verdict, model/checkpoint/video presence, run config basename and
    sha256, the parsed summary head, problems) and `campaign`.
  - **Run-record directories** are taken as written, with
    `verify_run_dir`'s verdict (`--skip-verify` leaves it unchecked).
  - **Older directories are backfilled** into the record's shape from
    `config.json` (through `build_run_record`), `stage_summary.txt`
    and `best_model_meta.json`, marked `version_inferred`. The
    summary's `X of Y budget (stopped early)` line makes an early stop
    `early_stop`; a stop reason on a full-budget run (train-3) becomes
    `completed` with the `full_budget: ...` reason `train()` now
    writes. No summary: `aborted`, and a stub without a final model; an
    empty run-named directory is an aborted stub. A directory without a
    run marker, or with a `config.json` that is not a courtside run
    config, is skipped and listed, never parsed. No machine path
    reaches a backfilled row.
  - **`experiments/legacy_levers.toml`** (new package data): each of
    the 20 PaddleTennis run directories mapped to one lever (or
    `baseline`/`infra`), and the 27 WallBall directories of the five
    era folders to their recipe era, each with its verdict and doc.
    The index joins it by run id. `tools/smoke_wheel.py` checks that an
    installed wheel carries it and that it parses.
  - Validated against a text-only mirror of the Drive archive (25
    directories: all 20 PaddleTennis and 5 WallBall): 8 aborted, 7 of
    them stubs; all 20 PaddleTennis runs joined to one lever; git sha,
    algorithm, budget against steps trained, best step and stop reason
    agree with the source files for all 17 runs with a summary. New
    [`docs/experiments.md`](docs/experiments.md) documents records,
    manifests, `verify` and the index, Colab use included.
- **Run records, sync manifests and run-directory trust (design L1,
  first half; 2026-10-17).** No env stream or training change: golden
  digests 22/22; the trainer digest against `09f202a` matches every
  policy, normalizer, monitor, evaluation, best-model, checkpoint,
  diagnosis, `resolved_model` and RNG component of the SAC, DemoSAC,
  PaddleTennis-SAC and PPO jobs, and `run_layout_artifacts` differs
  only by the two new files, `run_record` and `sync_manifest`. Recipe
  snapshot re-recorded: each of the 92 entries gains
  `sync_dir: null`; nothing else changes.
  - **`run_record.json`** (`courtside-run-record/1`, new package
    `courtside_dynamics.experiments`, which imports neither SB3 nor
    torch). Written atomically right after `config.json` with status
    `running`, rewritten on every terminal path: `completed`,
    `early_stop` (patience on either selection path, the degenerate
    guard or a gate stage budget; `stop_reason` says which),
    `interrupted`, `crashed` (the crash salvage, or a failure before or
    after `learn()` once `config.json` exists). It records the run id
    (the `log_dir` leaf), UTC start and finish, the budget (total,
    trained, SB3 `n_updates`), the algorithm, recipe, task (env
    classes, the sha256 of their constructor kwargs, the observation
    fingerprint, `eval_protocol: null`), the comparability key, seeds,
    git sha and dirty state, library versions, the warm start's source
    directory name and artifact digests, the selection from
    `best_model_meta.json`, the wall clock with the periodic and
    closing evaluations' share, the pre-flight warnings, the artifacts
    present, and `experiment` placeholders for the spec system.
    `stage_summary.txt` keeps `Status: completed` for an early stop
    (the campaign resumes a leg only from it); the record says which.
  - **`sync_manifest.json`** (`courtside-sync-manifest/1`): the path,
    size and sha256 of every other file, written atomically and last on
    every terminal path, once every env and the SB3 logger are closed.
    `experiments.records.verify_run_dir` and `python -m
    courtside_dynamics.experiments.verify <run_dir>` (exit 0 only when
    complete) call a directory complete when its manifest verifies and
    the run record it lists is terminal; files written after the
    manifest are listed as unlisted and fail only `--strict`.
  - **Local-then-sync.** New `TrainConfig.sync_dir` (default `None`,
    recorded in `config.json`, refused in run-config files): train under
    `log_dir` on local disk and mirror it to `sync_dir` at every
    checkpoint and at the end. Each copied file is re-hashed and the
    mirror's manifest is written only after every one matches; a failed
    pass leaves the previous manifest, so a partial mirror never
    verifies. A failed checkpoint pass is printed and training goes on;
    the final mirror retries twice, then raises once the local run
    directory is sealed (`experiments.sync.sync_run_dir` re-runs it).
    `sync_dir` must be absent or empty, outside `log_dir`, with an
    existing parent, checked before any output. Notebook knobs, default
    off: `sb3_training` `LOCAL_THEN_SYNC` (section 9d mirrors the
    post-training plots and video before the disconnect) and
    `paddle_tennis_campaign` `LOCAL_THEN_SYNC` (each leg trains locally
    and mirrors into its Drive attempt dir; operational, not
    fingerprinted).
  - **Run-directory trust fixes** (`repo_review_20261016`):
    `disconnect_runtime` flushes and unmounts Drive before tearing the
    VM down and leaves it up when the flush fails (critic-4);
    `check_run_artifacts` and the summary's Artifacts section count an
    empty directory artifact, such as `media/videos/`, as missing and
    say so (training-data-5); a guard that fires once the run has
    trained its whole budget (`num_timesteps >= total_timesteps`, e.g.
    at the final evaluation of a budget that is a multiple of
    `eval_freq`) now reads `completed` with a `full_budget: ...` stop
    reason that quotes it (one at the final scheduled evaluation of any
    other budget left steps untrained and reads `early_stop`), and
    `train()` warns
    before any output (and `config.json`'s
    `resolved_training.early_stop_reachability` and the run record keep
    it) when patience or the degenerate guard cannot fire inside the
    budget: BallBalance's recipe patience is evaluation 41 of 40
    (train-3, training-data-3); the reward-path early stop records its
    `stop_reason`, found through `EvalCallback`'s nested
    `callback_after_eval`, and stops exactly when it did (train-1);
    `reuse_log_dir=True` also rotates the run record, the sync
    manifest and the monitor and TensorBoard directories
    (`metrics/monitor.attempt_<UTC time>`), and a `log_dir` holding a
    previous attempt's run record is refused like one holding its
    evaluation rows (train-5); the `KeyboardInterrupt` salvage saves
    `final_model` and the normalizer first and runs the gate's
    `finalize`, the closing evaluation and the summary each
    log-and-continue, like the crash salvage (train-2); and
    `SubstepTennisEventSampler.reset()` discards a control step an
    exception left open instead of raising, so an interrupted env can
    be reset (`begin_control_step` still refuses, naming the cure),
    and a recovered PaddleTennis env replays a fresh env's stream byte
    for byte (rules-events-1). Interrupted summaries now carry a
    `Stop reason: interrupted: ...` line.
- **Verifier fixes on the PPO commits (2026-10-17).** No env stream or
  SAC-family training change: the trainer digest against `f920d8f`
  matches every component of the SAC, DemoSAC and PaddleTennis-SAC
  jobs, and the PPO job is unchanged from the previous commit. Every
  change below is on-policy only unless it says otherwise.
  - **The deployable best states its updates.** An evaluation runs
    before its rollout's update, so a best selected at the first
    evaluation of a one-rollout cadence was the untrained initial
    policy, and the BallBalance quick test kept it (438 at 4,096
    steps, 0 updates) while its summary advised deploying it.
    `best_model_meta.json` gains a top-level `n_updates` (SB3's count
    behind the selecting evaluation); the summary's "Best eval" and
    "Best model" lines give the count, a `best_model.zip` selected at 0
    is flagged on its line and printed, and "Final vs best" no longer
    advises deploying such a best.
  - **Quick test: two rollouts per evaluation.** A PPO quick test
    sizes `n_steps` by `eval_freq // (2 · n_envs)` (was
    `// n_envs`) and rounds its preset evaluation cadence to at least
    two rollouts, so evaluation 1 follows an update: 256 at 8 workers
    (was 512), 128 at 16 (was 256), 2,048 on one worker (unchanged).
    BallBalance's quick test now trains 13 rollouts (SB3 n_updates
    130) and selects its best after one update. Snapshot: the 10 PPO
    quick-test entries at 8 workers are re-recorded (`n_steps` 512 →
    256); nothing else changes.
  - **The last update is logged.** SB3 dumps an update's `train/*`
    metrics with the next rollout's row, so the final update's never
    reached `progress.csv` (the BallBalance quick test read
    `train/n_updates` 60 against SB3's 70). `train()` now dumps the
    logger once after `learn()` returns when they are pending, so
    TensorBoard, Training Health and the flags include it.
  - **Epochs per update.** Training Health gains an "Epochs per
    update" line (the steps of `train/n_updates`) and a flag when the
    last-quarter mean is below half of `n_epochs`: the gSDE profiles'
    `target_kl` ended every update in its first epoch while their mean
    approx_kl looked healthy (the PaddleTennis quick test now reads
    1.77 of 10, flagged). New `artifacts.epochs_per_update`;
    `on_policy_health_flags` takes `n_epochs` (optional).
  - **Resolved reward normalization in the summary.** Hyperparameters
    list `reward_norm` (the resolved switch, the configured value, and
    the normalizer's gamma and its source) from `config.json`'s
    `resolved_training`.
  - **Refused, not silent.** `degenerate_min_evals` must be `None` or
    a nonnegative integer, checked before any output for every
    family (an SAC run with −1 used to fail after `config.json` with a
    `TypeError` from the update-floor print, and a PPO run accepted
    it). The performance gate's `finalize` raises a pending advance
    that fails on a normal training end, after writing the stage
    history, and only reports it under the interrupt and crash
    salvage (`finalize(salvage=True)`). The on-policy pre-flight
    refuses a gate `stage_eval_budget` stop at or below the update
    floor (it has no warm-up to raise), and an actor-critic `net_arch`
    dict with keys other than `pi` and `vf` (SAC's `qf` built a PPO
    value network with no hidden layer). Without a run record,
    `load_run_model` refuses an `algo` whose family the zip's own
    policy class contradicts (`SAC.load` of a PPO checkpoint failed
    inside SB3).
  - **Docs.** `TrainConfig.vec_env` and the run-config spec say a
    plain script using `"subproc"` needs an `if __name__ ==
    "__main__":` guard. README's CI learning bar reads about 11 s, as
    elsewhere. The plan's Open list adds the gSDE profiles' learning
    rate against `target_kl`, the humanoid profile's move to 8
    subprocess workers × 512, and the humanoid notebook's `REPO_REF`
    bump, a merge prerequisite: until it lands, the notebook's default
    `AUTO_ADVANCE` refuses the pinned package before any stage trains.
- **PPO is first-class: loaders, provenance, health flags, learning
  bars, notebooks and docs (PPO plan Steps 10–12 and 14,
  2026-10-17).** No env stream or SAC-family training change: the
  trainer digest against `f920d8f` matches every component of the SAC,
  DemoSAC and PaddleTennis-SAC jobs. The PPO job's `resolved_model`
  gains the keys below (and its `eval_info.csv` keeps Step 6's
  `n_updates` rows); the recipe snapshot is unchanged.
  - **Loaders (Step 10).** New `algos.load_run_model(path, algo=None)`
    (with `algos.locate_run_config` and `algos.recorded_run_algo`)
    loads a model through the algorithm its run's `config.json` records:
    `SAC.load` for SAC and DemoSAC, `PPO.load` for PPO. An explicit
    `algo` that disagrees with the record raises, and so does a record
    without `train_config.algo`; a checkpoint outside any run needs
    `algo=`. It replaces the hard-coded `SAC.load` in
    `paddle_diagnosis.native_checkpoint_policy` (new `algo=` keyword),
    the P5 transfer champion loader and `score_paddle_stage`, and backs
    `_load_best_model`, `record_best_model_video` (default `algo` is now
    `None`, was `"SAC"`) and `evaluate_best_wall_ball` (a DemoSAC run
    now loads through `SAC.load`). The diagnosis, post-swing, k=2
    harvest, k=2 step-0 and P5 transfer tools take `--algo`.
    `score_paddle_stage`'s report names the scored algorithm and the
    run's comparability key.
  - **Provenance (Step 11).** `config.json` gains top-level `algorithm`
    (`name`, `family`, `implementation` = `sb3==<version>`) and
    `comparability_key` (`task`, `eval_protocol` = `null` for now,
    `algo_family`) blocks, for every run. On-policy `resolved_model`
    adds `target_kl`, `sde_sample_freq`, `log_std_init` and `ortho_init`,
    and is re-recorded when training ends with SB3's `n_updates` and
    `timesteps` (trained vs budget); SAC's block is byte-identical. An
    on-policy summary reads its timesteps against the budget, saying by
    how much the last rollout ran past it.
  - **Training Health flags (Step 11).** A PPO run's
    `stage_summary.txt` flags, over the last quarter of its updates,
    mean approx_kl above max(2 × target_kl, 0.05), mean clip_fraction
    above 0.3, mean explained_variance below 0, and a final log std
    more than 1 nat from `log_std_init` (also printed). New
    `artifacts.on_policy_health_flags`.
  - **Tests (Step 12).** A parametrized SAC/PPO end-to-end `train()`
    test, PPO determinism, `score_paddle_stage` on a real PPO
    checkpoint, and a CI learning bar: PPO learns a 1-D reach through
    `train()` in about 11 s (best paired return −3.2 to −3.4 over three
    seeds against a −6 bar; the zero action scores −26.7).
  - **`tools/ppo_learning_probe.py` (Step 12).** BallBalance on its
    recipe's PPO profile for at most 330 s, then the best model on 20
    held-out seeds; PASS at a mean of 700 of 750. Run once here: 750.0
    on all 20 seeds after 491,520 steps.
  - **Notebooks.** `paddle_tennis_campaign` admits `ALGO = "PPO"`: the
    legs use the recipe's PPO profile on new `PPO_N_ENVS` /
    `PPO_EVAL_FREQ` / `PPO_CHECKPOINT_FREQ` (8 workers, 24,576,
    98,304), no SAC key reaches them, the settings cell refuses
    `REQUIRE_DEVICE = "cuda"`, a temperature skip, a geometry the
    on-policy pre-flight refuses and a package without first-class PPO
    (the pinned `REPO_REF` included), a PPO campaign refuses a non-PPO
    warm-start lineage before its manifest exists, and it lives under
    `PaddleTennisCampaign/ppo/`. The SAC-family plan is unchanged.
    `sb3_training` keeps a recipe's PPO profile worker count when
    switching to PPO (it forced 16), and its worker table and
    training-health notes describe the PPO profiles and flags.
  - **Docs (Step 14).** README "Algorithms", the `TrainConfig.algo`
    and `build_train_config` docstrings, the run-config spec's
    per-family settings tables, a DECISIONS entry, and the plan's
    status line.
- **PPO runtime: stops counted in updates, matched normalizer gamma,
  CPU default, warm-start exploration, subprocess workers (PPO plan
  Steps 6–9 and 13, 2026-10-17).** No env stream changes, and every SAC
  and DemoSAC training path is unchanged: the trainer digest against
  `f920d8f` matches on the SAC, DemoSAC and PaddleTennis-SAC jobs.
  The PPO job differs only in `eval_info.csv`, which gains its
  `n_updates` rows.
  - **Stops in policy updates (Step 6).** New
    `TrainConfig.min_policy_updates_before_stop` (K, default 20). An
    on-policy run floors the early-stop warm-ups (the headline
    evaluator's patience warm-up, SB3's `StopTrainingOnNoModelImprovement`
    `min_evals`, and `degenerate_min_evals`) at
    ceil(K · n_steps · n_envs / eval_freq) evaluations, so no early stop
    fires before K updates; a raised warm-up is printed, and 0 disables
    the floor. Off-policy runs must leave the default (any other value
    is refused). On the shipped PPO profiles only the degenerate stop
    moves: on PaddleTennis, WallBallBootstrap, GoalRally and
    TrueBaseline it first fires at evaluation 11 (43 updates) instead
    of 6 (23) at 8 workers, and at 16 (31) instead of 6 (11) at 16;
    WallBallBaseline's moves only at 16 workers (27 to 31 updates).
  - **Rollout boundaries (Step 6).** Under an on-policy model,
    `PerformanceGatedEnvStagesCallback` applies an earned advance at the
    end of the rollout its evaluation landed in (`finalize` applies one
    a stop left pending, so the stage history matches an off-policy
    run's), and `LinearEnvAttrScheduleCallback` moves its value only at
    rollout starts. Both detect the model at training start; off-policy
    runs are unchanged.
  - **`eval_info.csv` `n_updates` (Step 6).** On-policy runs write one
    `n_updates` row per evaluation: SB3's `_n_updates` (the
    `train/n_updates` count) behind the evaluated policy. CSV only, so
    selection, the gate and TensorBoard never see it.
  - **Normalizer gamma (Step 7).** An on-policy run that normalizes
    rewards passes the model's gamma to all four `SelectiveVecNormalize`
    constructions and sets it on a warm-start normalizer; the
    pre-flight refuses a non-numeric gamma, and train() checks the
    normalizer's against the built model. Every other run keeps
    VecNormalize's 0.99 (SB3 updates `ret_rms` even with reward
    normalization off, so an SAC run's pickled stats would move).
  - **CPU by default (Step 8).** An on-policy run with SB3's MLP
    actor-critic policy, no `model_kwargs["device"]` and
    `require_device=None` is built with `device="cpu"`. CNN and
    multi-input policies, explicit devices and the SAC family keep
    their device.
  - **Warm-start exploration (Step 9).** New
    `WarmStartConfig.reset_log_std` (default False): an on-policy
    target refills the transferred `policy.log_std` from its own
    `log_std_init`. On-policy `transfer_log_ent_coef=False` (a no-op
    before) is refused with a pointer to `reset_log_std`; the SAC
    family refuses `reset_log_std`. On-policy provenance lists
    `policy.log_std` as transferred or reset and records
    `reset_log_std` (and `log_std_reset_value`). The humanoid notebook's
    AUTO_ADVANCE passes `reset_log_std=True` and refuses, before any
    stage trains, a package without it, which includes the pinned
    `REPO_REF` until its bump.
  - **Subprocess workers (Step 13).** New `TrainConfig.vec_env`
    (`"dummy"` default, or `"subproc"`) for the training workers only.
    `"subproc"` uses `train.DummyDtypeSubprocVecEnv`, which returns
    DummyVecEnv's float32 rewards and observation-space dtypes (stock
    `SubprocVecEnv` returns float64 rewards, which VecNormalize's return
    statistics would see), and runs `check_env` on a throwaway instance
    in the parent. A PPO run trains bit-identically either way.
  - **`config.json`.** `train_config` records the two new fields and
    `warm_start.reset_log_std`. A new top-level `resolved_training`
    block records the resolved reward normalization, the normalizer's
    gamma and its source, the device asked of SB3 and why, and an
    on-policy run's rollout and evaluation floor. `resolved_model` is
    unchanged.
  - Snapshot: each of the 92 resolved entries gains the two new fields
    at their defaults (`min_policy_updates_before_stop: 20`,
    `vec_env: "dummy"`); nothing else changes, and the 4 refusals are
    untouched.
- **PPO starting profiles, on-policy pre-flight and quick test (PPO
  plan Steps 3–5, 2026-10-17).** No env stream or SAC-family change:
  every SAC and DemoSAC snapshot entry is byte-identical, and the
  trainer digest against `f920d8f` matches on every job. The PPO job
  matches too, since its explicit settings override the new profile.
  - **Profiles (Step 3).** Every recipe's `algo_cfg["PPO"]` carries the
    design's starting profile (section 3), with clip_range 0.2,
    gae_lambda 0.95 and the recipe's own gamma:
    - BallBalance and BallBounce: 8 × 512 rollouts, minibatch 256,
      lr 3e-4, target_kl 0.03, log_std_init −1, ent_coef 0, reward
      normalization on.
    - WallBall family: 8 × 1,024 rollouts, otherwise the same.
    - PaddleTennis: 8 × 1,024, minibatch 256, 10 epochs, lr 3e-4,
      target_kl 0.02, gSDE resampled every 64 steps, log_std_init −2
      without orthogonal init, ent_coef 0, no reward normalization,
      CPU.
    - Humanoid Stages 0–2: 1 × 2,048, minibatch 256, 5 epochs, lr
      1e-4, target_kl 0.02, gSDE resampled every episode (150/300
      steps), log_std_init −2 without orthogonal init, ent_coef 0.001
      (was 0.01).
    - Humanoid smoke: the stage settings on 128-step rollouts.

    Cadences are whole rollouts at the profile's worker count and at 16
    (sb3_training's PPO count): `eval_freq` 24,576 (ball and humanoid)
    or 32,768 (WallBall, PaddleTennis), `checkpoint_freq` 245,760
    (98,304 on the humanoid stages, 4,096 on the smoke). No early stop
    can fire before about 20 updates. The degenerate-signal stop needs
    6 flat evaluations under PPO, not 5.
  - **Pre-flight (Step 4).** `train._validate_on_policy_config` runs
    in `train()` before any output and in `build_train_config`, and
    returns at once for the SAC family. It refuses the gate's SAC-only
    levers (update pause, replay wipe, entropy reset). It refuses
    non-integer `n_steps`/`batch_size`, `batch_size <= 1` with advantage
    normalization, `n_steps * n_envs <= 1`, and a minibatch that does
    not divide the rollout. It refuses `eval_freq // n_envs < n_steps`,
    a budget shorter than one rollout, evaluation or checkpoint
    cadences that are not whole rollouts, and `policy_kwargs` the
    policy class does not accept (new `algos.validate_policy_kwargs`).
    WallBallDepthCurriculum and its Aligned variant are refused under
    PPO for their gate levers. So is a hand-built
    `TrainConfig(algo="PPO")` left at the dataclass cadences
    (`eval_freq` 25,000 at 4 × 2,048): the message names the nearest
    whole-rollout value.
  - **Quick test (Step 5).** Unless the explicit `model_kwargs` set
    `n_steps`, a PPO quick test shrinks it to the largest power of two
    within min(current, budget // (4 · n_envs), eval_freq // n_envs).
    SB3's 2,048 becomes 512 at 8 workers and 256 at 16. The preset
    evaluation and checkpoint cadences then round down to whole
    rollouts. A PPO run's `stage_summary.txt` gains an "SB3 n_updates"
    line (SB3's count of optimization epochs), flagged and printed
    when it is 0. SAC summaries are unchanged.
  - **Starters.** Flat `eval_freq`/`checkpoint_freq` move into
    `[train.algo.sac]`, with `[train.algo.ppo]` counterparts (ball
    balance, paddle tennis, the humanoid stages and smoke). A run file
    copied from an older starter with a flat cadence is refused under
    PPO when the cadence is not a whole rollout; move it into
    `[train.algo.sac]`.
  - Snapshot: the 32 PPO entries are re-recorded; nothing else changes.
- **Per-algorithm config blocks (PPO plan Step 2, 2026-10-17).** No env
  stream changes. SAC and DemoSAC training paths are unchanged except
  an explicit SAC run of the humanoid stages (below); the trainer
  digest against `f920d8f` matches on every job.
  - `Recipe.algo_cfg[family]` holds `TrainConfig` fields for one
    algorithm family (`"SAC"` covers DemoSAC). Merge order:
    `extra_cfg`, then the run family's block (`model_kwargs` merged
    one level deep), then the run file, `quick_test`, and explicit
    arguments.
  - The PaddleTennis and WallBallBootstrap SAC exploration bundles move
    into `algo_cfg["SAC"]` (Bootstrap's family-neutral `gamma=0.995`
    stays shared). Every recipe now builds under `algo="PPO"`; both
    used to refuse it ("'target_entropy' is not accepted by PPO").
  - The humanoid stages' `{use_sde: True, ent_coef: 0.01}` bundle
    moves into `algo_cfg["PPO"]`. **Intended change:** an explicit
    `algo="SAC"` or `"DemoSAC"` run of Stages 0–2 now gets SB3's SAC
    defaults (auto entropy, no gSDE) instead of a fixed `ent_coef` of
    0.01 with gSDE. The stages default to PPO, whose resolution is
    unchanged, so no campaign path moves.
  - Run files gain `[train.algo.sac]` and `[train.algo.ppo]` tables,
    validated like `[train]` and merged after it only for runs of that
    family (DemoSAC reads `sac`). An unknown family table is refused;
    `RunFileConfig.train_algo` carries the tables. The WallBallBootstrap
    starter's `ent_coef` moves to `[train.algo.sac.model_kwargs]`.
  - `algos.DIVERGENT_SHARED_KWARGS` (`n_steps`, `batch_size`,
    `ent_coef`, `use_sde`, `sde_sample_freq`), which SAC and PPO both
    accept but read differently: a flat recipe or file `model_kwargs`
    key from the set is refused when the run's family is not the
    recipe default's. The hint names the family tables and says that
    an explicit `model_kwargs=` (exempt) replaces the whole bundle.
  - The recipe snapshot also records every recipe's PPO resolution
    (96 entries, was 70). Re-recorded: the 12 humanoid Stage 0–2 ×
    {SAC, DEMOSAC} entries, plus 26 new PPO entries. Every other entry
    is byte-identical.
  - The campaign notebook and the trainer digest's PaddleTennis job
    read the recipe's SAC bundle through `build_train_config` instead
    of `extra_cfg["model_kwargs"]`, so both work against the pinned
    package and this one.
- **Algorithm families (PPO plan Step 1, 2026-10-17).** No env stream
  or training path changes: the recipe snapshot is unchanged and the
  trainer digest against `f920d8f` matches on every job.
  - `algos.OFF_POLICY_ALGOS` and the new `ON_POLICY_ALGOS` are derived
    from the registered classes (`issubclass` on SB3's off- and
    on-policy bases) and equal the old hand-kept set. A registered
    algorithm that is neither refuses at import.
  - `algos.algo_family()` names each algorithm's config family after
    its most general registered ancestor: `"SAC"` for SAC and DemoSAC,
    `"PPO"` for PPO. The reward-normalization default, its twin for
    recorded configs and the warm-start family check now test families
    instead of the strings `"PPO"`/`"SAC"`.
  - One intended notebook fix: `sb3_training` re-applies the worker
    count only when `ALGO` switches family, and picks it by family. So
    `ALGO="DemoSAC"` keeps the SAC count (the recipe's own, 8 on every
    recipe but the 1-env humanoid smoke; 8 on the PPO-default humanoid
    stages) where it used to get PPO's 16. The notebooks use the
    derived `OFF_POLICY_ALGOS`, which the pinned `REPO_REF` exports
    too.
- **Refactor-gate leftovers (2026-10-17).** No env stream or training
  path changes: every existing golden entry reproduces unchanged, and
  the trainer digest against `f920d8f` matches on every job.
  - CI's typecheck passes again. Three errors (`Env.data`, an
    `np.full` shape that may be `None`, `BaseAlgorithm.log_ent_coef`)
    are narrowed with typed checks, and CI runs `python -m mypy`. A
    mypy without numpy, Gymnasium and SB3 beside it passes silently;
    README "Refactor gates" says how to run it as CI does.
  - Golden cases gain a reset mode. `"seed_once"` seeds the first
    reset only, as SB3's VecEnvs do in training, so the env RNG and
    PaddleTennis' serve alternation carry across episodes. New CI
    cases: the PaddleTennis recipe under random and oracle play over
    three episodes, and the WallBall TrueBaseline oracle. CI now runs
    22 cases in about 45 s.
  - The trainer digest also compares the best-model artifacts: the
    policy weights and `log_ent_coef` inside `best_model.zip`,
    `best_vec_normalize.pkl`'s stats, the meta's selection fields and
    bound artifact names (each side checks the bound sha256s against
    its files), which run-layout artifacts exist, and the checkpoint
    file names.
  - `digests.version_info()` records numpy's SIMD dispatch. The CI
    goldens match with AVX-512, or all SIMD dispatch, disabled (the
    env code calls no dispatch-sensitive numpy loop), and a test
    replays one case per env family with every dispatch target off.
  - `tests/test_packaging.py` checks that the installed dependencies,
    and the imported MuJoCo, satisfy the declared ranges ("mujoco
    3.16.0 violates the declared range ..."), and reads the CI
    workflow's install lines through quoting and flag variants.
- **Refactor-gate hardening (2026-10-16).** No env stream or training
  path changes: every recorded `stream_sha256` reproduces unchanged.
  Fixes from the gates' review (see DECISIONS):
  - The golden file moves to format `courtside-env-digests/2`. The info
    digest hashes every item, tuples and arrays included (they used to
    be skipped), and raises on a type it cannot hash. A new
    `interface_sha256` covers the spaces and the concrete obs, reward
    and flag types. Every PaddleTennis reward must equal the
    left-to-right sum of its nine `rew_*` terms, bit for bit.
  - Tests pin the summation order of both PaddleTennis reward sums on
    forced adversarial terms, and a third NaN witness keeps an
    order-sensitive reward in the CI streams.
  - New cases: `volley_rule="legal"` with the lead-charge oracle, and
    the k=2 drill in both launch arms from a library harvested
    in-process.
  - The digest and snapshot tools and the test suite refuse to run
    when `courtside_dynamics` imports from another checkout (a git
    worktree's editable install).
  - The recipe snapshot compares canonical JSON text, so `8` versus
    `8.0` is a diff. The trainer digest runs the two sides under
    different hash seeds.
- **Golden digests and the SAC freeze harness (2026-10-16).** No env
  stream or training path changes. Three refactor gates (see DECISIONS):
  - `courtside_dynamics.digests` plus `tools/golden_digest.py`
    (`--check`, `--write`, `--full`) check per-step env-stream and info
    digests against `tests/golden/env_digests.json`. CI runs the fast
    set (22 cases in about 45 s, with the cases added below) through
    `tests/test_golden_digests.py`, including the
    NaN-with-pending-escrow witnesses. The humanoid case has a declared
    MuJoCo 3.16 drift slot.
  - `tools/recipe_snapshot.py` freezes every recipe's resolved
    `TrainConfig` in `tests/golden/recipe_snapshot.json`.
  - `tools/trainer_digest.py --base <ref>` compares tiny seeded `train()`
    jobs (SAC, DemoSAC with demos off, the PaddleTennis SAC quick path,
    PPO) against a temporary worktree of `<ref>`.

  `courtside_dynamics.recipes` now imports SB3 and torch inside
  `build_train_config` instead of at module import, so the recipe
  registry and env factories load without the training stack. Every
  resolution is unchanged: the snapshot rebuilt from `4e2e019`'s source
  (before the import move) is identical.
- **Dependency hygiene (2026-10-16).** No code or env stream changes
  on the supported range, which is now exactly the range that
  reproduces the golden digests: `mujoco>=3.12,<3.16`. 3.16.0's
  `mj_setConst` refreshes body bounding volumes, which knocks the
  humanoid Stage-1 oracle off its calibration (138 steps, target
  missed), while the PaddleTennis, WallBall and BallBounce streams are
  identical from 3.12 through 3.16. Below 3.12 the streams move: 3.11
  changes the humanoid stream, and 3.10 every golden case but the
  WallBall TrueBaseline random stream (see DECISIONS).
  The `gymnasium` floor rises to 1.1.1, since 1.1.0 cannot construct
  any courtside env. The license is now the SPDX string `MIT` with no License
  classifier, built with `setuptools>=77`. CI's typecheck and test
  jobs install CPU torch instead of the CUDA build.
  `tests/test_packaging.py` pins the bounds and the CPU-torch order.
- **PaddleTennis instrument and bug-fix batch (2026-10-05).** Fixes
  the review findings in `docs/repo_review_20261005.md` §5b/§7.2 plus
  their adversarial follow-ups.
  - **Comparability.** Observations, rewards, `terminated` and
    `truncated` are bit-identical to the pre-batch env for the default
    and recipe kwargs, checked by per-step digests over random and
    oracle play. Saved policies and `VecNormalize` statistics stay
    valid. Eval *selection* and the info columns change, so learning
    curves and `best_model.zip` picks are not comparable across this
    batch.
  - **New info keys.** `episode_legal_hit_count_a`,
    `episode_valid_return_count_a`, `episode_rally_returns_a` (the
    policy's confirmed returns after its first in each point, i.e. the
    k≥2 conversions, counted from each point's launch),
    `contact_depth_a`, `episode_mean_contact_depth_a` (0.0 until the
    first hit), `episode_contact_depth_sum_a` and
    `episode_deep_returns_a`, plus a `deep_contact_depth` kwarg
    (default 5.5 m). The hit-weighted batch mean depth is
    `episode_contact_depth_sum_a_ep_mean /
    episode_legal_hit_count_a_ep_mean`.
  - **Reset.** `reset(seed=...)` now restarts serve alternation at
    side A, as the humanoid env does. `options={"serve_side": "a"|"b"}`
    forces the first serve. Probe tools pass the old schedule
    explicitly, so their outputs are unchanged.
  - **Rules labels (reward-neutral).** An untouched ball's second
    court contact is `second_bounce` even beyond the lines; before,
    that was `out_of_bounds`, which accounted for about 89% of oracle
    `out_of_bounds` endings. An un-crossed hit landing on the hitter's
    own side is `failed_to_cross`. Per-reason `term_*` flags are
    episode-scoped, and the non-finite guard reports `nonfinite_state`.
    A non-finite opponent action ends the episode as unsafe instead of
    raising. The serve-clearance nudge is clamped to the slide ranges
    and re-verified. The diagnostic court lines are painted inside the
    rules boundary.
  - **PaddleTennis recipe.**
    - Selection and success move from opponent-dominated `crossings`
      to `episode_rally_returns_a`, with per-key min-deltas sized to
      one conversion in 30 episodes (0.5/30) and 0.25 on reward.
    - The degenerate guard reads `episode_legal_hit_count_a_ep_mean`,
      and only the headline must be flat.
    - Evaluation is paired: every evaluation replays seeds
      `seed + 1_000_000 + i` with alternating serve sides, and the
      confirmation batch uses its own block, +100,000.
    - `reward_eval_episodes = 5`.
    - Training monitors log the episode counters.
    - An unseeded build warns and evaluates unpaired.
  - **Training infrastructure.**
    - New `TrainConfig` fields: `eval_seed`, `eval_reset_options`
      (opt-in pairing; `"none"` opts out), `degenerate_flat_keys`,
      `monitor_info_keywords` and `reuse_log_dir`.
    - `best_metric_min_delta` accepts a per-key mapping.
    - The final-info-eval stream stays fresh-random.
    - `config.json` records the observation-name fingerprint and the
      evaluation streams that actually run.
    - Warm starts and DemoSAC demo libraries are refused when their
      recorded observation names differ (legacy sources without a
      fingerprint load with a notice).
    - gSDE warm starts set `use_sde_at_warmup`, so warmup samples the
      pretrained policy rather than uniform-random actions.
    - Invalid eval reset options and misconfigured callbacks fail
      before any output is written.
    - A reused `log_dir` is refused unless `reuse_log_dir=True`.
    - The stage summary reports the selecting batch's metrics.
    - Checkpoint diagnosis skips a failing checkpoint and disables
      only after 3 consecutive failures.
  - **Notebooks and tools.**
    - The campaign scoring and resume paths verify the best-model and
      normalizer pair.
    - A resume re-scores only an attempt the scorer will accept.
    - The campaign fingerprint covers the gate settings.
    - There is one shared seed ledger (`tools/_seed_ledger.py`) that
      every probe consults.
    - Probes exit non-zero on a FAIL verdict.
    - `plot_learning_curve` marks the selected checkpoint.
  - **Dependencies.** The `gymnasium` floor is raised to 1.1.

- **PaddleTennis environment and recipe.** *(Partially superseded by
  the ground-rules bullet below: the default opponent, rally-rule
  profile, reference band, and certification described here are the
  volley era's, reproducible via `volley_rule="legal"`.)*
  `CourtsideDynamics/PaddleTennis` (`PaddleTennisEnv`): 1v1
  cooperative rally on the probe-frozen paddle court — half-length
  6.5 m, regulation 0.914 m net, singles width ±4.115 m, the wall-ball
  ball and paddles verbatim — with every task number frozen by the
  pre-committed probe battery before the env existed
  (`docs/paddle_tennis_probes_20260802.md`,
  `docs/paddle_tennis_probes_p3_p4_20260802.md`). The policy plays
  side A through the familiar 3-action target interface; side B is an
  injectable `opponent_controller` (default: the frozen `lead_charge`
  scripted controller) reading the exact side-B mirror of the
  48-value side-relative observation (24 physical + 12 rally-state +
  12 contact-memory values; probe P4 pinned the mirror bit-for-bit).
  One point per episode from the P3-measured serve band (origin
  3.25 m, 9 m/s, 21°; 100% legal, 100% returnable), serve side
  alternating every reset; +1 per rules-confirmed return by either
  side, −1 on the ending fault, −2 on unsafe/nonfinite physics. The
  `PaddleTennis` SAC recipe selects and succeeds on `crossings` (the
  P3 reference band: 3.15–3.42 for the scripted pair) and normalizes
  only the physical block; a `court_style` kwarg
  (`diagnostic`/`tennis`/`none`) mirrors the WallBall render-only
  presentation styles, with visibility lists derived from the
  compiled model. The frozen definition passed held-out
  certification (`tools/paddle_tennis_probes.py --certify`, reserved
  seeds 3100–3199: mean crossings 3.22 against the pre-registered
  2.6 floor, zero unsafe terminations;
  `docs/paddle_tennis_env_20260802.md`).
- **PaddleTennis n-point episodes (implemented, default off).** The
  committed follow-up from the shaping extension's verdict
  (`docs/design_paddle_tennis_npoint.md`): new env kwarg
  `points_per_episode` (default 1 — the frozen one-point task stays
  bit-identical until changed; `None`, spelled `"none"` in a TOML
  `[env]` table, fills the 1500-step cap with continuous play). A
  rally fault inside a multi-point episode pays its −1, claws back
  any pending escrow (the escrow's clawback boundary becomes the
  point), resets the rules machine, alternates the server, and
  relaunches the next feed with **paddle positions carried over** —
  the load-bearing decision that prices post-swing wander with the
  next point's already-sampled serve-return credit instead of the
  never-sampled k=2 return. The relaunch protocol clearance-checks
  the drawn serve cell against both paddle heads (0.45 m envelope,
  up to 10 redraws, then a minimal sanctioned nudge counted in
  `point_serve_nudged`) and re-primes the event sampler only after
  the ball teleport — the review's two reproduced hazards (a feed
  spawned into a parked paddle deflecting at 41 m/s with no events;
  a stale crossing detector faulting the new point as a reverse
  crossing). New info/CSV keys: `points_played` (completed points
  only), `completed_point_crossings`, `point_serve_nudged`, and
  seven cumulative `point_end_*` counters; the eight `term_*` group
  flags become strictly episode-ending descriptors. The diagnosis
  instrument segments episodes into per-point traces and adds the
  era's second recovery metric — **inter-point recovery travel**
  (point end to next feed arrival) — kept distinct from the
  within-point recovery hold. NP0 (bit-identity of the default,
  lockstep) and NP1 (30 mechanics witnesses: carryover continuity,
  nudge coincidence, spawn clearance/cell/velocity, the
  reverse-crossing hazard absent over 116 far-side slam points,
  exact escrow identities on bit-identical shaped/unshaped arms,
  statue point economics, strict alternation across points and
  episodes; racket–net contact measured **unreachable** on this
  scene, making the pinned cross-boundary latch semantics vacuously
  safe) all PASS (`tools/paddle_tennis_npoint_probe.py`, seeds
  5400–5499 now burned as the era's calibration block; NP0
  additionally verified cross-version — bit-identical
  obs/reward/info against the actual pre-amendment merge commit).
  NP2 recalibrated the era's reference band (100 episodes):
  crossings/episode 11.40 ± 1.58, completed points/episode 1.13,
  bridge metric 4.46 crossings per completed point, zero nudges
  over 113 relaunches, oracle inter-point recovery 2.10 m mean —
  and the L2 pilot's numeric bars plus its three-branch decision
  rule were frozen at NP2 time in the design doc §4a. NP3 held-out
  certification then **PASSED** every pre-registered floor
  (reserved seeds 4300–4399, now consumed: mean crossings 11.51 vs
  the 9.0 floor, 106 completed points vs 50, 0% nudges, zero
  unsafe). The recipe does NOT enable n-point play; adoption waits
  on the L2 pilot against the frozen bars. **L2 verdict
  (2026-08-15): Stop/pivot.** Two from-scratch pilots at the
  implementation SHA — `20260810_211754` (the pre-registered shape,
  seed 0, early-stopped 1.125M) and `20260815_015143` (seed 1,
  n_envs 8, full 2M) — both collapsed to zero policy contact from
  the first checkpoint (K FAIL every row; R2/P″/D2″ FAIL; M intact
  by the letter). n-point stays default-off; verdict, validation
  experiments (statue economics −4.4 vs oracle +11.6 in-mode; the
  from-scratch exploration cliff under carryover), and the
  recommended warm-started retest:
  `docs/paddle_tennis_npoint_pilot_20260815_review.md`. **The
  warm-started retest (L2W, run `20260815_180815`) then measured
  both halves** (review doc §6a): the transferred k=1 policy
  collects the design's inter-point credit at zero updates (+1.92
  reward, crossings 6.37 — the campaign's first positive learned
  eval), and SAC training under the same economics regresses it to
  a ~30%-engagement equilibrium with k=2 still exactly 0% — the
  stroke re-converges, positioning never improves. The design's
  mechanism is real; the next probed change is paying position
  (reach shaping), not more budget.
- **PaddleTennis escrowed reach shaping (implemented, default
  off).** The position→touch remedy the L2W warm-start verdict
  points at (`docs/design_paddle_tennis_reach_shaping.md`): new env
  kwargs `reach_shaping` (default 0.0 — the frozen task's reward
  stream is bit-identical until enabled) and `reach_shaping_radius`
  (default 3.0 m). When on: the incoming ball's live first bounce on
  side A pays `reach_shaping × max(0, 1 − d/radius)` into a pending
  escrow (d = recorded bounce position to the side-A paddle head),
  the side-A legal hit that takes the opportunity keeps it, and
  every ending path — n-point boundaries included — claws back the
  pending remainder, so episode totals are exactly the
  opportunity-taken proximity pay: camping is not farmable. New
  `rew_reach`/`rew_reach_clawback` components in the decomposition,
  info, and CSV schema. RS0 (default bit-identity, lockstep), RS1
  (six-witness identity battery on fresh block 5500–5599, now
  burned; `tools/paddle_tennis_reach_probe.py`), and RS2 (ending-path
  invariants, `TestReachShaping`) ship with the change; the LR1
  learning pilot — the single reward-side change against the L2W
  baseline, bars frozen in the design doc §4 — then ran
  (`20260816_135919`, 1M warm-started steps) and **the ADOPT branch
  fired** (design doc §4a): k=2 exchange survival above zero at five
  checkpoints (a confirmed second-exchange return at 700k — the
  campaign's first), touch 41%, ready error to 1.08 m, k=1 receiving
  68–83% across the last three rows, and the first positive learned
  final evaluation (+0.14; the from-scratch era's floor was −4.4).
  The recipe change (adopting n-point + both shapings + the hardened
  guard set) and the registered-run pre-registration on reserved
  4100–4199 were the committed follow-ups — **both now shipped**:
  the `PaddleTennis` recipe defaults to `points_per_episode=None`,
  `contact_shaping=0.25`, `reach_shaping=0.25`, success/guards on
  the side-A-only `legal_hit_count_a`, `best_metric_min_delta=0.25`,
  `confirm_best_eval`, and the 5-eval degenerate stop (the three
  recipe-pin tests flip to adoption pins; the ENV defaults are
  untouched, so direct construction and every lockstep identity stay
  frozen). The registered run — 3M warm-started steps, seed 1, bars
  frozen from the LR1 band, held-out gate on 4100–4199 — is
  pre-registered in
  `docs/paddle_tennis_registered_run_prereg_20260816.md` with its
  run config frozen verbatim in that document (Drive-side at launch,
  sha-pinned by run provenance, per the pilots' convention). The
  adoption code review then landed the reach escrow's **ordering
  amendment** (design doc §2a): a payment whose bounce and taking
  hit share one control step is kept immediately instead of escrowed
  — the original ordering could claw back a tight interception's pay
  — plus probe-tool hardening (NP2 band no longer crashes on zero
  completed points; the statue/camper witnesses consolidate into
  `envs/_paddle_court.py` so tools and tests measure one frozen
  policy; dead NP1 bookkeeping dropped). Both pilots ran the
  pre-amendment ordering; the registered run runs the amendment.
  The pre-registration also gains **LS1**, a frozen from-scratch
  gate pilot (§1a): the adopted recipe with no warm start is now a
  live from-scratch entry point (the reach escrow pays a positional
  gradient before any touch — the structure whose absence killed the
  L2 from-scratch pilots), and LS1's verdict decides whether the
  registered run launches warm-started (§2 as frozen) or from
  scratch under its own budget addendum.
- **PaddleTennis staged campaign notebook (tooling).**
  `notebooks/paddle_tennis_campaign.ipynb` runs the whole
  from-scratch campaign in one execution — a gate leg (the LS1
  shape) trained from scratch and scored in-process against the
  frozen §1a bars by the same diagnosis instrument the run's own
  checkpoint probes use, then a main leg warm-started from whichever
  lineage the verdict selects (PASS: the gate leg's own best, +2M —
  the 1M spent counts toward the 3M total; FAIL: the frozen
  `20260809_211147` reference at its full 3M) — with every leg,
  verdict, and branch decision recorded in a Drive-side
  `campaign_manifest.json`. Re-running all cells resumes: completed
  legs are skipped, crashed attempts stay in place as evidence
  (numbered `attempt_NN` dirs), the recorded branch decision is
  reused rather than re-decided, and a settings fingerprint refuses
  to resume a campaign whose frozen budgets/bars changed (the
  "changes after start void the run" doctrine, enforced in code).
  New unit-tested `notebook_utils` helpers carry the machinery:
  `paddle_campaign_metrics` (trace aggregation mirroring the
  instrument's `report()`), `score_campaign_bars` (three-valued
  PASS/MIDDLE/FAIL where NO_DATA never passes and a reading exactly
  on the fail line routes to MIDDLE — prereg edge ambiguity goes to
  the maintainer, not an automated branch), `score_paddle_stage`
  (best-checkpoint reading of a finished leg against a bar table,
  the env rebuilt from the run's own recorded constructor kwargs,
  report written under `reports/`), `resolve_warm_start_branch`, and
  the manifest/attempt-dir functions. Warm-started legs merge
  `learning_starts=25_000` into the recipe's calibrated SAC bundle
  instead of replacing it. The generic single-run driver stays
  `sb3_training.ipynb`, and a registered attempt still freezes its
  plan in a pre-registration doc first — the notebook executes a
  frozen protocol, it does not invent one.
- **PaddleTennis escrowed post-swing hold (implemented, default
  off).** The k=2 remedy the registered run's stop/amend verdict
  routes to (`docs/design_paddle_tennis_postswing_hold.md`): new env
  kwargs `hold_shaping` (default 0.0 — the frozen task's reward
  stream is bit-identical until enabled) and `hold_shaping_travel`
  (default 4.0 m). When on: a side-A legal hit arms a travel-metered
  window; the opponent's legal return strike pays `hold_shaping ×
  max(0, 1 − travel/budget)` into a pending escrow (travel = the
  side-A paddle head's accumulated XY path since the hit,
  follow-through included by design); the **next** side-A legal hit —
  by construction the k=2 hit — keeps the advance; and every ending
  path (episode endings, truncation, n-point boundaries, the
  non-finite guards) claws back the pending remainder, with windows
  disarmed at boundaries before the relaunch teleport. Hit-then-
  freeze farming nets exactly zero (collect, never keep — witnessed
  in the battery); serve returns against a non-hitter pay nothing
  (the escrow is strictly post-swing). New
  `rew_hold`/`rew_hold_clawback` components in the decomposition,
  info, and CSV schema. Motivation is the registered run's measured
  blocker: k=2 pinned at 1% across 3M steps while k=1 receiving
  reached 95% and post-swing travel sat at 7.5 m — on the k=2 hit the
  ladder now pays hold + reach + contact + the confirmed +1, every
  segment of the second exchange carrying its own farm-proof
  gradient. `TestHoldShaping` (11 tests) and the PH1 witness probe
  (`tools/paddle_tennis_hold_probe.py`, fresh block 6200–6299) ship
  with the change; the LH1 pilot — warm-started from the registered
  run's 2.4M best, bars frozen from its band — is pre-registered in
  the design doc §4. The pilot series closed 2026-08-22 without
  adoption (design doc §4a–§4c): at a 4.0 m travel budget the ramp
  sat entirely outside the policy's 6–10 m post-swing band and paid
  exactly zero (LH1b trained bit-identically to LH1 — the payment
  cliff, booked as a design error), and at the amended (0.5, 12.0)
  the gradient was live but the policy kept the wander — hold travel
  settled at 7.5–8.7 m and k=2 stayed at its 1% floor across 3M —
  while, as an unregistered-extension observation, setting the
  warm-started line's eval record (+2.483 at 2.425M with the first
  deep-training `best_model`; MuJoCo 3.11→3.12 confound noted). The
  kwargs stay default-off; the k=2 probe moves diagnosis-side.
- **PaddleTennis escrowed contact shaping (implemented, default
  off).** The touch→in remedy
  (docs/design_paddle_tennis_contact_shaping.md): new env kwarg
  `contact_shaping` (default 0.0 — the frozen task's reward stream
  is bit-identical until enabled). When on: a legal side-A hit pays
  the shaping immediately, the advance is kept when that return
  confirms and clawed back on **every** episode ending (termination,
  truncation, both nonfinite guards), so episode totals are exactly
  `shaping × side-A confirms` — farm-proof — while contact-time
  Q-values separate "hit then miss" from "never hit". New
  `rew_shaping`/`rew_shaping_clawback` components in the
  decomposition, info, and CSV schema; `scripted_hard_slam_witness`
  joins the scripted controllers. S1 (seeds 5300–5399, now burned)
  and S2 both PASS with exact identities. The L1 pilot (run
  `20260809_161704`, 1M steps, shaping via `[env]` TOML) landed in
  the rule's declared middle: real strokes emerged (receiving k=1
  93%, 8 landed-in shots per diagnosis row from 500k, touch peak
  46% vs the exploration pilot's 17%, crossings 1.23 vs 0.67) but
  short of the 55%-touch and 2.5-crossings bars, with k=2 survival
  still 0% (the policy wanders post-swing instead of re-readying —
  the next measured bottleneck). The frozen lean's single 2M
  extension (run `20260809_211147`) then nearly completed k=1:
  receiving survival 100%, serving 53%, policy shots 78% in at
  oracle depth, best crossings 1.77 and still climbing at the cap —
  but k=2 stayed exactly 0% in all twenty rows (recovery wander
  8.3 m), so the committed branch fired: the next probed change is
  n-point episodes, with between-point position carryover as its
  load-bearing design decision. The recipe still does NOT enable
  shaping. The notebook's replay cell
  now applies `court_style="tennis"` to every court that supports
  it (previously WallBall-only), so PaddleTennis replay footage
  renders on the mini tennis court.
- **PaddleTennis exploration package (training change).** The
  ground-era diagnosis's ranked remedy, shipped recipe-level:
  `model_kwargs` gains `use_sde=True`, `ent_coef="auto_0.02"`,
  `target_entropy=-1.5`, and `train_freq=(64, "step")`
  (docs/paddle_tennis_exploration_20260808.md). The raised target is
  the mechanism fix — the stock run's 5e-5 coefficient was SB3's
  tuner resting at the too-low default target (−3.0), so the target
  is what moves, not the coefficient — and the multi-step
  `train_freq` is what makes gSDE real under SAC at all (the
  off-policy collector resets the noise matrix every rollout;
  at `train_freq=1` that is every step — iid noise). iid per-step
  noise cannot produce coherent ball-reaching runs (83/100 points
  ended `policy_never_reached`). Task definition, reference band,
  and certification unchanged; learning curves start a new
  training-configuration regime. Pilot criteria pre-registered in
  the doc before the pilot ran; the pilot (run `20260809_005951`,
  1M steps) confirmed the mechanism (entropy sustained; the stock
  run's serving-side zero and macro memorization both gone) but
  acquired no stroke — every policy shot from 300k on landed
  9.2–16.0 m deep (2.7–9.5 m past the 6.5 m baseline) and
  engagement oscillated without an attractor
  (crossings ≤ 0.67 vs the ≥ 2.5 bar). The package stays; the
  evidence points the next probed change at the touch→in credit
  gap (doc §3).
- **Checkpoint behavioral diagnosis.** The exchange/positioning
  instrument that diagnosed the ground-era pilot (one memorized
  serve-return macro, no general ball-reaching;
  docs/paddle_tennis_diagnosis_20260808.md) now lives in the package
  (`training/paddle_diagnosis.py`; `tools/paddle_tennis_diagnosis_probe.py`
  stays as the CLI) and runs automatically at every checkpoint save
  via `TrainConfig.checkpoint_diagnosis` (enabled in the
  `PaddleTennis` recipe): per-checkpoint reports plus a cached oracle
  reference row under `reports/diagnosis/`, exception-isolated so a
  probe failure can never kill a run.
- **PaddleTennis ground rules (behavior change).** The first learned
  GPU run maximized cooperative return rate with a close-net volley
  loop (a net crossing every ~14 control steps, crossings 37.6 and
  climbing at the 2M budget). Pre-bounce returns are now
  `VOLLEY_RETURN` faults via `RallyRules.require_bounce_before_return`
  (off for wall-ball/humanoid consumers) and the new `PaddleTennisEnv`
  kwarg `volley_rule` — **default `"fault"`**, with `"legal"`
  reproducing the superseded volley era exactly. The scripted
  reference is the recalibrated ground oracle (run-up wait behind the
  predicted landing, soft `GROUND_SWING` stroke, hold-low recovery;
  the frozen P1 port's returns were measured to be largely volleys),
  giving a new reference band of 7.78 crossings at a 126-step cadence
  (probed: both volley witnesses collapse to 100% volley faults;
  ground play is rule-neutral; a volley fault confirms nothing, so
  touching a doomed ball earns nothing) and a new held-out
  certification (PASS 7.68 on reserved block 4200–4299, floors
  5.9/0.90 pre-registered from the pre-fix calibration band). `term_volley`
  joins the env's terminal flags and the recipe's eval keys.
  Volley-era artifacts (run `20260803_004559`, the local pilot, the
  3100-block certification at 3.22) are not comparable with
  ground-era numbers; docs/paddle_tennis_ground_rules_20260803.md is
  the amendment record.
- **P5 transfer instrument.** *(Recalibrated under ground rules —
  see the ground-rules snapshot §6; the numbers below are the
  volley-era stub calibration, and the baseline row now runs the
  ground oracle.)* `tools/paddle_tennis_p5_transfer.py`
  renders the paddle court as the wall-ball world (23-value
  observation, true-baseline action mapping) so the wall-ball
  champions can play `PaddleTennisEnv` unmodified. Scripted
  calibration (`docs/paddle_tennis_p5_transfer_20260802.md`): the
  `scaled` identification plus a serve-yield overlay is the only
  viable configuration; champion measurements and the pre-registered
  pool-admission rule run on Colab.
- **BallBalance nonfinite guard.** BallBalance now carries the same
  guard as every sibling env: actions are shape/finiteness-validated,
  the physics state is checked before stepping, and a nonfinite
  action/state ends the episode with reward 0 on the echoed last
  finite observation. Previously a NaN action reached MuJoCo, which
  warns and silently resets the simulation mid-episode — so on
  solver-blow-up seeds, episode lengths and termination counts differ
  from a stock 0.25.0 tree. Healthy episodes are bit-identical.
- **WallBall court-style rendering fix.** The 0.25.0 coordinate ticks
  (`court_tick_xm5..xm8`) escaped the hand-maintained style-visibility
  list and stayed visible in `court_style="tennis"`/`"none"` footage;
  the visibility lists are now derived from the compiled model, so the
  ticks hide correctly. Render-only (sites cannot collide); pinned by
  the cross-style observation-equality tests.
- **Run-summary stop reasons.** `stage_summary.txt` gains a
  `Stop reason` line naming which guard ended a stopped-early run
  (early-stop patience, degenerate signal, or stage budget). Artifact
  addition only; existing readers scan line-prefixes and are
  unaffected.
- **LD1′ launch blockers fixed (DemoSAC; 2026-09-27 review).** Four
  defects that blocked the demonstration-injection pilot
  (`docs/design_paddle_tennis_demo_injection.md` §5). Default
  (non-DemoSAC) training is bit-identical; no verdict in any
  pre-registered doc changes.
  - *`config.json` recorded `demo_transitions: 0` on every DemoSAC
    run.* The demo buffer builds lazily at the first `learn()`, but
    `train()` wrote `resolved_model` before that. `train()` now
    builds a DemoSAC model's demo buffer just before recording the
    model (a no-op with the surface off), so the recorded count is
    the real train-split count and is on disk before the first
    gradient step. Inference loaders still never need the library.
  - *SB3 floor.* `stable-baselines3>=2.5` → `>=2.7`: DemoSAC needs
    the off-policy `n_steps` attribute and
    `ReplayBufferSamples.discounts`, both new in 2.7.0.
    `DemoSAC.train()` mirrors SB3 2.9.0's `SAC.train`, which is
    unchanged since 2.7.0. The suite passes against 2.7.0,
    including the SD0 lockstep.
  - *The campaign notebook could not launch DemoSAC* (run-config
    TOMLs reject `algo` by design). `paddle_tennis_campaign.ipynb`
    gains a plan-level `ALGO` knob (default `None` = the recipe's
    SAC), passed to `build_train_config(algo=...)`, and per-leg
    `LEG1_MODEL_KWARGS` / `LEG2_MODEL_KWARGS`, merged over the bundle
    the leg already carries. The settings cell refuses a typo'd or
    cross-algorithm key, a `seed`, and a warm-started leg's
    `learning_starts` before the manifest is written. Each leg's
    frozen plan now pins `algo`, the leg's model kwargs and the demo
    library's digest. `validate_run_config_against_plan` gains
    `algo` (case-insensitive, like the registry) and `model_kwargs`
    (subset match) keys. The new knobs join the campaign fingerprint
    only when set, so a default-plan campaign created before them
    still resumes. `DemoSAC` gains a `demo_library_sha256`
    constructor pin (full digest or a lowercase-hex prefix of at
    least 8 chars), checked at construction, so a moved or
    re-harvested library aborts the launch instead of voiding the
    run afterwards. The whole `demo_*` surface is now suppliable
    through `model_kwargs`.
  - *`train()` leaked its SB3 logger.* The `progress.csv` and
    TensorBoard handles are now closed in `train()`'s cleanup path
    on every exit (they previously stayed open across notebook legs
    and showed up as `ResourceWarning`s under pytest). A caller that
    trains the returned model further must `set_logger` a fresh
    logger first.
- **Pipeline robustness closures (review 2026-08-28, after-LT1
  class).** Robustness only — no eval, reward, or selection semantics
  change for a correctly configured run; every shipped recipe passes
  the new start-of-run checks, and a seeded PaddleTennis run
  (4k steps, 2 evals) reproduces `main` byte-for-byte in
  `eval_info.csv` and `evaluations.npz`, with identical best-model
  selection and final/best policy weights.
  - *Miskeyed selection keys fail at start (§2.1).* `train()` probes
    the eval env's `info` schema on a throwaway instance (one reset +
    one step) and refuses a `headline_key`, `success_key`,
    `best_metric_keys` entry, or armed `degenerate_guard_keys` entry
    that no evaluation can emit, with a close-match hint. Previously a
    typo scored `-inf` at every evaluation and selection silently fell
    through to reward. `InfoDictEvalCallback` also prints a line on
    every evaluation missing a selection/guard key (hand-wired
    callers); scoring is unchanged. **Migration:** a config whose
    headline or success key its env never emits used to train on
    degraded selection; it now raises before training.
  - *`eval_info.csv` append is exception-isolated (§2.2)*: an
    `OSError` logs and continues, like the `evaluations.npz` writer
    (cardinal rule 7).
  - *Crash salvage (§2.13, §3).* Any non-`KeyboardInterrupt`
    exception out of `learn()` now saves `final_model.zip`,
    `vec_normalize.pkl`, the gate's stage history, and a
    `stage_summary.txt` with `Status: crashed` (the exception as the
    stop reason; closing eval skipped) before re-raising. Each salvage
    step is isolated and never masks the original error. The
    interrupt and crash paths are now integration-tested.
  - *`TrainConfig.require_device` (§2.3).* Opt-in `"cuda"` fails in
    milliseconds — before any env or run directory — when
    `torch.cuda.is_available()` is false or `model_kwargs["device"]`
    contradicts it, and asserts the built model's device. Recorded in
    `config.json`'s `train_config`. The campaign notebook sets
    `REQUIRE_DEVICE = "cuda"` (not fingerprinted, so existing
    campaigns still resume).
  - *Loud provenance (§2.5).* The `config.json` updaters and the
    summary's `best_model_meta.json` read print an `[artifacts]` line
    with the path and exception instead of returning silently; a read
    failure that would drop pinned provenance
    (`resolved_model.demo_library_sha256`, or the warm-start
    `initialization` block that the plan's `expected_artifact_sha256`
    and `transfer_log_ent_coef` checks read) raises at run start.
  - *No silent normalizer fallback (§2.6).* Best-model replay raises
    when a normalizer was expected — a snapshot that fails to load, or
    none on disk for a run whose `config.json` records
    `normalize_obs=True` — instead of replaying on raw observations.
  - *Also:* `train()` calls `validate_model_kwargs` itself, so a direct
    `TrainConfig` fails before any setup; `tools/smoke_wheel.py`'s
    env-count floor is 5, pinned to the live registry by a test.

## 0.25.0

The true-baseline extension
(`docs/design_wall_ball_true_baseline.md`). The paddle workspace, serve
energy, and in-play volume extend to the ITF baseline — **verified
bit-for-bit neutral for every pre-existing configuration**, so all
saved policies, `VecNormalize` statistics, and learning curves remain
comparable with 0.22.0–0.24.0. Every geometry and mechanics number was
measured by scripted probes before the change landed, and the frozen
era task passed a held-out certification (seeds 4000–4099) before its
first run.

- **`wall_ball.xml` workspace extension.** `paddle_slide_x` joint
  range and `paddle_target_x` ctrlrange widen (−3, 2) → (−6.5, 2),
  taking the physical world-space workspace from (−4.7, 0.3) to
  (−8.2, 0.3). The bold baseline marker moves to −8.2 (just behind
  the ITF baseline the tennis style draws at −7.985) and gains −5..−8
  ticks (render-only sites). **The default x action mapping does NOT
  follow**: it is now frozen at the historical workspace via
  `_DEFAULT_PADDLE_X_TARGET_RANGE = (-4.7, 0.3)` — previously it was
  derived from the XML ctrlrange, and widening the XML would have
  silently rescaled action semantics for every config without an
  explicit `paddle_x_target_range`. The extended workspace is strictly
  opt-in. Verified by hashing full obs/reward/termination streams,
  HEAD vs extended tree, on four legacy configs: all identical.
- **`WallBallEnv` gains `ball_in_play_min_x`** (default −6.0 =
  bit-for-bit historical behavior; validated finite and negative).
  The deep out-of-bounds edge is task geometry: an 11 m/s serve dies
  OOB at −6.0 before a paddle at −7.9 can touch it (measured 60/60).
  Only the new recipe widens it (−10.0).
- **`ladder_certification` gains `feasibility_ge2_floor`** (spec knob,
  default the stock 0.90, validated in (0, 1]; the resolved floor is
  recorded in every report's provenance). No scripted reference
  dominates the 11.8 m task (calibrated oracle band: 1.98 mean, 67%
  ≥2 rate), so the true-baseline recipe declares 0.50 — the band
  minus two 30-episode sampling standard deviations. Every other
  recipe keeps the stock floor.
- **`WallBallTrueBaseline` recipe + packaged starter**
  (`run_configs/wall_ball_true_baseline.toml`): the goal-rally
  structure (direct task, single-stage informational gate at the
  campaign's 3.0 mark, no curriculum) at the era task — fence
  (−8.2, −2.6), start −7.9, home −5.4 (fence midpoint), serve 11.0
  from origin 1.0, mapping (−8.2, 0.3), in-play bound −10.0,
  patience 60 (the replication's record seed banked its best at 3.2M
  under patience 60). Certification probe `lead_charge` 2.6 (gap
  sweep winner), 30 episodes, seeds 30000+. Held-out certification
  passes all blocking criteria: oracle 1.81 mean / 61% ≥2 (Wilson CI
  [0.51, 0.70]), crude 9% > 0, monotone references, clean landing
  probe. `WallBallGoalRally` and every other recipe are untouched.

## 0.24.0

The campaign diagnosis release
(`docs/wall_ball_rally_diagnosis_20260728_review.md`). No physics,
observation, action, or reward changes: **saved policies and
`VecNormalize` statistics remain comparable with 0.22.0/0.23.0.** The
existing recipes are untouched; a new recipe replaces the depth ladder
as the campaign's production configuration.

**`WallBallGoalRally` — the structural replacement for the
sliding-fence depth ladder is no curriculum at all.** Three ladder
generations stalled below a flat 3.0 promotion bar that no scripted
reference reaches at any rung (~70 controller configurations, best
2.7) and that four multi-million-step runs plateaued 0.2–0.9 under
everywhere past stage 0, while each promotion bought a measured
+0.35 m serve-landing jump and (in 0.22.0) an advance package whose
one exercised instance cost ~4.3M of 6M steps. Meanwhile the
constant-width 0.22.0 fence work had already made the goal task itself
learnable (crude placement-blind two-return rate 80–84% vs 9.5–16% on
the retired fence) — and no run ever trained there, because the
ladder was in the way. The pre-registered paired local SAC structure
A/B/C (direct vs aligned-serve vs serve-origin mixture, common seeds;
review doc §4) settled the rest: direct goal-task training scores
2.03/2.71 completed returns on 100 held-out seeds at 500k local steps
— versus 1.14 all-time from any 6M ladder run — with the mixture no
better (2.27) and aligned-only worse (1.09–1.51). The recipe
therefore trains the depth ladder's final stage AS the whole task
(fence (−4.7, −2.6), start −3.9, serve origin 1.0, speed 7.0;
evaluation equals training equals the historical goal task, so every
prior goal number stays comparable). A single-stage gate that can
never promote keeps stage stamping, `curriculum_stages.json`, and
startup certification (training geometry certified held-out twice:
96% / 91% oracle two-return rates on seed blocks 3000-3099 /
3100-3199); its informational 3.0 bar makes
`curriculum/gate_window_mean` crossing 3.0 the campaign-goal marker.

**`stage_eval_budget` promotion-staleness guard** on
`PerformanceGatedEnvStagesCallback` (and the gate spec): a non-final
stage that records N evaluations without promoting either stops the
run (`"stop"`, default — run `20260727_233859` burned ~4.5M steps in
exactly that state with nothing able to notice) or force-advances
(`"advance"`, recorded as `promoted: false` with
`advance_reason: "stage_eval_budget"` in `curriculum_stages.json`, so
an unearned rung is auditable). History rows gain `advance_reason` and
`stage_eval_budget_exhausted` fields.

**Startup-certification probes recalibrated for the constant-width
geometry.** New `lead_charge` oracle probe mode (charge-gap trigger +
ballistic y/z lead at the intercept while charging): the stock
y/z-tracking charge failed the 90% feasibility floor at stages 0 and 3
of the 0.22.0 ladder (36%/70% two-return rates). Frozen per-stage gaps
(3.0/0.8/1.0/1.2/3.0, chosen at n=200 on calibration seeds 0-199 by
rate-with-margin rather than raw argmax) certify held-out at
95–100% per stage (seeds 3000-3099) — `tools/depth_stage_sweep.py
--ladder release` and every gated run's startup certification now pass
their blocking criteria. Startup episodes 10 → 30 (the 0.90 floor sits
close to the probes' true rates; 10-episode verdicts flip on one
seed). The remaining advisory warnings on the depth ladder — no
scripted reference reaches its 3.0 bar; every adjacent stage jumps the
serve landing by ~0.35 m — are true findings about that ladder.

**`wall_ball_oracle_action` mapping parameterized.** It hardcoded the
retired fixed −1.7 pivot and old spans, silently scoring ~0.5 bounces
on any re-pivoted geometry (review 20260727_233859). New
`paddle_home_x` / `paddle_x_target_range` kwargs derive the inversion;
defaults preserve the legacy `WallBall` recipe contract byte-for-byte.

## 0.23.0

Startup ladder certification. No physics, observation, action, reward,
or recipe-geometry changes: **learning curves and saved policies remain
comparable with 0.22.0.**

Review of run `20260727_233859` (the first 0.22.0 run;
`docs/wall_ball_depth_curriculum_20260727_233859_review.md`) found the
0.22.0 ladder had shipped without a calibration sweep:
`tools/depth_stage_sweep.py` still hardcoded the retired 0.21 stage
tables and fixed pivot, nothing forced it to track the recipe, and the
mismatch surfaced only after a 17-hour run stalled at stage 1. The
sweep is no longer ad hoc:

**Training runs self-certify.** A new `TrainConfig.ladder_certification`
spec (set by both depth recipes) makes `train()` sweep every
`performance_gate` stage with the scripted reference cells — parked /
crude / calibrated oracle plus the policy-independent serve-landing
probe — *before* the worker fleet is built, on fresh `env_fn()`
instances with each stage applied the way the gate applies it. The
verdict lands in `reports/ladder_certification.json` (new run-layout
artifact, recorded in `config.json`) and is printed at startup.
Certification is derived entirely from the live gate spec and env
factory, so there is no second stage table to go stale.

**Stage-application integrity is a blocking criterion.** Every stage
key must already exist on the constructed env and must read back with
the written value — the exact class of the pre-0.22.0 `paddle_home_x`
silent no-op, now caught in seconds instead of by post-mortem.
Certification also warns when no scripted reference reaches the
promotion bar and when the serve-landing offset jumps more than 0.25 m
between adjacent stages (the measured stage-0→1 receive discontinuity).

**Advisory by default.** The stock probes are per-index calibrations
from the 0.21 geometry; with them the current ladder is *known* to fail
feasibility at stages 0 and 3 and the inversion detector at stage 1, so
failures print loudly and are recorded but do not abort unless the spec
sets `enforce`. Certification probes draw from a reserved seed block
(30000+), disjoint from every burned experiment range.

**`tools/depth_stage_sweep.py --ladder release` (new default)** carries
no stage table: it certifies the `WallBallDepthCurriculum` recipe's
live gate stages through the same package certifier. The
`aligned`/`baseline` tables remain pinned to the 0.21-era serve-origin
candidates for reproducing the alignment campaign's paired comparisons.
Also fixed: the recipe's usable-action-share comment now carries the
measured values (0.49/0.43/0.42/0.47/0.63); the previous figures
(0.52/0.48/0.49/0.54/0.66) matched neither the code nor the changelog.

## 0.22.0

Four changes to both depth recipes, all aimed at defects the
`20260727_004014` post-mortem measured directly. That run promoted three
times (a first — the previous best reached stage 1) but plateaued on the
goal task at **1.14 completed returns**: its 50-episode long-horizon
audit landed the serve in 100% of episodes, a second ball in 10%, a
third in 2%, and ended every single episode in a double bounce.
**Neither learning curves nor saved policies are comparable with any
earlier depth run.**

**Constant 2.1 m fence, ≥1.3 m of runway at every stage.** Return pace is
set by the paddle's forward speed at contact, and that speed needs
runway. A swept probe on the old goal fence scored 0 completed returns
from 0.0–0.4 m of forward travel, 1 from 0.6–0.9 m, and 2–3 from
1.2–1.6 m. The old ladder narrowed the fence 3.0 → 1.7 m as it receded
and left exactly 0.9 m at the goal — enough for one good shot and no
re-load, which is precisely the 100%-then-fail signature the audit
found. Fences now hold a constant 2.1 m and every stage clears 1.3 m.
2.1 m is the widest constant width that keeps the ladder free of an
all-stage refuge: the fence travels 2.4 m back (−2.3 → −4.7), so any
width ≥ 2.4 would hand every stage a shared front-court interval again —
the flaw run `20260722_124613` exposed. The 0.3 m separation matches the
ladder it replaces, and `paddle_start_x` is unchanged at every stage.

**`paddle_home_x` is a live property, re-centred per stage.** It is the
pivot of the normalized x action map, and it was a plain attribute — so a
stage that moved it was a silent no-op (`_control_home` is derived once
during setup, and `set_wrapper_attr(..., force=False)` reports success
for any existing attribute). With the pivot pinned at −1.7 against the
old goal fence, 71.7% of the x action range — all of action 0 and the
entire positive half — clamped onto the fence's front edge. Each stage
now sets the pivot to its fence midpoint, holding the usable action
share roughly flat (0.49, 0.43, 0.42, 0.48, 0.63) instead of collapsing
0.67 → 0.28. `paddle_x_target_range` is untouched, so action *scale*
still never drifts across stages.

**`return_shaping_scale` = 0.15 (new env kwarg, default 0.0).** The
outgoing leg had no dense reward at all: `accept_paddle_hit` closes the
tracking window and only wall contact reopens it, so "how hard did I hit
it" rode on one sparse +1 about 50 steps later. Measured consequence — a
1.5 m/s pop-up (which the fixed 10° face pitch sends mostly *upward*)
paid exactly what a 17 m/s drive paid, and 6 of 63 legal hits never
reached the wall. The new term is potential-based on the ball's
remaining gap to the wall and joins `_pending_shaping`, so it is
refunded unless the return actually completes. Deliberately far below
`track_shaping_scale` (0.5): the incoming term already crowds out the
flat +1 wall payment (57% → 73% of per-cycle reward as the fence
recedes), and 0.15 adds ~1.0 per completed return rather than dominating.

**`reset_entropy_on_advance` = True.** The run finished with
`train/ent_coef` at 0.0011 — effectively deterministic — after sitting on
stage 3 for 97 evaluations. Every advance had been dropping a policy
with no exploration budget into new geometry. The lever was added in
0.20.0 for exactly this failure and had never been exercised; it is
legal here because `model_kwargs` pins only `gamma`, leaving SB3's
`ent_coef` on `"auto"`.

Not changed: `serve_speed` still co-moves with depth (5.2 → 7.0), so
depth and ball speed remain confounded. `wall_reward_increment` stays at
0.0. Both are the next levers if the goal-task number stays flat.

## 0.21.0

Two changes to both depth recipes, aimed at the two defects the
2026-07-26 measurement pass identified. **Neither learning curves nor
saved policies are comparable with any earlier depth run.**

`gamma` 0.99 → 0.995, pinned in `model_kwargs`. Entropy stays on SB3
auto — this is a one-lever change. The exchange cadence in this task
measures 117–135 env steps on the calibrated oracle, so the SB3 default
prices the next completed return at `0.99^130 ≈ 0.27` of the one already
banked and the third at `≈ 0.07`. A policy that stops trying after
exchange two is behaving optimally under that discount, which is a
mechanical candidate for the ~2.0–2.4 plateau every depth run has hit.
The only run in this project that ever produced a long rally — WallBall
`20260713_192636`, `bounce_count_ep_mean` **12.30**, 13 exchanges with
zero floor bounces, every episode timing out rather than dying — used
gamma 0.995. That run appears nowhere in this changelog or in
`docs/`, and `lessons_learned.md`'s "five healthy runs bracket the same
~3.2–3.4 ceiling … the binding constraint is capability" omits it.
gamma was tested once before, in 0.13.0, bundled with the falsified
weak-return retry and reverted with it, so it has never been evaluated
alone.

`n_eval_episodes` 30 → 60 on the matched/selection stream, with
`final_eval_episodes` pinned at 30 and `best_metric_min_delta`
0.5/30 → 0.5/60. Promotion is a threshold crossing on a noisy statistic:
per-episode `bounce_count` sd is ~0.87, so a 30-episode batch has sd
~0.16 and a 3-eval window ~0.092 — and runs `20260724_152530` /
`20260725_171747` cleared the 3.0 bar at stage 1 by **0.011**, about a
tenth of a standard error. 60 episodes cut the window sd to ~0.065.

The threshold itself is deliberately unchanged. An earlier reading of
this evidence proposed lowering it, on the grounds that the sweep's
`ds._oracle` averages only 2.25–2.57 at every stage. That oracle is not
a feasibility ceiling: a predictive controller driving the *same* three
slide actuators scores **4.39** at stage 0 (max 8, ≥5 in 40% of
episodes, 49/200 episodes still alive at the 750-step cap). 3.0 is
reachable; the defect was the measurement, not the bar. The flat
~2.4-across-stages curve is likewise an artifact — each stage's oracle
probe is tuned individually and criterion 7 rejects any stage scoring
more than 1.5× its predecessor, so under one uniform controller the
stages read 4.39 / 2.85 / 1.52 at s0 / s2 / s4.

Cost: eval episodes per cycle go 60 → 90 (+50%). The three eval streams
already cost about as many env steps as training itself, so expect
throughput to fall roughly 15–20%; budget wall clock accordingly.

Metrics era: `bounce_count_ep_mean` keeps its meaning and gets less
noisy, but promotion *timing* is not comparable with 30-episode runs,
and no reward-bearing quantity is comparable across the gamma change.

## 0.20.0

Curriculum-harness changes aimed at the depth campaign's dominant cost:
per-stage promotion price, not total budget. Runs
`20260721_004722` (3M steps, reached stage 2 of 4) and
`20260724_152530` (6M budget, still on stage 1 at 2.55M) place the same
ladder position at twice the compute, so what binds is the cost of each
promotion.

One new `performance_gate` key, **default off**, so existing runs are
bit-identical:

- `reset_entropy_on_advance` restores SAC's auto-tuned entropy
  temperature to its initial value on every stage advance and clears the
  temperature optimizer's moment buffers. Run `20260721_004722` ended at
  `train/ent_coef` 9.2e-4 and never re-inflated after a promotion, so the
  policy met each new geometry very nearly deterministic — and
  `advance_update_pause_steps` sets `gradient_steps = 0`, freezing
  `log_ent_coef` too, so the tuner could not recover during the pause
  either. Restores the *pressure* to re-expand entropy, not the action
  noise itself: sampling spread lives in the policy's learned `log_std`
  and re-expands over the following gradient steps. Raises at training
  start when `ent_coef` is a fixed float, rather than being a silent
  no-op. The restore target is read from the **configured** `ent_coef`
  string rather than the live tensor: a warm-started continuation
  deliberately inherits the source run's collapsed `log_ent_coef`, so
  sampling the tensor would restore precisely the collapse the reset
  exists to undo. `entropy_reset_value` overrides that target when
  `"auto"`'s 1.0 is more exploration than a mid-campaign continuation
  wants.

`InfoDictEvalCallback` now publishes `last_confirmation_metrics` (cleared
per evaluation) so `confirm_best`'s second batch is observable instead of
vanishing except for the copy in `best_model_meta.json`.

A `pool_confirmation_samples` gate switch that folded that batch into the
promotion window was **built and then rejected** — see `docs/DECISIONS.md`.
The batch is conditionally sampled (it only runs when the primary batch
beat the running best), so averaging it in regresses exactly the high
draws toward the mean while leaving low draws alone. That biases the
window downward and silently raises the bar, the one thing run 1's review
said not to do. Simulated at the campaign's own numbers it cost **+27
evaluations ≈ +666k env steps per promotion** in the slow-climb regime
Run A is actually in.

Evaluation streams are consolidated. Under headline-metric selection the
reward `EvalCallback` and the final-config info-eval stream roll the
*same* distribution (the recipe's `eval_env_overrides`; the gate re-syncs
only the matched evaluator), and the reward stream is reporting-only
there. It is now retired and the final-config stream owns
`evaluations.npz`, emitting SB3's exact schema and mirroring
`eval/mean_reward` / `eval/mean_ep_length`, so TensorBoard dashboards,
`notebook_utils` plots, and `stage_summary.txt` read unchanged. One env
and one rollout pass fewer per evaluation; for run `20260724_152530`'s
config the three streams cost 50 episodes per 25k training steps.

New `final_eval_episodes` sizes that stream. It defaults to the **full
`n_eval_episodes`** when the merge happens — 30 for the depth recipe,
against the 5 + 15 the two split streams used — because the survivor is
then the only stream scoring the goal task during training. At 5 episodes
that estimate had a standard error of ~0.8–1.1 bounces, too noisy to
resolve the 0.30 → 0.98 → 1.76 transfer curve the campaign exists to buy.
Runs that keep both streams are unchanged (`n_eval_episodes // 2`).

Net eval cost per 25k training steps for the depth recipe: 50 episodes
across three rollout passes becomes 60 across two. Ten more episodes buys
a 6× larger sample on the campaign's target metric; set
`final_eval_episodes` explicitly to trade it back for wall clock.

Retiring the reward `EvalCallback` also removed the only periodic
progress a run printed, leaving a multi-hour job silent apart from stage
advances and the run-ending line. New `eval_verbose` restores it on the
info-eval streams alone, **default off** (`None` follows `verbose`):
`eval_verbose = 1` prints one line per evaluation per stream — step,
episodes, mean reward, episode length, curriculum stage, and the
selection metric — prefixed with the stream's `log_prefix` so the
matched and final-config lines stay apart. It is deliberately separate
from `verbose`, which drives SB3's per-rollout table: SAC emits that
every `log_interval` (4) episodes, thousands of tables over a
multi-million-step run, so it is not a usable substitute for progress at
`eval_freq` cadence.

Selection, the gate, `best_model.zip`, `best_model_meta.json`, and the
per-stage `stage_bests/` archive are all driven by the matched stream and
are unchanged.

Reporting and provenance:

- `curriculum_stages.json` records `reset_entropy_on_advance` (additive:
  existing readers are unaffected, but the file is not byte-identical to
  a 0.19.0 run's) and `curriculum/entropy_resets` is a new TensorBoard
  series.
- `config.json` now records `reward_eval_episodes` and
  `final_eval_episodes`. Both were absent from the hand-maintained
  `train_config` block, so a run's artifacts could not say whether its
  reward stream rolled 5 episodes or 30. A new test pins that the block
  covers every `TrainConfig` field except the four code-valued ones and
  the two recorded at top level, so the next field cannot drift the same
  way.
- `InfoDictEvalCallback` now rejects a multi-worker `eval_env`. Its
  rollout loop reads `infos[0]`/`rewards[0]` and counts episodes from
  worker 0 only, so extra workers were stepped but never measured — a
  silent mismeasurement, and the reason vectorizing evaluation is a
  rewrite of that loop rather than a bigger env.

Metrics era: unchanged for the matched stream and for `eval_info.csv`.
`evaluations.npz` keeps its schema but its rows come from a different
(larger) episode count, so absolute reward curves are comparable in
meaning and less noisy, not bit-identical.

## 0.19.0

`WallBallDepthCurriculum` now slides the paddle's entire movement
window toward the baseline instead of only extending its rear edge.
Run `20260722_124613` exposed the old ladder's loophole: all five
stages shared a front-court interval, so the policy could reach the
final stage while continuing to contact the ball at a shallow
position. The replacement ladder keeps adjacent stages
overlapping but removes any position common to every stage:

- paddle fences are now `[-2.7, 0.3]`, `[-3.2, -0.8]`,
  `[-3.7, -1.6]`, `[-4.2, -2.4]`, and `[-4.7, -3.0]`;
- paddle starts, serve speeds, open scoring, rewards, the
  three-evaluation promotion gate, and its replay-clear / 50k-update
  pause are unchanged;
- the default recipe and starter-config budget increase from 3M to 6M
  steps so the run has time to reach and train at the final stage; and
- fixed final evaluation now uses the new final-stage fence.

WallBall `info` now partitions legal paddle hits into pre- and
post-bounce counts, records whether the episode opened with a volley,
and counts completed returns initiated after a bounce. Matching
one-step event flags make contact ordering auditable without changing
the reward or observation. The long-horizon report advances to schema
3 and includes pooled contact-sequence rates alongside its existing
floor-bounce proxy diagnostics. The calibration sweep validates the
counter identities, reports the new behavior metrics, and adds a
non-blocking pre-bounce interception probe.

The replacement geometry passed the 200-episode-per-cell scripted sweep
on 2026-07-23. Oracle ≥2-return rates were 92–96% and crude-controller
rates remained nonzero at every stage; the diagnostic opening-volley
probe fell from 100% in stages 0–2 to 38% in stage 3 and 0% in stage 4.
There is no environment policy-space or reward change, so 0.18.0
checkpoints remain loadable; however, the new stage geometry and budget
make curriculum progression and aggregate learning curves a new
comparison era.

## 0.18.0

Performance-gated curriculum runs now keep their per-stage history.
Previously each stage advance called `reset_selection_state()` and the
next stage's first evaluation overwrote `best_model.zip` — destroying
the departing stage's champion. Runs `20260721_142121` and
`20260722_002913` each ended wanting exactly that checkpoint (the
stage-entry policy was the best either run measured at the next
stage's geometry), and only luck preserved one of them as the
run-level best.

- **`model/stage_bests/stage_NN/`**: on every advance the gate
  archives the departing stage's `best_model.zip` /
  `best_vec_normalize.pkl` / `best_model_meta.json` (immediately
  before the selection reset), and the final stage's best is
  duplicated there at training end — skipped when that stage recorded
  no evaluation, since the on-disk triple would belong to the previous
  stage. The run's `config.json` is copied alongside each triple, so a
  `stage_NN/` directory is a valid (legacy-flat-layout)
  `WarmStartConfig.source_run_dir`: a continuation really can
  warm-start from any archived stage's champion.
- **`reports/curriculum_stages.json`**: refreshed atomically on every
  stage close and finalized at training end, with each stage's
  entry/exit timesteps, evaluation count, promotion window values
  (kept under both promotion rules), streak, and archived best
  selection values — the table every campaign review previously
  reconstructed by hand from `eval_info.csv`. Interrupted runs keep
  it too: train()'s KeyboardInterrupt salvage path finalizes the gate
  (SB3 skips `on_training_end` when an exception escapes the training
  loop), and completed-stage rows are already durable on disk even
  against a hard runtime death.
- **`config.json` gate provenance**: the structured
  `performance_gate` block now records `promotion_rule`,
  `advance_update_pause_steps`, and `clear_replay_buffer_on_advance`
  (mirroring `train()`'s resolution defaults) — run
  `20260721_142121`'s warm-up package was active but provably absent
  from its artifact.

Both new artifacts are registered in `RUN_LAYOUT` as conditional
(gate-only) entries, so non-gated runs audit clean. No physics,
reward, or training-behavior change; all 0.17.0 artifacts and curves
remain comparable.

## 0.17.0

`WallBallDepthCurriculum` widens the promotion window: `sustain_evals`
2 → 3 (`window_mean`, so promotion now needs the mean of the last
three 30-episode evals ≥ 3.0). Run `20260721_142121` — the first
campaign run under `window_mean` — promoted out of stage 0 on a 2-eval
mean of 3.08, only ~0.3 standard errors above the bar (per-eval SE
near 3.0 is ~0.4 bounces), i.e. a variance-driven promotion; its
stage-1 exit (3.37) would have cleared any reasonable window. The
threshold itself stays 3.0: observed eval maxima at the easy stages
(3.27/3.47) leave no headroom to raise it without gating the whole
run, and the run's stage-2 stall was a post-advance recovery failure,
not premature promotion. Cost is ~50k extra steps per stage. Promotion
timing — and therefore `curriculum/stage_index` trajectories — are not
comparable with 2-eval runs; per-stage eval metrics remain comparable.
`WallBallBootstrap` (historical) keeps its 2-eval gate.

## 0.16.1

Version bump only; no physics, observation, action, or recipe changes.
All 0.16.0 artifacts and learning curves remain comparable.

## 0.16.0

Humanoid-tennis stage overhaul: the curriculum's racket forgiveness now
physically works, and the three PPO stage recipes adopt the open
remediations from `docs/humanoid_env_review.md` / `docs/DECISIONS.md`.
A new metric era for the tennis stages — contact behavior and rewards
changed, so prior tennis-stage artifacts and curves are not comparable.
WallBall and ball recipes are unchanged.

- **Stringbed collision bounds fixed.** The `racket_contact_scale`
  enlargement wrote `geom_size` and called `mj_setConst`, which never
  refreshes `geom_rbound`/`geom_aabb`; MuJoCo's broadphase culled every
  contact beyond the nominal extent, leaving the forgiveness inert.
  Both bounds are now recomputed from the scaled semi-axes, and a
  regression test pins that a ball in the enlarged-only annulus
  generates a contact (and does not against the nominal stringbed).
  The stage-1 feasibility oracle was recalibrated for the corrected
  physics (swing at control call 56, pitch 0.65; both mirrored sides
  land the target return in 129 steps).
- **Task-metric selection for Stages 0–2.** `headline_key =
  "stage_success"` (plus `confirm_best_eval` and 5-episode
  reporting-only reward evals) moves `best_model.zip` and early stop
  off the shaped eval reward — the run-20260712_190054 failure mode the
  WallBall recipes already guard against. `legal_hit_count_ep_mean`
  sits between the success keys and the reward tie-break so
  pre-success contact progress still resets patience and updates the
  best model.
- **Exploration and shaping unblockers.** Stock iid Gaussian
  exploration measured zero contacts in 264 random episodes while a
  constant held action succeeds ~15% — the recipes now set
  `use_sde = true` with a small `ent_coef = 0.01` floor, and Stages 1–2
  enable the (previously inert) escrowed `valid_hit_shaping = 0.25` so
  a hit that becomes a target return keeps a contact-time reward while
  episode-total farming stays impossible.
- **Reward normalization off for the tennis stages.** The PPO default
  (normalize returns) would amplify the first sparse success toward the
  clip ceiling on this hand-scaled terminal-only reward
  (`normalize_reward = false` in recipes and starters).
- **Early stop reachable per stage.** Patience scaled to each stage's
  eval budget (8 / 12 / 20 for 20 / 40 / 80 evals); the shared 20 was
  mathematically inert for Stages 0–1.
- **Eager config validation.** `build_train_config` now resolves the
  algo name and validates merged `model_kwargs` against the resolved
  algorithm's constructor before any environment or artifact work
  (cross-algorithm keys previously failed only inside `train()` after
  the env fleet was built, and a string `ent_coef` on PPO survived
  until the first gradient update).

## 0.15.1

First-round fixes from the depth-curriculum run-1 review
(`docs/wall_ball_depth_curriculum_run1_review.md`, run `20260721_004722`:
budget-bound at stage 2 of 4, monotone final-geometry transfer confirmed).
The gate gains an opt-in `promotion_rule = "window_mean"` (promote on the
mean of the last `sustain_evals` evaluations — stage 2 cleared 3.0 in four
separate 30-episode evals without ever managing two in a row) and a
promotion warm-up package (`advance_update_pause_steps`,
`clear_replay_buffer_on_advance`): each advance drops the stale-stage
replay buffer and pauses gradient updates until fresh frontier-stage data
exists, targeting the measured promotion shocks (matched eval 3.37→1.43
and 3.10→1.70, ~0.5–1M steps of recovery each). `WallBallDepthCurriculum`
adopts all three (window mean, 50k pause, clear) plus
`reward_eval_episodes = 5` (the reporting-only reward eval stream cost
~as many env steps as training itself). Policy-only `warm_start` now
supports SAC (actor/critics/targets via the policy state dict plus the
auto-tuned entropy temperature), enabling stage-2 continuation runs with
a truncated `[train.performance_gate]` stage table. The gate stamps
`curriculum_stage_index` into the matched eval stream (eval_info.csv,
TensorBoard, `best_model_meta.json`, stage summary) so scores are never
again ambiguous about their geometry. WallBall info dicts add
`legal_paddle_hit_x_sum` / `legal_paddle_hit_x_mean` (where legal hits
happen — the fence-front-camping tripwire for deep stages).
`VideoRecordCallback` no longer stops at the first episode end: milestone
videos/CSVs record up to `max_episodes` (default 3) within the
`video_length` cap. No physics, observation-space, or reward changes;
existing eval metrics remain comparable within the 0.15 era.

## 0.15.0

Adds `WallBallDepthCurriculum` — the campaign's answer to the falsified
cheap levers (budget, incentives, capacity; see `docs/lessons_learned.md`
lesson 19): curricularize *position* instead. The recipe runs the env's
`open` rally style (any paddle hit opens the gate, volleys are legal paid
returns, no early-touch fine) with a performance-gated five-stage depth
ladder that walks `paddle_x_fence` / `paddle_start_x` from volley range
(−2.7, 0.3) back to the workspace baseline (−4.7, −1.2) while serve speed
co-moves 5.2 → 7.0, each stage earned by sustaining ≥3 eval exchanges.
The action mapping stays pinned to the full workspace so action semantics
never drift. Every stage was calibrated by the new
`tools/depth_stage_sweep.py` scripted ladder (parked < crude full-swing <
charge-and-lead oracle strictly monotone; oracle ≥2 returns from 93–97%
of serves; crude second exchanges 66–83%; 200 episodes/cell).
`WallBallBootstrap` is marked historical/superseded (its cold-start
problem was solved by auto-entropy before it ever ran, and its reward
package bundles the falsified weak-return retry). Depth-ladder metrics
are a new era — not comparable to fixed-lane baseline runs, whose
reference remains `20260718_023737`. No physics or observation changes;
existing recipes untouched.

## 0.14.0

Reverts the falsified 0.13.0 `WallBallBaseline` recalibration: run
`20260718_213222` showed the asymmetric weak-return retry teaches soft,
unchainable returns (1.33 vs 3.23 eval bounces; 82% double-bounce), so
terminal weak returns and SB3-default gamma are restored, keeping only
the ≥5-rate selection tiebreak. Adds `wall_reward_increment` (the n-th
completed return banks `1 + (n−1)×increment`; default 0.0 is
bit-identical and it ships dark, ladder-calibrated at 0.5 — run
`20260719_223139` later showed it redistributes failure style without
moving the ceiling). Run directories are reorganized into `model/`,
`metrics/`, `reports/`, `media/`, `checkpoints/` via a shared registry
with legacy-flat fallback for existing runs, and the unreadable
`eval_info.png` grid is split into four themed, size-bounded pages.
Training-side reward semantics return to 0.12 behavior; eval metrics
remain comparable with 0.10+.

## 0.13.0

Recalibrates `WallBallBaseline` from the first two runs that learned the full
task (`20260717_165358`, `20260718_023737`): training fines weak returns
(`weak_return_penalty = 0.1` retries) while evaluation re-asserts the strict
terminal rule so scores stay comparable, the critic's horizon doubles
(`gamma = 0.995`, auto-entropy untouched), and best-model selection tie-breaks
on the ≥5-bounce rate the `023737` run actually improved instead of the
saturated ≥2 rate. WallBall also gains a render-only `court_style` kwarg —
`"tennis"` draws a to-size ITF half-court (baseline 11.885 m from the
wall-as-net, service line at 6.40 m, singles/doubles sidelines) for replay
footage, which the notebook now records by default — and the notebook resolves
the recipe's run config automatically (`CONFIG_FILE = "auto"` creates your
editable Drive copy from the packaged starter on first use). Training-side
reward semantics changed, so 0.12 baseline *learning curves* are not
comparable; eval metrics remain comparable with 0.10+.

## 0.12.0

Makes run configuration Colab-native: the starter TOMLs ship inside the
package and the SAC recipes carry their calibrated `n_envs = 8`, so the
training notebook no longer pins `MODEL_KWARGS` (the 20260717 A/B showed the
old pinned entropy bundle prevents WallBall from learning at all) and only
passes variables you explicitly set — a TOML's `[train]` table is not silently
overridden (`seed` stays notebook-owned: the loader rejects it in files,
loudly). Recipes now also carry the calibrated `early_stop_patience = 20`.
Physics unchanged; 0.11 artifacts remain comparable.

## 0.11.1

Adds render-only court markings to WallBall so videos show where on the court
the action is: bold white lines at the wall base (x = 3.9) and the deepest
paddle reach ("baseline", x = −4.7), faint 1 m coordinate ticks, a cyan strip
marking the preset's paddle lane with a yellow paddle-home line, orange fence
lines (visible only when a `paddle_x_fence` is set), and a warm line at the
serve drop x. The markers are MuJoCo sites — they cannot collide — and the
preset-dependent ones are repositioned every reset, so curriculum changes made
between episodes stay visible. No physics, observation, or reward change.

## 0.11.0

Targets the WallBall bootstrap failure (three runs in which SAC never touched
the ball while a scripted tracker contacts 100% of serves). New
`WallBallBootstrap` recipe: a `first_hit_bonus` paid outright once per episode
makes touching the ball strictly out-earn passivity for the first time;
`weak_return_penalty` softens the terminal `floor_before_wall` fault into a
fined retry so feeble early swings stop scoring identically to doing nothing; a
performance-gated ladder (new `PerformanceGatedEnvStagesCallback`) widens the
serve's lateral corridor from `serve_vy_max` 1.1 to the full 2.0 as competence
is demonstrated, with matched-stage evaluation driving selection and
`eval_info_final.csv` tracking the canonical serve. Depth and serve-pace
ladders were swept and rejected (close-court rebounds fly out; slow serves
underpower returns). WallBallEnv also gains `serve_start_x`, `paddle_start_x`,
`paddle_x_fence`, and a schedulable `paddle_joint_damping` for future
curricula, and run provenance now records the installed git commit on Colab
(pip VCS installs) and honors a `COURTSIDE_DYNAMICS_GIT_SHA` override. Existing
recipes are unchanged and 0.10 artifacts remain comparable.

## 0.10.0

Recalibrates `WallBallBaseline`'s geometry so the second rally exchange is
learnable, not just scriptable. The baseline recipe raises its paddle slide
damping from 5 to 8 (new `paddle_joint_damping` env kwarg), capping saturated
swings at 12.5 m/s and slowing returns ~15% so rebounds land shallower; it
widens its lane front from x=-2.1 to x=-1.6; and it softens the pre-bounce
paddle touch from a terminal style violation to a non-terminal
`early_touch_penalty` fine (new env kwarg; the default `None` keeps the
terminal rule). In the calibration sweep a placement-blind full-swing tracker
recovers a second exchange in 70% of episodes under the new geometry versus 0%
under the old, while the scripted oracle still returns 500/500 calibrated
serves and completes two or more returns from 92% of them. Observation and
action shapes are unchanged, but the physics and lane changes make 0.9
`WallBallBaseline` policies, `VecNormalize` statistics, and learning curves
non-comparable; start fresh baseline runs. The open `WallBall` and
`WallBallVolley` presets keep the shared XML's damping 5 and their existing
calibration. The recoverable-bounce placement score now projects to the new
baseline lane front (x=-1.6).

## 0.9.0

Adds recovery-focused training to the strict baseline style. WallBall supports
fixed per-run rally styles while keeping the same 3-action interface.
`WallBallVolley` forbids floor contacts. `WallBallBaseline` requires exactly
one bounce before each paddle return, starts the paddle farther back at world
x=-2.7, restricts it to the [-3.2, -2.1] baseline lane, and uses a calibrated
lower serve. Its training factory mixes normal serves with incoming-wall and
post-bounce recovery fragments, then tapers that practice using global training
steps. Checkpoint evaluation, videos, and post-training endurance scoring
always start from a normal serve. A one-bit recoverability flag expands
WallBall observations to 23 values, so all 0.8 `WallBall`, `WallBallVolley`,
and `WallBallBaseline` policies and `VecNormalize` statistics require a fresh
run. The original `WallBall` recipe remains the permissive `open` setup; train
separate policies for the strict volley and baseline recipes.

## 0.8.0

Simplifies WallBall to the paddle face alone at a fixed 10° upward pitch. Its
three `[-1, 1]` actions are absolute x/y/z position targets tracked by
force-limited servos, and its observation shrinks from 26 to 22 values after
removing yaw/pitch state. Previous WallBall models, `VecNormalize` statistics,
replay buffers, and raw MuJoCo states are incompatible; start a fresh run
rather than resuming a pre-0.8 artifact.

## 0.7.0

Corrects BallBounce's rotation units and actuator authority and replaces
control-boundary touch rewards with substep-resolved, top-face rebound events.
Earlier BallBounce policies and learning curves are not comparable and should
be retrained; the observation grows from 18 to 30 values to include ball spin
and the event detector's Markov state. Its training recipe reports success
after ten deliberate rebounds in one episode; passive paddle contacts do not
increment that metric.

## 0.6.0

Registered environment IDs become unversioned. Callers using the previous
version suffixes should switch to the unversioned IDs listed in the README
environment table.
