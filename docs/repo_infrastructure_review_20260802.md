# Repository & environment infrastructure review — 2026-08-02

Status: **Review snapshot**, pinned to `main`@`4bd715d`, v0.25.0.

Scope: requested as a between-campaigns infrastructure review before the
PaddleTennis era opens — code cleanup, consolidation, unit tests,
documentation, bug fixes, and logging, with training-outcome questions
explicitly out of scope (the wall-ball verdicts are already recorded in
their own review snapshots). Method: nine parallel scoped reviews
(core envs, tennis stack, training, callbacks, config/recipes,
notebooks/tools, tests, docs, CI/logging), every medium/high finding
adversarially re-verified against the code before inclusion (3 of 66
were refuted and are excluded); plus empirical checks and a Google
Drive inventory of the training-run corpus.

## 1. Baseline health

The empirical checks all pass at the pinned commit:

- `ruff check .` — clean. `mypy` — clean (34 files).
- `pytest -n 2` (pinned threads) — **831 passed, 1 skipped, 84 s**.
  The skip is the GL-context render test, expected headless.
- Five SB3 `UserWarning`s ("render() but no `render_mode`") leak from
  `tests/test_callbacks.py` — harmless, but worth silencing so real
  warnings stay visible.

Things the review set out to find and *did not*: no dead code in the
training package (the retired depth-ladder and advance-package knobs
remain live via shipped TOMLs and history-reproduction paths);
`env_attr_schedule` and the gate's advance package are still exercised
by live recipes; `tools/depth_stage_sweep.py` is not dead weight (its
`--ladder release` path delegates to the live certification module the
current-era recipes self-certify against). The rally reducer and the
wall-ball reward/termination logic survived hand-tracing without a live
correctness bug. The lesson-16 `[eval_env]` drift blind spot is
genuinely closed.

The findings below are therefore mostly *seams and drift*, concentrated
exactly where the next campaign will lean.

## 2. P0 — bugs worth fixing regardless of PaddleTennis

### 2.1 The TOML validator has drifted two releases behind the code (largest cluster)

`run_config.py`'s hand-maintained key allowlists lag the 0.24.0/0.25.0
features, so **a run-config TOML for the current era's certified task
cannot be written**:

- `run_config.py:322` rejects `lead_charge` oracle probes, though
  `ladder_certification.py:622` accepts them and every current recipe
  (DepthCurriculum, GoalRally, TrueBaseline) uses *only* `lead_charge`
  probes.
- `run_config.py:273` — `_LADDER_CERT_KEYS` lacks
  `feasibility_ge2_floor` (0.25.0). A TOML replacing
  `[train.ladder_certification]` on WallBallTrueBaseline silently loses
  the recipe's 0.50 floor and falls back to the stock 0.90 that no
  scripted reference reaches on the 11.8 m task — startup certification
  would then fail.
- `run_config.py:177` — `_GATE_OPTIONAL_KEYS` lacks
  `stage_eval_budget` / `stage_eval_budget_action` (0.24.0), which
  `train.py:1283-1284` accepts; the validator's error message
  ("train() reads exactly [...]") is now factually wrong.
- Downstream of the same drift: `artifacts.py:310`'s
  `performance_gate` provenance block and
  `performance_gate.py:567-580`'s `curriculum_stages.json` payload both
  omit the staleness-guard pair, so a run stopped by the guard has a
  `config.json` that shows no guard was configured — against the
  block's own "records the gate as actually run" comment.
- `docs/run_config_file_spec.md:135-150` (a **Living** spec)
  understates the accepted surface: no `lead_charge`, no
  `feasibility_ge2_floor`, and the gate optional-key list omits the
  0.20.0 entropy pair as well as the staleness pair.

**Recommended shape of the fix**: don't just patch the lists — derive
them. Export one authoritative key-set per consumer
(`train()`'s accepted gate keys, `ladder_certification._SPEC_KEYS`,
the probe-kind set) and make `run_config.py` import them, plus a drift
test asserting loader sets == consumer sets
(`tests/test_run_config.py:296` area currently pins only 0.15.1-era
keys). This is lessons-learned rule 16 applied to the surface
PaddleTennis is about to grow with new probe kinds and gate knobs.
Note the loader *validation* itself is well tested
(`tests/test_ladder_certification.py:305-345`) — the gap is the key
lists, not the machinery.

### 2.2 video_record.py cluster

- `video_record.py:268` — after `auto_sums.pop(k)` fires for a
  non-numeric info value, the next occurrence of that key raises an
  uncaught `KeyError` (the subscript load evaluates before
  `float(value)`, escaping the `except (TypeError, ValueError)`),
  crashing the run. Latent for WallBall's all-numeric infos; a
  PaddleTennis `winner`/outcome info field would hit it.
- `video_record.py:318` — the recording rollout has **no exception
  isolation** (try/finally only). A `mujoco.FatalError` or moviepy
  `DependencyNotInstalled` propagates out of `model.learn()` and loses
  the run's final artifacts. `docs/DECISIONS.md:706` claims this is
  already fixed ("has gained try/except around normalizer sync and the
  rollout") — the DECISIONS entry currently overstates the shipped
  code, and no test pins the isolation.
- `video_record.py:158` — a TOML can set `csv_header` but not
  `info_row_fn` (`run_config.py` rejects callables by design); that
  combination keeps the pinned header while installing the auto row
  formatter, silently writing rows of a different width than the
  header.
- `video_record.py:273` — `rec_env.render()` per rollout step is
  redundant under SB3 ≥ 2.5 (`VecVideoRecorder.step_wait` already
  captures frames), roughly doubling render cost of each recording
  pass. Delete the line.
- `info_dict_eval.py:372` (and `video_record.py:205-206`) — a failed
  `sync_envs_normalization` is silently swallowed; SB3's own
  EvalCallback raises here. Today `train()` always wraps both sides in
  `SelectiveVecNormalize` so it never fires, but a differently-wrapped
  PaddleTennis eval env would be scored on stale normalization with no
  trace — the silent-no-op class this repo explicitly bans.

### 2.3 Other confirmed bugs

- `train.py:1637` — the `final_info_eval` env is **never closed**: the
  inner `finally` closes train/eval/info_eval then calls
  `opened_envs.clear()`, erasing the outer `finally`'s only record of
  `final_info_env`. Leaks a MuJoCo env on every completed run that uses
  the final stream; every `_merged_eval_cfg` test silently tolerates
  it.
- `train.py:881` — `train()` never calls
  `algos.validate_model_kwargs`; only the recipe path does. A direct
  `TrainConfig` user (the module docstring's advertised usage) gets the
  late crash the validator exists to prevent — after the env fleet is
  built and artifacts written.
- `wall_ball.py:904` — the four 0.25.0 coordinate ticks
  (`court_tick_xm8/xm7/xm6/xm5`, `wall_ball.xml:34-37`) are missing
  from `_COURT_STATIC_SITES`, so `court_style="tennis"`/`"none"` never
  hide them; they render on top of the tennis surface in presentation
  footage. The mirror test iterates the same incomplete tuple, which is
  why it passed. Fix by deriving visibility lists from compiled-model
  `court_*` site names and asserting exhaustive coverage.
- `wall_ball.py:717` — `paddle_x_target_range` is the one derived-at-
  init config knob without a guarding property setter after the 0.22.0
  `paddle_home_x` lesson: a post-init `set_attr` is a silent no-op on
  action semantics *while the rendered lane markers move* at next
  reset — actively misleading video overlays.
- `tennis_rules.py:727,732,803,893` — `event.from_side or
  self._ball_side` discards an explicit `from_side=CourtSide.A`
  (IntEnum value 0 is falsy). Behaviorally masked today (validation
  forbids the contradictory cases), but semantically wrong at four
  sites; must be `is None` checks before the reducer is reused with
  side-relative mirroring or event replay.
- `training/tennis_curriculum.py:884` — `_git_sha` here is a weaker
  duplicate of `artifacts._git_sha` (no ls-files gate, no env
  override, no PEP 610 fallback): promotion provenance records
  `git_sha=None` on Colab VCS installs and can record an *enclosing
  repo's* SHA for a site-packages install. Delete and import the
  hardened resolver.
- `notebook_utils.py:722` — `_load_obs_normalizer` swallows any
  `VecNormalize.load` failure and returns identity, so
  `record_best_model_video` silently replays the best model on
  unnormalized observations — the replay looks like a catastrophically
  bad policy. Given how often obs shapes have changed (22→23 etc.),
  this failure mode is realistic; log loudly or raise.
- `notebooks/sb3_training.ipynb:205` — the notebook imports
  `courtside_dynamics.recipes` (→ `import mujoco`) *before*
  `setup_colab()` sets `MUJOCO_GL=egl`, violating the ordering contract
  the humanoid notebook documents and `test_notebooks.py` enforces
  only for the humanoid notebook. Either fix the ordering and extend
  the test to both notebooks, or (if provably harmless) fix the
  humanoid notebook's claim — today the two notebooks assert
  contradictory contracts.

## 3. P1 — consolidation that directly de-risks PaddleTennis

`docs/design_paddle_tennis.md` §2 claims the paddle "transfers as-is"
and the rules/events modules are "deliberately not welded to the
humanoid env". The review measured those claims; the code is currently
behind them in five places, and there are six patterns already
duplicated 2–4× that the new env would copy again (twice, for the
two-paddle case):

1. **Paddle interface extraction** (`wall_ball.py`, ~300 lines across
   `__init__`, three property setters, `_action_to_controls:1162`,
   `reset_model`): the calibrated normalized-target mapping, home-pivot
   world-space resolution, fence clamping, workspace validation, and
   schedulable damping belong in a composable `PaddleInterface` that
   WallBall instantiates once and PaddleTennis twice. Same for the
   asset: the calibrated paddle subtree (`wall_ball.xml:97-113,141-143`)
   should become an includable fragment rather than hand-mirrored XML.
2. **`_tennis_events.py:160-171` hard-requires humanoids** —
   `TennisSceneContactIndex._validate` rejects empty
   `humanoid_a/humanoid_b`/racket groups, and `from_model` hardcodes
   `ball_free`/`court_surface`/`player_a_racket_` names. As-is,
   PaddleTennis P1 cannot reuse the substep sampler, contact latching,
   or crossing detector. Make the robot groups optional and lift names
   into parameters.
3. **`_tennis_physics.py:45-55` freezes the regulation court** in
   module constants; the probes froze PaddleTennis at half-length
   6.5 m. A small frozen `CourtGeometry` dataclass threaded through
   `RallyStateMachine` (used at `tennis_rules.py:667,761,944`) keeps
   the humanoid env byte-identical while letting the new env reuse the
   reducer instead of forking it.
4. **Triplicated env guards**: the last-finite-observation echo
   (`humanoid_tennis.py:1261`, `wall_ball.py:1790`,
   `ball_bounce.py:439` — three names, drifting semantics) and the
   finite/nonnegative validator battery (`ball_bounce.py:112` static
   methods vs ~15 inline blocks in WallBall's constructor) belong in
   `_base.py`. BallBalance — the one env *without* the NaN guard
   (`ball_balance.py:34-42` returns the raw nonfinite obs and feeds
   unvalidated actions into `do_simulation`) — then inherits it.
5. **Piecewise action mapping** implemented twice
   (`humanoid_tennis.py:707`, `wall_ball.py:1179-1185`) and
   hand-inverted twice more in the scripted oracles — the documented
   retired-map oracle bug came from exactly that inversion. One shared
   `piecewise_targets`/`normalize_targets` pair in `_base.py`.
6. **Serve/feed machinery**: `TennisServeConfig` (inside
   `humanoid_tennis.py`) and `CurriculumLaunchConfig` duplicate the
   feed contract (same 1.1–1.5 m height window, on-half, lateral, noise
   validations) in incompatible parameterizations, with side-B
   mirroring implemented twice; WallBall's serve sampling is inlined a
   third way in `reset_model:2083`. PaddleTennis's P3 serve-rules probe
   needs precisely one side-local launch module with a
   `mirror_for_side` helper.
7. **`tennis_rules.py` fault pre-scan duplication** — `_crossing_/
   _court_fault_candidate` re-simulate `_handle_crossing`/
   `_handle_court_contact` transition logic in projected-local form
   (`:804-808` vs `:895-899`), and `_handle_crossing` repeats the
   strict-vs-forgiving block five times. Extract shared classification
   predicates so the projection and the handlers cannot diverge.
8. **`wall_ball.py` info-dict duplication** — `_nonfinite_termination`
   hand-copies `step()`'s ~58-key info dict; a missed key silently
   breaks CSV writers. One `_info_dict()` builder; the PaddleTennis
   info contract should be built the same way from day one.
9. **`recipes.py` templates** — `_make_goal_rally` and
   `_make_true_baseline` (`:1304`) are the same 100-line direct-task
   scaffold differing in task/probe/patience; a
   `_make_direct_task_recipe` helper is the template the first
   PaddleTennis recipe will want. The `rally_phase` block is repeated
   verbatim five times (`:380,415,558,667,981`) — module constants,
   as the humanoid recipes already do.
10. **`train.py` (1647 lines)** — the ~450-line warm-start subsystem
    has a clean seam and its own tests; extract to
    `training/warm_start.py` before opponent plumbing lands (keep
    `SelectiveVecNormalize` importable from its current path —
    saved `vec_normalize.pkl` files unpickle by module path).
11. **Triplicated numeric/provenance helpers** — three
    `_wilson_interval`s that have *already drifted* (clamped vs
    unclamped, different validation; `ladder_certification.py:368`,
    `tennis_curriculum.py:705`, `depth_stage_sweep.py:524` — only the
    tools copy has boundary tests), two chunked-sha256s, two
    `_git_sha`s. One shared implementation each; the DECISIONS Wilson
    entry explicitly calls for the boundary tests at
    n ∈ {25, 30, 50, 100}.
12. **`notebook_utils.py` (1745 lines)** — ~950 lines are the WallBall
    long-horizon evaluation harness; extract to e.g.
    `evaluation/wall_ball.py` (re-exported for compatibility) so the
    generic pieces (distribution summary, atomic writers, artifact
    pairing) become the skeleton the PaddleTennis evaluator reuses.

## 4. P2 — test-suite updates

- **No `tests/conftest.py` exists.** MuJoCo/event helpers are
  copy-pasted with drift already visible: `_event_batch`/
  `_inject_batches` ×2, `_fresh_model`/`_set_ball_state` ×2 (one
  spells the kwarg `linear_velocity`, the other `velocity`),
  `_has_active_contact` ×2, `_set_ball_toward_stringbed` ×2, the
  `get_env`-shaped `_FakeModel` ×2, and inside `test_envs.py` alone
  `_zero_action` ×5. PaddleTennis tests would mint the next copies.
- **`test_envs.py` (3,965 lines)** is ~87% deep WallBall suites behind
  an "intentionally short" smoke-test docstring. Split the WallBall
  classes into `tests/test_wall_ball.py` (mirroring
  `test_humanoid_tennis_env.py`) before the new env's suites land.
  Also: registering an env requires updating four parallel tables in
  this one file — collapse to one.
- **KeyboardInterrupt salvage has zero coverage** (`train.py:1588`):
  the path protecting ~20-hour Colab runs (save final model, eval,
  "interrupted" summary, gate finalize) has no test naming
  `KeyboardInterrupt`; it is the seam a train.py refactor is most
  likely to break silently. Related confirmed gap: `finalize()` runs
  only in the KeyboardInterrupt branch, so a NaN/OOM crash loses the
  in-flight stage history row.
- **Git-SHA provenance chain untested** (`artifacts.py:88`): the
  override/ls-files-gate/PEP 610 fallback logic that every run review
  depends on has no unit tests.
- **Same-substep crossing+landing** in the reducer
  (`tennis_rules.py:721-723`) is correct by hand-trace but untested —
  a regression silently turns legal feed-and-land into
  FAILED_TO_CROSS.
- **Rendering smoke covers one env on one Python version**: generalize
  the skipif-guarded render smoke over all registered ids (WallBall
  with `court_style="tennis"` included) under the existing xvfb CI
  step.
- **No coverage measurement** anywhere; add `pytest-cov` on one matrix
  leg (xdist-aware) so gaps are at least visible before the new env
  lands.
- Retirement watch: `tests/test_depth_stage_sweep.py` pins
  `ALIGNED_STAGES` as the default ladder of a retired campaign —
  fine as archival, but label it (or gate it) so it doesn't read as
  live contract.

## 5. P3 — documentation updates

- **README is two releases stale on recipes** (`README.md:102-115`):
  `WallBallGoalRally` (0.24.0 production recipe) and
  `WallBallTrueBaseline` (0.25.0 era task) appear nowhere in the
  README, while the retired depth ladder reads as current. Add a drift
  test (README list == `sorted(RECIPES)`) so the PaddleTennis recipe
  cannot be forgotten the same way.
- **Historical markers lag the 0.24.0 verdict** across four surfaces:
  `WallBallDepthCurriculum`'s description still reads as live
  (`recipes.py:989`, no HISTORICAL marker despite the established
  pattern and test); `WallBallBootstrap`'s supersession chain points at
  the *retired* DepthCurriculum (`recipes.py:680`; note
  `test_recipes.py:1038` pins the stale pointer — update together);
  the bootstrap starter TOML calls itself "the recommended WallBall
  trainer" (comment text is invisible to value-equality drift tests);
  `WallBallDepthCurriculumAligned` still says a pending A/B gates it,
  but Phase D closed the arm permanently on 2026-07-28. Clear status
  markers matter right now: PaddleTennis work will browse `RECIPES`
  for a template.
- **`docs/README.md` index**: the self-acknowledged backlog is exactly
  seven docs (verified exhaustive against disk); additionally the
  blank line at `:36` orphans the `design_wall_ball_paddle_orientation`
  row outside the rendered table, and the prose says "three kinds"
  while defining four.
- **`__init__.py` version-comment stack** stopped at 0.19.0, six
  releases behind the `__version__ = "0.25.0"` line directly below it.
  It duplicates the CHANGELOG and has demonstrably stopped being
  maintained — truncate to a two-line pointer before the PaddleTennis
  version series starts.
- **No git tags** exist for any release (0.6.0–0.25.0), so mapping a
  run's `git_sha` to a version requires checkout archaeology. Tag
  releases going forward.
- `index.html`/CNAME: the public site is still "Coming soon." while
  the repo has four working environments, a completed campaign, and
  GIFs — cheap win if the domain is meant to be live.
- `tools/smoke_wheel.py:7` inventory comment says 11 starter TOMLs;
  14 ship. Symptom of hand-maintained duplication (see §6).

## 6. P4 — CI, packaging, and infrastructure

- **The mypy job type-checks a weaker world than the README's
  documented flow**: CI installs `.[dev]` only, so torch/SB3/pandas
  become `Any` under `ignore_missing_imports = true`, and
  `warn_unused_ignores` can even disagree between CI and local runs.
  Install `.[train,dev]` in that job (the test matrix already pays the
  torch download three times per run).
- **No `schedule:`/`workflow_dispatch:`** — every dependency is an
  unpinned floor and the primary distribution path is fresh Colab
  resolves, so an upstream release breaks users silently between
  pushes (the glfw-stub mypy override is this class of breakage). A
  weekly cron matters more during campaign lulls when pushes pause.
- **No `timeout-minutes`** on any job: a hang outside pytest's own
  `--timeout` net (pip resolve, build, xvfb) burns the 6-hour default —
  the exact scenario the `addopts` comment worries about.
- **`smoke_wheel.py:40`'s `EXPECTED_ENV_IDS`** hand-duplicates the
  `register()` calls with no sync test. Derive from
  `gym.registry` with a minimum-count assert, so a newly registered
  PaddleTennis env — and its packaged XML — is smoke-tested
  automatically. (This is the script whose whole purpose is catching
  the missing-package-data failure mode.)
- Refuted for the record: the `cache: pip` keys *are* valid without
  requirements files (setup-python falls back to hashing
  `**/pyproject.toml`) — no action needed there.

## 7. Logging & observability

The `print`-based convention is applied consistently (deliberate, per
Colab-first operation), with these confirmed gaps:

- **Run-ending decisions are stdout-only and unrecoverable**: the
  early-stop patience message (`info_dict_eval.py:832`), degenerate-
  signal stop (`:817`), and the gate's budget stop
  (`performance_gate.py:785`) all vanish with the Colab runtime, while
  `stage_summary.txt` records only a reason-less "(stopped early)".
  Persist a machine-readable `stop_reason` (summary field or
  `reports/events.log`). PaddleTennis adds more decision points
  (opponent swaps, phase transitions) that deserve the same channel.
- **Training-worker Monitor CSVs persist only `r,l,t`**
  (`train.py:947` — `make_vec_env` gets no
  `monitor_kwargs={'info_keywords': ...}`), the top "mostly open" item
  in the DECISIONS post-hoc-attribution entry. A recipe-settable
  `monitor_info_keywords` forwarded to Monitor gives the very first
  PaddleTennis runs termination attribution for free
  (`load_monitor_episodes` already passes extra columns through).
- **Sampled serve parameters are unrecoverable** from the info stream
  (`wall_ball.py:2153` exposes only fragment + bonus eligibility)
  although every era review analyzes serve depth. Cheap to add to
  reset info.
- **Silently swallowed failures** (each contradicts the repo's own
  no-silent-no-op rule): `artifacts.py:436/456` return `None` on
  OSError/JSONDecodeError, dropping resolved-model/warm-start
  provenance from `config.json`; `colab_setup.py:38-41` swallows the
  ICD-write `PermissionError` and `:75`'s health check compiles an
  *empty* model (exercises neither EGL nor GPU — a broken runtime
  passes setup and fails hours later at first render);
  `notebook_utils.py:722` (§2.3). Convention suggestion: any
  `except`-and-continue must print one `[module]`-prefixed line.
- `train.py`'s own operational prints lack the `[ClassName]` prefix
  convention the callbacks and notebook_utils follow.

## 8. Training-session corpus (Google Drive)

Inventory: 40 run directories across 7 families (TennisWall ×1,
WallBall ×12, WallBallBaseline ×12, DepthCurriculum ×7, Aligned ×2,
GoalRally ×4, TrueBaseline ×2), all SAC on Colab (T4→L4 era), ~20 h
per 6M-step run at 75–87 FPS. Outcomes match the repo's recorded
verdicts (classic WallBall solved to the episode cap; GoalRally seed 1
collapse documented; TrueBaseline the current best line; no
ball_balance/ball_bounce/humanoid runs were ever synced). Infra
observations for whoever builds the next sync/report tooling:

- **Two artifact layouts coexist** (pre/post ~Jul 19 restructure), and
  the earliest run uses `run_summary.txt`/`run_config.json` naming.
  The 0.14.0 legacy-fallback readers handle this, but any new Drive
  tooling must too.
- **Three aborted-run stubs** (`20260717_164059`, `20260717_165144`,
  `20260728_224706`) contain only `config.json` + `monitor/` and are
  indistinguishable from live runs without listing contents. A
  convention (e.g. an `ABORTED` marker file, or a sweep that renames
  stubs) would keep the corpus clean.
- **Three orphan TOMLs** in Drive `configs/` have no corresponding run
  family (`*_keepbuffer.toml`, `*_keepbuffer_5m.toml`,
  `*_stage4_escalator_warmstart.toml`) — drafted-never-run, worth a
  note or deletion.
- Config TOMLs carry **three different MIME types**, implying three
  upload paths over time.
- `git_sha` is null/"unknown" in every run before ~Jul 20 —
  the archaeology risk the provenance work fixed; the §2.3
  `tennis_curriculum._git_sha` finding is the remaining hole in that
  guarantee.
- **Caution for future Drive tooling**: title-based Drive searches for
  `config.json`/`stage_summary.txt` return files from an unrelated
  project sharing the Drive; always scope by folder id.
- Every recent run shows negative "final vs best" — `best_model.zip`
  is the deployable artifact each time, which the selection machinery
  already handles; noted here because PaddleTennis reporting should
  keep leading with best-window numbers, not final.

## 9. Low-severity appendix (not adversarially verified)

Forty-nine further findings were reported at low severity and not put
through verification; spot-checks suggest they are broadly accurate.
The most useful, one line each:

- `wall_ball.py:394` `min_force` is the one reward-critical kwarg the
  constructor never validates (NaN/negative silently disables edges).
- `ball_balance.py:46` reset noise is applied without joint-limit
  clipping or quaternion normalization (`slider_z` starts out of range
  in ~half of resets).
- `test_envs.py:444` BallBalance's sole termination rule (drop below
  z=0) is asserted nowhere.
- `monitor_log.py:97` `worker_id` from a lexicographic glob mismatches
  SB3 worker rank at ≥10 workers.
- `info_dict_eval.py:68` `eval_freq` documented as training steps but
  triggered on vec-env `n_calls` (×n_envs divergence).
- `callbacks/__init__.py:8` `PerformanceGatedEnvStagesCallback` is the
  one public callback not exported.
- `performance_gate.py:611` `finalize()` rewrites the stage history
  file twice back-to-back.
- `env_attr_schedule.py:143` re-applies the scheduled attribute to
  every worker on every step even when unchanged.
- `run_config.py:263` `reset_entropy_on_advance`/`entropy_reset_value`
  accept any type (no validation).
- `humanoid_tennis.py:780` rebuilds the action-name index dict per
  call; `robot_models.py:190` stand height bypasses the spec registry;
  `scripted_policies.py:299` hardcodes layout indices the published
  constants already expose; `:116` the wall-ball oracle uniquely skips
  obs-shape validation.
- `ladder_certification.py:802` `None` landing offset misaligns the
  report table exactly in the failing case.
- `notebook_utils.py:78/129` duplicate the Drive-root resolution block;
  `:333` `best_model_meta` missing from `_ARTIFACT_HINTS`; `:1697`
  docstring names the wrong preferred normalizer file.
- Starter-TOML comment drift beyond §5: `wall_ball.toml:3` steers new
  work to the historical bootstrap; the depth-curriculum starter still
  recommends the closed aligned arm and shows discredited
  `episodes = 10` probes.
- `smoke_wheel.py:133` the "copyable" check exercises tempfile, not
  `copy_starter_config`.
- Humanoid notebook install cell predates the sb3 notebook's
  `REPO_REF` pattern.

## 10. Suggested sequencing

1. **Now (small, high value)**: §2.1 validator keys + spec + provenance
   block (one commit, with the drift test); §2.2 video_record fixes;
   `final_info_eval` leak; court ticks; README recipe list + historical
   markers; smoke_wheel registry derivation; CI mypy extras +
   timeouts + cron.
2. **Before writing PaddleTennis env code**: the §3 extractions in
   dependency order — base-env guards/validators and piecewise mapping
   first, then `PaddleInterface`, then the events/geometry
   parameterization (`_tennis_events` optional humanoids,
   `CourtGeometry`), then the serve/launch module. Each is
   behavior-preserving with existing tests pinning the values.
3. **Alongside the first PaddleTennis test file**: `tests/conftest.py`
   and the `test_envs.py` split, so the new suites start on shared
   helpers.
4. **Before the first long PaddleTennis run**: KeyboardInterrupt
   salvage test, stop-reason persistence, `monitor_info_keywords`,
   colab_setup render probe — the unattended-run protections.
