# Repository review: bugs, cleanup and consolidation (2026-10-16)

Kind: **Review snapshot**, pinned to `main`@`b9585e2` (2026-10-05) and
conducted 2026-10-16.

## Method

- **Scoped reviewers.** 13 scoped reviewers each read their modules in
  full and reproduced their claims with scratch probes. The scopes:
  train, callbacks, PaddleTennis env, rules and events, notebooks,
  artifacts and DemoSAC, recipes and config, tools, consolidation,
  tests, WallBall and simple envs, humanoid, and the Drive training
  data.
- **Verification.** Each scope had its own adversarial verifier, which
  tried to refute every finding with an independent reproduction. A
  completeness critic checked for gaps across the scopes, and its
  additions were verified the same way.
- **Training-data cross-check.** The Google Drive run archive
  (`courtside-dynamics/training_runs`) was checked against the code.
  Recorded artifacts served as test oracles.
- **Known items.** Every finding was checked against
  `repo_review_20261005.md`, `rl_pipeline_review_20260828.md`,
  `DECISIONS.md` and the CHANGELOG. Items already listed there appear
  only with new evidence, marked *known* in the appendix.
- **Medium findings.** These were re-checked by hand at HEAD.
- **Stack.** Every run used a fresh dependency resolve on 2026-10-16:
  Python 3.13, SB3 2.9.0, gymnasium 1.4.0, MuJoCo 3.16.0, numpy 2.5.3
  and CPU torch.

## 1. Summary

**Baseline.**
- ruff and mypy are clean.
- pytest: 1276 passed, 1 skipped (no GL) and **2 failed**. Both
  failures are MuJoCo 3.16 drift in a humanoid test (§2.1).

**98 findings, all confirmed:**
- 51 bugs, 8 consolidation and 39 cleanup items.
- After the verifiers re-rated severity, none is high. Ten unique
  findings are medium.
- 11 overlap known items with new evidence.

**Past verdicts stand. Nothing found changes env streams, rewards or
rules-engine calls on the active task:**
- PaddleTennis, WallBall and BallBounce per-step digests (random and
  oracle play, 4,500 steps) are bit-identical between MuJoCo 3.12, the
  August training stack, and 3.16.
- Two concurrent seeded PaddleTennis `train()` runs produced
  byte-identical `eval_info.csv`, `evaluations.npz` and monitor files.
- In the 1M LT1 run's `eval_info.csv`, the reward decomposition sums
  exactly to `episode_reward_mean` at all 40 evaluations.

**Where the defects are.** They sit at the edges:
- best-model selection after a warm start;
- the interrupt salvage paths;
- run reporting: `stage_summary`, the plots and the campaign report;
- pre-flight validation that fires too late;
- dependency floors.

**Correction to an in-session note.** The PaddleTennis recipe uses
`early_stop_patience = 20` at a 2M-step budget. The earliest stop is
evaluation 40 of 80, so early stopping is live. The patience of 60 on
run `20260828_113136` was set at that launch. What is broken is the
stop *label* (§3.1).

## 2. Fix first: the medium findings

| # | Finding | Where | Impact class |
|---|---|---|---|
| 2.1 | MuJoCo 3.16 (released 2026-10-14) breaks the humanoid Stage-1 oracle. CI goes red on the next weekly run | `tests/test_humanoid_tennis_curriculum.py:522`, `pyproject.toml:26` | tooling |
| 2.2 | Paired `confirm_best` compares a candidate's confirmation block with the first best's *standard* block | `callbacks/info_dict_eval.py:1273` | era-boundary (selection only) |
| 2.3 | Interrupt salvage calls `gate.finalize()` unguarded, before saving the model | `training/train.py:2617` | bit-identical |
| 2.4 | An interrupt inside an open control step wedges the event sampler | `envs/_tennis_events.py:298` | bit-identical |
| 2.5 | `stage_summary` "Final vs best" and its "deploy best_model.zip" flag use the reward-stream argmax | `training/artifacts.py:1163` | tooling |
| 2.6 | Campaign report `unsafe_terminations` misses `unsafe_physics` endings | `notebook_utils.py:2363` | tooling |
| 2.7 | `eval_headline.png` omits the recipe's headline metric for PaddleTennis and humanoid runs | `notebook_utils.py:563` | tooling |
| 2.8 | TOML gate values are validated only after `config.json` is written and the old CSVs are rotated | `run_config.py:200` | bit-identical |
| 2.9 | DemoSAC never checks the demo library's recorded `env_kwargs` against the training env (*known*, LD1′) | `training/demo_sac.py:370` | bit-identical |
| 2.10 | Humanoid Stage 1/2 promotion can never pass its canonical-spec check | `training/tennis_curriculum.py:942` | era-boundary (dormant) |

### 2.1 MuJoCo 3.16 drift (tests-1, humanoid-2)

**What fails.**
`test_physical_stage1_oracle_lands_a_mirrored_target_return[0,1]`
asserts an exact step count of 129. Under 3.16 the oracle takes 138
steps and misses the target.
- The test passes on the August stack (MuJoCo 3.12, numpy 2.1.3,
  gymnasium 1.3).
- The last green CI run was 2026-10-12, before 3.16 shipped.
- The humanoid notebook's Stage-1 preflight runs the same oracle, so
  it aborts the same way.

**Paddle is unaffected.** Its digests are bit-identical across 3.12 and
3.16 (§1).

**Fix.** Do not cap `mujoco` globally: Colab resolves it fresh, and the
paddle envs are unaffected. Either:
- move the humanoid code to the planned archive (2026-10-05 §7.1); or
- turn the exact-step assertion into a behavioral one (the target is
  landed) and record the MuJoCo version the oracle was calibrated on.

### 2.2 `confirm_best` compares across seed blocks until the first confirmation (callbacks-1, training-data-1)

**The cause.** Under paired evaluation the standard and confirmation
batches each replay their own fixed seed block. The first best is
banked without a confirmation score. Until a later candidate is
accepted, every candidate's *confirmation*-block score is compared with
that first best's *standard*-block score (`info_dict_eval.py:1273-1279`;
the code comment says so).

**Why it matters.** Block difficulty differs, and paired policies are
deterministic on a fixed block, so the gap is a persistent bias rather
than noise. A warm start makes it worse: evaluation 1 is the strong
source policy.
- A candidate that dominates the first best on *both* blocks is
  rejected whenever the confirmation block is harder by more than the
  candidate's margin.
- It then re-rolls a 30 × 1,500-step confirmation at every evaluation.

**Evidence.**
- The verifier reproduced 10 of 10 confirmations rolled and lost by
  such a candidate.
- A model-based estimate puts the rate at about 30% of warm-start runs
  permanently rejecting a +0.5 reward gain when conversions tie.
- Both 2026-08-28 runs kept evaluation 1 (step 25,000 =
  `learning_starts`, the untouched warm-start source) as `best_model`.
  Replaying the 1M run's data under HEAD's per-key deltas also selects
  25k, because no later evaluation reaches it on the primary key. The
  bias therefore did not change that run's verdict, but LD1′ is a warm
  start.

**Fix.** When the first best is banked with `confirm_best` and
`eval_seed` set, roll the confirmation block for it once and store
`_best_confirm_score`. Also:
- add a test where a candidate that dominates on both blocks is
  accepted at its first opportunity;
- record in `best_model_meta` when the selected checkpoint predates
  the first gradient update.

### 2.3 Interrupt salvage drops everything on one I/O error (train-2)

**The order today.** The `KeyboardInterrupt` handler calls
`gate_callback.finalize()` first, and only then saves `final_model`,
`vec_normalize`, the closing eval and the summary.
- `finalize()` does disk I/O on the Drive mount: it copies the best
  triple and reads `best_model_meta.json`.
- If that I/O raises, `train()` raises out of the handler and nothing
  is saved.

**Contrast.** The crash path (`_salvage_crashed_run`) saves the model
first and isolates each step. The verifier injected an OSError into
`archive_best`: the crash path kept all three artifacts, while the
interrupt path kept none.

**Fix.** Match the crash path: save the model and normalizer first,
then wrap `finalize()` and the closing evaluation in
log-and-continue blocks.

### 2.4 An interrupted step wedges the event sampler (rules-events-1)

**What wedges.** `SubstepTennisEventSampler.reset()` raises if a
control step is open, and neither env drains the sampler when an
exception escapes between `begin_control_step` and `end_control_step`.
A Colab stop (`KeyboardInterrupt`) lands mid-step most of the time.
After that, every `reset()` and `step()` on that env raises.

**Why it compounds 2.3.** `train()`'s interrupt salvage calls
`evaluate_policy` on the same `eval_env` the reward `EvalCallback`
steps, so the salvage itself crashes.

**Fix (bit-identical).** In `reset()`, discard an open control step
instead of raising. A healthy trajectory never reaches `reset()` with
an open step, and the verifier confirmed a recovered env replays a
fresh env's stream byte for byte.

### 2.5 "Final vs best" points at a checkpoint nobody saved (training-data-2, artifacts-demo-3)

**Two problems with the comparison.** `write_run_summary` compares the
closing eval with the argmax of `evaluations.npz`.
- **Wrong checkpoint.** Under headline selection, `best_model.zip` is
  the task-metric pick, not that argmax. The flag "policy regressed;
  deploy best_model.zip" therefore recommends a checkpoint other than
  the one compared. 2026-10-05 §5b #9 fixed the same drift in the plot
  but not here.
- **False alarms.** The batch moved PaddleTennis to a 5-episode reward
  stream. The maximum over 80–120 such batches is biased: a Monte
  Carlo with a flat policy flags a regression in about 20% of runs (0%
  with the old 30-episode stream).

**Fix.** Compare against `best_model_meta["metrics"]` and the selected
step. Then derive the noise floor from the closing eval, or record the
selecting batch's reward std.

### 2.6 Campaign `unsafe_terminations` undercounts (notebook-utils-1, consolidation-2)

**The drift.** `paddle_campaign_metrics` counts only
`termination == "nonfinite_state"`. The env treats both
`NONFINITE_STATE` and `UNSAFE_PHYSICS` as unsafe
(`paddle_tennis.py:134`), and the probe tools count both.

**Why it matters.** `score_paddle_stage` writes this field into
`campaign_gate.json` and `campaign_final_report.json`, which the
pre-registration's "zero unsafe terminations" criterion is read
against.

**Fix.** Count the env's own unsafe set through a public alias, and
switch `tools/paddle_tennis_npoint_probe.py:623` to the same alias.

### 2.7 `eval_headline.png` has no headline (consolidation-1, notebook-utils-2)

**The cause.** Panels are paged by a hard-coded list of WallBall stem
prefixes.

**The effect on PaddleTennis.** The k≥2 headline
(`episode_rally_returns_a`) and the conversion counters land on
`eval_diagnostics.png`. The headline page shows only reward and
episode length, the reward-led reading the selection design exists to
prevent. Humanoid `stage_success` is affected the same way.

**Fix.** Route panels by the `headline_key`, `success_key` and
selection keys recorded in `config.json`, matching by prefix. Keep the
stem list as a fallback.

### 2.8 Gate TOML validation happens too late (recipes-config-1)

**The gap.** `_validate_performance_gate` only type-checks. The value
rules (rule name, finite threshold, pause ≥ 0, flag dependencies,
`stage_eval_budget` ≥ `sustain_evals`) live only in the gate callback's
constructor. That constructor runs after `write_run_config` and after
`_check_log_dir_reuse` has rotated the previous attempt's CSVs.

**Evidence.** The verifier fed in 8 malformed gates, and all 8 passed
load, build and the pre-flight checks. The resulting error does not
name the file.

**Fix.** Extract a `validate_gate_spec()` into `performance_gate.py`
and call it from both places.

### 2.9 DemoSAC demo-library compatibility (artifacts-demo-1, known)

**The gap.** The library records the `env_kwargs` it was harvested
under (the reward definition), but `DemoSAC` never compares them with
the training env. A library harvested under different shaping weights
loads silently. This is already on the LD1′ list. The new evidence is
that the recorded kwargs are present and the check is cheap.

### 2.10 Humanoid Stage 1/2 promotion is unreachable (humanoid-1)

**The cause.** `_canonical_evaluation_spec_id` pins
`valid_hit_shaping = 0.0` for every stage. The 0.16.0 Stage 1/2
recipes use 0.25, so the spec ID resolves to `None` and
`assess_curriculum_promotion` raises at Stage 1. This happens after
about 7 CPU hours of training, with no promotion report written.

**Fix.** Only needed if the humanoid code is not archived: add a
per-stage shaping table next to `_CANONICAL_EPISODE_LENGTHS`, with one
helper for the spec ID on both the producer and consumer side.

## 3. Low-severity bugs worth fixing

### 3.1 Selection, stopping and reporting

**Wrong stop labels (train-3, training-data-3).** A guard firing on the
last scheduled evaluation labels a run that trained its full budget
`Stop reason: early_stop_patience`, as both 2026-08-28 runs show.
Nothing warns when a budget is too small for patience to fire at all
(the earliest stop is evaluation 2 × patience).

**Reward-path early stop drifted from the headline path (train-1):**
- it stops after N+1 evaluations instead of N;
- it ignores the `env_attr_schedules` warm-up;
- it records no stop reason;
- BallBalance's patience 20 can never fire at its 1M budget.

**Artifacts that hide what happened:**
- **training-data-6.** Rejected `confirm_best` batches are never
  persisted, despite the CHANGELOG.
- **training-data-4.** The best-checkpoint block renders the per-step
  `rew_*` components as terminal values, so penalties read 0.00.
- **training-data-5.** An empty videos directory counts as present.
  The 1M LT1 run lost all four periodic videos and its CSVs with no
  warning.
- **artifacts-demo-4.** The summary's Artifacts index omits the
  conditional artifacts the run wrote.

**Config pitfalls:**
- **train-6.** Headline-only knobs are silently inert without a
  `headline_key`, while their newer siblings are refused.
- **callbacks-2.** A bare string where a key sequence is expected is
  split into one-letter keys.

### 3.2 Run setup, salvage and dependencies

**Dependencies:**
- **critic-1.** The `gymnasium>=1.1` floor admits 1.1.0, under which
  every courtside env fails to construct, because its `MujocoEnv`
  asserts a four-mode `render_modes` list. 1.1.1 and later work. Raise
  the floor to `>=1.1.1`.
- **critic-2, critic-3.** Algorithm overrides fail late or with a
  wrong hint. See the PPO plan (§7).

**Colab and log directories:**
- **critic-4.** `disconnect_runtime` tears down the Colab VM without
  `drive.flush_and_unmount()`, so the last Drive writes can be lost:
  the campaign record and report, and the final leg's artifacts.
- **train-5.** `reuse_log_dir=True` leaves the previous attempt's
  monitor CSVs and TensorBoard events, which are read back into the
  new attempt's curves.

**Integrity checks:**
- **train-4** (known). `train()`'s warm start does not verify the
  best-model/normalizer pair against `best_model_meta.json`. Only the
  campaign notebook does.
- **callbacks-3, consolidation-5.** `csv_header = "none"` with a
  recipe `info_row_fn` misaligns every milestone CSV.

### 3.3 Environments

**WallBall and the simple envs:**
- **env-wallball-simple-2 (era-boundary).** `ball_balance.xml` has no
  `<compiler angle="radian"/>`.
  - The three tray hinge limits compile to ±0.25° (0.0044 rad), not
    the evidently intended ±0.25 rad (about 14°).
  - The tray is effectively locked flat. That is why the zero action
    already "balances" about 470 steps.
  - Fix it only with a declared BallBalance version.
- **env-wallball-simple-3.** When `gym.make` passes a larger
  `episode_len`, the registered `TimeLimit` cuts the episode first, so
  WallBall skips its terminal refund.
- **env-wallball-simple-4** (known). WallBall still lacks the pre-step
  physics guard that the CHANGELOG says every sibling env has.

**PaddleTennis and the rules engine:**
- **env-paddle-1.** `PaddleCourtServe` accepts nonfinite fields. The
  failure surfaces at reset as an unrelated OverflowError.
- **env-paddle-2.** The `serve_side` contract drifted from the humanoid
  env it mirrors: a label at reset, an int at step, and int options are
  refused.
- **rules-events-2.** The substep event loop exists in three copies.
  The probe scene's copy dropped the pre-step scan and the FatalError
  catch.
- **artifacts-demo-2.** Diagnosis ender attribution lost its side
  prefix for own-side landings after the `failed_to_cross` relabel.

### 3.4 Notebooks and tools

**Notebooks:**
- **notebook-utils-3.** The campaign "Reports & curves" cell raises
  KeyError on a leg that is not complete.
- **notebook-utils-5.** `_in_colab()` raises ModuleNotFoundError
  instead of returning False when no `google` package is installed.
- **env-wallball-simple-1.** The legacy WallBall constructor-defaults
  mirror lacks `return_shaping_scale`, so `evaluate_best_wall_ball`
  rejects older runs.

**Tools:**
- **tools-1.** The k2 harvest records opportunities from side-A-serving
  points. The drill loader refuses those, and `k2_step0` silently
  scores them.
- **tools-3.** The PT2 spectrum windows include the closing step, so
  they do not mirror PT1 as documented.
- **Seed-ledger gaps (tools-5, tools-2, consolidation-3).** The WallBall
  chapter's burned blocks are missing from the ledger, as are the P0–P4
  calibration blocks. The P3 sweep and `depth_stage_sweep` bypass it.
- **tools-6.** `depth_stage_sweep` opens its output paths only after
  the expensive run, and its release writer lacks the mkdir.
- **consolidation-4.** The three `_wilson_interval` copies have drifted
  in clamping and input validation.

### 3.5 Tests

- **tests-3.** `test_csv_logger_survives_learn` cannot detect the
  regression it names: `progress.csv` exists, empty, either way.
- **tests-2** (known). Both P4 mirror-identity tests stop before the
  first racket contact.

## 4. Consolidation

### 4.1 New since the existing registers

- **sha256 pin rule.** Lowercase hex, 8–64 characters, prefix match,
  hand-mirrored in 4–5 places (train-7, consolidation-8).
- **Nested-config defaults** re-spelled at every consumer:
  `checkpoint_diagnosis` 6 times, `performance_gate` 3 times
  (consolidation-7).
- **The campaign notebook** re-implements `build_train_config`'s
  `[env]` layering and eager env probe (consolidation-9).
- **The physical observation block,** its names and its accessors
  exist in both the env and `PaddleCourtScene`; only the scene copy is
  P4-certified (env-paddle-3).
- **Small pairs:**
  - selection validators duplicated between `train.py` and
    `InfoDictEvalCallback` (callbacks-6);
  - observation-name validation in two functions (artifacts-demo-6);
  - the video stat-sync block in two callbacks (callbacks-5).
- **Copies that drifted** (reported as bugs above): the substep event
  loop (rules-events-2), `_wilson_interval` (consolidation-4) and the
  unsafe-termination count (2.6).

### 4.2 Status of the existing registers

**`repo_review_20261005.md` §6.** No item is done, and one is partly
done: the seed ledger was extracted to `tools/_seed_ledger.py`, with
the gaps listed in §3.4. Several estimates have moved:

| Register item | Now |
|---|---|
| `notebook_utils.py` | grew from 2,760 to 3,206 lines |
| `train()` | still about 900 lines |
| normalizer loaders | at least 6 load sites, not 3 |
| `_wilson_interval` | 3 copies, now drifted |
| `_sha256` / git helpers | 5 copies in src and 3 in tools |
| `_draw_serve` vs `PaddleCourtScene.serve` | identical, no drift |

**`rl_pipeline_review_20260828.md` §6:**
- SM1, the seed ledger, is done, with gaps.
- SM2–SM5 and NT1–NT3 are open.
- Correction: the `video_record._scalar_info_keys` re-export is not
  dead; `video_record.py:162` calls it.

## 5. Cleanup

**Stale or wrong docs and comments.** These are worth one sweep:
- **Paddle stack (env-paddle-4/5/6).** It still describes a one-point
  episode and calls the module a prototype. `policy.predict(obs)` in
  the opponent-wiring recipe skips the VecNormalize transform every
  checkpoint needs.
- **`TerminationReason` (rules-events-3).** "Ordered by fault priority"
  is no longer true; the integers are a persisted CSV wire format.
- **`envs/__init__` export order (rules-events-5).**
- **Recipe comments (recipes-config-4, training-data-7).**
- **`run_config_file_spec.md` and the starter README (recipes-config-6).**
  Neither was updated for the batch.
- **`DECISIONS.md`:**
  - callbacks-9 still says the info evaluator never re-seeds.
  - env-wallball-simple-5 marks the WallBall normalization exclusion
    implemented, but no recipe sets it.
- **Seed-ledger citations (tools-4, tests-4).** All seven `DECISIONS.md`
  line citations are stale by 74 lines, and the test pins the stale
  strings.
- **WallBall docstrings (env-wallball-simple-6).**
- **`sb3_training` markdown (notebook-utils-8).**
- **The `record_best_model_video` docstring (consolidation-10).**
- **The package `__init__` history (recipes-config-7).**

**Dead code:**
- the grouped vulture results (consolidation-6);
- the `_collect_metrics` last-seen fallback (callbacks-4);
- the `stopped` flag in `_step_mujoco_simulation` (env-paddle-7);
- the `--oracle`/`--stub-oracle` flags, which are parsed but never read
  (tools-7);
- `WindowRecord.cmd_to_hit` (tools-8);
- `volley_probe.evaluate_criteria` (tools-9);
- the humanoid leftovers (humanoid-5).

**CI and packaging:**
- **CUDA torch in CI (recipes-config-3).** Every CI install pulls the
  CUDA torch build, about 2.7 GB of nvidia wheels and about 1m40s each.
  Use the CPU index.
- **Slow `test_train.py` (tests-5).** About 80% of its 152 s goes to
  SAC gradient steps no assertion depends on.
- **License metadata (recipes-config-2).** The license is declared as a
  TOML table plus a classifier. setuptools announces removal by
  2027-02-18, which would break every install.
- **Tautological assertions (tests-6),** including in CI's only GL
  render smoke test.

## 6. What the training data says

The Drive archive holds:
- about 28 WallBall run dirs (07-14 → 08-02);
- 20 PaddleTennis run dirs (08-03 → 08-29);
- the 2026-08-30 diagnostic streams.

Nothing has trained since 2026-08-29. The 2026-10 Drive activity
belongs to a different project (mesozoic-labs).

**Answers to the leads raised during this review:**
- **Stop reason.** Both 2026-08-28 runs stopped "by patience" at
  exactly their final evaluation: 2 × 20 = 40 of 40, and 2 × 60 = 120
  of 120. HEAD prints the same label (§3.1).
- **Best = step 25,000.** Step 25,000 is `learning_starts`, i.e. the
  warm-start source before any gradient update. Corroborating
  evidence:
  - `best_model.zip` is 1.6 MB, against 3.55 MB for the checkpoints,
    because it has no optimizer moments;
  - `n_updates` equals total − 25k exactly.
- **Final vs best.** −0.863 on the 3M companion run is the winner's
  curse: a flat-policy simulation predicts −0.84. The 1M run's −2.465
  is a real drop, consistent with its booked T1-FAIL.

**Checks that came up clean:**
- phase fractions sum to 1;
- the reward decomposition is exact;
- there are no NaNs;
- throughput is consistent (45.2 and 40.3 FPS);
- the provenance chain from parent `best_model_meta` to the
  children's expected sha256 values is intact;
- checkpoint and diagnosis cadence matches the config.

**Already fixed since the run SHAs:** the SB3 logger was never closed
(`b43b6ba`), and `best_model_meta` metrics were missing (`125c4ed`).

## 7. Related studies from this session

### PPO support audit

PPO runs end to end:
- time-limit truncation bootstraps correctly in every env;
- PPO→PPO warm start copies the weights exactly;
- BallBalance learned to 750/750 in about 5.5 CPU-minutes.

But it is not first-class. 20 unique findings were confirmed. The main
ones:
- the evaluation-counted guards can stop a PPO run after 0–7 policy
  updates (high);
- PaddleTennis refuses `algo="PPO"`, because its SAC-only
  `model_kwargs` sit in one flat dict;
- `quick_test` can finish "completed" with zero updates;
- the humanoid PPO recipes run outside the trust region
  (`approx_kl` 0.22–1.71);
- the reward normalizer always uses γ = 0.99;
- several loaders hard-code `SAC.load`;
- the campaign notebook rejects PPO;
- there is no end-to-end PPO test.

A 14-step plan (about 40 h) fixes these behind a SAC freeze harness,
so SAC paths stay bit-identical.

### Architecture study (toward humanoid doubles)

The recommendation is extraction behind golden digests, not a rewrite:
- freeze today's task as a versioned ID;
- build a per-agent court core, a player/team referee, observations by
  name, one racket-command interface, and MjSpec scene assembly, each
  only when its rung needs it;
- train a brain/body ladder: the tennis "brain" on CPU paddle courts
  through paddle doubles, and the G1 "body" once on a GPU stack after a
  throughput spike;
- make eras explicit as (task version, eval-protocol version), with a
  standalone re-scoring harness.

The full texts of both studies can be committed as `docs/` pages.

## 8. Suggested batches

| Batch | Contents | Impact |
|---|---|---|
| **A. Robustness and reporting** (about 1–2 days) | 2.3, 2.4, 2.5, 2.6, 2.7, 2.8; train-3/training-data-3 (stop label and an inert-patience warning); training-data-5 (empty videos); critic-4 (Drive flush); train-5; callbacks-2/3 | bit-identical for env streams and selection |
| **B. Dependency hygiene** (about half a day) | 2.1 (humanoid test, or archive); critic-1 (`gymnasium>=1.1.1`); recipes-config-3 (CPU torch in CI); recipes-config-2 (license metadata) | tooling |
| **C. Selection semantics** (needs a maintainer decision) | 2.2 (`confirm_best` first best); train-1 (one early-stop helper for both paths); training-data-6 (persist confirmation batches) | era-boundary for selection; env untouched |
| **D. Before LD1′** | 2.9 (demo-library kwargs check); train-4 (warm-start pair check) | bit-identical |
| **E. Declared env versions** | env-wallball-simple-2 (BallBalance radians); env-wallball-simple-3 (TimeLimit) | era-boundary |
| **F. Consolidation and cleanup** | §4 and §5. Do these opportunistically with the architecture migration's extraction steps, not as a standalone pass | bit-identical |

## Appendix: all findings

Severity is as re-rated by the verifiers. "Known" marks an overlap
with an earlier review, the DECISIONS or the CHANGELOG, reported here
with new evidence.

| ID | Kind | Severity | Known | Impact | Location | Finding |
|---|---|---|---|---|---|---|
| artifacts-demo-1 | bug | medium | yes | bit-identical | `src/courtside_dynamics/training/demo_sac.py:370` | DemoSAC never checks the demo library's recorded env_kwargs (reward definition) against the training env |
| callbacks-1 | bug | medium |  | era-boundary | `src/courtside_dynamics/callbacks/info_dict_eval.py:1275` | Paired confirm_best still compares the confirmation block with the standard-block best until a confirmation is first accepted (every run start and every stage reset) |
| consolidation-1 | bug | medium |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:563` | eval_headline.png omits the recipe's headline metric for PaddleTennis (and humanoid) runs |
| humanoid-1 | bug | medium |  | era-boundary | `src/courtside_dynamics/training/tennis_curriculum.py:942` | Stage 1/2 promotion can never pass the canonical-spec check: the spec pins valid_hit_shaping=0.0 but the 0.16.0 recipes use 0.25 |
| humanoid-2 | bug | medium |  | tooling-only | `pyproject.toml:26` | MuJoCo 3.16 (resolved by the unbounded mujoco>=3.3 pin) breaks the Stage-1 oracle: its test fails and the notebook's Stage 1 preflight aborts |
| notebook-utils-1 | bug | medium |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:2363` | Campaign report's unsafe_terminations misses unsafe_physics endings (drifted from the probe tool's same-named count) |
| recipes-config-1 | bug | medium |  | bit-identical | `src/courtside_dynamics/run_config.py:200` | TOML gate validator is a drifted partial copy of the gate callback's checks; malformed gate values pass every pre-output check and fail only after config.json is written and the previous attempt's CSVs are rotated |
| rules-events-1 | bug | medium |  | bit-identical | `src/courtside_dynamics/envs/_tennis_events.py:298` | An interrupt inside an open control step wedges the event sampler: every later reset()/step() raises, and train()'s interrupt salvage crashes in evaluate_policy |
| tests-1 | bug | medium |  | tooling-only | `tests/test_humanoid_tennis_curriculum.py:522` | Dormant humanoid oracle test pins an exact MuJoCo step count; fails on mujoco 3.16.0, which the unpinned floor now resolves (CI goes red on the next fresh resolve) |
| train-2 | bug | medium |  | bit-identical | `src/courtside_dynamics/training/train.py:2617` | KeyboardInterrupt salvage calls gate.finalize() unguarded before saving final_model, so one I/O error loses the model, normalizer and summary |
| training-data-1 | bug | medium | yes | era-boundary | `src/courtside_dynamics/callbacks/info_dict_eval.py:1275` | Paired confirm_best compares a candidate's confirmation block against the first best's standard block until one confirmation is accepted, which can pin best_model.zip to eval #1 and re-roll a 30-episode confirmation at every eval |
| training-data-2 | bug | medium |  | tooling-only | `src/courtside_dynamics/training/artifacts.py:1163` | stage_summary 'Final vs best' and its 'policy regressed; deploy best_model.zip' flag use the reward-stream argmax: wrong checkpoint under headline selection, and a ~20% false-alarm rate with HEAD's 5-episode reward stream |
| artifacts-demo-2 | bug | low |  | tooling-only | `src/courtside_dynamics/training/paddle_diagnosis.py:149` | Diagnosis ender attribution lost its side prefix for own-side landings after the failed_to_cross relabel |
| artifacts-demo-3 | bug | low |  | bit-identical | `src/courtside_dynamics/training/artifacts.py:1163` | stage_summary tells the operator to 'deploy best_model.zip' for a reward regression that best_model.zip does not address under headline selection |
| artifacts-demo-4 | bug | low |  | bit-identical | `src/courtside_dynamics/training/artifacts.py:1396` | stage_summary 'Artifacts' index omits every conditional artifact the run actually wrote (diagnosis reports, eval_info_final.csv, run_config.toml, ladder certification, stage_bests) |
| callbacks-2 | bug | low |  | bit-identical | `src/courtside_dynamics/callbacks/info_dict_eval.py:405` | Key-sequence parameters accept a bare string and silently split it into one-letter keys; _string_keys guards only degenerate_flat_keys |
| callbacks-3 | bug | low |  | tooling-only | `src/courtside_dynamics/callbacks/video_record.py:118` | VideoRecordCallback checks csv_header-without-info_row_fn but not the reverse: a recipe's info_row_fn with csv_header "none" writes a 3-column header over 38-column rows |
| consolidation-2 | bug | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:2363` | paddle_campaign_metrics.unsafe_terminations counts only nonfinite_state endings, unlike the other unsafe counters |
| consolidation-3 | bug | low | yes | tooling-only | `tools/paddle_tennis_probes.py:520` | Seed-ledger fix incomplete: the P3 sweep path skips the ledger, and the burned P0-P4 calibration blocks are missing from it |
| consolidation-4 | bug | low | yes | bit-identical | `src/courtside_dynamics/training/ladder_certification.py:377` | Three _wilson_interval copies have drifted (clamping and input validation) |
| consolidation-5 | bug | low |  | bit-identical | `src/courtside_dynamics/callbacks/video_record.py:118` | A TOML can clear csv_header while the recipe's info_row_fn stays, misaligning every milestone CSV |
| critic-1 | bug | low | yes | tooling-only | `pyproject.toml:29` | gymnasium>=1.1 floor admits 1.1.0, whose MujocoEnv asserts a 4-mode render_modes list: every courtside env fails to construct |
| critic-2 | bug | low |  | bit-identical | `src/courtside_dynamics/recipes.py:1937` | Notebook ALGO override to PPO fails for PaddleTennis and WallBallBootstrap: recipe SAC-only model_kwargs leak in, and the error suggests 'target_kl' |
| critic-3 | bug | low |  | bit-identical | `src/courtside_dynamics/callbacks/performance_gate.py:467` | Gate warm-up model-compatibility checks run only inside learn(): PPO, or a fixed ent_coef, on the depth-curriculum recipes crashes about 30 s in, after writing a 'crashed' run dir with an untrained final_model.zip |
| critic-4 | bug | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:269` | disconnect_runtime tears down the Colab VM without drive.flush_and_unmount(), so the last Drive writes (campaign record/report, final leg artifacts) can be lost |
| env-paddle-1 | bug | low |  | bit-identical | `src/courtside_dynamics/envs/_paddle_court.py:104` | PaddleCourtServe accepts nonfinite lateral/noise fields; failure surfaces at reset as an unrelated numpy OverflowError |
| env-paddle-2 | bug | low |  | bit-identical | `src/courtside_dynamics/envs/paddle_tennis.py:1698` | serve_side contract drifted from the humanoid env it claims to mirror: reset info is a label, step info an int, and int options are refused |
| env-wallball-simple-1 | bug | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:969` | _LEGACY_WALL_BALL_CONSTRUCTOR_DEFAULTS mirror drifted: missing return_shaping_scale, so evaluate_best_wall_ball rejects every run recorded before 2026-07-27 |
| env-wallball-simple-2 | bug | low |  | era-boundary | `src/courtside_dynamics/assets/ball_balance.xml:14` | ball_balance.xml has no <compiler angle="radian"/>: tray tilt limits compile to +/-0.25 degrees and the 1000-gain actuators drive through them by up to 55x |
| env-wallball-simple-3 | bug | low |  | bit-identical | `src/courtside_dynamics/__init__.py:141` | Registered TimeLimit pre-empts the envs' own truncation when gym.make passes a larger episode_len: WallBall skips its terminal refund and term_timeout |
| env-wallball-simple-4 | bug | low | yes | bit-identical | `src/courtside_dynamics/envs/wall_ball.py:1328` | WallBall still lacks the pre-step physics-state guard, yet the CHANGELOG and ball_balance.py now say every sibling env carries it |
| notebook-utils-2 | bug | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:563` | plot_eval_info puts the PaddleTennis headline on the diagnostics page and splits distribution metrics into misnamed panels |
| notebook-utils-3 | bug | low |  | tooling-only | `notebooks/paddle_tennis_campaign.ipynb:894` | Campaign 'Reports & curves' cell raises KeyError on any leg that is not complete |
| notebook-utils-5 | bug | low |  | tooling-only | `src/courtside_dynamics/colab_setup.py:28` | _in_colab() raises ModuleNotFoundError instead of returning False when no `google` package is installed |
| rules-events-2 | bug | low |  | tooling-only | `src/courtside_dynamics/envs/_paddle_court.py:330` | The substep event-sampling loop exists in three copies; the probe scene's copy drifted (no pre-step scan, no FatalError catch), so a blown-up state is silently teleported instead of ending the point |
| tests-2 | bug | low | yes | tooling-only | `tests/test_paddle_tennis.py:173` | Both P4 mirror-identity tests stop before the first racket contact, so contact asymmetry between sides A and B is untested despite their 'contact-rich physics' claims |
| tests-3 | bug | low |  | tooling-only | `tests/test_train.py:251` | test_csv_logger_survives_learn cannot detect the regression it names: progress.csv exists (empty) whether or not learn() kept train()'s logger |
| tools-1 | bug | low |  | tooling-only | `tools/paddle_tennis_k2_harvest.py:231` | k2 harvest records k=2 opportunities from side-A-serving points, which the drill loader refuses and k2_step0 silently scores |
| tools-2 | bug | low | yes | tooling-only | `tools/paddle_tennis_probes.py:519` | 'Every probe consults the seed ledger' is false: the P3 sweep and depth_stage_sweep draw any --seed-start unguarded |
| tools-3 | bug | low |  | tooling-only | `tools/paddle_tennis_command_spectrum_analysis.py:130` | PT2 spectrum tool's post-swing windows include the closing step, so they do not mirror PT1 as documented |
| tools-5 | bug | low |  | tooling-only | `tools/_seed_ledger.py:47` | Seed ledger claims every burned block but omits the WallBall chapter's burned blocks |
| tools-6 | bug | low |  | tooling-only | `tools/depth_stage_sweep.py:1099` | Output paths are opened only after the expensive run; depth_stage_sweep's release writer lacks the mkdir its candidate writer has |
| train-1 | bug | low |  | era-boundary | `src/courtside_dynamics/training/train.py:2037` | Reward-selection early stop drifts from the headline path: stops after N+1 evals, ignores env_attr_schedules, records no stop_reason; BallBalance's patience can never fire |
| train-3 | bug | low |  | bit-identical | `src/courtside_dynamics/training/train.py:2665` | A guard firing on the last scheduled eval labels a full-budget run with 'Stop reason: early_stop_patience' |
| train-4 | bug | low | yes | bit-identical | `src/courtside_dynamics/training/train.py:1720` | train()'s warm start never checks that best_model.zip and best_vec_normalize.pkl are the pair best_model_meta.json bound; only the campaign notebook does |
| train-5 | bug | low |  | bit-identical | `src/courtside_dynamics/training/train.py:1315` | reuse_log_dir=True leaves the previous attempt's monitor CSVs and TensorBoard events, which are read back into the new attempt's curves and summary |
| train-6 | bug | low |  | bit-identical | `src/courtside_dynamics/training/train.py:1018` | Headline-only selection and stop knobs are silently inert without headline_key, while their new siblings are refused |
| training-data-3 | bug | low |  | tooling-only | `src/courtside_dynamics/training/artifacts.py:1085` | Stop reason 'early_stop_patience' is recorded for runs that trained their full budget, while a reward-selected early stop records no reason |
| training-data-4 | bug | low |  | tooling-only | `src/courtside_dynamics/training/artifacts.py:1340` | Best-checkpoint block renders per-step reward components (rew_*) as terminal ep-mean/last-episode/max, which read 0.00 for penalties; the per-episode dose (_ep_sum_mean) is never shown |
| training-data-5 | bug | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:520` | check_run_artifacts and the summary's Artifacts section count an empty videos directory as present; LT1 lost all four periodic videos with no durable trace |
| training-data-6 | bug | low |  | tooling-only | `src/courtside_dynamics/callbacks/info_dict_eval.py:1260` | confirm_best batches are never persisted: a rejected candidate leaves no trace, despite the CHANGELOG saying the second batch is now observable |
| artifacts-demo-6 | consolidation | low |  | bit-identical | `src/courtside_dynamics/training/artifacts.py:184` | Observation-name validation is duplicated between env_observation_names and vec_env_observation_names |
| consolidation-7 | consolidation | low |  | bit-identical | `src/courtside_dynamics/training/train.py:2099` | Defaults for nested config mappings are re-spelled at every consumer (checkpoint_diagnosis 6×, performance_gate 3×) |
| consolidation-8 | consolidation | low |  | bit-identical | `src/courtside_dynamics/training/train.py:219` | The sha256 pin rule (lowercase hex, 8-64 chars, prefix match) is implemented 4× for validation and 4× for matching |
| consolidation-9 | consolidation | low |  | bit-identical | `notebooks/paddle_tennis_campaign.ipynb:0` | The campaign notebook re-implements build_train_config's [env]-table layering and eager env probe |
| env-paddle-3 | consolidation | low |  | bit-identical | `src/courtside_dynamics/envs/paddle_tennis.py:653` | Physical observation block, its names and its state accessors are implemented twice (env and PaddleCourtScene); only the scene copy is P4-certified |
| notebook-utils-6 | consolidation | low |  | tooling-only | `src/courtside_dynamics/notebook_utils.py:2011` | Best-model replay and the humanoid promotion loader skip the new pair check; record_best_model_video's docstring is stale |
| notebook-utils-7 | consolidation | low |  | tooling-only | `notebooks/humanoid_tennis_training.ipynb:341` | Humanoid notebook still hand-parses the stage_summary Status line that stage_summary_status() now owns |
| train-7 | consolidation | low |  | bit-identical | `src/courtside_dynamics/training/train.py:219` | The sha256 pin rule (lowercase hex, 8-64 chars, prefix match) is hand-mirrored in five places |
| artifacts-demo-5 | cleanup | low |  | tooling-only | `src/courtside_dynamics/training/monitor_log.py:106` | monitor_log's 'no finished episodes -> FileNotFoundError' contract never fires for real SB3 files; worker_id ordering is lexicographic |
| callbacks-4 | cleanup | low |  | bit-identical | `src/courtside_dynamics/callbacks/info_dict_eval.py:829` | Dead 'last seen step' fallback in _collect_metrics, with a stale video_length comment |
| callbacks-5 | cleanup | low |  | bit-identical | `src/courtside_dynamics/callbacks/video_record.py:252` | Video sync-failure message gives the wrong fallback stats; the sync-with-fallback block is duplicated across two callbacks |
| callbacks-6 | cleanup | low |  | bit-identical | `src/courtside_dynamics/training/train.py:1028` | Selection-config validators are duplicated between train._validate_evaluation_config and InfoDictEvalCallback.__init__; the eval_seed copies drifted, and train imports three private helpers |
| callbacks-7 | cleanup | low |  | bit-identical | `src/courtside_dynamics/callbacks/env_attr_schedule.py:137` | Schedule and gate claim their start value or stage 0 is in force from the start, but SB3 resets the training envs before on_training_start |
| callbacks-8 | cleanup | low |  | bit-identical | `src/courtside_dynamics/callbacks/info_dict_eval.py:969` | With eval_verbose, the confirm_best batch prints a heartbeat line indistinguishable from the selection batch's |
| callbacks-9 | cleanup | low |  | docs-only | `docs/DECISIONS.md:872` | DECISIONS still says the info evaluator never re-seeds ('Unpaired evaluation ... open (P1)') |
| consolidation-10 | cleanup | low |  | docs-only | `src/courtside_dynamics/notebook_utils.py:2014` | Stale docstring: record_best_model_video describes a vec_normalize.pkl contract it no longer follows |
| consolidation-6 | cleanup | low |  | bit-identical | `src/courtside_dynamics/callbacks/performance_gate.py:412` | Confirmed-dead or write-only code from the vulture sweep (grouped) |
| env-paddle-4 | cleanup | low |  | docs-only | `src/courtside_dynamics/envs/_paddle_court.py:431` | Misleading opponent-wiring recipe: 'policy.predict(obs)' skips the VecNormalize transform every PaddleTennis checkpoint needs |
| env-paddle-5 | cleanup | low |  | docs-only | `src/courtside_dynamics/recipes.py:296` | Recipe comment says terminal serve_side_is_policy is the fraction of episodes the policy served (~0.5); under n-point it is the last point's server |
| env-paddle-6 | cleanup | low |  | docs-only | `src/courtside_dynamics/envs/paddle_tennis.py:28` | Stale or contradictory docstrings in the paddle stack (one-point episode, 'Prototype' module, PROVISIONAL XML, ground oracle 'HOLDS position') |
| env-paddle-7 | cleanup | low |  | bit-identical | `src/courtside_dynamics/envs/paddle_tennis.py:722` | Dead `stopped` flag in PaddleTennisEnv._step_mujoco_simulation |
| env-wallball-simple-5 | cleanup | low |  | docs-only | `docs/DECISIONS.md:691` | DECISIONS marks the WallBall binary/counter obs normalization exclusion 'implemented (0.11.0)', but no WallBall recipe ever set it |
| env-wallball-simple-6 | cleanup | low |  | docs-only | `src/courtside_dynamics/envs/wall_ball.py:10` | Stale WallBall docstrings and comments that misdescribe the reward, stall, double-bounce bookkeeping and workspace |
| humanoid-3 | cleanup | low | yes | bit-identical | `src/courtside_dynamics/envs/__init__.py:9` | Import-time coupling is wider than §7.1 states: paddle-motivated edits to shared modules can break paddle imports through module-level validation of the humanoid presets |
| humanoid-4 | cleanup | low |  | docs-only | `docs/repo_review_20261005.md:326` | §7.1 humanoid archive plan: mostly accurate at HEAD, but one condition is stale, the size figures drifted, and the removal map omits files |
| humanoid-5 | cleanup | low |  | bit-identical | `src/courtside_dynamics/envs/tennis_curriculum.py:381` | Small dead and duplicated code in the humanoid modules (only worth fixing if the archive slips) |
| notebook-utils-4 | cleanup | low |  | tooling-only | `notebooks/paddle_tennis_campaign.ipynb:925` | Campaign artifact audit always flags media/best_model.mp4 as MISSING on healthy legs |
| notebook-utils-8 | cleanup | low |  | docs-only | `notebooks/sb3_training.ipynb:428` | Stale sb3_training markdown about where best_model is saved and the plot layout |
| recipes-config-2 | cleanup | low |  | tooling-only | `pyproject.toml:11` | pyproject declares the license as a TOML table plus a License classifier; setuptools announces removal by 2027-Feb-18, which would break every Colab git+ install |
| recipes-config-3 | cleanup | low |  | tooling-only | `.github/workflows/ci.yml:100` | CI installs the CUDA torch build (about 2.7 GB of nvidia wheels) in all four dependency installs, about 1m40s each |
| recipes-config-4 | cleanup | low |  | docs-only | `src/courtside_dynamics/recipes.py:296` | Recipe comments and docstring describe behavior the code no longer has (serve_side_is_policy terminal key, DepthCurriculum reward_eval_episodes, algo choices) |
| recipes-config-5 | cleanup | low |  | docs-only | `src/courtside_dynamics/run_configs/wall_ball_depth_curriculum_aligned.toml:38` | Aligned depth-curriculum starter's commented gate example is the retired pre-0.22.0 geometry with no paddle_home_x pivot |
| recipes-config-6 | cleanup | low |  | docs-only | `docs/run_config_file_spec.md:58` | run_config_file_spec.md and the starter README were not updated for the batch's run-config changes |
| recipes-config-7 | cleanup | low |  | docs-only | `src/courtside_dynamics/__init__.py:12` | Package __init__ version-history comment stops at 0.19.0 while __version__ is 0.25.0 |
| rules-events-3 | cleanup | low |  | docs-only | `src/courtside_dynamics/envs/tennis_rules.py:115` | TerminationReason docstring says 'ordered by fault priority', which the batch made false; the integers are actually a persisted CSV wire format that no test pins |
| rules-events-4 | cleanup | low |  | bit-identical | `src/courtside_dynamics/envs/tennis_rules.py:237` | RallySnapshot / to_info omit require_bounce_before_return, the only rule switch that varies in PaddleTennis |
| rules-events-5 | cleanup | low |  | docs-only | `src/courtside_dynamics/envs/__init__.py:3` | Stale registration-adjacent docstrings: envs/__init__ claims curriculum/registration export order (imports are isort-alphabetical) and the package's version log stops at 0.19.0 while __version__ is 0.25.0 |
| tests-4 | cleanup | low |  | docs-only | `tests/test_seed_ledger.py:56` | Seed-ledger 'cited' test is self-referential, and all seven DECISIONS.md line citations it pins went stale in the batch's own booking commit |
| tests-5 | cleanup | low |  | tooling-only | `tests/test_train.py:1297` | test_train.py spends about 80% of its 152 s on SAC gradient steps no assertion depends on; other slow outliers duplicate work |
| tests-6 | cleanup | low |  | tooling-only | `tests/test_humanoid_tennis_env.py:760` | Tautological assertions, including in the CI's only GL render smoke test |
| tools-10 | cleanup | low |  | docs-only | `tools/paddle_tennis_probes.py:399` | Stale comments: --certify claims to certify 'exactly the env training will see'; demo harvest says its reset seed 'carries no statistical content' |
| tools-4 | cleanup | low |  | docs-only | `tools/_seed_ledger.py:48` | Every DECISIONS.md line citation in the seed ledger is stale by +74 lines, and the test pins the stale strings |
| tools-7 | cleanup | low |  | tooling-only | `tools/paddle_tennis_diagnosis_probe.py:86` | Dead --oracle (diagnosis probe) and --stub-oracle (P5) flags: parsed, never read, silently overridden by --model |
| tools-8 | cleanup | low |  | tooling-only | `tools/paddle_tennis_postswing_target_probe.py:185` | PT1 WindowRecord.cmd_to_hit duplicates cmd_from_hit_mean and never reaches the report or JSON |
| tools-9 | cleanup | low |  | tooling-only | `tools/paddle_tennis_volley_probe.py:188` | volley_probe.evaluate_criteria is dead (no caller since adoption_verdict replaced it) |
| train-8 | cleanup | low |  | docs-only | `src/courtside_dynamics/training/train.py:152` | Stale or misleading warm-start docs and provenance labels in train.py |
| training-data-7 | cleanup | low | yes | bit-identical | `src/courtside_dynamics/recipes.py:299` | PaddleTennis terminal eval keys still describe term_* as 'the grouped fault distribution', but in continuous n-point play they are structurally constant |
