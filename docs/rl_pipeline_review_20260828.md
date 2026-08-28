# RL pipeline review — 2026-08-28

Kind: **Review snapshot**, pinned to `main`-lineage commit `ebce641`
(the LT1 registration commit), conducted 2026-08-28 while both LT1
runs were in flight. Method: nine review scopes (callbacks, train
core, the last-25-commit regression surface across ladder envs and
tools, the paddle env, the humanoid-rung status audit, config
plumbing, the three notebooks, tests+CI, and the DECISIONS gap
ledger) plus a three-scope cleanup survey (notebook/artifact
utilities, `src` training core, tools+tests). Every scope read its
modules in full; claims marked *confirmed-by-execution* were
reproduced with runnable probes, and each **material** finding was
then re-verified by an independent adversarial pass — **13 material
findings, 13/13 CONFIRMED** (plus 43 minor and 14 nit findings,
individually confirmed by reading or execution but not
re-adjudicated).

**Nothing in this review changed code.** The review branch is the
live `pip install` source for the two in-flight runs (registered LT1
`20260828_121324` and companion `20260828_113136`), so every fix
below is *routed*, not landed. Routing labels:

- **[after-LT1]** — safe to land as soon as the in-flight runs
  complete (no comparability impact; robustness/observability only).
- **[era-boundary]** — changes eval semantics, reward taxonomy, or a
  recipe's frozen shape; may only land with a declared comparability
  era break, per doctrine.
- **[humanoid-resume]** — blocks or shapes the humanoid rung; land
  before the next humanoid training attempt, irrelevant to the
  paddle campaign.
- **[frozen-instrument]** — touches a pre-registered probe cited by
  sha in verdict documents; only when that instrument is
  legitimately reopened.

## 1. In-flight-run safety (the question asked first)

**Verdict: nothing found corrupts or endangers the artifacts or
verdicts of `20260828_121324` (registered LT1) or `20260828_113136`
(companion).** Checked deliberately, per scope:

- No shared code used by the paddle recipe stack changed in the
  last 25 commits in a behavior-affecting way (scope 2's regression
  sweep; the paddle env's own 179 tests pass at `ebce641`).
- The LT1 configuration is immune to the miskeyed-selection finding
  (§2.1): the recipe hardcodes the correct `crossings` key and the
  campaign plan validator cross-checks `config.json`.
- The eval/normalizer/selection machinery the runs depend on was
  re-verified healthy end-to-end: normalizer freshness at every
  save site, eval-stream isolation, confirm-best banking, seed
  hygiene, TB↔CSV consistency, checkpoint-diagnosis ordering
  (scope 0/1 healthy lists).
- No notebook path can touch the in-flight artifacts (fresh
  timestamped run dirs; campaign namespaced by `CAMPAIGN_ID`).
- The real exposures that do exist are *loss-of-robustness*, not
  corruption: an unhandled Drive I/O error mid-run (§2.2, §2.5)
  or a non-KeyboardInterrupt crash (§3, train core) would lose
  end-of-run artifacts but cannot falsify anything already written —
  checkpoints, the protected best pair, and eval CSVs survive.

The double-bounce misattribution (§2.9) does affect the in-flight
runs' `term_out_of_bounds` / `term_second_bounce` eval columns — but
it is **era-consistent** (present in every prior campaign run), so
comparability is preserved. That is precisely why its fix is
**[era-boundary]**, never mid-campaign.

## 2. Material findings (13, all adversarially CONFIRMED)

### 2.1 Miskeyed selection metric silently degrades to reward selection — [after-LT1]

`src/courtside_dynamics/callbacks/info_dict_eval.py:840` (`_score_of`)
with `train.py:1310-1318` and `run_config.py:454-475`. A
`best_metric_key` absent from the eval metrics scores as `-inf` and
the lexicographic comparison falls through to the remaining keys —
selection silently degrades to success-rate/reward-based selection,
the exact reward-crowned-degenerate-policy failure (run
`20260712_190054`) headline selection exists to prevent. Executed: a
typo'd `headline_key = "corssings"` trains normally, the first eval
unconditionally crowns best, and a degenerate high-reward policy is
later crowned over a genuinely better task policy; nothing warns
anywhere. `run_config.py` validates field *names*, never metric
*values*, and no layer checks `headline_key` / `best_metric_keys` /
`degenerate_guard_keys` / `success_key` against emitted metrics.
**Fix:** after the first completed evaluation with selection keys
set, raise (or loud per-eval warning + a flag in
`best_model_meta.json`) when any selection or guard key is absent
from collected metrics.

### 2.2 `eval_info.csv` append is not exception-isolated — [after-LT1]

`info_dict_eval.py:1104` (`_append_csv`, invoked from `_on_step`).
Any `OSError` during the append propagates out of `model.learn()`,
and `train.py:1716-1739`'s salvage catches only `KeyboardInterrupt`
— the run dies losing `final_model.zip`, the closing eval, and
`stage_summary.txt`. This contradicts the same callback's own
lesson-7 treatment of the npz writer (log-and-continue,
lines 495-511), and both in-flight runs write this file to a Google
Drive FUSE mount where transient I/O errors are a known failure
mode, once per 25k-step evaluation. Confirmed by a contrast repro:
`_append_evaluations_npz` logs and continues on the same unwritable
path where `_append_csv` raises. **Fix:** wrap `_append_csv` in the
same try/except-OSError log-and-continue (the CSV is a diagnostic
artifact per the module's own doctrine; TB and `last_metrics` still
carry the values). Explicitly do **not** land while the branch is
the in-flight runs' install source.

### 2.3 No device assertion anywhere in the training path — [after-LT1]

`src/courtside_dynamics/colab_setup.py:44-67` +
`training/train.py` (no `torch.cuda` reference; device recorded only
into after-the-fact provenance at `train.py:1784`).
`setup_colab(require_gpu=True)` shells out to `nvidia-smi` only — a
check a `+cpu` torch build passes on a GPU runtime — and lives in an
optional mid-notebook cell that standalone launch cells bypass;
SB3's `device='auto'` silently resolves to CPU. This is today's
incident realized: two shape-conformant LT1 launches
(`20260828_114038`/`114815`) ran on a CPU-only runtime, were not
completable in wall-clock, and cost launch slots; the field fix was
a hand-added CUDA assert in the launch cell (LT1 prereg §4a).
**Fix:** an expected-device knob honored inside `train()` (fail
loud, cardinal rule 1), plus `torch.cuda.is_available()` checked in
`setup_colab` alongside `nvidia-smi` and echoed in the notebooks'
config-echo cells.

### 2.4 `sb3_training.ipynb` has no warm-start affordance — [after-LT1]

`notebooks/sb3_training.ipynb` (zero occurrences of
`WarmStartConfig`, sha pins, or plan validation). Warm-started
registered runs are now the campaign's standard shape, yet every
pilot is launched from a hand-maintained standalone cell that
bypasses the notebook's provenance, guard, and validation machinery.
The LT1 §4a launch record is this finding's failure scenario
realized three ways in one morning: an import-error abort, a
companion run whose frozen 1M budget was lost on a re-run (trained
3M) with an inherited `early_stop_patience=60`, and two CPU false
starts — while the stack already owns every countermeasure
(`WarmStartConfig` sha pins, `validate_run_config_against_plan`,
`build_train_config(warm_start=...)`). **Fix:** a warm-start block
in the settings cell (`WARM_START_RUN_DIR` /
`TRANSFER_LOG_ENT_COEF` / `EXPECTED_ARTIFACT_SHA256` →
`WarmStartConfig`), plus an optional expected-plan dict validated
post-launch — the campaign notebook's pattern, transplanted.

### 2.5 `artifacts.py` provenance writers swallow failures silently — [after-LT1]

`training/artifacts.py:453, 473, 601` — `except (OSError,
JSONDecodeError): return None` with no log line.
`update_run_config_with_model` / `..._with_initialization` silently
drop resolved-model and warm-start provenance from `config.json` on
I/O or JSON errors (confirmed by execution against a truncated
`config.json`: returned `None`, zero bytes of output, initialization
block permanently absent). A transient Drive error at an LT1-style
run start would leave the run unable to prove
`transfer_log_ent_coef=False` was applied; the campaign validator
would catch it loudly post-hoc, but only after the compute is spent.
**Fix:** one `[artifacts]`-prefixed print per swallowed exception
(the 2026-08-02 infrastructure review's own convention).

### 2.6 `_load_obs_normalizer` silent identity fallback — [after-LT1]

`notebook_utils.py:704-733` — `except Exception: return lambda obs:
obs`. A `VecNormalize` snapshot that *exists but fails to load*
(shape skew, truncated pickle, SB3 version skew) is silently
replaced by the identity, contradicting the docstring ("identity if
neither file exists") and the notebook's replay-fidelity promise —
the literal cardinal-rule-1 example ("silent normalizer fallbacks
that record garbage videos of a competent policy"), unfixed since
the 2026-08-02 review flagged it. Confirmed by execution:
loading a 4-dim normalizer against a 6-dim env returned the identity
with zero bytes of output, feeding straight into best-model replay.
Impact is diagnostic (videos are not verdict evidence), but a
campaign judgment call can be made from a misleading recording.
**Fix:** loud warning or raise on the except path.

### 2.7 Training-worker Monitors record only `r,l,t` — [after-LT1]

`train.py:1046-1051` — `make_vec_env` is called without
`monitor_kwargs`; no `info_keywords` plumbing exists anywhere in
`src` (grep-confirmed). Training-time terminations in the in-flight
runs therefore cannot be attributed post hoc: LT1's analysis of
which *training* episodes ended in faults vs timeouts vs k=2 chains
must fall back on 25k-cadence eval snapshots — the recurring pain
the DECISIONS "instrument for post-hoc attribution" entry records.
**Fix:** recipe-settable `monitor_info_keywords` forwarded as
`monitor_kwargs={'info_keywords': ...}`; `load_monitor_episodes`
already passes extra columns through.

### 2.8 Unpaired evaluation remains the untreated variance floor — [era-boundary]

`info_dict_eval.py:551` — a bare `self.eval_env.reset()`; every
evaluation (and every confirm-best re-sample) draws a fresh unpaired
episode set, so the full batch SE lands on every best-model
comparison, including LT1's checkpoint crowning. Mitigations exist
(quantized keys, min-delta, `confirm_best_eval`, degenerate guard)
but the variance itself is untreated; the DECISIONS entry
("Unpaired evaluation is the root of the gate noise — open (P1)")
describes exactly this line. **Fix:** re-seed the matched selection
stream to a fixed episode set per eval (paired common-random-numbers
comparisons), keeping the final-config stream fresh-random. This is
an eval-semantics change: **era boundary only, never mid-campaign.**

### 2.9 Double-bounce faults misreported as `out_of_bounds` — [era-boundary]

`envs/tennis_rules.py:988-1005` (`_handle_court_contact` checks
bounds before bounce count) with the priority map at `:779-783`.
Executed at `ebce641`: a rally whose in-bounds first bounce is
followed by a second hop landing out terminates
`out_of_bounds` with `bounce_count=1` — real-tennis truth is
`second_bounce`. The same `RallyStateMachine` backs
`PaddleTennisEnv`, so the in-flight runs' fault columns carry the
bias; it is era-consistent (every prior run identical), so
comparability holds — and that is why the fix **requires a declared
fault-taxonomy era break**. This was review-Major M1 in the humanoid
env review and is the only review-Major absent from DECISIONS'
humanoid status list. **Fix (at the boundary):** check bounce count
before bounds in `_handle_court_contact`; add "second hop lands out"
unit tests; add the item to DECISIONS' tracking list *now* (a docs
change, safe immediately).

### 2.10 Humanoid promotion gate still crashes at 0-success — [humanoid-resume]

`training/tennis_curriculum.py:717` (clamp), `:270-271` (invariant),
`:521-525`, `:1317/:1325` (crash sites). The `[0,1]` clamp does not
touch the actual case: `_wilson_interval(0, 50)` returns
`low = 6.94e-18` — a positive FP residual — which violates
`SideSuccessSummary`'s strict `0 <= low <= success_rate` invariant.
Executed sweep: zero-success sides crash at n∈{25,50,100,200};
all-success sides crash at n∈{10,25,30} (n=50 survives by FP luck).
An early humanoid policy scoring 0/50 on one serving side — exactly
what the review measured for every cold policy — burns all 100
held-out episodes and then loses the entire promotion evaluation to
a `ValueError` instead of reporting 0%. DECISIONS' "partially
addressed" overstates the state, and the gate tests DECISIONS itself
demands (0-success / all-success at n∈{25,30,50,100}) still do not
exist. The three `_wilson_interval` copies have drifted (humanoid
`[0,1]`-clamped, `ladder_certification.py:377` unclamped,
`tools/depth_stage_sweep.py:524` a third copy). **Fix:** clamp to
the rate (`low = min(max(low, 0), p)`, `high = max(min(high, 1),
p)`), add the demanded boundary tests, and consolidate to one shared
implementation.

### 2.11 Free-standing humanoids collapse with no fall fault — [humanoid-resume]

`envs/humanoid_tennis.py` (no fall/upright/pelvis-height code
anywhere; grep-confirmed) + `README.md:209`. Executed: under zero
action both free-standing G1 pelvises settle at z = 0.103 m and the
episode runs 211 steps to a *ball* fault; no fall or unsafe flag
ever fires. The README and module docstring still state the
zero-action standing contract without the anchored-only caveat.
Any free-standing training attempt starts every episode with an
unrewarded, unterminated balance collapse. **Fix:** fall detection
(pelvis height / torso orientation) with termination and/or an
upright-alive term for FREE_STANDING mode; re-document zero-action
as anchored-only; treat a standing controller as the true Stage 3
prerequisite.

### 2.12 Stage 0's 150-step truncation hides the miss penalty — [humanoid-resume]

`recipes.py:1259` (`episode_len=150`) +
`envs/humanoid_tennis.py:1244-1312`. Executed: a zero-action Stage 0
episode truncates at step 150 with total reward exactly 0.0; the
identical trajectory at `episode_len=400` terminates at step 194
with −1.0 (`second_bounce`). A non-contacting policy therefore
trains on a zero-variance reward stream — every episode exactly 0.0
below the first chance contact — the review's measured
blocked-learning regime, now mitigated only by the *unvalidated*
0.16.0 exploration remediations (gSDE + `ent_coef=0.01` are present
in the recipe, but **no humanoid stage training run has been
recorded anywhere since**). **Fix:** extend Stage 0 `episode_len`
past ~200 (400 verified to work) so misses pay −1, then run the
cheap 25k-step Stage 0 probe to validate gSDE before building
anything on this rung.

### 2.13 KeyboardInterrupt salvage has zero test coverage — [after-LT1]

`train.py:1716-1739` + `tests/` (grep finds no `KeyboardInterrupt`
anywhere in tests). The salvage path protecting ~20-hour Colab runs
(final model, final eval, interrupted summary, gate finalize) is
untested; a `train.py` refactor — such as the warm-start extraction
in the cleanup register — could silently break it, and the next
manually stopped long run (e.g. the 3M companion after its 1M
window) would lose its closing artifacts. Partially mitigated:
since `82a2afe`, completed-stage history rows are rewritten per
stage close and survive hard death. **Fix:** an integration test
injecting `KeyboardInterrupt` mid-learn and asserting the salvage
artifacts exist.

## 3. Minor findings register (43), by subsystem

Condensed to the decision-relevant core; each was individually
confirmed. Routing tags as in §2.

**Callbacks / train core** — [after-LT1] unless noted:
- Diagnosis probe self-disables permanently on any single failure
  with only a console print (`paddle_diagnosis.py:643-654`); retry
  on OSError, write a `diagnosis_probe_disabled.txt` marker.
- `<key>_final` is the *last episode's* terminal value (n=1 of 30),
  presented in `stage_summary`'s final/peak block as a headline
  (`info_dict_eval.py:622,656-657`; `artifacts.py:1057-1063`);
  document, prefer `_ep_mean` in the headline block.
- Performance-gate stage-close I/O (champion archive, config copy,
  stage history) is not exception-isolated and fires exactly at
  promotion boundaries (`performance_gate.py:533-545, 589-596`).
- `np.bool_` info values are silently dropped by
  `_scalar_info_keys` (`callbacks/_info.py:37-40`; latent — live
  envs wrap in `bool()`); accept `np.bool_`, add a regression test.
- Non-KI crash salvage absent: any exception besides
  KeyboardInterrupt out of `model.learn()` skips the epilogue
  (`train.py:1716-1739`); broaden to save-then-reraise with a
  `crashed` status.
- Reused `log_dir` silently mixes attempts: `eval_info.csv` appends
  across trainings while models overwrite (`train.py:997`,
  `info_dict_eval.py:1104`; confirmed by execution — 36 rows per
  timestep after two runs); refuse or loudly warn on a non-empty
  `log_dir`.
- Warm-start docstring hides SB3's uniform-random action warmup
  below `learning_starts` (`train.py:396-405`): LT1's first 25k
  steps are random-action transitions by design; document (no
  comparability damage — LH1 control identical).
- Warm-up eval-count derivation uses `ceil(steps/eval_freq)` not the
  realized cadence (`train.py:1341-1347, 1365-1372`) — off-by-one
  only when `n_envs` does not divide `eval_freq`; no current recipe
  affected (nit-adjacent).
- `train()` never calls `validate_model_kwargs` (only the recipe
  path does); a direct-`TrainConfig` typo crashes late inside SB3
  after the env fleet is built.

**Ladder envs + tools** — [after-LT1] except as tagged:
- `WallBallEnv` is the only ladder env without the pre-step
  `_physics_state_is_finite()` guard (`wall_ball.py:1321-1334`);
  reproduced: a hidden-state NaN (ball quaternion) triggers
  MuJoCo's BADQPOS auto-reset into a teleported-but-live episode
  with no `term_nonfinite`. Add the sibling guard (commit `c62727a`
  pattern) and the missing test.
- **Seed-ledger drift across probes** (the mechanism behind two
  findings): `paddle_tennis_hold_probe.py:63-71` is missing burned
  block **6300–6399** from `_RESERVED_BLOCKS` (verified: the guard
  accepts `--seed-start 6300`), and five older probes
  (shaping/volley/diagnosis + two more) have **no**
  `_refuse_reserved` at all — nothing refuses the sealed 4100–4199
  gate block on a mistyped `--seed-start`. Block tables have drifted
  to 2/5/7/8 entries across the four probes that do guard.
  **[frozen-instrument]** for edits to adjudicated probes; the
  clean route is the cleanup register's shared
  `seed_ledger.py` (§6), additive-only.
- Three probes exit 0 on a FAIL verdict (npoint, shaping, volley;
  confirmed by execution) — automation gating on exit status reads
  a failed battery as a pass. **[frozen-instrument]**.
- `BallBalanceEnv` returns `info={}` even on the nonfinite guard
  firing — the one env where the guard is unobservable (nit).
- `depth_stage_sweep.py`'s release path hardcodes the
  `WallBallDepthCurriculum` recipe; the other ladder-bearing recipes
  cannot be certified from the CLI.

**Paddle env** (both findings inert under the frozen config —
document-only):
- `_nudge_paddle_clear` writes slide qpos unclamped to joint ranges
  and never re-verifies clearance (`paddle_tennis.py:1133-1163`);
  surgically exercisable only outside the frozen serve envelope
  (drawn origins keep ≥ ~1 m margin; 26 oracle boundaries produced
  0 nudges).
- Two `term_*` families with different semantics coexist in the
  paddle info dict (point-scoped snapshot booleans vs episode-ending
  group floats; four colliding names contradict on absorbed
  boundaries). No current consumer reads the snapshot names;
  document the contract in the CSV/eval key comments.

**Config plumbing / recipes** — [after-LT1] unless noted:
- The packaged PaddleTennis starter TOML's header still describes
  the retired one-point era ("reference band is 7.78", "Everything
  here matches the recipe today") — two eras stale
  (`run_configs/paddle_tennis.toml:1-13`). Rewrite for the n-point
  era. (Also an erratum-adjacent doc item, §7.)
- `resolve_run_config_file` reuses any existing Drive copy with the
  same "reusing existing copy" message whether or not it matches
  the packaged starter (demonstrated: a stale-era file pinning 3M
  silently wins over the recipe); print a DIFFERS warning on
  byte mismatch.
- `quick_test` does not scale `checkpoint_diagnosis` — the
  diagnosis dominates the smoke test's wall clock (~135k serial env
  steps against a 25k-step budget); add `{"episodes": 3}` to the
  overrides.
- The PaddleTennis recipe never sets `reward_eval_episodes`, so
  every 25k steps rolls a redundant 45k-step full reward eval — the
  exact overhead the depth recipe documents and fixed.
  **[era-boundary]** (the in-flight and control runs share this
  shape; never mid-campaign).
- `validate_run_config_against_plan`: a missing/non-mapping
  `train_config` block is booked as drift instead of the documented
  plain `ValueError` (instrument-error misattribution, fails
  closed); and pin artifact *names* are not validated against the
  `WarmStartConfig` allowlist, so a typo'd plan key books drift
  against a healthy run.

**Notebooks** — [after-LT1]:
- `check_run_artifacts` unconditionally expects
  `best_model_meta.json`, which only headline-selection recipes
  write — healthy BallBalance runs audit as MISSING (executed).
- Humanoid notebook: the failure-prone best-model video runs
  *before* the promotion evaluation (cardinal-rule-7 inversion —
  a codec failure after multi-hour training aborts before any
  promotion artifact exists); reorder or isolate.
- Campaign notebook: `GATE_MIDDLE_ACTION` is decision-shaping but
  absent from the fingerprint and manifest; a resume can silently
  flip it. Add to `FINGERPRINT` and the decision record.
- Campaign notebook resume treats a `'trained'` leg like an absent
  one and retrains 1M–3M steps when minutes of re-scoring would do;
  re-enter at scoring when a validated attempt exists (cost, not
  corruption).
- `sb3_training`'s `EARLY_STOP_PATIENCE` comment says "20" while
  the notebook's default env deliberately carries 60 — the same
  stale-patience confusion that produced the companion run's booked
  deviation.
- `sb3_training` imports the package before `setup_colab()` sets
  `MUJOCO_GL` — the humanoid notebook documents the opposite
  contract; reconcile (mostly theoretical).
- Humanoid notebook's "warm-start advancement supports PPO only"
  guard reason is stale (SAC warm starts are production-proven);
  reword (nit).

**Tests + CI** — [after-LT1]:
- Random-rollout env tests are unreproducible: `reset(seed=0)` does
  not seed `action_space`, so a NaN failure cannot be replayed
  (confirmed: two seed-0 runs, different actions). Seed the action
  space in the three tests.
- The sha-mismatch warm-start test asserts the `ValueError` but not
  that the target dir stayed unwritten (the pattern its sibling
  tests use); one `assert not exists` line.
- CI thread-pinning env vars (`OMP_NUM_THREADS=1` etc.) are
  duplicated per-step instead of hoisted to job level — a future
  step silently loses them and runs 3–8× slower (nit; the repo's
  own measurements).

**DECISIONS-ledger residue** (still-open items re-verified, dormant
unless tagged): the weaker `_git_sha` duplicate in
`tennis_curriculum.py:881-893` (bites the first resumed humanoid
run's provenance, **[humanoid-resume]**); the whole PaddleTennis era
shipped under an unmoved `0.25.0` with no git tags (bump/tag at the
era boundary LT1's verdict creates); `paddle_x_target_range` remains
an unguarded plain attribute on the closed WallBall line; the Colab
health check still compiles an empty MJCF (exercising neither EGL
nor GPU render); `sb3_training` cell-8 layout tree omits newer
per-recipe reports (cosmetic).

## 4. Humanoid rung: status audit and the transfer chasm

The audit walked every item in DECISIONS' humanoid status list
against the code. Since-fixed items verified real (C1 stringbed
bounds, 0.16.0 exploration remediations present, M5 video isolation,
M7 headline surfacing); the still-open items are §2.10–§2.12 plus
the §3 register's minors (graded aim signal absent — a landed-out
hit and a whiff both pay the same flat penalty; `n_envs=1` pin whose
recorded remedy is superseded because `train()` never wires
`SubprocVecEnv`, so un-pinning alone buys nothing; body-local ball
spin observations; the promotion suite evaluating launch
perturbations Stage 0–1 training can never sample).

Beyond individual findings, four structural transfer gaps separate
the paddle campaign from the humanoid goal, none tracked as work
items before this review:

1. **Action interface**: the paddle policy speaks 3 task-space
   position targets into kp-300 servos; the humanoid speaks
   joint-space torques/PD across a G1. No shim exists or is
   designed.
2. **Task semantics**: the paddle era's default is
   `volley_rule='fault'` (bounce-first ground strokes); the
   humanoid task as configured does not enforce the same rally
   grammar the paddle policy was shaped on.
3. **Physical scale**: ball radius/mass/restitution/friction and
   court dimensions all differ measurably between the paddle court
   and the regulation humanoid court.
4. **No side-canonical view**: the paddle policy is side-relative
   with a bit-exact mirror (P4); the humanoid observation is a
   299-value centralized two-player vector with no side-local
   rendering.

And the rung-level facts: **no humanoid stage training run has been
recorded since the 0.16.0 remediations** (whether gSDE unblocks
Stage 0 is untested — the cheapest missing experiment in the repo,
~25k steps); Stages 3–5 remain `environment_available=False`; and
the promotion gate cannot currently gate an early policy at all
(§2.10 fires exactly at the 0%-per-side scores early policies
produce).

## 5. Capability gaps (what the stack cannot do yet)

- **The campaign's standing blocker**: k=2 has never exceeded noise
  floor. LT1 (in flight) is the optimizer-side attempt; the
  interface-side command-rate treatment exists only as a design
  (`design_paddle_tennis_command_rate.md`), gated behind LT1's
  T1-FAIL branch.
- **P5 champion transfer** remains open: instrument and tests
  exist, the Colab champion rows never ran, so the phase-P2
  frozen-opponent-pool decision cannot be taken.
- **Paddle-pitch loft authority** — the era's declared open
  capability question; no pitch actuation exists and no run has yet
  forced the decision.
- **Paired evaluation** (common random numbers) is designed in
  DECISIONS but unbuilt (§2.8) — the era buys gate reliability with
  latency instead of variance removal.
- **Eval parallelism**: all eval envs are `n_envs=1` while training
  runs 4–8 wide; claiming the wall-clock back requires rewriting
  the rollout loop to aggregate workers.
- **No single-run warm-start driver** (§2.4) and no
  plan-validation affordance in `sb3_training`; no expected-device
  guard anywhere (§2.3) — the three capability restatements of
  today's launch incidents.
- **No test-tier markers**: the ~12 training-loop tests (~110 s
  serial, a third of suite time) cannot be deselected for a fast
  local loop.
- **Durable degradation records**: non-fatal callback failures
  (video, diagnosis disablement) exist only as console prints;
  `stop_reason` covers run-ending events only, so an unattended
  run's artifact set cannot explain its own gaps.
- Coverage is measured but ungated; the Drive-corpus conventions
  proposed in the 2026-08-02 review (ABORTED stub markers, orphan
  triage) still have no implementation — and the LT1 launch morning
  minted four new non-registered run dirs of exactly that class
  (each one's status is at least named in the prereg §4a record).

## 6. Cleanup register (surveyed, none applied)

From the three-scope cleanup survey (~40k lines read; AST-based
dead-code and duplicate-block scans; every `notebook_utils` name
cross-grepped against notebooks, tests, tools, docs). **Zero dead
functions found** — the repo's debt is duplication and megafunction
structure, not corpses. Ranked within risk class:

**Safe-mechanical** (behavior-identical, land [after-LT1] in one
sweep):
1. **Shared seed ledger** — new `training/seed_ledger.py` holding
   the authoritative reserved/burned block table +
   `refuse_reserved()`; probes import it *additively* (existing
   frozen instruments untouched until legitimately reopened). The
   highest-value item: five drifting hand-copies currently guard
   the repo's most load-bearing invariant (§3, seed-ledger drift).
2. **One `_sha256_file`** — three byte-identical helpers
   (`notebook_utils.py:1278`, `info_dict_eval.py:51`,
   `train.py:745`) feed cross-checked provenance pins (warm-start
   pins validate digests `best_model_meta` wrote); one definition
   in `training/artifacts.py` removes a real drift channel.
3. **Parametrize the notebook hygiene tests** — three near-identical
   compile/hygiene test copies in `test_notebooks.py` have already
   drifted (the sb3 variant silently lost the cell-id-uniqueness
   assertion); one parametrized pair restores equal coverage.
4. **Shared wall-ball/tennis `phase_labels`** — the identical dicts
   are repeated verbatim in 5 + 3 recipes (an 8-site shotgun edit
   hazard).
5. Retire the `_tennis_physics.is_in_bounds` module-level shim
   (production callers all use the `CourtGeometry` method; only a
   parity test imports it), the `video_record._scalar_info_keys`
   re-export, the `performance_gate.extra_target_envs` never-passed
   parameter (preserve its load-bearing doc warning), the
   unreachable `evaluate_policy` list-branch in `train.py:1749`,
   and `plot_eval_info`'s redundant function-local csv/json
   re-imports; point `test_diagnosis_probe` at the package module
   instead of the CLI.

**Needs-tests** (behavior-preserving intent, but the moved logic
deserves direct unit coverage as part of the change):
1. **Decompose `train()`** (972–1805, ~833 lines) into
   `_resolve_eval_streams(cfg)`, `_selection_kwargs(...)`,
   `_apply_warm_start(...)` — the sizing and min-evals derivations
   are the two places a recipe change silently interacts with early
   stopping (two documented past incidents), currently testable
   only via 400-step integration runs. The highest-value structural
   item; do it together with the §2.13 salvage test so the refactor
   cannot silently break salvage.
2. **One eval-info row parser** — three independent parsers of
   `eval_info.csv` have already drifted on malformed-row tolerance
   (`plot_eval_info` / `_read_eval_info_series` /
   `_read_eval_info_at_step`); a shared `iter_eval_info_rows` fixes
   a latent `TypeError` divergence.
3. Extract the twice-written Drive-root convention in
   `notebook_utils` (`resolve_run_dir` vs
   `resolve_run_config_file`, messages already drifted); unify the
   three atomic-JSON-write idioms on the fsync'd variant (the gate
   and certification writers currently have weaker durability than
   the campaign manifest); a `_quiet_train_config()` builder for
   the 7 verbatim TrainConfig clusters in `test_train.py`; merge
   the three `_drive` lockstep-rollout helpers in
   `test_paddle_tennis.py`; hoist the duplicated normalization-env
   scaffolds in `test_tennis_curriculum_training.py` and the
   ball-at-wall teleport scaffold (5 copies) in `test_envs.py`.

**Touches-frozen-instrument** (only with a declared reopening):
- Splitting `notebook_utils.py` (2578 lines, five clearly-delimited
  eras; the WallBall instrument alone is 944 lines/37%) into a
  package behind a re-export facade — pure relocation, zero
  notebook/test edits required, but the WallBall section is a
  cited instrument; schedule deliberately.
- The hold-probe↔reach-probe shared witness harness and the
  P5↔diagnosis checkpoint-loader duplication: both are
  pre-registered instruments cited by sha in verdict docs.
  Recorded so nobody "fixes" them casually; extract a shared
  harness only *for future probes*, leaving shipped files bit-frozen.

**Deliberate leaves** (look like debt, are load-bearing — do not
"clean"): the legacy flat-layout fallback in `artifacts.py:664-708`
(the performance gate archives stage champions into exactly that
layout and warm start resolves them through it); the WallBall
long-horizon block in `notebook_utils` (dormant-but-keep:
`sb3_training` cell 27 still gates on it and a pinned test requires
it — it is the only way to re-audit historical WallBall runs); the
three HISTORICAL recipes (kept to reproduce campaign eras);
`tools/depth_stage_sweep.py` (pinned 0.21-era stage tables are
provenance, and two test suites import it).

## 7. Erratum candidates (bookable now; docs only)

1. **LT1 prereg §4a mis-attributes the companion's 3M budget.** The
   launch record says "the recipe default won over the frozen 1M
   when the budget override was lost" — but the PaddleTennis recipe
   default is and has always been **2,000,000** (verified at
   `cb0981f`, `0c05cf0`, and `ebce641`); a genuinely lost override
   yields 2M. The 3M came from a **stale explicit notebook value**
   (the registered-run era's 3M pairing). The companion's
   classification and the validator flag are unaffected; the frozen
   record of *why* is wrong. Book as an erratum when the campaign
   leg closes (the prereg forbids in-flight edits).
2. **The packaged `paddle_tennis.toml` starter header** describes
   the retired one-point era and claims currency (§3, config
   plumbing) — rewrite for the n-point era with an era-correct
   reference band.
3. **DECISIONS' `n_envs` un-pin recommendation** is superseded:
   without `SubprocVecEnv` wiring it buys ~nothing (the stage-0
   TOML already documents this); amend the entry.
4. **`sb3_training`'s patience comment** ("20, i.e. 500k steps")
   contradicts the notebook's own default env (60 by design).

## 8. Sequencing

1. **Now (docs only, no code):** this snapshot; the DECISIONS
   tracking additions (§2.9's missing entry; §7.3's amendment);
   nothing else touches the branch while it is the in-flight runs'
   install source.
2. **When LT1 completes** (registered run ~19:15 UTC; companion
   ~07:30 UTC next day): land the [after-LT1] class — §2.1–§2.7,
   §2.13, the §3 register's after-LT1 items, and the
   safe-mechanical cleanup sweep (§6), each with its named test.
   The LT1 *verdict* work (KT1/T1/R1 scoring, §4 decision rule)
   proceeds independently under the prereg.
3. **At the era boundary LT1's verdict creates:** §2.8 (paired
   eval), §2.9 (fault taxonomy), `reward_eval_episodes`, the
   version bump/tag — each declared in the boundary's
   pre-registration, per doctrine.
4. **Before any humanoid resumption:** §2.10–§2.12 plus the
   [humanoid-resume] register items, opened by the 25k-step Stage 0
   gSDE validation probe — the cheapest missing experiment in the
   repo.
