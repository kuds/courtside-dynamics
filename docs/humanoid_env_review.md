# Humanoid Tennis Environment Review

- **Date:** 2026-07-13
- **Reviewed at:** `main` @ `0d294f2` (package version 0.7.0)
- **Scope:** the humanoid tennis environment (`HumanoidTennisCoopEnv`) and the
  shared infrastructure it depends on — rally rules, event sampling, curriculum
  and promotion gate, training pipeline, logging/artifacts, video recording and
  replay, and learning feasibility. The ball tasks (BallBalance, BallBounce,
  WallBall) were reviewed only where they share that infrastructure.
- **Method:** six parallel static review passes (rules engine, environment
  core, curriculum, logging/diagnostics, video, feasibility), with each
  critical/major finding independently and adversarially verified against the
  code and, where possible, empirically reproduced; six live simulation probes
  against the installed package (MuJoCo 3.10, Gymnasium 1.3, SB3 2.9, 4-core
  CPU container, `MUJOCO_GL=osmesa`); one full end-to-end 25,000-step PPO
  training run of the Stage 0 recipe with a complete artifact audit and a
  SIGINT-salvage test; and the full test suite.
- **Test suite:** 422 passed, 0 failed (81 s). An initial segfault at ~90% of
  the suite was traced to the review container's broken `triton` wheel (pulled
  in by CUDA-enabled torch on a CPU-only box), not to repository code;
  uninstalling `triton` produced a clean run.

## 1. Verdict

**Environment quality — strong.** The rally rules engine, event sampler,
action/observation contracts, determinism, curriculum gating, artifact
capture, and replay design are rigorous and, in nearly every place probed,
behave exactly as documented. Trajectories are bitwise-reproducible under
seeding. The scripted oracles prove Stages 0–1 are physically solvable on both
mirrored serve sides. Physics is credible: court restitution 0.763 at drop
speeds (real hard court 0.73–0.76), drag within 0.4% of its own model, feeds
that clear the net by 0.79–0.92 m and land in bounds.

**Learning feasibility — currently blocked.** As configured, the recipes are
very unlikely to produce learned policies. Stage 0 exposes essentially zero
cold-start reward signal (0 hits in 30 random episodes; 1 success in 177
episodes of a real PPO run; every scheduled evaluation 0.000). Stages 1–2 are
worse: a *constant* −1 under naive play with the only bridging signal (hit
shaping) disabled by default. A confirmed contact-culling bug additionally
neutralizes the curriculum's stringbed forgiveness, making Stages 0–2
physically harder than designed. The README's "convergence is not claimed"
framing is accurate; this review explains why and what would unblock it.

No critical/major finding submitted for adversarial verification was refuted.

## 2. Findings

Severity legend:

- **Critical** — wrong behavior that corrupts training, evaluation, or rules
  outcomes.
- **Major** — significant correctness or design flaw.
- **Minor / Info** — quality, robustness, or documentation issues worth
  tracking.

Verification legend: *confirmed* = an independent adversarial pass traced the
code and/or reproduced the behavior; *measured* = demonstrated directly in
simulation during this review.

### 2.1 Critical

#### C1. Curriculum stringbed contact forgiveness is silently neutralized by stale collision bounds *(confirmed)*

- **Where:** `src/courtside_dynamics/envs/humanoid_tennis.py:459-467`
- **What:** For Stages 0–2 the constructor enlarges the stringbed ellipsoid's
  in-plane semi-axes (`racket_contact_scale` 1.5×/1.35×) by writing
  `model.geom_size` and calling `mj_setConst`. `mj_setConst` does **not**
  recompute the broadphase bounding sphere (`geom_rbound`) or the midphase
  AABB (`geom_aabb`); both keep the nominal extents. MuJoCo culls candidate
  contact pairs using these stale bounds, so the ball never collides with the
  outer band of the enlarged stringbed. Reproduced: along the long axis,
  contacts stop at ~0.2065 m center distance = stale rbound 0.17 + ball
  0.0335 + margin 0.003 — exactly the *nominal* touching distance. The
  renderer draws from `geom_size`, so videos show the ball passing through
  visible racket geometry.
- **Impact:** The documented "early contact forgiveness" has essentially no
  effect at the racket tip; Stages 0–2 are materially harder than configured;
  the `racket_contact_scale` value recorded in run metadata is misleading.
  The test suite only asserts `geom_size` values
  (`tests/test_humanoid_tennis_curriculum.py:402-415`), never that the
  enlarged reach actually generates contacts.
- **Recommended fix:** After scaling `geom_size`, update `geom_rbound` (and
  AABB) to match — or avoid runtime resizing entirely by compiling the scale
  into the model per stage (e.g. via `mujoco.MjSpec`). Add a regression test
  that fires a ball at the enlarged-only zone and asserts a racket contact
  event.

#### C2. Stage 0 produces essentially zero reward events under the configured PPO exploration *(measured)*

- **Where:** `src/courtside_dynamics/recipes.py:306-321` (recipe);
  reward structure in `src/courtside_dynamics/envs/humanoid_tennis.py`
- **What:** The Stage 0 recipe (500k steps, stock SB3 PPO: `ent_coef=0.0`, no
  gSDE, iid per-step Gaussian exploration at 100 Hz) faces a reward stream
  that is almost identically zero. Measured in this review: 0 valid hits and
  0 nonzero rewards in 30 uniform-random episodes (a parallel probe measured
  0 in 264 episodes across uniform and Gaussian action noise); the ball never
  came within 0.5 m of the learned player's racket (minimum observed
  distance 0.52 m). Because `episode_len=150` (1.5 s) ends before the
  second-bounce fault (~1.9 s), episodes truncate at reward exactly 0 — the
  reward has literally zero variance, so PPO's advantages and gradients are
  null until a hit occurs by chance.
- **Corroboration:** the end-to-end 25k-step PPO run found exactly 1 success
  in 177 training episodes; all five evaluations scored 0.000 ± 0.000 (100%
  timeouts); value-function explained variance was negative throughout
  (final −0.72). Meanwhile a *constant* random action held for a full episode
  succeeds ~15% of the time (18/120 in a probe) — the task itself is easy;
  per-step iid noise at 100 Hz averages out to a near-still arm.
- **Recommended fix (any of, ideally several):** enable temporally correlated
  exploration (`use_sde=True`) or frame-stacked action repeat in the humanoid
  recipes; enable a small `valid_hit_shaping` by default (the escrow/clawback
  machinery already exists and is verified sound); or add a
  distance-to-ball/racket shaping term. Also consider extending
  `episode_len` past the miss-fault so failures at least produce −1 instead
  of indistinguishable zeros.

### 2.2 Major

#### M1. Double-bounce faults are systematically misreported as `out_of_bounds` *(confirmed)*

- **Where:** `src/courtside_dynamics/envs/tennis_rules.py:944-961`; priority
  table at `tennis_rules.py:753-757`
- **What:** `_handle_court_contact` checks in-bounds before the bounce count,
  and `OUT_OF_BOUNDS` outranks `SECOND_BOUNCE` in same-substep priority. A
  good ball whose second hop lands past the lines — the single most common
  unreturned-ball trajectory — is labeled `out_of_bounds` instead of
  `second_bounce`. In real tennis, once the first bounce is in, the point
  ends by double bounce regardless of where the second lands. Reproduced in
  the pure state machine and end-to-end (6/6 neutral-action episodes end
  `out_of_bounds` with `bounces=1` and an in-bounds first bounce).
- **Impact:** Reward, termination, and rally counts are unaffected. But the
  fault distribution consumed by the curriculum evaluator's `fault_flags`
  (`training/tennis_curriculum.py:673-676`) and `eval_info.csv`'s `term_*`
  rates reports "shots landing out" when the truth is "receiver never
  reached the ball" — misleading exactly the diagnostics used to debug
  curriculum failures.
- **Recommended fix:** Check the bounce count before bounds in
  `_handle_court_contact` (equivalently: once a confirmed in-bounds first
  bounce exists, classify the rally-ending court contact as
  `SECOND_BOUNCE`). Add unit tests for "second hop lands out".

#### M2. Free-standing "PD standing hold" does not stand, and no fall termination exists *(measured)*

- **Where:** default (Stage 6) mode; zero-action contract documented in
  `humanoid_tennis.py` docstrings and README
- **What:** With zero actions in free-standing mode, both G1s sink from
  pelvis z = 0.78 m to below 0.5 m within 1.2 s and settle collapsed at
  ~0.11 m — bit-identically across seeds; confirmed visually in renders. No
  humanoid-fall fault fires (`term_unsafe`/`term_humanoid_net` stay False);
  episodes end ~2.1 s in on a *ball* fault. The anchored Stages 0–2 mask
  this because both pelvises are welded.
- **Impact:** "A zero action is the two-player standing-reference PD hold"
  (README/API docs) holds only for welded robots. Free-standing training
  starts every episode with an unrewarded, unterminated balance problem, and
  the Stage 2 → Stage 6 gap is even larger than the missing Stages 3–5
  imply.
- **Recommended fix:** Add fall detection (pelvis height / torso orientation
  threshold) with a termination and/or an upright-alive reward term for
  free-standing mode; re-document the zero-action contract as anchored-only;
  treat a standing/balance controller as the true Stage 3 prerequisite.

#### M3. Stages 1–2 pay a constant −1 for every failure mode; the only bridging signal is disabled *(measured)*

- **Where:** `src/courtside_dynamics/envs/humanoid_tennis.py:1295-1363`
  (`_reward`), `:144-161` (`TennisRewardConfig.valid_hit_shaping = 0.0`)
- **What:** Under `VALID_TARGET_RETURN`, a whiff, a hit that lands out, and a
  net fault all pay exactly −1; valid racket hits pay 0 because
  `valid_hit_shaping` defaults to 0.0 and `_reward` zeroes the per-return
  reward for target stages. Measured: 30/30 random Stage 1 episodes and
  30/30 Stage 2 episodes ended at total reward exactly −1.000 with zero
  valid hits — no reward variance at all.
- **Impact:** Cold-start learning has no gradient toward hitting, let alone
  aiming. Warm-starting from Stage 0 solves hit discovery, but even a policy
  that hits reliably receives no signal distinguishing "hit, landed out"
  from "whiff" — and oracle data shows that distinction is the entire Stage
  1→2 task (the Stage 1 swing under Stage 2 randomization: 12/20 hits,
  0/20 target returns).
- **Recommended fix:** Enable escrowed `valid_hit_shaping` by default in the
  humanoid recipes (the escrow/clawback accounting was audited and has no
  double-pay path); consider grading target misses by landing distance to
  the target region.

#### M4. Promotion gate crashes on a 0-success serving side (Wilson interval rounding) *(confirmed)*

- **Where:** `src/courtside_dynamics/training/tennis_curriculum.py:270`
  (`_wilson_interval` / `SideSuccessSummary.__post_init__`)
- **What:** `_wilson_interval(0, 50)` returns a lower bound of ~6.9e−18
  (floating-point residual), violating the invariant
  `0.0 <= low <= success_rate` and raising
  `ValueError("side-success confidence interval is invalid")`. With the
  canonical 50-episodes-per-side suite, any policy that never succeeds from
  one side crashes the whole evaluation instead of reporting 0%. A symmetric
  upper-edge failure exists for perfect sides at some episode counts (n=25
  and n=30 round below 1.0; n=50/100 survive by platform luck).
- **Impact:** The promotion machinery fails exactly at the extremes an early
  curriculum run is most likely to produce. Tests never exercise a
  0-success side.
- **Recommended fix:** Clamp: `low = min(max(low, 0.0), rate)` and
  `high = max(min(high, 1.0), rate)`. Add gate tests for 0-success and
  all-success sides at n ∈ {25, 30, 50, 100}.

#### M5. A missing GL backend or moviepy crashes the whole training run at the first video *(confirmed + measured)*

- **Where:** `src/courtside_dynamics/callbacks/video_record.py:152-250`;
  wiring at `src/courtside_dynamics/training/train.py:789-801`
- **What:** `VideoRecordCallback` has no error isolation. On a headless
  machine without `MUJOCO_GL=egl/osmesa`, the first scheduled recording
  raises `mujoco.FatalError`, which propagates out of `model.learn()`. Only
  `KeyboardInterrupt` is salvaged (`train.py:956`), so `final_model.zip`,
  the final evaluation, and `stage_summary.txt` are never written. At the
  default `video_freq=100_000` this kills a run ~45+ minutes in. Reproduced
  empirically (training steps run fine without GL; `model.learn()` raises at
  the first video). The recording env also leaks if `VecVideoRecorder`
  construction raises, because the `try/finally: rec_env.close()` begins
  only after recorder construction.
- **Recommended fix:** Wrap recording in try/except, log-and-continue on
  GL/moviepy failure (optionally disable further attempts after the first
  failure); move recording-env creation inside the try/finally.

#### M6. Ball spin is observed in an unrecoverable body-local frame *(confirmed)*

- **Where:** `src/courtside_dynamics/envs/humanoid_tennis.py:1515-1517`
  (`_get_obs` exposing free-joint `qvel[3:6]`)
- **What:** MuJoCo expresses free-joint rotational velocity in the body-local
  frame, and the ball's orientation quaternion is (deliberately) not
  observed. After any spin-imparting impact, the observed vector is the
  world spin rotated by an arbitrary, unobservable orientation: physically
  different world spins (topspin vs. sidespin) can produce identical
  observations yet diverge sharply at the bounce (the ball/court pair has
  torsional/rolling friction). Only the spin magnitude is reliable. Verified
  empirically (with the body yawed 90°, qvel angular (1,0,0) is world
  (0,1,0)).
- **Impact:** Bounded today — all curriculum launches use zero spin and
  bounce friction is low — but the observation under-delivers the README's
  "ball position, velocity, and spin" exactly where spin would matter.
- **Recommended fix:** Expose world-frame angular velocity (e.g.
  `mj_objectVelocity` on the ball body, or rotate qvel by the ball
  quaternion) under the existing observation names.

#### M7. `stage_summary`'s "Best Checkpoint Evaluation" reports single-episode noise and omits the task metric *(confirmed)*

- **Where:** `src/courtside_dynamics/training/artifacts.py:654`
  (`write_run_summary`); `callbacks/info_dict_eval.py:222`
- **What:** The section renders `<key>_final` values — the terminal value of
  the *last* eval episode only (5 episodes for humanoid info-eval), while the
  episode-aggregated metrics logged at the same step (`success_rate`,
  `<key>_ep_mean`, the `rally_count` distribution) are read by
  `_read_eval_info_at_step` but never rendered. Humanoid recipes set no
  `headline_key`, so no episode aggregate appears anywhere in
  `stage_summary.txt`. Confirmed against the live run's artifacts.
- **Impact:** The one file a human reads first shows single-episode noise for
  sparse counters and omits the configured task metric
  (`success_key='stage_success'`).
- **Recommended fix:** Render `success_rate` and `*_ep_mean` (and the
  distribution keys) in the best-checkpoint section; set a `headline_key`
  (e.g. `stage_success`) in the humanoid recipes.

#### M8. 88–97% of the 58-dim action space is optimizer dead weight, and `n_envs=1` forfeits easy wall-clock wins *(measured / analysis)*

- **Where:** `src/courtside_dynamics/envs/humanoid_tennis.py:685-700`
  (mask applied before physics), `src/courtside_dynamics/recipes.py:199`
  (`n_envs=1` pin)
- **What:** The env zeroes masked action dims before mapping to controls
  (verified bitwise-inert in simulation), but PPO and SAC still model all
  58 dims: Stage 0 trains 2 live dims against 56 dims of pure ratio noise
  (96.6% inactive); Stages 1–2 train 7 (87.9% inactive). SAC's
  `target_entropy='auto'` resolves to −58 vs. a task-relevant −2/−7 (the
  README acknowledges this; the recipes don't mitigate it). Separately, the
  fixed-stage recipes pin `n_envs=1`; measured end-to-end training
  throughput is 38 steps/s (raw stepping: 103–108 steps/s; eval/video
  callbacks account for the difference), so the 0.5M/1M/2M budgets are
  ≈3.6h/7.3h/14.6h wall-clock on a 4-core CPU — roughly 25 h for the
  curriculum, undocumented.
- **Recommended fix:** Provide a stage-aware wrapper exposing only active
  dims (the active set is fixed per env instance given a serve side; see
  also N-F2 on alternation), or set `target_entropy` explicitly for SAC and
  document the PPO ratio-noise cost. Un-pin `n_envs` for PPO recipes — the
  callback cadences are already `n_envs`-independent
  (`_env_steps_to_calls`) — for a ~4–8× wall-clock reduction. Document
  expected wall-clock budgets.

### 2.3 Minor and informational

#### Rules engine

- **N-R1.** A rule fault on the exact truncation step returns
  `terminated=True` **and** `truncated=True` (`humanoid_tennis.py:1265-1275`).
  SB3 resolves the ambiguity correctly; other consumers may bootstrap a fault
  state. *Fix:* suppress `truncated` when `terminated` is already true.
- **N-R2.** Events in substep groups after an in-batch fault vanish from all
  `RallyTransition` partitions and `event_count_*` info keys
  (`tennis_rules.py:557`) — inconsistent with the already-terminal path.
  *Fix:* `ignored.extend()` the remaining groups before breaking.
- **N-R3.** A dying/rolling ball never emits the rally-ending second bounce —
  the shared `ball_court` contact channel stays latched once contacts are
  closer than `release_substeps` (2 ms), so the episode idles to timeout
  (`_tennis_events.py:399`). *Fix:* add a low-speed/rolling detector that
  closes the rally, or a persistent-contact timeout.
- **N-R4.** README calls the observation tail `[193:298]` "bounded", but
  `rally_count` (obs index 216) is an unbounded raw count — relevant because
  this region is excluded from normalization. *Fix:* document, cap, or scale
  it (e.g. `rally_count / rally_target`).
- **I-R5.** Deliberate rule divergences from real tennis, correctly
  documented but worth restating: any ball-net touch is a fault (even a
  let-cord that lands in); any racket re-contact > 2 ms apart is a double
  hit (stricter than ITF). The curriculum evaluator's fault-flag classifier
  covers only 7 of 17 termination reasons
  (`training/tennis_curriculum.py:668`).

#### Environment core

- **N-E1.** Target-miss termination pays `fault_penalty` for a tennis-legal
  return, and rally info still reports the rally as unterminated
  (`humanoid_tennis.py:1345`) — coherent for the curriculum objective,
  confusing in logs. *Fix:* a distinct info flag / termination label for
  curriculum target misses.
- **N-E2.** A non-finite policy action raises `ValueError` mid-step instead of
  terminating with the unsafe-physics penalty like every other non-finite
  path (`humanoid_tennis.py:692`). *Fix:* route through the safety-event
  path.
- **I-E3.** Stage-success termination reports
  `termination_reason_name='none'`; success is only encoded in
  `term_stage_success`. Downstream code keying on the reason name will
  misread success terminals. *Fix:* add an explicit success reason.
- **I-E4.** Positive audit note: the shaping escrow has no double-pay path;
  clawback correctly fires on truncation of an in-flight return and is
  net-zero on Stage 0 success.

#### Physics (measured characteristics, not bugs)

- **I-P1.** Oblique-bounce friction is low: a 10 m/s skidding impact leaves
  with 70.6 rad/s topspin (rolling ≈ 242 rad/s; real balls leave near or
  above rolling) and retains 83% of horizontal speed (real hard courts
  ~60–80%). Spin-based play is weakly supported. *Fix if desired:* raise
  ball/court friction (incl. rolling/torsional) toward measured tennis
  values.
- **I-P2.** Vertical COR at high speed is slightly bouncy (0.795 at a fast
  oblique impact; real balls' COR decreases with speed). Ball penetration at
  a 16 m/s impact is 4.3 mm of the 33.5 mm radius — acceptable soft-contact
  artifact.

#### Curriculum & promotion

- **N-C1.** Git SHA provenance doesn't detect a dirty working tree — locally
  edited code records a clean upstream SHA
  (`training/tennis_curriculum.py:881`). *Fix:* record
  `git status --porcelain` emptiness alongside the SHA.
- **I-C2.** Held-out suite seeds are not enforced disjoint from training
  seeds (held-out-ness is per exact physical state); promotion evidence
  objects are unauthenticated (advisory gate, not tamper-proof). Documented
  behavior; listed for awareness.
- **I-C3.** Measured curriculum gaps: the Stage 1 timed swing is 0/20 under
  Stage 2 randomization (by design); a fixed reach motion intercepts only
  65% of Stage 2 launches, so Stage 2 genuinely requires adaptive reaching.
  No Stage 2 oracle exists — solvability across the full randomized launch
  box with the reduced 1.35× stringbed is unproven (and C1 currently reduces
  the effective stringbed further). *Fix:* add a Stage 2 oracle
  (feed-parameterized swing timing) once C1 is fixed.

#### Logging & diagnostics

- **N-L1.** `eval_info.csv` is append-only across runs; reusing a `log_dir`
  silently mixes stale rows into plots, headline stats, and the
  best-checkpoint lookup (`callbacks/info_dict_eval.py:329`). *Fix:*
  truncate on run start or include a run id column.
- **N-L2.** `plot_eval_info`'s metric-name splitter mangles the
  `_ep_min/_ep_p50/_ep_p90/_ep_max` distribution metrics the humanoid
  recipes emit (`notebook_utils.py:320`). *Fix:* split on the known suffix
  list.
- **N-L3.** KeyboardInterrupt salvage protects only `model.learn()`; a Ctrl-C
  during the save/eval epilogue loses `stage_summary` (and possibly
  `final_model`) (`train.py:956`). *Fix:* extend the handler around the
  epilogue.
- **N-L4.** `config.json` omits env constructor kwargs, so non-curriculum
  runs aren't reconstructible from disk; `git_sha` is `null` unless
  `train()` runs with the repo as CWD (measured: null when launched from
  another directory) (`artifacts.py:26,97`). *Fix:* serialize
  `env_kwargs`; resolve the git root from the installed package path (or
  `__file__`), not CWD.
- **N-L5.** The humanoid recording CSV omits `rew_stage_success` (and
  `rew_action_cost`), so a +1 success row shows all reward components 0
  (`recipes.py:70-101`). *Fix:* add the keys to
  `_HUMANOID_TENNIS_CSV_KEYS`.
- **I-L6.** Humanoid info-eval metrics (including `success_rate`) are
  computed from only `n_eval_episodes // 4` = 5 episodes (`train.py:822`);
  `monitor_log` assigns worker ids by lexicographic file order, mislabeling
  workers at `n_envs ≥ 11` (`monitor_log.py:97`).
- **I-L7.** Positive audit notes, verified live: the env's rich info keys
  genuinely reach TensorBoard and `eval_info.csv` (51 metrics, long format,
  no NaN columns); `evaluations.npz`/`eval_info.csv` timesteps align; a full
  curriculum config serializes into `config.json` (including all 106
  normalization-exclusion indices); SIGINT salvage works exactly as the
  README promises (7 files including summary and final model, ~8 s);
  `best_vec_normalize.pkl` snapshots at each new best;
  `check_run_artifacts` accurately explains missing artifacts.

#### Video recording & replay

- **N-V1.** `record_best_model_video` encodes at 60 fps against the 100 Hz
  control/render stream — best-model replays are 0.6× slow motion and
  inconsistent with training videos (`notebook_utils.py:539`). *Fix:* use
  `env.metadata["render_fps"]`.
- **N-V2.** Recording never selects the tuned court cameras: the default free
  camera frames the whole scene (robots ~10 px tall, ball invisible; mean
  inter-frame change 0.006/255). The model ships a `sideline` camera that
  clearly shows net, robot, racket, and ball
  (`video_record.py:152`, `assets/humanoid_tennis.xml:44`). *Fix:* pass
  `camera_name="sideline"` (or a recipe-configurable camera) to the
  recording env.
- **N-V3.** Every recorded step renders twice — an explicit
  `rec_env.render()` duplicates `VecVideoRecorder`'s internal capture
  (`video_record.py:246`). Dead code; at ~0.48 s/frame under OSMesa it
  doubles a cost that already runs ~48× slower than realtime. *Fix:* delete
  the extra call.
- **N-V4.** Silent normalizer fallbacks: `_load_obs_normalizer` falls back to
  identity on any load failure (`notebook_utils.py:520`), and a
  `sync_envs_normalization` failure is swallowed (`video_record.py:184`) —
  either would record garbage-behavior videos of a competent policy with no
  warning. *Fix:* warn loudly (or fail) on both paths.
- **N-V5.** Pinning `csv_header` without `info_row_fn` produces CSV rows
  misaligned with the header (`video_record.py:127`). *Fix:* validate the
  pair at config-build time.
- **N-V6.** All humanoid training videos show a single serve side and an
  identical launch (video_length ≥ episode length ⇒ exactly one episode per
  recording; the recording env's serve alternation restarts each rollout).
  Mirrored-side and Stage 2 randomization behavior is never visually
  sampled. Quick-test's `video_length=750` also makes filenames claim
  "step-0-to-step-750" for 151-frame videos. *Fix:* record ≥2 episodes per
  video or alternate recorded serve sides; name files from actual frames.
- **N-V7.** `record_best_model_video` leaks the MuJoCo env if the rollout
  raises (`notebook_utils.py:566`). *Fix:* try/finally.
- **I-V8.** Positive audit notes, verified live: videos record the *current*
  policy deterministically; the recording env is built from the same
  `env_fn` as training so stage/config always match; VecNormalize handling
  is exemplary (exact `SelectiveVecNormalize` reconstruction with the
  193–298 exclusion tail, frozen stats, live sync; paired
  `best_model.zip`/`best_vec_normalize.pkl`); encoding is exactly realtime
  (100 fps metadata = 100 Hz control, verified on disk); replay-from-info is
  bit-exact (max divergence 0.0 over 100 steps reproducing a randomized
  Stage 2 launch from reset-info on a fresh env). The reset-options replay
  path omits angular velocity, but all seven launch presets use zero spin
  and reset writes spin deterministically from env config, so replay is
  unaffected today.

#### Learning feasibility

- **N-F1.** Notebook `EARLY_STOP_PATIENCE=20` is mathematically inert for
  Stages 0–1 (their budgets contain fewer than 20 evaluations)
  (`train.py:761`). *Fix:* scale patience to the eval budget.
- **N-F2.** Alternating the learned side each episode doubles the task: two
  mirrored skills in disjoint action slices with no symmetry
  canonicalization (`humanoid_tennis.py:716`). *Fix option:* canonicalize
  observations/actions into the learned player's frame, or train one side
  and mirror weights.
- **N-F3.** The promotion suite evaluates launch perturbations Stages 0–1
  never sample in training (held-out noise vs. deterministic training
  launches) — the 80% gate implicitly demands generalization the training
  distribution doesn't teach. *Fix:* either add launch noise to Stage 0–1
  training or align the gate suite with the training distribution.
- **N-F4.** Curriculum dead-ends: Stages 3–5 are unimplemented and the Stage
  2 → 6 jump is unbridged — free-standing mode today means two robots that
  cannot stand (see M2).
- **I-F5.** VecNormalize reward normalization meets an all-zero reward
  stream: the running return variance is ~0, so the first sparse success is
  amplified to the clip ceiling — an instability spike exactly when learning
  should begin (`train.py:449`). *Mitigation:* disable reward normalization
  for sparse stages, or warm the return-variance estimate.
- **I-F6.** Positive evidence: Stage 0/1 oracles pass on both mirrored sides
  under their exact launch configs; the credit-assignment horizon
  (~0.5–1.2 s from action to outcome) is compatible with γ=0.99 at 100 Hz;
  the selective normalization scheme (indices 0–192 normalized, bounded tail
  raw) is the right design for cross-stage normalizer transfer, and the
  notebook's warm-start path correctly transfers policy+obs_rms while
  resetting reward normalization and optimizer state.

## 3. Empirical measurements

All numbers from a 4-core CPU container, MuJoCo 3.10, `MUJOCO_GL=osmesa`.

### Throughput and determinism

| Measurement | Value | Note |
|---|---|---|
| Raw env stepping (zero actions) | 103–108 steps/s | ~1.05× realtime at 100 Hz control; all stages ≈ free-standing |
| End-to-end PPO training | 38 steps/s | 25k quick run = 10 m 55 s; eval/video callbacks dominate |
| `reset()` latency | ~1.6 ms | mean of 5 |
| `render()` under OSMesa | 0.48 s/frame | recording ≈ 48× slower than realtime |
| Trajectory reproducibility | bitwise | SHA-256-identical obs/reward streams across env recreation (50 steps, Stage 2 and free-standing) |
| Serve alternation | exact | unseeded resets strictly alternate; new seed restarts at the initial side |
| Action-mask inertness | bitwise | random values in masked coords vs. zeros: identical 25-step trajectories |

### Physics plausibility

| Quantity | Measured | Reference |
|---|---|---|
| Court restitution (6 m/s drop) | 0.763 | 0.73–0.76 real hard court |
| Vertical COR, fast oblique impact | 0.795 | real COR decreases with speed |
| Horizontal speed retention, 10 m/s skid | 83% | ~60–80% real |
| Post-skid topspin | 70.6 rad/s | ~242 rad/s rolling |
| Drag decel @ 16 m/s | 5.45 m/s² | 5.27 (env Cd 0.55) / 4.79 (Cd 0.50) |
| Feed net clearance (all modes) | 0.79–0.92 m | all first bounces in bounds |
| Free-standing zero-action collapse | pelvis < 0.5 m at 1.2 s | settles ~0.11 m; identical across seeds; no fall fault |

### Scripted oracles

| Oracle | Result | Detail |
|---|---|---|
| Free-standing legal return | 10/10 | rally at step 239, reward +1.0, exact A/B mirror, seed-invariant |
| Stage 0 intercept | 10/10 | success at step 67 both sides; total reward exactly +1.0 |
| Stage 1 timed return | 10/10 | success at step 117, rally 1, reward +1.0 |
| Stage 1 swing under Stage 2 randomization | 0/20 success | 12/20 hit; 10× failed_to_cross, 2× own-body, 8× whiff; each −1.0 |
| Stage 0 reach under Stage 2 randomization | 26/40 hits | 65% interceptable with a fixed motion; identical per-seed across sides |

### Cold-start reward density

| Config | Random policy (30 eps; free: 10) | Zero action | Termination profile |
|---|---|---|---|
| Stage 0 (150 steps) | 0/30 success, reward 0.0 | 0.0 | 100% timeout; min ball–racket distance 0.52 m; zero near-misses < 0.5 m |
| Stage 1 (300 steps) | 0 hits, reward −1.0 constant | −1.0 | 28× second_bounce, 1× out_of_bounds, 1× failed_to_cross |
| Stage 2 (300 steps) | 0 hits, reward −1.0 constant | −1.0 | 27× second_bounce, 2× failed_to_cross, 1× out_of_bounds |
| Free-standing (1000 steps) | reward −1.0 constant | −1.0 | 100% "out_of_bounds" at ~205 steps (robots collapsed; see M1 for the label) |

### End-to-end PPO quick run (Stage 0 recipe, 25k steps, seed 0)

| Check | Result |
|---|---|
| Run completion | exit 0, zero warnings |
| Artifacts vs. README table | 12/12 present, well-formed |
| `notebook_utils` audit | accurate (correctly explains the 4 notebook-side artifacts) |
| SIGINT salvage | works: config, summary, monitor CSV, progress.csv, final model + normalizer, final eval — in ~8 s |
| Eval reward (all 5 evals) | 0.000 ± 0.000 (100% timeout) |
| Training successes | 1 of 177 episodes |
| PPO health (final) | approx_kl 0.107, clip_fraction 0.57, explained_variance −0.72 |
| `git_sha` in config.json | null (launched outside repo CWD; see N-L4) |
| Videos | 2 mp4 @ 100 fps, 151 frames each (filenames claim 750; see N-V6) |

## 4. Prioritized remediation plan

1. **C1 — stringbed collision bounds.** Highest leverage: it silently changes
   task difficulty and invalidates recorded metadata. Fix + regression test.
2. **C2/M3 — give the stages a learnable gradient.** Correlated exploration
   (gSDE/action repeat) for Stage 0; default-on escrowed hit shaping for
   Stages 1–2; consider graded target-miss distance. These are recipe/config
   changes, not env rewrites.
3. **M1 — fault attribution order.** Small, isolated rules fix; makes every
   downstream fault-distribution diagnostic truthful.
4. **M4 — Wilson interval clamp.** One-line fix plus edge-case tests; unblocks
   promotion evaluation of early (0%-on-a-side) policies.
5. **M5 — video callback isolation.** Try/except + graceful degradation;
   protects hours-long runs. Bundle N-V1 (60 fps), N-V2 (sideline camera),
   N-V3 (double render).
6. **M7 — surface the task metric** in `stage_summary`; set `headline_key`.
7. **M8 — un-pin `n_envs`** for PPO recipes; document wall-clock budgets;
   optionally add an active-dims wrapper / explicit SAC `target_entropy`.
8. **M2 — free-standing reality.** Fall termination + upright term (or
   re-document zero-action as anchored-only); plan a standing-controller
   stage before Stage 6 work.
9. **M6 + remaining minors** as convenient: world-frame spin, git-SHA
   resolution from package path, per-run `eval_info.csv`, success termination
   reason, recording CSV reward keys, plot suffix handling.

## Appendix: review environment

- Container: 4-core CPU, no GPU; Ubuntu 24.04; Python 3.11.15.
- Rendering: no system GL initially; `libosmesa6`/`libegl1` installed for the
  probes (`MUJOCO_GL=osmesa`; `egl` also works but emits cosmetic teardown
  errors on GPU-less boxes).
- Packages: mujoco 3.10.0, gymnasium 1.3.0, stable-baselines3 2.9.0,
  torch 2.13.0 (CPU).
- The review's probe scripts and raw outputs (feed trajectories, restitution
  and drag CSVs, workspace sweeps, oracle traces, reward-density stats, the
  full 25k-step run directory, and rendered stills/videos) were produced in
  the session workspace; the methodology above is sufficient to reproduce
  every number from a fresh clone.
