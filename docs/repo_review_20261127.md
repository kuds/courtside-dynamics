# Repository review: goals, training setups and next steps (2026-11-27)

Kind: **Review snapshot**, pinned to `main`@`8df9277` (the merge of
PR #81, "week-1 foundations", 2026-10-18) and conducted 2026-11-27.

## Method

- **Scoped reviewers.** Each scope read its code and docs and
  reproduced its claims with scratch scripts. The scopes were:
  - goals and history;
  - the training setups, with every recipe resolved through
    `build_train_config` without training;
  - what week 1 delivered against its three designs;
  - the status at HEAD of each of the 98 findings in
    `repo_review_20261016.md`;
  - the repository's quality gates;
  - run readiness: the planned next runs, executed at small scale;
  - the humanoid and GPU track.
- **Bug review.** The week-1 diff (`b9585e2..HEAD`, about 54k lines)
  was split into five scopes, each with a bug finder followed by an
  adversarial verifier. The scopes were run records and sync,
  offline evaluation, PPO and training, digests and linalg, and the
  Colab notebooks. Only findings the verifier could not refute are
  reported (§7).
- **External state.** The Google Drive run archive and the GitHub
  Actions history were checked through their APIs.
- **Stack.** Fresh resolve on 2026-11-27: Python 3.13, MuJoCo 3.15.0
  (the pyproject cap), gymnasium 1.4.0, SB3 2.9.2, CPU torch 2.15 and
  numpy 2.5.3, on 4 shared CPUs.
- **Snapshot scope.** The body describes `main`@`8df9277`. Fixes made
  afterwards on the same branch are listed in §11, and short inline
  notes point there.

## 1. Bottom line

1. **Where the ladder stands.**
   - Ball control and WallBall are done; WallBall's goal was met and
     replicated on 2/3 seeds.
   - PaddleTennis is the active rung, and it is stuck. Over about 25M
     SAC steps in 16 runs, k=1 receiving reached 90-100%, but the
     policy's own second return (k=2) never left the noise floor: 1.6%
     on fresh seeds at best, against a 5% bar.
   - Six levers were falsified on k=2 and two were retired before
     launch.
   - The top-ranked explanation, a context gate on rally bookkeeping,
     has never been tested by an intervention.
   - The humanoid track has scripted oracles only.
2. **Nothing has trained for 90 days, and nothing has been committed
   for 40.**
   - The last run ended 2026-08-29, and the Drive archive has had no
     write since 08-30.
   - Week 1 (10-16 to 10-18) delivered its infrastructure plan, about
     25k hand-written lines plus 21.6k of goldens, and ran none of the
     runs it booked. The day-0 PPO reference, the August re-score and
     spike S1 were all due that week.
   - That is the "runless stretch" the 2026-10-16 campaign design was
     written to prevent, and this review treats it as the main risk.
3. **CI on `main` has been red since 2026-11-16, although no code
   changed.**
   - Seven PaddleTennis golden digests fail on every Python leg since
     GitHub's runner moved to Ubuntu 26.04. Every gate passes locally.
   - A same-day A/B confirmed the runner image as the cause, most
     likely through the OS math library. CI is now pinned to
     `ubuntu-24.04` on this review's branch.
   - It means bit-identity is platform-dependent, and Colab's digests
     have never been recorded (§5).
4. **One training setup is current: SAC on PaddleTennis.**
   - The PaddleTennis PPO profile is unprobed. It also differs from the
     pre-registered day-0 PPO reference on 8-9 settings: γ, network,
     batch, epochs, gSDE and cadences.
   - The pre-registered call is not "no early stop" as written, and its
     reading table confounds algorithm with discount.
   - In-loop evaluation is the dominant cost: about 31-39% of L4 wall
     clock for SAC, and 75% of CPU wall clock at the recipe's PPO
     cadence (§4).
5. **The next runs are executable today, with two fixes.**
   - The PPO reference, the in-loop and `@v3` re-scores and `select()`
     all ran end to end here, including on a real August run.
   - **First fix:** on a host without OpenGL, any run that records
     video (the default) aborted at its second video attempt (N1).
     Fixed on this review's branch.
   - **Second fix:** the notebooks refused a Colab CPU runtime (N4).
     Fixed on this review's branch.
   - **Throughput** is better than the design assumed: PPO collects
     1.2-2.7k steps/s, so 3 seeds × 10M take about 3 h on 4 vCPUs
     (§8).
6. **The week-1 code holds up, with one notable new risk.**
   - All 18 bug-review findings survived verification, but most were
     downgraded to low.
   - **The new risk:** CPU re-scores depend on the host's MKL kernels,
     so the same snapshot under the same protocol scores differently on
     AVX2 and AVX-512 Colab hosts (N2). Pinning `MKL_CBWR` fixes it
     (fixed on this branch, §11).
   - **Older findings:** of the 98 from 2026-10-16, 82 remain open (81
     after this branch, §11),
     including 7 of the 10 medium ones. The open ones that matter are
     the paired `confirm_best` bias and the misleading
     reporting (§6.3).
7. **The GPU plan has a version problem.** mjlab pins MuJoCo 3.11,
   Playground 1.x needs ≥ 4.0, and the CPU court is capped at 3.15. S1
   is ready but has never run (§9).
8. **Next (§10):**
   - restore CI;
   - fix the video abort;
   - launch the PPO reference as pre-registered, with a γ control;
   - re-score the August runs under `@v3` with MKL pinned;
   - record the Colab digests;
   - run S1.

   Then test the context gate directly, and hold further
   infrastructure until runs are in flight.

## 2. Goal and where the ladder stands

The project has one stated goal: teach humanoid agents to rally and
play tennis (`README.md:5-11`). Since 2026-10-16 the long-run target
has been humanoid doubles, rung H4 of
`design_env_ladder_and_structure_20261016.md`. The strategy behind it:
- learn the tennis "brain" on cheap CPU paddle courts;
- learn the G1 "body" once, on GPU;
- join the two last;
- certify every humanoid rung with a scripted brain first, so doubles
  never waits on k=2. P1, today's PaddleTennis, therefore "gates
  nothing downstream" (§3).

| Rung | What success means | Status at HEAD |
|---|---|---|
| BallBalance, BallBounce | Trained agents; PPO BallBalance holds 750/750 on 20 held-out seeds | **Done** |
| WallBall | A sustained ≥3.0-return rally from the workspace baseline | **Done; chapter closed 2026-08-02.** Goal met on run `20260728_225217` and replicated on 2 of 3 seeds. The true-baseline era passed on reliability (2/2) but not on skill (1/2) |
| **PaddleTennis (P1)** | Registered bar k=2 ≥ 5% (RK1). Ladder claim: k=2 ≥ 10% on 3 held-out seeds, or a capped lever budget with a recorded falsification | **Active and stuck.** k=1 receiving reaches 90-100%, but k=2 has never left the noise floor. The best value, 5% on calibration seeds, re-measures at 1.6% [0.3, 4.5] on fresh seeds |
| R0 sanity tasks, P2 duel to P6 doubles | Per-rung bars in the ladder design §3 | **Planned.** None of the code exists: no `CourtEnv`, `ObsLayout` or task registry. Five env IDs are registered |
| H0 footwork to H4 G1 doubles (GPU) | H0: fewer than 1 fall per 100 s. H1: within 10 cm and 15° at strike. H4: bars pre-registered at entry | **Planned, behind spikes S1-S3.** S1 is built but has never run on an L4 |
| `HumanoidTennisCoop`, the centralized 58-dim env, Stages 0-2 | Advisory promotion gate | **Slated for archive (M8, not started).** Scripted oracles only, no learned result. The design forbids further training |

## 3. What the project has learned

### 3.1 Chronology

**WallBall** (2026-07-14 to 08-02):
- The baseline-era levers closed negative.
- The sliding-fence depth ladder was retired (0.24.0): no scripted
  reference reaches the bar on any rung.
- Easier aligned serves were falsified: they trained weaker policies.
- Direct goal-task training met the campaign goal (0.24.0) and
  replicated on 2/3 seeds.
- The true-baseline era (0.25.0) closed the chapter with "an
  environment's dominant strategy is set by its physics".

**PaddleTennis** (2026-08-02 to 09-05). About 25M SAC steps across 16
completed runs; the verdicts are in `docs/DECISIONS.md`:

| Date | Lever or era | Verdict |
|---|---|---|
| 08-02/03 | Env freeze; volley rules | The first run found a close-net patting loop. Pre-bounce returns now fault: the ground rules |
| 08-08 | Ground-rules baseline | One memorized macro: serve return at 98%, k=2 0% |
| 08-09/10 | Exploration package; contact escrow 0.25 | k=1 100% (one point per episode), k=2 0%. Both adopted |
| 08-10 to 15 | n-point play from scratch | Collapses to a zero-contact "statue". Stop/pivot |
| 08-16 | Reach escrow 0.25, r = 3 m | **ADOPT**: k=2 1% at five checkpoints |
| 08-16 to 22 | Registered 3M run | Learns from scratch, k=1 95%. **RK1 FAIL** (k=2 ≤ 1%) |
| 08-18 to 22 | Post-swing hold escrow | Closed without adoption |
| 08-28/29 | LT1, the temperature skip | Every bar FAILS. The optimizer-side lever is spent |
| 08-30 | Command-rate limiter | Killed by the PT2 measurement before any code |
| 09-02 to 05 | k=2 drill; LD1′ demo injection (`DemoSAC`) | The drill pilot was retired before freeze. LD1′ was built default-off and has never been frozen or run |
| 10-05 | Instrument batch | Selection follows the policy's own k≥2 returns; evaluation is paired |
| 10-16 to 18 | Week 1 | Infrastructure only (§6); no run |

### 3.2 The k=2 problem

**Definitions.**
- **`episode_rally_returns_a`** is the selection and success target
  since 2026-10-05. It counts side A's confirmed returns that were not
  its first of the point, summed over a 1,500-step n-point episode.
- **The booked k=2 numbers** use a different instrument: per-point
  exchange survival on calibration seeds.
- No August checkpoint has been scored on the new metric yet. That is
  the job of the planned re-score (§10).

**The best measured account (PT2, 2026-08-30) is a
directed-interception gap.** The policy's paddle sits 2.6-3.3 m from
the return's bounce; the oracle's sits 0.89 m away and converts 98.2%.
Opportunities are not scarce: 1.3-1.7 per episode.

| # | Hypothesis | Status |
|---|---|---|
| 1 | **Context gate.** The policy keys on rally bookkeeping (phase, counters, the clock) rather than physics. The same physical states convert 6.9% as a fresh feed and 1.3-2.0% in real rally context; the oracle converts 77.5% in both | **Open, top-ranked.** Measured on one checkpoint; no intervention has tested it (the context-blind pilot and LD1′ have not run) |
| 2 | Narrow incoming-ball distribution: a memorized serve-return macro | Open. The "ball machine" was never built |
| 3 | Credit spans more than the discount horizon: γ = 0.99 against about 210 steps between side-A hits (0.99^210 ≈ 0.12) | **Open and never varied.** Every SAC PaddleTennis run used SB3's default 0.99 |
| 4 | Home is the wrong place to stand | Open (task design) |
| 5 | SAC critic pathology: 55-61% dormant units, Q near action-blind at failure states | Open. The PPO reference is the designed test |
| 6 | v2 value-target biases: time-to-go in `obs[35]` while timeouts bootstrap; clawbacks paid on the truncation step | Judged small. The fix is v3 (M12), not built |
| — | Opportunity scarcity, stroke quality, opponent asymmetry, budget, post-swing drift or hold, command thrash, entropy collapse | **Refuted** |

**The most decision-relevant experiments are booked, and none has
run.** They are the context-blind pilot (named "next" since
2026-10-05), the PPO reference, the SAC γ=0.995 arm and the re-score.

### 3.3 Process lessons, and whether they held

The 2026-10-16 campaign design diagnosed three habits:
- **S6:** code comes before the cheap test.
- **S7:** prose serves as the experiment ledger.
- **S8:** runs are rare and n=1.

It ruled: "Don't let the migration become another runless stretch."

The record since then:
- **Week 1.** 24 commits, 21 of them infrastructure, 0 runs. About
  25k hand-written lines plus 21.6k lines of goldens and fixtures.
  `DECISIONS.md` grew from 1,159 to 2,746 lines, every new entry an
  infrastructure entry. Its PaddleTennis section has not changed since
  2026-10-05.
- **Since 2026-10-18.** No commits for 40 days. The last training run
  ended 2026-08-29, **90 days ago**.
- **Drive agrees.** The `courtside-dynamics` folder has no file
  created or modified after 2026-08-30.

The core of week 1 was justified, and some of it was needed directly:
the gates, the PPO fixes, the harness and `@v3`. But the habit the
design was written to stop has continued, and this review counts it as
the project's main risk.

## 4. Current training setups

Every number in this section comes from configs resolved at HEAD:
`build_train_config(recipe, algo=…)` for each recipe and algorithm,
without training. None of these setups has trained yet. The PPO
profiles, `paddle-npoint@v3`, snapshots, `sync_dir` and the campaign's
PPO plan all arrived after the last run.

### 4.1 Recipes

All 15 recipes build. Their status:

| Status | Recipes |
|---|---|
| **Current rung** | `PaddleTennis` (SAC by default, 2M steps, selection protocol `paddle-npoint@v3`) |
| Done | `BallBalance` (1M), `BallBounce` (1.5M) |
| Closed WallBall chapter | `WallBall`, `WallBallVolley`, `WallBallBaseline` (1.5M each); `WallBallGoalRally` and `WallBallTrueBaseline` (6M each, performance-gated, so they have no re-scorable protocol) |
| Historical | `WallBallBootstrap`; `WallBallDepthCurriculum` and `WallBallDepthCurriculumAligned` (the only two that refuse PPO: their gate levers act on SAC's replay buffer and temperature) |
| Experimental | Humanoid Stages 0-2 (PPO by default; 0.5M, 1M and 2M) and `HumanoidTennisCoopSmoke` (10k) |

Other facts that hold across the recipes:
- `DemoSAC` builds on every recipe.
- Every recipe resolves to `vec_env="dummy"`. The PPO plan's
  `subproc` workers are opt-in, and no recipe uses them.
- `eval_freq` and `checkpoint_freq` count total environment steps
  summed across workers (`train.py:1760-1768`).

### 4.2 The PaddleTennis recipe

**Task.**
- Continuous n-point play in fixed 1,500-step episodes (15 s). A fault
  pays its penalty and the next point launches from where play left
  off.
- The P3 serve band: 3.25 m behind the net, 9 ± 1 m/s, 21 ± 3°.
- The opponent is the bounce-waiting ground oracle on side B, played
  through an exact mirror.
- `volley_rule="fault"`.
- Rewards: +1 per confirmed return **by either side**, −1 per fault,
  −2 for unsafe physics.

**Spaces.**
- Observation: 48 dims. [0:24] is physical and the only normalized
  block; [24:36] is rally state; [36:48] are contact latches.
- Action: 3 dims.

**Training aids** (training env only):
- the contact escrow pays +0.25 per legal hit, clawed back unless the
  hit's return is confirmed;
- the reach escrow pays `0.25 × max(0, 1 − d/3 m)` at the incoming
  bounce, kept only if a legal hit follows.

`paddle-npoint@v3` strips both, so every evaluation reward is task
reward.

**SAC block.**
- `use_sde=True`, `ent_coef="auto_0.02"`, `target_entropy=-1.5`,
  `train_freq=(64, "step")`, `gradient_steps=-1` (1:1).
- γ 0.99 (the SB3 default), net 256×256, `learning_starts` 100,
  `device="auto"`.
- 8 workers.

**Selection (`paddle-npoint@v3`).**
- 30 paired, deterministic CPU episodes per evaluation, on seeds
  `seed + 1,000,000 + i`, with alternating serves.
- Lexicographic keys: the k≥2 headline (Δ 1/60), then `success_rate`
  (Δ 1/60), then task reward (Δ 0.25).
- A 30-episode confirmation batch runs on a second block whenever a
  candidate beats the best.
- **While k≥2 is zero, the third key decides.** That key is task
  reward: confirmed returns by either side, minus faults.

**Cadence.**
- An evaluation every 25k steps (80 per run).
- A checkpoint plus a 30-episode diagnosis every 250k.
- A snapshot at every evaluation: about 3.5 MB each, about 280 MB per
  run.
- Early stopping:
  - patience 20, so the earliest stop is evaluation 40 of 80;
  - the degenerate guard can stop a run from evaluation 5.

### 4.3 Evaluation dominates the wall clock

N-point evaluation episodes always run the full 1,500 steps, and every
evaluation env is a single serial worker (`train.py:3723-3727`). Rates
measured here on one thread:

| Stream | Rate |
|---|---|
| SAC training | 80 env steps/s |
| PPO training | 1,611 env steps/s |
| Batch-1 evaluation | 860-955 steps/s |

L4 rates were back-solved from the August FPS.

| Setup | Eval steps per training step | Eval share of wall clock (normal / confirming evaluation) |
|---|---|---|
| SAC recipe on an L4 (estimate) | 2.3× | **31-39% / 44-53%** |
| SAC recipe on this CPU | 2.3× | 18% / 28% |
| PPO recipe profile, CPU | 1.8× | **75% / 84%** |
| PPO campaign plan, CPU | 2.6× | **82% / 88%** |
| PPO day-0 reference cadence (eval every 245,760), CPU | 0.32× | 35% / 46% |

At the recipe's cadence, a 2M-step PPO run takes about 21 minutes of
training and 63 minutes of evaluation on this CPU.

The run record's `eval_share` times only the two evaluation callbacks.
It leaves out the 45k-step checkpoint diagnosis and the videos, so it
will under-report this cost.

### 4.4 PPO: the recipe profile is not the pre-registered reference

The day-0 PPO reference (`design_env_ladder_and_structure_20261016.md`
§7, item 1) still builds as a single call at HEAD. The recipe's own
PaddleTennis PPO profile (`recipes.py:1545-1576`) is marked "Not yet
probed", and it differs from that reference:

| Field | Day-0 reference | Recipe profile at HEAD |
|---|---|---|
| batch, epochs | 2048, 5 | 256, 10 |
| γ | 0.995 | 0.99 |
| gSDE resample, `log_std_init` | every 32 steps, −1.0 | every 64 steps, −2.0 |
| network | 256×256 (pi and vf) | SB3 default 64×64 (15k parameters, against 158k) |
| `ortho_init` | True (SB3 default) | False |
| eval and checkpoint cadence | 245,760 and 491,520 | 32,768 (every 4 rollouts) and 245,760 |
| patience; budget | none; 10M | 20; 2M |

Three consequences:
- **The recipe's 10 epochs never act.** `target_kl=0.02` ends every
  update inside its first epoch (measured: approx_kl 0.010-0.015 after
  one epoch).
- **The day-0 call is not "no early stop".** It overrides patience but
  not `early_stop_degenerate_evals`, so the inherited degenerate guard
  can stop it at evaluation 7 (1.72M steps), before the
  pre-registered 3M readout. Add `early_stop_degenerate_evals=0`.
- **The reading table confounds algorithm with discount.** Its row 2
  ("k=2 ≥ 3% on ≥ 2 seeds: SAC critic or entropy pathology") compares
  PPO at γ 0.995 with SAC runs that all used γ 0.99, so a PPO success
  would not separate the two. Either run the SAC γ=0.995 arm (§7 run 4
  of that design), or add a γ=0.99 PPO arm.

### 4.5 How runs are launched

| Notebook | Use | Pitfalls at HEAD |
|---|---|---|
| `sb3_training.ipynb` | Generic driver for any recipe | `ENV` defaults to `WallBallTrueBaseline`, a closed chapter (6M SAC on Run-All). No `REQUIRE_DEVICE` guard: off Colab, or when `nvidia-smi` passes but torch sees no GPU, an SAC run trains on CPU silently. `LOCAL_THEN_SYNC` defaults to False |
| `paddle_tennis_campaign.ipynb` | Multi-leg PaddleTennis campaign | Forces CUDA for SAC. Switching to PPO takes three edits, and its warm-start fallback points at an SAC run. Its `learning_starts` equals `eval_freq`, so evaluation 1 scores the untouched warm-start source (the 2026-08-28 "best = step 25,000") |
| `humanoid_tennis_training.ipynb` | Gated Stage 0-2 PPO | No sync option. Training is forbidden by the ladder design |
| `rescore_runs.ipynb` | CPU re-score of a Drive runs folder under `@v3` | Ready. A stale comment calls its pin a "Placeholder" (fixed on this branch, §11) |

- **Pins.** All notebooks pin `REPO_REF = 95fb930`. That predates the
  BLAS fix `c669f4d`. (This review's branch moves the four training and
  re-scoring notebooks to `069d000`; see §10.)
  - PaddleTennis streams are identical across OpenBLAS kernels and
    across the old and new code (verified on 150k steps).
  - The pin is therefore harmless for the active rung. BallBounce,
    WallBall and the humanoid env would differ.
- **GPU runs record no protocol id.** `eval_protocol_record` pins the
  device, so an SAC run on a Colab GPU records `eval_protocol.id:
  null`, with `nearest: paddle-npoint@v3` and `differs: ["device"]`.
  Every GPU SAC result needs a CPU re-score before it is comparable.
  This is by design (`e9575dc`); it needs to be planned for.

## 5. Code health and CI

### 5.1 Locally, every gate passes

| Check | Result |
|---|---|
| `ruff check .` | clean |
| `pytest -n 2` (`OMP_NUM_THREADS=1`) | **1809 passed**, 1 skipped (no GL context), 252 s |
| Gate 1: golden env digests | 22/22 (CI set) and 22/22 (`--full`) |
| Gate 2: recipe snapshot | 96/96 resolutions match |
| Gate 3: trainer digest | not run (needs a worktree checkout; CI never runs it) |
| `python -m mypy` on Python 3.11 (a CI replica) | clean, 73 files |
| `python -m mypy` on Python 3.13 with numpy 2.5 (as the README says to run it) | **aborts.** numpy 2.5's stubs use PEP 695 `type` statements, and the config pins `python_version = 3.11`. Retargeting to 3.12 surfaces 2 real errors at `tools/paddle_tennis_npoint_probe.py:637` |
| `DemoSAC.train` against SB3 2.9.2 | Still faithful. `SAC.train` is AST-identical in 2.9.0 and 2.9.2, so only the "2.9.0" wording is stale |
| `experiments.verify` and `index` on the archived fixtures | `verify` exits 1 on all 5 (INCOMPLETE, no manifest). That is expected: no archived run predates the run-record tooling. `index` backfills 5 rows |

The run-record, manifest and re-score path (L1-L3) has passed its
tests but has never touched a production run.

### 5.2 CI on `main` has been red since 2026-11-16

These facts were checked against the GitHub Actions API:
- **Runs #229 (2026-11-16) and #230 (2026-11-23) failed.** Both are
  scheduled runs at `8df9277`. In each, all three pytest legs
  (py3.11, py3.12, py3.13) fail on the same 7 golden-digest cases.
- **The same SHA passed weekly** from 2026-10-19 through 2026-11-09
  (#225-#228).
- **The failures are all PaddleTennis:**
  - the 7 failing cases are `recipe/random`, `recipe/camper`,
    `one_point/oracle`, `hold/oracle`, `drill_feed/random`,
    `drill_full/random` and `recipe/oracle_seed_once` (its info digest
    only);
  - the other 7 pass on the same runners: `recipe/zero`,
    `recipe/oracle`, `recipe/random_seed_once`, `legal_volley/oracle`
    and the three NaN witnesses;
  - every other env passes.
- **The wrong digests do not depend on CPU, numpy or Python.** They are
  bit-identical across X86_V3, AVX512_ICL and AVX512_SPR runners,
  across numpy 2.4.6 and 2.5.5, and across the three Pythons.

**Cause: the runner image** (confirmed after this review was first
drafted; see the update below). The health reviewer found the
following:
- Between #228 and #229, GitHub's `ubuntu-latest` image moved from
  Ubuntu 24.04 to 26.04 (glibc 2.39 → 2.43).
- The py3.11 leg is the control. Its numeric packages were the same in
  both runs (numpy 2.4.6, MuJoCo 3.15.0), and it failed identically.
  The py3.12 and py3.13 legs also moved numpy from 2.5.4 to 2.5.5,
  which matches 22/22 on glibc 2.39.
- A local replica of the failing py3.11 package set passes on glibc
  2.39.

The most likely mechanism is last-bit changes in the OS math library.
The PaddleTennis code calls `np.sin` and `np.cos` in the serve draw
and `np.hypot` in the reach and hold escrows; the scripted policies
and MuJoCo add more libm calls. A last-bit difference
shows up only when a rollout reaches an input whose rounding changed,
which would explain why some seeded cases move and others do not.
The exact libm call that moves has not been isolated.

**Update (same day).** The fix landed on this review's branch
(`619ea91`, with follow-ups). A `workflow_dispatch` A/B confirmed the
cause:
- **Run #231**, at the same stream code, passed every pytest leg on
  `ubuntu-24.04`, and the render smoke test ran and passed.
- **The new canary job on `ubuntu-latest`** (glibc 2.43) failed exactly
  the same 7 cases with the same digests, from a base-only install with
  no SB3 or torch.

So the runner image is the cause; numpy, SB3 and torch are not. Items
1-3 below are done. The canary now passes while exactly those 7 cases
move, so a change to that set is visible.

**Why it matters beyond CI:**
- **CI no longer guards the env streams.** Every future change to
  `main` is merged against a red baseline.
- **Bit-identity is platform-dependent beyond BLAS.** Week 1 removed
  the BLAS dependence (`c669f4d`), but the streams still depend on the
  OS. Every August run trained on Colab, and Colab's digests have never
  been recorded: M1's "Colab vs local recorded" clause is open. The
  "env streams bit-identical since August" claim that the re-score
  relies on has so far been checked only on the reviewers' machines.
- **The failure message points the wrong way.**
  `digests.version_info()` does not record libc, so the "recorded
  under" and "running under" lines look the same and steer the reader
  toward MuJoCo or numpy.

**Fix, in order:**
1. A/B one CI job on `ubuntu-24.04` against `ubuntu-26.04` to confirm
   the cause.
2. Pin `runs-on: ubuntu-24.04` as a stopgap, declared in DECISIONS.
   Do not re-record the goldens.
3. Add the libc version to `version_info()`.
4. Record the digests on Colab.
5. Then decide whether to make the affected path libm-independent (as
   was done for BLAS) or to keep per-platform locks (the ladder
   design's "one lock per platform" fallback).

### 5.3 Other CI and dependency notes

- **The render smoke test has been skipped since 2026-11-16** (#229,
  #230; fixed on this branch, §11). Its step
  has no `if: ${{ !cancelled() }}`, so it is skipped whenever the
  coverage step fails. The local suite skips it too (no GL).
- **What CI does not run:** golden `--full`, gate 3, the PPO learning
  probe (a "nightly job is planned"), and notebook execution. The
  notebook tests compile the notebooks and run some cells.
- **Node 20.** The actions run on deprecated Node 20.
- **Dependency drift:**
  - The MuJoCo cap `<3.16` now excludes 3.16.0, 3.17.0, 4.0.0 and
    4.0.1. On 3.17 only the humanoid golden moves; 4.0.1 moves all 22.
  - gymnasium 1.4.0, SB3 2.9.2 and torch 2.15 are the latest releases.

## 6. What week 1 delivered, and what it did not

### 6.1 Engineering: delivered, and faster than planned

Week 1 was 24 commits between 2026-10-16 19:45 and 2026-10-18 03:26
UTC. A re-run of 248 targeted gate tests at HEAD passed. So did
`tools/ppo_learning_probe.py`: 750/750 on 20 held-out seeds after
664k steps in 344 s.

| Plan | Done | Partial | Not started |
|---|---|---|---|
| Env ladder M0-M13 | M1 (except the Colab clause), M2 (via PPO Steps 1-14), M3 | **M0** (SHA pins, but no git tag exists locally or on origin); **M9** (pre-empted by PPO Steps 6, 9 and 11; `critic_warmup_rollouts`, the per-dimension std alarm and mid-leg resume are missing) | M4 (R0 tasks; about a third pre-empted by the 1-D reach test), M5-M8, M10-M12. M13 was never triggered |
| Campaign structure L0-L9 | L0, L1 | **L2** (audit mode deferred to L6); **L3** (`@v3` declared; the history not re-scored) | L4-L9 and the first spec-driven baseline |
| PPO plan Steps 0-14 | All 15 | Step 3's probes (only BallBalance probed); Step 12's nightly job | §4 extras (`reset_log_std_on_advance`, `critic_warmup_rollouts`) |
| Spikes S1-S4 | — | **S1** (built and smoke-run on CPU, never run on an L4) | S2, S3, S4 |
| "Training runs before changes" (ladder §7, items 1-5) | — | — | **All five** |

Notes on what was built:
- **Volume.** The +54,425 lines split into about 25.4k of hand-written
  code and tests, 21.6k of generated goldens and fixtures, and 4.8k of
  prose.
- **Fixup rounds.** Seven verifier-fixup commits account for about
  10.1k lines.
- **Growth.** `src` Python grew by 39% (28.7k → 40.0k lines), and
  `train.py` by 72% (2,708 → 4,662).
- **The lever debt is untouched.** `PaddleTennisEnv` still has 1,781
  lines and 19 parameters.
- **Some work came ahead of need:**
  - in-loop protocols for frozen WallBall and humanoid recipes;
  - the gated WallBall protocols, added and then withdrawn;
  - S1 built in the repo, with 496 lines of tests of its verdict
    branches (the design said off-repo);
  - four verifier rounds hardening interrupt windows that no run has
    exercised.
- **One plan conflict.** M8 (archive the humanoid code) runs against
  week 1's new humanoid PPO profiles, protocols and golden case.

### 6.2 Schedule

| Milestone (plan) | Due | Reality on 2026-11-27 |
|---|---|---|
| Day-0 PPO reference; the August re-score; S1 | 2026-10-16 to 10-23 | Not run (42 days late) |
| Pilot unlocked (after M6) | about 2026-10-26 | Locked: M5 and M6 not started |
| First spec-driven baseline | about 2026-10-30 | Not started |
| Certified paddle doubles | 70 working days in the ladder design, but **about 88** once the campaign design's +18 L-step days are added | 11.25 of about 88 days done. At plan rate from now, about 2027-03-15 |

### 6.3 The 98 findings of 2026-10-16

| Status | Count |
|---|---|
| Fixed | 12 |
| Partly fixed | 4 |
| Open | 82 |
| Obsolete | 0 |

**Medium findings.** Of the 10 unique medium ones, 3 are fixed:
- `train-2`: the interrupt salvage;
- `rules-events-1`: the wedged event sampler;
- `humanoid-2`: MuJoCo 3.16. This was fixed by a cap. The test still
  asserts exactly 129 steps, so lifting the cap brings the failure
  back.

**Batches.**
- Batch A's run-directory trust items are fixed: stop labels,
  `reuse_log_dir`, empty videos, the Drive flush.
- So is all of batch B except the exact-step test.
- Batch A's reporting bugs, batches C, D and E, and nearly all of F are
  open. The per-ID table is in Appendix A.

**Open items that still matter for the next runs:**
- **`callbacks-1`/`training-data-1`, the paired `confirm_best` bias.**
  The in-loop rule still compares a candidate's confirmation block
  with the first best's *standard*-block score. A scratch reproduction
  lost 10 of 10 confirmations. The PPO reference (a near-untrained
  first best) and the LD1′ warm start can both pin `best_model.zip` to
  an early evaluation. `experiments/select.py` says the in-loop fix
  waits on a maintainer decision and offers an offline within-block
  `select()` instead. Read every verdict through `select()` on the
  snapshots, not the in-loop pick.
- **`training-data-2`, "deploy best_model.zip".** `stage_summary`
  still compares the closing evaluation with the 5-episode
  reward-stream argmax. It can recommend deploying a step that
  `best_model.zip` does not hold. This was reproduced.
- **`consolidation-1`, the headline plot.** A PaddleTennis run's
  `eval_headline.png` still shows only reward and length, while the
  k≥2 headline lands on the diagnostics page. That is the reward-led
  reading the selection design exists to prevent.
- **`notebook-utils-1`, the unsafe count.** The campaign's
  `unsafe_terminations` still counts only `nonfinite_state`. That is
  the field the "zero unsafe terminations" pre-registered criterion is
  read against.
- **`recipes-config-1`.** Gate TOML values are validated only after
  `config.json` is written.
- **`artifacts-demo-1` and `train-4`, both LD1′ blockers.** `DemoSAC`
  does not check the demo library's recorded `env_kwargs`. `train()`'s
  warm start does not check that the best pair is bound by
  `best_model_meta.json`.
- **`humanoid-1`.** Stage 1/2 promotion can never pass. In addition,
  `_wilson_interval(0, 50)` breaks `SideSuccessSummary`'s invariant, so
  a 0% side crashes the promotion evaluation instead of reporting 0%
  (§9).

## 7. New findings in the week-1 code (verified)

**Method.** Five finders covered the week-1 diff: run records and
sync, offline evaluation, PPO and training, digests and linalg, and
the Colab notebooks. Between them they reported 18 findings. Each
scope's verifier re-read the code at HEAD, wrote its own independent
reproduction, and defaulted to "refuted" when it could not reproduce
a finding.
- **Result.** None was refuted. Four were confirmed only in part.
  Severity was re-rated for real use, and most findings dropped to low.
- **Two additions** come from other parts of the review: the
  video-retry abort, found by the readiness run (§8) and reproduced
  here, and the red CI (§5.2).

### 7.1 Findings that matter for the next runs

| # | Finding | Severity | Where |
|---|---|---|---|
| N1 | **Fixed on this branch (§11).** **A second video attempt on a host without OpenGL aborts the training process** (SIGABRT, uncatchable). The run is left `running`, with no manifest and no final model. The bug predates week 1, but PPO's move to CPU makes it bite now (§8) | **high** for CPU launches | `callbacks/video_record.py:173-190` |
| N2 | **Fixed on this branch (§11).** **CPU re-scores depend on the host.** torch's CPU `Linear` calls MKL SGEMM, and MKL's AVX2 and AVX-512 kernels round differently. PaddleTennis is chaotic, so one ulp in an action changes whole episodes. The same snapshot, seeds and protocol fingerprint gave 53 of 132 differing metrics on the two kernels. The protocol pins only `device="cpu"`, and score rows record no CPU model, MKL mode or package SHA | **medium** | `experiments/protocols.py:264`; `harness.py:649` |
| N3 | **Fixed on this branch (§11).** **A campaign leg whose final Drive mirror fails is never booked.** It goes unbooked even after the operator runs the printed `sync_run_dir(...)` fix. `run_leg` does not catch `train()`'s final-sync error, so on the next Run-All `campaign_leg_resume_point` returns "train" and the complete `attempt_01` is replaced by a fresh `attempt_02`. A Ctrl-C during the final mirror takes the same path | **medium** | `paddle_tennis_campaign.ipynb` (`run_leg`); `train.py:743-773`; `notebook_utils.py:2334` |
| N4 | **Fixed on this branch.** **The campaign and `sb3_training` notebooks refused a Colab CPU runtime.** `setup_colab()` defaults to `require_gpu=True` and runs before `ALGO` is read, although the PPO reference is a CPU run by design | low (but it blocks the planned launch path) | `colab_setup.py:51`; campaign cell `pc-gpu` |

**Fixes:**
- **N1:** after the first failure, set `save_freq = 0`.
- **N2:** set `MKL_CBWR` before torch is imported; under either
  `AVX2` or `COMPATIBLE`, both kernels give identical bits. Then
  record the numeric mode in the protocol and in every score row.
  - Pinning changes CPU training streams, so declare it as an era
    boundary for CPU-trained runs.
  - It does not corrupt within-run selection: each run is re-scored
    whole in one call.
  - It does break cross-session comparisons, the merged verdicts, and
    the in-loop identity check for CPU-trained runs re-scored on a
    host with the other ISA.
- **N3:** catch the error and book the leg as `trained_unsynced`, then
  re-sync on resume.
- **N4:** call `setup_colab(require_gpu=False)` and leave the GPU
  requirement to `REQUIRE_DEVICE`.

### 7.2 Lower-severity confirmed findings

| ID | Finding (all low after verification) |
|---|---|
| records-1 | `run_id` is the run directory's basename, so every campaign leg is `attempt_NN`. A manual `sync_run_dir` into the wrong leg's Drive directory therefore passes the "another run's mirror" guard: it overwrites that leg's files, deletes its checkpoints, and re-seals it as complete. Timestamp-named runs are protected. Fix: a unique run id at launch |
| records-2 (partly) | The mirror copies `run_record.json` (terminal) before `snapshots/` and `stage_summary.txt`. A final pass cut short leaves a directory that an unverified index (`verify=False`) reports as complete. `verify` itself catches it |
| records-3 | The index backfill turns every performance-gated run's `model/stage_bests/stage_NN/` (which holds a copy of `config.json`) into phantom "aborted" stub rows. This affects the WallBall gated era and future humanoid curricula |
| records-4 (partly) | `sync_run_dir` re-seals whatever the source holds, without checking the source's manifest |
| records-5 (partly) | Dead launches are recognized only in timestamp-named directories, so an early-dying campaign leg (`attempt_NN`, empty under `LOCAL_THEN_SYNC`) leaves no index row |
| eval-2 | A `rescore --steps` or `--which` quick look rewrites `<run>/scores/<protocol>.jsonl` whole, deleting an earlier full re-score. The notebook driver writes elsewhere, so only CLI use is exposed |
| eval-3 | `probe --block confirm` reports the standard block's first seed |
| eval-4 | `from_run_config` fingerprints JSON-round-tripped kwargs, so an IntEnum kwarg becomes an int and `HumanoidTennisCoopSmoke`'s in-loop protocol never matches its recorded one |
| ppo-1 | Ctrl-C on a `vec_env="subproc"` run raises `BrokenPipeError` during teardown, and the run is recorded as crashed instead of interrupted. The default `dummy` path is unaffected |
| digests-1 (partly) | The static BLAS guard misses `from numpy.linalg import norm`, `np.vecdot`, `np.matvec` and `@=` |
| digests-2 | glibc libm picks FMA or SSE2 variants of sin, cos, atan2 and pow per CPU, and no gate exercises that. This supports the libm explanation of §5.2 |
| digests-3 (partly) | `c669f4d` breaks the documented bit-for-bit `--protocol inloop` re-score for legacy WallBall and BallBounce runs evaluated on AVX-512 hosts. PaddleTennis is unaffected |
| colab-2 | *Fixed on this branch (§11).* In `sb3_training` with `LOCAL_THEN_SYNC`, a failing replay-video cell halts Run-All before the 9d mirror, so the audit and plots stay on the VM's disk |
| colab-4 | The humanoid notebook writes its promotion evidence at the run-directory root after sealing, so every gated stage fails `verify` |
| colab-5 (partly) | *Fixed on this branch (§11): the training and re-scoring notebooks pin `069d000`.* The `REPO_REF` pin predates `c669f4d`. Non-PaddleTennis streams on pinned notebooks still depend on the host's OpenBLAS kernel. Several "bump REPO_REF for PPO / LOCAL_THEN_SYNC" notes are stale, since the pin already ships both |

**What held up under review:**
- `_linalg` is equivalent to the pre-AVX2 numpy calls: 0 differences
  in 290k cases.
- The harness reproduces in-loop rows bit for bit on the same host.
- The PPO rollout-geometry checks, the update-counted stops, the
  normalizer gamma, `log_std` reset and `load_run_model` all hold.
- `DemoSAC` still matches SB3 2.9.2.

## 8. Run readiness: the next runs, executed here at small scale

The planned next runs were executed here at reduced scale, at
`8df9277`, on 4 shared CPUs with one torch thread each.

| Step | What ran | Result |
|---|---|---|
| Day-0 PPO reference (ladder §7 item 1) | The design's call builds unchanged for seeds 0-2 at 4, 8 and 16 workers. A reduced run (4 envs × 256-step rollouts, 8,192 steps, 4 evaluations) went through `train()` | **Pass.** 64.5 s, 40 updates. Wrote `run_record.json` (completed), `sync_manifest.json`, 4 snapshot pairs and `best_model_meta.json`. `verify` and `verify --strict` pass |
| In-loop identity | `rescore --protocol inloop` on that run | **Pass.** 132/132 keys equal `eval_info.csv` at all 4 snapshots |
| `@v3` re-score and `select()` | `rescore --protocol paddle-npoint@v3`, then `select()` | **Pass.** About 100 s per artifact. Both pick the same step on `@v3` rows. On the in-loop rows, `select()` and the in-loop rule disagree (2048 against 4096): the confirmation bias of §6.3 |
| Re-score of a real August run | Registered run `20260816_235141`: its best (2.4M) and the 2M checkpoint, fetched read-only from Drive (sha256 matched `best_model_meta`), scored under `@v3` and `@v2` | **Pass.** Every non-reward key is identical across the two protocols (117-119 per block), so L3's identity gate holds on real data. Under `@v3`: k≥2 = 0.000 for both, k=1 1.70 and 1.43, task reward 0.667 (1.445 under `@v2` with shaping) |
| SAC quick test | `build_train_config('PaddleTennis', algo='SAC', quick_test=True)` | **Crash** (exit 134) at the second video attempt, step 20k. With `record_video=False` it completes in 405 s and verifies (38 files) |
| PPO learning probe | `tools/ppo_learning_probe.py` | **Pass.** 750/750 on 20 held-out seeds after 688k steps. The CI 1-D reach test passes in 7.5 s |
| SAC γ=0.995 arm | Through a TOML `[train.algo.sac.model_kwargs] gamma = 0.995` | **Builds.** The value deep-merges onto the exploration bundle |

**The crash is the one real defect.**
- **What happens.** On a host where MuJoCo cannot create an OpenGL
  context, `VideoRecordCallback` catches the first failure and retries
  at the next cadence. MuJoCo 3.15 cannot retry: the second
  `mujoco.Renderer()` trips a C++ static-initialization guard and
  aborts the interpreter, which no Python `except` can catch.
- **Reproduced here** with two `Renderer()` calls (exit 134).
- **What is left behind.** A run record that says `running`, no
  manifest, no closing evaluation and no final model.
- **Where it applies.** `record_video` defaults to True. Any CPU host
  without working GL dies at its second video cadence: this container,
  CI runners and possibly Colab CPU runtimes. `setup_colab` forces
  `MUJOCO_GL=egl`, and whether that works without an NVIDIA driver was
  not tested.
- **For the PPO reference** at its 250k video cadence, that means death
  at 500k steps.
- **The fix.** After the first failure, disable recording for the rest
  of the run (`save_freq = 0`). Until then, pass `record_video=False`
  on CPU hosts.

**Throughput, measured.**

| Quantity | Measured | Design's figure |
|---|---|---|
| PPO collection, single process | 1,244-1,674 env steps/s | 632-832 |
| PPO collection, `subproc` (8 workers, 4 vCPUs) | 1,889-2,673 env steps/s | — |
| PPO update share | 7-14% | ≤ 4% |
| Single-env evaluation | 738-1,082 steps/s; the 30-episode checkpoint diagnosis takes about 51 s | — |
| SAC recipe, one torch thread | 80-88 env steps/s, 95% of it in gradient steps | — |

What the numbers imply:
- `subproc` gains only 1.4-1.6×, because policy inference and the
  in-loop evaluation stay in the main process.
- At the reference cadence, evaluation takes 20-48% of PPO wall clock.
- **PPO reference, one core:** about 0.8-1.0 h per 3M steps per seed
  and 2.6-3.5 h per 10M.
- **Three seeds × 10M** as parallel single-process runs on 4 vCPUs
  take about 2.6-3.5 h, against 6-8.6 h one after another.
- **SAC on CPU:** about 7.5-9.6 h for the recipe's 2M budget and
  11-14 h per 3M seed of the γ arm. That is comparable to the L4
  estimate of about 20 h per seed.

**The re-score is bigger than planned.** Drive holds about 237 legacy
PaddleTennis SAC checkpoint zips (2026-08-03 to 08-29). At 1.7-1.9
minutes each, scoring them all takes about 7-8 CPU-hours; the design
assumed about 100 checkpoints and "a few CPU-hours". The notebook has
no every-500k option, so use `QUICK`, `RUN_IDS`, or the CLI's
`--steps` with parallel processes.

**Smaller defects:**
- `stage_summary.txt` reads the git SHA when it is written, not at
  launch. A checkout that moves mid-run gets a summary that disagrees
  with its run record. This happened here: this review's own
  work-in-progress commit (`86ab95d`) landed during the SAC quick
  test, so its summary says `86ab95d` and its record `8df9277`.
- `rescore_runs.ipynb`'s pin comment is stale (fixed on this branch,
  §11): 95fb930 does ship
  `@v3`.

## 9. The humanoid and GPU track

**What is proven: scripted physics only.**
- On MuJoCo 3.15.0 both oracles succeed on both mirrored sides: Stage 0
  in 67 steps and Stage 1 in 129.
- The humanoid curriculum tests pass 34/34; the other humanoid and
  event test files pass 196, with 1 GL skip.
- The only learned evidence ever recorded is a 25k-step Stage 0 PPO
  run from July: 1 success in 177 episodes, every evaluation 0.000.
- The post-0.16.0 PPO profile has never trained.
- The humanoid code is about 12% of `src` (4.7k lines) and 10% of the
  tests. The G1 meshes are 19.7 MB.

**What is broken or blocked:**
- **Stage 1/2 promotion can never pass (`humanoid-1`, open).**
  `_canonical_evaluation_spec_id` pins `valid_hit_shaping = 0.0`, but
  the recipes use 0.25. A Stage 1 notebook run would train for about
  7 CPU-hours and then raise before writing a promotion report.
- **A 0% serve side crashes promotion.** `_wilson_interval` clamps only
  to [0, 1]. For 0/50 its lower bound is 6.9e-18, which is above the
  rate, so `SideSuccessSummary` raises. This was reproduced with a
  zero-action policy on the canonical Stage 0 suite. DECISIONS calls
  it "partially addressed".
- **The MuJoCo cap is still load-bearing.** On 3.17 the Stage 1 oracle
  takes 138 steps and misses; on 4.0.1 all 22 goldens move.

**The GPU plan has a version problem.** The design says to train on
GPU (mjlab first, Playground as the fallback) and certify on the CPU
court, using the same assets and torch weights. The stacks have moved
since:

| Stack | MuJoCo requirement | Fits the repo's `>=3.12,<3.16`? |
|---|---|---|
| mjlab 1.6.x | `mujoco~=3.11`, `mujoco-warp~=3.11`, Python <3.14 | No (below the floor) |
| MuJoCo Playground 1.0/1.1 (2026-11-06 and 11-23) | MuJoCo and MuJoCo-Warp ≥ 4.0; no longer depends on JAX/Brax | No (above the cap, and 4.0.1 moves every golden) |
| Playground 0.2.0 (S1's pin, the JAX/Brax line) | — | Installable, but no longer the maintained line |

Today no single environment can run a current GPU stack and the CPU
court together. A version strategy is needed before S2, S3 or H0 are
built: either separate environments with a CPU certification step, or
a declared era on MuJoCo 4.

**Spike S1 is well built but has one operational risk.**
- Its pins all exist on PyPI. Its bars, verdict rules, scene
  composition and results folder are tested (25 pass).
- The scene composes on CPU: 4×G1 plus a ball, nq 151. On one CPU
  thread it runs 347 control steps/s, 14× below the 5,000 bar, so the
  bar is GPU-only as expected.
- **The risk:** the `warp-lang` 1.18.0 PyPI wheel is built against
  CUDA 13.4 and needs an R580+ driver. On an older Colab driver every
  Warp row and all of mjlab drop out. The verdict would then rest on
  classic MJX alone, and a FAIL would reflect the driver, not the GPU.
- Its transitive dependencies (flax, optax, orbax, chex) are unpinned
  against Colab's preinstalled image.
- `DECISIONS.md` records the wrong S1 `REPO_REF` (b9585e2). The
  notebook pins 95fb930.

## 10. Next steps

The ordering principle is the one the project already wrote down:
run what can change a decision, with no new features, and keep a run
in flight. Items 1-6 need no design work. Most of them run on CPU.

### 10.1 This week: restore the baseline, then run

| # | Step | Cost | Why now |
|---|---|---|---|
| 1 | **Done on this branch** (`619ea91` and follow-up). **Restore green CI.** A/B one job on `ubuntu-24.04` against `ubuntu-26.04`, then pin `runs-on: ubuntu-24.04` as a stopgap declared in DECISIONS (do not re-record). Add the libc version to `digests.version_info()`. Add `if: ${{ !cancelled() }}` to the render step | hours | Every later merge needs a working stream gate (§5.2) |
| 2 | **Done on this branch** (`619ea91` and follow-up). **Fix the video-retry abort.** Disable recording after the first failure, with a regression test. Until then pass `record_video=False` on CPU hosts | 1 h | Otherwise a CPU run dies at its second video cadence (§8) |
| 3 | **Launch the day-0 PPO reference** exactly as pre-registered: the §7 call plus `early_stop_degenerate_evals=0` and `record_video=False`, 3 seeds as parallel single-process CPU runs with a Drive `sync_dir`. On a Colab CPU runtime, use a notebook pinned to this branch's fix (N4). Record `os.cpu_count()` and the CPU's ISA. Read at 3M against the frozen table | about 1 h to the 3M readout; 2.6-3.5 h for 3 × 10M on 4 vCPUs | The cheapest result that can change a decision: task-side or SAC-side, the default algorithm, and LD1′'s form |
| 3a | **Book the choice first.** Add a DECISIONS line saying the reference is the design-literal call, not the recipe's unprobed profile, and commit the config as a TOML or spec in git rather than on Drive. Report best models under both `select()` and the in-loop rule (§6.3) | minutes | Two configs disagree on 8-9 settings (§4.4) |
| 3b | **Add a γ control** so the reading table can separate algorithm from discount: 1-2 PPO seeds at γ=0.99 (about 1 h per 3M on CPU), or the SAC γ=0.995 arm (2 × 3M, about 11-14 h per seed on CPU or about 20 h on an L4) | as stated | Row 2 of the table otherwise confounds the two (§4.4) |
| 4 | **Re-score the August runs under `@v3`** with `rescore_runs.ipynb` on a Colab CPU runtime (`RUNS_SUBDIR='PaddleTennis/sac'`, `SERIES='checkpoints'`). Set `MKL_CBWR=COMPATIBLE` before torch is imported, or score every run on one host type and record its ISA, so that scores from different sessions are comparable (N2). Run `QUICK` first; then best models plus 500k multiples as parallel CLI processes. Book the `select()` verdicts | 1.5-2 CPU-h (QUICK); 7-8 CPU-h (all about 237 checkpoints) | Gives the current-era incumbent on `episode_rally_returns_a`, which every later comparison needs, and closes L3 |
| 5 | **Record the golden digests on Colab** (`tools/golden_digest.py --full` on a CPU runtime) | 5 min | Closes M1. Tests whether Colab reproduces the local streams, an assumption the re-score's comparability rests on (§5.2) |
| 6 | **Run spike S1** after a half-day preflight: print the `nvidia-smi` driver first, swap Warp to its `+cu12` 1.18.0 wheel if the driver is below R580, and pin flax, optax, orbax and chex. Book PASS, FAIL or no result | 0.5 day + 1-2 L4-h | Decides the whole humanoid half and the GPU platform (§9) |

### 10.2 Next two weeks: test the top hypothesis directly

| # | Step | Notes |
|---|---|---|
| 7 | **Context-blind pilot, built minimally.** Make it a pure observation projection, i.e. physics only, dropping the rally-state, latch and clock dims. Give it its own observation fingerprint so old checkpoints refuse it. Train 3 PPO seeds on CPU against the step-3 reference; SAC seeds against the γ arm when an L4 is free | It is the only direct test of hypothesis 1 (§3.2), and it does not need M5-M7. Score it on the 10-05 bar (k=2 ≥ 5% on fresh seeds) or the campaign design's pilot rule |
| 8 | **Decide LD1′.** Freeze it as a short pilot spec with a cap and a kill date, in whichever algorithm step 3 favours (`DemoSAC`, or PPO with BC or kickstarting via M13). Or retire it by tag-and-delete | Fix `artifacts-demo-1` and `train-4` first if it runs |
| 9 | **Fix what would mislead the reading of steps 3-8:** the paired `confirm_best` first-best bias (decide the rule; roll a confirmation for the first best); `stage_summary`'s "deploy best_model.zip"; the `eval_headline.png` routing; the campaign's unsafe count; and N2-N4 plus records-1 from §7 (pin `MKL_CBWR` and record the numeric mode in protocols and score rows; book an unsynced campaign leg; give runs a unique id) | Mostly bit-identical. The `confirm_best` rule and the `MKL_CBWR` pin are declared era decisions (for selection, and for CPU-trained streams) |
| 10 | **Widen the incoming-ball distribution** (the 10-05 "ball machine") if k=2 moves only partially | Hypothesis 2 |

### 10.3 Later, at the rung that needs it

- **M5-M7 with L4-L5** (`ObsLayout`, task registry, experiment
  specs). Build them when step 7's result shows what P1 needs. Count
  the total as about 88 working days, not 70.
- **L6-L7** (extract the escrows, then tag and delete the dead levers),
  before any new lever is added. `PaddleTennisEnv` still carries 19
  kwargs.
- **The MuJoCo version strategy and M8** (§9):
  - tag first;
  - fix `humanoid-1` and the Wilson clamp in the tagged commit, so the
    archived notebook can promote;
  - keep the G1 adapter for the GPU track;
  - lift the cap to `<4`, since 3.17 moves only the humanoid golden.
- **S2-S4 and H0** only after S1 passes. Budget about 30-90 L4-hours
  for H0's 3 seeds.

### 10.4 Process guardrails

- **Keep one run in flight at all times.** Merge no M4+/L4+
  infrastructure until steps 3-6 have launched.
- **Make runless stretches visible.** Generate a one-line status from
  the run index ("days since last run", "runs in flight") and put it at
  the top of `docs/README.md`.
- **Spend verifier rounds on code that has run.** Week 1's four
  interrupt-hardening rounds covered windows no run had exercised.
- **Refresh the stale docs:**
  - the design status headers, which still say "week 1 in progress";
  - the throughput figures (632-832 steps/s and ≤ 4% updates; §8);
  - the README's "nightly job is planned";
  - the S1 `REPO_REF` in DECISIONS;
  - the incomplete `docs/README.md` index.

## 11. Post-snapshot updates (same day, on this branch)

The fixes below landed on `claude/inspiring-lamport-bxvxn5` after the
snapshot. The body of this review still describes `main`@`8df9277`.

| Commits | What they close |
|---|---|
| `619ea91`, `5028bc4` | **§5.2 items 1-3.** CI's gating jobs run on `ubuntu-24.04`. A non-gating `golden-canary` job checks `ubuntu-latest` against its known 7-case drift. `digests.version_info()` records `libc`. The render smoke step runs when the coverage step fails. Run #231 (green on 24.04, the same 7 cases failing on `ubuntu-latest`, glibc 2.43) confirmed the runner image as the cause, and #232 was green throughout. **N1, the render guard:** where there is no display the recorder skips without trying, and a failed recording is never retried. A context failure poisons rendering for the process and sets `MUJOCO_GL=disable`, and `record_best_model_video` then refuses with a Python error. Both end-to-end scenarios die of SIGABRT on the old code and pass on the new |
| `069d000` and the commit after `957cc46` | **N4.** A PPO notebook runs on a Colab CPU runtime. `setup_colab(require_gpu=False)` leaves `MUJOCO_GL` alone there and reports what `nvidia-smi` found. The campaign refuses the SAC plan's `REQUIRE_DEVICE = "cuda"` on a runtime without a GPU, and `sb3_training` demands a GPU for the SAC family only. **colab-2:** the `sb3_training` replay cell no longer stops Run-All. **notebook-utils-5:** `_in_colab()` no longer raises without a `google` package |
| `957cc46` | **colab-5.** The four training and re-scoring notebooks pin `REPO_REF` to `069d000`, which ships these fixes and the BLAS-free streams. The stale "bump REPO_REF" notes and the re-score "Placeholder" comment are corrected. The S1 spike keeps its own pin |

**Still open from this review:**
- N2: CPU re-scores depend on MKL's host kernels; pin `MKL_CBWR` and
  record the numeric mode.
- N3: a campaign leg is unbooked after a failed final mirror.
- `records-1` to `records-5`, `eval-2` to `eval-4`, `ppo-1`,
  `digests-1` to `digests-3` and `colab-4`.
- The other 81 findings of 2026-10-16.
- Every research step of §10 from item 3 on: the PPO reference, the
  August re-score, the Colab digests, S1 and the context-blind pilot.

## Appendix A. Status of the 2026-10-16 findings at HEAD

Status as reviewed at HEAD. Evidence is a file:line or commit at `8df9277`; batches are from `repo_review_20261016.md` §8.

| ID | Kind | Severity | Batch | Status | Evidence |
|---|---|---|---|---|---|
| rules-events-1 | bug | medium | A | fixed | envs/_tennis_events.py:311-317 reset() discards an open step; tests/test_tennis_events.py:610,674 pass |
| train-2 | bug | medium | A | fixed | training/train.py:3088-3183 saves model/normalizer first, each step log-and-continue; tests/test_train.py:2894 passes |
| humanoid-2 | bug | medium | B | fixed | pyproject.toml:44 mujoco>=3.12,<3.16 (4e2e019), pinned by tests/test_packaging.py:42; a cap, not the review's archive/behavioral fix |
| critic-2 | bug | low | - | fixed | per-family algo_cfg (55ad948); probe: build_train_config('PaddleTennis'/'WallBallBootstrap', algo='PPO') resolves PPO-only model_kwargs |
| critic-4 | bug | low | A | fixed | notebook_utils.py:257-282 (flush_and_unmount at :275) before teardown, stays up on failure |
| train-3 | bug | low | A | fixed | train.py:4606-4615 full_budget label; early_stop_reachability warns pre-output (2042) |
| train-5 | bug | low | A | fixed | _check_log_dir_reuse (train.py:2750-2850) rotates monitor/TB/run record/manifest; tests/test_train.py:2783,2815 |
| training-data-3 | bug | low | A | fixed | train.py:4606-4615 full_budget; reward-path reason via _StopOnNoRewardImprovement (345-372) |
| training-data-5 | bug | low | A | fixed | notebook_utils.py:563-573 empty dir = MISSING; artifacts.py:2089-2093 '<-- EMPTY' |
| critic-1 | bug | low | B | fixed | pyproject.toml:50 gymnasium>=1.1.1 (4e2e019) |
| recipes-config-2 | cleanup | low | B | fixed | pyproject.toml:6,14 setuptools>=77 + license = 'MIT', no License classifier |
| recipes-config-3 | cleanup | low | B | fixed | .github/workflows/ci.yml:56,108 CPU torch index before the editable install |
| tests-1 | bug | medium | B | partly fixed | tests/test_humanoid_tennis_curriculum.py:522 still asserts steps == 129; passes only because of the <3.16 cap |
| critic-3 | bug | low | - | partly fixed | PPO on gated recipes now refused pre-output (train.py:2542-2555); repro: WallBallDepthCurriculum SAC with ent_coef=0.05 still raises TypeError inside learn() after writing config.json, run_record, crashed summary |
| train-1 | bug | low | C | partly fixed | stop_reason recorded (train.py:345-372, _stop_reason_of 375-401); inert-patience warning (2042-2095); reward path keeps SB3's N+1 rule and no schedule warm-up (train.py:2080-2085 vs headline 2075-2079); no shared helper |
| train-8 | cleanup | low | F | partly fixed | train.py:824 and :1198 still say 'same algorithm (PPO or SAC)', though :3301 accepts any SAC-family pair; PPO temperature text updated |
| humanoid-1 | bug | medium | - | open | training/tennis_curriculum.py:942 still pins valid_hit_shaping 0.0; recipes.py:1626,1648 use 0.25; no humanoid code change in b9585e2..HEAD |
| consolidation-1 | bug | medium | A | open | notebook_utils.py:608-635 still pages by WallBall stem prefixes; headline_key only reorders within a page (699-721); repro: episode_rally_returns_a and stage_success -> eval_diagnostics |
| notebook-utils-1 | bug | medium | A | open | notebook_utils.py:2441-2442 still counts only termination == 'nonfinite_state'; env set is paddle_tennis.py:135-138 |
| recipes-config-1 | bug | medium | A | open | run_config.py:214-270 type-only; train.py:2397-2420 keys only; gate ctor train.py:4109 runs after write_run_config (3603); comment at 3590-3599 concedes it |
| training-data-2 | bug | medium | A | open | training/artifacts.py:1699-1787: under headline selection held_step stays None, so the reward argmax is compared; repro: 'Best eval at 2,000 ... deploy best_model.zip' while best_model holds 1,000 |
| callbacks-1 | bug | medium | C | open | callbacks/info_dict_eval.py:1438-1445 still falls back to _best_score until a confirmation is accepted; scratch repro: dominating candidate lost 10/10 confirmations; offline experiments/select.py:19-31 has the within-block rule |
| training-data-1 | bug | medium | C | open | same code as callbacks-1 (info_dict_eval.py:1438-1445); best_model_meta n_updates only for on-policy (train.py:217, record_n_updates) |
| artifacts-demo-1 | bug | medium | D | open | training/demo_sac.py:250-277 _load_demo_library checks schema+obs fingerprint only; file untouched since b9585e2; harvest records env_kwargs (tools/paddle_tennis_k2_demo_harvest.py:319) |
| artifacts-demo-2 | bug | low | - | open | training/paddle_diagnosis.py:118-152 FAILED_TO_CROSS still falls through to the unprefixed termination name |
| artifacts-demo-4 | bug | low | - | open | training/artifacts.py:2082-2101 Artifacts index iterates EXPECTED_ARTIFACTS, which excludes _CONDITIONAL_ARTIFACTS (1375-1395) |
| consolidation-3 | bug | low | - | open | tools/paddle_tennis_probes.py and tools/_seed_ledger.py untouched since b9585e2 |
| env-paddle-1 | bug | low | - | open | envs/_paddle_court.py:104-109 checks only speed/elevation; probe: lateral_degrees/lateral_noise_degrees/elevation_noise_degrees = NaN accepted |
| env-paddle-2 | bug | low | - | open | envs/paddle_tennis.py:1683-1710 unchanged (only _linalg edits in b9585e2..HEAD) |
| env-wallball-simple-1 | bug | low | - | open | notebook_utils.py:1019-1022 _LEGACY_WALL_BALL_CONSTRUCTOR_DEFAULTS still lacks return_shaping_scale |
| env-wallball-simple-4 | bug | low | - | open | envs/wall_ball.py changed only for _linalg.norm (c669f4d); no pre-step guard added |
| notebook-utils-3 | bug | low | - | open | paddle_tennis_campaign.ipynb section 7 cell still reads record['report'] for every manifest stage |
| notebook-utils-5 | bug | low | - | open | colab_setup.py:27-28 find_spec('google.colab') unchanged at 8df9277; fixed on this branch (`069d000`, §11) |
| tests-2 | bug | low | - | open | tests/test_paddle_tennis.py:155 mirror test unchanged (diff only touches recipe/eval-env tests) |
| tests-3 | bug | low | - | open | tests/test_train.py:323-353 still asserts only that progress.csv exists |
| tools-1 | bug | low | - | open | tools/paddle_tennis_k2_harvest.py diff adds --algo only |
| tools-2 | bug | low | - | open | tools/paddle_tennis_probes.py and depth_stage_sweep.py untouched |
| tools-3 | bug | low | - | open | tools/paddle_tennis_command_spectrum_analysis.py untouched |
| tools-5 | bug | low | - | open | tools/_seed_ledger.py untouched |
| tools-6 | bug | low | - | open | tools/depth_stage_sweep.py untouched |
| train-6 | bug | low | - | open | probe: confirm_best_eval, early_stop_degenerate_evals, degenerate_guard_keys, scalar best_metric_min_delta all accepted with headline_key=None by every pre-output validator |
| training-data-4 | bug | low | - | open | training/artifacts.py:1985-2000 still prints every *_final key (rew_* included) as ep-mean/last/max; no _ep_sum_mean |
| artifacts-demo-3 | bug | low | A | open | training/artifacts.py:1783-1786 still advises 'deploy best_model.zip' under headline selection (same repro as training-data-2) |
| callbacks-2 | bug | low | A | open | info_dict_eval.py:762-764 tuple(dict.fromkeys(...)); probe: best_metric_keys='abc' -> ('a','b','c'), same for info_keys, degenerate_guard_keys |
| callbacks-3 | bug | low | A | open | callbacks/video_record.py:118 checks only csv_header-without-info_row_fn; file untouched since b9585e2 |
| consolidation-2 | bug | low | A | open | notebook_utils.py:2441-2442 (same as notebook-utils-1) |
| consolidation-5 | bug | low | A | open | callbacks/video_record.py:118 (same as callbacks-3) |
| notebook-utils-2 | bug | low | A | open | notebook_utils.py:608-635 (same as consolidation-1) |
| training-data-6 | bug | low | C | open | info_dict_eval.py:1002-1003 writes the CSV row before confirmation rolls; confirmation kept only in memory and in an accepted best's meta (1629-1639); offline harness.py:808 can recompute from snapshots |
| train-4 | bug | low | D | open | _prepare_warm_start (train.py:3213-3440) checks only explicit sha256 pins (3424), never best_model_meta.json |
| env-wallball-simple-2 | bug | low | E | open | assets/ball_balance.xml still has no <compiler> element |
| env-wallball-simple-3 | bug | low | E | open | __init__.py:124-144 registered max_episode_steps unchanged; file untouched |
| artifacts-demo-5 | cleanup | low | F | open | training/monitor_log.py untouched |
| artifacts-demo-6 | consolidation | low | F | open | training/artifacts.py:209-244 both functions still duplicate the validation |
| callbacks-4 | cleanup | low | F | open | fallback moved, not removed: info_dict_eval.py:367-376 |
| callbacks-5 | cleanup | low | F | open | callbacks/video_record.py untouched |
| callbacks-6 | cleanup | low | F | open | train.py:66-69 still imports _reset_options_tuple, _resolve_min_deltas, _string_keys |
| callbacks-7 | cleanup | low | F | open | env_attr_schedule.py:151-157 comment unchanged |
| callbacks-8 | cleanup | low | F | open | info_dict_eval.py:1141-1159 heartbeat has no confirmation marker |
| callbacks-9 | cleanup | low | F | open | docs/DECISIONS.md:2324-2326 still says the evaluator never re-seeds |
| consolidation-10 | cleanup | low | F | open | notebook_utils.py:2086-2089 docstring still names vec_normalize.pkl; loader prefers best_vec_normalize.pkl |
| consolidation-4 | bug | low | F | open | three copies remain: ladder_certification.py:377, tennis_curriculum.py:705, tools/depth_stage_sweep.py:524 |
| consolidation-6 | cleanup | low | F | open | e.g. performance_gate.py:430 stages_metadata still has no caller in src/tools/tests |
| consolidation-7 | consolidation | low | F | open | seed_start 5200 still at train.py:2383,3954, recipes.py:1494, notebook_utils.py:2616 |
| consolidation-8 | consolidation | low | F | open | pin rule still at demo_sac.py:111-113, train.py:910-911, notebook_utils.py:3038-3039,3190-3191, tennis_curriculum.py:336 |
| consolidation-9 | consolidation | low | F | open | campaign make_leg_config still layers ENV_KWARGS via make_env_fn/make_eval_env_fn and probes factories itself |
| env-paddle-3 | consolidation | low | F | open | envs/paddle_tennis.py and _paddle_court.py: no structural change |
| env-paddle-4 | cleanup | low | F | open | envs/_paddle_court.py:432 'policy.predict(obs, ...)' recipe unchanged |
| env-paddle-5 | cleanup | low | F | open | recipes.py:309-311 comment unchanged |
| env-paddle-6 | cleanup | low | F | open | envs/_paddle_court.py:1-3 'Prototype'; paddle_tennis.py:341 |
| env-paddle-7 | cleanup | low | F | open | envs/paddle_tennis.py:723-757 stopped set and del'd, never read |
| env-wallball-simple-5 | cleanup | low | F | open | docs/DECISIONS.md:691 heading still 'implemented (0.11.0)'; no WallBall recipe sets the exclusion |
| env-wallball-simple-6 | cleanup | low | F | open | envs/wall_ball.py only _linalg change |
| humanoid-3 | cleanup | low | F | open | envs/__init__.py untouched; M8 not executed |
| humanoid-4 | cleanup | low | F | open | repo_review_20261005 §7.1 unchanged; design M8 (design_env_ladder §4) restates the archive without a removal map |
| humanoid-5 | cleanup | low | F | open | envs/tennis_curriculum.py untouched |
| notebook-utils-4 | cleanup | low | F | open | best_model_video still in EXPECTED_ARTIFACTS (artifacts.py:1375-1406); campaign never records it |
| notebook-utils-6 | consolidation | low | F | open | record_best_model_video/_load_obs_normalizer (notebook_utils.py:2074-2110) do no pair check; only evaluate_best_wall_ball (1838) and score_paddle_stage (2607) verify |
| notebook-utils-7 | consolidation | low | F | open | notebooks/humanoid_tennis_training.ipynb:359 still parses 'Status:' by hand |
| notebook-utils-8 | cleanup | low | F | open | sb3_training.ipynb section 6 still says LOG_DIR/best_model.zip |
| recipes-config-4 | cleanup | low | F | open | recipes.py:309-314 comments unchanged |
| recipes-config-5 | cleanup | low | F | open | run_configs/wall_ball_depth_curriculum_aligned.toml untouched |
| recipes-config-6 | cleanup | low | F | open | run_config_file_spec.md updated only for v1.2 PPO family tables; batch-era settings (e.g. eval_reset_options 'none', 71f2d48) still absent |
| recipes-config-7 | cleanup | low | F | open | __init__.py:114 log stops at 0.19.0, __version__ 0.25.0 at :120 |
| rules-events-2 | bug | low | F | open | envs/_paddle_court.py untouched since b9585e2; three loop copies remain |
| rules-events-3 | cleanup | low | F | open | envs/tennis_rules.py:115 docstring unchanged |
| rules-events-4 | cleanup | low | F | open | envs/tennis_rules.py untouched |
| rules-events-5 | cleanup | low | F | open | envs/__init__.py:3-4 docstring unchanged |
| tests-4 | cleanup | low | F | open | tools/_seed_ledger.py citations unchanged (:48-121) while DECISIONS.md grew +1589 lines |
| tests-5 | cleanup | low | F | open | tests/test_train.py grew +2619 lines; no slimming |
| tests-6 | cleanup | low | F | open | tests/test_humanoid_tennis_env.py untouched |
| tools-10 | cleanup | low | F | open | tools/paddle_tennis_probes.py untouched |
| tools-4 | cleanup | low | F | open | tools/_seed_ledger.py untouched; e.g. cited DECISIONS.md:554 now lands mid-paragraph |
| tools-7 | cleanup | low | F | open | tools/paddle_tennis_diagnosis_probe.py:87 --oracle parsed, never read (109-116); p5_transfer.py:448 --stub-oracle same |
| tools-8 | cleanup | low | F | open | tools/paddle_tennis_postswing_target_probe.py:100,185 |
| tools-9 | cleanup | low | F | open | tools/paddle_tennis_volley_probe.py:188 no caller |
| train-7 | consolidation | low | F | open | same copies as consolidation-8 |
| training-data-7 | cleanup | low | F | open | recipes.py:313-314 comment unchanged |
