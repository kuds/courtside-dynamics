# Repository review and baseline-rally plan — 2026-10-05

Kind: **Review snapshot**, pinned to `main`@`08b86c2` (2026-09-27),
conducted 2026-10-05. Scope: the whole repository, with the PaddleTennis
campaign and the next milestone in front: **a paddle that rallies from
the baseline**. Method: four parallel review scopes, each reading its
modules in full:

- training history from the docs;
- the Google Drive run archive (`courtside-dynamics/training_runs`);
- the paddle env stack and its probe tools;
- the training, callback and notebook infrastructure.

Each material finding was then re-checked here, by reading the code or
with a scratch probe. Items marked *confirmed* were reproduced. Lint,
mypy and the full test suite (1073 passed, 1 skipped for no GL) were
green at the pinned commit.

Two commits on the review branch land the bit-identical and
infrastructure fixes (§5a). Everything else is routed rather than landed,
using the 2026-08-28 review's labels: **[era-boundary]** changes
observations, eval semantics or a recipe's frozen shape, and
**[cleanup]** has no behavioral impact.

## 1. Where training stands

- **The last training run ended 2026-08-29.** Nothing has trained in the
  37 days since; only CPU probes ran (PT2, the k=2 harvest/step-0,
  pre-freeze diagnostics, Phase 0 gates, the demo harvest).
- **Drive agrees.** The newest courtside-dynamics content is
  `diagnostics/pt1_ctrl_streams_20260829` (Aug 30). There is no LD1′,
  DemoSAC or demo-library run. The 2026-10-05 Drive activity is a
  different project (mesozoic-labs).
- **The headline has not moved since mid-August.**
  - k=1 receiving (returning the serve) is 90–95%.
  - k=2 (a second policy return in the same point) is ≤1% on every final
    checkpoint since 2026-08-09, with one 5% blip on LH1c at 2.5M that was
    1.6% [0.3, 4.5] on fresh seeds.
  - Crossings have plateaued at about 5.5–6.0 per episode, against the
    oracle pair's 11.4–12.3.
- **About 25M SAC steps over 16 runs** went into six levers. All were
  falsified on k=2: the exploration package, contact shaping, n-point
  play, reach shaping, post-swing hold, and temperature skip. Command-rate
  limiting and the k=2 drill were retired before launch.
- **LD1′ (DemoSAC demo injection) is built, and its launch blockers are
  closed (`b43b6ba`), but it is not frozen.**
  - `demo_fraction`, the BC coefficient and the KD bar numbers are
    unset.
  - The notebook's `ALGO` defaults to `None`.
  - The class-split diagnosis extension (D-G) and critic head-recycling
    (D-B) are not in code.

| Run | Date | Steps | Lever | k=1 / k=2 (final ckpt) | Crossings final/best |
|---|---|---|---|---|---|
| `20260809_211147` | 08-10 | 2M | contact shaping (1-pt) | 100% / 0% | 1.50 / 1.77 |
| `20260815_015143` | 08-15 | 2M | n-point from scratch | 0% / 0% (statue) | 3.00 / 3.20 |
| `20260816_235141` (registered) | 08-16 | 3M | n-point + contact + reach | 95% / 0% (2.4M: 95/1) | 5.63 / 5.97 |
| `20260821_013700` (LH1c) | 08-21 | 3M | + post-swing hold 0.5/12 m | 90% / 1% | 5.97 / 6.17 |
| `20260828_113136` (companion) | 08-28 | 3M | temperature skip | 90% / 0% | 5.20 / 5.67 |
| `20260828_121324` (LT1) | 08-28 | 1M warm | temperature skip | 46% / 1% | 4.37 / 5.57@25k |

The full chronology is in `paddle_tennis_npoint_pilot_20260815_appendix.md`
and the dated reviews. Drive folder of record:
`courtside-dynamics/training_runs/PaddleTennis/sac/`.

## 2. Why k=2 is stuck: the campaign's own evidence, read together

The diagnostics are already in the repo. What they add up to:

1. **The policy is gated on bookkeeping context, not physics.** The same
   physical k=2 state converts 6.9% when presented as a fresh feed and
   1.3–2.0% in its real mid-rally context. Swapping the rally-context
   block (obs dims 24–35) between the two presentations moves the
   deterministic action by 1.60 on average (`paddle_tennis_prefreeze_diagnostics_20260830.md`
   §4). The carriers are the phase one-hot, `rally_count`, the
   crossing flags and the episode clock. They are redundantly encoded,
   so they have to be removed as a block. The scripted oracle reads none
   of them; it uses physics plus `bounce_count` / `ball_side`, and is
   context-blind: 77.5% under both presentations.
2. **Even the fresh-feed number is low (6.9%).** k=2 balls are the
   opponent's soft ground returns, not the P3 serve band the policy
   practised on 50% of points. The policy learned "one memorized
   serve-return macro" (`paddle_tennis_diagnosis_20260808.md`), and
   nothing in training widens the incoming-ball distribution.
3. **Credit spans more than the discount horizon.** At 100 Hz control,
   SB3's default γ = 0.99 is about a 100-step horizon. The measured
   hit-to-hit cadence is about 105 steps, so consecutive side-A hits are
   about 210 steps apart (0.99^210 ≈ 0.12). WallBall found the same
   problem and moved to 0.995 (DECISIONS, 0.13.0). The PaddleTennis
   pre-registrations never vary γ.
4. **Home is the wrong place to stand.** Measured with one-point
   episodes and a constant action, 40 points per side:
   - **Receiving.** The zero action parks the paddle in the incoming
     flight path: 0/40 legal touches and 16/40 volley faults. Parked
     deep and low (x ≈ −5.5, z ≈ 0.5) it gets 15/40 legal touches and
     2/40 volley faults.
   - **Serving.** The ballistic serve launches from x = −3.25, behind
     the home paddle at −1.7, so it flies through the policy's own home
     column. The zero action clears it. Holding just 0.2 m higher
     intercepts it on 6/40 serve points (`wrong_hitter`).

LD1′ attacks (1) by giving the critic successful transitions in the
failing context. That is a legitimate fix, but an indirect one. The direct
fix for (1) is to **stop showing the policy the context it gates on**.
(2)–(4) are task-design levers that no lever so far has touched.

## 3. Baseline-rally gap (new measurement)

The oracle-vs-oracle pair rallies well (12.25 crossings per 1500-step
episode, n-point), but **from mid-court, not the baseline**. On the
13 m court (half-length 6.5 m), over 78 side-A contacts:

- paddle x at contact (percentiles 10/50/90): −5.46 / −4.25 / −3.15 m;
- side-A bounce x: −4.86 / −3.66 / −2.56 m.

The paddle's x reach ends at −6.4 m, inside the baseline. A deep ball
bouncing near the baseline would need contact behind it, which the
workspace cannot provide. **"Rally from the baseline" is therefore a new
task era, not a new lever on the current one:**

- a deeper home pivot;
- a workspace that extends behind the baseline (the
  `WallBallTrueBaseline` precedent);
- deep feeds;
- a metric that measures contact depth (none exists today).

There is also an open question only the maintainer can answer.
**"Tennis court" could mean the 13 m paddle court or a regulation 23.77 m
court.** The paddle tops out around 12.5 m/s, enough for
baseline-to-baseline on 13 m (ballistic range at 45° ≈ 15.9 m) but not
on 23.77 m without stronger actuators.

## 4. Recommended next steps (ordered)

### 4.1 Fix the instruments first (about 1 day) — [era-boundary]

- **Episode-cumulative policy counters.**
  - Add `episode_legal_hit_count_a` and `episode_valid_return_count_a`
    to the info, alongside `completed_point_crossings`.
  - Point `success_key` and the degenerate guard at them.
  - Today both read the *last, unfinished* point. The oracle playing
    side A made 6–7 hits per episode but scored 2–7 on the terminal
    `legal_hit_count_a`. *Confirmed.*
- **Select on what the goal is.**
  - Make the headline a policy-side rally measure (k≥2 rate, or the
    policy's confirmed returns per point), not opponent-dominated
    `crossings`.
  - Give `best_metric_min_delta` a per-key mapping. One 0.25 delta
    applied to `success_rate` means the tie-break needs a +25 pp swing
    (`info_dict_eval.py` `_improves`). *Confirmed.*
- **Depth.**
  - Publish contact |x| at each side-A legal hit.
  - Add a depth-gated success key, e.g. a confirmed return with contact
    |x| ≥ 5.5 m.
- **Reproducible eval.**
  - `reset(seed=s)` is not reproducible: the serve side comes from
    `_next_serving_side`, which seeding never touches, so the same seed
    twice gives different observations. *Confirmed.*
  - Accept `options={"serve_side": ...}` as the humanoid env does, and
    run eval on a fixed seed list so evaluations are paired.

### 4.2 A context-blind observation profile (the cheapest direct attack on §2.1)

- **Add `observation_profile="physical"`** as a default-off env kwarg.
  - Drop the gate-carrying dims: phase one-hot (24–27),
    `own_is_serving` (28), `feed_crossed_net` (31),
    `pending_return_crossed_net` (32), `rally_count` (34) and the
    episode clock (35).
  - Keep what the oracle needs and what read identically across
    presentations: `expected_returner_is_own` (29), `ball_side_is_own`
    (30), `bounce_count` (33) and the contact tail.
  - SB3 handles time-limit bootstrapping through `TimeLimit.truncated`,
    so the clock is not needed for correctness.
- **Land with it, since the observation changes anyway:**
  - world-frame ball spin (§5b);
  - clipped and scaled counters.
- **Pilot.**
  - The adopted recipe, trained from scratch: the registered run shows
    this shape reaches k=1 ≈ 95% by about 2.4M.
  - Two seeds × 3M steps, γ = 0.995.
  - Bar: k=2 ≥ 5% on fresh seeds = ADOPT.
- **If it fails, the gate was not the binding constraint.** LD1′ then
  stays the next lever, and §4.4 the one after.

### 4.3 Make iteration cheaper (before any long run) — [cleanup] / [era-boundary]

- **Eval is about 55–65% of wall clock.**
  - 1500-step n-point episodes run as two 30-episode streams per eval,
    each on the same distribution, plus `confirm_best` re-runs. That is
    about 90k eval steps per 25k training steps.
  - Fix: let the selection stream own `evaluations.npz` (or set
    `reward_eval_episodes=5`), and vectorize eval.
- **`SubprocVecEnv`.**
  - `n_envs=8` on `DummyVecEnv` steps serially (about 25% of training
    time).
- **Env step.**
  - The redundant per-substep safety scan is removed in this review's
    second commit; about 15–20% faster and bit-identical (§5a).
- **Warm starts sample uniform-random actions.**
  - Every warm-started leg (LD1′ included) spends its first
    `learning_starts` steps on uniform-random actions, because
    `use_sde_at_warmup=False`. *Confirmed; known since 2026-08-28, still
    open.*
  - Fix this before launching LD1′.
- **Exploration under gSDE.**
  - gSDE's `log_std_init=-3` gives an initial noise std of about 0.14,
    against plain SAC's 0.62, and `learning_starts=100` is effectively
    no warmup.
  - Worth one sweep point (`log_std_init` −2, 10k warmup) for
    from-scratch runs.

### 4.4 Widen the incoming-ball distribution ("ball machine")

Train the receive skill on a feed distribution that covers what the
opponent actually returns:

- origin depth, speed, elevation, lateral angle and spin;
- or the oracle's harvested returns at net crossing.

With the context-blind profile, the k=2 drill's *feed* arm becomes
exactly this, and its step-0 engagement objection loses its basis
because the context mismatch it measured is gone.

### 4.5 The baseline era

Build it only after 4.1–4.4 move k=2, because they all carry over.

1. **Probe first, as the campaign already does.**
   - Move the home pivot to about −5.0 to −5.5.
   - Extend the x workspace behind the baseline (for example to −8.0).
   - Move the feed origin deep.
   - Re-calibrate the ground oracle until the oracle pair rallies with
     a contact-depth median ≥ 5.5 m, and freeze that band as the
     reference.
2. **Keep the serve out of the server's own column.** A deep home
   would sit behind any mid-court launch origin, but a deep feed origin
   would collide again.
   - Either make the server's paddle non-colliding during the initial
     feed, or launch from a position the server's paddle cannot occupy.
3. **Reward depth.**
   - Pay for (or gate success on) returns that land beyond the service
     line, so rallies stay deep rather than collapsing to mid-court.
4. **Decide court size (§3) before any of this.**

### 4.6 Process

- The campaign's science is careful: pre-registration, adversarial
  verification, and falsified levers stay falsified. But its cadence has
  become the bottleneck.
  - 43 of 99 commits since 08-02 are docs-only.
  - There are 27 PaddleTennis docs, about 58k words.
  - No run has gone out in 37 days.
- Suggested split:
  - Run cheap, multi-seed **exploratory** pilots (§4.2, §4.4) without a
    full pre-registration.
  - Keep pre-registration for **claims** (a k=2 ADOPT, a promotion).
  - Keep a single living "PaddleTennis status" page instead of
    accreting dated docs.

## 5. Bugs

### 5a. Fixed on this branch

| Fix | Where | Evidence |
|---|---|---|
| Recipe dicts were aliased into every `TrainConfig` (shallow `update`), so mutating `cfg.model_kwargs` retuned every later build of that recipe in-process | `recipes.py` `build_train_config` | *Confirmed*; regression test fails before, passes after |
| `tennis_curriculum._git_sha` was a stale bare `git rev-parse` copy (null on Colab installs) | `training/tennis_curriculum.py` | now imports `artifacts._git_sha` |
| Reward `EvalCallback` printed "New best mean reward!" each eval under headline selection, where it saves nothing | `training/train.py` | `verbose` now wired |
| `gymnasium>=1.0` floor too low: callbacks pass `set_wrapper_attr(..., force=False)` and read its bool; 1.0.0 has neither | `pyproject.toml` | *Confirmed* against the 1.0.0/1.1.0 wheels; floor now 1.1 |
| Starter TOML header and README described the retired one-point/crossings recipe; `eval_freq` comments said "per env" | `run_configs/paddle_tennis.toml`, `ball_balance.toml`, `README.md` | doc-only |
| Pre-step safety scan ran on every substep, re-scanning the state the previous substep's sample had just passed | `envs/paddle_tennis.py` | bit-identical digest over 4,500 random steps; about 0.91 → 0.75 ms/step |

### 5b. Open, routed

| # | Finding | Where | Route |
|---|---|---|---|
| 1 | Ball spin observed in the ball's body frame. After the first contact it differs from world spin by up to 28 rad/s (*confirmed*), which is exactly the k=2 regime. Same pattern in `wall_ball.py`; already open for the humanoid env (DECISIONS). | `paddle_tennis.py` `_ball_angular_velocity` | era-boundary (land with §4.2) |
| 2 | n-point `success_rate` and the degenerate guard read the last, unfinished point | recipe `success_key`, info | era-boundary (§4.1) |
| 3 | `reset(seed)` is not reproducible (serve-side alternation ignores the seed) | `paddle_tennis.py` reset | era-boundary (§4.1) |
| 4 | A ball the receiver never touches is labelled `out_of_bounds`, not `second_bounce` (known, 2026-08-28 review §2.9) | `tennis_rules.py` | era-boundary |
| 5 | `term_*` info keys contradict each other at n-point boundaries (`term_volley_return=True` beside `term_volley=0.0`); the forced-nonfinite path reports `termination_reason_name="none"` | `paddle_tennis.py` `_build_info` | era-boundary |
| 6 | `rally_count` and `bounce_count` sit raw in the unnormalized tail (`rally_count` reached 13 in oracle rallies) | obs[33:35] | era-boundary (§4.2) |
| 7 | Warm-start warmup samples uniform-random actions (known since 2026-08-28) | `train.py` warm start + recipe | before LD1′ |
| 8 | One `best_metric_min_delta` is applied across keys of very different scale | `info_dict_eval.py` `_improves` | era-boundary (§4.1) |
| 9 | `plot_learning_curve` marks the reward argmax as "best checkpoint", but headline recipes select by the headline metric | `notebook_utils.py` | cleanup |
| 10 | The campaign gate verdict and warm start load model + normalizer without the sha256 pairing check that `evaluate_best_wall_ball` does | `notebook_utils.py` `score_paddle_stage` | cleanup |
| 11 | No mid-leg resume: the replay buffer is never checkpointed, so a Colab disconnect at 2.9M of 3M restarts the leg | `train.py` | cleanup |
| 12 | `k2_step0` restores `_crossings` but not `_crossings_offset`; the opponent controller raising on non-finite output crashes a vectorized rollout | tools, env | cleanup |

## 6. Cleanup and consolidation register

Ordered by value. Line counts are estimates. "Archive" means tag, then
delete, with a pointer in `DECISIONS.md`, so reproducibility survives.

| Item | Savings | Notes |
|---|---|---|
| Archive the retired WallBall recipes: Bootstrap, DepthCurriculum, DepthCurriculumAligned | about 480 recipe + 166 TOML lines | `WallBallGoalRally`/`TrueBaseline` derive from `RECIPES["WallBallDepthCurriculum"]`; inline about 70 lines of base kwargs first |
| Gate advance machinery used only by those ladders (pause, replay clear, entropy reset, `stage_eval_budget`) | about 400 of 809 lines in `performance_gate.py` | active recipes use a single-stage gate that never promotes |
| `tools/depth_stage_sweep.py`: only `--ladder release` is live | about 1,100 of 1,309 lines + tests | serves the closed serve-alignment and sliding-fence campaigns |
| Hold shaping (closed without adoption) | about 1,050 lines across env, tests and two tools | revisit the keep-in-env decision |
| One-off instruments for finished gates: `k2_step0`, `command_spectrum_analysis` | about 600 lines | command-rate was never implemented |
| `tools/_paddle_common.py`: `_refuse_reserved` (5 drifted seed lists; the 2026-08-28 review already ranked a shared seed ledger first), sha/git helpers (3×), recipe `env_kwargs` hand-copied in 6 tools | about 150 lines | removes drift channels |
| Merge the three escrow-witness probes (shaping, reach, hold) | about 450 lines | they differ mostly in key names |
| `src` helper duplication: sha256 (5×), frozen-normalizer loaders (3×), `_wilson_interval` (2×), `eval_info.csv` parsers (3×) | about 120 lines | `_git_sha` already drifted once (fixed in §5a) |
| Split `notebook_utils.py` (2,760 lines): Colab/plots/replay/WallBall instrument/paddle campaign, with re-exports | structural | notebooks need no change |
| Decompose `train()` (about 900 lines) | structural | as the 2026-08-28 review also says |
| `PaddleTennisEnv`: `_draw_serve` duplicates `PaddleCourtScene.serve`; joint/actuator names repeated 4×; nine reward-component kwargs and two clawback blocks could become a dict and one helper | about 250 lines; `step()` drops from 210 to about 120 lines | |
| WallBall → `PaddleInterface` delegation | about 200 lines + a drift test | |
| Docs | — | stale statements remain (CHANGELOG "recipe does NOT enable n-point/shaping"; LT1 prereg §4a 3M "recipe default"; `docs/README.md` says "three kinds" and lists four); replace dated PaddleTennis status docs with one living page |

## 7. Follow-up audit: humanoid removal, fix-first, cleanup-first (2026-10-05)

The maintainer's goal for PaddleTennis is **baseline rallies where the
ball bounces once in the singles court before being hit back across the
net**. Three questions followed. A second pass answered them with four
scoped audits: humanoid removal map, bug triage against the next run,
foundation readiness, and a rules-engine audit against that goal.

- Every medium-or-higher *new* finding and every load-bearing claim got
  an independent adversarial verifier. That was 14 verifications, 12
  confirmed and 2 refuted or downgraded; the corrections are applied
  below.
- A completeness critic then checked all four scopes.
- Scratch probes are under the session scratchpad; none are committed.

### 7.1 Remove the humanoid code now? Yes, as an archive, in parallel with the fixes, not as a prerequisite

**Evidence:**

- **No runtime coupling.** At runtime the PaddleTennis path never calls
  humanoid code; the coupling is import-time only (`envs/__init__.py`,
  `training/__init__.py`, `recipes.py`, `scripted_policies.py`, and the
  gym registration).
  - Today every paddle import also loads `humanoid_tennis`,
    `robot_models` and both `tennis_curriculum` modules, so a broken
    edit to dormant humanoid code can take down paddle training on
    Colab.
  - A deletion dry-run kept ruff and mypy clean and the paddle,
    DemoSAC, train and notebook_utils tests green. PaddleTennis obs,
    reward and full info stayed **byte-identical** across random,
    oracle and lead-charge play (*confirmed*).
- **Size.**
  - About 4.7k src lines go: 4.0k humanoid-only, plus about 0.7k in
    `recipes.py` and `scripted_policies.py`.
  - About 4.1k test lines go, and 1.07k XML lines.
  - The G1 assets are 19.7 MB, which is **96.7% of the wheel**
    (9.04 MB → about 0.3 MB) on every Colab `pip install`.
  - The CI saving is real but small: about 30 s per matrix leg (about
    15%).
- **Dormant either way.** No humanoid training has run since 0.16.0
  (2026-07-21). Resuming already needs the three [humanoid-resume] bugs
  and the four structural transfer gaps from the 2026-08-28 review, so
  resumption is a re-design regardless.

**Conditions:**

- **Tag first.** Create `humanoid-tennis-archive-v0.25.0` plus an
  archive branch, from a full clone (this one is shallow and the repo
  has no tags).
- **Port before deleting.** Port the ~29 shared-coverage tests to the
  paddle court *before* deleting anything.
  - 32 of 36 `test_tennis_events.py` items build the humanoid scene.
  - The markov-state and safety-drain sampler tests, the only GL render
    smoke (`ci.yml:128`) and the only gymnasium `env_checker` test are
    humanoid tests.
  - Porting them gives the paddle court its first `in_bounds` coverage
    at half-length 6.5 m.
- **Carry the reset pattern over.** Copy the seeded `reset(options)`
  serve-side pattern (`humanoid_tennis.py:741-811`) for §5b #3.
- **Keep the old docs working.** Pin the humanoid notebook's install to
  the tag, add a README roadmap section, and bump the version, since a
  registered ID is removed.
- **Pruning can ride along.** Pruning the dead humanoid enum members and
  contact channels may go in the same PR. The PaddleTennis recipe pins
  its CSV and eval keys, so dropping the 24 always-zero humanoid info
  keys changes no log. Keep the explicit `TerminationReason` integers.

### 7.2 Fix bugs before more training? Yes: a targeted 1–2 day batch, chosen by which run is next

Nothing found corrupts physics, the rules machine's legal/illegal calls,
or the reward-escrow identity, so past verdicts stand. What is broken are
the **instruments that judge, select and stop the next run**:

| Fix | Why (evidence) |
|---|---|
| Episode-cumulative policy hit/return counters; point `success_key` and the degenerate guard at them | They read the last, truncation-cut point. The oracle made 7 hits in each of 3 episodes, but the terminal key read 7/7/3 (*confirmed*) |
| Per-key `best_metric_min_delta`; a policy-side headline (k≥2 rate or policy returns per point) instead of opponent-dominated `crossings` | A +20 pp `success_rate` gain at tied crossings does not count as an improvement (*confirmed*) |
| Guard flatness: drop `episode_reward_mean` from the flatness test | A dead statue run is stopped at about eval 9–26, not the designed eval 5 (*confirmed*; wasted compute only) |
| Reproducible `reset(seed)` (`options={"serve_side": ...}`) and paired eval seeds | The same seed twice gives different observations (*confirmed*) |
| `reward_eval_episodes=5` for PaddleTennis (WallBall already does this) | 60 × 1,500 = 90k eval steps per 25k training steps, before `confirm_best` |
| Contact-depth info key and a depth-gated success key | Nothing measures "from the baseline" today |
| **Observation fingerprint**: record observation names and env kwargs in `config.json`; check them on warm start and on demo-library load | Today both check only shapes. A same-shape meaning change (world-frame spin, scaled counters) would silently load every old checkpoint and the LD1′ demo library onto a different task (*new*) |
| Tag the current era before merging the batch | Both notebooks install from `main`, so any merge changes what the next Colab session runs |

Then, depending on the next run:

- **§4.2 context-blind pilot (recommended next run).**
  - **Read the oracle's observation fields by name first.** The ground
    oracle and the diagnosis reference read `obs[30]` and `obs[33]` by
    literal index from the same vector the policy sees. Dropping dims
    without a layout object silently corrupts the opponent or the
    reference row: the oracle's rally count fell from 1–13 to 0–1
    (*confirmed*).
  - **Land the obs changes behind the `observation_profile` flag or
    one declared era break:** world-frame spin (§5b #1), scaled
    counters (#6) and the label fixes (#4, plus a new own-side
    `failed_to_cross` mislabel).
  - **Declare γ = 0.995 as part of the bundle,** or give it its own
    arm, because γ has never been varied on this task.
- **LD1′ first instead.**
  - Fix the uniform-random warm-start warmup (§5b #7).
  - Keep every obs or reward fix default-off and bit-identical.
- **Decide explicitly (low impact): truncation clawback.** The escrow
  claws back at time-limit truncation, which SB3 also bootstraps
  through, so the pending escrow is double-counted. That happens at
  most once per 1500-step episode.

### 7.3 Wider clean-up first? No: resume after the targeted fixes

**Health is good.**
- ruff and mypy are clean, and 1073 tests pass.
- CI covers ruff, mypy, a wheel smoke test, pytest on 3.11–3.13 with
  thread pinning, a render smoke test and a weekly re-resolve.
- Validation fails loudly.

**The §6 register can wait.** It is almost entirely off the paddle
critical path (WallBall ladders, `depth_stage_sweep`, gate ladder
machinery, splitting `notebook_utils`, decomposing `train()`). Doing it
now adds regression risk and makes no run more trustworthy. Do it
opportunistically after the first new pilots.

**Targeted structural work before the baseline era** (not before the
§4.2 pilot), about 3–5 days with the oracle re-probe:

- **A single source of truth for paddle geometry.**
  `PADDLE_HOME_X`/`PADDLE_LOCAL_*` are hand-copied from the XML, and no
  test names them. They calibrate the oracle, which is also the
  opponent. Add an A/B mirror test of the XML.
- **One serve-draw function, with a clearance check on every launch.**
  `_draw_serve` duplicates `PaddleCourtScene.serve`, and reset does no
  clearance check. A serve origin 0.2 m from the server's home ends
  38/40 points `wrong_hitter` with no warning (latent under the frozen
  serve).
- **A per-instance observation layout object,** read by name.
- **Retire hold shaping.** Keep the k=2 drill while LD1′ is still a
  candidate: it is LD1′'s named RE-AIM escalation, and the demo harvest
  loads through it.
- **Probe tools build the env from the recipe,** with one seed ledger.

### 7.4 Rules engine against the stated goal

The engine already enforces the target drill correctly. Probes set the
ball state directly on the env; the sampler had 0 of 2,381 bounces on
the wrong side and no tunnelling. Specifically:

- **In/out:** judged on the singles court (|y| ≤ 4.115, |x| ≤ 6.5). A
  line ball is in, using the ball-centre convention, about 2.5 cm
  stricter than tennis.
- **Bounce rule:** exactly one bounce before the return. A pre-bounce
  hit is `volley_return` and a second bounce is a fault.
- **Return credit:** only when the shot crosses the net and its first
  bounce is in.
- **Baseline-era positions:** contact from behind the baseline or
  outside the sideline is legal.

Gaps to close when the baseline-era env is built:

| Gap | Evidence | Route |
|---|---|---|
| Lateral reach is \|y\| ≤ 3.2 m (head ±3.0 plus a 0.2 m face) against the 4.115 m singles half-width; x reach stops at −6.4, inside the −6.5 baseline | Probe; dormant today because oracle landings reach at most \|y\| 2.23 | baseline era |
| The net panel ends at the singles sideline, so a low ball can pass *beside* it and count | Probe: crossing at y = 4.49, z = 0.65, then a confirmed return | before any y-workspace widening |
| Racket–net contacts are never generated (collision bits 8&2 \| 4&2 = 0), so the paddle would pass *through* the net; obs dims 40, 41, 46 and 47 are constant zero | Probe; unreachable today by a 1.7 cm workspace gap | baseline era; drop the dims with §4.2 |
| Bad feeds (long, wide, net) are charged as point-ending faults with no let, and feed origins behind the baseline are refused | 3 of 2,000 frozen-band feeds land out; re-centring at 5.5–6 m would put about 10–25% long | baseline era: feed faults become lets |
| Any net touch ends the rally (`ball_net_is_fault=True`); the rigid tape pops clipped balls up at 60–93° | `ball_net` ends 86% of oracle one-point rallies, and 20–34% of those would be play-on in tennis; oracle net clearance is the binding limit on reference rally length | keep as a deliberate drill rule; document it in DECISIONS |
| An untouched ball's second bounce past the line is labelled `out_of_bounds` (§5b #4), and a shot landing behind the hitter's own baseline is labelled `out_of_bounds`, not `failed_to_cross` (*new*) | About 89% of oracle `out_of_bounds` labels are really second bounces | era boundary, one fault-taxonomy change |
| About 1% of tape grazes deflect the ball with no event (contacts sampled only after each RK4 step) | 1 unexplained deflection against about 86 detected net faults in 100 oracle points | park until a declared break |

### 7.5 The decision that sets the before-training list

The scopes disagreed only because none of them chose the next run.
Recommended order:

1. Tag the current era. The humanoid archive can proceed in parallel.
2. Land the §7.2 instrument batch and the observation fingerprint.
3. Run the §4.2 context-blind pilot: from scratch, 2 seeds × 3M,
   γ = 0.995, with the bundle declared.
4. Whatever k=2 does, do the §7.3 foundation pass and the §7.4 gaps,
   re-probe the oracle pair on the baseline geometry, then train the
   baseline era.

LD1′ stays the fallback if the pilot leaves k=2 at or below 1%.

## 8. Status: the bug-fix batch is landed (2026-10-05)

The maintainer approved fixing every bug before the pilot.

- **Scope.** Everything in §5b and §7.2, plus the label fixes from
  §7.4. The exceptions are the baseline-era geometry gaps and the
  items deferred to the observation profile (spin frame, counter
  scaling).
- **How it was built.** Three parallel implementers (env/rules,
  training infrastructure, notebooks/tools) and an integrator, followed
  by a six-lens adversarial review.
  - 14 medium findings were verified by execution and fixed. The most
    important: paired evaluation had been switched on for every seeded
    run, the selection deltas were too coarse to see one conversion,
    and confirmation compared different seed blocks.
  - A follow-up round fixed 11 low findings, each re-checked by an
    independent verifier against mutants.
- **State at the end.** ruff and mypy are clean. The suite has 1278
  passed and 1 skipped (no GL), in 145 s against 159 s before the
  batch. The task observations, rewards and endings are bit-identical
  to `de02d13`. The decisions are booked in `DECISIONS.md` ("The
  instrument batch …"), the changes in `CHANGELOG.md`.
- **Next:** the §4.2 context-blind observation-profile pilot.
