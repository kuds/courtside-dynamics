# WallBall Baseline Experiment Review

Review of the two `WallBallBaseline` SAC runs uploaded to Google Drive on
2026-07-14 (`WallBallBaseline/sac/20260714_050506` and
`.../20260714_211111`), cross-referenced against the environment, recipe,
training, and callback code at `cdb17d4` (HEAD, v0.9.0). Evidence comes
from each run's `config.json`, `stage_summary.txt`, `eval_info.csv`,
`evaluations.npz`, Monitor logs, per-checkpoint rollout CSVs, best-model
videos, and the 50-episode long-horizon endurance evals.

## TL;DR

- **Run 1** (pre-0.9.0 code at `1eec0d4`, obs 22, success = `bounce_count >= 1`)
  converged by ~225k steps to a degenerate "one-and-done" policy: exactly one
  legal paddle hit, one wall return, then it lets the ball double-bounce. It
  never improved again over the remaining 875k steps. 50/50 long-horizon
  episodes show exactly 1 return; the max training return in 1,152 episodes
  was 2.04 — a second exchange was effectively never visited.
- **Run 2** (v0.9.0, recovery curriculum + `recoverable_bounce_bonus`,
  success = `bounce_count >= 2`) is a **total eval failure**: the deterministic
  policy never touched the ball after the first eval — 0 paddle contacts in
  900/900 eval episodes, every episode ending in double-bounce at ~112 steps
  with return exactly −1.0, while *training* reward climbed to ~1.8–2.2 by
  farming recovery-reset starts. Its `best_model.zip` was selected by a
  ~1e-8 floating-point tie-break and is untrained-equivalent.
- The single most important finding is a **confirmed code bug**: the
  recovery-curriculum schedule never executed. Run 2 trained at
  `recovery_reset_probability = 0.6` for its entire duration instead of
  tapering 0.6 → 0.15. Details in Finding 1.
- The second most important finding is **geometric**: with
  `paddle_x_target_range = (-3.2, -2.1)`, ~73% of second-exchange balls are
  physically uninterceptable (their second bounce lands in front of the
  reachable lane), so the raised `bounce_count >= 2` success bar is close to
  unreachable regardless of reward tuning — and no oracle/solvability test
  proves it reachable. Details in Finding 2.

Prioritized changes are in the [Recommendations](#recommendations) section;
a suggested run sequence is at the end.

## What the runs actually did

### Run 1 — `20260714_050506` (labeled v0.8.0; actually unreleased `1eec0d4` code)

| Item | Value |
|---|---|
| Steps executed | 1,100,000 of 1,500,000 (early stop) |
| Best model | step 600k, `bounce_count_ep_mean` 1.033 (exactly one 2-bounce episode among 30) |
| Eval plateau | reward ≈ 1.96 ± 0.07, `bounce_count` pinned at 1.0 from 225k → 1.1M |
| Success rate (≥1) | saturated at ~1.0 from 225k — no signal for 80% of the run |
| Long-horizon (50 eps, 5000-step cap) | every episode: 1 legal hit, 1 wall return, then death; term causes: double_bounce 48, oob 2; mean length 261 of 5000 |
| Reward composition | shaping +1.74 (87.9% of net), wall +1.0, paddle +0.25, terminal penalty −1.0 → ≈ +2.0 |

The rollout CSVs at 250k/500k/750k/1M are behaviorally identical (hit at step
~70, wall at ~150–156, double-bounce out at ~259–273): the policy froze at
250k. 70–100% of plateau eval episodes end in the *penalized* double-bounce —
the −1.0 penalty is simply priced in against the ~+3.0 income of one cycle.

### Run 2 — `20260714_211111` (v0.9.0)

| Item | Value |
|---|---|
| Steps executed | 750,000 of 1,500,000 (early stop) |
| Eval reward | exactly −1.000 at all 30 evals (spread 2.5e-8 = float noise) |
| Paddle contact | eval 1 (25k): 0.2 hits/ep from the near-random policy; evals 2–30: **zero contact in 870 episodes** |
| Best model | step 250k, selected by a −0.9999999897 vs −1.0000000 tie-break; video shows a static paddle and a whiffed serve |
| Training reward | rose to ~1.8–2.2 (bands up to 12.4) on the recovery-reset distribution; 38% of late training episodes (the standard-serve ones) still failed |
| Long-horizon (50 eps) | 0 paddle hits, 0 wall contacts, 100% double_bounce at ~112 steps, return exactly −1.0 |

Training actively moved the policy *away* from ball contact as measured at
eval: the initial near-random policy interacted with the ball more than every
trained checkpoint.

## Root-cause findings

### Finding 1 (bug, confirmed): the curriculum schedule never reached the env

`LinearEnvAttrScheduleCallback._apply_current_value`
(`src/courtside_dynamics/callbacks/env_attr_schedule.py:106`) calls
`self.training_env.set_attr("recovery_reset_probability", value)`. SB3 2.9.0's
`DummyVecEnv.set_attr` does a plain `setattr` on the outermost wrapper — the
`Monitor` that `make_vec_env` always adds — and gymnasium 1.3.0's `Wrapper`
defines no `__setattr__` forwarding. The value becomes a shadow attribute on
the Monitor; `WallBallEnv`'s property setter (`wall_ball.py:511`) is never
invoked, and `reset()` keeps reading the constructor value. This was
**reproduced locally** under the exact library versions recorded in the run
config. Consequences:

- Run 2 trained at `recovery_reset_probability = 0.6` for all 750k steps; the
  documented 0.6 → 0.15 taper in `config.json` never physically happened.
  Only ~40% of training episodes started from the standard serve that eval
  (`recovery_reset_probability = 0.0`) measures.
- `get_attr` would read the wrapper's shadow copy, so even a diagnostic probe
  would have masked the bug, and the callback never logs its applied value.
- The unit test (`tests/test_env_attr_schedule.py`) uses a fake recording
  VecEnv, so the Monitor path is untested.

### Finding 2 (design): the second exchange is mostly physically unreachable

Measured with scripted rollouts against the eval configuration: after a real
completed return, the rebound's first bounce lands at x ≈ 2.1 (p10 0.74,
p90 3.06) with |y| up to 3.7, and — left untouched — its **second bounce lands
in front of the lane front x = −2.1 in ~73% of cases**. Only ~27% of post-wall
balls cross x = −2.1 airborne after their required bounce, at low height
(z ≈ 0.39). A scripted tracker+full-swing policy reproduces the Run-1 plateau
exactly (`bounce_count` {1: 35, 2: 5} over 40 episodes). The
`bounce_count >= 2` success bar is therefore luck-bounded, not skill-bounded,
under the current lane — and the solvability test
(`test_calibrated_baseline_serve_is_physically_solvable`) only asserts
`bounce_count >= 1`, *and* instantiates the env with
`recovery_reset_probability = 0.6`, so up to ~60% of its seeds don't even test
the serve.

### Finding 3 (design): the reward ladder is flat at −1.0 below the first completed cycle

Because the paddle bonus and tracking shaping are refundable "pending
advances" clawed back on any failed cycle or episode end
(`wall_ball.py:894-905, 1105-1110`), **every failure mode nets exactly −1.0**:
no-op = −1.000, random policy = −1.000 (200/200 episodes), touch-then-miss =
−1.000. The only signal distinguishing "touched the ball" from "did nothing"
is discount leakage of the temporarily-paid advance (γ^130 ≈ 0.52). On top of
that, in `one_bounce` style, touching the serve before its bounce is an
instant −1.0 termination (`paddle_before_bounce`, `wall_ball.py:968-973`);
Run 2's eval-1 policy hit style violations in 13.3% of episodes, i.e. early
contact attempts were punished *as hard as total passivity* — after which
contact stopped entirely. A random policy makes zero paddle contacts in 200
episodes, so SAC's critic sees a constant −1 return with nothing to climb.

Note what the claw-back does right: it makes shaping farming impossible
(verified — failed episodes net exactly −1.0 regardless of length). The
problem is not farming; it is that the first two rungs of the ladder (touch;
touch-and-return) are separated from "did nothing" only by discounting.

### Finding 4 (design): the one placement-gradient channel is inert

`recoverable_bounce_bonus` (Run 2's key addition) pays
`0.25 * clip(1 − |proj_y|/2.0, 0, 1)` at the first required bounce after a
legal return. Measured over 131 scripted completed returns: score mean 0.035,
**exactly zero in 76% of cases** (real rebound |y| runs up to 3.7 against the
2.0 hard clip), expected payment ≈ 0.009 per return — ~0.2% of per-cycle
reward. The one channel that grades the actual bottleneck skill (return
placement) provides essentially no gradient, and the zero-clipped region has
literally zero slope.

### Finding 5 (design): recovery fragments don't match real recovery states

`incoming_wall` fragments bounce at x ≈ −0.62 [−1.62, 0.28] with |vy| ≤ 0.4
and centered y; **real** post-return rebounds bounce at x ≈ 2.1 with |y| mean
1.9 (max 3.7). `post_bounce` fragments give ~30–45 steps of reaction time vs
~60–70 for a serve. So 60% of Run 2's training practiced intercepting balls
the policy rarely faces, while the balls it actually creates are mostly
unsaveable (Finding 2). The schedule also never tapers to 0.0 (end value
0.15), so even a bug-fixed run permanently trains on a distribution eval
never sees.

### Finding 6 (harness): model selection and early stopping ran on noise

- `info_dict_eval.py:434-441` compares raw float tuples with strict `>` and
  no epsilon, with `episode_reward_mean` as the final key. Run 2's "best" at
  250k won on ~1e-8 of shaping-telescoping residue; that single noise event
  reset the patience counter and extended a provably dead run from a would-be
  525k stop to 750k. Run 1's best (600k) beat the plateau by exactly one
  lucky 2-bounce episode in 30 (1/30 metric resolution).
- The docstring contract "at least 2N evaluations before a stop can fire"
  (`train.py:264-276`) is violated: `_evals_since_best` accrues during the
  warm-up, so the earliest stop is eval N+1 = 21.
- The schedule endpoint (750k = eval 30) coincides exactly with the earliest
  reachable stop under best-at-250k + patience 20, so by construction Run 2
  could never train at the final curriculum distribution.
- `success_rate` had zero resolution in both runs: saturated at 1.0 from 225k
  in Run 1 (threshold ≥ 1; this saturation was already documented from the
  20260712 run in commit `474c91e` before Run 1 launched) and pinned at 0.0
  in Run 2 (threshold ≥ 2). The only Run-2 metric that moved (final-eval
  episode length 114 → 205) is not used for selection.

### Finding 7 (SAC config)

- **Fixed `ent_coef = 0.02`** disables entropy auto-tuning (the recorded
  `target_entropy = −3.0` is inert for a float coefficient — verified against
  SB3 2.9.0). Both runs show terminal exploration failure: Run 1's training
  returns stayed bimodal {−1.0, ~1.99} for 3,500+ episodes; Run 2 stopped
  touching the ball. The pin was motivated by auto-α collapse (0.004/0.0005)
  measured on the *legacy 5-action env* and never re-validated on the
  3-action env.
- **`learning_starts = 100`** (SB3 default, never overridden): with 8 envs and
  `gradient_steps = -1`, batch-256 updates begin after ~104 stored
  transitions, sampled with replacement, while VecNormalize stats are fit on
  <2 episodes.
- **`buffer_size = 2M` never evicts** within a ≤1.5M-step run: transitions
  stored under early normalization statistics and under the 60% recovery-reset
  regime are replayed uniformly all run (at Run 2's stop, ~33% of buffer
  content predated 250k).
- **VecNormalize normalizes binary flags and counters** (obs dims 12
  `paddle_hit_since_last_wall`, 13 `floor_bounce_count`, 20 `stall_progress`,
  21 `pending_advance`, 22 `recoverable_bounce_eligible`) whose base rates
  drift with the curriculum; `normalize_obs_excluded_indices` exists and is
  used by HumanoidTennis recipes but not WallBall.
- UTD ratio (1.0), γ = 0.995, lr 3e-4, and critic health are all fine —
  optimizer metrics look nominal in both runs; this was not an optimization
  instability.

### Finding 8 (instrumentation)

- Monitor CSVs persist only `r,l,t` — no `info_keywords` — so training-time
  termination attribution is impossible post-hoc.
- The `*_ep_mean` eval aggregate is the mean of *final-step* values, so
  instantaneous flags read nonsense (`event_completed_return` ≡ 0.0 despite
  the events occurring); `rew_*_mean` per-step aggregates read 0 while
  `_final` is nonzero.
- `one_bounce_recovery_count` increments on every first qualified paddle hit
  (`wall_ball.py:863-864`), so it reads ~1.0 with zero rallies — misnamed and
  misleading.
- All 7 rollout CSVs and both videos share one identically-seeded serve
  (first bounce at step 49 in every file), and the video recorder stops at
  the first episode end (240/106 frames vs `video_length = 10000`) — each
  checkpoint's behavioral record is n = 1 fixed trajectory.
- `git_sha` is null in both configs (`artifacts.py:26-29` runs
  `git rev-parse HEAD` without `-C <repo>`, so it fails outside a checkout
  cwd). Run 1 provably ran on unreleased post-0.8.0 code (`1eec0d4`) while
  reporting "0.8.0" — the "0.8.0 vs 0.9.0" comparison framing in the
  artifacts is wrong.
- ~0.66 non-legal paddle contacts per episode (contact chatter within ~6
  steps of the legal hit) never shrink and are unexplained — worth a
  debounce/investigation.
- Run 1 → Run 2 changed five things at once (success threshold, obs dim,
  reward channel, reset distribution, lane front), so nothing about the
  regression is attributable from these two runs alone; both used a single
  seed.

## Recommendations

### P0 — fix before any re-run

1. **Fix the schedule callback** (`env_attr_schedule.py:106`): replace
   `self.training_env.set_attr(self.attr_name, value)` with
   `self.training_env.env_method("set_wrapper_attr", self.attr_name, value)`
   (gymnasium ≥ 1.0 walks the wrapper stack to the owning env), or set on
   `env.unwrapped`. Log the applied value each eval
   (`self.logger.record(f"schedule/{self.attr_name}", value)`) and log the
   realized per-rollout `reset_mode_id` mix so applied-vs-intended curriculum
   is auditable. Add an integration test that goes through
   `make_vec_env` + `Monitor` and asserts
   `env.unwrapped.recovery_reset_probability` actually changes — the current
   fake-VecEnv test cannot catch this.

2. **Make selection and early stopping noise-proof**
   (`info_dict_eval.py:434-450`, `train.py:905-924`):
   - Quantize selection keys before comparison (round `bounce_count_ep_mean`
     and rates to 1/n_eval_episodes; require improvement > ε) so 1e-8 noise
     cannot crown a best model or reset patience.
   - Drop `episode_reward_mean` from `best_metric_keys` (it is ~88% tracking
     shaping in Run 1 and pure penalty noise in Run 2); select on
     `(bounce_count_ep_mean, bounce_count_ep_ge_2_rate)` — both already
     logged.
   - Add a degenerate-signal guard: if all selection keys have zero variance
     over the last ~5 evals *and* the bottom competence rung
     (`paddle_hit_count_ep_mean`) is 0, stop early. Run 2 would have died at
     ~150k instead of 750k.
   - For recipes with `env_attr_schedules`, set
     `early_stop_min_evals ≥ ceil(max(end_timesteps)/eval_freq) + patience`
     (here 30 + 20 = 50) so patience only counts after the curriculum
     reaches its final distribution; and fix the "2N evaluations" warm-up so
     the implementation matches the documented contract.
   - Optionally confirm a candidate best with a second independent eval
     batch before overwriting `best_model.zip` (kills single-episode flukes
     like Run 1's 600k).

3. **Prove `bounce_count >= 2` is reachable before scoring runs against it.**
   Extend the oracle gate to demonstrate ≥ 2 consecutive returns from a
   standard serve with `recovery_reset_probability = 0` (the current test
   asserts only ≥ 1 and lets 60% of its seeds start from fragments). If the
   scripted oracle cannot do it, fix the geometry first:
   - Widen the paddle lane forward (`recipes.py` `paddle_x_target_range`
     front edge −2.1 → around −1.6 to −1.4, set by re-running the calibration
     sweep). The measured second-bounce distribution (p10 −3.19, p50 −1.35)
     means a moderate widening makes roughly half of rebounds reachable;
     going all the way to −1.0 would put the *serve's own first bounce*
     (x ∈ [−2.01, −1.50]) inside the lane and aggravate pre-bounce faults,
     so it must be paired with item 4 and re-calibrated. Note the
     recoverable-bounce projection plane derives from the lane front
     (`wall_ball.py:646-647`), so placement-score statistics change with it.
   - Alternatively/additionally, slow returns so rebounds land shallower:
     raise paddle slide-joint damping (`wall_ball.xml:30-37`, damping 5 → ~12;
     terminal velocity 100/5 = 20 m/s today, which permits unphysical
     16.6 m/s swings) or reduce actuator forcerange. Verify the scripted
     serve-return calibration (500/500) still passes.

4. **Give the pre-cycle rungs of the reward ladder nonzero net value**
   (`wall_ball.py`):
   - Soften `paddle_before_bounce` from instant −1.0 termination to a small
     non-terminal penalty (e.g. −0.25 in a new `rew_early_touch` channel,
     rally continues, gate stays closed). It currently makes early contact
     attempts exactly as costly as total passivity — the trap Run 2 fell
     into — and it deters the forward positioning the lane widening needs.
     Keep `floor_before_wall` terminal (it fires only on the agent's own
     outgoing shot).
   - Consider paying the first legal paddle hit outright (drop
     `_pending_bonus` for it) so touch-then-fail nets −0.75 instead of
     −1.00. **Caveat (history-informed):** an outright bonus + timeout
     claw-back exemption re-opens a juggle-to-truncation exploit of the
     same class as the 20260712 catch-and-stall policy that motivated
     refundable advances. If adopted: keep the bonus refundable on
     truncation (only non-refundable on failure terminations), watch
     `term_timeout` (currently 0 everywhere) as a tripwire, and revert if it
     grows without `bounce_count` growth.

### P1 — high value, ship with the next run

5. **Make the placement channel carry gradient**
   (`wall_ball.py:622-656`, `recipes.py`): replace the hard
   `clip(1 − |proj_y|/2.0, 0, 1)` with a smooth kernel (e.g.
   `exp(−|proj_y|/limit)`), raise `recoverable_bounce_lateral_limit`
    2.0 → 3.0 to match the measured |y| distribution, and raise the bonus
   0.25 → ~0.75 so a well-placed return earns reward commensurate with the
   +1.0 wall bonus instead of the current expected ~0.009. A depth factor
   (does the projected second bounce land inside the lane?) targets the
   bottleneck even more directly but requires new ballistic-projection code —
   worth doing, not free.
6. **Rebalance shaping vs events**: cut `track_shaping_scale` 0.5 → 0.2 (kept
   shaping is 87.9% of Run 1's net return; gradient mass rewards approach,
   not rallying) and make the wall reward escalate per exchange
   (`wall_ball.py:889`: `1.0 + 0.5 * (bounce_count − 1)`, capped ~3.0) so each
   additional exchange is worth strictly more than the last.
7. **Fix the curriculum distribution** (`wall_ball.py:1337-1361`,
   `recipes.py:379,404-411`): widen fragment `vy`/`y` ranges and move
   `post_bounce` fragment x toward the measured real-rebound footprint
   (bounce x ~2.1, |y| up to ~3.7); lower the starting share
   (`recovery_reset_probability` 0.6 → ~0.3) and **taper the schedule to
   0.0** (not 0.15) so late training matches eval. Keep the constructor value
   and the schedule `start_value` in sync — after the callback fix, the
   schedule's `start_value` overwrites the constructor at training start.
8. **SAC config** (`notebooks/sb3_training.ipynb` cell 5 / `train.py`):
   `ent_coef="auto_0.02"` with `target_entropy=−1.5` (keeps the safe init,
   restores adaptation; fall back to a fixed sweep {0.02, 0.05, 0.1} if α
   collapses again — check `train/ent_coef`); `learning_starts=10_000`;
   `buffer_size=500_000`; add
   `normalize_obs_excluded_indices=(12, 13, 20, 21, 22)` to the WallBall
   recipe (mechanism already exists in `SelectiveVecNormalize`).
9. **Add a curriculum-matched eval stream**: evaluate each checkpoint on the
   *current training* reset distribution alongside the canonical from-serve
   eval; the gap between the two curves is the transfer deficit, visible
   within 1–2 evals instead of post-mortem. Also add a stochastic-eval arm
   (`deterministic=False`) to distinguish "mean policy is degenerate" from
   "competence lives in the action noise" — Run 2's training success may
   have lived entirely in noise + recovery starts.

### P2 — hygiene and diagnostics

10. **Report the competence ladder, not one threshold**: log and plot paddle
    contact rate → legal hit rate → return-1 rate → return-2 rate (plus the
    survival thresholds already in v0.9.0). Selection stays on the graded
    keys from item 2; `success_threshold` remains a reported KPI only.
11. **Cut the redundant eval pass**: when headline selection is active,
    `EvalCallback` is reporting-only (`train.py:845-857`) yet still runs 30
    deterministic episodes every 25k steps; cut to ~5 episodes (or have
    InfoDictEval write `evaluations.npz`). Saves roughly 15% of run
    env-steps and removes the second, disagreeing "best" from reports.
    Consider raising the *selection* stream to 50 episodes.
12. **Statistical power and provenance**: ≥3 seeds per configuration; fix git
    SHA capture (`artifacts.py:26-29` needs `-C <package repo>` like
    `tennis_curriculum.py:884-885`, or embed the commit at install time);
    bump the package version on any behavior-changing env merge (Run 1 ran
    on unreleased code while reporting "0.8.0"); vary the video/rollout env
    seed and record ≥3 episodes per checkpoint (all current recordings are
    one identical serve); fix the recorder's `video_length` truncation.
13. **Instrumentation fixes**: pass
    `info_keywords=("term_*", "rew_*", "bounce_count", "reset_mode_id")`-style
    keys to Monitor; add per-episode *sum* aggregates for `rew_*` channels
    (the `_ep_mean` final-step semantics are misleading); rename or fix
    `one_bounce_recovery_count` (fires on first hit, not on recoveries);
    investigate the ~0.66/episode non-legal paddle-contact chatter (possible
    double-count or unpenalized double-hit); note the sign-flip refund oddity
    (a net-negative pending advance is refunded as *positive* reward on
    failure steps, `wall_ball.py:899,1106`).
14. **Training-only multi-serve mode** (optional, larger change): re-serve
    in-place after double_bounce/oob/stall instead of terminating (env kwarg,
    off in eval). Episodes currently use 112–273 of 750 steps, so the replay
    buffer is starved of exchange-2+ states; this multiplies rally-relevant
    transitions ~3–6× without touching eval semantics.

## Suggested experiment sequence

The Run 1 → Run 2 delta changed five variables at once and is unattributable.
Sequence the next runs to restore attribution (each step ~2–3h on the L4, and
with the flat-metric guard, dead runs die in minutes, not hours):

1. **A: bug-fix re-run** — v0.9.0 + callback fix + selection/early-stop fixes
   (items 1–2) + SAC config (item 8), everything else unchanged. This is the
   honest baseline for "does the curriculum, as actually designed, work?"
2. **B: no-curriculum ablation** — same as A with
   `recovery_reset_probability = 0` everywhere. Isolates the curriculum's
   net effect (Run 2 accidentally tested "always 0.6," which is neither).
3. **C: geometry** — A + lane widening + softened `paddle_before_bounce` +
   damping fix (items 3–4), gated on the extended oracle proving ≥ 2 returns
   is scriptable first.
4. **D: reward rebalance** — C + placement-channel and shaping/wall changes
   (items 5–6), then the fragment redesign (item 7) if the matched-eval gap
   (item 9) shows the curriculum still transferring poorly.

Run ≥3 seeds for whichever arm graduates to a "baseline" label, and compare
on `bounce_count_ep_mean` and the survival ladder — the only metrics
comparable across these configs.

## Addendum: geometry bundle calibration (2026-07-16)

The geometry bundle (recommendations 3–4) was calibrated with a damping ×
lane-front sweep (100 seeded standard-serve episodes per config, scripted
policies, `recovery_reset_probability = 0`) and is implemented on this
branch. Two results refine the original findings:

**Correction to Finding 2.** The packaged oracle — which retreats before the
bounce and intercepts at the lane front — already completes ≥ 2 returns from
**95%** of standard serves at the *old* geometry. The "~73% of second
exchanges physically unsaveable" measurement applies to *untouched* rebounds
from fast, flat returns; well-placed returns were always recoverable. The
real defect is that the old geometry is unforgiving of mediocre play: a
placement-blind track-then-full-swing policy (the behavioral archetype Run 1
actually learned) recovers a second exchange in **exactly 0%** of episodes
under the old geometry — there was nothing for RL to reinforce past exchange
one.

**Chosen configuration** — lane `(-3.2, -1.6)` + paddle slide damping 8
(new `paddle_joint_damping` env kwarg, baseline recipe only) +
`early_touch_penalty 0.25` (new env kwarg softening `paddle_before_bounce`
to a non-terminal fine):

| Config (front, damping) | Oracle ≥2 | Crude tracker ≥2 |
|---|---|---|
| (−2.1, 5) — old | 0.95 | **0.00** |
| (−1.8, 5) | 0.57 | — |
| (−2.1, 8) | 0.49 | — |
| (−1.8, 8) | 0.91 | 0.28 |
| **(−1.6, 8) — new** | **0.92** | **0.70** |
| (−1.4, 8) | 0.89 | 0.77 |

The two changes only work together (either alone collapses oracle second
returns to ~50%). (−1.4, 8) marginally beats (−1.6, 8) for crude play but
puts the *entire* serve-bounce footprint (x ∈ [−2.01, −1.50]) inside the
lane — a naive front-camper hits pre-bounce faults in 94% of episodes there
vs 39% at −1.6 — and drags the recoverable-bounce projection plane furthest
forward. Verified at the final recipe configuration: oracle 500/500 first
returns and 92% ≥ 2 (n=500); crude tracker 70.5% ≥ 2 (n=200); parked paddle
0 contacts and 0 returns (n=200), so the no-op ≤ 0 invariant holds. The
`bounce_count ≥ 2` success bar is now pinned by a solvability test that runs
from standard serves.

## Addendum 2: the bootstrap failure and the 2026-07-17 package

Two further runs on the fixed harness sharpened the diagnosis:

- **Run 20260717_025611** (v0.10.0, fragments 0.6 tapering): reproduced Run
  2's split — training reward rose on fragment mastery while the from-serve
  eval policy never touched the ball; the late-training failure rate matched
  the standard-serve reset share. Fragment skill does not transfer to serve
  receipt.
- **Run 20260717_040824 ("Arm B", fragments off)**: total bootstrap failure —
  zero paddle contact in *training* (reward −1.000 ± 0.000), killed by the
  degenerate guard at 125k steps in 22 minutes.

Calibration probes then rejected both intuitive curricula and found the real
barrier:

- **Depth ladder rejected**: difficulty in distance-to-wall is U-shaped.
  Close-court play collapses to ~50% oracle second returns (short-range
  returns hit the wall hot and rebound out of the court), and mid-court to
  baseline is flat.
- **Serve-pace ladder rejected**: return power rides on incoming momentum,
  so slow serves *underpower* returns (oracle first returns fall from 100%
  at speed 5.5 to 12% at 3.5).
- **The actual barrier**: a scripted y-tracker contacts 100% of serves but
  died 120/120 by terminal `floor_before_wall` at exactly the passivity
  return (−1) — and with every refundable advance clawed back on failure,
  *all* pre-scoring behaviors tie at −1. The track → touch → swing learning
  path crossed a dead-flat valley; SAC had no episode-return gradient toward
  contact at any point.

The 0.11.0 bootstrap package (implemented, `WallBallBootstrap` recipe) makes
the competence ladder strictly monotone at episode level for the first time
— stage-0 serve, n=120: parked −1.00 < weak-swing tracker −0.85 <
placement-blind full swing +7.63 < oracle +12.07 — via a once-per-episode
outright `first_hit_bonus` (0.25), a `weak_return_penalty` (0.1) fined retry
(with a fresh shaping window) replacing the terminal weak-return fault, the
already-shipped early-touch softening, a performance-gated serve-spread
ladder (`serve_vy_max` 1.1 → 2.0, nested distributions, gate:
`bounce_count_ep_mean ≥ 1.3` sustained 2 evals) with matched-stage selection
plus a canonical-serve `eval_info_final.csv` stream, and the never-tried
exploration configuration (`ent_coef="auto_0.02"`, `target_entropy=-1.5`,
`learning_starts=10k`, 500k buffer). The adversarial review of the bundle
caught and closed two high-severity holes before commit: repeat paddle taps
no longer reset the stall clock (touch-then-deaden possession measured a
risk-free +0.25 ride to truncation; it now nets −0.85), and a curriculum
stage advance resets the evaluator's best-model selection state (an
easy-stage score otherwise permanently bars better final-stage policies —
measured ~0.6–0.7 metric inflation between the narrowest and full serve).
Success metrics and the no-op invariant are unchanged (parked paddle still
scores 0 contacts and −1).

## Provenance notes

- Run 1's `config.json` reports v0.8.0 but the run required features merged
  in `1eec0d4` (2026-07-14 00:03 −0500), after the 0.8.0 bump; Run 2 matches
  `cdb17d4` (v0.9.0). Neither records a git SHA.
- No fixes postdated the runs at the time of this review (HEAD ==
  `cdb17d4`, 2026-07-16). The P0 harness fixes (recommendations 1–2, plus
  the git-SHA provenance fix from item 12) are implemented on this branch
  in the commit following this document; the env/reward/geometry changes
  (items 3–9) and the experiment sequence remain open.
- Analysis artifacts (decoded eval CSVs, npz, video frames) were regenerated
  from the Drive originals; per-row reward-channel sums reconstruct episode
  returns exactly (max residual ~5e-8) in all seven rollout CSVs, so the
  channel accounting above is exact, not approximate.
