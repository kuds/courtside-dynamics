# Lessons learned: WallBall training campaign (v0.9 – v0.14)

Distilled from nine training runs (`20260714_050506` through
`20260719_223139`), the baseline review (`wall_ball_baseline_review.md`),
and the engineering work around them. Each lesson records the evidence
and the operational rule this repo now follows because of it.

## Reward design

**1. When two opposite errors bracket the desired behavior, keep their
costs symmetric.**
Undershooting (weak return) and overshooting (OOB) both ended episodes
at net −1, which pushed the policy toward the committed middle. Making
undershooting a −0.1 non-terminal retry (0.13.0) made it 10× cheaper
than overshooting *and* kept the shaping stream alive — SAC rationally
converged to soft, unchainable returns (run `213222`: 82% double
bounce, 8% OOB, `recoverable_bounce_score` peak 0.44 vs ~0.99, plateau
at one exchange). *Rule: any penalty softening must be checked against
its opposite-error sibling; asymmetry is a steering wheel.*

**2. A flat reward valley is invisible until you measure competence
between the floors.**
With refundable advances clawed back, every pre-scoring behavior —
parked, tracking, touching, weak swinging — netted exactly −1.0. The
track→touch→swing path had zero gradient, and three runs sat in that
valley. *Rule: before training, score a ladder of scripted policies
(parked / tracker / crude swing / oracle) and require strict
monotonicity; ties at the bottom mean the valley is still there.*

**3. Scripted-ladder calibration validates reward ordering, not
learning dynamics.**
The weak-return retry passed its (bootstrap-context) calibration and
still produced a degenerate optimum in baseline training. The ladder
proves better play scores better *at fixed behavior*; it cannot reveal
which behavior gradient descent will find cheapest to improve. *Rule:
the ladder is a necessary pre-flight, and only a training run is a
verdict — budget one run per reward change.*

**4. Shaping should shrink as competence grows, or it becomes the
job.**
Even in the best run, tracking shaping was ~75% of the episode return;
returns pay +1 regardless of whether they set up the next exchange.
The double-bounce ceiling is the policy optimizing what it is actually
paid for. *Rule: pay the target behavior directly rather than paying
approach ever more — but see lesson 19 for what that did and did not
buy when tried (run `223139`).*

## Exploration and hyperparameters

**5. Compare the entropy budget against the reward scale per episode.**
Fixed `ent_coef=0.02` over a 750-step episode is worth tens of reward
units — an order of magnitude above the ±1 task signal — so "stay
random" was literally optimal, and the policy never touched the ball
(run `025611`). Auto-entropy collapsed the coefficient to ~0.0007 and
the identical setup learned the full task (run `165358`). *Rule:
prefer auto-tuned entropy; if pinning, budget `ent_coef × episode_len`
against the reward scale first.*

**6. Change one variable per run.**
The decisive entropy finding came from an *accidental* clean A/B (a
dropped `MODEL_KWARGS` cell). The 0.13.0 regression cost an extra
decomposition because gamma and the weak-return retry shipped
together. *Rule: recipe changes that alter training dynamics ship one
lever at a time, each with its own run.*

**7. Same seed is not same run on a GPU.**
Two seed-0 runs of an identical stack diverged within the first
gradient steps (CUDA nondeterminism) and differed by up to 1.7
bounces mid-training, while agreeing at the endpoint (3.23 = 3.23).
*Rule: treat < ~0.5 eval bounces as noise; compare endpoints and
long-horizon audits, not mid-curve snapshots; never claim replication
from seed identity.*

**8. Budget cures nothing after a plateau.**
Doubling the budget (1.5M→3M) moved the eval mean by +0.1 and the run
early-stopped; the tail improved instead (≥5-rally 8%→22%). *Rule:
when two runs agree on a ceiling, the next run changes the task or the
reward, not the step count.*

## Environment and calibration

**9. Calibrate learnability with weak scripted policies, not just an
oracle.**
The oracle returned 500/500 serves under the old geometry while a
placement-blind tracker recovered 0% of second exchanges — and
training matched the tracker, not the oracle. The damping×lane sweep
also showed paired parameters only work together. *Rule: feasibility
(oracle) and learnability (weak tracker) are different measurements;
sweep them jointly before training on new geometry.*

**10. Render-only must be provably render-only.**
Court markings and the tennis court are MuJoCo sites (cannot collide),
pinned by cross-style observation-equality tests, and the marked env
bit-reproduces a pre-marking run's eval. *Rule: cosmetic changes carry
a proof (mechanism + regression test), not an assurance.*

## Evaluation and selection

**11. Select on task metrics; reward is shaped and will lie.**
Eval reward was ~88% tracking shaping in one run; in another a ~1e-8
reward difference crowned a best model and reset early-stop patience.
Delta-tolerant lexicographic task keys + an independent confirmation
batch + a degenerate-signal guard fixed selection. *Rule: headline and
selection are task counts; reward is diagnostic only.*

**12. Keep the eval task frozen; keep tiebreakers alive.**
Strict, unchanged eval is what made a five-run comparison (and the
0.13.0 falsification) possible — training-side changes must never leak
into scoring (`eval_env_overrides` re-asserts, drift-tested). And a
saturated tiebreaker (`ge_2` at 100%) is a dead one — retire it for
the metric where headroom actually lives (`ge_5`). *Rule: eval
semantics change only with a documented metric-era break; audit
selection keys for saturation after every run.*

**13. The eval horizon caps what selection can see.**
The 750-step cap truncates exactly the deep rallies later runs
improved; the mean compresses while the tail moves. *Rule: pair the
capped eval with the 50-seed long-horizon audit — it is the sensitive
instrument for tail effects.*

## Engineering

**14. Silent no-ops are the default failure mode; make everything fail
loudly.**
The catalog: `VecEnv.set_attr` shadow-writing onto Monitor (a whole
run's curriculum never applied), stale notebook kwargs silently
replacing calibrated bundles, truthy `"false"` strings, sentinel
`None`s smuggled into non-Optional fields, explicit kwargs silently
beating config files. Every fix is the same shape: validate, raise
with the file/key named, suggest the fix. *Rule: a config knob that
can be ignored will eventually be ignored — wire it to raise.*

**15. Provenance must be automatic and byte-exact.**
Early runs recorded `git_sha: null` and archaeology was guesswork;
now config.json records the SHA, the TOML's path+hash+content, and the
run dir keeps a byte-exact `run_config.toml` copy. The 0.13.0
falsification was diagnosable in minutes because provenance said
exactly what ran. *Rule: if the artifact doesn't say what produced it,
the run didn't happen.*

**16. Lock documentation to code with drift tests — and mind their
blind spots.**
Starter TOMLs are pinned to recipes, notebook cells to contract tests.
The pattern works — and its one gap (the `[eval_env]` table the drift
test never compared) was exactly where an unnoticed edit could have
silently changed what evaluation measures. *Rule: a drift test guards
only what it compares; when adding a new configuration surface, extend
the drift test in the same commit.*

**17. Adversarial review before every commit pays for itself.**
Every multi-agent review round of this campaign found confirmed,
material defects that tests and types had passed: reward exploits
(touch-then-deaden), selection-state bugs, a workflow-crashing
notebook path, a silent PPO worker-count regression, the `[eval_env]`
blind spot. *Rule: substantive diffs get an adversarial find→verify
pass before merge; findings become regression tests, not just fixes.*

**18. Design → approve → implement → review → run → record the
verdict.**
The 0.13.0 recalibration was designed, reviewed, and still falsified —
which is the process working: strict eval caught it in one 5-hour run,
and the falsification is recorded in the same design doc that proposed
it. *Rule: designs carry a Status line that is updated with the
outcome, including failure; a falsified idea documented is worth more
than a quiet revert.*

## Addendum (v0.14, escalation experiment)

**19. Reward magnitude steers failure *style*; only capability moves
the ceiling.**
The escalating wall reward (run `223139`: n-th return pays
1 + (n−1)×0.5, scoring unchanged) worked exactly as calibrated and
changed the policy's character — double-bounce failures fell to 30%
(best on record; the policy commits to the next exchange) while
out-of-bounds rose to 58% (it now overhits) — yet the long-horizon
tail did not improve (2.80 completed returns, survival ≥5 at 10%, vs
the flat reference's 3.42 and 22%). Five healthy runs now bracket the
same ~3.2–3.4 eval-bounce ceiling across budgets (1.5M/3M) and
return-reward magnitudes. *Rule: when correctly-aimed incentive
changes only redistribute the failure taxonomy without raising the
ceiling, stop tuning rewards — the binding constraint is capability
(capacity, observations, actions, or task geometry), and the failure
mix becomes a style knob to revisit only after capability moves.*
