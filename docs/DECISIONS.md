# Decisions & Lessons Learned

A running journal of the non-obvious engineering decisions, bugs, and dead ends
behind Courtside Dynamics — the institutional memory that would otherwise be
scattered across review docs, commit messages, and version notes. When a design
choice, a calibration value, or a "we tried X and it failed" finding is worth
remembering, it goes here.

**How to use this file**

- Newest entries first, within each themed section.
- Each entry names the **decision or lesson**, the **evidence** behind it (with
  the concrete numbers that make it actionable), and a **Status**:
  *implemented* (shipped, with the version), *open* (recommended but not done),
  or *characteristic* (a property to be aware of, not a bug).
- The deep source material lives in the review snapshots under `docs/` (see
  [`docs/README.md`](README.md)); this file distills their durable conclusions.
  Per-release migration notes live in the root [`CHANGELOG.md`](../CHANGELOG.md).
- **Status caveat:** WallBall entries were tracked through implementation
  (0.9.0 → 0.13.0) and their statuses are current. The humanoid-tennis entries
  are from the v0.7.0 review; spot-checks against the current tree are noted
  inline, but they have **not** been fully re-audited — treat "open" there as
  "open as of 0.7.0, not since disproven."

---

## Cardinal rules

The meta-lessons that keep recurring across the project. Most specific entries
below are an instance of one of these.

1. **Fail loud, never silent.** This repo's history is a catalog of silent
   no-ops: a `VecEnv.set_attr` that wrote a shadow attribute instead of the real
   one (a whole training run at the wrong curriculum value), silent normalizer
   fallbacks that record garbage videos of a competent policy, an inert
   `clip_reward`, `git_sha` quietly `null`. Every config loader, callback, and
   schedule should raise or log its applied value, not swallow the failure. The
   run-config system ([spec](run_config_file_spec.md)) was built on this
   principle.

2. **Keep the reward ladder strictly monotone at the episode level.** The single
   most expensive failure mode here: every pre-scoring behavior — do nothing,
   touch and miss, weak return — netting *exactly* `-1.0` after claw-backs, so
   SAC saw a dead-flat valley with no episode-return gradient toward contact.
   Before launching a run, verify `parked < weak-swing < full-swing < oracle` as
   distinct returns (the 0.11.0 bootstrap package measures exactly this ladder).

3. **Prove solvability with a scripted oracle before scoring against a bar.**
   A `bounce_count >= 2` success threshold was treated as a skill gap for
   thousands of steps when part of it was geometry. Gate any new success bar on
   an oracle demonstrating it is reachable *from the evaluation distribution*
   (standard serve, `recovery_reset_probability = 0`).

4. **Make model-selection and early-stopping noise-proof.** A "best" checkpoint
   was once crowned on ~1e-8 of floating-point telescoping residue, which reset
   the patience counter and kept a provably-dead run alive for 500k extra steps.
   Quantize selection keys to `1/n_eval_episodes`, require improvement `> ε`,
   drop `episode_reward_mean` from the selection keys, and add a degenerate-signal
   guard (zero variance + zero paddle contacts ⇒ stop).

5. **Don't carry hyperparameters across an action-space or physics change.** The
   fixed `ent_coef` that "worked" was tuned on the legacy 5-action WallBall; on
   the 3-action env it disabled auto-entropy and helped block learning entirely.
   Re-validate exploration/entropy config whenever the action space changes.

6. **Match curriculum fragments to the distribution the policy actually
   creates/faces.** 60% of one run trained on synthetic "recovery" balls
   (bounce x ≈ −0.6, centered y) the policy almost never faces, while the balls
   it *does* create (bounce x ≈ 2.1, |y| up to 3.7) were barely practiced.
   Fragment skill did not transfer to serve receipt at all.

7. **Isolate side-effectful callbacks so they cannot kill a run.** A missing
   headless-GL backend made the first scheduled video recording raise
   `mujoco.FatalError` out of `model.learn()`, losing `final_model.zip`, the
   final eval, and `stage_summary.txt` ~45 min in. Recording (and any optional
   diagnostic) must be try/except log-and-continue.

8. **Change one variable at a time, and run ≥3 seeds before calling something a
   baseline.** One regression changed five things at once (success threshold,
   obs dim, reward channel, reset distribution, lane front) on a single seed —
   nothing about it was attributable.

---

## PaddleTennis — the rally campaign

Source: the dated `paddle_tennis_*` docs and `design_paddle_tennis_*`
verdict sections indexed in [`docs/README.md`](README.md); this
section distills the campaign's era verdicts (2026-08-15 → 09-02;
the earlier exploration/diagnosis-era verdicts remain in their
source docs' status lines).
The P0–P2 court-scaling entry predates the campaign and stays in the
WallBall section below.

### The instrument batch: select on the policy's own k≥2 conversions, evaluate paired, keep the task bit-identical — *implemented (2026-10-05)*
The 2026-10-05 review (`repo_review_20261005.md` §5b, §7) found the
task sound but the instruments that judge, select and stop a run
broken. Examples:
- success and the dead-run guard read only the last, truncation-cut
  point;
- one 0.25 min-delta hid a +20 pp success gain;
- selection keyed on opponent-dominated `crossings`;
- `reset(seed)` was not reproducible;
- warm starts checked only observation *shapes*.

The batch was then put through an adversarial review: six lenses,
each finding verified by execution, 25 fixes. The decisions worth
remembering:

- **The task stays bit-identical; only info and selection move.**
  Per-step obs, reward, terminated and truncated digests match the
  pre-batch env over random and oracle play, for the default and
  recipe kwargs, seeded and unseeded resets, and drill arms. Old
  checkpoints, normalizers and the LD1′ demo library stay loadable.
  `best_model` picks and eval curves are not comparable across the
  batch.
- **Selection and success follow `episode_rally_returns_a`.** This
  counts the policy's confirmed returns after its first in each point,
  counted from the point's *launch*, so a full-context drill's restored
  rules count cannot inflate it.
  - The min-deltas are 0.5/30 on the headline and `success_rate`, so
    one conversion in 30 episodes registers (0.05 could not), and
    0.25 on reward.
  - The guard reads legal hits and requires only the headline to be
    flat, so a run learning k=1 is never killed.
  - **LD1′ consequence:** its design's §6 claim that "selection keeps
    its meaning" no longer holds. Its gate scores a checkpoint
    selected on k≥2 conversions.
- **Paired evaluation is opt-in.** Only recipes that set
  `eval_reset_options` (PaddleTennis) derive a paired seed block at
  `seed + 1_000_000`, with the confirmation batch at +100,000.
  - Each block's best is compared only with its own block.
  - Turning pairing on for every seeded run would have served every
    humanoid evaluation from side A, and made the WallBall long-horizon
    "held-out" audit replay its own selection seeds. That audit now
    refuses overlapping seeds.
  - The final-info-eval stream stays fresh-random, per "Unpaired
    evaluation is the root of the gate noise" above.
- **Truncation clawback is kept (decided, not fixed).** The escrows
  claw back at time-limit truncation, and SB3 also bootstraps through
  truncation, so the pending amount is counted twice.
  - The effect is at most one event per 1500-step episode, about
    1.7e-4 per step of expected bias.
  - Removing it would break the escrow's undiscounted net-zero
    identity, which the S2 tests pin.
  - Revisit only if the clock leaves the observation and the clawback
    shows up in value diagnostics.
- **A net cord ends the rally (deliberate drill rule).**
  `ball_net_is_fault=True` ends 86% of oracle one-point rallies; 20–34%
  of those would be play-on in tennis. The rigid tape is unrealistic,
  popping clipped balls up at 60–93°. The rule teaches net clearance,
  so it stays until a compliant tape exists (baseline era).
- **Spin frame and raw counters wait for the observation profile.**
  Ball spin is body-frame and `rally_count`/`bounce_count` are raw.
  Fixing either in place would silently re-mean every 48-dim
  checkpoint. They ship with the context-blind
  `observation_profile`, whose different shape and new
  observation-name fingerprint refuse old checkpoints loudly.
- **Seed blocks.** Paired evaluation seeds are ≥ 1,000,000, clear of
  every ledger block. `tools/_seed_ledger.py` is the single ledger.
  - It adds the consumed k=2 scratch blocks 9030–9099, 9148–9167 and
    9168–9187.
  - It keeps 6400–6499 reserved, because D-G re-booked it after its
    08-30 release.
  - `k2_harvest` keeps 9030–9099 reproducible via its allowance.
  - Reproducing the LH1c cross-check library (9148–) now needs an
    explicit allowance.

### The pure-drill pilot is retired before freeze on its own gates; the campaign routes injection-first — *booked (2026-09-02); LD1′*
Three maintainer-blessed Phase 0 gates
([`paddle_tennis_ld1prime_freeze_brief_20260830.md`](paddle_tennis_ld1prime_freeze_brief_20260830.md),
each verified) settled the k=2 routing on measurement rather than
prediction: the registered checkpoint's critics are **near
action-blind at the k=2 failure states** (the oracle-vs-policy Q
gap sits inside the ~0.02 range that *uniform-random* actions span —
a Q-filtered BC term cannot engage at launch); **critic penultimate
dormancy holds on the replay-like training stream** (55/59% dormant,
41/44% exactly zero, vs <5%/~0% for fresh nets of the same
architecture — the recycling prescription passes the gate the
diagnostics note demanded); and the **stochastic behavior policy
reproduces the deterministic step-0 engagement rows in both drill
arms** (feed 6.5%, full 1.3%), so gSDE jitter does not rescue the
drill's engagement premise. Booked: (D-A) LD1-as-drafted RETIRED
pre-freeze, injection-first LD1′
([`design_paddle_tennis_demo_injection.md`](design_paddle_tennis_demo_injection.md));
(D-B) critic head-recycling at fine-tune start, full re-init only as
escalation, LayerNorm dropped; (D-C) BC term unfiltered at launch,
Q-filter armed by a pre-registered ordering measurement; (D-D) no
temperature re-heat — the T1-FAIL era law binds, ent_coef and
saturation become observables; (D-E) demos are oracle completions
from the policy's OWN harvested failure states, harvested through
the conversion's confirmation so the conversion payment is in the
buffer, with the post-confirmation rally recorded to the point's
end, the episode's truncation, or a 300-step cap, whichever comes
first (wording **amended and booked 2026-09-05**; as first booked it read "through point
termination", which the measured harvest does not meet literally —
design §3);
(D-F) drill OFF at launch, arm (b) the pre-registered RE-AIM
escalation, the D2 fork deferred; (D-G) battery block 6400–6499,
scratch extension 9200–9299, and the pre-pilot instrument list —
the injection fork with its gradient_steps regression test, the
model-side provenance hook and plan key, the SD0 certificate, and
the demo harvest tool shipped default-off the same day (an
adversarial review of that surface found the booked pilot could
not launch through the warm-start algo gates and that the first
demo library carried a fresh-episode clock and a mislabeled cap —
the material findings fixed and the library re-harvested; the
first library's table (142 kept, "0 censored") had gone into the
design's §3 the same day and is superseded; a second verification
pass's residual minors are recorded in the design, including the
amendment of D-E's "through point termination" wording booked
2026-09-05, since most kept demos end at the recording cap while
the payment clause holds by construction); the class-split
diagnosis extension is the outstanding prerequisite.
Lessons booked: (a) **the
pre-freeze diagnostics' first-draft headlines were wrong in four
places and the verification caught every one** — the "collapsed
gSDE noise" was a 30× misread of the exploration-matrix scale (the
true marginal noise is ~0.60 and never collapsed), "sane
calibration" was a systematic −0.46 pessimism, the per-feature
context-gate ranking was non-decomposable, and "arm (c) decisively
negative" was statistically indeterminate — a freeze input is not
a freeze input until it survives its own adversarial pass; (b) **a
recommendation red-team is cheap and found blocking flaws** in the
first composite proposal (an unimplementable actor-only transfer,
a LayerNorm critic the warm-start guard refuses, a demo harvest with
the exact distribution mismatch the proposal used to reject arm
(c), and a temperature component the campaign had already booked
as spent) — the corrected plan is a single lever with pre-named
escalations, which is what one-lever discipline demanded all along;
(c) **measure engagement in the mode the learner actually runs**
(behavior policy, replay-like stream) before pricing a pilot on
eval-mode rows.

### The command-rate design dies on measurement before a line of code: k=2 is a directed-interception gap, not a thrash gap — *closed without implementation (booked 2026-08-30); PT2*
The frozen limiter design's premise was falsified by a
pre-implementation probe
([`paddle_tennis_command_spectrum_20260830.md`](paddle_tennis_command_spectrum_20260830.md),
one Colab CPU session, every headline independently re-derived
twice): the measured thrash is heavy-tailed **jump-then-dwell**
(p50 per-step command delta ≤ 0.033 m, p99 3.4–5.3 m), not the
per-step alternation Δ = 0.15 was priced against — no
travel-reduction mechanism in evidence at the frozen candidate,
and the Δ small enough to bite (≤ 0.05, sub-plant) ramps the
oracle's own strike jumps (p99 1.76 m). Decisively: on measured
k=2 opportunities (~1.3–1.7 per episode — **opportunity is not
scarce**) the trained paddle sits ~3 m from the return's bounce
and converts ~1%, the oracle sits 0.89 m and converts 98% — and
the **frozen-at-hit counterfactual is no better than the thrash**
(≤ 1 m on 1.5–6%), while the oracle's own frozen counterfactual
collapses to 39%: its k=2 mechanism is *directed recovery toward
the return's landing point*, which removing motion cannot produce.
Lessons booked: (a) **price a mechanism against the measured
spectrum, not the assumed one** — the design's CR2 witness
(symmetric per-step bang-bang) was the one spectrum its limiter
trivially defeats and could never have caught this; (b) **a
cheap falsification probe in front of implementation is worth
running whenever the mechanism's premise is measurable from
recorded data** — this one cost hours and saved an
implementation, a battery, a 1M pilot, and an era break; (c) the
campaign's target is now a number: **paddle → return-bounce
~3 m → ~1 m**. Seed block 6400–6499 returned to the pool
unconsumed. Routing: the k=2 drill curriculum is taken up first
(maintainer direction, 2026-08-30), demonstration/buffer
injection the named alternative.

### The temperature-skip lever is spent: LT1 fails every bar and the campaign routes to the interface side — *adjudicated (2026-08-29, run 20260828_121324); command-rate design frozen*
The pre-registered temperature-skip pilot
([`paddle_tennis_lt1_prereg_20260823.md`](paddle_tennis_lt1_prereg_20260823.md)
§4b): warm-started from the registered 2.4M best with
`transfer_log_ent_coef=False`, the fresh α = 0.02 bought only a
**~11k-step live window** (below 1e-3 at 36k, inside the predicted
10–20k band) before re-collapsing to the source's inherited value
(1.474e-4 vs 1.589e-4). **KT1 FAIL** (k=2 ≤ 1% at every checkpoint,
either parity), **T1 FAIL** (strike-ended post-swing saturation
never < 85% — minimum 86.0% vs the 87.6% source anchor, scored by
replaying all ten 100k checkpoints through the PT1 instrument),
**R1 FAIL** (k=1 receiving peaks at 57%). The frozen rule's T1-FAIL
branch booked: **the optimizer-side lever is spent — no further
temperature retry** — and
[`design_paddle_tennis_command_rate.md`](design_paddle_tennis_command_rate.md)
was frozen the same day (Δ = 0.15 rate limit, CR0–CR3 battery, CR2
block 6400–6499 assigned; LC1's final freeze — shape confirmation
and bars — deferred to after the battery). Two observations sharpen
the routing. The 3M same-seed companion's first 1M **reproduced the
registered run byte-identically on all ten diagnosis rows** (GPU
nondeterminism produced no measurable divergence — this rerun was a
reproduction, not a replicate, exactly as the prereg's §4a
cautioned: zero independent evidence). And its 1M–3M tail is the
campaign's core dissociation at its sharpest: k=1 receiving climbed
monotonically to **90% at 3M while k=2 never exceeded 1%** in
twenty rows — first-exchange mastery is fully learnable under a
collapsed temperature and buys nothing at the second exchange,
consistent with the blocker being neither shot quality nor budget.
Era law extended: **restoring α buys a window, not a regime** —
under appendix D.6's tanh-saturation mechanism, a warm start that
begins ~88% saturated re-collapses a fresh auto-tuned temperature
(confirmed at α = 0.02; a fixed α would be a new design with its
own document, per the prereg's §4); treating the temperature
without unsaturating the head is spent as a lever.

### The review snapshot corrects the closing record; verdicts stand — *booked (2026-08-23)*
Independent re-derivation of every closing number
([`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md)):
RK1/KH1 FAIL and the hold-line closure all reproduce, but §4c's
extension observation had missed that k=2 receiving read **5% at the
2.5M probe** — the era's strongest reading, at the PASS-bar values,
booked as noise floor. Fresh-seed re-measurement: 1.6% [0.3, 4.5]
(real but small, seed-conditioned at its peak, not via stillness —
thrash unchanged at 92% saturation, hold dose +0.211 paid and all
clawed back in the audited trace). Also measured: **MuJoCo
3.11 ≡ 3.12 on this task** (digit-exact deterministic replays), so
LH1c's record eval belongs to the run, not the physics; and
task-metric selection does not track k=2 — the crowned best (2.325M)
is not the k=2-richest checkpoint (2.5M). Lesson 13a bit again in
draft (caught by the review's own verification pass, commit
`ec0606a`): the first draft's pooled estimate blended the seeds that
nominated the checkpoint with the fresh ones; only the fresh arm
scores.

### The post-swing wander is commanded thrash, not drift — *diagnosis (PT1, 2026-08-22)*
`data.ctrl` replay of the best checkpoints: commanded XY path 3–4×
the paddle's actual travel (31.0 vs 7.8 m; oracle ratio 1.0), 88–91%
of post-swing steps saturated (|action| > 0.9; the 2026-08-23
review's 2.5M row extends the band to 93%), 0.26–0.38 m/step
command saccades, no attractor (≈4 m from hit point and home, 5–6 m
from the ball). The dynamics-side hypothesis is dead; the action head is
emitting bang-bang under a dead temperature (ent_coef ~1.7e-4, std
~0.014). Era law refined: **paid windows organize only where
exploration can find the paying behavior** — the hold escrow needed
~80 consecutive near-zero commands from a saturated head with no
noise left to discover stillness. Routes: temperature-skip warm
start first, an interface-side command-rate/low-pass treatment
second (a new comparability era), no further escrow scale.

### The hold-escrow line closes without adoption — and a shaping term must pay where the policy already is — *closed (2026-08-22)*
Three pilots (`design_paddle_tennis_postswing_hold.md` §4–§4c): LH1
(0.25, 4.0 m) and LH1b (0.5, 4.0 m) delivered **exactly zero
gradient** — the 4.0 m travel budget was a cliff entirely outside
the policy's 6–10 m band, proven by LH1b training byte-identical to
LH1 (doubling a zero); the design error is recorded plainly: choosing
the budget so the measured wander "pays zero" is precisely what makes
it unlearnable. LH1c (0.5, 12.0 m — ramp moved onto the occupied
band) delivered the dose and the policy kept the wander (registered
1M window: KH1 FAIL, hold travel back to 8.1–8.3 m). Honest
counterpoint, unregistered extension: campaign-record eval **+2.483
at 2.425M**, the first warm-started run whose best came from deep
training, k=1 receiving to 90–98%, and (per the 2026-08-23 erratum)
a small real late k=2 rise by a non-hold route. Kwargs stay in the
env, default off, certified farm-proof.

### The registered n-point run: from-scratch bootstrapping proven, k=2 is the standing blocker — *adjudicated (2026-08-22, run 20260816_235141)*
LS1 PASS (the adopted task trains from scratch — the campaign no
longer depends on the transfer lineage), RE3 PASS (k=1 receiving to
95%), M PASS (77.5% of evals), RE1/RS2 middle, **RK1 FAIL** (k=2
≤ 1% at every checkpoint — 1% on eleven, 0% on nineteen) → the
stop/amend branch fired into the hold-escrow line. R2 watch read
7.5–9.1 m all run: the post-swing window was the one unpaid segment
of the k=2 chain. Held-out block 4100–4199 not opened; stays sealed
for the first registered-result branch.

### Reach shaping fires the ADOPT branch; the recipe enters the n-point era — *implemented + adopted (2026-08-16)*
LR1 (escrowed reach at 0.25, radius 3.0, warm-started): first
positive learned final eval (+0.14), k=2 above zero at five
checkpoints, touch 41%, ready error 1.08 m. Recipe defaults since
the adoption commit: `points_per_episode=None` (continuous n-point),
`contact_shaping=0.25`, `reach_shaping=0.25`, side-A-only
success/guards (`legal_hit_count_a`, `best_metric_min_delta=0.25`,
`confirm_best_eval`, 5-eval degenerate stop). Env defaults stay
frozen (everything default-off at env level).

### The n-point L2 verdict: continuous play un-samples contact from scratch; transfer collects the credit — *stop/pivot (2026-08-15)*
Both from-scratch n-point pilots collapsed to zero contact (3.1M
cumulative steps, one shot in bounds, ever). The validation battery
localized it: the reward pays competence (oracle +11.6 vs statue
−4.4 in-mode) but contact is **undiscoverable by exploration noise
at any std** (only full-range held-64 uniform touches, ~1/10k
steps), noise is locally punished, so entropy collapse was the
correct local gradient; the volley trap runs 3–10× (ball-chasing
toucher −52/episode). L2W then proved the design's inter-point
credit mechanism by transfer: the k=1-mastered checkpoint scored
+1.92 with **zero gradient updates** — and training regressed it
(k=1 100% → ~30% equilibrium). Chronic α-collapse is on the record
here too: 0.02 → 9.8e-4 inside 12k steps; appendix D.6's
tanh-saturation mechanism (saturated squashed mean inflates latent
−log π, annealing α to zero) empirically falsifies the target −1.5
rationale. At the verdict there was no warm-start temperature-skip
flag in code; it shipped 2026-08-23 as
`WarmStartConfig.transfer_log_ent_coef`, with LT1
(`paddle_tennis_lt1_prereg_20260823.md`) as its pre-registered
pilot (frozen 2026-08-28).

## WallBall — reward, curriculum & geometry

Source: [`wall_ball_baseline_review.md`](wall_ball_baseline_review.md) (runs from
2026-07-14, reviewed at `cdb17d4`/v0.9.0, with addenda through the 0.11.0
package). See also `CHANGELOG.md` 0.9.0 → 0.13.0. Depth-campaign entries
below distill [`wall_ball_rally_diagnosis_20260728_review.md`](wall_ball_rally_diagnosis_20260728_review.md).

### PaddleTennis P0–P2: premise measured, court scaled to paddle physics, loft is the frontier — *probe result (2026-08-02)*
The pivot's premise is now a measurement, not an argument: on a
two-paddle prototype (wall-ball calibrations verbatim, net at x=0),
in-rally strokes land **mean 3.3 m past the net, 50% deeper than
3 m** — the depth distribution the wall never produced. Two design
facts came out non-obvious: (1) the **full-scale ITF court is
infeasible** at the paddles' ~12.5 m/s ceiling (nothing clears from
11.9 m at any net height) — the era's court freezes at half-length
6.5 m with the regulation 0.914 m net, placing baseline strokes
exactly at the power ceiling; (2) with a fixed-pitch face, **depth
control spans a knife's edge** (hard ↔ long, soft ↔ net), and a
strike-height offset (~0.12 m below ball center) measurably widens
it — the first quantitative sign that paddle-pitch actuation may
graduate from parked consistency lever to era prerequisite, a
question for the first learned runs. Scripted reference band:
2.0 crossings/point, max 10, ≥4 in 33% (bimodal — half of points die
on serve-return, the wall-ball pattern repeating). P3–P5 still gate
env code. Details:
[`paddle_tennis_probes_20260802.md`](paddle_tennis_probes_20260802.md).

### The replication splits — reliability 2/2, era skill 1/2 — and the volley loophole closes the wall-ball chapter — *result (run 20260801_144043); chapter closed*
Seed 1 ran its full 6M budget, out-scored seed 0 everywhere (best
window 3.072 — the first ≥3.0 window in campaign history, essentially
uncompressed; uncapped audit 3.08; champion banked at literally step
6,000,000 and confirmed at 3.417, the project's highest), and **failed
the deep-receive criterion completely**: measured first contact mean
x = −4.49 (bar ≤ −6.0), 0% of episodes deep. Mechanism: in 48% of
episodes it *volleys the serve out of the air* at −3.4 before the
ball bounces; the rest intercept the post-bounce leg early at −5.5.
Two seeds, two coherent opposite styles — and the style that refuses
the era's skill scores higher. Training reliability is settled at 2/2
(no basin, no collapse, 384 evals). The durable lesson: **an
environment's dominant strategy is set by its physics, not by reward
or rule decree** — the wall only hits short (rebounds mean −1.0,
never deeper than −6.1), so depth can never be instrumentally useful
against it, and every decree that forces it (fences, serve energy,
style bans) invites the next exploit. Decision: the 0.26.0
loophole-closing and episode-cap changes are **rejected as moot**, the
wall-ball chapter closes with its deliverables banked (calibrated
physics, the probes-first methodology, 2/2 unattended training
safety, two stylistically opposite champions), and the campaign moves
to opponent play, where deep balls exist because someone hits them:
[`design_paddle_tennis.md`](design_paddle_tennis.md). Details:
[`wall_ball_true_baseline_replication_20260801_review.md`](wall_ball_true_baseline_replication_20260801_review.md).

### The true-baseline era opens on the first run — and the episode cap becomes the measured ceiling — *result (run 20260731_132322)*
The first `WallBallTrueBaseline` GPU run passed every pre-registered
primary criterion: best window 2.856 (bar 2.0), uncapped audit **3.02
mean completed returns** (100% of episodes ≥1, 98% ≥2, max 9), and a
measured mean first contact of **x = −6.49** (bar ≤ −6.0; 98% of
episodes deeper) — 2.8 m deeper than the goal era, with zero opening
volleys. Band-relative it is the project's strongest policy (1.53×
its oracle band vs 1.31–1.39× for the goal-era seeds). The champion
banked at 2.1M; early stop at 3.6M with the final policy off-peak
(1.98) — best-model banking again saved the result. The stretch (a
≥3.0 window) was NOT met, and three independent observations pin the
cause on the 750-step episode cap, not the policy: eval max rallies
sat at *exactly* 5 for ~120 consecutive evals (750 ÷ 156-step cadence
= 4.8 exchanges), timeout terminations tracked the ≥5-rate
one-for-one on strong evals, and 14% of uncapped audit episodes
outran the cap (longest 1,487 steps, 9 returns). Lesson for future
eras: when cadence changes, the episode cap silently rescales the
headline metric's ceiling — re-derive `episode_len` from cadence
(target ~6 exchanges) whenever geometry moves. `episode_len` 750 →
~1,100 is the pre-registered 0.26.0 candidate, to land between runs;
the seed-1 replication runs first, unchanged. Details:
[`wall_ball_true_baseline_20260731_132322_review.md`](wall_ball_true_baseline_20260731_132322_review.md).

### The true-baseline era task is probe-frozen and certified before any run — *implemented (0.25.0)*
The extension to the ITF baseline (workspace (-8.2, 0.3), fence
(-8.2, -2.6), start -7.9, home -5.4, serve 11.0, in-play bound -10.0 —
`WallBallTrueBaseline`) was measured before it was built, and three of
the measurements overturned the "obvious" design:
- **Serve speed is the depth lever, and it has a floor.** Landing
  scales ~0.5 m per +1 m/s from origin 1.0; below 10 m/s the ball dies
  short of a paddle at -7.9, so the goal-era serve (7.0) is
  unreturnable there because it *never arrives*. At 11.0 the
  post-bounce leg crosses -7.9 at 0.54 m and 9.4 m/s in 100% of
  episodes — reachable, and (from -7.9) impossible to intercept
  pre-bounce, so the deep receive is forced without any fence trick.
- **A baseline-only fence is geometrically unsustainable.** Rebounds
  off legal deep returns land at mean x = -1.0 and are *never* deeper
  than -6.1 — the rally always comes forward. The era fence adds the
  deep band to the goal fence instead of relocating play.
- **The in-play bound is task geometry, not a constant.** The
  historical OOB edge (-6.0) sits 1.9 m in front of the new paddle
  start: the serve terminates out-of-bounds before the receiver can
  touch it (measured 60/60 dead serves). `ball_in_play_min_x` is now a
  per-task kwarg (default -6.0, bit-for-bit historical), -10.0 here.
- The certification feasibility floor became a spec knob
  (`feasibility_ge2_floor`, default 0.90): no scripted reference
  dominates an 11.8 m court (oracle band mean 1.98, >=2 rate 67%,
  cadence 156 steps), so the recipe declares 0.50 = band minus two
  30-episode sampling sd, stamped into every report. Held-out
  certification on fresh seeds 4000-4099 passes every blocking
  criterion (oracle 61%, CI [0.51, 0.70]). Probe-design lesson en
  route: score the serve by *crossing-state kinematics* at the receive
  depth, not by arrival at a parked face — the post-bounce hop
  (apex ~0.6 m) passes under a parked face (bottom 0.95 m) and reads
  as a false 0% reachable.
  Details: [`design_wall_ball_true_baseline.md`](design_wall_ball_true_baseline.md).

### The default action mapping must never be derived from the XML — *implemented (0.25.0)*
Widening `paddle_slide_x`/ctrlrange for the workspace extension
silently rescaled the **default** x action mapping (action -1 went
from world -4.7 to -8.2) for any env built without an explicit
`paddle_x_target_range` — caught by the bit-for-bit default-config
guard test, which exists for exactly this. The default mapping is now
a frozen module constant (`_DEFAULT_PADDLE_X_TARGET_RANGE =
(-4.7, 0.3)`); the extended workspace is opt-in per recipe. Verified
by hashing full obs/reward/termination streams across HEAD vs the
extended tree on four legacy configs — all bitwise identical. The
general rule: physical capability (XML ranges) and action semantics
(the mapping) are separate contracts; growing the first must never
move the second implicitly.

### The campaign goal is met: a sustained rally from the workspace baseline — *result (run 20260728_225217); replicated 2026-07-30*
**Replication update (runs 20260729_140112 / 20260730_005134):** 2 of 3
seeds pass, and the third success beats the first everywhere (best window
3.750 vs 3.311; audit 3.76 mean / median 4 / max 11 / 32% >=5-survival;
84% of audit episodes now end with the ball running out AFTER a rally).
Seed 1 exposed a new failure mode — one-and-done basin capture for 1.4M
steps, unaided escape to 1.45, then full collapse to zero paddle contact
at ~2.5M, ended by the degenerate-signal guard at 3.175M with the
pre-collapse champion banked. All measured policies play DEEP (contact-x
-3.4 to -3.95; nobody camps the fence front), settling that the
fixed-pitch face suffices for deep rally play. Reliability (~2/3), not
capability, is the recipe's open item. Details:
[`wall_ball_goal_rally_replication_20260730_review.md`](wall_ball_goal_rally_replication_20260730_review.md).

Original seed-0 entry follows.
The first `WallBallGoalRally` run resolved the campaign: goal-task eval
sustained `bounce_count_ep_mean ≥ 3.0` over 3×60-episode windows from 1.25M
steps (best window 3.311; best model 3.333 confirmed 3.267 at 1.325M), and
the 50-seed long-horizon audit reads **3.54 mean completed returns** (median
3, p90 6, max 8; ≥5-survival 26%; 100% of episodes complete ≥1 return; zero
opening volleys, 95% post-bounce play) — versus 1.14 and
every-episode-double-bounce for the best ladder policy on the identical
task. Every pre-registered success criterion passed including the stretch
bar; early stop ended the run at 1.825M of 6M (~6.6 h) on a settled plateau.
The bar that five curriculum runs treated as an unreachable promotion gate
was a reachable performance level once training happened on the task itself.
Caveats: seed 0, n=1 (second seed is the standing next step); plateau
~3.0–3.3 with the ≥5 tail at 26% — consistency, not competence, is the
remaining headroom on this geometry. Snapshot:
[`wall_ball_goal_rally_20260728_225217_review.md`](wall_ball_goal_rally_20260728_225217_review.md).

### The sliding-fence depth ladder is retired; the goal task is trained directly — *implemented (0.24.0); vindicated by run 20260728_225217*
Three ladder generations (0.15/0.19-0.21/0.22) stalled the same way: a flat
`bounce_count_ep_mean >= 3.0` bar that **no scripted reference reaches at any
rung** (~70 controller configurations swept in the diagnosis, best 2.70;
held-out oracle means 2.47-2.89) and that four multi-million-step learned
runs plateaued 0.2-0.9 below at every rung past stage 0 (0.22.0 stage 1:
201 evals, best 2.82, still climbing at 6M; 0.19-aligned stage 2: 0 of 128
evals >= 2.9). Meanwhile each promotion moved the serve landing +0.35 m —
above the measured ~0.25 m approach-room threshold — so every rung swapped
the receive task at the exact moment the 0.22.0 advance package deleted the
buffer and re-randomized the policy (measured cost: train reward below
untrained levels for ~250k steps and not back to pre-advance level until
~5.3M of a 6M run). The replacement (`WallBallGoalRally`) trains the
goal task directly — the constant-width 0.22.0 fence had already restored
crude learnability there (80-84% placement-blind two-return rate vs
9.5-16% on the retired narrow fence), which no run ever saw because none
trained past stage 1. A serve-origin curriculum (the one axis measured to
control receive difficulty, with rungs that change only the reset
distribution) was designed, certified, and then *tested against direct
training* in the pre-registered paired local SAC A/B/C: direct goal-task
training won — 2.03/2.71 completed returns on 100 held-out seeds at 500k
local steps (all-time ladder best: 1.14 from 6M Colab steps), with a
serve-origin mixture no better (2.27) and an aligned-serve entry rung
strictly worse on goal skill (1.09-1.51, and near-zero zero-shot transfer
for most of training). **Two lessons:** (1) when a curriculum's guard
rails outlive the hazard they guarded (the 0.22.0 geometry fix removed
the goal-task learnability collapse), the curriculum itself becomes the
obstacle — re-test the direct task after any geometry change; (2) if a
curriculum axis is ever needed again, pick one that leaves replayed
experience valid across rungs (reset-distribution changes, not dynamics
changes) — the fence axis taxed every promotion with off-policy
staleness by construction.

### Serve alignment is falsified as a learning aid — easier receives train weaker policies — *closed (0.24.0)*
The serve-alignment hypothesis (move the serve landing to the paddle so
receive is easy; `design_wall_ball_serve_alignment.md`) was tested at both
of its pre-registered sites in the diagnosis campaign's paired local SAC
battery. Stage-1 A/B: pooled paired Δ −0.34 against alignment, both pairs
negative, and the 2×2 cross-eval shows baseline-trained policies beating
aligned-trained ones on BOTH serve geometries — including the aligned
serve itself. Goal-fence A/B/C: the aligned-entry arm was the worst of
three conditions on goal-task skill (held-out 1.09-1.51 vs 2.03-2.71 for
direct training). Mechanism: approach-to-receive training generalizes
down to easy receives; easy-receive training does not generalize up — the
alignment removes exactly the skill the campaign needs. **Do not
re-propose serve-origin alignment (full or partial) as a training aid;**
the λ-blend feasibility maps remain valid as probe-calibration reference.

### A promotion bar must be calibrated against references on the rung it gates — *implemented (0.24.0)*
The 3.0 bar was calibrated once, on the retired 0.21 stage 0 (a predictive
controller scored 4.39 there), then inherited by geometries where the
scripted ceiling is ~2.5-2.9 and SAC's own multi-million-step asymptotes
are 2.1-2.8. The standing "do not lower the bar on scripted-oracle evidence
alone" rule was honored: no bar was lowered — the gated ladder was retired
outright, and `WallBallGoalRally`'s single-stage gate keeps 3.0 as a purely
informational campaign-goal marker (`curriculum/gate_window_mean` crossing
it in TensorBoard) that can never gate anything. Note the arithmetic that
made 3.0 unreachable-in-practice as a *promotion* bar: at the goal cadence
(~125 steps/exchange, measured) `episode_len = 750` caps an episode at ~5
returns, so 3.0 demands ~60% of the physical cap sustained across a
3x60-episode window.

### Promotion staleness needs its own clock — *implemented (0.24.0)*
Run `20260727_233859` sat on stage 1 for 201 evaluations (~5M steps) with
no mechanism able to notice: early stopping watches best-model improvement
(and noise kept resetting it via a `best_metric_min_delta` ~1/13 of a batch
SE), while promotion staleness was unmonitored. `stage_eval_budget` on the
gate now bounds a non-final stage's residency: `"stop"` ends the run (the
right default when the bar is load-bearing), `"advance"` force-promotes
with `advance_reason: "stage_eval_budget"` recorded in
`curriculum_stages.json` (the right choice when rungs are benign
reset-distribution shifts and the anneal must complete inside the budget).

### Scripted probes must lead the intercept, and their gaps are seed-sensitive — *implemented (0.24.0)*
The 0.21-era certification probes tracked the ball's current y/z while
charging; on the constant-width ladder that scores 36%/70% two-return rates
at stages 0/3 — probe defects indistinguishable from geometry defects until
recalibrated. The `lead_charge` mode (same commit trigger, ballistic y/z
lead at the closing-speed intercept) clears 90% everywhere; gaps were
frozen at n=200 by rate-with-margin rather than raw argmax (calibration
argmaxes shed 3-5 points held-out here, twice measured), then certified
once on held-out seeds 3000-3099 at 95-100%. Startup certification runs 30
episodes/cell, not 10: a 0.90 floor against true rates of 92-98% flips on
one unlucky seed at n=10.

### The 0.11.0 bootstrap package — a strictly monotone competence ladder — *implemented (0.11.0)*
The fix for the flat `-1` valley. On the stage-0 serve (n=120) the ladder now
reads **parked −1.00 < weak-swing tracker −0.85 < placement-blind full swing
+7.63 < oracle +12.07**. Mechanisms: a once-per-episode outright
`first_hit_bonus` (0.25); a `weak_return_penalty` (0.1) fined retry replacing the
terminal weak-return fault; early-touch softening; a performance-gated serve
ladder (`serve_vy_max` 1.1 → 2.0, gate `bounce_count_ep_mean ≥ 1.3` sustained
2 evals) with matched-stage selection; and the previously-untried exploration
config (`ent_coef="auto_0.02"`, `target_entropy=-1.5`, `learning_starts=10k`,
buffer 500k). Adversarial review closed two high-severity exploits before
commit: repeat paddle taps no longer reset the stall clock (touch-then-deaden
possession was a risk-free +0.25 ride to truncation), and a stage advance now
resets best-model selection state (an easy-stage score otherwise permanently
barred better final-stage policies, ~0.6–0.7 metric inflation). The no-op
invariant holds: a parked paddle still scores 0 contacts and −1.

### Geometry: the old lane was unforgiving, not unsolvable — *implemented (0.10.0)*
The chosen bundle: baseline lane `(-3.2, -1.6)` + paddle slide damping `8`
(new `paddle_joint_damping`, baseline recipe only) + `early_touch_penalty 0.25`
(non-terminal fine replacing the terminal `paddle_before_bounce` fault). The two
geometry changes only work *together* — either alone collapses oracle second
returns to ~50%. Calibration sweep (config → oracle ≥2 / crude-tracker ≥2):
old `(-2.1, damping 5)` → **0.95 / 0.00**; chosen `(-1.6, damping 8)` →
**0.92 / 0.70**. `(-1.4, 8)` scored marginally better for crude play but put the
whole serve-bounce footprint inside the lane (94% pre-bounce faults for a naive
front-camper vs 39% at −1.6). The earlier "~73% physically unsaveable" framing
was a *correction target*: that figure is for untouched rebounds from fast/flat
returns; a retreating oracle always recovered ≥2 from 95% of standard serves.
**Lesson:** distinguish "unsolvable" from "unforgiving of mediocre play" — the
crude-tracker column, not the oracle column, is what RL has to climb.

### Difficulty ladders that were swept and rejected — *rejected*
- **Depth ladder:** difficulty vs distance-to-wall is U-shaped, not monotone.
  Close-court returns hit the wall hot and rebound out (oracle ≥2 collapses to
  ~50%); mid-court to baseline is flat. Moving the paddle closer does not ease
  the task.
- **Serve-pace ladder:** return power rides on incoming momentum, so *slower*
  serves underpower returns (oracle first-returns fall from 100% at serve speed
  5.5 to 12% at 3.5). Slowing the serve makes it harder, not easier.

### The recovery-schedule bug: forward attr writes through the wrapper stack — *implemented (P0)*
`LinearEnvAttrScheduleCallback` called `training_env.set_attr(...)`, which under
SB3 2.9.0 / gymnasium 1.3.0 `setattr`s the outer `Monitor` wrapper — a shadow
attribute the `WallBallEnv` property setter never sees. One run trained at
`recovery_reset_probability = 0.6` for all 750k steps instead of tapering to
0.15, and `get_attr` read the shadow copy so probes were masked too. Fix:
`env_method("set_wrapper_attr", name, value)` (walks the stack), log the applied
value each eval, and integration-test *through* `make_vec_env` + `Monitor`, not a
fake recording VecEnv. **General rule:** never `set_attr` a semantic env
attribute on a VecEnv; go through `set_wrapper_attr`/`unwrapped`.

### Reward channels must actually pay out on the real distribution — *open (P1)*
`recoverable_bounce_bonus` graded the bottleneck skill (return placement) with a
hard `clip(1 − |proj_y|/2.0, 0, 1)` — but real rebounds run to |y| = 3.7 against
the 2.0 clip, so it paid **exactly zero in 76%** of completed returns (~0.2% of
per-cycle reward, zero slope where it mattered). Recommended: smooth kernel
(`exp(−|proj_y|/limit)`), raise the lateral limit 2.0 → 3.0, raise the bonus
0.25 → ~0.75 to be commensurate with the +1.0 wall bonus.

### Keep `ent_coef` adaptive; exclude binary/counter obs dims from normalization — *implemented (0.11.0)*
Fixed `ent_coef = 0.02` disables SAC auto-entropy (the recorded
`target_entropy = -3.0` is inert for a float coefficient) and was carried over
from the legacy 5-action env. `learning_starts` was never raised off the SB3
default of 100 (batch-256 updates began after ~104 transitions, VecNormalize
fit on <2 episodes), and WallBall normalized binary flags/counters (obs dims
12, 13, 20, 21, 22) whose base rates drift with the curriculum, even though
`SelectiveVecNormalize` already supports `normalize_obs_excluded_indices`.
Shipped in the bootstrap package as `auto_0.02` / `target_entropy=-1.5` /
`learning_starts=10k` / buffer 500k.

### Make each additional exchange worth strictly more than the last — *open (P1)*
In the "one-and-done" run, kept shaping was +1.74 = **87.9%** of net return; the
−1.0 double-bounce penalty was simply priced in against ~+3.0 of one-cycle
income, so a second exchange was never worth visiting (max training return in
1,152 episodes: 2.04). Recommended: cut `track_shaping_scale` 0.5 → 0.2 and make
the wall reward escalate per exchange (`1.0 + 0.5*(bounce_count − 1)`, capped
~3.0).

### The "one-and-done" degenerate policy — *diagnosed; remediated by the bundle above*
Signature to recognize: converges by ~225k steps to exactly one legal hit, one
wall return, then lets the ball double-bounce; rollout CSVs at 250k/500k/750k/1M
are behaviorally identical (froze at 250k); 50/50 long-horizon episodes show
exactly one return. It is not one bug — it is the confluence of rules 2, 3, and
"each exchange must pay more" above.

---

## Training harness & instrumentation

### A run directory says what ran and seals itself; a mirror is trusted only through its manifest — *implemented (unreleased, 2026-10-17); design L1, first half*
`design_campaign_structure_20261016.md` R3: 7 of about 48 archived run
directories were stubs that looked live, one run lost all its videos
with no trace, and nothing could tell a Drive copy from a partial one.
Every env stream and training path is unchanged (golden digests 22/22;
trainer digest against `09f202a`: every component matches but the
layout list, which gains `run_record` and `sync_manifest`).
- **Two files, one rule.** `run_record.json` says what ran and how it
  ended; `sync_manifest.json` (path, size and sha256 of every other
  file) is written last, after the final record, once every env and
  the SB3 logger are closed. A directory is *complete* when its
  manifest verifies and the record it lists is terminal. Complete
  means "exactly what `train()` left when the run ended", a crashed
  or interrupted run included; it does not mean "succeeded". A
  SIGKILL, a dead VM or a run still training leaves no manifest, so
  the directory is incomplete by definition.
- **The record copies `config.json`, it never re-derives.** Algorithm,
  task fingerprints, comparability key, seeds and versions come from
  the `config.json` payload, so the two cannot disagree. The task's
  constructor kwargs are recorded as the sha256 of their canonical
  JSON, and the warm start by its source directory's *name*, not its
  machine path. `eval_protocol` stays `null` until phase 4 versions
  protocols.
- **Status is finer than the summary's.** The record separates
  `early_stop` from `completed`; `stage_summary.txt` keeps
  `completed` for both, because the campaign resumes (re-scores) a
  leg only from that line. A guard whose condition first holds at
  the last scheduled evaluation trained the whole budget: `completed`,
  with a `full_budget: ...` reason that quotes the guard. A callback
  that stops early without a reason gets one that says so.
- **Unlisted files are reported, not fatal by default.** The notebooks
  write plots, the replay video and the long-horizon audit into the run
  directory after `train()` sealed it. Failing those would make every
  notebook run incomplete. Every file `train()` wrote is still checked,
  and `--strict` fails extras. A partial *mirror* still never verifies:
  its terminal record only arrives in the final pass, and a pass that
  dies leaves the previous manifest, which then lists a changed file
  or a `running` record.
- **Mirror rules.** Opt-in (`sync_dir`, default off). The target must be
  absent or empty at launch, so every file in it is this run's, and
  its parent must exist, so an unmounted Drive fails before output
  instead of filling a local folder of the same name. Each file is
  hashed while it is read, written to a temporary file, renamed, then
  read back and re-hashed. A checkpoint pass that fails is printed and
  training continues. The final pass gets three attempts (5 s and 30
  s apart), then raises once the local directory is sealed: the run
  is safe locally, but the caller must learn that its durable copy is
  not. `sync_dir` is refused in run-config files, like `log_dir`: it
  belongs to the machine, not the experiment.
- **The trust fixes, and why each is shaped as it is.**
  `disconnect_runtime` keeps the VM up when `flush_and_unmount()`
  fails: an idle GPU costs less than the last writes of a run.
  `reuse_log_dir` rotates the monitor and TensorBoard *directories*
  whole, because TensorBoard reads any file whose name contains
  `tfevents`, so a renamed event file would still be read. A previous
  attempt's run record now refuses reuse like its evaluation rows, so
  an attempt that died before its first evaluation is caught. The
  interrupt salvage saves the model and normalizer before the gate's
  `finalize` (an `OSError` there used to lose all three).
  `SubstepTennisEventSampler.reset()` discards an open control step;
  a healthy trajectory never reaches it, so the goldens hold, and a
  recovered PaddleTennis env replays a fresh env's stream byte for
  byte. Early stops the budget cannot reach are warned before output
  and recorded (`resolved_training.early_stop_reachability`); the
  warm-up arithmetic moved into helpers that `train()` and the record
  share.
- **Eval share is measured, not estimated.** Each evaluation
  callback's evaluating calls are timed on the instance with
  `time.monotonic` (no stream or RNG moves), plus the closing
  evaluation: the number L8 needs before trimming in-loop
  evaluation. The checkpoint diagnosis is not counted.
- **Open (L1, second half).** The regenerated run index, the backfill of
  the ~48 legacy run dirs (the 7 stubs marked aborted), and mapping
  each PaddleTennis run to its lever.

### The PPO verifier pass: a deployable best states its updates, and nothing fails quietly — *implemented (unreleased, 2026-10-17)*
Verifiers re-ran the phase-2 PPO commits (since `f920d8f`). The fixes
keep every env stream and SAC-family training path unchanged (trainer
digest: the SAC, DemoSAC and PaddleTennis-SAC jobs match).
- **empirical-2, the other half.** Step 5 ended zero-update runs, but
  `best_model.zip` could still be the untrained initial policy: an
  evaluation runs before its rollout's update, and the quick test
  evaluated every rollout, so evaluation 1 scored the initial
  weights. On BallBalance that was the best score (438 at 4,096 steps,
  then down to 206), and the summary said "deploy best_model.zip".
  Two changes. The record and the warning: `best_model_meta.json`
  carries the selecting evaluation's SB3 `n_updates` (top level, so
  selection, the gate and TensorBoard still never see it), the
  summary's "Best eval" and "Best model" lines print it, and a
  `best_model.zip` selected at 0 is flagged on its line and printed.
  Which checkpoint `best_model.zip` holds follows the selector:
  `best_model_meta.json` under headline selection, the reward
  stream's best otherwise (read from `eval_info.csv`'s `n_updates` row
  at that step). And the cause: the quick test fits `n_steps` to
  `eval_freq // (2 · n_envs)` and rounds its preset evaluation cadence
  to at least two rollouts, so evaluation 1 follows one update (256 at
  8 workers, 128 at 16; one worker unchanged at 2,048). Halving the
  rollout, rather than doubling the cadence, keeps the quick test's
  six evaluations and doubles its updates. Measured here: the
  BallBalance quick test trains 13 rollouts (SB3 n_updates 130), its
  best is evaluation 1 at 10 updates (385), and it still regresses
  after it (to 133), which the summary now reports without the
  untrained caveat; WallBallTrueBaseline selects at 10 updates.
- **The last update reaches progress.csv.** SB3's on-policy `learn()`
  dumps the logger before each `train()`, so every update's `train/*`
  row is written with the next rollout's, and the final update's never
  is. `train()` dumps once more after `learn()` returns (not after an
  interrupt or crash) when a `train/*` value is pending, with
  `time/total_timesteps` as the row's x value, as SB3's EvalCallback
  does. Labelling the block "as of the last logged update" was the
  alternative; the dump makes the health block, the flags and
  `train/n_updates` agree with SB3's count instead.
- **Epochs per update.** With gSDE, `target_kl` ended every update
  inside its first epoch (PaddleTennis: one SB3 update per rollout at
  `n_epochs` 10; humanoid Stage 0: 13 over 13 at 5) while the mean
  approx_kl, averaged only over the minibatches that ran, sat under
  the flag. The steps of the cumulative `train/n_updates` series are
  the per-update epoch counts; a last-quarter mean below half of
  `n_epochs` is flagged. The PaddleTennis quick test on the new
  geometry raises it: 23 SB3 updates over 13 rollouts, 1.77 epochs per
  update of 10. Calibrating these profiles' learning rate against
  `target_kl` joins the plan's open probing item.
- **Fail loud, by family.** A negative `degenerate_min_evals` used to
  crash an SAC run inside the update-floor print (`f"{None:,}"`) after
  `config.json`, and a PPO run accepted it; the callback-config
  pre-flight now refuses a non-integer or negative value, and the floor
  helper passes a warm-up through untouched when there is no floor.
  `finalize(salvage=False)`: a pending advance that fails on a normal
  training end raises after the stage history is written, while the
  interrupt and crash salvage (which still have a model to save) only
  report it. The gate's `stage_eval_budget` stop counts a stage's own
  evaluations with no warm-up the floor could raise, so the pre-flight
  refuses a stop budget at or below the floor on a multi-stage gate
  (narrowing the docstring's "no early stop before K updates" was the
  alternative). An actor-critic `net_arch` dict may name only `pi` and
  `vf`. A checkpoint outside any run is checked against its zip's
  recorded policy class (deserialized with SB3's own `json_to_data`)
  before loading.
- **Accepted as documented, no change.** Steps 6 and 13 added two
  `TrainConfig` fields at their defaults to every snapshot entry, SAC
  ones included; the verifier confirmed the change is purely additive
  (no removed or changed value outside the documented humanoid SAC
  entries). The humanoid notebook's default run refuses the pinned
  `REPO_REF` (`b9585e2`, before `reset_log_std`) by design until the
  week-1 pin bump, which is now a merge prerequisite in the plan's
  Open list.

### PPO is first-class: loaders follow the record, runs carry their family, and two learning bars guard it — *implemented (unreleased, 2026-10-17)*
PPO plan Steps 10–12 and 14 (`design_ppo_first_class_20261016.md`,
static-6/9/10/11, empirical-3, tests-gaps-9/13), which close the plan.
PPO is now selected, configured, checked, run, recorded, loaded and
tested per family like SAC; every env stream and SAC-family training
path is unchanged (golden digests 22/22, recipe snapshot unchanged,
trainer digest matches on the SAC, DemoSAC and PaddleTennis-SAC jobs).
- **Loaders read the run's record.** `algos.load_run_model(path,
  algo=None)` finds the run's `config.json` from the model path, looking
  only where the run layout puts a model (the root of a pre-0.14 run,
  `model/`, `checkpoints/`, a stage archive), and loads each family
  through its root class: SAC and DemoSAC through `SAC.load`
  (DemoSAC's inference path, which needs no demo library), PPO through
  `PPO.load`. An explicit `algo` must name the recorded algorithm
  (DemoSAC is not SAC there, although both load alike). A record that
  says nothing (unreadable JSON, no `train_config.algo`) raises rather
  than guessing, and a checkpoint outside any run needs `algo=` (the
  tools' new `--algo`). It replaces `SAC.load` in
  `native_checkpoint_policy` (and so in the diagnosis, post-swing, k=2
  harvest and k=2 step-0 tools), the P5 champion loader,
  `score_paddle_stage`, `_load_best_model` and `record_best_model_video`
  (whose default was `algo="SAC"`, now `None`). `evaluate_best_wall_ball`
  already followed the record but loaded DemoSAC through
  `DemoSAC.load`; it uses the shared loader too. SAC keeps the exact
  call it made.
- **On-policy provenance.** `resolved_model` gains `target_kl`
  (shared key list: SAC has no such attribute, so it is skipped there)
  and, gated on `OnPolicyAlgorithm`, `sde_sample_freq` and the
  policy's `log_std_init` and `ortho_init`. The gate matters: SAC
  models do carry `sde_sample_freq` and `_n_updates`, and the trainer
  digest freezes SAC's record. When an on-policy run's training ends
  (also in the crash salvage), the block is re-recorded with SB3's
  `n_updates` and `timesteps` (`trained` vs `budget`). PPO checks the
  budget between whole rollouts, so it can overrun by up to one; the
  summary's Timesteps line now says by how much (on-policy runs only).
- **Family and comparability in `config.json`.** A top-level
  `algorithm` block (`name` = the registered class, `family`,
  `implementation` = `sb3==<version>`, since a GPU-stack PPO would be
  another implementation under the same name) and a
  `comparability_key` (`task` = recipe or env class, `eval_protocol`,
  `algo_family`). `eval_protocol` is `null` until the evaluation
  protocols are versioned. Both are top-level because `train_config`
  must hold only configured fields and `resolved_model` is frozen for
  SAC. The campaign scorer's report names the algorithm and copies the
  key. No reader ranks runs yet; the run index (week 1, phase 3) must
  key on the whole block, not on the task alone.
- **Training Health flags (on-policy).** Four flags, each on the mean
  of the last quarter of the logged updates, so one noisy update
  neither raises nor clears one: approx_kl > max(2 · target_kl, 0.05)
  (SB3 cuts an update's epochs at 1.5 · target_kl, so twice the target
  means the cut is not holding); clip_fraction > 0.3; explained_variance
  < 0; and the final `train/std` more than 1 nat from `log_std_init`
  (`train/std` is SB3's mean of exp(log_std), read on its log). The
  references come from the run's own `resolved_model`. A non-finite
  window is flagged, never averaged away. Flags print as well as land
  in the summary; the SAC summary is unchanged. A humanoid-shaped
  progress.csv (approx_kl 0.22–1.71, clip_fraction 0.72–0.83, the log
  std up 1.6 nats) fires all four. The BallBalance probe below fires
  none: last-quarter approx_kl 0.013, clip_fraction 0.16,
  explained_variance 0.38, log std −1.36 against −1.
- **Characteristic: explained_variance is ill-conditioned once returns
  stop varying.** In that probe's last quarter, after every episode had
  reached 750, it swung between −2.0 and 0.98 and ended at −1.995 while
  the value loss was 2e-5. A run kept going long past convergence can
  raise the flag; read it next to the reward curve.
- **Tests that cross the whole path.** A parametrized SAC/PPO
  `train()` test on 12-step BallBalance (one worker, PPO on 8-step
  rollouts) covers headline selection with `confirm_best` and a paired
  `eval_seed`, the final-config stream, VecNormalize checkpoints, a
  one-stage gate, the patience stop and the closing evaluation. It
  checks the artifacts, `resolved_model`, the stop step (evaluation 4
  for SAC, 7 for PPO, whose warm-up the update floor raises to 5),
  `norm_reward` in every pickled normalizer, and that every saved model
  reloads through its own class. Two seeded PPO runs train identical
  weights and normalizers. A real tiny PPO PaddleTennis run trained by
  `train()` is scored by `score_paddle_stage`.
- **CI learning bar.** A 1-D reach defined in the test (reward
  −|goal − x|, truncated at 40 steps, never terminated) learns through
  `train()` with PPO on 4 × 256 rollouts, minibatch 256, lr 1e-3 and
  reward normalization on, in about 11 s of one thread. Measured best
  paired-evaluation return over 24,576 steps: −3.41, −3.22 and −3.23
  for seeds 0, 1 and 2. The optimum, full speed to the goal and then
  hold, averages about −3.3; the zero action −26.7. Max approx_kl was
  0.014–0.022. The bar is best ≥ −6 and approx_kl < 0.1. Selection is
  on the paired evaluator's reward (`best_metric_keys=
  ("episode_reward_mean",)`), so the reloaded best pair replays the
  selecting batch's seeds and must reproduce its mean exactly. The
  plan's −1.8 was the audit run's unpaired reward stream (10 fresh
  episodes per evaluation, −15.6 to −1.77), an easier draw than these
  paired batches.
- **Manual learning bar.** `tools/ppo_learning_probe.py` trains
  BallBalance on its recipe's PPO profile as `build_train_config`
  resolves it, capped at 330 s of wall clock, and scores the best model
  behind its normalizer on held-out seeds 70,000–70,019 (bar: mean ≥
  700 of 750). Run once here on one thread: 491,520 steps and 1,190 SB3
  updates in the cap (349 s with the closing evaluation), evaluations
  at 749.4 from 294,912 steps and 750.0 from 319,488, held-out mean
  750.0 on all 20 seeds: PASS. Exit status 1 on a FAIL. Scheduling it
  nightly or by workflow_dispatch is left open.
- **Notebooks.** The campaign notebook admits `ALGO = "PPO"`. The legs
  build on the recipe's PPO profile, at 8 workers with evaluations
  every 24,576 steps and checkpoints every 98,304: the whole-rollout
  cadences nearest the SAC plan's 25,000 and 100,000, so a PPO gate
  reads about as many evaluations and diagnosis probes. The profile's
  own 32,768 and 245,760 are sized for 16 workers too. No SAC key
  reaches a PPO leg (no `learning_starts` on a warm-started one). The
  settings cell refuses `REQUIRE_DEVICE` (PPO trains on the CPU), a
  temperature skip, a package without first-class PPO (the pinned
  `REPO_REF` included) and a geometry the on-policy pre-flight
  refuses. After Drive mounts, every warm-start lineage must record a
  PPO algorithm: train() would refuse the frozen SAC fallback only at
  the main leg, a whole gate leg later. PPO campaigns live under
  `PaddleTennisCampaign/ppo/`. `sb3_training` no longer gives a family
  switch 16 workers when the recipe has a block for the new family:
  its PPO profile sizes `n_envs` with `n_steps`, and BallBalance under
  PPO used to train its measured 8 × 512 profile at 16 × 512.
- **Left open.** A campaign `reset_log_std` knob (the PaddleTennis
  profile has no measured warm-start story yet); the humanoid
  notebook's own `resolve_algo(cfg.algo).load` (its automatic
  advancement is PPO-only, so it already loads the recorded class);
  the plan's section 4 items beyond Step 9.

### PPO's stops count updates, and its task changes only between rollouts — *implemented (unreleased, 2026-10-17)*
PPO plan Steps 6–9 and 13 (`design_ppo_first_class_20261016.md`,
static-3/4/5, empirical-1/4, tests-gaps-8). The runtime still treated
an on-policy run like SAC in five places: stops sized in evaluations,
stage and schedule changes at any step, a reward normalizer
discounting at 0.99 whatever the model's gamma, SB3's `"auto"` device,
and a warm start that carried the source's exploration noise.
- **Stops wait for K updates.** `TrainConfig.min_policy_updates_before_stop`
  (K = 20) floors both early-stop warm-ups and `degenerate_min_evals`
  at ceil(K · rollout / eval_freq) evaluations, as the plan specifies.
  Each stop needs at least one evaluation past its warm-up, and
  evaluation e sees e · R − 1 updates (R rollouts per evaluation), so
  the earliest stop sees ≥ K + R − 1 ≥ K. A test reproduces the
  audit's stop at `_n_updates == 0` with K = 0 and shows K = 3 moving
  it to the 4th evaluation, 3 updates in. The floor adds to the
  degenerate window rather than replacing it, so on PaddleTennis,
  WallBallBootstrap, GoalRally and TrueBaseline the 6-evaluation
  window (Step 3's sizing for ~20 updates) now first fires at 43
  updates at 8 workers instead of 23; at 16 workers it fired at 11,
  below the target, and fires at 31 now. WallBallBaseline's
  schedule-derived warm-up already exceeds the floor at 8 workers (55
  updates, unchanged) and moves from 27 to 31 at 16. Patience never
  binds on a shipped profile. Retuning the windows now that the floor
  exists is left to the profile work. An off-policy run that sets K to
  anything but the default is refused: the field would otherwise be a
  silent no-op there.
- **`n_updates` means SB3's count.** The new `eval_info.csv` row is
  SB3's `_n_updates`, the counter `progress.csv` logs as
  `train/n_updates` and the summary as "SB3 n_updates" (optimization
  epochs), not the rollout count K is measured in; with `n_epochs` E
  and no `target_kl` cut it reads E × rollouts trained. A second row
  for rollouts was left out: it follows from the timestep. The row is
  written to the CSV only, not added to the metrics, so selection,
  the gate and TensorBoard are unchanged (`best_model_meta.json`
  gained a top-level `n_updates` in the verifier pass).
- **Rollout boundaries, detected not configured.** The gate and the
  schedule callback check `isinstance(model, OnPolicyAlgorithm)` at
  training start; there is no knob that could put a PPO run back on
  per-step changes. A gate advance earned mid-rollout waits for
  `_on_rollout_end`. Under `train()` the pre-flight already makes every
  evaluation end a rollout, so no step separates the two; the deferral
  makes the callback correct by construction, also when it is wired by
  hand. When another callback stops the run at the earning
  evaluation, `finalize` applies the pending advance (a failure
  raises on a normal training end and is only reported under the
  salvage paths, since the verifier pass), so
  `curriculum_stages.json` reads as an off-policy run's would.
  Schedules apply `value_at(start)` at each rollout start.
- **Normalizer gamma, on-policy with reward normalization only.** SB3
  updates `ret_rms` whenever the normalizer trains, even with
  `norm_reward=False`, so giving every run its model's gamma would
  change SAC's pickled statistics. The four `SelectiveVecNormalize`
  constructions all get the gamma, so a synced evaluation normalizer
  never disagrees with its source, and `train()` checks the
  normalizer's gamma against the built model.
- **CPU by default** when SB3 itself would warn: its
  `_maybe_recommend_cpu` compares the policy class name with
  `ActorCriticPolicy`, so CNN and multi-input policies keep `"auto"`.
  An explicit device or `require_device` wins. PaddleTennis's PPO
  profile keeps its explicit `device="cpu"`, recorded as
  `model_kwargs`.
- **`reset_log_std`** refills `policy.log_std` in place (the optimizer
  keeps its parameter) from the target policy's own `log_std_init`,
  after the strict state-dict load. On-policy
  `transfer_log_ent_coef=False`, a documented no-op until now, is
  refused because it reads like an exploration restart and is not one.
  The humanoid notebook needs the option, and its pinned `REPO_REF`
  (`b9585e2`) predates it, so AUTO_ADVANCE refuses that package at
  configuration time rather than at the first advance hours in. The
  week-1 pin bump lifts that.
- **Subprocess workers must not change the run.** Stock
  `SubprocVecEnv` returns float64 rewards where `DummyVecEnv`'s buffers
  hold float32, so VecNormalize's return statistics, and with them a
  normalized PPO run, drift from step 1. Measured on BallBalance,
  BallBounce, WallBall TrueBaseline, PaddleTennis and humanoid Stage 0
  (2 workers, 600 to 1,700 steps, random actions): the stock streams
  differ at step 1, and `DummyDtypeSubprocVecEnv`, which fills
  DummyVecEnv's buffers by the same assignment, matches every
  observation, reward, done and terminal observation. `check_env` runs
  on a throwaway parent instance instead of worker 0; the same
  measurement shows the streams match with that difference. A tiny
  PPO run trains bit-identically on both (weights, normalizer,
  monitors).
- **Where the records go.** `config.json`'s `train_config` stays
  exactly the configured fields (a test forbids derived keys), and the
  trainer digest compares `resolved_model`, so new derived values go to
  a new top-level `resolved_training` block. That also keeps the SAC
  jobs' `resolved_model` identical.
- **Snapshot.** Two new `TrainConfig` fields add a key to every resolved
  entry, SAC ones included. The re-record was checked to be purely
  additive: 184 inserted lines, two per entry, with every
  pre-existing field byte-identical.

### PPO runs are configured and checked in whole rollouts — *implemented (unreleased, 2026-10-17)*
PPO plan Steps 3–5 (`design_ppo_first_class_20261016.md`, static-2,
static-3(1), static-8, empirical-2). PPO updates once per rollout of
`n_steps * n_envs` env steps, but every cadence and guard was set in
env steps for SAC. At 8 workers × SB3's 2,048-step rollout, the
degenerate stop at 5 evaluations × 25k fired after 0 to 7 updates, a
25k-step quick test could finish "completed" with one update or none,
and evaluations landed mid-rollout.
- **Profiles.** Each recipe's `algo_cfg["PPO"]` holds the design's
  section-3 values, with these choices where the design left room:
  - Cadences are whole rollouts at the profile's worker count and at
    16, sb3_training's PPO count. 25k is not a multiple of any
    rollout, so `eval_freq` is 24,576 (6 rollouts of 4,096; 12 of
    2,048) or 32,768 (4 of 8,192), and `checkpoint_freq` is 245,760
    (98,304 on the humanoid stages).
  - Guards are sized so none fires before about 20 updates at the
    profile's own worker count. An evaluation ending rollout k runs
    before rollout k's update, so evaluation e has seen
    e × (rollouts per evaluation) − 1. The degenerate stop needs 6
    evaluations (23 updates) instead of 5 (19); patience 20 first
    stops at 159–479. At 16 workers an evaluation spans half as many
    rollouts; Step 6 will floor the guards in code.
  - The humanoid `ent_coef` is 0.001, the top of the design's 0–0.001
    range: a token floor against std collapse at a tenth of the drift
    measured at 0.01. The 1-env humanoid stages are refused at 16
    workers (`eval_freq // n_envs` is less than `n_steps` 2,048) until
    Step 13's 8 × 512 geometry.
  - The design has no smoke row. HumanoidTennisCoopSmoke gets the
    stage settings on 128-step rollouts, so its 10k budget sees about
    80 updates.
  - No profile pins gamma: the recipe's own (a flat `extra_cfg` gamma,
    else SB3's 0.99) applies to both families. PPO's reward
    normalization is pinned explicitly even where it equals the
    default. `device="cpu"` is set on PaddleTennis only; Step 8 makes
    it the on-policy default.
- **Pre-flight.** `_validate_on_policy_config` refuses rather than
  rounds. A config whose cadence is not a whole rollout says so, with
  the nearest multiple; silently moving an evaluation would change
  what selection compares. Besides the design's list it refuses
  cadences that are not whole rollouts and non-integer
  `n_steps`/`batch_size`. It runs in `train()` before any output and
  in `build_train_config`, so a recipe refuses at build time. The
  depth-ladder recipes' SAC-only gate levers make them PPO refusals
  by design; giving a retired SAC ladder an invented PPO gate would be
  a new experiment, not a port.
- **Quick test.** The design's `n_steps` shrink alone left the 5,000-
  and 10,000-step presets off the rollout grid, which the pre-flight
  refuses. So the presets, never an explicit value, also round down to
  whole rollouts. An explicit cadence that misses is refused instead.
- **Zero updates** are flagged in the summary and on the console for
  the PPO family only. SAC summaries stay byte-for-byte as they were.
  The summary line is labelled "SB3 n_updates", not "policy updates":
  SB3's `_n_updates` counts optimization epochs (`n_epochs` per
  rollout, fewer under `target_kl`). A BallBalance quick test reads 70
  after 7 rollouts.
- **Starters stay family-safe.** A flat cadence in a starter beats the
  recipe's PPO block and would be refused under PPO. So the starters
  keep cadences in `[train.algo.*]` tables, and a test checks that
  each starter resolves like its recipe under both families.

### Recipes and run files configure each algorithm family separately — *implemented (unreleased, 2026-10-17)*
PPO plan Step 2 (`design_ppo_first_class_20261016.md`, tests-gaps-1,
static-1, empirical-9). A recipe carried one flat
`extra_cfg["model_kwargs"]`, written for its default algorithm. PPO was
therefore refused on PaddleTennis and WallBallBootstrap (their SAC
bundles hold SAC-only keys), got SB3 defaults everywhere else, and an
explicit SAC run of a humanoid stage inherited the PPO bundle's fixed
`ent_coef`. SB3 2.9's SAC also accepts `n_steps`, so
`validate_model_kwargs` cannot catch a shared key whose meaning differs
between the two.
- **Decision.** `Recipe.algo_cfg` maps a family (`algo_family`) to a
  block of `TrainConfig` fields applied over `extra_cfg`, with
  `model_kwargs` merged one level deep. Blocks are keyed by family,
  not algorithm, so DemoSAC inherits SAC's calibration. `extra_cfg`
  keeps only what both families share (WallBall's `gamma`, the
  humanoid stages' `normalize_reward=False`). Run files get matching
  `[train.algo.sac]` / `[train.algo.ppo]` tables, merged after `[train]`
  for the run's family only, so one file can carry both profiles.
- **Divergent shared keys are refused, not reinterpreted.**
  `DIVERGENT_SHARED_KWARGS` lists the keys both constructors accept
  with different meanings. A flat key from it (recipe or file) is
  written for the recipe's default family, so under the other family
  `build_train_config` refuses it and points to the family tables.
  Warning instead would let a SAC `batch_size=512` silently become a
  PPO minibatch. An explicit `model_kwargs=` argument is exempt: it
  replaces every lower layer, flat keys included, and the hint says so.
- **The one intended SAC-family change** is the humanoid Stage 0–2 ×
  {SAC, DemoSAC} resolution: the gSDE + `ent_coef=0.01` bundle was
  chosen and measured under PPO, and those recipes default to PPO. An
  explicit SAC run now gets SB3's SAC defaults. It is off the campaign
  path; every other SAC-family snapshot entry is byte-identical, and
  the trainer digest matches.
- **The snapshot records PPO too.** With every recipe now resolving
  under PPO, `tools/recipe_snapshot.py` adds PPO to its matrix (96
  resolutions), so each later PPO step's change is a reviewed diff
  rather than an unrecorded one.
- **Readers of the old location.** The campaign notebook and the
  trainer digest's PaddleTennis job read the SAC bundle through
  `build_train_config` rather than from `extra_cfg` or `algo_cfg`, so
  they work against both the pinned `REPO_REF` and this package.

### Algorithm-dependent code tests families derived from the registry — *implemented (unreleased, 2026-10-17)*
PPO plan Step 1 (`design_ppo_first_class_20261016.md`, static-7). The
trainer and the notebooks decided algorithm-dependent behaviour with
hand-written strings (`algo.upper() == "PPO"`, `!= "PPO"`, a
`("SAC", "DEMOSAC")` tuple, a hand-kept `OFF_POLICY_ALGOS`). Each
string test is a place where a new registration can silently take the
wrong branch, and one already did: `sb3_training` gave
`ALGO="DemoSAC"` PPO's 16 workers because DemoSAC is not the string
`"SAC"`.
- **Decision.** `OFF_POLICY_ALGOS` and `ON_POLICY_ALGOS` are derived
  with `issubclass` from SB3's `OffPolicyAlgorithm` and
  `OnPolicyAlgorithm`, and an algorithm that is neither refuses at
  import. `algo_family()` names an algorithm's config family after its
  most general registered ancestor, so DemoSAC (a SAC subclass) is in
  the SAC family and reads SAC's settings. Two unrelated registered
  ancestors would make the family ambiguous and refuse at import.
  Every resolved value is unchanged: the derived sets equal the old
  set, and the recipe snapshot and trainer digest match.
- **Notebooks** test the derived `OFF_POLICY_ALGOS`, not
  `algo_family()`: they install the package at a pinned `REPO_REF`
  (`b9585e2`), which exports that name but not the new function. The
  humanoid notebook's `RESOLVED_ALGO != "PPO"` stays: automatic
  advancement supports exactly PPO, not a family.

### Golden digests and the freeze harness gate every refactor — *implemented (unreleased, 2026-10-16)*
Week 1 restructures the env and training code by extraction (design
M1 + L0, PPO plan Step 0). The rule is that nothing on the campaign path
moves unless the change is declared, and three gates make that checkable:
- **Env streams.** `courtside_dynamics.digests` hashes obs (float64),
  reward, terminated, truncated, qpos and qvel at every reset and step.
  Separate digests cover every info item (non-scalars recursively, by
  type, dtype and shape) and the interface the float64 hash erases:
  the spaces' dtype, shape and bounds, and the concrete obs, reward and
  flag types. All nine `rew_*` keys are required at every PaddleTennis
  step. `tools/golden_digest.py` checks against
  `tests/golden/env_digests.json`, and `tests/test_golden_digests.py`
  gates CI. The 22 CI cases take about 45 s serial: the recipe ×
  {zero, random, oracle, camper}, one-point with alternating serves,
  the +hold arm, `volley_rule="legal"` with the lead-charge oracle, the
  k=2 drill in both launch arms, three NaN witnesses, the recipe's
  random and oracle streams reset the way training resets them,
  WallBall TrueBaseline (random, oracle, and the oracle reset the
  training way), BallBounce, BallBalance and 300 humanoid steps.
  `--full` runs five spaced seeds per case (two for the humanoid case;
  the NaN witnesses stay on the one seed each is calibrated on).
- **Reset modes.** The original cases reseed every episode
  (`env.reset(seed=seed + episode)`), so each episode reproduces on its
  own, but training never resets that way: SB3's `DummyVecEnv` seeds
  its first reset and auto-resets with `reset()` (no seed, no options)
  from then on, so the env RNG and any state the env keeps across
  resets carry over. PaddleTennis keeps its serve alternation, which a
  seeded reset restarts at side A. A refactor that, say, re-derived the
  serve side on every reset would pass every reseeded case. A case's
  `reset` field picks `"seed_each"` (the default, so every older
  golden reproduces unchanged) or `"seed_once"`. Three seed-once CI
  cases run the training path: the PaddleTennis recipe under random
  and oracle play for 3,200 steps (two 1,500-step episodes and 200
  steps of a third, so two unseeded resets), and the WallBall
  TrueBaseline oracle for 3,000 steps (12 episodes). On seed 0 the
  oracle's three episodes serve from sides A, B and A; a test pins
  that seed-once resets alternate the serve and draw later serves
  from the carried RNG, where seed-each resets serve from A every
  time. The new cases are recorded as new entries; no existing entry
  changed.
- **SIMD dispatch.** numpy picks AVX-512, AVX2 or baseline ufunc
  loops at import, and some of them round differently. On numpy 2.5.3
  (x86-64, 2×10⁵ random inputs) these float64 loops give other last
  bits without AVX-512: `exp` (4.7% of inputs), `log1p`, `expm1`,
  `power`, `arcsin`, `arccos`, `arctan2`, `cbrt`, and rarely `log`,
  `tan` and `arctan`. `tanh` differs without AVX2 (20%). Float64
  `sin`, `cos`, `sqrt` and `hypot` do not move. The env code calls
  only those four (and `np.linalg.norm`) from numpy's math; its few
  other transcendentals are Python `math` calls (libm, no dispatch),
  and the physics is MuJoCo's compiled code. So the goldens,
  recorded on an AVX-512 machine, still match on GitHub's runners,
  which may lack AVX-512 (often an AMD EPYC 7763). Measured: all 22 CI cases match
  with AVX-512 disabled (`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL
  AVX512_SPR"`, numpy 2.x names; numpy 1.26 calls them `AVX512F
  AVX512CD AVX512_SKX AVX512_CLX`), with AVX2 disabled as well, and the
  full set matches with AVX-512 disabled. `version_info()` now records
  the dispatch (`numpy_simd`, e.g. `baseline X86_V2; dispatch X86_V3
  X86_V4`), and a CI test replays one case per env family with every
  dispatch target disabled, so a refactor that brings, say, an
  `np.exp` into a stream is caught on any CPU before a runner with
  other SIMD finds it.
- **NaN witnesses.** These sit on pending escrows, and a test re-proves
  them from the stream's own info. A NaN action at step 250 of the
  seed-0 recipe oracle stream claws back the contact escrow (−0.25). At
  step 430 on the hold arm it claws back reach (−0.1753) and hold
  (−0.0099) at once. At step 443 of the seed-1 hold-arm stream it claws
  back reach (−0.17532) and hold (−0.01969), and that reward's bits
  depend on the summation order.
- **Reward summation order.** Float addition is not associative, and
  L6 (TermStack) and M11 (`step()` split into terms) must re-associate
  the PaddleTennis reward under a "digests identical" gate. Before this
  fix no digested step was order-sensitive (the count is 0 for every
  format-1 case in both sets), so swapping two terms passed the gate.
  Three checks now hold the order:
  - every digested PaddleTennis reward must equal the left-to-right sum
    of its nine `rew_*` info terms, bit for bit;
  - `reward_order_sensitive_steps` is a gated field, and the seed-1
    witness keeps it at 1 in CI;
  - two tests force every term of `step()` (nine) and of the nonfinite
    guard (four) nonzero at once, with magnitudes searched so that any
    exchange of two terms moves the bits. Only the first pair is
    exempt, since float addition commutes. `math.fsum`, and `sum()`
    from Python 3.12 on (compensated), fail those tests too.
- **Drill cases.** The drill library is harvested in-process from
  seed 1000 (two k=2 entries plus one clearance-violating copy), so no
  pickle is committed. At `drill_fraction=0.5` both arms record
  drilled points, undrilled policy-receiving points and the clearance
  fallback, and the counts are gated fields. Two drill mutants stay
  invisible. A `<` → `<=` eligibility test differs only when a uniform
  draw equals 0.5 exactly. Dropping the full arm's solver warm-start
  restore changes nothing at these two launch states, which hold a
  single active constraint.
- **Recipe snapshot** (`tools/recipe_snapshot.py`,
  `tests/golden/recipe_snapshot.json`). It records 70 resolutions: every
  recipe × {default algo, SAC, DEMOSAC} × quick_test, plus the seeded
  PaddleTennis builds that keep paired evaluation. Env kwargs are
  recovered from the factory closures; no env is built, and it runs in
  0.05 s after import.
- **Trainer digest** (`tools/trainer_digest.py --base <ref>`). It runs
  four seeded `train()` jobs (SAC BallBalance, DemoSAC with demos off,
  the PaddleTennis SAC quick path, PPO BallBalance) on this checkout
  and on a temporary worktree of the ref. It compares policy weights,
  `log_ent_coef`, normalizer stats, monitor, evaluations,
  `eval_info.csv`, selection meta, diagnosis reports, RNG states and
  `resolved_model`. It also compares the best-model artifacts that
  replay, evaluation, warm start and promotion load: the policy weights
  and torch variables (`log_ent_coef`) read back from `best_model.zip`,
  `best_vec_normalize.pkl`'s stats and settings, and the selection
  fields and bound artifact names of `best_model_meta.json`. The
  meta's sha256s cannot be compared across sides, since SB3 zips embed
  timestamps, so each side checks them against its own files, as the
  loaders do, and a failed check never counts as a match. Finally it
  compares which `RUN_LAYOUT` artifacts exist and the checkpoint file
  names. Both sides take about 100 s together, under
  different hash seeds (`PYTHONHASHSEED` 1 and 0), so a str-hash-order
  dependence shows up as a mismatch.
- **Typecheck.** Run mypy as CI does: `python -m mypy` in an
  environment with the train extra (CI: Python 3.11). With
  `ignore_missing_imports`, a mypy that cannot import numpy, Gymnasium
  and SB3 (a global or `uv tool` install) treats them as `Any` and
  passes. That is how three errors reached CI's typecheck (an
  `Env.data` access, an `np.full` shape that may be `None`, and
  `BaseAlgorithm.log_ent_coef`), each now narrowed with a typed check.
  CI runs `python -m mypy`, and `tests/test_packaging.py` pins that and
  the CPU-torch order, reading the workflow's install lines through
  quoting and flag variants.
- **Names.** The PPO design's Step 0 names `tests/test_sac_freeze.py`
  and `tools/sac_freeze_digest.py`. Part 1 shipped as
  `tools/recipe_snapshot.py` plus `tests/test_recipe_snapshot.py`, and
  part 2 as `tools/trainer_digest.py` plus `tests/test_trainer_digest.py`.
  A later step's "must pass Step 0" means those four files.
- **Evidence.** Every paddle, WallBall, BallBounce and BallBalance
  digest in both sets is bit-identical across MuJoCo 3.15.0/3.16.0,
  Python 3.11/3.13 and numpy 2.4.6/2.5.x. The humanoid random stream
  diverges at step 49 under 3.16, which is the body-BVH refresh behind
  the `<3.16` cap (Humanoid section). Against `4e2e019`, every trainer
  component matches, and DemoSAC-off equals SAC hash for hash.
  Format 2 (all info items, the interface digest, four new cases)
  re-recorded every entry, primaries under MuJoCo 3.15. All 30 format-1
  entries kept their `stream_sha256`, step, episode, ending and
  reward-sum values, so the re-record changed no stream. The format-2
  CI set matches 19/19 under MuJoCo 3.12.0, 3.15.0 and 3.16.0, Python
  3.11 and 3.13, numpy 1.26.4, 2.4.6 and 2.5.x, and gymnasium 1.1.1 and
  1.4.0; the full set matches under 3.16. With HEAD and the base on
  different hash seeds, every trainer component still matches. With
  the seed-once cases, the 22-case CI set matches 22/22 under MuJoCo
  3.12.0, 3.15.0 and 3.16.0, under Python 3.11 with numpy 2.4.6, and
  with AVX-512 or all SIMD dispatch disabled.
- **Decisions.**
  - Primary digests are recorded under the supported MuJoCo (what CI
    resolves). A version-specific difference lives only in a
    declared-drift slot whose reason is in code (`DECLARED_DRIFT`), and
    every other version must reproduce the primary.
  - The PaddleTennis camper is the frozen reach-camper witness. A
    camper "at the home target" is the zero action exactly (zero maps
    to the pivot), so it would only duplicate the zero digest.
  - `recipes.py` imports the training stack inside
    `build_train_config`, so the registry, and therefore the digests,
    load without SB3 or torch. All 70 resolutions are identical to
    those of `4e2e019`, the commit before the import move.
  - `--write` leaves every reproduced entry untouched, versions
    included, and prints each change. The mismatch text says to
    declare, never to re-record. Entries recorded before `numpy_simd`
    joined `version_info()` keep versions without it: the CPU that
    recorded them is unknown, so it is not backfilled.
  - Info values with no canonical form raise instead of being skipped.
    Every info key counts in the census, so adding, renaming or
    removing an array- or tuple-valued item is visible.
  - The digest and snapshot tools, and the test suite (in
    `tests/conftest.py`), stop when `courtside_dynamics` imports from
    outside this checkout's `src/`. In a git worktree the editable
    install resolves to the main checkout, and a mutant there used to
    pass the worktree's gates. Run them with `PYTHONPATH=src`.
  - The recipe snapshot compares canonical JSON text, so a type-only
    change (`8` to `8.0`, `True` to `1`, `0.0` to `0`) is a diff.
- **Lesson: perturb in the precision the computation runs in.** The
  self-test's one-ULP learning-rate bump (float64) on the PaddleTennis
  recipe left every trained output bit-identical: 3e-4 and its float64
  neighbour round to the same float32 step size. Only `resolved_model`
  caught it, which is why that component exists. A float32 ULP moves
  the policy, `log_ent_coef`, normalizer, evaluations, selection and
  diagnosis.
- **Not covered.**
  - The trainer digest is a manual pre-merge gate, not CI, because it
    takes minutes. It does not exercise warm start or resume, DemoSAC
    with demos on, or the TOML run-config path.
  - Monitor columns can stay equal under a real change when the tiny
    budget's episodes end before learning matters; the weight and RNG
    hashes carry those cases.
  - Colab digests are not yet recorded (M1 asks for Colab vs local).
    Run `python tools/golden_digest.py --full` in a Colab runtime and
    record the result.

### The 2026-08-28 review's after-LT1 robustness findings are closed — *implemented (unreleased, commit `0cdd79f`)*
Commit `0cdd79f` closes these findings from
[`rl_pipeline_review_20260828.md`](rl_pipeline_review_20260828.md) (the
snapshot itself stays unchanged; a 2026-09-27 audit had confirmed none
had landed): §2.1 (miskeyed selection keys), §2.2 (`eval_info.csv`
append), §2.3 (`require_device`), §2.5 (provenance writers), §2.6
(normalizer fallback), §2.13 (salvage tests), plus from §3 the non-KI
crash salvage, `validate_model_kwargs` in `train()`, and the
smoke-wheel floor (4 → 5). Still open from the after-LT1 class: §2.4,
§2.7, the rest of the §3 register, and the §6 cleanup sweep.

### Certification artifacts must be derived from the live spec, never maintained in parallel — *implemented (0.23.0)*

The 0.22.0 ladder shipped without a calibration sweep because
`tools/depth_stage_sweep.py` carried its own copy of the stage tables and
nothing forced that copy to track the recipe; the drift surfaced only when
review `20260727_233859` re-swept the live geometry after a 17-hour run
stalled at stage 1 (oracle clears the 3.0 bar at **no** stage; feasibility
fails at stages 0 and 3). Evidence that parallel tables rot: the tool's
pivot (−1.7) and fences were two releases stale, `wall_ball_oracle_action`
still inverts the retired fixed-pivot map, and the recipe's own comment
carried share numbers matching nothing. Since 0.23.0, gated runs
self-certify at startup — stages read from `performance_gate`, envs built
by the run's `env_fn`, verdict written to
`reports/ladder_certification.json` — and stage-application integrity
(attribute exists + round-trips) is a blocking criterion, closing the
`paddle_home_x` silent-no-op class for good. Certification is advisory
until the probes are recalibrated for the constant-width geometry:
enforcing criteria that the stock probes are known to fail would brick
legitimate runs.

### An argmax over a logged eval stream is a hypothesis, not a result — *characteristic*
Second occurrence in this campaign, so it is a pattern rather than a
mishap. Run `20260727_004014`'s unsynced goal-geometry stream made the
late-run region look better than the selected checkpoint (8-eval trailing
means: 1.163 at the selected 4,650,000 vs 1.283 at 5,750,000 and 1.254 at
6,000,000), and the run was called wrong for having selected on
matched-stage geometry. Re-scored on **200 fresh paired seeds**
(`design_wall_ball_checkpoint_selection_audit.md`), `final_model` came
back **worse** (−0.080, t = −1.80) and the best candidate was not
significant (+0.085, t = +1.62, p ≈ 0.11 before any multiplicity
correction). The trailing means had been computed from a 30-episode
estimator over the same 240 evaluations the maximum was drawn from. The
first occurrence is in `plan_wall_ball_aligned_deep_stages.md`: a control
arm's calibration argmax (94.0%) fell to 89.0% held-out and failed while
the stale shipped value passed at 94.0%. *Rule: re-score the candidate
region on seeds no selection has touched, pair the arms on identical
seeds, report the paired statistic, and size the re-score against the
effect claimed — 3–5 point calibration-to-held-out swings at n = 200 are
routine here, which is the same order as every margin that has mattered.*
Two useful side effects of that audit: `best_model.zip` **stands** (no
re-selection), and four checkpoints spanning 1.35M steps all landing at
1.07–1.30 returns is direct evidence the depth plateau is structural, not
a choice-of-snapshot artifact.

### Per-stage promotion price, not total budget, is what binds the depth campaign — *characteristic*
Run `20260721_004722` reached stage 2 of 4 on 3M steps; run
`20260724_152530` was still on stage 1 at 2.55M of a 6M budget, with
stage 0 costing 1,425,000 steps in *both* (within 0.02%). Doubling the
budget did not move the ladder position, so the lever is the cost of each
promotion, not the total. Two of the three components are self-inflicted:
promotion-shock recovery (~0.5–1M steps per advance) and gate latency.
Note also that `20260724_152530` cold-started (`warm_start: null`)
despite run 1's review making warm continuation its P0 #1 and 0.15.1
shipping SAC warm start — so it re-paid stage 0's 1.43M-step bill for
nothing. **Lesson:** for a gated ladder, measure steps-per-promotion and
project it against the budget *before* launching; a cold start on an
already-mastered stage 0 is ~24% of a 6M run spent re-learning a solved
problem.

### Resetting the entropy temperature on advance — *implemented (0.20.0, default off)*
`train/ent_coef` sat at 0.0007–0.0009 from ~100k steps onward in run
`20260721_004722` and never re-inflated after a promotion, so the policy
met each new geometry very nearly deterministic. The warm-up package made
that worse: `advance_update_pause_steps` sets `gradient_steps = 0`, which
freezes `log_ent_coef` along with everything else, so the tuner could not
recover *during* the pause either. Run 1's review filed this as a watch
item gated on "if a de-noised continuation still recovers slowly from
stage shocks"; `20260724_152530` fired it (stage 1 >1.4M steps, vs 0.97M
for the same stage in run 1). Shipped as the `reset_entropy_on_advance`
gate key. Two implementation notes worth keeping: clearing the Adam moment
buffers matters as much as restoring the value (stale moments push the
restored coefficient straight back down), and this restores *pressure*
to re-expand entropy — sampling spread lives in the policy's learned
`log_std` and re-expands over subsequent gradient steps, so expect a ramp,
not a step.

**Default-on for both depth recipes from 0.22.0.** The lever shipped in
0.20.0 and was then never exercised: run `20260727_004014` finished with
`train/ent_coef` at **0.0011** after 97 evaluations on stage 3, so every
one of its three advances still handed new geometry a policy with no
exploration budget. Legal there because `model_kwargs` pins only `gamma`,
leaving SB3's `ent_coef` on `"auto"` — a recipe that pins a float
coefficient cannot use this key.

### A sliding fence turns a growing slab of the action range into a zero-gradient plateau — *characteristic; remediated (0.22.0)*
`_action_to_controls` clamps the *target*, not the mapping
(`wall_ball.py:1097-1105`) — correct, and it keeps action semantics fixed
across stages. The consequence, unrecorded until now: with
`paddle_x_target_range = (-4.7, 0.3)` and `paddle_home_x = -1.7` the map
is piecewise linear about `a = 0`, so the fraction of the x-action range
that saturates against the fence grows **33.3% → 52.5% → 64.2% → 70.0% →
71.7%** across stages 0–4 (live band shrinking 66.7% → 28.3%). Inside a
saturated region `∂target/∂a = 0`, so Q is flat and the actor's
reparameterized gradient vanishes. The policy a fence slide clamps is
exactly the front-camper the ladder exists to break, and once clamped it
has no gradient pointing deeper — it can only escape on action noise,
which by then is nearly gone (see the entropy entry above). This is a
**third** promotion-shock mechanism alongside the two already recorded
(stale replay buffer, unobservable fence).

**Remediated in 0.22.0**, after run `20260727_004014` paid the predicted
price (three promotions, then 97 evaluations stalled on stage 3). Two
independent re-derivations reproduced the band above to the digit, which
is worth noting only because the mechanism was already on record here and
was rediscovered from the run data rather than read off this entry —
*check this journal before re-deriving.* The fix has two halves.
`paddle_home_x` became a property whose setter recomputes
`_control_home[0]`: it had been a plain attribute, so a stage that moved
it was a **silent no-op** (`set_wrapper_attr(..., force=False)` reports
success for any existing attribute) — cardinal rule 1, again. Each stage
now pivots the map on its own fence midpoint, holding the live band
roughly flat (0.49 / 0.43 / 0.42 / 0.48 / 0.63) instead of collapsing
0.67 → 0.28. `paddle_x_target_range` is untouched, so action *scale*
still never drifts across stages — the property this entry was written to
protect survives.

### Don't pool the `confirm_best` batch into the promotion window — *built, then rejected (0.20.0)*
Tempting and wrong. `confirm_best` re-rolls a full `n_eval_episodes` batch
on the current stage before dethroning a best, and the gate never saw it,
so folding it into the promotion window looks like free evidence — the
switch was written, tested, and reverted before shipping. The flaw: that
batch is **conditionally sampled**. It only runs when the primary batch
beat the running best, so pooling averages a deliberately-selected *high*
draw with an independent one (regressing it toward the mean) while leaving
low draws untouched. The window mean is therefore biased downward, which
silently *raises* the bar by an unquantified amount — precisely what run
1's review forbade ("do not lower the bar; de-noise the estimator
instead", and raising it invisibly is worse). Simulated at the campaign's
own numbers (true mean climbing toward a 3.0 bar, per-episode std 2.0,
n=30, 3-eval window mean, `min_delta` 0.5/30): mean window entry
**3.005 → 2.954** (bias −0.046), and evaluations-to-promotion **41.8 →
47.6** on a fast climb and **132 → 159** on a slow one — **+27 evals ≈
+666k env steps per promotion** in the regime Run A is actually in, on a
campaign whose binding constraint *is* steps-per-promotion. **Lesson:**
extra episodes only reduce variance when whether you collect them is
independent of what the first sample said. The honest way to buy gate
evidence is unconditional — raise `n_eval_episodes` or `sustain_evals`,
or pair the episode set (next entry). `last_confirmation_metrics` is still
published as diagnostic surface; nothing consumes it for decisions.

### Unpaired evaluation is the root of the gate noise — *open (P1)*
`InfoDictEvalCallback` calls `self.eval_env.reset()` with no seed and
never re-seeds (`info_dict_eval.py`), so **every evaluation draws a fresh
30 episodes**. Consecutive evals are therefore unpaired and the full
~0.4-bounce batch SE lands on every promotion decision and every
best-model comparison. The project's response was to widen the promotion
window (2 → 3 evals), which buys reliability at 75k steps of latency per
decision — treating the variance rather than removing it. Re-seeding the
*matched* stream to a fixed episode set each evaluation makes the
comparison paired (common random numbers): the variance of eval-to-eval
*differences* collapses, and `sustain_evals` could go back to 2. Keep the
final-config stream fresh-random, where an unbiased estimate is what is
wanted. Recommended alongside 0.20.0's `pool_confirmation_samples`, which
attacks the same SE from the sample-size side.

### Three eval streams, two of them on the same distribution — *implemented (0.20.0)*
Under headline selection a gated run stood up three periodic evaluators;
the reward `EvalCallback` and the final-config info-eval both rolled the
recipe's `eval_env_overrides` distribution (the gate re-syncs only the
matched evaluator). For run `20260724_152530`: 5 + 30 + 15 = 50 episodes
per 25k training steps, ~17,500 env steps of evaluation per 25,000 of
training at ~350-step episodes — and up to ~112% overhead on evals that
trigger a confirmation. `InfoDictEvalCallback` was already a strict
superset of the reward evaluator (it collects per-episode returns for its
own reward tie-break), so the duplicate is retired and the final-config
stream owns `evaluations.npz`. 0.20.0 also makes the single-worker
requirement explicit: the rollout loop reads `infos[0]`/`rewards[0]` and
counted worker 0 only, so a multi-worker eval env was stepped in full and
three quarters of it went unmeasured — now rejected at construction.
**Still open:** all eval envs are built `n_envs=1` (`train.py`), so
evaluation runs batch-1 and serial while training runs 8-wide. That is the
real wall-clock lever (~21,000 serial eval steps per 25,000 training steps
at the 0.20.0 episode counts), and claiming it means rewriting the rollout
loop to aggregate every worker — not passing a bigger env.

### `config.json`'s `train_config` is a hand-maintained allowlist — *partially implemented (0.20.0)*
`artifacts.py` enumerates the fields to serialize by hand, so a new
`TrainConfig` field is silently absent from every run's provenance
snapshot until someone remembers to add it. `reward_eval_episodes` — set
to 5 by two recipes and directly changing how much evidence a run's
reward stream collects — was missing for its whole life. 0.20.0 records
it and `final_eval_episodes`, and a test now pins that the block covers
every field except the four code-valued ones (`env_fn`, `eval_env_fn`,
`extra_callbacks`, `info_row_fn`) and the two recorded at top level
(`recipe_name`, `run_config_file`) — in both directions, so a derived
value cannot be smuggled into the block either. Adding a field without
recording it is now a test failure rather than a discovery made months
later while reading a run.

### `final_stage_index` reports the departing stage mid-run — *open (P3)*
`_write_stage_history` reads `self._stage_index` but is called from
`_close_stage_record`, which runs *before* the increment on an advance, so
a live or hard-killed run's `curriculum_stages.json` names the stage that
just ended. Run `20260724_152530` showed `final_stage_index: 0` while
`best_model_meta.json` recorded `curriculum_stage_index: 1.0`.
Self-corrects at `finalize()`; only misleads while a run is in flight.

### Instrument for post-hoc attribution — *mostly open (P2)*
Recurring pain: Monitor CSVs persisted only `r,l,t` (no `info_keywords`), so
training-time terminations couldn't be attributed after the fact; `*_ep_mean`
eval aggregates are means of *final-step* values (instantaneous flags read as
nonsense); `one_bounce_recovery_count` incremented on every first paddle hit
(reads ~1.0 with zero rallies); rollout/video used a single fixed seed
(n = 1 behavioral record per checkpoint). Recommended: pass
`info_keywords=("term_*","rew_*","bounce_count","reset_mode_id")` to Monitor,
add per-episode *sum* aggregates for `rew_*`, and vary the video/rollout seed.

### Provenance: record the git SHA correctly and bump the version on behavior changes — *partially implemented*
`artifacts.py` ran `git rev-parse HEAD` without `-C <repo>`, so `git_sha` was
`null` unless launched from the repo CWD (the SHA-capture fix landed on the
post-review branch). Worse for reproducibility: one run's `config.json` reported
`v0.8.0` while running features merged *after* the 0.8.0 bump (commit `1eec0d4`)
— so its "0.8.0 vs 0.9.0" comparison framing is wrong. **Rule:** bump the
package version on any behavior-changing merge, and never trust a version string
that isn't backed by a recorded SHA.

### Cut the redundant eval pass; add a curriculum-matched eval stream — *open (P1/P2)*
When headline selection is active, `EvalCallback` is reporting-only yet still
runs 30 deterministic episodes every 25k steps (~15% of run env-steps).
Recommended: cut it to ~5, raise the *selection* stream toward 50, and add an
eval stream on the *current training reset distribution* — the gap to the
canonical from-serve eval is the transfer deficit, visible within 1–2 evals. A
`deterministic=False` arm would also separate "mean policy degenerate" from
"competence lives in action noise."

---

## Humanoid tennis environment

Source: [`humanoid_env_review.md`](humanoid_env_review.md) (main @ `0d294f2`,
v0.7.0, 2026-07-13). Environment correctness, determinism, and physics rated
strong (trajectories bitwise-reproducible; Stages 0–1 proven solvable by
scripted oracles); **learning feasibility was blocked as configured**. Statuses
below were last updated at 0.16.0; items still marked open were spot-checked as
still present in the current tree.

### MuJoCo is capped below 3.16: its bounds refresh moves the Stage-1 oracle — *implemented (unreleased, 2026-10-16)*
MuJoCo 3.16.0 (2026-10-14; changelog item 7, commit `d823f9212`) made
`mj_setConst` compute the bounding volumes of bodies as well as geoms. The
env enlarges each racket stringbed 1.5× at construction and refreshes the
geom's own bounds (entry below). Through 3.15, though, the racket *body's*
BVH (`bvh_aabb`) kept the nominal stringbed: in the racket frame its root
box spans |y| ≤ 0.152 m and z ≤ 0.322 m, against the scaled stringbed's
0.195 m and 0.385 m. Contacts in that outer strip are still culled on
≤3.15, so the 0.16.0 forgiveness fix is only partial there. The Stage-1
oracle (129 steps, landed target) was calibrated on those stale bounds.
On 3.16 it takes 138 steps and misses the target, which also aborts the
humanoid notebook's Stage-1 preflight.
- **Evidence.** Installing only the other version's `bvh_aabb` makes the
  3.15 and 3.16 oracle streams bit-identical in both directions (obs and
  reward sha256, both serve sides). No other compiled array the two
  versions share differs, except the declarative `actuator_ctrlspec`.
  PaddleTennis, WallBall and BallBounce per-step digests are identical
  from 3.12 through 3.16 (`repo_review_20261016.md` §1).
- **Floor evidence.** The golden CI set (22 cases, seed-once cases
  included) matches 22/22 on 3.12.0 and 21/22 on 3.11.0, where the
  humanoid random stream moves. On 3.10.0 it matches 1/22: every
  PaddleTennis, BallBounce, BallBalance and humanoid case moves, and
  so do both WallBall TrueBaseline oracle cases; only its random
  stream reproduces. (The gates' review measured the format-1 set at
  the same 1/15 there, the one match being that WallBall random case.)
  On 3.3.0 the format-1 set matched 0/15, and the Stage-1 oracle takes
  127 steps instead of 129.
- **Decision.** Bound `mujoco>=3.12,<3.16`: the supported range is
  exactly the range that reproduces the golden digests, so "every other
  version must reproduce the primary" holds on all of it. 3.12 is also
  the August training stack (`repo_review_20261016.md` §2.1). The floor was `>=3.3` until the gates'
  review, which found it admitted versions the goldens fail on.
  `tests/test_packaging.py` pins both ends. The oracle really misses on 3.16,
  so this is not a test-expectation artifact, and the test is not
  loosened. Refreshing the body BVH in the env would change the humanoid
  stream on every supported version and still need a recalibrated oracle:
  a declared task change, not hygiene. The review (§2.1) advised against a
  global cap. On this evidence it costs the paddle work nothing, and
  3.15.0 ships wheels for cp310–cp315.
- **Lift when** the humanoid code is archived (design M8), or the env
  refreshes the racket body's bounds itself and the oracle is
  recalibrated on them. The golden digests (M1) then prove the paddle
  streams unchanged under the new MuJoCo. Notebooks pinned to
  `REPO_REF` = `b9585e2` still resolve 3.16 until that pin moves past
  this change.

### Update `geom_rbound`/AABB when resizing a MuJoCo geom at runtime — *implemented (0.16.0)*
The stringbed enlargement for "early contact forgiveness" writes `geom_size` and
calls `mj_setConst`, which does **not** recompute the broadphase bounding sphere
(`geom_rbound`) or midphase AABB. MuJoCo culls contact pairs on the stale
bounds, so the enlarged outer band never generates contacts — the forgiveness is
essentially inert at the racket tip and the recorded `racket_contact_scale`
metadata is misleading. Videos show the ball passing through the visibly-enlarged
racket. Fix: recompute `geom_rbound`/AABB after scaling, or compile the scale
per-stage via `mujoco.MjSpec`; add a regression test firing a ball at the
enlarged-only zone. *(0.16.0 recomputes both bounds from the scaled semi-axes
when scaling — `mj_setConst` does not refresh them;
`test_racket_forgiveness_band_generates_contacts` pins a ball contact in the
enlarged-only annulus, and the stage-1 oracle was recalibrated for the
corrected contact timing. Still partial on MuJoCo ≤3.15, where the racket
body's BVH keeps the nominal size; see the entry above.)*

### Sparse reward + iid per-step Gaussian exploration at 100 Hz = zero gradient — *partially implemented (0.16.0)*
0 valid hits / 0 nonzero rewards in 264 random episodes; the end-to-end 25k run
scored 1 success in 177 episodes, all evals `0.000 ± 0.000`. The tell: a
*constant* random action held for a whole episode succeeds ~15% of the time —
iid noise averages to a near-still arm, so the failure is exploration structure,
not task difficulty. Recommended (several together): `use_sde=True` or action
repeat; enable escrowed `valid_hit_shaping` by default; add distance-to-ball
shaping; extend `episode_len` past the miss-fault so a miss pays −1, not an
indistinguishable 0. *(0.16.0 ships `use_sde=True` + `ent_coef=0.01` and
Stage 1–2 recipe-level `valid_hit_shaping=0.25`; distance shaping, action
repeat, and the `episode_len` extension remain open, and the env-level
shaping default stays 0.0.)*

### A flat −1 for all Stage 1–2 outcomes gives no aim gradient — *partially implemented (0.16.0)*
Under `VALID_TARGET_RETURN`, a whiff, a hit landing out, and a net fault all pay
exactly −1, and valid hits pay 0 (`valid_hit_shaping` defaults 0.0). The Stage 1
oracle under Stage 2 randomization got 12/20 *hits* but 0/20 *target returns* —
hit-vs-aim is the entire Stage 1→2 task and it is unrewarded. (Same shape as
WallBall's flat-valley problem, Cardinal Rule 2.) *(0.16.0 sets the escrowed
`valid_hit_shaping=0.25` in the Stage 1–2 recipes, separating hit-then-miss
from never-hitting at contact time; hit-landing-out vs whiff still terminate
at the same −1, so a graded aim signal remains open.)*

### The zero-action "PD standing hold" only holds welded robots — *open (major)*
The documented "zero action = two-player standing-reference PD hold" is
anchored-only. Free-standing (Stage 6), zero actions collapse both G1s from
pelvis z = 0.78 m to ~0.11 m within ~1.2 s, and no fall fault fires. Free-standing
training begins every episode with an unrewarded, unterminated balance problem —
the Stage 2 → 6 gap is larger than the missing Stages 3–5 imply. Fix: fall
detection + termination and/or an upright-alive reward; treat a standing
controller as the true Stage 3 prerequisite.

### Clamp Wilson-interval bounds into `[0, rate]` / `[rate, 1]` — *partially addressed*
`_wilson_interval(0, 50)` returned a lower bound of ~6.9e-18, violating
`0 <= low <= success_rate` and crashing the promotion gate exactly at the
0%/100%-on-a-side extremes an early curriculum run produces most. The current
function clamps to `[0, 1]` (`max(0.0, …), min(1.0, …)`) but not to the
per-side rate — add gate tests for 0-success and all-success at
n ∈ {25, 30, 50, 100}.

### Isolate the video callback (and fix its render defaults) — *partially implemented*
At 0.7.0 the callback had no error isolation, so a missing GL backend lost the
run's artifacts (Cardinal Rule 7). *(2026-08-02: the isolation actually landed —
an earlier revision of this entry claimed try/except around the rollout that the
code did not have. The whole record pass is now wrapped in
except-Exception/log-and-continue with a regression test, the normalizer-sync
fallback warns loudly instead of passing silently, and the duplicate
`rec_env.render()` — `VecVideoRecorder.step_wait` already captures the frame —
is gone.)* Still worth doing: encode at `env.metadata["render_fps"]` (100), not
60 (replays were 0.6× slow-mo), and pass `camera_name="sideline"` (the default
free camera renders the ball invisible).

### Surface the task metric in `stage_summary`; un-pin `n_envs`; observe world-frame spin — *open (major)*
- `stage_summary.txt` renders `<key>_final` (the last eval *episode's* terminal
  value) and humanoid recipes set no `headline_key`, so the configured task
  metric (`stage_success`) never appears — the first file a human reads shows
  single-episode noise. Render `success_rate`/`*_ep_mean` and set `headline_key`.
  *(0.16.0 sets `headline_key = "stage_success"` on the three stage recipes, so
  selection, early stop, and `best_model_meta.json` follow the task metric, and
  the existing headline block now surfaces `stage_success_ep_mean` in
  `stage_summary.txt`; still open: rendering `success_rate` and the
  non-headline `*_ep_mean` keys, which remain single-episode `*_final`
  values.)*
- Fixed-stage recipes pin `n_envs = 1`; Stage 0 trains 2 live action dims among
  56 inactive (96.6% dead weight), and the 0.5M/1M/2M budgets are ≈
  3.6h/7.3h/14.6h on 4-core CPU (undocumented). Un-pin `n_envs` for PPO
  (callback cadences are already `n_envs`-independent) for ~4–8× wall-clock.
- Observed ball "spin" is free-joint `qvel[3:6]`, a body-*local* frame rotated by
  the unobserved orientation quaternion — topspin and sidespin can read
  identically. Only magnitude is reliable today (bounded because all launches use
  zero spin). Expose world-frame angular velocity under the existing obs names.

### Don't re-litigate these (verified correct) — *characteristic*
Shaping escrow has no double-pay path and is net-zero on Stage 0 success;
selective normalization (indices 0–192 normalized, bounded rally/contact/mask
tail 193–298 raw) is the right design for cross-stage normalizer transfer; the
warm-start path correctly transfers policy + obs_rms while resetting reward
normalization and the optimizer; SIGINT salvage works; replay-from-info is
bit-exact. The many remaining minor findings (rules-engine edge cases, logging
hygiene, CSV column gaps) live in the review doc's §2.3.

### Environment gotcha: a broken CUDA `triton` wheel segfaults the suite on CPU boxes — *characteristic*
A suite segfault at ~90% traced to the container's `triton` wheel (pulled in by
CUDA torch on a CPU-only box), not repo code; uninstalling `triton` gave a clean
run (422 passed, 81 s). Don't chase phantom repo bugs on CPU-only environments.
`MUJOCO_GL=osmesa` works headless (`egl` works but emits cosmetic teardown errors
without a GPU).

---

## Physics reference values

Measured characteristics of the ball/court model (not bugs — reference points for
anyone tuning spin or bounce play). Source: `humanoid_env_review.md` §3.

| Quantity | Measured | Real-world reference |
|---|---|---|
| Court restitution (6 m/s drop) | 0.763 | hard court 0.73–0.76 |
| Vertical COR (fast oblique) | 0.795 | decreases with speed |
| Horizontal speed retained (10 m/s skid) | 83% | hard court ~60–80% |
| Topspin from a 10 m/s skid | 70.6 rad/s (rolling ≈ 242) | — |
| Drag deceleration @ 16 m/s | 5.45 m/s² (env Cd 0.55 → 5.27) | — |

Oblique-bounce friction is low, so spin-based play is only weakly supported today;
raising ball/court friction (including rolling/torsional) is the lever if spin
tactics matter later.

---

## Run-configuration system

Source: [`run_config_file_spec.md`](run_config_file_spec.md) (v1.1, implemented
in 0.13.0) and [`design_court_and_config_updates.md`](design_court_and_config_updates.md).

### One editable TOML per experiment, deep-merged, loud on error — *implemented (0.13.0)*
Motivated directly by Cardinal Rule 1: the `WallBallBootstrap` recipe made the
notebook's `model_kwargs=MODEL_KWARGS` cell a live footgun — an explicit kwarg
*replaces* the recipe's whole exploration package (auto-entropy, `learning_starts`,
buffer) silently. The TOML layer sits between recipe defaults and `quick_test`
(precedence: `recipe < file < quick_test < explicit kwargs`) and **deep-merges**
mapping-valued fields so "tweak one hyperparameter" cannot discard a calibrated
bundle. `performance_gate` replaces wholesale (ordered stage ladders can't merge
element-wise). The loader fails loudly on every class of mistake (unknown key
with a `difflib` suggestion, rejected builder-owned field, typo'd env kwarg
caught by an eager probe env), and `config.json` records the file path, sha256,
and full content.

### Guard against invisible configuration; TOML has no `None` — *implemented (0.13.0, spec v1.1)*
v1 required the config path to always be explicit — an implicitly-discovered
config that silently changes a run is exactly the failure mode this repo keeps
paying for. v1.1 softens this to *assisted-explicit* for the notebook only
(`CONFIG_FILE = "auto"` materializes the packaged starter into Drive on first
use and prints path + sha256 every run); `build_train_config` itself still never
discovers anything. TOML gotcha worth remembering: there is no `None`, so a kwarg
that needs `None` (e.g. `early_touch_penalty` for the legacy terminal rule) uses
the sentinel string `"none"`, converted recursively — and `"true"`/`"false"` are
*rejected* because both are truthy Python strings that would silently enable what
the file tried to disable.

### `weak_return_penalty` in training, strict rule in eval — *implemented (0.13.0)*
So training gets practice reps out of near-misses (12–16% of best-model episodes)
while evaluation keeps the exact task every prior run was measured on
(`bounce_count_ep_mean` stays comparable with the 165358/023737 runs). Paired
with `gamma = 0.995` (credit horizon ~200 steps > one ~130-step exchange,
targeting the now-dominant 54% double-bounce failure), and best-model tie-break
moved from the saturated `ge_2_rate` to the `ge_5_rate` the long run actually
improved (8% → 22%). This is why *learning curves* across 0.12 → 0.13 baseline
runs are not comparable but *eval metrics* still are.

### Court markings are render-only MuJoCo sites — *implemented (0.11.1 / 0.13.0)*
The `court_style` kwarg (`diagnostic` / `tennis` / `none`) and all lane/serve
markers are MuJoCo *sites* — they cannot collide, so they have provably zero
physics/observation/reward impact and are safe to reposition every reset. The
`tennis` style fits a to-size ITF half-court onto the existing 16 m × 12 m floor
with no scaling (wall face = net at x = 3.9, baseline at x = −7.985); the service
line lands at x = −2.50, *inside* the paddle lane — a happy accident worth
keeping. Metrics-producing paths always keep the default style; only the recorded
video differs.

### PaddleTennis freezes as one cooperative alternating-serve point per episode — *implemented (unreleased)*
The phase-P1 env (`CourtsideDynamics/PaddleTennis`) shipped only after P0–P4
measured every number it froze (`paddle_tennis_env_20260802.md`). The
non-obvious calls: **either side's fault pays the shared −1** (a cooperative
rally has no useful "whose fault" asymmetry — an opponent fault usually
punishes a poor incoming ball; asymmetry waits for the scoring phase), **one
point per episode with strict serve alternation** (keeps `crossings` an
un-averaged per-point tail and the side statistics provably 50/50 — the P4
mirror makes alternation exactly fair), **episode_len 1500** (the wall-ball
750-cap lesson: never truncate a healthy rally; the scripted tail ends in
~270 steps), **the bounded rally/contact observation tail stays unnormalized**
(the humanoid stage-boundary variance lesson), and **no shaping at the
freeze** (every pre-evidence shaping term this repo added was later falsified
or exploited). The policy physically trains side A only; P4's bit-for-bit
mirror is the reason that loses nothing, and `serve_side_is_policy` records
the alternation in every info stream. `ladder_certification` stays
WallBall-only — PaddleTennis certifies through the probes harness on the
reserved held-out seed blocks instead.
