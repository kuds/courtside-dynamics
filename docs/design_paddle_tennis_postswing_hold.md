# Design: PaddleTennis escrowed post-swing hold

Status: **design frozen 2026-08-17 before implementation**; the §3
battery and §4 pilot bars are pre-registered here before any probe or
training step. The standing doctrine applies: changes after a
pre-registered run starts void that run as campaign evidence.

## 1. Why this change, and why now

The registered n-point run (`20260816_235141`, from-scratch 3M, seed
1) booked **RK1 FAIL under §5's stop/amend branch**
(`paddle_tennis_registered_run_prereg_20260816.md`): k=2 exchange
survival sat at exactly 1% on eleven checkpoints and 0% on the other
nineteen (count corrected 2026-08-23, from the diagnosis rows; the
original text said nine/twenty-one), while every upstream rung was
climbing or solved —
LS1 gate PASS (the from-scratch bootstrap works), RE3 PASS with k=1
receiving at 81–95% across the last third, first-hit shot quality at
81% crossed / 71% in by 3M (the scripted oracle plays 92%/84%), M
PASS at 77.5%, campaign-record evaluations (best +1.822; this
figure read "+1.85" until the 2026-08-23 review re-derived it from
evaluations.npz). A channel
that stays flat at 1% for two million steps while its prerequisites
triple is not budget-limited; it is reward-limited.

The instrument names the limiting behavior precisely, and has all
run: **post-swing wander**. After its own legal hit, the policy
travels 7.5 m mean (p90 10.7) while its shot is away, so the
opponent's return arrives to an empty court. Both completed
second-exchange events (and all nine k=2 survivals) happened on the
rare points where it stayed home. This window — from the policy's
hit to the opponent's return strike — is the one segment of the
k=2 causal chain the reward does not price:

| segment | paid by |
|---|---|
| move toward the incoming bounce | reach escrow (at the bounce) |
| touch it legally | contact escrow (at the hit) |
| the return crosses and lands in | +1 confirmed return |
| **hold court while the shot is away** | **nothing** |

R2 (inter-point recovery ≤ 6.0 m) was pre-registered as the watch
metric that routes this design: it read 7.5–9.1 m at every
checkpoint. This document is that route: the **post-swing-hold
escrow**, the third member of the event-escrow family, built to the
same contract as its siblings (default off, bit-identical until
enabled, farm-proof by keep/clawback construction).

## 2. Mechanism (frozen)

Two new constructor kwargs on `PaddleTennisEnv`:

- `hold_shaping: float = 0.0` — payment scale. `0.0` (default)
  leaves the frozen task's reward stream **bit-identical**; validated
  finite and non-negative.
- `hold_shaping_travel: float = 4.0` — the accumulated post-swing
  travel (meters of XY path) at which the pay ramps to zero;
  validated finite and positive.

**Window lifecycle.** A side-A legal hit *arms* a hold window: the
env records the side-A paddle head's XY at that step's end and
starts accumulating the head's per-step XY path length. The window
*pays* at the opponent's next legal racket hit (side B in
`valid_racket_hits` — the event that makes the next incoming ball
real):

```
payment = hold_shaping × max(0, 1 − travel / hold_shaping_travel)
```

and disarms (one payment per window). The window *disarms unpaid* at
every point ending and episode ending — a shot that dies before the
opponent strikes it earns nothing for the wait.

**Escrow.** The payment enters the pending-hold escrow. It is *kept*
by the policy's next side-A legal hit — which, arriving after the
opponent's return strike, **is by construction the k=2 hit** — and
whatever is pending *claws back* at every ending path: episode
terminations, truncation, n-point boundaries, and the forced
non-finite guard, exactly the reach escrow's clawback set. Episode
totals therefore satisfy the family's sum identity: `rew_hold +
rew_hold_clawback` totals exactly the hold pay whose follow-up hit
happened. Camping is unpaid (no window without a hit); hit-once-
then-freeze collects a payment but never keeps it (the point ends
before another legal hit) and nets zero.

**Ordering pins** (the reach §2a lesson, applied at design time):

- A side-A legal hit on the same step both *keeps* the prior pending
  escrow (keep runs first) and *arms* the next window (arm runs
  second). A payment coexisting with a side-A legal hit on one step
  is kept immediately, never escrowed — physically the B strike and
  the A follow-up hit are separated by a court crossing, but the
  edge is pinned rather than assumed, with a unit test.
- Travel accumulates from the arming step's end position, so swing
  follow-through counts against the budget; `hold_shaping_travel`'s
  default absorbs it (§2a below).
- A side-B racket hit with **no armed window** (the opponent
  returning a serve or feed) pays nothing: the escrow is strictly
  post-swing.
- The n-point relaunch teleport never crosses a window: boundaries
  disarm before the next point launches, so serve placement cannot
  leak into travel.

New reward components `rew_hold` / `rew_hold_clawback` extend the
decomposition to nine components in `info` and the CSV schema, all
exact zeros when the kwarg is off.

### 2a. Why travel, and why 4.0 m

The payment variable is accumulated **path length**, not
displacement or distance-to-an-anchor: displacement forgives orbits;
an anchor point presumes a "correct" ready position the design has
no basis to freeze (the reach escrow already prices where the policy
ends up, at bounce time — this escrow prices the flailing itself,
which is what the instrument measures as recovery-hold travel and
what the failed channel exhibits at 7.5 m). Follow-through
(~1–1.5 m of carry inside the window) plus legitimate repositioning
(~1–2 m) lands a disciplined exchange near 2–3 m of travel, paying
0.25–0.5 of scale; the measured wander (≥ 7.5 m instrument-metric,
more with follow-through included) pays zero. `4.0` is that
boundary, frozen from the run's band before any probe.

On the k=2 hit the full ladder now pays: kept hold (≤ 0.25) + reach
at the return's bounce (≤ 0.25) + contact at the hit (0.25) + the
confirmed +1 — every segment of the second exchange carries its own
gradient, each farm-proof under its own keep rule.

## 3. Certification battery (pre-registered; fresh seed block
6200–6299, burned by this battery)

- **PH0 — bit-identity.** Default-off lockstep: `hold_shaping=0.0`
  arm against the pre-change definition, exact observation/reward/
  info equality (the family's standing bar), plus the shaped arm's
  non-hold components identical to the unshaped arm's.
- **PH1 — witness economics** (`tools/paddle_tennis_hold_probe.py`,
  100 episodes × lockstep shaped/unshaped arms, seeds 6200+):
  - `statue` and `volley_patting`: zero hold pay, zero clawback —
    no window ever arms (statue) or survives legally (patting);
  - `camper`: zero everywhere (no hit, no window);
  - a `hit_then_freeze` witness (the farming attempt this design
    must defeat): collects payments, keeps none, **nets exactly
    zero**;
  - `ground_oracle`: collects **and keeps** (the only witness that
    completes second exchanges) — kept pay strictly positive;
  - `hard_slam` and the stacked arm (`hold+reach+contact`): exact
    per-episode sum identities; component streams bit-equal to the
    single-shaping arms.
- **PH2 — ending-path invariants** (`TestHoldShaping` unit tests):
  clawback at episode endings, truncation, n-point boundaries, and
  the non-finite guard; window disarm at boundaries and reset;
  the same-step keep pin; kwarg validation; travel accounting
  against hand-computed paddle paths; stacking identity.

Adoption of any recipe default waits, as always, on a pilot verdict
— nothing in this document changes the frozen task or the recipe.

### 3a. Battery results (recorded 2026-08-17; **PH0–PH2 PASS**)

PH0/PH2 via `TestHoldShaping` (11 tests: lockstep default
bit-identity, formula-exact payments against an externally
re-accumulated travel, every clawback path, boundary disarm,
the same-step keep pin, validation) — all green. PH1
(`tools/paddle_tennis_hold_probe.py`, 100 episodes, seeds
6200–6299 now burned):

| witness | paid | clawed | kept | eps paid | eps kept |
|---|---|---|---|---|---|
| statue | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| camper | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| hit_then_freeze | 2.577 | −2.577 | **0.000** | 41/100 | 0/100 |
| ground_oracle | 21.256 | −4.001 | **17.256** | 89/100 | 83/100 |
| hard_slam | 0.023 | 0.000 | 0.023 | 1/100 | 1/100 |
| volley_patting | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| oracle_stacked | 21.256 | −4.001 | 17.256 | 89/100 | 83/100 |

Worst per-episode identity gap 1.11e-16 (float ulp); every payment
coincided with an opponent return strike; the stacked arm composed
the three escrows exactly. The economics read precisely as designed:
the farming witness collects and keeps nothing; the only witness
that completes second exchanges keeps 81% of what it is paid; the
slam witness's 101 hits earned a single kept payment (its shots are
almost never returnable — hold pay follows genuine rally
continuation, nothing else). The §4 pilot is cleared to launch.

## 4. LH1 — the pilot (pre-registered; frozen before launch)

**Shape.** Warm start from the registered run's protected best
(`training_runs/PaddleTennis/sac/20260816_235141`, best checkpoint
step 2,400,000, `best_model.zip` sha256 `838997fb…`,
`best_vec_normalize.pkl` sha256 `d0502c14…` — the loader validates
the pairing), **seed 0** (pilot convention), **1,000,000 steps**,
n_envs 4, eval 25k, checkpoint/diagnosis 100k. The single change
against the source run's task is the hold escrow. Run config, frozen
verbatim (Drive-side copy as `paddle_tennis_lh1_hold.toml`, sha
recorded by provenance as usual):

```toml
# LH1 — post-swing-hold pilot (docs/design_paddle_tennis_postswing_hold.md §4).
# Code-side pairing: seed 0, total_timesteps 1_000_000, warm start from
# training_runs/PaddleTennis/sac/20260816_235141.

[env]
hold_shaping = 0.25
hold_shaping_travel = 4.0

[train]
n_envs = 4
eval_freq = 25_000
checkpoint_freq = 100_000

[train.model_kwargs]
learning_starts = 25_000
```

**Bars (frozen; anchors: the registered run's measured band).**

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| KH1 (headline) | k=2 exchange survival, either parity | ≤ 1% at every checkpoint (no advance over the registered band) | one checkpoint at 2% | ≥ 3% at some checkpoint |
| H1 (mechanism) | recovery-hold travel, instrument metric | never < 6.0 m (the registered band's floor was 4.4–7.9 m after 1M) | (3.5, 6.0) everywhere | ≤ 3.5 m at some checkpoint |
| R1 (retention) | k=1 receiving survival | < 60% at every checkpoint (the transferred stroke broke) | [60%, 80%) everywhere | ≥ 80% at some checkpoint |
| guards | recipe defaults | an abort books KH1 FAIL directly | — | — |

**Decision rule.** KH1 PASS ∧ H1 PASS → adopt the hold escrow as a
recipe default (its own adoption commit + pin tests) and pre-register
the era's RK1 retry. KH1 FAIL with H1 PASS → the hold behavior
trained but did not unlock k=2: the block is elsewhere (opponent
return distribution, timing) — book it and re-diagnose before any
further reward change. H1 FAIL → the escrow could not beat the
wander's attractor at 0.25: one re-pair at `hold_shaping = 0.5` is
sanctioned (the same single-knob re-pair convention as the
exploration era), its verdict re-applying this rule without that
branch. Middles → maintainer's call, documented post-hoc.

**Launch checklist (notebook deltas).** `REPO_REF` = the
implementation commit on main; `ENV = "PaddleTennis"`, `SEED = 0`,
`TOTAL_TIMESTEPS = 1_000_000`, `CONFIG_FILE` = the TOML above; §5
config-build cell adds
`overrides["warm_start"] = WarmStartConfig(source_run_dir=".../20260816_235141")`
(the same cell both prior warm-started pilots used). Validate the
run's `config.json` against this section before the first checkpoint.

### 4a. LH1 verdict (recorded 2026-08-19) and the sanctioned re-pair

Run `20260818_210727` (config validated against §4 at launch; git
`e13a8f8`, TOML sha `4fd84f05…`, warm-start artifacts sha-verified)
completed its full 1M in 6h 26m, zero unsafe. The checkpoint series:

| ckpt | k=1 recv | k=2 | touch | hold travel (m) |
|---|---|---|---|---|
| 100k | 31% | 0% | 17% | 4.97 |
| 200k | 32% | 0% | 19% | 5.36 |
| 300k | 58% | 1% | 32% | 5.15 |
| 400k | 61% | 1% | 33% | 7.04 |
| 500k | 59% | 0% | 32% | 5.82 |
| 600k | 70% | 0% | 36% | 6.36 |
| 700k | 71% | 1% | 37% | 6.07 |
| 800k | 74% | 0% | 37% | 6.91 |
| 900k | 78% | 1% | 41% | 5.81 |
| 1M | 72% | 0% | 36% | 6.04 |

- **KH1 FAIL** — every k=2 reading ≤ 1% (four 1% events; no advance
  over the registered band).
- **H1: no PASS, and the honest reading is that the mechanism did
  not take at 0.25.** No checkpoint reached ≤ 3.5 m; the sub-6.0
  readings that block a by-the-letter FAIL all belong to the early
  churn window when the policy was barely engaging (k=1 31–58%) —
  once engagement recovered, travel returned to 5.8–7.0 m against
  the source's 7.46 m. The bands do not tile this outcome; per the
  standing convention the gap routes to the maintainer.
- **R1: no PASS by the letter** (peak 78% at 900k against the 80%
  bar), not FAIL (≥ 60% from 400k on). The recovery arc was
  LR1-shaped, not L2W-shaped — but incomplete: task-metric selection
  kept **the untouched 25k checkpoint as best_model for the whole
  run** (crossings 5.57 / reward +1.07 at eval 1; nothing later beat
  it; final eval −1.16). Training under the transferred temperature
  (1.7e-4 throughout, `train/std` 0.016) never re-attained the
  source's eval quality, let alone improved on it.

**Decision (maintainer, 2026-08-19): the §4-sanctioned re-pair fires
— LH1b at `hold_shaping = 0.5`,** the single-knob re-pair the rule
pre-declared for exactly this substance. The supporting precedent:
LR1 trained successfully under this identical optimizer regime when
its escrow's signal was dense; the hold signal is sparse (pay only
when the policy's own shot comes back) — dose is the cheapest
untested variable. The named alternative (warm-starting without the
transferred temperature — the L2W review's "strict version" flag)
stays on the shelf as the next single change if LH1b's verdict
re-applies the rule, which it does **without this branch**: the
re-pair is once, per the convention.

**LH1b shape (frozen):** identical to §4 in every respect — same
warm-start source and artifacts, seed 0, 1M steps, cadence, bars,
and decision rule — except the TOML's `hold_shaping = 0.5`
(`hold_shaping_travel` stays 4.0). Drive-side copy as
`paddle_tennis_lh1b_hold.toml`:

```toml
# LH1b — post-swing-hold re-pair at 0.5
# (docs/design_paddle_tennis_postswing_hold.md §4a). Code-side
# pairing unchanged: seed 0, total_timesteps 1_000_000, warm start
# from training_runs/PaddleTennis/sac/20260816_235141.

[env]
hold_shaping = 0.5
hold_shaping_travel = 4.0

[train]
n_envs = 4
eval_freq = 25_000
checkpoint_freq = 100_000

[train.model_kwargs]
learning_starts = 25_000
```

### 4b. LH1b voided: the payment cliff (recorded 2026-08-20), and
LH1c

LH1b (run `20260820_013545`, `hold_shaping = 0.5`, config otherwise
conformant) was stopped early on a proof of degeneracy: its 100k
diagnosis probe is **byte-identical** to LH1's, every behavioral
eval metric through 175k matches LH1 to the last digit, and the eval
rewards differ only in floating-point residue. Two runs with
different reward scales can only train bit-identically if the scaled
component contributed **exactly zero** to every training step — and
it did. The §2 formula `max(0, 1 − travel/4.0)` clamps to zero
whenever the window's travel exceeds 4.0 m, and the learned policy's
post-swing travel (follow-through included) lives at ~6–10 m:
**the ramp is a cliff entirely outside the region the policy
occupies, so the escrow pays nothing and carries no gradient at any
scale.** The occasional sub-budget window under the *deterministic*
eval policy produced the fp-dust reward differences; the *stochastic*
training policy never paid once. LH1b is therefore **void as a dose
test** (it doubled zero), and by the same proof LH1's own verdict is
sharpened: the mechanism was never delivered, not rejected — its
sanctioned re-pair consumed by an outcome the §2a design analysis
failed to anticipate. The design error is recorded plainly: §2a
chose the 4.0 m budget so the measured wander "pays zero," which is
precisely what makes it unlearnable — a shaping term must pay
*something* where the policy already is.

**Amendment (frozen before LH1c): the travel budget moves onto the
occupied band.** `hold_shaping_travel = 12.0` with
`hold_shaping = 0.5`: the current behavior (~6–10 m) pays
0.08–0.25 of the 0.5 scale with a monotone gradient toward stillness,
and a disciplined hold (~2–3 m) pays 0.35–0.45. The linear form is
kept (no code change — both values are constructor kwargs). The PH1
battery re-runs at the amended values on fresh block **6300–6399**
before launch; identities are structural and must hold unchanged.

**Re-battery at (0.5, 12.0): PASS** (recorded 2026-08-20; 100
episodes, seeds 6300–6399 burned; identities exact to 2.22e-16).
The amended geometry pays where the cliff did not: `hit_then_freeze`
now collects in 50/100 episodes (18.478 paid) and still keeps
**0.000** — farming stays dead under the denser pay; the oracle
collects 131.585 and keeps 113.202 (86%) across 85/100 episodes;
statue/camper/volley-patting remain exact zeros; the stacked arm
composes exactly. LH1c is cleared to launch.

**LH1c shape (frozen):** identical to §4/§4a in every other respect
— same warm-start source and artifacts, seed 0, cadence, bars
(KH1/H1/R1 unchanged, scored on the 1M window), decision rule with
no remaining re-pair branch. Drive-side TOML as
`paddle_tennis_lh1c_hold.toml`:

```toml
# LH1c — post-swing-hold with the ramp on the occupied band
# (docs/design_paddle_tennis_postswing_hold.md §4b). Code-side
# pairing unchanged: seed 0, total_timesteps 1_000_000, warm start
# from training_runs/PaddleTennis/sac/20260816_235141.

[env]
hold_shaping = 0.5
hold_shaping_travel = 12.0

[train]
n_envs = 4
eval_freq = 25_000
checkpoint_freq = 100_000

[train.model_kwargs]
learning_starts = 25_000
```

### 4c. LH1c verdict (recorded 2026-08-22): the line closes

Run `20260821_013700` (config validated against §4b at launch; git
`b3667c5`, TOML `paddle_tennis_lh1c_hold.toml` sha `84412dc2…`,
warm-start artifacts sha-verified) completed with one launch
deviation, **declared before any data landed**: the maintainer set
`TOTAL_TIMESTEPS = 3_000_000` for an overnight budget against the
frozen 1M. Handling as declared at launch: the registered verdict
below scores the 1,000,000-step window only; 1M–3M is an
unregistered extension, reported as exploratory observation. One
further named confound: MuJoCo moved 3.11.0 → 3.12.0 between LH1b
and LH1c, so comparisons against the earlier runs carry a
physics-version asterisk. The run finished 3M steps in 19h 51m at
41 FPS, zero unsafe terminations.

**The dose was delivered this time.** Unlike LH1b, the 100k
diagnosis probe diverged from the LH1 replay — the §4b amendment put
the ramp on the occupied band and the escrow carried real gradient.
LH1c is therefore a valid test of the mechanism, which neither LH1
nor LH1b ever was.

**Registered 1M window, against the frozen §4 bars:**

- **KH1 FAIL** — every k=2 reading in the window ≤ 1% (four 1%
  events, the rest 0%); no advance over the registered band.
- **H1: no PASS — the mechanism was paid and declined.** Window
  minimum 5.20 m, but as in §4a the sub-6.0 readings belong to the
  early churn; once engagement recovered, hold travel returned to
  8.1–8.3 m — *above* the source checkpoint's 7.46 m. The bands
  again do not tile the outcome; the maintainer's reading is
  rejection at a delivered dose, not non-delivery.
- **R1 middle** — ≥ 60% once engagement recovered, peak 68% in the
  window, never ≥ 80%.
- Guards quiet throughout.
- Selection echo of §4a: inside the registered window the
  transferred checkpoint was never beaten (eval-series window best
  +1.067 at eval 1 — the 25k transfer itself; eval-info window max
  +1.19).

**Extension observations (1M–3M, unregistered, exploratory):**

- k=1 receiving climbed steadily to **90%** at the 3M probe — the
  warm-started line's best first-return channel.
- k=2 never left the noise floor: two isolated 2% readings (1.3M,
  2.9M) among ~19 extension probes otherwise at 0–1% — the pattern
  probe-sized sampling noise produces, not an onset. **[This reading
  was wrong — corrected in §4d.]**
- Hold travel settled at 7.5–8.7 m (3M probe: 8.71 m mean, p90
  11.37 m). Paid, not bought. [Corrected in §4d: the extension series
  actually spans 6.38–8.71 m.]
- **Training dynamics, the honest counterpoint:** best eval
  **+2.483 ± 0.829 at step 2,425,000** — the campaign record (prior
  best +1.822 across all runs; originally recorded here as +1.85) —
  and `best_model` selected from step
  **2,325,000** (crossings 5.63): the first warm-started run whose
  best came from deep training rather than the untouched transfer.
  Four of the eval-info channel's top-5 sit in the 2.0–2.6M band
  (rank 5 is 1.725M; corrected in §4d — the original text claimed
  the entire top-5) (reward +1.766 and crossings 6.17 at 2.3M; final
  policy 5.97 crossings, 30% success, closing eval +0.976 ± 1.626).
  Whether the live hold gradient improved the optimization or MuJoCo
  3.12 moved the task is not separable post-hoc and no attempt is
  made here. [Separated in §4d: the physics did not move.]

**Decision (maintainer, 2026-08-22): the hold-escrow line closes
without adoption.** The §4 rule re-applies with no re-pair branch
remaining. The three-pilot record: LH1 (0.25, 4.0) and LH1b (0.5,
4.0) never delivered the gradient (§4b's cliff; LH1b void), and
LH1c (0.5, 12.0) delivered it and the policy kept the wander — the
post-swing travel is worth more to the current policy than the
escrow pays, at a scale (0.5) already rivaling the terminal ±1 that
we decline to raise further. The kwargs stay in the env, default
off, certified farm-proof, available to a later era. The k=2
blocker stands, and its assessed locus moves from "unpaid window"
(§1) to **the policy's own post-swing action targets**: the next
probe is diagnosis-side — replay the best checkpoints through the
instrument and read where the paddle is being *commanded* after the
swing — before any further reward-side change. The §4a shelf item
(warm start without the transferred temperature) remains the named
training-dynamics candidate for the next registered attempt.

### 4d. Erratum and post-closure measurements (2026-08-23)

The review snapshot
[`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md)
re-derived every §4c number from the run's raw artifacts and re-ran
the instruments locally against sha-verified checkpoints. The KH1
verdict and the closure decision stand unchanged. Three recorded
observations were wrong and are corrected here:

- **The extension k=2 series.** The diagnosis rows actually read: 2%
  at 1.3M, **2.1M**, and 2.9M; **5% at 2.5M** (4 second-hits in 85
  receiving points, one landing in); 1% at 1.2M, 1.6M, 2.0M, 2.2M,
  2.3M, 2.6M, and 3.0M; 0% elsewhere. The 5% reading — the era's
  strongest — sits at the KH1 (≥ 3%) and RK1 (≥ 5%) PASS-bar values,
  outside the registered window. Re-measured by the review (CPU
  replay, same env kwargs): the same-seed replication reproduces it
  (6.2%, 5/81 — real, not a probe artifact), but on fresh seeds
  5230–5299 the checkpoint scores **1.6% (3/191, 95% CI
  [0.3%, 4.5%])** — off the floor (the registered 1M window ran
  ≈ 0.4%), below every PASS bar, and with no measured edge over the
  crowned 2.325M best (1.1%, 3/274, on the full block). The
  corrected reading: late-run k=2 rose off the floor by a non-hold
  route — real but small, and seed-conditioned at its peak.
- **Extension hold travel** spans 6.38–8.71 m, not "settled at
  7.5–8.7 m".
- **The top-5 claim**: four of five in the 2.0–2.6M band on either
  instrument (eval-info rank 5 at 1.725M; npz rank 2 at 2.925M),
  never five.

Two §4c questions the review answered with new measurements:

- **The MuJoCo confound is retired.** PT1's oracle and LH1c-best rows
  replicate to every printed digit under MuJoCo 3.12.0 versus the
  doc's 3.11.0 — the version change is behaviorally nil on this task,
  so the campaign-record evals belong to the run, not to physics
  moving.
- **The dose is directly audited.** The 2.5M milestone-video per-step
  trace (the one artifact carrying the reward decomposition) shows
  `rew_hold` paying 4 windows for +0.211 total (mean 0.053 of the
  0.5 scale ≈ 10.7 m travel inside paying windows) and **all of it
  clawed back — net kept 0.000**. "Delivered and declined" now rests
  on a logged number. PT1 on the same checkpoint reads cmd path
  25.35 m, 92.4% saturation, 0.290 m/step saccade — the late k=2 is
  not via stillness.

## 5. Seed ledger

**6200–6299 burned** by the PH1 battery at (0.25, 4.0);
**6300–6399 burned** by the §4b re-battery at (0.5, 12.0). Training
seed 0 per the pilot convention; diagnosis stays on calibration
5200+. LH1c burned no new blocks. **4100–4199 remains sealed** —
the registered run's stop/amend booking did not open it, and it
stays reserved for the first run whose registered-result branch
fires.
