# Design: PaddleTennis k=2 drill — training the second ball in its real context

Status: **Proposed — the §3a harvest and step-0 instruments AND the
§2 env mechanism are shipped (2026-08-30, default-off, both D2 arms
selectable behind the flag; KD0-certified: default
obs/reward/termination streams bit-identical by same-code lockstep
and the launch path's RNG discipline certified against an
independent twin generator, with the default info stream gaining
three constant provenance keys; §2a records the implementation
pins, hardened same-day by an adversarial review whose 17 confirmed
findings — one live counter-corruption bug included — were all
fixed with mutation-verified tests). **Re-booked 2026-09-02 on the
Phase 0 gates
([`paddle_tennis_ld1prime_freeze_brief_20260830.md`](paddle_tennis_ld1prime_freeze_brief_20260830.md)):
LD1-as-drafted (§5) is RETIRED before freeze — the campaign routes
injection-first
([`design_paddle_tennis_demo_injection.md`](design_paddle_tennis_demo_injection.md));
the shipped mechanism stays ready as the pre-registered RE-AIM
escalation with the drill OFF at the LD1′ launch, and the §2 D2
fork is deferred, not decided.** Drafted 2026-08-30 from the PT2 routing
([`paddle_tennis_command_spectrum_20260830.md`](paddle_tennis_command_spectrum_20260830.md)
§7, taken up first per the maintainer direction booked with the
command-rate closure), **revised the same day after its own
adversarial design review measured a fact that reshapes the
mechanism (§1a)**. A **training-distribution change with the eval
task bit-frozen** — the smallest era surface yet priced (§6).
Standing doctrine from any freeze: changes after a pre-registered
run starts void it as campaign evidence.

## 1. Why this change, and why a drill

PT2 measured the k=2 blocker as a **directed-interception gap**: on
real second-ball opportunities the trained paddle sits ~3 m from
the return's bounce and converts ~1%, while the oracle sits 0.89 m
and converts 98%. Opportunities are real but modest: **4.5
points/episode mean** (measured from the PT2 streams; min 4, max 6),
~2.2 of them policy-receiving, of which ~60% produce a k=2
opportunity — 1.3–1.7 per episode. A drill — launching a
policy-receiving point directly into a **harvested real k=2
scenario** with probability `drill_fraction` — raises k=2-class
exposure by **~1.6–2× at `drill_fraction = 0.5`** (ceiling ~2.4× at
1.0; ~750 drilled launches in a 1M pilot). Density alone is
therefore *not* this design's payload; what the drill uniquely buys
is stated in §1a, and it was discovered by measurement.

## 1a. The measured fact that reshapes the mechanism: k=2 is context-gated

The design review's matched-pair probe (2026-08-30; scripts
archived at the session workpapers, methodology in §3) took **47
harvested k=2 instants** (the opponent's return one control step
before crossing to side A, from the LH1c crowned best's own PT2
streams) and compared the **same deterministic policy** on the same
physics under two presentations:

- **Real recorded continuation** (rally flags as play produced
  them): **0/47 conversions**, ≤ 1 m at bounce 15.9%.
- **Replayed as a fresh-feed launch** (rules machine restarted, so
  the observation's phase one-hot reads `INITIAL_FEED`): **16–17/47
  legal hits (34–36%)**, touch 38%.

**Magnitude correction (same day, §3a):** the review's probe
launched *simplified physics* — spinless finite-difference ball
velocities, paddle velocities zeroed, opponent paddle parked (the
streams carry positions only). Re-measured with the harvest tool's
**faithful full-physics launches** (real ball spin, real paddle
qpos+qvel, both sides), the feed-context rate is **~7% for both
subjects** (registered 7/102; LH1c 2/30), and applying the review's
simplification stack to the same states raises it to 9–20%
(reaching the review's 36% band on its own n=47 stream states,
within overlapping CIs). So roughly **half to two-thirds of the
headline 34% was easier physics, not context** — spinless,
still-paddle balls evoke the receive program much more readily. The
**context gate survives at corrected magnitude**: faithful
feed-context ~7% vs full-context 0–2% (registered: 7/102 vs 2/102,
§3a) — real, ~3×, but nothing like 0-vs-34. The gate is carried by
the rally-context observation components (indices 24–47, raw and
normalization-excluded: the phase one-hot, `feed_crossed_net`,
`pending_return_crossed_net`, `rally_count`); the 3M companion's
k=1-to-90%-while-k=2-stays-0 dissociation remains the gate seen
longitudinally.

This measurement cuts both ways for a drill, and the two edges are
the design's central fork (§2 D2):

- A **feed-context drill** (harvested physics relaunched as a
  fresh feed) starts from the measured **6.9% step-0 legal-hit
  rate** (registered library, §3a) — inside the low half of the
  from-scratch 3–17% band that organized k=1, though that
  precedent ran under live exploration noise and this policy's
  temperature is collapsed (LR1's collapsed-temperature ADOPT
  started from 18–41% — two different regimes, cited separately).
  Its transfer must still cross the context gate (now ~7% vs
  0–2%, not 0-vs-34), and it *reduces* genuine-context k=2
  exposure at step 0 (a drilled point reaches a real-flag k=2
  moment only via its own drill-feed conversion, vs the
  ~90%-converting serve receive it replaced).
- A **full-context drill** (restore the harvested instant exactly:
  physics *and* rules-machine *and* event-sampler state — measured
  bit-exact, §3a) trains the genuinely failing context — but from
  a **2.0% step-0 rate** (2/102), below the from-scratch band,
  squarely in PT1's F4 regime: gradients need successes the
  deterministic-and-wrong policy rarely produces, and **no
  campaign precedent shows the escrow ladder engaging from ~2%
  touch under a collapsed temperature**.

**Honest consequence, stated before any freeze:** the matched-pair
discovery *strengthens the demonstration/buffer-injection lever's
case* — injected oracle k=2 transitions carry the real context
*with successful outcomes*, which neither drill variant provides.
This document proceeds on the drill per the maintainer's routing,
puts the fork to the maintainer with the measurement in hand, and
names demonstration injection as the routed alternative in the §5
decision rule rather than a someday-fallback.

## 2. Mechanism (proposed — decision points named; D2 is a maintainer fork)

Two new constructor kwargs on `PaddleTennisEnv`, both default-off
(the shipped implementation adds a third, `drill_context`, holding
the D2 fork behind the same flag — §2a):

- `drill_library: str | None = None` — path to a versioned harvest
  artifact; loaded at construction, sha256 recorded into run
  provenance. `drill_fraction > 0` with `drill_library=None` (or
  the converse) is rejected loudly at validation.
- `drill_fraction: float = 0.0` — probability that an eligible
  point launches as a drill scenario; validated in [0.0, 1.0].
  **Defaults are bit-identical off** (KD0), including RNG
  discipline: the eligibility draw consumes no RNG when the drill
  is off, and its stream position when on is pinned at freeze.

**D1 — scenario source: harvested, never synthetic.** The harvest
instrument (`tools/paddle_tennis_k2_harvest.py`, shipped 2026-08-30,
§3a) replays a named checkpoint deterministically and
records, at the **last pre-net-crossing control step** of every
k=2-opportunity return: full ball state (position + 6-dof qvel,
directly captured), both paddles' slide qpos + qvel, **the rules
machine and event-sampler state** (D2's full-context arm needs
them; the harvest captures them regardless so one library serves
both arms), the recorded observation vector (KD1's fidelity
target), and metadata (source run/checkpoint shas, seed, step, the
return's recorded first-bounce position). **Harvest seeds: a
declared scratch block, NOT calibration 5200+** — harvesting from
the diagnosis seeds would make the in-run diagnosis instrument's
own scenarios training data (train-on-test against the campaign's
primary k=2 instrument). This document proposes booking
**9000–9199 as the campaign's scratch/workpaper block** (the
2026-08-30 feasibility and review probes already consumed
9000–9029 and 9100–9146 there; §7 books them), with the harvest
drawing from its unconsumed range at freeze.
**Library size is a named constraint:** ~100 replay episodes yield
only **~120–150 entries**, i.e. ~5–6 bit-exact repeats per entry
across a 1M pilot's ~750 drilled launches — memorization is
representable and KD3's marginals cannot see it. The freeze sets
the harvest episode count to reach a maintainer-chosen
repeats-per-entry bound, and the library ships with a
**train/held-out split** (held-out entries never launched in
training; drilled-point metrics on them are the memorization
control).

**D2 — the fork: what "launching a harvested scenario" restores.**
- **(a) Feed-context** (the original draft): physics only; fresh
  `RallyStateMachine(serving_side=B)`; the drill ball is the
  point's feed. Measured step-0 on the registered library, faithful
  full physics: **touch 6.9%, legal hit 6.9%** (§3a; the review
  probe's 34–38% was its simplified-physics variant, §1a).
  Cheap (reuses `_launch_point`'s mid-episode body with the drill
  draw replacing `_draw_serve()`; feasibility measured, §3), but
  carries the §1a transfer discontinuity and the genuine-context
  exposure reduction.
- **(b) Full-context** (the revision's recommendation to *evaluate
  first at the battery*): restore the harvested instant exactly —
  physics, rules machine, event sampler — so the launch
  observation equals the recorded observation (KD1 asserts it
  within tolerance). Trains the real failing context; step-0 touch
  ~0%; the F4 engagement risk is priced in §1a and scored by the
  §5 mechanism observables against the step-0 baseline rather
  than assumed away.
- The maintainer picks the arm (or a mixed fraction, its own
  decision) **at freeze, with KD-battery step-0 rows from both
  arms in hand** — both are cheap to measure before any training.

**D3 — eligibility: policy-receiving points only**, drilled with
probability `drill_fraction`; serving-side points untouched; the
serve-alternation ledger is preserved by construction (the drill
substitutes *what* B's feed is, never *whose* feed it is). Whether
the episode's *first* point (the `mid_episode=False` reset path) is
drill-eligible is pinned at freeze; the reset-info contract gains
the `drill_point` marker and the library entry index so every
launch stays reproducible from the info stream (the standing
serve-provenance contract, extended).

**D4 — state written at a drill launch and the carryover price.**
Ball qpos/qvel and both paddles' slide qpos + qvel from the
harvest tuple (self-consistent real joint state); arm (b)
additionally restores rules/sampler state. **This suspends the
n-point era's position-carryover law on drilled boundaries** — the
inter-point wander price the n-point design exists to charge
vanishes on ~half of receiving boundaries at `drill_fraction=0.5`,
SAC bootstraps the boundary fault step's Q-target into a
teleported state, and the R2/inter-point-recovery instrument reads
garbage across drilled boundaries. All three are **priced, not
hidden**: the class-split instrument extension (§5 prerequisite)
splits every boundary metric by `drill_point`, the drilled-boundary
fraction is a recorded observable, and the bootstrap-across-
teleport term is accepted for the pilot (recorded, revisited if
LD1's training-health series degrades).

**D5 — clearance at harvest time, not launch time.** Measured: all
60 feasibility tuples put the ball ≥ 1.11 m from the harvested
side-A head (median 4.7 m) — real joint states are clear by
construction. The clearance filter therefore runs **in the harvest
tool** (drop any tuple violating the envelope against either head;
count recorded in library metadata for KD3); the launch-time check
becomes an assert, and the serve-fallback path
(`drill_fallback_count`, `drill_point` ∈ {1.0, 0.0} in info) exists
only for the assert's failure, fail-loud per cardinal rule 1.
*(Shipped divergence, §2a: the launch-time check is a counted
serve-fallback rather than an assert — fail-loud is carried by the
load-time validation at construction plus the KD4 gate requiring
`drill_fallback_count` = 0; the freeze may reinstate the assert.)*

**D6 — training-only wiring, pinned three ways.** The recipe sets
the drill kwargs on the training env only; its `eval_env_overrides`
force them off (the PaddleTennis recipe gains that dict); and
`validate_run_config_against_plan` pins both kwargs plus the
library sha. Selection, periodic/final eval, and the checkpoint
diagnosis all run the frozen standard task (verified against the
current wiring: every decision-feeding reader is eval-side).

**D7 — library staticness.** Static for the pilot; periodic
re-harvest is the named follow-on, only if LD1's verdict reads as
distribution-drift starvation.

## 2a. Implementation pins (shipped 2026-08-30, default-off; binding unless the freeze overrides them)

The mechanism landed in `PaddleTennisEnv` ahead of the freeze so the
battery can run the moment the maintainer decides D2 — behind
`drill_library=None`/`drill_fraction=0.0` defaults, the same
default-off pattern as every escrow. Pins the code adds to §2, each a
freeze input rather than a freeze:

- **Both D2 arms behind one kwarg**: `drill_context ∈ {"feed",
  "full"}` (default `"feed"`; only meaningful with the drill on —
  the fork stays the maintainer's, the flag just makes both arms
  measurable through the same battery).
- **RNG stream order, pinned**: on a policy-receiving launch with
  the drill on, the eligibility uniform draws first, then (when
  drilled) the entry index — before any serve draw. Policy-serving
  launches and the drill-off env draw nothing extra (KD0's
  discipline — certified in
  `tests/test_paddle_tennis.py::TestK2Drill` against an INDEPENDENT
  twin generator replaying the specified draw sequence, since a
  same-code lockstep cannot see a draw both arms share; the
  stream-order, extra-draw, and swapped-draw mutants are all
  measured killed).
- **First-point eligibility ON** (D3's open pin, pre-freeze
  default): a policy-receiving episode reset drills like any other
  policy-receiving launch.
- **The full arm restores context, never the episode frame**:
  physics + rules machine + event sampler + solver warm-start from
  the harvest tuple; the episode's own clock, step budget, reward
  escrows, and every published point/crossing counter are untouched
  (a drilled point substitutes what the point *is*, not where the
  episode stands). Consequences, measured on the registered library
  through the real launch body: the launch observation reproduces
  the recorded one to ≤ 8e-6 (max 7.903e-6) on every component
  except index 35 (`episode_remaining_fraction`, which reads the
  drilled episode's own clock), and the harvested rally's crossing
  count is absorbed by an INTERNAL continuity offset
  (`_crossings_offset`) so the episode's `crossings` counter never
  jumps while `completed_point_crossings` keeps its
  completed-point meaning — the offset was originally rebased onto
  `_crossings_base` itself, which corrupted that published key to
  negative values; caught by the same-day adversarial review,
  fixed, and pinned by a mutation-verified regression test. Escrow
  non-restoration is exact for this scenario class: all 102
  registered entries carry zero pending escrow at the harvested
  instant (the k=1 advance is already confirmed by the time the
  opponent's return is in flight).
- **Load-time validation, fail-loud at construction**: schema
  `k2-drill-library-v0`; per-arm required fields; every entry
  harvested from a policy-receiving (side-B-serving) point — D3's
  slot contract, and true of all 102 registered entries; every
  entry's harvested rally-rule profile matching the env's
  `volley_rule` (mixing profiles inside one env is refused); half-
  configured kwarg pairs and out-of-range fractions rejected. The
  library file's construction-time sha256 lands on the env as
  `drill_library_sha256` AND is banked into the run's
  `config.json` by the env probe — the record of what the run
  actually loaded, not what the file at the path contains at audit
  time — pinnable at freeze via
  `validate_run_config_against_plan`'s new `drill_library_sha256`
  key alongside the `env_kwargs` subset match.
- **D5 as shipped**: the launch-time clearance check runs against
  the harvest-recorded heads *before* any state mutation; a
  violating entry falls back to a drawn serve and increments the
  per-episode `drill_fallback_count` (0 across every measured
  launch of the registered library).
- **Provenance**: step info gains `drill_point` (the step's own
  point, the same pre-flip convention as `serve_side_is_policy`),
  `drill_entry_index` (−1 undrilled), and `drill_fallback_count`;
  reset info gains `drill_point` + `drill_entry_index`, with the
  serve keys carrying the harvested launch ball state — every
  launch reproducible from the info stream. The D6 recipe overrides
  are shipped (`eval_env_overrides` forces both kwargs off for the
  PaddleTennis recipe) and cover any `[env]`-table override; an
  explicit `[eval_env]` table still wins last by the documented
  layering, so the freeze checklist forbids drill keys there and
  the frozen plan pins the RECORDED evaluation env via
  `validate_run_config_against_plan`'s new `eval_env_kwargs` key
  (asserting `drill_library=None`, `drill_fraction=0.0` on the
  run's own `evaluation_env` block).
- **Train/held-out split (D1)**: realized as separate artifacts at
  freeze — the env consumes a train-only library file verbatim; no
  split field in the schema.

## 3. Measured anchors (all 2026-08-30, CPU)

- **PT2 geometry** (the target): paddle → return-bounce 3.0 m /
  7.5% ≤ 1 m vs oracle 0.89 m / 99.8%; conversion 1.5% vs 98.2%;
  flux 1.3–1.7 opportunities/episode at 4.5 points/episode.
- **Feasibility probe** (pre-design; archived
  `drill_feasibility.py`): 60 pre-crossing return states harvested
  by finite difference from the PT2 registered-checkpoint streams;
  **30/30 replayed feed-context launches produced a legal side-A
  feed bounce; the scripted oracle hit 27/30 (90%)**. *Disclosure:
  a partial-state probe* — side-A paddle qpos only (qvel zeroed),
  side-B paddle at reset park (the streams carry A-only qpos),
  spinless finite-difference velocities, launch hand-rolled on the
  reset path; KD2 re-anchors with the full D4 state through the
  real launch body before any band freezes. The at-crossing launch
  variant is rejected by the event sampler's side validation
  (code: `_tennis_events.py` reset; measured), fixing D2's
  pre-crossing convention.
- **The matched-pair context measurement** (§1a; review probes
  `drill_policy_probe*.py`, 47 states from the LH1c streams,
  scratch seeds 9100–9146): real-context 0/47 convert, 15.9%
  ≤ 1 m; feed-context 34–36% legal hits, 38% touch, robust across
  its three paddle-state arms — **all simplified physics, superseded
  at corrected magnitude by §3a's faithful rows. The step-0
  baselines for both D2 arms come from
  `tools/paddle_tennis_k2_step0.py`'s faithful launches, re-run on
  the pilot's own library at battery time** — drilled-point
  observables are scored against them, never against an assumed
  floor or the simplified variant (the draft's touch-accrual
  arithmetic is retired; its ≤ 1 m-as-touch proxy overstated
  real-context touch ~10×).

## 3a. The harvest tool, the registered library, and both-arm step-0 rows (2026-08-30, banked)

Implemented per the maintainer's direction ahead of the freeze:
[`tools/paddle_tennis_k2_harvest.py`](../tools/paddle_tennis_k2_harvest.py)
(the D1 instrument — full-physics scenario capture with rules/sampler
deep-copies, refuse-reserved guard including the calibration block,
provenance-stamped pickle library, schema `k2-drill-library-v0`) and
[`tools/paddle_tennis_k2_step0.py`](../tools/paddle_tennis_k2_step0.py)
(both D2 launch arms + fidelity checks + step-0 scoring).

**The registered library**: harvested from the registered 2.4M
protected best (pins `838997fb…`/`d0502c14…` verified before
capture) on scratch seeds **9030–9099** (70 episodes) — **102
entries, 1.46/episode** (matching PT2's opportunity flux), library
sha `43975265…` (regenerated at the hardened-instrument commit
`c8bc07d` with clean-tree git stamping after verification caught the
first capture's dirty-tree provenance; entry content verified
field-identical across regenerations). Harvest exclusion counters:
clearance-dropped 0, unarmed crossings 0. Reproducible
deterministically from the tool + the recorded pins and seed range.

**Restore validation (KD1-grade, measured at scale):** the
full-context restore is **exact** — launch-observation max
deviation ≤ 8e-6 (max 7.903e-6) across all 102 entries, and the restored
continuation reproduces the harvest-recorded ball track
**bit-exactly** (per-entry max divergence p50 = 0.0; overall max
4.6e-5 on one entry; outcome agreement 102/102). Requirements
discovered and shipped in the tool: restoring MuJoCo's
`qacc_warmstart` (without it, solver-warmstart differences amplify
through bounces to meters) and step-number-aligned comparison.

**Step-0 rows (registered checkpoint, faithful full physics):**

| arm | touch | legal hit | ≤ 1 m at bounce | n |
|---|---|---|---|---|
| (a) feed-context | 6.9% | **6.9%** | 5.9% | 102 |
| (b) full-context | 2.9% | **2.0%** | 4.3% | 102 |
| oracle on the same launches (feed) | 77.5% | **77.5%** | 84.3% (0.98 m mean) | 102 |

The oracle row certifies the drill balls winnable; the 2/102
full-context conversions equal the recorded live-play outcomes
(outcome-match 100%), i.e. arm (b)'s step-0 row *is* the real-task
baseline by construction. Like-for-like caveat, surfaced by the
instrument: 11 of the full arm's 102 entries are
truncation-censored (harvested near the episode cap, so their
restored continuation inherits the short remaining budget while the
feed arm grants a fresh one); excluding them moves the comparison
only to ~2.2% vs 6.9%.

**Env-launch cross-check (2026-08-30, after the §2a mechanism
shipped):** the same 102 entries scored through
`PaddleTennisEnv._launch_drill` itself (the body training will run,
fresh episode clock and budget) reproduce the tool SUMMARY RATES —
feed 6.9%/6.9%; full legal-hit 2.0%, touch 3.9% vs the tool's 2.9%
— with launch-observation fidelity ≤ 8e-6 off-clock (max
7.903e-6). At the per-entry level the full arm's outcomes are NOT
the tool's (verified against the banked
`env_launch_crosscheck.json`): the fresh episode clock — obs index
35, the sole launch input that differs, by up to 0.99 — flips the
deterministic checkpoint's outcomes on non-censored entries. Both
recorded live-play conversions (entries 39, 67) are lost, two new
conversions (3, 15) and one extra touch (16) appear, the 11
truncation-censored entries play out under the fresh budget to
zero touches, and bounce-within-1m moves 4.3% → 12.0% (only one
gain from a censored entry). The matching 2.0% is a coincidence of
counts, not a reproduction — so the earlier statement that arm
(b)'s step-0 row "is the real-task baseline by construction" holds
only for the tool's clock-restored replay; **the freeze's KD2
full-arm baseline must be the env-body row measured through
`_launch_drill`**, which §3's re-anchor rule already requires. The oracle scores **identically under
both arms** (touch = legal = 77.5%, same enders, same geometry):
the oracle reads only the physical block plus `bounce_count` and
`ball_side`, so it is structurally blind to the §1a context gate —
a clean dissociation pinning the 6.9%-vs-2.0% gap entirely to the
learned policy's use of the rally-context features. Oracle-miss
taxonomy on these launches (freeze-relevant for the KD2 oracle
band): 23/102 never touched, all wide/fast tails (missed balls
bounce at |y| ≈ 3.4 m vs 1.5 m for converted ones; the oracle still
gets within ~1.1 m of the bounce); 20 of the 21 with recorded
geometry are kinematically reachable within the playable window
(≤ 8 m/s average required vs the 12.5 m/s ceiling), so the 77.5%
row under-certifies winnability — it measures the frozen oracle's
cold-start rescue rate, not ball quality.

**Oracle witness-parameter probe (same day, same launches; the
frozen `scripted_ground_opponent` untouched — variants are
parameterizations of the public
`ground_lead_charge_local_action`):** halving the run-up wait
margin (0.9 → 0.45 m behind the predicted landing) lifts
conversion 77.5% → **80.4%**; cutting it to 0.2 m raises touches
but breaks legality (73.5% — the face re-enters the descent path);
widening the charge gap (0.8 → 1.2 m) changes nothing at either
margin. Reading: the wait margin is a small real lever, the charge
gap is not binding, and the residual ~20% miss tail (wide/fast
balls the oracle trails by ~1.1 m at bounce) needs a different
charge program — direct post-bounce path interception rather than
wait-behind-the-landing. Consequence, explicitly NOT an action:
the frozen oracle is the task opponent and a certified reference
(bands, P4 fairness, every instrument row) — it does not change.
If the freeze wants a tighter KD2 winnable-cert or (under the
demonstration-injection route) a higher-coverage demonstrator on
drill states, that is a NEW named witness function calibrated at
battery time, with these rows as its starting point.

**The simplification-confound attribution** (the §1a magnitude
correction): applying the review probe's simplification stack
(ball spin zeroed, paddle velocities zeroed, opponent parked) to
the same faithful states raises feed-context legal hits from 6.9%
to 9% (registered, n=102) and from 6.7% to 20% (an LH1c
cross-check library, n=30, seeds 9148–9167; the review's own 36%
on its n=47 stream states re-verified, CIs overlapping the 20%).
Spin alone roughly doubles the LH1c rate (6.7% → 13%). Consequence
for D1, binding on any freeze: **the drill must launch faithful
physics — real spin, real paddle velocities — or it trains easier
balls than the task serves** (the serve-alignment falsification's
exact trap).

## 4. Certification battery (KD0 shipped with the mechanism; KD1–KD4 proposed — the maintainer freezes the battery before the pilot)

- **KD0 — bit-identity when off**, including the RNG-stream
  discipline of §2. *Shipped as standing regression tests with the
  mechanism* (`tests/test_paddle_tennis.py::TestK2Drill`: full-info
  lockstep off-vs-default; the twin-generator RNG certificate for
  undrilled, ineligible, and drilled launches; stream shared
  through ineligible launches with divergence exactly at the first
  drilled one; hardened by the 2026-08-30 adversarial review — six
  launch-site mutants, all measured killed); the frozen battery
  re-runs them. The same-code lockstep cannot certify cross-commit
  identity by itself: the default info stream legitimately gained
  three constant provenance keys (§2a), and the review's verifier
  additionally locksteped the default env against the pre-change
  commit (966dba8) including RNG state as a one-time check.
- **KD1 — launch fidelity.** Arm (a): the relaunched ball's first
  side-A bounce within a measured tolerance of the harvest-recorded
  bounce. Arm (b): the launch observation equals the recorded
  observation within tolerance (the full-context restore's whole
  point), plus the bounce check.
- **KD2 — witness economics + step-0 policy rows on drilled
  points.** `statue`: kept escrow exactly 0 on every drilled point,
  clawback identities across drill boundaries (measured already on
  30 feed-context points: 10 paid reach, 0 kept). `oracle`:
  conversion within a band of the re-anchored measurement.
  **Step-0 warm-start-policy rows for BOTH D2 arms** (touch, legal
  hit, ≤ 1 m on the pilot library) — the numbers the D2 fork and
  the §5 mechanism bars are set from. Witness `drill_fraction`
  pinned (candidate 1.0 for density).
- **KD3 — distribution and fairness audit.** Library marginals vs
  the source checkpoint's PT2-measured opportunity geometry;
  clearance-filter and harvest-yield counts from metadata;
  serve-alternation ledger unchanged at every fraction; the
  A-side-only training asymmetry booked (eval-side P4 mirror
  untouched); train/held-out split integrity.
- **KD4 — realized drill economics at pilot settings** (the stage
  the draft lacked): ~20–50 episodes at the pilot's exact config,
  recording drilled draws/episode, fallback count (must be 0 under
  D5), per-entry repeat factor, library coverage of the PT2
  depth/lateral tails, and drilled-point reach-pay incidence —
  gating the pilot on the §1 arithmetic as corrected, so exposure
  starvation or a silent fallback drift cannot pass the battery.
- **Seed block: 6400–6499, booked by the maintainer 2026-09-02**
  (the battery block for KD2–KD4 and the LD1′ SD battery alike). Harvest and probe seeds:
  the §7 scratch block. **4100–4199 stays sealed.**

## 5. LD1 — pilot shape (RETIRED as drafted, 2026-09-02, on the Phase 0 gates; the rule SHAPE below is inherited by LD1′, with the drill as its RE-AIM escalation)

Warm start from the **registered 2.4M protected best** (shas pinned
via `expected_artifact_sha256`; also the library's source — one
lever); 1M steps, seed 0, n_envs 4, eval 25k,
checkpoint/diagnosis 100k; recipe defaults plus the pinned library
and **`drill_fraction = 0.5`** (one sanctioned pre-pilot re-anchor
from KD4's measured economics, the hold design's pattern).
*Prerequisite still to ship before the pilot* (deliberately not part
of the 2026-08-30 mechanism landing — it consumes the freeze's
choices): the diagnosis instrument's class-split extension (PT2 §7)
— ready-position / touch / boundary metrics split by ball class and
`drill_point` (now a standing info key, §2a), so the mechanism bars
below are standing per-run observables.

**Decision-rule shape, committed now (numbers at freeze):**

- **KD-primary (the only verdict-bearing k bar): k=2 exchange
  survival on the STANDARD task** at the standing KT1 bar shape —
  with the **false-PASS guard the source's own noise floor
  demands**: the warm-start source reads 1.6% [0.3, 4.5] on fresh
  seeds, so a bar-clearing checkpoint **must be confirmed on a
  fresh, untouched seed arm** (the 2026-08-23 pattern: only the
  fresh arm scores; a PASS from the diagnosis stream alone is a
  hypothesis, per the standing lesson) or the bar set above the
  floor's CI upper. Drill-point statistics are mechanism
  observables, never verdicts.
- **KD-mechanism:** paddle → return-bounce on standard-task k=2
  opportunities (band from the battery, target direction 3 m →
  0.89 m), and drilled-point touch/conversion scored against the
  KD2 step-0 baselines (measured on the registered library, §3a:
  ≈6.9% for arm (a), ≈2–3% for arm (b) — only movement *from the
  baseline* counts).
- **KD-retention:** k=1 receiving holds its band on the standard
  task, with a **mid-run abort convention** named at freeze (the
  existing eval-side selection/patience machinery on
  `legal_hit_count_a` bounds the damage before 1M steps; the
  ~6.5 GPU-hour pilot cost bounds the rest).
- **Branches:** **ADOPT** (KD-primary PASS confirmed fresh-arm +
  retention holds) → pre-register the era's registered-scale retry
  with the drill as recipe convention. **RE-AIM** (mechanism moves
  — drilled-point gains beyond baseline, or standard-task
  geometry improves — but KD-primary FAIL) → the context gate held
  against this arm: route to the **other D2 arm** if unpiloted,
  else to **demonstration/buffer injection** with the measured
  context numbers as its targets. **STOP** (no mechanism movement
  beyond step-0 baselines) → the drill class is spent;
  demonstration injection inherits directly. Middles: maintainer's
  call, documented post-hoc.
- **Step-0 rows** (standard-task diagnosis + both-arm drilled-point
  rows) are the pilot's baselines, banked at battery time.

## 6. What this does not do, and what the era break costs

- **No reward, opponent, dynamics, action-interface, or
  observation change.** Obs stays 48; the lineage stays
  warm-startable; the three escrows and their identities are
  untouched (the drill only changes when the ladder gets its
  chance to pay).
- **The eval task is bit-frozen** (D6): selection metrics, eval
  series, diagnosis rows, and historical comparisons keep their
  meaning. What breaks, completely enumerated: **training-side
  series** (monitor returns, train-phase fractions, train-time
  event rates — drilled episodes change composition), and **the
  n-point carryover law + R2-family boundary metrics on drilled
  boundaries** (§2 D4 — split by `drill_point`, so the undrilled
  series stays meaningful). obs_rms is fit on the drilled stream
  (part of the policy artifact, not a series break — noted for
  completeness).
- **What survives a null result:** every §5 branch routes — ADOPT,
  the other arm, or demonstration injection with sharpened
  targets. The §1a measurement already banked this design's
  largest information gain: the blocker is a context gate, and
  that fact outlives any LD1 outcome.

## 7. Seed ledger

This document proposes **9000–9199 as a booked scratch/workpaper
block**, with the following consumption recorded (all 2026-08-30):
feasibility probe **9000–9029**; review probes **9100–9146**; the
registered harvest **9030–9099** (the §3a library's source
episodes); step-0 replay resets **9147** (launch draws discarded);
the LH1c cross-check harvest **9148–9167**; the arm-(c) roll-in
probe **9168–9187**
([`paddle_tennis_prefreeze_diagnostics_20260830.md`](paddle_tennis_prefreeze_diagnostics_20260830.md)).
Unconsumed remainder: **9188–9199** (12 seeds). **Scratch extension
9200–9299 booked by the maintainer 2026-09-02** (first consumer: the
demo-harvest diversity block, recorded in the injection design's
ledger). Calibration 5200+ stays clean for the diagnosis
instrument (no train-on-test). The KD battery block is assigned at
freeze (§4). Nothing else burned; **4100–4199 remains sealed**.
