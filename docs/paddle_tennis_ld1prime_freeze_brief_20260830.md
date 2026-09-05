# LD1′ freeze brief: the Phase 0 gates and the decisions they resolve (2026-08-30)

Status: **BOOKED by the maintainer 2026-09-02 — decisions D-A
through D-G below are the campaign's routing; D-E's harvest wording
amended and re-booked 2026-09-05 (its bullet)** (the LD1′ pilot
itself is pre-registered separately in
[`design_paddle_tennis_demo_injection.md`](design_paddle_tennis_demo_injection.md)
once its instruments certify; the numbers here are its inputs). Phase 0 was
maintainer-blessed; its three gates ran on banked artifacts and
standing seed conventions (env resets: 5200–5209 diagnosis-class
reads and the 9147 discarded-draw launch convention; torch noise
seeds 0–2, recorded; **zero new env seeds consumed**). All three
results passed an adversarial verification pass (7 minor precision
findings, all incorporated below; 0 material).

## The three gates, verified

**G1 — Q-ordering: the kept critics are near action-blind at the
failure states.** On the 102 full-context launch states, min(Q1,Q2)
ranks the oracle's action above the policy's on only 42% (45% along
oracle-play trajectories from those states, first ~120 steps); the
mean gap is −0.003. The verified scale reference: sweeping uniform
random actions across the whole [−1,1]³ cube moves min-Q by only
~0.02 total per state (mean |Δ| vs the policy action 0.007), so the
oracle-policy gap sits **inside the critics' action-insensitivity
floor** — they barely separate random actions from policy actions
here, under every read (per-critic, mean-Q, min-Q).
*Consequence:* a Q-filtered BC term **cannot engage on day one**
(the filter would pass ~half the demos arbitrarily); demos must
first teach the critics through TD. There is also no k=2
action-ordering worth preserving in the critics.

**G2 — training-stream dormancy: the recycling gate passes.** On
the replay-like stream (stochastic gSDE rollouts at the measured
~0.60 pre-tanh noise, critics evaluated at the actions actually
taken — the convention the diagnostics note demanded), critic
penultimate dormancy reads 55%/59% at τ=0.025 with 41%/44% exactly
zero — **matching or slightly exceeding** the deterministic slice
(55.5/61.3% and 38.3/40.6%). The dormancy is a property of the
training stream, not a measurement convention. (Cadence caveat:
the probe's 64-step noise blocks are episode-aligned rather than
globally aligned as in training; immaterial to the fractions.)

**G3 — behavior-policy step-0: jitter does not rescue engagement.**
Re-scoring both drill arms under the stochastic behavior policy
(3 torch-seed replicates; noise verified flowing, every action
perturbed): feed pooled legal **6.5%**, full **1.3%** — on top of
the deterministic 6.9%/2.0%. Statistically fair under the
conservative read: the replicates cluster on the same 102 states
(4 feed entries convert in all reps; effective n ≈ 100–140, not
306), and even cluster-conservative CIs (feed [3.2%, 13.1%], full
[0.3%, 5.9%]) contain the deterministic references. The pure-drill
pilot's low-engagement premise now stands **measured in both
modes**, closing the red-team's strongest objection to retiring it.

## Decisions these resolve (maintainer bookings, in order)

- **D-A. Re-book the §5 routing: injection-first.** Retire
  LD1-as-drafted pre-freeze (it was never frozen). The case no
  longer rests on predictions: G3 (both modes), the oracle
  context-blindness dissociation, and the longitudinal k=1/k=2
  dissociation carry it.
- **D-B. Critic treatment: head-reset / ReDo-style recycling at
  fine-tune start.** G2 passed the note's own gate; G1 shows no
  k=2 action-ordering to preserve. Full re-initialization stays a
  RE-AIM escalation only (and would need its own transfer-path
  design — the repo's warm-start loader is all-or-nothing).
  LayerNorm critics: dropped (not expressible in stock SB3;
  refused by the warm-start guard; breaks the lineage).
- **D-C. BC term: unfiltered at launch, Q-filter armed by
  measurement.** Pre-register the G1 ordering metric
  (frac-oracle-higher on held-out demo states) as the arming
  condition — the filter turns on when the critics demonstrably
  learn the ordering from injected data, not by schedule.
- **D-D. No temperature re-heat.** The booked T1-FAIL era law is
  mechanistic and data-independent (auto-α re-collapses under the
  ~88%-saturated head). ent_coef and post-swing saturation become
  pre-registered observables; any temperature variant is its own
  designed document per the LT1 prereg's named path.
- **D-E. Demo harvest spec:** oracle completions launched from the
  policy-harvested full-context library states (the policy's OWN
  failure distribution — not oracle-native play), harvested through
  the conversion's confirmation so the conversion payment is in the
  buffer (fail-loud assert per trajectory), with the
  post-confirmation rally recorded to the point's end, the episode's
  truncation, or a 300-step cap, whichever comes first (**amended
  and booked 2026-09-05**: as
  first booked this read "through point termination", which the
  measured harvest does not meet literally — two scripted players
  rally on after the conversion; design §3), budgeted from the measured
  77.5–80.4% oracle conversion (fresh-clock figures at booking; the
  clock-restored harvest touched 77.2% and kept 64.4% — design §3),
  with a train/held-out split.
  **Booking needed: the 9200–9299 scratch extension** (remainder is
  12 seeds) for harvest diversity beyond the 102 deterministic
  completions.
- **D-F. Drill OFF at LD1′ launch;** drill-on (arm (b)) is a
  pre-registered RE-AIM escalation. The D2 fork is thereby
  deferred, not decided; the shipped mechanism waits ready.
- **D-G. Instruments to ship before the pilot** (≈ the drill
  mechanism's shipping cost, priced honestly): the SAC.train()
  injection fork + algo-registry entry **with a regression test on
  the resolved gradient_steps** (the silent 256× trap), the
  model-side demo-library provenance hook (construction-time sha →
  config.json + a `demo_library_sha256` plan key), the SD0
  bit-identity certificate for the training-loop change (scoped:
  parameter-stream lockstep with global-RNG accounting), the demo
  harvest tool, and the class-split diagnosis extension (standing
  prerequisite). **Battery block: 6400–6499** (maintainer assigns).

## What Phase 0 did NOT test

Training dynamics under injection (a short injection-only mini-run
remains the optional Phase-2.5 gate before the 1M pilot); the
arm-(c) discriminating sample (parked, no mechanism case); masking
(named RE-AIM alternative, own design if it fires).
