# Design: PaddleTennis command-rate limit — capping the target saccade

Status: **FROZEN 2026-08-29 (maintainer decision, recorded via the
review session), before any implementation.** The §5 gate is
satisfied: LT1 completed and was scored 2026-08-28, and its
**T1-FAIL branch was booked 2026-08-29**
([`paddle_tennis_lt1_prereg_20260823.md`](paddle_tennis_lt1_prereg_20260823.md)
§4b — α back below 1e-3 at 36k steps, strike-ended saturation never
< 85% across all ten checkpoints, minimum 86.0% against the 87.6%
anchor). Binding as drafted: the §2 mechanism decisions D1–D4, with
**D5 at (i)** per the drafted recommendation; the §3 candidate
**Δ = 0.15** and its one-knob re-pair convention (now in force per
§3's own terms); and the §4 battery, whose freeze-time slot is
filled — **CR2 seed block 6400–6499 assigned** (next free above
6300–6399, verified against the ledger at freeze). Nothing is
implemented yet; implementation proceeds in battery order (CR0
first). Still a **NEW COMPARABILITY ERA if adopted** (action
semantics change), and the §5 pilot's final freeze — shape
confirmation and bars — stays deferred to the post-battery freeze,
per §5. Drafted 2026-08-23 from the PT1
routing
([`paddle_tennis_postswing_targets_20260822.md`](paddle_tennis_postswing_targets_20260822.md)
§6 item 2 — whose warning this Status repeats: not a TOML toggle —
commissioned by
[`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md) §4
step 4). Standing doctrine from this freeze: changes after a
pre-registered run starts void it as campaign evidence.

## 1. Why this change, and why interface-side

PT1 measured what the k=2 blocker's post-swing wander *is*: not
drift, not a wrong destination — **commanded thrash**. On identical
physics and seeds (strike-ended windows, PT1 §4): the trained
checkpoints command 31.04 m (registered 2.4M) and 22.71 m (LH1c
2.325M) of XY target path per ~80-step window while the paddle
actually travels 7.84 m and 7.55 m — the commanded path is **3–4×
the actual** (oracle ratio 1.0: 2.56 vs 2.55 m). The action head is
saturated (|a| > 0.9) on **87.6–92.4%** of post-swing steps across
the three measured checkpoints (registered 2.4M, LH1c 2.325M, the
2.5M k=2 peak — the review's "88–93%" band), the command jumps
**0.26–0.38 m per step** (saccades 0.381 / 0.258 / 0.290; oracle
**0.033**), and the mean servo gap is **2.2–2.7 m** (2.70 / 2.23 m).
The servo tracks **≲ 0.1 m/step** (the review's measured band, §4
step 4): the plant low-pass-filters a command it can never follow.

PT1's F4 names why every reward-side fix starved: earning the hold
pay took **~80 consecutive near-zero-displacement commands from a
saturated head** under a dead temperature (ent_coef ~1.7e-4, std
~0.014) — no noise to discover partial stillness, no local gradient
to unsaturate 80 outputs at once. The reward-side line is closed:
all three event-escrow designs ran (contact and reach adopted; hold
delivered its dose and the policy declined it —
[`design_paddle_tennis_postswing_hold.md`](design_paddle_tennis_postswing_hold.md)
§4c, audited in §4d); PT1 §6 closes further escrow scale,
opponent-side, and dynamics-side explanations.

Why interface-side rather than a fourth reward term (the
action-smoothness tax PT1 §6 also names): **the thrash costs the
policy nothing** — no reward reads `ctrl`, the plant filters the
saccades away — and a smoothness penalty must be *descended* by
exactly the dead optimizer F4 indicted. An interface-side limit
changes what actions *do*, mechanically, on step one, nothing to
learn: stillness becomes the head's default output instead of an
80-step feat; deliberate motion passes through (§2, §3).

## 2. Mechanism (frozen 2026-08-29 — D1–D4 as drafted, D5 at (i); decision points named, not silently chosen)

One new constructor kwarg on `PaddleTennisEnv`:

- `target_rate_limit: float | None = None` — meters of position
  target motion per control step, per axis; validated `None` or
  finite positive. **`None` (default) is bit-identical off** (CR0).

The env forwards the value to **both** `PaddleInterface` instances
it constructs; the limiter lives in `PaddleInterface`
(`src/courtside_dynamics/envs/_paddle.py`) as per-instance state in
the ctrl/qpos frame. One knob, both sides — no per-side asymmetry
is expressible — and the mirror's negation is exact (IEEE), so the
clamp commutes with it bit-for-bit: the P4 fairness identity
([`paddle_tennis_probes_p3_p4_20260802.md`](paddle_tennis_probes_p3_p4_20260802.md))
extends to the limiter state; CR3 asserts it.

**D1 — rate limit vs first-order lag (the routing named both):
rate limit, recommended.** A first-order lag (`eff += β(raw − eff)`)
attenuates *every* command — small corrective motions included —
never settles exactly on a held target, and has no statement in the
measured unit (m/step). A per-step rate limit clamps exactly what
PT1 measured (the saccade), passes sub-limit motion through
untouched, and makes exact stillness reachable. Update rule, pinned
for bit-exactness:

```
if |raw − eff| <= Δ (per axis):  eff = raw        # exact passthrough
else:                            eff = eff ± Δ
```

**D2 — placement in the chain: last, after the fence.** The mapping
today is clip → `piecewise_targets` → fence clamp
(`PaddleInterface.targets`, `_paddle.py:299–326`). The limiter runs
**fourth, on the fence-clamped target**: its state then tracks a
point the servo could actually be asked to hold — limiting *before*
the fence would let a saturated command wind up meters of pending
motion that later commands must un-walk — and the effective target
stays in-fence by construction (unit-pinned).

**D3 — one scalar, per axis, all three axes.** Three decoupled
slides, a per-axis sign mirror: a component-wise clamp keeps the
mechanism one line and the mirror argument trivial, z (strike lift)
governed by the same number; an XY-norm limit couples axes for
nothing, and the diagonal worst case (√2·Δ) is priced in §3.

**D4 — state lifecycle, pinned.** At episode reset the state seeds
from the paddle's **current joint qpos** (the named alternatives —
seed-from-home, first-step passthrough — each grant a free step-one
saccade). Across an absorbed n-point boundary the state **carries
over** with the paddles (the relaunch teleports the ball only); the
sanctioned serve-clearance nudge displaces qpos and does **not**
touch the limiter state, exactly as it never touched `ctrl`.

**D5 — the observation question, put to the design review by
name.** Today `ctrl` is a memoryless function of the current
action; under the limiter the effective target becomes genuine
hidden state (the 48-obs vector carries no target). Does the policy
need to see it? (i) Leave it hidden — obs stays 48, the lineage
stays warm-startable, and the POMDP gap is bounded: below the Δ
band effective and raw targets coincide (D1's passthrough), so the
hidden state is largest exactly in the thrash regime the design
removes. (ii) Expose the own-side effective target (3 side-local
values via the mirror) — Markov restored, but obs 48 → 51 breaks
the warm-start policy weights, forfeiting the lineage.
**Recommendation: pilot at (i)**; (ii) is the named follow-on if
the pilot's verdict reads as Markov starvation.

## 3. Parameter anchors from measurement

- **Control step = 0.01 s; physical ceiling = 0.125 m/step.**
  `paddle_court.xml` timestep 0.002 s × `frame_skip` 5 (the
  `CourtsideMujocoEnv` default; PaddleTennis does not override it);
  the ±100 N force cap over slide damping 8 gives the
  probe-measured ~12.5 m/s terminal velocity, and
  12.5 m/s × 0.01 s = **0.125 m/step** — the fastest any paddle can
  move, whatever the target does. Any Δ ≥ 0.125 leaves the *plant*,
  not the limiter, binding on sustained travel (CR2's sprint
  witness measures it).
- **The servo's tracked band ≲ 0.1 m/step** (review §4 step 4).
- **Oracle mean saccade 0.033 m/step** (PT1 §4) — must sit
  comfortably inside Δ. Its *strike* demands are unpublished (its
  1.8–2.1% saturated steps are the lunges), so CR1 first extends
  the probe to record the saccade distribution (p99/max). Bounded:
  at kp 300 under the 100 N cap, target error beyond 100/300 ≈
  0.33 m adds force *duration*, not peak force, so a target ramping
  at Δ ≥ 0.125 m/step still reaches terminal velocity in a few
  steps — the credible impairment is strike *onset* timing (a 2 m
  jump becomes a ~13-step ramp at 0.15), what CR1 exists to catch.
- **Policy thrash 0.26–0.38 m/step** (0.381 / 0.258; 2.5M: 0.290)
  — must sit outside Δ, including the diagonal worst case √2·Δ.

**Candidate: `target_rate_limit = 0.15` m/step.** 4.5× the oracle's
mean saccade, 1.2× the physical ceiling (plant stays binding), 1.5×
the servo band, 42–61% below the measured thrash; diagonal worst
case 0.212 < 0.26. Alternating saturated commands collapse to a
±0.15 m target dither the servo low-passes to near-stillness; a
sustained one-direction command still crosses the court at full
speed. Stillness by default, motion by intent.

**One-knob re-pair convention** (the hold design's pattern; binding
only once this document is frozen): one re-pair to
**Δ = 0.20 m/step**, fired **only** by a CR1
oracle-impairment failure at 0.15 (the measured risk on the tight
side), with CR1/CR2 re-run at the new value. A CR2 failure (the
thrash witness's paddle path fails to collapse) has **no** re-pair
— it falsifies the mechanism and closes the line. No pilot-time
re-pair: the knob is geometric; the battery settles it first.

## 4. Certification battery (frozen 2026-08-29)

- **CR0 — bit-identity when off.** The default in lockstep against
  the pre-change env: exact observation/reward/info equality (the
  family's standing bar); the `_paddle.py` ↔ WallBall drift pin
  green; every existing suite unchanged.
- **CR1 — oracle-behavior preservation.** The PT1 oracle row must
  reproduce under the limit — the oracle must NOT be impaired.
  Replay the scripted ground oracle (30 episodes, deterministic,
  calibration seeds 5200–5229 — reproducing the PT1 row requires
  its seeds) at Δ = 0.15 vs `None`. If the unlimited arm's measured
  max demanded saccade ≤ Δ, the bar is D1's passthrough:
  **bit-identical trajectories**. Otherwise, on the strike-ended
  group against PT1 §4: windows (164) ±5%; commanded 2.56 m and
  actual 2.55 m XY path ±0.25 m; servo gap 0.54 m ±0.15 m; saccade
  0.033 ±0.01; mean crossings and legal-hit counts ±10% of the
  unlimited arm; no termination class shifted more than 5 points;
  failure fires the §3 re-pair, once.
- **CR2 — witness economics** (the PT1 instrument plus limiter
  assertions; 100 episodes per witness per arm, Δ = 0.15 vs
  `None`): `statue` (zero demanded saccade — arms expected
  bit-identical, asserted not assumed); a new `sprint` witness
  (full-court dash; time-to-cross within +10% of its unlimited arm
  — deliberate travel preserved); a new **`thrash` witness**
  (alternating ±1 bang-bang — F2 scripted) whose **commanded path
  is mechanically capped**: effective saccade ≤ Δ every step,
  effective path ≤ Δ × steps per window, actual paddle path
  collapsing versus its unlimited arm (the stillness-by-default
  number anchoring the §5 mechanism bar); `ground_oracle` (CR1's
  row).
- **CR3 — mirror fairness, P4-style.** With the limiter ON,
  `TestP4MirroringIdentity`'s contracts extend: mirrored action
  sequences produce **sign-exact** mirrored effective targets, and
  mirrored trajectories stay mirrored to the P4 bar (1e-6 through
  40 control frames of contact) — bit-for-bit fairness preserved.
- **Seed block: 6400–6499, assigned at the 2026-08-29 freeze for
  CR2** — the next free block above the highest burned (6300–6399,
  the hold design's §4b re-battery), verified free by a repo-wide
  sweep at freeze time (**4100–4199 stays sealed**; 4300–4399 was
  consumed by the NP3 certification, not reserved). The block is
  consumed when CR2 first runs; CR1 alone reuses calibration
  5200–5229, as stated.

## 5. LC1 — pilot shape (drafted; final freeze — shape confirmation and bars — follows the §4 battery)

**Gate: satisfied 2026-08-29.** LC1 launches **only** if LT1 routes
here — its §4 T1-FAIL branch (α re-collapsed, no saturation
movement: the optimizer-side lever spent), per the review's step 3;
LT1's "T1 PASS, KT1 FAIL" branch re-diagnoses, not this design.
**The T1-FAIL branch was booked** (LT1 §4b, maintainer,
2026-08-29): the gate is open. Per this section's own terms, the
pilot's final freeze — shape confirmed and bars set from the
battery's measured band and the step-0 replay row — happens only
after the §4 battery runs; nothing below is binding yet.

Proposed shape, frozen only after the §4 battery: **1M steps**, seed
0, n_envs 4, eval 25k, checkpoint/diagnosis 100k (the standing pilot
cadence), recipe defaults plus `target_rate_limit = 0.15`,
**warm-started from the standing lineage** at LT1-verdict time
(default: the registered 2.4M protected best, `838997fb…`/
`d0502c14…`, sha-pinned — LT1 §2's one-lever reasoning; any other
source is its own pre-registered decision).

**The transfer question, named — this is the new era's caveat.** A
policy trained without the limit and replayed under it changes
behavior on step one: mechanically calmer post-swing, possibly
degraded strike onset. The first eval measures *transfer*, not
learning; the source run's curves are not this run's baseline. The
pilot must book a **step-0 replay row** — the warm-started
checkpoint under the limiter, through the PT1 instrument and the
k-ladder, before any training — and score learning against it.

**Bars: to be frozen from the battery's measured band** at the
post-battery LC1 freeze, not here. The candidates, with anchor sources: KC1 headline
(k=2 exchange survival ≥ 3% at some checkpoint — the KH1/KT1 bar);
C1 mechanism (post-swing *actual* paddle travel / recovery-hold,
its bar set from CR2's thrash-collapse measurement and the step-0
row — raw-head saturation stops being the mechanism metric: a
saturated head can now be behaviorally calm); R1 retention (k=1
receiving ≥ 80% at some checkpoint). No Δ re-pair at pilot time.

## 6. What this does not do, and what the era break costs

- **No reward change.** All three escrows and the nine-component
  decomposition keep their certified economics, untouched.
- **No opponent change — with the honest caveat.** The controller's
  code is untouched but its targets pass through the same limiter
  (fairness *is* the point); CR1 certifies its behavior survives,
  and if CR1 fails at both sanctioned values the design dies rather
  than going asymmetric.
- **No dynamics change.** kp/kv, the ±100 N cap, damping 8, the
  timestep: untouched. The limiter shapes targets, never forces;
  the 12.5 m/s ceiling stands.
- **The era break is the cost being bought, priced here.** With the
  kwarg ON, action semantics change: training curves, eval-reward
  series, and the campaign-record evaluations (+2.483, +1.822) stop
  being comparable across the break; every PT1 command-side row
  changes meaning; the oracle's certified band gets a new-era row
  from CR1; old checkpoints replayed under the limit are new-era
  *subjects*, not baselines; every behavioral bar (k-ladder,
  travel, touch) re-anchors from the battery and the step-0 row.
  What survives: default OFF keeps every past run reproducible
  bit-for-bit; the instruments keep their units.

Seed ledger: **6400–6499 assigned to CR2 at the 2026-08-29 freeze**
(consumed when CR2 first runs; nothing burned yet); CR1 reuses
calibration 5200–5229; **4100–4199 remains sealed**.

## 7. Pre-implementation falsification (2026-08-30) — PROPOSED disposition

**Status of this section: PROPOSAL — the maintainer books the
disposition.** No CR battery stage has run, no pre-registered run
has started, and no line of the limiter has been implemented, so a
disposition decision now voids nothing; §1–§6 above stand as the
frozen record.

Before CR0, the PT2 diagnosis probe
([`paddle_tennis_command_spectrum_20260830.md`](paddle_tennis_command_spectrum_20260830.md))
measured, from recorded per-step command streams of the three
standing checkpoints and the oracle, the two quantities this design
had assumed:

- **F1/F2** — the thrash is heavy-tailed *jump-then-dwell* (p50
  per-step command delta ≤ 0.033 m; p99 3.4–5.3 m), not the
  per-step alternation §3 priced Δ = 0.15 against; the D1 rule
  applied open-loop at the frozen candidate shows **no
  paddle-travel-reduction mechanism in evidence** (effective
  commands still slew at ≥ the 0.125 m/step plant rate on ~47% of
  window steps; ~100% simulated ON/OFF travel, undecidable
  closed-loop but unsupported open-loop), while the effective
  target *walks* 3.1–3.3 m net per window. Material bite appears
  only at sub-plant Δ (≤ 0.05) — exactly where **F4** shows the
  oracle's own strike jumps (p99 1.78 m) begin ramping (87%/68% of
  its unlimited travel at 0.05/0.03): the CR1 impairment this
  design's §6 names as line-ending.
- **F3 (decisive)** — on measured k=2 opportunities the paddle sits
  2.6–3.3 m from the return's bounce (≤ 1 m on 7.5–13.5%) and
  converts ~1%, while the oracle sits 0.89 m (99.8%) and converts
  98%; the **frozen-at-hit counterfactual — the zero-motion
  end-member of this design's mechanism — is no better (≤ 1 m on
  1.5–6.0%; 3.0–13.5% at the follow-through-exempt +30 anchor)**,
  and the oracle's own frozen counterfactual collapses to 38.7%:
  its k=2 mechanism is *directed recovery toward the return's
  landing point* (strict-k=2 anchor, n=150), which a rate limit cannot produce for commands
  that (per PT1) track no attractor. The §1 premise (mechanized by §2) — stillness by
  default pays the k=2 chain via the certified escrow ladder — is
  contradicted at every Δ.

**Proposed disposition: CLOSED without implementation** — falsified
by measurement before CR0, at the cost of one CPU replay session.
On booking: the CR2 seed block **6400–6499 returns to the free pool
unconsumed** (this document then burns nothing); the LT1 §4 routing
("route to the interface-side treatment") is recorded as satisfied
and exhausted by this measurement; the campaign's next lever is a
new design decision, with PT2 §7 naming the measured target (paddle
→ return-bounce ~3 m → ~1 m) and the candidate lever classes
(demonstration/buffer injection; the k=2 drill curriculum).
