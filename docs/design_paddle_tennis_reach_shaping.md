# Design: escrowed reach shaping — paying the position→touch gap

Status: **Implemented; RS0–RS2 PASS (§3a); LR1 run — the ADOPT branch
fires (§4a): K, E1, E2, E3 PASS**, 2026-08-16. The era's number
moved: k=2 exchange survival above zero at five checkpoints
(including a confirmed, in-bounds second-exchange return at 700k —
the first in the campaign), the first positive learned final
evaluation (+0.14), and the recipe is cleared to adopt
`points_per_episode=None` + `contact_shaping=0.25` +
`reach_shaping=0.25` with the 6M registered run to be pre-registered
on the standing 4100–4199 held-out gate. The remedy
the L2W warm-start pilot points at
([`paddle_tennis_npoint_pilot_20260815_review.md`](paddle_tennis_npoint_pilot_20260815_review.md)
§6a): handed a k=1-mastered policy, n-point training re-converged the
*stroke* on its own (86% crossed / 79% in at oracle depth by 1M) while
every *positioning* metric stayed flat or worsened — touch ≤ 15%,
ready error 3.8 m, post-swing wander 6.6 m, inter-point travel 9.1 m —
and k=2 stayed exactly 0% through 4M cumulative n-point steps across
three runs. The contact escrow pays the touch→in gap; **nothing pays
the position→touch gap**, and the review's exploration battery
measured that noise cannot cross it (zero contacts at any std; the
bounce lands 4.8 m away with ~1 s of flight). This design pays that
gap without making it farmable, one rung below the contact escrow on
the same audited pattern.

## 1. Mechanism — the contact escrow, one rung down

Provenance, stated honestly: the repo **measured the problem** (L2W
§6a: stroke re-converges, position never improves; the review §4b:
reach is the exploration cliff's face) and owns the **audited answer
pattern** — the escrowed advance of
[`design_paddle_tennis_contact_shaping.md`](design_paddle_tennis_contact_shaping.md)
§1, whose accounting identities have now passed S1/S2, NP1 (across
point boundaries), and two GPU campaigns. This design instantiates
the same pattern at the next milestone down the skill ladder:

- when the incoming ball's **live first bounce** lands on side A's
  court (a return opportunity — the rules machine records
  `first_bounces` only for non-terminal first bounces, so a feed
  that lands out or fails to cross pays nothing durable): pay
  `+reach_shaping × max(0, 1 − d / reach_shaping_radius)` **now**
  and add it to a pending-reach escrow. `d` is the XY distance from
  the bounce (the qualifying `BALL_COURT_A` event's recorded contact
  position) to the side-A paddle head at step end — at most one
  control step of paddle motion past the bounce substep (≤ 0.125 m
  at the 12.5 m/s actuation ceiling), the documented, decidable
  convention;
- when side A makes a **legal racket hit** (the opportunity is
  taken): clear the pending escrow — the advance is kept. Commit
  before opening, the contact escrow's ordering, so a keep and the
  next bounce can share a step;
- at **every** ending — point boundaries under n-point play,
  termination, truncation, and both nonfinite guards: claw back
  whatever is pending (`rew_reach_clawback = −pending`).

The sum over any episode is exactly the proximity pay of the bounces
whose ball side A then legally hit — an opportunity not taken
contributes **zero** to the undiscounted total, so camping cannot be
farmed. What it changes is the *positioning-time* value: being near
the bounce is worth up to `reach_shaping × (1 − γ^d)` at the moment
of the bounce even when the subsequent swing fails to confirm, and
worth the full advance when the hit lands — so "in position and
swinging" strictly dominates "in position, statue" dominates "out of
position", which is precisely the gradient every run so far has
lacked. Chained with the contact escrow, the ladder now pays every
milestone: position (this design) → touch (kept reach) → in (kept
contact + the shared +1).

Why the gradient reaches the policy this time (the L2 lesson): the
contact escrow's gradient was dead from scratch because touching was
unsampleable. Reach pay is **sampled on every receiving point by any
policy** — the statue collects (and forfeits) it; a warm-started
policy that touches ~30% of reachable balls converts proximity into
kept advances at a rate that *grows with the very behavior being
paid*. The pay is dense where the failure is.

## 2. PaddleTennis specifics

- **Policy-side only (side A).** Same rationale and same asymmetry
  as the contact escrow: opponent bounces on side B pay nothing.
- **The qualifying event is decidable and unique.** A live first
  bounce on side A (`CourtSide.A in transition.first_bounces`)
  implies exactly one processed `BALL_COURT_A` event this step, and
  implies side A is the returner: a side-A shot bouncing on side A
  is a `FAILED_TO_CROSS` fault (terminal → excluded), a serve
  bouncing on the server's side likewise. Under ground rules every
  incoming ball offers at most one such bounce (the second bounce
  ends the point), so at most one pending-reach advance exists per
  opportunity and the double-hit rule keeps hits unambiguous.
- **Terminal-bounce steps net zero by construction.** A first bounce
  that coexists with the point's ending in the same step (e.g. a
  same-step second bounce) pays and is clawed back in the same
  reward — visible in the decomposition, zero in the total.
- **Clawback on every ending path — the §2 contract of the contact
  design, inherited verbatim.** The three reward assembly sites
  (main step with truncation resolved first, forced-nonfinite,
  early-return guard) all fire the reach clawback; under n-point
  play the absorbed point boundary claws back exactly like the
  contact escrow (NP1's boundary semantics). RS2 witnesses each
  path.
- **Magnitude 0.25, radius 3.0 m.** The magnitude is the contact
  escrow's audited value: at most one advance per opportunity,
  bounded by `0.25 × opportunities-taken` per episode in totals —
  still strictly dominated by the ±1 task stream (an oracle episode
  keeps ~1.2; the task stream moves ~±7). The radius is set from the
  measured geometry: the L2W policy's ready error averaged 2.15–3.83
  m (grazing the ramp's edge — gradient live exactly where the
  policy operates), the statue-at-home sits 2.85 m from the mean
  landing (small but nonzero pay — the pay path is exercised from
  step one), and the oracle's 0.89 m earns 70% of full pay. A wider
  radius pays indifference; a narrower one recreates the cliff.
  Both are env kwargs; the radius is frozen for the era once LR1
  runs.
- **Plumbing.** New env kwargs `reach_shaping: float = 0.0`
  (finite non-negative; **default off — the frozen task and every
  earlier era are bit-identical**, asserted by lockstep test) and
  `reach_shaping_radius: float = 3.0` (finite positive). New reward
  components `rew_reach` and `rew_reach_clawback` join the
  seven-way decomposition identity, the info dict, and the recipe
  CSV header. `config.json` records both kwargs via constructor
  provenance.
- **Comparability.** Enabling reach shaping starts a new
  reward-comparability era (stacked on the contact era when both are
  on). Behavioral metrics, the scripted bands, and every
  certification are reward-independent and remain valid. Selection
  follows task metrics (and the L2W-hardened guards), never eval
  reward.

### 2a. Ordering amendment (2026-08-16, post-LR1, pre-registered-run)

The shipped §1 ordering ("commit before opening") had a blind edge
the adoption code review caught: a qualifying bounce and the legal
hit that takes it can share one 50 ms control step (the tight
interception the shaping exists to teach), and the original ordering
escrowed that payment — to be clawed back at the next boundary
unless another hit followed. Coexistence in a batch always orders
bounce → hit (one ball; an untaken earlier bounce would have ended
the point as a second bounce), so the amendment keeps a same-step
payment immediately: only an *untaken* payment enters escrow
(`_reach_escrow_step`, unit-pinned). This changes the shaped reward
stream in exactly this edge; both pilots ran the original ordering
(their recorded results stand as-is) and the registered run runs the
amendment. The RS1 battery re-ran under the amended tracker rule and
passed with results **byte-identical to §3a's table** (identity gap
0.00e+00) — no same-step take occurs on the 5500 block for these
witnesses, so the edge is pinned by the
`test_same_step_take_is_kept_not_escrowed` unit witness rather than
the block. The default-off stream is untouched (RS0 unaffected).

## 3. Pre-registered probe battery

Calibration seed block **5500–5599 is reserved for RS1** (verified
fresh: no prior doc, tool, or test uses it). Ledger note: the
2026-08-15 review's local validation workpapers used ad-hoc blocks
**5600–6199**; they are hereby recorded as burned. Reserved 4100–4199
and 4300–4399 stay untouched; 5200–5299 (diagnosis), 5300–5399 (S1),
5400–5499 (NP) stay burned.

- **RS0 — bit-identity of the default.** `reach_shaping=0.0` (the
  default) produces bit-identical trajectories, rewards, and info
  streams to the pre-amendment env (lockstep, the shaping batch's
  pattern; shipped as the mirror-arm assertions in
  `TestReachShaping` — every existing suite also passes unchanged).
- **RS1 — incentive-ordering witnesses** (scripted,
  `tools/paddle_tennis_reach_probe.py`, 100 episodes per witness per
  arm on 5500–5599, reach 0.25/3.0 vs 0.0; every criterion an exact
  per-seed identity):
  - *every witness*: per episode,
    `Σ rew_reach + Σ rew_reach_clawback == kept` exactly, where
    `kept` is the tracker-recomputed sum of payments whose
    opportunity a side-A legal hit took; per seed, the
    shaped-minus-unshaped total equals the same quantity; no payment
    ever fires on a step without a processed first-bounce event;
  - *statue* (zero action): `kept == 0` on every episode — net
    exactly zero — with ≥ 1 paying episode on the block (the pay
    path exercised);
  - *off-line camper* (parked at landing depth, ~1 m off the ball
    line — the anti-farming witness): `kept == 0` on every episode
    with payments in ≥ 50% of episodes — proximity without intent
    collects nothing;
  - *ground oracle*: keeps ≥ 90% of what it is paid;
  - *hard-slam* (touch-then-out): keeps ≥ 50% of what it is paid —
    reach pays positioning, not aim, by design;
  - *volley-patting*: identities hold (its pre-bounce faults create
    no live bounce; pay only ever follows balls it fails to volley);
  - *stacked arm* (oracle, contact 0.25 + reach 0.25):
    shaped-minus-unshaped total equals
    `kept_reach + 0.25 × side-A confirms` exactly — the escrows
    compose additively.
- **RS2 — decomposition invariants on every ending path** (shipped
  as `TestReachShaping`): the every-step seven-component identity;
  truncation-with-pending (clawback fires); a NaN action with a
  pending advance (the early-return guard claws back beside the −2);
  the n-point point boundary (statue economics stay exactly the
  frozen ones, paid > 0 and net == 0); formula exactness against the
  recorded event position; kwarg validation; the recipe still ships
  both shapings OFF.

## 3a. RS results — PASS (2026-08-16)

RS0/RS2 pass as `TestReachShaping` in `tests/test_paddle_tennis.py`
(10 tests: lockstep default bit-identity and shaped-arm bit-identity,
the seven-component every-step identity, truncation / NaN-guard /
point-boundary clawbacks, n-point statue economics exactly frozen,
formula exactness against the recorded event position, kwarg
validation, recipe-off pin), with every prior suite unchanged (all
142 paddle tests, 306 adjacent train/config/notebook/env tests, ruff
and mypy clean).

RS1 ran as pre-registered (`tools/paddle_tennis_reach_probe.py`, 100
episodes per witness per arm, seeds 5500–5599, reach 0.25/3.0 vs
0.0). **Every criterion passed**, every identity exact to 0.00e+00,
all arms bit-identical:

| witness | paid | clawed | kept | hits | confirms | eps paid | mean shaped reward |
|---|---|---|---|---|---|---|---|
| statue | 2.62 | −2.62 | **0.00** | 0 | 0 | 34/100 | −0.620 |
| off-line camper | 13.30 | −13.30 | **0.00** | 0 | 0 | 94/100 | −0.530 |
| ground oracle | 75.03 | −1.51 | 73.51 (98%) | 418 | 379 | 98/100 | 7.575 |
| hard-slam | 17.92 | −0.59 | 17.32 (97%) | 98 | 6 | 98/100 | −0.257 |
| volley-patting | 0.00 | 0.00 | 0.00 | 0 | 0 | 0/100 | −1.000 |
| oracle, stacked | 75.03 | −1.51 | 73.51 | 418 | 379 | 98/100 | 8.523 |

- Anti-farming witnessed at scale: the camper collects a payment in
  94% of episodes and keeps **exactly zero**; same for the statue's
  34 paying episodes. Camping is worthless, as designed.
- The oracle and the hard-slam keep 98%/97% of their pay — the
  advance rides on taking the opportunity, not on the shot's
  outcome (the slam confirms only 6 of 98 hits yet keeps its reach
  pay): position→touch is paid independently of touch→in, exactly
  the ladder separation intended.
- The stacked arm's per-episode uplift decomposes exactly as
  `kept_reach + 0.25 × confirms` (8.523 − 7.575 ≈ 0.948 =
  0.25 × 379/100): the escrows compose additively.
- No payment ever fired without a processed first-bounce event
  (0 violations across all arms).

Seed ledger: block **5500–5599 burned** (RS1 calibration); the
review's ad-hoc workpaper blocks 5600–6199 recorded burned (§3).
Reserved 4100–4199 and 4300–4399 remain untouched. Next: the LR1
pilot per §4 — launchable from the notebook exactly like L2W
(review doc §6 recipe) with the one-line TOML addition.

## 4. LR1 — the learning pilot (pre-registered here, before any run)

**One reward-side change against the L2W baseline.** Identical run
shape to L2W (`20260815_180815`): warm start from `20260809_211147`
best (policy + obs_rms + temperature, per the loader), seed 0,
n_envs 4, 1M steps, eval 25k, checkpoint/diagnosis cadence 100k, the
L2W guard set (degenerate guard on `legal_hit_count_a_ep_mean`,
`best_metric_min_delta 0.25`, `confirm_best_eval`, success on
`legal_hit_count_a`), TOML adding only:

```toml
[env]
points_per_episode = "none"
contact_shaping = 0.25
reach_shaping = 0.25          # the one change vs L2W
```

(`reach_shaping_radius` stays at its 3.0 default.) No update-ratio or
entropy change rides along — LR1 must isolate the reward amendment;
the churn control is the *named fallback arm*, not a passenger.

**Criteria, frozen before the run** (anchors: L2W measured values;
each metric reads from the automated diagnosis rows / eval CSV at its
own best checkpoint):

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| **K** (headline) | k=2 exchange survival, either parity | 0% at every checkpoint | — (binary) | > 0% at some checkpoint |
| E1 engagement | touched-after-bounce | ≤ 27% everywhere (the L2W peak) | (27%, 40%) | ≥ 40% at some checkpoint |
| E2 position | ready-position error mean | ≥ 3.4 m everywhere (L2W late plateau) | (2.0, 3.4) | ≤ 2.0 m at some checkpoint |
| E3 retention | k=1 receiving survival | < 50% everywhere after 100k | one checkpoint ≥ 60% | ≥ 60% at two consecutive checkpoints (L2W: single 50% peak) |
| R2 | inter-point recovery mean | > 7.6 m everywhere (L2W floor) | (6.0, 7.6] | ≤ 6.0 m at some checkpoint |
| M | behavioral mechanism | `legal_hit_count_a_ep_mean` = 0 at ≥ half the evals | — | > 0 at ≥ half the evals (std ≥ 5e-3 stays a secondary log-check, not a verdict input) |

**Decision rule** (three branches, the L1 shape):

- **Adopt**: K PASS, M intact, and ≥ 2 of E1/E2/E3/R2 PASS → the
  recipe adopts `points_per_episode=None` + `contact_shaping=0.25` +
  `reach_shaping=0.25`, and the 6M registered run is pre-registered
  with its held-out gate on the standing reserved block 4100–4199.
- **Extend once** (declared middle, non-forcing): M intact, K FAIL,
  ≥ 2 of E1/E2/E3/R2 in PASS-or-middle, and the diagnosis rows still
  improving at 1M → a single 2M extension of the identical
  configuration; its verdict re-applies this rule without the extend
  branch.
- **Stop/pivot**: M broken, or K FAIL with < 2 others in
  PASS-or-middle. The pivot is chosen by the failure signature,
  named now: if reach pay is collected but forfeited (paid ≫ kept,
  flat — the escrow too strict for the current touch rate), the next
  probe is the **non-escrowed reach variant** (pay kept
  unconditionally; new statue economics, so its own RS battery); if
  the L2W crash-then-partial-recovery signature reappears in the
  first 100k regardless of reach credit, the next arm is the
  **update-ratio churn control** (cap `gradient_steps` early;
  training-side, no era change) on the unchanged reward.

Budget: ~6.5 GPU-hours on an L4 at L2W's measured 43 FPS; the armed
guards bound a dead run at ~1 GPU-hour.

## 4a. LR1 results — the ADOPT branch fires (2026-08-16)

Run `20260816_135919` (Colab L4, SHA 9d76f6a, TOML sha `84669842`,
seed 0, n_envs 4, 1M steps in 6h17m at 44 FPS; the single reward-side
change against the L2W baseline, everything else byte-matched
including the warm-start source hashes). Scored against the §4
frozen bars, each at its own best checkpoint:

| criterion | bar | measured | outcome |
|---|---|---|---|
| **K** (headline) | k=2 > 0% at some checkpoint | **1% at five checkpoints** (400k, 700k, 800k, 900k, 1M); the 700k instance a confirmed in-bounds second-exchange return at oracle depth (3.90 m) — the campaign's first | **PASS** |
| E1 engagement | touch ≥ 40% | **41% at 900k** (series 18→26→23→29→23→29→30→36→41→38; L2W peak 27%) | **PASS** |
| E2 position | ready error ≤ 2.0 m | ≤ 2.0 at **eight consecutive checkpoints**, best **1.08 m** at 700k (oracle 0.89; L2W never below 2.15 and worsening) | **PASS** |
| E3 retention | k=1 receiving ≥ 60% twice consecutive | **68% → 83% → 68%** (800k–1M; L2W matched steps: ~20%) | **PASS** |
| R2 | inter-point recovery ≤ 6.0 m | 8.0–9.3 m, flat all run | FAIL |
| M | `legal_hit_count_a` > 0 at ≥ half the evals | nonzero at 39 of 40 evals | intact |

**Decision: K PASS + M intact + three of E1/E2/E3/R2 PASS →
ADOPT.** Both committed follow-ups have since shipped: the recipe
adopts `points_per_episode=None`, `contact_shaping=0.25`, and
`reach_shaping=0.25` with the L2W-hardened guard set (recipe-pin
tests flipped to adoption pins), and the registered run is
pre-registered —
[`paddle_tennis_registered_run_prereg_20260816.md`](paddle_tennis_registered_run_prereg_20260816.md)
(3M warm-started steps, bars frozen from this run's band, held-out
gate on reserved 4100–4199).

The run's headline numbers, for the era record: final eval
**+0.143 ± 2.12** (the campaign's first positive learned final;
statue −4.4, L2W final −1.88), recent-train −0.51, headline
crossings final 4.97 (best model remains the pristine 25k transfer
at 6.37 — no trained eval crossed the 6.62 confirmation bar).
Behavioral endpoint (1M row): 72 policy hits per 30-episode probe,
receiving 75% crossed / 53% in at 3.23 m, touch 38%,
`policy_never_reached` down to 61% of points (from 84% at the L2
era's floor), the opponent losing 5–7 points per row from 800k, and
— first sign of life in the dead channel — **serving-side k=1 12%**
at 1M (11 serve-side hits, up from 0–3% everywhere since the
one-point era).

Mechanism read, consistent across all ten rows: the paid window
organized (ready error 2.49 → 1.08 m, the cleanest monotone
improvement in the campaign) while the unpaid windows stayed noisy
(recovery-hold 4.5–7.3 m, inter-point 8.0–9.3 m — R2's FAIL). The
same paid-vs-unpaid split that ranked this design now names the next
amendment if the registered run stalls at k=2 ≈ 1%: extend the
event-escrow family to the post-swing hold window (pay position at
the opponent's strike), not a motion tax — the statue attractor
stays the failure mode to respect. The quiet-optimizer pattern
persists even in success (ent_coef 1.5e-4, std 0.021 at end) and
remains a logged non-signal under the behavioral M.

Seed ledger: unchanged (run on seed 0; diagnosis calibration 5200+;
4100–4199 and 4300–4399 still reserved and untouched).

## 5. What this is not

Not a change to the frozen task semantics (default off; rules,
serve, observations, termination untouched — the rules machine
already recorded every quantity this design reads), not a
restructuring of the shared `+1` or the fault penalty (the
own-credit and fault-asymmetry directions stay behind this, per the
review's ranking), not a serve or opponent curriculum (measured
insufficient alone / rejected respectively), and not the churn
control (named as LR1's fallback arm, kept training-side and
separate so the reward change is measured clean).
