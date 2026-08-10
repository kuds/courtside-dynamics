# Design: n-point episodes — continuous play with position carryover

Status: **Implemented and certified** (default off), 2026-08-10 —
NP0/NP1 PASS, the NP2 band recorded with the frozen L2 bars, and
NP3 held-out certification PASS (all in §4a); the L2 pilot
remains. The committed
follow-up from the contact-shaping extension verdict
([`design_paddle_tennis_contact_shaping.md`](design_paddle_tennis_contact_shaping.md)
§5): k=1 is close to mastered on both sides (receiving survival
100%, serving 53%, strokes 78% in at oracle depth), and the sole
remaining barrier is post-swing wander (recovery-hold travel 8.33 m
vs the oracle's 2.2) — the policy has never sampled a k=2 return in
3M cumulative steps because it is out of position when the reply
arrives. This amendment converts that never-sampled k=2 credit into
densely-sampled inter-point credit.

## 1. The load-bearing decision: positions carry over

An episode becomes a fixed 1500-step block of **continuous play**:
when a point ends in a rally fault, the episode does not end — the
fault pays its −1, the rules state resets, the serve alternates,
and the next point's feed launches — with both paddles where play
left them (one narrow exception in §2's relaunch protocol).

That property is what makes the amendment answer the measured
failure, through **both serve parities** (strict alternation means
the wanderer faces each on alternating boundaries):

- Wander after a *receiving-point* serve-return (the archetype the
  extension measured): the policy **serves** the next point, so its
  next collectible is the serving-side k=1 reply — collected at
  53% at the extension's end, worth `+1 (+0.25 kept escrow)` —
  arriving after the serve's and the opponent-return's flight
  (~2 flights of re-ready time).
- Wander after a *serving-point* reply: the policy **receives** the
  next point, and the collectible is the receiving serve-return it
  already collects at 100% when in position, ~1 flight away.

Either way the wander now has a near-term, already-sampled price —
unlike the k=2 return, which has never been sampled at all. The
gradient is denser on receiving boundaries (100% channel) than
serving ones (53% channel); the probes measure both rather than
assuming one number covers both parities.

## 2. Definition changes (a new comparability era)

- **Env kwarg `points_per_episode: int | None = 1`** (validated:
  `None` or int ≥ 1). Default **1 = the frozen current behavior,
  bit-identical** — the amendment ships off, exactly like
  `volley_rule` and `contact_shaping` did. `None` means "as many
  points as fit the step cap": the episode always runs to the
  1500-step truncation (or an unsafe termination), uniform-length
  continuous play. The recipe adopts `None` only after the probes
  and pilot pass. Intermediate integers exist for probes, not for
  the recipe.
- **Point-boundary semantics.** A rally fault inside a multi-point
  episode: pays the fault penalty that step, claws back any pending
  escrow (**the escrow's clawback boundary becomes the point** — a
  hit that has not confirmed by its point's end can never confirm),
  resets the rules machine, alternates the server, and relaunches
  per the protocol below. Unsafe / nonfinite endings still
  terminate the episode immediately, exactly as today.
- **The point relaunch protocol** (the delicate mechanism, spelled
  out because two measured hazards live here):
  1. *Launch-cell clearance.* The serve cell (side-local
     x = −3.25 ± 0.25, y = 0 ± 0.5, z = 1.3 ± 0.05) lies inside
     the paddles' reachable workspace, and today's clearance comes
     entirely from the per-episode re-park this design removes —
     the P3/P4 probe snapshot already prescribed the missing
     guarantee ("keep the launch clear of the server's paddle
     envelope explicitly"). At each relaunch the serve draw is
     clearance-checked against both paddle envelopes (with a
     margin); up to K re-draws of the jittered origin are taken,
     and if none clears, the offending paddle is displaced the
     minimal distance out of the launch envelope — the one
     sanctioned exception to carryover, counted in info
     (`point_serve_nudged`) so the probes and diagnosis can see
     how often the referee had to step in. A paddle merely in the
     serve *corridor* (not the launch envelope) is deliberately
     NOT cleared: deflecting or self-touching the feed from out of
     position is legitimate, *event-visible* gameplay cost —
     wrong-hitter/premature faults, not silent corruption.
  2. *Sampler re-priming order.* The crossing detector and
     contact latch are stateful: the relaunch must teleport the
     ball to the cleared draw **first**, then re-prime the event
     sampler from that state, then resume stepping — never
     mid-control-step. Without the re-prime, the first substep of
     the new point emits a spurious net crossing from the dead
     ball's final side and the fresh rules machine faults it as a
     reverse crossing; with re-priming *before* the teleport, a
     latched stale contact can silently suppress real feed events.
     The clearance check in (1) guarantees no **ball** contact of
     any latchable channel exists at prime time. One non-ball
     channel could in principle latch across a boundary: a paddle
     in sustained contact with the net. Its semantics are pinned
     rather than cleared: net-touch faults fire on contact rising
     edges, so a net touch persisting across a boundary is not
     re-faulted in the new point (one fault per contact episode —
     and if the prior point ended on a *different* fault in the
     same step batch, the sustained touch goes unfaulted
     entirely). Accepted: no reward channel involves the net, so
     leaning on it gains nothing. **Measured at NP1**: on the
     paddle-court scene the case is unreachable — the collision
     masks never pair the racket faces with the net panel
     (contype/conaffinity product 0 both sides) and the frozen
     workspace keeps both faces 0.017 m short of the net plane —
     so the probe asserts both from the compiled model instead of
     staging a live hold-against-net witness; the pinned semantics
     stay the contract for any future scene where the channel is
     live (the humanoid court).
  3. *Alternation and step budget.* The server flips per point,
     within and across episodes; **a truncation-cut partial point
     consumes its alternation turn** (the next episode's first
     server is the opponent of the last server used, completed or
     not). `step_number` and the cap are episode-scoped and never
     reset at boundaries.
- **Observations: unchanged.** The frozen 48-value layout already
  carries the rally-state block that resets per point; the P4
  mirror contract is untouched.
- **Rewards: unchanged per event.** +1 shared confirm, −1 per point
  fault (now potentially several per episode), −2 unsafe, escrow
  shaping with the point-boundary clawback. Episode return becomes
  the sum over its points.
- **Info/metrics contract.**
  - `crossings` accumulates across the whole episode (offset
    carried across rules resets).
  - New keys: `points_played` (**completed points only** — the
    truncation-cut partial point is excluded), `completed_point_crossings`
    (cumulative crossings as of the last completed point's end, so
    the bridge metric's numerator and denominator are both
    completed-point-scoped and derivable from the info stream
    alone), `point_serve_nudged`, and cumulative per-group
    point-ending counters (`point_end_out_of_bounds`, …) mirroring
    the `term_*` taxonomy. Per-point rates everywhere divide
    completed-point numerators by completed-point counts; episode
    totals (`crossings`) still include the partial point's events.
  - The env's eight `term_*` group flags become strictly
    episode-ending descriptors: they fire only for what actually
    ended the episode — timeout at the cap, nonfinite/unsafe from
    the guards, and a rally-fault group only when the episode ends
    on a point fault (never under `None`; the final point under a
    finite `points_per_episode`). Note the rules
    snapshot's own `to_info` emits per-reason `term_<name>`
    booleans — four byte-identical to env group names, which the
    env's floats already overwrite today; that overwrite is part
    of the contract and stays. On absorbed-boundary steps the
    snapshot's `rally_terminal`/`termination_reason*` keys DO
    describe the point's fault (they reset with the machine next
    step); the `point_end_*` counters are the durable record.
    `serve_side_is_policy` likewise names the step's own
    (just-ended) point on boundary steps, agreeing with the
    snapshot's `serve_side` keys — the returned observation is
    what describes the relaunched point.
  - Recipe CSV/eval keys extend accordingly.
- **Diagnosis instrument: point segmentation.** The rules machine
  still emits a terminated transition at each point's end (the env
  absorbs it), so the instrument segments episodes into points on
  those transitions and reports the same per-point ledger as today,
  plus **two recovery metrics kept distinct**: the existing
  within-point recovery-hold travel (the 8.33 m instrument), and
  the new *inter-point recovery* (travel between a point's end and
  the next feed's arrival). The instrument upgrade ships with the
  env change.

**Comparability:** a task-definition change — a new era. Nothing
crosses the boundary: the ground-era band (7.78/point) and
certification (7.68) describe one-point episodes; the n-point era
gets its own scripted band, held-out certification, and
registered-run pre-registration. The per-point crossings rate is
the bridge metric reported on both sides of the boundary.

## 3. Seed-ledger assignments (frozen here)

- **5400–5499**: n-point scripted-band calibration (verified fresh:
  no prior doc, tool, or test uses the block).
- **4300–4399**: RESERVED — n-point held-out certification. Never
  touched by probes or tests. (Adjacent 4200–4299 is burned by the
  ground-era certification, and **4000–4099 is burned** by the
  wall-ball true-baseline certification — neither is clean.)
- **4100–4199**: RESERVED — re-dedicated to the n-point era's
  registered run's held-out gate. Named by exactly one prior
  pre-registration (the volley-era first-run plan, superseded
  unconsumed; the ground-era registered run was never
  pre-registered — its issuing branches never fired), and verified
  untouched by every tool and test.
- All other burned blocks in the repo ledger stay burned — the
  paddle-era burns (1000–1119, 1200–2639, 3100–3199, 4200–4299,
  5000–5099, 5100–5199, 5200–5299, 5300–5399) and the earlier
  wall-ball/humanoid-era blocks (4000–4099 among them); this list
  extends, never replaces, the corpus ledger.

## 4. Pre-registered probe battery (before the recipe adopts it)

- **NP0 — bit-identity of the default.** `points_per_episode=1`
  must produce bit-identical trajectories, rewards, and info
  streams to the pre-amendment env on the same seeds (lockstep
  test, the shaping batch's pattern).
- **NP1 — mechanics witnesses** (scripted, seeds 5400+):
  - *carryover*: at each boundary, paddle positions exactly
    continuous (except a sanctioned nudge, which must coincide
    with `point_serve_nudged` incrementing) while rules state
    resets and the server flips — asserted from state;
  - *relaunch hazards* (the review's reproduced failure modes,
    kept as witnesses): a paddle forced into the launch envelope
    at a boundary → the feed's events are intact (no silent
    latch-suppressed contact; feed lands where the serve model
    says), the nudge fires and is counted; a point ending with
    the ball on the far side → the next point's opening steps emit
    no spurious reverse-crossing fault; the net-touch case → the
    probe asserts from the compiled model that racket–net contact
    is unreachable on this scene (mask product 0, workspace short
    of the net plane), making the pinned one-fault-per-contact
    semantics vacuously safe here;
  - *escrow point-boundary identity*: per point, S1-style exact,
    with a pending-escrow-at-point-end case witnessed;
  - *statue under `None`*: nets exactly −1 on each of its receiving
    points; a serving point nets exactly 0 when the opponent's
    serve-return confirms (the shared +1 offsets the fault —
    measured at implementation smoke) and exactly −1 when the feed
    or the return itself faults, so serving points read in
    {0, −1} with the 0 case witnessed; `points_played` matches the
    completed-fault count, episodes always truncate at the cap;
  - *alternation*: strict across points and episode boundaries,
    including the partial-point-consumes-a-turn rule.
- **NP2 — the oracle-pair band, recalibrated** (seeds 5400+, 100
  episodes): crossings per episode and per completed point, points
  per episode, nudge frequency, and the oracle's own inter-point
  recovery travel (the policy's target for R2). The scripted pair
  under carryover is a genuinely new measurement — the serving
  oracle starts each point from wherever play left it, not the
  engineered park — so NP2's numbers are taken as the era's
  reference, not assumed near the ground band.
- **NP3 — held-out certification** (reserved 4300–4399, floors
  pre-registered from NP2's band before the block is opened).
- **L2 — learning pilot** (recipe + `points_per_episode=None` +
  `contact_shaping=0.25` via TOML, seed 0, 2M steps, n_envs 4,
  cadence 100k). Numeric bars are frozen in an L2 pre-registration
  addendum at NP2 time (they must be stated in the new era's
  units); the criteria set and decision-rule shape are committed
  now, middles to be declared alongside the numbers:
  - **K (the point of the era): k=2 exchange survival > 0%** at
    some checkpoint — the number that has never moved;
  - **R1: within-point recovery-hold travel** falling from the
    extension's 8.33 m (same instrument, commensurable);
  - **R2: inter-point recovery travel** approaching the oracle's
    NP2-measured value (the new instrument, its own bar);
  - **P″/D2″**: crossings-per-completed-point and touch bars from
    NP2;
  - **M**: mechanism intact (unchanged two-part check).

## 4a. Results and the frozen L2 pre-registration (2026-08-10)

**NP0 — PASS, cross-version.** Beyond the in-tree lockstep test,
the default env was run against the actual pre-amendment code (the
merge commit, in a separate worktree) on seeds 5400–5402:
observations, rewards, and every pre-era info column bit-identical
over 2,613 steps (episodes of 1500/794/319); the post-amendment
info stream differs only by the ten additive n-point keys.

**NP1 — PASS (30/30)** (`tools/paddle_tennis_npoint_probe.py`,
seeds 5400+): 315 un-nudged
boundaries continuous (max jump 0.123 m vs the 0.5 m bound); every
relaunch spawn inside the mirrored serve cell, clearing both
paddles by ≥ 0.451 m at ≤ 9.96 m/s (a serve draw, never the
41 m/s deflection hazard); the parked-paddle witness drew 6 nudges
over 107 boundaries, every one coincident with its counter; 116
hard-slam far-side points opened with zero illegal-hit faults (the
reverse-crossing hazard absent); escrow identities exact
(0.00e+00) on bit-identical shaped/unshaped arms with 58 boundary
clawbacks witnessed; statue economics exact — every receiving
point −1, serving points 30 offset to 0 / 10 at −1 (the feed or
return faulting kills the offset; the probe's criterion was
corrected from "serving points net 0" to "read in {0, −1} with
the 0 case witnessed" before the battery ran); alternation strict
across 397 launches. The net-touch case measured **unreachable**
on this scene: collision-mask pairing product 0 both sides and a
0.017 m workspace margin short of the net plane, asserted from
the compiled model (§2's pinned semantics stay the contract for
scenes with a live channel).

**NP2 — the era's reference band** (100 episodes, seeds
5400–5499, `points_per_episode=None`, scripted pair):

- crossings/episode mean **11.40** (std 1.58, min 8, max 14);
- completed points/episode mean **1.13** (std 0.91) — the pair's
  rallies often outlive the cap; 113 completed points total;
- crossings per completed point (bridge metric) **4.46**;
- nudges **0** of 113 relaunches (the referee never stepped in on
  the scripted pair; the parker witness covers the path);
- inter-point recovery travel mean **2.10 m** (p90 3.97) — the R2
  oracle reference;
- within-point recovery hold mean **2.22 m** (p90 3.59) — the R1
  oracle reference, matching the old era's 2.2;
- touched-after-bounce 99% (the D2″ oracle reference); ready
  error 0.89 m vs the designed 0.90 (instrument self-check);
- zero unsafe episodes.

Under carryover the oracle pair's own exchange survival drops
(receiving k=1 92%/k=2 75%; serving 88%/68% vs ~100% from the
engineered park) — exactly why NP2 is the era's reference and the
ground band retires at the boundary.

**The frozen L2 bars** (pre-registered here, before any L2 step
runs; each metric reads from the automated diagnosis rows/CSV at
its own best checkpoint; run shape: recipe + TOML
`[env] points_per_episode = "none"`, `contact_shaping = 0.25`,
seed 0, 2M steps, n_envs 4, cadence 100k):

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| **K** (headline) | k=2 exchange survival, either parity | 0% at every checkpoint | — (binary) | > 0% at some checkpoint |
| R1 | within-point recovery-hold mean | ≥ 8.33 m (the extension's end) | (5.0, 8.33) | ≤ 5.0 m |
| R2 | inter-point recovery mean | > 8.4 m (4× oracle) | (4.2, 8.4] | ≤ 4.2 m (2× oracle) |
| P″ | crossings per completed point | < 1.8 (k=1-only ceiling) | [1.8, 2.5) | ≥ 2.5 |
| D2″ | touched-after-bounce rate | < 46% (the L1 peak) | [46%, 60%) | ≥ 60% |
| M | mechanism (ent-coef anneal + train/std ≥ 5e-3) | broken | — | intact |

The P″ FAIL floor is the k=1-only ceiling translated into
per-completed-point units from the extension's measured rates
(receiving ≈ 1.7 crossings/point at 100%·78%-in, serving ≈
1.5–1.9 at 53%; an approximation, labeled as such). R2 has no
policy baseline (the metric is new), so its bars anchor to the
only measured reference, the oracle, at a priori factors 2× and
4×. Two PASS bars are labeled a-priori choices rather than NP2
read-offs (deviating from §4's "bars from NP2" shorthand): P″
PASS 2.5 is the L1-era P′ crossings bar restated in
per-completed-point units (≈ 0.56× the oracle's 4.46 bridge
metric), and D2″ PASS 60% sits between the L1-era measured touch
peak (46%) and the oracle's 99% reference; their FAIL floors are
the measured anchors.

**Decision rule** (three branches, the L1 shape):

- **Adopt**: K PASS, M intact, and ≥ 2 of R1/R2/P″/D2″ PASS →
  the recipe adopts `points_per_episode=None` (+ shaping) and the
  6M registered run is pre-registered with its held-out gate on
  4100–4199.
- **Extend once** (declared middle, non-forcing): M intact and
  either K PASS with < 2 others passing, or K FAIL with ≥ 2
  others in PASS-or-middle → a single 2M extension of the frozen
  lean (the shaping playbook); its verdict re-applies this rule
  with the extend branch removed.
- **Stop/pivot**: M broken, or K FAIL with < 2 others in
  PASS-or-middle → n-point stays default-off; the next probed
  change targets the opponent/curriculum side.

**NP3 floors** (pre-registered here from NP2's band, before the
reserved block 4300–4399 was opened; scripted pair, 100 episodes,
`points_per_episode=None`, one shot):

- mean crossings/episode ≥ **9.0** (0.79× the NP2 mean — the
  prior certifications' convention);
- total completed points ≥ **50** (NP2 measured 113);
- nudge rate ≤ **2%** of relaunches (NP2 measured 0%);
- zero unsafe terminations.

**NP3 — PASS** (`--certify`, the block's single sanctioned
opening, 2026-08-10): mean crossings/episode **11.51** (std 1.47),
**106** completed points, **0%** nudges, **0** unsafe — every
floor cleared, and the held-out band replicates the calibration
band closely (crossings 11.51 vs 11.40; bridge metric 4.56 vs
4.46; inter-point recovery 1.99 m vs 2.10 m; ready error 0.89
both). Seeds 4300–4399 are now consumed. The definition is
certified; what remains before recipe adoption is the L2 pilot
against the bars above.

## 5. What this is not

Not a reward change (the shaping design owns that; its semantics
extend unchanged to point boundaries), not an observation or
mirror change, not an opponent change, and not the adversarial
scoring phase (P3 of the original design sketch) — this is still
the cooperative rally, played continuously.
