# PaddleTennis probes P3–P4 — a clean serve band, a rally tail above P1, an exact mirror

Status: review snapshot, 2026-08-02, of probes P3 (serve rules) and
P4 (mirroring identity) from the pre-committed battery
([`design_paddle_tennis.md`](design_paddle_tennis.md) §6), run on the
committed prototype scene (`envs/_paddle_court.py`,
`assets/paddle_court.xml`) — the P0–P2-frozen geometry wired through
the shared rules/events/paddle machinery. P5 (champion transfer)
remains open and gates the opponent-pool decision; the env-definition
freeze can now proceed on P3/P4's numbers.

## TL;DR

- **P4 passes at the strictest bar.** A mirrored state produces
  **bit-for-bit identical** side-relative observations, a mirrored
  action commands the **sign-exact** mirrored world target, and
  mirrored trajectories stay mirrored through 40 control frames of
  contact-rich physics to `1e-6` (MuJoCo constraint-ordering ulps, far
  below anything a policy or the rules can resolve). One policy can
  play either side through the mirror; deterministic tests pin all
  three properties (`tests/test_paddle_court.py`).
- **P3 finds a clean serve band**: origin **3.25 m behind the net,
  9 m/s, 18–24°** — **98–100% of jittered draws are legal**, land mean
  **4.3–4.8 m deep**, and **92–100% of legal serves are returnable**
  by the frozen receiver. Secondary band: origin 4.5 m, 10 m/s, 21°
  (98% legal, 4.65 m, 97% returnable).
- **The rally tail sits ABOVE the P0–P2 reference band**: mean
  cumulative return crossings **3.15–3.42** per point at the primary
  band (P1's scratchpad band was 2.02, max 10). The committed
  controller port plus the measured serve band out-rallies the
  prototype battery; these are the era's committed, seed-reproducible
  reference numbers.
- **Server self-touch is a flight-path hazard, not an origin rule.**
  Origin 2.0 m is disqualified outright (`wrong_hitter` 14–21/40 at
  every speed: the serve flies through the server's own yielded
  paddle), but soft-flat serves self-touch from 4.5 m too (10/40 at
  8 m/s / 18°). The rule for the env freeze: the serve's flight must
  clear the server's home column, which the primary band's loft does
  by construction.
- **The loft knife-edge governs serve-return.** Soft serves (8 m/s)
  arrive short (2.1–2.7 m) and the receiver's returns die in the net
  (`ball_net` 22–31/40, returnable 0–20%); 11 m/s serves overshoot
  legality from every origin (parked-pass long, 0–35% legal except
  4.5/11/18). Exactly the P0–P2 finding, now measured on the
  serve-return.
- **Alternation is provably fair**: every side-B twin cell reproduces
  its side-A primary's statistics **exactly** (36/36 pairs) — P4's
  identity observed end to end through serve, physics, contacts, and
  the rules reducer.

## 1. Method

Committed harness (`tools/paddle_tennis_probes.py`), scripted only.
Grid: origin depth {2.0, 3.25, 4.5} × speed {8, 9, 10, 11} m/s ×
elevation {18, 21, 24}° with the probe-standard jitter (position
(0.25, 0.5, 0.05), speed ±1, elevation ±3°, lateral ±4°), 40 points
per cell, every cell run from both serving sides on a shared seed
block. Each point is played twice on the same serve draw:

- **Parked pass** — both paddles **corner-parked outside the flight
  envelope**; the serve's crossing, first-bounce legality, and landing
  depth are policy-independent (the wall-ball certification's
  parked-paddle instrument, moved clear of play).
- **Active pass** — both sides run the P1-frozen `lead_charge` port
  (gap 0.8, swing-through 0.4, strike −0.12): returnable = the rules
  confirm the receiver's return of a parked-legal serve; crossings =
  cumulative return crossings (net crossings minus the feed's own).

Two measurement pitfalls were caught by the pre-push adversarial
review and are worth recording for the certification design:

1. **A parked paddle is an obstacle.** The first draft parked paddles
   at their home columns; the receiver's parked face passively
   volleyed in-flight serves and the deflections were scored as serve
   faults — the primary band's entire apparent 22% "fault budget" was
   this censoring artifact (it is 98–100% legal measured clear of
   play). Any "policy-independent" landing instrument on this court
   must move the paddles out of the envelope, not just zero their
   actions.
2. **`RallySnapshot.shot_crossing_count` is an end-state latch** (was
   the *final* shot across the net at termination), not a cumulative
   count. Rally-tail metrics must derive from `net_crossing_count`
   minus the feed. The first draft under-reported the tail by ~4×.

Controller notes baked into the port and its tests: the server
**yields** (neutral park) while the ball is outgoing — serves launch
from the server's own half and pass through the home column — and the
receiver pre-positions (tracks y/z) only while the ball is incoming.

## 2. P3 — the serve map (per-cell, 40 points, side A shown; every
side-B twin is identical)

Selected rows; the full 72-cell table is reproducible via
`python tools/paddle_tennis_probes.py --points 40 --seed-start 1200`.

| origin | speed | elev | legal | landing depth | returnable | crossings | note |
|---|---|---|---|---|---|---|---|
| 2.00 | any | any | 0–62% | 4.1–6.3 | 50–100% | 0.6–1.7 | `wrong_hitter` 14–21/40: serve fouls the server's own paddle |
| 3.25 | 8 | 18 | 100% | 2.70 | **20%** | 0.53 | short serve; receiver's returns die in the net |
| 3.25 | 9 | 18 | 100% | 4.27 | 92% | **3.42** | primary band; best rally tail of the grid |
| **3.25** | **9** | **21** | **100%** | **4.55** | **100%** | **3.23** | **recommended primary band** |
| 3.25 | 9 | 24 | 98% | 4.83 | 100% | 3.15 | primary band, deeper |
| 3.25 | 10 | 18 | 90% | 5.28 | 100% | 2.85 | hotter variant |
| 3.25 | 11 | any | 10–35% | 5.8–6.3 | 100% | 1.5–2.3 | overshoots legality |
| 4.50 | 8 | 18 | 52% | 2.12 | 0% | 0.00 | deep-soft-flat: short, netted, and 10/40 self-touch |
| 4.50 | 9 | 21 | 98% | 3.37 | 54% | 2.02 | mid band |
| 4.50 | 10 | 21 | 98% | 4.65 | 97% | 3.23 | secondary band |
| 4.50 | 11 | 18 | 80% | 5.32 | 100% | 3.00 | secondary band, flat |

Findings for the env freeze:

1. **Serve rule**: origin 3.25 m behind the net, 9 m/s, 21° ± the
   probe-standard jitter — 100% legal, mean landing 4.55 m (the deep
   arrival the campaign pivoted for), 100% returnable, rally tail
   3.23. No re-serve/fault machinery is needed at this band; service
   faults can wait for the scoring phase.
2. **The self-touch constraint is about flight path, not distance**:
   any draw whose trajectory passes low through the server's own home
   column risks fouling the yielded paddle (all of origin 2.0; soft-
   flat draws from 4.5). The primary band's loft clears the column by
   construction. The env's serve validation should keep the launch
   clear of the server's paddle envelope explicitly rather than
   trusting an origin-distance heuristic.
3. **Speed is the legality lever, elevation is the depth lever**:
   legality collapses above 10 m/s from every origin (the 6.5 m court
   against the ~12.5 m/s ceiling), while within the legal band
   elevation trades legality for landing depth (4.27 → 4.83 across
   18–24° at 9 m/s).
4. **Alternation**: strict per-point alternation is fair by
   construction — the mirror makes the two sides' statistics
   identical, observed exactly in all 36 cell pairs.
5. **Reference band for certification**: the committed
   port + primary band produce mean crossings 3.15–3.42 with failure
   taxonomy dominated by strokes landing out (36–40/40), `ball_net`
   taking over as swings soften — the P1 taxonomy at a higher tail.
   Certification floors for the first learned runs should be set from
   these committed numbers (band minus sampling error), not from the
   scratchpad battery's 2.02.

## 3. P4 — mirroring identity (the design's §6 bit-for-bit check)

Implemented as deterministic tests over the committed candidate
contracts (`tests/test_paddle_court.py::TestP4MirroringIdentity`):

- **Observations**: `obs(A, S) == obs(B, mirror(S))` with **zero**
  tolerance, across 25 randomized states (both pairings). The
  side-local frame is side A's world frame; side B reads the world
  through the exact 180° court rotation via the shared
  `mirror_for_side`.
- **Actions**: the same side-local action commands world targets that
  mirror **sign-exactly** through either paddle interface (IEEE
  negation is exact; the two paddles' ctrlranges are exact mirrors by
  XML construction).
- **Dynamics**: from mirrored states under the same side-local policy,
  trajectories remain mirrored through 40 control frames (200 physics
  substeps) of paddle-ball-court contact to `1e-6` — not bit-for-bit,
  because MuJoCo assembles the two paddles' contacts in different
  orders on the two sides; the drift is constraint-ordering ulps.
- **End to end**: the P3 sweep's 36 side-B alternation twins reproduce
  their side-A primaries' statistics exactly, through serve, physics,
  contacts, and the rules reducer.

## 4. What this unfreezes, what stays open

With P3 and P4 done, per the design doctrine the env definition can
now be frozen and certified: geometry (P0), reference band (P1, now
superseded by §2 finding 5's committed band), premise (P2), serve
rules (P3), and the observation/action mirroring contract (P4) are
all measured. Still open:

- **P5 — champion transfer**: both wall-ball champions
  (`20260731_132322`, `20260801_144043`) dropped onto the new court vs
  the oracle. Requires pulling the checkpoints + normalizers from
  Drive and an observation shim from the paddle-court state to the
  wall-ball 23-dim template; decides whether the phase-P2 opponent
  pool starts warm. Gates the opponent pool, not the env definition.
- The observation layout freeze (the committed candidate is physical
  state only, pinned to its names tuple; the env adds the
  rally/bookkeeping tail the way wall-ball did).
- The P0–P2 open question (paddle-pitch actuation as loft authority)
  is untouched by this battery and still belongs to the first learned
  runs.

## Seed ledger

Calibration only, no clean blocks touched: this battery burned
**1200–2639** (36 configurations × 40 points, shared between each
side-A/side-B twin pair and between the parked/active passes) plus
1000–1119 in harness bring-up diagnostics. The reserved held-out
blocks **3100–3199** and **4100–4199** remain untouched, as does every
block in the wall-ball ledger. P4's randomized-state tests generate
states (not evaluation episodes) from seeds 8100/8150, outside every
ledgered range.
