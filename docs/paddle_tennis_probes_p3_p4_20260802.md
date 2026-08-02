# PaddleTennis probes P3–P4 — the serve band is measured, the mirror is exact

Status: review snapshot, 2026-08-02, of probes P3 (serve rules) and
P4 (mirroring identity) from the pre-committed battery
([`design_paddle_tennis.md`](design_paddle_tennis.md) §6), run on the
now-committed prototype scene (`envs/_paddle_court.py`,
`assets/paddle_court.xml`) — the P0–P2-frozen geometry wired through
the shared rules/events/paddle machinery. P5 (champion transfer)
remains open and still gates the opponent-pool decision; the
env-definition freeze can now proceed on P3/P4's numbers.

## TL;DR

- **P4 passes at the strictest bar.** A mirrored state produces
  **bit-for-bit identical** side-relative observations, a mirrored
  action commands the **sign-exact** mirrored world target, and
  mirrored trajectories stay mirrored through 40 control frames of
  contact-rich physics to `1e-6` (MuJoCo constraint-ordering ulps, far
  below anything a policy or the rules can resolve). One policy can
  play either side through the mirror; deterministic tests pin all
  three properties (`tests/test_paddle_court.py`).
- **P3 finds a real serve band**: origin **3.25 m behind the net,
  9 m/s, 21–24°** — 78% of jittered draws are legal, land mean
  **4.7–5.0 m deep**, and **100% of legal serves are returnable** by
  the frozen receiver. A second band sits at origin 4.5, 10 m/s,
  21–24° (68–70% legal, 4.8–5.2 m, 96–100% returnable).
- **The serve origin must clear the server's own home column.**
  From origin 2.0 m (0.3 m behind the paddle's home at 1.7 m),
  14–21 of 40 points ended `wrong_hitter` — the serve flies through
  the server's own paddle even in the neutral yield posture. Rule:
  serve from ≥ ~1.5 m behind the home column.
- **The loft knife-edge governs serve-return too.** Soft serves
  (8 m/s) from deep origins arrive short (2.1–2.6 m) and the frozen
  receiver's returns die in the net (`ball_net` 22–31/40, returnable
  0–12%); fast serves (11 m/s) overshoot legality almost everywhere.
  Exactly the P1 finding, now measured on the serve-return.
- **Alternation is provably fair**: every side-B twin cell reproduces
  its side-A primary's statistics **exactly** (36/36 pairs) — P4's
  identity observed end to end through serve, rally, and rules.

## 1. Method

Committed harness (`tools/paddle_tennis_probes.py`), scripted only.
Grid: origin depth {2.0, 3.25, 4.5} × speed {8, 9, 10, 11} m/s ×
elevation {18, 21, 24}° with the probe-standard jitter (position
(0.25, 0.5, 0.05), speed ±1, elevation ±3°, lateral ±4°), 40 points
per cell, each cell run from both serving sides on a shared seed
block. Each point is played twice on the same serve draw:

- **Parked pass** — both paddles hold the neutral yield posture; the
  serve's crossing, first-bounce legality, and landing depth are
  policy-independent (the wall-ball certification's parked-paddle
  instrument).
- **Active pass** — both sides run the P1-frozen `lead_charge` port
  (gap 0.8, swing-through 0.4, strike −0.12): returnable = the rules
  confirm the receiver's return of a parked-legal serve.

Two court-specific controller lessons are baked into the port and its
tests: the server **yields** (neutral park) while the ball is
outgoing, because serves launch from the server's own half and pass
through the home column; the receiver pre-positions (tracks y/z) only
while the ball is incoming.

## 2. P3 — the serve map (per-cell, 40 points, side A shown; every
side-B twin is identical)

Selected rows; the full 72-cell table is reproducible via
`python tools/paddle_tennis_probes.py --points 40 --seed-start 1200`.

| origin | speed | elev | legal | landing depth | returnable | note |
|---|---|---|---|---|---|---|
| 2.00 | 8–10 | any | 5–50% | 4.1–6.2 | 94–100% | `wrong_hitter` 14–21/40: origin fouls the server's own paddle |
| 3.25 | 8 | 18 | 72% | 2.56 | **10%** | short serve; receiver's returns die in the net |
| **3.25** | **9** | **21** | **78%** | **4.69** | **100%** | **recommended primary band** |
| 3.25 | 9 | 24 | 78% | 5.03 | 100% | primary band, deeper |
| 3.25 | 10 | 18 | 75% | 5.48 | 100% | hotter variant, deep |
| 3.25 | 11 | any | 10–32% | 5.9–6.3 | 100% | overshoots legality |
| 4.50 | 8 | any | 52–75% | 2.1–2.2 | 0–4% | deep-soft = short + net faults |
| 4.50 | 9 | 24 | 80% | 3.87 | 66% | best legality of the grid |
| 4.50 | 10 | 21–24 | 68–70% | 4.8–5.2 | 96–100% | secondary band |
| 4.50 | 11 | 18 | 68% | 5.48 | 100% | secondary band, flat |

Findings for the env freeze:

1. **Serve origin**: mid-court, ≥ ~1.5 m behind the server's home
   column. 3.25 m behind the net is the measured sweet spot; 2.0 m is
   disqualified outright (self-touch), and 4.5 m narrows the loft
   window (soft draws arrive short).
2. **Speed/elevation**: 9 m/s at 21–24° from 3.25 m. Legality falls
   off a cliff above 10 m/s (the court is 6.5 m deep and the P0
   ceiling is ~12.5 m/s: serve pace and court depth are the same
   currency).
3. **Fault budget**: at the probe-standard jitter, ~22% of primary-band
   draws are faults (long). The env definition must choose between a
   tighter draw (jitter is the biggest lever: the band's failures are
   almost all `out_of_bounds` long) and an explicit re-serve/fault
   rule. Recommendation: tighten the draw for P1-phase cooperative
   rally (a fault serve teaches nothing) and revisit service faults
   when scoring arrives (design doc phase P3).
4. **Alternation**: strict per-point alternation is fair by
   construction — the mirror makes the two sides' serve statistics
   identical, observed exactly in all 36 cell pairs.
5. **Rally tail with the port**: mean return crossings peak at 0.93
   (active pass, hot serves) — below the P1 prototype's 2.02 band.
   The committed port is more conservative than the scratchpad
   controller the probes doc froze; its numbers are the *committed*
   reference going forward and live in this harness, reproducible to
   the seed. (The P1 lesson stands: returns overwhelmingly die
   `out_of_bounds` long, with `ball_net` taking over as swings
   soften.)

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
now be frozen and certified: geometry (P0), reference band (P1),
premise (P2), serve rules (P3), and the observation/action mirroring
contract (P4) are all measured. Still open:

- **P5 — champion transfer**: both wall-ball champions
  (`20260731_132322`, `20260801_144043`) dropped onto the new court vs
  the oracle. Requires pulling the checkpoints + normalizers from
  Drive and an observation shim from the paddle-court state to the
  wall-ball 23-dim template; decides whether the phase-P2 opponent
  pool starts warm. Gates the opponent pool, not the env definition.
- The observation layout freeze (the committed candidate is physical
  state only; the env adds the rally/bookkeeping tail the way
  wall-ball did) and the serve-jitter tightening from finding 3.
- The P0–P2 open question (paddle-pitch actuation as loft authority)
  is untouched by this battery and still belongs to the first learned
  runs.

## Seed ledger

Calibration only, no clean blocks touched: this battery burned
**1200–2639** (36 configurations × 40 points, shared between each
side-A/side-B twin pair) plus 1000–1119 in harness bring-up
diagnostics. The reserved held-out blocks **3100–3199** and
**4100–4199** remain untouched, as does every block in the wall-ball
ledger. P4's randomized-state tests generate states (not evaluation
episodes) from seeds 8100/8150, outside every ledgered range.
