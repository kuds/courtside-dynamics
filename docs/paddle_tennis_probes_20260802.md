# PaddleTennis probes P0–P2 — the premise holds, the court shrinks, and loft is the frontier

Status: review snapshot, 2026-08-02, of the first PaddleTennis probe
battery ([`design_paddle_tennis.md`](design_paddle_tennis.md) §6,
P0–P2) run on a scratchpad prototype: the wall-ball-calibrated ball
and paddles (verbatim geom/actuator parameters, face pitch ±10°)
mirrored across a net at x = 0, raw MuJoCo, scripted controllers, no
learning, calibration-seed jitter only. No repo code shipped; these
numbers exist to freeze (or kill) the task definition before any env
class is written.

## TL;DR

- **P2, the premise the pivot rests on, holds.** In-rally strokes
  land at mean **3.3 m** past the net (p10 2.3, p90 4.4, 50% deeper
  than 3 m) — a real depth *distribution*, produced by an opponent's
  strokes. The wall's rebounds landed at mean 1.0 m and never deep;
  the reason wall-ball could not host a baseline era does not exist
  on the two-paddle court.
- **P0: the full-scale ITF court is infeasible for these paddles.**
  From x = −11.885 essentially nothing clears any net height at the
  paddle's ~12.5 m/s ceiling (real racquets triple that). The
  feasible geometry is a **~6.5 m half-court with the regulation
  0.914 m net**, where strokes from every depth have net-clearing,
  in-court cells. Net height barely matters between 0.914 and 1.07;
  lowering to 0.70 mostly widens the *shallow* envelope.
- **P1: scripted rallies exist — best band 2.0 crossings/point,
  max 10, ≥4 crossings in 33% of points.** Points are bimodal
  (median 0): roughly half die on the serve-return, the rest develop
  into genuine multi-exchange rallies. Serve-return is the hard
  skill, exactly as it was in wall-ball.
- **The era's core difficulty is identified and it is loft.** With a
  fixed-pitch face, hard swings sail long and soft swings find the
  net; the playable window between the two is narrow. A strike-height
  offset (meeting the ball ~0.12 m below center) measurably widens
  it — net faults 20 → 5, landings deepened to the 3.3 m band —
  proof that strike height is a real control channel a learned
  policy can exploit, and the first quantitative motivation for the
  parked paddle-pitch actuation design if learned policies saturate
  the same way.

## 1. Method

Prototype XML (scratchpad only): floor, net box at x = 0 (height
settable), the wall-ball ball verbatim (mass/restitution/priority
contact params), two wall-ball paddles verbatim (kp 300 / kv 18 /
±100 N position servos, ±10° fixed face pitch, damping 8) at bases
x = ∓1.7 with mirrored slide ranges. Controllers are a compact
world-frame port of the certification `lead_charge` family: ballistic
y/z lead, home-park when the ball is outgoing, charge-and-swing
through the ball toward the net when it comes inside a trigger gap.
Serve: ballistic launch from side A's mid-court drawn from the
P0-viable envelope (10–12 m/s, 18–24°, wall-ball-style jitter);
the rally begins with B's return. 60 points per configuration,
calibration seeds.

## 2. P0 — flight/net feasibility map

450-cell deterministic grid: launch position × height × speed
(8/10/12) × elevation (5–25°) × net height (0.70/0.914/1.07),
measuring net clearance and landing depth.

| launch x | net 0.914: cells clearing + landing in ±6.5 court | note |
|---|---|---|
| −3 | 6/30 | wide envelope, needs elevation ≥10° |
| −5 | 5/30 | v ≥ 10 and 15–25° |
| −7 | 4/30 | **v = 12 required**, 20–25° |
| −9 | 2/30 | v = 12 at 20–25° only |
| −11.885 (ITF) | ~0/30 at every net height | full court infeasible |

Findings: depth costs speed ~linearly and the paddle's ceiling
(~12.5 m/s terminal under the ±100 N / damping-8 servo) is exactly at
the "hit from the baseline" requirement of the 6.5 m court — deep
play is *possible but maximally demanding*, which is the right
difficulty placement for the skill the campaign wants. Elevation
(15–25°) matters more than raw speed; the fixed 10° face supplies
part of it, ball-reflection geometry and strike height the rest.

## 3. P1 — scripted rally band (the era's reference numbers)

Controller sweep over charge gap × swing-through × strike offset;
frozen best configuration **gap 0.8 m, swing 0.4 m, strike −0.12 m**
(60 points, seeds 1000–1059):

- **Crossings per point: mean 2.02, median 0, max 10; ≥2 in 40%,
  ≥4 in 33%.**
- Failure taxonomy: 55/60 points end with a stroke landing out,
  5/60 in the net, 0 double bounces, 0 timeouts.
- The sweep's shape is the finding: swing 1.0 → every point ends
  "out"; swing 0.3 → net faults explode (20/60) while the landings
  that do occur are deepest. Depth control spans a knife's edge that
  the strike-height offset visibly widens.

The band is deliberately conservative — a hand-tuned controller with
three scalar knobs. Wall-ball's history says learned policies beat
their scripted bands by 1.3–1.5×; a learned rally policy has strictly
more control channels (continuous swing modulation, per-stroke strike
height) than the controller that produced these numbers.

## 4. P2 — return-landing depth (the premise measurement)

Landings of in-rally crossings at the frozen configuration: **mean
3.33 m past the net, p10 2.25, p50 3.33, p90 4.41; 50% deeper than
3 m** (softer-swing configs reach p90 6.1). Compare the wall: rebound
landings mean 1.0 m, never past 6.1 m *from a court whose play was
8 m deep* — effectively never deep. On the two-paddle court, deep
balls arrive because the opponent hits them, at a rate that makes
covering depth instrumentally necessary. The pivot's premise is
measured, not assumed.

## 5. What freezes, what stays open

Frozen by this battery (pending P3–P5 before any env ships):
- Court: **half-length 6.5 m, net 0.914 m**, singles width ±4.115.
- Paddles/ball: wall-ball calibrations verbatim; no power changes —
  the ceiling *is* the difficulty placement.
- Reference band to calibrate certification against: the P1 numbers
  above, with the certification floor philosophy from 0.25.0
  (`feasibility_ge2_floor` equivalent set from the measured band,
  never from aspiration).

Open, assigned to the remaining probes: serve rules and alternation
(P3 — the ballistic serve is a probe device, not a design); the
mirroring identity (P4); champion transfer (P5); and the observation
layout. One new open question raised by the battery: whether
strike-height alone gives a learned policy enough loft authority, or
whether paddle-pitch actuation
([`design_wall_ball_paddle_orientation.md`](design_wall_ball_paddle_orientation.md),
parked since 0.22.0) graduates from consistency lever to era
prerequisite. That question should be answered by the first learned
runs, not pre-empted.

## Seed ledger

Scripted probes only, on burned calibration ranges (P0 deterministic
grid; P1/P2 seeds 1000–1059). No clean blocks touched; 3100–3199 and
4100–4199 remain reserved.
