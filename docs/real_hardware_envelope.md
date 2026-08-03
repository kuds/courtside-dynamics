# Real-Hardware Envelope — the mounted-arm bin shot

**Kind:** Reference (external hardware) · **Date:** 2026-08-03 · **Pinned to:**
`main`@`aec8cd1`, v0.25.0 + unreleased PaddleTennis work

**Status: PROVISIONAL — every number below is analytic, none is measured.**
This document derives the physical envelope of a bench-top task from first
principles and from the calibrations already frozen in this repo's XML. Nothing
here has been run on hardware. Read it as a pre-registration of what to measure
(§7), in the same spirit as the P0–P5 probe battery that gated the PaddleTennis
env — not as a result.

## 0. The task

> A **mounted** (fixed-base) robot arm holding a real tennis racket. A human
> underhand-tosses a tennis ball toward it. The arm intercepts the ball in
> flight and returns it into a bucket or bin on the floor.

Fixed base — no locomotion, no mobile platform. The ball is *tossed*, not
served or machine-fed. Success is **landing in the bin**, not merely making
contact. That last clause is what makes this an *accuracy* task rather than a
*power* task, and it inverts the usual robot-tennis difficulty ordering.

Design point used throughout, chosen to be a comfortable underhand toss:

| Parameter | Value |
|---|---|
| Release height / distance | 1.0 m / 5.0 m from the arm |
| Release speed / elevation | 7.0 m/s at 40° |
| Flight time to the arm | **0.93 s** |
| Apex | 2.03 m at t = 0.46 s |
| Arrival at the arm | 0.95 m height, **7.1 m/s descending at 41°** |
| Bin | 0.4 m diameter, rim 0.5 m high, 5 m away |

## 1. Why this task is ~1/7th of a groundstroke

The headline result of the whole analysis:

| Stroke | Racket-head speed at contact |
|---|---|
| **This task (5 m bin shot)** | **2.5–3.5 m/s** |
| Recreational forehand | 18–25 m/s |
| Pro forehand | 30–35 m/s |
| Pro serve | 40–50 m/s |

**Derivation.** To land at horizontal distance *D* with a 0.5 m drop to the bin
rim, launched at the range-optimal ~45°:

```
D = (v cosθ / g) · ( v sinθ + sqrt(v² sin²θ + 2 g Δh) )
```

| Bin distance *D* | Required exit speed |
|---|---|
| 3 m | 5.0 m/s |
| 4 m | 5.9 m/s |
| **5 m** | **6.6 m/s** |
| 6 m | 7.4 m/s |

The racket speed that produces that exit speed follows from the apparent
coefficient of restitution *e*:

```
v_out = V·(1 + e) + e·u
```

where *V* is stringbed speed along the outgoing axis and *u* is the incoming
ball speed component being reversed (≈ 5 m/s of the 7.1 m/s arrival). With
*e* = 0.45, `6.6 = 1.45·V + 2.25` ⇒ **V = 3.0 m/s**. Sweeping the plausible
restitution range *e* ∈ [0.35, 0.60] gives V ∈ [2.3, 3.6] m/s.

**Consequence: a collaborative arm is sufficient for this task.** The usual
"cobots are too slow for tennis" verdict is about groundstrokes at 20+ m/s. At
3 m/s the constraint disappears — and it disappears *before* you exploit the
0.49 m lever from wrist to stringbed center, which converts modest wrist
angular rate into stringbed speed for free.

## 2. What is actually hard: aiming

Speed is not the binding constraint. Sensitivity is. All partials taken at the
5 m design point, bin half-width ±0.2 m:

| Error source | Sensitivity | Tolerance to hit the bin |
|---|---|---|
| Exit speed | dD/dv ≈ **1.4 m per m/s** | **±0.14 m/s (±2%)** |
| Racket speed | dv_out/dV = 1 + e ≈ 1.45 | **±0.10 m/s** |
| **Incoming ball speed** | dv_out/du = e ≈ 0.45 | **±0.31 m/s of toss variation** |
| Launch elevation | 45° is the range maximum; ±5° costs ~0.5–0.7 m | **±2°**, and errors are one-signed (always short) |
| Face azimuth | maps ~1:1–2:1 into outgoing direction | **±1°** |

Three things fall out of that table.

**(a) Per-throw compensation is mandatory, not optional.** A human underhand
toss varies by well over ±0.5 m/s throw to throw. At *e* = 0.45 that is ±0.22
m/s of exit speed, i.e. ±0.31 m of range — comparable to the whole bin. The
controller must **measure *u* from the observed track and correct the commanded
racket speed**:

```
ΔV = −( e / (1 + e) ) · Δu  ≈  −0.31 · Δu
```

An open-loop swing that ignores incoming speed will scatter around the bin even
with a perfect arm.

**(b) Bias for the short side.** Because 45° is the range *maximum*, elevation
error is one-signed: every angle error lands short. Aim slightly beyond the bin
center and let the error distribution fall into it.

**(c) Spin is the error source this repo cannot model at all.** The toss carries
spin, and any oblique (non-normal) contact adds more. Magnus force curves the
outgoing ball laterally, and it is entirely absent from every simulation here
(§4). Mitigation: strike as close to face-normal as the geometry allows,
accepting a lower *e*, to suppress spin generation.

## 3. Timing and perception budget

0.93 s of flight is luxurious compared to a real stroke — this is the second
way the task is easier than it sounds.

| Phase | Window | Note |
|---|---|---|
| Detect + track | t = 0 → 0.40 s | 43% of flight; ~50 frames at 120 fps |
| Fit + predict intercept | → 0.50 s | ballistic + quadratic drag |
| **Commit** | t ≈ 0.55 s | ~0.35 s before contact; do not re-plan after |
| Swing | 0.55 → 0.93 s | 0.5 m travel + 3 m/s build ⇒ < 10 m/s² |

**Cameras: 90–120 fps global-shutter stereo is enough.** A groundstroke-return
robot needs 200+ fps because it has ~0.4 s of flight; here, even 60 fps yields
~55 frames. This is the cheapest part of the build and does not warrant machine
vision-class hardware.

**Budget latency jitter, not latency mean.** At 7 m/s, 60 ms of end-to-end lag
is 0.42 m of ball travel — but a *constant* 60 ms is just a fixed offset you
calibrate out. Variance is what misses the bin. Measure the distribution
(probe B1, §7), not the average.

**Drag matters at this scale, mildly.** Evaluating the repo's own
`ball_drag_force()` (`_tennis_physics.py:123`, Cd 0.55, ρ 1.225, r 0.0335 m,
m 0.0577 kg) at 7 m/s gives **1.01 m/s², about 10% of gravity** — consistent
with the 5.27 m/s² @ 16 m/s row in `DECISIONS.md` §Physics reference values,
since the term is quadratic. Over 0.93 s of flight that costs roughly 0.6–0.9
m/s and ~0.25 m of range — **larger than the bin's radius**, so a drag-free
predictor will systematically overshoot. Include quadratic drag in the
trajectory fit (the repo's function is directly reusable), or calibrate the bias
out empirically. For contrast, at a 30 m/s serve speed the same function returns
~1.9× gravity — which is why serve return is a different problem, not a faster
version of this one.

## 4. Sim → real gap

The load-bearing section. What this repo simulates and what a bench would do:

| Quantity | This repo | Physical bench | Consequence |
|---|---|---|---|
| Ball radius | HumanoidTennis: **0.0335 m** ✓ (`humanoid_tennis.xml:113`). PaddleTennis / WallBall: **0.07 m** (`paddle_court.xml:130`) | 0.0335 m | The humanoid scene is regulation. The paddle/wall scenes run a **2.1× oversized** ball — ~4.4× the frontal area and a far larger contact target, so their contact rates are optimistic. |
| Ball mass | 0.0577 kg (humanoid) / 0.057 kg (paddle) | 0.057 kg | Matches. |
| Aerodynamic drag | HumanoidTennis: **modeled correctly** — `ball_drag_force()` applies quadratic drag via `xfrc_applied` every substep at Cd 0.55, ρ 1.225, regulation cross-section (`_tennis_physics.py:123`). PaddleTennis / WallBall / BallBounce: **none** — the function has no caller there, and no scene sets a fluid density | ~10% of g at 7 m/s | In the paddle/wall scenes trajectories are exact parabolas. **Do not port a paddle-court trajectory model to the bench**: a drag-free predictor lands the bin shot ~0.25 m short, which is larger than the bin's radius. The humanoid env's drag model is directly reusable. |
| Magnus / spin force | **none anywhere** — no lift term exists; `xfrc_applied` carries drag only | dominant lateral-error source | The repo cannot model the effect most likely to make you miss the bin, even in the otherwise-faithful humanoid scene. |
| Actuator | 3 slide joints, pure translation, ±100 N, ~12.5 m/s terminal velocity, in a 6.3 × 6 × 2.9 m box (`paddle_court.xml:105–144`) | 6-DoF revolute arm, ~0.85 m reach sphere, 1–3 m/s tool speed | **The shipped envs' "paddle" is a six-metre gantry, not an arm.** No reach limit, no rotational inertia, no gravity torque, no configuration-dependent Jacobian. |
| Hitting surface | 0.04 × 0.4 × 0.5 m slab, 0.35 kg, fixed 10° pitch | racket: 0.68 m long, 0.26 × 0.34 m face, 0.30 kg | The **racket asset** (`tennis_racket.xml`) is a good match for a real racket. The **paddle** every shipped env actually drives is not. |
| G1 arm torque | shoulder/elbow 25 N·m, wrist pitch/yaw 5 N·m (`unitree_g1_tennis.xml:65–92`) | same nominal spec | But the model has no joint-velocity ceiling, no thermal derating, and no backlash. |
| Contact restitution | `solref="0.01 0.15"`, calibrated for the wall-ball paddle | *e* ≈ 0.35–0.60, varying with impact speed and off-center distance | **Measure your own *e*.** It is the single most load-bearing number in §2's aiming model. |

**A useful cross-check on the humanoid side.** Taking the racket at ~0.9 m from
the G1's shoulder gives roughly 0.26 kg·m² for the racket alone, ~0.4 kg·m²
with the arm links. The 25 N·m shoulder limit then yields α ≈ 60 rad/s²; over a
0.25 s swing that is ~15 rad/s, or **~19 m/s at the racket tip and ~16 m/s at
the stringbed center**, before the 88 N·m waist yaw adds anything. So the
simulated humanoid sits at roughly *recreational-groundstroke* authority and
nowhere near a serve — the model is not wildly cheating on torque. It is
cheating on velocity limits, thermal limits, and gearbox tolerance to impact,
none of which it expresses.

### What transfers to a bench build

- ❌ **No trained policy transfers.** Every shipped env's action space commands
  a translating slab inside a 6 m box. There is no correspondence to an arm's
  joint or task-space commands. This is a hard architectural gap, not a
  fine-tuning gap.
- ❌ **PaddleTennis / WallBall contact-rate and rally-length numbers do not
  transfer** — oversized ball, no drag, no spin.
- ✅ **`ball_drag_force()` transfers as-is** for the trajectory predictor
  (§3). It is the correct quadratic model at regulation scale; it simply is not
  wired into the paddle/wall scenes.
- ✅ **`tennis_racket.xml` transfers as a specification.** 0.30 kg, CoM 0.36 m
  up the grip, tip at 0.682 m, diaginertia `0.0105 0.012 0.0018`. Use it to
  size the physical racket mount and to estimate the wrist's added inertia.
- ✅ **The methodology transfers, and is the most valuable export.** Cardinal
  rule 3 — *prove solvability with a scripted oracle before scoring against a
  bar* — is exactly the right discipline for a hardware bring-up, and §7 is that
  rule applied.

## 5. Hardware classes, for *this* task

Verify every number against the current datasheet before buying; the figures
below are order-of-magnitude guidance for sizing, not quoted specs.

**Collaborative arms — UR5e, Franka FR3, xArm 6, Kinova Gen3. Sufficient.**
Roughly 1–2 m/s at the tool flange, which clears the 2.5–3.5 m/s stringbed
requirement once the ~0.5 m grip-to-stringbed lever contributes. Real-time
interfaces exist and are the thing to confirm before purchase: Franka FCI
(~1 kHz), UR RTDE with `servoJ` (~500 Hz), xArm (~250 Hz). Two gotchas:

- **Collision detection will protective-stop on every hit.** Ball impulse is
  `m·Δv ≈ 0.057 × (5 + 6.6) ≈ 0.66 N·s` over a ~5 ms contact — roughly **130 N
  peak**, applied ~0.5 m from the wrist, so a ~65 N·m moment transient. Force
  and torque thresholds must be raised or the arm faults on contact. **Raising
  them means the arm is no longer collaborative in the safety-rated sense** (§8).
- **Safety-rated speed limits may cap you below the datasheet.** Check the
  configured limit, not the brochure.

**DIY quasi-direct-drive — 3× AK80-9 / RMD-X8 class, ~$1.5–3k. Sufficient, and
the best value if you intend to escalate.** Low gear ratios (6:1–9:1) are
inherently impact-tolerant, 1 kHz torque control over CAN is available, and
yaw + shoulder pitch + wrist on a carbon tube reaches 5–8 m/s at the racket —
well past what the bin shot needs, and into ball-machine territory later.

**Industrial 6-axis — used ABB IRB 1100/1200, FANUC M-10iD, KUKA AGILUS.
Overkill here.** Correct only if the roadmap goes to a machine-fed groundstroke.
If so, the external-guidance interface is the gating question (FANUC Stream
Motion, ABB EGM, KUKA RSI — ~4–12 ms command cycles), along with a compliant or
sacrificial racket coupler to keep repeated shock out of the reducers.

**Check joint velocity before buying any arm marketed on payload.** Payload and
reach say nothing about whether it can swing. The number that matters is degrees
per second at the shoulder and wrist, converted to m/s at your intercept radius.

## 6. Recommended configuration

Cheapest build that plausibly completes the task:

- Cobot or QDD arm, base bolted to a rigid table, intercept point at ~1.0 m.
- Racket on a **compliant, sacrificial coupler** — cheap to replace, and it
  attenuates the shock transient into the wrist.
- Two global-shutter cameras at 90–120 fps, wide baseline, both seeing the full
  toss corridor; calibrate extrinsics against the arm base, not against each
  other.
- Trajectory fit: ballistic + quadratic drag, refit every frame until commit.
- Aim policy: solve for exit speed from the bin distance, invert §1's restitution
  equation for *V*, apply the §2(a) per-throw correction from the measured *u*,
  bias long per §2(b).
- Contact as close to face-normal as reachable, to suppress spin.

## 7. Bench probes to run before scoring anything (B0–B5)

Pre-registered, in dependency order. Each gates the next; **B0 gates every
number in §2.**

| Probe | Measures | Why it gates |
|---|---|---|
| **B0** | Apparent restitution *e*: drop test, then static-racket returns of a tossed ball at several speeds and off-center distances | Every aiming equation in §1–§2 is parameterized on *e*. A 0.35 vs 0.60 error moves required racket speed by 50%. |
| **B1** | End-to-end perception latency **and its jitter** — strobe or LED against the commanded motion | §3: the mean is calibratable, the variance is not. |
| **B2** | Achievable stringbed speed and repeatability at the *actual* intercept pose, on the *actual* arm | Datasheet tool speed is not pose-independent; the Jacobian is worst near singularities. |
| **B3** | Human toss variability: *u*, arrival point, arrival angle over ≥100 tosses | Sets the compensation range §2(a) must cover, and tells you whether a human toss is even repeatable enough. |
| **B4** | **Open-loop bin shot**: fixed pre-programmed swing, ball hand-placed at the intercept point, no perception in the loop | Isolates aiming from sensing. If B4 cannot hit the bin, no amount of tracking will help. |
| **B5** | Closed-loop: live toss, full pipeline | The task. |

**Suggested pre-registered bar for B5**, in the style of this repo's run
pre-registrations: **≥ 30% of 50 tosses land in a 0.4 m bin at 5 m, with a
contact rate ≥ 80%**, reported as two separate numbers. Reporting only "it hit
the ball" would repeat the wall-ball chapter's mistake of scoring against a bar
nobody calibrated.

## 8. Safety

A 3 m/s racket is far less dangerous than a 20 m/s one, but this is still a
whipping rigid object at head height, and §5's threshold change is the specific
risk: **an arm with its collision detection turned down is no longer a
collaborative robot.** Netting between the arm and the operator, hardware e-stop
in reach of the person tossing, the toss made from behind a marked line outside
the arm's reach envelope, and nobody else in the cell.

## 9. Provenance

Repo values are read from the tree at `main`@`aec8cd1`:
`assets/paddle_court.xml` (ball geom line 130; paddle bodies and position
servos lines 105–144), `assets/tennis_racket.xml` (inertial and geom block),
`assets/humanoid_tennis.xml` (ball geom line 113),
`assets/robots/unitree_g1_tennis.xml` (arm joint classes, lines 65–92), and
`envs/_tennis_physics.py` (drag constants and `ball_drag_force`).

The drag and Magnus claims were checked twice, and the first check was wrong in
a way worth recording. Grepping `<option`, `density`, and `viscosity` across
`assets/` finds no fluid density in any scene, which reads as "the project has
no aerodynamics" — but drag is applied **in Python**, via `xfrc_applied` in
`humanoid_tennis.py:1069`, deliberately (the docstring explains that MuJoCo's
ellipsoid-fluid approximation is not calibrated for a felt ball). An
XML-only search misses it. The corrected statement is the one in §4: drag is
modeled, faithfully, in exactly one env. Magnus really is absent everywhere —
`ball_drag_force` is the only force written to `xfrc_applied`, and it has no
lift term.

Everything else — toss model, exit-speed table, restitution algebra,
sensitivity partials, impulse and latency budgets — is analytic, derived here,
and **unmeasured**. Drag coefficient (Cd ≈ 0.55) and restitution range
(*e* ≈ 0.35–0.60) are textbook values for a tennis ball, not measurements of
your ball on your racket; B0 replaces them. Hardware figures in §5 are
order-of-magnitude and must be checked against current datasheets.

When B0–B5 produce numbers, they belong in a dated review snapshot, with the
durable conclusions lifted into [`DECISIONS.md`](DECISIONS.md).
