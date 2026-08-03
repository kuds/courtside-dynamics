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

**The sport is a free parameter, and it is the highest-leverage choice in the
build.** §1–§4 work the tennis case throughout, because that is the project's
destination. §5.5 evaluates tennis against pickleball and ping-pong *with their
matched balls* and recommends **starting with ping-pong** — a 2.7 g ball cuts
the impact moment 65× and the required paddle speed to a third, moving the
entire difficulty from mechanics into aerodynamics, where it is iterable.

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

The racket speed that produces that exit speed follows from a two-body impact
with the implement's **effective mass at the impact point**, `M_eff`. Writing
`r = m_ball/M_eff` for the mass ratio and *e* for the *local* coefficient of
restitution (ball against the striking surface, ≈ 0.75–0.78 — the repo's own
measured court COR is 0.763):

```
v_out = [ V·(1 + e) + u·(e − r) ] / (1 + r)
```

where *V* is stringbed speed along the outgoing axis and *u* is the incoming
ball speed component being reversed (≈ 5 m/s of the 7.1 m/s arrival). This
reduces to the familiar `V(1+e) + e·u` only in the heavy-implement limit
`r → 0`, which is **not** a safe assumption for light implements (§5.5).

For a tennis racket struck at the stringbed center,
`M_eff = 1/(1/m + d²/I_cm) = 1/(1/0.30 + 0.13²/0.012) ≈ 0.211 kg`, so
`r ≈ 0.27` and:

```
v_out = 1.40·V + 0.40·u
```

**The model self-validates:** setting V = 0 gives an apparent COR (ACOR) of
**0.40**, which is the textbook measured value for a tennis racket struck at the
sweet spot. Solving for the 5 m bin: `6.6 = 1.40V + 2.0` ⇒ **V = 3.3 m/s**.
Sweeping plausible restitution and mass-ratio values keeps V within
**2.5–3.6 m/s**.

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
| Racket speed | dv_out/dV ≈ **1.40** (§1) | **±0.10 m/s** |
| **Incoming ball speed** | dv_out/du ≈ **0.40** (§1) | **±0.35 m/s of toss variation** |
| Launch elevation | 45° is the range maximum; ±5° costs ~0.5–0.7 m | **±2°**, and errors are one-signed (always short) |
| Face azimuth | maps ~1:1–2:1 into outgoing direction | **±1°** |

Three things fall out of that table.

**(a) Per-throw compensation is mandatory, not optional.** A human underhand
toss varies by well over ±0.5 m/s throw to throw. At a 0.40 coefficient that is
±0.20 m/s of exit speed, i.e. ±0.28 m of range — comparable to the whole bin. The
controller must **measure *u* from the observed track and correct the commanded
racket speed**:

```
ΔV = −(0.40 / 1.40) · Δu  ≈  −0.29 · Δu
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

## 5. Candidate arms — surveyed 2026-08-03

### 5.1 Torque is not the constraint

State this first because it is the question everyone asks and the answer is
boring. A tennis racket is 0.30 kg with its CoM 0.36 m from the mount
(`tennis_racket.xml`):

| Demand on the wrist | Value |
|---|---|
| Static holding moment | 0.30 × 9.81 × 0.36 = **1.06 N·m** |
| Racket inertia about the mount | 0.30 × 0.36² + 0.0105 ≈ **0.049 kg·m²** |
| Torque to reach 5 rad/s in 0.2 s | 0.049 × 25 = **1.2 N·m** |

Every arm in §5.3 rates its wrist joints in the 5–33 N·m range — **5× to 30×
margin**. No arm on the market that can hold a 1.5 kg payload will struggle to
*swing* a racket. The two things that actually gate the build are **joint
angular velocity** (§5.2) and **surviving the impact** (§5.4).

### 5.2 Reading the speed spec correctly

The relevant question is not "max TCP speed" but **max joint velocity converted
to m/s at the stringbed**, which sits ~0.49 m beyond the flange
(`tennis_racket.xml`, stringbed center at z = 0.49). Two corrections matter:

- **Published "max TCP speed" understates the swing.** UR quotes 1 m/s for the
  UR5e, but that is a conservative *rated* figure: the joints run at 180 °/s
  (3.14 rad/s) and the documented maximum TCP speed is **4 m/s**. UR's own
  forum notes that a coordinated shoulder+elbow move exceeds 1 m/s readily. The
  racket's 0.49 m lever adds to that again.
- **Distal speed beats proximal speed for a swing.** An arm whose *wrist* axes
  are fast whips better than one with a fast base, because the wrist sits at the
  end of the longest lever. This is what makes the FR3 and PiPER interesting.

A useful bound: a joint at ω rad/s puts the stringbed at `ω × (r_joint→flange +
0.49)` m/s, and coordinated joints add.

### 5.3 The shortlist

Prices are single-unit street/list as advertised on 2026-08-03 and move
constantly; treat them as ±20% and re-check before buying. "Sim model" means a
maintained MJCF in **MuJoCo Menagerie**, which matters here because this repo is
MuJoCo-native and already vendors a third-party model (the G1) under
`assets/third_party/` with provenance and checksums.

| Arm | Price (USD) | Reach | Payload | Max joint speed | Est. stringbed speed | Menagerie model |
|---|---|---|---|---|---|---|
| **AgileX PiPER** | **$1,999** | 626 mm | 1.5 kg | **225 °/s (J4–J6)** | ~3–4 m/s | ✅ `agilex_piper` (MIT) |
| ARX L5 | ~¥29,800 (~$4.1k) | ~620 mm | 2–3 kg | **not published** | unknown | ✅ `arx_l5` (BSD-3) |
| **UFACTORY xArm 6** | **$9,500** | 700 mm | 5 kg | 180 °/s | ~4 m/s | ⚠️ **xArm 7 only** |
| Unitree Z1 | $15,999 | 740 mm | 2–5 kg | 180 °/s, 33 N·m/joint | ~4 m/s | ✅ `unitree_z1` (BSD-3) |
| **Franka FR3** | ~$30k | 855 mm | 3 kg | 150 °/s A1–A4, **301 °/s A5–A7** | ~4–5 m/s | ✅ `franka_fr3` (Apache-2.0) |
| Universal Robots UR5e | ~$30–37k | 850 mm | 5 kg | 180 °/s (TCP max 4 m/s) | ~4 m/s | ✅ `universal_robots_ur5e` |

Stringbed-speed estimates are derived, not measured, and assume a coordinated
multi-joint swing with the arm near full extension. Probe B2 (§7) is what
replaces them.

Also in Menagerie and *not* recommended here: Kinova Gen3, KUKA LBR iiwa 14,
Rethink Sawyer, Flexiv Rizon 4/4S (all fine arms, none cheaper or faster than
the above for this task), and the ViperX 300 / WidowX 250 / SO-ARM100 /
Low-Cost Robot Arm class (Dynamixel-driven, too slow and too light to swing a
0.3 kg racket at 3 m/s).

**Verdicts.**

- **Recommended: UFACTORY xArm 6 at $9,500.** 5 kg payload, 700 mm reach,
  180 °/s. It wins on the axis that §5.4 shows actually decides this — **moment
  margin** — with roughly 3× PiPER's wrist-moment budget, and its 700 mm reach
  supplies enough speed that a *shorter, lighter* implement stays viable, which
  is the cheapest way to cut shock. One real annoyance: **Menagerie ships
  xArm 7, not xArm 6** — the sim model would be a 7-DoF proxy for a 6-DoF arm,
  so the kinematics do not match. Buy the xArm 7 instead if sim fidelity matters
  more than ~$2k.
- **Best research platform: Franka FR3 (~$30k).** Fast distal joints
  (301 °/s on A5–A7), 1 kHz torque control over FCI, a maintained Menagerie
  model, and — relevant here — **joint torque sensing in all seven axes**, so it
  is the one arm on this list explicitly engineered for contact-rich work. If
  the budget exists, this is the arm that will not be the reason the project
  fails.
- **AgileX PiPER at $1,999 — attractive on paper, and I no longer recommend it
  for this task.** Its J4–J6 genuinely run at 225 °/s, faster than any cobot
  here, and the MIT-licensed MJCF matches the hardware. But §5.4 shows it is
  caught in a bind: it is **simultaneously speed-marginal and shock-marginal,
  and the fix for each makes the other worse.** Reaching 3 m/s requires the full
  0.49 m racket lever, and that same lever is what multiplies the ball's impulse
  into the wrist. A 0.4 kg racket at a 0.36 m CoM offset already consumes
  ~1.4 kg of its 1.5 kg payload budget in moment terms *before the ball
  arrives*. Viable only as a consumable, with every mitigation in §5.4 applied.
- **Skip the Unitree Z1 at $15,999.** Same 180 °/s as a $9,500 xArm 6, less
  payload than a UR5e, for more money than either.
- **ARX L5 is the unresolved one.** Cheap (~$4.1k), force-controlled, 500 Hz
  control loop via the open-source `arx5-sdk`, and in Menagerie — but neither
  ARX nor the SDK publishes joint velocity limits, and the SDK explicitly says
  its default safety limits are placeholders. Ask ARX directly.

### 5.4 The wrist-moment budget — the constraint that actually decides

§5.1 showed that *swinging* a racket is torque-trivial. That is not the whole
story, because a racket is a long lever and the ball arrives at the far end of
it. Three separate loads, in increasing order of severity.

**(a) Static hold — small in force, large in moment.** Take a strung racket plus
a mounting adapter at 0.4 kg, CoM 0.36 m from the flange:

```
M_static = 0.4 × 9.81 × 0.36 ≈ 1.4 N·m
```

Payload ratings are quoted at a *nominal CoM offset*, typically 50–100 mm. At
360 mm, a 0.4 kg racket produces the same wrist moment as **1.4 kg hung at
100 mm** — so on a 1.5 kg-rated arm the racket consumes ~90% of the moment
budget while weighing barely a quarter of the rated mass. **This is the trap in
reading "1.5 kg payload, racket is only 0.3 kg, plenty of margin."** It is not
plenty; it is nearly all of it, before the ball is involved.

**(b) Swing torque — comfortable.** Racket inertia about the flange
`I ≈ 0.4 × 0.36² + 0.0105 ≈ 0.062 kg·m²`. Reaching 3 m/s at the stringbed
(ω = 3/0.49 = 6.1 rad/s) in 0.2 s needs α = 30 rad/s²:

```
τ_swing = 0.062 × 30 ≈ 1.9 N·m,  plus gravity → ~3.3 N·m peak
```

**(c) Ball impact — the severe one.**

| Quantity | Value |
|---|---|
| Impulse `m·Δv = 0.057 × (5 + 6.6)` | **0.66 N·s** |
| Contact duration (tennis ball on strings) | ~4–5 ms |
| Mean force | **~132 N** |
| Peak force (half-sine, ~1.6× mean) | **~210 N** |
| Mean moment at 0.49 m (butt-mounted) | **~65 N·m** |
| Peak moment | **~100 N·m for ~5 ms** |
| Angular impulse into the wrist axis | **0.32 N·m·s** |

Two things stop that 100 N·m from being a 100 N·m gearbox load, and neither
makes it safe:

- **The joint cannot respond.** A 5 ms event is far inside any servo's
  bandwidth (~10–20 ms), so this is a structural/inertial event, not a control
  one. No amount of good control helps.
- **Link inertia and structural compliance filter it.** A light arm's structural
  modes sit around 20–50 Hz (20–50 ms period), so a 5 ms impulse is substantially
  absorbed by link flex and inertia rather than delivered to the gear teeth.

What reaches the reducer is therefore a *fraction* of 0.32 N·m·s, spread over
tens of milliseconds — plausibly tens of N·m instantaneous. Against a wrist
rated at single-digit N·m continuous and perhaps 2–3× that momentary, this lands
in the **"survivable occasionally, damaging repeatedly"** band. And repetition is
the whole point: a bring-up is hundreds to thousands of impacts, which is a
fatigue and backlash-growth regime, not a one-off strength question.

**The bind that disqualifies a small arm.** Lever length helps speed and hurts
shock, and both effects are linear in the same number:

| Implement | Lever to strike point | Speed from a 3.93 rad/s wrist | Impact moment |
|---|---|---|---|
| Tennis racket, butt-mounted | 0.49 m | 1.9 m/s | 65 N·m |
| Tennis racket, gripped mid-handle | ~0.34 m | 1.3 m/s | 45 N·m |
| Pickleball-style paddle | ~0.28 m | 1.1 m/s | 37 N·m |

An arm with **speed margin** can spend it on a shorter implement and cut the
moment ~40%. An arm that needs the full 0.49 m lever just to reach 3 m/s cannot.
That is precisely PiPER's position, and it is why moment margin — not price,
not joint speed — is the right primary selection criterion **for the tennis
case**. §5.5 shows that switching sports dissolves this section entirely: with a
matched ping-pong ball the impact moment falls from 65 N·m to 1 N·m.

**Mitigations, in order of leverage.**

1. **A compliant coupler between flange and racket. Mandatory, not optional.**
   Elastomer bushings or a torsion element sized for a ~5 Hz natural frequency
   with the racket inertia (`k = I·ω_n² ≈ 0.062 × 31² ≈ 60 N·m/rad`) stretches
   the 5 ms impulse into a ~100 ms push, cutting peak joint torque roughly
   5–10×. The obvious objection — that compliance ruins the ±1° aiming
   requirement of §2 — does not bite: during the 5 ms of contact the racket
   deflects only `½ · (65/0.062) · 0.005² ≈ 0.013 rad ≈ 0.75°`. The face angle
   *at* contact is what aims the ball, and it barely moves. The coupler mostly
   costs a little effective restitution, which B0 calibrates out.
2. **Rigid-mount the racket. Do not hold it in a gripper.** A parallel-jaw
   gripper on a round handle develops maybe 25–50 N of friction hold against a
   ~210 N impact applied 0.3 m away. It will slip and rotate, ruining aim
   repeatability, and grippers are not rated for shock. Machine an adapter that
   bolts the butt to the tool flange.
3. **Align the strike point near the racket's center of percussion.** For a
   racket pivoted at the flange, `q = I/(m·d) = 0.062/(0.4 × 0.36) ≈ 0.43 m` —
   comfortably inside the stringbed (0.32–0.66 m). Striking there nulls the
   *transverse reaction force* at the wrist bearing. Be precise about what this
   buys: it removes bearing force and vibration, **not** the joint torque
   impulse, which is `J × q` regardless. Worth having, not a substitute for (1).
4. **Shorten the shot.** Exit speed scales the impulse, and the bin distance
   sets the exit speed: moving the bin from 5 m to 3 m drops required exit speed
   from 6.6 to 5.0 m/s and the impulse ~20%. Free margin during bring-up.

**Two further integration risks, unchanged:**

- **Collision detection will protective-stop on every hit.** The transient
  comfortably exceeds any default threshold. Raise it, and note that **doing so
  means the arm is no longer collaborative in the safety-rated sense** (§8).
- **Speed limits are enforced with a Cat 0 stop.** On UR hardware, exceeding the
  *configured* TCP speed limit cuts drive power immediately. The 4 m/s
  kinematic capability is only available if the safety configuration allows it.
  Verify the configured limit, not the brochure.

### 5.5 Sport choice: tennis, pickleball, or ping-pong — with matched balls

The implement is not chosen alone. Picking a paddle means picking **its ball**,
and the ball changes every number in this document. This section supersedes an
earlier version that compared paddles against a *tennis* ball; that comparison
answered a question nobody was asking, and its conclusion about ping-pong was
backwards.

Two independent physical scalings drive everything:

- **Impact scales with ball mass.** A 2.7 g ball delivers 1/21 the impulse of a
  57 g one at the same speed. The entire §5.4 shock problem is a *tennis ball*
  problem.
- **Aerodynamics scale with area/mass.** `k = ½ρC_dA/m` is 6× larger for a
  ping-pong ball than a tennis ball, and Magnus scales the same way. What the
  light ball saves mechanically, it spends aerodynamically.

Each sport is evaluated at its own natural scale (bin and toss distance
matched to the sport), 45° launch, 1.0 m contact height, 0.5 m bin rim,
integrating quadratic drag numerically:

| | **Tennis** | **Pickleball** | **Ping-pong** |
|---|---|---|---|
| Ball mass / diameter | 57 g / 67 mm | 24 g / 74 mm | 2.7 g / 40 mm |
| Bin & toss distance | 5.0 m | 4.0 m | 2.5 m |
| Drag at 6.6 m/s (as % of *g*) | 9% | 22% | **57%** |
| Magnus force / weight | 4% | 12% | **32%** |
| Required exit speed | 6.96 m/s | 6.40 m/s | 5.17 m/s |
| Local COR *e* | 0.78 | 0.40 | 0.90 |
| Mass ratio `r` | 0.27 | 0.13 | **0.023** |
| **ACOR** | 0.40 | 0.24 | **0.86** |
| **Paddle speed needed** | 3.61 m/s | **4.35 m/s** | **1.35 m/s** |
| Toss speed / arrival | 7.30 / 4.75 m/s | 6.77 / 4.10 m/s | 5.61 / 3.11 m/s |
| Flight time | 0.95 s | 0.86 s | 0.70 s |
| **Impact impulse** | 0.667 N·s | 0.252 N·s | **0.022 N·s** |
| **Impact moment** | **65 N·m** | 14 N·m | **1 N·m** |
| Speed tolerance (0.4 m bin) | ±0.16 m/s (2.3%) | ±0.20 m/s (3.2%) | **±0.30 m/s (5.8%)** |
| Catch window (face + ball) | 0.367 m | 0.274 m | 0.190 m |

**Ping-pong makes the mechanical problem disappear.** A 2.7 g ball against a
120 g effective paddle mass gives `r = 0.023` — essentially the infinite-mass
limit — so the ACOR is **0.86**, and the paddle behaves like a wall. Consequences:

- **1 N·m of impact moment**, 65× less than tennis. Every arm in §5.3 clears
  this by two orders of magnitude, PiPER included and over-specified. §5.4, the
  section that drove the whole arm selection, simply stops applying.
- **1.35 m/s of paddle speed**, a third of what tennis needs. Speed stops being
  a selection criterion too.
- **Speed control is 2.5× more forgiving** (±5.8% vs ±2.3%), because heavy drag
  flattens the range-versus-speed curve — it acts as a governor.
- **The whole rig is desk-scale** (2.5 m), and a 2.7 g projectile is harmless,
  which changes the §8 safety picture from "netting and e-stop" to "reasonable
  care."

**What ping-pong costs, and it is not trivial.** Magnus force reaches **32% of
the ball's weight** — 8× tennis. Spin, not speed, becomes the dominant aiming
error, and it arrives from two uncontrolled sources: the toss, and any oblique
contact. A ping-pong ball with modest sidespin curves by tens of centimetres
over 2.5 m. Additionally the catch window is the smallest (0.190 m vs 0.367 m),
the 40 mm ball is the hardest to track, and at 2.7 g it is genuinely blown
off course by room air currents. Mitigations: use an **anti-spin or bare-wood
blade rather than tacky spin rubber** (this is the high-leverage one — it
attacks spin *generation* directly), strike near-normal, and turn the HVAC off.

**What the ping-pong speed envelope actually looks like.** Elite play smashes at
roughly **25–30 m/s**, with measured records around 32–34 m/s (Guinness: 116 km/h).
Against that, the PiPER's ~2.5–3 m/s of paddle speed produces, via
`v_out = 1.857·V + 0.857·u`:

| Paddle speed *V* | vs stationary ball | vs 6 m/s incoming | vs 10 m/s incoming | vs 15 m/s incoming |
|---|---|---|---|---|
| 1.35 m/s (bin shot) | 2.5 m/s | 7.6 m/s | 11.1 m/s | 15.4 m/s |
| 3.0 m/s (PiPER max) | **5.6 m/s** | 10.7 m/s | 14.1 m/s | **18.4 m/s** |

**The asymmetry is the useful part.** At ACOR 0.858, incoming speed returns at
**0.857×** — so *blocking and countering a fast ball is nearly free*, while
*generating* speed from a slow ball is entirely paddle-limited. A PiPER can
serve at ~5.6 m/s (a realistic table-tennis serve pace) and can counter-hit a
15 m/s ball back at 18 m/s, but it will never smash. This is the same reason
real table-tennis robots block competently long before they loop.

**Drag caps the useful speed anyway.** At `k = 0.128 m⁻¹`, a ping-pong ball
loses **30% of its speed over one table length (2.74 m) regardless of how hard
it is hit** — the decay is exponential in distance, not speed. Carry distance
from a 0.9 m launch, at the best angle:

| Launch speed | 5 | 10 | 15 | 20 | 30 m/s |
|---|---|---|---|---|---|
| Carry | 2.7 m | 6.0 m | 8.8 m | 11.0 m | 14.2 m |
| Drag as multiple of *g* | 0.3× | 1.3× | 2.9× | 5.2× | **11.8×** |

Doubling speed from 15 to 30 m/s buys only 60% more range, because drag rises
as *v²*. Above ~15 m/s you are mostly heating the air. For a 2.5 m target task,
anything past ~6 m/s is wasted.

**Pickleball is the surprise loser.** It needs **more paddle speed than tennis**
(4.35 vs 3.61 m/s), because a pickleball is a dead ball — local COR ≈ 0.40
against tennis's 0.78 — so the ACOR falls to 0.24 even though the ball is
lighter. It buys a real 4.6× reduction in impact moment (14 vs 65 N·m) and it
has the largest ball of the three (74 mm, easiest to track), but it is not the
free middle option it looks like. *Caveat:* this treats the paddle face as
rigid. Real pickleball paddles have measurable face deflection, which would
raise the ACOR toward 0.3–0.4 and pull the required paddle speed down to
~3.5–4.4 m/s. B0 settles it.

**Recommendation: build the ping-pong rig first, then escalate.**

The argument is about *which kind of risk you are taking*. Tennis puts the risk
in the mechanics — gearbox wear, which is irreversible, expensive, and silent
until it is not. Ping-pong moves the risk into aerodynamics and control, which
is software: free to retry, iterable, and instrumented. **For a bring-up, trade
irreversible risk for iterable risk every time.**

Concretely: a ping-pong rig runs the entire B0–B5 battery (§7) — restitution,
latency, achievable speed, toss variability, open-loop aiming, closed-loop —
on a desk, on the cheapest arm, with no shock risk and no safety cell. Every
component of the pipeline except the aerodynamic model transfers directly to
tennis. Then escalate to tennis with a working pipeline and, by then, actual
measurements telling you whether the arm needs replacing.

This also mirrors the repo's own methodology. The project is a progression —
BallBalance → BallBounce → WallBall → PaddleTennis → HumanoidTennis — and every
rung was justified by proving the mechanism cheaply before scaling it. A
ping-pong bench rig is that same argument in hardware. Note too that every
shipped env drives a **paddle** (a 0.4 × 0.5 m slab), not a racket: a physical
paddle rig is closer to what this repo already simulates than a tennis racket is.

*(Numbers in this section come from `impl2.py`-style numerical integration of
quadratic drag; the script is short enough to reproduce from the constants in
the table. Drag coefficients — 0.55 tennis, ~0.45 pickleball and ping-pong —
are literature values, and the pickleball figure is the least certain of the
three because of the perforations.)*

### 5.6 If the roadmap escalates

Beyond the bin shot — machine-fed groundstrokes at 15–20 m/s — the published
precedent is worth copying rather than re-deriving. DeepMind's 2024
human-competitive table-tennis system uses a **6-DoF ABB IRB 1100 mounted on two
Festo linear gantries**, with perception from a pair of Ximea MQ013CG-ON cameras
at **125 Hz**. Two lessons transfer directly: the speed comes from *adding a
linear axis*, not from buying a faster arm; and 125 Hz sufficed for a sport with
roughly half this task's flight time, which independently corroborates §3's
90–120 fps recommendation. A DIY quasi-direct-drive build (3× AK80-9 / RMD-X8
class, ~$1.5–3k) is the other escalation path — low gear ratios are inherently
impact-tolerant, and yaw + shoulder + wrist on a carbon tube reaches 5–8 m/s.

**Check joint velocity before buying any arm marketed on payload.** Payload and
reach say nothing about whether it can swing.

## 6. Recommended configuration

Cheapest build that plausibly completes the task:

**Stage 1 — ping-pong bench rig (§5.5).** Desk-scale, 2.5 m. The load case is
tiny: a ~0.25 kg blade-plus-adapter with CoM ~0.15 m out is **0.37 N·m** static,
the strike needs **1.35 m/s**, and the ball delivers **1 N·m** of impact moment.

**Recommended arm: AgileX PiPER, $1,999.** Margins against the above:

| Constraint | Required | PiPER | Margin |
|---|---|---|---|
| Paddle speed | 1.35 m/s | ~2.5–3 m/s | **~2×** |
| Static wrist moment | 0.37 N·m | ~1.5 N·m equiv. | **~4×** |
| Impact moment | 1 N·m | ≥3–5 N·m | **~3–5×** |
| Reach vs intercept volume | ~0.5 × 0.5 × 0.4 m | 626 mm | adequate |

Those are healthy margins, and the J1–J3 speed figure AgileX does not publish —
the unknown that disqualified PiPER for tennis in §5.3 — stops mattering when
the bar is 1.35 m/s. It also carries an exact, MIT-licensed Menagerie model.

**Buy the UFACTORY xArm 7 ($11,000) instead only if tennis is likely within the
year.** It is not more *margin* for ping-pong — it is escalation headroom, which
is a different purchase. Note the option cost is small: PiPER now plus an xArm 7
later is $12,999 against $11,000 today, so **preserving the choice costs about
$2,000**. The xArm 7 is the right escalation target rather than the xArm 6
despite lower payload (3.5 vs 5 kg, still ~10× what ping-pong needs): its
seventh joint gives a null space for holding paddle orientation across the
intercept volume without approaching singularities, and it is the variant
Menagerie actually ships.

**Not recommended: UFACTORY Lite 6 (~$3,000).** 440 mm reach is tight against
the intercept volume, and a 600 g payload rating leaves the 0.37 N·m paddle
moment with little or no margin depending on the rated CoM offset.

**Buy workspace margin for $200, not $9,000.** The failure mode most likely to
bite at Stage 1 is the intercept point landing outside the reachable volume —
and that is driven by *toss scatter*, not by arm reach. A commodity ping-pong
ball launcher has far lower variance than a human arm, shrinking the required
volume directly. Use one through B0–B4 to isolate the pipeline, then reintroduce
the human toss for B5, which is the actual task. This addresses the real risk
far more cheaply than a larger arm does.

Use an anti-spin or bare-wood blade, and turn the HVAC off (§5.5).

**Stage 2 — tennis, once the pipeline works.** Then and only then does the arm
selection below bind:

- Arm from §5.3 — **xArm 6/7** as the default, FR3 if the budget allows — base
  bolted to a rigid table, intercept point at ~1.0 m. Select on **wrist-moment
  margin** (§5.4a), not on payload rating or joint speed.
- **Racket bolted to the flange through a compliant coupler** (§5.4 mitigation
  1–2). Never gripped.
- **Vendor its Menagerie MJCF into `assets/third_party/`** following the exact
  pattern already used for the G1 (`PROVENANCE.md`, `SHA256SUMS`,
  `CHANGELOG.upstream.md`). All the candidate licenses are permissive
  (MIT / BSD-3 / Apache-2.0). This gives a sim of the *real* arm, which is the
  thing no shipped env currently has (§4).
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
| **B4-w** | **Shock wear**: log joint current/torque through 500 impacts, then re-measure backlash and repeatability against the B2 baseline | §5.4 is an estimate of a *fatigue* regime. This is the cheap experiment that turns "probably survivable" into a number, and it should run before committing to a long campaign. |
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

**Hardware survey (§5), conducted 2026-08-03.** Vendor sources: the
[MuJoCo Menagerie model list](https://github.com/google-deepmind/mujoco_menagerie);
[UFACTORY xArm 6](https://www.ufactory.us/product/ufactory-xarm-6) ($9,500,
180 °/s, 5 kg, 700 mm); [AgileX PiPER](https://global.agilex.ai/products/piper)
($1,999, 1.5 kg, 626 mm) with the 225 °/s J4–J6 figure from the AgileX manual
table surfaced via the `piper_sdk` / `piper_ros` documentation; the
[Franka Research 3 datasheet](https://franka.de/hubfs/Digital_Datasheet%20Franka%20Research%203_R02212_2.1_EN.pdf)
(150 °/s A1–A4, 301 °/s A5–A7, A6 239 °/s under FCI);
[UR5e technical specifications](https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf)
plus the [UR forum thread on rated vs achievable TCP speed](https://forum.universal-robots.com/t/understanding-tcp-max-speed-from-datasheet/39665);
[Unitree Z1](https://shop.unitree.com/products/unitree-z1) ($15,999, 180 °/s,
33 N·m); ARX L5 pricing from Chinese trade press (¥29,800 / ¥49,800 Pro) with
control-rate and safety-limit caveats from
[`real-stanford/arx5-sdk`](https://github.com/real-stanford/arx5-sdk). The
escalation precedent is
[*Achieving Human Level Competitive Robot Table Tennis*](https://arxiv.org/html/2408.03906v2)
(ABB IRB 1100 on two Festo gantries, Ximea cameras at 125 Hz). **Prices are
advertised single-unit figures on one day and are the least durable numbers in
this document.**

Everything else — toss model, exit-speed table, restitution algebra,
sensitivity partials, impulse and latency budgets, and the stringbed-speed
estimates in §5.3 — is analytic, derived here, and **unmeasured**. Drag coefficient (Cd ≈ 0.55) and restitution range
(*e* ≈ 0.35–0.60) are textbook values for a tennis ball, not measurements of
your ball on your racket; B0 replaces them. Hardware figures in §5 are
order-of-magnitude and must be checked against current datasheets.

When B0–B5 produce numbers, they belong in a dated review snapshot, with the
durable conclusions lifted into [`DECISIONS.md`](DECISIONS.md).
