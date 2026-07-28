# Design: actuated paddle pitch and yaw

**Status: proposed, not implemented. Explicitly gated — see
[Prerequisites](#prerequisites).** Written 2026-07-27 against `0.22.0`.
The XML in this note was compiled and exercised in MuJoCo 3.10.0; the
numbers below are measured, not sketched.

## Why this is worth designing now and building later

The paddle face is currently pinned at a 10° upward pitch. That is not
an oversight — it is a hand-tuned prior doing real work: it guarantees
that *any* contact sends the ball up and forward. Run `20260727_004014`'s
post-mortem showed the flip side, that a slow contact is not merely soft
but *upward* (measured `qvel [1.51, -0.73, 3.52]` at 1.5 m/s paddle speed
vs `[15.79, -0.72, 4.61]` at 10 m/s), so the weak second shot pops up and
lands short. Face angle is a genuine lever on exactly the failure the
campaign is stuck on.

It is also the wrong lever to pull next. Freeing pitch enlarges the
action space of a task that has never completed three returns from the
baseline, and most of the new volume drives the ball into the floor or
over the wall. The campaign has already exhausted budget, incentives and
network capacity as levers; adding DOFs before the current geometry works
spends exploration budget in the opposite direction. This note exists so
that the work is *ready* when the prerequisite is met, not so it starts
early.

## Current model

```xml
<body name="paddle_base" pos="-1.7 0 1.2">
    <joint name="paddle_slide_x" type="slide" axis="1 0 0" damping="5" />
    <joint name="paddle_slide_y" type="slide" axis="0 1 0" damping="5" />
    <joint name="paddle_slide_z" type="slide" axis="0 0 1" damping="5" />
    <body name="paddle_head" pos="0 0 0" euler="0 -10 0">
        <site name="paddle_sensor" type="box" size="0.02 0.2 0.25" />
        <geom name="paddle_face" type="box" size="0.02 0.2 0.25" mass="0.35"
              solref="0.01 0.5" />
    </body>
</body>
```

The pitch is a **static body frame**, not a joint —
`model.body('paddle_head').quat` is `[0.996195, 0, -0.087156, 0]`, a −10°
rotation about y. There is no DOF to actuate. The model has four joints
total: three paddle slides plus the ball's free joint. Actuators are
three `position` servos (`kp=300 kv=18`, force-limited ±100 N) on the
slides. `<compiler angle="degree">`, so XML ranges are degrees while
`ctrlrange` is radians.

## Proposed change

```xml
 <body name="paddle_base" pos="-1.7 0 1.2">
     <joint name="paddle_slide_x" .../>
     <joint name="paddle_slide_y" .../>
     <joint name="paddle_slide_z" .../>
+    <!-- paddle_head stops being rigidly attached once it has its own
+         joints, so paddle_base needs inertia of its own (see below). -->
+    <inertial pos="0 0 0" mass="0.05" diaginertia="0.002 0.002 0.002" />
     <body name="paddle_head" pos="0 0 0" euler="0 -10 0">
+        <joint name="paddle_hinge_pitch" type="hinge" axis="0 1 0"
+               range="-10 10" damping="1.5" armature="0.01" />
+        <joint name="paddle_hinge_yaw" type="hinge" axis="0 0 1"
+               range="-10 10" damping="1.5" armature="0.01" />
         ...
```

```xml
+<position name="paddle_target_pitch" joint="paddle_hinge_pitch"
+          kp="8" kv="0.8" ctrlrange="-0.175 0.175" ctrllimited="true"
+          forcerange="-12 12" forcelimited="true" />
+<position name="paddle_target_yaw" joint="paddle_hinge_yaw"
+          kp="8" kv="0.8" ctrlrange="-0.175 0.175" ctrllimited="true"
+          forcerange="-12 12" forcelimited="true" />
```

**Keep `euler="0 -10 0"` on the body and hang the hinges off it.** At
`qpos = 0` the body sits at its XML frame, so a zero action reproduces
today's pose exactly; the joint only adds deviation. MuJoCo's `ref`
attribute can express the same thing but inverts confusingly (rotation is
`qpos − ref`), and the body's local z is tilted 10° from world z, which
matters for the yaw axis. Verified: zero-action `paddle_head` `xquat` and
`xpos` are bit-identical to the current model.

### The gotcha that is not obvious

Adding joints to `paddle_head` makes `paddle_base` a moving body with no
mass of its own — the 0.35 kg lives on the child's geom — and MuJoCo
refuses to compile:

```
Error: mass and inertia of moving bodies must be larger than mjMINVAL
Element name 'paddle_base', id 2
```

The change is therefore **not purely additive**. Giving `paddle_base` an
explicit `<inertial>` fixes it. Put the new mass on the *base*, not by
splitting the face's: `paddle_face` is the only paddle collision geom, so
keeping it at 0.35 kg leaves contact response untouched — verified, the
`paddle_head` body mass reads 0.35 before and after. The added base mass
only changes how the slide servos accelerate the assembly, which is a
tuning question, not a physics-of-contact question.

### Measured behaviour of the compiled variant

| | before | after |
|---|---:|---:|
| `nu` (actuators) | 3 | 5 |
| `nq` | 10 | 12 |
| `njnt` | 4 | 6 |
| `paddle_head` mass | 0.35 | 0.35 |
| zero-action pose | — | bit-identical |

With `kp=8`, a pitch command settles within ~0.25° of target (ctrl
−0.30 rad → −16.95°; ctrl +0.20 rad → +11.30°), the shortfall being
gravity torque on the offset face. Those gains are a starting point, not
a calibration.

## What it ripples into

1. **Action dim 3 → 5**, so SAC's `target_entropy` moves **−3.0 → −5.0**.
   Given that `gamma` alone (0.99 → 0.995) was worth three curriculum
   promotions in this task, the entropy target is not a free parameter and
   should be re-examined rather than inherited.
2. **Observation 23 → 27.** `_get_obs` builds paddle state from an
   explicit `self._joints_obs("paddle_slide_x", ...)` call, so the hinges
   extend it naturally (qpos + qvel per joint). Angles are continuous, so
   no `normalize_obs_excluded_indices` entry is needed. *The policy cannot
   control what it cannot see — this is not optional.*
3. **Every saved policy becomes incompatible.** A different action
   dimension means no warm start from anything in the archive, including
   the depth ladder's `stage_bests`.
4. **Rotational gains need their own calibration.** The linear swing cap
   was hand-tuned (`paddle_joint_damping = 8.0`, chosen so full-power
   returns rebound recoverably at ~12.5 m/s). A face that can snap 40°
   inside one 10 ms control step produces contact velocities nothing in
   the reward is calibrated against. Note also that the shared `<default>`
   block's `armature="0.1"` is sized for slides; for a hinge, armature has
   units of inertia and 0.1 is far too stiff — the sketch above overrides
   it to 0.01.
5. **`paddle_joint_damping` semantics.** The kwarg currently means "slide
   damping". Decide explicitly whether it also covers the hinges or
   whether a second kwarg is introduced; silently applying a
   swing-calibrated linear damping to a rotational DOF would be a quiet
   miscalibration of exactly the kind this repo keeps finding.

## Pitch before yaw

They are not symmetric, and the reward is why.

**Pitch has an immediate, dense payoff.** It directly controls launch
angle — the variable behind the measured pop-up failure — and 0.22.0's
`return_shaping_scale` already pays for driving the ball wallward. There
is a dense gradient waiting for pitch to exploit.

**Yaw has almost no gradient today.** `rew_wall` pays a flat +1 for any
legal wall contact; nothing in the reward cares where on the wall the
ball lands. Yaw matters only indirectly — angling the face changes where
the ball returns, which changes the next exchange's difficulty — and that
credit path runs through a full exchange, discounted `0.995^130 ≈ 0.52`.
It would be mostly exploration cost against a weak, delayed payoff.

*Recommendation: ship pitch alone first. Add yaw only alongside a
placement objective, or defer it to the humanoid tennis environment where
placement genuinely scores.*

## Calibration gate before any training

Lesson 9 exists for this: calibrate learnability with scripted policies
before spending GPU-hours. Concretely, before the first training run:

1. Extend `tools/depth_stage_sweep.py`'s oracle with a pitch argument.
2. On the same seeds, score the fixed-pitch oracle against a
   pitch-using oracle at the goal geometry.
3. **Acceptance criterion: the pitch oracle must beat the fixed-pitch
   oracle on completed returns by a paired margin that survives held-out
   seeds.** Seeds 0–199, 1000–1199, 10,000–10,049 and 20,000–20,199 are
   burned (see `design_wall_ball_checkpoint_selection_audit.md`); 2000–2199
   and 3000–3199 remain clean.

If a hand-written controller cannot exploit the extra DOF, a learner will
not either, and that costs an afternoon instead of a 21-hour run. Report
the comparison paired, per that same note — this campaign has produced two
non-replicating argmax results, and a scripted-ladder margin is exactly
the kind of number that invites a third.

## Rollout: open the range as a curriculum

The performance gate already applies arbitrary env attributes per stage —
that is how the depth fence recedes. Reuse it rather than starting from
full freedom:

| stage | pitch range | note |
|---|---|---|
| 0 | ±0° | frozen: bit-identical to today's behaviour |
| 1 | ±5° | refinement of a working policy |
| 2 | ±10° | |
| 3 | ±15° | widen only on demonstrated competence |

This turns the extra DOF into a refinement rather than a new search
problem, and gives a behavioural warm start without needing weight
compatibility. **Keep the range centred on −10°** — the current value is
calibrated; do not make the policy rediscover it.

Requires the same treatment `paddle_home_x` needed in 0.22.0: a range
that a stage sets must actually take effect. If the range is enforced
anywhere derived (an actuator `ctrlrange`, a cached array), it needs a
property setter that recomputes it, or the curriculum is a silent no-op
again.

## Open questions

* Does the added `paddle_base` mass (0.05 kg here) disturb the calibrated
  slide response enough to need `kp`/`kv`/`damping` re-tuning? Cheap to
  check against the existing scripted ladder.
* Should pitch be a *position* target (as sketched) or a *velocity*
  target? Position matches the existing interface and keeps action
  semantics stable across stages; velocity would make "swing through the
  ball" directly expressible. Position is the conservative default.
* Does the 10° prior remain correct once pitch is free, or does the
  optimum shift with the 0.22.0 runway change?

## Prerequisites

**Do not start this until the depth ladder is working.** Concretely: the
goal-geometry evaluator must break past its `20260727_004014` ceiling
(peak 1.400 completed returns, plateauing at ~1.14) and show a non-zero
≥3 rate on held-out seeds. Until then the binding constraint is geometry
and reward, which 0.22.0 addresses, and adding DOFs would confound the
measurement of whether those changes worked.
