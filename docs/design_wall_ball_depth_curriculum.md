# Design: WallBallDepthCurriculum — rally from the workspace baseline

Status: implemented and pre-flight validated for 0.19.0, 2026-07-23.
The first depth curriculum shipped in 0.15.0 and its gate/artifact
refinements shipped through 0.18.0. The latest run showed that the policy
can rally at the final stage, but also exposed a positional shortcut.
This revision removes that shortcut before adding another reward or rule.

## Evidence and decision

The latest run, `20260722_124613`, reached stage 4 at about 4.175M steps
and produced a best matched-stage evaluation of about 3.33 completed
returns. That is useful evidence of rally skill. It is not yet evidence
of baseline play:

- Stage 4 started the paddle around x = −3.9, but legal contacts averaged
  around x = −1.85.
- The old final fence was (−4.7, −1.2), so the policy could sprint forward
  after reset and play near the fence front.
- More importantly, every old stage shared the interval (−2.7, −1.2).
  The curriculum changed the back of the range while retaining a
  front-court refuge that was legal for the entire run.

The next experiment therefore changes the physical opportunity set, not
the objective. Each stage slides the whole movement window backward. The
windows still overlap enough to transfer adjacent-stage skill, but no x
position is legal at every stage. At the final stage, a legal hit is
mechanically a workspace-baseline hit.

This is the smallest intervention that directly addresses the observed
shortcut. A depth reward could still be optimized at the old front edge,
and a must-bounce rule would change the task before establishing whether
geometry alone already produces the desired style.

## Sliding-window ladder

The replacement ladder keeps the existing starts and calibrated flat-serve
schedule while moving each fence front backward:

| Stage | Paddle fence (x) | Start | Serve speed | Adjacent transfer |
|---|---:|---:|---:|---|
| 0 | (−2.7, 0.3) | −1.6 | 5.2 | Existing shallow entry task |
| 1 | (−3.2, −0.8) | −2.1 | 5.5 | Overlap with s0: (−2.7, −0.8) |
| 2 | (−3.7, −1.6) | −2.7 | 6.0 | Overlap with s1: (−3.2, −1.6) |
| 3 | (−4.2, −2.4) | −3.3 | 6.5 | Overlap with s2: (−3.7, −2.4) |
| 4 | (−4.7, −3.0) | −3.9 | 7.0 | Overlap with s3: (−4.2, −3.0) |

The fixed control geometry remains:

- `paddle_x_target_range = (-4.7, 0.3)` and
  `paddle_home_x = -1.7`, so action semantics do not drift.
- `serve_start_x = 1.0`, `serve_lob = 0.0`, and
  `paddle_joint_damping = 8.0`.
- `paddle_start_x` places the body inside the current fence; the fixed
  control home is part of the action mapping, not a stage ready position.

The final fence is the baseline of the current paddle workspace, not the
true ITF baseline at x = −7.985. Extending play beyond x = −4.7 still
requires an XML workspace and serve-energy change and remains a later
campaign phase.

## What deliberately does not change

The pilot isolates geometry:

- Rally style remains `open`. Pre-bounce hits are legal and paid; the
  telemetry measures them rather than silently changing their reward.
- Reward coefficients and reward components remain unchanged.
- The promotion metric remains `bounce_count_ep_mean` with threshold
  3.0, the current three-evaluation `window_mean`, and matched-stage
  evaluation.
- The existing promotion warm-up remains: clear replay, collect 50k
  frontier-stage steps, then resume updates. Network weights still
  transfer.
- Best-model selection, per-stage archives, final-stage scoring, and
  long-horizon evaluation remain unchanged.

This is important experimentally: if the policy moves backward and keeps
rallying, the result belongs to the sliding constraint rather than a
simultaneous reward or gate change.

## Telemetry contract

The environment reports cumulative episode counters:

- `pre_bounce_legal_paddle_hit_count`: legal gate-opening hits before a
  floor bounce in that rally cycle.
- `post_bounce_legal_paddle_hit_count`: legal gate-opening hits after a
  floor bounce in that rally cycle.
- `opening_volley_count`: opening-cycle legal hits made before the serve
  bounces.
- `post_bounce_completed_return_count`: completed returns whose legal
  paddle hit occurred after the floor bounce.

These complement the existing legal-hit count and contact-x sum/mean. The
following identities are part of the telemetry contract and are checked by
the calibration sweep:

```text
pre_bounce_legal_paddle_hit_count
  + post_bounce_legal_paddle_hit_count
  == legal_paddle_hit_count

opening_volley_count <= pre_bounce_legal_paddle_hit_count

post_bounce_completed_return_count <= bounce_count
post_bounce_completed_return_count
  <= post_bounce_legal_paddle_hit_count
```

For review, a baseline return means:

```text
completed return
AND its legal paddle hit followed a floor bounce
AND contact_x was inside the stage's baseline target
```

The final condition is mechanical at stage 4 because the complete legal
range is (−4.7, −3.0). Contact x and the new counters should still be
reported separately: one answers where the policy played, the other
answers whether it waited for the bounce.

## Calibration outcome and sweep contract

`tools/depth_stage_sweep.py` was run at 200 episodes per policy and stage
(4,000 episodes total) on 2026-07-23. The old 2026-07-20 numbers did not
certify these narrower, fully sliding windows, so the replacement ladder
received a fresh pre-flight.

The script first performs static checks:

1. Every `paddle_start_x` lies inside its stage fence.
2. Every adjacent pair of fences has positive-width overlap.
3. The intersection of all stage fences is empty.

It then runs four scripted cells per stage:

- parked;
- placement-blind crude full swing, the historical learnability bar;
- stage-calibrated run-up or timed-charge oracle, the feasibility bar;
- an uncalibrated pre-bounce chase probe, which asks whether the opening
  serve can still be volleyed from that fence.

The existing dynamic criteria remain blocking:

1. parked reward < crude reward < oracle reward within every stage;
2. the oracle completes at least two returns on at least 90% of serves;
3. the crude controller completes a second exchange in a nonzero fraction
   of episodes;
4. no adjacent oracle bounce mean jumps above 1.5 times its predecessor.

Telemetry identities are also blocking. The sweep prints mean pre-bounce
hits, mean post-bounce hits, opening-volley episode rate, and mean
post-bounce completed returns for every cell. The chase probe's
opening-volley rate is diagnostic, not a pass/fail threshold: it has not
yet been calibrated well enough to turn a zero into proof of mechanical
impossibility.

If the oracle cannot clear the narrower stages, retune overlap or serve
landing geometry before training. Do not make an infeasible ladder
learnable by adding reward.

The fresh sweep passed every static and dynamic criterion:

| Stage | Crude ≥2 returns | Oracle ≥2 returns | Oracle mean returns | Opening-volley probe |
|---|---:|---:|---:|---:|
| 0 | 83% | 94% | 2.58 | 100% |
| 1 | 73% | 96% | 2.44 | 100% |
| 2 | 69% | 92% | 2.35 | 100% |
| 3 | 58% | 94% | 2.25 | 38% |
| 4 | 14% | 95% | 2.29 | 0% |

All telemetry identities passed. The stage-4 result is especially useful:
post-bounce rallying remains feasible, while the diagnostic controller
could no longer reach the opening flight. Because that probe is
deliberately non-blocking and not a learned policy, this is evidence that
the geometry strongly suppresses volleying—not proof that a trained policy
can never discover one.

## Six-million-step pilot

Run one full SAC pilot with a 6M-step budget. The latest run did not reach
stage 4 until roughly 4.175M, so the previous 3M default could end before
the experiment reaches its actual target. Early stopping can still finish
a settled final plateau.

The pilot review targets are:

- reach stage 4 and retain independent rally performance near the current
  three-return level;
- all stage-4 contacts fall inside (−4.7, −3.0), by construction;
- at least 70% of completed stage-4 returns are post-bounce returns;
- learned-policy opening volleys are near zero at stage 4;
- promotion shocks recover within roughly 250k steps rather than consuming
  most of a stage residency;
- final-stage and long-horizon results do not show a material rally
  regression relative to the latest run.

These behavioral targets are review criteria, not new promotion gates for
the first pilot. The current bounce-count gate stays the only training
gate so the experiment remains interpretable.

## Pre-registered escalation rules

Add complexity only for the failure that appears:

- **Opening volleys persist at stage 4:** add an explicit must-bounce rule.
  This is a task-rule problem; a small depth reward is not a reliable way
  to prohibit a legal volley.
- **The scripted oracle fails a stage:** widen or subdivide the transition,
  or adjust serve landing geometry. Do not start RL on a failed geometry.
- **The oracle passes but RL stalls:** reduce the shift between adjacent
  windows first. If interception remains the binding failure, reconsider
  the queued landing-point observation feature.
- **Contacts are post-bounce and deep but completed returns collapse:** add
  a small bonus only for a successfully completed post-bounce baseline
  return. Do not pay raw position or an uncompleted touch.
- **The policy hugs each stage's front boundary:** that is acceptable when
  the final boundary itself is the workspace baseline. If x = −3.0 is not
  deep enough for the next campaign goal, extend the physical workspace
  rather than reopening a common front-court refuge.
- **Replay clearing causes excessive relearning:** run a matched
  clear-versus-keep comparison before changing observations or rewards.

If the pilot passes, compare the old and sliding ladders over three seeds
at 6M steps, then confirm the winner on fresh seeds with a fixed
long-horizon audit.

## Historical calibration note

The original five-stage ladder was:

```text
(-2.7,  0.3)
(-3.2, -0.6)
(-3.7, -1.0)
(-4.2, -1.2)
(-4.7, -1.2)
```

Its 2026-07-20 sweep passed the scripted feasibility and learnability
checks: oracle ≥2-return rates were 93–97%, crude ≥2-return rates were
66–83%, and no adjacent oracle bounce mean jumped by 1.5 times. Those
results established that the serve schedule and open-style curriculum
could support deeper starts.

They did not test the positional objective. The shared interval
(−2.7, −1.2) let every stage collapse to the same front-court strategy,
which the latest contact-x telemetry finally exposed. The sliding ladder
keeps the useful calibrated serve schedule while removing that shared
shortcut.
