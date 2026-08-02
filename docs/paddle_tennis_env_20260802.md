# PaddleTennis env freeze — the phase-P1 task definition

Status: task-definition record, 2026-08-02, for the registered
`CourtsideDynamics/PaddleTennis` environment
(`envs/paddle_tennis.py`) and the `PaddleTennis` recipe. Every number
here was measured by the pre-committed probe battery before the env
existed ([`design_paddle_tennis.md`](design_paddle_tennis.md) §6;
results in
[`paddle_tennis_probes_20260802.md`](paddle_tennis_probes_20260802.md)
and
[`paddle_tennis_probes_p3_p4_20260802.md`](paddle_tennis_probes_p3_p4_20260802.md)).
This document is the freeze: changes to anything in §1–§5 start a new
comparability era and must say so in the changelog.

## 1. Physical task (frozen by P0–P2)

Court half-length **6.5 m** per side, net **0.914 m** at x = 0,
singles width **±4.115 m**; the wall-ball ball and face-only paddles
verbatim (kp 300 / kv 18 / ±100 N targets, +10° face pitch, slide
damping 8 baked into `assets/paddle_court.xml`). Paddle homes at
x = ∓1.7; each paddle's workspace spans its own half (side-local
x ∈ [−6.4, −0.1], y ∈ ±3, z ∈ [−0.9, 2.0] about home). Sites cannot
collide, so all court markings are render-only.

## 2. Interface

- **Action** (3, in [−1, 1]): the wall-ball piecewise position-target
  mapping for the policy's own paddle. The policy always acts in
  side-local coordinates; physically it plays side A.
- **Observation** (48, all side-relative; P4 pinned the side mirror
  bit-for-bit):
  - `[0:24]` physical: ball position/velocity/spin, own paddle
    position/velocity, opponent paddle position/velocity, ball−own
    paddle;
  - `[24:36]` rally state: phase one-hot (4), own-relative
    serving/returner/ball-side flags, feed and pending-return
    crossing flags, bounce count, rally count, episode remaining
    fraction;
  - `[36:48]` contact memory: latch + release-progress for the six
    live sampler channels (own/opponent racket, court, net,
    own/opponent racket–net), own-relative. Exposed so the
    observation stays Markov (the humanoid convention).
  - Recipes normalize `[0:24]` only; the bounded tail stays raw.
- **Opponent**: side B is `opponent_controller(observation) -> action`
  reading the exact side-B mirror of the same observation; default is
  the frozen `lead_charge` scripted controller. Both sides act on the
  same pre-step state each control frame.

## 3. Episode and serve (frozen by P3)

One point per episode. The serve is ballistic from the serving side's
half: origin **3.25 m** behind the net, **9 m/s** at **21°**, with the
probe-standard jitter (position (0.25, 0.5, 0.05), speed ±1 m/s,
elevation ±3°, lateral ±4°) — measured 100% legal, mean landing
4.55 m, 100% returnable by the scripted opponent. The serving side
alternates on every reset (the humanoid alternation contract; P4/P3
proved alternation exactly fair, 36/36 mirrored cells identical).
`episode_len` 1500 control steps (15 s) — the wall-ball 750-cap
lesson says never truncate a healthy rally; the P3 tail terminates
naturally in ~270 steps.

## 4. Reward and termination

- **+`return_reward`** (1.0) per rules-confirmed legal return by
  **either side** — cooperative phase P1: the pair keeps the rally
  alive together, the shared-outcome design of
  `HumanoidTennisCoopEnv`'s rally target.
- **−`fault_penalty`** (1.0) on the terminal rules fault, whichever
  side faulted (same shared-outcome reasoning; there is no "my
  fault/your fault" asymmetry in a cooperative rally).
- **−`unsafe_physics_penalty`** (2.0) on unsafe/nonfinite physics;
  nonfinite actions/states end the episode without stepping MuJoCo,
  on the echoed last finite observation.
- No shaping terms of any kind at the freeze (campaign lessons 4/19:
  every shaping term this repo added before evidence demanded it was
  later falsified or exploited).
- Termination groups exposed as mutually exclusive `term_*` info
  flags: out_of_bounds, ball_net, second_bounce, failed_to_cross,
  illegal_hit (wrong hitter / double hit / premature /
  simultaneous / reverse crossing), net_touch (either racket),
  nonfinite, timeout.

## 5. Selection metric and reference band

`crossings` = cumulative return crossings (net crossings minus the
feed's own — **not** `shot_crossing_count`, an end-state latch; P3
measurement pitfall #2). The recipe's success (≥1) and headline
selection both follow it. Scripted-pair reference band at the frozen
serve: **3.15–3.42 mean crossings** (P3, seeds 1200–2639); a smoke
reproduction through the registered env at bring-up measured 3.05 on
calibration seeds 1000–1039. Certification floors derive from this
committed band minus sampling error — never from the P0–P2 scratchpad
band (2.02).

**Held-out certification: PASS (2026-08-02).**
`python tools/paddle_tennis_probes.py --certify` played the frozen
definition (registered env, both sides scripted) once per seed on the
reserved block **3100–3199**, against floors pre-registered from
calibration data only (mean crossings ≥ 2.6; ≥1-crossing rate ≥ 0.85;
zero unsafe/nonfinite terminations): mean crossings **3.22**
(std 0.81) — inside the committed band — ≥1-crossing **99%**, valid
returns 2.49 mean, serve sides exactly 50/50, zero unsafe, taxonomy
out_of_bounds 92 / ball_net 5 / failed_to_cross 3 (the P3 shape).
That burn retires block 3100–3199; **4100–4199 stays reserved** for
the first learned runs' held-out evaluation.

## 6. Presentation

`court_style` kwarg, render-only, the WallBall contract:
`"diagnostic"` (default; baselines, side lines, metre ticks, home
columns, serve-origin markers repositioned from the live serve
config), `"tennis"` (a to-size mini-court overlay: apron, surface,
baselines, singles lines, center marks), `"none"` (bare floor; TOML's
`"none"` sentinel maps to it). Visibility lists are derived from the
compiled model (`court_tennis_*` prefix vs other `court_*` sites), so
new markers cannot escape a hand-maintained list.

## 7. Decisions taken at the freeze (and their alternatives)

1. **Shared cooperative fault penalty** — either side's fault pays
   −1. Alternative (penalize only the policy's own faults) rejected:
   in phase P1 the objective is the *pair's* rally, and an opponent
   fault usually punishes a poor incoming ball. Revisit at scoring
   (phase P3), where asymmetry is the point.
2. **One point per episode** — matches the humanoid env; keeps
   `crossings` an un-averaged per-point tail and serve alternation a
   clean 50/50 across resets. Alternative (multi-point episodes with
   in-episode re-serve) deferred to the scoring phase.
3. **Policy physically on side A** — the mirror makes the sides
   provably identical (P4), so training only side A loses nothing;
   `serve_side_is_policy` records the alternation in every info dict.
4. **Serve jitter kept at the probe standard** — the measured band is
   the task; widening it is a curriculum decision for later, made
   against these committed numbers.
5. **No certification ladder in the recipe** — `ladder_certification`
   is WallBall-specific machinery; PaddleTennis held-out
   certification runs through the probes harness on the reserved seed
   blocks instead.

## 8. Open questions (tracked, not blocking)

- **P5 — champion transfer** (design §6): gates only the phase-P2
  opponent pool; requires the Drive checkpoints, normalizers, and a
  23-dim observation shim.
- **Paddle-pitch actuation as loft authority** (P0–P2 headline
  difficulty): explicitly a question for the first learned runs, on
  the frozen fixed-pitch interface first.
