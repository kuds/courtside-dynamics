# Design sketch: PaddleTennis — 1v1 rally play on the full court

Status: **Adopted — phase-P1 environment shipped**, 2026-08-02
(originally proposed the same day against v0.25.0). The probe battery
ran first, per doctrine: P0–P4 froze the geometry, reference band,
premise, serve rules, and mirroring contract (§6 status notes), and
the registered `CourtsideDynamics/PaddleTennis` env implements that
frozen definition
([`paddle_tennis_env_20260802.md`](paddle_tennis_env_20260802.md)).
P5 (champion transfer) remains open and gates only the phase-P2
opponent pool.

## 1. Why leave the wall

The true-baseline replication closed the wall-ball chapter with a
measured structural verdict
([`wall_ball_true_baseline_replication_20260801_review.md`](wall_ball_true_baseline_replication_20260801_review.md)):
the wall only hits short (rebounds land mean x = −1.0, never deeper
than −6.1), so deep play can never be instrumentally useful against
it — it must be decreed, and decrees invite exploits (seed 1's serve
volley). Against an *opponent*, returns land with genuine depth
variance, so covering the baseline becomes useful because of what the
opponent does, not because a fence forbids the alternative. Baseline
play becomes something to *learn*, not to legislate.

PaddleTennis is also the missing rung in the repo's ladder: the envs
jump from "paddle vs wall" (solved) to "two G1 humanoids on a court"
(built, never seriously trained, known feasibility concerns per the
v0.7.0 review). A two-paddle full-court env de-risks the humanoid
destination directly — same court, same rules machinery, trivial
embodiment.

## 2. What already exists to reuse

- **Court**: `humanoid_tennis.xml` carries the full ITF-court
  geometry; the wall-ball tennis overlay has the half-court to-size.
  The net replaces the wall as the central obstacle.
- **Rules/events**: `tennis_rules.py` (`RallyRules`),
  `_tennis_events.py`, `_tennis_physics.py`, `TennisServeConfig` are
  shared modules, deliberately not welded to the humanoid env.
- **Paddle + ball physics**: wall-ball's calibrated face-only paddle
  (3-action position-target interface, ±100 N, damping calibrations)
  and ball (restitution-calibrated, priority contact params) transfer
  as-is.
- **Opponents, day one**: the certification oracle family
  (`lead_charge`) is a competent scripted player, and the era left
  two confirmed champions with opposite styles — the deep receiver
  (`20260731_132322`) and the volleyer (`20260801_144043`) — a
  ready-made frozen opponent pool. (Champion transfer to the new
  court is an experiment, not an assumption — their observations were
  learned against a wall at +4; §6 P5 measures this.)
- **Methodology**: probe battery → frozen task → held-out
  certification → pre-registered run, plus the training-safety
  machinery that has now run 2-for-2 unattended.

## 3. Environment sketch

- **Geometry**: full court, net at x = 0 (height to be probed; the
  ITF 0.914 m as the starting candidate — high enough to shape arcs,
  low enough that wall-ball-power returns clear it). Side A paddle
  workspace on x < 0, side B mirrored on x > 0. Court width from the
  ITF overlay (singles ±4.115) rather than wall-ball's ±5.5 — to be
  probed.
- **Agents**: phase 1 is **single-agent** — the policy controls side
  A with the familiar 3-action normalized interface; side B is an
  opponent controller (scripted oracle or frozen policy) driven
  through the same interface with mirrored coordinates.
- **Observation**: side-relative frame (own-side coordinates flipped
  so one policy can play either side), extending the wall-ball 23-dim
  template with the opponent paddle's state (+3 pos, likely +3 vel).
  Exact layout frozen after probes.
- **Objective, phase 1**: cooperative rally — reward per legal
  crossing of the net that lands in, mirroring wall-ball's completed
  return and `HumanoidTennisCoopEnv`'s rally-target design. Point
  scoring/adversarial play is explicitly deferred (sparse win/loss is
  a bad first signal, and rally skill is the prerequisite either
  way).
- **Rules**: `RallyRules` semantics — serve, single bounce allowed
  per side, net faults, out landings. No fences: the court and the
  opponent are the geometry.

## 4. Phases

1. **P1 — vs scripted oracle, cooperative rally.** Plain SB3
   single-agent; the oracle returns what it can reach. Success =
   sustained multi-crossing rallies on a certified-feasible task.
2. **P2 — frozen opponent pool.** Swap the scripted opponent for the
   wall-ball champions (if P5 shows transfer) and P1's own graduates;
   opponent sampled per episode. Robustness across styles is the
   metric.
3. **P3 — self-play and scoring** (only if P1/P2 earn it): periodic
   frozen-copy self-play, then adversarial point scoring.
4. **P4 — hand the court to the humanoids**: the PaddleTennis task
   definitions, curricula, and opponents become the humanoid env's
   graduation targets.

## 5. What this costs (honest scope)

New XML (court + net + two paddles), a new env class (~WallBallEnv
scale, minus its accumulated preset machinery, plus mirroring and
opponent plumbing), a new metric/telemetry suite, a new probe
battery, and a fresh comparability corpus (no cross-era number
survives the move). Opponent management is a genuinely new
infrastructure axis — kept small in P1 (a callable controller on the
B side) and grown only as phases earn it.

## 6. Pre-committed probe battery (before any env code ships)

> **Status update, 2026-08-02: P0–P2 have run** — results in
> [`paddle_tennis_probes_20260802.md`](paddle_tennis_probes_20260802.md).
> Headlines: the premise holds (in-rally landings mean 3.3 m deep,
> 50% past 3 m); the full ITF court is infeasible at the paddles'
> power ceiling and the geometry freezes at half-length 6.5 m /
> net 0.914 m; the scripted rally band is 2.0 crossings/point
> (max 10, ≥4 in 33%); and loft control with the fixed-pitch face is
> identified as the era's core difficulty, with strike height as a
> measured control channel.
>
> **Status update, 2026-08-02 (later): P3–P4 have also run** — on the
> committed probe substrate (`envs/_paddle_court.py`, an unregistered,
> reward-free prototype scene; results in
> [`paddle_tennis_probes_p3_p4_20260802.md`](paddle_tennis_probes_p3_p4_20260802.md)).
> The serve band and the mirroring identity are measured; **P5
> (champion transfer) remains open** and gates the opponent-pool
> decision. The env *definition* (registered id, rewards, recipe)
> remains unshipped pending the freeze + certification this doctrine
> requires.
>
> **Status update, 2026-08-02 (env freeze): the phase-P1 environment
> has shipped** on the P0–P4 numbers —
> `CourtsideDynamics/PaddleTennis` (`envs/paddle_tennis.py`), the
> `PaddleTennis` recipe, and the frozen task definition recorded in
> [`paddle_tennis_env_20260802.md`](paddle_tennis_env_20260802.md).
>
> **Status update, 2026-08-08 (pilot diagnosis):** the ground-era
> pilot early-stopped at 1.37 crossings (band 7.78); the behavioral
> probe ([`paddle_tennis_diagnosis_20260808.md`](paddle_tennis_diagnosis_20260808.md))
> shows one memorized serve-return macro and no general
> ball-reaching (serving-side survival 0%, touch rate 37%), rejects
> the stroke-authority and opponent-asymmetry explanations, and
> ranks sustained exploration and n-point episodes as the next
> probed changes. The instrument now runs at every checkpoint.
>
> **Status update, 2026-08-03 (ground rules):** the first GPU run
> (volley era, seed 0, unregistered) maximized return rate with a
> close-net volley loop — crossings 37.6 at a 14-step cadence. The
> probed fix
> ([`paddle_tennis_ground_rules_20260803.md`](paddle_tennis_ground_rules_20260803.md))
> makes pre-bounce returns a fault (`volley_rule="fault"`, now the
> registered default), recalibrates the scripted reference (ground
> band 7.78; the frozen P1 oracle's returns were largely volleys),
> and re-certifies held-out (7.68 on block 4200–4299). The volley-era
> first-run pre-registration is superseded unconsumed; 4100–4199
> stays reserved for the ground-era first run.
>
> **Status update, 2026-08-02 (first learned evidence + first-run
> pre-registration):** the local SAC pilot of the frozen recipe
> passes the scripted band at its 175k eval and reaches crossings
> 6.40 best (final quarter oscillating 5.2–6.4)
> ([`paddle_tennis_pilot_and_first_run_20260802.md`](paddle_tennis_pilot_and_first_run_20260802.md)),
> which also pre-registers the first GPU run (seed 1, stock TOML,
> primary ≥ 6.0, held-out gate on block 4100–4199).
>
> **Status update, 2026-08-02 (P5 instrument):** the transfer shim
> shipped and its scripted calibration ran
> ([`paddle_tennis_p5_transfer_20260802.md`](paddle_tennis_p5_transfer_20260802.md)):
> the `scaled + yield` configuration is the only viable one (rigid
> translation is broken by command-range geometry; the serve-yield
> overlay is mandatory — wall-ball players never learned to stand
> down during their own serve). The champion measurements themselves
> run on Colab against a pre-registered pool-admission rule; until
> they do, P5's opponent-pool decision stays open.

All scripted, no learning, calibration seed blocks; numbers frozen
into the task definition the way T1–T7 froze the true-baseline era:

- **P0 — net/ball feasibility map**: for candidate net heights and
  serve configs, measure clearance rates and landing depths of
  wall-ball-calibrated strokes launched from each side. Kills or
  confirms the 0.914 m net.
- **P1 — oracle-vs-oracle rally band**: two `lead_charge`-family
  controllers across the net; measure sustained crossings, cadence,
  failure taxonomy. This is the era's reference band and the
  certification bar's calibration — if scripted play cannot rally at
  all, the task definition changes before any training.
- **P2 — return-landing distribution**: where do a rallying
  opponent's returns actually land? The whole premise is that this
  distribution has real depth mass (unlike the wall's −1.0 mean);
  measure it, don't assume it.
- **P3 — serve rules**: who serves, alternation, serve origin/speed
  ranges that produce legal, returnable serves on the full court.
- **P4 — observation/mirroring identity**: bit-for-bit check that a
  mirrored state produces mirrored observations and that a policy
  playing side B through the mirror is equivalent to side A.
- **P5 — champion transfer**: both wall-ball champions dropped onto
  the new court (own side, mirrored) vs the oracle — do they play at
  all? Sets whether P2's pool starts warm or must be regrown.

## 7. Open questions the probes must answer

Net height and restitution; court width; whether the opponent paddle
needs full state in the observation or position only; serve
alternation semantics; episode structure (fixed length vs
first-to-N-crossings); cooperative reward shaping scale; and whether
the volleyer champion's style is an asset (aggressive net play) or a
degenerate attractor on the real court — the first environment where
that question has a fair answer.
