"""NP1/NP2 — n-point mechanics witnesses and the recalibrated band.

The pre-registered battery from
``docs/design_paddle_tennis_npoint.md`` §4, run on the fresh
calibration block **5400–5499** (the reserved blocks 4100–4199 and
4300–4399 are never touched; the CLI refuses seed ranges that
intersect them).

**NP1 — mechanics witnesses** (scripted, PASS/FAIL rows):

- *carryover*: at every un-nudged point boundary both paddles stay
  continuous (no re-park jump); nudges increment
  ``point_serve_nudged`` only on boundary steps.
- *relaunch integrity*: after every relaunch the ball sits inside
  the mirrored serve cell, clears both paddle heads by the frozen
  0.45 m envelope, and carries a sane serve velocity — the measured
  hazard (a feed spawned into a paddle resolving at 41 m/s with no
  events) is witnessed absent. The ``serve-cell parker`` witness
  parks side A inside its own launch cell so the redraw/nudge path
  is exercised live, not just at unit level.
- *reverse-crossing hazard*: the hard-slam witness ends points with
  the ball far beyond the opponent's baseline; the next point must
  open with zero illegal-hit-group faults (the review's reproduced
  stale-sampler failure).
- *net-touch reachability*: the design pinned one-fault-per-contact
  semantics for a racket–net touch persisting across a boundary.
  On the paddle-court scene that hazard is **unreachable**: the
  collision masks (racket faces contype 8 / conaffinity 2 vs net
  panel contype 4 / conaffinity 2) never pair, and the workspace
  keeps both faces short of the net plane — both asserted here from
  the compiled model, so the pinned semantics are witnessed vacuous
  rather than assumed.
- *escrow point-boundary identity*: shaped (0.25) vs unshaped arms
  are bit-identical in trajectory; per episode
  ``paid + clawback == 0.25 x side-A confirms`` exactly, with a
  pending-escrow-at-boundary clawback witnessed at least once.
- *statue economics under None*: every completed receiving point
  nets exactly −1 and every completed serving point reads in
  {0, −1} with the 0 case witnessed (the shared +1 from the
  opponent's serve-return offsets the fault when it lands; a
  faulting feed or unconfirmed return leaves −1); zero shaping
  ever paid; every episode truncates at the cap.
- *alternation*: the server strictly alternates across points AND
  episodes, the truncation-cut partial point consuming its turn.

**NP2 — the oracle-pair band, recalibrated** (100 episodes): the
scripted pair under carryover is a genuinely new measurement, taken
here as the era's reference: crossings per episode and per completed
point, points per episode, nudge frequency, and the oracle's own
inter-point recovery travel (the L2 pilot's R2 target), plus the
full diagnosis-instrument report (whose within-point recovery hold
is the R1 oracle reference).

Usage::

    python tools/paddle_tennis_npoint_probe.py
        [--np1-episodes N] [--parker-episodes N]
        [--np2-episodes N] [--seed-start S] [--skip-np2]

    # NP3 held-out certification (reserved 4300-4399, single
    # sanctioned opening, floors pre-registered from NP2):
    python tools/paddle_tennis_npoint_probe.py --certify
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable

import numpy as np

from courtside_dynamics.envs._base import invert_piecewise_target
from courtside_dynamics.envs._paddle_court import (
    PADDLE_HOME_X,
    PADDLE_LOCAL_X_RANGE,
    PADDLE_LOCAL_Z,
    scripted_ground_opponent,
    scripted_hard_slam_witness,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import CourtSide

#: Fresh calibration block frozen in the design doc §3.
PROBE_SEED_START = 5400
NP1_EPISODES = 12
#: The parker's nudge path is stochastic (redraws usually clear a
#: single parked paddle); more episodes keep the observed-nudge
#: expectation comfortably above zero.
PARKER_EPISODES = 16
NP2_EPISODES = 100
SHAPING = 0.25

#: Reserved held-out blocks (design doc §3) this probe must refuse.
RESERVED_BLOCKS = ((4100, 4199), (4300, 4399))

#: NP3 held-out certification (design doc §4a): floors pre-registered
#: from NP2's band (11.40 crossings/episode, 113 completed points, 0%
#: nudges) BEFORE the reserved block was opened. ``--certify`` is the
#: block's single sanctioned opening.
NP3_CERT_SEED_START = 4300
NP3_CERT_EPISODES = 100
NP3_MEAN_CROSSINGS_FLOOR = 9.0
NP3_COMPLETED_POINTS_FLOOR = 50
NP3_NUDGE_RATE_CEILING = 0.02

_IDENTITY_TOL = 1e-9
#: Serve draws are 9 +/- 1 m/s; anything materially faster right at
#: relaunch would be the penetration-resolution hazard, not a serve.
_SPAWN_SPEED_BOUND = 10.5
#: The carryover bound: one control step of legitimate paddle motion
#: (matches the lockstep test), far below the ~1.5 m re-park jump.
_CARRYOVER_JUMP_BOUND = 0.5


def _statue(_observation: np.ndarray) -> np.ndarray:
    return np.zeros(3)


def _serve_cell_parker(_observation: np.ndarray) -> np.ndarray:
    """Park side A inside its own serve cell (side-local −3.25, 0, 1.3).

    Forces the relaunch clearance machinery every time side A serves;
    mid-flight feed touches it causes are legitimate event-visible
    faults (the design deliberately does not clear the corridor).
    """
    x_low, x_high = PADDLE_LOCAL_X_RANGE
    z_base, z_low, z_high = PADDLE_LOCAL_Z
    return np.array(
        [
            invert_piecewise_target(-3.25, x_low, -PADDLE_HOME_X, x_high),
            0.0,
            invert_piecewise_target(1.3 - z_base, z_low, 0.0, z_high),
        ],
        dtype=np.float64,
    )


_POINT_END_NAMES = (
    "out_of_bounds",
    "ball_net",
    "second_bounce",
    "failed_to_cross",
    "illegal_hit",
    "net_touch",
    "volley",
)


@dataclasses.dataclass
class EpisodeRow:
    """One battery episode's full witness record."""

    seed: int
    steps: int
    total_reward: float
    shaping_paid: float
    clawback: float
    confirms_a: int
    crossings: int
    points_played: int
    nudges: int
    servers: list[str]  # server of each launched point, in order
    point_rewards: list[float]  # per completed point
    partial_reward: float | None  # cap-cut partial point, if any
    boundary_jumps: list[float]  # per un-nudged boundary, both paddles
    nudged_boundaries: int
    nudges_off_boundary: int
    boundary_clawbacks: int
    spawn_clearance_min: float
    spawn_speed_max: float
    spawn_cell_violations: int
    point_end: dict[str, float]
    term_timeout: bool
    term_nonfinite: bool


def run_battery_episode(
    env: PaddleTennisEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> EpisodeRow:
    """Walk one n-point episode recording every NP1 observable."""
    observation, _ = env.reset(seed=seed)
    servers = [env._serving_side.name]
    steps = confirms = 0
    total = paid = claw = point_reward = 0.0
    prev_points = prev_nudges = 0
    point_rewards: list[float] = []
    boundary_jumps: list[float] = []
    nudged_boundaries = nudges_off_boundary = 0
    boundary_clawbacks = spawn_cell_violations = 0
    spawn_clearance_min = np.inf
    spawn_speed_max = 0.0
    point_ended = False

    while True:
        before = {
            side: env._paddle_position(side).copy()
            for side in (CourtSide.A, CourtSide.B)
        }
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        steps += 1
        total += float(reward)
        paid += float(info["rew_shaping"])
        claw += float(info["rew_shaping_clawback"])
        confirms += int(bool(info["event_valid_return_a"]))
        point_reward += float(reward)

        new_points = int(info["points_played"])
        nudges_now = int(info["point_serve_nudged"])
        nudge_delta = nudges_now - prev_nudges
        point_ended = new_points > prev_points
        if nudge_delta and not point_ended:
            nudges_off_boundary += 1
        if point_ended:
            point_rewards.append(point_reward)
            point_reward = 0.0
            if not (terminated or truncated):
                # A relaunch happened inside this step.
                if float(info["rew_shaping_clawback"]) < 0.0:
                    boundary_clawbacks += 1
                if nudge_delta:
                    nudged_boundaries += 1
                else:
                    boundary_jumps.append(
                        max(
                            float(
                                np.linalg.norm(
                                    env._paddle_position(side) - before[side]
                                )
                            )
                            for side in (CourtSide.A, CourtSide.B)
                        )
                    )
                servers.append(env._serving_side.name)
                ball = env._ball_position()
                spawn_clearance_min = min(
                    spawn_clearance_min,
                    min(
                        float(np.linalg.norm(env._paddle_position(side) - ball))
                        for side in (CourtSide.A, CourtSide.B)
                    ),
                )
                spawn_speed_max = max(
                    spawn_speed_max,
                    float(np.linalg.norm(env._ball_velocity())),
                )
                sign = -1.0 if env._serving_side is CourtSide.A else 1.0
                inside = (
                    2.99 <= sign * float(ball[0]) <= 3.51
                    and abs(float(ball[1])) <= 0.51
                    and 1.24 <= float(ball[2]) <= 1.36
                )
                if not inside:
                    spawn_cell_violations += 1
        prev_points = new_points
        prev_nudges = nudges_now
        if terminated or truncated:
            return EpisodeRow(
                seed=seed,
                steps=steps,
                total_reward=total,
                shaping_paid=paid,
                clawback=claw,
                confirms_a=confirms,
                crossings=int(info["crossings"]),
                points_played=new_points,
                nudges=nudges_now,
                servers=servers,
                point_rewards=point_rewards,
                partial_reward=None if point_ended else point_reward,
                boundary_jumps=boundary_jumps,
                nudged_boundaries=nudged_boundaries,
                nudges_off_boundary=nudges_off_boundary,
                boundary_clawbacks=boundary_clawbacks,
                spawn_clearance_min=float(spawn_clearance_min),
                spawn_speed_max=spawn_speed_max,
                spawn_cell_violations=spawn_cell_violations,
                point_end={
                    name: float(info[f"point_end_{name}"]) for name in _POINT_END_NAMES
                },
                term_timeout=bool(info["term_timeout"]),
                term_nonfinite=bool(info["term_nonfinite"]),
            )


def run_pass(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    shaping: float,
    episodes: int,
    seed_start: int,
) -> list[EpisodeRow]:
    env = PaddleTennisEnv(points_per_episode=None, contact_shaping=shaping)
    try:
        return [
            run_battery_episode(env, policy, seed)
            for seed in range(seed_start, seed_start + episodes)
        ]
    finally:
        env.close()


def scene_net_touch_rows(
    checks: list[tuple[str, bool, str]],
) -> None:
    """Measure racket–net reachability from the compiled model.

    The design's pinned one-fault-per-contact semantics for a net
    touch persisting across a boundary are vacuous on this scene iff
    (a) the collision masks never pair racket faces with the net
    panel and (b) the workspace keeps the faces short of the net
    plane; both are asserted from model constants, not assumed.
    """
    env = PaddleTennisEnv()
    try:
        env.reset(seed=PROBE_SEED_START)
        net = env.model.geom("net_panel")
        net_xmat = env.data.geom("net_panel").xmat.reshape(3, 3)
        net_extent = float(np.abs(net_xmat[0]) @ np.asarray(net.size, dtype=np.float64))
        net_x = float(env.data.geom("net_panel").xpos[0])
        for side, face_name, base_name, joint_name in (
            (
                "A",
                "player_a_racket_face",
                "player_a_paddle_base",
                "player_a_slide_x",
            ),
            (
                "B",
                "player_b_racket_face",
                "player_b_paddle_base",
                "player_b_slide_x",
            ),
        ):
            face = env.model.geom(face_name)
            mask = (int(face.contype[0]) & int(net.conaffinity[0])) | (
                int(net.contype[0]) & int(face.conaffinity[0])
            )
            checks.append(
                (
                    f"net-touch: collision mask never pairs face {side} with the net",
                    mask == 0,
                    f"pairing product {mask}",
                )
            )
            xmat = env.data.geom(face_name).xmat.reshape(3, 3)
            extent = float(np.abs(xmat[0]) @ np.asarray(face.size, dtype=np.float64))
            base_x = float(env.model.body(base_name).pos[0])
            low, high = (float(v) for v in env.model.joint(joint_name).range)
            if side == "A":
                closest = base_x + high + extent
                margin = (net_x - net_extent) - closest
            else:
                closest = base_x + low - extent
                margin = closest - (net_x + net_extent)
            checks.append(
                (
                    f"net-touch: workspace keeps face {side} short of the net plane",
                    margin > 0.0,
                    f"margin {margin:.3f} m",
                )
            )
    finally:
        env.close()


def evaluate_np1(
    passes: dict[str, list[EpisodeRow]],
) -> list[tuple[str, bool, str]]:
    """The pre-registered NP1 criteria as (name, passed, detail) rows.

    ``passes`` keys: statue, oracle_shaped, oracle_unshaped,
    slam_shaped, slam_unshaped, parker.
    """
    checks: list[tuple[str, bool, str]] = []
    scene_net_touch_rows(checks)

    every_row = [row for rows in passes.values() for row in rows]
    net_touch_ends = sum(row.point_end["net_touch"] for row in every_row)
    checks.append(
        (
            "net-touch: zero point_end_net_touch across the battery",
            net_touch_ends == 0.0,
            f"{net_touch_ends:.0f} endings",
        )
    )

    jumps = [j for row in every_row for j in row.boundary_jumps]
    checks.append(
        (
            "carryover: un-nudged boundaries continuous",
            bool(jumps) and max(jumps) < _CARRYOVER_JUMP_BOUND,
            f"{len(jumps)} boundaries, max jump "
            f"{max(jumps, default=float('nan')):.3f} m",
        )
    )
    off_boundary = sum(row.nudges_off_boundary for row in every_row)
    checks.append(
        (
            "carryover: nudges only ever fire on boundary steps",
            off_boundary == 0,
            f"{off_boundary} off-boundary increments",
        )
    )

    clearance = min(
        (
            row.spawn_clearance_min
            for row in every_row
            if np.isfinite(row.spawn_clearance_min)
        ),
        default=float("inf"),
    )
    checks.append(
        (
            "relaunch: every spawn clears both paddles by 0.45 m",
            clearance >= 0.45 - 1e-9,
            f"min clearance {clearance:.3f} m",
        )
    )
    speed = max((row.spawn_speed_max for row in every_row), default=0.0)
    checks.append(
        (
            "relaunch: spawn velocity is a serve draw, never a deflection",
            speed <= _SPAWN_SPEED_BOUND,
            f"max spawn speed {speed:.2f} m/s",
        )
    )
    cell_violations = sum(row.spawn_cell_violations for row in every_row)
    checks.append(
        (
            "relaunch: every spawn lands inside the mirrored serve cell",
            cell_violations == 0,
            f"{cell_violations} violations",
        )
    )
    parker_nudges = sum(row.nudges for row in passes["parker"])
    parker_boundaries = sum(row.points_played for row in passes["parker"])
    checks.append(
        (
            "relaunch: parked-paddle witness exercises the nudge path",
            parker_nudges >= 1,
            f"{parker_nudges} nudges over {parker_boundaries} boundaries",
        )
    )

    slam_rows = passes["slam_shaped"] + passes["slam_unshaped"]
    slam_illegal = sum(row.point_end["illegal_hit"] for row in slam_rows)
    slam_points = sum(row.points_played for row in slam_rows)
    checks.append(
        (
            "reverse-crossing hazard: far-side deaths open cleanly",
            slam_illegal == 0.0 and all(row.points_played >= 1 for row in slam_rows),
            f"{slam_points} slam points, {slam_illegal:.0f} illegal-hit endings",
        )
    )

    for name in ("oracle", "slam"):
        shaped = passes[f"{name}_shaped"]
        unshaped = passes[f"{name}_unshaped"]
        worst_escrow = worst_total = 0.0
        identical = True
        for srow, urow in zip(shaped, unshaped, strict=True):
            expected = SHAPING * srow.confirms_a
            worst_escrow = max(
                worst_escrow,
                abs(srow.shaping_paid + srow.clawback - expected),
            )
            worst_total = max(
                worst_total,
                abs(srow.total_reward - urow.total_reward - expected),
            )
            identical = identical and (
                srow.steps == urow.steps
                and srow.points_played == urow.points_played
                and srow.confirms_a == urow.confirms_a
                and srow.crossings == urow.crossings
            )
        checks.append(
            (
                f"escrow: {name} per-episode identity across boundaries",
                worst_escrow <= _IDENTITY_TOL,
                f"worst |paid+clawback - 0.25*confirms| = {worst_escrow:.2e}",
            )
        )
        checks.append(
            (
                f"escrow: {name} shaped-minus-unshaped total is the kept escrow",
                worst_total <= _IDENTITY_TOL,
                f"worst deviation {worst_total:.2e}",
            )
        )
        checks.append(
            (
                f"escrow: {name} arms bit-identical",
                identical,
                "steps/points/confirms/crossings all equal",
            )
        )
    boundary_clawbacks = sum(row.boundary_clawbacks for row in passes["slam_shaped"])
    checks.append(
        (
            "escrow: pending-at-point-end clawback witnessed",
            boundary_clawbacks >= 1,
            f"{boundary_clawbacks} nonzero boundary clawbacks",
        )
    )

    statue = passes["statue"]
    worst_receiving = 0.0
    serving_zero = serving_minus_one = serving_other = 0
    for row in statue:
        for server, point_reward in zip(row.servers, row.point_rewards, strict=False):
            if server == "B":
                worst_receiving = max(worst_receiving, abs(point_reward - -1.0))
            elif abs(point_reward) <= _IDENTITY_TOL:
                serving_zero += 1
            elif abs(point_reward - -1.0) <= _IDENTITY_TOL:
                # No offsetting +1: the feed itself faulted or the
                # opponent's serve-return failed to confirm.
                serving_minus_one += 1
            else:
                serving_other += 1
    checks.append(
        (
            "statue: every completed receiving point nets exactly -1",
            worst_receiving <= _IDENTITY_TOL,
            f"worst deviation {worst_receiving:.2e}",
        )
    )
    checks.append(
        (
            "statue: serving points read in {0, -1}, offset economics witnessed",
            serving_other == 0 and serving_zero >= 1,
            f"{serving_zero} offset to 0, {serving_minus_one} at -1,"
            f" {serving_other} other",
        )
    )
    statue_shaping = sum(abs(row.shaping_paid) + abs(row.clawback) for row in statue)
    checks.append(
        (
            "statue: zero shaping under n-point play",
            statue_shaping == 0.0,
            f"|paid|+|clawback| = {statue_shaping}",
        )
    )
    checks.append(
        (
            "statue: every episode truncates at the cap",
            all(row.term_timeout and not row.term_nonfinite for row in statue),
            f"{len(statue)} episodes",
        )
    )
    counter_drift = max(
        (
            abs(sum(row.point_end.values()) - float(row.points_played))
            for row in every_row
        ),
        default=0.0,
    )
    checks.append(
        (
            "points_played equals the point_end counter total",
            counter_drift == 0.0,
            f"max drift {counter_drift:.0f}",
        )
    )

    for name, rows in passes.items():
        sequence: list[str] = []
        for row in rows:
            sequence.extend(row.servers)
        alternates = all(
            first != second
            for first, second in zip(sequence, sequence[1:], strict=False)
        )
        checks.append(
            (
                f"alternation: strict across points and episodes ({name})",
                alternates,
                f"{len(sequence)} launches",
            )
        )
    return checks


def run_np2(
    *, episodes: int, seed_start: int
) -> tuple[str, list[str], dict[str, float]]:
    """The oracle-pair band; returns (report, band lines, aggregates)."""
    from courtside_dynamics.training.paddle_diagnosis import (
        report as diag_report,
    )
    from courtside_dynamics.training.paddle_diagnosis import (
        run_episode as diag_run_episode,
    )

    env = PaddleTennisEnv(points_per_episode=None)
    crossings: list[int] = []
    points: list[int] = []
    nudges: list[int] = []
    completed_crossings: list[int] = []
    unsafe = 0
    relaunches = 0
    all_traces = []
    all_travels: list[float] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            traces, travels = diag_run_episode(env, scripted_ground_opponent, seed)
            crossings.append(int(env._crossings))
            points.append(int(env._points_played))
            nudges.append(int(env._point_serve_nudged))
            completed_crossings.append(int(env._crossings_base))
            if traces[-1].termination in (
                "nonfinite_state",
                "unsafe_physics",
            ):
                unsafe += 1
            # The instrument opens exactly one inter-point window per
            # mid-episode relaunch, so travels counts relaunches (a
            # point absorbed exactly at the cap relaunches nothing).
            relaunches += len(travels)
            all_traces.extend(traces)
            all_travels.extend(travels)
    finally:
        env.close()

    per_point = float(np.sum(completed_crossings) / max(np.sum(points), 1))
    band = [
        f"NP2 band ({episodes} episodes, seeds {seed_start}-"
        f"{seed_start + episodes - 1}):",
        "  crossings/episode        mean "
        f"{np.mean(crossings):6.2f}  std {np.std(crossings):5.2f}"
        f"  min {min(crossings)}  max {max(crossings)}",
        "  completed points/episode mean "
        f"{np.mean(points):6.2f}  std {np.std(points):5.2f}",
        f"  crossings per completed point (bridge metric) {per_point:.2f}",
        f"  nudges: {sum(nudges)} over {relaunches} relaunches"
        f" ({sum(nudges) / max(relaunches, 1):.1%})",
        "  inter-point recovery travel mean "
        f"{np.mean(all_travels):.2f} m  p90 "
        f"{np.percentile(all_travels, 90):.2f} m"
        f"  (n={len(all_travels)})",
        f"  unsafe episodes: {unsafe}",
    ]
    return (
        diag_report(
            all_traces,
            "NP2 oracle band (n-point carryover)",
            interpoint_travels=all_travels,
        ),
        band,
        {
            "mean_crossings": float(np.mean(crossings)),
            "completed_points": float(np.sum(points)),
            "nudge_rate": float(sum(nudges) / max(relaunches, 1)),
            "unsafe": float(unsafe),
        },
    )


def _refuse_reserved(seed_start: int, episodes: int) -> None:
    last = seed_start + episodes - 1
    for low, high in RESERVED_BLOCKS:
        if seed_start <= high and last >= low:
            raise SystemExit(
                f"seed range {seed_start}-{last} intersects the"
                f" reserved held-out block {low}-{high}"
            )


def certify() -> int:
    """NP3: the reserved block's single sanctioned opening.

    Runs the band measurement on held-out seeds 4300–4399 and holds
    it to the floors pre-registered from NP2 (design doc §4a).
    """
    print(
        "NP3 held-out certification (reserved seeds "
        f"{NP3_CERT_SEED_START}-"
        f"{NP3_CERT_SEED_START + NP3_CERT_EPISODES - 1})"
    )
    instrument_report, band, stats = run_np2(
        episodes=NP3_CERT_EPISODES, seed_start=NP3_CERT_SEED_START
    )
    print("\n".join(band))
    print()
    print(instrument_report)
    print()
    floors = (
        (
            f"mean crossings/episode >= {NP3_MEAN_CROSSINGS_FLOOR}",
            stats["mean_crossings"] >= NP3_MEAN_CROSSINGS_FLOOR,
            f"measured {stats['mean_crossings']:.2f}",
        ),
        (
            f"completed points >= {NP3_COMPLETED_POINTS_FLOOR}",
            stats["completed_points"] >= NP3_COMPLETED_POINTS_FLOOR,
            f"measured {stats['completed_points']:.0f}",
        ),
        (
            f"nudge rate <= {NP3_NUDGE_RATE_CEILING:.0%} of relaunches",
            stats["nudge_rate"] <= NP3_NUDGE_RATE_CEILING,
            f"measured {stats['nudge_rate']:.1%}",
        ),
        (
            "zero unsafe terminations",
            stats["unsafe"] == 0.0,
            f"measured {stats['unsafe']:.0f}",
        ),
    )
    for name, passed, detail in floors:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({detail})")
    verdict = all(passed for _name, passed, _detail in floors)
    print()
    print(f"NP3 verdict: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np1-episodes", type=int, default=NP1_EPISODES)
    parser.add_argument("--parker-episodes", type=int, default=PARKER_EPISODES)
    parser.add_argument("--np2-episodes", type=int, default=NP2_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PROBE_SEED_START)
    parser.add_argument("--skip-np2", action="store_true")
    parser.add_argument(
        "--certify",
        action="store_true",
        help="NP3 held-out certification on the reserved block "
        "4300-4399 against the pre-registered floors (the block's "
        "single sanctioned opening; ignores the other flags)",
    )
    args = parser.parse_args()
    if args.certify:
        raise SystemExit(certify())
    episodes_max = max(
        args.np1_episodes,
        args.parker_episodes,
        0 if args.skip_np2 else args.np2_episodes,
    )
    _refuse_reserved(args.seed_start, episodes_max)

    passes: dict[str, list[EpisodeRow]] = {}
    for name, policy, shaping, episodes in (
        ("statue", _statue, SHAPING, args.np1_episodes),
        (
            "oracle_shaped",
            scripted_ground_opponent,
            SHAPING,
            args.np1_episodes,
        ),
        (
            "oracle_unshaped",
            scripted_ground_opponent,
            0.0,
            args.np1_episodes,
        ),
        (
            "slam_shaped",
            scripted_hard_slam_witness,
            SHAPING,
            args.np1_episodes,
        ),
        (
            "slam_unshaped",
            scripted_hard_slam_witness,
            0.0,
            args.np1_episodes,
        ),
        (
            "parker",
            _serve_cell_parker,
            0.0,
            args.parker_episodes,
        ),
    ):
        passes[name] = run_pass(
            policy,
            shaping=shaping,
            episodes=episodes,
            seed_start=args.seed_start,
        )
        points = sum(row.points_played for row in passes[name])
        print(f"[{name}] {episodes} episodes, {points} completed points")

    print()
    print("NP1 mechanics witnesses")
    checks = evaluate_np1(passes)
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} ({detail})")
    verdict = all(passed for _name, passed, _detail in checks)
    print()
    print(f"NP1 verdict: {'PASS' if verdict else 'FAIL'}")

    if not args.skip_np2:
        print()
        instrument_report, band, _stats = run_np2(
            episodes=args.np2_episodes, seed_start=args.seed_start
        )
        print("\n".join(band))
        print()
        print(instrument_report)


if __name__ == "__main__":
    main()
