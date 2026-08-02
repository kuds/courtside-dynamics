"""PaddleTennis probe P3: serve rules on the frozen court.

Design doc contract (docs/design_paddle_tennis.md §6): before any env
ships, P3 must measure "who serves, alternation, serve origin/speed
ranges that produce legal, returnable serves on the full court". The
P0-P2 battery froze the court (half-length 6.5, net 0.914, width
4.115) and the scripted reference controller (`lead_charge` family:
gap 0.8, swing-through 0.4, strike offset -0.12); this harness sweeps
ballistic serve configurations against that frozen receiver and
reports, per cell:

- legal-serve rate: the serve crosses the net and its first bounce
  lands inside the receiver's court;
- returnable rate: the frozen receiver makes a legal racket return of
  a legal serve (rules-confirmed crossing of the return);
- rally tail: mean crossings per point and the failure taxonomy.

Alternation: every cell is also run with the sides swapped. Probe P4
proved the mirror identity bit-for-bit at the state level, so the
mirrored cells double as an end-to-end empirical check — their
statistics must match the primary cells up to simulation ulps.

Scripted probes only. Seeds come from the calibration block recorded
in the probe doc's ledger; the reserved held-out blocks (3100-3199,
4100-4199) are never touched here.

Usage::

    python tools/paddle_tennis_probes.py            # full P3 sweep
    python tools/paddle_tennis_probes.py --quick    # smoke (tests/CI)
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter

import numpy as np

from courtside_dynamics.envs._base import invert_piecewise_target
from courtside_dynamics.envs._paddle_court import (
    PADDLE_COURT,
    PaddleCourtScene,
    PaddleCourtServe,
)
from courtside_dynamics.envs.tennis_rules import (
    COURT_EVENT_KINDS as _COURT_EVENT_KINDS,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyPhase,
    TerminationReason,
)

_GRAVITY = 9.81

#: The P1-frozen scripted receiver (probe doc §3): commit gap, swing-
#: through past the ball toward the net, strike-height offset below
#: ball center. Values are the frozen best configuration.
LEAD_CHARGE_GAP = 0.8
LEAD_CHARGE_SWING = 0.4
LEAD_CHARGE_STRIKE = -0.12

#: Side-local paddle constants (paddle_court.xml): home pivot, the
#: mapping span, and the z slide's local range around base height 1.2.
_HOME_X = -1.7
_X_LOW, _X_HIGH = -6.4, -0.1
_Y_SPAN = 3.0
_Z_BASE, _Z_LOW, _Z_HIGH = 1.2, -0.9, 2.0

#: Default control-frame budget per point: 6 s of simulated time at
#: the 100 Hz control rate, comfortably past the longest probe rallies.
MAX_CONTROL_STEPS = 600


def lead_charge_local_action(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    paddle_position: np.ndarray,
    *,
    gap: float = LEAD_CHARGE_GAP,
    swing: float = LEAD_CHARGE_SWING,
    strike: float = LEAD_CHARGE_STRIKE,
) -> np.ndarray:
    """The frozen P1 controller, in the side-local frame.

    A world-frame port of the certification ``lead_charge`` oracle
    (training/ladder_certification.py): park at home while the ball is
    outgoing, and once an incoming ball is within ``gap`` of the
    paddle, charge with a ballistic y/z lead aimed ``strike`` below the
    projected ball height, driving the target ``swing`` beyond the
    ball toward the net. Volleys are legal on this court, so unlike the
    wall-ball oracle there is no bounce-count precondition.
    """
    ball_x, ball_y, ball_z = ball_position
    vx, vy, vz = ball_velocity
    paddle_x = paddle_position[0]
    incoming = vx < -0.1

    if incoming and (ball_x - paddle_x) <= gap:
        target_x = min(ball_x + swing, _X_HIGH)
        t_hit = float(
            np.clip((ball_x - paddle_x) / max(0.5, -vx + 2.5), 0.0, 1.2)
        )
        target_y = ball_y + vy * t_hit
        target_z = (
            max(0.25, ball_z + vz * t_hit - 0.5 * _GRAVITY * t_hit**2)
            + strike
        )
    elif incoming:
        # Pre-position while the ball closes: track its y and height
        # from the home column (the wall-ball oracle's waiting posture)
        # so the strike is reachable when the commit gap trips.
        target_x = _HOME_X
        target_y = ball_y
        target_z = max(0.25, ball_z)
    else:
        # Yield while the ball is outgoing: the serve launches from the
        # server's own half and passes through the home column, so a
        # ball-tracking park (safe for the wall-ball oracle, whose
        # serves launched beyond the paddle) would swat the server's
        # own feed -- an illegal WRONG_HITTER touch.
        target_x = _HOME_X
        target_y = 0.0
        target_z = _Z_BASE

    action_x = invert_piecewise_target(target_x, _X_LOW, _HOME_X, _X_HIGH)
    action_y = float(np.clip(target_y / _Y_SPAN, -1.0, 1.0))
    action_z = invert_piecewise_target(
        target_z - _Z_BASE, _Z_LOW, 0.0, _Z_HIGH
    )
    return np.array([action_x, action_y, action_z], dtype=np.float64)


@dataclasses.dataclass(slots=True)
class PointResult:
    """Rules-scored outcome of one served point."""

    serve_crossed: bool
    serve_first_bounce_in: bool
    first_bounce_depth: float | None
    receiver_returned: bool
    crossings: int
    termination: TerminationReason
    control_steps: int


def play_point(
    scene: PaddleCourtScene,
    serve: PaddleCourtServe,
    *,
    serving_side: CourtSide,
    rng: np.random.Generator,
    active_receiver: bool = True,
    max_control_steps: int = MAX_CONTROL_STEPS,
) -> PointResult:
    """Serve and play one point.

    ``active_receiver=False`` parks both paddles in the neutral yield
    posture for the whole point: the policy-independent serve-landing
    measurement, exactly the parked-paddle probe the wall-ball
    certification uses. The same seed draws the same serve in both
    modes, so a cell can pair each landing verdict with its active
    outcome.
    """
    serving_side = CourtSide(serving_side)
    receiver = serving_side.opponent
    scene.serve(serve, serving_side=serving_side, rng=rng)

    serve_crossed = False
    serve_first_bounce_in = False
    first_bounce_depth: float | None = None
    receiver_returned = False
    yield_action = np.array([0.0, 0.0, 0.0])
    steps_played = max_control_steps
    for step in range(max_control_steps):
        for side in (CourtSide.A, CourtSide.B):
            if not active_receiver:
                scene.apply_action(side, yield_action)
                continue
            ball = scene.ball_position()
            ball_vel = scene.ball_velocity()
            paddle = scene.paddle_position(side)
            if side is CourtSide.B:
                ball = ball.copy()
                ball_vel = ball_vel.copy()
                paddle = paddle.copy()
                ball[:2] *= -1.0
                ball_vel[:2] *= -1.0
                paddle[:2] *= -1.0
            scene.apply_action(
                side, lead_charge_local_action(ball, ball_vel, paddle)
            )
        batch = scene.step_control_frame(step)
        transition = scene.rules.advance(
            batch.events, contact_peaks=batch.contact_peaks
        )
        after = transition.after
        if not serve_crossed and after.feed_crossed_net:
            serve_crossed = True
        if (
            not serve_first_bounce_in
            and after.bounce_count > 0
            and after.phase is not RallyPhase.TERMINAL
            and after.bounce_side is receiver
        ):
            serve_first_bounce_in = True
            for event in transition.processed_events:
                if event.kind in _COURT_EVENT_KINDS and event.position:
                    # Depth past the net, in the receiver's own frame.
                    first_bounce_depth = abs(float(event.position[0]))
                    break
        if receiver in transition.confirmed_returns:
            receiver_returned = True
        if after.terminated:
            steps_played = step + 1
            break

    return PointResult(
        serve_crossed=serve_crossed,
        serve_first_bounce_in=serve_first_bounce_in,
        first_bounce_depth=first_bounce_depth,
        receiver_returned=receiver_returned,
        # The P1 metric counts RETURN crossings (median 0 = unreturned
        # serve), not the feed itself.
        crossings=int(after.shot_crossing_count),
        termination=after.termination_reason,
        control_steps=steps_played,
    )


@dataclasses.dataclass(slots=True)
class CellResult:
    """Aggregated P3 metrics for one serve configuration."""

    serve: PaddleCourtServe
    serving_side: CourtSide
    points: int
    legal_serve_rate: float
    mean_landing_depth: float
    returnable_rate: float
    mean_crossings: float
    terminations: Counter

    def row(self) -> str:
        top = ", ".join(
            f"{reason.name.lower()} {count}"
            for reason, count in self.terminations.most_common(2)
        )
        return (
            f"| {self.serve.start_distance_from_net:.2f} "
            f"| {self.serve.speed:.0f} "
            f"| {self.serve.elevation_degrees:.0f} "
            f"| {self.serving_side.label} "
            f"| {self.legal_serve_rate:.0%} "
            f"| {self.mean_landing_depth:.2f} "
            f"| {self.returnable_rate:.0%} "
            f"| {self.mean_crossings:.2f} "
            f"| {top} |"
        )


def run_cell(
    scene: PaddleCourtScene,
    serve: PaddleCourtServe,
    *,
    serving_side: CourtSide,
    points: int,
    seed_start: int,
) -> CellResult:
    """Play ``points`` seeded points, each as a parked/active pair.

    The parked pass measures the serve itself (crossed + first bounce
    inside the receiver's court, policy-independent); the active pass,
    on the same seed and therefore the same serve draw, measures what
    the frozen receiver makes of it.
    """
    parked = [
        play_point(
            scene,
            serve,
            serving_side=serving_side,
            rng=np.random.default_rng(seed_start + index),
            active_receiver=False,
        )
        for index in range(points)
    ]
    active = [
        play_point(
            scene,
            serve,
            serving_side=serving_side,
            rng=np.random.default_rng(seed_start + index),
        )
        for index in range(points)
    ]
    legal_indices = [
        index
        for index, r in enumerate(parked)
        if r.serve_crossed and r.serve_first_bounce_in
    ]
    depths = [
        parked[index].first_bounce_depth
        for index in legal_indices
        if parked[index].first_bounce_depth is not None
    ]
    returned = [
        index
        for index in legal_indices
        if active[index].receiver_returned
    ]
    return CellResult(
        serve=serve,
        serving_side=serving_side,
        points=points,
        legal_serve_rate=len(legal_indices) / points,
        mean_landing_depth=(
            float(np.mean(depths)) if depths else float("nan")
        ),
        returnable_rate=(
            len(returned) / len(legal_indices) if legal_indices else 0.0
        ),
        mean_crossings=float(np.mean([r.crossings for r in active])),
        terminations=Counter(r.termination for r in active),
    )


def sweep_serve_rules(
    *,
    points: int = 40,
    seed_start: int = 1200,
    quick: bool = False,
) -> list[CellResult]:
    """The P3 grid: origin depth x speed x elevation, both serving sides.

    The grid brackets the P0-viable envelope (10-12 m/s, 18-24 deg)
    from three origin depths; each primary (side A) cell has a
    side-B alternation twin sharing its seed block.
    """
    scene = PaddleCourtScene()
    depths: tuple[float, ...] = (2.0, 3.25, 4.5)
    speeds: tuple[float, ...] = (8.0, 9.0, 10.0, 11.0)
    elevations: tuple[float, ...] = (18.0, 21.0, 24.0)
    if quick:
        depths, speeds, elevations = (3.25,), (11.0,), (21.0,)
        points = min(points, 4)

    cells: list[CellResult] = []
    seed = seed_start
    for depth in depths:
        for speed in speeds:
            for elevation in elevations:
                serve = PaddleCourtServe(
                    start_distance_from_net=depth,
                    speed=speed,
                    elevation_degrees=elevation,
                )
                for side in (CourtSide.A, CourtSide.B):
                    cells.append(
                        run_cell(
                            scene,
                            serve,
                            serving_side=side,
                            points=points,
                            seed_start=seed,
                        )
                    )
                seed += points
    return cells


def format_report(cells: list[CellResult]) -> str:
    lines = [
        "P3 serve-rules sweep "
        f"(court half-length {PADDLE_COURT.half_length}, "
        f"net 0.914, frozen lead_charge receiver)",
        "",
        "| depth | speed | elev | side | legal | land-depth "
        "| returnable | crossings | top terminations |",
        "|---|---|---|---|---|---|---|---|---|",
        *(cell.row() for cell in cells),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points", type=int, default=40, help="points per cell"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1200,
        help="first calibration seed (see the probe doc's ledger)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="single-cell smoke run"
    )
    args = parser.parse_args(argv)
    cells = sweep_serve_rules(
        points=args.points, seed_start=args.seed_start, quick=args.quick
    )
    print(format_report(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
