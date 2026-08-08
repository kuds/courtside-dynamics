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

The harness also owns the held-out certification of the frozen env
(``--certify``): the registered ``PaddleTennisEnv`` with both sides
scripted, on the reserved seed block, against floors pre-registered
from calibration data only (see ``certify_frozen_env``). The repo's
``ladder_certification`` machinery is WallBall-specific and is not
used here.

Usage::

    python tools/paddle_tennis_probes.py            # full P3 sweep
    python tools/paddle_tennis_probes.py --quick    # smoke (tests/CI)
    python tools/paddle_tennis_probes.py --certify  # held-out cert
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter

import numpy as np

from courtside_dynamics.envs._paddle_court import (
    NET_HEIGHT,
    PADDLE_COURT,
    PaddleCourtScene,
    PaddleCourtServe,
    lead_charge_local_action,
    scripted_ground_opponent,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import (
    COURT_EVENT_KINDS as _COURT_EVENT_KINDS,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyPhase,
    TerminationReason,
)

#: Default control-frame budget per point: 12 s of simulated time at
#: the 100 Hz control rate. A point that exhausts it shows up honestly
#: in the taxonomy as termination NONE (truncated), never silently.
MAX_CONTROL_STEPS = 1200


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

    ``active_receiver=False`` corner-parks both paddles outside the
    flight envelope for the whole point: the policy-independent
    serve-landing measurement (the wall-ball certification's
    parked-paddle instrument, moved clear of play). The same seed draws
    the same serve in both modes, so a cell can pair each landing
    verdict with its active outcome.
    """
    serving_side = CourtSide(serving_side)
    receiver = serving_side.opponent
    scene.serve(serve, serving_side=serving_side, rng=rng)

    serve_crossed = False
    serve_first_bounce_in = False
    first_bounce_depth: float | None = None
    receiver_returned = False
    # The parked pass measures ballistics, so the paddles must be OUT
    # of the flight envelope entirely: a home-column park is itself an
    # obstacle (the receiver's parked face passively volleys in-flight
    # serves, censoring perfectly legal draws -- the first committed
    # sweep's whole 'fault budget' at the primary band was this
    # artifact). Corner-park: deep, full-left, floor.
    corner_park = np.array([-1.0, -1.0, -1.0])
    steps_played = max_control_steps
    for step in range(max_control_steps):
        for side in (CourtSide.A, CourtSide.B):
            if not active_receiver:
                scene.apply_action(side, corner_park)
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
        # The P1 metric counts cumulative RETURN crossings (median 0 =
        # unreturned serve): total net crossings minus the feed's own.
        # (snapshot.shot_crossing_count is an end-state latch -- was the
        # FINAL shot across at termination -- not a cumulative count;
        # the first committed sweep misused it and under-reported the
        # rally tail.)
        crossings=max(
            0,
            int(after.net_crossing_count) - int(after.feed_crossed_net),
        ),
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
        depth
        for index in legal_indices
        if (depth := parked[index].first_bounce_depth) is not None
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


#: Held-out certification contract for the frozen env definition,
#: GROUND-RULES era (volley_rule="fault" default; the superseded
#: volley-era contract -- seeds 3100-3199, floors 2.6 / 0.85, PASS at
#: 3.22 -- is recorded in docs/paddle_tennis_env_20260802.md). The
#: seed block is the reserved one designated in the ground-rules
#: snapshot's ledger; every floor below was pre-registered from
#: CALIBRATION data only (the ground-rules probe's ground/fault cell
#: on seeds 5100-5199: mean crossings 7.04, per-episode std 3.96,
#: >=1-crossing rate 0.97), before any reserved seed was drawn.
CERTIFICATION_SEED_START = 4200
CERTIFICATION_EPISODES = 100
#: Probe mean (7.04, same instrument as this certification) minus two
#: combined sampling standard errors of two 100-episode means
#: (2 x sqrt(2) x 0.396), rounded down.
CERTIFICATION_MEAN_CROSSINGS_FLOOR = 5.9
#: Probe returned-serve rate (0.97) minus two combined binomial
#: sampling deviations (2 x ~0.024), rounded down.
CERTIFICATION_GE1_RATE_FLOOR = 0.90


@dataclasses.dataclass(slots=True)
class CertificationResult:
    """Held-out verdict for the frozen ``PaddleTennisEnv`` definition."""

    episodes: int
    seed_start: int
    mean_crossings: float
    std_crossings: float
    ge1_rate: float
    mean_valid_returns: float
    serve_side_a_fraction: float
    unsafe_terminations: int
    terminations: Counter
    passed: bool

    def report(self) -> str:
        taxonomy = ", ".join(
            f"{name} {count}"
            for name, count in self.terminations.most_common()
        )
        verdict = "PASS" if self.passed else "FAIL"
        return "\n".join(
            [
                "PaddleTennis held-out certification "
                f"(registered env, scripted pair, seeds "
                f"{self.seed_start}-{self.seed_start + self.episodes - 1})",
                f"  mean crossings   {self.mean_crossings:.2f} "
                f"(floor {CERTIFICATION_MEAN_CROSSINGS_FLOOR}; "
                f"std {self.std_crossings:.2f})",
                f"  >=1-crossing     {self.ge1_rate:.0%} "
                f"(floor {CERTIFICATION_GE1_RATE_FLOOR:.0%})",
                f"  valid returns    {self.mean_valid_returns:.2f} mean",
                f"  serve side A     {self.serve_side_a_fraction:.0%}",
                f"  unsafe/nonfinite {self.unsafe_terminations} "
                "(floor: exactly 0)",
                f"  taxonomy         {taxonomy}",
                f"  verdict          {verdict}",
            ]
        )


def certify_frozen_env(
    *,
    episodes: int = CERTIFICATION_EPISODES,
    seed_start: int = CERTIFICATION_SEED_START,
) -> CertificationResult:
    """Play the frozen task, both sides scripted, one seed per episode.

    Constructs the registered env with its frozen defaults -- the
    definition being certified is exactly the one training will see.
    Serve sides alternate across resets by the env's own contract, so
    the block splits 50/50 between policy-serving and receiving. The
    scripted player is the era's reference controller (the ground
    oracle under ground rules).
    """
    env = PaddleTennisEnv()
    crossings: list[int] = []
    valid_returns: list[float] = []
    serve_side_a = 0
    unsafe = 0
    terminations: Counter = Counter()
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, reset_info = env.reset(seed=seed)
            if reset_info["serve_side"] == CourtSide.A.label:
                serve_side_a += 1
            info: dict = {}
            while True:
                observation, _, terminated, truncated, info = env.step(
                    scripted_ground_opponent(observation)
                )
                if terminated or truncated:
                    break
            crossings.append(int(info["crossings"]))
            valid_returns.append(float(info["valid_return_count"]))
            terminations[info["termination_reason_name"]] += 1
            if info["term_nonfinite"]:
                unsafe += 1
    finally:
        env.close()

    values = np.asarray(crossings, dtype=np.float64)
    mean_crossings = float(values.mean())
    ge1_rate = float((values >= 1).mean())
    passed = (
        mean_crossings >= CERTIFICATION_MEAN_CROSSINGS_FLOOR
        and ge1_rate >= CERTIFICATION_GE1_RATE_FLOOR
        and unsafe == 0
    )
    return CertificationResult(
        episodes=episodes,
        seed_start=seed_start,
        mean_crossings=mean_crossings,
        std_crossings=float(values.std(ddof=1)) if episodes > 1 else 0.0,
        ge1_rate=ge1_rate,
        mean_valid_returns=float(np.mean(valid_returns)),
        serve_side_a_fraction=serve_side_a / episodes,
        unsafe_terminations=unsafe,
        terminations=terminations,
        passed=passed,
    )


def format_report(cells: list[CellResult]) -> str:
    lines = [
        "P3 serve-rules sweep "
        f"(court half-length {PADDLE_COURT.half_length}, "
        f"net {NET_HEIGHT}, frozen lead_charge receiver)",
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
    parser.add_argument(
        "--certify",
        action="store_true",
        help=(
            "held-out certification of the frozen env on the reserved "
            "seed block (burns it; run once per definition freeze)"
        ),
    )
    parser.add_argument(
        "--certify-episodes",
        type=int,
        default=CERTIFICATION_EPISODES,
        help="episodes for --certify",
    )
    parser.add_argument(
        "--certify-seed-start",
        type=int,
        default=CERTIFICATION_SEED_START,
        help="first seed for --certify (default: the reserved block)",
    )
    args = parser.parse_args(argv)
    if args.certify:
        result = certify_frozen_env(
            episodes=args.certify_episodes,
            seed_start=args.certify_seed_start,
        )
        print(result.report())
        return 0 if result.passed else 1
    cells = sweep_serve_rules(
        points=args.points, seed_start=args.seed_start, quick=args.quick
    )
    print(format_report(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
