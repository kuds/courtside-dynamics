"""PaddleTennis probe P5: wall-ball champion transfer to the paddle court.

Design contract (docs/design_paddle_tennis.md §6 P5): drop the
wall-ball era's two champions (runs ``20260731_132322`` and
``20260801_144043``, both ``WallBallTrueBaseline``) onto the frozen
paddle court against the scripted oracle, and measure whether the
phase-P2 opponent pool can start warm. The champions were trained on
the 23-value wall-ball observation against a wall at x = +3.9; this
harness is the observation/action shim that lets them play side A of
``PaddleTennisEnv`` unmodified.

Two pre-declared shim variants (both anchored on the net-as-wall
identification; picking between them is part of what P5 measures):

- ``net``    -- rigid translation: the net plane maps to the wall
  face, distances preserved. The champion sees the whole paddle court
  compressed into the wall-ball front court (its deep band is empty).
- ``scaled`` -- affine x: net maps to the wall face AND the paddle
  court's baseline (6.5 m) maps to the wall-ball baseline (11.885 m
  from the face), x velocities scaled by the same factor. Depth
  distribution matches training at the cost of physical distortion.

The wall-ball rally-state tail has no exact analog; the mapping is
documented per index in :func:`wall_ball_observation` (the gate flag
derives from possession, the stall/curriculum tail is zero).

Instruments::

    # Shim fidelity, no checkpoint needed: the certified wall-ball
    # lead-charge oracle plays through the shim. If the shim is
    # faithful, this scripted player -- which sustains the true-
    # baseline reference band natively -- should rally here too.
    python tools/paddle_tennis_p5_transfer.py --stub-oracle

    # The real champions (run where the Drive artifacts are mounted,
    # e.g. Colab):
    python tools/paddle_tennis_p5_transfer.py \
        --model .../model/best_model.zip \
        --vec-normalize .../model/best_vec_normalize.pkl

Seeds default to the calibration block 5000-5099 (recorded in the P5
snapshot's ledger); the reserved held-out block 4100-4199 is never
touched here.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Callable

import numpy as np

from courtside_dynamics.envs._base import (
    invert_piecewise_target,
    piecewise_targets,
)
from courtside_dynamics.envs._paddle_court import (
    PADDLE_HOME_X,
    PADDLE_LOCAL_X_RANGE,
    scripted_lead_charge_opponent,
)
from courtside_dynamics.envs.paddle_tennis import (
    PADDLE_TENNIS_OBSERVATION_NAMES,
    PaddleTennisEnv,
)
from courtside_dynamics.training.ladder_certification import _oracle_action

#: Index of every paddle-court observation, by its frozen name.
_IDX = {
    name: index
    for index, name in enumerate(PADDLE_TENNIS_OBSERVATION_NAMES)
}

#: Wall-ball world geometry the champions were trained against
#: (WallBallTrueBaseline, 0.25.0): the wall FACE the ball rebounds
#: off, and the paddle head's zero-qpos origin.
WALL_FACE_X = 3.9
WB_BASELINE_X = -7.985
WB_PADDLE_ORIGIN = (-1.7, 0.0, 1.2)

#: The champions' action mapping (low, home, high) on the x axis; the
#: y/z mappings are the shared paddle calibration and pass through
#: unchanged (y: home 0 span 3; z: home 1.2, +2.0/-0.9).
WB_X_MAPPING = (-8.2, -5.4, 0.3)
#: The fence the champions TRAINED under (WallBallTrueBaseline env
#: kwargs). In training every decoded x target was clamped to this
#: interval before it reached the actuator, so any action >= +0.491
#: physically meant "paddle to -2.6". The decode must reproduce that
#: clamp: without it the (0.491, 1.0] action interval -- which a
#: saturating SAC policy lives in, and which the scripted stub never
#: emits -- would command up to 1.6 m in front of any position the
#: champions ever reached (adversarial-review finding, 2026-08-03).
WB_TRAINED_FENCE = (-8.2, -2.6)
#: The largest ball x the champions ever observed: the wall face
#: minus the ball radius. During opponent possession the paddle
#: court's far half would map beyond it (physically impossible in
#: wall-ball), so the rendered ball is held at the face instead --
#: the ball "waits at the wall" until the opponent's return sends it
#: back, which is exactly the rebound moment wall-ball trained on.
WB_BALL_MAX_X = 3.83
#: The certified true-baseline oracle configuration (recipes.py
#: ladder_certification: lead_charge 2.6 at the (-8.2, -2.6) fence).
WB_ORACLE_FENCE = (-8.2, -2.6)
WB_ORACLE_LEAD_CHARGE = 2.6

#: Default calibration seed block for this probe (see the P5 snapshot
#: ledger). Never the reserved held-out blocks.
P5_SEED_START = 5000
P5_EPISODES = 100


@dataclasses.dataclass(frozen=True, slots=True)
class ShimVariant:
    """An affine identification of the paddle court with the wall court.

    ``x_wb = WALL_FACE_X + scale * x_local`` for positions;
    x velocities scale by ``scale``; y and z map identically.
    """

    name: str
    scale: float

    def to_wall_x(self, x_local: float) -> float:
        return WALL_FACE_X + self.scale * x_local

    def to_local_x(self, x_wall: float) -> float:
        return (x_wall - WALL_FACE_X) / self.scale


SHIM_VARIANTS = {
    "net": ShimVariant("net", 1.0),
    "scaled": ShimVariant(
        "scaled", (WALL_FACE_X - WB_BASELINE_X) / 6.5
    ),
}


def wall_ball_observation(
    observation: np.ndarray, shim: ShimVariant
) -> np.ndarray:
    """Render the side-local paddle-court state as a wall-ball obs (23,).

    Index map (wall-ball name <- paddle-court source):

    - 0-2   ball x/y/z        <- ball position, x through the shim,
      then held at the wall face (``WB_BALL_MAX_X``) while the ball
      is on the opponent's half -- the far half of the paddle court
      does not exist in the champion's world, and an unclamped map
      would show the ball up to 6 m beyond the wall at scaled
      velocities (impossible-in-training states over every
      receive-preparation window; adversarial-review finding)
    - 3-5   ball vx/vy/vz     <- ball velocity, vx scaled
    - 6-11  paddle qpos/qvel  <- own paddle world pos/vel minus the
      wall-ball origin (-1.7, 0, 1.2), interleaved per axis
    - 12    hit-since-wall    <- ball on own side AND own side not the
      expected returner (own shot in flight toward the net; the
      net crossing plays the wall touch's gate-closing role)
    - 13    floor bounces     <- rules bounce_count (per-possession)
    - 14-16 paddle-to-ball    <- ball minus own paddle, dx scaled
    - 17-19 ball spin         <- unchanged (not meaningfully scalable)
    - 20-22 stall/advance/recovery tail <- 0 (no stall clock, no
      curriculum, no recovery resets on the paddle court)
    """
    o = observation
    ball_x = min(
        shim.to_wall_x(o[_IDX["ball_position_x"]]), WB_BALL_MAX_X
    )
    paddle_x = shim.to_wall_x(o[_IDX["own_paddle_position_x"]])
    return np.array(
        [
            ball_x,
            o[_IDX["ball_position_y"]],
            o[_IDX["ball_position_z"]],
            shim.scale * o[_IDX["ball_linear_velocity_x"]],
            o[_IDX["ball_linear_velocity_y"]],
            o[_IDX["ball_linear_velocity_z"]],
            paddle_x - WB_PADDLE_ORIGIN[0],
            shim.scale * o[_IDX["own_paddle_velocity_x"]],
            o[_IDX["own_paddle_position_y"]] - WB_PADDLE_ORIGIN[1],
            o[_IDX["own_paddle_velocity_y"]],
            o[_IDX["own_paddle_position_z"]] - WB_PADDLE_ORIGIN[2],
            o[_IDX["own_paddle_velocity_z"]],
            (
                1.0
                if o[_IDX["ball_side_is_own"]] >= 0.5
                and o[_IDX["expected_returner_is_own"]] < 0.5
                else 0.0
            ),
            o[_IDX["bounce_count"]],
            ball_x - paddle_x,
            o[_IDX["ball_minus_own_paddle_y"]],
            o[_IDX["ball_minus_own_paddle_z"]],
            o[_IDX["ball_angular_velocity_x"]],
            o[_IDX["ball_angular_velocity_y"]],
            o[_IDX["ball_angular_velocity_z"]],
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def paddle_court_action(
    action: np.ndarray, shim: ShimVariant
) -> np.ndarray:
    """Re-express a wall-ball action as a paddle-court local action.

    The x component decodes through the champion's own asymmetric
    mapping to a wall-ball world target, crosses the shim, clamps to
    the paddle-court workspace, and re-encodes through the court's
    mapping. y and z are the identical shared calibration on both
    courts and pass through unchanged.
    """
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    low, home, high = WB_X_MAPPING
    target_wb_x = float(
        piecewise_targets(
            np.array([action[0]]),
            np.array([low]),
            np.array([home]),
            np.array([high]),
        )[0]
    )
    # Training semantics: the true-baseline env clamped every decoded
    # target to the fence before the actuator saw it.
    target_wb_x = float(
        np.clip(target_wb_x, WB_TRAINED_FENCE[0], WB_TRAINED_FENCE[1])
    )
    target_local_x = float(
        np.clip(
            shim.to_local_x(target_wb_x),
            PADDLE_LOCAL_X_RANGE[0],
            PADDLE_LOCAL_X_RANGE[1],
        )
    )
    ax = invert_piecewise_target(
        target_local_x,
        PADDLE_LOCAL_X_RANGE[0],
        -PADDLE_HOME_X,
        PADDLE_LOCAL_X_RANGE[1],
    )
    return np.array([ax, action[1], action[2]], dtype=np.float64)


WallBallPolicy = Callable[[np.ndarray], np.ndarray]


def transfer_policy(
    policy: WallBallPolicy,
    shim: ShimVariant,
    *,
    yield_overlay: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a wall-ball-frame policy as a paddle-court side-A player.

    ``yield_overlay`` parks the player at the CHAMPION'S trained
    neutral -- champion action zero, i.e. wall-ball home -5.4 decoded
    through the shim -- whenever the ball is not its to hit. Wall
    ball had no yield concept (its serves always arrived FROM the
    wall side) and the first stub sweep measured the consequence:
    every serving point lost to ``wrong_hitter``, the transferred
    player diving through its own serve's flight path (P3's
    documented self-touch hazard). Parking at the champion's own
    neutral (not the paddle-court home) keeps the release-step
    self-state inside the champion's lifelong workspace AND clears
    the serve column deep (adversarial-review finding, 2026-08-03).

    Known limitation, acknowledged in the P5 snapshot: the overlay is
    turn-scoped (it also suppresses between-shot repositioning after
    every own hit), so overlay-off arms are always reported beside it.
    """
    park = paddle_court_action(np.zeros(3), shim)

    def act(observation: np.ndarray) -> np.ndarray:
        if (
            yield_overlay
            and observation[_IDX["expected_returner_is_own"]] < 0.5
        ):
            return park.copy()
        wall_obs = wall_ball_observation(observation, shim)
        return paddle_court_action(policy(wall_obs), shim)

    return act


def stub_oracle_policy() -> WallBallPolicy:
    """The certified true-baseline lead-charge oracle, wall-ball frame."""

    def act(wall_obs: np.ndarray) -> np.ndarray:
        return _oracle_action(
            wall_obs,
            fence=WB_ORACLE_FENCE,
            home=WB_X_MAPPING[1],
            mapping=(WB_X_MAPPING[0], WB_X_MAPPING[2]),
            run_up=None,
            charge_gap=None,
            lead_charge=WB_ORACLE_LEAD_CHARGE,
        )

    return act


def champion_policy(
    model_path: str, vec_normalize_path: str
) -> WallBallPolicy:
    """A saved SB3 SAC champion behind its frozen normalizer."""
    import pickle

    from stable_baselines3 import SAC

    model = SAC.load(model_path, device="cpu")
    with open(vec_normalize_path, "rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False

    def act(wall_obs: np.ndarray) -> np.ndarray:
        normalized = normalizer.normalize_obs(
            wall_obs[None, :].astype(np.float64)
        )[0]
        action, _ = model.predict(normalized, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return act


@dataclasses.dataclass(slots=True)
class TransferResult:
    """Aggregated P5 metrics for one player under one shim variant."""

    label: str
    episodes: int
    seed_start: int
    mean_crossings: float
    ge1_rate: float
    mean_returns_side_a: float
    mean_returns_side_b: float
    terminations: Counter

    def row(self) -> str:
        taxonomy = ", ".join(
            f"{name} {count}"
            for name, count in self.terminations.most_common(3)
        )
        return (
            f"| {self.label} | {self.mean_crossings:.2f} "
            f"| {self.ge1_rate:.0%} | {self.mean_returns_side_a:.2f} "
            f"| {self.mean_returns_side_b:.2f} | {taxonomy} |"
        )


def run_transfer(
    player: Callable[[np.ndarray], np.ndarray],
    *,
    label: str,
    episodes: int,
    seed_start: int,
) -> TransferResult:
    """Play the frozen task with ``player`` on side A, oracle on side B."""
    env = PaddleTennisEnv()
    crossings: list[int] = []
    returns_a: list[float] = []
    returns_b: list[float] = []
    terminations: Counter = Counter()
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            info: dict = {}
            while True:
                observation, _, terminated, truncated, info = env.step(
                    player(observation)
                )
                if terminated or truncated:
                    break
            crossings.append(int(info["crossings"]))
            returns_a.append(float(info["valid_return_count_a"]))
            returns_b.append(float(info["valid_return_count_b"]))
            terminations[info["termination_reason_name"]] += 1
    finally:
        env.close()
    values = np.asarray(crossings, dtype=np.float64)
    return TransferResult(
        label=label,
        episodes=episodes,
        seed_start=seed_start,
        mean_crossings=float(values.mean()),
        ge1_rate=float((values >= 1).mean()),
        mean_returns_side_a=float(np.mean(returns_a)),
        mean_returns_side_b=float(np.mean(returns_b)),
        terminations=terminations,
    )


def format_report(results: list[TransferResult]) -> str:
    header = results[0]
    return "\n".join(
        [
            "P5 champion-transfer probe "
            f"({header.episodes} episodes/row, seeds "
            f"{header.seed_start}-"
            f"{header.seed_start + header.episodes - 1}; "
            "side A = transferred player, side B = native oracle)",
            "",
            "| player | crossings | >=1 | returns A | returns B "
            "| top terminations |",
            "|---|---|---|---|---|---|",
            *(result.row() for result in results),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=P5_EPISODES
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=P5_SEED_START,
        help="first calibration seed (see the P5 snapshot's ledger)",
    )
    parser.add_argument(
        "--variant",
        choices=(*SHIM_VARIANTS, "both"),
        default="both",
    )
    parser.add_argument(
        "--stub-oracle",
        action="store_true",
        help=(
            "play the certified wall-ball lead-charge oracle through "
            "the shim (fidelity instrument; the default when no "
            "--model is given)"
        ),
    )
    parser.add_argument(
        "--model", help="SB3 SAC checkpoint (best_model.zip)"
    )
    parser.add_argument(
        "--vec-normalize",
        help="frozen normalizer (best_vec_normalize.pkl)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also run the native paddle-court oracle on the same seeds",
    )
    parser.add_argument(
        "--yield-overlay",
        choices=("on", "off", "both"),
        default="both",
        help=(
            "neutral-park the player when the ball is not its to hit "
            "(the serve-yield rule wall-ball never needed)"
        ),
    )
    args = parser.parse_args(argv)
    if args.model and not args.vec_normalize:
        parser.error("--model requires --vec-normalize")

    if args.model:
        policy = champion_policy(args.model, args.vec_normalize)
        player_name = f"champion:{args.model}"
    else:
        policy = stub_oracle_policy()
        player_name = "wall-ball oracle (lead_charge 2.6)"

    variants = (
        tuple(SHIM_VARIANTS)
        if args.variant == "both"
        else (args.variant,)
    )
    overlays = {
        "on": (True,),
        "off": (False,),
        "both": (False, True),
    }[args.yield_overlay]
    results = [
        run_transfer(
            transfer_policy(
                policy,
                SHIM_VARIANTS[variant],
                yield_overlay=yield_overlay,
            ),
            label=(
                f"{player_name} via {variant}"
                + (" + yield" if yield_overlay else "")
            ),
            episodes=args.episodes,
            seed_start=args.seed_start,
        )
        for variant in variants
        for yield_overlay in overlays
    ]
    if args.baseline:
        results.append(
            run_transfer(
                scripted_lead_charge_opponent,
                label="native paddle-court oracle",
                episodes=args.episodes,
                seed_start=args.seed_start,
            )
        )
    print(format_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
