"""Hand-coded reference policies used to validate env design.

These are *not* training baselines — they exist so unit tests and humans
can answer the question "is this env even solvable?" without running RL.
If a scripted policy with full state access can't beat random on an env,
the env's reward shaping or physics are broken and no amount of SAC will
fix that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from courtside_dynamics.envs.humanoid_tennis import (
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    HumanoidTennisCoopEnv,
)
from courtside_dynamics.envs.tennis_rules import CourtSide

#: Gravity used by the oracle's ballistic lead prediction. Matches
#: MuJoCo's default ``option.gravity`` z component.
_WALL_BALL_GRAVITY = 9.81


def ball_bounce_oracle_action(obs: np.ndarray) -> np.ndarray:
    """Track the ball and time powered top-face strikes in BallBounce.

    The action order is rotate x/y/z, slide z/y/x.  Small PD terms keep the
    paddle level, lateral PD tracks the ball, and the vertical motor alternates
    between resetting low and striking upward as a descending ball enters the
    hitting corridor.  This is a deterministic feasibility fixture, not a
    training baseline.
    """
    observation = np.asarray(obs)
    if observation.shape != (30,):
        raise ValueError("BallBounce observation must have shape (30,)")

    action = np.zeros(6, dtype=np.float32)
    for action_index, (qpos_index, qvel_index) in enumerate(
        ((6, 7), (8, 9), (10, 11))
    ):
        action[action_index] = np.clip(
            -observation[qpos_index] - 0.03 * observation[qvel_index],
            -1.0,
            1.0,
        )

    # Actuator order is z/y/x while observation order is x/y/z.
    action[4] = np.clip(
        2.0 * (observation[1] - observation[14])
        - 0.3 * observation[15],
        -1.0,
        1.0,
    )
    action[5] = np.clip(
        2.0 * (observation[0] - observation[12])
        - 0.3 * observation[13],
        -1.0,
        1.0,
    )
    ball_is_descending = observation[5] < 0.0
    ball_paddle_gap = observation[2] - observation[16]
    paddle_can_strike = observation[16] < 0.35
    action[3] = (
        1.0
        if ball_is_descending and ball_paddle_gap < 0.12 and paddle_can_strike
        else -1.0
    )
    return action


def wall_ball_oracle_action(obs: np.ndarray) -> np.ndarray:
    """Intercept-and-swing target controller for :class:`WallBallEnv`.

    The three normalized actions are absolute position targets for the
    paddle's x/y/z slides. The face has a fixed 10-degree upward pitch
    and is anchored at world ``(-1.7, 0, 1.2)``, so its position is
    ``(-1.7 + slide_x_qpos, slide_y_qpos, 1.2 + slide_z_qpos)``.
    Inverting that transform gives the physical qpos targets below; the
    env's force-limited servo tracks them.

    Two behaviours, keyed on the ball's x velocity:

    - **Inbound** (ball flying toward the paddle): hold x near the home
      column while using a ballistic lead in y/z to park the face where
      the ball is *going to be*. Keeping x retracted preserves room to
      accelerate through every contact; chasing the incoming ball to
      the positive x limit produced one strong serve return followed by
      weak, static-face rebounds. A small early z lift gives the low
      post-wall returns some upward paddle velocity before the strike.
      Once aligned and close, swing toward the wall.
    - **Outbound**: recover to the home column and track the ball's
      y/z, waiting for the wall rebound.

    This is a feasibility oracle, not a competitive baseline.
    """
    ball_x, ball_y, ball_z = obs[0], obs[1], obs[2]
    ball_vx, ball_vy, ball_vz = obs[3], obs[4], obs[5]
    rel_dx, rel_dy, rel_dz = obs[14], obs[15], obs[16]

    inbound = ball_vx < -0.1

    if inbound:
        # Ballistic lead to the home contact plane (roughly 5 cm in
        # front of the face centred at world x=-1.7).
        t_hit = max(0.0, (ball_x + 1.65) / max(0.1, -ball_vx))
        pred_y = ball_y + ball_vy * t_hit
        pred_z = (
            ball_z + ball_vz * t_hit - 0.5 * _WALL_BALL_GRAVITY * t_hit**2
        )
        if pred_z < 0.1:
            # The ball will bounce before arrival; the parabola above
            # is wrong past the bounce, so fall back to tracking its
            # current height (never below a reachable floor pickup).
            pred_z = max(ball_z, 0.15)
        target_x = 0.0
        target_y = float(np.clip(pred_y, -3.0, 3.0))
        target_z = float(np.clip(pred_z - 1.2, -0.9, 2.0))
        if 0.0 < rel_dx <= 1.3:
            target_z = float(np.clip(target_z + 0.2, -0.9, 2.0))
        aligned = abs(rel_dy) < 0.25 and abs(rel_dz) < 0.30
        if aligned and 0.0 < rel_dx <= 0.55:
            # Swing through the contact.
            target_x = 2.0
    else:
        target_x = 0.0
        target_y = float(np.clip(ball_y, -3.0, 3.0))
        target_z = float(np.clip(ball_z - 1.2, -0.9, 2.0))

    targets = np.array([target_x, target_y, target_z], dtype=np.float64)
    # Invert WallBallEnv's piecewise mapping around the qpos=0 home
    # target. Negative and positive spans differ for x/z.
    negative_spans = np.array([3.0, 3.0, 0.9])
    positive_spans = np.array([2.0, 3.0, 2.0])
    spans = np.where(targets >= 0.0, positive_spans, negative_spans)
    return np.clip(targets / spans, -1.0, 1.0).astype(np.float32)


# The receiving G1's right shoulder is pre-positioned to -0.2 radians by the
# reset-only oracle fixture.  Under HumanoidTennisCoopEnv's piecewise action
# mapping, this normalized value keeps that raw target fixed: stand=0.2,
# lower=-3.0892, so (-0.2 - 0.2) / (0.2 - -3.0892) = -0.12161012.
_ORACLE_SHOULDER_ACTION = np.float32(-0.12161012)

# Stage 0 begins at the standing shoulder targets (0.2, -0.2 rad) and
# physically drives them to (0.0, -1.8 rad). These normalized values are
# derived from the G1 actuator ctrlranges; neutral action misses the feed.
_STAGE0_SHOULDER_PITCH_ACTION = np.float32(-0.0608051)
_STAGE0_SHOULDER_ROLL_ACTION = np.float32(-0.7799171)
_STAGE1_SWING_SHOULDER_PITCH_ACTION = np.float32(0.4857513)
_STAGE1_SWING_CONTROL_STEP = 57


def humanoid_tennis_stage0_action(obs: np.ndarray) -> np.ndarray:
    """Return the mirrored non-neutral physical Stage 0 intercept action."""
    observation = np.asarray(obs)
    expected_shape = (HUMANOID_TENNIS_OBSERVATION_LAYOUT.total_size,)
    if observation.shape != expected_shape:
        raise ValueError(
            f"humanoid-tennis observation must have shape {expected_shape}"
        )
    action = np.zeros(58, dtype=np.float32)
    rally_start = HUMANOID_TENNIS_OBSERVATION_LAYOUT.rally_state.start
    assert rally_start is not None
    serving_side_a = observation[rally_start + 4] >= 0.5
    shoulder_start = 51 if serving_side_a else 22
    action[shoulder_start] = _STAGE0_SHOULDER_PITCH_ACTION
    action[shoulder_start + 1] = _STAGE0_SHOULDER_ROLL_ACTION
    return action


def humanoid_tennis_stage1_action(
    obs: np.ndarray,
    *,
    control_step: int,
) -> np.ndarray:
    """Return the two-phase deterministic physical Stage 1 swing.

    The shoulder prepares for 57 control calls, then swings forward. The
    timing is a feasibility fixture, not a robust target-return controller.
    """
    if isinstance(control_step, bool) or not isinstance(control_step, int):
        raise TypeError("control_step must be an integer")
    if control_step < 0:
        raise ValueError("control_step must be non-negative")
    action = humanoid_tennis_stage0_action(obs)
    if control_step >= _STAGE1_SWING_CONTROL_STEP:
        active = np.flatnonzero(action)
        if active.size != 2:
            raise RuntimeError("Stage 1 oracle expected two shoulder controls")
        action[int(active[0])] = _STAGE1_SWING_SHOULDER_PITCH_ACTION
    return action


def humanoid_tennis_oracle_reset_options(
    serving_side: CourtSide | int | str = CourtSide.A,
) -> dict[str, Any]:
    """Return a mirrored reset-only physical-return fixture.

    The ball begins near the serving baseline and the receiving racket begins
    in a reachable static pose.  Once :meth:`HumanoidTennisCoopEnv.step`
    starts, the harness writes no qpos/qvel or body state: the racket strike,
    rebound, net crossing, and landing are all MuJoCo contacts/dynamics.
    """
    side = HumanoidTennisCoopEnv._coerce_side(serving_side)
    if side is CourtSide.A:
        return {
            "serve_side": "a",
            "ball_position": (-11.35, -1.5, 1.3),
            "ball_velocity": (50.0, 5.05, 1.43),
            "joint_positions": {
                "player_b_right_shoulder_pitch_joint": -0.2,
            },
        }
    return {
        "serve_side": "b",
        "ball_position": (11.35, 1.5, 1.3),
        "ball_velocity": (-50.0, -5.05, 1.43),
        "joint_positions": {
            "player_a_right_shoulder_pitch_joint": -0.2,
        },
    }


def humanoid_tennis_oracle_action(obs: np.ndarray) -> np.ndarray:
    """Hold the mirrored physical-return fixture using the central action.

    This is a feasibility oracle, not a learned-tennis baseline.  It uses the
    observed serving-side one-hot to hold the opposing G1's pre-positioned
    right shoulder while all other controls remain at their standing targets.
    """
    observation = np.asarray(obs)
    expected_shape = (HUMANOID_TENNIS_OBSERVATION_LAYOUT.total_size,)
    if observation.shape != expected_shape:
        raise ValueError(
            f"humanoid-tennis observation must have shape {expected_shape}"
        )
    action = np.zeros(58, dtype=np.float32)
    rally_start = HUMANOID_TENNIS_OBSERVATION_LAYOUT.rally_state.start
    assert rally_start is not None
    serving_side_a = observation[rally_start + 4] >= 0.5
    action[51 if serving_side_a else 22] = _ORACLE_SHOULDER_ACTION
    return action


@dataclass(frozen=True, slots=True)
class HumanoidTennisStage0Result:
    """Outcome of the constrained physical intercept oracle."""

    serving_side: CourtSide
    steps: int
    total_reward: float
    stage_success: bool
    terminated: bool
    truncated: bool
    event_kinds: tuple[str, ...]
    final_info: dict[str, Any]


def run_humanoid_tennis_stage0_oracle(
    env: Any,
    *,
    serving_side: CourtSide | int | str = CourtSide.A,
    seed: int = 0,
    max_steps: int = 150,
) -> HumanoidTennisStage0Result:
    """Drive the real wrist racket through the deterministic Stage 0 feed."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    side = HumanoidTennisCoopEnv._coerce_side(serving_side)
    observation, info = env.reset(
        seed=seed,
        options={"serve_side": side.label},
    )
    if info.get("curriculum_stage") != 0:
        raise ValueError("Stage 0 oracle requires curriculum stage 0")
    total_reward = 0.0
    events: list[str] = []
    terminated = False
    truncated = False
    steps = 0
    for loop_step in range(1, max_steps + 1):
        steps = loop_step
        action = humanoid_tennis_stage0_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        events.extend(info["step_event_kinds"])
        if terminated or truncated:
            break
    return HumanoidTennisStage0Result(
        serving_side=side,
        steps=steps,
        total_reward=total_reward,
        stage_success=bool(info["stage_success"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        event_kinds=tuple(events),
        final_info=dict(info),
    )


@dataclass(frozen=True, slots=True)
class HumanoidTennisStage1Result:
    """Outcome of the timed physical anchored-return feasibility oracle."""

    serving_side: CourtSide
    steps: int
    total_reward: float
    stage_success: bool
    rally_count: int
    terminated: bool
    truncated: bool
    event_kinds: tuple[str, ...]
    final_info: dict[str, Any]


def run_humanoid_tennis_stage1_oracle(
    env: Any,
    *,
    serving_side: CourtSide | int | str = CourtSide.A,
    seed: int = 0,
    max_steps: int = 300,
) -> HumanoidTennisStage1Result:
    """Run one timed physical target return without simulation-state writes."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    side = HumanoidTennisCoopEnv._coerce_side(serving_side)
    observation, info = env.reset(
        seed=seed,
        options={"serve_side": side.label},
    )
    if info.get("curriculum_stage") != 1:
        raise ValueError("Stage 1 oracle requires curriculum stage 1")
    total_reward = 0.0
    events: list[str] = []
    terminated = False
    truncated = False
    steps = 0
    for control_step in range(max_steps):
        steps = control_step + 1
        action = humanoid_tennis_stage1_action(
            observation,
            control_step=control_step,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        events.extend(info["step_event_kinds"])
        if terminated or truncated:
            break
    return HumanoidTennisStage1Result(
        serving_side=side,
        steps=steps,
        total_reward=total_reward,
        stage_success=bool(info["stage_success"]),
        rally_count=int(info["rally_count"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        event_kinds=tuple(events),
        final_info=dict(info),
    )


@dataclass(frozen=True, slots=True)
class HumanoidTennisOracleResult:
    """Compact outcome returned by :func:`run_humanoid_tennis_oracle`."""

    serving_side: CourtSide
    steps: int
    total_reward: float
    rally_count: int
    terminated: bool
    truncated: bool
    event_kinds: tuple[str, ...]
    final_info: dict[str, Any]


def run_humanoid_tennis_oracle(
    env: Any,
    *,
    serving_side: CourtSide | int | str = CourtSide.A,
    seed: int = 0,
    max_steps: int = 300,
) -> HumanoidTennisOracleResult:
    """Run one deterministic physical legal return without mid-rally writes."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    side = HumanoidTennisCoopEnv._coerce_side(serving_side)
    observation, info = env.reset(
        seed=seed,
        options=humanoid_tennis_oracle_reset_options(side),
    )
    total_reward = 0.0
    events: list[str] = []
    terminated = False
    truncated = False
    steps = 0
    for loop_step in range(1, max_steps + 1):
        steps = loop_step
        action = humanoid_tennis_oracle_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        events.extend(info["step_event_kinds"])
        if info["rally_count"] >= 1 or terminated or truncated:
            break
    return HumanoidTennisOracleResult(
        serving_side=side,
        steps=steps,
        total_reward=total_reward,
        rally_count=int(info["rally_count"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        event_kinds=tuple(events),
        final_info=dict(info),
    )


__all__ = [
    "HumanoidTennisOracleResult",
    "HumanoidTennisStage0Result",
    "HumanoidTennisStage1Result",
    "ball_bounce_oracle_action",
    "humanoid_tennis_oracle_action",
    "humanoid_tennis_oracle_reset_options",
    "humanoid_tennis_stage0_action",
    "humanoid_tennis_stage1_action",
    "run_humanoid_tennis_oracle",
    "run_humanoid_tennis_stage0_oracle",
    "run_humanoid_tennis_stage1_oracle",
    "wall_ball_oracle_action",
]
