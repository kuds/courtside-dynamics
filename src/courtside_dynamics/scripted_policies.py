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


def wall_ball_oracle_action(obs: np.ndarray) -> np.ndarray:
    """PD-controlled paddle tracker for :class:`WallBallEnv`.

    The 5 actions correspond to ``paddle_slide_x``, ``paddle_slide_y``,
    ``paddle_slide_z``, ``paddle_yaw``, and ``paddle_pitch`` motors with
    ``ctrlrange=[-1, 1]``. The slide qpos values are offsets from the
    paddle_base anchor at world ``(-2, 0, 1.2)``; with yaw=pitch=0 the
    paddle_head face sits ``(0.3, 0, 0)`` ahead of the base, so the
    face's world position is ``(-1.7 + slide_x_qpos, slide_y_qpos,
    1.2 + slide_z_qpos)``. Inverting that gives the qpos targets below.

    The paddle face is positioned slightly past the ball on the wall
    side so that on the inbound trip the ball flies into the face,
    instead of the paddle chasing the ball from behind.
    """
    ball_x = obs[0]
    ball_y = obs[1]
    ball_z = obs[2]
    paddle_x_qpos, paddle_x_qvel = obs[6], obs[7]
    paddle_y_qpos, paddle_y_qvel = obs[8], obs[9]
    paddle_z_qpos, paddle_z_qvel = obs[10], obs[11]
    paddle_yaw_qpos, paddle_yaw_qvel = obs[12], obs[13]
    paddle_pitch_qpos, paddle_pitch_qvel = obs[14], obs[15]

    # Place the face just past the ball (toward the wall, +x). The
    # ball arrives moving in -x, so face_x slightly larger than ball_x
    # is wrong — it's slightly *less* than ball_x that puts the face
    # in the ball's path. Use ball_x - 0.05 in world coords.
    desired_face_x = ball_x - 0.05
    target_x = float(np.clip(desired_face_x + 1.7, -3.0, 2.0))
    target_y = float(np.clip(ball_y, -3.0, 3.0))
    target_z = float(np.clip(ball_z - 1.2, -0.9, 2.0))
    target_yaw = 0.0
    target_pitch = 0.0

    kp, kd = 8.0, 1.0
    raw = np.array(
        [
            kp * (target_x - paddle_x_qpos) - kd * paddle_x_qvel,
            kp * (target_y - paddle_y_qpos) - kd * paddle_y_qvel,
            kp * (target_z - paddle_z_qpos) - kd * paddle_z_qvel,
            kp * (target_yaw - paddle_yaw_qpos) - kd * paddle_yaw_qvel,
            kp * (target_pitch - paddle_pitch_qpos) - kd * paddle_pitch_qvel,
        ],
        dtype=np.float32,
    )
    return np.clip(raw, -1.0, 1.0)


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
    "humanoid_tennis_oracle_action",
    "humanoid_tennis_oracle_reset_options",
    "humanoid_tennis_stage0_action",
    "humanoid_tennis_stage1_action",
    "run_humanoid_tennis_oracle",
    "run_humanoid_tennis_stage0_oracle",
    "run_humanoid_tennis_stage1_oracle",
    "wall_ball_oracle_action",
]
