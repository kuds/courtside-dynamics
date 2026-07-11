"""Hand-coded reference policies used to validate env design.

These are *not* training baselines — they exist so unit tests and humans
can answer the question "is this env even solvable?" without running RL.
If a scripted policy with full state access can't beat random on an env,
the env's reward shaping or physics are broken and no amount of SAC will
fix that.
"""
from __future__ import annotations

import numpy as np


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
