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

    The 4 actions correspond to ``paddle_slide_x``, ``paddle_slide_y``,
    ``paddle_slide_z``, and ``paddle_rotate_x`` motors with
    ``ctrlrange=[-1, 1]``. The slide qpos values are offsets from the
    paddle_base anchor, *not* world coordinates, which is what makes
    the target math nontrivial. With the anchor at world ``(-2, 0, 1)``
    and the head face sitting ``(0, 0, 0.35)`` above the base body, the
    face's world position is ``(-2 + slide_x_qpos, slide_y_qpos,
    1.35 + slide_z_qpos)``. Inverting that gives the qpos targets
    below.

    The paddle is positioned slightly forward (toward the wall) of the
    ball so that on the return trip the ball flies into it, instead of
    chasing the ball from behind.
    """
    ball_x = obs[0]
    ball_y = obs[1]
    ball_z = obs[2]
    paddle_x_qpos, paddle_x_qvel = obs[6], obs[7]
    paddle_y_qpos, paddle_y_qvel = obs[8], obs[9]
    paddle_z_qpos, paddle_z_qvel = obs[10], obs[11]
    paddle_rot_qpos, paddle_rot_qvel = obs[12], obs[13]

    # Convert desired *world* paddle-face position to slide qpos targets.
    # paddle_base anchor: (-2, 0, 1); head face offset: (0, 0, 0.35).
    desired_face_x = ball_x + 0.3  # sit just past the ball on the wall side
    target_x = float(np.clip(desired_face_x + 2.0, -1.0, 4.0))
    target_y = float(np.clip(ball_y, -2.0, 2.0))
    target_z = float(np.clip(ball_z - 1.35, -0.9, 1.5))
    target_rot = 0.0

    kp, kd = 8.0, 1.0
    raw = np.array(
        [
            kp * (target_x - paddle_x_qpos) - kd * paddle_x_qvel,
            kp * (target_y - paddle_y_qpos) - kd * paddle_y_qvel,
            kp * (target_z - paddle_z_qpos) - kd * paddle_z_qvel,
            kp * (target_rot - paddle_rot_qpos) - kd * paddle_rot_qvel,
        ],
        dtype=np.float32,
    )
    return np.clip(raw, -1.0, 1.0)
