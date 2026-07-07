"""Ball Balance environment.

The agent controls a flat tray with 3 translational + 3 rotational DOFs and
must keep a ball from rolling off. The reward is +1 for every step the ball
stays above the tray; the episode terminates when the ball falls below z=0.

This is the simplest environment in the Courtside Dynamics curriculum and is
intended as a sanity check for the overall training pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import CourtsideMujocoEnv


class BallBalanceEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Balance a ball on a tray with 6 DOFs."""

    def __init__(self, episode_len: int = 750, **kwargs: Any) -> None:
        utils.EzPickle.__init__(self, episode_len=episode_len, **kwargs)
        CourtsideMujocoEnv.__init__(
            self,
            asset_path("ball_balance.xml"),
            episode_len=episode_len,
            obs_dim=18,
            **kwargs,
        )

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        obs = self._get_obs()
        terminated = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        truncated = self.step_number >= self.episode_len
        return obs, reward, terminated, truncated, {}

    def reset_model(self):
        self.step_number = 0
        self.set_state(*self._noisy_init_state())
        return self._get_obs()

    #: Human-readable labels matching each element of ``_get_obs``. The
    #: length equals ``observation_space.shape[0]``.
    observation_names: tuple[str, ...] = (
        "ball_x", "ball_y", "ball_z",
        "ball_vx", "ball_vy", "ball_vz",
        "rotate_x_qpos", "rotate_x_qvel",
        "rotate_y_qpos", "rotate_y_qvel",
        "rotate_z_qpos", "rotate_z_qvel",
        "slider_x_qpos", "slider_x_qvel",
        "slider_y_qpos", "slider_y_qvel",
        "slider_z_qpos", "slider_z_qvel",
    )

    def _get_obs(self) -> np.ndarray:
        ball = self.data.joint("ball")
        return np.concatenate(
            (
                np.asarray(ball.qpos[:3]),
                np.asarray(ball.qvel[:3]),
                self._joints_obs(
                    "rotate_x", "rotate_y", "rotate_z",
                    "slider_x", "slider_y", "slider_z",
                ),
            )
        )
