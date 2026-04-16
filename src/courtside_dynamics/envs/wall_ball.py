"""Wall Ball environment.

The agent controls a paddle with two translational DOFs (y, z) and one
rotational DOF (pitch) and must hit a ball into a wall. Reward is +1 on
every rising edge of the wall touch sensor.

.. note::

    The XML/reward for this environment are preserved verbatim from the
    original notebook implementation. As currently written the ball is
    spawned without any initial velocity, so the agent never receives
    reward -- see the project review for a discussion of the fix. A
    subsequent commit will address this.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path


class WallBallEnv(MujocoEnv, utils.EzPickle):
    """Hit a ball into a wall with a 3-DOF paddle."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        episode_len: int = 750,
        min_force: float = 0.0,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self, episode_len=episode_len, min_force=min_force, **kwargs
        )

        self.min_force = min_force
        self.bounce_count = 0
        self.previous_touch_value = 0.0

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            asset_path("wall_ball.xml"),
            5,
            observation_space=observation_space,
            **kwargs,
        )
        self.step_number = 0
        self.episode_len = episode_len

    def step(self, a):
        reward = 0.0
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        current_touch_value = float(self.data.sensor("touch_sensor").data[0])
        if (
            current_touch_value >= self.min_force
            and self.previous_touch_value <= 0
        ):
            self.bounce_count += 1
            reward = 1.0
        self.previous_touch_value = current_touch_value

        obs = self._get_obs()
        terminated = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        truncated = self.step_number > self.episode_len
        return (
            obs,
            reward,
            terminated,
            truncated,
            {"sensor_data": current_touch_value},
        )

    def reset_model(self):
        self.step_number = 0
        self.bounce_count = 0
        self.previous_touch_value = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            (
                np.array(self.data.joint("ball_x").qpos[:3]),
                np.array(self.data.joint("ball_x").qvel[:3]),
                np.array(self.data.joint("paddle_slide_y").qpos),
                np.array(self.data.joint("paddle_slide_y").qvel),
                np.array(self.data.joint("paddle_slide_z").qpos),
                np.array(self.data.joint("paddle_slide_z").qvel),
                np.array(self.data.joint("paddle_rotate_x").qpos),
                np.array(self.data.joint("paddle_rotate_x").qvel),
            ),
            axis=0,
        )
