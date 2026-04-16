"""Ball Bounce environment.

The agent controls a 6-DOF paddle and must keep juggling a ball upward on
the paddle's face. Reward is +1 on every rising edge of the paddle's touch
sensor (i.e. each fresh contact with the ball). The episode terminates if
the ball falls below the paddle.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path


class BallBounceEnv(MujocoEnv, utils.EzPickle):
    """Juggle a ball on a 6-DOF paddle."""

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
        episode_len: int = 1_000,
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
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            asset_path("ball_bounce.xml"),
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
        current_velocity_value = np.array(self.data.sensor("ball_velocity").data)
        current_accelerometer_value = np.array(
            self.data.sensor("ball_accelerometer").data
        )
        current_fromto_value = np.array(self.data.sensor("ball_to_paddle").data)

        # Bounce detection: rising edge on the touch sensor above ``min_force``.
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
        info = {
            "touch_sensor": current_touch_value,
            "ball_velocity": current_velocity_value,
            "ball_accelerometer": current_accelerometer_value,
            "ball_to_paddle": current_fromto_value,
        }
        return obs, reward, terminated, truncated, info

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
                np.array(self.data.joint("ball_freejoint").qpos[:3]),
                np.array(self.data.joint("ball_freejoint").qvel[:3]),
                np.array(self.data.joint("rotate_x").qpos),
                np.array(self.data.joint("rotate_x").qvel),
                np.array(self.data.joint("rotate_y").qpos),
                np.array(self.data.joint("rotate_y").qvel),
                np.array(self.data.joint("rotate_z").qpos),
                np.array(self.data.joint("rotate_z").qvel),
                np.array(self.data.joint("slider_x").qpos),
                np.array(self.data.joint("slider_x").qvel),
                np.array(self.data.joint("slider_y").qpos),
                np.array(self.data.joint("slider_y").qvel),
                np.array(self.data.joint("slider_z").qpos),
                np.array(self.data.joint("slider_z").qvel),
            ),
            axis=0,
        )
