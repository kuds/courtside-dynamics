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

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import CourtsideMujocoEnv


class BallBounceEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Juggle a ball on a 6-DOF paddle."""

    def __init__(
        self,
        episode_len: int = 1_000,
        min_force: float = 0.0,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self, episode_len=episode_len, min_force=min_force, **kwargs
        )

        self.min_force = float(min_force)
        self.bounce_count = 0
        self.previous_touch_value = 0.0

        CourtsideMujocoEnv.__init__(
            self,
            asset_path("ball_bounce.xml"),
            episode_len=episode_len,
            obs_dim=18,
            **kwargs,
        )

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

        # Bounce detection: rising edge on the touch sensor across the
        # ``min_force`` threshold (same hysteresis as WallBall). The strict
        # ``>`` matters at the default ``min_force=0``: with the old
        # ``current >= min_force and prev <= 0`` form, a step with *no
        # contact at all* (current == prev == 0) satisfied both clauses and
        # paid +1 every airborne step.
        if (
            current_touch_value > self.min_force
            and self.previous_touch_value <= self.min_force
        ):
            self.bounce_count += 1
            reward = 1.0
        self.previous_touch_value = current_touch_value

        obs = self._get_obs()
        terminated = bool(not np.isfinite(obs).all() or (obs[2] < 0))
        truncated = self.step_number >= self.episode_len
        info = {
            # ``bounce_count`` is the true task metric (contacts so far this
            # episode); exposing it lets InfoDictEvalCallback aggregate it
            # and ``success_key="bounce_count"`` compute a success rate.
            "bounce_count": self.bounce_count,
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
        self.set_state(*self._noisy_init_state())
        return self._get_obs()

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
        ball = self.data.joint("ball_freejoint")
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
