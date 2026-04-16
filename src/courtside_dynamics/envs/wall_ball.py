"""Wall Ball environment.

A 4-DOF paddle (x/y/z slide + pitch hinge) must rally a ball against a
wall. The ball is served on reset with an initial velocity toward the wall;
the agent earns +1 each time the ball makes a fresh contact with the wall,
which can happen either from the serve or from paddle returns. Longer
rallies therefore produce higher episode returns.

Bugs fixed in this version relative to the original notebook
implementation:

- The ball is now given a non-zero initial velocity on reset, so the
  reward signal is reachable. (Previously the ball was spawned at rest
  far from any agent-reachable geometry and no touch event could fire.)
- The paddle has a new ``paddle_slide_x`` DOF so it can actually move
  along the rally axis.
- Termination no longer triggers on ``obs[2] < 0`` (which would have
  ended episodes the first time the ball fell below the ball's starting
  z). Episodes now end when the ball leaves the valid play volume
  (behind the paddle or below the floor).
- The rising-edge check on the touch sensor uses a strict ``>`` on the
  current reading so ``min_force=0`` no longer spams a reward every step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path

# Cartesian bounds for "ball is still in play". Outside these, the episode
# terminates. The numbers are chosen generously so a bad shot doesn't
# instantly end training, but a ball that's fundamentally out of reach
# (behind the paddle by >1m, below the floor, etc.) does.
_BALL_MIN_X = -5.0
_BALL_MAX_X = 5.0
_BALL_MIN_Y = -4.5
_BALL_MAX_Y = 4.5
_BALL_MIN_Z = -0.5


class WallBallEnv(MujocoEnv, utils.EzPickle):
    """Rally a ball against a wall with a 4-DOF paddle."""

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
        serve_speed: float = 12.0,
        serve_lob: float = 2.0,
        serve_speed_jitter: float = 1.0,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            min_force=min_force,
            serve_speed=serve_speed,
            serve_lob=serve_lob,
            serve_speed_jitter=serve_speed_jitter,
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        self.bounce_count = 0
        self.previous_touch_value = 0.0

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64
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

        # Cache the ball's DOF offset so serve velocities don't depend on
        # a hard-coded index into qvel.
        self._ball_dofadr = int(self.model.joint("ball_x").dofadr[0])

    def step(self, a):
        reward = 0.0
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        current_touch_value = float(self.data.sensor("touch_sensor").data[0])
        # Rising edge: fire exactly once per fresh above-threshold contact.
        if (
            current_touch_value > self.min_force
            and self.previous_touch_value <= self.min_force
        ):
            self.bounce_count += 1
            reward = 1.0
        self.previous_touch_value = current_touch_value

        obs = self._get_obs()
        ball_x, ball_y, ball_z = obs[0], obs[1], obs[2]
        ball_out_of_bounds = (
            ball_x < _BALL_MIN_X
            or ball_x > _BALL_MAX_X
            or ball_y < _BALL_MIN_Y
            or ball_y > _BALL_MAX_Y
            or ball_z < _BALL_MIN_Z
        )
        terminated = bool(not np.isfinite(obs).all() or ball_out_of_bounds)
        truncated = self.step_number > self.episode_len

        info = {
            "sensor_data": current_touch_value,
            "bounce_count": self.bounce_count,
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

        # Serve: give the ball a forward velocity toward the wall (+x) with
        # a small upward lob so it clears any floor bounces en route. The
        # speed is randomized mildly so the agent can't memorize a single
        # serve trajectory.
        vx = self.serve_speed + self.np_random.uniform(
            -self.serve_speed_jitter, self.serve_speed_jitter
        )
        vy = self.np_random.uniform(-0.2, 0.2)
        vz = self.serve_lob
        qvel[self._ball_dofadr + 0] = vx
        qvel[self._ball_dofadr + 1] = vy
        qvel[self._ball_dofadr + 2] = vz

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            (
                np.array(self.data.joint("ball_x").qpos[:3]),
                np.array(self.data.joint("ball_x").qvel[:3]),
                np.array(self.data.joint("paddle_slide_x").qpos),
                np.array(self.data.joint("paddle_slide_x").qvel),
                np.array(self.data.joint("paddle_slide_y").qpos),
                np.array(self.data.joint("paddle_slide_y").qvel),
                np.array(self.data.joint("paddle_slide_z").qpos),
                np.array(self.data.joint("paddle_slide_z").qvel),
                np.array(self.data.joint("paddle_rotate_x").qpos),
                np.array(self.data.joint("paddle_rotate_x").qvel),
            ),
            axis=0,
        )
