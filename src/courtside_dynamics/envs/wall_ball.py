"""Wall Ball environment.

A 4-DOF paddle (x/y/z slide + pitch hinge) must rally a ball against a
wall. The ball is served on reset with an initial velocity toward the
wall. The reward is intentionally narrow:

- The **first** wall contact (the serve) earns nothing. With a free
  serve bonus the policy gradient is flat — every random rollout would
  hit reward 1.0 and SAC's critic would have zero variance to fit.
- Subsequent wall contacts earn +1, but only after the paddle has
  touched the ball at least once. This forces the agent to learn the
  paddle-redirect-to-wall loop instead of free-riding on the serve.
- Each fresh paddle contact earns ``paddle_hit_bonus`` so the agent has
  a dense gradient long before it can close the full loop.

Episodes also terminate early when the ball goes "dead": if no new
paddle/wall rising-edge fires for ``stall_steps`` consecutive steps, the
remaining timesteps would be wasted compute on a ball stuck rolling on
the floor, so we cut the episode short.

Bugs fixed in earlier revisions (kept for reference):

- The ball is now given a non-zero initial velocity on reset, so the
  reward signal is reachable.
- The paddle has a ``paddle_slide_x`` DOF so it can actually move along
  the rally axis.
- Termination no longer triggers on ``obs[2] < 0``; episodes end when
  the ball leaves the valid play volume.
- The rising-edge check on the touch sensor uses a strict ``>`` so
  ``min_force=0`` no longer spams a reward every step.
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
        serve_speed: float = 10.0,
        serve_lob: float = 4.0,
        serve_speed_jitter: float = 1.0,
        paddle_hit_bonus: float = 0.25,
        stall_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            min_force=min_force,
            serve_speed=serve_speed,
            serve_lob=serve_lob,
            serve_speed_jitter=serve_speed_jitter,
            paddle_hit_bonus=paddle_hit_bonus,
            stall_steps=stall_steps,
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        self.paddle_hit_bonus = float(paddle_hit_bonus)
        self.stall_steps = int(stall_steps)

        # Runtime bookkeeping (reset in ``reset_model``).
        self.bounce_count = 0           # rewarded wall contacts (post-paddle)
        self.wall_contact_count = 0     # all wall contacts incl. the serve
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._steps_since_event = 0

        # Obs: ball pos(3) + ball vel(3) + paddle qpos/qvel(8) +
        # paddle_engaged flag(1) = 15. The flag tells the value function
        # whether the *next* wall contact is rewardable, which an MLP
        # policy can't infer from raw state alone.
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64
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
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        wall_touch = float(self.data.sensor("wall_touch").data[0])
        paddle_touch = float(self.data.sensor("paddle_touch").data[0])

        wall_edge = (
            wall_touch > self.min_force
            and self._prev_wall_touch <= self.min_force
        )
        paddle_edge = (
            paddle_touch > self.min_force
            and self._prev_paddle_touch <= self.min_force
        )
        self._prev_wall_touch = wall_touch
        self._prev_paddle_touch = paddle_touch

        reward = 0.0
        event_this_step = False

        if paddle_edge:
            self.paddle_hit_count += 1
            reward += self.paddle_hit_bonus
            event_this_step = True

        if wall_edge:
            self.wall_contact_count += 1
            # Gate: only reward wall contacts that came after a paddle
            # touch. The first wall contact (the serve) yields nothing.
            if self.paddle_hit_count >= 1:
                self.bounce_count += 1
                reward += 1.0
            event_this_step = True

        if event_this_step:
            self._steps_since_event = 0
        else:
            self._steps_since_event += 1

        obs = self._get_obs()
        ball_x, ball_y, ball_z = obs[0], obs[1], obs[2]
        ball_out_of_bounds = (
            ball_x < _BALL_MIN_X
            or ball_x > _BALL_MAX_X
            or ball_y < _BALL_MIN_Y
            or ball_y > _BALL_MAX_Y
            or ball_z < _BALL_MIN_Z
        )
        # Stall: cut the episode if the ball has gone dead. Only counts
        # *after* the serve has registered, otherwise a slow serve would
        # spuriously trip it.
        stalled = (
            self.wall_contact_count >= 1
            and self._steps_since_event >= self.stall_steps
        )
        terminated = bool(
            not np.isfinite(obs).all() or ball_out_of_bounds or stalled
        )
        truncated = self.step_number > self.episode_len

        info = {
            # Backward-compat keys consumed by callbacks/CSV writers.
            "sensor_data": wall_touch,
            "bounce_count": self.bounce_count,
            # New diagnostic keys.
            "wall_contact_count": self.wall_contact_count,
            "paddle_hit_count": self.paddle_hit_count,
            "paddle_touch": paddle_touch,
            "wall_touch": wall_touch,
            "stalled": bool(stalled),
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.step_number = 0
        self.bounce_count = 0
        self.wall_contact_count = 0
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._steps_since_event = 0

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

    observation_names: tuple[str, ...] = (
        "ball_x", "ball_y", "ball_z",
        "ball_vx", "ball_vy", "ball_vz",
        "paddle_slide_x_qpos", "paddle_slide_x_qvel",
        "paddle_slide_y_qpos", "paddle_slide_y_qvel",
        "paddle_slide_z_qpos", "paddle_slide_z_qvel",
        "paddle_pitch_qpos", "paddle_pitch_qvel",
        "paddle_engaged",
    )

    def _get_obs(self) -> np.ndarray:
        paddle_engaged = np.array([float(self.paddle_hit_count >= 1)])
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
                np.array(self.data.joint("paddle_pitch").qpos),
                np.array(self.data.joint("paddle_pitch").qvel),
                paddle_engaged,
            ),
            axis=0,
        )
