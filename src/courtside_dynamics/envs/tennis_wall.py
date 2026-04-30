"""Tennis Wall environment.

A 5-DOF racket (x/y/z slide + yaw + pitch) must rally a ball against a
wall.  Unlike :class:`WallBallEnv`, which gives a flat +1 on every wall
contact, ``TennisWallEnv`` uses a **state-machine reward** that shapes
behaviour through the full rally cycle:

Phase 0 – *APPROACH_PADDLE*
    Ball is heading toward the racket after a serve or wall return.
    A potential-based shaping reward encourages the agent to close the gap
    between the ball and the racket face.

Phase 1 – *APPROACH_WALL*
    The paddle has just struck the ball.  The agent earns a paddle-hit
    bonus and the shaping target switches to the wall.

Transitions
    * paddle contact (rising edge) during Phase 0 → Phase 1 + bonus
    * wall contact (rising edge) during Phase 1 → Phase 0 + bonus + rally++

The rally counter tallies complete paddle→wall round-trips; achieving a
high rally count is the ultimate goal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path

# Phase constants (also used as observation flags).
_PHASE_APPROACH_PADDLE = 0
_PHASE_APPROACH_WALL = 1

# Play-volume limits.  The ball must stay inside these bounds or the
# episode terminates.
_BALL_MIN_X = -6.0
_BALL_MAX_X = 5.0
_BALL_MIN_Y = -5.5
_BALL_MAX_Y = 5.5
_BALL_MIN_Z = -0.5


class TennisWallEnv(MujocoEnv, utils.EzPickle):
    """Rally a ball against a wall with a 5-DOF racket and shaped reward."""

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
        serve_speed: float = 6.0,
        serve_lob: float = 2.0,
        serve_speed_jitter: float = 1.0,
        shaping_scale: float = 1.0,
        paddle_hit_bonus: float = 2.0,
        wall_hit_bonus: float = 3.0,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            min_force=min_force,
            serve_speed=serve_speed,
            serve_lob=serve_lob,
            serve_speed_jitter=serve_speed_jitter,
            shaping_scale=shaping_scale,
            paddle_hit_bonus=paddle_hit_bonus,
            wall_hit_bonus=wall_hit_bonus,
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        self.shaping_scale = float(shaping_scale)
        self.paddle_hit_bonus = float(paddle_hit_bonus)
        self.wall_hit_bonus = float(wall_hit_bonus)

        # Runtime bookkeeping (reset in ``reset_model``).
        self.phase = _PHASE_APPROACH_PADDLE
        self.rally_count = 0
        self.paddle_hit_count = 0
        self.wall_hit_count = 0
        self._prev_paddle_touch = 0.0
        self._prev_wall_touch = 0.0
        self._prev_dist: float | None = None

        # Observation: ball pos(3) + vel(3) + paddle joints qpos(5) +
        # qvel(5) + phase one-hot(2) = 18.
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            asset_path("tennis_wall.xml"),
            5,
            observation_space=observation_space,
            **kwargs,
        )
        self.step_number = 0
        self.episode_len = episode_len

        # Cache DOF offsets.
        self._ball_dofadr = int(self.model.joint("ball_x").dofadr[0])

        # Wall x-position (centre of the wall geom) for shaping target.
        self._wall_x = float(self.data.body("wall").xpos[0])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        # -- sensor readings --
        paddle_touch = float(self.data.sensor("paddle_touch").data[0])
        wall_touch = float(self.data.sensor("wall_touch").data[0])

        # -- rising-edge detection --
        paddle_edge = (
            paddle_touch > self.min_force
            and self._prev_paddle_touch <= self.min_force
        )
        wall_edge = (
            wall_touch > self.min_force
            and self._prev_wall_touch <= self.min_force
        )
        self._prev_paddle_touch = paddle_touch
        self._prev_wall_touch = wall_touch

        # -- current ball position --
        ball_pos = np.array(self.data.joint("ball_x").qpos[:3])

        # -- shaping reward: encourage ball to approach current target --
        reward = 0.0
        if self.phase == _PHASE_APPROACH_PADDLE:
            target = np.array(self.data.body("paddle_base").xpos)
        else:
            target = np.array([self._wall_x, ball_pos[1], ball_pos[2]])

        dist = float(np.linalg.norm(ball_pos - target))
        if self._prev_dist is not None:
            reward += self.shaping_scale * (self._prev_dist - dist)
        self._prev_dist = dist

        # -- state-machine transitions --
        if self.phase == _PHASE_APPROACH_PADDLE and paddle_edge:
            reward += self.paddle_hit_bonus
            self.paddle_hit_count += 1
            self.phase = _PHASE_APPROACH_WALL
            self._prev_dist = None  # reset shaping baseline

        elif self.phase == _PHASE_APPROACH_WALL and wall_edge:
            reward += self.wall_hit_bonus
            self.wall_hit_count += 1
            self.rally_count += 1
            self.phase = _PHASE_APPROACH_PADDLE
            self._prev_dist = None

        # -- observations & termination --
        obs = self._get_obs()

        bx, by, bz = ball_pos[0], ball_pos[1], ball_pos[2]
        ball_out = (
            bx < _BALL_MIN_X
            or bx > _BALL_MAX_X
            or by < _BALL_MIN_Y
            or by > _BALL_MAX_Y
            or bz < _BALL_MIN_Z
        )
        terminated = bool(not np.isfinite(obs).all() or ball_out)
        truncated = self.step_number > self.episode_len

        info = {
            "phase": self.phase,
            "rally_count": self.rally_count,
            "paddle_hit_count": self.paddle_hit_count,
            "wall_hit_count": self.wall_hit_count,
            "paddle_touch": paddle_touch,
            "wall_touch": wall_touch,
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.step_number = 0
        self.phase = _PHASE_APPROACH_PADDLE
        self.rally_count = 0
        self.paddle_hit_count = 0
        self.wall_hit_count = 0
        self._prev_paddle_touch = 0.0
        self._prev_wall_touch = 0.0
        self._prev_dist = None

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )

        # Serve: ball moves from near the wall toward the paddle (-x).
        vx = -(
            self.serve_speed
            + self.np_random.uniform(
                -self.serve_speed_jitter, self.serve_speed_jitter
            )
        )
        vy = self.np_random.uniform(-0.5, 0.5)
        vz = self.serve_lob
        qvel[self._ball_dofadr + 0] = vx
        qvel[self._ball_dofadr + 1] = vy
        qvel[self._ball_dofadr + 2] = vz

        self.set_state(qpos, qvel)
        return self._get_obs()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    observation_names: tuple[str, ...] = (
        "ball_x", "ball_y", "ball_z",
        "ball_vx", "ball_vy", "ball_vz",
        "paddle_slide_x_qpos", "paddle_slide_x_qvel",
        "paddle_slide_y_qpos", "paddle_slide_y_qvel",
        "paddle_slide_z_qpos", "paddle_slide_z_qvel",
        "paddle_yaw_qpos", "paddle_yaw_qvel",
        "paddle_pitch_qpos", "paddle_pitch_qvel",
        "phase_approach_paddle", "phase_approach_wall",
    )

    def _get_obs(self) -> np.ndarray:
        phase_onehot = np.array(
            [
                float(self.phase == _PHASE_APPROACH_PADDLE),
                float(self.phase == _PHASE_APPROACH_WALL),
            ]
        )
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
                np.array(self.data.joint("paddle_yaw").qpos),
                np.array(self.data.joint("paddle_yaw").qvel),
                np.array(self.data.joint("paddle_pitch").qpos),
                np.array(self.data.joint("paddle_pitch").qvel),
                phase_onehot,
            ),
            axis=0,
        )
