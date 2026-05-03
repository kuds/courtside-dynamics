"""Wall Ball environment.

A 4-DOF paddle (x/y/z slide + pitch hinge) must rally a ball against a
wall. The ball is served on reset with an initial velocity toward the wall;
the agent earns +1 each time the ball makes a fresh contact with the wall,
plus shaping rewards that encourage tracking and returning the ball.

Reward design
-------------

The base reward is +1 per fresh wall contact (rising edge on the wall
touch sensor). On its own that signal is too sparse for SAC to bootstrap
from random play, because the only "free" wall contact comes from the
serve and the agent must discover the full track-the-ball-and-swing
sequence before any further reward fires. Three shaping terms address
that:

* ``paddle_hit_bonus`` — one-shot bonus the first time the paddle face
  contacts the ball during the *return* phase (i.e. while the ball is
  travelling back toward the paddle after a wall bounce).
* ``track_shaping_scale`` — potential-based shaping that rewards
  reductions in the paddle→ball distance while the ball is heading back
  toward the paddle. Telescopes to zero over an episode so it does not
  bias optimal-policy returns.
* ``out_of_bounds_penalty`` — flat negative reward when the ball leaves
  the play volume, so the agent is incentivised to keep the rally alive
  rather than passively waiting for the episode to end.

``info`` keys
-------------

Beyond ``sensor_data`` and ``bounce_count``, the env now emits
``paddle_touch``, ``paddle_hit_count``, ``wall_touch``,
``wall_contact_count``, and ``stalled`` so external dashboards can plot
return rate and rally health alongside the bare reward curve. ``stalled``
is a 0/1 flag that flips to 1 when the agent hasn't produced any fresh
wall or paddle contact for ``stall_steps`` consecutive sim steps; it
exists purely for diagnostics (it does not terminate the episode).

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
        track_shaping_scale: float = 0.5,
        paddle_hit_bonus: float = 1.0,
        out_of_bounds_penalty: float = 1.0,
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
            track_shaping_scale=track_shaping_scale,
            paddle_hit_bonus=paddle_hit_bonus,
            out_of_bounds_penalty=out_of_bounds_penalty,
            stall_steps=stall_steps,
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        self.track_shaping_scale = float(track_shaping_scale)
        self.paddle_hit_bonus = float(paddle_hit_bonus)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.stall_steps = int(stall_steps)

        self.bounce_count = 0
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        # Tracks whether the ball is currently heading back toward the
        # paddle (i.e. has bounced off the wall and not yet been
        # returned). Shaping only fires in this state so the agent isn't
        # rewarded for crowding the wall.
        self._returning = False
        self._prev_paddle_to_ball: float | None = None
        # Cumulative tracking shaping awarded since the last wall bounce.
        # Used for PBRS terminal correction: if the episode ends before the
        # agent returns the ball, the accumulated shaping is clawed back so
        # a no-op policy cannot earn net-positive shaping from the ball
        # naturally drifting toward the stationary paddle after a bounce.
        self._return_shaping_total: float = 0.0
        self._steps_since_event = 0

        # Observation: ball xyz(3) + ball xyz vel(3) + paddle joints
        # qpos(4) + qvel(4) + paddle_head→ball relative xyz(3) = 17.
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
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

        wall_touch = float(self.data.sensor("touch_sensor").data[0])
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

        if wall_edge:
            self.bounce_count += 1
            reward += 1.0
            # Wall hit -> ball will rebound toward the paddle. Open a
            # shaping window and reset the tracking baseline.
            self._returning = True
            self._prev_paddle_to_ball = None
            self._return_shaping_total = 0.0
            self._steps_since_event = 0

        if paddle_edge:
            self.paddle_hit_count += 1
            if self._returning:
                reward += self.paddle_hit_bonus
            self._returning = False
            self._prev_paddle_to_ball = None
            self._return_shaping_total = 0.0
            self._steps_since_event = 0

        # Potential-based tracking shaping: reward reductions in the
        # paddle→ball distance while a return is in progress.
        # Each delta is Φ(s) - Φ(s') = scale * (prev_dist - dist), which
        # telescopes to scale * (d_init - d_final) over the full window.
        # PBRS terminal correction below ensures a missed return earns zero
        # net shaping by clawing back the accumulated total on episode end.
        ball_pos = np.array(self.data.joint("ball_x").qpos[:3])
        paddle_head_pos = np.array(self.data.body("paddle_head").xpos)
        dist = float(np.linalg.norm(ball_pos - paddle_head_pos))
        if self._returning and self.track_shaping_scale > 0.0:
            if self._prev_paddle_to_ball is not None:
                delta = self.track_shaping_scale * (
                    self._prev_paddle_to_ball - dist
                )
                reward += delta
                self._return_shaping_total += delta
            self._prev_paddle_to_ball = dist

        if not (wall_edge or paddle_edge):
            self._steps_since_event += 1
        stalled = int(self._steps_since_event >= self.stall_steps)

        obs = self._get_obs(ball_pos, paddle_head_pos)
        ball_x, ball_y, ball_z = ball_pos[0], ball_pos[1], ball_pos[2]
        ball_out_of_bounds = (
            ball_x < _BALL_MIN_X
            or ball_x > _BALL_MAX_X
            or ball_y < _BALL_MIN_Y
            or ball_y > _BALL_MAX_Y
            or ball_z < _BALL_MIN_Z
        )
        if ball_out_of_bounds:
            reward -= self.out_of_bounds_penalty

        terminated = bool(not np.isfinite(obs).all() or ball_out_of_bounds)
        truncated = self.step_number > self.episode_len

        # PBRS terminal correction: if the episode ends while a return is
        # still in progress (the agent never hit the ball back), claw back
        # the accumulated tracking shaping so the no-op policy earns zero
        # net shaping from the ball passively drifting toward the paddle.
        if (terminated or truncated) and self._returning and abs(self._return_shaping_total) > 1e-9:
            reward -= self._return_shaping_total
            self._return_shaping_total = 0.0

        info = {
            "sensor_data": wall_touch,
            "wall_touch": wall_touch,
            "paddle_touch": paddle_touch,
            "bounce_count": self.bounce_count,
            "wall_contact_count": self.bounce_count,
            "paddle_hit_count": self.paddle_hit_count,
            "stalled": stalled,
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.step_number = 0
        self.bounce_count = 0
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._returning = False
        self._prev_paddle_to_ball = None
        self._return_shaping_total = 0.0
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
        "paddle_rotate_x_qpos", "paddle_rotate_x_qvel",
        "paddle_to_ball_dx", "paddle_to_ball_dy", "paddle_to_ball_dz",
    )

    def _get_obs(
        self,
        ball_pos: np.ndarray | None = None,
        paddle_head_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        if ball_pos is None:
            ball_pos = np.array(self.data.joint("ball_x").qpos[:3])
        if paddle_head_pos is None:
            paddle_head_pos = np.array(self.data.body("paddle_head").xpos)
        rel = ball_pos - paddle_head_pos
        return np.concatenate(
            (
                ball_pos,
                np.array(self.data.joint("ball_x").qvel[:3]),
                np.array(self.data.joint("paddle_slide_x").qpos),
                np.array(self.data.joint("paddle_slide_x").qvel),
                np.array(self.data.joint("paddle_slide_y").qpos),
                np.array(self.data.joint("paddle_slide_y").qvel),
                np.array(self.data.joint("paddle_slide_z").qpos),
                np.array(self.data.joint("paddle_slide_z").qvel),
                np.array(self.data.joint("paddle_rotate_x").qpos),
                np.array(self.data.joint("paddle_rotate_x").qvel),
                rel,
            ),
            axis=0,
        )
