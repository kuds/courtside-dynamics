"""Wall Ball environment.

A 4-DOF paddle (x/y/z slide + pitch hinge) must rally a ball against a
wall. The ball is served on reset with an initial velocity toward the
wall. The reward is intentionally narrow:

- The **first** wall contact (the serve) earns nothing. With a free
  serve bonus the policy gradient is flat — every random rollout would
  hit reward 1.0 and SAC's critic would have zero variance to fit.
- Subsequent wall contacts earn +1, but **only** if a paddle hit
  happened since the previous wall contact. A "wall → wall" sequence
  with no paddle return between is a failed rally and pays nothing,
  forcing the agent to actually return the ball each cycle rather than
  free-riding on a serve that ricochets back to the wall on its own.
- Each fresh paddle contact earns ``paddle_hit_bonus`` so the agent has
  a dense gradient long before it can close the full loop.
- ``track_shaping_scale`` adds potential-based shaping that rewards
  reductions in the paddle→ball distance while a return is in progress
  (i.e. between a wall contact and the next paddle hit). Strict PBRS is
  ``F = γ·Φ(s') − Φ(s)``; this implementation uses ``Φ(s) − Φ(s')``
  (γ treated as 1) for simplicity. With SAC's typical γ=0.99 the policy
  bias is small but not exactly zero — keep ``track_shaping_scale``
  modest relative to the +1 wall and paddle-hit rewards. The shaping
  also has a terminal correction: any accumulated tracking shaping is
  clawed back if the return window ends without a paddle hit (either
  because the ball drifted out of the play volume or because a second
  wall bounce happened with no return between). That preserves the
  invariant that a no-op policy nets zero tracking shaping.
- ``out_of_bounds_penalty`` subtracts a flat amount on the terminating
  step when the ball leaves the play volume, so the agent is
  incentivised to keep the rally alive rather than letting the ball
  escape.

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
        track_shaping_scale: float = 0.5,
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
            paddle_hit_bonus=paddle_hit_bonus,
            track_shaping_scale=track_shaping_scale,
            out_of_bounds_penalty=out_of_bounds_penalty,
            stall_steps=stall_steps,
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        self.paddle_hit_bonus = float(paddle_hit_bonus)
        self.track_shaping_scale = float(track_shaping_scale)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.stall_steps = int(stall_steps)

        # Runtime bookkeeping (reset in ``reset_model``).
        self.bounce_count = 0           # rewarded wall contacts (post-paddle)
        self.wall_contact_count = 0     # all wall contacts incl. the serve
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._steps_since_event = 0
        # Strict per-cycle gate: True iff a paddle hit has happened
        # since the most recent wall contact (or, before the first wall
        # contact, never — so the serve earns nothing). Wall +1 fires
        # only when this is True at the moment of the wall edge.
        self._paddle_hit_since_last_wall = False
        # Tracking-shaping window: True between a wall hit and the next
        # paddle hit (the ball is heading back toward the paddle).
        self._returning = False
        self._prev_paddle_to_ball: float | None = None
        # Cumulative tracking shaping awarded since the current return
        # window opened. Clawed back if the window ends without a paddle
        # hit (either via a second wall bounce or via episode end), so
        # passive ball drift toward the paddle nets zero shaping.
        self._return_shaping_total: float = 0.0

        # Obs: ball pos(3) + ball vel(3) + paddle qpos/qvel(8) +
        # paddle_hit_since_last_wall flag(1) + paddle_head→ball
        # relative xyz(3) = 18. The flag exposes the wall-reward gate
        # state: True iff the next wall contact will pay +1, which an
        # MLP policy can't infer from raw state alone. The relative
        # xyz spares the policy from learning the joint→world mapping
        # by hand.
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
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
            self._paddle_hit_since_last_wall = True
            self._returning = False
            self._prev_paddle_to_ball = None
            self._return_shaping_total = 0.0
            event_this_step = True

        if wall_edge:
            self.wall_contact_count += 1
            # If the previous return window didn't end in a paddle hit,
            # the tracking shaping accumulated during it was unearned —
            # claw it back now before opening the next window. This
            # preserves the no-op zero-net-shaping invariant across
            # multi-bounce episodes.
            if self._returning and abs(self._return_shaping_total) > 1e-9:
                reward -= self._return_shaping_total
            # Strict gate: only pay the +1 if a paddle hit happened
            # since the last wall contact (or it's the serve and the
            # gate has never been opened).
            if self._paddle_hit_since_last_wall:
                self.bounce_count += 1
                reward += 1.0
            self._paddle_hit_since_last_wall = False
            # Open a fresh shaping window.
            self._returning = True
            self._prev_paddle_to_ball = None
            self._return_shaping_total = 0.0
            event_this_step = True

        # Potential-based tracking shaping: reward reductions in the
        # paddle→ball distance while a return is in progress. Each delta
        # is scale * (prev_dist - dist), which telescopes over the
        # window to scale * (d_init - d_final). Clawback (above on
        # consecutive wall, below on episode end) ensures a window that
        # doesn't terminate in a paddle hit nets zero.
        ball_pos = np.array(self.data.joint("ball_x").qpos[:3])
        paddle_head_pos = np.array(self.data.body("paddle_head").xpos)
        if self._returning and self.track_shaping_scale > 0.0:
            dist = float(np.linalg.norm(ball_pos - paddle_head_pos))
            if self._prev_paddle_to_ball is not None:
                delta = self.track_shaping_scale * (
                    self._prev_paddle_to_ball - dist
                )
                reward += delta
                self._return_shaping_total += delta
            self._prev_paddle_to_ball = dist

        if event_this_step:
            self._steps_since_event = 0
        else:
            self._steps_since_event += 1

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

        # PBRS terminal correction: if the episode ends mid-return, the
        # accumulated shaping was unearned (no paddle hit closed the
        # window), so claw it back now.
        if (
            (terminated or truncated)
            and self._returning
            and abs(self._return_shaping_total) > 1e-9
        ):
            reward -= self._return_shaping_total
            self._return_shaping_total = 0.0

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
        self._paddle_hit_since_last_wall = False
        self._returning = False
        self._prev_paddle_to_ball = None
        self._return_shaping_total = 0.0

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
        "paddle_hit_since_last_wall",
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
        gate_open = np.array([float(self._paddle_hit_since_last_wall)])
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
                np.array(self.data.joint("paddle_pitch").qpos),
                np.array(self.data.joint("paddle_pitch").qvel),
                gate_open,
                rel,
            ),
            axis=0,
        )
