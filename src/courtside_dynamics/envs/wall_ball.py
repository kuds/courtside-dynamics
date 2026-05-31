"""Wall Ball environment.

A 5-DOF racket (x/y/z slide + yaw + pitch hinges) must rally a ball
against a wall. The ball is served toward the racket on reset, so the
agent's first job is to make contact, then drive the ball back into the
wall, then receive the rebound, and so on.

The reward is intentionally narrow:

- Wall contacts earn +1, but **only** if a paddle hit happened since the
  previous wall contact. A "wall → wall" sequence with no paddle return
  between is a failed rally and pays nothing, forcing the agent to
  actually return the ball each cycle rather than free-riding on a ball
  that ricochets off the wall on its own. The serve, in this env, comes
  *toward* the paddle, so the very first event is expected to be a
  paddle hit; the gate is initialised closed (False) on reset so any
  pre-paddle wall contact would pay nothing.
- Each fresh paddle contact earns ``paddle_hit_bonus`` so the agent has
  a dense gradient long before it can close the full loop.
- ``track_shaping_scale`` adds potential-based shaping that rewards
  reductions in the paddle→ball distance while a return is in progress.
  The shaping window is opened **on reset** as if a virtual wall hit
  had just happened — that way the agent gets dense PBRS signal during
  the serve flight, before any sensor edge has fired. Strict PBRS is
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
paddle/wall rising-edge fires for ``stall_steps`` consecutive steps
*after the first event*, the remaining timesteps would be wasted compute
on a ball stuck rolling on the floor, so we cut the episode short.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path

# Cartesian bounds for "ball is still in play". Outside these, the
# episode terminates. Paddle starts near x=-2, wall sits at x=4, so
# we leave a generous margin behind the paddle.
_BALL_MIN_X = -6.0
_BALL_MAX_X = 5.0
_BALL_MIN_Y = -5.5
_BALL_MAX_Y = 5.5
_BALL_MIN_Z = -0.5


class WallBallEnv(MujocoEnv, utils.EzPickle):
    """Rally a ball against a wall with a 5-DOF racket."""

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
        self.wall_contact_count = 0     # all wall contacts incl. stray hits
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._steps_since_event = 0
        self._first_event_seen = False
        # Strict per-cycle gate: True iff a paddle hit has happened
        # since the most recent wall contact. Wall +1 fires only when
        # this is True at the moment of the wall edge.
        self._paddle_hit_since_last_wall = False
        # Tracking-shaping window: True between (virtual or real) wall
        # hit and the next paddle hit. Opened on reset so the serve
        # flight produces dense shaping signal.
        self._returning = False
        self._prev_paddle_to_ball: float | None = None
        # Cumulative tracking shaping awarded since the current return
        # window opened. Clawed back if the window ends without a paddle
        # hit (either via a second wall bounce or via episode end), so
        # passive ball drift toward the paddle nets zero shaping.
        self._return_shaping_total: float = 0.0

        # Obs: ball pos(3) + ball vel(3) + paddle qpos/qvel(10) +
        # paddle_hit_since_last_wall flag(1) + paddle_head→ball
        # relative xyz(3) = 20. The flag exposes the wall-reward gate
        # state: True iff the next wall contact will pay +1, which an
        # MLP policy can't infer from raw state alone. The relative
        # xyz spares the policy from learning the joint→world mapping
        # by hand.
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
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
        # Per-component reward breakdown, surfaced in ``info`` so training
        # diagnostics can tell whether the agent is completing rallies
        # (rew_wall) or just farming the dense shaping term (rew_shaping).
        # The four components sum exactly to ``reward``.
        rew_wall = 0.0
        rew_paddle = 0.0
        rew_shaping = 0.0
        rew_oob = 0.0
        event_this_step = False

        if paddle_edge:
            self.paddle_hit_count += 1
            reward += self.paddle_hit_bonus
            rew_paddle += self.paddle_hit_bonus
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
                rew_shaping -= self._return_shaping_total
            # Strict gate: only pay the +1 if a paddle hit happened
            # since the last wall contact.
            if self._paddle_hit_since_last_wall:
                self.bounce_count += 1
                reward += 1.0
                rew_wall += 1.0
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
                rew_shaping += delta
                self._return_shaping_total += delta
            self._prev_paddle_to_ball = dist

        if event_this_step:
            self._steps_since_event = 0
            self._first_event_seen = True
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
            rew_oob -= self.out_of_bounds_penalty

        # Stall: cut the episode if the ball has gone dead. Only counts
        # *after* the first event has fired, otherwise a slow serve
        # flight would spuriously trip it before the agent has had a
        # chance to make contact.
        stalled = (
            self._first_event_seen
            and self._steps_since_event >= self.stall_steps
        )
        obs_nonfinite = not bool(np.isfinite(obs).all())
        terminated = bool(obs_nonfinite or ball_out_of_bounds or stalled)
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
            rew_shaping -= self._return_shaping_total
            self._return_shaping_total = 0.0

        info = {
            # Backward-compat keys consumed by callbacks/CSV writers.
            "sensor_data": wall_touch,
            "bounce_count": self.bounce_count,
            # Diagnostic counters / sensors.
            "wall_contact_count": self.wall_contact_count,
            "paddle_hit_count": self.paddle_hit_count,
            "paddle_touch": paddle_touch,
            "wall_touch": wall_touch,
            "stalled": bool(stalled),
            # Per-component reward breakdown (sums to ``reward``).
            "rew_wall": rew_wall,
            "rew_paddle": rew_paddle,
            "rew_shaping": rew_shaping,
            "rew_oob": rew_oob,
            # Termination-cause flags. At most one is True, and exactly
            # one on the final step of an episode; per-episode aggregation
            # turns these into OOB / stall / timeout / nonfinite fractions.
            "term_oob": bool(ball_out_of_bounds),
            "term_stall": bool(stalled),
            "term_timeout": bool(truncated),
            "term_nonfinite": bool(obs_nonfinite),
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
        self._first_event_seen = False
        self._paddle_hit_since_last_wall = False
        # Open the PBRS shaping window from t=0 as if a virtual wall
        # hit had just happened. Without this the serve flight earns
        # no shaping and the policy has to find the ball cold.
        self._returning = True
        self._prev_paddle_to_ball = None
        self._return_shaping_total = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )

        # Serve: throw the ball *toward* the paddle (negative x) with a
        # small upward lob and mild lateral jitter so the agent can't
        # memorize a single trajectory.
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

    observation_names: tuple[str, ...] = (
        "ball_x", "ball_y", "ball_z",
        "ball_vx", "ball_vy", "ball_vz",
        "paddle_slide_x_qpos", "paddle_slide_x_qvel",
        "paddle_slide_y_qpos", "paddle_slide_y_qvel",
        "paddle_slide_z_qpos", "paddle_slide_z_qvel",
        "paddle_yaw_qpos", "paddle_yaw_qvel",
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
                np.array(self.data.joint("paddle_yaw").qpos),
                np.array(self.data.joint("paddle_yaw").qvel),
                np.array(self.data.joint("paddle_pitch").qpos),
                np.array(self.data.joint("paddle_pitch").qvel),
                gate_open,
                rel,
            ),
            axis=0,
        )
