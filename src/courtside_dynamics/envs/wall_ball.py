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
- The *first* paddle contact of each cycle earns ``paddle_hit_bonus``
  so the agent has a dense gradient long before it can close the full
  loop. Repeat paddle contacts before the next wall hit pay nothing:
  without that gate, juggling the ball on the paddle (the previous
  curriculum stage's skill!) farms the bonus indefinitely and is
  competitive with actually rallying.
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
- ``double_bounce_penalty`` subtracts a flat amount on the terminating
  step when the rally dies via a second consecutive floor bounce, so
  letting the ball drop is never a cheaper escape than hitting it out.

Episodes terminate early when the rally is over:

- **Double floor bounce**: as in real wall ball, the ball may touch the
  floor at most once between consecutive paddle/wall contacts; the
  second consecutive floor bounce ends the episode immediately. Bounces
  are detected at substep resolution as ball-floor contact onsets with
  a pre-impact downward speed above ``floor_bounce_min_speed`` — the
  speed gate debounces the contact chatter of a ball settling or
  rolling on the floor, which would otherwise read as a rapid string
  of "bounces".
- **Out of bounds**: the ball left the play volume.
- **Stall**: no paddle/wall rising edge for ``stall_steps`` consecutive
  steps. This catches balls too slow to bounce (rolling along the
  floor) and whiffed serves alike. The counter runs from reset;
  ``stall_steps`` is far longer than the serve flight, so a normal
  serve cannot trip it before the agent has had a chance to make
  contact.

Contact events are filtered to *ball* contacts: the raw touch sensors
sum every contact on their site, so without the filter an unpowered
paddle sagging until its face scrapes the floor reads as a paddle "hit"
(paying the bonus, opening the wall +1 gate, and resetting the stall
clock with the ball metres away).
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import utils

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import CourtsideMujocoEnv

# Cartesian bounds for "ball is still in play". Outside these, the
# episode terminates. Paddle starts near x=-2, wall sits at x=4, so
# we leave a generous margin behind the paddle. The floor geom is a
# MuJoCo plane, which collides as an infinite half-space (its size only
# affects rendering), so the z bound cannot fire from ordinary play --
# it is a guard against solver blow-ups that produce large-but-finite
# states (NaN/inf ones are caught by the nonfinite-obs check instead).
_BALL_MIN_X = -6.0
_BALL_MAX_X = 5.0
_BALL_MIN_Y = -5.5
_BALL_MAX_Y = 5.5
_BALL_MIN_Z = -0.5


class WallBallEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Rally a ball against a wall with a 5-DOF racket."""

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
        double_bounce_penalty: float = 1.0,
        floor_bounce_min_speed: float = 0.5,
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
            double_bounce_penalty=double_bounce_penalty,
            floor_bounce_min_speed=floor_bounce_min_speed,
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
        self.double_bounce_penalty = float(double_bounce_penalty)
        # Minimum pre-impact downward speed for a ball-floor contact
        # onset to count as a bounce. A settling/rolling ball chatters
        # through many near-zero-energy contact onsets that must not
        # count toward the double-bounce rule.
        self.floor_bounce_min_speed = float(floor_bounce_min_speed)
        self.stall_steps = int(stall_steps)

        # Runtime bookkeeping (reset in ``reset_model``).
        self.bounce_count = 0           # rewarded wall contacts (post-paddle)
        self.wall_contact_count = 0     # all wall contacts incl. stray hits
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._steps_since_event = 0
        # Consecutive floor bounces with no paddle/wall contact between
        # them (the wall-ball rally rule: the second one is a dead
        # rally) and the per-episode total, for diagnostics.
        self.floor_bounce_count = 0
        self.floor_bounce_total = 0
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
        # paddle_hit_since_last_wall flag(1) + floor_bounce_count(1) +
        # paddle_head→ball relative xyz(3) = 21. The flag exposes the
        # wall-reward gate state: True iff the next wall contact will
        # pay +1 (and, equivalently, that this cycle's paddle bonus is
        # already consumed), which an MLP policy can't infer from raw
        # state alone. floor_bounce_count exposes the double-bounce
        # termination state — when it reads 1 the next floor bounce
        # ends the episode, which is likewise not inferable from the
        # ball's instantaneous position/velocity. The relative xyz
        # spares the policy from learning the joint→world mapping by
        # hand.
        CourtsideMujocoEnv.__init__(
            self,
            asset_path("wall_ball.xml"),
            episode_len=episode_len,
            obs_dim=21,
            **kwargs,
        )

        # Cache the ball's DOF offset so serve velocities don't depend on
        # a hard-coded index into qvel.
        self._ball_dofadr = int(self.model.joint("ball_x").dofadr[0])
        # Geom ids for the contact-pair checks in
        # _step_mujoco_simulation (floor-bounce detection and filtering
        # touch events down to real ball contacts).
        self._ball_geom_id = int(self.model.geom("ball_geom").id)
        self._floor_geom_id = int(self.model.geom("floor").id)
        self._paddle_geom_id = int(self.model.geom("paddle_face").id)
        self._wall_geom_id = int(self.model.geom("wall_geom").id)
        # Whether the ball was in floor contact on the previous substep,
        # persisted across frames so a contact spanning a frame boundary
        # isn't double-counted as a fresh bounce.
        self._ball_on_floor = False
        # Peak touch-sensor readings across the substeps of the most
        # recent frame (see _step_mujoco_simulation).
        self._substep_wall_touch = 0.0
        self._substep_paddle_touch = 0.0

    def _ball_contacts(self) -> tuple[bool, bool, bool]:
        """Return (floor, paddle, wall) flags for the ball's active contacts."""
        ncon = int(self.data.ncon)
        if ncon == 0:
            return False, False, False
        geom1 = self.data.contact.geom1[:ncon]
        geom2 = self.data.contact.geom2[:ncon]
        ball = self._ball_geom_id
        involves_ball = (geom1 == ball) | (geom2 == ball)
        if not involves_ball.any():
            return False, False, False
        other = np.where(geom1 == ball, geom2, geom1)[involves_ball]
        return (
            bool((other == self._floor_geom_id).any()),
            bool((other == self._paddle_geom_id).any()),
            bool((other == self._wall_geom_id).any()),
        )

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """Step the physics one substep at a time, tracking contact events.

        The ball's contact with the paddle/wall is stiff and brief --
        with realistic restitution it can begin and end entirely inside
        one ``frame_skip`` window (5 x 2 ms). Gymnasium's default runs
        all substeps in one ``mj_step`` call and the env then reads the
        sensors once, so such a hit would be invisible: the reward gate
        would silently fail to open on a real paddle contact. Sampling
        the two touch sensors at every substep and keeping the peak
        makes edge detection immune to contact duration.

        Two refinements happen here, at substep resolution, because both
        are invisible at frame resolution:

        - Touch peaks are sampled only in substeps where the ball is
          actually touching the corresponding geom. The raw sensors sum
          every contact on their site, so an unpowered paddle sagging
          until its face scraped the FLOOR used to register as a paddle
          "hit" with the ball metres away.
        - Floor bounces are detected as ball-floor contact onsets whose
          pre-impact downward speed exceeds ``floor_bounce_min_speed``
          (the debounce for settling/rolling chatter). Any paddle/wall
          contact resets the consecutive-bounce count, mirroring the
          wall-ball rally rule; the reset runs after bounce detection so
          a racket contact in the same substep as a floor touch absolves
          the bounce rather than terminating on a scooped pickup.
        """
        self.data.ctrl[:] = ctrl
        wall_peak = 0.0
        paddle_peak = 0.0
        for _ in range(n_frames):
            ball_vz = float(self.data.qvel[self._ball_dofadr + 2])
            mujoco.mj_step(self.model, self.data)
            on_floor, on_paddle, on_wall = self._ball_contacts()
            if on_wall:
                wall_peak = max(
                    wall_peak, float(self.data.sensor("wall_touch").data[0])
                )
            if on_paddle:
                paddle_peak = max(
                    paddle_peak,
                    float(self.data.sensor("paddle_touch").data[0]),
                )
            if (
                on_floor
                and not self._ball_on_floor
                and ball_vz < -self.floor_bounce_min_speed
            ):
                self.floor_bounce_count += 1
                self.floor_bounce_total += 1
            self._ball_on_floor = on_floor
            if on_paddle or on_wall:
                self.floor_bounce_count = 0
        self._substep_wall_touch = wall_peak
        self._substep_paddle_touch = paddle_peak
        # Matches gymnasium's MujocoEnv: force-related quantities are
        # only computed on demand after stepping.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        self.step_number += 1

        wall_touch = self._substep_wall_touch
        paddle_touch = self._substep_paddle_touch

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
        # The five components sum exactly to ``reward``.
        rew_wall = 0.0
        rew_paddle = 0.0
        rew_shaping = 0.0
        rew_oob = 0.0
        rew_double_bounce = 0.0
        event_this_step = False

        if paddle_edge:
            self.paddle_hit_count += 1
            # Bonus only on the first paddle hit per wall cycle. The gate
            # flag doubles as "bonus consumed": repeat contacts before the
            # next wall hit would otherwise let the agent farm the bonus
            # by juggling the ball on the paddle without ever rallying.
            if not self._paddle_hit_since_last_wall:
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

        # Double bounce: the rally is dead once the ball has touched the
        # floor twice with no paddle/wall contact between (the counter
        # is maintained at substep resolution in
        # _step_mujoco_simulation). Penalized like OOB so neither
        # failure mode is a cheaper escape than the other; skipped when
        # OOB fires on the same step so one failure pays one penalty.
        double_bounce = self.floor_bounce_count >= 2
        if double_bounce and not ball_out_of_bounds:
            reward -= self.double_bounce_penalty
            rew_double_bounce -= self.double_bounce_penalty

        # Stall: cut the episode if the ball has gone dead (e.g. rolling
        # along the floor too slowly for a debounced floor bounce to
        # register). The counter runs from reset: ``stall_steps`` is far
        # longer than the serve flight, so a normal serve can't trip it
        # before the agent has had a chance to make contact, and a
        # whiffed serve that stays in bounds is still cut off instead of
        # burning the whole episode.
        stalled = self._steps_since_event >= self.stall_steps
        obs_nonfinite = not bool(np.isfinite(obs).all())
        terminated = bool(
            obs_nonfinite or ball_out_of_bounds or double_bounce or stalled
        )
        truncated = self.step_number >= self.episode_len

        # Mutually-exclusive termination cause, in priority order, so the
        # per-episode fractions in eval aggregation partition cleanly (sum
        # to <= 1). A terminated step never also counts as a timeout (gym
        # semantics: termination wins over truncation when both fire).
        term_nonfinite = obs_nonfinite
        term_oob = ball_out_of_bounds and not term_nonfinite
        term_double_bounce = double_bounce and not (
            term_nonfinite or ball_out_of_bounds
        )
        term_stall = stalled and not (
            term_nonfinite or ball_out_of_bounds or double_bounce
        )
        term_timeout = truncated and not terminated

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
            # Diagnostic counters / sensors. ``floor_bounce_count`` is
            # the consecutive count driving the double-bounce rule (it
            # resets on every paddle/wall contact); ``floor_bounce_total``
            # accumulates over the whole episode.
            "wall_contact_count": self.wall_contact_count,
            "paddle_hit_count": self.paddle_hit_count,
            "floor_bounce_count": self.floor_bounce_count,
            "floor_bounce_total": self.floor_bounce_total,
            "paddle_touch": paddle_touch,
            "wall_touch": wall_touch,
            "stalled": bool(stalled),
            # Per-component reward breakdown (sums to ``reward``).
            "rew_wall": rew_wall,
            "rew_paddle": rew_paddle,
            "rew_shaping": rew_shaping,
            "rew_oob": rew_oob,
            "rew_double_bounce": rew_double_bounce,
            # Termination-cause flags. Mutually exclusive: at most one is
            # True (exactly one on the terminating step), so per-episode
            # aggregation turns these into a clean OOB / double-bounce /
            # stall / timeout / nonfinite breakdown that sums to <= 1.
            "term_oob": bool(term_oob),
            "term_double_bounce": bool(term_double_bounce),
            "term_stall": bool(term_stall),
            "term_timeout": bool(term_timeout),
            "term_nonfinite": bool(term_nonfinite),
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.step_number = 0
        self.bounce_count = 0
        self.wall_contact_count = 0
        self.paddle_hit_count = 0
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._substep_wall_touch = 0.0
        self._substep_paddle_touch = 0.0
        self._steps_since_event = 0
        self.floor_bounce_count = 0
        self.floor_bounce_total = 0
        self._ball_on_floor = False
        self._paddle_hit_since_last_wall = False
        # Open the PBRS shaping window from t=0 as if a virtual wall
        # hit had just happened. Without this the serve flight earns
        # no shaping and the policy has to find the ball cold.
        self._returning = True
        self._prev_paddle_to_ball = None
        self._return_shaping_total = 0.0

        qpos, qvel = self._noisy_init_state()

        # Serve: throw the ball *toward* the paddle (negative x) with a
        # small upward lob and lateral jitter so the agent can't
        # memorize a single trajectory.
        vx = -(
            self.serve_speed
            + self.np_random.uniform(
                -self.serve_speed_jitter, self.serve_speed_jitter
            )
        )
        # The lateral component is always off-centre (|vy| in
        # [0.8, 1.8], random sign) so the serve can never intersect a
        # racket parked at its reset pose. With realistic ball
        # restitution a straight serve rebounds off a *static* racket,
        # paying the paddle bonus (and sometimes a full rally) for
        # doing nothing -- the no-op baseline must stay <= 0 for the
        # gated reward to mean anything. The minimum clears the parked
        # face (half-width 0.2 + ball radius 0.07) across the whole
        # serve-speed jitter range.
        vy = self.np_random.uniform(0.8, 1.8)
        if self.np_random.random() < 0.5:
            vy = -vy
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
        "floor_bounce_count",
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
        floor_bounces = np.array([float(self.floor_bounce_count)])
        return np.concatenate(
            (
                ball_pos,
                np.asarray(self.data.joint("ball_x").qvel[:3]),
                self._joints_obs(
                    "paddle_slide_x", "paddle_slide_y", "paddle_slide_z",
                    "paddle_yaw", "paddle_pitch",
                ),
                gate_open,
                floor_bounces,
                rel,
            ),
            axis=0,
        )
