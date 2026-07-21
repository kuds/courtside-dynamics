"""Wall Ball environment.

A face-only paddle with a fixed 10-degree upward pitch must rally a ball
against a wall. The policy supplies three normalized absolute position
targets for the paddle's x/y/z slides; a force-limited position servo
handles the low-level motion. The ball is served toward the paddle on
reset, so the agent's first job is to make contact, then drive the ball
back into the wall, then receive the rebound, and so on.

The reward is intentionally narrow. Only the gated wall +1 is paid
outright; everything else is a refundable *advance* against completing
the rally cycle it belongs to:

``rally_style`` selects the contact sequence for a whole run without changing
the policy spaces. ``open`` preserves the original permissive rule,
``volley`` ends the rally on any floor bounce, and ``one_bounce`` requires
exactly one floor bounce before every paddle return. Strict styles process
floor/paddle/wall contacts in MuJoCo substep order, so a later contact inside
the same control frame cannot erase an earlier fault. Paddle pitch remains
fixed at 10 degrees in every style; recipes may only change its world-space x
home and target lane. ``one_bounce`` training can optionally replace a
configurable fraction of normal serves with narrow ``incoming_wall`` or
``post_bounce`` recovery fragments. Evaluation keeps that probability at zero.

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
- A legal, policy-generated wall return arms one
  ``recoverable_bounce_bonus``. Its first required floor bounce consumes the
  opportunity and pays the bonus scaled by how centrally the ball projects
  into ``recoverable_bounce_lateral_limit`` at the paddle's forward x plane.
  Normal serves and synthetic recovery resets never arm it, so curriculum
  state cannot receive credit for a trajectory the policy did not create;
  contact chatter and later bounces cannot pay it twice.
- In ``one_bounce`` style, touching the ball before its required floor
  bounce is by default a terminal ``paddle_before_bounce`` style
  violation. Setting ``early_touch_penalty`` softens it to a
  non-terminal fine (paid outright per contact edge, never refunded)
  while the return gate stays shut: punishing an early contact
  *attempt* exactly as hard as total passivity removes the incentive
  to approach the ball at all, and it deters the forward positioning
  a wide paddle lane requires.
- Likewise, an outgoing shot that drops short of the wall is by default
  a terminal ``floor_before_wall`` violation. Setting
  ``weak_return_penalty`` softens it into a fined retry: the failed
  cycle's advances are clawed back as usual, but the bounced ball stays
  live and the next paddle hit is legal. A tracker whose first swings
  are feeble otherwise dies at the same -1 as total passivity on every
  episode -- the wall in the track -> touch -> swing learning path.
- ``first_hit_bonus`` pays outright, once per episode, on the first
  gate-opening paddle hit and is never clawed back. With every
  refundable advance reversed on failure, all pre-scoring behaviors
  otherwise tie at the failure-penalty floor and episode returns carry
  no gradient toward ball contact. Once per episode so a
  drop-and-retry dribble cannot farm it; the amount should stay well
  below a completed cycle's ~+2 so rallying still dominates.
- ``track_shaping_scale`` adds tracking shaping that rewards reductions
  in the paddle→ball distance while a return is in progress. The
  shaping window is opened **on reset** as if a virtual wall hit had
  just happened — that way the agent gets dense signal during the serve
  flight, before any sensor edge has fired. The per-step deltas
  telescope to ``scale * (d_initial - d_final)`` over each window.
- **Advances are refunded on a failed cycle.** The paddle bonus and the
  tracking shaping are paid the moment they are earned (a from-scratch
  learner needs the dense signal), but they are only *kept* if the
  cycle completes with a gated wall contact. If the cycle instead ends
  any other way — a wall→wall with no return between, out of bounds,
  double bounce, stall, or the episode ending first — every advance
  paid since the last completed cycle is clawed back on that step.
  This is the fix for the exploit run ``20260712_190054`` converged
  to: catch the serve (banking ~1.25 of serve-flight shaping plus the
  0.25 touch bonus), deaden the ball, and wait out the stall clock for
  a risk-free ~1.5 per episode while never touching the wall. Under
  the refund rule a touch followed by anything but a completed return
  nets zero advance, so no prefix of a rally can out-earn the rally.
- ``out_of_bounds_penalty`` subtracts a flat amount on the terminating
  step when the ball leaves the play volume, so the agent is
  incentivised to keep the rally alive rather than letting the ball
  escape.
- ``double_bounce_penalty`` subtracts a flat amount on the terminating
  step when the rally dies via a second consecutive floor bounce, so
  letting the ball drop is never a cheaper escape than hitting it out.
- ``stall_penalty`` subtracts a flat amount on the terminating step
  when the episode ends via the stall cut. Before this existed, stall
  was the one *free* way to end an episode, which is exactly why the
  catch-and-stall policy chose it; now every early-termination path
  costs the same and the only non-negative strategy is to keep
  rallying.

Episodes terminate early when the rally is over:

- **Double floor bounce**: as in real wall ball, the ball may touch the
  floor at most once between consecutive paddle/wall contacts; the
  second consecutive floor bounce ends the episode immediately. Bounces
  are detected at substep resolution as ball-floor contact onsets with
  a pre-impact downward speed above ``floor_bounce_min_speed`` — the
  speed gate debounces the contact chatter of a ball settling or
  rolling on the floor, which would otherwise read as a rapid string
  of "bounces".
- **Style violation**: a volley touched the floor, a one-bounce return
  contacted the paddle before its bounce, or an outgoing one-bounce shot
  hit the floor before reaching the wall.
- **Out of bounds**: the ball left the play volume.
- **Stall**: no paddle/wall rising edge (or required one-bounce floor event)
  for ``stall_steps`` consecutive steps. This catches balls too slow to
  bounce (rolling along the floor) and whiffed serves alike. The counter runs
  from reset; ``stall_steps`` is far longer than the serve flight, so a normal
  serve cannot trip it before the agent has had a chance to make contact.

Nonfinite values never leave the env: a nonfinite action terminates the
episode without stepping the physics, and if the simulation itself
produces a nonfinite state the step returns the last finite observation
instead. Either way the step is flagged ``term_nonfinite``. This
matters because a single NaN observation reaching ``VecNormalize``
permanently poisons its running statistics and silently ruins the rest
of a training run.

Paddle touch events are filtered to *ball* contacts: the raw touch
sensor sums every contact on its site, so without the filter an
unpowered paddle sagging until its face scrapes the floor reads as a
paddle "hit" (paying the bonus, opening the wall +1 gate, and resetting
the stall clock with the ball metres away). The wall sensor needs no
filter — nothing but the ball can physically reach the wall.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box

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

_RALLY_STYLES = frozenset({"open", "volley", "one_bounce"})
_AWAIT_BOUNCE = 0
_AWAIT_PADDLE = 1
_AWAIT_WALL = 2
_RALLY_PHASE_NAMES = {
    _AWAIT_BOUNCE: "await_bounce",
    _AWAIT_PADDLE: "await_paddle",
    _AWAIT_WALL: "await_wall",
}
_CONTACT_EVENT_PRIORITY = {"floor": 0, "paddle": 1, "wall": 2}
_RESET_NORMAL = 0
_RESET_INCOMING_WALL = 1
_RESET_POST_BOUNCE = 2
_RESET_MODE_NAMES = {
    _RESET_NORMAL: "normal",
    _RESET_INCOMING_WALL: "incoming_wall",
    _RESET_POST_BOUNCE: "post_bounce",
}


class WallBallEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Rally with a 10-degree upward-pitched face and three target actions."""

    action_names: tuple[str, ...] = (
        "paddle_target_x",
        "paddle_target_y",
        "paddle_target_z",
    )

    # Reset-scoped episode flag (assigned in reset_model, read by step's
    # accept_paddle_hit closure, which mypy visits first).
    _first_hit_paid: bool

    def __init__(
        self,
        episode_len: int = 750,
        min_force: float = 0.0,
        serve_speed: float = 6.0,
        serve_lob: float = 2.0,
        serve_speed_jitter: float = 1.0,
        serve_vy_min: float = 0.8,
        serve_vy_max: float = 1.8,
        paddle_hit_bonus: float = 0.25,
        track_shaping_scale: float = 0.5,
        out_of_bounds_penalty: float = 1.0,
        double_bounce_penalty: float = 1.0,
        stall_penalty: float = 1.0,
        style_violation_penalty: float = 1.0,
        floor_bounce_min_speed: float = 0.5,
        stall_steps: int = 200,
        rally_style: str = "open",
        paddle_home_x: float = -1.7,
        paddle_x_target_range: tuple[float, float] | None = None,
        recovery_reset_probability: float = 0.0,
        post_bounce_reset_fraction: float = 0.5,
        recoverable_bounce_bonus: float = 0.0,
        recoverable_bounce_lateral_limit: float = 0.0,
        early_touch_penalty: float | None = None,
        weak_return_penalty: float | None = None,
        first_hit_bonus: float = 0.0,
        paddle_joint_damping: float | None = None,
        serve_start_x: float = 1.0,
        paddle_start_x: float | None = None,
        paddle_x_fence: tuple[float, float] | None = None,
        court_style: str = "diagnostic",
        wall_reward_increment: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not isinstance(rally_style, str) or rally_style not in _RALLY_STYLES:
            raise ValueError(
                f"rally_style must be one of {sorted(_RALLY_STYLES)}, "
                f"got {rally_style!r}"
            )
        if not np.isfinite(wall_reward_increment) or wall_reward_increment < 0:
            raise ValueError(
                "wall_reward_increment must be finite and non-negative"
            )
        if not np.isfinite(paddle_home_x):
            raise ValueError("paddle_home_x must be finite")
        if (
            not np.isfinite(style_violation_penalty)
            or style_violation_penalty < 0.0
        ):
            raise ValueError(
                "style_violation_penalty must be finite and nonnegative"
            )
        for name, value in (
            ("recovery_reset_probability", recovery_reset_probability),
            ("post_bounce_reset_fraction", post_bounce_reset_fraction),
        ):
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not np.isfinite(recoverable_bounce_bonus)
            or recoverable_bounce_bonus < 0.0
        ):
            raise ValueError(
                "recoverable_bounce_bonus must be finite and nonnegative"
            )
        if (
            not np.isfinite(recoverable_bounce_lateral_limit)
            or recoverable_bounce_lateral_limit < 0.0
        ):
            raise ValueError(
                "recoverable_bounce_lateral_limit must be finite and "
                "nonnegative"
            )
        if early_touch_penalty is not None and (
            not np.isfinite(early_touch_penalty) or early_touch_penalty < 0.0
        ):
            raise ValueError(
                "early_touch_penalty must be None or finite and nonnegative"
            )
        if weak_return_penalty is not None and (
            not np.isfinite(weak_return_penalty) or weak_return_penalty < 0.0
        ):
            raise ValueError(
                "weak_return_penalty must be None or finite and nonnegative"
            )
        if not np.isfinite(first_hit_bonus) or first_hit_bonus < 0.0:
            raise ValueError(
                "first_hit_bonus must be finite and nonnegative"
            )
        if paddle_joint_damping is not None and (
            not np.isfinite(paddle_joint_damping) or paddle_joint_damping <= 0.0
        ):
            raise ValueError(
                "paddle_joint_damping must be None or finite and positive"
            )
        if not np.isfinite(serve_start_x):
            raise ValueError("serve_start_x must be finite")
        if paddle_start_x is not None and not np.isfinite(paddle_start_x):
            raise ValueError("paddle_start_x must be finite")
        resolved_fence: tuple[float, float] | None = None
        if paddle_x_fence is not None:
            if len(paddle_x_fence) != 2:
                raise ValueError("paddle_x_fence must contain two values")
            resolved_fence = (
                float(paddle_x_fence[0]),
                float(paddle_x_fence[1]),
            )
            if not np.isfinite(resolved_fence).all():
                raise ValueError("paddle_x_fence must be finite")
            if resolved_fence[0] >= resolved_fence[1]:
                raise ValueError("paddle_x_fence must be strictly increasing")
        resolved_target_range: tuple[float, float] | None = None
        if paddle_x_target_range is not None:
            if len(paddle_x_target_range) != 2:
                raise ValueError("paddle_x_target_range must contain two values")
            resolved_target_range = (
                float(paddle_x_target_range[0]),
                float(paddle_x_target_range[1]),
            )
            if not np.isfinite(resolved_target_range).all():
                raise ValueError("paddle_x_target_range must be finite")
            if resolved_target_range[0] >= resolved_target_range[1]:
                raise ValueError(
                    "paddle_x_target_range must be strictly increasing"
                )

        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            min_force=min_force,
            serve_speed=serve_speed,
            serve_lob=serve_lob,
            serve_speed_jitter=serve_speed_jitter,
            serve_vy_min=serve_vy_min,
            serve_vy_max=serve_vy_max,
            paddle_hit_bonus=paddle_hit_bonus,
            track_shaping_scale=track_shaping_scale,
            out_of_bounds_penalty=out_of_bounds_penalty,
            double_bounce_penalty=double_bounce_penalty,
            stall_penalty=stall_penalty,
            style_violation_penalty=style_violation_penalty,
            floor_bounce_min_speed=floor_bounce_min_speed,
            stall_steps=stall_steps,
            rally_style=rally_style,
            paddle_home_x=paddle_home_x,
            paddle_x_target_range=resolved_target_range,
            recovery_reset_probability=float(recovery_reset_probability),
            post_bounce_reset_fraction=float(post_bounce_reset_fraction),
            recoverable_bounce_bonus=float(recoverable_bounce_bonus),
            recoverable_bounce_lateral_limit=float(
                recoverable_bounce_lateral_limit
            ),
            early_touch_penalty=(
                None
                if early_touch_penalty is None
                else float(early_touch_penalty)
            ),
            weak_return_penalty=(
                None
                if weak_return_penalty is None
                else float(weak_return_penalty)
            ),
            first_hit_bonus=float(first_hit_bonus),
            paddle_joint_damping=(
                None
                if paddle_joint_damping is None
                else float(paddle_joint_damping)
            ),
            serve_start_x=float(serve_start_x),
            paddle_start_x=(
                None if paddle_start_x is None else float(paddle_start_x)
            ),
            paddle_x_fence=resolved_fence,
            court_style=court_style,
            wall_reward_increment=float(wall_reward_increment),
            **kwargs,
        )

        self.min_force = float(min_force)
        self.serve_speed = float(serve_speed)
        self.serve_lob = float(serve_lob)
        self.serve_speed_jitter = float(serve_speed_jitter)
        # Lateral serve speed range (|vy|, sign randomized). The 0.8
        # default floor keeps the serve off a racket parked at its
        # reset pose (see reset_model); the ceiling controls how
        # angled the incoming ball can be. Widening the ceiling puts
        # angled receive states -- which a rallying agent otherwise
        # only creates itself, several exchanges in -- into the serve
        # distribution from step one.
        if not 0.0 < serve_vy_min <= serve_vy_max:
            raise ValueError(
                "serve_vy_min must satisfy 0 < serve_vy_min <= "
                f"serve_vy_max, got ({serve_vy_min}, {serve_vy_max})"
            )
        self.serve_vy_min = float(serve_vy_min)
        self.serve_vy_max = float(serve_vy_max)
        self.paddle_hit_bonus = float(paddle_hit_bonus)
        self.track_shaping_scale = float(track_shaping_scale)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.double_bounce_penalty = float(double_bounce_penalty)
        self.stall_penalty = float(stall_penalty)
        self.style_violation_penalty = float(style_violation_penalty)
        self.rally_style = rally_style
        self.paddle_home_x = float(paddle_home_x)
        self.paddle_x_target_range = resolved_target_range
        self.recovery_reset_probability = float(recovery_reset_probability)
        self.post_bounce_reset_fraction = float(post_bounce_reset_fraction)
        self.recoverable_bounce_bonus = float(recoverable_bounce_bonus)
        self.recoverable_bounce_lateral_limit = float(
            recoverable_bounce_lateral_limit
        )
        # ``None`` keeps the historical rule: a pre-bounce paddle touch in
        # one_bounce style is a terminal style violation. A float softens
        # it to a non-terminal fine (see step); the gate stays shut either
        # way. Softening exists because the terminal rule punished early
        # ball-contact *attempts* exactly as hard as total passivity --
        # run 20260714_211111's policy stopped touching the ball entirely
        # after eval-1 style violations -- and because it deters the
        # forward positioning a wider paddle lane requires.
        self.early_touch_penalty = (
            None if early_touch_penalty is None else float(early_touch_penalty)
        )
        # ``None`` keeps the historical rule: an outgoing shot that hits
        # the floor before the wall is a terminal style violation. A float
        # softens it into a fined retry (see step): the 2026-07-17
        # calibration probe showed a y-tracking policy that touches every
        # serve but swings weakly dying 120/120 by this rule at exactly
        # the passivity return (-1), which walls off the natural
        # track -> touch -> swing learning path.
        self.weak_return_penalty = (
            None if weak_return_penalty is None else float(weak_return_penalty)
        )
        # Paid outright on the episode's FIRST gate-opening paddle hit and
        # never clawed back: with every refundable advance reversed on
        # failure, all pre-scoring behaviors tie at the -1 penalty floor
        # and SAC gets no episode-return gradient toward ball contact
        # (run 20260717_040824 never touched the ball in 125k steps).
        # Once per episode -- a per-cycle payment would let a
        # drop-and-retry dribble farm it.
        self.first_hit_bonus = float(first_hit_bonus)
        # ``None`` keeps the shared XML's slide damping (5, the open/volley
        # calibration). Presets override per-instance: the servo's kv acts
        # inside the +/-100 N force clamp, so a pegged actuator's terminal
        # velocity is 100/damping -- at 5 that allows 20 m/s bang-bang
        # swings whose fast, flat returns rebound too deep to recover.
        # WallBallBaseline sets 8 (12.5 m/s cap, ~15% slower returns).
        # A property so a curriculum can reschedule it mid-run; before the
        # model exists this only records the value.
        self.paddle_joint_damping = paddle_joint_damping
        # Serve origin and paddle start position, world-space. The depth
        # curriculum co-moves them (serve_start_x - paddle_start_x fixed)
        # so serve-receipt geometry is identical at every stage while the
        # distance to the wall anneals. Both only affect future resets.
        self.serve_start_x = float(serve_start_x)
        self.paddle_start_x = (
            None if paddle_start_x is None else float(paddle_start_x)
        )
        # Floor-marking presentation (render-only; see
        # _refresh_court_markers). Validated via the property setter.
        self.court_style = court_style
        # Escalating rally payment: the n-th completed return of an
        # episode banks 1 + (n - 1) * increment, so chaining exchanges
        # out-earns restarting them. 0.0 reproduces the flat +1
        # bit-for-bit. Banked outright like the base wall reward (never
        # pending, never clawed back); counters reset with the episode,
        # so recovery fragments cannot farm the escalator.
        self.wall_reward_increment = float(wall_reward_increment)
        # World-space clamp on the x position target, decoupled from the
        # action mapping: the mapping (paddle_x_target_range) stays fixed
        # for a whole run so action semantics never drift, while the fence
        # confines where targets may land -- without it a policy receives
        # the deep serve, then sprints forward and rallies front-court,
        # and an annealed depth would only ever govern the first exchange.
        # Range-validated against the physical workspace after model load.
        self._paddle_x_fence = resolved_fence
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
        self.legal_paddle_hit_count = 0
        # World-space paddle x summed over the episode's legal
        # (gate-opening) paddle hits, so eval aggregation can measure
        # WHERE the policy actually plays -- e.g. whether a deep-fence
        # stage is mastered from the back court or by camping the fence
        # front. Sampled at the end of the control frame that registered
        # the hit (the paddle moves at most one frame's travel past the
        # true contact point). A sum, not a mean: 0.0 with a zero
        # ``legal_paddle_hit_count`` is exact, whereas a fake 0.0 "mean"
        # would sit indistinguishably inside the court's x range.
        self.legal_paddle_hit_x_sum = 0.0
        self.one_bounce_recovery_count = 0
        self.one_bounce_return_count = 0
        self._rally_phase = (
            _AWAIT_BOUNCE if rally_style == "one_bounce" else _AWAIT_PADDLE
        )
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
        # Refundable advances paid during the current rally cycle (last
        # completed gated wall contact -> next one). Both are paid
        # immediately but clawed back if the cycle ends any way other
        # than a gated wall contact, so no prefix of a rally -- catch
        # the serve, juggle, stall out -- can out-earn the rally.
        # Tracked separately so the clawback lands in the matching
        # ``rew_*`` component and the breakdown still sums to reward.
        self._pending_shaping: float = 0.0
        self._pending_bonus: float = 0.0
        # A legal, policy-generated wall return arms exactly one bonus for
        # its next required floor bounce. Synthetic recovery resets never
        # arm it: they teach interception without pretending the policy
        # created the incoming trajectory.
        self._recoverable_bounce_eligible = False
        self._reset_mode = _RESET_NORMAL
        # Most recent finite observation, echoed back if the sim ever
        # produces a nonfinite state so NaNs can't reach VecNormalize.
        self._last_finite_obs: np.ndarray | None = None

        # Obs: ball pos(3) + ball vel(3) + paddle qpos/qvel(6) +
        # paddle_hit_since_last_wall flag(1) + floor_bounce_count(1) +
        # paddle_head→ball relative xyz(3) + ball angular velocity(3) +
        # stall progress(1) + pending advance(1) + recoverable-bounce
        # eligibility(1) = 23. Everything the
        # env's own dynamics or reward depends on is observable — the
        # policy should never face two identical observations with
        # different futures:
        #
        # - The flag exposes the wall-reward gate state: True iff the
        #   next wall contact will pay +1 (and, equivalently, that this
        #   cycle's paddle bonus is already consumed), which an MLP
        #   policy can't infer from raw state alone.
        # - floor_bounce_count exposes the double-bounce termination
        #   state — when it reads 1 the next floor bounce ends the
        #   episode, likewise not inferable from the ball's
        #   instantaneous position/velocity.
        # - The relative xyz spares the policy from learning the
        #   joint→world mapping by hand.
        # - The ball's angular velocity (the free joint's rotational
        #   DOFs) exposes spin: with sliding friction on every ball
        #   contact, spin changes rebound direction, so two balls with
        #   identical position+velocity but different spin behave
        #   differently.
        # - stall_progress is _steps_since_event / stall_steps: the
        #   stall cut is *penalized*, so the clock that triggers the
        #   fine must be visible to the policy being fined.
        # - pending_advance is the shaping+bonus total that a failed
        #   cycle would claw back on the terminating step; the refund
        #   rule makes terminal reward depend on this otherwise-hidden
        #   accumulator.
        # - recoverable_bounce_eligible distinguishes a genuine post-wall
        #   trajectory (whose first legal bounce can earn the configured
        #   bonus) from an otherwise identical synthetic recovery reset.
        CourtsideMujocoEnv.__init__(
            self,
            asset_path("wall_ball.xml"),
            episode_len=episode_len,
            obs_dim=23,
            **kwargs,
        )

        # Capture the XML's compiled damping before any override so a
        # later ``paddle_joint_damping = None`` can restore it exactly.
        self._xml_paddle_damping = {
            joint_name: float(
                self.model.dof_damping[
                    int(self.model.joint(joint_name).dofadr[0])
                ]
            )
            for joint_name in (
                "paddle_slide_x",
                "paddle_slide_y",
                "paddle_slide_z",
            )
        }
        # Re-assign through the property now that the model exists: this
        # writes the override (or the XML default) into dof_damping.
        self.paddle_joint_damping = self.paddle_joint_damping

        # MuJoCo exposes the position actuators' physical qpos ranges as
        # its default action space. The policy API stays normalized and
        # centred on the home pose: zero targets qpos=0 even though the
        # x and z joint limits are asymmetric. This is deliberately a
        # target interface, not a raw force interface.
        compiled_actions = tuple(
            self.model.actuator(index).name for index in range(self.model.nu)
        )
        if compiled_actions != self.action_names:
            raise ValueError("WallBall actuator order differs from the public API")
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.action_names),),
            dtype=np.float32,
        )
        self._control_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._control_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._control_home = np.zeros(self.model.nu, dtype=np.float64)

        # ``paddle_home_x`` and ``paddle_x_target_range`` are public
        # world-space coordinates. Convert them to the x slide's qpos space
        # only after MuJoCo has compiled the model. This lets recipes move and
        # restrict the paddle without editing the shared XML or changing the
        # policy's normalized three-action interface.
        mujoco.mj_forward(self.model, self.data)
        x_joint = self.data.joint("paddle_slide_x")
        self._paddle_x_qposadr = int(self.model.joint("paddle_slide_x").qposadr[0])
        self._paddle_x_origin = float(
            self.data.body("paddle_head").xpos[0] - x_joint.qpos[0]
        )
        physical_world_range = (
            self._paddle_x_origin + float(self._control_low[0]),
            self._paddle_x_origin + float(self._control_high[0]),
        )
        target_world_range = (
            physical_world_range
            if resolved_target_range is None
            else resolved_target_range
        )
        tolerance = 1e-9
        if (
            target_world_range[0] < physical_world_range[0] - tolerance
            or target_world_range[1] > physical_world_range[1] + tolerance
        ):
            raise ValueError(
                "paddle_x_target_range must stay within the physical "
                f"world-space range {physical_world_range}, got "
                f"{target_world_range}"
            )
        if not (
            target_world_range[0] - tolerance
            <= self.paddle_home_x
            <= target_world_range[1] + tolerance
        ):
            raise ValueError(
                "paddle_home_x must lie inside paddle_x_target_range; "
                f"got home {self.paddle_home_x} and range "
                f"{target_world_range}"
            )
        self._control_low[0] = target_world_range[0] - self._paddle_x_origin
        self._control_high[0] = target_world_range[1] - self._paddle_x_origin
        self._control_home[0] = self.paddle_home_x - self._paddle_x_origin
        self.paddle_x_target_range = target_world_range
        # Fence and paddle start are validated against the *physical*
        # workspace (not the possibly narrower mapping range): the fence
        # deliberately confines a full-workspace mapping, and a curriculum
        # start position may sit anywhere the paddle can physically be.
        self._paddle_x_world_range = physical_world_range
        self.paddle_x_fence = self._paddle_x_fence
        if self.paddle_start_x is not None and not (
            physical_world_range[0] - tolerance
            <= self.paddle_start_x
            <= physical_world_range[1] + tolerance
        ):
            raise ValueError(
                "paddle_start_x must lie inside the physical world-space "
                f"range {physical_world_range}, got {self.paddle_start_x}"
            )

        # Court markings are render-only sites (sites never generate
        # contacts); capture their XML alphas so markers hidden by a
        # style (or the fence lines when no fence is set) can be
        # restored exactly.
        self._court_marker_alpha = {
            name: float(self.model.site(name).rgba[3])
            for name in (
                self._COURT_MARKER_SITES
                + self._COURT_STATIC_SITES
                + self._COURT_TENNIS_SITES
                + self._SENSOR_TINT_SITES
            )
        }
        self._refresh_court_markers()

        # Cache the ball's DOF offset so serve velocities don't depend on
        # a hard-coded index into qvel.
        self._ball_qposadr = int(self.model.joint("ball_x").qposadr[0])
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
        # Ball-paddle-face contact on the previous substep. Under RK4
        # the touch sensor lags the contact snapshot by one substep
        # (see _step_mujoco_simulation), so the peak-sampling window
        # must include it or brief contacts lose their force reading.
        self._ball_on_paddle = False
        self._ball_on_wall = False
        # Peak touch-sensor readings across the substeps of the most
        # recent frame (see _step_mujoco_simulation).
        self._substep_wall_touch = 0.0
        self._substep_paddle_touch = 0.0
        # Candidate contact onsets from the most recent control frame.
        # Strict rally modes combine these substep indices with the existing
        # force-qualified paddle/wall edges so floor/paddle/wall order cannot
        # be erased inside one five-substep frame.
        self._substep_contact_events: list[tuple[int, int, str]] = []
        self._substep_floor_bounce_state: tuple[
            np.ndarray, np.ndarray
        ] | None = None

    @property
    def rally_phase_name(self) -> str:
        """Human-readable name for the current fixed-recipe rally phase."""
        return _RALLY_PHASE_NAMES[self._rally_phase]

    @property
    def reset_mode(self) -> str:
        """Reset fragment selected for the current episode."""
        return _RESET_MODE_NAMES[self._reset_mode]

    @property
    def paddle_joint_damping(self) -> float | None:
        """Per-instance paddle slide damping; ``None`` keeps the XML value."""
        return self._paddle_joint_damping

    @paddle_joint_damping.setter
    def paddle_joint_damping(self, value: float | None) -> None:
        if value is not None and (
            not np.isfinite(value) or float(value) <= 0.0
        ):
            raise ValueError(
                "paddle_joint_damping must be None or finite and positive"
            )
        self._paddle_joint_damping = None if value is None else float(value)
        model = getattr(self, "model", None)
        if model is None:
            # Constructor path: the value is applied once the model exists.
            return
        for joint_name, xml_default in self._xml_paddle_damping.items():
            dofadr = int(model.joint(joint_name).dofadr[0])
            model.dof_damping[dofadr] = (
                xml_default
                if self._paddle_joint_damping is None
                else self._paddle_joint_damping
            )

    @property
    def paddle_x_fence(self) -> tuple[float, float] | None:
        """World-space clamp on the x position target; ``None`` disables."""
        return self._paddle_x_fence

    @paddle_x_fence.setter
    def paddle_x_fence(self, value: tuple[float, float] | None) -> None:
        if value is None:
            self._paddle_x_fence = None
            return
        if len(value) != 2:
            raise ValueError("paddle_x_fence must contain two values")
        fence = (float(value[0]), float(value[1]))
        if not np.isfinite(fence).all():
            raise ValueError("paddle_x_fence must be finite")
        if fence[0] >= fence[1]:
            raise ValueError("paddle_x_fence must be strictly increasing")
        world_range = getattr(self, "_paddle_x_world_range", None)
        if world_range is not None:
            tolerance = 1e-9
            if (
                fence[0] < world_range[0] - tolerance
                or fence[1] > world_range[1] + tolerance
            ):
                raise ValueError(
                    "paddle_x_fence must stay within the physical "
                    f"world-space range {world_range}, got {fence}"
                )
        self._paddle_x_fence = fence

    # Court markings (see assets/wall_ball.xml), all render-only sites.
    # The preset-dependent diagnostic markers are rewritten from the
    # resolved kwargs so videos show where the active lane, home, fence,
    # and serve drop actually are for THIS preset; the static diagnostic
    # geography and the tennis-court set only toggle visibility.
    _COURT_MARKER_SITES = (
        "court_lane_strip",
        "court_line_lane_min",
        "court_line_lane_max",
        "court_line_home",
        "court_line_fence_min",
        "court_line_fence_max",
        "court_line_serve",
    )
    _COURT_STATIC_SITES = (
        "court_line_wall_base",
        "court_line_baseline",
        "court_tick_xm4",
        "court_tick_xm3",
        "court_tick_xm2",
        "court_tick_xm1",
        "court_tick_x0",
        "court_tick_xp1",
        "court_tick_xp2",
        "court_tick_xp3",
    )
    _COURT_TENNIS_SITES = (
        "court_tennis_apron",
        "court_tennis_surface",
        "court_tennis_baseline",
        "court_tennis_service_line",
        "court_tennis_singles_left",
        "court_tennis_singles_right",
        "court_tennis_doubles_left",
        "court_tennis_doubles_right",
        "court_tennis_center_service",
        "court_tennis_center_mark",
    )
    # Debug tints hidden in tennis presentation footage. Alpha affects
    # rendering only; the touch sensors attached to these sites keep
    # working (pinned by the cross-style observation-equality test).
    _SENSOR_TINT_SITES = ("wall_sensor", "paddle_sensor")
    _COURT_STYLES = ("diagnostic", "tennis", "none")

    @property
    def court_style(self) -> str:
        """Floor-marking presentation: diagnostic overlays (default), a
        to-size tennis half-court with the wall as the net, or a bare
        floor. Render-only; takes effect at the next reset when changed
        between episodes."""
        return self._court_style

    @court_style.setter
    def court_style(self, value: str | None) -> None:
        # A TOML run config writes court_style = "none", which the
        # loader's None-sentinel converts to None before the env sees
        # it; treat None as the "none" style rather than rejecting it.
        if value is None:
            value = "none"
        if value not in self._COURT_STYLES:
            raise ValueError(
                f"court_style must be one of {self._COURT_STYLES}, "
                f"got {value!r}"
            )
        self._court_style = value

    def _refresh_court_markers(self) -> None:
        """Apply the court style and preset-dependent marker positions.

        Called at construction and every reset so curriculum changes
        applied between episodes (``paddle_x_fence``, ``serve_start_x``,
        ``court_style`` via ``set_wrapper_attr``) stay visible in
        rendered videos. Sites are render-only, so this cannot affect
        physics, observations, or rewards.
        """
        diagnostic = self._court_style == "diagnostic"
        tennis = self._court_style == "tennis"

        def _show(name: str, visible: bool) -> None:
            site_id = int(self.model.site(name).id)
            self.model.site_rgba[site_id][3] = (
                self._court_marker_alpha[name] if visible else 0.0
            )

        for name in self._COURT_STATIC_SITES:
            _show(name, diagnostic)
        for name in self._COURT_TENNIS_SITES:
            _show(name, tennis)
        for name in self._SENSOR_TINT_SITES:
            _show(name, not tennis)

        def _place(name: str, x: float, *, visible: bool = True) -> None:
            site_id = int(self.model.site(name).id)
            self.model.site_pos[site_id][0] = float(x)
            _show(name, visible and diagnostic)

        lane = self.paddle_x_target_range
        # Resolved to the physical range in __init__ before the first
        # call; only the pre-resolution annotation admits None.
        assert lane is not None
        lane_low, lane_high = lane
        _place("court_line_lane_min", lane_low)
        _place("court_line_lane_max", lane_high)
        _place("court_line_home", self.paddle_home_x)
        strip_id = int(self.model.site("court_lane_strip").id)
        self.model.site_pos[strip_id][0] = (lane_low + lane_high) / 2.0
        self.model.site_size[strip_id][0] = max(
            (lane_high - lane_low) / 2.0, 1e-6
        )
        _show("court_lane_strip", diagnostic)
        _place("court_line_serve", self.serve_start_x)
        fence = self.paddle_x_fence
        if fence is None:
            _place("court_line_fence_min", 0.0, visible=False)
            _place("court_line_fence_max", 0.0, visible=False)
        else:
            _place("court_line_fence_min", fence[0])
            _place("court_line_fence_max", fence[1])

    @property
    def recovery_reset_probability(self) -> float:
        """Probability that a one-bounce reset starts in recovery state."""
        return self._recovery_reset_probability

    @recovery_reset_probability.setter
    def recovery_reset_probability(self, value: float) -> None:
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                "recovery_reset_probability must be finite and in [0, 1]"
            )
        self._recovery_reset_probability = float(value)

    @property
    def post_bounce_reset_fraction(self) -> float:
        """Recovery-reset share that starts immediately after the bounce."""
        return self._post_bounce_reset_fraction

    @post_bounce_reset_fraction.setter
    def post_bounce_reset_fraction(self, value: float) -> None:
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                "post_bounce_reset_fraction must be finite and in [0, 1]"
            )
        self._post_bounce_reset_fraction = float(value)

    def _ball_contacts(
        self,
    ) -> tuple[float, float, bool, bool, bool]:
        """Scan the ball's active contacts for one substep.

        Returns ``(face_force, wall_force, on_floor, on_face, on_wall)``:
        the largest normal force among ball-face and ball-wall contact pairs
        (0.0 when none), plus contact flags for those three surfaces.
        The face is the paddle's only collision geometry. Forces come
        from ``mj_contactForce``, which shares ``data.contact``'s
        snapshot: unlike the touch sensors it neither lags the contact
        list under RK4 nor loses edge hits whose contact point falls
        outside the sensor site's thin box, and a ball-pair force is
        inherently ball-specific.
        """
        face_force = 0.0
        wall_force = 0.0
        on_floor = on_face = on_wall = False
        force6 = np.zeros(6)
        ball = self._ball_geom_id
        for i in range(int(self.data.ncon)):
            geom1 = int(self.data.contact.geom1[i])
            geom2 = int(self.data.contact.geom2[i])
            if geom1 == ball:
                other = geom2
            elif geom2 == ball:
                other = geom1
            else:
                continue
            if other == self._floor_geom_id:
                on_floor = True
            elif other == self._paddle_geom_id:
                on_face = True
                mujoco.mj_contactForce(self.model, self.data, i, force6)
                face_force = max(face_force, float(force6[0]))
            elif other == self._wall_geom_id:
                on_wall = True
                mujoco.mj_contactForce(self.model, self.data, i, force6)
                wall_force = max(wall_force, float(force6[0]))
        return face_force, wall_force, on_floor, on_face, on_wall

    def _ordered_frame_events(
        self, *, paddle_edge: bool, wall_edge: bool
    ) -> list[str]:
        """Return force-qualified strict-mode events in substep order.

        Floor events are already qualified by pre-impact speed. Paddle and
        wall candidates are admitted only when the frame's existing force
        edge qualifies them, preserving the established ``min_force``
        contract. When two contacts share a MuJoCo snapshot, deterministic
        floor -> paddle -> wall precedence makes a half-volley count as a
        bounce and prevents a later wall from rescuing a dead outgoing shot.
        """
        selected = [
            event
            for event in self._substep_contact_events
            if event[2] == "floor"
        ]
        fallback_substep = self.frame_skip
        for event_name, qualified in (
            ("paddle", paddle_edge),
            ("wall", wall_edge),
        ):
            if not qualified:
                continue
            candidates = [
                event
                for event in self._substep_contact_events
                if event[2] == event_name
            ]
            if candidates:
                selected.append(min(candidates))
            else:
                # Sensor/contact-force timing can qualify an edge one
                # substep after the physical contact snapshot disappears.
                # Keep the event rather than silently dropping a real hit.
                selected.append(
                    (
                        fallback_substep,
                        _CONTACT_EVENT_PRIORITY[event_name],
                        event_name,
                    )
                )
        selected.sort()
        return [event_name for _, _, event_name in selected]

    def _recoverable_bounce_score(self) -> float:
        """Score the bounce's projected lateral intercept at paddle front.

        MuJoCo's floor contact is sampled at the exact onset substep. Its
        x/y velocity is sufficiently stable for a lateral projection even
        when the soft contact has not yet settled enough for a trustworthy
        post-impact z velocity. If the ball has already crossed the forward
        paddle plane, its bounce y is the conservative fallback. A zero
        lateral limit disables the shaping term.
        """
        limit = self.recoverable_bounce_lateral_limit
        state = self._substep_floor_bounce_state
        if limit <= 0.0 or state is None:
            return 0.0
        position, velocity = state
        if not bool(np.isfinite(position).all() and np.isfinite(velocity).all()):
            return 0.0
        vx = float(velocity[0])
        if vx >= -1e-9:
            return 0.0
        # Ball position is world-space; actuator limits are slide-joint qpos.
        # Convert the resolved workspace back to world coordinates before
        # projecting, and reject a bounce that is already behind the entire
        # paddle lane (the ball is travelling toward decreasing x).
        back_x = float(self._paddle_x_origin + self._control_low[0])
        front_x = float(self._paddle_x_origin + self._control_high[0])
        if self._paddle_x_fence is not None:
            # The reachable region -- and therefore the placement target
            # this score projects to -- is the fence, not the mapping.
            back_x = max(back_x, self._paddle_x_fence[0])
            front_x = min(front_x, self._paddle_x_fence[1])
        ball_x = float(position[0])
        if ball_x < back_x:
            return 0.0
        time_to_front = (front_x - ball_x) / vx
        if time_to_front > 0.0:
            projected_y = float(position[1] + velocity[1] * time_to_front)
        else:
            projected_y = float(position[1])
        return float(np.clip(1.0 - abs(projected_y) / limit, 0.0, 1.0))

    def _action_to_controls(self, action: np.ndarray) -> np.ndarray:
        """Map normalized x/y/z targets to physical slide coordinates.

        Mapping each half of the action range independently keeps action
        zero at the qpos=0 home pose despite asymmetric x/z limits.
        Finite out-of-range inputs are clipped, matching Gymnasium's
        bounded action-space convention.
        """
        normalized = np.asarray(action, dtype=np.float64)
        if normalized.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, "
                f"got {normalized.shape}"
            )
        if not bool(np.isfinite(normalized).all()):
            raise ValueError("action must contain only finite values")
        normalized = np.clip(normalized, -1.0, 1.0)
        positive_span = self._control_high - self._control_home
        negative_span = self._control_home - self._control_low
        targets = np.where(
            normalized >= 0.0,
            self._control_home + normalized * positive_span,
            self._control_home + normalized * negative_span,
        )
        if self._paddle_x_fence is not None:
            # The fence clamps the *target*, not the mapping: in-window
            # actions mean exactly the same thing at every curriculum
            # stage, and only where out-of-window targets saturate moves.
            targets[0] = np.clip(
                targets[0],
                self._paddle_x_fence[0] - self._paddle_x_origin,
                self._paddle_x_fence[1] - self._paddle_x_origin,
            )
        return targets

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

        - The paddle "touch" measurement is primarily the ball-face
          contact normal force read via ``mj_contactForce`` (see
          ``_ball_contacts``): the raw touch sensor sums every contact
          on its site, so an unpowered paddle sagging until its face
          scraped the FLOOR used to register as a paddle "hit" with the
          ball metres away — and conversely the sensor misses genuine
          edge hits (contact point outside its thin site box) and, under
          RK4, lags ``data.contact`` by one substep. The sensor is kept
          as a supplementary reading, sampled only while the ball
          touches the face now or did on the previous substep (the
          one-substep grace bridges the RK4 lag). The wall reading takes
          the max of the pair force and the raw sensor unconditionally:
          nothing but the ball can physically reach the wall, and each
          signal catches brief contacts the other misses.
        - Floor bounces are detected as ball-floor contact onsets whose
          pre-impact downward speed exceeds ``floor_bounce_min_speed``
          (the debounce for settling/rolling chatter). ``open`` retains its
          historical paddle/wall reset behavior. Strict modes defer those
          updates to :meth:`step`, which processes substep-indexed events in
          floor -> paddle -> wall order and latches the first rule fault.
        """
        self.data.ctrl[:] = self._action_to_controls(ctrl)
        wall_peak = 0.0
        paddle_peak = 0.0
        self._substep_contact_events = []
        self._substep_floor_bounce_state = None
        for substep_index in range(n_frames):
            ball_vz = float(self.data.qvel[self._ball_dofadr + 2])
            mujoco.mj_step(self.model, self.data)
            (
                face_force,
                wall_force,
                on_floor,
                on_face,
                on_wall,
            ) = self._ball_contacts()
            wall_sensed = max(
                wall_force, float(self.data.sensor("wall_touch").data[0])
            )
            wall_peak = max(wall_peak, wall_sensed)
            paddle_peak = max(paddle_peak, face_force)
            if on_face or self._ball_on_paddle:
                paddle_peak = max(
                    paddle_peak,
                    float(self.data.sensor("paddle_touch").data[0]),
                )
            floor_onset = (
                on_floor
                and not self._ball_on_floor
                and ball_vz < -self.floor_bounce_min_speed
            )
            if floor_onset:
                if self.rally_style == "open":
                    self.floor_bounce_count += 1
                    self.floor_bounce_total += 1
                self._substep_contact_events.append(
                    (
                        substep_index,
                        _CONTACT_EVENT_PRIORITY["floor"],
                        "floor",
                    )
                )
                if self._substep_floor_bounce_state is None:
                    ball_joint = self.data.joint("ball_x")
                    self._substep_floor_bounce_state = (
                        np.asarray(ball_joint.qpos[:3]).copy(),
                        np.asarray(ball_joint.qvel[:3]).copy(),
                    )
            if self.rally_style != "open":
                if on_face and not self._ball_on_paddle:
                    self._substep_contact_events.append(
                        (
                            substep_index,
                            _CONTACT_EVENT_PRIORITY["paddle"],
                            "paddle",
                        )
                    )
                wall_active = on_wall or wall_sensed > 0.0
                if wall_active and not self._ball_on_wall:
                    self._substep_contact_events.append(
                        (
                            substep_index,
                            _CONTACT_EVENT_PRIORITY["wall"],
                            "wall",
                        )
                    )
            self._ball_on_floor = on_floor
            self._ball_on_paddle = on_face
            self._ball_on_wall = on_wall or wall_sensed > 0.0
            if self.rally_style == "open" and (
                on_face or wall_sensed > 0.0
            ):
                self.floor_bounce_count = 0
        self._substep_wall_touch = wall_peak
        self._substep_paddle_touch = paddle_peak
        # Matches gymnasium's MujocoEnv: force-related quantities are
        # only computed on demand after stepping.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def step(self, a):
        action = np.asarray(a, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, "
                f"got {action.shape}"
            )
        if not bool(np.isfinite(action).all()):
            # A nonfinite action means the policy or a wrapper upstream
            # has blown up. Stepping MuJoCo with it would corrupt the
            # physics state; end the episode instead, without physics.
            return self._nonfinite_termination()
        self.do_simulation(action, self.frame_skip)
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
        # The components sum exactly to ``reward``.
        rew_wall = 0.0
        rew_paddle = 0.0
        rew_recoverable_bounce = 0.0
        rew_shaping = 0.0
        rew_oob = 0.0
        rew_double_bounce = 0.0
        rew_stall = 0.0
        rew_style_violation = 0.0
        rew_early_touch = 0.0
        rew_weak_return = 0.0
        rew_first_hit = 0.0
        event_early_touch = False
        event_weak_return = False
        event_this_step = False
        event_floor_bounce = any(
            event_name == "floor"
            for _, _, event_name in self._substep_contact_events
        )
        event_legal_paddle_hit = False
        event_completed_return = False
        event_recoverable_bounce = False
        event_post_wall_bounce = False
        recoverable_bounce_score = 0.0
        event_style_violation = False
        style_violation_reason: str | None = None
        double_bounce_latched = False

        def accept_paddle_hit() -> None:
            """Open the gated return and pay its refundable first-hit bonus."""
            nonlocal reward, rew_paddle, rew_first_hit
            nonlocal event_legal_paddle_hit
            first_hit = not self._paddle_hit_since_last_wall
            if first_hit:
                self.legal_paddle_hit_count += 1
                self.legal_paddle_hit_x_sum += float(
                    self.data.body("paddle_head").xpos[0]
                )
                reward += self.paddle_hit_bonus
                rew_paddle += self.paddle_hit_bonus
                self._pending_bonus += self.paddle_hit_bonus
                event_legal_paddle_hit = True
                if self.first_hit_bonus > 0.0 and not self._first_hit_paid:
                    # Outright and unconditional from here on: the
                    # episode has proven ball contact once.
                    self._first_hit_paid = True
                    reward += self.first_hit_bonus
                    rew_first_hit += self.first_hit_bonus
                if self.rally_style == "one_bounce":
                    self.one_bounce_recovery_count += 1
            self._paddle_hit_since_last_wall = True
            self._rally_phase = _AWAIT_WALL
            self._returning = False
            self._prev_paddle_to_ball = None

        def accept_wall_contact() -> None:
            """Close the current cycle, paying only a gated legal return."""
            nonlocal reward, rew_wall, rew_shaping, rew_paddle
            nonlocal event_completed_return
            self.wall_contact_count += 1
            legal_return = (
                self._paddle_hit_since_last_wall
                and (
                    self.rally_style == "open"
                    or self._rally_phase == _AWAIT_WALL
                )
            )
            if legal_return:
                # Gated wall contact: the rally cycle completed, so the
                # +1 pays and every advance from this cycle is earned.
                self.bounce_count += 1
                if self.rally_style == "one_bounce":
                    self.one_bounce_return_count += 1
                    self._recoverable_bounce_eligible = True
                # bounce_count was just incremented: this is the n-th
                # completed return, worth 1 + (n-1)*increment.
                payment = 1.0 + (
                    (self.bounce_count - 1) * self.wall_reward_increment
                )
                reward += payment
                rew_wall += payment
                self._pending_shaping = 0.0
                self._pending_bonus = 0.0
                event_completed_return = True
            else:
                # Wall -> wall with no paddle return between: a failed
                # cycle. Claw back the advances paid during it (only
                # shaping can be pending here -- a paddle bonus would
                # have opened the gate) before opening the next window.
                reward -= self._pending_shaping + self._pending_bonus
                rew_shaping -= self._pending_shaping
                rew_paddle -= self._pending_bonus
                self._pending_shaping = 0.0
                self._pending_bonus = 0.0
                if self.rally_style == "one_bounce":
                    self._recoverable_bounce_eligible = False
            self._paddle_hit_since_last_wall = False
            self.floor_bounce_count = 0
            self._rally_phase = (
                _AWAIT_BOUNCE
                if self.rally_style == "one_bounce"
                else _AWAIT_PADDLE
            )
            # Open a fresh shaping window.
            self._returning = True
            self._prev_paddle_to_ball = None

        if self.rally_style == "open":
            # Preserve the original frame-edge behavior exactly for the
            # backwards-compatible setup and its already-trained policies.
            if paddle_edge:
                self.paddle_hit_count += 1
                accept_paddle_hit()
                event_this_step = True
            if wall_edge:
                accept_wall_contact()
                event_this_step = True
        else:
            # Strict modes process floor/paddle/wall contacts in MuJoCo
            # substep order. Once a fault occurs, later contacts in the same
            # control frame cannot rescue the rally or earn reward.
            for contact_event in self._ordered_frame_events(
                paddle_edge=paddle_edge, wall_edge=wall_edge
            ):
                if contact_event == "floor":
                    self.floor_bounce_count += 1
                    self.floor_bounce_total += 1
                    if self.floor_bounce_count >= 2:
                        double_bounce_latched = True
                        break
                    if self.rally_style == "volley":
                        style_violation_reason = "floor_in_volley"
                        break
                    if self._rally_phase == _AWAIT_BOUNCE:
                        if self._recoverable_bounce_eligible:
                            event_post_wall_bounce = True
                            recoverable_bounce_score = (
                                self._recoverable_bounce_score()
                            )
                            if recoverable_bounce_score > 0.0:
                                bounce_reward = (
                                    self.recoverable_bounce_bonus
                                    * recoverable_bounce_score
                                )
                                reward += bounce_reward
                                rew_recoverable_bounce += bounce_reward
                                event_recoverable_bounce = True
                            self._recoverable_bounce_eligible = False
                        self._rally_phase = _AWAIT_PADDLE
                        # A required, legal bounce is rally progress. In the
                        # rear-court preset it also prevents the wall-to-floor
                        # flight from consuming the whole stall budget.
                        event_this_step = True
                    elif self._rally_phase == _AWAIT_WALL:
                        if self.weak_return_penalty is None:
                            style_violation_reason = "floor_before_wall"
                            break
                        # Softened weak return: the outgoing shot dropped
                        # short of the wall. Terminating here priced a
                        # tracker's first feeble swings exactly like
                        # total passivity (-1 either way), walling off
                        # the track -> touch -> swing learning path; the
                        # softened rule fines the drop, claws back the
                        # failed cycle's advances like a wall->wall
                        # failure, and leaves the bounced ball live so
                        # the next paddle hit is a legal retry. A second
                        # bounce before that retry still ends the episode
                        # via the double-bounce rule.
                        self.weak_return_count += 1
                        reward -= self.weak_return_penalty
                        rew_weak_return -= self.weak_return_penalty
                        event_weak_return = True
                        reward -= self._pending_shaping + self._pending_bonus
                        rew_shaping -= self._pending_shaping
                        rew_paddle -= self._pending_bonus
                        self._pending_shaping = 0.0
                        self._pending_bonus = 0.0
                        self._paddle_hit_since_last_wall = False
                        self._recoverable_bounce_eligible = False
                        self._rally_phase = _AWAIT_PADDLE
                        # Open a fresh shaping window, mirroring the
                        # wall->wall failure path: the retry is exactly
                        # where the dense approach signal is needed.
                        self._returning = True
                        self._prev_paddle_to_ball = None
                        event_this_step = True
                elif contact_event == "paddle":
                    self.paddle_hit_count += 1
                    if (
                        self.rally_style == "one_bounce"
                        and self._rally_phase == _AWAIT_BOUNCE
                    ):
                        if self.early_touch_penalty is None:
                            style_violation_reason = "paddle_before_bounce"
                            break
                        # Softened early touch: fine the contact, keep
                        # the return gate shut (the required bounce has
                        # not happened), and let the rally continue.
                        # Paid outright, not as a refundable advance --
                        # a fine that could be handed back would be no
                        # fine at all.
                        self.early_touch_count += 1
                        reward -= self.early_touch_penalty
                        rew_early_touch -= self.early_touch_penalty
                        event_early_touch = True
                        # A fined touch is a fault, not rally progress:
                        # it must not reset the stall clock (a periodic
                        # tap would otherwise hold a dead ball to the
                        # penalty-free timeout, undercutting the stall
                        # fine -- the catch-and-stall exploit class),
                        # and it disarms the recoverable-bounce bonus
                        # (the paddle just re-aimed the rebound, so the
                        # placement is no longer the return's doing).
                        # The shaping window closes like any other
                        # paddle contact: the deflection invalidates
                        # the tracked approach distance.
                        self._recoverable_bounce_eligible = False
                        self._returning = False
                        self._prev_paddle_to_ball = None
                        continue
                    if self._rally_phase == _AWAIT_PADDLE:
                        accept_paddle_hit()
                        # Only the gate-opening hit is rally progress.
                        event_this_step = True
                    # Repeat contacts while awaiting the wall remain
                    # tolerated, but never pay another bonus -- and never
                    # reset the stall clock. A free clock reset let
                    # touch-then-deaden ride a banked outright bonus to
                    # the penalty-free truncation with periodic taps
                    # (measured +0.25 under the bootstrap recipe, above
                    # the weak-swing tracker rung); without it, held-ball
                    # possession stalls out at -1 like any dead rally.
                    self.floor_bounce_count = 0
                    self._returning = False
                    self._prev_paddle_to_ball = None
                elif contact_event == "wall":
                    accept_wall_contact()
                    event_this_step = True

            if style_violation_reason is not None:
                event_style_violation = True
                self._returning = False
                self._prev_paddle_to_ball = None

        # Tracking shaping: reward reductions in the paddle→ball
        # distance while a return is in progress. Each delta is
        # scale * (prev_dist - dist), which telescopes over the window
        # to scale * (d_init - d_final). Every delta joins the pending
        # advance, so shaping is only kept when the cycle completes.
        # Skipped when the state has gone nonfinite -- a NaN distance
        # would otherwise poison the reward and the pending total.
        ball_pos = np.array(self.data.joint("ball_x").qpos[:3])
        paddle_head_pos = np.array(self.data.body("paddle_head").xpos)
        if (
            self._returning
            and style_violation_reason is None
            and not double_bounce_latched
            and self.track_shaping_scale > 0.0
        ):
            dist = float(np.linalg.norm(ball_pos - paddle_head_pos))
            if np.isfinite(dist):
                if self._prev_paddle_to_ball is not None:
                    delta = self.track_shaping_scale * (
                        self._prev_paddle_to_ball - dist
                    )
                    reward += delta
                    rew_shaping += delta
                    self._pending_shaping += delta
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
        # Double bounce: the rally is dead once the ball has touched the
        # floor twice with no paddle/wall contact between (the counter
        # is maintained at substep resolution in
        # _step_mujoco_simulation). Penalized like OOB so neither
        # failure mode is a cheaper escape than the other; skipped when
        # OOB fires on the same step so one failure pays one penalty.
        double_bounce = bool(
            double_bounce_latched or self.floor_bounce_count >= 2
        )
        style_violation = style_violation_reason is not None

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
            obs_nonfinite
            or ball_out_of_bounds
            or style_violation
            or double_bounce
            or stalled
        )
        truncated = self.step_number >= self.episode_len

        # Mutually-exclusive termination cause, in priority order, so the
        # per-episode fractions in eval aggregation partition cleanly (sum
        # to <= 1). A terminated step never also counts as a timeout (gym
        # semantics: termination wins over truncation when both fire).
        term_nonfinite = obs_nonfinite
        term_oob = ball_out_of_bounds and not term_nonfinite
        term_style_violation = style_violation and not (
            term_nonfinite or ball_out_of_bounds
        )
        term_double_bounce = double_bounce and not (
            term_nonfinite or ball_out_of_bounds or style_violation
        )
        term_stall = stalled and not (
            term_nonfinite
            or ball_out_of_bounds
            or style_violation
            or double_bounce
        )
        term_timeout = truncated and not terminated

        # Apply exactly one failure penalty using the same priority as the
        # mutually-exclusive terminal flags. This keeps the breakdown
        # auditable even when two physical failure conditions coincide.
        if term_oob:
            reward -= self.out_of_bounds_penalty
            rew_oob -= self.out_of_bounds_penalty
        elif term_style_violation:
            reward -= self.style_violation_penalty
            rew_style_violation -= self.style_violation_penalty
        elif term_double_bounce:
            reward -= self.double_bounce_penalty
            rew_double_bounce -= self.double_bounce_penalty

        # Stall is a failed rally like OOB and double bounce, and it is
        # fined like one. Without this it was the sole free exit, and
        # the optimal policy was to deaden the ball and wait it out.
        if term_stall:
            reward -= self.stall_penalty
            rew_stall -= self.stall_penalty

        # Terminal refund: the episode is ending, so whatever advance is
        # still pending belongs to a cycle that never completed -- take
        # it back. Ensures a failed cycle nets zero advance no matter
        # how the episode ends (OOB, double bounce, stall, timeout,
        # nonfinite state).
        if terminated or truncated:
            reward -= self._pending_shaping + self._pending_bonus
            rew_shaping -= self._pending_shaping
            rew_paddle -= self._pending_bonus
            self._pending_shaping = 0.0
            self._pending_bonus = 0.0

        # Never emit a nonfinite observation: echo the last finite one
        # instead (the episode is terminating via term_nonfinite). A
        # single NaN obs ingested by VecNormalize's running statistics
        # would silently corrupt every subsequent normalized step.
        if obs_nonfinite and self._last_finite_obs is not None:
            obs = self._last_finite_obs.copy()
        else:
            self._last_finite_obs = obs.copy()

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
            "legal_paddle_hit_count": self.legal_paddle_hit_count,
            # Positional play diagnostics: where (world x) the legal hits
            # happened. The mean divides by the hit count (0.0 when no
            # hits yet -- filter on legal_paddle_hit_count > 0 before
            # aggregating positions across episodes).
            "legal_paddle_hit_x_sum": self.legal_paddle_hit_x_sum,
            "legal_paddle_hit_x_mean": (
                self.legal_paddle_hit_x_sum / self.legal_paddle_hit_count
                if self.legal_paddle_hit_count > 0
                else 0.0
            ),
            "one_bounce_recovery_count": self.one_bounce_recovery_count,
            "one_bounce_return_count": self.one_bounce_return_count,
            # Friendly alias: ``bounce_count`` is retained for callbacks and
            # best-model selection, while ``return_count`` reads naturally in
            # mode-comparison reports.
            "return_count": self.bounce_count,
            "floor_bounce_count": self.floor_bounce_count,
            "floor_bounce_total": self.floor_bounce_total,
            "paddle_touch": paddle_touch,
            "wall_touch": wall_touch,
            "stalled": bool(stalled),
            "rally_style": self.rally_style,
            "rally_phase": self._rally_phase,
            "rally_phase_name": self.rally_phase_name,
            "style_violation_reason": style_violation_reason,
            "early_touch_count": self.early_touch_count,
            "weak_return_count": self.weak_return_count,
            "event_floor_bounce": bool(event_floor_bounce),
            "event_legal_paddle_hit": bool(event_legal_paddle_hit),
            "event_completed_return": bool(event_completed_return),
            "event_recoverable_bounce": bool(event_recoverable_bounce),
            "event_post_wall_bounce": bool(event_post_wall_bounce),
            "event_early_touch": bool(event_early_touch),
            "event_weak_return": bool(event_weak_return),
            "event_style_violation": bool(event_style_violation),
            "recoverable_bounce_score": recoverable_bounce_score,
            "recoverable_bounce_eligible": bool(
                self._recoverable_bounce_eligible
            ),
            "reset_mode": self.reset_mode,
            "reset_mode_id": self._reset_mode,
            # Per-component reward breakdown (sums to ``reward``).
            "rew_wall": rew_wall,
            "rew_paddle": rew_paddle,
            "rew_recoverable_bounce": rew_recoverable_bounce,
            "rew_shaping": rew_shaping,
            "rew_oob": rew_oob,
            "rew_double_bounce": rew_double_bounce,
            "rew_stall": rew_stall,
            "rew_style_violation": rew_style_violation,
            "rew_early_touch": rew_early_touch,
            "rew_weak_return": rew_weak_return,
            "rew_first_hit": rew_first_hit,
            # Termination-cause flags. Mutually exclusive: at most one is
            # True (exactly one on the terminating step), so per-episode
            # aggregation turns these into a clean OOB / double-bounce /
            # stall / timeout / nonfinite breakdown that sums to <= 1.
            "term_oob": bool(term_oob),
            "term_double_bounce": bool(term_double_bounce),
            "term_stall": bool(term_stall),
            "term_style_violation": bool(term_style_violation),
            "term_timeout": bool(term_timeout),
            "term_nonfinite": bool(term_nonfinite),
        }
        return obs, reward, terminated, truncated, info

    def _nonfinite_termination(self):
        """Terminal transition for a nonfinite action, without physics.

        Mirrors ``step``'s terminal bookkeeping: pending advances are
        refunded, the observation is the last finite one, and the cause
        is reported as ``term_nonfinite``. No fine on top -- a NaN
        action is an upstream numerical failure, not a strategy the
        penalty structure needs to price.
        """
        reward = -(self._pending_shaping + self._pending_bonus)
        rew_shaping = -self._pending_shaping
        rew_paddle = -self._pending_bonus
        self._pending_shaping = 0.0
        self._pending_bonus = 0.0
        obs = (
            self._last_finite_obs.copy()
            if self._last_finite_obs is not None
            else self._get_obs()
        )
        info = {
            "sensor_data": 0.0,
            "bounce_count": self.bounce_count,
            "wall_contact_count": self.wall_contact_count,
            "paddle_hit_count": self.paddle_hit_count,
            "legal_paddle_hit_count": self.legal_paddle_hit_count,
            "legal_paddle_hit_x_sum": self.legal_paddle_hit_x_sum,
            "legal_paddle_hit_x_mean": (
                self.legal_paddle_hit_x_sum / self.legal_paddle_hit_count
                if self.legal_paddle_hit_count > 0
                else 0.0
            ),
            "one_bounce_recovery_count": self.one_bounce_recovery_count,
            "one_bounce_return_count": self.one_bounce_return_count,
            "return_count": self.bounce_count,
            "floor_bounce_count": self.floor_bounce_count,
            "floor_bounce_total": self.floor_bounce_total,
            "paddle_touch": 0.0,
            "wall_touch": 0.0,
            "stalled": False,
            "rally_style": self.rally_style,
            "rally_phase": self._rally_phase,
            "rally_phase_name": self.rally_phase_name,
            "style_violation_reason": None,
            "early_touch_count": self.early_touch_count,
            "weak_return_count": self.weak_return_count,
            "event_floor_bounce": False,
            "event_legal_paddle_hit": False,
            "event_completed_return": False,
            "event_recoverable_bounce": False,
            "event_post_wall_bounce": False,
            "event_early_touch": False,
            "event_weak_return": False,
            "event_style_violation": False,
            "recoverable_bounce_score": 0.0,
            "recoverable_bounce_eligible": bool(
                self._recoverable_bounce_eligible
            ),
            "reset_mode": self.reset_mode,
            "reset_mode_id": self._reset_mode,
            "rew_wall": 0.0,
            "rew_paddle": rew_paddle,
            "rew_recoverable_bounce": 0.0,
            "rew_shaping": rew_shaping,
            "rew_oob": 0.0,
            "rew_double_bounce": 0.0,
            "rew_stall": 0.0,
            "rew_style_violation": 0.0,
            "rew_early_touch": 0.0,
            "rew_weak_return": 0.0,
            "rew_first_hit": 0.0,
            "term_oob": False,
            "term_double_bounce": False,
            "term_stall": False,
            "term_style_violation": False,
            "term_timeout": False,
            "term_nonfinite": True,
        }
        return obs, reward, True, False, info

    def reset_model(self):
        self._refresh_court_markers()
        self.step_number = 0
        self.bounce_count = 0
        self.wall_contact_count = 0
        self.paddle_hit_count = 0
        self.legal_paddle_hit_count = 0
        self.legal_paddle_hit_x_sum = 0.0
        self.one_bounce_recovery_count = 0
        self.one_bounce_return_count = 0
        self.early_touch_count = 0
        self.weak_return_count = 0
        self._first_hit_paid = False
        self._rally_phase = (
            _AWAIT_BOUNCE
            if self.rally_style == "one_bounce"
            else _AWAIT_PADDLE
        )
        self._prev_wall_touch = 0.0
        self._prev_paddle_touch = 0.0
        self._substep_wall_touch = 0.0
        self._substep_paddle_touch = 0.0
        self._substep_contact_events = []
        self._substep_floor_bounce_state = None
        self._steps_since_event = 0
        self.floor_bounce_count = 0
        self.floor_bounce_total = 0
        self._ball_on_floor = False
        self._ball_on_paddle = False
        self._ball_on_wall = False
        self._paddle_hit_since_last_wall = False
        # Open the shaping window from t=0 as if a virtual wall hit
        # had just happened. Without this the serve flight earns no
        # shaping and the policy has to find the ball cold.
        self._returning = True
        self._prev_paddle_to_ball = None
        self._pending_shaping = 0.0
        self._pending_bonus = 0.0
        self._recoverable_bounce_eligible = False
        self._reset_mode = _RESET_NORMAL

        # Recovery fragments are a training-only curriculum for the strict
        # one-bounce task. A zero probability preserves the original reset
        # RNG stream exactly; open/volley modes always retain their normal
        # serve even if a caller sets the probability dynamically.
        if (
            self.rally_style == "one_bounce"
            and self.recovery_reset_probability > 0.0
            and self.np_random.random() < self.recovery_reset_probability
        ):
            if self.post_bounce_reset_fraction <= 0.0:
                self._reset_mode = _RESET_INCOMING_WALL
            elif self.post_bounce_reset_fraction >= 1.0:
                self._reset_mode = _RESET_POST_BOUNCE
            elif self.np_random.random() < self.post_bounce_reset_fraction:
                self._reset_mode = _RESET_POST_BOUNCE
            else:
                self._reset_mode = _RESET_INCOMING_WALL

        qpos, qvel = self._noisy_init_state()
        # Move the *actual* reset pose to the configured start (the mapping
        # home unless a curriculum sets paddle_start_x), retaining the same
        # small randomized offset used by every other joint. Merely
        # remapping action zero would leave the baseline paddle at the old
        # front-court XML pose for its first control frame.
        x_noise = (
            qpos[self._paddle_x_qposadr]
            - self.init_qpos[self._paddle_x_qposadr]
        )
        if self.paddle_start_x is None:
            start_qpos = self._control_home[0]
            start_low = self._control_low[0]
            start_high = self._control_high[0]
        else:
            # An explicit start is validated against -- and clamps to --
            # the *physical* workspace, not the possibly narrower action
            # mapping: a curriculum may deliberately start the paddle
            # outside the mapping's lane.
            start_qpos = self.paddle_start_x - self._paddle_x_origin
            start_low = self._paddle_x_world_range[0] - self._paddle_x_origin
            start_high = self._paddle_x_world_range[1] - self._paddle_x_origin
        if self._paddle_x_fence is not None:
            start_low = max(
                start_low, self._paddle_x_fence[0] - self._paddle_x_origin
            )
            start_high = min(
                start_high, self._paddle_x_fence[1] - self._paddle_x_origin
            )
        qpos[self._paddle_x_qposadr] = np.clip(
            start_qpos + x_noise, start_low, start_high
        )

        if self._reset_mode == _RESET_NORMAL:
            # Serve: throw the ball *toward* the paddle (negative x) with a
            # small upward lob and lateral jitter so the agent can't
            # memorize a single trajectory. The origin follows
            # serve_start_x (curriculum stages translate it with the
            # paddle start), keeping the same pose noise as the XML serve.
            # Skipped entirely at the XML origin: the add/subtract round
            # trip is not bit-exact, and it would silently break same-seed
            # reproducibility for every preset that never moves the serve.
            if self.serve_start_x != self.init_qpos[self._ball_qposadr]:
                qpos[self._ball_qposadr] = (
                    self.serve_start_x
                    + qpos[self._ball_qposadr]
                    - self.init_qpos[self._ball_qposadr]
                )
            vx = -(
                self.serve_speed
                + self.np_random.uniform(
                    -self.serve_speed_jitter, self.serve_speed_jitter
                )
            )
            # The lateral component is always off-centre (|vy| in
            # [serve_vy_min, serve_vy_max], random sign) so the serve can
            # never intersect a racket parked at its reset pose. With
            # realistic ball restitution a straight serve rebounds off a
            # *static* racket, paying the paddle bonus (and sometimes a
            # full rally) for doing nothing -- the no-op baseline must stay
            # <= 0 for the gated reward to mean anything.
            vy = self.np_random.uniform(self.serve_vy_min, self.serve_vy_max)
            if self.np_random.random() < 0.5:
                vy = -vy
            vz = self.serve_lob
        elif self._reset_mode == _RESET_INCOMING_WALL:
            # A narrow, centred post-wall flight. It still has to make its
            # required floor bounce before the paddle may return it.
            qpos[self._ball_qposadr : self._ball_qposadr + 3] = (
                self.np_random.uniform(
                    low=np.array([3.1, -0.75, 1.3]),
                    high=np.array([3.5, 0.75, 1.7]),
                )
            )
            vx = self.np_random.uniform(-8.5, -7.5)
            vy = self.np_random.uniform(-0.4, 0.4)
            vz = self.np_random.uniform(-1.5, 0.3)
        else:
            # Begin just after a legal floor bounce. The synthetic bounce is
            # represented in the phase/counters, but it is not credited as a
            # policy recovery and cannot pay recoverable_bounce_bonus.
            qpos[self._ball_qposadr : self._ball_qposadr + 3] = (
                self.np_random.uniform(
                    low=np.array([-0.5, -0.75, 0.10]),
                    high=np.array([0.5, 0.75, 0.15]),
                )
            )
            vx = self.np_random.uniform(-7.5, -6.0)
            vy = self.np_random.uniform(-0.4, 0.4)
            vz = self.np_random.uniform(2.5, 3.5)
            self._rally_phase = _AWAIT_PADDLE
            self.floor_bounce_count = 1
            self.floor_bounce_total = 1
        qvel[self._ball_dofadr + 0] = vx
        qvel[self._ball_dofadr + 1] = vy
        qvel[self._ball_dofadr + 2] = vz

        self.set_state(qpos, qvel)
        obs = self._get_obs()
        self._last_finite_obs = obs.copy()
        return obs

    def _get_reset_info(self) -> dict[str, Any]:
        """Expose the selected fragment without inventing rally credit."""
        return {
            "reset_mode": self.reset_mode,
            "reset_mode_id": self._reset_mode,
            "recoverable_bounce_eligible": bool(
                self._recoverable_bounce_eligible
            ),
        }

    observation_names: tuple[str, ...] = (
        "ball_x", "ball_y", "ball_z",
        "ball_vx", "ball_vy", "ball_vz",
        "paddle_slide_x_qpos", "paddle_slide_x_qvel",
        "paddle_slide_y_qpos", "paddle_slide_y_qvel",
        "paddle_slide_z_qpos", "paddle_slide_z_qvel",
        "paddle_hit_since_last_wall",
        "floor_bounce_count",
        "paddle_to_ball_dx", "paddle_to_ball_dy", "paddle_to_ball_dz",
        "ball_wx", "ball_wy", "ball_wz",
        "stall_progress",
        "pending_advance",
        "recoverable_bounce_eligible",
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
        # Free-joint qvel is [vx, vy, vz, wx, wy, wz]; [3:6] is spin.
        ball_qvel = np.asarray(self.data.joint("ball_x").qvel)
        stall_progress = np.array(
            [min(1.0, self._steps_since_event / max(1, self.stall_steps))]
        )
        pending_advance = np.array(
            [self._pending_shaping + self._pending_bonus]
        )
        recoverable_bounce_eligible = np.array(
            [float(self._recoverable_bounce_eligible)]
        )
        return np.concatenate(
            (
                ball_pos,
                ball_qvel[:3],
                self._joints_obs(
                    "paddle_slide_x", "paddle_slide_y", "paddle_slide_z",
                ),
                gate_open,
                floor_bounces,
                rel,
                ball_qvel[3:6],
                stall_progress,
                pending_advance,
                recoverable_bounce_eligible,
            ),
            axis=0,
        )
