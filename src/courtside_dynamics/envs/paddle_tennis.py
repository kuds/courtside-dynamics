"""PaddleTennis: 1v1 cooperative rally on the probe-frozen paddle court.

The P1-phase environment of the PaddleTennis era
(docs/design_paddle_tennis.md), frozen on the probe battery's numbers
(P0–P4; docs/paddle_tennis_probes_20260802.md and
docs/paddle_tennis_probes_p3_p4_20260802.md):

- **Court**: half-length 6.5 m, regulation 0.914 m net, singles width
  ±4.115 m; wall-ball ball and paddles verbatim, slide damping 8
  (``assets/paddle_court.xml``).
- **Task**: cooperative rally against an opponent controller on side
  B. The policy plays side A through the familiar normalized 3-action
  paddle interface; every rules-confirmed legal return — by either
  side, this is *cooperative* play — pays ``+return_reward``. A fault
  ending the point pays ``-fault_penalty`` (whichever side faulted:
  the pair keeps the rally alive together, exactly the shared-outcome
  design of ``HumanoidTennisCoopEnv``'s rally target), and unsafe or
  non-finite physics pays ``-unsafe_physics_penalty``.
- **Serve**: the P3-measured band (origin 3.25 m behind the net,
  9 m/s, 21°, probe-standard jitter — 100% legal, mean landing
  4.55 m, 100% returnable by the scripted opponent). One point per
  episode; the serving side alternates on every reset, matching the
  humanoid env's alternation contract. The episode terminates when
  the rules machine terminates the point and truncates at
  ``episode_len``.
- **Opponent**: side B is driven through the side-relative mirror by
  ``opponent_controller`` (default: the frozen ``lead_charge``
  scripted controller). A frozen policy drops in via the same
  observation-in/action-out signature once probe P5 settles champion
  transfer.

Observation (48 values, all side-A-local; probe P4 pinned the
mirroring identity of the physical block bit-for-bit):

- ``[0:24]``   physical state: ball position/velocity/spin, own and
  opponent paddle position/velocity, ball-minus-own-paddle;
- ``[24:36]``  rally state: phase one-hot, serving/returner/ball-side
  flags (own-relative), feed and pending-return crossing flags,
  bounce count, rally count, episode remaining fraction;
- ``[36:48]``  contact-latch and release-progress state for the six
  live channels (own racket, opponent racket, court, net, own
  racket–net, opponent racket–net) — the sampler's rising-edge
  suppression state, exposed so the observation stays Markov, the
  same reasoning as the humanoid env's contact tail.

Recipes normalize the continuous physical block ``[0:24]`` and leave
the bounded tail raw (the humanoid convention: freshly active flags
must not inherit near-zero variance from a transferred normalizer).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import (
    CourtsideMujocoEnv,
    finite_nonnegative,
)
from courtside_dynamics.envs._paddle import PaddleInterface
from courtside_dynamics.envs._paddle_court import (
    PADDLE_COURT,
    PADDLE_HOME_X,
    PaddleCourtServe,
    scripted_lead_charge_opponent,
)
from courtside_dynamics.envs._serve import mirror_for_side
from courtside_dynamics.envs._tennis_events import (
    TENNIS_CONTACT_CHANNELS,
    SubstepTennisEventSampler,
    TennisSceneContactIndex,
    TennisStepEventBatch,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
    RallyPhase,
    RallyStateMachine,
    RallyTransition,
    TerminationReason,
)

#: The six sampler channels that exist on a robot-free court, as
#: indices into ``TENNIS_CONTACT_CHANNELS`` (the humanoid channels are
#: structurally zero here and are not observed).
_LIVE_CHANNEL_INDICES = tuple(
    TENNIS_CONTACT_CHANNELS.index(name)
    for name in (
        "ball_racket_a",
        "ball_racket_b",
        "ball_court",
        "ball_net",
        "racket_a_net",
        "racket_b_net",
    )
)
#: The same channels viewed from side B: own/opponent rackets swap.
_LIVE_CHANNEL_INDICES_MIRRORED = tuple(
    TENNIS_CONTACT_CHANNELS.index(name)
    for name in (
        "ball_racket_b",
        "ball_racket_a",
        "ball_court",
        "ball_net",
        "racket_b_net",
        "racket_a_net",
    )
)

_UNSAFE_REASONS = frozenset(
    {
        TerminationReason.NONFINITE_STATE,
        TerminationReason.UNSAFE_PHYSICS,
    }
)

#: Termination-reason groups exposed as ``term_*`` info flags (the
#: wall-ball convention: exactly one is 1 on a terminal step).
_TERM_GROUPS: tuple[tuple[str, frozenset[TerminationReason]], ...] = (
    ("term_out_of_bounds", frozenset({TerminationReason.OUT_OF_BOUNDS})),
    ("term_ball_net", frozenset({TerminationReason.BALL_NET})),
    ("term_second_bounce", frozenset({TerminationReason.SECOND_BOUNCE})),
    (
        "term_failed_to_cross",
        frozenset({TerminationReason.FAILED_TO_CROSS}),
    ),
    (
        "term_illegal_hit",
        frozenset(
            {
                TerminationReason.WRONG_HITTER,
                TerminationReason.DOUBLE_HIT,
                TerminationReason.PREMATURE_HIT,
                TerminationReason.SIMULTANEOUS_RACKET_CONTACT,
                TerminationReason.REVERSE_CROSSING,
            }
        ),
    ),
    (
        "term_net_touch",
        frozenset(
            {
                TerminationReason.RACKET_A_NET,
                TerminationReason.RACKET_B_NET,
            }
        ),
    ),
    ("term_nonfinite", _UNSAFE_REASONS),
)

_PHYSICAL_NAMES = (
    *(f"ball_position_{axis}" for axis in "xyz"),
    *(f"ball_linear_velocity_{axis}" for axis in "xyz"),
    *(f"ball_angular_velocity_{axis}" for axis in "xyz"),
    *(f"own_paddle_position_{axis}" for axis in "xyz"),
    *(f"own_paddle_velocity_{axis}" for axis in "xyz"),
    *(f"opponent_paddle_position_{axis}" for axis in "xyz"),
    *(f"opponent_paddle_velocity_{axis}" for axis in "xyz"),
    *(f"ball_minus_own_paddle_{axis}" for axis in "xyz"),
)

_RALLY_NAMES = (
    *(f"rally_phase_{phase.name.lower()}" for phase in RallyPhase),
    "own_is_serving",
    "expected_returner_is_own",
    "ball_side_is_own",
    "feed_crossed_net",
    "pending_return_crossed_net",
    "bounce_count",
    "rally_count",
    "episode_remaining_fraction",
)

_CONTACT_CHANNEL_LABELS = (
    "own_racket",
    "opponent_racket",
    "court",
    "net",
    "own_racket_net",
    "opponent_racket_net",
)
_CONTACT_NAMES = tuple(
    f"contact_latched_{label}" for label in _CONTACT_CHANNEL_LABELS
) + tuple(
    f"contact_release_progress_{label}"
    for label in _CONTACT_CHANNEL_LABELS
)

PADDLE_TENNIS_OBSERVATION_NAMES = (
    _PHYSICAL_NAMES + _RALLY_NAMES + _CONTACT_NAMES
)

#: Slice of the observation the recipes normalize (the continuous
#: physical block); the bounded rally/contact tail stays raw.
PADDLE_TENNIS_NORMALIZED_SLICE = slice(0, len(_PHYSICAL_NAMES))

PADDLE_TENNIS_ACTION_NAMES = ("target_x", "target_y", "target_z")

OpponentController = Callable[[np.ndarray], np.ndarray]


class PaddleTennisEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Cooperative 1v1 paddle rally on the probe-frozen court."""

    def __init__(
        self,
        episode_len: int = 1500,
        serve_config: PaddleCourtServe | None = None,
        opponent_controller: OpponentController | None = None,
        court_style: str = "diagnostic",
        return_reward: float = 1.0,
        fault_penalty: float = 1.0,
        unsafe_physics_penalty: float = 2.0,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            serve_config=serve_config,
            opponent_controller=opponent_controller,
            court_style=court_style,
            return_reward=return_reward,
            fault_penalty=fault_penalty,
            unsafe_physics_penalty=unsafe_physics_penalty,
            **kwargs,
        )
        self.return_reward = finite_nonnegative(
            "return_reward", return_reward
        )
        self.fault_penalty = finite_nonnegative(
            "fault_penalty", fault_penalty
        )
        self.unsafe_physics_penalty = finite_nonnegative(
            "unsafe_physics_penalty", unsafe_physics_penalty
        )
        self.serve_config = serve_config or PaddleCourtServe()
        self.opponent_controller: OpponentController = (
            opponent_controller
            if opponent_controller is not None
            else scripted_lead_charge_opponent
        )
        self.court_style = court_style  # validated by the setter

        CourtsideMujocoEnv.__init__(
            self,
            asset_path("paddle_court.xml"),
            episode_len=episode_len,
            obs_dim=len(PADDLE_TENNIS_OBSERVATION_NAMES),
            **kwargs,
        )

        # The policy commands only its own paddle; the compiled model
        # carries both sides' actuators, so the public action space is
        # NOT MuJoCo's default.
        compiled = tuple(
            self.model.actuator(index).name
            for index in range(self.model.nu)
        )
        expected = tuple(
            f"player_{side}_target_{axis}"
            for side in ("a", "b")
            for axis in ("x", "y", "z")
        )
        if compiled != expected:
            raise ValueError(
                "paddle_court actuator order differs from the public API"
            )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        mujoco.mj_forward(self.model, self.data)
        self._paddles = {
            CourtSide.A: PaddleInterface(
                self.model,
                self.data,
                home_x=-PADDLE_HOME_X,
                joint_names=(
                    "player_a_slide_x",
                    "player_a_slide_y",
                    "player_a_slide_z",
                ),
                actuator_names=(
                    "player_a_target_x",
                    "player_a_target_y",
                    "player_a_target_z",
                ),
                head_body_name="player_a_head",
            ),
            CourtSide.B: PaddleInterface(
                self.model,
                self.data,
                home_x=PADDLE_HOME_X,
                joint_names=(
                    "player_b_slide_x",
                    "player_b_slide_y",
                    "player_b_slide_z",
                ),
                actuator_names=(
                    "player_b_target_x",
                    "player_b_target_y",
                    "player_b_target_z",
                ),
                head_body_name="player_b_head",
            ),
        }
        self._index = TennisSceneContactIndex.from_model(
            self.model, require_robots=False
        )
        self._event_sampler = SubstepTennisEventSampler(
            self.model, index=self._index, court=PADDLE_COURT
        )
        self._rules = RallyStateMachine(
            serving_side=CourtSide.A, court=PADDLE_COURT
        )
        ball_joint = self.model.joint("ball_free")
        self._ball_qposadr = int(ball_joint.qposadr[0])
        self._ball_dofadr = int(ball_joint.dofadr[0])

        # Serve sides alternate across resets (the humanoid contract).
        self._next_serving_side = CourtSide.A
        self._serving_side = CourtSide.A
        self._latest_event_batch: TennisStepEventBatch | None = None
        self._last_transition: RallyTransition | None = None
        self._crossings = 0
        self._last_serve_state: tuple[np.ndarray, np.ndarray] | None = None

        # Court-style visibility lists are derived from the compiled
        # model (the WallBall mechanism): court_tennis_* is the
        # presentation set; every other court_* site is diagnostic,
        # minus the preset-dependent serve markers repositioned at
        # reset.
        court_sites = tuple(
            name
            for name in (
                self.model.site(i).name for i in range(self.model.nsite)
            )
            if name.startswith("court_")
        )
        self._court_preset_sites = ("court_line_serve_a", "court_line_serve_b")
        missing = [
            name
            for name in self._court_preset_sites
            if name not in court_sites
        ]
        if missing:
            raise ValueError(
                f"paddle_court.xml is missing preset marker site(s) "
                f"{missing}"
            )
        self._court_tennis_sites = tuple(
            name
            for name in court_sites
            if name.startswith("court_tennis_")
        )
        self._court_static_sites = tuple(
            name
            for name in court_sites
            if not name.startswith("court_tennis_")
            and name not in self._court_preset_sites
        )
        self._court_marker_alpha = {
            name: float(self.model.site(name).rgba[3])
            for name in court_sites
        }
        self._refresh_court_markers()

    # -- court style (render-only) ----------------------------------------

    _COURT_STYLES = ("diagnostic", "tennis", "none")

    @property
    def court_style(self) -> str:
        """Floor-marking presentation: diagnostic overlays (default), a
        to-size tennis mini-court, or a bare floor. Render-only; takes
        effect at the next reset when changed between episodes."""
        return self._court_style

    @court_style.setter
    def court_style(self, value: str | None) -> None:
        # TOML's "none" sentinel arrives as None; treat it as the
        # "none" style (the WallBall contract).
        if value is None:
            value = "none"
        if value not in self._COURT_STYLES:
            raise ValueError(
                f"court_style must be one of {self._COURT_STYLES}, "
                f"got {value!r}"
            )
        self._court_style = value

    def _refresh_court_markers(self) -> None:
        """Apply the court style; render-only (sites cannot collide)."""
        diagnostic = self._court_style == "diagnostic"
        tennis = self._court_style == "tennis"

        def _show(name: str, visible: bool) -> None:
            site_id = int(self.model.site(name).id)
            self.model.site_rgba[site_id][3] = (
                self._court_marker_alpha[name] if visible else 0.0
            )

        for name in self._court_static_sites:
            _show(name, diagnostic)
        for name in self._court_tennis_sites:
            _show(name, tennis)
        # Preset-dependent serve markers: repositioned from the
        # resolved serve config, visible in the diagnostic style only.
        for name, sign in (
            ("court_line_serve_a", -1.0),
            ("court_line_serve_b", 1.0),
        ):
            site_id = int(self.model.site(name).id)
            self.model.site_pos[site_id][0] = (
                sign * self.serve_config.start_distance_from_net
            )
            _show(name, diagnostic)

    # -- observations (side-relative; P4-pinned mirroring) ----------------

    def _ball_position(self) -> np.ndarray:
        return self.data.qpos[
            self._ball_qposadr : self._ball_qposadr + 3
        ].copy()

    def _ball_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self._ball_dofadr : self._ball_dofadr + 3
        ].copy()

    def _ball_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self._ball_dofadr + 3 : self._ball_dofadr + 6
        ].copy()

    def _paddle_position(self, side: CourtSide) -> np.ndarray:
        head = "player_a_head" if side is CourtSide.A else "player_b_head"
        return np.asarray(
            self.data.body(head).xpos, dtype=np.float64
        ).copy()

    def _paddle_velocity(self, side: CourtSide) -> np.ndarray:
        joints = (
            ("player_a_slide_x", "player_a_slide_y", "player_a_slide_z")
            if side is CourtSide.A
            else (
                "player_b_slide_x",
                "player_b_slide_y",
                "player_b_slide_z",
            )
        )
        return np.array(
            [
                float(
                    self.data.qvel[int(self.model.joint(name).dofadr[0])]
                )
                for name in joints
            ],
            dtype=np.float64,
        )

    def observation_for_side(self, side: CourtSide) -> np.ndarray:
        """The full side-local observation (the policy reads side A)."""
        side = CourtSide(side)
        opponent = side.opponent
        snapshot = self._rules.snapshot()

        physical = np.concatenate(
            (
                mirror_for_side(self._ball_position(), side),
                mirror_for_side(self._ball_velocity(), side),
                mirror_for_side(self._ball_angular_velocity(), side),
                mirror_for_side(self._paddle_position(side), side),
                mirror_for_side(self._paddle_velocity(side), side),
                mirror_for_side(self._paddle_position(opponent), side),
                mirror_for_side(self._paddle_velocity(opponent), side),
                mirror_for_side(
                    self._ball_position() - self._paddle_position(side),
                    side,
                ),
            )
        )

        phase_one_hot = np.zeros(len(RallyPhase), dtype=np.float64)
        phase_one_hot[int(snapshot.phase)] = 1.0
        remaining = 1.0 - min(1.0, self.step_number / self.episode_len)
        rally = np.array(
            [
                *phase_one_hot,
                1.0 if snapshot.serving_side is side else 0.0,
                1.0 if snapshot.expected_returner is side else 0.0,
                1.0 if snapshot.ball_side is side else 0.0,
                1.0 if snapshot.feed_crossed_net else 0.0,
                1.0 if snapshot.pending_return_crossed_net else 0.0,
                float(snapshot.bounce_count),
                float(snapshot.rally_count),
                remaining,
            ],
            dtype=np.float64,
        )

        markov = np.asarray(
            self._event_sampler.markov_state(), dtype=np.float64
        )
        channel_count = len(TENNIS_CONTACT_CHANNELS)
        indices = (
            _LIVE_CHANNEL_INDICES
            if side is CourtSide.A
            else _LIVE_CHANNEL_INDICES_MIRRORED
        )
        contact = np.concatenate(
            (
                markov[list(indices)],
                markov[[channel_count + i for i in indices]],
            )
        )
        return np.concatenate((physical, rally, contact))

    def _get_obs(self) -> np.ndarray:
        return self.observation_for_side(CourtSide.A)

    # -- stepping ----------------------------------------------------------

    def _apply_side_action(
        self, side: CourtSide, local_action: np.ndarray
    ) -> None:
        world_action = mirror_for_side(
            np.asarray(local_action, dtype=np.float64).copy(), side
        )
        self._paddles[side].apply(self.data, world_action)

    def _step_mujoco_simulation(self, ctrl, n_frames) -> None:
        """Step substep-by-substep, sampling contact/crossing events.

        ``ctrl`` is ignored: both paddles' controls were written by
        :meth:`step` before ``do_simulation`` ran (the policy's side A
        action plus the opponent's side B action).
        """
        del ctrl
        self._event_sampler.begin_control_step(self.step_number)
        stopped = False
        for control_substep in range(n_frames):
            pre_step = self._event_sampler.safety_event_kind(self.data)
            if pre_step is not None:
                self._event_sampler.record_safety_event(
                    pre_step, control_substep=control_substep
                )
                stopped = True
                break
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError:
                self._event_sampler.record_safety_event(
                    RallyEventKind.UNSAFE_PHYSICS,
                    control_substep=control_substep,
                )
                stopped = True
                break
            safety = self._event_sampler.sample_substep(
                self.data, control_substep=control_substep
            )
            if safety is not None:
                stopped = True
                break
        del stopped
        self._latest_event_batch = self._event_sampler.end_control_step()

    def step(self, a):
        action = np.asarray(a, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, "
                f"got {action.shape}"
            )
        if not bool(np.isfinite(action).all()):
            # Never step physics from a blown-up policy/wrapper: end
            # the episode on the echoed observation (the shared guard).
            return self._nonfinite_termination()
        if not self._physics_state_is_finite():
            return self._nonfinite_termination()

        # The opponent reads its side-local view BEFORE physics, like
        # the policy did (both act on the same pre-step state).
        opponent_action = np.asarray(
            self.opponent_controller(
                self.observation_for_side(CourtSide.B)
            ),
            dtype=np.float64,
        )
        if opponent_action.shape != (3,) or not bool(
            np.isfinite(opponent_action).all()
        ):
            raise ValueError(
                "opponent_controller must return a finite action of "
                f"shape (3,), got {opponent_action!r}"
            )
        self._apply_side_action(CourtSide.A, np.clip(action, -1.0, 1.0))
        self._apply_side_action(
            CourtSide.B, np.clip(opponent_action, -1.0, 1.0)
        )
        # Not gymnasium's do_simulation: its ctrl-shape gate compares
        # against model.nu (both sides' actuators), but the public
        # action is one side's three targets and both controls were
        # already written above. The override ignores ctrl entirely.
        self._step_mujoco_simulation(None, self.frame_skip)
        self.step_number += 1

        batch = self._latest_event_batch
        assert batch is not None
        transition = self._rules.advance(
            batch.events, contact_peaks=batch.contact_peaks
        )
        self._last_transition = transition
        after = transition.after
        self._crossings = max(
            0,
            int(after.net_crossing_count) - int(after.feed_crossed_net),
        )

        rew_return = self.return_reward * len(
            transition.confirmed_returns
        )
        rew_fault = 0.0
        rew_unsafe = 0.0
        if transition.terminated_now:
            if after.termination_reason in _UNSAFE_REASONS:
                rew_unsafe = -self.unsafe_physics_penalty
            else:
                rew_fault = -self.fault_penalty
        reward = rew_return + rew_fault + rew_unsafe

        raw_observation = self._get_obs()
        nonfinite = not bool(np.isfinite(raw_observation).all())
        obs = self._record_or_echo_observation(raw_observation, nonfinite)
        forced_nonfinite = nonfinite and not transition.terminated_now
        if forced_nonfinite:
            # A nonfinite observation without a rules-side safety
            # termination still ends the episode with the unsafe
            # penalty; the guard means VecNormalize never sees it.
            rew_unsafe = -self.unsafe_physics_penalty
            reward = rew_return + rew_fault + rew_unsafe

        terminated = bool(after.terminated or nonfinite)
        truncated = bool(
            not terminated and self.step_number >= self.episode_len
        )
        info = self._build_info(
            transition,
            rew_return=rew_return,
            rew_fault=rew_fault,
            rew_unsafe=rew_unsafe,
            truncated=truncated,
            forced_nonfinite=forced_nonfinite,
        )
        return obs, float(reward), terminated, truncated, info

    def _nonfinite_termination(self):
        """End the episode without physics on a nonfinite action/state."""
        self.step_number += 1
        # Route the failure through the rules machine exactly as the
        # sampler would have (the humanoid env's convention): the info
        # stream keeps its full ``to_info`` schema and the termination
        # reason reads ``nonfinite_state`` from every key.
        snapshot = self._rules.snapshot()
        transition = self._rules.advance(
            [
                RallyEvent(
                    kind=RallyEventKind.NONFINITE_STATE,
                    substep=max(0, snapshot.last_event_substep + 1),
                )
            ]
        )
        self._last_transition = transition
        obs = self._record_or_echo_observation(self._get_obs(), True)
        rew_unsafe = -self.unsafe_physics_penalty
        info = self._build_info(
            transition,
            rew_return=0.0,
            rew_fault=0.0,
            rew_unsafe=rew_unsafe,
            truncated=False,
            forced_nonfinite=True,
        )
        return obs, float(rew_unsafe), True, False, info

    def _build_info(
        self,
        transition: RallyTransition | None,
        *,
        rew_return: float,
        rew_fault: float,
        rew_unsafe: float,
        truncated: bool,
        forced_nonfinite: bool = False,
    ) -> dict[str, Any]:
        if transition is not None:
            info = transition.to_info()
            reason = transition.after.termination_reason
        else:
            info = {}
            reason = TerminationReason.NONE
        if forced_nonfinite:
            reason = TerminationReason.NONFINITE_STATE
        info.update(
            {
                "crossings": self._crossings,
                "serve_side_is_policy": (
                    1.0 if self._serving_side is CourtSide.A else 0.0
                ),
                "rew_return": rew_return,
                "rew_fault": rew_fault,
                "rew_unsafe": rew_unsafe,
                "term_timeout": 1.0 if truncated else 0.0,
            }
        )
        for name, reasons in _TERM_GROUPS:
            info[name] = 1.0 if reason in reasons else 0.0
        return info

    # -- reset -------------------------------------------------------------

    def reset_model(self):
        self.step_number = 0
        self._crossings = 0
        self._last_transition = None
        self._serving_side = self._next_serving_side
        self._next_serving_side = self._serving_side.opponent

        serve = self.serve_config
        rng = self.np_random
        position_noise = rng.uniform(
            low=-np.asarray(serve.position_noise),
            high=np.asarray(serve.position_noise),
        )
        speed = float(
            rng.uniform(
                serve.speed - serve.speed_noise,
                serve.speed + serve.speed_noise,
            )
        )
        elevation = np.radians(
            rng.uniform(
                serve.elevation_degrees - serve.elevation_noise_degrees,
                serve.elevation_degrees + serve.elevation_noise_degrees,
            )
        )
        lateral = np.radians(
            rng.uniform(
                serve.lateral_degrees - serve.lateral_noise_degrees,
                serve.lateral_degrees + serve.lateral_noise_degrees,
            )
        )
        position = (
            np.array(
                [
                    -serve.start_distance_from_net,
                    serve.lateral_position,
                    serve.height,
                ]
            )
            + position_noise
        )
        horizontal = speed * np.cos(elevation)
        velocity = np.array(
            [
                horizontal * np.cos(lateral),
                horizontal * np.sin(lateral),
                speed * np.sin(elevation),
            ]
        )
        mirror_for_side(position, self._serving_side)
        mirror_for_side(velocity, self._serving_side)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[self._ball_qposadr : self._ball_qposadr + 3] = position
        qpos[self._ball_qposadr + 3 : self._ball_qposadr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        qvel[self._ball_dofadr : self._ball_dofadr + 6] = 0.0
        qvel[self._ball_dofadr : self._ball_dofadr + 3] = velocity
        self.set_state(qpos, qvel)

        self._rules = RallyStateMachine(
            serving_side=self._serving_side, court=PADDLE_COURT
        )
        self._event_sampler.reset(
            self.data, ball_side=self._serving_side
        )
        self._latest_event_batch = None
        self._last_serve_state = (position.copy(), velocity.copy())
        self._refresh_court_markers()

        observation = self._get_obs()
        if not bool(np.isfinite(observation).all()):
            raise RuntimeError("reset produced a non-finite observation")
        self._remember_finite_observation(observation)
        return observation

    def _get_reset_info(self) -> dict[str, Any]:
        assert self._last_serve_state is not None
        position, velocity = self._last_serve_state
        return {
            "serve_side": self._serving_side.label,
            "serve_side_is_policy": (
                1.0 if self._serving_side is CourtSide.A else 0.0
            ),
            # Full initial ball state, so recordings and audits can
            # reproduce the exact serve (the humanoid contract, and the
            # wall-ball review lesson: serve draws must be recoverable
            # from the info stream).
            "serve_ball_position": tuple(float(v) for v in position),
            "serve_ball_velocity": tuple(float(v) for v in velocity),
        }

    #: Human-readable labels matching the observation vector.
    observation_names: tuple[str, ...] = PADDLE_TENNIS_OBSERVATION_NAMES
    action_names: tuple[str, ...] = PADDLE_TENNIS_ACTION_NAMES


__all__ = [
    "PADDLE_TENNIS_ACTION_NAMES",
    "PADDLE_TENNIS_NORMALIZED_SLICE",
    "PADDLE_TENNIS_OBSERVATION_NAMES",
    "PaddleCourtServe",
    "PaddleTennisEnv",
]
