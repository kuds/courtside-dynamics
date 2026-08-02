"""Centralized cooperative two-humanoid tennis environment.

The environment deliberately presents one ordinary Gymnasium interface: one
58-value action vector, one centralized observation, one shared reward, and
one termination/truncation pair.  It is an environment-correctness and
curriculum milestone, not a claim that unconstrained two-G1 tennis is
trainable from scratch with a stock PPO or SAC configuration.

Action values are normalized position targets.  Zero holds both Unitree G1s
at their registry standing references; each half of the vector is mapped
piecewise to that player's asymmetric actuator ranges.  The public slices are
available through :attr:`action_layout` and :attr:`action_names`.

The observation layout is fixed at 299 values.  Besides both players'
proprioception and the physical racket/ball state, it includes every hidden
rally-state and contact-latch value that can affect the next event.  This
keeps later curriculum stages from silently changing the registered
observation contract when joints or a scripted partner are masked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import (
    CourtsideMujocoEnv,
    piecewise_targets,
)
from courtside_dynamics.envs._tennis_events import (
    TENNIS_CONTACT_CHANNELS,
    TENNIS_CONTACT_MARKOV_LABELS,
    SubstepTennisEventSampler,
    TennisEventSamplerConfig,
    TennisStepEventBatch,
)
from courtside_dynamics.envs._tennis_physics import (
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
    ball_drag_force,
)
from courtside_dynamics.envs.robot_models import (
    UNITREE_G1_ACTION_LAYOUT,
    UNITREE_G1_ACTUATOR_NAMES,
    UNITREE_G1_STAND_JOINT_POSITIONS,
    CentralizedActionLayout,
    get_robot_model_spec,
    initialize_humanoid_tennis_home,
)
from courtside_dynamics.envs.tennis_curriculum import (
    CurriculumConfig,
    CurriculumStage,
    LearnedPlayerMode,
    MobilityMode,
    PartnerMode,
    ServeMode,
    StageObjective,
    get_curriculum_config,
    get_curriculum_preset,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
    RallyPhase,
    RallyRules,
    RallySnapshot,
    RallyStateMachine,
    RallyTransition,
    TerminationReason,
)


@dataclass(frozen=True, slots=True)
class TennisServeConfig:
    """Seeded initial-feed distribution in side-A-local court coordinates.

    Side B is the exact 180-degree court rotation of side A.  The default ball
    starts 1.635 m inside the baseline, just courtward of the serving-side G1;
    a centered feed launched from behind that robot would immediately hit its
    torso rather than enter the rally.
    """

    start_distance_from_net: float = 10.25
    lateral_position: float = 0.0
    height: float = 1.3
    local_velocity: tuple[float, float, float] = (16.0, 0.0, 4.5)
    position_noise: tuple[float, float, float] = (0.1, 0.25, 0.1)
    velocity_noise: tuple[float, float, float] = (0.5, 0.25, 0.25)
    local_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name, values in (
            ("local_velocity", self.local_velocity),
            ("position_noise", self.position_noise),
            ("velocity_noise", self.velocity_noise),
            ("local_angular_velocity", self.local_angular_velocity),
        ):
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly three values")
        scalar_values = (
            self.start_distance_from_net,
            self.lateral_position,
            self.height,
            *self.local_velocity,
            *self.position_noise,
            *self.velocity_noise,
            *self.local_angular_velocity,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("serve configuration must contain only finite values")
        if not 0.0 < self.start_distance_from_net <= COURT_HALF_LENGTH:
            raise ValueError("start_distance_from_net must lie on one court half")
        if self.height <= 0.0:
            raise ValueError("serve height must be positive")
        if any(value < 0.0 for value in self.position_noise):
            raise ValueError("position_noise values must be non-negative")
        if any(value < 0.0 for value in self.velocity_noise):
            raise ValueError("velocity_noise values must be non-negative")
        if self.local_velocity[0] <= self.velocity_noise[0]:
            raise ValueError(
                "forward serve velocity must remain positive across its noise range"
            )
        if self.start_distance_from_net <= self.position_noise[0]:
            raise ValueError(
                "longitudinal position noise must keep the ball on its serving side"
            )
        if abs(self.lateral_position) + self.position_noise[1] > COURT_HALF_WIDTH:
            raise ValueError("serve lateral position/noise must start in bounds")
        if not (
            1.1 <= self.height - self.position_noise[2]
            and self.height + self.position_noise[2] <= 1.5
        ):
            raise ValueError(
                "serve height and noise must keep reset height within 1.1–1.5 m"
            )


@dataclass(frozen=True, slots=True)
class TennisRewardConfig:
    """Shared cooperative reward, with transactional optional hit shaping."""

    valid_return_reward: float = 1.0
    valid_hit_shaping: float = 0.0
    fault_penalty: float = 1.0
    unsafe_physics_penalty: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.valid_return_reward,
            self.valid_hit_shaping,
            self.fault_penalty,
            self.unsafe_physics_penalty,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("reward values must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CentralizedObservationLayout:
    """Stable slices of the centralized observation vector."""

    player_a_proprio: slice
    player_b_proprio: slice
    player_a_racket: slice
    player_b_racket: slice
    ball: slice
    relative: slice
    rally_state: slice
    contact_latches: slice
    contact_release_progress: slice
    active_action_mask: slice
    total_size: int


HUMANOID_TENNIS_ACTION_NAMES: Final = tuple(
    f"player_a_{name}" for name in UNITREE_G1_ACTUATOR_NAMES
) + tuple(f"player_b_{name}" for name in UNITREE_G1_ACTUATOR_NAMES)


def _proprioception_names(player: str) -> tuple[str, ...]:
    return (
        *(f"{player}_base_position_{axis}" for axis in "xyz"),
        *(f"{player}_base_quaternion_{axis}" for axis in ("w", "x", "y", "z")),
        *(f"{player}_{name}_position" for name in UNITREE_G1_ACTUATOR_NAMES),
        *(f"{player}_base_linear_velocity_{axis}" for axis in "xyz"),
        *(f"{player}_base_angular_velocity_{axis}" for axis in "xyz"),
        *(f"{player}_{name}_velocity" for name in UNITREE_G1_ACTUATOR_NAMES),
    )


def _racket_names(player: str) -> tuple[str, ...]:
    return (
        *(f"{player}_racket_center_{axis}" for axis in "xyz"),
        *(f"{player}_racket_normal_{axis}" for axis in "xyz"),
        *(f"{player}_racket_long_axis_{axis}" for axis in "xyz"),
        *(f"{player}_racket_linear_velocity_{axis}" for axis in "xyz"),
        *(f"{player}_racket_angular_velocity_{axis}" for axis in "xyz"),
    )


_BALL_OBSERVATION_NAMES: Final = (
    *(f"ball_position_{axis}" for axis in "xyz"),
    *(f"ball_linear_velocity_{axis}" for axis in "xyz"),
    *(f"ball_angular_velocity_{axis}" for axis in "xyz"),
)

_RELATIVE_OBSERVATION_NAMES: Final = tuple(
    f"ball_minus_{target}_{axis}"
    for target in (
        "player_a_pelvis",
        "player_a_racket",
        "player_b_pelvis",
        "player_b_racket",
    )
    for axis in "xyz"
)

_RALLY_OBSERVATION_NAMES: Final = (
    *(f"rally_phase_{phase.name.lower()}" for phase in RallyPhase),
    "serving_side_a",
    "serving_side_b",
    "expected_returner_a",
    "expected_returner_b",
    "ball_side_a",
    "ball_side_b",
    "last_valid_hitter_none",
    "last_valid_hitter_a",
    "last_valid_hitter_b",
    "pending_return_hitter_none",
    "pending_return_hitter_a",
    "pending_return_hitter_b",
    "bounce_side_none",
    "bounce_side_a",
    "bounce_side_b",
    "pending_return_crossed_net",
    "feed_crossed_net",
    "bounce_count",
    "shot_crossing_count",
    "rally_count",
    "rule_ball_net_is_fault",
    "rule_strict_reverse_crossing",
    "rule_strict_double_hit",
    "episode_remaining_fraction",
)

HUMANOID_TENNIS_OBSERVATION_NAMES: Final = (
    _proprioception_names("player_a")
    + _proprioception_names("player_b")
    + _racket_names("player_a")
    + _racket_names("player_b")
    + _BALL_OBSERVATION_NAMES
    + _RELATIVE_OBSERVATION_NAMES
    + _RALLY_OBSERVATION_NAMES
    + TENNIS_CONTACT_MARKOV_LABELS[: len(TENNIS_CONTACT_CHANNELS)]
    + TENNIS_CONTACT_MARKOV_LABELS[len(TENNIS_CONTACT_CHANNELS) :]
    + tuple(f"active_action_{name}" for name in HUMANOID_TENNIS_ACTION_NAMES)
)

HUMANOID_TENNIS_OBSERVATION_LAYOUT: Final = CentralizedObservationLayout(
    player_a_proprio=slice(0, 71),
    player_b_proprio=slice(71, 142),
    player_a_racket=slice(142, 157),
    player_b_racket=slice(157, 172),
    ball=slice(172, 181),
    relative=slice(181, 193),
    rally_state=slice(193, 221),
    contact_latches=slice(221, 231),
    contact_release_progress=slice(231, 241),
    active_action_mask=slice(241, 299),
    total_size=299,
)

if len(HUMANOID_TENNIS_OBSERVATION_NAMES) != 299:
    raise RuntimeError(
        "humanoid-tennis observation labels drifted from the registered API"
    )


class HumanoidTennisCoopEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Cooperate as two physical G1 humanoids to sustain a legal rally."""

    action_layout: Final[CentralizedActionLayout] = UNITREE_G1_ACTION_LAYOUT
    action_names: Final = HUMANOID_TENNIS_ACTION_NAMES
    observation_layout: Final = HUMANOID_TENNIS_OBSERVATION_LAYOUT
    observation_names: Final = HUMANOID_TENNIS_OBSERVATION_NAMES

    def __init__(
        self,
        robot_model: str = "unitree_g1",
        episode_len: int = 1_000,
        frame_skip: int = 10,
        initial_serve_side: CourtSide | int | str = CourtSide.A,
        serve_config: TennisServeConfig | None = None,
        reward_config: TennisRewardConfig | None = None,
        rally_rules: RallyRules | None = None,
        sampler_config: TennisEventSamplerConfig | None = None,
        rally_target: int | None = None,
        curriculum_config: (
            CurriculumConfig | CurriculumStage | int | str | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        initial_side = self._coerce_side(initial_serve_side)
        resolved_curriculum = self._resolve_curriculum(curriculum_config)
        resolved_serve = serve_config or TennisServeConfig()
        resolved_reward = reward_config or TennisRewardConfig()
        resolved_rules = rally_rules or (
            resolved_curriculum.rally_rules()
            if resolved_curriculum is not None
            else RallyRules()
        )
        resolved_sampler = sampler_config or TennisEventSamplerConfig()
        if rally_target is None:
            resolved_rally_target = (
                resolved_curriculum.rally_target
                if resolved_curriculum is not None
                else 1
            )
        else:
            resolved_rally_target = int(rally_target)
        if episode_len < 1:
            raise ValueError("episode_len must be at least one")
        if frame_skip != 10:
            raise ValueError(
                "frame_skip must be 10 for the 0.001 s model and 100 fps API"
            )
        if resolved_rally_target < 1:
            raise ValueError("rally_target must be at least one")

        utils.EzPickle.__init__(
            self,
            robot_model=robot_model,
            episode_len=episode_len,
            frame_skip=frame_skip,
            initial_serve_side=initial_side,
            serve_config=serve_config,
            reward_config=reward_config,
            rally_rules=rally_rules,
            sampler_config=sampler_config,
            rally_target=rally_target,
            curriculum_config=resolved_curriculum,
            **kwargs,
        )

        self.robot_model = robot_model
        self.robot_spec = get_robot_model_spec(robot_model)
        self.serve_config = resolved_serve
        self.reward_config = resolved_reward
        self.rally_rules = resolved_rules
        self.sampler_config = resolved_sampler
        self.rally_target = resolved_rally_target
        self.curriculum_config = resolved_curriculum
        self._curriculum_launch_overridden = (
            resolved_curriculum is not None and serve_config is not None
        )
        self.initial_serve_side = initial_side
        self._next_serve_side = initial_side
        self._pending_reset_options: dict[str, Any] = {}
        self._last_reset_info: dict[str, Any] = {}

        CourtsideMujocoEnv.__init__(
            self,
            asset_path("humanoid_tennis.xml"),
            episode_len=episode_len,
            obs_dim=self.observation_layout.total_size,
            frame_skip=frame_skip,
            **kwargs,
        )
        if (self.model.nq, self.model.nv, self.model.nu) != (79, 76, 58):
            raise ValueError(
                "unitree_g1 composite dimensions changed: expected "
                "nq=79, nv=76, nu=58"
            )
        compiled_actuators = tuple(
            self.model.actuator(index).name for index in range(self.model.nu)
        )
        if compiled_actuators != self.action_names:
            raise ValueError("compiled actuator order differs from the public API")

        # MujocoEnv exposes raw actuator ranges by default.  The centralized
        # policy API is normalized so zero is a meaningful two-player PD hold.
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_layout.total_size,),
            dtype=np.float32,
        )
        self._active_action_mask = np.ones(self.model.nu, dtype=np.float64)
        self._stand_controls = np.asarray(
            UNITREE_G1_STAND_JOINT_POSITIONS * 2,
            dtype=np.float64,
        )
        self._control_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._control_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._action_name_to_index = {
            name: index for index, name in enumerate(self.action_names)
        }
        if not (
            np.all(self._control_low <= self._stand_controls)
            and np.all(self._stand_controls <= self._control_high)
        ):
            raise ValueError("standing references must lie inside actuator ranges")

        self._player_qpos_slices: dict[CourtSide, slice] = {}
        self._player_qvel_slices: dict[CourtSide, slice] = {}
        for side, player in ((CourtSide.A, "player_a"), (CourtSide.B, "player_b")):
            base_joint = self.model.joint(f"{player}_floating_base_joint")
            qpos_start = int(base_joint.qposadr[0])
            qvel_start = int(base_joint.dofadr[0])
            self._player_qpos_slices[side] = slice(
                qpos_start, qpos_start + self.robot_spec.nq
            )
            self._player_qvel_slices[side] = slice(
                qvel_start, qvel_start + self.robot_spec.nv
            )

        ball_joint = self.model.joint("ball_free")
        self._ball_qposadr = int(ball_joint.qposadr[0])
        self._ball_dofadr = int(ball_joint.dofadr[0])
        self._ball_body_id = int(self.model.body("ball").id)
        self._pelvis_body_ids = {
            CourtSide.A: int(self.model.body("player_a_pelvis").id),
            CourtSide.B: int(self.model.body("player_b_pelvis").id),
        }
        self._racket_body_ids = {
            CourtSide.A: int(self.model.body("player_a_racket_tennis_racket").id),
            CourtSide.B: int(self.model.body("player_b_racket_tennis_racket").id),
        }
        self._racket_site_ids = {
            CourtSide.A: int(self.model.site("player_a_racket_racket_center").id),
            CourtSide.B: int(self.model.site("player_b_racket_racket_center").id),
        }
        self._racket_stringbed_geom_ids = {
            CourtSide.A: int(
                self.model.geom("player_a_racket_racket_stringbed").id
            ),
            CourtSide.B: int(
                self.model.geom("player_b_racket_racket_stringbed").id
            ),
        }
        self._pelvis_weld_ids = {
            CourtSide.A: int(
                self.model.equality("player_a_pelvis_curriculum_weld").id
            ),
            CourtSide.B: int(
                self.model.equality("player_b_pelvis_curriculum_weld").id
            ),
        }
        self._nominal_stringbed_sizes = {
            side: self.model.geom_size[geom_id].copy()
            for side, geom_id in self._racket_stringbed_geom_ids.items()
        }
        if self.curriculum_config is not None:
            scale = self.curriculum_config.racket_contact_scale
            for side, geom_id in self._racket_stringbed_geom_ids.items():
                nominal = self._nominal_stringbed_sizes[side]
                # Keep thickness, rigid-body mass, and inertia unchanged;
                # only the physical stringbed's in-plane reach is forgiven.
                self.model.geom_size[geom_id, 0] = nominal[0]
                self.model.geom_size[geom_id, 1:3] = nominal[1:3] * scale
                # Neither the size write nor mj_setConst refreshes the
                # compile-time collision bounds, and MuJoCo's broadphase
                # culls candidate pairs on them -- without this, contacts
                # in the enlarged outer band are silently dropped and the
                # forgiveness is inert. For an ellipsoid the bounding
                # sphere radius is the largest semi-axis and the
                # geom-frame AABB half-sizes are the semi-axes.
                self.model.geom_rbound[geom_id] = float(
                    np.max(self.model.geom_size[geom_id])
                )
                self.model.geom_aabb[geom_id, 3:6] = self.model.geom_size[
                    geom_id
                ]
            mujoco.mj_setConst(self.model, self.data)

        self._state_machine = RallyStateMachine(
            serving_side=initial_side,
            rules=self.rally_rules,
        )
        self._event_sampler = SubstepTennisEventSampler(
            self.model,
            config=self.sampler_config,
        )
        self._latest_event_batch = TennisStepEventBatch(
            control_step=0,
            substeps_sampled=0,
            events=(),
            contact_peaks={},
            active_contact_latches=frozenset(),
        )
        self._pending_shaping = {CourtSide.A: 0.0, CourtSide.B: 0.0}
        self._learned_returner: CourtSide | None = None
        self._stage_success = False
        self._stage_attempt_complete = False
        self._target_hit = False
        self._target_miss = False

    @property
    def neutral_action(self) -> np.ndarray:
        """Return the normalized action that PD-holds both standing poses."""
        return np.zeros(
            (self.action_layout.total_size,),
            dtype=np.float32,
        )

    @property
    def active_action_mask(self) -> np.ndarray:
        """Return a copy of the current policy-controlled action mask."""
        return self._active_action_mask.astype(np.float32, copy=True)

    @property
    def curriculum_metadata(self) -> dict[str, Any]:
        """Return fixed-stage provenance suitable for run artifacts."""
        config = self.curriculum_config
        if config is None:
            return {"curriculum_enabled": False}
        return {
            "curriculum_enabled": True,
            "curriculum_stage": int(config.stage),
            "curriculum_stage_name": config.name,
            "curriculum_matches_canonical_preset": (
                config == get_curriculum_preset(config.stage).config
            ),
            "robot_model": self.robot_model,
            "episode_len": self.episode_len,
            "frame_skip": self.frame_skip,
            "model_timestep": float(self.model.opt.timestep),
            "control_timestep": float(self.dt),
            "action_size": self.action_layout.total_size,
            "observation_size": self.observation_layout.total_size,
            "curriculum_objective": config.objective.value,
            "curriculum_mobility": config.mobility.value,
            "learned_player_mode": config.learned_players.value,
            "partner_mode": config.partner_mode.value,
            "launch_mode": config.launch.mode.value,
            "launch_start_distance_from_net": (
                config.launch.start_distance_from_net
            ),
            "launch_lateral_position": config.launch.lateral_position,
            "launch_height": config.launch.height,
            "launch_speed": config.launch.speed,
            "launch_elevation_degrees": config.launch.elevation_degrees,
            "launch_lateral_degrees": config.launch.lateral_degrees,
            "launch_position_noise": config.launch.position_noise,
            "launch_speed_noise": config.launch.speed_noise,
            "launch_elevation_noise_degrees": (
                config.launch.elevation_noise_degrees
            ),
            "launch_lateral_noise_degrees": (
                config.launch.lateral_noise_degrees
            ),
            "launch_angular_velocity": config.launch.angular_velocity,
            "racket_contact_scale": config.racket_contact_scale,
            "receiver_distance_from_net": config.receiver_distance_from_net,
            "playable_half_length": config.court.playable_half_length,
            "playable_half_width": config.court.playable_half_width,
            "target_min_depth": config.court.target_min_depth,
            "target_max_depth": config.court.target_max_depth,
            "target_center_y": config.court.target_center_y,
            "target_half_width": config.court.target_half_width,
            "target_miss_is_terminal": config.court.target_miss_is_terminal,
            "rally_target": self.rally_target,
            "fault_strictness": config.fault_strictness.value,
            "effective_ball_net_is_fault": self.rally_rules.ball_net_is_fault,
            "effective_strict_reverse_crossing": (
                self.rally_rules.strict_reverse_crossing
            ),
            "effective_strict_double_hit": self.rally_rules.strict_double_hit,
            "sampler_min_contact_force": self.sampler_config.min_contact_force,
            "sampler_release_substeps": self.sampler_config.release_substeps,
            "sampler_crossing_deadband": self.sampler_config.crossing_deadband,
            "safety_max_ball_linear_speed": (
                self.sampler_config.safety.max_ball_linear_speed
            ),
            "safety_max_abs_qvel": self.sampler_config.safety.max_abs_qvel,
            "safety_max_abs_qacc": self.sampler_config.safety.max_abs_qacc,
            "stage_success_reward": config.stage_success_reward,
            "terminate_on_success": config.terminate_on_success,
            "valid_return_reward": self.reward_config.valid_return_reward,
            "valid_hit_shaping": self.reward_config.valid_hit_shaping,
            "fault_penalty": self.reward_config.fault_penalty,
            "unsafe_physics_penalty": (
                self.reward_config.unsafe_physics_penalty
            ),
            "active_joint_names": config.active_joint_names,
            "pd_hold_joint_names": config.pd_hold_joint_names,
            "curriculum_launch_overridden": self._curriculum_launch_overridden,
            "override_serve_start_distance_from_net": (
                self.serve_config.start_distance_from_net
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_lateral_position": (
                self.serve_config.lateral_position
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_height": (
                self.serve_config.height
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_local_velocity": (
                self.serve_config.local_velocity
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_position_noise": (
                self.serve_config.position_noise
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_velocity_noise": (
                self.serve_config.velocity_noise
                if self._curriculum_launch_overridden
                else None
            ),
            "override_serve_angular_velocity": (
                self.serve_config.local_angular_velocity
                if self._curriculum_launch_overridden
                else None
            ),
        }

    @property
    def rally_state(self) -> RallySnapshot:
        """Return an immutable snapshot for diagnostics and scripted harnesses."""
        return self._state_machine.snapshot()

    @staticmethod
    def _coerce_side(value: CourtSide | int | str) -> CourtSide:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"a", "player_a"}:
                return CourtSide.A
            if normalized in {"b", "player_b"}:
                return CourtSide.B
            raise ValueError("serve side must be 'a' or 'b'")
        return CourtSide(value)

    @staticmethod
    def _resolve_curriculum(
        value: CurriculumConfig | CurriculumStage | int | str | None,
    ) -> CurriculumConfig | None:
        if value is None:
            return None
        config = (
            value
            if isinstance(value, CurriculumConfig)
            else get_curriculum_config(value)
        )
        if config.stage in {
            CurriculumStage.CONSTRAINED_STANDING_RETURN,
            CurriculumStage.MOBILE_WITH_SCRIPTED_PARTNER,
            CurriculumStage.TWO_CONSTRAINED_LEARNED,
        }:
            # Custom configs cannot silently turn a planned stage into a
            # claimed implementation. Those stages need dedicated physical
            # constraints/controllers and validation first.
            get_curriculum_config(config.stage, require_available=True)
        if config.stage in {
            CurriculumStage.ANCHORED_RACKET_INTERCEPT,
            CurriculumStage.ANCHORED_RETURN,
            CurriculumStage.ANCHORED_RANDOMIZED_RETURN,
        } and config.mobility is not MobilityMode.FIXED_BASE:
            raise ValueError("implemented stages 0–2 require fixed-base mobility")
        if config.stage in {
            CurriculumStage.ANCHORED_RACKET_INTERCEPT,
            CurriculumStage.ANCHORED_RETURN,
            CurriculumStage.ANCHORED_RANDOMIZED_RETURN,
        } and config.partner_mode is not PartnerMode.PD_HOLD:
            raise ValueError("implemented stages 0–2 require a PD-held partner")
        if (
            config.stage is CurriculumStage.TWO_FREE_STANDING
            and config.mobility is not MobilityMode.FREE_STANDING
        ):
            raise ValueError("stage 6 requires free-standing mobility")
        if config.stage is CurriculumStage.TWO_FREE_STANDING and (
            config.learned_players is not LearnedPlayerMode.BOTH
            or config.partner_mode is not PartnerMode.LEARNED
            or config.objective is not StageObjective.RALLY_TARGET
        ):
            raise ValueError(
                "stage 6 requires two learned players and a rally-target objective"
            )
        return config

    def _action_to_controls(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float64)
        if normalized.shape != (self.model.nu,):
            raise ValueError(
                f"action must have shape {(self.model.nu,)}, got {normalized.shape}"
            )
        if not bool(np.isfinite(normalized).all()):
            raise ValueError("action must contain only finite values")
        normalized = np.clip(normalized, -1.0, 1.0) * self._active_action_mask
        return piecewise_targets(
            normalized,
            self._control_low,
            self._stand_controls,
            self._control_high,
        )

    def _configure_action_mask(self, serving_side: CourtSide) -> None:
        """Select fixed-size policy slices for this episode's learned side."""
        config = self.curriculum_config
        if config is None:
            self._learned_returner = None
            self._active_action_mask.fill(1.0)
            return

        self._active_action_mask.fill(0.0)
        learned_sides: tuple[CourtSide, ...]
        if config.learned_players is LearnedPlayerMode.BOTH:
            learned_sides = (CourtSide.A, CourtSide.B)
            self._learned_returner = None
        else:
            self._learned_returner = serving_side.opponent
            learned_sides = (self._learned_returner,)
        for side in learned_sides:
            prefix = "player_a" if side is CourtSide.A else "player_b"
            for joint_name in config.active_joint_names:
                index = self._action_name_to_index[f"{prefix}_{joint_name}"]
                self._active_action_mask[index] = 1.0

    def _configure_curriculum_mobility(self, serving_side: CourtSide) -> None:
        """Place/anchor curriculum robots once, before simulation begins."""
        for equality_id in self._pelvis_weld_ids.values():
            self.data.eq_active[equality_id] = 0

        config = self.curriculum_config
        if config is None or config.mobility is MobilityMode.FREE_STANDING:
            return
        if config.mobility is not MobilityMode.FIXED_BASE:
            raise NotImplementedError(
                "foot-constrained humanoid mobility is not implemented"
            )

        receiver = serving_side.opponent
        receiver_qpos = self.data.qpos[self._player_qpos_slices[receiver]]
        receiver_qvel = self.data.qvel[self._player_qvel_slices[receiver]]
        receiver_qpos[0] = (
            -config.receiver_distance_from_net
            if receiver is CourtSide.A
            else config.receiver_distance_from_net
        )
        receiver_qpos[1] = 0.0
        receiver_qvel[:6] = 0.0

        # Both bases are physically welded for anchored one-player stages.
        # The non-learned G1 is an irrelevant PD-held feed-side partner; its
        # anchor prevents a balance failure from contaminating the arm task.
        for side, equality_id in self._pelvis_weld_ids.items():
            player_qpos = self.data.qpos[self._player_qpos_slices[side]]
            equality_data = self.model.eq_data[equality_id]
            equality_data[0:3] = player_qpos[0:3]
            equality_data[3:6] = 0.0
            equality_data[6:10] = player_qpos[3:7]
            self.data.eq_active[equality_id] = 1

    def action_for_control_targets(self, targets: Mapping[str, float]) -> np.ndarray:
        """Build a normalized action for selected named actuator targets.

        Unspecified actuators remain at the standing reference.  This helper is
        intended for deterministic physical oracles and future curriculum
        controllers; it does not mutate simulation state.
        """
        action = np.zeros(self.model.nu, dtype=np.float64)
        name_to_index = {name: index for index, name in enumerate(self.action_names)}
        for name, target_value in targets.items():
            try:
                index = name_to_index[name]
            except KeyError as error:
                raise ValueError(f"unknown actuator target {name!r}") from error
            target = float(target_value)
            if not math.isfinite(target):
                raise ValueError("control targets must be finite")
            if not self._control_low[index] <= target <= self._control_high[index]:
                raise ValueError(f"control target for {name} lies outside ctrlrange")
            stand = self._stand_controls[index]
            if target >= stand:
                span = self._control_high[index] - stand
            else:
                span = stand - self._control_low[index]
            action[index] = 0.0 if span == 0.0 else (target - stand) / span
        return action.astype(self.action_space.dtype)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset a seeded alternating-serve sequence.

        Passing a new seed restarts serve alternation at
        ``initial_serve_side``.  Normal episode resets omit ``seed`` and
        alternate thereafter.  Reset-only ``options`` support the physical
        scripted oracle without any mid-rally state write:

        ``serve_side`` (``"a"``/``"b"``), ``ball_position`` (world xyz),
        ``ball_velocity`` (world linear xyz), and ``joint_positions`` (a
        mapping of fully-prefixed one-DoF joint names to reset positions).
        """
        if seed is not None:
            self._next_serve_side = self.initial_serve_side
        self._pending_reset_options = self._validate_reset_options(options or {})
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self._pending_reset_options = {}

    def _validate_reset_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"serve_side", "ball_position", "ball_velocity", "joint_positions"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        validated: dict[str, Any] = {}
        if "serve_side" in options:
            validated["serve_side"] = self._coerce_side(options["serve_side"])
        for key in ("ball_position", "ball_velocity"):
            if key not in options:
                continue
            values = np.asarray(options[key], dtype=np.float64)
            if values.shape != (3,) or not bool(np.isfinite(values).all()):
                raise ValueError(f"{key} must be a finite three-vector")
            validated[key] = values.copy()
        if "joint_positions" in options:
            positions = options["joint_positions"]
            if not isinstance(positions, Mapping):
                raise ValueError("joint_positions must be a mapping")
            validated_positions: dict[str, float] = {}
            for name, value in positions.items():
                if not isinstance(name, str):
                    raise ValueError("joint position names must be strings")
                try:
                    joint = self.model.joint(name)
                except KeyError as error:
                    raise ValueError(f"unknown reset joint {name!r}") from error
                if int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("only one-DoF hinge joints may be overridden")
                target = float(value)
                if not math.isfinite(target):
                    raise ValueError("joint reset positions must be finite")
                if bool(joint.limited[0]) and not (
                    float(joint.range[0]) <= target <= float(joint.range[1])
                ):
                    raise ValueError(f"reset position for {name} lies outside range")
                validated_positions[name] = target
            validated["joint_positions"] = validated_positions
        return validated

    def reset_model(self) -> np.ndarray:
        self.step_number = 0
        options = self._pending_reset_options
        serving_side = options.get("serve_side", self._next_serve_side)
        serving_side = CourtSide(serving_side)
        self._next_serve_side = serving_side.opponent

        initialize_humanoid_tennis_home(
            self.model,
            self.data,
            robot_model=self.robot_model,
        )
        self._configure_action_mask(serving_side)
        self._configure_curriculum_mobility(serving_side)
        for name, target in options.get("joint_positions", {}).items():
            joint = self.model.joint(name)
            self.data.qpos[int(joint.qposadr[0])] = target
            try:
                actuator = self.model.actuator(name)
            except KeyError:
                continue
            self.data.ctrl[int(actuator.id)] = target

        position, velocity = self._sample_initial_ball_state(serving_side)
        if "ball_position" in options:
            position = options["ball_position"].copy()
        if "ball_velocity" in options:
            velocity = options["ball_velocity"].copy()
        if CourtSide.from_x(float(position[0]), tie_breaker=serving_side) is not serving_side:
            raise ValueError("reset ball position must lie on the serving side")
        if velocity[0] == 0.0 or math.copysign(1.0, float(velocity[0])) != (
            1.0 if serving_side is CourtSide.A else -1.0
        ):
            raise ValueError("reset ball velocity must point toward the other side")

        ball_qpos = self.data.qpos[self._ball_qposadr : self._ball_qposadr + 7]
        ball_qvel = self.data.qvel[self._ball_dofadr : self._ball_dofadr + 6]
        ball_qpos[:] = (*position, 1.0, 0.0, 0.0, 0.0)
        ball_qvel[:] = (*velocity, *self._world_angular_velocity(serving_side))
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._state_machine.reset(serving_side)
        self._event_sampler.reset(self.data, ball_side=serving_side)
        self._latest_event_batch = TennisStepEventBatch(
            control_step=0,
            substeps_sampled=0,
            events=(),
            contact_peaks={},
            active_contact_latches=frozenset(),
        )
        self._pending_shaping = {CourtSide.A: 0.0, CourtSide.B: 0.0}
        self._stage_success = False
        self._stage_attempt_complete = False
        self._target_hit = False
        self._target_miss = False
        self._last_reset_info = self._build_initial_state_info(serving_side)

        observation = self._get_obs()
        if not bool(np.isfinite(observation).all()):
            raise RuntimeError("reset produced a non-finite observation")
        self._remember_finite_observation(observation)
        return observation

    def _sample_initial_ball_state(
        self,
        serving_side: CourtSide,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.curriculum_config is not None
            and not self._curriculum_launch_overridden
        ):
            return self._sample_curriculum_launch(serving_side)
        config = self.serve_config
        position_noise = self.np_random.uniform(
            low=-np.asarray(config.position_noise),
            high=np.asarray(config.position_noise),
        )
        velocity_noise = self.np_random.uniform(
            low=-np.asarray(config.velocity_noise),
            high=np.asarray(config.velocity_noise),
        )
        canonical_position = np.array(
            [
                -config.start_distance_from_net,
                config.lateral_position,
                config.height,
            ],
            dtype=np.float64,
        ) + position_noise
        canonical_velocity = np.asarray(
            config.local_velocity,
            dtype=np.float64,
        ) + velocity_noise
        if serving_side is CourtSide.B:
            canonical_position[:2] *= -1.0
            canonical_velocity[:2] *= -1.0
        return canonical_position, canonical_velocity

    def _sample_curriculum_launch(
        self,
        serving_side: CourtSide,
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.curriculum_config
        assert config is not None
        launch = config.launch
        if launch.mode is ServeMode.RANDOMIZED:
            position_noise = self.np_random.uniform(
                low=-np.asarray(launch.position_noise),
                high=np.asarray(launch.position_noise),
            )
            speed = float(
                self.np_random.uniform(
                    launch.speed - launch.speed_noise,
                    launch.speed + launch.speed_noise,
                )
            )
            elevation = float(
                self.np_random.uniform(
                    launch.elevation_degrees - launch.elevation_noise_degrees,
                    launch.elevation_degrees + launch.elevation_noise_degrees,
                )
            )
            lateral = float(
                self.np_random.uniform(
                    launch.lateral_degrees - launch.lateral_noise_degrees,
                    launch.lateral_degrees + launch.lateral_noise_degrees,
                )
            )
        else:
            position_noise = np.zeros(3, dtype=np.float64)
            speed = launch.speed
            elevation = launch.elevation_degrees
            lateral = launch.lateral_degrees
        canonical_position = np.array(
            [
                -launch.start_distance_from_net,
                launch.lateral_position,
                launch.height,
            ],
            dtype=np.float64,
        ) + position_noise
        canonical_velocity = np.asarray(
            launch.velocity(
                speed=speed,
                elevation_degrees=elevation,
                lateral_degrees=lateral,
            ),
            dtype=np.float64,
        )
        if serving_side is CourtSide.B:
            canonical_position[:2] *= -1.0
            canonical_velocity[:2] *= -1.0
        return canonical_position, canonical_velocity

    def _world_angular_velocity(self, serving_side: CourtSide) -> np.ndarray:
        if (
            self.curriculum_config is not None
            and not self._curriculum_launch_overridden
        ):
            values = self.curriculum_config.launch.angular_velocity
        else:
            values = self.serve_config.local_angular_velocity
        angular = np.asarray(values, dtype=np.float64).copy()
        if serving_side is CourtSide.B:
            angular[:2] *= -1.0
        return angular

    def _build_initial_state_info(self, serving_side: CourtSide) -> dict[str, Any]:
        qpos = self.data.qpos[self._ball_qposadr : self._ball_qposadr + 7].copy()
        qvel = self.data.qvel[self._ball_dofadr : self._ball_dofadr + 6].copy()
        labels = (
            "initial_ball_x",
            "initial_ball_y",
            "initial_ball_z",
            "initial_ball_quat_w",
            "initial_ball_quat_x",
            "initial_ball_quat_y",
            "initial_ball_quat_z",
        )
        velocity_labels = (
            "initial_ball_vx",
            "initial_ball_vy",
            "initial_ball_vz",
            "initial_ball_wx",
            "initial_ball_wy",
            "initial_ball_wz",
        )
        info: dict[str, Any] = {
            "serve_side": int(serving_side),
            "serve_side_name": serving_side.label,
            "initial_ball_qpos": tuple(float(value) for value in qpos),
            "initial_ball_qvel": tuple(float(value) for value in qvel),
        }
        info.update(zip(labels, (float(value) for value in qpos), strict=True))
        info.update(
            zip(velocity_labels, (float(value) for value in qvel), strict=True)
        )
        return info

    def _curriculum_info(self) -> dict[str, Any]:
        config = self.curriculum_config
        if config is None:
            return {}
        player_a = self.action_layout.player_a
        player_b = self.action_layout.player_b
        return {
            **self.curriculum_metadata,
            "active_action_count": int(np.count_nonzero(self._active_action_mask)),
            "active_action_count_a": int(
                np.count_nonzero(self._active_action_mask[player_a])
            ),
            "active_action_count_b": int(
                np.count_nonzero(self._active_action_mask[player_b])
            ),
            "pd_held_action_count": int(
                self.model.nu - np.count_nonzero(self._active_action_mask)
            ),
            "learned_returner": (
                -1 if self._learned_returner is None else int(self._learned_returner)
            ),
            "learned_returner_name": (
                "both"
                if self._learned_returner is None
                else self._learned_returner.label
            ),
            "stage_success": self._stage_success,
            "stage_attempt_complete": self._stage_attempt_complete,
            "target_hit": self._target_hit,
            "target_miss": self._target_miss,
            "term_stage_success": False,
            "term_target_miss": False,
        }

    def _get_reset_info(self) -> dict[str, Any]:
        info = self._state_machine.snapshot().to_info()
        info.update(self._last_reset_info)
        info.update(
            {
                "episode_step": 0,
                "episode_length": self.episode_len,
                "rally_target": self.rally_target,
                "rally_target_reached": False,
            }
        )
        info.update(self._curriculum_info())
        return info

    def _step_mujoco_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        controls = self._action_to_controls(ctrl)
        self.data.ctrl[:] = controls
        self._event_sampler.begin_control_step(self.step_number)
        stopped_for_safety = False
        for control_substep in range(n_frames):
            pre_step_safety = self._event_sampler.safety_event_kind(self.data)
            if pre_step_safety is not None:
                self._event_sampler.record_safety_event(
                    pre_step_safety,
                    control_substep=control_substep,
                )
                stopped_for_safety = True
                break
            self.data.xfrc_applied[self._ball_body_id] = 0.0
            ball_velocity = self.data.qvel[
                self._ball_dofadr : self._ball_dofadr + 3
            ]
            if bool(np.isfinite(ball_velocity).all()):
                self.data.xfrc_applied[self._ball_body_id, :3] = ball_drag_force(
                    ball_velocity
                )
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError:
                self._event_sampler.record_safety_event(
                    RallyEventKind.UNSAFE_PHYSICS,
                    control_substep=control_substep,
                )
                stopped_for_safety = True
                break
            safety_kind = self._event_sampler.sample_substep(
                self.data,
                control_substep=control_substep,
            )
            if safety_kind is not None:
                stopped_for_safety = True
                break
        self.data.xfrc_applied[self._ball_body_id] = 0.0
        self._latest_event_batch = self._event_sampler.end_control_step()
        if not stopped_for_safety and all(
            bool(np.isfinite(values).all())
            for values in (self.data.qpos, self.data.qvel, self.data.qacc)
        ):
            mujoco.mj_rnePostConstraint(self.model, self.data)

    def _evaluate_curriculum_objective(
        self,
        transition: RallyTransition,
    ) -> tuple[bool, bool, bool, bool]:
        """Return success, attempt-complete, target-hit, and target-miss.

        The tennis state machine is reduced first. A rule fault therefore
        always wins over a curriculum success that appears in the same
        substep batch.
        """
        config = self.curriculum_config
        if (
            config is None
            or transition.after.terminated
            or self._stage_success
            or (
                self._target_miss
                and config.court.target_miss_is_terminal
            )
        ):
            return False, False, False, False

        if config.objective is StageObjective.FIRST_VALID_RACKET_HIT:
            assert self._learned_returner is not None
            success = self._learned_returner in transition.valid_racket_hits
            return success, success, False, False

        if config.objective is StageObjective.RALLY_TARGET:
            success = transition.after.rally_count >= self.rally_target
            return success, success, False, False

        assert config.objective is StageObjective.VALID_TARGET_RETURN
        assert self._learned_returner is not None
        if self._learned_returner not in transition.confirmed_returns:
            return False, False, False, False

        target_side = self._learned_returner.opponent
        target_kind = (
            RallyEventKind.BALL_COURT_A
            if target_side is CourtSide.A
            else RallyEventKind.BALL_COURT_B
        )
        for event in transition.processed_events:
            if event.kind is not target_kind or event.position is None:
                continue
            playable = config.court.contains_playable(event.position)
            target_hit = playable and config.court.contains_target(
                event.position,
                side=target_side,
            )
            return target_hit, True, target_hit, not target_hit
        # A partner volley can confirm a tennis return, but it cannot satisfy
        # the curriculum's explicit physical landing target.
        return False, False, False, False

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.step_number += 1

        # Inspect all observation-producing MuJoCo fields before reducing the
        # event batch.  If a derived pose/velocity is non-finite even though
        # MuJoCo's core arrays escaped the sampler guard, append one safety
        # event and advance the rule machine exactly once.  This preserves any
        # earlier same-step rally events and keeps step-event diagnostics
        # consistent with the transition that determined reward/termination.
        observation_probe = self._get_obs()
        step_events = list(self._latest_event_batch.events)
        if not bool(np.isfinite(observation_probe).all()) and not any(
            event.kind
            in {RallyEventKind.NONFINITE_STATE, RallyEventKind.UNSAFE_PHYSICS}
            for event in step_events
        ):
            latest_substep = max(
                (
                    event.substep
                    for event in step_events
                ),
                default=self._state_machine.snapshot().last_event_substep,
            )
            step_events.append(
                RallyEvent(
                    kind=RallyEventKind.NONFINITE_STATE,
                    substep=max(0, latest_substep + 1),
                )
            )
        transition = self._state_machine.advance(
            step_events,
            contact_peaks=self._latest_event_batch.contact_peaks,
        )
        (
            stage_success_now,
            stage_attempt_complete,
            target_hit,
            target_miss,
        ) = self._evaluate_curriculum_objective(transition)
        self._stage_success = self._stage_success or stage_success_now
        self._stage_attempt_complete = (
            self._stage_attempt_complete or stage_attempt_complete
        )
        self._target_hit = self._target_hit or target_hit
        self._target_miss = self._target_miss or target_miss

        raw_observation = self._get_obs()
        observation_sanitized = not bool(np.isfinite(raw_observation).all())
        observation = self._record_or_echo_observation(
            raw_observation, observation_sanitized
        )

        stage_success_terminal = bool(
            stage_success_now
            and self.curriculum_config is not None
            and self.curriculum_config.terminate_on_success
        )
        target_miss_terminal = bool(
            target_miss
            and self.curriculum_config is not None
            and self.curriculum_config.court.target_miss_is_terminal
        )
        terminated = bool(
            transition.after.terminated
            or stage_success_terminal
            or target_miss_terminal
        )
        # A stage outcome on the final control step is a terminal task result,
        # not an ambiguous simultaneous timeout.
        truncated = bool(
            self.step_number >= self.episode_len
            and not (stage_success_terminal or target_miss_terminal)
        )
        reward, reward_info = self._reward(
            transition,
            terminated or truncated,
            stage_success_now=stage_success_now,
            target_miss_terminal=target_miss_terminal,
        )
        info = self._build_step_info(
            transition,
            reward_info=reward_info,
            terminated=terminated,
            truncated=truncated,
            observation_sanitized=observation_sanitized,
            stage_success_terminal=stage_success_terminal,
            target_miss_terminal=target_miss_terminal,
        )
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _reward(
        self,
        transition: RallyTransition,
        episode_done: bool,
        *,
        stage_success_now: bool = False,
        target_miss_terminal: bool = False,
    ) -> tuple[float, dict[str, float]]:
        if (
            self.curriculum_config is not None
            and self.curriculum_config.objective
            is StageObjective.VALID_TARGET_RETURN
        ):
            # Target-return stages pay once, only after the physical landing
            # satisfies the curriculum overlay.
            rew_return = 0.0
        else:
            rew_return = (
                self.reward_config.valid_return_reward
                * len(transition.confirmed_returns)
            )
        rew_stage_success = (
            self.curriculum_config.stage_success_reward
            if stage_success_now and self.curriculum_config is not None
            else 0.0
        )
        rew_shaping = 0.0
        rew_clawback = 0.0
        rew_fault = 0.0

        # A volley may confirm the previous player's pending return and open a
        # new one in the same substep.  Commit old escrow before adding new.
        for side in transition.confirmed_returns:
            if (
                self.curriculum_config is None
                or self.curriculum_config.objective
                is not StageObjective.VALID_TARGET_RETURN
                or stage_success_now
            ):
                self._pending_shaping[side] = 0.0
        for side in transition.valid_racket_hits:
            amount = self.reward_config.valid_hit_shaping
            rew_shaping += amount
            self._pending_shaping[side] += amount

        if episode_done:
            pending = sum(self._pending_shaping.values())
            rew_clawback = -pending
            self._pending_shaping = {CourtSide.A: 0.0, CourtSide.B: 0.0}

        if transition.terminated_now or target_miss_terminal:
            if transition.after.termination_reason in {
                TerminationReason.NONFINITE_STATE,
                TerminationReason.UNSAFE_PHYSICS,
            }:
                rew_fault = -self.reward_config.unsafe_physics_penalty
            else:
                rew_fault = -self.reward_config.fault_penalty

        components = {
            "rew_valid_return": float(rew_return),
            "rew_stage_success": float(rew_stage_success),
            "rew_shaping": float(rew_shaping),
            "rew_shaping_clawback": float(rew_clawback),
            "rew_fault": float(rew_fault),
            "rew_action_cost": 0.0,
        }
        reward = float(sum(components.values()))
        return reward, components

    def _build_step_info(
        self,
        transition: RallyTransition,
        *,
        reward_info: Mapping[str, float],
        terminated: bool,
        truncated: bool,
        observation_sanitized: bool,
        stage_success_terminal: bool = False,
        target_miss_terminal: bool = False,
    ) -> dict[str, Any]:
        info: dict[str, Any] = transition.to_info()
        info.update(self._last_reset_info)
        info.update(reward_info)
        if stage_success_terminal or target_miss_terminal:
            # A curriculum outcome stops before the newly opened next-player
            # opportunity can be attempted. Count completed task attempts,
            # not that unreachable future return.
            opportunities = max(1, transition.after.rally_count)
        else:
            opportunities = (
                int(transition.after.feed_crossed_net)
                + transition.after.rally_count
            )
        info.update(
            {
                "episode_step": self.step_number,
                "episode_length": self.episode_len,
                "term_timeout": bool(truncated and not terminated),
                "term_unsafe": transition.after.termination_reason
                in {
                    TerminationReason.NONFINITE_STATE,
                    TerminationReason.UNSAFE_PHYSICS,
                },
                "term_humanoid_net": transition.after.termination_reason
                in {
                    TerminationReason.HUMANOID_A_NET,
                    TerminationReason.HUMANOID_B_NET,
                },
                "term_racket_net": transition.after.termination_reason
                in {
                    TerminationReason.RACKET_A_NET,
                    TerminationReason.RACKET_B_NET,
                },
                "return_opportunity_count": opportunities,
                "valid_return_rate": (
                    transition.after.rally_count / opportunities
                    if opportunities
                    else 0.0
                ),
                "rally_target": self.rally_target,
                "rally_target_reached": (
                    transition.after.rally_count >= self.rally_target
                ),
                "pending_shaping_balance": float(
                    sum(self._pending_shaping.values())
                ),
                "observation_sanitized": observation_sanitized,
                "step_event_count": len(transition.observed_events),
                "step_event_kinds": tuple(
                    event.kind.value for event in transition.observed_events
                ),
                "active_contact_latches": tuple(
                    sorted(self._latest_event_batch.active_contact_latches)
                ),
            }
        )
        for label, value in zip(
            TENNIS_CONTACT_MARKOV_LABELS,
            self._event_sampler.markov_state(),
            strict=True,
        ):
            info[label] = float(value)
        info.update(self._curriculum_info())
        if self.curriculum_config is not None:
            info["term_stage_success"] = stage_success_terminal
            info["term_target_miss"] = target_miss_terminal
        return info

    @staticmethod
    def _side_one_hot(side: CourtSide) -> tuple[float, float]:
        return (1.0, 0.0) if side is CourtSide.A else (0.0, 1.0)

    @staticmethod
    def _optional_side_one_hot(side: CourtSide | None) -> tuple[float, float, float]:
        if side is None:
            return (1.0, 0.0, 0.0)
        if side is CourtSide.A:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def _player_proprioception(self, side: CourtSide) -> np.ndarray:
        return np.concatenate(
            (
                self.data.qpos[self._player_qpos_slices[side]],
                self.data.qvel[self._player_qvel_slices[side]],
            )
        )

    def _racket_observation(self, side: CourtSide) -> np.ndarray:
        site_id = self._racket_site_ids[side]
        body_id = self._racket_body_ids[side]
        rotation = self.data.xmat[body_id].reshape(3, 3)
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            spatial_velocity,
            0,
        )
        return np.concatenate(
            (
                self.data.site_xpos[site_id],
                rotation[:, 0],
                rotation[:, 2],
                spatial_velocity[3:6],
                spatial_velocity[0:3],
            )
        )

    def _rally_observation(self, snapshot: RallySnapshot) -> np.ndarray:
        phase = tuple(float(snapshot.phase is value) for value in RallyPhase)
        remaining = max(
            0.0,
            1.0 - self.step_number / float(self.episode_len),
        )
        return np.asarray(
            (
                *phase,
                *self._side_one_hot(snapshot.serving_side),
                *self._side_one_hot(snapshot.expected_returner),
                *self._side_one_hot(snapshot.ball_side),
                *self._optional_side_one_hot(snapshot.last_valid_racket_hitter),
                *self._optional_side_one_hot(snapshot.pending_return_hitter),
                *self._optional_side_one_hot(snapshot.bounce_side),
                float(snapshot.pending_return_crossed_net),
                float(snapshot.feed_crossed_net),
                float(snapshot.bounce_count),
                float(snapshot.shot_crossing_count),
                float(snapshot.rally_count),
                float(snapshot.ball_net_is_fault),
                float(snapshot.strict_reverse_crossing),
                float(snapshot.strict_double_hit),
                remaining,
            ),
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        ball_qpos = self.data.qpos[self._ball_qposadr : self._ball_qposadr + 3]
        ball_qvel = self.data.qvel[self._ball_dofadr : self._ball_dofadr + 6]
        racket_a = self._racket_observation(CourtSide.A)
        racket_b = self._racket_observation(CourtSide.B)
        pelvis_a = self.data.xpos[self._pelvis_body_ids[CourtSide.A]]
        pelvis_b = self.data.xpos[self._pelvis_body_ids[CourtSide.B]]
        relative = np.concatenate(
            (
                ball_qpos - pelvis_a,
                ball_qpos - racket_a[:3],
                ball_qpos - pelvis_b,
                ball_qpos - racket_b[:3],
            )
        )
        observation = np.concatenate(
            (
                self._player_proprioception(CourtSide.A),
                self._player_proprioception(CourtSide.B),
                racket_a,
                racket_b,
                ball_qpos,
                ball_qvel,
                relative,
                self._rally_observation(self._state_machine.snapshot()),
                np.asarray(self._event_sampler.markov_state(), dtype=np.float64),
                self._active_action_mask,
            )
        ).astype(np.float64, copy=False)
        if observation.shape != (self.observation_layout.total_size,):
            raise RuntimeError(
                f"observation has shape {observation.shape}; expected "
                f"{(self.observation_layout.total_size,)}"
            )
        return np.ascontiguousarray(observation)


__all__ = [
    "CentralizedObservationLayout",
    "HUMANOID_TENNIS_ACTION_NAMES",
    "HUMANOID_TENNIS_OBSERVATION_LAYOUT",
    "HUMANOID_TENNIS_OBSERVATION_NAMES",
    "HumanoidTennisCoopEnv",
    "TennisRewardConfig",
    "TennisServeConfig",
]
