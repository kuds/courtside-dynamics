"""Immutable curriculum contracts for cooperative humanoid tennis.

Curriculum configuration is fixed for the lifetime of an environment
instance.  The registered observation exposes the active-action mask, but it
does not encode target-region, launch-distribution, or contact-forgiveness
parameters; sampling different stages at reset would therefore make the task
partially observed.  Phase 4 provides separate fixed-stage environments and
held-out promotion evaluation instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Final

from courtside_dynamics.envs._serve import validate_feed_geometry
from courtside_dynamics.envs._tennis_physics import (
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
    LINE_TOLERANCE,
)
from courtside_dynamics.envs.robot_models import UNITREE_G1_ACTUATOR_NAMES
from courtside_dynamics.envs.tennis_rules import CourtSide, RallyRules


class CurriculumStage(IntEnum):
    """The ordered cooperative-tennis learning progression."""

    ANCHORED_RACKET_INTERCEPT = 0
    ANCHORED_RETURN = 1
    ANCHORED_RANDOMIZED_RETURN = 2
    CONSTRAINED_STANDING_RETURN = 3
    MOBILE_WITH_SCRIPTED_PARTNER = 4
    TWO_CONSTRAINED_LEARNED = 5
    TWO_FREE_STANDING = 6


class MobilityMode(StrEnum):
    """Physical mobility constraint applied to the humanoids."""

    FIXED_BASE = "fixed_base"
    FOOT_CONSTRAINED = "foot_constrained"
    FREE_STANDING = "free_standing"


class StageObjective(StrEnum):
    """Terminal success condition layered over the tennis rule machine."""

    FIRST_VALID_RACKET_HIT = "first_valid_racket_hit"
    VALID_TARGET_RETURN = "valid_target_return"
    RALLY_TARGET = "rally_target"


class LearnedPlayerMode(StrEnum):
    """Which centralized action slices are controlled by the policy."""

    CURRENT_RETURNER = "current_returner"
    BOTH = "both"


class PartnerMode(StrEnum):
    """How a player not controlled by the policy is driven."""

    PD_HOLD = "pd_hold"
    SCRIPTED = "scripted"
    LEARNED = "learned"


class ServeMode(StrEnum):
    """Whether the initial physical feed is deterministic or randomized."""

    SCRIPTED = "scripted"
    RANDOMIZED = "randomized"


class FaultStrictness(StrEnum):
    """Named bundles of deterministic rally-rule switches."""

    REGULATION = "regulation"
    FORGIVING = "forgiving"

    def rally_rules(self) -> RallyRules:
        """Build the rule bundle represented by this setting."""
        if self is FaultStrictness.REGULATION:
            return RallyRules()
        return RallyRules(
            ball_net_is_fault=False,
            strict_reverse_crossing=False,
            strict_double_hit=False,
        )


@dataclass(frozen=True, slots=True)
class CurriculumLaunchConfig:
    """Side-A-local launch distribution, mirrored 180 degrees for side B.

    ``speed`` is the three-dimensional speed.  Elevation is measured above
    the court plane and lateral angle is measured away from the forward court
    axis.  Random draws are uniform and are performed by the environment's
    seeded ``np_random`` generator.
    """

    mode: ServeMode
    start_distance_from_net: float
    lateral_position: float
    height: float
    speed: float
    elevation_degrees: float
    lateral_degrees: float = 0.0
    position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_noise: float = 0.0
    elevation_noise_degrees: float = 0.0
    lateral_noise_degrees: float = 0.0
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ServeMode(self.mode))
        position_noise = tuple(self.position_noise)
        angular_velocity = tuple(self.angular_velocity)
        object.__setattr__(self, "position_noise", position_noise)
        object.__setattr__(self, "angular_velocity", angular_velocity)
        vectors = (position_noise, angular_velocity)
        if any(len(values) != 3 for values in vectors):
            raise ValueError("launch vectors must contain exactly three values")
        values = (
            self.start_distance_from_net,
            self.lateral_position,
            self.height,
            self.speed,
            self.elevation_degrees,
            self.lateral_degrees,
            *self.position_noise,
            self.speed_noise,
            self.elevation_noise_degrees,
            self.lateral_noise_degrees,
            *self.angular_velocity,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("launch configuration must contain only finite values")
        # The geometric feed contract is shared with TennisServeConfig
        # (envs._serve): one implementation for on-half, in-width,
        # off-net-plane, inside-baseline, and the 1.1-1.5 m height
        # window.
        validate_feed_geometry(
            kind="launch",
            start_distance_from_net=self.start_distance_from_net,
            lateral_position=self.lateral_position,
            height=self.height,
            position_noise=self.position_noise,
            require_inside_baseline=True,
        )
        if self.speed <= 0.0 or not 0.0 <= self.speed_noise < self.speed:
            raise ValueError("launch speed must remain positive across its noise")
        if self.elevation_noise_degrees < 0.0:
            raise ValueError("elevation noise must be non-negative")
        if self.lateral_noise_degrees < 0.0:
            raise ValueError("lateral noise must be non-negative")
        if abs(self.elevation_degrees) + self.elevation_noise_degrees >= 89.0:
            raise ValueError("launch elevation must retain a forward component")
        if abs(self.lateral_degrees) + self.lateral_noise_degrees >= 89.0:
            raise ValueError("launch lateral angle must retain a forward component")
        if self.mode is ServeMode.SCRIPTED and any(
            value != 0.0
            for value in (
                *self.position_noise,
                self.speed_noise,
                self.elevation_noise_degrees,
                self.lateral_noise_degrees,
            )
        ):
            raise ValueError("scripted launches cannot include random noise")

    def velocity(
        self,
        *,
        speed: float | None = None,
        elevation_degrees: float | None = None,
        lateral_degrees: float | None = None,
    ) -> tuple[float, float, float]:
        """Convert configured polar launch values to a local xyz velocity."""
        resolved_speed = self.speed if speed is None else float(speed)
        elevation = math.radians(
            self.elevation_degrees
            if elevation_degrees is None
            else float(elevation_degrees)
        )
        lateral = math.radians(
            self.lateral_degrees
            if lateral_degrees is None
            else float(lateral_degrees)
        )
        horizontal = resolved_speed * math.cos(elevation)
        return (
            horizontal * math.cos(lateral),
            horizontal * math.sin(lateral),
            resolved_speed * math.sin(elevation),
        )


@dataclass(frozen=True, slots=True)
class CurriculumCourtConfig:
    """Mirrored logical playable area and target overlay.

    The regulation court remains the physical surface and the tennis state
    machine still owns tennis faults.  This overlay only decides whether a
    physically legal shot satisfies the current curriculum task.  All edges
    are inclusive, matching tennis line semantics.
    """

    playable_half_length: float
    playable_half_width: float
    target_min_depth: float
    target_max_depth: float
    target_center_y: float = 0.0
    target_half_width: float = COURT_HALF_WIDTH
    target_miss_is_terminal: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target_miss_is_terminal, bool):
            raise TypeError("target_miss_is_terminal must be a bool")
        values = (
            self.playable_half_length,
            self.playable_half_width,
            self.target_min_depth,
            self.target_max_depth,
            self.target_center_y,
            self.target_half_width,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("court configuration must contain only finite values")
        if not 0.0 < self.playable_half_length <= COURT_HALF_LENGTH:
            raise ValueError("playable half length must fit the regulation court")
        if not 0.0 < self.playable_half_width <= COURT_HALF_WIDTH:
            raise ValueError("playable half width must fit the regulation court")
        if not 0.0 <= self.target_min_depth <= self.target_max_depth:
            raise ValueError("target depth interval is invalid")
        if self.target_max_depth > self.playable_half_length:
            raise ValueError("target depth must fit the playable region")
        if self.target_half_width <= 0.0:
            raise ValueError("target half width must be positive")
        if (
            abs(self.target_center_y) + self.target_half_width
            > self.playable_half_width
        ):
            raise ValueError("target width must fit the playable region")

    def contains_playable(
        self,
        position: tuple[float, float, float],
    ) -> bool:
        """Return whether a landing is within the reduced stage court."""
        x, y, _ = position
        return (
            abs(float(x)) <= self.playable_half_length + LINE_TOLERANCE
            and abs(float(y)) <= self.playable_half_width + LINE_TOLERANCE
        )

    def contains_target(
        self,
        position: tuple[float, float, float],
        *,
        side: CourtSide,
    ) -> bool:
        """Return whether a landing is inside the mirrored target region."""
        x, y, _ = position
        depth = abs(float(x))
        local_y = float(y) if side is CourtSide.B else -float(y)
        return (
            self.target_min_depth - LINE_TOLERANCE
            <= depth
            <= self.target_max_depth + LINE_TOLERANCE
            and abs(local_y - self.target_center_y)
            <= self.target_half_width + LINE_TOLERANCE
        )


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Complete fixed-stage behavioral contract for one environment."""

    stage: CurriculumStage
    name: str
    mobility: MobilityMode
    objective: StageObjective
    learned_players: LearnedPlayerMode
    partner_mode: PartnerMode
    active_joint_names: tuple[str, ...]
    pd_hold_joint_names: tuple[str, ...]
    launch: CurriculumLaunchConfig
    court: CurriculumCourtConfig
    racket_contact_scale: float
    receiver_distance_from_net: float
    rally_target: int = 1
    fault_strictness: FaultStrictness = FaultStrictness.REGULATION
    stage_success_reward: float = 1.0
    terminate_on_success: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", CurriculumStage(self.stage))
        object.__setattr__(self, "mobility", MobilityMode(self.mobility))
        object.__setattr__(self, "objective", StageObjective(self.objective))
        object.__setattr__(
            self,
            "learned_players",
            LearnedPlayerMode(self.learned_players),
        )
        object.__setattr__(self, "partner_mode", PartnerMode(self.partner_mode))
        object.__setattr__(
            self,
            "fault_strictness",
            FaultStrictness(self.fault_strictness),
        )
        if not isinstance(self.launch, CurriculumLaunchConfig):
            raise TypeError("launch must be a CurriculumLaunchConfig")
        if not isinstance(self.court, CurriculumCourtConfig):
            raise TypeError("court must be a CurriculumCourtConfig")
        if not isinstance(self.terminate_on_success, bool):
            raise TypeError("terminate_on_success must be a bool")
        if not self.terminate_on_success:
            raise ValueError(
                "the fixed observation contract requires "
                "terminate_on_success=True because it "
                "does not expose a success latch"
            )
        if not isinstance(self.name, str) or not isinstance(self.description, str):
            raise TypeError("curriculum name and description must be strings")
        if not self.name.strip():
            raise ValueError("curriculum name cannot be empty")
        known = set(UNITREE_G1_ACTUATOR_NAMES)
        active = tuple(self.active_joint_names)
        held = tuple(self.pd_hold_joint_names)
        object.__setattr__(self, "active_joint_names", active)
        object.__setattr__(self, "pd_hold_joint_names", held)
        if len(set(active)) != len(active) or len(set(held)) != len(held):
            raise ValueError("curriculum joint partitions cannot contain duplicates")
        unknown = (set(active) | set(held)) - known
        if unknown:
            raise ValueError(f"unknown curriculum joints: {sorted(unknown)}")
        overlap = set(active) & set(held)
        if overlap:
            raise ValueError(f"active and PD-held joints overlap: {sorted(overlap)}")
        if set(active) | set(held) != known:
            missing = known - (set(active) | set(held))
            raise ValueError(f"curriculum joint partition is incomplete: {sorted(missing)}")
        if self.learned_players is LearnedPlayerMode.CURRENT_RETURNER:
            if self.partner_mode is PartnerMode.LEARNED:
                raise ValueError("a single learned returner cannot have a learned partner")
        elif self.partner_mode is not PartnerMode.LEARNED:
            raise ValueError("two learned players require partner_mode='learned'")
        if (
            self.learned_players is LearnedPlayerMode.BOTH
            and self.objective is not StageObjective.RALLY_TARGET
        ):
            raise ValueError("two learned players require a rally-target objective")
        finite_values = (
            self.racket_contact_scale,
            self.receiver_distance_from_net,
            self.stage_success_reward,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("curriculum scalars must be finite")
        if self.racket_contact_scale < 1.0:
            raise ValueError("racket contact scale cannot be smaller than regulation")
        if not 0.0 < self.receiver_distance_from_net <= COURT_HALF_LENGTH:
            raise ValueError("receiver distance must lie on one physical court half")
        if isinstance(self.rally_target, bool) or not isinstance(
            self.rally_target,
            int,
        ):
            raise TypeError("rally target must be an integer")
        if self.rally_target < 1:
            raise ValueError("rally target must be at least one")
        if self.stage_success_reward < 0.0:
            raise ValueError("stage success reward must be non-negative")

    @property
    def active_joint_count_per_learned_player(self) -> int:
        """Number of policy-controlled joints in each learned slice."""
        return len(self.active_joint_names)

    def rally_rules(self) -> RallyRules:
        """Return a fresh immutable rule bundle for this stage."""
        return self.fault_strictness.rally_rules()


@dataclass(frozen=True, slots=True)
class CurriculumPreset:
    """Availability and limitation metadata around one stage config."""

    config: CurriculumConfig
    environment_available: bool
    training_recipe_available: bool
    limitation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, CurriculumConfig):
            raise TypeError("preset config must be a CurriculumConfig")
        if not isinstance(self.environment_available, bool) or not isinstance(
            self.training_recipe_available,
            bool,
        ):
            raise TypeError("preset availability fields must be bools")
        if self.limitation is not None and not isinstance(self.limitation, str):
            raise TypeError("preset limitation must be a string or None")
        if self.environment_available and self.limitation and self.training_recipe_available:
            raise ValueError("an available training preset cannot have a limitation")
        if not self.environment_available and not self.limitation:
            raise ValueError("unavailable presets require an explicit limitation")


_ALL_JOINTS: Final = UNITREE_G1_ACTUATOR_NAMES
_SHOULDER_INTERCEPT: Final = UNITREE_G1_ACTUATOR_NAMES[22:24]
_RIGHT_ARM: Final = UNITREE_G1_ACTUATOR_NAMES[22:29]
_BALANCE_AND_RACKET: Final = (
    UNITREE_G1_ACTUATOR_NAMES[0:15] + UNITREE_G1_ACTUATOR_NAMES[22:29]
)


def _held(active: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in _ALL_JOINTS if name not in set(active))


_ANCHOR_SPEED: Final = math.hypot(6.0, 3.9)
_ANCHOR_ELEVATION: Final = math.degrees(math.atan2(3.9, 6.0))
_ANCHOR_LAUNCH: Final = CurriculumLaunchConfig(
    mode=ServeMode.SCRIPTED,
    start_distance_from_net=1.0,
    lateral_position=0.5,
    height=1.25,
    speed=_ANCHOR_SPEED,
    elevation_degrees=_ANCHOR_ELEVATION,
)
_RANDOMIZED_ANCHOR_LAUNCH: Final = CurriculumLaunchConfig(
    mode=ServeMode.RANDOMIZED,
    start_distance_from_net=1.0,
    lateral_position=0.5,
    height=1.25,
    speed=_ANCHOR_SPEED,
    elevation_degrees=_ANCHOR_ELEVATION,
    position_noise=(0.10, 0.15, 0.05),
    speed_noise=0.50,
    elevation_noise_degrees=2.5,
    lateral_noise_degrees=2.0,
)
_ANCHOR_COURT: Final = CurriculumCourtConfig(
    playable_half_length=4.0,
    playable_half_width=2.5,
    target_min_depth=0.25,
    target_max_depth=4.0,
    target_half_width=2.5,
)
_REGULATION_TARGET: Final = CurriculumCourtConfig(
    playable_half_length=COURT_HALF_LENGTH,
    playable_half_width=COURT_HALF_WIDTH,
    target_min_depth=0.0,
    target_max_depth=COURT_HALF_LENGTH,
    target_half_width=COURT_HALF_WIDTH,
    target_miss_is_terminal=False,
)
_FULL_LAUNCH: Final = CurriculumLaunchConfig(
    mode=ServeMode.RANDOMIZED,
    start_distance_from_net=10.25,
    lateral_position=0.0,
    height=1.3,
    speed=math.sqrt(16.0**2 + 4.5**2),
    elevation_degrees=math.degrees(math.atan2(4.5, 16.0)),
    position_noise=(0.1, 0.25, 0.1),
    speed_noise=0.55,
    elevation_noise_degrees=1.0,
    lateral_noise_degrees=1.0,
)


def _config(
    stage: CurriculumStage,
    *,
    name: str,
    mobility: MobilityMode,
    objective: StageObjective,
    learned_players: LearnedPlayerMode,
    partner_mode: PartnerMode,
    active: tuple[str, ...],
    launch: CurriculumLaunchConfig,
    court: CurriculumCourtConfig,
    racket_scale: float,
    receiver_distance: float,
    rally_target: int = 1,
    description: str,
) -> CurriculumConfig:
    return CurriculumConfig(
        stage=stage,
        name=name,
        mobility=mobility,
        objective=objective,
        learned_players=learned_players,
        partner_mode=partner_mode,
        active_joint_names=active,
        pd_hold_joint_names=_held(active),
        launch=launch,
        court=court,
        racket_contact_scale=racket_scale,
        receiver_distance_from_net=receiver_distance,
        rally_target=rally_target,
        description=description,
    )


TENNIS_CURRICULUM_PRESETS: Final[Mapping[CurriculumStage, CurriculumPreset]] = (
    MappingProxyType(
        {
            CurriculumStage.ANCHORED_RACKET_INTERCEPT: CurriculumPreset(
                config=_config(
                    CurriculumStage.ANCHORED_RACKET_INTERCEPT,
                    name="anchored_racket_intercept",
                    mobility=MobilityMode.FIXED_BASE,
                    objective=StageObjective.FIRST_VALID_RACKET_HIT,
                    learned_players=LearnedPlayerMode.CURRENT_RETURNER,
                    partner_mode=PartnerMode.PD_HOLD,
                    active=_SHOULDER_INTERCEPT,
                    launch=_ANCHOR_LAUNCH,
                    court=_ANCHOR_COURT,
                    racket_scale=1.5,
                    receiver_distance=3.0,
                    description=(
                        "Fixed-pelvis two-shoulder intercept of one slow, "
                        "deterministic physical feed."
                    ),
                ),
                environment_available=True,
                training_recipe_available=True,
            ),
            CurriculumStage.ANCHORED_RETURN: CurriculumPreset(
                config=_config(
                    CurriculumStage.ANCHORED_RETURN,
                    name="anchored_return",
                    mobility=MobilityMode.FIXED_BASE,
                    objective=StageObjective.VALID_TARGET_RETURN,
                    learned_players=LearnedPlayerMode.CURRENT_RETURNER,
                    partner_mode=PartnerMode.PD_HOLD,
                    active=_RIGHT_ARM,
                    launch=_ANCHOR_LAUNCH,
                    court=_ANCHOR_COURT,
                    racket_scale=1.5,
                    receiver_distance=3.0,
                    description=(
                        "Fixed-pelvis right-arm return into a generous reduced-court "
                        "target."
                    ),
                ),
                environment_available=True,
                training_recipe_available=True,
            ),
            CurriculumStage.ANCHORED_RANDOMIZED_RETURN: CurriculumPreset(
                config=_config(
                    CurriculumStage.ANCHORED_RANDOMIZED_RETURN,
                    name="anchored_randomized_return",
                    mobility=MobilityMode.FIXED_BASE,
                    objective=StageObjective.VALID_TARGET_RETURN,
                    learned_players=LearnedPlayerMode.CURRENT_RETURNER,
                    partner_mode=PartnerMode.PD_HOLD,
                    active=_RIGHT_ARM,
                    launch=_RANDOMIZED_ANCHOR_LAUNCH,
                    court=_ANCHOR_COURT,
                    racket_scale=1.35,
                    receiver_distance=3.0,
                    description=(
                        "Fixed-pelvis target return with bounded, seeded launch "
                        "randomization."
                    ),
                ),
                environment_available=True,
                training_recipe_available=True,
            ),
            CurriculumStage.CONSTRAINED_STANDING_RETURN: CurriculumPreset(
                config=_config(
                    CurriculumStage.CONSTRAINED_STANDING_RETURN,
                    name="constrained_standing_return",
                    mobility=MobilityMode.FOOT_CONSTRAINED,
                    objective=StageObjective.VALID_TARGET_RETURN,
                    learned_players=LearnedPlayerMode.CURRENT_RETURNER,
                    partner_mode=PartnerMode.PD_HOLD,
                    active=_BALANCE_AND_RACKET,
                    launch=_RANDOMIZED_ANCHOR_LAUNCH,
                    court=_ANCHOR_COURT,
                    racket_scale=1.25,
                    receiver_distance=5.0,
                    description="Planned foot-constrained balance and return stage.",
                ),
                environment_available=False,
                training_recipe_available=False,
                limitation=(
                    "foot equality constraints and long-horizon standing stability "
                    "have not been implemented or validated"
                ),
            ),
            CurriculumStage.MOBILE_WITH_SCRIPTED_PARTNER: CurriculumPreset(
                config=_config(
                    CurriculumStage.MOBILE_WITH_SCRIPTED_PARTNER,
                    name="mobile_with_scripted_partner",
                    mobility=MobilityMode.FREE_STANDING,
                    objective=StageObjective.VALID_TARGET_RETURN,
                    learned_players=LearnedPlayerMode.CURRENT_RETURNER,
                    partner_mode=PartnerMode.SCRIPTED,
                    active=_ALL_JOINTS,
                    launch=_FULL_LAUNCH,
                    court=_REGULATION_TARGET,
                    racket_scale=1.0,
                    receiver_distance=10.5,
                    description="Planned mobile player with a deterministic partner.",
                ),
                environment_available=False,
                training_recipe_available=False,
                limitation=(
                    "no general physical scripted-partner controller is available; "
                    "the Phase 3 oracle is only a single feasibility fixture"
                ),
            ),
            CurriculumStage.TWO_CONSTRAINED_LEARNED: CurriculumPreset(
                config=_config(
                    CurriculumStage.TWO_CONSTRAINED_LEARNED,
                    name="two_constrained_learned",
                    mobility=MobilityMode.FOOT_CONSTRAINED,
                    objective=StageObjective.RALLY_TARGET,
                    learned_players=LearnedPlayerMode.BOTH,
                    partner_mode=PartnerMode.LEARNED,
                    active=_BALANCE_AND_RACKET,
                    launch=_FULL_LAUNCH,
                    court=_REGULATION_TARGET,
                    racket_scale=1.0,
                    receiver_distance=10.5,
                    rally_target=2,
                    description="Planned short-rally task for two constrained policies.",
                ),
                environment_available=False,
                training_recipe_available=False,
                limitation=(
                    "two-policy constrained control depends on the unimplemented "
                    "standing constraints and has not passed promotion evaluation"
                ),
            ),
            CurriculumStage.TWO_FREE_STANDING: CurriculumPreset(
                config=_config(
                    CurriculumStage.TWO_FREE_STANDING,
                    name="two_free_standing",
                    mobility=MobilityMode.FREE_STANDING,
                    objective=StageObjective.RALLY_TARGET,
                    learned_players=LearnedPlayerMode.BOTH,
                    partner_mode=PartnerMode.LEARNED,
                    active=_ALL_JOINTS,
                    launch=_FULL_LAUNCH,
                    court=_REGULATION_TARGET,
                    racket_scale=1.0,
                    receiver_distance=10.5,
                    rally_target=4,
                    description=(
                        "Full free-standing environment-correctness mode; not a "
                        "validated training recipe."
                    ),
                ),
                environment_available=True,
                training_recipe_available=False,
                limitation=(
                    "the environment is available, but end-to-end free-standing "
                    "two-G1 training has not been demonstrated"
                ),
            ),
        }
    )
)


def coerce_curriculum_stage(value: CurriculumStage | int | str) -> CurriculumStage:
    """Normalize an integer, enum name, preset name, or ``stage_N`` string."""
    if isinstance(value, CurriculumStage):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized.startswith("stage_") and normalized[6:].isdigit():
            return CurriculumStage(int(normalized[6:]))
        if normalized.isdigit():
            return CurriculumStage(int(normalized))
        for stage, preset in TENNIS_CURRICULUM_PRESETS.items():
            if normalized in {stage.name.lower(), preset.config.name}:
                return stage
        raise ValueError(f"unknown tennis curriculum stage {value!r}")
    return CurriculumStage(value)


def get_curriculum_preset(
    stage: CurriculumStage | int | str,
) -> CurriculumPreset:
    """Return preset metadata without claiming that the stage is available."""
    return TENNIS_CURRICULUM_PRESETS[coerce_curriculum_stage(stage)]


def get_curriculum_config(
    stage: CurriculumStage | int | str,
    *,
    require_available: bool = True,
) -> CurriculumConfig:
    """Return a preset config, optionally rejecting planned-only stages."""
    preset = get_curriculum_preset(stage)
    if require_available and not preset.environment_available:
        assert preset.limitation is not None
        raise NotImplementedError(
            f"curriculum stage {int(preset.config.stage)} "
            f"({preset.config.name}) is not available: {preset.limitation}"
        )
    return preset.config


__all__ = [
    "CurriculumConfig",
    "CurriculumCourtConfig",
    "CurriculumLaunchConfig",
    "CurriculumPreset",
    "CurriculumStage",
    "FaultStrictness",
    "LearnedPlayerMode",
    "MobilityMode",
    "PartnerMode",
    "ServeMode",
    "StageObjective",
    "TENNIS_CURRICULUM_PRESETS",
    "coerce_curriculum_stage",
    "get_curriculum_config",
    "get_curriculum_preset",
]
