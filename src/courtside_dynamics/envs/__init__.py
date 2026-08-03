"""Custom MuJoCo environments for the Courtside Dynamics curriculum.

Exports are listed in curriculum order (simplest → most complex), matching
the registration order in :mod:`courtside_dynamics`.
"""

from courtside_dynamics.envs.ball_balance import BallBalanceEnv
from courtside_dynamics.envs.ball_bounce import BallBounceEnv
from courtside_dynamics.envs.humanoid_tennis import (
    HUMANOID_TENNIS_ACTION_NAMES,
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    HUMANOID_TENNIS_OBSERVATION_NAMES,
    CentralizedObservationLayout,
    HumanoidTennisCoopEnv,
    TennisRewardConfig,
    TennisServeConfig,
)
from courtside_dynamics.envs.paddle_tennis import (
    PADDLE_TENNIS_ACTION_NAMES,
    PADDLE_TENNIS_NORMALIZED_SLICE,
    PADDLE_TENNIS_OBSERVATION_NAMES,
    PaddleCourtServe,
    PaddleTennisEnv,
)
from courtside_dynamics.envs.robot_models import (
    ROBOT_MODELS,
    SUPPORTED_ROBOT_MODELS,
    UNITREE_G1_ACTION_LAYOUT,
    RobotModelSpec,
    get_robot_model_spec,
    initialize_humanoid_tennis_home,
)
from courtside_dynamics.envs.tennis_curriculum import (
    TENNIS_CURRICULUM_PRESETS,
    CurriculumConfig,
    CurriculumCourtConfig,
    CurriculumLaunchConfig,
    CurriculumPreset,
    CurriculumStage,
    FaultStrictness,
    LearnedPlayerMode,
    MobilityMode,
    PartnerMode,
    ServeMode,
    StageObjective,
    coerce_curriculum_stage,
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
from courtside_dynamics.envs.wall_ball import WallBallEnv

__all__ = [
    "BallBalanceEnv",
    "BallBounceEnv",
    "CourtSide",
    "CentralizedObservationLayout",
    "CurriculumConfig",
    "CurriculumCourtConfig",
    "CurriculumLaunchConfig",
    "CurriculumPreset",
    "CurriculumStage",
    "FaultStrictness",
    "HUMANOID_TENNIS_ACTION_NAMES",
    "HUMANOID_TENNIS_OBSERVATION_LAYOUT",
    "HUMANOID_TENNIS_OBSERVATION_NAMES",
    "HumanoidTennisCoopEnv",
    "LearnedPlayerMode",
    "MobilityMode",
    "PADDLE_TENNIS_ACTION_NAMES",
    "PADDLE_TENNIS_NORMALIZED_SLICE",
    "PADDLE_TENNIS_OBSERVATION_NAMES",
    "PaddleCourtServe",
    "PaddleTennisEnv",
    "PartnerMode",
    "ROBOT_MODELS",
    "RallyEvent",
    "RallyEventKind",
    "RallyPhase",
    "RallyRules",
    "RallySnapshot",
    "RallyStateMachine",
    "RallyTransition",
    "RobotModelSpec",
    "SUPPORTED_ROBOT_MODELS",
    "ServeMode",
    "StageObjective",
    "TENNIS_CURRICULUM_PRESETS",
    "UNITREE_G1_ACTION_LAYOUT",
    "TerminationReason",
    "TennisRewardConfig",
    "TennisServeConfig",
    "WallBallEnv",
    "coerce_curriculum_stage",
    "get_curriculum_config",
    "get_curriculum_preset",
    "get_robot_model_spec",
    "initialize_humanoid_tennis_home",
]
