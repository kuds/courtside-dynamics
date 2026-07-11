"""Custom MuJoCo environments for the Courtside Dynamics curriculum.

Exports are listed in curriculum order (simplest → most complex), matching
the registration order in :mod:`courtside_dynamics`.
"""

from courtside_dynamics.envs.ball_balance import BallBalanceEnv
from courtside_dynamics.envs.ball_bounce import BallBounceEnv
from courtside_dynamics.envs.robot_models import (
    ROBOT_MODELS,
    SUPPORTED_ROBOT_MODELS,
    UNITREE_G1_ACTION_LAYOUT,
    RobotModelSpec,
    get_robot_model_spec,
    initialize_humanoid_tennis_home,
)
from courtside_dynamics.envs.wall_ball import WallBallEnv

__all__ = [
    "BallBalanceEnv",
    "BallBounceEnv",
    "ROBOT_MODELS",
    "RobotModelSpec",
    "SUPPORTED_ROBOT_MODELS",
    "UNITREE_G1_ACTION_LAYOUT",
    "WallBallEnv",
    "get_robot_model_spec",
    "initialize_humanoid_tennis_home",
]
