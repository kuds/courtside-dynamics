"""Courtside Dynamics: MuJoCo environments for learning racket sports with RL.

The long-term goal of this project is to train humanoid agents that can rally
and play tennis. The environments in this package form a progression of
stepping stones toward that goal, from simple ball balancing and paddle
juggling up to full wall-rally and (eventually) humanoid tennis.
"""
from __future__ import annotations

from gymnasium.envs.registration import register

__version__ = "0.1.0"

# Register environments with gymnasium so they can be created via
# ``gymnasium.make("CourtsideDynamics/BallBalance-v0")`` etc.
register(
    id="CourtsideDynamics/BallBalance-v0",
    entry_point="courtside_dynamics.envs.ball_balance:BallBalanceEnv",
    max_episode_steps=1000,
)

register(
    id="CourtsideDynamics/BallBounce-v0",
    entry_point="courtside_dynamics.envs.ball_bounce:BallBounceEnv",
    max_episode_steps=1000,
)

register(
    id="CourtsideDynamics/WallBall-v0",
    entry_point="courtside_dynamics.envs.wall_ball:WallBallEnv",
    max_episode_steps=1000,
)

register(
    id="CourtsideDynamics/TennisWall-v0",
    entry_point="courtside_dynamics.envs.tennis_wall:TennisWallEnv",
    max_episode_steps=1000,
)

__all__ = ["__version__"]
