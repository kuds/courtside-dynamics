"""Courtside Dynamics: MuJoCo environments for learning racket sports with RL.

The long-term goal of this project is to train humanoid agents that can rally
and play tennis. The environments in this package form a progression of
stepping stones toward that goal, from simple ball balancing and paddle
juggling up to full wall-rally and (eventually) humanoid tennis.
"""
from __future__ import annotations

from gymnasium.envs.registration import register

# 0.2.0: WallBall contact physics made realistic (ball contact priority
# + solref tuned to regulation-like restitution; racket reaches a
# grounded ball). Task dynamics changed, so the env id bumped to
# WallBall-v2 -- results are not comparable with v1 runs.
__version__ = "0.2.0"

# Register environments with gymnasium so they can be created via
# ``gymnasium.make("CourtsideDynamics/BallBalance-v0")`` etc.
# ``max_episode_steps`` matches each env's internal ``episode_len``
# default so the TimeLimit wrapper and the env's own truncation agree.
register(
    id="CourtsideDynamics/BallBalance-v0",
    entry_point="courtside_dynamics.envs.ball_balance:BallBalanceEnv",
    max_episode_steps=750,
)

register(
    id="CourtsideDynamics/BallBounce-v0",
    entry_point="courtside_dynamics.envs.ball_bounce:BallBounceEnv",
    max_episode_steps=1000,
)

# v2: realistic ball restitution (v1's ball was nearly perfectly
# inelastic -- it died on the first wall contact, capping every rally
# at one exchange) and a slide_z range that can address a grounded
# ball. Results are not comparable with v1.
register(
    id="CourtsideDynamics/WallBall-v2",
    entry_point="courtside_dynamics.envs.wall_ball:WallBallEnv",
    max_episode_steps=750,
)

__all__ = ["__version__"]
