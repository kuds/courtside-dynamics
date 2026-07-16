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
# grounded ball). Task dynamics changed, so results are not comparable
# with earlier runs.
# 0.3.0: first registered centralized cooperative two-G1 tennis environment.
# This adds a new environment/API without changing the existing task IDs.
# 0.4.0: WallBall enforces the wall-ball rally rules it was always meant
# to have: the episode terminates on the second consecutive floor
# bounce (with ``double_bounce_penalty``), paddle/wall touch readings
# come from ball contact forces (a sagging paddle scraping the floor no
# longer pays the paddle bonus, opens the wall gate, or resets the
# stall clock), the stall cutoff runs from reset instead of arming only
# after the first contact, and the obs gains ``floor_bounce_count``
# (20->21 dims). This realizes the env's intended termination spec rather
# than changing it, but
# reward/episode-length curves (and saved policies, due to the obs
# change) are not comparable with earlier runs.
# 0.5.0: fixed-stage humanoid-tennis curriculum contracts, physical pelvis
# welds, action masking/PD holds, Stage 0–2 objectives and recipes, and held-out
# promotion metrics. The registered action/observation dimensions remain
# 58/299; stages 3–5 stay explicit planned-only presets.
# 0.6.0: registered Gymnasium IDs are unversioned.
# 0.7.0: BallBounce uses substep ball-paddle pair contacts, rewards only
# deliberate upward top-face rebounds, terminates relative to the paddle, and
# corrects its rotation units and vertical actuator authority. Its observation
# grows from 18 to 30 values to expose spin, contact-event state, and time.
# Existing BallBounce policies are incompatible and should be retrained.
# 0.8.0: WallBall replaces its raw 5-motor racket with a face-only paddle at a
# fixed 10-degree upward pitch. Three normalized actions now command
# force-limited x/y/z position targets, and yaw/pitch removal shrinks
# observations from 26 to 22 values. Existing WallBall policies, normalizers,
# replay buffers, and raw simulator states are incompatible and must not be
# resumed.
# 0.9.0: WallBall one-bounce training adds tapered recovery-state resets and
# recoverable-bounce credit while keeping evaluation on full normal serves.
# The bonus-eligibility flag grows WallBall observations from 22 to 23 values,
# so 0.8 WallBall policies and VecNormalize statistics must be retrained.
__version__ = "0.10.0"

# Register environments with gymnasium so they can be created via
# ``gymnasium.make("CourtsideDynamics/BallBalance")`` etc.
# ``max_episode_steps`` matches each env's internal ``episode_len``
# default so the TimeLimit wrapper and the env's own truncation agree.
register(
    id="CourtsideDynamics/BallBalance",
    entry_point="courtside_dynamics.envs.ball_balance:BallBalanceEnv",
    max_episode_steps=750,
)

register(
    id="CourtsideDynamics/BallBounce",
    entry_point="courtside_dynamics.envs.ball_bounce:BallBounceEnv",
    max_episode_steps=1000,
)

# WallBall uses realistic ball restitution and a slide_z range that can
# address a grounded ball. Results from the earlier, nearly inelastic
# implementation are not comparable.
register(
    id="CourtsideDynamics/WallBall",
    entry_point="courtside_dynamics.envs.wall_ball:WallBallEnv",
    max_episode_steps=750,
)

register(
    id="CourtsideDynamics/HumanoidTennisCoop",
    entry_point=(
        "courtside_dynamics.envs.humanoid_tennis:HumanoidTennisCoopEnv"
    ),
    max_episode_steps=1000,
)

__all__ = ["__version__"]
