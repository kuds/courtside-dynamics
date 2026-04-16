"""Smoke tests for the Courtside Dynamics environments.

These tests are intentionally short. They catch three classes of bug:

1. The env doesn't construct at all (XML, observation space, MujocoEnv init).
2. The env violates the Gymnasium API (caught by ``check_env``).
3. The env can't produce any reward signal at all (caught by the random
   rollout below).

Point 3 is specifically there to flag the Wall Ball regression documented
in the review: as originally written the ball has no initial velocity and
will never generate a touch-sensor event, so the policy has no learning
signal whatsoever. The random-rollout test starts out permissive (it only
asserts the env runs), with a stricter reward-nonzero check marked xfail
for Wall Ball until the fix lands.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import courtside_dynamics  # noqa: F401  (triggers registration)
from courtside_dynamics.envs import (
    BallBalanceEnv,
    BallBounceEnv,
    WallBallEnv,
)


ENV_CLASSES = [BallBalanceEnv, BallBounceEnv, WallBallEnv]

# Construction kwargs mirroring what each notebook uses at training time.
# In particular ``min_force=100.0`` makes the touch-sensor rising-edge check
# actually discriminate between "in contact" and "not in contact" --
# with ``min_force=0.0`` (the XML default) the check fires every step, which
# masks reward bugs. See the ``test_random_rollout_produces_some_reward``
# docstring for the relevance to Wall Ball.
ENV_CLASSES_WITH_KWARGS = [
    (BallBalanceEnv, {}),
    (BallBounceEnv, {"min_force": 100.0}),
    (WallBallEnv, {"min_force": 100.0}),
]


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_env_constructs_and_steps(env_cls):
    env = env_cls()
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            assert np.isfinite(reward)
    finally:
        env.close()


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_env_passes_sb3_check_env(env_cls):
    """Run the Stable-Baselines3 env checker on each env."""
    sb3_env_checker = importlib.import_module(
        "stable_baselines3.common.env_checker"
    )
    env = env_cls()
    try:
        sb3_env_checker.check_env(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_random_rollout_runs_without_nan(env_cls, rng):
    env = env_cls()
    try:
        env.reset(seed=0)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                env.reset(seed=0)
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_cls,kwargs",
    [
        (BallBalanceEnv, {}),
        (BallBounceEnv, {"min_force": 100.0}),
        pytest.param(
            WallBallEnv,
            {"min_force": 100.0},
            marks=pytest.mark.xfail(
                reason=(
                    "Wall Ball reward signal is unreachable: the ball is "
                    "spawned at rest at x=-4 while the paddle can only reach "
                    "x=-2 (no x-slide joint) and the wall is at x=3, so no "
                    "touch event can fire with random (or any) actions. "
                    "Tracked for fix."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_random_rollout_produces_some_reward(env_cls, kwargs):
    """Random actions should hit at least one nonzero reward over 2000 steps.

    For Ball Balance the reward is +1/step so this is trivial. For Ball
    Bounce gravity pulls the ball onto the paddle and an early contact
    usually lands. For Wall Ball this currently fails -- exactly the
    regression the upcoming fix needs to close.
    """
    env = env_cls(**kwargs)
    try:
        env.reset(seed=0)
        total = 0.0
        for _ in range(2_000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                env.reset(seed=0)
        assert total > 0.0, f"No reward observed in random rollout of {env_cls.__name__}"
    finally:
        env.close()


def test_gymnasium_make_ids():
    """The registered env ids resolve via ``gymnasium.make``."""
    import gymnasium

    for env_id in (
        "CourtsideDynamics/BallBalance-v0",
        "CourtsideDynamics/BallBounce-v0",
        "CourtsideDynamics/WallBall-v0",
    ):
        env = gymnasium.make(env_id)
        try:
            env.reset(seed=0)
        finally:
            env.close()


def test_asset_path_returns_existing_files():
    from pathlib import Path

    from courtside_dynamics.assets import asset_path

    for name in ("ball_balance.xml", "ball_bounce.xml", "wall_ball.xml"):
        p = Path(asset_path(name))
        assert p.is_file(), f"Missing asset: {name}"
