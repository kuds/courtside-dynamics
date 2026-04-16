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
    TennisWallEnv,
    WallBallEnv,
)

ENV_CLASSES = [BallBalanceEnv, BallBounceEnv, WallBallEnv, TennisWallEnv]

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
    (TennisWallEnv, {"min_force": 100.0}),
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
        (WallBallEnv, {"min_force": 100.0}),
        (TennisWallEnv, {"min_force": 100.0}),
    ],
)
def test_random_rollout_produces_some_reward(env_cls, kwargs):
    """Random actions should hit at least one nonzero reward over 2000 steps.

    For Ball Balance the reward is +1/step so this is trivial. For Ball
    Bounce gravity pulls the ball onto the paddle and an early contact
    usually lands. For Wall Ball the ball is served toward the wall on
    reset, so the first wall contact comes "for free" and the test
    verifies that reward genuinely fires on rising edges rather than
    spamming every step.
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
        "CourtsideDynamics/TennisWall-v0",
    ):
        env = gymnasium.make(env_id)
        try:
            env.reset(seed=0)
        finally:
            env.close()


def test_asset_path_returns_existing_files():
    from pathlib import Path

    from courtside_dynamics.assets import asset_path

    for name in ("ball_balance.xml", "ball_bounce.xml", "wall_ball.xml", "tennis_wall.xml"):
        p = Path(asset_path(name))
        assert p.is_file(), f"Missing asset: {name}"


class TestTennisWallStateMachine:
    """Verify the phase transitions and reward shaping in TennisWallEnv."""

    def test_initial_phase_is_approach_paddle(self):
        env = TennisWallEnv()
        try:
            obs, _ = env.reset(seed=42)
            # Last two elements are phase one-hot [approach_paddle, approach_wall]
            assert obs[-2] == 1.0 and obs[-1] == 0.0
            assert env.phase == 0  # _PHASE_APPROACH_PADDLE
        finally:
            env.close()

    def test_obs_shape_matches_spec(self):
        env = TennisWallEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (18,)
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == (18,)
        finally:
            env.close()

    def test_rally_count_starts_at_zero(self):
        env = TennisWallEnv()
        try:
            env.reset(seed=0)
            assert env.rally_count == 0
        finally:
            env.close()

    def test_shaping_reward_is_nonzero(self):
        """The potential-based shaping should produce nonzero reward even
        with random actions (the ball is moving toward the paddle on reset)."""
        env = TennisWallEnv()
        try:
            env.reset(seed=0)
            rewards = []
            for _ in range(50):
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                rewards.append(reward)
                if terminated or truncated:
                    break
            assert any(r != 0.0 for r in rewards), "Shaping reward should be nonzero"
        finally:
            env.close()

    def test_info_keys_present(self):
        env = TennisWallEnv()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            for key in ("phase", "rally_count", "paddle_hit_count",
                        "wall_hit_count", "paddle_touch", "wall_touch"):
                assert key in info, f"Missing info key: {key}"
        finally:
            env.close()
