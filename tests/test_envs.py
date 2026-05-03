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


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_observation_names_match_obs_shape(env_cls):
    """Each env declares labels in lockstep with its observation vector."""
    env = env_cls()
    try:
        names = env.observation_names
        assert len(names) == env.observation_space.shape[0], (
            f"{env_cls.__name__}.observation_names has {len(names)} labels "
            f"but observation_space.shape is {env.observation_space.shape}"
        )
        assert len(set(names)) == len(names), (
            f"{env_cls.__name__}.observation_names has duplicate labels: "
            f"{[n for n in names if names.count(n) > 1]}"
        )
    finally:
        env.close()


def test_asset_path_returns_existing_files():
    from pathlib import Path

    from courtside_dynamics.assets import asset_path

    for name in ("ball_balance.xml", "ball_bounce.xml", "wall_ball.xml", "tennis_wall.xml"):
        p = Path(asset_path(name))
        assert p.is_file(), f"Missing asset: {name}"


class TestWallBallShaping:
    """Verify the WallBall reward shaping and diagnostic info keys."""

    def test_obs_shape_includes_paddle_to_ball_offset(self):
        env = WallBallEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (17,)
            # Last three obs dims = paddle_head -> ball relative xyz.
            ball = np.array(env.data.joint("ball_x").qpos[:3])
            paddle = np.array(env.data.body("paddle_head").xpos)
            np.testing.assert_allclose(obs[-3:], ball - paddle, atol=1e-6)
        finally:
            env.close()

    def test_info_keys_present(self):
        env = WallBallEnv(min_force=100.0)
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            for key in (
                "sensor_data",
                "wall_touch",
                "paddle_touch",
                "bounce_count",
                "wall_contact_count",
                "paddle_hit_count",
                "stalled",
            ):
                assert key in info, f"Missing info key: {key}"
        finally:
            env.close()

    def test_out_of_bounds_applies_penalty(self):
        """Driving the ball out of the play volume should subtract the
        configured penalty from the terminating step's reward."""
        env = WallBallEnv(
            min_force=100.0,
            out_of_bounds_penalty=2.5,
            track_shaping_scale=0.0,
        )
        try:
            env.reset(seed=0)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            # Park the ball just inside the +y boundary and shove it out.
            qpos[ball_qposadr : ball_qposadr + 3] = [0.0, 4.4, 1.5]
            qvel[ball_dofadr : ball_dofadr + 3] = [0.0, 50.0, 0.0]
            env.set_state(qpos, qvel)

            terminating_reward = None
            for _ in range(50):
                _, reward, terminated, truncated, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                if terminated or truncated:
                    terminating_reward = reward
                    break
            assert terminating_reward is not None, "Episode never terminated"
            assert terminating_reward <= -2.4, (
                f"Expected ~-2.5 penalty on OOB termination, got {terminating_reward}"
            )
        finally:
            env.close()

    def test_paddle_hit_bonus_fires_on_return(self):
        """Place the ball on the paddle face mid-return and confirm the
        paddle-hit bonus is paid out."""
        env = WallBallEnv(
            min_force=0.0,
            paddle_hit_bonus=2.0,
            track_shaping_scale=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            # Force the env into "returning" state as if a wall bounce just
            # happened, so the paddle bonus is eligible.
            env._returning = True
            env._prev_paddle_to_ball = None

            paddle_pos = env.data.body("paddle_head").xpos.copy()
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = paddle_pos + np.array(
                [0.0, 0.2, 0.05]
            )
            qvel[ball_dofadr : ball_dofadr + 3] = [0.0, -8.0, 0.0]
            env.set_state(qpos, qvel)

            cumulative = 0.0
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if info["paddle_hit_count"] >= 1:
                    break
                if terminated or truncated:
                    break
            assert info["paddle_hit_count"] >= 1, (
                "Paddle contact never registered; rising-edge detector or "
                "paddle touch sensor is broken"
            )
            assert cumulative >= 1.5, (
                f"Expected paddle-hit bonus (~2.0) to accumulate, got {cumulative}"
            )
        finally:
            env.close()


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

    def test_phase_transitions_on_paddle_contact(self):
        """Teleport the ball into the paddle face and verify phase advances.

        The paddle face sits at ``paddle_base + (0.3, 0, 0)`` (see
        ``tennis_wall.xml``). Placing the ball there with a small inward
        velocity guarantees the paddle touch sensor fires, which should
        bump ``paddle_hit_count`` and flip the phase from APPROACH_PADDLE
        (0) to APPROACH_WALL (1). Testing this deterministically is the
        only way to catch a broken rising-edge detector: random rollouts
        leave convergence to luck.
        """
        env = TennisWallEnv(min_force=0.0)
        try:
            env.reset(seed=0)
            # Place the ball in front of the paddle face (which sits at
            # paddle_base + +0.3 on x) and send it toward the racket.
            ball_start = env.data.body("paddle_base").xpos.copy()
            ball_start[0] += 0.5

            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = ball_start
            qvel[ball_dofadr : ball_dofadr + 3] = [-5.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            assert env.phase == 0
            transitioned = False
            for _ in range(20):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                if info["paddle_hit_count"] >= 1:
                    transitioned = True
                    break
                if terminated or truncated:
                    break
            assert transitioned, (
                "Paddle contact never registered in 20 steps; rising-edge "
                "detector or touch sensor is broken"
            )
            assert env.phase == 1  # APPROACH_WALL
        finally:
            env.close()
