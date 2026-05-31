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
# In particular ``min_force`` makes the touch-sensor rising-edge check
# actually discriminate between "in contact" and "not in contact" --
# with ``min_force=0.0`` (the XML default) the check fires every step,
# which masks reward bugs. See the
# ``test_random_rollout_produces_some_reward`` docstring for the
# relevance to Wall Ball.
ENV_CLASSES_WITH_KWARGS = [
    (BallBalanceEnv, {}),
    (BallBounceEnv, {"min_force": 100.0}),
    (WallBallEnv, {"min_force": 1.0}),
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
    ],
)
def test_random_rollout_produces_some_reward(env_cls, kwargs):
    """Random actions should hit at least one nonzero reward over 2000 steps.

    For Ball Balance the reward is +1/step so this is trivial. For Ball
    Bounce gravity pulls the ball onto the paddle and an early contact
    usually lands. Wall Ball is intentionally excluded: under the
    post-fix reward gate (no reward until the paddle has touched the
    ball) random actions can't earn anything within 2000 steps. See
    ``TestWallBallRewardGate`` for the targeted oracle-vs-noop check.
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
        "CourtsideDynamics/WallBall-v1",
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

    for name in ("ball_balance.xml", "ball_bounce.xml", "wall_ball.xml"):
        p = Path(asset_path(name))
        assert p.is_file(), f"Missing asset: {name}"


class TestWallBallRewardGate:
    """Verify the post-fix WallBall reward design.

    The original env gave +1 on every wall contact, including the serve
    bounce. SAC training stalled at reward=1.000 ± 0.000 because the
    serve alone produced 1 reward and a dead-ball floor rally produced
    no further events — i.e. random and trained policies were
    indistinguishable. These tests pin down the new contract:

    * a no-op policy gets the serve bounce but earns nothing,
    * the oracle (paddle tracker) earns strictly more than no-op,
    * a stalled ball terminates instead of burning the full episode,
    * paddle-face contacts grant the shaping bonus.
    """

    @staticmethod
    def _zero_action(env):
        return np.zeros(env.action_space.shape, dtype=np.float32)

    def test_noop_earns_no_wall_reward(self):
        """Hold the paddle still and verify no wall +1 ever pays out.

        The serve flies *toward* the paddle, so a still paddle either
        (a) gets brushed by the ball as it passes, possibly registering
        a paddle-hit edge, or (b) lets the ball drift past and OOB.
        Either way, the gated wall reward must remain at zero —
        ``bounce_count`` (which only counts *rewarded* wall hits) must
        stay at 0 across the whole episode.
        """
        env = WallBallEnv(
            min_force=1.0,
            episode_len=400,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            total_reward = 0.0
            for _ in range(400):
                _, reward, terminated, truncated, _ = env.step(
                    self._zero_action(env)
                )
                total_reward += reward
                if terminated or truncated:
                    break
            assert env.bounce_count == 0, (
                f"No-op policy somehow earned a gated wall hit: "
                f"bounce_count={env.bounce_count}"
            )
            assert total_reward == 0.0, (
                f"No-op total reward should be 0 (all bonuses zeroed), "
                f"got {total_reward}"
            )
        finally:
            env.close()

    def test_oracle_outscores_noop(self):
        """A hand-coded paddle tracker must score strictly above no-op.

        This is the canary for the whole fix: if a controller with full
        state access can't get nonzero reward, the env is unsolvable and
        no RL agent will fix it.
        """
        from courtside_dynamics.scripted_policies import (
            wall_ball_oracle_action,
        )

        def run(action_fn, seed):
            env = WallBallEnv(min_force=1.0, episode_len=750)
            try:
                obs, _ = env.reset(seed=seed)
                total = 0.0
                for _ in range(750):
                    obs, reward, terminated, truncated, _ = env.step(
                        action_fn(obs)
                    )
                    total += reward
                    if terminated or truncated:
                        break
                return total
            finally:
                env.close()

        noop_total = run(lambda o: np.zeros(5, dtype=np.float32), seed=0)
        oracle_total = run(wall_ball_oracle_action, seed=0)
        # With the OOB penalty + tracking-shaping clawback, a no-op
        # policy that lets the ball escape can land slightly negative;
        # the strict "no positive reward" claim is what matters here.
        assert noop_total <= 0.0, (
            f"No-op policy earned {noop_total}, expected <= 0"
        )
        assert oracle_total > noop_total, (
            f"Oracle ({oracle_total}) did not beat no-op ({noop_total}) — "
            "env is not solvable by paddle tracking"
        )

    def test_stalled_ball_terminates_episode(self):
        """A dead ball should end the episode well before episode_len.

        The serve flies *at* the paddle, so a no-op policy normally
        either grazes the paddle or lets the ball pass and OOB. To
        exercise the stall path deterministically, we pin the ball to
        the floor at rest and flip ``_first_event_seen`` on (as if a
        paddle hit had already happened) — the env should then count
        ``stall_steps`` of no events and terminate via stall.
        """
        env = WallBallEnv(
            min_force=1.0, episode_len=2000, stall_steps=50
        )
        try:
            env.reset(seed=0)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = [0.0, 0.0, 0.07]
            qvel[ball_dofadr : ball_dofadr + 6] = 0.0
            env.set_state(qpos, qvel)
            env._first_event_seen = True
            env._steps_since_event = 0

            terminated = False
            info = {"stalled": False}
            steps = 0
            for step in range(1, 600):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                steps = step
                if terminated or truncated:
                    break
            assert terminated, "Stalled ball never terminated the episode"
            assert info["stalled"], "Termination wasn't flagged as a stall"
            assert steps < 600, (
                "Episode ran past stall_steps=50 budget"
            )
        finally:
            env.close()

    def test_paddle_contact_grants_bonus(self):
        """Teleport the ball into the paddle and confirm the bonus fires.

        The paddle face sits at world ``(-1.7, 0, 1.2)`` when all
        paddle joints are at qpos=0 (paddle_base at (-2, 0, 1.2) +
        paddle_head body offset (0.3, 0, 0)). Spawning the ball just
        in front of the face on the wall side and giving it an inward
        velocity is a deterministic way to verify the rising-edge
        detector and bonus wiring without depending on luck.
        """
        env = WallBallEnv(min_force=0.0, paddle_hit_bonus=0.5)
        try:
            env.reset(seed=0)

            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = [-1.0, 0.0, 1.3]
            qvel[ball_dofadr : ball_dofadr + 3] = [-3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            total_reward = 0.0
            hit = False
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total_reward += reward
                if info["paddle_hit_count"] >= 1:
                    hit = True
                    break
                if terminated or truncated:
                    break
            assert hit, "Ball-into-paddle never registered a contact"
            assert total_reward >= 0.5, (
                f"Paddle bonus didn't fire (total reward {total_reward})"
            )
        finally:
            env.close()

    def test_obs_shape_includes_paddle_to_ball_offset(self):
        """The 20-dim obs ends with the paddle_head→ball relative xyz."""
        env = WallBallEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (20,)
            ball = np.array(env.data.joint("ball_x").qpos[:3])
            paddle = np.array(env.data.body("paddle_head").xpos)
            np.testing.assert_allclose(obs[-3:], ball - paddle, atol=1e-6)
        finally:
            env.close()

    def test_out_of_bounds_applies_penalty(self):
        """Driving the ball out of the play volume subtracts the
        configured penalty from the terminating step's reward."""
        env = WallBallEnv(
            min_force=1.0,
            out_of_bounds_penalty=2.5,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
        )
        try:
            env.reset(seed=0)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = [0.0, 5.4, 1.5]
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
                f"Expected ~-2.5 OOB penalty, got {terminating_reward}"
            )
        finally:
            env.close()

    def test_track_shaping_zero_net_on_missed_return(self):
        """A no-op policy must net zero tracking shaping on a missed
        return — the PBRS terminal correction must claw back the
        accumulated shaping when the episode ends mid-window."""
        env = WallBallEnv(
            min_force=1.0,
            track_shaping_scale=1.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            # Place the ball post-wall, heading back but offset in y so
            # it misses the paddle and exits OOB.
            qpos[ball_qposadr : ball_qposadr + 3] = [2.5, 2.0, 1.0]
            qvel[ball_dofadr : ball_dofadr + 3] = [-15.0, 0.0, 0.0]
            env.set_state(qpos, qvel)
            env._returning = True
            env._prev_paddle_to_ball = None
            env._return_shaping_total = 0.0

            cumulative = 0.0
            episode_done = False
            for _ in range(300):
                _, reward, terminated, truncated, _ = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if terminated or truncated:
                    episode_done = True
                    break
            assert episode_done, "Episode did not end in 300 steps"
            assert abs(cumulative) < 1e-9, (
                f"Expected zero net shaping on missed return, got {cumulative}"
            )
        finally:
            env.close()

    def test_track_shaping_positive_on_successful_return(self):
        """Net shaping is positive when the paddle returns the ball:
        no clawback fires on a successful return, so the agent keeps
        the positive shaping accumulated while the ball approached."""
        env = WallBallEnv(
            min_force=0.0,
            track_shaping_scale=1.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            env._returning = True
            env._prev_paddle_to_ball = None
            env._return_shaping_total = 0.0

            # The paddle's broad face is normal to ±x (per the
            # reorientation on main); approach it from +x with a small
            # velocity so the stiff contact peak is captured.
            paddle_pos = env.data.body("paddle_head").xpos.copy()
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = paddle_pos + np.array(
                [0.5, 0.0, 0.05]
            )
            qvel[ball_dofadr : ball_dofadr + 3] = [-3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            cumulative = 0.0
            hit_registered = False
            for _ in range(50):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if info["paddle_hit_count"] >= 1:
                    hit_registered = True
                    break
                if terminated or truncated:
                    break
            assert hit_registered, "Paddle contact never registered"
            assert cumulative > 0.0, (
                f"Expected positive net shaping on successful return, got {cumulative}"
            )
        finally:
            env.close()

    def test_consecutive_wall_hit_does_not_pay(self):
        """A wall contact with no paddle hit since the last wall does
        not earn the +1 — tightening main's "any paddle hit this episode"
        gate to "since the last wall hit" so a ball that bounces off
        the wall twice with no return between fails the rally."""
        env = WallBallEnv(
            min_force=1.0,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            # Force the gate closed (as if a previous wall already
            # fired with no paddle return) and drive a fresh wall hit.
            env._paddle_hit_since_last_wall = False
            env._returning = False
            env._prev_wall_touch = 0.0

            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            # Park the ball just shy of the wall sensor (face at x≈2.85)
            # and shove it at the wall.
            # Slow velocity so the stiff-contact peak coincides with a
            # control-step boundary and the touch sensor reads nonzero.
            qpos[ball_qposadr : ball_qposadr + 3] = [2.7, 0.0, 1.5]
            qvel[ball_dofadr : ball_dofadr + 3] = [3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            cumulative = 0.0
            initial_count = env.wall_contact_count
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if info["wall_contact_count"] > initial_count:
                    break
                if terminated or truncated:
                    break
            assert info["wall_contact_count"] > initial_count, (
                "Wall contact never registered"
            )
            # bounce_count tracks *rewarded* wall contacts; with the
            # gate closed this contact must not increment it.
            assert env.bounce_count == 0
            assert cumulative < 0.5, (
                f"Expected gated wall hit to pay nothing, got {cumulative}"
            )
        finally:
            env.close()

    def test_consecutive_wall_hit_claws_back_stale_shaping(self):
        """A consecutive wall contact must claw back any tracking
        shaping accumulated in the previous return window, preserving
        the no-op zero-net invariant across multi-bounce episodes."""
        env = WallBallEnv(
            min_force=1.0,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            # Pretend the previous return window paid out +0.5 of
            # shaping but ended without a paddle return.
            env._paddle_hit_since_last_wall = False
            env._returning = True
            env._return_shaping_total = 0.5
            env._prev_paddle_to_ball = None
            env._prev_wall_touch = 0.0

            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            # Slow velocity so the stiff-contact peak coincides with a
            # control-step boundary and the touch sensor reads nonzero.
            qpos[ball_qposadr : ball_qposadr + 3] = [2.7, 0.0, 1.5]
            qvel[ball_dofadr : ball_dofadr + 3] = [3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            cumulative = 0.0
            initial_count = env.wall_contact_count
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if info["wall_contact_count"] > initial_count:
                    break
                if terminated or truncated:
                    break
            assert info["wall_contact_count"] > initial_count
            # Clawback: -0.5. No +1 (gate closed). Other rewards zeroed.
            assert cumulative < -0.4, (
                f"Expected ~-0.5 clawback on consecutive wall, got {cumulative}"
            )
        finally:
            env.close()

    def test_wall_reward_resumes_after_paddle_hit(self):
        """After a paddle hit, the next wall contact pays +1 again."""
        env = WallBallEnv(
            min_force=1.0,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            env._paddle_hit_since_last_wall = True
            env._returning = False
            env._return_shaping_total = 0.0
            env._prev_wall_touch = 0.0

            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            # Slow velocity so the stiff-contact peak coincides with a
            # control-step boundary and the touch sensor reads nonzero.
            qpos[ball_qposadr : ball_qposadr + 3] = [2.7, 0.0, 1.5]
            qvel[ball_dofadr : ball_dofadr + 3] = [3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)

            cumulative = 0.0
            initial_count = env.wall_contact_count
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                cumulative += reward
                if info["wall_contact_count"] > initial_count:
                    break
                if terminated or truncated:
                    break
            assert info["wall_contact_count"] > initial_count
            assert cumulative >= 0.9, (
                f"Expected +1 wall reward after paddle hit, got {cumulative}"
            )
            # Gate should now be closed again until the next paddle hit.
            assert env._paddle_hit_since_last_wall is False
        finally:
            env.close()

    def test_reward_components_sum_to_total(self):
        """The per-component reward breakdown in info must sum to reward.

        Whatever the dynamics do, ``rew_wall + rew_paddle + rew_shaping +
        rew_oob`` has to equal the scalar reward on every step -- that's
        the invariant that makes the composition plots trustworthy.
        """
        env = WallBallEnv(min_force=1.0)
        try:
            env.reset(seed=0)
            rng = np.random.default_rng(0)
            for _ in range(250):
                action = rng.uniform(
                    -1.0, 1.0, size=env.action_space.shape
                ).astype(np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                components = (
                    info["rew_wall"]
                    + info["rew_paddle"]
                    + info["rew_shaping"]
                    + info["rew_oob"]
                )
                assert abs(components - reward) < 1e-9, (
                    f"components {components} != reward {reward}"
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_termination_flag_set_on_out_of_bounds(self):
        """An OOB exit flags ``term_oob`` (and not the others) at the end."""
        env = WallBallEnv(
            min_force=1.0,
            out_of_bounds_penalty=1.0,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
        )
        try:
            env.reset(seed=0)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos[ball_qposadr : ball_qposadr + 3] = [0.0, 5.4, 1.5]
            qvel[ball_dofadr : ball_dofadr + 3] = [0.0, 50.0, 0.0]
            env.set_state(qpos, qvel)

            for _ in range(50):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                if terminated or truncated:
                    assert info["term_oob"] is True
                    assert info["term_stall"] is False
                    assert info["term_timeout"] is False
                    break
            else:
                raise AssertionError("episode never terminated")
        finally:
            env.close()
