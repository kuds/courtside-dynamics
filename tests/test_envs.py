"""Smoke tests for the Courtside Dynamics environments.

These tests are intentionally short. They catch three classes of bug:

1. The env doesn't construct at all (XML, observation space, MujocoEnv init).
2. The env violates the Gymnasium API (caught by ``check_env``).
3. The reward function pays when it shouldn't or stays silent when it
   should pay (the random rollout below plus the targeted
   ``TestWallBallRewardGate`` suite).

Wall Ball is excluded from the random-rollout reward check on purpose:
under the gated reward (no wall +1 until the paddle has touched the
ball) random actions can't earn anything in 2000 steps, so the env gets
its own oracle-vs-noop check instead.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import courtside_dynamics  # noqa: F401  (triggers registration)
from courtside_dynamics.envs import (
    BallBalanceEnv,
    BallBounceEnv,
    HumanoidTennisCoopEnv,
    WallBallEnv,
)

ENV_CLASSES = [
    BallBalanceEnv,
    BallBounceEnv,
    WallBallEnv,
    HumanoidTennisCoopEnv,
]

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
    (HumanoidTennisCoopEnv, {}),
]


@pytest.mark.parametrize("env_cls,kwargs", ENV_CLASSES_WITH_KWARGS)
def test_env_constructs_and_steps(env_cls, kwargs):
    env = env_cls(**kwargs)
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
def test_random_rollout_runs_without_nan(env_cls):
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


def test_ball_bounce_info_exposes_bounce_count():
    """``info["bounce_count"]`` tracks the env counter step-for-step.

    The counter is BallBounce's true task metric; exposing it in ``info``
    is what lets ``InfoDictEvalCallback`` aggregate it and the recipe's
    ``success_key="bounce_count"`` compute an eval success rate. The ball
    spawns above the paddle, so a held-still paddle receives at least one
    contact within a few hundred steps.
    """
    env = BallBounceEnv()
    try:
        env.reset(seed=0)
        saw_bounce = False
        for _ in range(500):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert info["bounce_count"] == env.bounce_count
            if info["bounce_count"] >= 1:
                saw_bounce = True
                break
            if terminated or truncated:
                break
        assert saw_bounce, "Ball never contacted the paddle under no-op"
    finally:
        env.close()


def test_ball_bounce_no_reward_without_contact():
    """No touch, no reward -- even at the default ``min_force=0``.

    Regression test: the old rising-edge check (``current >= min_force
    and prev <= 0``) was satisfied by two consecutive zero-touch steps,
    so a default-constructed env (e.g. via ``gymnasium.make``) paid +1
    on every airborne step. The ball spawns ~0.3 above the paddle, so
    the first few steps are guaranteed contact-free.
    """
    env = BallBounceEnv()
    try:
        env.reset(seed=0)
        for _ in range(5):
            _, reward, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert info["touch_sensor"] == 0.0, (
                "Test premise broken: ball already in contact"
            )
            assert reward == 0.0, (
                f"Reward {reward} paid with zero touch force"
            )
    finally:
        env.close()


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_episode_truncates_at_exactly_episode_len(env_cls):
    """``episode_len=N`` must mean N steps, not N+1.

    Regression test for the ``step_number > episode_len`` off-by-one
    that made every env run one extra step (and the README report 751
    for a 750-step BallBalance episode).
    """
    env = env_cls(episode_len=20)
    try:
        env.reset(seed=0)
        steps = 0
        truncated = False
        for _ in range(40):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            steps += 1
            if terminated or truncated:
                break
        # BallBalance can't terminate early under zero action; the other
        # envs might (e.g. WallBall OOB), in which case the truncation
        # boundary isn't exercised -- only assert when truncation fired.
        if truncated:
            assert steps == 20, (
                f"{env_cls.__name__} truncated after {steps} steps, "
                "expected exactly episode_len=20"
            )
        elif env_cls is BallBalanceEnv:
            raise AssertionError(
                "BallBalance episode neither terminated nor truncated "
                "within 2x episode_len"
            )
    finally:
        env.close()


def test_gymnasium_make_ids():
    """The registered env ids resolve via ``gymnasium.make``."""
    import gymnasium

    for env_id in (
        "CourtsideDynamics/BallBalance",
        "CourtsideDynamics/BallBounce",
        "CourtsideDynamics/WallBall",
        "CourtsideDynamics/HumanoidTennisCoop",
    ):
        env = gymnasium.make(env_id)
        try:
            assert env.spec is not None
            assert env.spec.id == env_id
            assert env.spec.version is None
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

    for name in (
        "ball_balance.xml",
        "ball_bounce.xml",
        "wall_ball.xml",
        "humanoid_tennis.xml",
    ):
        p = Path(asset_path(name))
        assert p.is_file(), f"Missing asset: {name}"


class TestWallBallContactPhysics:
    """Pin down the ball's contact restitution with physical measurements.

    In the v1 model, MuJoCo's contact mixing averaged the ball's bouncy
    ``solref`` with the wall/floor critically-damped defaults: measured
    restitution was 0.13 off the floor and 0.00 off the wall, so the
    ball died at the first wall contact and rallies beyond one exchange
    were physically impossible (every training run capped at
    ``bounce_count`` ~1). The ball geom now carries ``priority="1"`` and
    a ``solref`` tuned to a regulation-like bounce; these drop/impact
    tests keep contact deadness from silently returning.
    """

    @staticmethod
    def _fresh_model():
        import mujoco

        from courtside_dynamics.assets import asset_path

        model = mujoco.MjModel.from_xml_path(asset_path("wall_ball.xml"))
        return mujoco, model, mujoco.MjData(model)

    def test_floor_restitution_is_tennis_like(self):
        """A 1 m drop must rebound like a real ball on hard court
        (e ~ 0.73). Below the window the rally dies on the floor;
        above it the soft contact is injecting energy."""
        mujoco, model, data = self._fresh_model()
        qadr = int(model.joint("ball_x").qposadr[0])
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, 1.07]  # 1 m above floor

        prev_z, falling, apex = 1.07, True, None
        for _ in range(3000):
            mujoco.mj_step(model, data)
            z = float(data.qpos[qadr + 2])
            if falling and z > prev_z + 1e-6:
                falling = False
            if not falling and z < prev_z:
                apex = prev_z - 0.07
                break
            prev_z = z
        assert apex is not None, "ball never rebounded off the floor"
        e = np.sqrt(apex / 1.0)
        assert 0.55 <= e <= 0.9, (
            f"floor restitution {e:.2f} outside tennis-like window "
            "(v1 regression measured 0.13)"
        )

    def test_wall_returns_the_ball(self):
        """An 8 m/s wall impact must come back with meaningful speed.
        The v1 wall measured e=0.00: the ball dropped dead at its base
        and a second exchange could never happen."""
        mujoco, model, data = self._fresh_model()
        qadr = int(model.joint("ball_x").qposadr[0])
        dadr = int(model.joint("ball_x").dofadr[0])
        data.qpos[qadr : qadr + 3] = [2.0, 0.0, 1.5]
        data.qvel[dadr] = 8.0  # toward the wall

        v_out = None
        for _ in range(3000):
            mujoco.mj_step(model, data)
            if data.qvel[dadr] < -1e-3:
                v_out = -float(data.qvel[dadr])
                break
        assert v_out is not None, "ball never came back off the wall"
        assert v_out / 8.0 >= 0.3, (
            f"wall restitution {v_out / 8.0:.2f} too dead for rallies "
            "(v1 regression measured 0.00)"
        )

    def test_racket_can_address_grounded_ball(self):
        """At the bottom of its z range the face's lower edge must reach
        below a rolling ball's centre (z=0.07). At the old range the
        face bottomed out at z=0.15, making a grounded ball unreachable
        and 'rolling' an absorbing state."""
        import mujoco

        env = WallBallEnv()
        try:
            env.reset(seed=0)
            zadr = int(env.model.joint("paddle_slide_z").qposadr[0])
            lo = float(env.model.joint("paddle_slide_z").range[0])
            env.data.qpos[zadr] = lo
            mujoco.mj_forward(env.model, env.data)
            face_bottom = float(env.data.body("paddle_head").xpos[2]) - 0.25
            assert face_bottom <= 0.07, (
                f"lowest face edge z={face_bottom:.2f} cannot reach a "
                "rolling ball (centre z=0.07)"
            )
            assert face_bottom >= 0.0, (
                "face bottom dips below the floor at min slide_z"
            )
        finally:
            env.close()


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
            double_bounce_penalty=0.0,
            stall_penalty=0.0,
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

    def test_oracle_completes_gated_returns_under_recipe_settings(self):
        """The oracle must complete real rallies, on the recipe's env.

        This is the canary for the whole reward design, and it must be
        strict: the pre-2026-07 version of this test only required
        ``oracle > no-op``, which the oracle satisfied purely on
        tracking shaping and the touch bonus — zero completed returns.
        It therefore certified an env in which the best available
        strategy was the catch-and-farm exploit that run
        20260712_190054 then learned. The claim that actually matters
        is: a full-state controller can close the rally loop (paddle
        hit -> gated wall contact) under the *training* configuration,
        and doing so out-earns both doing nothing and touch-then-fail.
        """
        from courtside_dynamics.recipes import RECIPES
        from courtside_dynamics.scripted_policies import (
            wall_ball_oracle_action,
        )

        env_kwargs = dict(RECIPES["WallBall"].env_kwargs)
        env_kwargs.pop("render_mode", None)

        def run(action_fn, seed):
            env = WallBallEnv(**env_kwargs)
            try:
                obs, _ = env.reset(seed=seed)
                total = 0.0
                info: dict = {}
                for _ in range(env.episode_len):
                    obs, reward, terminated, truncated, info = env.step(
                        action_fn(obs)
                    )
                    total += reward
                    if terminated or truncated:
                        break
                return total, int(info.get("bounce_count", 0))
            finally:
                env.close()

        noop = lambda o: np.zeros(5, dtype=np.float32)  # noqa: E731
        for seed in (0, 1, 2):
            noop_total, noop_bounces = run(noop, seed=seed)
            oracle_total, oracle_bounces = run(
                wall_ball_oracle_action, seed=seed
            )
            assert noop_bounces == 0, (
                f"seed {seed}: no-op completed a rally?! premise broken"
            )
            assert noop_total <= 0.0, (
                f"seed {seed}: no-op earned {noop_total}, expected <= 0"
            )
            assert oracle_bounces >= 1, (
                f"seed {seed}: oracle completed {oracle_bounces} gated "
                "returns; the env is not demonstrably solvable"
            )
            assert oracle_total > 0.0, (
                f"seed {seed}: oracle completed a rally yet earned "
                f"{oracle_total} — completed returns must pay"
            )
            assert oracle_total > noop_total

    def test_stalled_ball_terminates_episode(self):
        """A dead ball should end the episode well before episode_len.

        The serve flies *at* the paddle, so a no-op policy normally
        either grazes the paddle or lets the ball pass and OOB. To
        exercise the stall path deterministically, we pin the ball to
        the floor and let it settle to contact equilibrium with raw
        ``mj_step`` calls (a ball *placed* at exact surface height is
        not at rest — the underdamped soft contact ejects it into real
        centimetre-scale hops, which correctly count as bounces). Once
        settled, its residual chatter is below the floor-bounce
        debounce speed, so no double bounce fires and the env counts
        ``stall_steps`` of no events and terminates via stall. (The
        stall clock runs from reset; it used to arm only after a first
        paddle/wall event, which left ball-never-touched episodes with
        no stall cutoff at all.)
        """
        import mujoco

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
            # Settle to contact equilibrium outside the env's counters.
            for _ in range(1000):
                mujoco.mj_step(env.model, env.data)
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

    def test_repeat_paddle_hits_pay_bonus_once_per_cycle(self):
        """Only the first paddle contact per wall cycle pays the bonus.

        Regression test for the juggling exploit: every fresh paddle
        contact used to pay ``paddle_hit_bonus``, so bouncing the ball
        on the paddle (no wall involved) farmed unbounded reward. We
        shoot the ball into the paddle face twice with no wall contact
        between; the second contact must register (hit count = 2) but
        pay nothing.
        """
        env = WallBallEnv(
            min_force=0.0,
            paddle_hit_bonus=0.5,
            track_shaping_scale=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])

            def shoot_ball_at_paddle():
                qpos = env.data.qpos.copy()
                qvel = env.data.qvel.copy()
                face = env.data.body("paddle_head").xpos.copy()
                qpos[ball_qposadr : ball_qposadr + 3] = face + np.array(
                    [0.5, 0.0, 0.05]
                )
                qvel[ball_dofadr : ball_dofadr + 6] = 0.0
                qvel[ball_dofadr : ball_dofadr + 3] = [-3.0, 0.0, 0.0]
                env.set_state(qpos, qvel)

            total_reward = 0.0
            hits_seen = 0
            for _ in range(2):
                shoot_ball_at_paddle()
                for _ in range(60):
                    _, reward, terminated, truncated, info = env.step(
                        self._zero_action(env)
                    )
                    total_reward += reward
                    if info["paddle_hit_count"] > hits_seen:
                        hits_seen = info["paddle_hit_count"]
                        break
                    if terminated or truncated:
                        break

            assert hits_seen == 2, (
                f"Expected two paddle contacts, saw {hits_seen}"
            )
            assert env.wall_contact_count == 0, (
                "Test premise broken: ball reached the wall"
            )
            assert abs(total_reward - 0.5) < 1e-9, (
                f"Second paddle hit in the same cycle paid a bonus: "
                f"total reward {total_reward}, expected 0.5"
            )
        finally:
            env.close()

    def test_obs_shape_includes_paddle_to_ball_offset(self):
        """The 21-dim obs ends with the paddle_head→ball relative xyz."""
        env = WallBallEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (21,)
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
            env._pending_shaping = 0.0

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
            env._pending_shaping = 0.0

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
            # Pretend the current cycle paid out +0.5 of shaping
            # advance but produced no paddle return.
            env._paddle_hit_since_last_wall = False
            env._returning = True
            env._pending_shaping = 0.5
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
            env._pending_shaping = 0.0
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
        rew_oob + rew_double_bounce`` has to equal the scalar reward on
        every step -- that's the invariant that makes the composition
        plots trustworthy.
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
                    + info["rew_double_bounce"]
                )
                assert abs(components - reward) < 1e-9, (
                    f"components {components} != reward {reward}"
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_termination_flags_are_mutually_exclusive(self):
        """At most one term_* flag may be True on any step, so the eval
        per-episode fractions partition cleanly (sum <= 1)."""
        env = WallBallEnv(min_force=1.0, episode_len=200)
        try:
            env.reset(seed=0)
            rng = np.random.default_rng(1)
            for _ in range(250):
                action = rng.uniform(
                    -1.0, 1.0, size=env.action_space.shape
                ).astype(np.float32)
                _, _, terminated, truncated, info = env.step(action)
                flags = (
                    int(info["term_oob"])
                    + int(info["term_double_bounce"])
                    + int(info["term_stall"])
                    + int(info["term_timeout"])
                    + int(info["term_nonfinite"])
                )
                assert flags <= 1, f"overlapping termination flags: {info}"
                if terminated or truncated:
                    # Exactly one cause on the terminating step.
                    assert flags == 1, f"no termination cause flagged: {info}"
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
                    assert info["term_double_bounce"] is False
                    assert info["term_stall"] is False
                    assert info["term_timeout"] is False
                    break
            else:
                raise AssertionError("episode never terminated")
        finally:
            env.close()


class TestWallBallAdvanceRefund:
    """Pin down the refundable-advance rules that killed the stall exploit.

    Run 20260712_190054 converged to: catch the serve (banking ~1.25 of
    serve-flight shaping plus the 0.25 touch bonus), deaden the ball,
    and wait out the stall clock — ~1.5 reward per episode, risk-free,
    zero completed rallies, and reward-based selection crowned it
    best_model.zip. These tests assert the two properties that make
    that impossible: a touch followed by anything but a completed
    return nets <= 0, and a completed return strictly out-earns it.
    """

    @staticmethod
    def _zero_action(env):
        return np.zeros(env.action_space.shape, dtype=np.float32)

    @staticmethod
    def _ball_addrs(env):
        return (
            int(env.model.joint("ball_x").qposadr[0]),
            int(env.model.joint("ball_x").dofadr[0]),
        )

    @classmethod
    def _shoot_ball_at_paddle(cls, env):
        """Teleport the ball just in front of the face, flying into it."""
        qposadr, dofadr = cls._ball_addrs(env)
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        face = env.data.body("paddle_head").xpos.copy()
        qpos[qposadr : qposadr + 3] = face + np.array([0.5, 0.0, 0.05])
        qvel[dofadr : dofadr + 6] = 0.0
        qvel[dofadr : dofadr + 3] = [-3.0, 0.0, 0.0]
        env.set_state(qpos, qvel)

    @classmethod
    def _deaden_ball(cls, env):
        """Teleport the ball to a dead drop mid-court, in bounds.

        Placed 5 mm above the floor rather than in exact contact:
        placement *at* the contact distance makes the soft-contact
        solver eject the ball upward fast enough (~0.65 m/s) that its
        re-landings register as genuine debounced floor bounces and
        the episode dies by double bounce instead of the stall these
        tests are about. A 5 mm drop lands at ~0.3 m/s, under the
        ``floor_bounce_min_speed=0.5`` gate, and settles silently.
        """
        qposadr, dofadr = cls._ball_addrs(env)
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        qpos[qposadr : qposadr + 3] = [1.0, 0.0, 0.075]
        qvel[dofadr : dofadr + 6] = 0.0
        env.set_state(qpos, qvel)

    @classmethod
    def _run_until_paddle_hit(cls, env, max_steps=80):
        """Shoot the ball at the paddle; return reward accrued to the hit."""
        cls._shoot_ball_at_paddle(env)
        total = 0.0
        for _ in range(max_steps):
            _, reward, terminated, truncated, info = env.step(
                cls._zero_action(env)
            )
            total += reward
            if info["paddle_hit_count"] >= 1:
                return total
            assert not (terminated or truncated), (
                "Episode ended before the paddle hit; premise broken"
            )
        raise AssertionError("Ball never contacted the paddle")

    def _touch_then_stall_total(self, env):
        """Reward for the 20260712 exploit: touch once, kill the ball."""
        env.reset(seed=0)
        total = self._run_until_paddle_hit(env)
        self._deaden_ball(env)
        for _ in range(env.stall_steps + 50):
            _, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            total += reward
            if terminated or truncated:
                assert info["term_stall"], (
                    f"Expected a stall termination, got "
                    f"{ {k: v for k, v in info.items() if k.startswith('term_')} }"
                )
                assert info["bounce_count"] == 0
                return total
        raise AssertionError("Stall cut never fired")

    def test_touch_then_stall_nets_nonpositive(self):
        """The exploit episode must not out-earn doing nothing.

        Under the pre-fix reward this scenario banked the paddle bonus
        plus all approach shaping (~+1.5 on a real serve) and ended
        free of charge. Now the advances are refunded at the stall cut
        and the stall itself is fined, so the total must come out at
        (or below) -stall_penalty.
        """
        env = WallBallEnv(min_force=1.0)
        try:
            total = self._touch_then_stall_total(env)
            assert total <= 0.0, (
                f"Touch-then-stall netted {total:+.3f}; the exploit "
                "is profitable again"
            )
            assert total <= -0.5 * env.stall_penalty, (
                f"Touch-then-stall netted {total:+.3f}; expected the "
                f"stall fine (~-{env.stall_penalty}) to dominate"
            )
        finally:
            env.close()

    def test_completed_return_outscores_touch_then_fail(self):
        """Closing the rally loop must strictly beat every prefix of it.

        Same setup as the exploit test, except the ball is sent into
        the wall after the paddle hit (a gated wall contact), and the
        episode then times out. The advances stay earned, the +1 pays,
        and the timeout refunds only the (empty) next cycle.
        """
        stall_env = WallBallEnv(min_force=1.0)
        try:
            stall_total = self._touch_then_stall_total(stall_env)
        finally:
            stall_env.close()

        env = WallBallEnv(min_force=1.0, episode_len=160)
        try:
            env.reset(seed=0)
            total = self._run_until_paddle_hit(env)
            # Send the ball into the wall: gate is open, so this wall
            # contact completes the cycle.
            qposadr, dofadr = self._ball_addrs(env)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            qpos[qposadr : qposadr + 3] = [2.7, 0.0, 1.5]
            qvel[dofadr : dofadr + 6] = 0.0
            qvel[dofadr : dofadr + 3] = [3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)
            saw_wall = False
            for _ in range(40):
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total += reward
                if info["bounce_count"] >= 1:
                    saw_wall = True
                    break
                assert not (terminated or truncated)
            assert saw_wall, "Ball never reached the wall; premise broken"
            # Let the episode run out via timeout with a dead ball.
            self._deaden_ball(env)
            while True:
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total += reward
                if terminated or truncated:
                    break
            assert info["term_timeout"], (
                "Expected a timeout ending; premise broken"
            )
            assert info["bounce_count"] == 1
            assert total >= 1.0, (
                f"Completed return netted only {total:+.3f}; the +1 and "
                "earned advances should survive the timeout refund"
            )
            assert total > stall_total + 1.0, (
                f"Completed return ({total:+.3f}) does not clearly beat "
                f"touch-then-stall ({stall_total:+.3f})"
            )
        finally:
            env.close()

    def test_stall_termination_pays_stall_penalty(self):
        """A stall ending costs ``stall_penalty``, like OOB/double bounce."""
        env = WallBallEnv(
            min_force=1.0,
            paddle_hit_bonus=0.0,
            track_shaping_scale=0.0,
            out_of_bounds_penalty=0.0,
            double_bounce_penalty=0.0,
            stall_penalty=2.0,
        )
        try:
            env.reset(seed=0)
            self._deaden_ball(env)
            total = 0.0
            for _ in range(env.stall_steps + 50):
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total += reward
                if terminated or truncated:
                    break
            assert info["term_stall"]
            assert abs(total - (-2.0)) < 1e-6, (
                f"Expected exactly the -2.0 stall fine, got {total:+.4f}"
            )
        finally:
            env.close()

    def test_timeout_refunds_pending_advances(self):
        """An unfinished cycle's advances are refunded at truncation."""
        env = WallBallEnv(
            min_force=1.0,
            episode_len=100,
            paddle_hit_bonus=0.5,
            track_shaping_scale=0.0,
            out_of_bounds_penalty=0.0,
            double_bounce_penalty=0.0,
            stall_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            total = self._run_until_paddle_hit(env)
            assert total >= 0.5 - 1e-9, "Paddle bonus was not paid upfront"
            self._deaden_ball(env)
            while True:
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total += reward
                if terminated or truncated:
                    break
            assert info["term_timeout"]
            assert abs(total) < 1e-6, (
                f"Timeout should refund the pending bonus to net 0, "
                f"got {total:+.4f}"
            )
        finally:
            env.close()

    def test_nonfinite_action_terminates_without_physics(self):
        """A NaN action ends the episode cleanly instead of stepping."""
        env = WallBallEnv(min_force=1.0)
        try:
            obs_before, _ = env.reset(seed=0)
            qpos_before = env.data.qpos.copy()
            obs, reward, terminated, truncated, info = env.step(
                np.full(env.action_space.shape, np.nan)
            )
            assert terminated and not truncated
            assert info["term_nonfinite"]
            assert np.isfinite(obs).all()
            np.testing.assert_allclose(obs, obs_before)
            np.testing.assert_allclose(env.data.qpos, qpos_before)
            assert reward <= 0.0
        finally:
            env.close()

    def test_nonfinite_observation_is_never_emitted(self):
        """If the sim goes nonfinite, the last finite obs is echoed.

        One NaN observation reaching VecNormalize permanently corrupts
        its running statistics, so the guard is load-bearing for every
        normalized training run.
        """
        env = WallBallEnv(min_force=1.0)
        try:
            env.reset(seed=0)
            obs_ok, *_ = env.step(self._zero_action(env))
            assert np.isfinite(obs_ok).all()
            env._get_obs = lambda *a, **k: np.full(  # type: ignore[method-assign]
                env.observation_space.shape, np.nan
            )
            obs, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            assert terminated
            assert info["term_nonfinite"]
            assert np.isfinite(obs).all()
            np.testing.assert_allclose(obs, obs_ok)
        finally:
            env.close()


class TestWallBallDoubleBounce:
    """Pin the wall-ball rally termination rules.

    The ball may touch the floor at most once between consecutive
    paddle/wall contacts; the second consecutive floor bounce ends the
    episode immediately, as in real wall ball. Bounce detection runs at
    substep resolution with a pre-impact-speed debounce (so the contact
    chatter of a settling or rolling ball doesn't read as bounces), and
    touch events are filtered to real ball contacts (so a paddle sagging
    onto the floor doesn't fake a hit).
    """

    @staticmethod
    def _zero_action(env):
        return np.zeros(env.action_space.shape, dtype=np.float32)

    @staticmethod
    def _place_ball(env, pos, vel=(0.0, 0.0, 0.0)):
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
        ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
        qpos[ball_qposadr : ball_qposadr + 3] = pos
        qvel[ball_dofadr : ball_dofadr + 6] = 0.0
        qvel[ball_dofadr : ball_dofadr + 3] = vel
        env.set_state(qpos, qvel)

    @staticmethod
    def _settle(env, substeps=1000):
        """Run raw physics (no env counters) until placed objects rest.

        A ball *placed* at exact surface height is not at rest: the
        underdamped soft contact ejects it into real centimetre-scale
        hops that correctly count as bounces. Rollouts never teleport a
        ball onto the surface, so tests that want a genuinely resting
        ball must settle it first.
        """
        import mujoco

        for _ in range(substeps):
            mujoco.mj_step(env.model, env.data)

    def test_second_floor_bounce_terminates(self):
        """Drop the ball mid-court: bounce 1 plays on, bounce 2 ends it."""
        env = WallBallEnv(min_force=1.0)
        try:
            env.reset(seed=0)
            # In-bounds, clear of both the paddle and the wall, so the
            # ball just bounces vertically in place.
            self._place_ball(env, [1.5, 0.0, 1.5])

            terminated = truncated = False
            info: dict = {}
            for _ in range(400):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if info["floor_bounce_total"] == 1:
                    assert not terminated, (
                        "episode ended on the FIRST floor bounce"
                    )
                if terminated or truncated:
                    break
            assert terminated, (
                "second floor bounce never terminated the episode"
            )
            assert info["term_double_bounce"] is True
            assert info["floor_bounce_total"] == 2
            assert info["floor_bounce_count"] == 2
        finally:
            env.close()

    def test_paddle_contact_resets_consecutive_count(self):
        """A paddle hit between floor bounces restarts the rally count,
        so 'one bounce, return, one bounce' is legal play and only
        *consecutive* floor bounces terminate."""
        env = WallBallEnv(min_force=0.0)
        try:
            env.reset(seed=0)
            # First floor bounce.
            self._place_ball(env, [1.5, 0.0, 1.0])
            terminated = truncated = False
            info: dict = {}
            for _ in range(200):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                assert not (terminated or truncated)
                if info["floor_bounce_count"] == 1:
                    break
            assert info["floor_bounce_count"] == 1

            # Paddle contact: shoot the ball into the face.
            face = env.data.body("paddle_head").xpos.copy()
            self._place_ball(
                env, face + np.array([0.5, 0.0, 0.05]), vel=(-3.0, 0.0, 0.0)
            )
            hit = False
            for _ in range(60):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if info["paddle_hit_count"] >= 1:
                    hit = True
                    break
                if terminated or truncated:
                    break
            assert hit, "ball-into-paddle never registered a contact"
            assert info["floor_bounce_count"] == 0, (
                "paddle contact did not reset the consecutive bounce count"
            )

            # A fresh bounce after the reset is the FIRST of a new
            # sequence: the episode continues despite two total bounces.
            self._place_ball(env, [1.5, 0.0, 1.0])
            for _ in range(200):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if info["floor_bounce_count"] == 1:
                    break
                if terminated or truncated:
                    break
            assert info["floor_bounce_count"] == 1
            assert not terminated, (
                "bounce after a paddle reset still terminated the episode"
            )
        finally:
            env.close()

    def test_wall_contact_resets_consecutive_count(self):
        """A wall rebound between floor bounces restarts the rally
        count, mirroring the paddle-reset rule — 'bounce, wall, bounce'
        is legal play and must not terminate."""
        env = WallBallEnv(min_force=1.0)
        try:
            env.reset(seed=0)
            # First floor bounce.
            self._place_ball(env, [1.5, 0.0, 1.0])
            terminated = truncated = False
            info: dict = {}
            for _ in range(200):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                assert not (terminated or truncated)
                if info["floor_bounce_count"] == 1:
                    break
            assert info["floor_bounce_count"] == 1

            # Wall rebound: drive the ball into the wall (same setup as
            # the wall-edge reward tests).
            self._place_ball(env, [2.7, 0.0, 1.5], vel=(3.0, 0.0, 0.0))
            reached_wall = False
            for _ in range(60):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if info["wall_contact_count"] >= 1:
                    reached_wall = True
                    break
                if terminated or truncated:
                    break
            assert reached_wall, "ball-into-wall never registered a contact"
            assert info["floor_bounce_count"] == 0, (
                "wall contact did not reset the consecutive bounce count"
            )

            # The next bounce is again the FIRST of a new sequence.
            self._place_ball(env, [1.5, 0.0, 1.0])
            for _ in range(200):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if info["floor_bounce_count"] == 1:
                    break
                if terminated or truncated:
                    break
            assert info["floor_bounce_count"] == 1
            assert not terminated, (
                "bounce after a wall reset still terminated the episode"
            )
        finally:
            env.close()

    def test_resting_ball_chatter_is_not_a_bounce(self):
        """A ball resting on the floor must not accumulate bounces from
        contact chatter — the pre-impact-speed debounce filters the
        near-zero-energy contact onsets of a settling ball, which a
        naive rising-edge counter reads as a rapid string of bounces."""
        env = WallBallEnv(min_force=1.0, stall_steps=300)
        try:
            env.reset(seed=0)
            self._place_ball(env, [1.5, 0.0, 0.07])
            self._settle(env)
            info: dict = {}
            for _ in range(120):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                assert info["floor_bounce_total"] == 0, (
                    "resting-ball contact chatter was counted as a bounce"
                )
                assert not terminated
            assert info["term_double_bounce"] is False
        finally:
            env.close()

    def test_double_bounce_applies_penalty(self):
        """The terminating double-bounce step subtracts the configured
        penalty (surfaced in ``rew_double_bounce``)."""
        env = WallBallEnv(
            min_force=1.0,
            double_bounce_penalty=2.5,
            track_shaping_scale=0.0,
            paddle_hit_bonus=0.0,
            out_of_bounds_penalty=0.0,
        )
        try:
            env.reset(seed=0)
            self._place_ball(env, [1.5, 0.0, 1.5])
            terminating_reward = None
            for _ in range(400):
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                if terminated or truncated:
                    terminating_reward = reward
                    assert info["term_double_bounce"] is True
                    assert abs(info["rew_double_bounce"] + 2.5) < 1e-9
                    break
            assert terminating_reward is not None, "episode never terminated"
            assert terminating_reward <= -2.4, (
                f"Expected ~-2.5 double-bounce penalty, got {terminating_reward}"
            )
        finally:
            env.close()

    def test_paddle_floor_scrape_is_not_a_paddle_hit(self):
        """An unpowered paddle sags until its face slams the floor.
        That transient used to fire the paddle touch sensor — paying the
        bonus, opening the wall +1 gate, and resetting the stall clock —
        with the ball metres away. With touch events filtered to real
        ball contacts the slam is ignored, and with the ball parked
        in-bounds (settled, so its chatter is below the bounce debounce)
        the episode ends via stall exactly ``stall_steps`` after reset.
        The paddle is re-raised to its reset pose after the settle phase
        so the sag and slam happen *inside* the env loop, where the
        filter is exercised: under unfiltered sampling this rollout
        registers a phantom paddle hit (~35 N at roughly step 131) and
        the stall clock resets, failing both assertions."""
        env = WallBallEnv(min_force=0.0, stall_steps=150)
        try:
            env.reset(seed=0)
            # Park the ball in-bounds, at rest, away from paddle & wall.
            self._place_ball(env, [3.0, 4.0, 0.07])
            self._settle(env)
            # Re-raise the paddle so it sags afresh during the env loop.
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            for name in (
                "paddle_slide_x", "paddle_slide_y", "paddle_slide_z",
                "paddle_yaw", "paddle_pitch",
            ):
                joint = env.model.joint(name)
                qpos[int(joint.qposadr[0])] = 0.0
                qvel[int(joint.dofadr[0])] = 0.0
            env.set_state(qpos, qvel)

            floor_geom = int(env.model.geom("floor").id)
            face_geom = int(env.model.geom("paddle_face").id)
            face_scraped_floor = False
            terminated = truncated = False
            info: dict = {}
            steps = 0
            for step in range(1, 400):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                steps = step
                for i in range(int(env.data.ncon)):
                    pair = {
                        int(env.data.contact.geom1[i]),
                        int(env.data.contact.geom2[i]),
                    }
                    if pair == {floor_geom, face_geom}:
                        face_scraped_floor = True
                if terminated or truncated:
                    break
            assert face_scraped_floor, (
                "test premise broken: the sagging paddle never touched "
                "the floor"
            )
            assert info["paddle_hit_count"] == 0, (
                "paddle-floor scrape was counted as a paddle hit"
            )
            assert terminated and info["term_stall"] is True
            assert steps == 150, (
                f"stall should fire exactly stall_steps after reset, "
                f"got {steps}"
            )
        finally:
            env.close()


class TestWallBallServe:
    """Pin the parameterized serve-angle distribution."""

    def test_serve_vy_within_configured_range(self):
        """|vy| after reset stays inside [serve_vy_min, serve_vy_max]
        (sign randomized), and widening the ceiling actually produces
        serves beyond the old 1.8 default."""
        env = WallBallEnv(serve_vy_min=0.8, serve_vy_max=2.6)
        try:
            dofadr = int(env.model.joint("ball_x").dofadr[0])
            vys = []
            for seed in range(40):
                env.reset(seed=seed)
                vys.append(float(env.data.qvel[dofadr + 1]))
            abs_vys = [abs(v) for v in vys]
            assert min(abs_vys) >= 0.8 - 1e-6
            assert max(abs_vys) <= 2.6 + 1e-6
            assert max(abs_vys) > 1.8, (
                "widened ceiling never produced a serve beyond the old "
                "default range"
            )
            assert any(v < 0 for v in vys) and any(v > 0 for v in vys), (
                "serve side is not randomized"
            )
        finally:
            env.close()

    def test_serve_vy_range_validated(self):
        """A floor of 0 (or an inverted range) would let straight
        serves hit a parked racket and re-open the no-op free-ride, so
        construction must fail fast."""
        with pytest.raises(ValueError, match="serve_vy_min"):
            WallBallEnv(serve_vy_min=0.0)
        with pytest.raises(ValueError, match="serve_vy_min"):
            WallBallEnv(serve_vy_min=2.0, serve_vy_max=1.0)
