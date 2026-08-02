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
# Contact-force thresholds are kept aligned with the corresponding recipes.
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
    Bounce random powered paddle motion produces deliberate rebound events.
    Wall Ball is intentionally excluded: under the
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


class TestBallBounceEvents:
    """Pin down deliberate, substep-resolved BallBounce events."""

    @staticmethod
    def _zero_action(env):
        return np.zeros(env.action_space.shape, dtype=np.float32)

    @staticmethod
    def _set_ball_state(env, *, position, velocity):
        qpos = env.init_qpos.copy()
        qvel = env.init_qvel.copy()
        joint = env.model.joint("ball_freejoint")
        qposadr = int(joint.qposadr[0])
        dofadr = int(joint.dofadr[0])
        qpos[qposadr : qposadr + 3] = position
        qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        qvel[:] = 0.0
        qvel[dofadr : dofadr + 3] = velocity
        env.set_state(qpos, qvel)

    def test_controlled_upward_strike_earns_one_bounce(self):
        """A powered top-face rebound is detected once and exposed in info."""
        env = BallBounceEnv(min_force=100.0, release_substeps=5)
        try:
            env.reset(seed=0)
            env.set_state(env.init_qpos.copy(), env.init_qvel.copy())
            action = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)

            info = {}
            reward = 0.0
            saw_contact = False
            for _ in range(60):
                _, reward, terminated, truncated, info = env.step(action)
                if info["contact_started"]:
                    saw_contact = True
                    assert reward == 0.0
                    assert info["contact_latched"]
                if reward > 0.0 or terminated or truncated:
                    break

            assert saw_contact
            assert reward == 1.0
            assert info["valid_bounce"] == 1
            assert info["bounce_count"] == env.bounce_count == 1
            assert info["contact_peak_force"] > 100.0
            assert info["impact_relative_speed"] <= -env.min_impact_speed
            assert (
                info["paddle_normal_speed_at_impact"]
                >= env.min_paddle_upward_speed
            )
            assert info["rebound_relative_speed"] >= env.min_rebound_speed
            assert not info["contact_latched"]

            # The same physical contact cannot pay again while the ball is
            # separating from the paddle.
            for _ in range(10):
                _, reward, terminated, truncated, info = env.step(action)
                assert reward == 0.0
                assert info["bounce_count"] == 1
                assert not terminated and not truncated
        finally:
            env.close()

    def test_rotation_only_top_face_strike_uses_contact_point_velocity(self):
        """Angular paddle motion at an edge can qualify without origin motion."""
        env = BallBounceEnv(min_force=20.0)
        try:
            env.reset(seed=0)
            qpos = env.init_qpos.copy()
            qvel = env.init_qvel.copy()
            rotate_y = env.model.joint("rotate_y")
            qvel[int(rotate_y.dofadr[0])] = 2.0
            ball = env.model.joint("ball_freejoint")
            qposadr = int(ball.qposadr[0])
            dofadr = int(ball.dofadr[0])
            qpos[qposadr : qposadr + 3] = [-0.15, 0.0, 0.068]
            qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[dofadr : dofadr + 6] = 0.0
            qvel[dofadr + 2] = -2.0
            env.set_state(qpos, qvel)

            total_reward = 0.0
            info = {}
            for _ in range(5):
                _, reward, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                total_reward += reward
                if reward > 0.0 or terminated or truncated:
                    break

            assert total_reward == 1.0
            assert info["paddle_normal_speed_at_impact"] > 0.05
            assert info["bounce_count"] == 1
        finally:
            env.close()

    def test_brief_between_frame_contact_is_sampled_but_not_rewarded(self):
        """A passive impact that is zero at the frame boundary is still seen."""
        env = BallBounceEnv(min_force=100.0)
        try:
            env.reset(seed=0)
            self._set_ball_state(
                env,
                position=(0.0, 0.0, 0.068),
                velocity=(0.0, 0.0, -2.0),
            )

            _, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            assert float(env.data.sensor("touch_sensor").data[0]) == 0.0
            assert info["touch_sensor"] > 100.0
            assert info["contact_started"]
            assert info["contact_episode_count"] == 1
            assert info["contact_rejection_reason"] == "passive_paddle"
            assert reward == 0.0
            assert info["bounce_count"] == 0
            assert not terminated and not truncated
        finally:
            env.close()

    def test_underside_contact_is_rejected_and_terminates(self):
        env = BallBounceEnv()
        try:
            env.reset(seed=0)
            qpos = env.init_qpos.copy()
            qvel = env.init_qvel.copy()
            slider = env.model.joint("slider_z")
            qpos[int(slider.qposadr[0])] = 0.2
            ball = env.model.joint("ball_freejoint")
            qposadr = int(ball.qposadr[0])
            dofadr = int(ball.dofadr[0])
            qpos[qposadr : qposadr + 3] = [0.0, 0.0, 0.132]
            qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[:] = 0.0
            qvel[dofadr + 2] = 2.0
            env.set_state(qpos, qvel)

            _, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            assert info["touch_sensor"] > 0.0
            assert info["contact_rejection_reason"] == "underside_or_edge"
            assert info["bounce_count"] == 0
            assert reward == 0.0
            assert terminated and not truncated
            assert info["termination_reason"] == "ball_below_paddle"
            assert info["term_ball_dropped"]
            assert not info["term_nonfinite"]
            assert not info["term_timeout"]
        finally:
            env.close()

    def test_lateral_clearance_rearms_after_two_substeps(self):
        """Leaving over an edge is a real release even near paddle height."""
        env = BallBounceEnv(release_substeps=2, release_clearance=0.002)
        try:
            env.reset(seed=0)
            self._set_ball_state(
                env,
                position=(0.4, 0.0, 0.06),
                velocity=(0.0, 0.0, 0.0),
            )
            env._contact_latched = True
            env._contact_candidate = False
            env._contact_local_position[:] = (0.2, 0.0, 0.02)

            env._step_mujoco_simulation(self._zero_action(env), 1)
            assert env._contact_latched
            assert env._contact_release_progress == 1

            env._step_mujoco_simulation(self._zero_action(env), 1)
            assert not env._contact_latched
            assert env._contact_release_progress == 0
        finally:
            env.close()

    def test_ball_below_elevated_paddle_terminates_without_contact(self):
        env = BallBounceEnv()
        try:
            env.reset(seed=0)
            qpos = env.init_qpos.copy()
            qvel = env.init_qvel.copy()
            slider = env.model.joint("slider_z")
            qpos[int(slider.qposadr[0])] = 0.2
            ball = env.model.joint("ball_freejoint")
            qposadr = int(ball.qposadr[0])
            qpos[qposadr : qposadr + 3] = [0.5, 0.0, 0.15]
            qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[:] = 0.0
            env.set_state(qpos, qvel)

            obs, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            assert obs[2] > 0.0
            assert info["contact_episode_count"] == 0
            assert reward == 0.0
            assert terminated and not truncated
            assert info["term_ball_dropped"]
        finally:
            env.close()

    def test_hidden_nonfinite_state_terminates_with_finite_outputs(self):
        env = BallBounceEnv()
        try:
            env.reset(seed=0)
            ball = env.model.joint("ball_freejoint")
            dofadr = int(ball.dofadr[0])
            env.data.qvel[dofadr + 3] = np.nan

            obs, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert terminated and not truncated
            assert info["term_nonfinite"]
            assert info["termination_reason"] == "nonfinite_state"
        finally:
            env.close()

    def test_noop_cannot_earn_a_valid_bounce(self):
        """Gravity/passive restitution must not satisfy the training metric."""
        for seed in range(20):
            env = BallBounceEnv(min_force=100.0, episode_len=300)
            try:
                env.reset(seed=seed)
                info = {}
                while True:
                    _, reward, terminated, truncated, info = env.step(
                        self._zero_action(env)
                    )
                    assert reward == 0.0
                    if terminated or truncated:
                        break
                assert info["bounce_count"] == env.bounce_count == 0
            finally:
                env.close()

    def test_scripted_oracle_reaches_sustained_juggling_gate(self):
        from courtside_dynamics.scripted_policies import (
            ball_bounce_oracle_action,
        )

        counts = []
        for seed in range(20):
            env = BallBounceEnv(min_force=100.0)
            try:
                observation, _ = env.reset(seed=seed)
                while True:
                    observation, _, terminated, truncated, info = env.step(
                        ball_bounce_oracle_action(observation)
                    )
                    if terminated or truncated:
                        break
                counts.append(info["bounce_count"])
            finally:
                env.close()

        assert np.mean(counts) >= 9.0
        assert sum(count >= 10 for count in counts) >= 15

    def test_reset_states_respect_limits_and_unit_quaternion(self):
        env = BallBounceEnv()
        try:
            for seed in range(100):
                env.reset(seed=seed)
                for name in (
                    "rotate_x",
                    "rotate_y",
                    "rotate_z",
                    "slider_x",
                    "slider_y",
                    "slider_z",
                ):
                    joint = env.model.joint(name)
                    value = float(env.data.qpos[int(joint.qposadr[0])])
                    assert float(joint.range[0]) <= value <= float(joint.range[1])
                quaternion = env.data.joint("ball_freejoint").qpos[3:7]
                assert np.linalg.norm(quaternion) == pytest.approx(1.0)
        finally:
            env.close()

    def test_rotation_units_and_vertical_motor_authority(self):
        env = BallBounceEnv()
        try:
            np.testing.assert_allclose(
                env.model.joint("rotate_x").range, [-0.3, 0.3]
            )
            assert float(env.model.actuator("slider_z_motor").gear[0]) == 100.0
            assert len(env.action_names) == env.action_space.shape[0]
            assert env.action_names == tuple(
                env.model.actuator(index).name
                for index in range(env.model.nu)
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


class TestWallBallPaddleInterface:
    """Pin the simplified face-only target-control contract."""

    def test_three_normalized_targets_map_around_home(self):
        env = WallBallEnv()
        try:
            assert env.action_space.shape == (3,)
            np.testing.assert_allclose(env.action_space.low, -1.0)
            np.testing.assert_allclose(env.action_space.high, 1.0)
            assert env.action_space.dtype == np.float32
            assert env.action_names == (
                "paddle_target_x",
                "paddle_target_y",
                "paddle_target_z",
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.full(3, -1.0)),
                [-3.0, -3.0, -0.9],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.zeros(3)),
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.ones(3)),
                [2.0, 3.0, 2.0],
            )
            # Finite inputs outside the declared Box are clipped.
            np.testing.assert_allclose(
                env._action_to_controls(np.array([2.0, -2.0, 0.0])),
                [2.0, -3.0, 0.0],
            )
            with pytest.raises(ValueError, match="action must have shape"):
                env.step(np.zeros(5, dtype=np.float32))
        finally:
            env.close()

    def test_face_is_only_paddle_geom_and_upward_pitch_is_fixed(self):
        env = WallBallEnv()
        try:
            joint_names = {
                env.model.joint(index).name for index in range(env.model.njnt)
            }
            geom_names = {
                env.model.geom(index).name for index in range(env.model.ngeom)
            }
            assert "paddle_yaw" not in joint_names
            assert "paddle_pitch" not in joint_names
            assert "paddle_grip" not in geom_names
            assert "paddle_shaft" not in geom_names
            assert {name for name in geom_names if name.startswith("paddle_")} == {
                "paddle_face"
            }

            env.reset(seed=0)
            for _ in range(30):
                env.do_simulation(
                    np.array([0.25, -0.25, 0.25], dtype=np.float32),
                    env.frame_skip,
                )
            targets = env._action_to_controls(
                np.array([0.25, -0.25, 0.25], dtype=np.float32)
            )
            actual = np.array(
                [
                    float(env.data.joint(name).qpos[0])
                    for name in (
                        "paddle_slide_x",
                        "paddle_slide_y",
                        "paddle_slide_z",
                    )
                ]
            )
            np.testing.assert_allclose(actual, targets, atol=0.03)
            angle = np.deg2rad(10.0)
            expected_rotation = np.array(
                [
                    [np.cos(angle), 0.0, -np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [np.sin(angle), 0.0, np.cos(angle)],
                ]
            )
            rotation = env.data.body("paddle_head").xmat.reshape(3, 3)
            np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
            np.testing.assert_allclose(
                rotation[:, 0],
                [np.cos(angle), 0.0, np.sin(angle)],
                atol=1e-12,
            )
            assert np.max(np.abs(env.data.actuator_force)) <= 100.0 + 1e-9
            assert np.isfinite(env.data.qpos).all()
            assert np.isfinite(env.data.qvel).all()
        finally:
            env.close()


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

    def test_fixed_paddle_pitch_lifts_a_horizontal_return(self):
        """A horizontal inbound ball must leave the fixed face upward."""
        mujoco, model, data = self._fresh_model()
        qadr = int(model.joint("ball_x").qposadr[0])
        dadr = int(model.joint("ball_x").dofadr[0])
        data.qpos[qadr : qadr + 3] = [-1.1, 0.0, 1.2]
        data.qvel[dadr : dadr + 3] = [-6.0, 0.0, 0.0]
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        touched_paddle = False
        rebound_velocity = None
        for _ in range(1000):
            mujoco.mj_step(model, data)
            for contact in data.contact:
                geom_names = {
                    model.geom(contact.geom1).name,
                    model.geom(contact.geom2).name,
                }
                if geom_names == {"ball_geom", "paddle_face"}:
                    touched_paddle = True
                    break
            if touched_paddle and data.qvel[dadr] > 0.1:
                rebound_velocity = data.qvel[dadr : dadr + 3].copy()
                break

        assert touched_paddle, "horizontal test ball never touched the paddle"
        assert rebound_velocity is not None, "ball never rebounded from paddle"
        assert rebound_velocity[2] > 0.1, (
            f"fixed pitch did not lift return: vz={rebound_velocity[2]:.2f}"
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
            paddle = env.data.body("paddle_head")
            rotation = paddle.xmat.reshape(3, 3)
            half_sizes = np.array([0.02, 0.2, 0.25])
            z_half_extent = float(np.abs(rotation[2]) @ half_sizes)
            face_bottom = float(paddle.xpos[2]) - z_half_extent
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

        noop = lambda o: np.zeros(3, dtype=np.float32)  # noqa: E731
        oracle_bounces_seen = []
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
            oracle_bounces_seen.append(oracle_bounces)

        # This controller change is intended to unlock more than the
        # one-return ceiling seen in the failed training run. Keep a
        # small deterministic multi-return canary in addition to the
        # per-seed feasibility floor above.
        assert np.median(oracle_bounces_seen) >= 2.0
        assert max(oracle_bounces_seen) >= 3

    def test_legal_hit_x_instrumentation_tracks_contact_positions(self):
        """Positional-play diagnostics: legal hits accumulate their world
        x so eval can measure WHERE the policy plays (a deep-fence stage
        "mastered" by camping the fence front must be visible as such).
        The sum/mean pair is exact for zero-hit episodes: sum 0.0 with
        count 0 carries no fake position, and the mean only divides by
        real hits."""
        from courtside_dynamics.recipes import RECIPES
        from courtside_dynamics.scripted_policies import (
            wall_ball_oracle_action,
        )

        env_kwargs = dict(RECIPES["WallBall"].env_kwargs)
        env_kwargs.pop("render_mode", None)
        env = WallBallEnv(**env_kwargs)
        try:
            obs, _ = env.reset(seed=0)
            # Pre-hit: both keys present and exactly zero.
            obs, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
            assert info["legal_paddle_hit_x_sum"] == 0.0
            assert info["legal_paddle_hit_x_mean"] == 0.0

            obs, _ = env.reset(seed=0)
            info = {}
            for _ in range(env.episode_len):
                obs, _, terminated, truncated, info = env.step(
                    wall_ball_oracle_action(obs)
                )
                if terminated or truncated:
                    break
            count = info["legal_paddle_hit_count"]
            assert count >= 1, "oracle made no legal hit; premise broken"
            mean_x = info["legal_paddle_hit_x_mean"]
            assert mean_x == pytest.approx(
                info["legal_paddle_hit_x_sum"] / count
            )
            # Contact positions must be physically plausible: inside the
            # paddle's world-space x workspace (with a small margin for
            # the end-of-frame sampling).
            low, high = env._paddle_x_world_range
            assert low - 0.2 <= mean_x <= high + 0.2
        finally:
            env.close()

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

    def test_obs_layout_and_markov_fields(self):
        """The 23-dim observation removes the deleted yaw/pitch state
        while retaining rel-xyz, spin, stall progress, and the pending
        advance and recovery-bonus eligibility needed to make the task
        Markovian."""
        env = WallBallEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert obs.shape == (23,)
            ball = np.array(env.data.joint("ball_x").qpos[:3])
            paddle = np.array(env.data.body("paddle_head").xpos)
            np.testing.assert_allclose(obs[14:17], ball - paddle, atol=1e-6)
            # Spin mirrors the free joint's rotational qvel.
            np.testing.assert_allclose(
                obs[17:20],
                np.asarray(env.data.joint("ball_x").qvel)[3:6],
                atol=1e-6,
            )
            # Fresh reset: stall clock and pending advance both zero.
            assert obs[20] == 0.0
            assert obs[21] == 0.0
            assert obs[22] == 0.0

            # Under no-op the stall clock must tick up monotonically
            # (the serve produces no paddle/wall edge for many steps).
            _, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
            obs2, *_ = env.step(np.zeros(3, dtype=np.float32))
            assert 0.0 < obs2[20] <= 1.0
            assert obs2[20] == env._steps_since_event / env.stall_steps
        finally:
            env.close()

    def test_obs_pending_advance_tracks_paddle_bonus(self):
        """After a paddle hit, the pending-advance obs must show the
        banked bonus (the amount a failed cycle would claw back)."""
        env = WallBallEnv(
            min_force=1.0,
            paddle_hit_bonus=0.5,
            track_shaping_scale=0.0,
        )
        try:
            env.reset(seed=0)
            ball_qposadr = int(env.model.joint("ball_x").qposadr[0])
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            face = env.data.body("paddle_head").xpos.copy()
            qpos[ball_qposadr : ball_qposadr + 3] = face + np.array(
                [0.5, 0.0, 0.05]
            )
            qvel[ball_dofadr : ball_dofadr + 6] = 0.0
            qvel[ball_dofadr : ball_dofadr + 3] = [-3.0, 0.0, 0.0]
            env.set_state(qpos, qvel)
            for _ in range(60):
                obs, _, terminated, truncated, info = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                assert not (terminated or truncated)
                if info["paddle_hit_count"] >= 1:
                    break
            else:
                raise AssertionError("Ball never contacted the paddle")
            assert obs[21] == 0.5, (
                f"pending_advance obs is {obs[21]}, expected the 0.5 bonus"
            )
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

        Whatever the dynamics do, every ``rew_*`` component has to sum
        to the scalar reward on every step -- that's the invariant that
        makes the composition plots trustworthy.
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
                    + info["rew_recoverable_bounce"]
                    + info["rew_shaping"]
                    + info["rew_oob"]
                    + info["rew_double_bounce"]
                    + info["rew_stall"]
                    + info["rew_style_violation"]
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
                    + int(info["term_style_violation"])
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
            assert info["pre_bounce_legal_paddle_hit_count"] == 0
            assert info["post_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 0
            assert info["post_bounce_completed_return_count"] == 0
            assert info["event_pre_bounce_legal_paddle_hit"] is False
            assert info["event_post_bounce_legal_paddle_hit"] is False
            assert info["event_opening_volley"] is False
            assert info["event_post_bounce_completed_return"] is False
        finally:
            env.close()


class TestWallBallDoubleBounce:
    """Pin the wall-ball rally termination rules.

    The ball may touch the floor at most once between consecutive
    paddle/wall contacts; the second consecutive floor bounce ends the
    episode immediately, as in real wall ball. Bounce detection runs at
    substep resolution with a pre-impact-speed debounce (so the contact
    chatter of a settling or rolling ball doesn't read as bounces), and
    touch events are filtered to real ball contacts.
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

    def test_zero_target_holds_face_clear_and_does_not_fake_a_hit(self):
        """The normalized zero action is an active home-position hold.

        With the ball parked and settled far away, the target-controlled
        face must stay clear of the floor, never register a phantom hit,
        and let the stall clock fire exactly on schedule.
        """
        env = WallBallEnv(min_force=0.0, stall_steps=150)
        try:
            env.reset(seed=0)
            # Park the ball in-bounds, at rest, away from paddle & wall.
            self._place_ball(env, [3.0, 4.0, 0.07])
            self._settle(env)

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
            assert not face_scraped_floor, "home-position hold scraped the floor"
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


class TestWallBallRallyStyleConfiguration:
    """Pin the public configuration shared by all WallBall presets.

    ``paddle_home_x`` and ``paddle_x_target_range`` are deliberately
    world-space values: recipe authors should not need to know that the
    XML's slide joint is measured relative to a base at x=-1.7.  The
    policy still emits the same three normalized targets in every mode.
    """

    @pytest.mark.parametrize("rally_style", ["open", "volley", "one_bounce"])
    def test_all_rally_styles_construct_with_same_spaces(self, rally_style):
        env = WallBallEnv(rally_style=rally_style)
        try:
            obs, _ = env.reset(seed=0)
            assert env.rally_style == rally_style
            assert env.action_space.shape == (3,)
            assert obs.shape == (23,)
        finally:
            env.close()

    def test_invalid_rally_style_is_rejected(self):
        with pytest.raises(ValueError, match="rally_style"):
            WallBallEnv(rally_style="serve_and_volley")

    @pytest.mark.parametrize(
        "kwargs,parameter",
        [
            # -5.0 sits inside the extended physical workspace but
            # outside the frozen default mapping range (-4.7, 0.3).
            ({"paddle_home_x": -5.0}, "paddle_home_x"),
            (
                {
                    "paddle_home_x": -2.7,
                    "paddle_x_target_range": (-3.2, -3.0),
                },
                "paddle_home_x",
            ),
            # -9.0 is beyond even the extended physical workspace
            # (-8.2), so this exercises the physical-bound check.
            ({"paddle_x_target_range": (-9.0, -2.0)}, "paddle_x_target_range"),
            ({"paddle_x_target_range": (-2.0, 0.5)}, "paddle_x_target_range"),
            ({"paddle_x_target_range": (-2.0, -3.0)}, "paddle_x_target_range"),
            ({"paddle_x_target_range": (-2.0,)}, "paddle_x_target_range"),
            ({"paddle_x_target_range": (-2.0, np.inf)}, "paddle_x_target_range"),
        ],
    )
    def test_world_space_paddle_configuration_is_validated(
        self, kwargs, parameter
    ):
        with pytest.raises((TypeError, ValueError), match=parameter):
            WallBallEnv(**kwargs)

    @pytest.mark.parametrize("penalty", [-1.0, np.nan, np.inf])
    def test_style_violation_penalty_is_finite_and_nonnegative(self, penalty):
        with pytest.raises(ValueError, match="style_violation_penalty"):
            WallBallEnv(style_violation_penalty=penalty)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("recovery_reset_probability", -0.01),
            ("recovery_reset_probability", 1.01),
            ("recovery_reset_probability", np.nan),
            ("post_bounce_reset_fraction", -0.01),
            ("post_bounce_reset_fraction", 1.01),
            ("post_bounce_reset_fraction", np.inf),
        ],
    )
    def test_recovery_reset_probabilities_are_validated(self, name, value):
        with pytest.raises(ValueError, match=name):
            WallBallEnv(**{name: value})

    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    def test_recoverable_bounce_bonus_is_validated(self, value):
        with pytest.raises(ValueError, match="recoverable_bounce_bonus"):
            WallBallEnv(recoverable_bounce_bonus=value)

    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    def test_recoverable_bounce_lateral_limit_is_validated(self, value):
        with pytest.raises(
            ValueError, match="recoverable_bounce_lateral_limit"
        ):
            WallBallEnv(recoverable_bounce_lateral_limit=value)

    def test_recovery_probability_can_be_changed_safely(self):
        env = WallBallEnv(rally_style="one_bounce")
        try:
            env.recovery_reset_probability = 0.75
            env.post_bounce_reset_fraction = 0.25
            assert env.recovery_reset_probability == 0.75
            assert env.post_bounce_reset_fraction == 0.25
            with pytest.raises(ValueError, match="recovery_reset_probability"):
                env.recovery_reset_probability = 1.1
            with pytest.raises(ValueError, match="post_bounce_reset_fraction"):
                env.post_bounce_reset_fraction = -0.1
        finally:
            env.close()

    def test_baseline_world_range_maps_around_configured_home(self):
        env = WallBallEnv(
            rally_style="one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -2.2),
        )
        try:
            np.testing.assert_allclose(
                env._action_to_controls(np.array([-1.0, 0.0, 0.0])),
                [-1.5, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.zeros(3)),
                [-1.0, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.array([1.0, 0.0, 0.0])),
                [-0.5, 0.0, 0.0],
            )
            # Configuring x must not alter the y/z action contract.
            np.testing.assert_allclose(
                env._action_to_controls(np.array([0.0, -1.0, -1.0])),
                [-1.0, -3.0, -0.9],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.array([0.0, 1.0, 1.0])),
                [-1.0, 3.0, 2.0],
            )
        finally:
            env.close()

    def test_reset_places_paddle_at_configured_world_home(self):
        env = WallBallEnv(
            rally_style="one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -2.2),
        )
        try:
            for seed in range(10):
                env.reset(seed=seed)
                slide_x = float(env.data.joint("paddle_slide_x").qpos[0])
                head_x = float(env.data.body("paddle_head").xpos[0])
                # Base x=-1.7 plus configured qpos=-1.0 gives the
                # requested world home x=-2.7.  Reset noise remains.
                assert abs(slide_x - (-1.0)) <= 0.011
                assert abs(head_x - (-2.7)) <= 0.011
                assert -3.2 <= head_x <= -2.2
        finally:
            env.close()

    def test_default_open_configuration_remains_bit_for_bit_compatible(self):
        env = WallBallEnv()
        try:
            env.reset(seed=0)
            assert env.rally_style == "open"
            np.testing.assert_allclose(
                env._action_to_controls(np.full(3, -1.0)),
                [-3.0, -3.0, -0.9],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.zeros(3)),
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                env._action_to_controls(np.ones(3)),
                [2.0, 3.0, 2.0],
            )
        finally:
            env.close()


class TestWallBallRecoveryResetCurriculum:
    """Recovery fragments must teach state, not manufacture rally credit."""

    @staticmethod
    def _ball_state(env):
        return (
            np.asarray(env.data.joint("ball_x").qpos[:3]).copy(),
            np.asarray(env.data.joint("ball_x").qvel[:3]).copy(),
        )

    def test_zero_probability_always_preserves_normal_one_bounce_reset(self):
        env = WallBallEnv(
            rally_style="one_bounce",
            recovery_reset_probability=0.0,
            post_bounce_reset_fraction=1.0,
        )
        try:
            for seed in range(20):
                obs, reset_info = env.reset(seed=seed)
                assert env.reset_mode == "normal"
                assert reset_info["reset_mode"] == "normal"
                assert reset_info["reset_mode_id"] == 0
                assert env.rally_phase_name == "await_bounce"
                assert env.floor_bounce_count == 0
                assert env.floor_bounce_total == 0
                assert env.bounce_count == 0
                assert env.one_bounce_recovery_count == 0
                assert env.one_bounce_return_count == 0
                assert env._recoverable_bounce_eligible is False
                assert obs[-1] == 0.0
        finally:
            env.close()

    @pytest.mark.parametrize("rally_style", ["open", "volley"])
    def test_recovery_resets_are_never_sampled_outside_one_bounce(
        self, rally_style
    ):
        env = WallBallEnv(
            rally_style=rally_style,
            recovery_reset_probability=1.0,
            post_bounce_reset_fraction=1.0,
        )
        try:
            for seed in range(10):
                _, reset_info = env.reset(seed=seed)
                assert env.reset_mode == "normal"
                assert reset_info["reset_mode"] == "normal"
        finally:
            env.close()

    def test_incoming_wall_reset_uses_calibrated_state_and_clean_counters(self):
        env = WallBallEnv(
            rally_style="one_bounce",
            recovery_reset_probability=1.0,
            post_bounce_reset_fraction=0.0,
        )
        try:
            for seed in range(20):
                obs, reset_info = env.reset(seed=seed)
                pos, vel = self._ball_state(env)
                assert env.reset_mode == "incoming_wall"
                assert reset_info["reset_mode_id"] == 1
                assert 3.1 <= pos[0] <= 3.5
                assert -0.75 <= pos[1] <= 0.75
                assert 1.3 <= pos[2] <= 1.7
                assert -8.5 <= vel[0] <= -7.5
                assert -0.4 <= vel[1] <= 0.4
                assert -1.5 <= vel[2] <= 0.3
                assert env.rally_phase_name == "await_bounce"
                assert env.floor_bounce_count == 0
                assert env.floor_bounce_total == 0
                assert env.bounce_count == 0
                assert env.wall_contact_count == 0
                assert env.legal_paddle_hit_count == 0
                assert env.pre_bounce_legal_paddle_hit_count == 0
                assert env.post_bounce_legal_paddle_hit_count == 0
                assert env.opening_volley_count == 0
                assert env.post_bounce_completed_return_count == 0
                assert env._floor_bounce_since_last_wall_or_reset is False
                assert env.one_bounce_recovery_count == 0
                assert env.one_bounce_return_count == 0
                assert env._recoverable_bounce_eligible is False
                assert obs[-1] == 0.0
        finally:
            env.close()

    def test_post_bounce_reset_starts_awaiting_paddle_without_credit(self):
        env = WallBallEnv(
            rally_style="one_bounce",
            recovery_reset_probability=1.0,
            post_bounce_reset_fraction=1.0,
        )
        try:
            for seed in range(20):
                obs, reset_info = env.reset(seed=seed)
                pos, vel = self._ball_state(env)
                assert env.reset_mode == "post_bounce"
                assert reset_info["reset_mode_id"] == 2
                assert -0.5 <= pos[0] <= 0.5
                assert -0.75 <= pos[1] <= 0.75
                assert 0.10 <= pos[2] <= 0.15
                assert -7.5 <= vel[0] <= -6.0
                assert -0.4 <= vel[1] <= 0.4
                assert 2.5 <= vel[2] <= 3.5
                assert env.rally_phase_name == "await_paddle"
                assert env.floor_bounce_count == 1
                assert env.floor_bounce_total == 1
                assert env.bounce_count == 0
                assert env.wall_contact_count == 0
                assert env.legal_paddle_hit_count == 0
                assert env.one_bounce_recovery_count == 0
                assert env.one_bounce_return_count == 0
                assert env._recoverable_bounce_eligible is False
                assert obs[13] == 1.0
                assert obs[-1] == 0.0
        finally:
            env.close()


class TestWallBallRallyStyleInfoContract:
    """Keep the training/evaluation diagnostics stable across modes."""

    _EVENT_KEYS = (
        "event_floor_bounce",
        "event_legal_paddle_hit",
        "event_pre_bounce_legal_paddle_hit",
        "event_post_bounce_legal_paddle_hit",
        "event_opening_volley",
        "event_completed_return",
        "event_post_bounce_completed_return",
        "event_recoverable_bounce",
        "event_post_wall_bounce",
        "event_style_violation",
    )

    @pytest.mark.parametrize("rally_style", ["open", "volley", "one_bounce"])
    def test_info_exposes_style_state_counters_and_step_events(
        self, rally_style
    ):
        env = WallBallEnv(rally_style=rally_style)
        try:
            env.reset(seed=0)
            _, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert np.isfinite(reward)
            assert not (terminated or truncated)
            assert isinstance(info["rally_phase_name"], str)
            assert info["legal_paddle_hit_count"] == 0
            assert info["pre_bounce_legal_paddle_hit_count"] == 0
            assert info["post_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 0
            assert info["post_bounce_completed_return_count"] == 0
            assert info["one_bounce_recovery_count"] == 0
            assert info["one_bounce_return_count"] == 0
            assert info["return_count"] == 0
            assert info["style_violation_reason"] is None
            assert info["rew_style_violation"] == 0.0
            assert info["rew_recoverable_bounce"] == 0.0
            assert info["recoverable_bounce_score"] == 0.0
            assert info["recoverable_bounce_eligible"] is False
            assert info["reset_mode"] == "normal"
            assert info["reset_mode_id"] == 0
            assert info["term_style_violation"] is False
            for key in self._EVENT_KEYS:
                assert info[key] is False
        finally:
            env.close()

    @pytest.mark.parametrize(
        "rally_style,reset_phase",
        [
            ("open", "await_paddle"),
            ("volley", "await_paddle"),
            ("one_bounce", "await_bounce"),
        ],
    )
    def test_reset_clears_style_metrics_and_restores_initial_phase(
        self, rally_style, reset_phase
    ):
        env = WallBallEnv(rally_style=rally_style)
        try:
            env.reset(seed=0)
            env.legal_paddle_hit_count = 4
            env.pre_bounce_legal_paddle_hit_count = 1
            env.post_bounce_legal_paddle_hit_count = 3
            env.opening_volley_count = 1
            env.post_bounce_completed_return_count = 2
            env.one_bounce_recovery_count = 3
            env.one_bounce_return_count = 2
            env._rally_phase = 2
            env.reset(seed=1)
            _, _, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert info["legal_paddle_hit_count"] == 0
            assert info["pre_bounce_legal_paddle_hit_count"] == 0
            assert info["post_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 0
            assert info["post_bounce_completed_return_count"] == 0
            assert info["one_bounce_recovery_count"] == 0
            assert info["one_bounce_return_count"] == 0
            assert info["rally_phase_name"] == reset_phase
        finally:
            env.close()


class TestWallBallRallyStyleSemantics:
    """Exercise strict volley/baseline rules with real MuJoCo contacts."""

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

    def _step_until(self, env, event_key, limit=120):
        for _ in range(limit):
            transition = env.step(self._zero_action(env))
            if transition[4][event_key]:
                return transition
            if transition[2] or transition[3]:
                pytest.fail(
                    f"episode ended before {event_key}: {transition[4]}"
                )
        pytest.fail(f"physics never produced {event_key} in {limit} steps")

    def _floor_bounce(self, env):
        self._place_ball(env, [1.5, 0.0, 0.7], vel=[0.0, 0.0, -2.0])
        return self._step_until(env, "event_floor_bounce")

    def _paddle_hit(self, env, *, legal=True):
        face = env.data.body("paddle_head").xpos.copy()
        self._place_ball(
            env,
            face + np.array([0.5, 0.0, 0.05]),
            vel=[-3.0, 0.0, 0.0],
        )
        key = "event_legal_paddle_hit" if legal else "event_style_violation"
        return self._step_until(env, key)

    def _wall_hit(self, env):
        self._place_ball(env, [2.7, 0.0, 1.5], vel=[3.0, 0.0, 0.0])
        return self._step_until(env, "event_completed_return")

    @staticmethod
    def _strict_env(rally_style, **kwargs):
        defaults = {
            "rally_style": rally_style,
            "min_force": 1.0,
            "track_shaping_scale": 0.0,
            "paddle_hit_bonus": 0.0,
            "out_of_bounds_penalty": 0.0,
            "double_bounce_penalty": 0.0,
            "stall_penalty": 0.0,
            "style_violation_penalty": 2.5,
            "stall_steps": 300,
        }
        defaults.update(kwargs)
        return WallBallEnv(**defaults)

    def test_one_bounce_cycle_requires_floor_then_paddle_then_wall(self):
        env = self._strict_env(
            "one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -2.2),
        )
        try:
            env.reset(seed=0)

            env._steps_since_event = 17
            _, _, terminated, truncated, info = self._floor_bounce(env)
            assert not (terminated or truncated)
            assert info["event_floor_bounce"] is True
            assert info["event_style_violation"] is False
            assert info["rally_phase_name"] == "await_paddle"
            assert info["floor_bounce_count"] == 1
            # A required baseline bounce is progress, so it restarts
            # the stall window just like a paddle/wall edge.
            assert env._steps_since_event == 0

            _, _, terminated, truncated, info = self._paddle_hit(env)
            assert not (terminated or truncated)
            assert info["event_legal_paddle_hit"] is True
            assert info["event_style_violation"] is False
            assert info["rally_phase_name"] == "await_wall"
            assert info["paddle_hit_count"] == 1
            assert info["legal_paddle_hit_count"] == 1
            assert info["one_bounce_recovery_count"] == 1
            assert info["one_bounce_return_count"] == 0

            _, reward, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert reward == pytest.approx(1.0)
            assert info["event_completed_return"] is True
            assert info["rally_phase_name"] == "await_bounce"
            assert info["one_bounce_return_count"] == 1
            assert info["return_count"] == 1
            assert info["bounce_count"] == 1
        finally:
            env.close()

    def test_opening_volley_and_post_bounce_hit_are_distinguished(self):
        env = self._strict_env("open")
        try:
            env.reset(seed=0)

            _, _, terminated, truncated, info = self._paddle_hit(env)
            assert not (terminated or truncated)
            assert info["event_legal_paddle_hit"] is True
            assert info["event_pre_bounce_legal_paddle_hit"] is True
            assert info["event_post_bounce_legal_paddle_hit"] is False
            assert info["event_opening_volley"] is True
            assert info["pre_bounce_legal_paddle_hit_count"] == 1
            assert info["post_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 1

            _, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["event_completed_return"] is True
            assert info["event_post_bounce_completed_return"] is False
            assert info["post_bounce_completed_return_count"] == 0

            self._floor_bounce(env)
            _, _, terminated, truncated, info = self._paddle_hit(env)
            assert not (terminated or truncated)
            assert info["event_legal_paddle_hit"] is True
            assert info["event_pre_bounce_legal_paddle_hit"] is False
            assert info["event_post_bounce_legal_paddle_hit"] is True
            assert info["event_opening_volley"] is False
            assert info["pre_bounce_legal_paddle_hit_count"] == 1
            assert info["post_bounce_legal_paddle_hit_count"] == 1
            assert info["opening_volley_count"] == 1
            assert (
                info["pre_bounce_legal_paddle_hit_count"]
                + info["post_bounce_legal_paddle_hit_count"]
                == info["legal_paddle_hit_count"]
            )

            _, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["event_completed_return"] is True
            assert info["event_post_bounce_completed_return"] is True
            assert info["post_bounce_completed_return_count"] == 1
            assert info["bounce_count"] == 2
        finally:
            env.close()

    @pytest.mark.parametrize(
        ("events", "expected_post_bounce"),
        [
            (
                [
                    (0, 0, "floor"),
                    (1, 1, "paddle"),
                    (2, 2, "wall"),
                ],
                True,
            ),
            (
                [
                    (0, 1, "paddle"),
                    (1, 0, "floor"),
                    (2, 2, "wall"),
                ],
                False,
            ),
        ],
    )
    def test_open_same_frame_sequence_uses_physical_contact_order(
        self, monkeypatch, events, expected_post_bounce
    ):
        env = self._strict_env("open")
        try:
            env.reset(seed=0)

            def fake_simulation(_action, _frame_skip):
                env._substep_contact_events = events
                env._substep_paddle_touch = env.min_force + 1.0
                env._substep_wall_touch = env.min_force + 1.0

            monkeypatch.setattr(env, "do_simulation", fake_simulation)
            _, reward, terminated, truncated, info = env.step(
                self._zero_action(env)
            )

            assert not (terminated or truncated)
            assert reward == pytest.approx(1.0)
            assert info["event_floor_bounce"] is True
            assert info["event_legal_paddle_hit"] is True
            assert info["event_completed_return"] is True
            assert (
                info["event_post_bounce_legal_paddle_hit"]
                is expected_post_bounce
            )
            assert (
                info["event_pre_bounce_legal_paddle_hit"]
                is not expected_post_bounce
            )
            assert (
                info["event_post_bounce_completed_return"]
                is expected_post_bounce
            )
        finally:
            env.close()

    def test_post_bounce_reset_seeds_sequence_state_without_hit_credit(self):
        env = self._strict_env(
            "one_bounce",
            recovery_reset_probability=1.0,
            post_bounce_reset_fraction=1.0,
        )
        try:
            _, reset_info = env.reset(seed=0)
            assert reset_info["reset_mode"] == "post_bounce"
            assert env.floor_bounce_total == 1
            assert env.legal_paddle_hit_count == 0
            assert env.post_bounce_legal_paddle_hit_count == 0

            _, _, terminated, truncated, info = self._paddle_hit(env)
            assert not (terminated or truncated)
            assert info["event_post_bounce_legal_paddle_hit"] is True
            assert info["event_pre_bounce_legal_paddle_hit"] is False
            assert info["event_opening_volley"] is False
            assert info["post_bounce_legal_paddle_hit_count"] == 1
            assert info["pre_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 0

            _, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["event_post_bounce_completed_return"] is True
            assert info["post_bounce_completed_return_count"] == 1
        finally:
            env.close()

    def test_nonfinite_action_preserves_sequence_counts_without_events(self):
        env = self._strict_env("open")
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)

            _, reward, terminated, truncated, info = env.step(
                np.full(env.action_space.shape, np.nan, dtype=np.float32)
            )
            assert np.isfinite(reward)
            assert terminated and not truncated
            assert info["term_nonfinite"] is True
            assert info["legal_paddle_hit_count"] == 1
            assert info["post_bounce_legal_paddle_hit_count"] == 1
            assert info["pre_bounce_legal_paddle_hit_count"] == 0
            assert info["opening_volley_count"] == 0
            assert info["post_bounce_completed_return_count"] == 0
            assert info["event_pre_bounce_legal_paddle_hit"] is False
            assert info["event_post_bounce_legal_paddle_hit"] is False
            assert info["event_opening_volley"] is False
            assert info["event_post_bounce_completed_return"] is False
        finally:
            env.close()

    def test_synthetic_recovery_bounce_never_earns_agent_bonus(self):
        env = self._strict_env(
            "one_bounce",
            recovery_reset_probability=1.0,
            post_bounce_reset_fraction=0.0,
            recoverable_bounce_bonus=0.6,
            recoverable_bounce_lateral_limit=2.0,
        )
        try:
            env.reset(seed=0)
            assert env.reset_mode == "incoming_wall"
            assert env._recoverable_bounce_eligible is False
            self._place_ball(
                env,
                [1.5, 0.0, 0.7],
                vel=[-4.0, 0.0, -2.0],
            )
            _, reward, terminated, truncated, info = self._step_until(
                env, "event_floor_bounce"
            )
            assert not (terminated or truncated)
            assert reward == pytest.approx(0.0)
            assert info["event_post_wall_bounce"] is False
            assert info["event_recoverable_bounce"] is False
            assert info["recoverable_bounce_score"] == 0.0
            assert info["rew_recoverable_bounce"] == 0.0
        finally:
            env.close()

    def test_agent_wall_return_arms_one_central_recoverable_bounce_bonus(self):
        env = self._strict_env(
            "one_bounce",
            recoverable_bounce_bonus=0.6,
            recoverable_bounce_lateral_limit=2.0,
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)
            obs, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["event_completed_return"] is True
            assert info["recoverable_bounce_eligible"] is True
            assert obs[-1] == 1.0

            self._place_ball(
                env,
                [1.5, 0.0, 0.7],
                vel=[-4.0, 0.0, -2.0],
            )
            obs, reward, terminated, truncated, info = self._step_until(
                env, "event_floor_bounce"
            )
            assert not (terminated or truncated)
            assert info["event_post_wall_bounce"] is True
            assert info["event_recoverable_bounce"] is True
            assert info["recoverable_bounce_score"] == pytest.approx(1.0)
            assert info["rew_recoverable_bounce"] == pytest.approx(0.6)
            assert reward == pytest.approx(0.6)
            assert info["recoverable_bounce_eligible"] is False
            assert obs[-1] == 0.0

            # A later bounce cannot farm the same policy-generated return.
            self._place_ball(
                env,
                [1.5, 0.0, 0.7],
                vel=[-4.0, 0.0, -2.0],
            )
            _, reward, terminated, truncated, info = self._step_until(
                env, "event_floor_bounce"
            )
            assert terminated and not truncated
            assert info["event_post_wall_bounce"] is False
            assert info["event_recoverable_bounce"] is False
            assert info["recoverable_bounce_score"] == 0.0
            assert info["rew_recoverable_bounce"] == 0.0
            assert reward == pytest.approx(0.0)
        finally:
            env.close()

    def test_post_wall_bounce_outside_lateral_limit_consumes_without_bonus(self):
        env = self._strict_env(
            "one_bounce",
            recoverable_bounce_bonus=0.6,
            recoverable_bounce_lateral_limit=0.5,
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)
            self._wall_hit(env)
            assert env._recoverable_bounce_eligible is True

            self._place_ball(
                env,
                [1.5, 1.0, 0.7],
                vel=[-4.0, 0.0, -2.0],
            )
            _, reward, terminated, truncated, info = self._step_until(
                env, "event_floor_bounce"
            )
            assert not (terminated or truncated)
            assert info["event_post_wall_bounce"] is True
            assert info["event_recoverable_bounce"] is False
            assert info["recoverable_bounce_score"] == 0.0
            assert info["rew_recoverable_bounce"] == 0.0
            assert reward == pytest.approx(0.0)
            assert env._recoverable_bounce_eligible is False
        finally:
            env.close()

    def test_recoverable_projection_uses_world_space_paddle_front(self):
        env = self._strict_env(
            "one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -2.1),
            recoverable_bounce_lateral_limit=2.0,
        )
        try:
            env.reset(seed=0)
            env._substep_floor_bounce_state = (
                np.array([1.0, 0.0, 0.07]),
                np.array([-4.0, 1.0, -2.0]),
            )
            # World front x=-2.1 gives t=(1.0 - -2.1)/4=.775,
            # projected y=.775, and score=1-.775/2=.6125.
            assert env._recoverable_bounce_score() == pytest.approx(0.6125)
        finally:
            env.close()

    def test_bounce_behind_entire_paddle_lane_is_not_recoverable(self):
        env = self._strict_env(
            "one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -2.1),
            recoverable_bounce_lateral_limit=2.0,
        )
        try:
            env.reset(seed=0)
            env._substep_floor_bounce_state = (
                np.array([-3.21, 0.0, 0.07]),
                np.array([-4.0, 0.0, -2.0]),
            )
            assert env._recoverable_bounce_score() == 0.0
        finally:
            env.close()

    @staticmethod
    def _baseline_serve_env_and_oracle(recovery_reset_probability=0.0):
        """WallBallBaseline env + matched oracle fn.

        ``recovery_reset_probability`` defaults to zero so serve tests
        exercise the calibrated serve itself -- with the recipe value
        (0.6) most seeds would start from synthetic recovery fragments
        and a serve test would not prove what its name claims. Pass 1.0
        to force every reset onto a fragment instead.
        """
        from courtside_dynamics.recipes import RECIPES
        from courtside_dynamics.scripted_policies import (
            wall_ball_baseline_oracle_action,
        )

        env_kwargs = dict(RECIPES["WallBallBaseline"].env_kwargs)
        env_kwargs.pop("render_mode", None)
        env_kwargs["recovery_reset_probability"] = recovery_reset_probability
        lane = env_kwargs["paddle_x_target_range"]
        home = env_kwargs["paddle_home_x"]

        def oracle(obs):
            return wall_ball_baseline_oracle_action(
                obs, paddle_x_target_range=lane, paddle_home_x=home
            )

        return WallBallEnv(**env_kwargs), oracle

    def test_calibrated_baseline_serve_is_physically_solvable(self):
        """An observation-only controller closes a legal cycle on real serves."""
        env, oracle = self._baseline_serve_env_and_oracle()
        try:
            for seed in range(5):
                obs, _ = env.reset(seed=seed)
                for _ in range(env.episode_len):
                    obs, _, terminated, truncated, info = env.step(
                        oracle(obs)
                    )
                    if info["bounce_count"] >= 1:
                        break
                    assert not (terminated or truncated), (
                        f"seed {seed}: calibrated serve ended before a "
                        f"legal return: {info}"
                    )
                assert info["bounce_count"] >= 1
                assert info["one_bounce_recovery_count"] >= 1
                assert info["one_bounce_return_count"] >= 1
                assert info["floor_bounce_total"] >= 1
        finally:
            env.close()

    def test_raised_success_bar_is_scriptable_from_standard_serves(self):
        """The recipe's success threshold (bounce_count >= 2) is provably
        reachable by an observation-only controller from standard serves.

        Guards against scoring training runs on a bar no policy can hit:
        the 2026-07-16 calibration sweep measured the oracle at 92%
        second returns over 500 serves under the (-3.2, -1.6) lane with
        slide damping 8 (seeds 0-9 reach [4, 2, 3, 4, 2, 2, 1, 3, 2, 2]
        returns), while at the pre-sweep (-3.2, -2.1)/damping-5 geometry
        a placement-blind tracker recovered a second exchange in exactly
        0% of episodes -- the run 20260714_050506 plateau.
        """
        env, oracle = self._baseline_serve_env_and_oracle()
        try:
            reached_two = 0
            for seed in range(10):
                obs, _ = env.reset(seed=seed)
                info = {}
                for _ in range(env.episode_len):
                    obs, _, terminated, truncated, info = env.step(
                        oracle(obs)
                    )
                    if terminated or truncated or info["bounce_count"] >= 2:
                        break
                assert info["bounce_count"] >= 1, (
                    f"seed {seed}: oracle failed even the first return: "
                    f"{info}"
                )
                if info["bounce_count"] >= 2:
                    reached_two += 1
            # 9/10 measured; >= 7 tolerates minor cross-platform contact
            # jitter without letting the bar drift out of reach.
            assert reached_two >= 7, (
                f"only {reached_two}/10 seeds completed a second return"
            )
        finally:
            env.close()

    def test_recovery_fragments_remain_solvable_by_the_oracle(self):
        """Fragment starts must stay closable under the recipe geometry.

        The serve tests above force recovery_reset_probability to zero,
        so this is the only oracle coverage of the synthetic
        incoming-wall / post-bounce fragments the training curriculum
        serves. Measured 2026-07-16: 47/50 fragment starts close a
        legal cycle at the (-3.2, -1.6)/damping-8 geometry.
        """
        env, oracle = self._baseline_serve_env_and_oracle(
            recovery_reset_probability=1.0
        )
        try:
            closed = 0
            for seed in range(10):
                obs, reset_info = env.reset(seed=seed)
                assert reset_info["reset_mode"] in {
                    "incoming_wall",
                    "post_bounce",
                }
                info = {}
                for _ in range(env.episode_len):
                    obs, _, terminated, truncated, info = env.step(
                        oracle(obs)
                    )
                    if terminated or truncated or info["bounce_count"] >= 1:
                        break
                if info["bounce_count"] >= 1:
                    closed += 1
            assert closed >= 8, (
                f"only {closed}/10 fragment starts closed a legal cycle"
            )
        finally:
            env.close()

    def test_open_mode_reports_floor_event_without_changing_old_rule(self):
        env = self._strict_env("open")
        try:
            env.reset(seed=0)
            _, _, terminated, truncated, info = self._floor_bounce(env)
            assert not (terminated or truncated)
            assert info["event_floor_bounce"] is True
            assert info["event_style_violation"] is False
            assert info["style_violation_reason"] is None
            assert info["term_style_violation"] is False
            assert info["floor_bounce_count"] == 1
        finally:
            env.close()

    def test_volley_floor_contact_is_terminal_style_violation(self):
        env = self._strict_env("volley")
        try:
            env.reset(seed=0)
            _, reward, terminated, truncated, info = self._floor_bounce(env)
            assert terminated and not truncated
            assert reward == pytest.approx(-2.5)
            assert info["event_floor_bounce"] is True
            assert info["event_style_violation"] is True
            assert info["style_violation_reason"] == "floor_in_volley"
            assert info["rew_style_violation"] == pytest.approx(-2.5)
            assert info["term_style_violation"] is True
            assert info["term_double_bounce"] is False
            assert info["legal_paddle_hit_count"] == 0
        finally:
            env.close()

    def test_one_bounce_paddle_before_floor_is_terminal_violation(self):
        env = self._strict_env("one_bounce")
        try:
            env.reset(seed=0)
            _, reward, terminated, truncated, info = self._paddle_hit(
                env, legal=False
            )
            assert terminated and not truncated
            assert reward == pytest.approx(-2.5)
            assert info["event_legal_paddle_hit"] is False
            assert info["event_style_violation"] is True
            assert info["style_violation_reason"] == "paddle_before_bounce"
            assert info["rew_style_violation"] == pytest.approx(-2.5)
            assert info["term_style_violation"] is True
            # Raw contact telemetry is preserved even though this hit
            # is not allowed to open the reward gate.
            assert info["paddle_hit_count"] == 1
            assert info["legal_paddle_hit_count"] == 0
            assert info["one_bounce_return_count"] == 0
        finally:
            env.close()

    def test_one_bounce_early_touch_softened_is_fine_not_termination(self):
        """With ``early_touch_penalty`` set, a pre-bounce paddle touch pays
        a non-terminal fine, keeps the gate shut, and the rally can still
        be completed legally afterwards."""
        env = self._strict_env("one_bounce", early_touch_penalty=0.25)
        try:
            env.reset(seed=0)
            face = env.data.body("paddle_head").xpos.copy()
            self._place_ball(
                env,
                face + np.array([0.5, 0.0, 0.05]),
                vel=[-3.0, 0.0, 0.0],
            )
            _, reward, terminated, truncated, info = self._step_until(
                env, "event_early_touch"
            )
            assert not terminated and not truncated
            assert reward == pytest.approx(-0.25)
            assert info["rew_early_touch"] == pytest.approx(-0.25)
            assert info["early_touch_count"] == 1
            assert info["event_style_violation"] is False
            assert info["style_violation_reason"] is None
            assert info["term_style_violation"] is False
            # The touch never opens the return gate.
            assert info["event_legal_paddle_hit"] is False
            assert info["legal_paddle_hit_count"] == 0
            assert info["paddle_hit_count"] == 1
            assert info["rally_phase_name"] == "await_bounce"

            # The rally continues: bounce -> paddle -> wall still pays.
            self._floor_bounce(env)
            self._paddle_hit(env)
            _, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["bounce_count"] == 1
            assert info["early_touch_count"] == 1
        finally:
            env.close()

    def test_early_touch_penalty_is_validated(self):
        for bad in (-0.1, np.nan, np.inf):
            with pytest.raises(ValueError, match="early_touch_penalty"):
                WallBallEnv(rally_style="one_bounce", early_touch_penalty=bad)

    def test_one_bounce_early_touch_does_not_reset_stall_clock(self):
        """A fined touch is a fault, not rally progress. If it reset the
        stall window, a periodic tap (0.25 per <=stall_steps) would hold
        a dead ball to the penalty-free timeout for less than the stall
        fine -- the catch-and-stall exploit class run 20260712_190054
        converged to."""
        env = self._strict_env("one_bounce", early_touch_penalty=0.25)
        try:
            env.reset(seed=0)
            env._steps_since_event = 17
            face = env.data.body("paddle_head").xpos.copy()
            self._place_ball(
                env,
                face + np.array([0.5, 0.0, 0.05]),
                vel=[-3.0, 0.0, 0.0],
            )
            self._step_until(env, "event_early_touch")
            # The counter kept accruing through the fined touch instead
            # of being zeroed by it.
            assert env._steps_since_event > 17
        finally:
            env.close()

    def test_one_bounce_early_touch_disarms_recoverable_bounce(self):
        """A pre-bounce touch re-aims the rebound, so the placement bonus
        armed by the preceding legal return must not pay on the
        paddle-steered trajectory (otherwise a cheap tap converts any
        armed rebound into a perfectly-placed 'recoverable' bounce)."""
        env = self._strict_env(
            "one_bounce",
            early_touch_penalty=0.25,
            recoverable_bounce_bonus=1.0,
            recoverable_bounce_lateral_limit=2.0,
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)
            _, _, _, _, info = self._wall_hit(env)
            assert info["recoverable_bounce_eligible"] is True

            face = env.data.body("paddle_head").xpos.copy()
            self._place_ball(
                env,
                face + np.array([0.5, 0.0, 0.05]),
                vel=[-3.0, 0.0, 0.0],
            )
            _, _, _, _, info = self._step_until(env, "event_early_touch")
            assert info["recoverable_bounce_eligible"] is False

            _, _, _, _, info = self._floor_bounce(env)
            assert info["event_recoverable_bounce"] is False
            assert info["rew_recoverable_bounce"] == 0.0
        finally:
            env.close()

    def test_paddle_joint_damping_overrides_model_per_instance(self):
        """The kwarg rewrites the compiled model's slide damping; the
        default None keeps the shared XML calibration (5) so open/volley
        presets are untouched by a baseline-only physics change."""
        slide_joints = ("paddle_slide_x", "paddle_slide_y", "paddle_slide_z")

        env = WallBallEnv()
        try:
            assert env.paddle_joint_damping is None
            for joint in slide_joints:
                dofadr = int(env.model.joint(joint).dofadr[0])
                assert env.model.dof_damping[dofadr] == pytest.approx(5.0)
        finally:
            env.close()

        env = WallBallEnv(paddle_joint_damping=8.0)
        try:
            for joint in slide_joints:
                dofadr = int(env.model.joint(joint).dofadr[0])
                assert env.model.dof_damping[dofadr] == pytest.approx(8.0)
            # The ball's free joint keeps its zero flight damping.
            ball_dofadr = int(env.model.joint("ball_x").dofadr[0])
            assert env.model.dof_damping[ball_dofadr] == pytest.approx(0.0)
        finally:
            env.close()

    def test_paddle_joint_damping_is_validated(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match="paddle_joint_damping"):
                WallBallEnv(paddle_joint_damping=bad)

    def test_one_bounce_weak_return_softened_allows_fined_retry(self):
        """With ``weak_return_penalty`` set, an outgoing shot that drops
        short is a fined retry, not a termination: the failed cycle's
        advances claw back, the ball stays live, and the next paddle hit
        legally reopens the gate."""
        env = self._strict_env(
            "one_bounce", weak_return_penalty=0.1, paddle_hit_bonus=0.25
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)  # opens the gate, +0.25 pending
            _, reward, terminated, truncated, info = self._floor_bounce(env)
            assert not terminated and not truncated
            # Fine plus the clawed-back pending paddle bonus.
            assert reward == pytest.approx(-0.1 - 0.25)
            assert info["rew_weak_return"] == pytest.approx(-0.1)
            assert info["rew_paddle"] == pytest.approx(-0.25)
            assert info["event_weak_return"] is True
            assert info["weak_return_count"] == 1
            assert info["event_style_violation"] is False
            assert info["term_style_violation"] is False
            assert info["rally_phase_name"] == "await_paddle"
            assert info["floor_bounce_count"] == 1

            # The retry closes a legal cycle.
            self._paddle_hit(env)
            _, _, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert info["bounce_count"] == 1
            assert info["weak_return_count"] == 1
        finally:
            env.close()

    def test_repeat_paddle_contact_does_not_reset_stall_clock(self):
        """Only the gate-opening hit is rally progress. A free stall-clock
        reset on repeat contacts let touch-then-deaden ride a banked
        outright bonus to the penalty-free truncation with periodic taps
        (measured +0.25 under the bootstrap recipe)."""
        env = self._strict_env("one_bounce", first_hit_bonus=0.25)
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)  # gate opens: this IS progress
            assert env._steps_since_event == 0
            assert env.rally_phase_name == "await_wall"

            env._steps_since_event = 17
            face = env.data.body("paddle_head").xpos.copy()
            self._place_ball(
                env,
                face + np.array([0.5, 0.0, 0.05]),
                vel=[-3.0, 0.0, 0.0],
            )
            before = env.paddle_hit_count
            for _ in range(120):
                _, _, terminated, truncated, info = env.step(
                    self._zero_action(env)
                )
                assert not (terminated or truncated)
                if info["paddle_hit_count"] > before:
                    break
            else:
                pytest.fail("physics never produced a repeat contact")
            # The repeat contact was tolerated but the stall clock kept
            # counting through it.
            assert env._steps_since_event > 17
        finally:
            env.close()

    def test_paddle_start_x_may_sit_outside_the_action_mapping(self):
        """A curriculum start clamps to the physical workspace, not the
        (possibly narrower) mapping lane."""
        env = WallBallEnv(
            rally_style="one_bounce",
            paddle_home_x=-2.7,
            paddle_x_target_range=(-3.2, -1.6),
            paddle_start_x=-0.5,  # far in front of the mapping lane
        )
        try:
            env.reset(seed=0)
            world_x = float(
                env.data.joint("paddle_slide_x").qpos[0]
                + env._paddle_x_origin
            )
            assert world_x == pytest.approx(-0.5, abs=0.05)
        finally:
            env.close()

    def test_first_hit_bonus_pays_outright_once_per_episode(self):
        """The first gate-opening hit pays an unconditional bonus exactly
        once per episode: later cycles' first hits earn only the
        refundable advance, and a fresh episode pays again."""
        env = self._strict_env(
            "one_bounce", first_hit_bonus=0.5, paddle_hit_bonus=0.0
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            _, reward, _, _, info = self._paddle_hit(env)
            assert reward == pytest.approx(0.5)
            assert info["rew_first_hit"] == pytest.approx(0.5)
            self._wall_hit(env)

            # Cycle 2's first hit: no second payment.
            self._floor_bounce(env)
            _, _, _, _, info = self._paddle_hit(env)
            assert info["rew_first_hit"] == pytest.approx(0.0)

            env.reset(seed=1)
            self._floor_bounce(env)
            _, _, _, _, info = self._paddle_hit(env)
            assert info["rew_first_hit"] == pytest.approx(0.5)
        finally:
            env.close()

    def test_first_hit_bonus_survives_cycle_failure(self):
        """Unlike the refundable paddle advance, the first-hit bonus is
        never clawed back -- that hard separation between touching and
        never touching is its whole purpose (run 20260717_040824's SAC
        saw identical -1 returns for both and never learned contact)."""
        env = self._strict_env(
            "one_bounce",
            first_hit_bonus=0.5,
            paddle_hit_bonus=0.25,
            style_violation_penalty=1.0,
        )
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)
            # Weak return: terminal here (weak_return_penalty unset). The
            # pending paddle advance claws back; the outright bonus stays.
            _, reward, terminated, _, info = self._floor_bounce(env)
            assert terminated
            assert info["rew_paddle"] == pytest.approx(-0.25)
            assert info["rew_first_hit"] == pytest.approx(0.0)
            assert reward == pytest.approx(-1.0 - 0.25)
        finally:
            env.close()

    def test_weak_return_and_first_hit_kwargs_are_validated(self):
        for bad in (-0.1, np.nan, np.inf):
            with pytest.raises(ValueError, match="weak_return_penalty"):
                WallBallEnv(rally_style="one_bounce", weak_return_penalty=bad)
            with pytest.raises(ValueError, match="first_hit_bonus"):
                WallBallEnv(rally_style="one_bounce", first_hit_bonus=bad)

    def test_floor_after_legal_paddle_before_wall_is_terminal_violation(self):
        env = self._strict_env("one_bounce")
        try:
            env.reset(seed=0)
            self._floor_bounce(env)
            self._paddle_hit(env)
            _, reward, terminated, truncated, info = self._floor_bounce(env)
            assert terminated and not truncated
            assert reward == pytest.approx(-2.5)
            assert info["event_floor_bounce"] is True
            assert info["event_style_violation"] is True
            assert info["style_violation_reason"] == "floor_before_wall"
            assert info["rew_style_violation"] == pytest.approx(-2.5)
            assert info["term_style_violation"] is True
            assert info["term_double_bounce"] is False
            assert info["one_bounce_return_count"] == 0
        finally:
            env.close()

    def test_legal_volley_cycle_never_counts_one_bounce_recovery(self):
        env = self._strict_env("volley")
        try:
            env.reset(seed=0)
            _, _, terminated, truncated, info = self._paddle_hit(env)
            assert not (terminated or truncated)
            assert info["event_legal_paddle_hit"] is True
            assert info["rally_phase_name"] == "await_wall"
            assert info["legal_paddle_hit_count"] == 1
            assert info["one_bounce_recovery_count"] == 0

            _, reward, terminated, truncated, info = self._wall_hit(env)
            assert not (terminated or truncated)
            assert reward == pytest.approx(1.0)
            assert info["event_completed_return"] is True
            assert info["rally_phase_name"] == "await_paddle"
            assert info["one_bounce_return_count"] == 0
            assert info["return_count"] == 1
            assert info["one_bounce_recovery_count"] == 0
        finally:
            env.close()

    def test_same_substep_events_are_ordered_floor_paddle_wall(self):
        env = WallBallEnv(rally_style="one_bounce")
        try:
            env.reset(seed=0)
            # Intentionally scrambled input proves sorting uses substep
            # then explicit contact precedence, not insertion order.
            env._substep_contact_events = [
                (2, 2, "wall"),
                (2, 0, "floor"),
                (2, 1, "paddle"),
            ]
            assert env._ordered_frame_events(
                paddle_edge=True, wall_edge=True
            ) == ["floor", "paddle", "wall"]
        finally:
            env.close()


class TestWallBallCourtMarkers:
    """Render-only court markings track the resolved preset kwargs."""

    def _site_x(self, env, name):
        return float(env.model.site_pos[int(env.model.site(name).id)][0])

    def _site_alpha(self, env, name):
        return float(env.model.site_rgba[int(env.model.site(name).id)][3])

    def test_markers_follow_preset_lane_home_and_serve(self):
        env = WallBallEnv(
            paddle_x_target_range=(-3.2, -1.6),
            paddle_home_x=-2.7,
            serve_start_x=0.8,
        )
        try:
            assert self._site_x(env, "court_line_lane_min") == -3.2
            assert self._site_x(env, "court_line_lane_max") == -1.6
            assert self._site_x(env, "court_line_home") == -2.7
            assert self._site_x(env, "court_line_serve") == 0.8
            strip_id = int(env.model.site("court_lane_strip").id)
            assert env.model.site_pos[strip_id][0] == pytest.approx(-2.4)
            assert env.model.site_size[strip_id][0] == pytest.approx(0.8)
            # Static geography stays authored in the XML.
            assert self._site_x(env, "court_line_wall_base") == 3.9
            assert self._site_x(env, "court_line_baseline") == -8.2
        finally:
            env.close()

    def test_default_lane_markers_span_the_physical_workspace(self):
        env = WallBallEnv()
        try:
            assert self._site_x(env, "court_line_lane_min") == pytest.approx(
                env.paddle_x_target_range[0]
            )
            assert self._site_x(env, "court_line_lane_max") == pytest.approx(
                env.paddle_x_target_range[1]
            )
        finally:
            env.close()

    def test_fence_markers_hidden_until_set_and_track_runtime_changes(self):
        env = WallBallEnv(paddle_x_target_range=(-3.2, -1.6))
        try:
            assert self._site_alpha(env, "court_line_fence_min") == 0.0
            assert self._site_alpha(env, "court_line_fence_max") == 0.0
            # A curriculum sets the fence between episodes via
            # set_wrapper_attr; markers refresh on the next reset.
            env.paddle_x_fence = (-3.0, -2.0)
            env.serve_start_x = 1.4
            env.reset(seed=0)
            assert self._site_x(env, "court_line_fence_min") == -3.0
            assert self._site_x(env, "court_line_fence_max") == -2.0
            assert self._site_alpha(env, "court_line_fence_min") > 0.0
            assert self._site_x(env, "court_line_serve") == 1.4
            # Clearing the fence hides the lines again.
            env.paddle_x_fence = None
            env.reset(seed=1)
            assert self._site_alpha(env, "court_line_fence_min") == 0.0
        finally:
            env.close()

    def test_markers_are_sites_only_and_leave_dynamics_unchanged(self):
        # Sites never generate contacts in MuJoCo, so the markings are
        # provably render-only as long as no marker is a geom.
        env = WallBallEnv()
        try:
            geom_names = {
                env.model.geom(i).name for i in range(env.model.ngeom)
            }
            assert not any(name.startswith("court_") for name in geom_names)
            # And the serve trajectory is bit-identical across resets
            # regardless of marker repositioning between them.
            env.reset(seed=7)
            first = [env.step(np.zeros(3))[0].copy() for _ in range(5)]
            env.paddle_x_fence = (-3.0, -2.0)
            env.paddle_x_fence = None
            env.reset(seed=7)
            second = [env.step(np.zeros(3))[0].copy() for _ in range(5)]
            for a, b in zip(first, second, strict=True):
                np.testing.assert_array_equal(a, b)
        finally:
            env.close()


class TestWallBallInPlayBound:
    """ball_in_play_min_x widens the deep OOB edge per-task (0.25.0)."""

    def test_default_keeps_the_historical_bound(self):
        env = WallBallEnv()
        try:
            assert env.ball_in_play_min_x == -6.0
        finally:
            env.close()

    @pytest.mark.parametrize("value", [0.0, 0.5, np.nan, np.inf, -np.inf])
    def test_bound_must_be_finite_and_negative(self, value):
        with pytest.raises(ValueError, match="ball_in_play_min_x"):
            WallBallEnv(ball_in_play_min_x=value)

    def _drive_ball_deep(self, env):
        """Send the ball flat and fast past the paddle toward deep x;
        returns (ball_x_at_termination, terminated_within_budget)."""
        env.reset(seed=0)
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        adr = int(env.model.joint("ball_x").qposadr[0])
        dof = int(env.model.joint("ball_x").dofadr[0])
        # Fast and flat: reaches x=-10 from -5.0 in ~0.2 s, well before
        # the drop from z=0.5 can produce a second floor bounce.
        qpos[adr : adr + 3] = [-5.0, 2.0, 0.5]
        qvel[dof : dof + 3] = [-25.0, 0.0, 0.0]
        env.set_state(qpos, qvel)
        for _ in range(40):
            obs, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                return float(obs[0]), bool(info["term_oob"])
        raise AssertionError("ball never left play")

    def test_widened_bound_extends_play_and_still_terminates(self):
        default_env = WallBallEnv(out_of_bounds_penalty=0.0)
        wide_env = WallBallEnv(
            out_of_bounds_penalty=0.0, ball_in_play_min_x=-9.5
        )
        try:
            x_default, oob_default = self._drive_ball_deep(default_env)
            x_wide, oob_wide = self._drive_ball_deep(wide_env)
            assert oob_default and x_default <= -6.0
            # The widened env keeps the same ball alive well past the
            # historical edge, then terminates at its own bound.
            assert oob_wide and x_wide <= -9.5
        finally:
            default_env.close()
            wide_env.close()


class TestWallBallCourtStyle:
    """court_style toggles render-only marking sets; tennis is to-size."""

    def _alpha(self, env, name):
        return float(env.model.site_rgba[int(env.model.site(name).id)][3])

    def _pos_size(self, env, name):
        sid = int(env.model.site(name).id)
        return env.model.site_pos[sid].copy(), env.model.site_size[sid].copy()

    def test_style_visibility_matrix(self):
        """Every court_* site in the COMPILED MODEL obeys its style.

        Iterating the model's own site names (not a Python tuple) is
        the point: the 0.25.0 workspace extension added
        court_tick_xm5..xm8 to the XML but not to the hand-maintained
        visibility tuple, and the old version of this test iterated the
        same incomplete tuple -- so the ticks stayed visible on top of
        the tennis surface while the suite passed. The lists are now
        derived from the model in __init__; this test re-derives the
        expectation independently so a partition bug cannot hide.
        """
        for style, diag_visible, tennis_visible in (
            ("diagnostic", True, False),
            ("tennis", False, True),
            ("none", False, False),
        ):
            env = WallBallEnv(court_style=style)
            try:
                model = env.unwrapped.model
                court_sites = [
                    name
                    for name in (
                        model.site(i).name for i in range(model.nsite)
                    )
                    if name.startswith("court_")
                ]
                assert len(court_sites) >= 25, court_sites
                for name in court_sites:
                    if name.startswith("court_line_fence"):
                        continue  # fence lines also need a fence set
                    expected = (
                        tennis_visible
                        if name.startswith("court_tennis_")
                        else diag_visible
                    )
                    assert (self._alpha(env, name) > 0) == expected, (
                        f"{name} in {style}"
                    )
                # The one the stale tuple missed for a whole release:
                assert (
                    self._alpha(env, "court_tick_xm8") > 0
                ) == diag_visible
                # Sensor-debug tints are hidden only in presentation
                # (tennis) footage.
                for name in WallBallEnv._SENSOR_TINT_SITES:
                    assert (self._alpha(env, name) > 0) == (
                        style != "tennis"
                    ), f"{name} in {style}"
            finally:
                env.close()

    def test_fence_lines_stay_hidden_in_tennis_even_with_a_fence(self):
        env = WallBallEnv(
            court_style="tennis", paddle_x_fence=(-3.0, -2.0)
        )
        try:
            assert self._alpha(env, "court_line_fence_min") == 0.0
            env.court_style = "diagnostic"
            env.reset(seed=0)
            assert self._alpha(env, "court_line_fence_min") > 0
        finally:
            env.close()

    def test_none_alias_and_ezpickle_round_trip(self):
        import pickle

        # The TOML sentinel hands the env None for court_style = "none".
        env = WallBallEnv(court_style=None)
        try:
            assert env.court_style == "none"
        finally:
            env.close()
        env = WallBallEnv(court_style="tennis")
        try:
            clone = pickle.loads(pickle.dumps(env))
            try:
                # EzPickle must capture the style: a SubprocVecEnv or
                # deepcopy clone may not silently revert to diagnostic.
                assert clone.court_style == "tennis"
            finally:
                clone.close()
        finally:
            env.close()

    def test_invalid_style_rejected(self):
        with pytest.raises(ValueError, match="court_style"):
            WallBallEnv(court_style="grass")

    def test_runtime_style_switch_applies_at_reset(self):
        env = WallBallEnv()
        try:
            assert self._alpha(env, "court_tennis_surface") == 0.0
            # The curriculum path: set_wrapper_attr between episodes,
            # visible after the next reset.
            env.court_style = "tennis"
            env.reset(seed=0)
            assert self._alpha(env, "court_tennis_surface") > 0
            assert self._alpha(env, "court_line_wall_base") == 0.0
            env.court_style = "none"
            env.reset(seed=1)
            assert self._alpha(env, "court_tennis_surface") == 0.0
            assert self._alpha(env, "court_line_wall_base") == 0.0
        finally:
            env.close()

    def test_tennis_court_is_to_itf_size(self):
        env = WallBallEnv(court_style="tennis")
        try:
            wall_x = 3.9  # wall face plane == the net
            pos, size = self._pos_size(env, "court_tennis_baseline")
            # ITF measures the 11.885 m half-length to the OUTER edge of
            # the baseline, which must not overhang the floor (x=-8).
            assert wall_x - (pos[0] - size[0]) == pytest.approx(11.885)
            assert pos[0] - size[0] >= -8.0
            assert size[1] == pytest.approx(5.485)  # doubles half-width
            mark_pos, mark_size = self._pos_size(env, "court_tennis_center_mark")
            # The 10 cm center-mark stub extends into the court from the
            # baseline's inner edge.
            assert mark_pos[0] - mark_size[0] == pytest.approx(
                pos[0] + size[0]
            )
            pos, size = self._pos_size(env, "court_tennis_service_line")
            assert wall_x - pos[0] == pytest.approx(6.40)
            assert size[1] == pytest.approx(4.115)  # singles half-width
            for name, y in (
                ("court_tennis_singles_right", 4.115),
                ("court_tennis_doubles_right", 5.485),
            ):
                pos, size = self._pos_size(env, name)
                assert pos[1] == pytest.approx(y)
                # Sidelines span net to baseline.
                assert pos[0] + size[0] == pytest.approx(wall_x)
                assert pos[0] - size[0] == pytest.approx(wall_x - 11.885)
            pos, size = self._pos_size(env, "court_tennis_center_service")
            assert pos[0] + size[0] == pytest.approx(wall_x)
            assert pos[0] - size[0] == pytest.approx(wall_x - 6.40)
        finally:
            env.close()

    def test_court_style_is_render_only(self):
        obs_by_style = {}
        for style in ("diagnostic", "tennis", "none"):
            env = WallBallEnv(court_style=style)
            try:
                env.reset(seed=11)
                for _ in range(5):
                    obs, *_ = env.step(np.zeros(3))
                obs_by_style[style] = obs.copy()
            finally:
                env.close()
        np.testing.assert_array_equal(
            obs_by_style["diagnostic"], obs_by_style["tennis"]
        )
        np.testing.assert_array_equal(
            obs_by_style["diagnostic"], obs_by_style["none"]
        )


class TestWallBallEscalatingWallReward:
    """wall_reward_increment: the n-th return banks 1 + (n-1)*increment."""

    def _run_oracle_episode(self, increment, seed):
        from courtside_dynamics.recipes import RECIPES
        from courtside_dynamics.scripted_policies import (
            wall_ball_baseline_oracle_action,
        )

        kwargs = dict(RECIPES["WallBallBaseline"].env_kwargs)
        kwargs.pop("render_mode", None)
        kwargs["recovery_reset_probability"] = 0.0
        kwargs["wall_reward_increment"] = increment
        env = WallBallEnv(**kwargs)
        total, wall_total, returns = 0.0, 0.0, 0
        try:
            obs, _ = env.reset(seed=seed)
            for _ in range(env.episode_len):
                obs, reward, term, trunc, info = env.step(
                    wall_ball_baseline_oracle_action(obs)
                )
                total += reward
                wall_total += info["rew_wall"]
                if term or trunc:
                    returns = int(info["bounce_count"])
                    break
        finally:
            env.close()
        return total, wall_total, returns

    def test_increment_zero_is_todays_reward_bit_for_bit(self):
        total_a, wall_a, n_a = self._run_oracle_episode(0.0, seed=3)
        # Explicit 0.0 and the default must agree exactly.
        from courtside_dynamics.recipes import RECIPES

        kwargs = dict(RECIPES["WallBallBaseline"].env_kwargs)
        kwargs.pop("render_mode", None)
        assert "wall_reward_increment" not in kwargs  # recipe ships dark
        assert wall_a == n_a  # flat +1 per completed return

    def test_escalation_pays_the_triangular_bonus_exactly(self):
        # Scripted actions are reward-independent, so the same seed
        # produces the same trajectory at both increments; the reward
        # difference is exactly 0.5 * (0 + 1 + ... + (n-1)).
        for seed in (3, 7):
            total_flat, wall_flat, n_flat = self._run_oracle_episode(0.0, seed)
            total_esc, wall_esc, n_esc = self._run_oracle_episode(0.5, seed)
            assert n_esc == n_flat and n_flat >= 2  # needs a real rally
            expected_bonus = 0.5 * (n_flat * (n_flat - 1) / 2)
            assert total_esc - total_flat == pytest.approx(expected_bonus)
            assert wall_esc - wall_flat == pytest.approx(expected_bonus)

    def test_invalid_increment_rejected(self):
        with pytest.raises(ValueError, match="wall_reward_increment"):
            WallBallEnv(wall_reward_increment=-0.1)
        with pytest.raises(ValueError, match="wall_reward_increment"):
            WallBallEnv(wall_reward_increment=float("nan"))

    def test_increment_survives_pickle(self):
        import pickle

        env = WallBallEnv(wall_reward_increment=0.5)
        try:
            clone = pickle.loads(pickle.dumps(env))
            try:
                assert clone.wall_reward_increment == 0.5
            finally:
                clone.close()
        finally:
            env.close()


class TestPaddleHomeIsALiveMappingPivot:
    """``paddle_home_x`` is the pivot of the normalized x action map.

    It was a plain attribute until 0.22.0, which made a curriculum stage
    that moved it a silent no-op: ``_control_home`` is derived once
    during setup, and ``set_wrapper_attr(..., force=False)`` reports
    success for any attribute that already exists. Run 20260727_004014
    paid for it -- the gate advanced to a (-4.7, -3.0) fence while the
    pivot stayed at -1.7, so 71.7% of the x action range (all of action
    0 and the whole positive half) clamped onto the fence's front edge.
    """

    @staticmethod
    def _zero_action_x(env):
        return float(
            env._action_to_controls(np.zeros(3, dtype=np.float32))[0]
            + env._paddle_x_origin
        )

    def _deep_env(self, **kwargs):
        return WallBallEnv(
            paddle_home_x=-1.7,
            paddle_x_target_range=(-4.7, 0.3),
            paddle_x_fence=(-4.7, -2.6),
            paddle_start_x=-3.9,
            **kwargs,
        )

    def test_assignment_moves_the_mapping_not_just_the_attribute(self):
        env = self._deep_env()
        try:
            # Pivot outside the fence: action 0 lands on the clamp.
            assert self._zero_action_x(env) == pytest.approx(-2.6)
            env.paddle_home_x = -3.65
            assert self._zero_action_x(env) == pytest.approx(-3.65)
        finally:
            env.close()

    def test_gate_path_moves_the_mapping(self):
        """``set_wrapper_attr`` is how PerformanceGatedEnvStagesCallback
        applies a stage; it must reach the same setter."""
        env = self._deep_env()
        try:
            env.set_wrapper_attr("paddle_home_x", -3.65, force=False)
            assert env.paddle_home_x == -3.65
            assert self._zero_action_x(env) == pytest.approx(-3.65)
        finally:
            env.close()

    @staticmethod
    def _usable_action_share(home, fence):
        """Fraction of the normalized x action range landing inside the
        fence rather than clamped onto one of its edges."""
        env = WallBallEnv(
            paddle_home_x=home,
            paddle_x_target_range=(-4.7, 0.3),
            paddle_x_fence=fence,
            paddle_start_x=-3.9,
        )
        try:
            xs = np.array([
                float(
                    env._action_to_controls(
                        np.array([a, 0.0, 0.0], dtype=np.float32)
                    )[0]
                    + env._paddle_x_origin
                )
                for a in np.linspace(-1.0, 1.0, 4001)
            ])
            low, high = fence
            return float(((xs > low + 1e-9) & (xs < high - 1e-9)).mean())
        finally:
            env.close()

    def test_recentring_restores_usable_action_range(self):
        # What run 20260727_004014 actually shipped: pivot pinned at
        # -1.7 against the old, narrow goal fence.
        assert self._usable_action_share(
            -1.7, (-4.7, -3.0)
        ) == pytest.approx(0.283, abs=0.01)
        # Widening the fence alone recovers only part of it, because the
        # pivot is still outside the fence.
        assert self._usable_action_share(
            -1.7, (-4.7, -2.6)
        ) == pytest.approx(0.350, abs=0.01)
        # Re-centring the pivot on the wider fence is what restores it.
        assert self._usable_action_share(
            -3.65, (-4.7, -2.6)
        ) == pytest.approx(0.633, abs=0.01)

    def test_rejects_nonfinite_and_out_of_range(self):
        env = self._deep_env()
        try:
            with pytest.raises(ValueError, match="paddle_home_x"):
                env.paddle_home_x = float("nan")
            with pytest.raises(ValueError, match="paddle_x_target_range"):
                env.paddle_home_x = -9.0
            # A rejected assignment must not have moved the mapping.
            assert self._zero_action_x(env) == pytest.approx(-2.6)
        finally:
            env.close()

    def test_constructor_value_survives_pickle(self):
        """Only the *constructed* pivot round-trips.

        Pickling rebuilds the env from its recorded constructor kwargs,
        so no runtime mutation survives -- true of ``paddle_x_fence`` and
        ``serve_speed`` long before this property existed, and pinned
        here so nobody reads the setter as making stage state durable.
        The gate never relies on it: SubprocVecEnv pickles the env
        *factory*, and stages are applied inside each worker via
        ``env_method("set_wrapper_attr", ...)``.
        """
        import pickle

        env = WallBallEnv(
            paddle_home_x=-3.65,
            paddle_x_target_range=(-4.7, 0.3),
            paddle_x_fence=(-4.7, -2.6),
            paddle_start_x=-3.9,
        )
        try:
            env.paddle_home_x = -3.0
            clone = pickle.loads(pickle.dumps(env))
            try:
                assert clone.paddle_home_x == -3.65
                assert self._zero_action_x(clone) == pytest.approx(-3.65)
            finally:
                clone.close()
        finally:
            env.close()


class TestReturnShapingScale:
    """Dense credit on the outgoing leg (paddle hit -> wall).

    The incoming leg has always been shaped; the outgoing one had none,
    so "how hard did I hit it" rode on a single sparse +1 about 50 steps
    later. Run 20260727_004014's long-horizon audit measured the result:
    6 of 63 legal hits never reached the wall, and a 1.5 m/s pop-up paid
    exactly what a 17 m/s drive paid.
    """

    @staticmethod
    def _outgoing(scale, ball_vx, steps=12):
        env = WallBallEnv(
            return_shaping_scale=scale,
            paddle_x_fence=(-4.7, -2.6),
            paddle_start_x=-3.9,
        )
        try:
            env.reset(seed=0)
            env._returning = False
            env._paddle_hit_since_last_wall = True
            env._prev_ball_to_wall = None
            env.data.joint("ball_x").qpos[:3] = [-2.0, 0.0, 1.0]
            env.data.joint("ball_x").qvel[:3] = [ball_vx, 0.0, 0.5]
            shaping = 0.0
            for _ in range(steps):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(3, dtype=np.float32)
                )
                shaping += float(info.get("rew_shaping", 0.0))
                if terminated or truncated:
                    break
            return shaping, float(env._pending_shaping)
        finally:
            env.close()

    def test_default_is_off(self):
        assert WallBallEnv.__init__.__defaults__ is not None
        env = WallBallEnv()
        try:
            assert env.return_shaping_scale == 0.0
        finally:
            env.close()
        assert self._outgoing(0.0, 16.0)[0] == 0.0

    def test_pace_is_rewarded(self):
        slow, _ = self._outgoing(0.15, 1.5)
        fast, _ = self._outgoing(0.15, 16.0)
        assert 0.0 < slow < fast
        # The drive closes an order of magnitude more gap per step.
        assert fast > 5 * slow

    def test_advance_is_refundable(self):
        """It must join ``_pending_shaping`` so a shot that drifts
        wallward and then dies banks nothing."""
        shaping, pending = self._outgoing(0.15, 16.0)
        assert pending == pytest.approx(shaping)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="return_shaping_scale"):
            WallBallEnv(return_shaping_scale=-0.1)
        with pytest.raises(ValueError, match="return_shaping_scale"):
            WallBallEnv(return_shaping_scale=float("nan"))


class TestSharedBaseGuards:
    """The _base.py helpers every env (and the next one) shares."""

    def test_piecewise_mapping_and_inverse_round_trip(self):
        from courtside_dynamics.envs._base import (
            invert_piecewise_target,
            piecewise_targets,
        )

        low, home, high = -4.7, -1.7, 0.3  # the calibrated x asymmetry
        # Action zero is exactly home; the endpoints map to the limits
        # (to float associativity -- home + (high - home) regroups).
        assert piecewise_targets(0.0, low, home, high) == home
        assert float(
            piecewise_targets(-1.0, low, home, high)
        ) == pytest.approx(low, abs=1e-12)
        assert float(
            piecewise_targets(1.0, low, home, high)
        ) == pytest.approx(high, abs=1e-12)
        for normalized in (-1.0, -0.5, -0.1, 0.0, 0.25, 0.9, 1.0):
            target = float(
                piecewise_targets(normalized, low, home, high)
            )
            recovered = invert_piecewise_target(target, low, home, high)
            assert recovered == pytest.approx(normalized, abs=1e-12)
        # Vectorized form matches per-element scalars.
        lows = np.array([low, -3.0, 0.0])
        homes = np.array([home, 0.0, 2.0])
        highs = np.array([high, 3.0, 2.9])
        actions = np.array([-0.75, 0.5, -0.25])
        vec = piecewise_targets(actions, lows, homes, highs)
        for index in range(3):
            assert vec[index] == pytest.approx(
                float(
                    piecewise_targets(
                        actions[index],
                        lows[index],
                        homes[index],
                        highs[index],
                    )
                )
            )
        # A home pivoted onto a mapping endpoint zeroes one half-span;
        # the inverse stays finite (the certification-ladder case).
        assert np.isfinite(
            invert_piecewise_target(-5.0, -4.7, -4.7, 0.3)
        )

    def test_shared_validators(self):
        from courtside_dynamics.envs._base import (
            finite_nonnegative,
            finite_positive,
        )

        assert finite_nonnegative("knob", 0) == 0.0
        assert finite_positive("knob", 2) == 2.0
        for bad in (float("nan"), float("inf"), -1.0):
            with pytest.raises(ValueError, match="knob"):
                finite_nonnegative("knob", bad)
        for bad in (float("nan"), 0.0, -1.0):
            with pytest.raises(ValueError, match="knob"):
                finite_positive("knob", bad)

    def test_ball_balance_rejects_nonfinite_actions_without_physics(self):
        """MuJoCo reacts to NaN ctrl by warning and resetting its state
        mid-episode; BallBalance -- previously the one env without the
        guard its siblings carry -- must end the episode on the last
        finite observation instead of stepping physics with the NaN."""
        env = BallBalanceEnv(episode_len=50)
        try:
            env.reset(seed=0)
            healthy_obs, *_ = env.step(np.zeros(6))
            qpos_before = env.unwrapped.data.qpos.copy()
            bad = np.full(6, np.nan)
            obs, reward, terminated, truncated, _ = env.step(bad)
            assert terminated
            assert reward == 0.0
            assert bool(np.isfinite(obs).all())
            assert np.array_equal(obs, healthy_obs)
            # Physics was never stepped with the NaN control.
            assert np.array_equal(env.unwrapped.data.qpos, qpos_before)
        finally:
            env.close()

    def test_ball_balance_echoes_last_finite_observation(self):
        """A solver blow-up must terminate on the echoed observation,
        not deliver NaNs to VecNormalize (the guard wall_ball and
        ball_bounce already had)."""
        env = BallBalanceEnv(episode_len=50)
        try:
            env.reset(seed=0)
            healthy_obs, *_ = env.step(np.zeros(6))
            env.unwrapped.data.qpos[:] = np.nan
            obs, _, terminated, _, _ = env.step(np.zeros(6))
            assert terminated
            assert bool(np.isfinite(obs).all())
            assert np.array_equal(obs, healthy_obs)
        finally:
            env.close()
