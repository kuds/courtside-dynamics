"""Phase 3 API and end-to-end tests for cooperative humanoid tennis."""

from __future__ import annotations

import importlib
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import courtside_dynamics  # noqa: F401  (triggers Gymnasium registration)
from courtside_dynamics.envs._tennis_events import (
    TENNIS_CONTACT_CHANNELS,
)
from courtside_dynamics.envs.humanoid_tennis import (
    HUMANOID_TENNIS_ACTION_NAMES,
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    HUMANOID_TENNIS_OBSERVATION_NAMES,
    HumanoidTennisCoopEnv,
    TennisRewardConfig,
    TennisServeConfig,
)
from courtside_dynamics.envs.robot_models import (
    UNITREE_G1_ACTION_LAYOUT,
    UNITREE_G1_STAND_JOINT_POSITIONS,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
)
from courtside_dynamics.scripted_policies import run_humanoid_tennis_oracle
from tests._helpers import event_batch as _event_batch
from tests._helpers import inject_batches as _inject_batches


def test_centralized_action_contract_is_exact_and_player_partitioned():
    env = HumanoidTennisCoopEnv()
    try:
        assert env.action_space.shape == (58,)
        assert env.action_space.dtype == np.dtype(np.float32)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        assert env.action_layout == UNITREE_G1_ACTION_LAYOUT
        assert env.action_layout.player_a == slice(0, 29)
        assert env.action_layout.player_b == slice(29, 58)
        assert env.action_layout.player_a_right_arm == slice(22, 29)
        assert env.action_layout.player_b_right_arm == slice(51, 58)
        assert len(HUMANOID_TENNIS_ACTION_NAMES) == 58
        assert len(set(HUMANOID_TENNIS_ACTION_NAMES)) == 58
        assert all(
            name.startswith("player_a_")
            for name in HUMANOID_TENNIS_ACTION_NAMES[:29]
        )
        assert all(
            name.startswith("player_b_")
            for name in HUMANOID_TENNIS_ACTION_NAMES[29:]
        )
    finally:
        env.close()


def test_centralized_observation_slices_and_labels_are_locked_together():
    env = HumanoidTennisCoopEnv()
    try:
        observation, _ = env.reset(seed=0)
        layout = HUMANOID_TENNIS_OBSERVATION_LAYOUT
        slices = (
            layout.player_a_proprio,
            layout.player_b_proprio,
            layout.player_a_racket,
            layout.player_b_racket,
            layout.ball,
            layout.relative,
            layout.rally_state,
            layout.contact_latches,
            layout.contact_release_progress,
            layout.active_action_mask,
        )
        boundaries = [(value.start, value.stop) for value in slices]

        assert boundaries == [
            (0, 71),
            (71, 142),
            (142, 157),
            (157, 172),
            (172, 181),
            (181, 193),
            (193, 221),
            (221, 231),
            (231, 241),
            (241, 299),
        ]
        assert layout.total_size == 299
        assert observation.shape == env.observation_space.shape == (299,)
        assert observation.dtype == env.observation_space.dtype == np.float64
        assert len(HUMANOID_TENNIS_OBSERVATION_NAMES) == 299
        assert len(set(HUMANOID_TENNIS_OBSERVATION_NAMES)) == 299
        assert len(observation[layout.contact_latches]) == len(
            TENNIS_CONTACT_CHANNELS
        )
        assert len(observation[layout.contact_release_progress]) == len(
            TENNIS_CONTACT_CHANNELS
        )
        assert np.all(observation[layout.contact_latches] == 0.0)
        assert np.all(observation[layout.contact_release_progress] == 0.0)
        assert np.all(observation[layout.active_action_mask] == 1.0)
        assert np.isfinite(observation).all()
    finally:
        env.close()


def test_normalized_actions_map_piecewise_to_stand_and_control_limits():
    env = HumanoidTennisCoopEnv()
    try:
        expected_stand = np.asarray(
            UNITREE_G1_STAND_JOINT_POSITIONS * 2,
            dtype=np.float64,
        )
        assert env.neutral_action.shape == (58,)
        assert env.neutral_action.dtype == np.float32
        assert env._action_to_controls(env.neutral_action) == pytest.approx(
            expected_stand
        )
        assert env._action_to_controls(np.ones(58)) == pytest.approx(
            env.model.actuator_ctrlrange[:, 1]
        )
        assert env._action_to_controls(-np.ones(58)) == pytest.approx(
            env.model.actuator_ctrlrange[:, 0]
        )

        probe = np.full(58, 0.5)
        expected_positive = expected_stand + 0.5 * (
            env.model.actuator_ctrlrange[:, 1] - expected_stand
        )
        assert env._action_to_controls(probe) == pytest.approx(expected_positive)
        probe *= -1.0
        expected_negative = expected_stand - 0.5 * (
            expected_stand - env.model.actuator_ctrlrange[:, 0]
        )
        assert env._action_to_controls(probe) == pytest.approx(expected_negative)
    finally:
        env.close()


def test_named_control_target_helper_round_trips_and_rejects_bad_targets():
    env = HumanoidTennisCoopEnv()
    try:
        name = "player_b_right_shoulder_pitch_joint"
        index = HUMANOID_TENNIS_ACTION_NAMES.index(name)
        target = -0.2
        action = env.action_for_control_targets({name: target})
        controls = env._action_to_controls(action)
        assert controls[index] == pytest.approx(target)
        assert np.delete(controls, index) == pytest.approx(
            np.delete(env._stand_controls, index)
        )

        with pytest.raises(ValueError, match="unknown actuator"):
            env.action_for_control_targets({"player_c_missing_joint": 0.0})
        with pytest.raises(ValueError, match="finite"):
            env.action_for_control_targets({name: float("nan")})
        with pytest.raises(ValueError, match="outside ctrlrange"):
            env.action_for_control_targets({name: 100.0})
    finally:
        env.close()


def test_seeded_resets_are_reproducible_and_alternate_serving_side():
    env = HumanoidTennisCoopEnv(initial_serve_side="a")
    try:
        observation_a1, info_a1 = env.reset(seed=37)
        observation_b, info_b = env.reset()
        observation_a2, info_a2 = env.reset()
        observation_reseeded, info_reseeded = env.reset(seed=37)

        assert [
            info_a1["serve_side_name"],
            info_b["serve_side_name"],
            info_a2["serve_side_name"],
            info_reseeded["serve_side_name"],
        ] == ["a", "b", "a", "a"]
        assert observation_reseeded == pytest.approx(observation_a1)
        assert info_reseeded["initial_ball_qpos"] == pytest.approx(
            info_a1["initial_ball_qpos"]
        )
        assert info_reseeded["initial_ball_qvel"] == pytest.approx(
            info_a1["initial_ball_qvel"]
        )
        assert not np.array_equal(observation_b, observation_a2)
    finally:
        env.close()


def test_reset_info_logs_full_seeded_ball_state_and_mirrors_by_side():
    env = HumanoidTennisCoopEnv()
    try:
        _, info_a = env.reset(seed=11, options={"serve_side": "a"})
        _, info_b = env.reset(seed=11, options={"serve_side": "b"})

        qpos_a = np.asarray(info_a["initial_ball_qpos"])
        qpos_b = np.asarray(info_b["initial_ball_qpos"])
        qvel_a = np.asarray(info_a["initial_ball_qvel"])
        qvel_b = np.asarray(info_b["initial_ball_qvel"])
        assert qpos_a.shape == qpos_b.shape == (7,)
        assert qvel_a.shape == qvel_b.shape == (6,)
        assert qpos_b[:2] == pytest.approx(-qpos_a[:2])
        assert qpos_b[2:] == pytest.approx(qpos_a[2:])
        assert qvel_b[:2] == pytest.approx(-qvel_a[:2])
        assert qvel_b[2:] == pytest.approx(qvel_a[2:])

        for label, expected in zip(
            (
                "initial_ball_x",
                "initial_ball_y",
                "initial_ball_z",
                "initial_ball_quat_w",
                "initial_ball_quat_x",
                "initial_ball_quat_y",
                "initial_ball_quat_z",
            ),
            qpos_b,
            strict=True,
        ):
            assert info_b[label] == pytest.approx(expected)
        for label, expected in zip(
            (
                "initial_ball_vx",
                "initial_ball_vy",
                "initial_ball_vz",
                "initial_ball_wx",
                "initial_ball_wy",
                "initial_ball_wz",
            ),
            qvel_b,
            strict=True,
        ):
            assert info_b[label] == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": 1}, "unsupported reset options"),
        ({"serve_side": "center"}, "serve side"),
        ({"ball_position": (1.0, 2.0)}, "finite three-vector"),
        ({"ball_velocity": (1.0, float("nan"), 2.0)}, "finite three-vector"),
        ({"joint_positions": []}, "must be a mapping"),
        ({"joint_positions": {"missing_joint": 0.0}}, "unknown reset joint"),
        (
            {"joint_positions": {"player_a_floating_base_joint": 0.0}},
            "only one-DoF hinge",
        ),
        (
            {"joint_positions": {"player_a_left_knee_joint": 100.0}},
            "outside range",
        ),
    ],
)
def test_reset_options_reject_invalid_shapes_names_and_ranges(options, message):
    env = HumanoidTennisCoopEnv()
    try:
        with pytest.raises(ValueError, match=message):
            env.reset(seed=0, options=options)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "serve_side": "a",
                "ball_position": (1.0, 0.0, 1.3),
                "ball_velocity": (10.0, 0.0, 1.0),
            },
            "serving side",
        ),
        (
            {
                "serve_side": "a",
                "ball_position": (-10.0, 0.0, 1.3),
                "ball_velocity": (-10.0, 0.0, 1.0),
            },
            "other side",
        ),
    ],
)
def test_reset_options_require_a_cross_court_initial_feed(options, message):
    env = HumanoidTennisCoopEnv()
    try:
        with pytest.raises(ValueError, match=message):
            env.reset(seed=0, options=options)
        observation, info = env.reset(seed=0)
        assert np.isfinite(observation).all()
        assert info["serve_side_name"] == "a"
    finally:
        env.close()


def test_episode_truncates_on_exact_internal_length_without_terminal_fault():
    env = HumanoidTennisCoopEnv(episode_len=7)
    try:
        env.reset(seed=0)
        for step in range(1, 8):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            assert np.isfinite(reward)
            assert terminated is False
            assert truncated is (step == 7)
            assert info["episode_step"] == step
        assert info["term_timeout"] is True
        assert info["termination_reason_name"] == "none"
    finally:
        env.close()


def test_step_reward_components_sum_exactly_and_repeat_initial_ball_scalars():
    env = HumanoidTennisCoopEnv(episode_len=20)
    try:
        _, reset_info = env.reset(seed=19)
        _, reward, terminated, truncated, step_info = env.step(env.neutral_action)

        reward_keys = (
            "rew_valid_return",
            "rew_stage_success",
            "rew_shaping",
            "rew_shaping_clawback",
            "rew_fault",
            "rew_action_cost",
        )
        assert reward == pytest.approx(sum(step_info[key] for key in reward_keys))
        assert not terminated
        assert not truncated
        for key in (
            "initial_ball_x",
            "initial_ball_y",
            "initial_ball_z",
            "initial_ball_quat_w",
            "initial_ball_quat_x",
            "initial_ball_quat_y",
            "initial_ball_quat_z",
            "initial_ball_vx",
            "initial_ball_vy",
            "initial_ball_vz",
            "initial_ball_wx",
            "initial_ball_wy",
            "initial_ball_wz",
        ):
            assert step_info[key] == reset_info[key]
        assert step_info["serve_side"] == reset_info["serve_side"]
        assert step_info["serve_side_name"] == reset_info["serve_side_name"]
    finally:
        env.close()


def test_nonfinite_observation_returns_last_finite_state_and_terminal_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    env = HumanoidTennisCoopEnv(episode_len=20)
    try:
        initial_observation, _ = env.reset(seed=0)

        def inject_nonfinite_state(_ctrl: np.ndarray, _n_frames: int) -> None:
            env._latest_event_batch = _event_batch(0)
            env.data.qpos[0] = np.nan

        monkeypatch.setattr(
            env,
            "_step_mujoco_simulation",
            inject_nonfinite_state,
        )
        observation, reward, terminated, truncated, info = env.step(
            env.neutral_action
        )

        assert np.isfinite(observation).all()
        assert observation == pytest.approx(initial_observation)
        assert reward == pytest.approx(-2.0)
        assert terminated is True
        assert truncated is False
        assert info["observation_sanitized"] is True
        assert info["termination_reason_name"] == "nonfinite_state"
        assert info["term_nonfinite_state"] is True
        assert info["term_unsafe"] is True
        assert info["observed_event_nonfinite_state"] is True
        assert info["step_event_count"] == 1
        assert info["step_event_kinds"] == ("nonfinite_state",)
    finally:
        env.close()


def test_duplicate_racket_event_cannot_farm_shaping_and_fault_claws_it_back(
    monkeypatch: pytest.MonkeyPatch,
):
    env = HumanoidTennisCoopEnv(
        episode_len=100,
        reward_config=TennisRewardConfig(
            valid_return_reward=2.0,
            valid_hit_shaping=0.25,
            fault_penalty=1.0,
        ),
    )
    try:
        env.reset(seed=0)
        crossing = RallyEvent(
            RallyEventKind.NET_CROSSING_TO_B,
            substep=0,
            event_id=0,
            from_side=CourtSide.A,
            to_side=CourtSide.B,
        )
        hit = RallyEvent(
            RallyEventKind.BALL_RACKET_B,
            substep=1,
            event_id=1,
            contact_episode=0,
            peak_force=5.0,
        )
        fault = RallyEvent(
            RallyEventKind.RACKET_B_NET,
            substep=2,
            event_id=2,
            contact_episode=0,
            peak_force=3.0,
        )
        batches = iter(
            (
                _event_batch(0, crossing),
                _event_batch(1, hit),
                _event_batch(2, hit),
                _event_batch(3, fault),
            )
        )
        _inject_batches(monkeypatch, env, batches)

        _, reward_crossing, _, _, _ = env.step(env.neutral_action)
        _, reward_hit, _, _, info_hit = env.step(env.neutral_action)
        _, reward_duplicate, _, _, info_duplicate = env.step(env.neutral_action)
        _, reward_fault, terminated, _, info_fault = env.step(env.neutral_action)

        assert reward_crossing == 0.0
        assert reward_hit == pytest.approx(0.25)
        assert info_hit["pending_shaping_balance"] == pytest.approx(0.25)
        assert reward_duplicate == 0.0
        assert info_duplicate["event_duplicate_suppressed"] is True
        assert info_duplicate["legal_hit_count_b"] == 1
        assert terminated is True
        assert reward_fault == pytest.approx(-1.25)
        assert info_fault["rew_fault"] == pytest.approx(-1.0)
        assert info_fault["rew_shaping_clawback"] == pytest.approx(-0.25)
        assert info_fault["pending_shaping_balance"] == 0.0
    finally:
        env.close()


def test_confirmed_return_commits_shaping_and_rewards_only_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    env = HumanoidTennisCoopEnv(
        episode_len=100,
        reward_config=TennisRewardConfig(
            valid_return_reward=2.0,
            valid_hit_shaping=0.25,
            fault_penalty=1.0,
        ),
    )
    try:
        env.reset(seed=0)
        batches = iter(
            (
                _event_batch(
                    0,
                    RallyEvent(
                        RallyEventKind.NET_CROSSING_TO_B,
                        substep=0,
                        event_id=0,
                        from_side=CourtSide.A,
                        to_side=CourtSide.B,
                    ),
                ),
                _event_batch(
                    1,
                    RallyEvent(
                        RallyEventKind.BALL_RACKET_B,
                        substep=1,
                        event_id=1,
                        contact_episode=0,
                        peak_force=5.0,
                    ),
                ),
                _event_batch(
                    2,
                    RallyEvent(
                        RallyEventKind.NET_CROSSING_TO_A,
                        substep=2,
                        event_id=2,
                        from_side=CourtSide.B,
                        to_side=CourtSide.A,
                    ),
                ),
                _event_batch(
                    3,
                    RallyEvent(
                        RallyEventKind.BALL_COURT_A,
                        substep=3,
                        event_id=3,
                        contact_episode=0,
                        position=(-4.0, 0.0, 0.0),
                        in_bounds=True,
                        peak_force=4.0,
                    ),
                ),
            )
        )
        _inject_batches(monkeypatch, env, batches)

        rewards = []
        infos = []
        for _ in range(4):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            assert not terminated
            assert not truncated
            rewards.append(reward)
            infos.append(info)

        assert rewards == pytest.approx([0.0, 0.25, 0.0, 2.0])
        assert sum(rewards) == pytest.approx(2.25)
        assert infos[-1]["event_valid_return_b"] is True
        assert infos[-1]["rally_count"] == 1
        assert infos[-1]["pending_shaping_balance"] == 0.0
        assert infos[-1]["rew_shaping_clawback"] == 0.0
    finally:
        env.close()


def test_neutral_policy_cannot_earn_survival_or_local_tap_reward():
    env = HumanoidTennisCoopEnv(episode_len=300)
    try:
        env.reset(seed=0)
        total_reward = 0.0
        for _ in range(300):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            total_reward += reward
            if terminated or truncated:
                break

        assert info["rally_count"] == 0
        assert info["legal_hit_count"] == 0
        assert total_reward <= 0.0
    finally:
        env.close()


def test_out_ball_clearing_visible_runoff_lands_on_physical_catch_plane():
    """A powerful out cannot fall forever and be mislabeled as a timeout."""
    env = HumanoidTennisCoopEnv(episode_len=1_000)
    try:
        env.reset(
            seed=0,
            options={
                "serve_side": "a",
                "ball_position": (-10.0, 3.0, 1.3),
                "ball_velocity": (80.0, 0.0, 20.0),
            },
        )
        for _ in range(1_000):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            if terminated or truncated:
                break

        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(-1.0)
        assert info["termination_reason_name"] == "out_of_bounds"
        assert info["event_ball_court_b"] is True
        assert info["step_contact_peak_ball_court_b"] > 0.0
        assert abs(float(env.data.qpos[env._ball_qposadr])) > 18.285
    finally:
        env.close()


def test_random_actions_produce_finite_observations_rewards_and_resets():
    env = HumanoidTennisCoopEnv(episode_len=60)
    try:
        env.action_space.seed(123)
        observation, _ = env.reset(seed=123)
        episodes = 0
        for _ in range(120):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert observation.shape == (299,)
            assert np.isfinite(observation).all()
            assert np.isfinite(reward)
            assert np.isfinite(info["valid_return_rate"])
            if terminated or truncated:
                episodes += 1
                observation, _ = env.reset()
                assert np.isfinite(observation).all()
        assert episodes >= 1
    finally:
        env.close()


def test_sampler_markov_state_exposes_contact_release_hysteresis():
    env = HumanoidTennisCoopEnv()
    try:
        env.reset(seed=0)
        sampler = env._event_sampler
        ball_qposadr = env._ball_qposadr
        ball_dofadr = env._ball_dofadr
        court_channel = TENNIS_CONTACT_CHANNELS.index("ball_court")
        release_offset = len(TENNIS_CONTACT_CHANNELS)

        env.data.qpos[ball_qposadr : ball_qposadr + 3] = (-2.0, 0.0, 0.03)
        env.data.qvel[ball_dofadr : ball_dofadr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        sampler.begin_control_step(0)
        sampler.sample_substep(env.data, control_substep=0)
        contact_batch = sampler.end_control_step()
        contact_state = sampler.markov_state()

        assert [event.kind for event in contact_batch.events] == [
            RallyEventKind.BALL_COURT_A
        ]
        assert contact_state[court_channel] == 1.0
        assert contact_state[release_offset + court_channel] == 0.0

        env.data.qpos[ball_qposadr + 2] = 1.0
        mujoco.mj_forward(env.model, env.data)
        sampler.begin_control_step(1)
        sampler.sample_substep(env.data, control_substep=0)
        first_release_batch = sampler.end_control_step()
        first_release_state = sampler.markov_state()

        assert "ball_court" in first_release_batch.active_contact_latches
        assert first_release_state[court_channel] == 1.0
        assert first_release_state[release_offset + court_channel] == pytest.approx(
            0.5
        )

        sampler.begin_control_step(2)
        sampler.sample_substep(env.data, control_substep=0)
        released_batch = sampler.end_control_step()
        released_state = sampler.markov_state()

        assert "ball_court" not in released_batch.active_contact_latches
        assert released_state[court_channel] == 0.0
        assert released_state[release_offset + court_channel] == 0.0
    finally:
        env.close()


def test_sampler_can_drain_one_explicit_safety_event_after_failed_substep():
    env = HumanoidTennisCoopEnv()
    try:
        env.reset(seed=0)
        sampler = env._event_sampler
        sampler.begin_control_step(0)
        sampler.record_safety_event(
            RallyEventKind.UNSAFE_PHYSICS,
            control_substep=0,
        )
        sampler.record_safety_event(
            RallyEventKind.NONFINITE_STATE,
            control_substep=1,
        )
        batch = sampler.end_control_step()

        assert batch.substeps_sampled == 2
        assert [event.kind for event in batch.events] == [
            RallyEventKind.UNSAFE_PHYSICS
        ]
        assert not batch.contact_peaks

        sampler.begin_control_step(1)
        with pytest.raises(ValueError, match="only safety kinds"):
            sampler.record_safety_event(
                RallyEventKind.BALL_NET,
                control_substep=0,
            )
        empty_batch = sampler.end_control_step()
        assert not empty_batch.events
    finally:
        env.close()


def test_environment_passes_stable_baselines3_checker():
    env_checker = importlib.import_module("stable_baselines3.common.env_checker")
    env = HumanoidTennisCoopEnv(episode_len=30)
    try:
        env_checker.check_env(env)
    finally:
        env.close()


def test_environment_passes_gymnasium_checker():
    env_checker = importlib.import_module("gymnasium.utils.env_checker")
    env = HumanoidTennisCoopEnv(episode_len=30)
    try:
        with pytest.warns(UserWarning, match="observation space"):
            env_checker.check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_stable_baselines3_dummy_vecenv_and_vecnormalize_smoke():
    vec_env_module = importlib.import_module("stable_baselines3.common.vec_env")
    dummy_vec_env = vec_env_module.DummyVecEnv(
        [lambda: HumanoidTennisCoopEnv(episode_len=20)]
    )
    normalized_env = vec_env_module.VecNormalize(
        dummy_vec_env,
        norm_obs=True,
        norm_reward=True,
    )
    try:
        observation = normalized_env.reset()
        assert observation.shape == (1, 299)
        assert np.isfinite(observation).all()

        observation, rewards, dones, infos = normalized_env.step(
            np.zeros((1, 58), dtype=np.float32)
        )
        assert observation.shape == (1, 299)
        assert rewards.shape == dones.shape == (1,)
        assert len(infos) == 1
        assert np.isfinite(observation).all()
        assert np.isfinite(rewards).all()
    finally:
        normalized_env.close()


@pytest.mark.skipif(
    sys.platform == "darwin"
    or (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}
    ),
    reason="test runner does not provide a MuJoCo GL context",
)
def test_rgb_array_render_smoke():
    env = HumanoidTennisCoopEnv(
        render_mode="rgb_array",
        width=320,
        height=240,
    )
    try:
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert np.isfinite(frame).all()
    finally:
        env.close()


def test_registered_environment_constructs_with_expected_time_limit():
    spec = gym.spec("CourtsideDynamics/HumanoidTennisCoop")
    assert spec.max_episode_steps == 1_000
    assert spec.entry_point == (
        "courtside_dynamics.envs.humanoid_tennis:HumanoidTennisCoopEnv"
    )

    env = gym.make("CourtsideDynamics/HumanoidTennisCoop")
    try:
        observation, info = env.reset(seed=0)
        assert isinstance(env.unwrapped, HumanoidTennisCoopEnv)
        assert observation.shape == (299,)
        assert info["serve_side_name"] == "a"
    finally:
        env.close()


def test_physical_oracle_accepts_registered_time_limit_wrapper():
    env = gym.make(
        "CourtsideDynamics/HumanoidTennisCoop",
        episode_len=300,
    )
    try:
        result = run_humanoid_tennis_oracle(env, serving_side="a")
        assert result.rally_count == 1
        assert result.event_kinds == (
            "net_crossing_to_b",
            "ball_racket_b",
            "net_crossing_to_a",
            "ball_court_a",
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("side", "expected_events"),
    [
        (
            CourtSide.A,
            (
                "net_crossing_to_b",
                "ball_racket_b",
                "net_crossing_to_a",
                "ball_court_a",
            ),
        ),
        (
            CourtSide.B,
            (
                "net_crossing_to_a",
                "ball_racket_a",
                "net_crossing_to_b",
                "ball_court_b",
            ),
        ),
    ],
)
def test_mirrored_physical_oracle_completes_one_legal_return(
    side: CourtSide,
    expected_events: tuple[str, ...],
):
    env = HumanoidTennisCoopEnv(episode_len=300)
    try:
        result = run_humanoid_tennis_oracle(
            env,
            serving_side=side,
            seed=0,
            max_steps=300,
        )

        assert result.rally_count == 1
        assert result.total_reward == pytest.approx(1.0)
        assert result.terminated is False
        assert result.truncated is False
        assert result.event_kinds == expected_events
        assert result.final_info["event_valid_return"] is True
        assert result.final_info["rally_target_reached"] is True
        assert result.final_info["termination_reason_name"] == "none"
    finally:
        env.close()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: HumanoidTennisCoopEnv(robot_model="unsupported"),
        lambda: HumanoidTennisCoopEnv(episode_len=0),
        lambda: HumanoidTennisCoopEnv(frame_skip=0),
        lambda: HumanoidTennisCoopEnv(rally_target=0),
        lambda: HumanoidTennisCoopEnv(
            serve_config=TennisServeConfig(position_noise=(-0.1, 0.0, 0.0))
        ),
        lambda: HumanoidTennisCoopEnv(
            serve_config=TennisServeConfig(
                start_distance_from_net=0.01,
                position_noise=(0.1, 0.0, 0.0),
            )
        ),
        lambda: HumanoidTennisCoopEnv(
            reward_config=TennisRewardConfig(fault_penalty=-1.0)
        ),
    ],
)
def test_invalid_configuration_is_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()
