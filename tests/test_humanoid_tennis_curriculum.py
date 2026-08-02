"""Physics and Gymnasium integration tests for humanoid-tennis curricula."""

from __future__ import annotations

import math
import pickle
from dataclasses import replace

import numpy as np
import pytest

from courtside_dynamics.envs.humanoid_tennis import (
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    HumanoidTennisCoopEnv,
    TennisRewardConfig,
)
from courtside_dynamics.envs.robot_models import UNITREE_G1_STAND_JOINT_POSITIONS
from courtside_dynamics.envs.tennis_curriculum import (
    get_curriculum_config,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
)
from courtside_dynamics.scripted_policies import (
    humanoid_tennis_stage0_action,
    run_humanoid_tennis_stage0_oracle,
    run_humanoid_tennis_stage1_oracle,
)
from courtside_dynamics.training.tennis_curriculum import (
    HeldOutSeedSuite,
    evaluate_curriculum_stage,
)
from tests._helpers import event_batch as _event_batch
from tests._helpers import inject_batches as _inject_batches


def _feed_crossing_to_b(substep: int = 0) -> RallyEvent:
    return RallyEvent(
        RallyEventKind.NET_CROSSING_TO_B,
        substep=substep,
        event_id=substep,
        from_side=CourtSide.A,
        to_side=CourtSide.B,
    )


def _racket_b_hit(substep: int = 1, *, event_id: int | None = None) -> RallyEvent:
    return RallyEvent(
        RallyEventKind.BALL_RACKET_B,
        substep=substep,
        event_id=substep if event_id is None else event_id,
        contact_episode=0,
        peak_force=5.0,
    )


def _return_crossing_to_a(substep: int = 2) -> RallyEvent:
    return RallyEvent(
        RallyEventKind.NET_CROSSING_TO_A,
        substep=substep,
        event_id=substep,
        from_side=CourtSide.B,
        to_side=CourtSide.A,
    )


def _court_a_landing(
    position: tuple[float, float, float],
    *,
    substep: int = 3,
) -> RallyEvent:
    return RallyEvent(
        RallyEventKind.BALL_COURT_A,
        substep=substep,
        event_id=substep,
        contact_episode=0,
        position=position,
        in_bounds=True,
        peak_force=4.0,
    )


@pytest.mark.parametrize("stage", [0, 1, 2, 6])
def test_supported_stages_preserve_fixed_action_and_observation_contract(
    stage: int,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=stage, episode_len=20)
    try:
        observation, info = env.reset(seed=0)
        assert env.action_space.shape == (58,)
        assert env.action_space.dtype == np.dtype(np.float32)
        assert env.observation_space.shape == (299,)
        assert observation.shape == (299,)
        assert observation.dtype == np.float64
        assert np.array_equal(
            observation[HUMANOID_TENNIS_OBSERVATION_LAYOUT.active_action_mask],
            env.active_action_mask,
        )
        assert info["curriculum_stage"] == stage
    finally:
        env.close()


def test_stage0_passes_sb3_checker_and_ezpickle_round_trip() -> None:
    from stable_baselines3.common.env_checker import check_env

    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=150)
    try:
        check_env(env)
        rebuilt = pickle.loads(pickle.dumps(env))
        try:
            observation, info = rebuilt.reset(seed=0)
            assert observation.shape == (299,)
            assert rebuilt.action_space.shape == (58,)
            assert info["curriculum_stage"] == 0
            assert info["active_action_count"] == 2
            assert info["curriculum_launch_overridden"] is False
            assert info["initial_ball_qpos"][:3] == pytest.approx(
                (-1.0, 0.5, 1.25)
            )
            metadata = rebuilt.curriculum_metadata
            assert metadata["curriculum_matches_canonical_preset"] is True
            assert metadata["robot_model"] == "unitree_g1"
            assert metadata["episode_len"] == 150
            assert metadata["frame_skip"] == 10
            assert metadata["sampler_release_substeps"] == 2
            assert metadata["model_timestep"] == pytest.approx(0.001)
            assert metadata["control_timestep"] == pytest.approx(0.01)
        finally:
            rebuilt.close()
    finally:
        env.close()


def test_custom_easier_curriculum_is_not_labeled_as_canonical() -> None:
    custom = replace(get_curriculum_config(0), racket_contact_scale=1.6)
    env = HumanoidTennisCoopEnv(curriculum_config=custom, episode_len=150)
    try:
        assert env.curriculum_metadata["curriculum_matches_canonical_preset"] is False
    finally:
        env.close()


@pytest.mark.parametrize("stage", [3, 4, 5])
def test_environment_rejects_planned_only_stages_before_simulation(stage: int) -> None:
    with pytest.raises(NotImplementedError, match="is not available"):
        HumanoidTennisCoopEnv(curriculum_config=stage)


@pytest.mark.parametrize(
    ("stage", "local_active"),
    [
        (0, np.arange(22, 24)),
        (1, np.arange(22, 29)),
        (2, np.arange(22, 29)),
    ],
)
def test_alternating_serve_switches_exact_returner_action_mask(
    stage: int,
    local_active: np.ndarray,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=stage)
    try:
        observation_a, info_a = env.reset(seed=7)
        expected_b = local_active + 29
        assert np.flatnonzero(env.active_action_mask).tolist() == expected_b.tolist()
        assert info_a["serve_side_name"] == "a"
        assert info_a["learned_returner_name"] == "b"
        assert info_a["active_action_count_a"] == 0
        assert info_a["active_action_count_b"] == len(local_active)
        assert np.array_equal(
            observation_a[HUMANOID_TENNIS_OBSERVATION_LAYOUT.active_action_mask],
            env.active_action_mask,
        )

        observation_b, info_b = env.reset()
        assert np.flatnonzero(env.active_action_mask).tolist() == local_active.tolist()
        assert info_b["serve_side_name"] == "b"
        assert info_b["learned_returner_name"] == "a"
        assert info_b["active_action_count_a"] == len(local_active)
        assert info_b["active_action_count_b"] == 0
        assert np.array_equal(
            observation_b[HUMANOID_TENNIS_OBSERVATION_LAYOUT.active_action_mask],
            env.active_action_mask,
        )
    finally:
        env.close()


def test_stage6_exposes_all_controls_and_no_welds() -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=6)
    try:
        _, info = env.reset(seed=0)
        assert np.all(env.active_action_mask == 1.0)
        assert info["active_action_count"] == 58
        assert info["active_action_count_a"] == 29
        assert info["active_action_count_b"] == 29
        assert info["learned_returner_name"] == "both"
        assert all(
            not bool(env.data.eq_active[index])
            for index in env._pelvis_weld_ids.values()
        )
    finally:
        env.close()


def test_default_environment_remains_non_curriculum_full_action_regression() -> None:
    env = HumanoidTennisCoopEnv(episode_len=10)
    try:
        observation, info = env.reset(seed=0)
        assert env.curriculum_config is None
        assert env.curriculum_metadata == {"curriculum_enabled": False}
        assert np.all(env.active_action_mask == 1.0)
        assert "curriculum_stage" not in info
        next_observation, reward, terminated, truncated, step_info = env.step(
            env.neutral_action
        )
        assert next_observation.shape == observation.shape == (299,)
        assert np.isfinite(next_observation).all()
        assert math.isfinite(reward)
        assert not terminated
        assert not truncated
        assert "stage_success" not in step_info
    finally:
        env.close()


def test_stage0_scripted_launch_is_seed_independent_and_mirrored() -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0)
    try:
        _, info_a1 = env.reset(seed=1, options={"serve_side": "a"})
        _, info_a2 = env.reset(seed=999, options={"serve_side": "a"})
        _, info_b = env.reset(seed=42, options={"serve_side": "b"})
        qpos_a1 = np.asarray(info_a1["initial_ball_qpos"])
        qvel_a1 = np.asarray(info_a1["initial_ball_qvel"])
        qpos_a2 = np.asarray(info_a2["initial_ball_qpos"])
        qvel_a2 = np.asarray(info_a2["initial_ball_qvel"])
        qpos_b = np.asarray(info_b["initial_ball_qpos"])
        qvel_b = np.asarray(info_b["initial_ball_qvel"])

        assert qpos_a2 == pytest.approx(qpos_a1)
        assert qvel_a2 == pytest.approx(qvel_a1)
        assert qpos_b[:2] == pytest.approx(-qpos_a1[:2])
        assert qpos_b[2:] == pytest.approx(qpos_a1[2:])
        assert qvel_b[:2] == pytest.approx(-qvel_a1[:2])
        assert qvel_b[2:] == pytest.approx(qvel_a1[2:])
    finally:
        env.close()


def test_stage2_launch_is_seeded_reproducible_and_bounded() -> None:
    config = get_curriculum_config(2)
    launch = config.launch
    env = HumanoidTennisCoopEnv(curriculum_config=config)
    try:
        _, first = env.reset(seed=37, options={"serve_side": "a"})
        _, replay = env.reset(seed=37, options={"serve_side": "a"})
        assert replay["initial_ball_qpos"] == pytest.approx(first["initial_ball_qpos"])
        assert replay["initial_ball_qvel"] == pytest.approx(first["initial_ball_qvel"])

        states: set[tuple[float, ...]] = set()
        for seed in range(20):
            _, info = env.reset(seed=seed, options={"serve_side": "a"})
            position = np.asarray(info["initial_ball_qpos"][:3])
            velocity = np.asarray(info["initial_ball_qvel"][:3])
            states.add(tuple(position) + tuple(velocity))
            nominal_position = np.array(
                [
                    -launch.start_distance_from_net,
                    launch.lateral_position,
                    launch.height,
                ]
            )
            assert np.all(
                np.abs(position - nominal_position)
                <= np.asarray(launch.position_noise) + 1e-12
            )
            speed = float(np.linalg.norm(velocity))
            horizontal = float(np.linalg.norm(velocity[:2]))
            elevation = math.degrees(math.atan2(float(velocity[2]), horizontal))
            lateral = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
            assert (
                launch.speed - launch.speed_noise
                <= speed
                <= launch.speed + launch.speed_noise
            )
            assert (
                launch.elevation_degrees - launch.elevation_noise_degrees
                <= elevation
                <= launch.elevation_degrees + launch.elevation_noise_degrees
            )
            assert (
                launch.lateral_degrees - launch.lateral_noise_degrees
                <= lateral
                <= launch.lateral_degrees + launch.lateral_noise_degrees
            )
        assert len(states) > 1
    finally:
        env.close()


def test_inactive_actions_are_ignored_and_receive_standing_pd_targets() -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        mask = env.active_action_mask.astype(bool)
        action = np.zeros(58, dtype=np.float32)
        action[~mask] = 1.0
        controls = env._action_to_controls(action)
        standing = np.asarray(UNITREE_G1_STAND_JOINT_POSITIONS * 2)
        assert controls == pytest.approx(standing)

        active_action = action.copy()
        active_action[mask] = 0.5
        active_controls = env._action_to_controls(active_action)
        assert active_controls[~mask] == pytest.approx(standing[~mask])
        assert np.any(np.abs(active_controls[mask] - standing[mask]) > 0.0)
    finally:
        env.close()


def test_inactive_policy_values_do_not_change_one_step_trajectory() -> None:
    baseline = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=20)
    probe = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=20)
    try:
        baseline.reset(seed=9, options={"serve_side": "a"})
        probe.reset(seed=9, options={"serve_side": "a"})
        mask = baseline.active_action_mask.astype(bool)
        ignored = np.zeros(58, dtype=np.float32)
        ignored[~mask] = 1.0
        obs_a, reward_a, term_a, trunc_a, info_a = baseline.step(
            baseline.neutral_action
        )
        obs_b, reward_b, term_b, trunc_b, info_b = probe.step(ignored)
        assert obs_b == pytest.approx(obs_a, abs=1e-12)
        assert reward_b == pytest.approx(reward_a)
        assert (term_b, trunc_b) == (term_a, trunc_a)
        assert info_b["step_event_kinds"] == info_a["step_event_kinds"]
        assert probe.data.ctrl == pytest.approx(baseline.data.ctrl)
    finally:
        baseline.close()
        probe.close()


def test_fixed_base_welds_are_active_and_hold_both_pelvises() -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=100)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        assert all(
            bool(env.data.eq_active[index]) for index in env._pelvis_weld_ids.values()
        )
        initial = {
            side: env.data.qpos[player_slice][:7].copy()
            for side, player_slice in env._player_qpos_slices.items()
        }
        action = np.zeros(58, dtype=np.float32)
        action[env.active_action_mask.astype(bool)] = 0.5
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated
        for side, player_slice in env._player_qpos_slices.items():
            assert env.data.qpos[player_slice][:3] == pytest.approx(
                initial[side][:3], abs=2e-4
            )
            # Quaternions may flip sign without changing orientation.
            assert abs(
                float(np.dot(env.data.qpos[player_slice][3:7], initial[side][3:7]))
            ) == pytest.approx(1.0, abs=2e-4)
    finally:
        env.close()


def test_racket_forgiveness_changes_only_stringbed_reach_not_rigid_mass() -> None:
    default = HumanoidTennisCoopEnv()
    stage0 = HumanoidTennisCoopEnv(curriculum_config=0)
    try:
        scale = get_curriculum_config(0).racket_contact_scale
        for side in CourtSide:
            default_geom = default._racket_stringbed_geom_ids[side]
            stage_geom = stage0._racket_stringbed_geom_ids[side]
            assert stage0.model.geom_size[stage_geom, 0] == pytest.approx(
                default.model.geom_size[default_geom, 0]
            )
            assert stage0.model.geom_size[stage_geom, 1:3] == pytest.approx(
                default.model.geom_size[default_geom, 1:3] * scale
            )
            default_body = default._racket_body_ids[side]
            stage_body = stage0._racket_body_ids[side]
            assert stage0.model.body_mass[stage_body] == pytest.approx(
                default.model.body_mass[default_body]
            )
            assert stage0.model.body_inertia[stage_body] == pytest.approx(
                default.model.body_inertia[default_body]
            )
    finally:
        default.close()
        stage0.close()


def _ball_touches_stringbed(env, offset: float) -> bool:
    """Teleport the ball ``offset`` along the stringbed's long in-plane
    axis (on the stringbed plane) and report whether a ball/stringbed
    contact is generated."""
    import mujoco

    stringbed_id = env._racket_stringbed_geom_ids[CourtSide.A]
    ball_geom_id = int(env.model.geom("ball_geom").id)
    center = env.data.geom_xpos[stringbed_id]
    frame = env.data.geom_xmat[stringbed_id].reshape(3, 3)
    qposadr = env._ball_qposadr
    env.data.qpos[qposadr : qposadr + 3] = center + frame[:, 2] * offset
    env.data.qvel[env._ball_dofadr : env._ball_dofadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    pairs = {
        frozenset((int(contact.geom1), int(contact.geom2)))
        for contact in env.data.contact[: env.data.ncon]
    }
    return frozenset((stringbed_id, ball_geom_id)) in pairs


def test_racket_forgiveness_band_generates_contacts() -> None:
    """The enlarged stringbed annulus must physically collide with the ball.

    Writing ``geom_size`` and calling ``mj_setConst`` leaves the
    compile-time broadphase bounds (``geom_rbound`` / ``geom_aabb``)
    stale, and MuJoCo culls candidate contact pairs on those bounds --
    the regression pinned here is the ball passing straight through the
    visibly enlarged outer band, making the forgiveness inert.
    """
    stage0 = HumanoidTennisCoopEnv(curriculum_config=0)
    try:
        stage0.reset(seed=0)
        stringbed_id = stage0._racket_stringbed_geom_ids[CourtSide.A]
        nominal = stage0._nominal_stringbed_sizes[CourtSide.A]
        scaled = stage0.model.geom_size[stringbed_id]
        assert scaled[2] > nominal[2]
        # The collision bounds must track the enlarged semi-axes.
        assert stage0.model.geom_rbound[stringbed_id] == pytest.approx(
            float(np.max(scaled))
        )
        assert stage0.model.geom_aabb[stringbed_id, 3:6] == pytest.approx(
            scaled
        )
        # Midpoint of the enlarged-only annulus: inside the scaled
        # ellipsoid, but clear of the nominal one by more than the ball
        # radius plus contact margins.
        offset = float(nominal[2] + scaled[2]) / 2.0
        assert _ball_touches_stringbed(stage0, offset)
    finally:
        stage0.close()

    default = HumanoidTennisCoopEnv()
    try:
        default.reset(seed=0)
        # Same absolute offset misses the nominal stringbed: the band is
        # genuinely enlarged-only, not a nominal-geometry artifact.
        assert not _ball_touches_stringbed(default, offset)
    finally:
        default.close()


@pytest.mark.parametrize("serving_side", [CourtSide.A, CourtSide.B])
def test_physical_stage0_oracle_succeeds_for_both_mirrored_sides(
    serving_side: CourtSide,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=150)
    try:
        result = run_humanoid_tennis_stage0_oracle(
            env,
            serving_side=serving_side,
            max_steps=150,
        )
        assert result.stage_success is True
        assert result.terminated is True
        assert result.truncated is False
        assert result.total_reward == pytest.approx(1.0)
        assert f"ball_racket_{serving_side.opponent.label}" in result.event_kinds
        assert result.final_info["term_stage_success"] is True
        assert result.final_info["rew_stage_success"] == pytest.approx(1.0)
        assert result.final_info["termination_reason_name"] == "none"
    finally:
        env.close()


@pytest.mark.parametrize("serving_side", [CourtSide.A, CourtSide.B])
def test_neutral_stage0_policy_does_not_satisfy_intercept(
    serving_side: CourtSide,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=150)
    try:
        _, info = env.reset(
            seed=0,
            options={"serve_side": serving_side.label},
        )
        total_reward = 0.0
        for _ in range(150):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            total_reward += reward
            if terminated or truncated:
                break
        assert info["stage_success"] is False
        assert info["term_stage_success"] is False
        assert total_reward <= 0.0
    finally:
        env.close()


@pytest.mark.parametrize("serving_side", [CourtSide.A, CourtSide.B])
def test_physical_stage1_oracle_lands_a_mirrored_target_return(
    serving_side: CourtSide,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=1, episode_len=300)
    try:
        result = run_humanoid_tennis_stage1_oracle(
            env,
            serving_side=serving_side,
            max_steps=300,
        )
        assert result.steps == 129
        assert result.stage_success is True
        assert result.rally_count == 1
        assert result.total_reward == pytest.approx(1.0)
        assert result.terminated is True
        assert result.truncated is False
        assert result.event_kinds == (
            f"net_crossing_to_{serving_side.opponent.label}",
            f"ball_racket_{serving_side.opponent.label}",
            f"net_crossing_to_{serving_side.label}",
            f"ball_court_{serving_side.label}",
        )
        assert result.final_info["target_hit"] is True
        assert result.final_info["term_stage_success"] is True
        assert result.final_info["return_opportunity_count"] == 1
        assert result.final_info["valid_return_rate"] == pytest.approx(1.0)
        assert result.final_info["termination_reason_name"] == "none"
    finally:
        env.close()


def test_stage0_duplicate_contact_cannot_multiply_success_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=20)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        hit = _racket_b_hit(1, event_id=1)
        duplicate = _racket_b_hit(1, event_id=2)
        _inject_batches(
            monkeypatch,
            env,
            iter((_event_batch(0, _feed_crossing_to_b(), hit, duplicate),)),
        )
        _, reward, terminated, truncated, info = env.step(env.neutral_action)
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(1.0)
        assert info["rew_stage_success"] == pytest.approx(1.0)
        assert info["legal_hit_count_b"] == 1
        assert info["duplicate_contact_suppressed_count"] == 1
    finally:
        env.close()


def test_tennis_fault_wins_over_same_batch_stage0_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=20)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        net_fault = RallyEvent(
            RallyEventKind.RACKET_B_NET,
            substep=1,
            event_id=2,
            contact_episode=0,
            peak_force=3.0,
        )
        _inject_batches(
            monkeypatch,
            env,
            iter(
                (
                    _event_batch(
                        0,
                        _feed_crossing_to_b(),
                        _racket_b_hit(),
                        net_fault,
                    ),
                )
            ),
        )
        _, reward, terminated, truncated, info = env.step(env.neutral_action)
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(-1.0)
        assert info["stage_success"] is False
        assert info["rew_stage_success"] == 0.0
        assert info["rew_fault"] == pytest.approx(-1.0)
        assert info["termination_reason_name"] == "racket_b_net"
    finally:
        env.close()


def _install_stage1_return_sequence(
    monkeypatch: pytest.MonkeyPatch,
    env: HumanoidTennisCoopEnv,
    landing: tuple[float, float, float],
) -> None:
    _inject_batches(
        monkeypatch,
        env,
        iter(
            (
                _event_batch(0, _feed_crossing_to_b(0)),
                _event_batch(1, _racket_b_hit(1)),
                _event_batch(2, _return_crossing_to_a(2)),
                _event_batch(3, _court_a_landing(landing, substep=3)),
            )
        ),
    )


@pytest.mark.parametrize(
    "landing",
    [
        (-2.0, 0.0, 0.0),
        (-4.0, 2.5, 0.0),
    ],
)
def test_stage1_in_target_landing_is_terminal_success(
    monkeypatch: pytest.MonkeyPatch,
    landing: tuple[float, float, float],
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=1, episode_len=20)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        _install_stage1_return_sequence(monkeypatch, env, landing)
        for _ in range(3):
            _, reward, terminated, truncated, _ = env.step(env.neutral_action)
            assert reward == 0.0
            assert not terminated
            assert not truncated
        _, reward, terminated, truncated, info = env.step(env.neutral_action)
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(1.0)
        assert info["stage_success"] is True
        assert info["stage_attempt_complete"] is True
        assert info["target_hit"] is True
        assert info["target_miss"] is False
        assert info["term_stage_success"] is True
        assert info["term_target_miss"] is False
        assert info["rew_valid_return"] == 0.0
        assert info["rew_stage_success"] == pytest.approx(1.0)
        assert info["termination_reason_name"] == "none"
    finally:
        env.close()


def test_stage1_target_miss_is_terminal_curriculum_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HumanoidTennisCoopEnv(curriculum_config=1, episode_len=20)
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        # This is legal on the regulation court but outside Stage 1's
        # four-metre playable/target overlay.
        _install_stage1_return_sequence(monkeypatch, env, (-5.0, 0.0, 0.0))
        for _ in range(3):
            env.step(env.neutral_action)
        _, reward, terminated, truncated, info = env.step(env.neutral_action)
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(-1.0)
        assert info["rally_count"] == 1
        assert info["stage_success"] is False
        assert info["stage_attempt_complete"] is True
        assert info["target_hit"] is False
        assert info["target_miss"] is True
        assert info["term_stage_success"] is False
        assert info["term_target_miss"] is True
        assert info["rew_stage_success"] == 0.0
        assert info["rew_fault"] == pytest.approx(-1.0)
        # Curriculum failure does not falsify the tennis rule reducer.
        assert info["termination_reason_name"] == "none"
    finally:
        env.close()


def test_stage1_target_miss_claws_back_hit_shaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HumanoidTennisCoopEnv(
        curriculum_config=1,
        episode_len=20,
        reward_config=TennisRewardConfig(valid_hit_shaping=0.25),
    )
    try:
        env.reset(seed=0, options={"serve_side": "a"})
        _install_stage1_return_sequence(monkeypatch, env, (-5.0, 0.0, 0.0))
        total_reward = 0.0
        for _ in range(4):
            _, reward, terminated, truncated, info = env.step(env.neutral_action)
            total_reward += reward
            if terminated or truncated:
                break
        assert info["term_target_miss"] is True
        assert info["rew_shaping_clawback"] == pytest.approx(-0.25)
        assert info["rew_fault"] == pytest.approx(-1.0)
        # The earlier +0.25 shaping and terminal -0.25 clawback cancel.
        assert total_reward == pytest.approx(-1.0)
    finally:
        env.close()


def test_held_out_evaluator_runs_physical_mirrored_stage0_oracle() -> None:
    suite = HeldOutSeedSuite(
        base_seed=90_000,
        launch_count=2,
        max_position_offset=(0.0, 0.0, 0.0),
        max_velocity_offset=(0.0, 0.0, 0.0),
    )
    summary = evaluate_curriculum_stage(
        humanoid_tennis_stage0_action,
        lambda: HumanoidTennisCoopEnv(curriculum_config=0, episode_len=150),
        policy_id="stage0-physical-oracle-v1",
        normalization_id="raw",
        stage_name="anchored_racket_intercept",
        suite=suite,
        max_steps_per_episode=150,
    )
    assert len(summary.episodes) == 4
    assert summary.success_rate == pytest.approx(1.0)
    assert summary.valid_return_rate == 0.0
    assert summary.fault_rate("ball_net") == 0.0
    assert summary.conditions == suite.conditions
