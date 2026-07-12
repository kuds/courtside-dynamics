"""Pure contract tests for the humanoid-tennis curriculum presets."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from courtside_dynamics.envs._tennis_physics import (
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
    LINE_TOLERANCE,
)
from courtside_dynamics.envs.robot_models import UNITREE_G1_ACTUATOR_NAMES
from courtside_dynamics.envs.tennis_curriculum import (
    TENNIS_CURRICULUM_PRESETS,
    CurriculumCourtConfig,
    CurriculumLaunchConfig,
    CurriculumStage,
    FaultStrictness,
    LearnedPlayerMode,
    MobilityMode,
    PartnerMode,
    ServeMode,
    StageObjective,
    coerce_curriculum_stage,
    get_curriculum_config,
    get_curriculum_preset,
)
from courtside_dynamics.envs.tennis_rules import CourtSide


def test_preset_registry_covers_every_stage_and_states_availability() -> None:
    assert tuple(TENNIS_CURRICULUM_PRESETS) == tuple(CurriculumStage)

    available = {
        stage
        for stage, preset in TENNIS_CURRICULUM_PRESETS.items()
        if preset.environment_available
    }
    trainable = {
        stage
        for stage, preset in TENNIS_CURRICULUM_PRESETS.items()
        if preset.training_recipe_available
    }
    assert available == {
        CurriculumStage.ANCHORED_RACKET_INTERCEPT,
        CurriculumStage.ANCHORED_RETURN,
        CurriculumStage.ANCHORED_RANDOMIZED_RETURN,
        CurriculumStage.TWO_FREE_STANDING,
    }
    assert trainable == {
        CurriculumStage.ANCHORED_RACKET_INTERCEPT,
        CurriculumStage.ANCHORED_RETURN,
        CurriculumStage.ANCHORED_RANDOMIZED_RETURN,
    }
    assert get_curriculum_preset(6).limitation is not None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, CurriculumStage.ANCHORED_RACKET_INTERCEPT),
        ("0", CurriculumStage.ANCHORED_RACKET_INTERCEPT),
        ("stage-1", CurriculumStage.ANCHORED_RETURN),
        ("anchored randomized return", CurriculumStage.ANCHORED_RANDOMIZED_RETURN),
        ("TWO_FREE_STANDING", CurriculumStage.TWO_FREE_STANDING),
    ],
)
def test_stage_coercion_accepts_public_spellings(value, expected) -> None:
    assert coerce_curriculum_stage(value) is expected


def test_stage_coercion_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown tennis curriculum stage"):
        coerce_curriculum_stage("almost_tennis")


@pytest.mark.parametrize(
    "stage",
    [
        CurriculumStage.CONSTRAINED_STANDING_RETURN,
        CurriculumStage.MOBILE_WITH_SCRIPTED_PARTNER,
        CurriculumStage.TWO_CONSTRAINED_LEARNED,
    ],
)
def test_planned_stages_fail_explicitly(stage: CurriculumStage) -> None:
    preset = get_curriculum_preset(stage)
    assert preset.environment_available is False
    assert preset.training_recipe_available is False
    assert preset.limitation
    with pytest.raises(NotImplementedError, match="is not available"):
        get_curriculum_config(stage)
    assert get_curriculum_config(stage, require_available=False) is preset.config


def test_supported_stage_contracts_lock_progression() -> None:
    stage0 = get_curriculum_config(0)
    stage1 = get_curriculum_config(1)
    stage2 = get_curriculum_config(2)
    stage6 = get_curriculum_config(6)

    assert stage0.mobility is MobilityMode.FIXED_BASE
    assert stage0.objective is StageObjective.FIRST_VALID_RACKET_HIT
    assert stage0.learned_players is LearnedPlayerMode.CURRENT_RETURNER
    assert stage0.partner_mode is PartnerMode.PD_HOLD
    assert stage0.active_joint_names == UNITREE_G1_ACTUATOR_NAMES[22:24]
    assert stage0.launch.mode is ServeMode.SCRIPTED

    for config in (stage1, stage2):
        assert config.mobility is MobilityMode.FIXED_BASE
        assert config.objective is StageObjective.VALID_TARGET_RETURN
        assert config.active_joint_names == UNITREE_G1_ACTUATOR_NAMES[22:29]
    assert stage1.launch.mode is ServeMode.SCRIPTED
    assert stage2.launch.mode is ServeMode.RANDOMIZED
    assert stage0.racket_contact_scale >= stage1.racket_contact_scale
    assert stage1.racket_contact_scale > stage2.racket_contact_scale > 1.0

    assert stage6.mobility is MobilityMode.FREE_STANDING
    assert stage6.objective is StageObjective.RALLY_TARGET
    assert stage6.learned_players is LearnedPlayerMode.BOTH
    assert stage6.partner_mode is PartnerMode.LEARNED
    assert stage6.active_joint_names == UNITREE_G1_ACTUATOR_NAMES
    assert stage6.racket_contact_scale == 1.0
    assert stage6.rally_target == 4


@pytest.mark.parametrize("stage", list(CurriculumStage))
def test_every_preset_has_a_complete_disjoint_joint_partition(
    stage: CurriculumStage,
) -> None:
    config = get_curriculum_preset(stage).config
    active = set(config.active_joint_names)
    held = set(config.pd_hold_joint_names)
    assert not active & held
    assert active | held == set(UNITREE_G1_ACTUATOR_NAMES)
    assert len(active) == len(config.active_joint_names)
    assert len(held) == len(config.pd_hold_joint_names)


def test_config_validation_rejects_invalid_joint_partitions_and_roles() -> None:
    config = get_curriculum_config(0)
    with pytest.raises(ValueError, match="overlap"):
        replace(
            config,
            pd_hold_joint_names=(
                config.active_joint_names[0],
                *config.pd_hold_joint_names,
            ),
        )
    with pytest.raises(ValueError, match="incomplete"):
        replace(config, pd_hold_joint_names=config.pd_hold_joint_names[:-1])
    with pytest.raises(ValueError, match="unknown curriculum joints"):
        replace(
            config,
            active_joint_names=("unknown_joint", *config.active_joint_names[1:]),
        )
    with pytest.raises(ValueError, match="cannot have a learned partner"):
        replace(config, partner_mode=PartnerMode.LEARNED)

    stage6 = get_curriculum_config(6)
    with pytest.raises(ValueError, match="require partner_mode='learned'"):
        replace(stage6, partner_mode=PartnerMode.PD_HOLD)
    with pytest.raises(ValueError, match="rally-target objective"):
        replace(stage6, objective=StageObjective.FIRST_VALID_RACKET_HIT)
    with pytest.raises(ValueError, match="success latch"):
        replace(config, terminate_on_success=False)


def test_fault_strictness_maps_to_explicit_rule_bundles() -> None:
    strict = get_curriculum_config(0).rally_rules()
    assert strict.ball_net_is_fault
    assert strict.strict_reverse_crossing
    assert strict.strict_double_hit

    forgiving = replace(
        get_curriculum_config(0),
        fault_strictness=FaultStrictness.FORGIVING,
    ).rally_rules()
    assert not forgiving.ball_net_is_fault
    assert not forgiving.strict_reverse_crossing
    assert not forgiving.strict_double_hit


def test_launch_validation_and_velocity_conversion() -> None:
    launch = CurriculumLaunchConfig(
        mode=ServeMode.SCRIPTED,
        start_distance_from_net=2.0,
        lateral_position=0.0,
        height=1.3,
        speed=10.0,
        elevation_degrees=30.0,
        lateral_degrees=0.0,
    )
    vx, vy, vz = launch.velocity()
    assert vx == pytest.approx(10.0 * math.cos(math.radians(30.0)))
    assert vy == pytest.approx(0.0)
    assert vz == pytest.approx(5.0)
    assert math.sqrt(vx * vx + vy * vy + vz * vz) == pytest.approx(10.0)

    with pytest.raises(ValueError, match="scripted launches cannot include"):
        replace(launch, position_noise=(0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="remain positive"):
        replace(launch, speed_noise=10.0, mode=ServeMode.RANDOMIZED)
    with pytest.raises(ValueError, match="retain a forward component"):
        replace(launch, elevation_degrees=89.0)
    with pytest.raises(ValueError, match="1.1–1.5"):
        replace(launch, height=1.0)
    with pytest.raises(ValueError, match="inside the baseline"):
        replace(
            launch,
            mode=ServeMode.RANDOMIZED,
            start_distance_from_net=COURT_HALF_LENGTH - 0.05,
            position_noise=(0.1, 0.0, 0.0),
        )


def test_court_validation_rejects_regions_outside_regulation_or_playable() -> None:
    with pytest.raises(ValueError, match="fit the regulation court"):
        CurriculumCourtConfig(
            playable_half_length=COURT_HALF_LENGTH + 0.01,
            playable_half_width=COURT_HALF_WIDTH,
            target_min_depth=0.0,
            target_max_depth=COURT_HALF_LENGTH,
        )
    with pytest.raises(ValueError, match="target depth"):
        CurriculumCourtConfig(
            playable_half_length=4.0,
            playable_half_width=2.0,
            target_min_depth=0.0,
            target_max_depth=4.1,
        )
    with pytest.raises(ValueError, match="target width"):
        CurriculumCourtConfig(
            playable_half_length=4.0,
            playable_half_width=2.0,
            target_min_depth=0.0,
            target_max_depth=4.0,
            target_center_y=1.0,
            target_half_width=1.1,
        )


def test_playable_and_mirrored_target_boundaries_are_inclusive() -> None:
    court = CurriculumCourtConfig(
        playable_half_length=4.0,
        playable_half_width=2.5,
        target_min_depth=1.0,
        target_max_depth=3.0,
        target_center_y=0.5,
        target_half_width=1.0,
    )
    assert court.contains_playable((4.0, 2.5, 0.0))
    assert court.contains_playable((-4.0, -2.5, 0.0))
    assert not court.contains_playable((4.0 + 2 * LINE_TOLERANCE, 0.0, 0.0))

    # Positive local target y mirrors to negative world y on side A.
    assert court.contains_target((1.0, 1.5, 0.0), side=CourtSide.B)
    assert court.contains_target((3.0, -0.5, 0.0), side=CourtSide.B)
    assert court.contains_target((-1.0, -1.5, 0.0), side=CourtSide.A)
    assert court.contains_target((-3.0, 0.5, 0.0), side=CourtSide.A)
    assert not court.contains_target(
        (3.0 + 2 * LINE_TOLERANCE, 0.5, 0.0),
        side=CourtSide.B,
    )
    assert not court.contains_target(
        (-2.0, -1.5 - 2 * LINE_TOLERANCE, 0.0),
        side=CourtSide.A,
    )
