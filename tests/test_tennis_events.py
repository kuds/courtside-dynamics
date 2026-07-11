"""Substep contact-event tests against the physical tennis scene."""

from __future__ import annotations

import pickle

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._tennis_events import (
    SubstepTennisEventSampler,
    TennisEventSamplerConfig,
    TennisSafetyLimits,
    TennisSceneContactIndex,
)
from courtside_dynamics.envs._tennis_physics import (
    BALL_RADIUS,
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
)
from courtside_dynamics.envs.robot_models import initialize_humanoid_tennis_home
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEventKind,
    RallyPhase,
    RallyStateMachine,
)


def _fresh_model():
    model = mujoco.MjModel.from_xml_path(asset_path("humanoid_tennis.xml"))
    data = mujoco.MjData(model)
    initialize_humanoid_tennis_home(model, data)
    return model, data


def _set_ball_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    position: np.ndarray | tuple[float, float, float],
    velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    dofadr = int(joint.dofadr[0])
    data.qpos[qposadr : qposadr + 3] = position
    data.qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dofadr : dofadr + 6] = 0.0
    data.qvel[dofadr : dofadr + 3] = velocity
    mujoco.mj_forward(model, data)


def _set_ball_toward_stringbed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    player: str,
    *,
    speed: float,
) -> None:
    stringbed = model.geom(f"{player}_racket_racket_stringbed")
    center = data.geom_xpos[stringbed.id].copy()
    normal = data.geom_xmat[stringbed.id].reshape(3, 3)[:, 0].copy()
    _set_ball_state(
        model,
        data,
        position=center - 0.2 * normal,
        velocity=speed * normal,
    )


def _sample_steps(
    sampler: SubstepTennisEventSampler,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    control_step: int,
    count: int,
):
    sampler.begin_control_step(control_step)
    for substep in range(count):
        mujoco.mj_step(model, data)
        sampler.sample_substep(data, control_substep=substep)
    return sampler.end_control_step()


def test_scene_contact_index_resolves_all_required_semantic_groups():
    model, _ = _fresh_model()
    index = TennisSceneContactIndex.from_model(model)
    assert len(index.net_geoms) == 7
    assert len(index.racket_a_geoms) == 8
    assert len(index.racket_b_geoms) == 8
    assert len(index.humanoid_a_geoms) == 27
    assert len(index.humanoid_b_geoms) == 27
    groups = (
        {index.ball_geom},
        {index.court_geom},
        index.net_geoms,
        index.racket_a_geoms,
        index.racket_b_geoms,
        index.humanoid_a_geoms,
        index.humanoid_b_geoms,
    )
    assert all(
        not left & right
        for left_index, left in enumerate(groups)
        for right in groups[left_index + 1 :]
    )


@pytest.mark.parametrize(
    "player,expected_kind",
    [
        ("player_a", RallyEventKind.BALL_RACKET_A),
        ("player_b", RallyEventKind.BALL_RACKET_B),
    ],
)
def test_high_speed_racket_impact_inside_control_frame_is_not_missed(
    player,
    expected_kind,
):
    model, data = _fresh_model()
    _set_ball_toward_stringbed(model, data, player, speed=60.0)
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(
        data,
        ball_side=CourtSide.A if player == "player_a" else CourtSide.B,
    )

    batch = _sample_steps(sampler, model, data, control_step=0, count=8)
    racket_events = [event for event in batch.events if event.kind is expected_kind]
    assert len(racket_events) == 1
    event = racket_events[0]
    assert 0 < event.substep < batch.substeps_sampled - 1
    assert event.peak_force > 0.0
    assert batch.contact_peaks[expected_kind] == pytest.approx(event.peak_force)
    assert expected_kind not in batch.active_contact_latches


def test_persistent_contact_across_control_boundary_emits_one_edge():
    model, data = _fresh_model()
    _set_ball_state(
        model,
        data,
        position=(-2.0, 0.0, 0.04),
        velocity=(0.0, 0.0, -1.0),
    )
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)

    first = _sample_steps(sampler, model, data, control_step=0, count=5)
    assert not first.events
    contact = _sample_steps(sampler, model, data, control_step=1, count=5)
    assert [event.kind for event in contact.events] == [
        RallyEventKind.BALL_COURT_A
    ]
    assert "ball_court" in contact.active_contact_latches
    after = _sample_steps(sampler, model, data, control_step=2, count=5)
    assert RallyEventKind.BALL_COURT_A not in {
        event.kind for event in after.events
    }


def test_one_substep_contact_chatter_is_suppressed_then_real_recontact_rearms():
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-2.0, 0.0, 1.0))
    sampler = SubstepTennisEventSampler(
        model,
        config=TennisEventSamplerConfig(release_substeps=2),
    )
    sampler.reset(data, ball_side=CourtSide.A)
    ball_joint = model.joint("ball_free")
    qposadr = int(ball_joint.qposadr[0])

    sampler.begin_control_step(0)
    for substep, height in enumerate((0.03, 1.0, 0.03, 1.0, 1.0, 0.03)):
        data.qpos[qposadr + 2] = height
        mujoco.mj_forward(model, data)
        sampler.sample_substep(data, control_substep=substep)
    batch = sampler.end_control_step()

    court_events = [
        event
        for event in batch.events
        if event.kind is RallyEventKind.BALL_COURT_A
    ]
    assert [event.contact_episode for event in court_events] == [0, 1]


def test_net_crossing_uses_deadband_and_interpolates_crossing_position():
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-1.0, 0.2, 1.3))
    sampler = SubstepTennisEventSampler(
        model,
        config=TennisEventSamplerConfig(crossing_deadband=1e-4),
    )
    sampler.reset(data, ball_side=CourtSide.A)
    ball_joint = model.joint("ball_free")
    qposadr = int(ball_joint.qposadr[0])

    sampler.begin_control_step(0)
    for substep, x in enumerate((-5e-5, 5e-5, -2e-5, 0.2)):
        data.qpos[qposadr : qposadr + 3] = (x, 0.4, 1.1)
        mujoco.mj_forward(model, data)
        sampler.sample_substep(data, control_substep=substep)
    batch = sampler.end_control_step()
    crossings = [
        event
        for event in batch.events
        if event.kind is RallyEventKind.NET_CROSSING_TO_B
    ]
    assert len(crossings) == 1
    assert crossings[0].from_side is CourtSide.A
    assert crossings[0].to_side is CourtSide.B
    assert crossings[0].position is not None
    assert crossings[0].position[0] == 0.0
    assert 0.2 <= crossings[0].position[1] <= 0.4


@pytest.mark.parametrize(
    "position,expected_kind,expected_in_bounds",
    [
        (
            (-COURT_HALF_LENGTH, 0.0, BALL_RADIUS - 0.004),
            RallyEventKind.BALL_COURT_A,
            True,
        ),
        (
            (COURT_HALF_LENGTH, COURT_HALF_WIDTH, BALL_RADIUS - 0.004),
            RallyEventKind.BALL_COURT_B,
            True,
        ),
        (
            (
                COURT_HALF_LENGTH + 1e-5,
                0.0,
                BALL_RADIUS - 0.004,
            ),
            RallyEventKind.BALL_COURT_B,
            False,
        ),
        (
            (
                -2.0,
                -COURT_HALF_WIDTH - 1e-5,
                BALL_RADIUS - 0.004,
            ),
            RallyEventKind.BALL_COURT_A,
            False,
        ),
    ],
)
def test_court_event_uses_contact_position_for_side_and_bounds(
    position,
    expected_kind,
    expected_in_bounds,
):
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(position[0], position[1], 1.0))
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(
        data,
        ball_side=(
            CourtSide.A
            if expected_kind is RallyEventKind.BALL_COURT_A
            else CourtSide.B
        ),
    )
    _set_ball_state(model, data, position=position)

    sampler.begin_control_step(0)
    sampler.sample_substep(data, control_substep=0)
    batch = sampler.end_control_step()
    event = next(event for event in batch.events if event.kind is expected_kind)
    assert event.in_bounds is expected_in_bounds
    assert event.position is not None
    assert event.position[0] == pytest.approx(position[0])
    assert event.position[1] == pytest.approx(position[1])


def test_high_speed_ball_net_contact_is_sampled_with_positive_peak():
    model, data = _fresh_model()
    _set_ball_state(
        model,
        data,
        position=(-0.25, 0.0, 0.5),
        velocity=(20.0, 0.0, 0.0),
    )
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)
    batch = _sample_steps(sampler, model, data, control_step=0, count=50)
    net_events = [
        event for event in batch.events if event.kind is RallyEventKind.BALL_NET
    ]
    assert len(net_events) == 1
    assert net_events[0].peak_force > 0.0
    assert any(name.startswith("net_") for name in net_events[0].geom_names or ())


@pytest.mark.parametrize("player", ["player_a", "player_b"])
@pytest.mark.parametrize("object_kind", ["racket", "humanoid"])
def test_player_and_racket_net_contacts_are_semantically_classified(
    player,
    object_kind,
):
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    if object_kind == "racket":
        geom_name = f"{player}_racket_racket_stringbed"
        expected = (
            RallyEventKind.RACKET_A_NET
            if player == "player_a"
            else RallyEventKind.RACKET_B_NET
        )
    else:
        geom_name = f"{player}_torso_collision"
        expected = (
            RallyEventKind.HUMANOID_A_NET
            if player == "player_a"
            else RallyEventKind.HUMANOID_B_NET
        )
    geom = model.geom(geom_name)
    root = model.joint(f"{player}_floating_base_joint")
    root_qposadr = int(root.qposadr[0])
    data.qpos[root_qposadr] -= data.geom_xpos[geom.id, 0]
    mujoco.mj_forward(model, data)

    batch = _sample_steps(sampler, model, data, control_step=0, count=1)
    assert expected in {event.kind for event in batch.events}
    assert batch.contact_peaks[expected] > 0.0


@pytest.mark.parametrize(
    "player,expected",
    [
        ("player_a", RallyEventKind.BALL_HUMANOID_A),
        ("player_b", RallyEventKind.BALL_HUMANOID_B),
    ],
)
def test_ball_humanoid_contact_is_recorded_for_reward_hacking_prevention(
    player,
    expected,
):
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    torso = model.geom(f"{player}_torso_collision")
    _set_ball_state(model, data, position=data.geom_xpos[torso.id].copy())

    sampler.begin_control_step(0)
    sampler.sample_substep(data, control_substep=0)
    batch = sampler.end_control_step()
    assert expected in {event.kind for event in batch.events}


@pytest.mark.parametrize(
    "mutate,expected",
    [
        ("nonfinite_qpos", RallyEventKind.NONFINITE_STATE),
        ("nonfinite_qvel", RallyEventKind.NONFINITE_STATE),
        ("nonfinite_ctrl", RallyEventKind.NONFINITE_STATE),
        ("nonfinite_applied_force", RallyEventKind.NONFINITE_STATE),
        ("unsafe_speed", RallyEventKind.UNSAFE_PHYSICS),
    ],
)
def test_sampler_emits_safety_event_before_contact_parsing(mutate, expected):
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    if mutate == "nonfinite_qpos":
        data.qpos[0] = np.nan
    elif mutate == "nonfinite_qvel":
        data.qvel[0] = np.inf
    elif mutate == "nonfinite_ctrl":
        data.ctrl[0] = np.nan
    elif mutate == "nonfinite_applied_force":
        data.xfrc_applied[0, 0] = np.inf
    else:
        ball_dofadr = int(model.joint("ball_free").dofadr[0])
        data.qvel[ball_dofadr] = 121.0

    sampler.begin_control_step(0)
    emitted = sampler.sample_substep(data, control_substep=0)
    batch = sampler.end_control_step()
    assert emitted is expected
    assert [event.kind for event in batch.events] == [expected]
    assert not batch.contact_peaks


def test_zero_normal_force_constraint_does_not_emit_contact(monkeypatch):
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-2.0, 0.0, 1.0))
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)
    _set_ball_state(model, data, position=(-2.0, 0.0, 0.03))

    def zero_contact_force(model, data, contact_index, force):
        del model, data, contact_index
        force[:] = 0.0

    monkeypatch.setattr(mujoco, "mj_contactForce", zero_contact_force)
    sampler.begin_control_step(0)
    assert sampler.sample_substep(data, control_substep=0) is None
    batch = sampler.end_control_step()
    assert not batch.events
    assert not batch.contact_peaks
    assert "ball_court" not in batch.active_contact_latches


def test_nonfinite_solver_contact_force_becomes_safety_event(monkeypatch):
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-2.0, 0.0, 1.0))
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)
    _set_ball_state(model, data, position=(-2.0, 0.0, 0.03))

    def nan_contact_force(model, data, contact_index, force):
        del model, data, contact_index
        force[:] = np.nan

    monkeypatch.setattr(mujoco, "mj_contactForce", nan_contact_force)
    sampler.begin_control_step(0)
    assert (
        sampler.sample_substep(data, control_substep=0)
        is RallyEventKind.NONFINITE_STATE
    )
    batch = sampler.end_control_step()
    assert [event.kind for event in batch.events] == [
        RallyEventKind.NONFINITE_STATE
    ]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_sampler_config_rejects_nonfinite_thresholds(value):
    with pytest.raises(ValueError, match="finite"):
        TennisEventSamplerConfig(min_contact_force=value)
    with pytest.raises(ValueError, match="finite"):
        TennisEventSamplerConfig(crossing_deadband=value)
    with pytest.raises(ValueError, match="positive"):
        TennisSafetyLimits(max_ball_linear_speed=value)


def test_every_semantic_geom_routes_independent_of_contact_order():
    model, _ = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    index = sampler.index

    def assert_pair(geom_a, geom_b, expected):
        for left, right in ((geom_a, geom_b), (geom_b, geom_a)):
            assert (
                sampler._classify_pair(  # noqa: SLF001 - internal contract test
                    left,
                    right,
                    contact_x=-1.0,
                    tie_breaker=CourtSide.A,
                )
                is expected
            )

    for geom in index.net_geoms:
        assert_pair(index.ball_geom, geom, RallyEventKind.BALL_NET)
        for racket in index.racket_a_geoms:
            assert_pair(racket, geom, RallyEventKind.RACKET_A_NET)
        for racket in index.racket_b_geoms:
            assert_pair(racket, geom, RallyEventKind.RACKET_B_NET)
        for humanoid in index.humanoid_a_geoms:
            assert_pair(humanoid, geom, RallyEventKind.HUMANOID_A_NET)
        for humanoid in index.humanoid_b_geoms:
            assert_pair(humanoid, geom, RallyEventKind.HUMANOID_B_NET)
    for racket in index.racket_a_geoms:
        assert_pair(index.ball_geom, racket, RallyEventKind.BALL_RACKET_A)
    for racket in index.racket_b_geoms:
        assert_pair(index.ball_geom, racket, RallyEventKind.BALL_RACKET_B)
    for humanoid in index.humanoid_a_geoms:
        assert_pair(index.ball_geom, humanoid, RallyEventKind.BALL_HUMANOID_A)
    for humanoid in index.humanoid_b_geoms:
        assert_pair(index.ball_geom, humanoid, RallyEventKind.BALL_HUMANOID_B)


def test_home_pose_foot_contacts_do_not_create_tennis_events():
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    batch = _sample_steps(sampler, model, data, control_step=0, count=1)
    assert not batch.events
    assert not batch.contact_peaks


def test_continuous_court_contact_crossing_center_does_not_create_second_bounce():
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-1.0, 0.0, 1.0))
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)
    sampler.begin_control_step(0)
    for substep, x in enumerate((-0.2, 0.2)):
        _set_ball_state(model, data, position=(x, 0.0, 0.03))
        sampler.sample_substep(data, control_substep=substep)
    batch = sampler.end_control_step()
    court_events = [
        event for event in batch.events if event.kind in {
            RallyEventKind.BALL_COURT_A,
            RallyEventKind.BALL_COURT_B,
        }
    ]
    assert [event.kind for event in court_events] == [
        RallyEventKind.BALL_COURT_A
    ]


def test_reset_rejects_side_that_contradicts_ball_position():
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(2.0, 0.0, 1.0))
    sampler = SubstepTennisEventSampler(model)
    with pytest.raises(ValueError, match="contradicts"):
        sampler.reset(data, ball_side=CourtSide.A)


def test_event_batch_is_picklable_for_vectorized_environment_transport():
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    batch = _sample_steps(sampler, model, data, control_step=0, count=1)
    restored = pickle.loads(pickle.dumps(batch))
    assert restored == batch


def test_physics_batch_feeds_directly_into_pure_rally_machine():
    model, data = _fresh_model()
    _set_ball_state(model, data, position=(-1.0, 0.0, 1.2))
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data, ball_side=CourtSide.A)
    _set_ball_state(model, data, position=(1.0, 0.0, 1.2))
    sampler.begin_control_step(0)
    sampler.sample_substep(data, control_substep=0)
    batch = sampler.end_control_step()

    machine = RallyStateMachine(serving_side=CourtSide.A)
    transition = machine.advance(
        batch.events,
        contact_peaks=batch.contact_peaks,
    )
    assert transition.after.phase is RallyPhase.AWAITING_RETURN
    assert transition.after.ball_side is CourtSide.B
    assert transition.after.net_crossing_count == 1


def test_control_step_buffer_must_be_opened_and_drained_exactly_once():
    model, data = _fresh_model()
    sampler = SubstepTennisEventSampler(model)
    sampler.reset(data)
    with pytest.raises(RuntimeError, match="begin_control_step"):
        sampler.sample_substep(data, control_substep=0)
    sampler.begin_control_step(0)
    with pytest.raises(RuntimeError, match="previous control step"):
        sampler.begin_control_step(1)
    sampler.sample_substep(data, control_substep=0)
    sampler.end_control_step()
    with pytest.raises(RuntimeError, match="no open control step"):
        sampler.end_control_step()
