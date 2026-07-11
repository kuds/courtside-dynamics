"""Phase 1 physics tests for the cooperative humanoid-tennis scene."""

from __future__ import annotations

import math
from html import escape

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._tennis_physics import (
    AIR_DENSITY,
    BALL_CROSS_SECTION,
    BALL_DRAG_COEFFICIENT,
    BALL_MASS,
    BALL_RADIUS,
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
    COURT_LENGTH,
    COURT_WIDTH,
    NET_CENTER_HEIGHT,
    NET_POST_HEIGHT,
    NET_POST_Y,
    ball_drag_force,
    is_in_bounds,
    net_height_at,
)
from courtside_dynamics.envs.robot_models import initialize_humanoid_tennis_home


def _fresh_model():
    model = mujoco.MjModel.from_xml_path(asset_path("humanoid_tennis.xml"))
    data = mujoco.MjData(model)
    initialize_humanoid_tennis_home(model, data)
    return model, data


def _set_ball_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    position: tuple[float, float, float],
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    dofadr = int(joint.dofadr[0])
    data.qpos[qposadr : qposadr + 3] = position
    data.qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dofadr : dofadr + 6] = 0.0
    data.qvel[dofadr : dofadr + 3] = linear_velocity
    mujoco.mj_forward(model, data)


def _has_active_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a: str,
    geom_b: str,
) -> bool:
    expected = {int(model.geom(geom_a).id), int(model.geom(geom_b).id)}
    for index in range(data.ncon):
        contact = data.contact[index]
        actual = {int(contact.geom1), int(contact.geom2)}
        if int(contact.efc_address) >= 0 and actual == expected:
            return True
    return False


def _masks_allow_contact(geom_a, geom_b) -> bool:
    return bool(
        int(geom_a.contype[0]) & int(geom_b.conaffinity[0])
        or int(geom_b.contype[0]) & int(geom_a.conaffinity[0])
    )


def _fixed_racket_rebound(speed: float) -> tuple[bool, float]:
    racket_path = escape(asset_path("tennis_racket.xml"), quote=True)
    xml = f"""
    <mujoco model="fixed_racket_rebound">
        <compiler angle="degree" />
        <option timestep="0.001" integrator="RK4" gravity="0 0 0" />
        <asset>
            <model name="racket_model" file="{racket_path}" />
        </asset>
        <worldbody>
            <attach model="racket_model" body="tennis_racket" prefix="fixture_" />
            <body name="fixture_ball" pos="-0.3 0 0.49">
                <freejoint name="fixture_ball_free" />
                <geom name="fixture_ball_geom" type="sphere" size="0.0335"
                      mass="0.0577" shellinertia="true" contype="2"
                      conaffinity="29" priority="1" solref="0.02 0.075" />
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dofadr = int(model.joint("fixture_ball_free").dofadr[0])
    data.qvel[dofadr] = speed
    mujoco.mj_forward(model, data)

    saw_stringbed = False
    for _ in range(1_000):
        mujoco.mj_step(model, data)
        touching = _has_active_contact(
            model,
            data,
            "fixture_ball_geom",
            "fixture_racket_stringbed",
        )
        saw_stringbed |= touching
        if saw_stringbed and not touching:
            return True, -float(data.qvel[dofadr])
    return saw_stringbed, -float(data.qvel[dofadr])


def test_scene_constructs_with_two_g1s_rackets_and_ball():
    model, data = _fresh_model()
    assert model.nq == 79  # 2 * (free-base 7 + 29 hinges) + ball free joint.
    assert model.nv == 76  # 2 * (free-base 6 + 29 hinges) + ball velocity.
    assert model.nu == 58
    assert data.qpos.shape == (79,)


def test_regulation_court_dimensions_and_semantic_regions():
    model, _ = _fresh_model()
    play_area = model.site("court_play_area")
    assert 2.0 * float(play_area.size[0]) == pytest.approx(COURT_LENGTH)
    assert 2.0 * float(play_area.size[1]) == pytest.approx(COURT_WIDTH)

    side_a = model.site("court_side_a")
    side_b = model.site("court_side_b")
    assert float(side_a.pos[0] + side_a.size[0]) == pytest.approx(0.0)
    assert float(side_b.pos[0] - side_b.size[0]) == pytest.approx(0.0)

    for name in (
        "baseline_a",
        "baseline_b",
        "sideline_negative",
        "sideline_positive",
        "out_of_bounds_back_a",
        "out_of_bounds_back_b",
        "out_of_bounds_sideline_negative",
        "out_of_bounds_sideline_positive",
    ):
        assert model.site(name).id >= 0

    surface = model.geom("court_surface")
    assert float(surface.pos[2] + surface.size[2]) == pytest.approx(0.0)


def test_visible_line_sites_lock_to_regulation_outer_edges():
    model, _ = _fresh_model()
    baseline_a = model.site("baseline_a")
    baseline_b = model.site("baseline_b")
    sideline_negative = model.site("sideline_negative")
    sideline_positive = model.site("sideline_positive")

    assert float(baseline_a.pos[0] - baseline_a.size[0]) == pytest.approx(
        -COURT_HALF_LENGTH
    )
    assert float(baseline_b.pos[0] + baseline_b.size[0]) == pytest.approx(
        COURT_HALF_LENGTH
    )
    assert float(sideline_negative.pos[1] - sideline_negative.size[1]) == pytest.approx(
        -COURT_HALF_WIDTH
    )
    assert float(sideline_positive.pos[1] + sideline_positive.size[1]) == pytest.approx(
        COURT_HALF_WIDTH
    )
    assert all(
        int(site.group[0]) == 0
        for site in (
            baseline_a,
            baseline_b,
            sideline_negative,
            sideline_positive,
        )
    )


def test_net_center_posts_and_height_profile():
    model, _ = _fresh_model()
    assert float(model.site("net_center_height").pos[2]) == pytest.approx(
        NET_CENTER_HEIGHT
    )
    assert float(model.site("net_post_negative_height").pos[2]) == pytest.approx(
        NET_POST_HEIGHT
    )
    assert float(model.site("net_post_positive_height").pos[2]) == pytest.approx(
        NET_POST_HEIGHT
    )
    assert abs(float(model.geom("net_post_negative").pos[1])) == pytest.approx(
        NET_POST_Y
    )
    assert abs(float(model.geom("net_post_positive").pos[1])) == pytest.approx(
        NET_POST_Y
    )

    assert net_height_at(0.0) == pytest.approx(NET_CENTER_HEIGHT)
    assert net_height_at(NET_POST_Y) == pytest.approx(NET_POST_HEIGHT)
    assert net_height_at(-NET_POST_Y) == pytest.approx(NET_POST_HEIGHT)
    assert net_height_at(NET_POST_Y / 2.0) == pytest.approx(
        (NET_CENTER_HEIGHT + NET_POST_HEIGHT) / 2.0
    )


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 0.0),
        (-COURT_HALF_LENGTH, 0.0),
        (COURT_HALF_LENGTH, 0.0),
        (0.0, -COURT_HALF_WIDTH),
        (0.0, COURT_HALF_WIDTH),
        (-COURT_HALF_LENGTH, -COURT_HALF_WIDTH),
        (COURT_HALF_LENGTH, COURT_HALF_WIDTH),
    ],
)
def test_contact_point_on_any_boundary_line_is_in_bounds(x, y):
    assert is_in_bounds(x, y)


@pytest.mark.parametrize(
    "x,y",
    [
        (COURT_HALF_LENGTH + 1e-5, 0.0),
        (-COURT_HALF_LENGTH - 1e-5, 0.0),
        (0.0, COURT_HALF_WIDTH + 1e-5),
        (0.0, -COURT_HALF_WIDTH - 1e-5),
        (COURT_HALF_LENGTH + 1e-5, COURT_HALF_WIDTH + 1e-5),
    ],
)
def test_contact_point_beyond_outer_line_edge_is_out(x, y):
    assert not is_in_bounds(x, y)


def test_ball_uses_regulation_scale_mass_and_radius():
    model, _ = _fresh_model()
    ball_geom = model.geom("ball_geom")
    ball_body = model.body("ball")
    assert float(ball_geom.size[0]) == pytest.approx(BALL_RADIUS)
    assert float(ball_body.mass[0]) == pytest.approx(BALL_MASS)
    expected_shell_inertia = 2.0 / 3.0 * BALL_MASS * BALL_RADIUS**2
    assert ball_body.inertia == pytest.approx(np.full(3, expected_shell_inertia))


def test_quadratic_ball_drag_is_finite_and_opposes_motion():
    velocity = np.array([12.0, -3.0, 4.0])
    drag = ball_drag_force(velocity)
    expected_magnitude = (
        0.5
        * AIR_DENSITY
        * BALL_DRAG_COEFFICIENT
        * BALL_CROSS_SECTION
        * np.linalg.norm(velocity) ** 2
    )
    assert np.isfinite(drag).all()
    assert float(np.dot(drag, velocity)) < 0.0
    assert np.linalg.norm(drag) == pytest.approx(expected_magnitude)
    assert ball_drag_force(np.zeros(3)) == pytest.approx(np.zeros(3))


def test_collision_masks_allow_required_phase1_pairs():
    court_model, _ = _fresh_model()
    racket_model = mujoco.MjModel.from_xml_path(asset_path("tennis_racket.xml"))
    ball = court_model.geom("ball_geom")
    court = court_model.geom("court_surface")
    net = court_model.geom("net_cloth_negative")
    robot = court_model.geom("player_a_torso_collision")
    racket = racket_model.geom("racket_stringbed")

    assert _masks_allow_contact(ball, court)
    assert _masks_allow_contact(ball, net)
    assert _masks_allow_contact(ball, racket)
    assert _masks_allow_contact(racket, court)
    assert _masks_allow_contact(racket, net)
    assert _masks_allow_contact(robot, court)
    assert _masks_allow_contact(robot, net)
    assert _masks_allow_contact(robot, ball)
    assert not _masks_allow_contact(robot, racket)


def test_racket_prefab_is_physical_and_rigid():
    model = mujoco.MjModel.from_xml_path(asset_path("tennis_racket.xml"))
    assert model.nq == 0
    assert model.nv == 0
    assert float(model.body("tennis_racket").mass[0]) == pytest.approx(0.30)

    stringbed = model.geom("racket_stringbed")
    assert int(stringbed.contype[0]) != 0
    assert int(stringbed.conaffinity[0]) != 0
    assert float(stringbed.size[0]) > 0.0
    assert float(stringbed.size[1]) == pytest.approx(0.13)
    assert float(stringbed.size[2]) == pytest.approx(0.17)
    assert float(model.site("racket_center").pos[2]) == pytest.approx(0.49)


@pytest.mark.parametrize("impact_speed", [10.0, 20.0, 30.0, 40.0])
def test_fixed_racket_rebounds_ball_without_injecting_energy(impact_speed):
    saw_stringbed, rebound_speed = _fixed_racket_rebound(impact_speed)
    assert saw_stringbed, f"no stringbed contact at {impact_speed} m/s"
    rebound_ratio = rebound_speed / impact_speed
    assert 0.55 <= rebound_ratio <= 0.9, (
        f"racket rebound ratio {rebound_ratio:.3f} is implausible at {impact_speed} m/s"
    )


def test_type2_ball_drop_has_plausible_hard_court_rebound():
    """A 2.54 m drop should rebound in the regulation Type 2 window."""
    model, data = _fresh_model()
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    dofadr = int(joint.dofadr[0])
    ball_body_id = int(model.body("ball").id)
    _set_ball_state(
        model,
        data,
        position=(2.0, 0.0, 2.54 + BALL_RADIUS),
    )

    bounced = False
    rebound_apex = None
    previous_vz = float(data.qvel[dofadr + 2])
    for _ in range(8_000):
        data.xfrc_applied[ball_body_id, :3] = ball_drag_force(
            data.qvel[dofadr : dofadr + 3]
        )
        mujoco.mj_step(model, data)
        vz = float(data.qvel[dofadr + 2])
        if not bounced and previous_vz < 0.0 <= vz:
            bounced = True
        elif bounced and previous_vz > 0.0 >= vz:
            rebound_apex = float(data.qpos[qposadr + 2]) - BALL_RADIUS
            break
        previous_vz = vz

    assert rebound_apex is not None, "ball never completed a court rebound"
    assert 1.35 <= rebound_apex <= 1.47, (
        f"Type 2 rebound apex {rebound_apex:.3f} m outside 1.35-1.47 m"
    )
    effective_restitution = math.sqrt(rebound_apex / 2.54)
    assert 0.72 <= effective_restitution <= 0.77


def test_successive_court_rebounds_do_not_inject_energy():
    model, data = _fresh_model()
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    dofadr = int(joint.dofadr[0])
    ball_body_id = int(model.body("ball").id)
    _set_ball_state(
        model,
        data,
        position=(2.0, 0.0, 2.54 + BALL_RADIUS),
    )

    rebound_apices: list[float] = []
    rising_after_bounce = False
    previous_vz = float(data.qvel[dofadr + 2])
    for _ in range(20_000):
        data.xfrc_applied[ball_body_id, :3] = ball_drag_force(
            data.qvel[dofadr : dofadr + 3]
        )
        mujoco.mj_step(model, data)
        vz = float(data.qvel[dofadr + 2])
        if previous_vz < 0.0 <= vz:
            rising_after_bounce = True
        elif rising_after_bounce and previous_vz > 0.0 >= vz:
            rebound_apices.append(float(data.qpos[qposadr + 2]) - BALL_RADIUS)
            rising_after_bounce = False
            if len(rebound_apices) == 4:
                break
        previous_vz = vz

    assert len(rebound_apices) == 4
    assert all(
        later < earlier
        for earlier, later in zip(
            rebound_apices[:-1],
            rebound_apices[1:],
            strict=True,
        )
    )


def test_oblique_court_bounce_reduces_slip_and_generates_spin():
    model, data = _fresh_model()
    joint = model.joint("ball_free")
    dofadr = int(joint.dofadr[0])
    ball_body_id = int(model.body("ball").id)
    initial_horizontal_speed = 10.0
    _set_ball_state(
        model,
        data,
        position=(2.0, 0.0, 0.5),
        linear_velocity=(initial_horizontal_speed, 0.0, -5.0),
    )

    touched_court = False
    for _ in range(3_000):
        data.xfrc_applied[ball_body_id, :3] = ball_drag_force(
            data.qvel[dofadr : dofadr + 3]
        )
        mujoco.mj_step(model, data)
        touching = _has_active_contact(
            model,
            data,
            "ball_geom",
            "court_surface",
        )
        touched_court |= touching
        if touched_court and not touching and float(data.qvel[dofadr + 2]) > 0.0:
            break

    assert touched_court
    assert 0.0 < float(data.qvel[dofadr]) < initial_horizontal_speed
    assert float(data.qvel[dofadr + 2]) > 0.0
    assert float(data.qvel[dofadr + 4]) > 0.0
    assert np.isfinite(data.qvel).all()


@pytest.mark.parametrize("impact_speed", [20.0, 40.0, 60.0])
@pytest.mark.parametrize(
    "lateral_position",
    [0.0, NET_POST_Y / 2.0, NET_POST_Y],
)
def test_ball_net_contact_is_physically_detectable(
    impact_speed,
    lateral_position,
):
    model, data = _fresh_model()
    _set_ball_state(
        model,
        data,
        position=(-0.25, lateral_position, 0.5),
        linear_velocity=(impact_speed, 0.0, 0.0),
    )

    saw_ball_net = False
    for _ in range(100):
        mujoco.mj_step(model, data)
        if any(
            _has_active_contact(model, data, "ball_geom", net_name)
            for net_name in (
                "net_cloth_negative",
                "net_cloth_positive",
                "net_top_cable_negative",
                "net_top_cable_positive",
                "net_center_strap",
                "net_post_negative",
                "net_post_positive",
            )
        ):
            saw_ball_net = True
            break

    assert saw_ball_net, "high-speed ball crossed the physical net undetected"


@pytest.mark.parametrize(
    "lateral_position",
    [0.0, NET_POST_Y / 2.0, NET_POST_Y],
)
def test_ball_above_net_crosses_without_contact(lateral_position):
    model, data = _fresh_model()
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    _set_ball_state(
        model,
        data,
        position=(
            -0.25,
            lateral_position,
            net_height_at(lateral_position) + BALL_RADIUS + 0.02,
        ),
        linear_velocity=(20.0, 0.0, 0.0),
    )

    saw_net_contact = False
    crossed = False
    for _ in range(100):
        mujoco.mj_step(model, data)
        saw_net_contact |= any(
            _has_active_contact(model, data, "ball_geom", net_name)
            for net_name in (
                "net_cloth_negative",
                "net_cloth_positive",
                "net_top_cable_negative",
                "net_top_cable_positive",
                "net_center_strap",
                "net_post_negative",
                "net_post_positive",
            )
        )
        if float(data.qpos[qposadr]) > BALL_RADIUS:
            crossed = True
            break

    assert crossed
    assert not saw_net_contact
