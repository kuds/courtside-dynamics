"""Robot-registry and two-G1 Phase 1 integration tests."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._tennis_physics import (
    COLLISION_BALL,
    COLLISION_COURT,
    COLLISION_NET,
    COLLISION_RACKET,
    COLLISION_ROBOT,
)
from courtside_dynamics.envs.robot_models import (
    ROBOT_MODELS,
    SUPPORTED_ROBOT_MODELS,
    UNITREE_G1,
    UNITREE_G1_ACTION_LAYOUT,
    UNITREE_G1_ACTUATOR_NAMES,
    get_robot_model_spec,
    initialize_humanoid_tennis_home,
)

ROBOT_PREFIXES = ("player_a_", "player_b_")


def _composite():
    model = mujoco.MjModel.from_xml_path(asset_path("humanoid_tennis.xml"))
    data = mujoco.MjData(model)
    initialize_humanoid_tennis_home(model, data)
    return model, data


def _has_active_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a: str,
    geom_b: str,
) -> bool:
    expected = {int(model.geom(geom_a).id), int(model.geom(geom_b).id)}
    return any(
        int(data.contact[index].efc_address) >= 0
        and {
            int(data.contact[index].geom1),
            int(data.contact[index].geom2),
        }
        == expected
        for index in range(data.ncon)
    )


def _active_contact_categories(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> list[frozenset[int]]:
    categories: list[frozenset[int]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if int(contact.efc_address) < 0:
            continue
        categories.append(
            frozenset(
                {
                    int(model.geom(int(contact.geom1)).contype[0]),
                    int(model.geom(int(contact.geom2)).contype[0]),
                }
            )
        )
    return categories


def _object_names(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    count: int,
) -> list[str]:
    return [
        name
        for object_id in range(count)
        if (
            name := mujoco.mj_id2name(
                model,
                object_type,
                object_id,
            )
        )
        is not None
    ]


def _set_ball_toward_stringbed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    player: str,
    *,
    speed: float = 20.0,
) -> None:
    stringbed = model.geom(f"{player}_racket_racket_stringbed")
    center = data.geom_xpos[stringbed.id].copy()
    normal = data.geom_xmat[stringbed.id].reshape(3, 3)[:, 0].copy()
    ball_joint = model.joint("ball_free")
    qposadr = int(ball_joint.qposadr[0])
    dofadr = int(ball_joint.dofadr[0])
    data.qpos[qposadr : qposadr + 3] = center - 0.2 * normal
    data.qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dofadr : dofadr + 6] = 0.0
    data.qvel[dofadr : dofadr + 3] = speed * normal
    mujoco.mj_forward(model, data)


def test_unitree_g1_is_the_only_complete_v1_registry_entry():
    assert SUPPORTED_ROBOT_MODELS == ("unitree_g1",)
    assert tuple(ROBOT_MODELS) == SUPPORTED_ROBOT_MODELS
    assert get_robot_model_spec() is UNITREE_G1
    assert get_robot_model_spec("unitree_g1") is UNITREE_G1

    with pytest.raises(ValueError, match="Unsupported robot_model='atlas'"):
        get_robot_model_spec("atlas")
    with pytest.raises(TypeError):
        ROBOT_MODELS["atlas"] = UNITREE_G1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        UNITREE_G1.nu = 30  # type: ignore[misc]


def test_single_g1_adapter_matches_registry_and_transmissions():
    model = mujoco.MjModel.from_xml_path(asset_path(UNITREE_G1.mjcf_asset))
    assert (model.nq, model.nv, model.nu) == (
        UNITREE_G1.nq,
        UNITREE_G1.nv,
        UNITREE_G1.nu,
    )

    actuator_names = tuple(model.actuator(index).name for index in range(model.nu))
    assert actuator_names == UNITREE_G1_ACTUATOR_NAMES
    assert len(UNITREE_G1.stand_joint_positions) == model.nu
    assert np.isfinite(model.actuator_ctrlrange).all()

    for actuator_id, actuator_name in enumerate(actuator_names):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        assert model.joint(joint_id).name == actuator_name

    assert model.body(UNITREE_G1.right_wrist_body).id >= 0
    assert model.body(UNITREE_G1.racket_body).id >= 0
    assert model.geom(UNITREE_G1.racket_stringbed_geom).id >= 0


def test_vendored_g1_manifest_license_and_checksums_resolve_offline():
    root = Path(asset_path("third_party/unitree_g1/PROVENANCE.md")).parent
    assert (root / "LICENSE").is_file()
    assert Path(asset_path("THIRD_PARTY_NOTICES.md")).is_file()

    manifest_lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 39
    for line in manifest_lines:
        expected, relative_path = line.split(maxsplit=1)
        payload = (root / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_composite_dimensions_namespaces_and_action_slices_are_locked():
    model, data = _composite()
    assert (model.nq, model.nv, model.nu) == (79, 76, 58)
    assert model.nkey == 0

    actuator_names = tuple(model.actuator(index).name for index in range(model.nu))
    assert actuator_names[:29] == tuple(
        f"player_a_{name}" for name in UNITREE_G1_ACTUATOR_NAMES
    )
    assert actuator_names[29:] == tuple(
        f"player_b_{name}" for name in UNITREE_G1_ACTUATOR_NAMES
    )

    layout = UNITREE_G1_ACTION_LAYOUT
    assert layout.player_a == slice(0, 29)
    assert layout.player_b == slice(29, 58)
    assert layout.player_a_right_arm == slice(22, 29)
    assert layout.player_b_right_arm == slice(51, 58)
    assert layout.total_size == model.nu

    data.ctrl[:] = 0.0
    data.ctrl[layout.player_a] = 0.25
    data.ctrl[layout.player_b] = -0.5
    assert data.ctrl[:29] == pytest.approx(np.full(29, 0.25))
    assert data.ctrl[29:] == pytest.approx(np.full(29, -0.5))


@pytest.mark.parametrize(
    "object_type,count_attribute",
    [
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
        (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
        (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
        (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
        (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
    ],
)
def test_player_object_suffix_sets_are_symmetric(object_type, count_attribute):
    model, _ = _composite()
    names = _object_names(model, object_type, int(getattr(model, count_attribute)))
    suffix_sets = [
        {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
        for prefix in ROBOT_PREFIXES
    ]
    assert suffix_sets[0]
    assert suffix_sets[0] == suffix_sets[1]
    assert len(names) == len(set(names))


def test_players_start_mirrored_facing_the_net():
    model, data = _composite()
    player_a = model.body("player_a_pelvis")
    player_b = model.body("player_b_pelvis")
    assert data.xpos[player_a.id, :2] == pytest.approx(-data.xpos[player_b.id, :2])
    assert data.xpos[player_a.id, 2] == pytest.approx(data.xpos[player_b.id, 2])

    rotation_a = data.xmat[player_a.id].reshape(3, 3)
    rotation_b = data.xmat[player_b.id].reshape(3, 3)
    assert rotation_a[:, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert rotation_b[:, 0] == pytest.approx([-1.0, 0.0, 0.0])


def test_all_composite_geoms_follow_the_collision_category_contract():
    model, _ = _composite()
    assert int(model.geom("court_surface").contype[0]) == COLLISION_COURT
    assert int(model.geom("ball_geom").contype[0]) == COLLISION_BALL
    assert int(model.geom("net_cloth_negative").contype[0]) == COLLISION_NET

    for geom_id in range(model.ngeom):
        geom = model.geom(geom_id)
        name = geom.name
        if not name.startswith(ROBOT_PREFIXES):
            continue
        if "_racket_" in name:
            assert int(geom.contype[0]) == COLLISION_RACKET
            assert int(geom.conaffinity[0]) == (
                COLLISION_COURT | COLLISION_BALL | COLLISION_NET
            )
        elif int(geom.group[0]) == 2:
            assert int(geom.contype[0]) == 0
            assert int(geom.conaffinity[0]) == 0
        else:
            assert int(geom.contype[0]) == COLLISION_ROBOT
            assert int(geom.conaffinity[0]) == (
                COLLISION_COURT | COLLISION_BALL | COLLISION_NET
            )


@pytest.mark.parametrize("player", ["player_a", "player_b"])
def test_racket_is_a_rigid_right_wrist_descendant(player):
    model, data = _composite()
    wrist = model.body(f"{player}_right_wrist_yaw_link")
    mount = model.body(f"{player}_right_racket_mount")
    racket = model.body(f"{player}_racket_tennis_racket")
    assert int(model.body_parentid[mount.id]) == wrist.id
    assert int(model.body_parentid[racket.id]) == mount.id
    assert int(mount.jntnum[0]) == 0
    assert int(racket.jntnum[0]) == 0
    assert float(racket.mass[0]) == pytest.approx(0.30)

    local_mount_position = np.array([0.1, 0.0, 0.0])
    expected_position = (
        data.xpos[wrist.id] + data.xmat[wrist.id].reshape(3, 3) @ local_mount_position
    )
    assert data.xpos[mount.id] == pytest.approx(expected_position)

    shoulder = model.joint(f"{player}_right_shoulder_pitch_joint")
    data.qpos[int(shoulder.qposadr[0])] += 0.15
    mujoco.mj_forward(model, data)
    expected_position = (
        data.xpos[wrist.id] + data.xmat[wrist.id].reshape(3, 3) @ local_mount_position
    )
    assert data.xpos[mount.id] == pytest.approx(expected_position)


def test_home_pose_has_only_expected_robot_court_contacts():
    model, data = _composite()
    categories = _active_contact_categories(model, data)
    assert categories
    assert all(
        category == frozenset({COLLISION_COURT, COLLISION_ROBOT})
        for category in categories
    )


@pytest.mark.parametrize("player", ["player_a", "player_b"])
def test_ball_reaches_each_physical_racket_stringbed(player):
    model, data = _composite()
    _set_ball_toward_stringbed(model, data, player)
    stringbed_name = f"{player}_racket_racket_stringbed"

    for _ in range(100):
        mujoco.mj_step(model, data)
        if _has_active_contact(model, data, "ball_geom", stringbed_name):
            break
    else:
        pytest.fail(f"ball did not contact {stringbed_name}")


@pytest.mark.parametrize("player", ["player_a", "player_b"])
def test_racket_and_robot_net_contacts_are_physically_enabled(player):
    net_names = (
        "net_cloth_negative",
        "net_cloth_positive",
        "net_top_cable_negative",
        "net_top_cable_positive",
        "net_center_strap",
    )

    racket_model, racket_data = _composite()
    racket_geom = racket_model.geom(f"{player}_racket_racket_stringbed")
    root_joint = racket_model.joint(f"{player}_floating_base_joint")
    root_qposadr = int(root_joint.qposadr[0])
    racket_data.qpos[root_qposadr] -= racket_data.geom_xpos[racket_geom.id, 0]
    mujoco.mj_forward(racket_model, racket_data)
    assert any(
        _has_active_contact(
            racket_model,
            racket_data,
            f"{player}_racket_racket_stringbed",
            net_name,
        )
        for net_name in net_names
    )

    robot_model, robot_data = _composite()
    torso_geom = robot_model.geom(f"{player}_torso_collision")
    root_joint = robot_model.joint(f"{player}_floating_base_joint")
    root_qposadr = int(root_joint.qposadr[0])
    robot_data.qpos[root_qposadr] -= robot_data.geom_xpos[torso_geom.id, 0]
    mujoco.mj_forward(robot_model, robot_data)
    assert any(
        category == frozenset({COLLISION_NET, COLLISION_ROBOT})
        for category in _active_contact_categories(robot_model, robot_data)
    )


def test_short_pd_held_rollout_is_finite_symmetric_and_deterministic():
    model, data_a = _composite()
    data_b = mujoco.MjData(model)
    initialize_humanoid_tennis_home(model, data_b)

    for _ in range(250):
        mujoco.mj_step(model, data_a)
        mujoco.mj_step(model, data_b)

    for values in (data_a.qpos, data_a.qvel, data_a.qacc, data_a.ctrl):
        assert np.isfinite(values).all()
    assert data_a.qpos == pytest.approx(data_b.qpos, abs=1e-12)
    assert data_a.qvel == pytest.approx(data_b.qvel, abs=1e-12)

    pelvis_a = data_a.xpos[model.body("player_a_pelvis").id]
    pelvis_b = data_a.xpos[model.body("player_b_pelvis").id]
    assert pelvis_a[:2] == pytest.approx(-pelvis_b[:2], abs=1e-9)
    assert pelvis_a[2] == pytest.approx(pelvis_b[2], abs=1e-9)
    assert not any(
        COLLISION_NET in category
        for category in _active_contact_categories(model, data_a)
    )
