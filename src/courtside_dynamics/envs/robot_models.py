"""Robot-model contracts for the humanoid-tennis environments.

The registry is intentionally small in v1. A model is admitted only when its
licensed assets, actuator ordering, standing reference, collision adapter, and
racket mount are all available in the installed package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import mujoco


@dataclass(frozen=True)
class RobotModelSpec:
    """Immutable dimensions and names locked to a packaged robot adapter."""

    name: str
    mjcf_asset: str
    nq: int
    nv: int
    nu: int
    actuator_names: tuple[str, ...]
    stand_joint_positions: tuple[float, ...]
    right_wrist_body: str
    racket_body: str
    racket_stringbed_geom: str
    source_commit: str
    license_expression: str

    def __post_init__(self) -> None:
        """Reject incomplete registry entries at import time."""
        if self.nu != len(self.actuator_names):
            raise ValueError(
                f"{self.name}: nu={self.nu} but "
                f"{len(self.actuator_names)} actuator names were provided"
            )
        if self.nu != len(self.stand_joint_positions):
            raise ValueError(
                f"{self.name}: nu={self.nu} but "
                f"{len(self.stand_joint_positions)} stand targets were provided"
            )


@dataclass(frozen=True)
class CentralizedActionLayout:
    """Documented slices of the one centralized two-player action vector."""

    player_a: slice
    player_b: slice
    player_a_right_arm: slice
    player_b_right_arm: slice
    total_size: int


UNITREE_G1_ACTUATOR_NAMES: Final = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

UNITREE_G1_STAND_JOINT_POSITIONS: Final = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)
UNITREE_G1_STAND_BASE_HEIGHT: Final = 0.783675

UNITREE_G1: Final = RobotModelSpec(
    name="unitree_g1",
    mjcf_asset="robots/unitree_g1_tennis.xml",
    nq=36,
    nv=35,
    nu=29,
    actuator_names=UNITREE_G1_ACTUATOR_NAMES,
    stand_joint_positions=UNITREE_G1_STAND_JOINT_POSITIONS,
    right_wrist_body="right_wrist_yaw_link",
    racket_body="racket_tennis_racket",
    racket_stringbed_geom="racket_racket_stringbed",
    source_commit="71f066ad0be9cd271f7ed58c030243ef157af9f4",
    license_expression="BSD-3-Clause",
)

ROBOT_MODELS: Final[Mapping[str, RobotModelSpec]] = MappingProxyType(
    {UNITREE_G1.name: UNITREE_G1}
)
SUPPORTED_ROBOT_MODELS: Final = tuple(ROBOT_MODELS)

# The right shoulder-through-wrist controls occupy indices 22:29 in each G1.
UNITREE_G1_ACTION_LAYOUT: Final = CentralizedActionLayout(
    player_a=slice(0, 29),
    player_b=slice(29, 58),
    player_a_right_arm=slice(22, 29),
    player_b_right_arm=slice(51, 58),
    total_size=58,
)


def get_robot_model_spec(robot_model: str = "unitree_g1") -> RobotModelSpec:
    """Return a complete robot contract or reject unsupported promises."""
    try:
        return ROBOT_MODELS[robot_model]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_ROBOT_MODELS)
        raise ValueError(
            f"Unsupported robot_model={robot_model!r}; supported: {supported}"
        ) from error


def initialize_humanoid_tennis_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_model: str = "unitree_g1",
) -> None:
    """Reset the two-player composite to its deterministic standing reference.

    The player root x/y positions and orientations come from ``model.qpos0``
    so court spawn transforms remain an XML concern. Scalar joint targets and
    controls come from the registry's locked actuator order.
    """
    spec = get_robot_model_spec(robot_model)
    expected_nu = 2 * spec.nu
    if model.nu != expected_nu:
        raise ValueError(
            f"Composite has nu={model.nu}; expected {expected_nu} for two "
            f"{robot_model} players"
        )

    mujoco.mj_resetData(model, data)
    for player_index, player in enumerate(("player_a", "player_b")):
        base_joint = model.joint(f"{player}_floating_base_joint")
        base_qposadr = int(base_joint.qposadr[0])
        data.qpos[base_qposadr + 2] = UNITREE_G1_STAND_BASE_HEIGHT

        for joint_name, target in zip(
            spec.actuator_names,
            spec.stand_joint_positions,
            strict=True,
        ):
            joint = model.joint(f"{player}_{joint_name}")
            data.qpos[int(joint.qposadr[0])] = target

        control_start = player_index * spec.nu
        data.ctrl[control_start : control_start + spec.nu] = spec.stand_joint_positions

    mujoco.mj_forward(model, data)


__all__ = [
    "CentralizedActionLayout",
    "ROBOT_MODELS",
    "RobotModelSpec",
    "SUPPORTED_ROBOT_MODELS",
    "UNITREE_G1",
    "UNITREE_G1_ACTION_LAYOUT",
    "UNITREE_G1_ACTUATOR_NAMES",
    "UNITREE_G1_STAND_BASE_HEIGHT",
    "UNITREE_G1_STAND_JOINT_POSITIONS",
    "get_robot_model_spec",
    "initialize_humanoid_tennis_home",
]
