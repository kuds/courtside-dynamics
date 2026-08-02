"""Shared MuJoCo/event test helpers.

These utilities used to be copy-pasted per test file and had already
drifted (one ``_set_ball_state`` spelled the kwarg ``linear_velocity``,
another ``velocity``; two ``_has_active_contact`` styles). One copy
lives here so the next environment's test files -- which exercise
exactly the same ball-state, contact-inspection, and event-injection
surface -- import instead of re-minting.

Import with local-underscore aliases to keep test bodies reading as
before::

    from tests._helpers import event_batch as _event_batch
"""
from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._tennis_events import TennisStepEventBatch
from courtside_dynamics.envs.robot_models import (
    initialize_humanoid_tennis_home,
)
from courtside_dynamics.envs.tennis_rules import RallyEvent


def event_batch(
    control_step: int,
    *events: RallyEvent,
) -> TennisStepEventBatch:
    """Wrap synthetic rally events as one sampled control-step batch."""
    return TennisStepEventBatch(
        control_step=control_step,
        substeps_sampled=1,
        events=events,
        contact_peaks={
            event.kind: event.peak_force
            for event in events
            if event.peak_force > 0.0
        },
        active_contact_latches=frozenset(),
    )


def inject_batches(
    monkeypatch: pytest.MonkeyPatch,
    env,
    batches: Iterator[TennisStepEventBatch],
) -> None:
    """Replace the env's physics step with a scripted event-batch feed."""

    def set_next_batch(_ctrl: np.ndarray, _n_frames: int) -> None:
        env._latest_event_batch = next(batches)

    monkeypatch.setattr(env, "_step_mujoco_simulation", set_next_batch)


def fresh_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the humanoid-tennis scene and pose both players at home."""
    model = mujoco.MjModel.from_xml_path(asset_path("humanoid_tennis.xml"))
    data = mujoco.MjData(model)
    initialize_humanoid_tennis_home(model, data)
    return model, data


def set_ball_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    position: np.ndarray | tuple[float, float, float],
    velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Teleport the free ball, zero its spin, and re-run ``mj_forward``."""
    joint = model.joint("ball_free")
    qposadr = int(joint.qposadr[0])
    dofadr = int(joint.dofadr[0])
    data.qpos[qposadr : qposadr + 3] = position
    data.qpos[qposadr + 3 : qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dofadr : dofadr + 6] = 0.0
    data.qvel[dofadr : dofadr + 3] = velocity
    mujoco.mj_forward(model, data)


def set_ball_toward_stringbed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    player: str,
    *,
    speed: float = 20.0,
) -> None:
    """Place the ball just off a player's stringbed, flying into it."""
    stringbed = model.geom(f"{player}_racket_racket_stringbed")
    center = data.geom_xpos[stringbed.id].copy()
    normal = data.geom_xmat[stringbed.id].reshape(3, 3)[:, 0].copy()
    set_ball_state(
        model,
        data,
        position=center - 0.2 * normal,
        velocity=speed * normal,
    )


def has_active_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_a: str,
    geom_b: str,
) -> bool:
    """Whether the solver currently carries an active a<->b contact."""
    expected = {int(model.geom(geom_a).id), int(model.geom(geom_b).id)}
    for index in range(data.ncon):
        contact = data.contact[index]
        actual = {int(contact.geom1), int(contact.geom2)}
        if int(contact.efc_address) >= 0 and actual == expected:
            return True
    return False


class FakeGetEnvModel:
    """The ``model.get_env()`` shape SB3 callbacks read a training env
    through -- just enough for callback unit tests."""

    def __init__(self, training_env) -> None:
        self._training_env = training_env

    def get_env(self):
        return self._training_env
