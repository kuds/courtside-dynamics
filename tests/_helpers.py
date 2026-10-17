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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._tennis_events import TennisStepEventBatch
from courtside_dynamics.envs.robot_models import (
    initialize_humanoid_tennis_home,
)
from courtside_dynamics.envs.tennis_rules import RallyEvent

if TYPE_CHECKING:
    from courtside_dynamics.training.train import TrainConfig


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


#: The tiny PaddleTennis runs' episode length, evaluation cadence and
#: budget (:func:`tiny_paddle_config`): four evaluations.
TINY_EPISODE_LEN = 300
TINY_EVAL_FREQ = 256
TINY_BUDGET = 1024


def tiny_paddle_config(algo: str, log_dir: Path) -> TrainConfig:
    """A PaddleTennis-shaped run: the recipe's paired selection evaluation
    (alternating serves, confirm_best, its keys and instruments) on
    300-step episodes, 2 evaluation episodes, 4 evaluations, small nets.

    The deltas are 0 so the reward tie-break decides between the
    untrained policies (their headline reads 0) and confirmations roll;
    the degenerate stop is off so all four evaluations run. Snapshots
    are on (the recipe's default) and checkpoints fall every 512 steps.
    """
    from courtside_dynamics.recipes import (
        build_train_config,
        make_env_fn,
        make_eval_env_fn,
    )

    short = {"episode_len": TINY_EPISODE_LEN}
    recipe_kwargs = build_train_config(
        "PaddleTennis", algo=algo, log_dir=str(log_dir), seed=0
    ).model_kwargs
    if algo == "SAC":
        model_kwargs = {
            **recipe_kwargs,
            "buffer_size": 5_000,
            "learning_starts": 64,
            "batch_size": 64,
            "train_freq": (16, "step"),
            "policy_kwargs": {"net_arch": [32, 32]},
        }
        extra: dict[str, Any] = {}
    else:
        model_kwargs = {
            **recipe_kwargs,
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 3e-3,
            "policy_kwargs": {**recipe_kwargs["policy_kwargs"], "net_arch": [32, 32]},
        }
        extra = {"min_policy_updates_before_stop": 0}
    return build_train_config(
        "PaddleTennis",
        algo=algo,
        log_dir=str(log_dir),
        seed=0,
        n_envs=2,
        total_timesteps=TINY_BUDGET,
        eval_freq=TINY_EVAL_FREQ,
        checkpoint_freq=512,
        n_eval_episodes=2,
        reward_eval_episodes=1,
        best_metric_min_delta=0.0,
        early_stop_degenerate_evals=0,
        checkpoint_diagnosis=None,
        video_freq=0,
        record_video=False,
        env_fn=make_env_fn("PaddleTennis", env_overrides=short),
        eval_env_fn=make_eval_env_fn("PaddleTennis", env_overrides=short),
        model_kwargs=model_kwargs,
        **extra,
    )
