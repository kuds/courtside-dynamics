"""Focused tests for the reusable numeric environment-attribute schedule."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env

from courtside_dynamics.callbacks.env_attr_schedule import (
    LinearEnvAttrScheduleCallback,
)
from tests._helpers import FakeGetEnvModel as _FakeModel


class _RecordingVecEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def env_method(
        self, method_name: str, attr_name: str, value: float, **kwargs
    ) -> list[bool]:
        # The callback must route writes through the wrapper-stack-aware
        # ``set_wrapper_attr`` with ``force=False``; ``VecEnv.set_attr``
        # (or ``force=True``) writes a shadow attribute on the outermost
        # wrapper instead (see the Monitor integration tests below).
        assert method_name == "set_wrapper_attr"
        assert kwargs == {"force": False}
        self.calls.append((attr_name, value))
        return [True]


def _callback(**overrides) -> LinearEnvAttrScheduleCallback:
    kwargs = {
        "attr_name": "recovery_reset_probability",
        "start_value": 0.6,
        "end_value": 0.1,
        "hold_until_timesteps": 100,
        "end_timesteps": 600,
    }
    kwargs.update(overrides)
    return LinearEnvAttrScheduleCallback(**kwargs)


def test_value_at_holds_interpolates_and_clamps() -> None:
    callback = _callback()

    assert callback.value_at(-1) == pytest.approx(0.6)
    assert callback.value_at(0) == pytest.approx(0.6)
    assert callback.value_at(100) == pytest.approx(0.6)
    assert callback.value_at(350) == pytest.approx(0.35)
    assert callback.value_at(600) == pytest.approx(0.1)
    assert callback.value_at(10_000) == pytest.approx(0.1)


def test_training_start_and_each_step_update_all_envs() -> None:
    callback = _callback()
    training_env = _RecordingVecEnv()
    callback.model = _FakeModel(training_env)  # type: ignore[assignment]

    callback.num_timesteps = 0
    callback._on_training_start()
    callback.num_timesteps = 350
    assert callback._on_step() is True
    callback.num_timesteps = 900
    assert callback._on_step() is True

    assert [name for name, _ in training_env.calls] == [
        "recovery_reset_probability",
        "recovery_reset_probability",
        "recovery_reset_probability",
    ]
    np.testing.assert_allclose(
        [value for _, value in training_env.calls],
        [0.6, 0.35, 0.1],
    )
    assert callback.last_value == pytest.approx(0.1)


def test_training_start_respects_existing_global_timestep() -> None:
    """A continued SB3 run must resume rather than restart the schedule."""
    callback = _callback()
    training_env = _RecordingVecEnv()
    callback.model = _FakeModel(training_env)  # type: ignore[assignment]
    callback.num_timesteps = 350

    callback._on_training_start()

    assert training_env.calls == [
        ("recovery_reset_probability", pytest.approx(0.35))
    ]


class _ScheduledEnv(gym.Env):
    """Minimal env exposing the scheduled attribute as a real property."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(
            -1.0, 1.0, (1,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, (1,), dtype=np.float32
        )
        self._recovery_reset_probability = 0.6
        self.setter_calls: list[float] = []

    @property
    def recovery_reset_probability(self) -> float:
        return self._recovery_reset_probability

    @recovery_reset_probability.setter
    def recovery_reset_probability(self, value: float) -> None:
        self._recovery_reset_probability = float(value)
        self.setter_calls.append(float(value))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_schedule_reaches_env_through_monitor_wrapper() -> None:
    """The scheduled value must reach the wrapped env's own attribute.

    Regression test for run 20260714_211111: ``VecEnv.set_attr`` does a
    plain ``setattr`` on each worker's outermost wrapper (the ``Monitor``
    that ``make_vec_env`` always adds), and gymnasium wrappers do not
    forward attribute writes -- the schedule became a shadow attribute on
    the Monitor and the run trained its entire budget at the start value.
    """
    vec_env = make_vec_env(_ScheduledEnv, n_envs=2)
    try:
        callback = _callback()
        callback.model = _FakeModel(vec_env)  # type: ignore[assignment]
        callback.num_timesteps = 350

        assert callback._on_step() is True

        for worker in vec_env.envs:  # Monitor-wrapped by make_vec_env
            inner = worker.unwrapped
            assert inner.recovery_reset_probability == pytest.approx(0.35)
            assert inner.setter_calls == [pytest.approx(0.35)]
            # The write must not be shadowed on the wrapper itself.
            assert "recovery_reset_probability" not in vars(worker)
    finally:
        vec_env.close()


def test_schedule_rejects_attribute_no_env_owns() -> None:
    """A typo'd ``attr_name`` must fail loudly at training start.

    With gymnasium's default ``force=True``, ``set_wrapper_attr`` would
    silently create the misnamed attribute on the outermost wrapper --
    recreating the exact silent-no-op failure mode the callback fix
    eliminates.
    """
    vec_env = make_vec_env(_ScheduledEnv, n_envs=1)
    try:
        callback = _callback(attr_name="recovery_reset_probablity")  # typo
        callback.model = _FakeModel(vec_env)  # type: ignore[assignment]
        callback.num_timesteps = 0

        with pytest.raises(
            AttributeError, match="recovery_reset_probablity"
        ):
            callback._on_training_start()

        # No shadow attribute may be left behind on any wrapper layer.
        assert "recovery_reset_probablity" not in vars(vec_env.envs[0])
        assert "recovery_reset_probablity" not in vars(
            vec_env.envs[0].unwrapped
        )
    finally:
        vec_env.close()


def test_schedule_metadata_is_json_serializable() -> None:
    callback = _callback()

    encoded = json.dumps(callback.schedule_metadata, sort_keys=True)

    assert '"schedule": "linear_after_hold"' in encoded
    assert callback.schedule_metadata["attr_name"] == (
        "recovery_reset_probability"
    )


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"attr_name": ""}, ValueError, "attr_name"),
        ({"attr_name": "   "}, ValueError, "attr_name"),
        ({"start_value": True}, TypeError, "start_value"),
        ({"end_value": np.inf}, ValueError, "end_value"),
        ({"hold_until_timesteps": True}, TypeError, "hold_until_timesteps"),
        ({"hold_until_timesteps": -1}, ValueError, "hold_until_timesteps"),
        ({"end_timesteps": 100}, ValueError, "end_timesteps"),
        ({"end_timesteps": 99}, ValueError, "end_timesteps"),
    ],
)
def test_invalid_schedule_configuration_is_rejected(overrides, error, match) -> None:
    with pytest.raises(error, match=match):
        _callback(**overrides)
