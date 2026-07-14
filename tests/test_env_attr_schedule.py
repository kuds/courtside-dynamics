"""Focused tests for the reusable numeric environment-attribute schedule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from courtside_dynamics.callbacks.env_attr_schedule import (
    LinearEnvAttrScheduleCallback,
)


class _RecordingVecEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def set_attr(self, attr_name: str, value: float) -> None:
        self.calls.append((attr_name, value))


class _FakeModel:
    def __init__(self, training_env: _RecordingVecEnv) -> None:
        self._training_env = training_env

    def get_env(self) -> _RecordingVecEnv:
        return self._training_env


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
