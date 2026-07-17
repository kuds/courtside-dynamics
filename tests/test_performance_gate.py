"""Tests for the performance-gated curriculum stage callback."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env

from courtside_dynamics.callbacks.performance_gate import (
    PerformanceGatedEnvStagesCallback,
)


class _StagedEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.serve_vy_max = 0.0
        self.serve_speed_jitter = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _FakeInfoEval:
    """Just enough of InfoDictEvalCallback for the gate to consume."""

    def __init__(self, eval_env) -> None:
        self.eval_env = eval_env
        self.completed_evals = 0
        self.last_metrics: dict[str, float] | None = None
        self.selection_resets = 0

    def reset_selection_state(self) -> None:
        self.selection_resets += 1

    def finish_eval(self, metrics: dict[str, float]) -> None:
        self.last_metrics = dict(metrics)
        self.completed_evals += 1


class _FakeModel:
    def __init__(self, training_env) -> None:
        self._training_env = training_env

    def get_env(self):
        return self._training_env


STAGES = (
    {"serve_vy_max": 1.1, "serve_speed_jitter": 0.2},
    {"serve_vy_max": 1.4, "serve_speed_jitter": 0.3},
    {"serve_vy_max": 2.0, "serve_speed_jitter": 0.5},
)


def _gate(train_env, info_eval, **overrides):
    kwargs = dict(
        stages=STAGES,
        metric_key="bounce_count_ep_mean",
        threshold=1.3,
        sustain_evals=2,
        info_eval=info_eval,
    )
    kwargs.update(overrides)
    callback = PerformanceGatedEnvStagesCallback(**kwargs)
    callback.model = _FakeModel(train_env)  # type: ignore[assignment]
    return callback


def _stage_values(vec_env, attr):
    return [getattr(worker.unwrapped, attr) for worker in vec_env.envs]


def test_gate_applies_stage_zero_and_advances_on_sustained_metric():
    train_env = make_vec_env(_StagedEnv, n_envs=2)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(train_env, info_eval)

        gate._on_training_start()
        assert gate.stage_index == 0
        assert _stage_values(train_env, "serve_vy_max") == [1.1, 1.1]
        assert _stage_values(eval_env, "serve_vy_max") == [1.1]

        # A miss resets the streak; sustained passes advance one stage.
        for metric, expected_stage in (
            (1.5, 0),  # pass 1/2
            (0.9, 0),  # miss -- streak resets
            (1.5, 0),  # pass 1/2 again
            (1.3, 1),  # pass 2/2 (threshold inclusive) -> advance
            (2.0, 1),
            (2.0, 2),  # advance to final stage
            (2.0, 2),  # ladder is exhausted; no further movement
            (2.0, 2),
        ):
            info_eval.finish_eval({"bounce_count_ep_mean": metric})
            assert gate._on_step() is True
            assert gate.stage_index == expected_stage, metric

        assert _stage_values(train_env, "serve_vy_max") == [2.0, 2.0]
        assert _stage_values(train_env, "serve_speed_jitter") == [0.5, 0.5]
        assert _stage_values(eval_env, "serve_vy_max") == [2.0]
        # Each advance forgets the previous stage's selection state:
        # scores across serve distributions are not comparable.
        assert info_eval.selection_resets == 2
    finally:
        train_env.close()
        eval_env.close()


def test_gate_ignores_steps_without_a_fresh_evaluation():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(train_env, info_eval, sustain_evals=1)
        gate._on_training_start()

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True
        assert gate.stage_index == 1
        # Steps between evaluations must not re-consume the same result.
        for _ in range(5):
            assert gate._on_step() is True
        assert gate.stage_index == 1
    finally:
        train_env.close()
        eval_env.close()


def test_gate_missing_metric_resets_streak():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(train_env, info_eval)
        gate._on_training_start()

        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()
        info_eval.finish_eval({})  # metric absent: cannot certify mastery
        gate._on_step()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()
        assert gate.stage_index == 0
    finally:
        train_env.close()
        eval_env.close()


def test_gate_rejects_attribute_no_env_owns():
    """force=False makes a typo'd stage attribute fail loudly instead of
    shadow-writing the Monitor wrapper -- the same regression class as
    run 20260714_211111's silent curriculum."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            stages=({"serve_vy_maximum": 1.1},),  # typo
        )
        with pytest.raises(AttributeError, match="serve_vy_maximum"):
            gate._on_training_start()
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"stages": ()}, ValueError, "stages"),
        ({"stages": ({},)}, ValueError, "at least one attribute"),
        ({"metric_key": ""}, ValueError, "metric_key"),
        ({"threshold": float("nan")}, ValueError, "threshold"),
        ({"sustain_evals": 0}, ValueError, "sustain_evals"),
        ({"sustain_evals": True}, ValueError, "sustain_evals"),
    ],
)
def test_gate_configuration_is_validated(overrides, error, match):
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        with pytest.raises(error, match=match):
            _gate(None, _FakeInfoEval(eval_env), **overrides)
    finally:
        eval_env.close()
