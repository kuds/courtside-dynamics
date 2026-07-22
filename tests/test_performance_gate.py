"""Tests for the performance-gated curriculum stage callback."""

from __future__ import annotations

import json
import os

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
        self.context_metrics: dict[str, float] = {}
        # (destination, selection_resets at call time): the second slot
        # pins the archive-before-reset ordering the gate must honor.
        self.archives: list[tuple[str, int]] = []

    def reset_selection_state(self) -> None:
        self.selection_resets += 1

    def archive_best(self, destination_dir: str) -> dict[str, int]:
        os.makedirs(destination_dir, exist_ok=True)
        meta = {"timestep": self.completed_evals}
        with open(
            os.path.join(destination_dir, "best_model_meta.json"), "w"
        ) as stream:
            json.dump(meta, stream)
        self.archives.append((destination_dir, self.selection_resets))
        return meta

    def set_context_metric(self, name: str, value: float) -> None:
        self.context_metrics[name] = float(value)

    def finish_eval(self, metrics: dict[str, float]) -> None:
        self.last_metrics = dict(metrics)
        self.completed_evals += 1


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class _FakeModel:
    def __init__(self, training_env) -> None:
        self._training_env = training_env

    def get_env(self):
        return self._training_env


class _FakeOffPolicyModel(_FakeModel):
    """Adds the SAC-shaped attributes the warm-up package touches."""

    def __init__(self, training_env) -> None:
        super().__init__(training_env)
        self.gradient_steps = -1
        self.replay_buffer = _FakeReplayBuffer()
        self.num_timesteps = 0


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
        ({"stage_bests_dir": ""}, ValueError, "stage_bests_dir"),
        ({"stage_history_path": "  "}, ValueError, "stage_history_path"),
        ({"promotion_rule": "always"}, ValueError, "promotion_rule"),
        (
            {"advance_update_pause_steps": -1},
            ValueError,
            "advance_update_pause_steps",
        ),
        (
            {"advance_update_pause_steps": True},
            ValueError,
            "advance_update_pause_steps",
        ),
        (
            {"clear_replay_buffer_on_advance": 1},
            ValueError,
            "clear_replay_buffer_on_advance",
        ),
        (
            {"clear_replay_buffer_on_advance": True},
            ValueError,
            "requires.*advance_update_pause_steps",
        ),
    ],
)
def test_gate_configuration_is_validated(overrides, error, match):
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        with pytest.raises(error, match=match):
            _gate(None, _FakeInfoEval(eval_env), **overrides)
    finally:
        eval_env.close()


def test_gate_stamps_stage_context_on_the_driving_evaluator():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(train_env, info_eval, sustain_evals=1)
        gate._on_training_start()
        assert info_eval.context_metrics == {"curriculum_stage_index": 0.0}

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 1
        assert info_eval.context_metrics == {"curriculum_stage_index": 1.0}
    finally:
        train_env.close()
        eval_env.close()


def test_gate_window_mean_promotes_on_the_average_not_each_eval():
    """Run 20260721_004722's stage 2 cleared 3.0 four separate times but
    never twice consecutively; the window-mean rule promotes exactly that
    profile while a below-bar average still holds the stage."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            promotion_rule="window_mean",
            threshold=3.0,
            sustain_evals=2,
        )
        gate._on_training_start()

        for metric in (
            3.2,  # window not yet full
            2.5,  # mean 2.85 < 3.0: hold
            3.3,  # mean (2.5 + 3.3)/2 = 2.9 < 3.0: hold
        ):
            info_eval.finish_eval({"bounce_count_ep_mean": metric})
            gate._on_step()
            assert gate.stage_index == 0, metric

        # 3.3 is already in the window; 2.9 brings the mean to 3.1 --
        # promoted despite the second eval sitting below the bar, which
        # the consecutive rule would have refused forever.
        info_eval.finish_eval({"bounce_count_ep_mean": 2.9})
        gate._on_step()
        assert gate.stage_index == 1

        # The window restarts at the new stage: one strong eval alone
        # cannot promote again.
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 1
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 2
    finally:
        train_env.close()
        eval_env.close()


def test_gate_window_mean_missing_metric_clears_the_window():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            promotion_rule="window_mean",
            threshold=3.0,
            sustain_evals=2,
        )
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        info_eval.finish_eval({})  # absent metric: evidence window resets
        gate._on_step()
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 0
    finally:
        train_env.close()
        eval_env.close()


def test_gate_warmup_clears_buffer_and_pauses_updates_until_deadline():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=1_000,
            clear_replay_buffer_on_advance=True,
        )
        model = _FakeOffPolicyModel(train_env)
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()
        assert model.gradient_steps == -1

        gate.num_timesteps = 500
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 1
        assert model.replay_buffer.resets == 1
        assert model.gradient_steps == 0  # updates paused

        # Still inside the pause window: stays frozen.
        gate.num_timesteps = 1_400
        gate._on_step()
        assert model.gradient_steps == 0

        # Deadline reached: original gradient_steps restored.
        gate.num_timesteps = 1_500
        gate._on_step()
        assert model.gradient_steps == -1

        # The next advance re-arms the whole package.
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        gate._on_step()
        assert gate.stage_index == 2
        assert model.replay_buffer.resets == 2
        assert model.gradient_steps == 0
    finally:
        train_env.close()
        eval_env.close()


def test_gate_warmup_requires_capable_model_at_training_start():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            advance_update_pause_steps=1_000,
        )
        # _FakeModel has no gradient_steps: must fail before training.
        with pytest.raises(TypeError, match="gradient_steps"):
            gate._on_training_start()

        gate_clear = _gate(
            train_env,
            info_eval,
            advance_update_pause_steps=1_000,
            clear_replay_buffer_on_advance=True,
        )
        model = _FakeOffPolicyModel(train_env)
        model.replay_buffer = None  # type: ignore[assignment]
        gate_clear.model = model  # type: ignore[assignment]
        with pytest.raises(TypeError, match="replay_buffer"):
            gate_clear._on_training_start()
    finally:
        train_env.close()
        eval_env.close()


def test_gate_archives_stage_bests_and_writes_history(tmp_path):
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            stage_bests_dir=str(tmp_path / "model" / "stage_bests"),
            stage_history_path=str(
                tmp_path / "reports" / "curriculum_stages.json"
            ),
        )
        gate._on_training_start()

        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})  # promote 0->1
        gate._on_step()
        info_eval.finish_eval({"bounce_count_ep_mean": 0.5})  # stay at 1
        gate._on_step()
        gate._on_training_end()

        # Stage 0 archived on the advance, stage 1 at training end -- and
        # the advance-time archive ran BEFORE the selection reset, while
        # best_model.zip still held the departing stage's champion.
        assert [
            os.path.basename(dest) for dest, _ in info_eval.archives
        ] == ["stage_00", "stage_01"]
        assert info_eval.archives[0][1] == 0
        assert info_eval.selection_resets == 1

        history = json.loads(
            (tmp_path / "reports" / "curriculum_stages.json").read_text()
        )
        assert history["metric_key"] == "bounce_count_ep_mean"
        assert history["threshold"] == 1.3
        assert history["stage_count"] == len(STAGES)
        assert history["final_stage_index"] == 1
        rows = history["stages"]
        assert [row["stage_index"] for row in rows] == [0, 1]
        assert rows[0]["promoted"] is True
        assert rows[0]["evals"] == 1
        assert rows[0]["streak"] == 1
        # The evidence window is kept under the consecutive rule too
        # (deque maxlen == sustain_evals == 1 here).
        assert rows[0]["promotion_window"] == [2.0]
        assert rows[0]["stage"] == dict(STAGES[0])
        assert rows[0]["best"] == {"timestep": 1}
        assert rows[1]["promoted"] is False
        assert rows[1]["evals"] == 1
        assert rows[1]["best"] == {"timestep": 2}
    finally:
        train_env.close()
        eval_env.close()


def test_gate_without_archive_paths_stays_inert(tmp_path):
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(train_env, info_eval, sustain_evals=1)
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()
        gate._on_training_end()
        assert info_eval.archives == []
        assert list(tmp_path.iterdir()) == []
    finally:
        train_env.close()
        eval_env.close()


def test_gate_history_records_window_mean_promotion_evidence(tmp_path):
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            promotion_rule="window_mean",
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        for metric in (1.1, 1.6):  # mean 1.35 >= 1.3 -> promote
            info_eval.finish_eval({"bounce_count_ep_mean": metric})
            gate._on_step()
        gate._on_training_end()

        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert rows[0]["promoted"] is True
        assert rows[0]["promotion_window"] == [1.1, 1.6]
    finally:
        train_env.close()
        eval_env.close()


def test_gate_skips_archiving_a_stage_with_zero_evaluations(tmp_path):
    """Training ending right after a promotion must not file the
    PREVIOUS stage's champion under the new stage's label: the on-disk
    best triple predates the stage whenever the stage saw no eval."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            stage_bests_dir=str(tmp_path / "stage_bests"),
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})  # promote 0->1
        gate._on_step()
        gate._on_training_end()  # zero evals at stage 1

        assert [
            os.path.basename(dest) for dest, _ in info_eval.archives
        ] == ["stage_00"]
        assert not (tmp_path / "stage_bests" / "stage_01").exists()
        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert rows[1]["evals"] == 0
        assert rows[1]["best"] is None
    finally:
        train_env.close()
        eval_env.close()


def test_gate_finalize_is_idempotent_and_covers_interrupts(tmp_path):
    """finalize() is the interrupt-salvage entry point; a later normal
    on_training_end must not append a duplicate final-stage row."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()

        gate.finalize()  # the interrupt path
        gate._on_training_end()  # a hypothetical later normal close
        gate.finalize()

        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert [row["stage_index"] for row in rows] == [0, 1]
    finally:
        train_env.close()
        eval_env.close()


def test_gate_history_survives_before_training_end(tmp_path):
    """Completed-stage rows are durable at advance time (atomic
    refresh), so a hard runtime death cannot lose them."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()  # advance closes stage 0 and refreshes the file

        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert [row["stage_index"] for row in rows] == [0]
        assert rows[0]["promoted"] is True
    finally:
        train_env.close()
        eval_env.close()


def test_gate_copies_run_config_into_stage_archives(tmp_path):
    """With config.json alongside the archived triple, a stage_NN dir is
    a valid legacy-flat-layout WarmStartConfig.source_run_dir."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        run_config = tmp_path / "config.json"
        run_config.write_text('{"train_config": {"algo": "SAC"}}\n')
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            stage_bests_dir=str(tmp_path / "stage_bests"),
            run_config_path=str(run_config),
        )
        gate._on_training_start()
        info_eval.finish_eval({"bounce_count_ep_mean": 2.0})
        gate._on_step()

        copied = tmp_path / "stage_bests" / "stage_00" / "config.json"
        assert copied.read_text() == run_config.read_text()
    finally:
        train_env.close()
        eval_env.close()
