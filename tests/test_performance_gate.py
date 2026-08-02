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
from tests._helpers import FakeGetEnvModel as _FakeModel


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
        self.last_confirmation_metrics: dict[str, float] | None = None
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

    def finish_eval(
        self,
        metrics: dict[str, float],
        confirmation: dict[str, float] | None = None,
    ) -> None:
        self.last_metrics = dict(metrics)
        # Mirrors InfoDictEvalCallback: the confirmation slot is cleared
        # on every evaluation, so a gate can never pool a stale batch.
        self.last_confirmation_metrics = (
            None if confirmation is None else dict(confirmation)
        )
        self.completed_evals += 1


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class _FakeOffPolicyModel(_FakeModel):
    """Adds the SAC-shaped attributes the warm-up package touches."""

    def __init__(self, training_env) -> None:
        super().__init__(training_env)
        self.gradient_steps = -1
        self.replay_buffer = _FakeReplayBuffer()
        self.num_timesteps = 0


class _FakeEntropyOptimizer:
    """Adam-shaped: the gate only clears accumulated moment state."""

    def __init__(self) -> None:
        self.state: dict[str, object] = {}


class _FakeAutoEntropyModel(_FakeOffPolicyModel):
    """SAC with an auto-tuned temperature (log_ent_coef + its optimizer).

    Shaped like the real thing in the way that matters here: SB3 keeps the
    *configured* ``ent_coef`` string on the model and the live temperature
    only in ``log_ent_coef``, so the two can disagree -- which is exactly
    the case on a warm-started continuation.
    """

    def __init__(
        self,
        training_env,
        init_log_ent_coef: float = 0.0,
        ent_coef: str = "auto",
    ) -> None:
        super().__init__(training_env)
        import torch as th

        self.log_ent_coef = th.tensor(
            [init_log_ent_coef], requires_grad=True
        )
        self.ent_coef_optimizer = _FakeEntropyOptimizer()
        self.ent_coef = ent_coef

    @property
    def alpha(self) -> float:
        """The live temperature, as ``train/ent_coef`` reports it."""
        return float(self.log_ent_coef.detach().exp().item())


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
        # The header records every optional gate lever at its resolved
        # value -- a budget-stopped run's history must show the budget
        # that stopped it (the 0.24.0 staleness pair was absent from
        # this payload for two releases).
        assert history["stage_eval_budget"] is None
        assert history["stage_eval_budget_action"] == "stop"
        assert history["entropy_reset_value"] is None
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


def test_entropy_reset_restores_initial_temperature_on_advance():
    """A collapsed alpha is restored, and its optimizer moments dropped.

    Run 20260721_004722 ended at ent_coef 9.2e-4 and never re-inflated
    after a promotion; the policy then met each new geometry nearly
    deterministic.
    """
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=10,
            reset_entropy_on_advance=True,
        )
        model = _FakeAutoEntropyModel(train_env, init_log_ent_coef=0.0)
        gate.model = model  # type: ignore[assignment]

        gate._on_training_start()
        assert gate._initial_log_ent_coef == pytest.approx(0.0)
        assert model.alpha == pytest.approx(1.0)

        # Simulate a long stage that decays the temperature to ~1e-3 and
        # leaves Adam moment state behind.
        model.log_ent_coef.detach().fill_(float(np.log(9.2e-4)))
        model.ent_coef_optimizer.state["stale"] = object()
        assert model.alpha == pytest.approx(9.2e-4, rel=1e-3)

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True

        assert gate.stage_index == 1
        assert model.alpha == pytest.approx(1.0)
        # Clearing the moments matters as much as the value: Adam would
        # otherwise push the restored coefficient straight back down.
        assert model.ent_coef_optimizer.state == {}
        assert gate._entropy_resets == 1
        # The restore must survive the update pause it is paired with.
        assert gate._update_pause_until == 10
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_keeps_gradients_flowing_through_log_ent_coef():
    """The reset writes through .detach(), so the leaf stays trainable."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=10,
            reset_entropy_on_advance=True,
        )
        model = _FakeAutoEntropyModel(
            train_env, init_log_ent_coef=-7.0, ent_coef="auto_0.02"
        )
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True

        # Writing through .detach() keeps the same trainable leaf, so the
        # temperature optimizer's param group stays valid afterwards.
        assert model.log_ent_coef.requires_grad is True
        assert model.log_ent_coef.is_leaf is True
        assert float(model.log_ent_coef.detach()[0]) == pytest.approx(
            float(np.log(0.02))
        )
        # And a gradient can still reach it.
        model.log_ent_coef.exp().sum().backward()
        assert model.log_ent_coef.grad is not None
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_requires_an_auto_tuned_coefficient():
    """A fixed float ent_coef has nothing to restore -- fail, never no-op."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            advance_update_pause_steps=10,
            reset_entropy_on_advance=True,
        )
        # No log_ent_coef / ent_coef_optimizer: SAC with ent_coef=0.02.
        gate.model = _FakeOffPolicyModel(train_env)  # type: ignore[assignment]
        with pytest.raises(TypeError, match="auto-tuned entropy coefficient"):
            gate._on_training_start()
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_rejects_non_boolean():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        with pytest.raises(
            ValueError, match="reset_entropy_on_advance must be a boolean"
        ):
            _gate(train_env, info_eval, reset_entropy_on_advance=1)
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_defaults_off_and_leaves_temperature_alone():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=10,
        )
        model = _FakeAutoEntropyModel(train_env, init_log_ent_coef=0.0)
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()
        assert gate._initial_log_ent_coef is None

        model.log_ent_coef.detach().fill_(float(np.log(9.2e-4)))
        model.ent_coef_optimizer.state["stale"] = object()
        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True

        assert gate.stage_index == 1
        assert model.alpha == pytest.approx(9.2e-4, rel=1e-3)
        assert "stale" in model.ent_coef_optimizer.state
        assert gate._entropy_resets == 0
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_against_a_real_sac_temperature_optimizer(tmp_path):
    """The reset must work on SB3's actual SAC internals, not just a fake.

    ``log_ent_coef`` is a leaf Tensor and ``ent_coef_optimizer`` is an Adam
    whose ``state`` is keyed by that very tensor, so the reset has to
    preserve param identity (or the optimizer would be updating a stale
    object) while dropping the accumulated moments.
    """
    from stable_baselines3 import SAC

    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        model = SAC(
            "MlpPolicy",
            train_env,
            ent_coef="auto_0.02",
            learning_starts=8,
            buffer_size=200,
            batch_size=8,
            seed=0,
            verbose=0,
        )
        # A short learn initializes the logger and accumulates Adam moment
        # state on log_ent_coef, which is what a long stage leaves behind.
        model.learn(total_timesteps=64)
        assert len(model.ent_coef_optimizer.state) == 1

        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=32,
            clear_replay_buffer_on_advance=True,
            reset_entropy_on_advance=True,
        )
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()
        # Target comes from the configured ent_coef string, so it is
        # exactly auto_0.02 even though the short learn already nudged the
        # live tensor away from it.
        assert gate._initial_log_ent_coef == pytest.approx(
            float(np.log(0.02)), rel=1e-9
        )
        assert float(model.log_ent_coef.detach().exp()) != pytest.approx(
            0.02, rel=1e-9
        )

        # Collapse the temperature the way a 1.4M-step stage does.
        model.log_ent_coef.detach().fill_(float(np.log(9.2e-4)))

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True

        assert gate.stage_index == 1
        assert float(model.log_ent_coef.detach().exp()) == pytest.approx(
            0.02, rel=1e-5
        )
        assert len(model.ent_coef_optimizer.state) == 0
        # The optimizer must still be pointed at the same tensor.
        assert (
            model.ent_coef_optimizer.param_groups[0]["params"][0]
            is model.log_ent_coef
        )
        assert model.log_ent_coef.requires_grad is True
        assert model.log_ent_coef.is_leaf is True
        # Updates are paused, so gradient_steps is 0 and the restored
        # temperature is what the learner resumes with.
        assert model.gradient_steps == 0

        # Training must resume cleanly and rebuild the moment state.
        model.learn(total_timesteps=64, reset_num_timesteps=False)
        assert np.isfinite(float(model.log_ent_coef.detach()))
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_target_ignores_a_warm_started_collapse():
    """A continuation inherits a collapsed alpha; the target must not.

    ``train()`` deliberately transfers the source run's ``log_ent_coef``
    so a fresh ``"auto"`` does not restart at 1.0, which means a warm
    continuation begins already collapsed. Reading the live tensor at
    training start would capture ~1e-3 and make every later reset restore
    exactly the collapse it exists to undo.
    """
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=10,
            reset_entropy_on_advance=True,
        )
        # Warm-started: the tensor already holds the inherited collapse.
        model = _FakeAutoEntropyModel(
            train_env,
            init_log_ent_coef=float(np.log(9.2e-4)),
            ent_coef="auto_0.02",
        )
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()
        assert gate._initial_log_ent_coef == pytest.approx(
            float(np.log(0.02))
        )

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True
        assert float(model.log_ent_coef.detach().exp()) == pytest.approx(0.02)
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("auto", 1.0), ("auto_0.02", 0.02), ("auto_1.0", 1.0)],
)
def test_configured_ent_coef_init_parses_the_auto_forms(configured, expected):
    from courtside_dynamics.callbacks.performance_gate import (
        _configured_ent_coef_init,
    )

    class _M:
        ent_coef = configured

    assert _configured_ent_coef_init(_M()) == pytest.approx(expected)


@pytest.mark.parametrize("configured", [0.02, "0.02", "auto_", "auto_x"])
def test_configured_ent_coef_init_rejects_unparseable_forms(configured):
    from courtside_dynamics.callbacks.performance_gate import (
        _configured_ent_coef_init,
    )

    class _M:
        ent_coef = configured

    with pytest.raises((TypeError, ValueError)):
        _configured_ent_coef_init(_M())


def test_entropy_reset_value_overrides_the_configured_target():
    """A continuation may not want "auto"'s 1.0 worth of exploration."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            sustain_evals=1,
            advance_update_pause_steps=10,
            reset_entropy_on_advance=True,
            entropy_reset_value=0.05,
        )
        # ent_coef "auto" would resolve to 1.0; the override wins.
        model = _FakeAutoEntropyModel(train_env, init_log_ent_coef=-7.0)
        gate.model = model  # type: ignore[assignment]
        gate._on_training_start()

        info_eval.finish_eval({"bounce_count_ep_mean": 9.0})
        assert gate._on_step() is True
        assert float(model.log_ent_coef.detach().exp()) == pytest.approx(0.05)
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), True, "0.02"])
def test_entropy_reset_value_rejects_invalid(bad):
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        with pytest.raises(ValueError, match="entropy_reset_value"):
            _gate(
                train_env,
                info_eval,
                reset_entropy_on_advance=True,
                entropy_reset_value=bad,
            )
    finally:
        train_env.close()
        eval_env.close()


def test_entropy_reset_value_requires_the_reset_to_be_enabled():
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        with pytest.raises(
            ValueError, match="requires reset_entropy_on_advance"
        ):
            _gate(train_env, info_eval, entropy_reset_value=0.02)
    finally:
        train_env.close()
        eval_env.close()


def test_stage_eval_budget_stop_ends_training_and_records_history(tmp_path):
    """A stalled non-final stage must stop the run once its evaluation
    budget is spent: run 20260727_233859 sat on stage 1 for 181 evals
    (~4.5M steps) with nothing able to notice."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            stage_eval_budget=3,
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        for i in range(2):
            info_eval.finish_eval({"bounce_count_ep_mean": 0.5})
            assert gate._on_step() is True, f"eval {i} within budget"
        info_eval.finish_eval({"bounce_count_ep_mean": 0.5})
        assert gate._on_step() is False  # budget exhausted -> stop
        assert gate.stage_index == 0
        # The stop records its reason for the run summary (the console
        # print vanishes with the Colab runtime).
        assert gate.stop_reason is not None
        assert gate.stop_reason.startswith("stage_eval_budget")

        gate._on_training_end()
        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert rows[0]["promoted"] is False
        assert rows[0]["advance_reason"] is None
        assert rows[0]["stage_eval_budget_exhausted"] is True
        assert rows[0]["evals"] == 3
    finally:
        train_env.close()
        eval_env.close()


def test_stage_eval_budget_advance_forces_promotion_with_reason(tmp_path):
    """budget_action='advance' degrades the hard gate to
    earned-or-scheduled: the stage advances with the full advance
    package and the history row says it was not earned."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env,
            info_eval,
            stage_eval_budget=2,
            stage_eval_budget_action="advance",
            stage_history_path=str(tmp_path / "curriculum_stages.json"),
        )
        gate._on_training_start()
        for _ in range(2):
            info_eval.finish_eval({"bounce_count_ep_mean": 0.5})
            assert gate._on_step() is True
        assert gate.stage_index == 1
        # The forced advance resets selection like an earned one.
        assert info_eval.selection_resets == 1

        # An earned promotion afterwards still records "gate".
        for _ in range(2):
            info_eval.finish_eval({"bounce_count_ep_mean": 5.0})
            gate._on_step()
        assert gate.stage_index == 2

        gate._on_training_end()
        rows = json.loads(
            (tmp_path / "curriculum_stages.json").read_text()
        )["stages"]
        assert rows[0]["promoted"] is False
        assert rows[0]["advance_reason"] == "stage_eval_budget"
        assert rows[1]["promoted"] is True
        assert rows[1]["advance_reason"] == "gate"
        # A budget-forced advance must not leak the exhausted flag into
        # the next stage's row.
        assert rows[1]["stage_eval_budget_exhausted"] is False
    finally:
        train_env.close()
        eval_env.close()


def test_stage_eval_budget_exempts_the_final_stage():
    """The last stage has nothing to promote to; early_stop_patience
    owns terminal convergence, so the budget must not end the run."""
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        gate = _gate(
            train_env, info_eval, sustain_evals=1, stage_eval_budget=2
        )
        gate._on_training_start()
        # Ride earned promotions to the final stage.
        for _ in range(2):
            info_eval.finish_eval({"bounce_count_ep_mean": 5.0})
            assert gate._on_step() is True
        assert gate.stage_index == 2
        # Far past the budget on the final stage: never stops.
        for _ in range(5):
            info_eval.finish_eval({"bounce_count_ep_mean": 0.0})
            assert gate._on_step() is True
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"stage_eval_budget": 0}, "positive integer"),
        ({"stage_eval_budget": True}, "positive integer"),
        ({"stage_eval_budget": 1}, ">= sustain_evals"),
        (
            {"stage_eval_budget": 5, "stage_eval_budget_action": "warn"},
            "'stop' or 'advance'",
        ),
    ),
)
def test_stage_eval_budget_configuration_is_validated(overrides, match):
    train_env = make_vec_env(_StagedEnv, n_envs=1)
    eval_env = make_vec_env(_StagedEnv, n_envs=1)
    try:
        info_eval = _FakeInfoEval(eval_env)
        with pytest.raises(ValueError, match=match):
            _gate(train_env, info_eval, **overrides)
    finally:
        train_env.close()
        eval_env.close()
