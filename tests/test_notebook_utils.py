"""Tests for the notebook-facing run-report and artifact-audit helpers.

``print_stage_summary`` and ``check_run_artifacts`` are the notebook's
post-training troubleshooting surface: the first replays the end-of-run
report inline, the second audits the run directory against the shared
``EXPECTED_ARTIFACTS`` registry and explains what a missing artifact
usually means. Both must degrade gracefully on partial/crashed runs.
"""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from courtside_dynamics.notebook_utils import (
    RunConfigPlanMismatch,
    _rollout_wall_ball_seed,
    _summarize_wall_ball_episodes,
    _wall_ball_constructor_kwargs_match,
    check_run_artifacts,
    evaluate_best_wall_ball,
    load_campaign_manifest,
    next_stage_attempt_dir,
    paddle_campaign_metrics,
    print_stage_summary,
    require_campaign_fingerprint,
    resolve_warm_start_branch,
    score_campaign_bars,
    score_paddle_stage,
    validate_run_config_against_plan,
    write_campaign_manifest,
)
from courtside_dynamics.training.artifacts import EXPECTED_ARTIFACTS, RUN_LAYOUT


def test_print_stage_summary_prints_report(tmp_path, capsys):
    (tmp_path / "stage_summary.txt").write_text("Final eval: 1.23\n")
    print_stage_summary(tmp_path)
    assert "Final eval: 1.23" in capsys.readouterr().out


def test_print_stage_summary_missing_file_explains(tmp_path, capsys):
    print_stage_summary(tmp_path)
    out = capsys.readouterr().out
    assert "no stage_summary.txt" in out
    assert "crashed" in out


def test_check_run_artifacts_empty_dir_reports_all_missing(tmp_path, capsys):
    missing = check_run_artifacts(tmp_path)
    assert set(missing) == {rel for _, rel in EXPECTED_ARTIFACTS}
    out = capsys.readouterr().out
    assert out.count("MISSING") == len(EXPECTED_ARTIFACTS)
    # Missing artifacts come with troubleshooting hints.
    assert "moviepy" in out
    assert "EvalCallback" in out


def test_check_run_artifacts_detects_files_and_dirs(tmp_path, capsys):
    # A legacy flat-layout run: best_model.zip and monitor/ at the run
    # root. The audit must resolve them through the locate_artifact
    # fallback while reporting missing entries at their new locations.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "best_model.zip").write_bytes(b"x" * 2048)
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (monitor / "0.monitor.csv").write_text("#header\n")

    missing = check_run_artifacts(tmp_path)
    assert RUN_LAYOUT["config"] not in missing
    assert RUN_LAYOUT["best_model"] not in missing
    assert RUN_LAYOUT["monitor_dir"] not in missing
    assert RUN_LAYOUT["final_model"] in missing

    out = capsys.readouterr().out
    assert "2.0 KB" in out  # file size rendered human-readably
    assert "1 file(s)" in out  # directories report their file count


def test_check_run_artifacts_all_present(tmp_path, capsys):
    for _, rel in EXPECTED_ARTIFACTS:
        full = tmp_path / rel
        # RUN_LAYOUT entries without a file extension are directories.
        if not full.suffix:
            full.mkdir(parents=True)
        else:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text("x")
    assert check_run_artifacts(tmp_path) == []
    assert "All expected artifacts present." in capsys.readouterr().out


def test_check_run_artifacts_accepts_conditional_post_training_outputs(
    tmp_path, capsys
):
    extras = (("long_eval", "best_model_long_horizon_eval.json"),)
    missing = check_run_artifacts(tmp_path, extra_artifacts=extras)
    assert "best_model_long_horizon_eval.json" in missing
    assert "long_eval:" in capsys.readouterr().out


def _write_synthetic_evaluations(tmp_path, best_mean=6.0, tail_mean=-1.0):
    """An evaluations.npz whose curve peaks mid-run then collapses."""
    import numpy as np

    timesteps = np.arange(1, 11) * 100_000
    means = np.full(10, tail_mean, dtype=float)
    means[3] = best_mean  # best checkpoint at 400k
    results = means[:, None] + np.array([[-0.1, 0.0, 0.1]])
    np.savez(
        tmp_path / "evaluations.npz",
        timesteps=timesteps,
        results=results,
        ep_lengths=np.full_like(results, 100),
    )
    return 400_000


def test_write_run_summary_flags_post_best_regression(tmp_path):
    """A final eval far below the best must be called out explicitly --
    the operator shouldn't have to diff two report lines to notice a
    collapse (as with the first WallBall run, final 7 points below best)."""
    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig
    from courtside_dynamics.training.artifacts import write_run_summary

    _write_synthetic_evaluations(tmp_path, best_mean=6.0, tail_mean=-1.0)
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        log_dir=str(tmp_path),
        total_timesteps=1_000_000,
    )
    out = write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=-1.0,
        final_std_reward=0.1,
        duration_seconds=100.0,
        actual_timesteps=650_000,
    )
    text = open(out).read()
    assert "Final vs best" in text
    assert "-7.000" in text
    assert "regressed after best" in text
    # Early-stopped runs report actual vs budgeted steps, and
    # throughput is computed from steps actually trained.
    assert "650,000 of 1,000,000 budget" in text
    assert "6500 FPS" in text


def test_write_run_summary_no_regression_flag_when_final_matches_best(tmp_path):
    """A healthy run (final ~ best) must not carry the scary flag."""
    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig
    from courtside_dynamics.training.artifacts import write_run_summary

    _write_synthetic_evaluations(tmp_path, best_mean=6.0, tail_mean=5.8)
    cfg = TrainConfig(env_fn=lambda: BallBalanceEnv(), log_dir=str(tmp_path))
    out = write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=5.9,
        final_std_reward=0.1,
        duration_seconds=100.0,
    )
    text = open(out).read()
    assert "Final vs best" in text
    assert "regressed after best" not in text
    assert "stopped early" not in text
    assert "Stop reason" not in text


def test_write_run_summary_records_the_stop_reason(tmp_path):
    """A stopped-early summary must say WHICH guard fired: the stopping
    callback printed it, but the console vanishes with the Colab
    runtime, and a bare "(stopped early)" sends the run review back to
    the curves to reconstruct why."""
    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig
    from courtside_dynamics.training.artifacts import write_run_summary

    _write_synthetic_evaluations(tmp_path, best_mean=6.0, tail_mean=5.8)
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        log_dir=str(tmp_path),
        total_timesteps=1_000_000,
    )
    out = write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=5.9,
        final_std_reward=0.1,
        duration_seconds=100.0,
        actual_timesteps=650_000,
        stop_reason=(
            "early_stop_patience: no improvement in bounce_count_ep_mean "
            "for the last 20 evaluations"
        ),
    )
    text = open(out).read()
    assert "Stop reason" in text
    assert "early_stop_patience" in text


def test_plot_learning_curve_marks_best_checkpoint(tmp_path):
    """The eval panels carry a best-checkpoint marker so post-best
    collapse is visible at a glance."""
    import matplotlib

    matplotlib.use("Agg")

    from courtside_dynamics.notebook_utils import plot_learning_curve

    best_step = _write_synthetic_evaluations(tmp_path)
    fig = plot_learning_curve(tmp_path, show=False)
    try:
        eval_ax = fig.axes[2]  # bottom-left: evaluation rewards
        labels = [line.get_label() for line in eval_ax.get_lines()]
        assert "best checkpoint" in labels
        marker = eval_ax.get_lines()[labels.index("best checkpoint")]
        assert marker.get_xdata()[0] == best_step
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_write_run_summary_lists_artifacts_from_shared_registry(tmp_path):
    """The report's Artifacts section iterates ``EXPECTED_ARTIFACTS`` (minus
    the report itself), so it and ``check_run_artifacts`` can't drift."""
    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig
    from courtside_dynamics.training.artifacts import write_run_summary

    (tmp_path / "monitor").mkdir()
    (tmp_path / "videos").mkdir()
    (tmp_path / "best_model.zip").write_text("x")

    cfg = TrainConfig(env_fn=lambda: BallBalanceEnv(), log_dir=str(tmp_path))
    out_path = write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.0,
        duration_seconds=10.0,
    )
    text = open(out_path).read()
    assert "monitor_dir" in text
    assert "videos_dir" in text
    assert "best_model" in text
    # The report never lists itself as an artifact.
    assert "stage_summary:" not in text


def test_wall_ball_constructor_comparison_accepts_only_omitted_new_defaults():
    legacy_training = {
        "episode_len": 750,
        "min_force": 20.0,
        "nested": {"range": [-3.2, -2.2]},
    }
    evaluated_defaults = {
        "episode_len": 5_000,
        "min_force": 20.0,
        "nested": {"range": (-3.2, -2.2)},
        "style_violation_penalty": 1.0,
        "rally_style": "open",
        "paddle_home_x": -1.7,
        "paddle_x_target_range": None,
        "recovery_reset_probability": 0.0,
        "post_bounce_reset_fraction": 0.5,
        "recoverable_bounce_bonus": 0.0,
        "recoverable_bounce_lateral_limit": 0.0,
    }

    assert _wall_ball_constructor_kwargs_match(
        legacy_training, evaluated_defaults
    )
    assert not _wall_ball_constructor_kwargs_match(
        legacy_training,
        {**evaluated_defaults, "rally_style": "one_bounce"},
    )
    assert not _wall_ball_constructor_kwargs_match(
        legacy_training,
        {**evaluated_defaults, "unrelated_new_setting": 1},
    )
    assert _wall_ball_constructor_kwargs_match(
        {**legacy_training, "rally_style": "one_bounce"},
        {**evaluated_defaults, "rally_style": "one_bounce"},
    )


def _wall_ball_row(
    *,
    reward: float,
    length: int,
    returns: int,
    floor_total: int,
    terminal_floor: int,
    reason: str,
    legal_paddle_hits: int | None = None,
    pre_bounce_legal_hits: int = 0,
    post_bounce_legal_hits: int | None = None,
    opening_volleys: int = 0,
    post_bounce_completed_returns: int = 0,
    one_bounce_recoveries: int = 0,
    one_bounce_returns: int = 0,
    style_violation_reason: str | None = None,
) -> dict[str, object]:
    resolved_legal_hits = (
        returns + 1 if legal_paddle_hits is None else legal_paddle_hits
    )
    resolved_post_bounce_hits = (
        resolved_legal_hits - pre_bounce_legal_hits
        if post_bounce_legal_hits is None
        else post_bounce_legal_hits
    )
    row: dict[str, object] = {
        "episode_reward": reward,
        "episode_length": length,
        "completed_returns": returns,
        "paddle_hit_count": returns + 1,
        "legal_paddle_hit_count": resolved_legal_hits,
        "pre_bounce_legal_paddle_hit_count": pre_bounce_legal_hits,
        "post_bounce_legal_paddle_hit_count": resolved_post_bounce_hits,
        "opening_volley_count": opening_volleys,
        "post_bounce_completed_return_count": (
            post_bounce_completed_returns
        ),
        "wall_contact_count": returns,
        "floor_bounce_total": floor_total,
        "terminal_floor_bounce_count": terminal_floor,
        "one_bounce_recovery_count": one_bounce_recoveries,
        "one_bounce_return_count": one_bounce_returns,
        "post_floor_bounce_paddle_recoveries": 0,
        "post_floor_bounce_completed_returns": 0,
        "unrecovered_floor_bounces": floor_total,
        "termination_reason": reason,
        "term_style_violation": reason == "style_violation",
        "style_violation_reason": style_violation_reason,
    }
    for key in (
        "rew_wall",
        "rew_paddle",
        "rew_shaping",
        "rew_oob",
        "rew_double_bounce",
        "rew_stall",
        "rew_style_violation",
        "rew_recoverable_bounce",
        "rew_early_touch",
        "rew_weak_return",
        "rew_first_hit",
    ):
        row[f"{key}_total"] = reward if key == "rew_wall" else 0.0
    return row


def test_summarize_wall_ball_episodes_survival_and_floor_proxy():
    rows = [
        _wall_ball_row(
            reward=10.0,
            length=5_000,
            returns=20,
            floor_total=2,
            terminal_floor=0,
            reason="timeout",
        ),
        _wall_ball_row(
            reward=4.0,
            length=1_200,
            returns=5,
            floor_total=2,
            terminal_floor=2,
            reason="double_bounce",
        ),
    ]

    summary = _summarize_wall_ball_episodes(
        rows, survival_steps=(750, 1_500, 3_000, 5_000)
    )

    assert summary["metrics"]["completed_returns"]["mean"] == 12.5
    assert summary["step_survival"]["750"] == {"count": 2, "rate": 1.0}
    assert summary["step_survival"]["1500"] == {"count": 1, "rate": 0.5}
    assert summary["step_survival"]["5000"] == {"count": 1, "rate": 0.5}
    assert summary["return_survival_curve"]["5"]["rate"] == 1.0
    assert summary["return_survival_curve"]["6"]["rate"] == 0.5
    assert summary["terminations"]["timeout"] == {"count": 1, "rate": 0.5}
    assert summary["terminations"]["out_of_bounds"] == {"count": 0, "rate": 0.0}
    assert summary["terminations"]["style_violation"] == {
        "count": 0,
        "rate": 0.0,
    }
    assert summary["style_violation_reasons"] == {}
    assert summary["floor_bounce_diagnostics"] == {
        "total_contacts": 4,
        "episodes_with_any": 2,
        "episodes_with_any_rate": 1.0,
        "post_floor_bounce_paddle_recoveries": 0,
        "episodes_with_paddle_recovery": 0,
        "episodes_with_paddle_recovery_rate": 0.0,
        "post_floor_bounce_completed_returns": 0,
        "episodes_with_completed_return": 0,
        "episodes_with_completed_return_rate": 0.0,
        "contacts_reset_before_terminal_upper_bound": 2,
    }


def test_summarize_wall_ball_episodes_reports_strict_style_metrics():
    rows = [
        _wall_ball_row(
            reward=-1.0,
            length=100,
            returns=0,
            floor_total=0,
            terminal_floor=0,
            reason="style_violation",
            legal_paddle_hits=0,
            style_violation_reason="premature_volley",
        ),
        _wall_ball_row(
            reward=3.0,
            length=500,
            returns=2,
            floor_total=2,
            terminal_floor=0,
            reason="timeout",
            legal_paddle_hits=2,
            post_bounce_completed_returns=2,
            one_bounce_recoveries=2,
            one_bounce_returns=2,
        ),
    ]

    summary = _summarize_wall_ball_episodes(rows, survival_steps=(100, 500))

    assert summary["metrics"]["legal_paddle_hit_count"]["mean"] == 1.0
    assert summary["metrics"]["one_bounce_recovery_count"]["mean"] == 1.0
    assert summary["metrics"]["one_bounce_return_count"]["mean"] == 1.0
    assert summary["contact_sequence_diagnostics"][
        "post_bounce_completed_return_rate"
    ] == 1.0
    assert summary["terminations"]["style_violation"] == {
        "count": 1,
        "rate": 0.5,
    }
    assert summary["style_violation_reasons"] == {
        "premature_volley": {"count": 1, "rate": 0.5}
    }
    assert "rew_style_violation" in summary["reward_components"]
    assert summary["return_survival_curve"]["3"] == {
        "count": 0,
        "rate": 0.0,
    }
    assert summary["return_survival_curve"]["5"] == {
        "count": 0,
        "rate": 0.0,
    }


class _FakeWallBallEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_len: int, instances: list[_FakeWallBallEnv]):
        self.episode_len = episode_len
        self._ezpickle_kwargs = {
            "episode_len": episode_len,
            "min_force": 20.0,
            "paddle_x_target_range": (-3.2, -2.2),
        }
        self.observation_space = Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.reset_seeds: list[int | None] = []
        self.closed = False
        self._step = 0
        instances.append(self)

    def reset(self, *, seed=None, options=None):
        del options
        self.reset_seeds.append(seed)
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        assert np.asarray(action).shape == (1,)
        self._step += 1
        truncated = self._step >= self.episode_len
        info = {
            "bounce_count": self._step,
            "paddle_hit_count": self._step,
            "wall_contact_count": self._step,
            "floor_bounce_total": 0,
            "floor_bounce_count": 0,
            "legal_paddle_hit_count": self._step,
            "pre_bounce_legal_paddle_hit_count": self._step,
            "post_bounce_legal_paddle_hit_count": 0,
            "opening_volley_count": 1,
            "post_bounce_completed_return_count": 0,
            "one_bounce_recovery_count": 0,
            "one_bounce_return_count": 0,
            "style_violation_reason": None,
            "rew_wall": 1.0,
            "rew_paddle": 0.0,
            "rew_shaping": 0.0,
            "rew_oob": 0.0,
            "rew_double_bounce": 0.0,
            "rew_stall": 0.0,
            "rew_style_violation": 0.0,
            "rew_recoverable_bounce": 0.0,
            "rew_early_touch": 0.0,
            "rew_weak_return": 0.0,
            "rew_first_hit": 0.0,
            "term_oob": False,
            "term_double_bounce": False,
            "term_stall": False,
            "term_style_violation": False,
            "term_timeout": truncated,
            "term_nonfinite": False,
        }
        obs = np.full(2, self._step, dtype=np.float32)
        return obs, 1.0, False, truncated, info

    def close(self):
        self.closed = True


class _FakeModel:
    observation_space = Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.num_timesteps = 123
        self.deterministic_values: list[bool] = []
        self.observations: list[np.ndarray] = []

    def predict(self, observation, *, deterministic):
        obs = np.asarray(observation)
        assert bool((obs >= 10.0).all()), "paired normalizer was not applied"
        self.observations.append(obs.copy())
        self.deterministic_values.append(deterministic)
        return np.zeros(1, dtype=np.float32), None


def test_rollout_counts_post_floor_bounce_recovery_and_return():
    instances: list[_FakeWallBallEnv] = []

    class ScriptedEnv(_FakeWallBallEnv):
        def step(self, action):
            assert np.asarray(action).shape == (1,)
            self._step += 1
            snapshots = (
                (0, 0, 0, 1),  # floor bounce
                (0, 1, 0, 1),  # paddle recovers it
                (1, 1, 1, 1),  # gated wall return completes the cycle
            )
            returns, paddle_hits, wall_contacts, floor_total = snapshots[
                self._step - 1
            ]
            truncated = self._step == len(snapshots)
            info = {
                "bounce_count": returns,
                "paddle_hit_count": paddle_hits,
                "wall_contact_count": wall_contacts,
                "floor_bounce_total": floor_total,
                "floor_bounce_count": 0,
                "legal_paddle_hit_count": paddle_hits,
                "pre_bounce_legal_paddle_hit_count": 0,
                "post_bounce_legal_paddle_hit_count": paddle_hits,
                "opening_volley_count": 0,
                "post_bounce_completed_return_count": returns,
                "one_bounce_recovery_count": int(self._step >= 2),
                "one_bounce_return_count": int(self._step >= 3),
                "style_violation_reason": None,
                "rew_wall": float(returns),
                "rew_paddle": 0.0,
                "rew_shaping": 0.0,
                "rew_oob": 0.0,
                "rew_double_bounce": 0.0,
                "rew_stall": 0.0,
                "rew_style_violation": 0.0,
                "rew_recoverable_bounce": 0.0,
                "rew_early_touch": 0.0,
                "rew_weak_return": 0.0,
                "rew_first_hit": 0.0,
                "term_oob": False,
                "term_double_bounce": False,
                "term_stall": False,
                "term_style_violation": False,
                "term_timeout": truncated,
                "term_nonfinite": False,
            }
            return np.zeros(2, dtype=np.float32), float(returns), False, truncated, info

    class ZeroModel:
        @staticmethod
        def predict(_observation, *, deterministic):
            assert deterministic is True
            return np.zeros(1, dtype=np.float32), None

    env = ScriptedEnv(3, instances)
    try:
        row = _rollout_wall_ball_seed(
            env,
            ZeroModel(),
            lambda obs: obs,
            seed=9,
            episode_len=3,
            deterministic=True,
        )
    finally:
        env.close()

    assert row["floor_bounce_total"] == 1
    assert row["post_floor_bounce_paddle_recoveries"] == 1
    assert row["post_floor_bounce_completed_returns"] == 1
    assert row["unrecovered_floor_bounces"] == 0
    assert row["legal_paddle_hit_count"] == 1
    assert row["pre_bounce_legal_paddle_hit_count"] == 0
    assert row["post_bounce_legal_paddle_hit_count"] == 1
    assert row["opening_volley_count"] == 0
    assert row["post_bounce_completed_return_count"] == 1
    assert row["one_bounce_recovery_count"] == 1
    assert row["one_bounce_return_count"] == 1


def _write_wall_ball_best_artifacts(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "env": {
                    "class": "WallBallEnv",
                    "constructor_kwargs": {
                        "episode_len": 750,
                        "min_force": 20.0,
                        "paddle_x_target_range": [-3.2, -2.2],
                    },
                },
                "train_config": {
                    "algo": "SAC",
                    "headline_key": "bounce_count",
                    "normalize_obs": True,
                    "clip_obs": 10.0,
                    "normalize_obs_excluded_indices": [],
                },
            }
        )
    )
    model_bytes = b"model"
    normalizer_bytes = b"normalizer"
    (tmp_path / "best_model.zip").write_bytes(model_bytes)
    (tmp_path / "best_vec_normalize.pkl").write_bytes(normalizer_bytes)
    (tmp_path / "best_model_meta.json").write_text(
        json.dumps(
            {
                "timestep": 123,
                "selection_keys": [
                    "bounce_count_ep_mean",
                    "success_rate",
                    "episode_reward_mean",
                ],
                "artifacts": {
                    "best_model.zip": {
                        "sha256": hashlib.sha256(model_bytes).hexdigest(),
                    },
                    "best_vec_normalize.pkl": {
                        "sha256": hashlib.sha256(normalizer_bytes).hexdigest(),
                    },
                },
            }
        )
    )


def test_evaluate_best_wall_ball_writes_drive_ready_metrics(tmp_path, monkeypatch):
    from stable_baselines3.common.vec_env import VecNormalize

    from courtside_dynamics.training import algos

    _write_wall_ball_best_artifacts(tmp_path)
    instances: list[_FakeWallBallEnv] = []
    model = _FakeModel()

    class FakeAlgo:
        @staticmethod
        def load(path, *, device):
            assert path.endswith("best_model.zip")
            assert device == "cpu"
            return model

    class FakeNormalizer:
        training = True
        norm_reward = True
        norm_obs = True
        clip_obs = 10.0
        normalize_obs_excluded_indices = ()

        @staticmethod
        def normalize_obs(obs):
            return np.asarray(obs) + 10.0

    monkeypatch.setattr(algos, "resolve_algo", lambda _name: FakeAlgo)
    monkeypatch.setattr(
        VecNormalize,
        "load",
        staticmethod(lambda _path, _venv: FakeNormalizer()),
    )

    payload = evaluate_best_wall_ball(
        tmp_path,
        lambda: _FakeWallBallEnv(3, instances),
        episode_len=3,
        seeds=(101, 102),
    )

    assert payload["policy"]["best_model_meta"]["timestep"] == 123
    assert payload["policy"]["pair_verification"] == (
        "verified_by_best_model_meta_sha256"
    )
    assert payload["environment"]["verification"] == (
        "verified_against_training_constructor"
    )
    assert payload["metrics"]["completed_returns"]["mean"] == 3.0
    assert payload["schema_version"] == 3
    assert payload["contact_sequence_diagnostics"] == {
        "legal_hit_total": 6,
        "pre_bounce_legal_hit_total": 6,
        "post_bounce_legal_hit_total": 0,
        "post_bounce_legal_hit_rate": 0.0,
        "opening_volley_total": 2,
        "episodes_with_opening_volley": 2,
        "opening_volley_episode_rate": 1.0,
        "completed_return_total": 6,
        "post_bounce_completed_return_total": 0,
        "post_bounce_completed_return_rate": 0.0,
    }
    assert payload["evaluation"]["return_survival_thresholds"] == [1, 2, 3, 5]
    assert payload["return_survival_curve"]["5"] == {
        "count": 0,
        "rate": 0.0,
    }
    assert payload["terminations"]["timeout"] == {"count": 2, "rate": 1.0}
    assert payload["terminations"]["double_bounce"] == {
        "count": 0,
        "rate": 0.0,
    }
    assert payload["step_survival"]["3"] == {"count": 2, "rate": 1.0}
    assert model.deterministic_values == [True] * 6
    assert any(instance.reset_seeds == [101, 102] for instance in instances)
    assert all(instance.closed for instance in instances)

    # The registry places the long-horizon outputs under reports/.
    summary_path = tmp_path / RUN_LAYOUT["best_long_eval"]
    episodes_path = tmp_path / RUN_LAYOUT["best_long_eval_episodes"]
    assert json.loads(summary_path.read_text()) == payload
    with episodes_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["evaluation_id"] for row in rows} == {payload["evaluation_id"]}
    assert payload["outputs"]["episodes_csv"]["sha256"]
    assert [int(row["seed"]) for row in rows] == [101, 102]
    assert [int(row["completed_returns"]) for row in rows] == [3, 3]


def test_evaluate_best_wall_ball_requires_task_selected_artifacts(tmp_path):
    _write_wall_ball_best_artifacts(tmp_path)
    (tmp_path / "best_vec_normalize.pkl").unlink()

    try:
        evaluate_best_wall_ball(
            tmp_path,
            lambda: None,
            episode_len=3,
            seeds=(1,),
        )
    except FileNotFoundError as exc:
        assert "best_vec_normalize.pkl" in str(exc)
    else:
        raise AssertionError("missing paired normalizer did not fail")


def test_evaluate_best_wall_ball_rejects_hash_mismatched_pair(tmp_path):
    _write_wall_ball_best_artifacts(tmp_path)
    (tmp_path / "best_vec_normalize.pkl").write_bytes(b"stale-normalizer")

    try:
        evaluate_best_wall_ball(
            tmp_path,
            lambda: None,
            episode_len=3,
            seeds=(1,),
        )
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("hash-mismatched model/normalizer pair did not fail")


class TestResolveRunConfigFile:
    def test_creates_from_starter_then_reuses_edits(self, tmp_path, capsys):
        from courtside_dynamics.notebook_utils import resolve_run_config_file

        root = tmp_path / "configs"
        path = resolve_run_config_file(
            "WallBallBaseline", local_root=str(root)
        )
        assert path == root / "wall_ball_baseline.toml"
        assert path.exists()
        assert "created from packaged starter" in capsys.readouterr().out

        # A second resolution returns the same file untouched -- even
        # after the user edits it (their experiment file wins).
        path.write_text("[train]\nn_envs = 2\n", encoding="utf-8")
        again = resolve_run_config_file(
            "WallBallBaseline", local_root=str(root)
        )
        assert again == path
        assert path.read_text(encoding="utf-8") == "[train]\nn_envs = 2\n"
        out = capsys.readouterr().out
        assert "reusing existing copy" in out
        assert "sha256" in out

    def test_unknown_recipe_returns_none_with_reason(self, tmp_path, capsys):
        from courtside_dynamics.notebook_utils import resolve_run_config_file

        assert (
            resolve_run_config_file("NoSuchEnv", local_root=str(tmp_path))
            is None
        )
        assert "no packaged starter" in capsys.readouterr().out

    def test_drive_requested_but_unmounted_falls_back(self, tmp_path, capsys):
        from courtside_dynamics.notebook_utils import resolve_run_config_file

        # No /content/drive in test environments: must fall back loudly.
        path = resolve_run_config_file(
            "WallBallBaseline",
            use_drive=True,
            local_root=str(tmp_path / "cfg"),
        )
        assert path is not None and path.exists()
        assert "not mounted" in capsys.readouterr().out


def _write_synthetic_eval_info(csv_path, stems, timesteps=(25_000, 50_000, 75_000)):
    """Long-format eval_info.csv with a ``_mean`` series per stem."""
    lines = ["timestep,metric,value"]
    for step_index, timestep in enumerate(timesteps):
        for stem_index, stem in enumerate(stems):
            lines.append(f"{timestep},{stem}_mean,{stem_index + 0.1 * step_index}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n")


def test_split_eval_metric_knows_every_variant_suffix():
    """``_ep_sum_mean`` splits ahead of its own ``_mean`` suffix, so
    ``rew_hold_ep_sum_mean`` overlays on the ``rew_hold`` panel instead
    of spawning a phantom ``rew_hold_ep_sum`` stem."""
    from courtside_dynamics.notebook_utils import _split_eval_metric

    assert _split_eval_metric("rew_hold_mean") == ("rew_hold", "mean")
    assert _split_eval_metric("rew_hold_ep_sum_mean") == (
        "rew_hold",
        "ep_sum_mean",
    )
    assert _split_eval_metric("rally_count_ep_mean") == (
        "rally_count",
        "ep_mean",
    )
    assert _split_eval_metric("phase_frac_rally") == ("phase_frac", "rally")
    assert _split_eval_metric("episode_length") == ("episode_length", "")


def test_plot_eval_info_overlays_ep_sum_mean_on_the_stem_panel(tmp_path):
    """The per-episode reward sum renders as a labeled variant line on
    its stem's panel, not as a separate panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from courtside_dynamics.notebook_utils import plot_eval_info

    csv_path = tmp_path / "metrics" / "eval_info.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "timestep,metric,value\n"
        "25000,rew_hold_mean,0.1\n"
        "25000,rew_hold_ep_sum_mean,12.5\n"
        "50000,rew_hold_mean,0.2\n"
        "50000,rew_hold_ep_sum_mean,25.0\n"
    )
    figures = plot_eval_info(tmp_path, show=False)
    assert figures is not None
    try:
        rewards_page = figures[2]
        titled = [axis for axis in rewards_page.axes if axis.get_title()]
        assert [axis.get_title() for axis in titled] == ["rew_hold"]
        labels = {line.get_label() for line in titled[0].get_lines()}
        assert labels == {"mean", "ep_sum_mean"}
    finally:
        for figure in figures:
            plt.close(figure)


def test_plot_eval_info_writes_four_bounded_pages(tmp_path):
    """The mega-plot is split into four themed pages, each capped at 24
    panels with numbered continuation pages, and every file stays small
    enough to casually download (the old single grid weighed 1.6-2.4 MB)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from courtside_dynamics.notebook_utils import plot_eval_info

    stems = [
        # eval_headline: task counters + reward/length context.
        "bounce_count",
        "bounce_count_ep_ge_2_rate",
        "episode_reward",
        "episode_length_hist",
        "paddle_hit_count",
        "legal_paddle_hit_count",
        "return_count",
        # eval_terminations: term_* rates and the shared phase panel.
        "term_oob",
        "term_double_bounce",
        "term_stall",
        "phase_frac_rally",
        "phase_frac_serve",
        # eval_rewards: reward components.
        "rew_wall",
        "rew_paddle",
        "rew_shaping",
        # eval_diagnostics: everything else -- enough stems to spill
        # past the 24-panel cap onto a continuation page.
        *(f"sensor_{index:02d}" for index in range(30)),
    ]
    # Legacy flat location on purpose: the reader must resolve it via
    # the locate_artifact fallback.
    _write_synthetic_eval_info(tmp_path / "eval_info.csv", stems)

    figures = plot_eval_info(
        tmp_path,
        save_path=str(tmp_path / "reports" / "eval_headline.png"),
        show=False,
    )
    assert figures is not None
    try:
        # 24-panel cap on every page (axes include blank grid slots).
        assert all(len(figure.axes) <= 24 for figure in figures)
        # headline + terminations + rewards + diagnostics x2.
        assert len(figures) == 5
    finally:
        for figure in figures:
            plt.close(figure)

    reports = tmp_path / "reports"
    for page in (
        "eval_headline",
        "eval_terminations",
        "eval_rewards",
        "eval_diagnostics",
    ):
        assert (reports / f"{page}.png").is_file(), page
    # 30 diagnostics stems -> 24 on the first page, 6 spill over.
    assert (reports / "eval_diagnostics_2.png").is_file()
    assert not (reports / "eval_diagnostics_3.png").exists()
    assert not (reports / "eval_info.png").exists()
    for png in reports.glob("*.png"):
        assert png.stat().st_size < 600_000, (
            f"{png.name} is {png.stat().st_size} bytes -- pages must stay "
            "casually downloadable"
        )


def test_plot_eval_info_groups_phase_and_term_metrics_together(tmp_path):
    """``term_*`` and ``phase_frac_*`` land on the terminations page and
    the ``phase_frac_<label>`` series still share one panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from courtside_dynamics.notebook_utils import plot_eval_info

    _write_synthetic_eval_info(
        tmp_path / "metrics" / "eval_info.csv",
        ["bounce_count", "term_oob", "phase_frac_rally", "phase_frac_serve"],
    )
    figures = plot_eval_info(tmp_path, save_path=str(tmp_path / "reports"), show=False)
    assert figures is not None
    try:
        # Always the four-page contract; empty groups render placeholders.
        assert len(figures) == 4
        terminations = figures[1]
        titles = {
            axis.get_title()
            for axis in terminations.axes
            if axis.get_title()
        }
        assert titles == {"term_oob", "phase_frac"}
    finally:
        for figure in figures:
            plt.close(figure)
    for page in (
        "eval_headline",
        "eval_terminations",
        "eval_rewards",
        "eval_diagnostics",
    ):
        assert (tmp_path / "reports" / f"{page}.png").is_file(), page


# ---------------------------------------------------------------------------
# PaddleTennis staged-campaign helpers
# ---------------------------------------------------------------------------


def _trace(
    *,
    serving: bool,
    hits: int,
    touched=(),
    ready=(),
    recovery=(),
    termination: str = "opponent_fault",
):
    """A lightweight EpisodeTrace stand-in for the aggregation tests."""
    from courtside_dynamics.training.paddle_diagnosis import EpisodeTrace

    return EpisodeTrace(
        serve_side_is_policy=serving,
        shots=[],
        policy_hits=hits,
        crossings=hits,
        termination=termination,
        ender="opponent",
        recovery_travel=list(recovery),
        ready_errors=list(ready),
        touched_after_bounce=list(touched),
    )


def test_paddle_campaign_metrics_mirrors_the_instrument():
    traces = [
        _trace(serving=False, hits=2, touched=[True, True], ready=[1.0, 3.0]),
        _trace(serving=False, hits=0, touched=[False], ready=[2.0]),
        _trace(serving=True, hits=1),
        _trace(serving=True, hits=3, termination="nonfinite_state"),
    ]
    metrics = paddle_campaign_metrics(traces, [4.0, 6.0])
    assert metrics["points"] == 4
    assert metrics["receiving_points"] == 2
    assert metrics["serving_points"] == 2
    assert metrics["k1_receiving_survival"] == 0.5
    assert metrics["k2_receiving_survival"] == 0.5
    assert metrics["k3_receiving_survival"] == 0.0
    assert metrics["k1_serving_survival"] == 1.0
    assert metrics["k2_serving_survival"] == 0.5
    assert metrics["k3_serving_survival"] == 0.5
    # RK1/RK2 read "either parity": the better split wins.
    assert metrics["k2_either_survival"] == 0.5
    assert metrics["k3_either_survival"] == 0.5
    assert metrics["touched_after_bounce_rate"] == pytest.approx(2 / 3)
    assert metrics["touched_after_bounce_samples"] == 3
    assert metrics["ready_error_mean"] == pytest.approx(2.0)
    assert metrics["ready_error_samples"] == 3
    assert metrics["unsafe_terminations"] == 1
    assert metrics["interpoint_travel_mean"] == pytest.approx(5.0)
    assert metrics["interpoint_boundaries"] == 2


def test_paddle_campaign_metrics_empty_split_reads_none_not_zero():
    traces = [_trace(serving=False, hits=1)]
    metrics = paddle_campaign_metrics(traces, [])
    assert metrics["k1_serving_survival"] is None
    assert metrics["k2_either_survival"] == 0.0  # from receiving alone
    assert metrics["touched_after_bounce_rate"] is None
    assert metrics["ready_error_mean"] is None
    assert metrics["interpoint_travel_mean"] is None
    with pytest.raises(ValueError, match="at least one diagnosis trace"):
        paddle_campaign_metrics([], [])


def test_paddle_campaign_metrics_recovery_hold_mirrors_the_report():
    # Pooled hold-window travels: [1, 2] + [3, 10] = [1, 2, 3, 10].
    traces = [
        _trace(serving=False, hits=2, recovery=[1.0, 2.0]),
        _trace(serving=True, hits=2, recovery=[3.0, 10.0]),
    ]
    metrics = paddle_campaign_metrics(traces, [])
    pooled = [1.0, 2.0, 3.0, 10.0]
    # Hand-computed: mean 4.0; linear-interpolated p90 sits 0.7 of the
    # way from 3 to 10 (rank (4-1)*0.9 = 2.7), i.e. 7.9.
    assert metrics["recovery_hold_travel_mean"] == pytest.approx(4.0)
    assert metrics["recovery_hold_travel_p90"] == pytest.approx(7.9)
    # Same formula as paddle_diagnosis.report's "recovery hold" line
    # (numpy's default linear percentile), bit-for-bit.
    assert metrics["recovery_hold_travel_mean"] == float(np.mean(pooled))
    assert metrics["recovery_hold_travel_p90"] == float(
        np.percentile(pooled, 90)
    )
    assert metrics["recovery_hold_samples"] == 4


def test_paddle_campaign_metrics_absent_travels_read_none():
    traces = [_trace(serving=False, hits=1)]
    # No travels list at all (e.g. a caller that only has traces): the
    # inter-point metrics read None, never a fabricated zero.
    metrics = paddle_campaign_metrics(traces)
    assert metrics["interpoint_travel_mean"] is None
    assert metrics["interpoint_boundaries"] is None
    # A measured-but-empty travels list keeps its existing reading.
    metrics = paddle_campaign_metrics(traces, [])
    assert metrics["interpoint_travel_mean"] is None
    assert metrics["interpoint_boundaries"] == 0
    # No hold windows observed (zero policy hits recorded): None too.
    assert metrics["recovery_hold_travel_mean"] is None
    assert metrics["recovery_hold_travel_p90"] is None
    assert metrics["recovery_hold_samples"] == 0


def test_score_campaign_bars_gates_on_recovery_hold_travel():
    # Lower-is-better gating on the post-swing wander, the same
    # mechanism LS-G uses for ready_error_mean.
    bars = {
        "H1_hold": {
            "metric": "recovery_hold_travel_mean",
            "pass_at": 0.5,
            "fail_at": 2.0,
            "higher_is_better": False,
            "gating": True,
        }
    }
    still = [_trace(serving=False, hits=2, recovery=[0.2, 0.4])]
    scored = score_campaign_bars(paddle_campaign_metrics(still), bars)
    assert scored["bars"]["H1_hold"]["verdict"] == "PASS"
    assert scored["verdict"] == "PASS"

    wandering = [_trace(serving=False, hits=2, recovery=[3.0, 5.0])]
    scored = score_campaign_bars(paddle_campaign_metrics(wandering), bars)
    assert scored["bars"]["H1_hold"]["verdict"] == "FAIL"
    assert scored["verdict"] == "FAIL"

    # No hold windows observed: NO_DATA, which is never a PASS.
    no_windows = [_trace(serving=False, hits=0)]
    scored = score_campaign_bars(paddle_campaign_metrics(no_windows), bars)
    assert scored["bars"]["H1_hold"]["verdict"] == "NO_DATA"
    assert scored["verdict"] == "MIDDLE"


def test_score_campaign_bars_three_valued_with_no_data():
    metrics = {"a": 0.12, "b": 0.03, "c": 2.2, "d": None}
    bars = {
        "A": {
            "metric": "a",
            "pass_at": 0.10,
            "fail_at": 0.05,
            "higher_is_better": True,
            "gating": True,
        },
        "B": {
            "metric": "b",
            "pass_at": 0.10,
            "fail_at": 0.02,
            "higher_is_better": True,
            "gating": True,
        },
        "C": {
            "metric": "c",
            "pass_at": 2.0,
            "fail_at": 2.4,
            "higher_is_better": False,
            "gating": False,
        },
        "D": {
            "metric": "d",
            "pass_at": 0.5,
            "fail_at": 0.1,
            "higher_is_better": True,
            "gating": False,
        },
    }
    scored = score_campaign_bars(metrics, bars)
    assert scored["bars"]["A"]["verdict"] == "PASS"
    assert scored["bars"]["B"]["verdict"] == "MIDDLE"
    assert scored["bars"]["C"]["verdict"] == "MIDDLE"
    assert scored["bars"]["D"]["verdict"] == "NO_DATA"
    assert scored["verdict"] == "MIDDLE"  # over gating bars only

    scored = score_campaign_bars({**metrics, "b": 0.01}, bars)
    assert scored["bars"]["B"]["verdict"] == "FAIL"
    assert scored["verdict"] == "FAIL"

    scored = score_campaign_bars({**metrics, "b": 0.5}, bars)
    assert scored["verdict"] == "PASS"

    record_only = {
        name: {**spec, "gating": False} for name, spec in bars.items()
    }
    assert score_campaign_bars(metrics, record_only)["verdict"] == "RECORDED"


def test_score_campaign_bars_edge_readings_land_middle():
    higher = {
        "H": {
            "metric": "m",
            "pass_at": 0.10,
            "fail_at": 0.05,
            "higher_is_better": True,
            "gating": True,
        }
    }
    lower = {
        "L": {
            "metric": "m",
            "pass_at": 2.0,
            "fail_at": 2.4,
            "higher_is_better": False,
            "gating": True,
        }
    }
    # PASS wins its inclusive edge; a reading exactly on the fail line
    # lands MIDDLE (prereg edge ambiguity routes to the maintainer).
    assert score_campaign_bars({"m": 0.10}, higher)["verdict"] == "PASS"
    assert score_campaign_bars({"m": 0.05}, higher)["verdict"] == "MIDDLE"
    assert score_campaign_bars({"m": 2.0}, lower)["verdict"] == "PASS"
    assert score_campaign_bars({"m": 2.4}, lower)["verdict"] == "MIDDLE"
    assert score_campaign_bars({"m": 2.41}, lower)["verdict"] == "FAIL"


def test_score_campaign_bars_rejects_bad_specs():
    spec = {
        "metric": "m",
        "pass_at": 0.1,
        "fail_at": 0.05,
        "higher_is_better": True,
        "gating": True,
    }
    with pytest.raises(ValueError, match="at least one bar"):
        score_campaign_bars({"m": 1.0}, {})
    with pytest.raises(KeyError, match="unknown metric"):
        score_campaign_bars({"other": 1.0}, {"X": spec})
    with pytest.raises(ValueError, match="spec keys mismatch"):
        score_campaign_bars({"m": 1.0}, {"X": {**spec, "extra": 1}})
    with pytest.raises(ValueError, match="spec keys mismatch"):
        missing = {k: v for k, v in spec.items() if k != "gating"}
        score_campaign_bars({"m": 1.0}, {"X": missing})
    with pytest.raises(ValueError, match="fail_at <= pass_at"):
        score_campaign_bars({"m": 1.0}, {"X": {**spec, "fail_at": 0.2}})
    with pytest.raises(ValueError, match="pass_at <= fail_at"):
        flipped = {**spec, "higher_is_better": False, "fail_at": 0.05}
        score_campaign_bars({"m": 1.0}, {"X": flipped})


def test_resolve_warm_start_branch_matrix():
    kwargs = {"stage_run_dir": "/runs/gate", "fallback_run_dir": "/runs/ref"}
    assert resolve_warm_start_branch("PASS", **kwargs) == (
        "continue",
        "/runs/gate",
    )
    assert resolve_warm_start_branch("FAIL", **kwargs) == (
        "fallback",
        "/runs/ref",
    )
    assert resolve_warm_start_branch(
        "FAIL", stage_run_dir="/runs/gate", fallback_run_dir=None
    ) == ("stop", None)
    for undecided in ("MIDDLE", "NO_DATA"):
        assert resolve_warm_start_branch(undecided, **kwargs) == ("stop", None)
        assert resolve_warm_start_branch(
            undecided, middle_action="continue", **kwargs
        ) == ("continue", "/runs/gate")
        assert resolve_warm_start_branch(
            undecided, middle_action="fallback", **kwargs
        ) == ("fallback", "/runs/ref")
        assert resolve_warm_start_branch(
            undecided,
            middle_action="fallback",
            stage_run_dir="/runs/gate",
            fallback_run_dir=None,
        ) == ("stop", None)
    with pytest.raises(ValueError, match="middle_action"):
        resolve_warm_start_branch("PASS", middle_action="punt", **kwargs)
    with pytest.raises(ValueError, match="unknown gate verdict"):
        resolve_warm_start_branch("RECORDED", **kwargs)


def test_campaign_manifest_roundtrip_and_fingerprint(tmp_path):
    assert load_campaign_manifest(tmp_path) is None
    fingerprint = {"seed": 0, "gate_bars": {"LS-C": (0.10, 0.05)}}
    manifest = {"status": "running", "fingerprint": fingerprint, "stages": {}}
    path = write_campaign_manifest(tmp_path, manifest)
    assert path == str(tmp_path / "campaign_manifest.json")
    loaded = load_campaign_manifest(tmp_path)
    assert loaded is not None
    assert loaded["status"] == "running"

    # Tuples JSON-normalize to lists; an identical protocol must match.
    require_campaign_fingerprint(loaded, fingerprint)
    with pytest.raises(ValueError, match="seed"):
        require_campaign_fingerprint(loaded, {**fingerprint, "seed": 1})
    with pytest.raises(ValueError, match="no fingerprint"):
        require_campaign_fingerprint({"status": "running"}, fingerprint)


def test_next_stage_attempt_dir_numbers_attempts(tmp_path):
    first = next_stage_attempt_dir(tmp_path, "leg1_scratch_gate")
    assert first == str(tmp_path / "leg1_scratch_gate" / "attempt_01")
    (tmp_path / "leg1_scratch_gate" / "notes.txt").write_text("x")
    (tmp_path / "leg1_scratch_gate" / "attempt_xx").mkdir()
    second = next_stage_attempt_dir(tmp_path, "leg1_scratch_gate")
    assert second == str(tmp_path / "leg1_scratch_gate" / "attempt_02")
    with pytest.raises(ValueError, match="bare directory name"):
        next_stage_attempt_dir(tmp_path, "nested/name")


def test_score_paddle_stage_requires_finished_paddle_run(tmp_path):
    bars = {
        "LS-C": {
            "metric": "touched_after_bounce_rate",
            "pass_at": 0.10,
            "fail_at": 0.05,
            "higher_is_better": True,
            "gating": True,
        }
    }
    with pytest.raises(FileNotFoundError, match="best checkpoint"):
        score_paddle_stage(tmp_path, bars=bars)

    # Legacy flat-layout artifacts satisfy location; the recorded env
    # class is validated before any model bytes are read.
    (tmp_path / "best_model.zip").write_bytes(b"zip")
    (tmp_path / "best_vec_normalize.pkl").write_bytes(b"pkl")
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluation_env": {"class": "WallBallEnv"}})
    )
    with pytest.raises(ValueError, match="PaddleTennisEnv"):
        score_paddle_stage(tmp_path, bars=bars)

    (tmp_path / "config.json").write_text(
        json.dumps({"evaluation_env": {"class": "PaddleTennisEnv"}})
    )
    with pytest.raises(ValueError, match="constructor kwargs"):
        score_paddle_stage(tmp_path, bars=bars)

    with pytest.raises(ValueError, match="episodes"):
        score_paddle_stage(tmp_path, bars=bars, episodes=0)


def _plan_run_config(*, transfer_log_ent_coef=False, algo="sac"):
    """A recorded config.json in the LT1 shape (warm-started leg).

    ``algo="ppo"`` models the no-temperature warm start ``train()``
    records: the flag verbatim, ``log_ent_coef`` in neither
    ``transferred`` nor ``reset``, and no ``transferred_ent_coef``
    (the flag is a documented no-op there).
    """
    transferred = ["policy.state_dict", "vec_normalize.obs_rms"]
    reset = [
        "policy.optimizer_state",
        "replay_buffer" if algo == "sac" else "rollout_buffer",
    ]
    temperature_evidence = {}
    if algo == "sac":
        if transfer_log_ent_coef:
            transferred.append("log_ent_coef")
            # train() records the moved temperature's value.
            temperature_evidence["transferred_ent_coef"] = 0.014
        else:
            reset.append("log_ent_coef")
    return {
        "train_config": {
            "seed": 0,
            "total_timesteps": 1_000_000,
            "n_envs": 4,
            "eval_freq": 25_000,
            "checkpoint_freq": 100_000,
            "warm_start": {
                "source_run_dir": "/drive/PaddleTennis/sac/20260816_235141",
            },
        },
        "env": {
            "class": "PaddleTennisEnv",
            "constructor_kwargs": {
                "render_mode": "rgb_array",
                "contact_reward_scale": 0.25,
                "paddle_x_target_range": [-8.2, 0.3],
            },
        },
        "initialization": {
            "source_run_dir": "/drive/PaddleTennis/sac/20260816_235141",
            "transfer_log_ent_coef": transfer_log_ent_coef,
            **temperature_evidence,
            "transferred": transferred,
            "reset": reset,
            "source_artifacts": {
                "best_model.zip": {
                    "path": "model/best_model.zip",
                    "sha256": "838997fb" + "0" * 56,
                },
                "best_vec_normalize.pkl": {
                    "path": "model/best_vec_normalize.pkl",
                    "sha256": "d0502c14" + "1" * 56,
                },
                "config.json": {
                    "path": "config.json",
                    "sha256": "ab" * 32,
                },
            },
        },
    }


def _write_plan_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_validate_run_config_against_plan_accepts_the_lt1_shape(tmp_path):
    path = _write_plan_config(tmp_path, _plan_run_config())
    validate_run_config_against_plan(
        path,
        {
            "seed": 0,
            "total_timesteps": 1_000_000,
            "n_envs": 4,
            "eval_freq": 25_000,
            "checkpoint_freq": 100_000,
            "env_class": "PaddleTennisEnv",
            # Subset match, and a planned tuple equals its recorded
            # JSON-list form.
            "env_kwargs": {
                "contact_reward_scale": 0.25,
                "paddle_x_target_range": (-8.2, 0.3),
            },
            "warm_start": {
                "source_run_dir_suffix": "sac/20260816_235141",
                "transfer_log_ent_coef": False,
                # Prefix and full-digest pins both match; None skips.
                "expected_artifact_sha256": {
                    "best_model.zip": "838997fb",
                    "best_vec_normalize.pkl": "d0502c14" + "1" * 56,
                },
            },
        },
    )


def test_validate_run_config_against_plan_eval_env_and_drill_sha_pins(
    tmp_path,
):
    """The k=2 drill design's D6 eval-side pin and consumed-library
    digest: ``eval_env_kwargs`` subset-matches the recorded
    ``evaluation_env`` block (the [eval_env] table layers above the
    recipe's eval overrides, so the frozen plan must be able to
    assert the drill is off there), and ``drill_library_sha256``
    matches what the training env actually loaded — by full digest
    or prefix — not what the file at the path contains at audit
    time."""
    config = _plan_run_config()
    config["env"]["drill_library_sha256"] = "43975265" + "e" * 56
    config["evaluation_env"] = {
        "class": "PaddleTennisEnv",
        "constructor_kwargs": {
            "render_mode": "rgb_array",
            "drill_library": None,
            "drill_fraction": 0.0,
        },
    }
    path = _write_plan_config(tmp_path, config)
    validate_run_config_against_plan(
        path,
        {
            "eval_env_kwargs": {"drill_library": None, "drill_fraction": 0.0},
            "drill_library_sha256": "43975265",
        },
    )
    validate_run_config_against_plan(
        path, {"drill_library_sha256": "43975265" + "e" * 56}
    )
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path,
            {
                "eval_env_kwargs": {"drill_fraction": 0.5},
                "drill_library_sha256": "deadbeef",
            },
        )
    message = str(excinfo.value)
    assert "evaluation_env kwarg 'drill_fraction': expected 0.5" in message
    assert "env.drill_library_sha256: expected 'deadbeef'" in message

    # An absent evaluation_env block or consumed digest is a
    # mismatch, never a silent pass.
    bare = _plan_run_config()
    bare_path = tmp_path / "bare.json"
    bare_path.write_text(json.dumps(bare))
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            bare_path,
            {
                "eval_env_kwargs": {"drill_library": None},
                "drill_library_sha256": "43975265",
            },
        )
    message = str(excinfo.value)
    assert "evaluation_env.constructor_kwargs" in message
    assert "records no consumed drill-library digest" in message

    # A malformed pin is a bad plan, not a mismatch.
    with pytest.raises(ValueError, match="lowercase hex"):
        validate_run_config_against_plan(
            path, {"drill_library_sha256": "NOT-HEX"}
        )


def test_validate_run_config_against_plan_accepts_from_scratch(tmp_path):
    config = _plan_run_config()
    del config["initialization"]
    config["train_config"]["warm_start"] = None
    path = _write_plan_config(tmp_path, config)
    validate_run_config_against_plan(
        path, {"seed": 0, "n_envs": 4, "warm_start": None}
    )
    # An empty warm-start expectation only demands the run WAS
    # warm-started, so it fails on the from-scratch record.
    with pytest.raises(RunConfigPlanMismatch, match="no initialization block"):
        validate_run_config_against_plan(path, {"warm_start": {}})


def test_validate_run_config_against_plan_lists_every_mismatch(tmp_path):
    path = _write_plan_config(tmp_path, _plan_run_config())
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path,
            {
                "seed": 1,
                "total_timesteps": 3_000_000,
                "n_envs": 4,  # matches: not reported
                "env_class": "WallBallEnv",
                "env_kwargs": {
                    "contact_reward_scale": 0.5,
                    "hold_reward_scale": 0.25,
                },
                "warm_start": None,
            },
        )
    message = str(excinfo.value)
    # EVERY divergence in one message, not just the first.
    assert "7 place(s)" in message
    assert "train_config.seed: expected 1" in message
    assert "config.json records 0" in message
    assert "train_config.total_timesteps: expected 3000000" in message
    assert "env.class: expected 'WallBallEnv'" in message
    assert "env kwarg 'contact_reward_scale': expected 0.5" in message
    assert (
        "env kwarg 'hold_reward_scale': expected 0.25, config.json records "
        "no such kwarg"
    ) in message
    assert "expected a from-scratch run" in message
    assert "train_config.warm_start: expected None" in message
    assert "n_envs" not in message


def test_validate_run_config_against_plan_reports_absent_fields(tmp_path):
    path = _write_plan_config(tmp_path, {"train_config": {"seed": 0}})
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path,
            {
                "seed": 0,
                "checkpoint_freq": 100_000,
                "env_class": "PaddleTennisEnv",
                "env_kwargs": {"contact_reward_scale": 0.25},
            },
        )
    message = str(excinfo.value)
    assert "train_config.checkpoint_freq: expected 100000" in message
    assert "config.json records nothing" in message
    assert "records no constructor kwargs" in message
    # env_kwargs pinning nothing has nothing to mismatch against.
    validate_run_config_against_plan(
        path, {"seed": 0, "env_kwargs": {}}
    )


def test_validate_run_config_against_plan_temperature_skip_both_ways(
    tmp_path,
):
    # The run transferred the temperature (SAC auto -> auto; the moved
    # temperature leaves ``transferred_ent_coef`` behind); the plan
    # expected the skip.
    path = _write_plan_config(
        tmp_path, _plan_run_config(transfer_log_ent_coef=True)
    )
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path, {"warm_start": {"transfer_log_ent_coef": False}}
        )
    message = str(excinfo.value)
    assert "3 place(s)" in message
    assert (
        "initialization.transfer_log_ent_coef: expected False, "
        "config.json records True"
    ) in message
    assert "config.json records it transferred" in message
    assert "initialization.transferred_ent_coef: expected no moved" in message

    # And the reverse: the run skipped, the plan expected a transfer.
    # Reset membership is evidence, not a requirement -- but a recorded
    # reset must not coexist with an expected transfer.
    path = _write_plan_config(
        tmp_path, _plan_run_config(transfer_log_ent_coef=False)
    )
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path, {"warm_start": {"transfer_log_ent_coef": True}}
        )
    message = str(excinfo.value)
    assert "2 place(s)" in message
    assert (
        "initialization.transfer_log_ent_coef: expected True, "
        "config.json records False"
    ) in message
    assert "config.json records it reset" in message

    # Matching expectations pass on the same records, both ways.
    validate_run_config_against_plan(
        path, {"warm_start": {"transfer_log_ent_coef": False}}
    )
    path = _write_plan_config(
        tmp_path, _plan_run_config(transfer_log_ent_coef=True)
    )
    validate_run_config_against_plan(
        path, {"warm_start": {"transfer_log_ent_coef": True}}
    )


def test_validate_run_config_against_plan_ppo_flag_is_verbatim_only(tmp_path):
    """A PPO warm start records ``transfer_log_ent_coef`` verbatim but
    files ``log_ent_coef`` in neither ``transferred`` nor ``reset`` and
    never records ``transferred_ent_coef`` (no temperature on either
    side -- the documented no-op). Agreeing flags must pass in both
    directions; disagreeing flags must fail only via the verbatim flag
    comparison."""
    for recorded_flag in (True, False):
        path = _write_plan_config(
            tmp_path,
            _plan_run_config(
                transfer_log_ent_coef=recorded_flag, algo="ppo"
            ),
        )
        validate_run_config_against_plan(
            path, {"warm_start": {"transfer_log_ent_coef": recorded_flag}}
        )
        with pytest.raises(RunConfigPlanMismatch) as excinfo:
            validate_run_config_against_plan(
                path,
                {"warm_start": {"transfer_log_ent_coef": not recorded_flag}},
            )
        message = str(excinfo.value)
        assert "1 place(s)" in message
        assert (
            f"initialization.transfer_log_ent_coef: expected "
            f"{not recorded_flag}, config.json records {recorded_flag}"
        ) in message


def test_validate_run_config_against_plan_pins_suffix_and_shas(tmp_path):
    path = _write_plan_config(tmp_path, _plan_run_config())
    with pytest.raises(RunConfigPlanMismatch) as excinfo:
        validate_run_config_against_plan(
            path,
            {
                "warm_start": {
                    "source_run_dir_suffix": "sac/20260809_211147",
                    "expected_artifact_sha256": {
                        "best_model.zip": "deadbeef",
                        "config.json": "ab" * 32,  # matches: not reported
                    },
                }
            },
        )
    message = str(excinfo.value)
    assert "2 place(s)" in message
    assert (
        "initialization.source_run_dir: expected a path ending with "
        "'sac/20260809_211147'"
    ) in message
    assert (
        "initialization.source_artifacts['best_model.zip']: expected "
        "sha256 starting 'deadbeef'"
    ) in message

    # A pinned artifact the provenance never recorded is a mismatch.
    config = _plan_run_config()
    del config["initialization"]["source_artifacts"]["best_model.zip"]
    path = _write_plan_config(tmp_path, config)
    with pytest.raises(
        RunConfigPlanMismatch, match="config.json\\s+records none"
    ):
        validate_run_config_against_plan(
            path,
            {
                "warm_start": {
                    "expected_artifact_sha256": {"best_model.zip": "838997fb"}
                }
            },
        )
    # None pins skip the digest check entirely (the WarmStartConfig
    # default), and a trailing slash on the suffix is tolerated.
    validate_run_config_against_plan(
        path,
        {
            "warm_start": {
                "source_run_dir_suffix": "sac/20260816_235141/",
                "expected_artifact_sha256": None,
            }
        },
    )


def test_validate_run_config_against_plan_suffix_matches_whole_components(
    tmp_path,
):
    """The suffix pin binds whole path components: 'sac/<leaf>' must not
    match an evil sibling like '.../evilsac/<leaf>'."""
    config = _plan_run_config()
    config["initialization"]["source_run_dir"] = (
        "/drive/PaddleTennis/evilsac/20260816_235141"
    )
    path = _write_plan_config(tmp_path, config)
    with pytest.raises(RunConfigPlanMismatch, match="expected a path ending"):
        validate_run_config_against_plan(
            path,
            {"warm_start": {"source_run_dir_suffix": "sac/20260816_235141"}},
        )
    # Exact equality matches, and a leading slash on the pin is
    # normalized away rather than demanding an absolute record.
    config = _plan_run_config()
    path = _write_plan_config(tmp_path, config)
    validate_run_config_against_plan(
        path,
        {
            "warm_start": {
                "source_run_dir_suffix": (
                    "/drive/PaddleTennis/sac/20260816_235141"
                )
            }
        },
    )
    config["initialization"]["source_run_dir"] = "sac/20260816_235141"
    path = _write_plan_config(tmp_path, config)
    validate_run_config_against_plan(
        path,
        {"warm_start": {"source_run_dir_suffix": "sac/20260816_235141"}},
    )


def test_validate_run_config_against_plan_validates_sha_pin_values(tmp_path):
    """Plan-side pins obey the WarmStartConfig rule (lowercase hex, 8 to
    64 chars): an empty pin would prefix-match every digest, and short
    or uppercase pins silently weaken the check. A bad pin is a bad
    plan -- plain ValueError, never a mismatch entry."""
    path = _write_plan_config(tmp_path, _plan_run_config())
    for bad_pin in ("", "838997f", "838997FB"):
        with pytest.raises(
            ValueError, match="lowercase\\s+hex, 8 to 64 chars"
        ) as excinfo:
            validate_run_config_against_plan(
                path,
                {
                    "warm_start": {
                        "expected_artifact_sha256": {
                            "best_model.zip": bad_pin
                        }
                    }
                },
            )
        assert type(excinfo.value) is ValueError, bad_pin
    # A valid 8-char lowercase prefix still matches the recorded digest.
    validate_run_config_against_plan(
        path,
        {
            "warm_start": {
                "expected_artifact_sha256": {"best_model.zip": "838997fb"}
            }
        },
    )


def test_validate_run_config_against_plan_rejects_bad_plans(tmp_path):
    path = _write_plan_config(tmp_path, _plan_run_config())

    def expect_bad_plan(match, config_path, expected):
        # Bad plans and malformed configs are instrument errors: plain
        # ValueError, never the RunConfigPlanMismatch subclass the
        # campaign notebook books as config drift.
        with pytest.raises(ValueError, match=match) as excinfo:
            validate_run_config_against_plan(config_path, expected)
        assert not isinstance(excinfo.value, RunConfigPlanMismatch)

    expect_bad_plan("at least one key", path, {})
    expect_bad_plan("unknown expected-plan keys", path, {"sedd": 0})
    expect_bad_plan(
        "unknown expected warm_start keys",
        path,
        {"warm_start": {"source_dir": "x"}},
    )
    expect_bad_plan(
        "must be a bool", path, {"warm_start": {"transfer_log_ent_coef": "no"}}
    )
    expect_bad_plan("env_kwargs must be a mapping", path, {"env_kwargs": [1]})
    expect_bad_plan(
        "mapping of\\s+artifact name",
        path,
        {"warm_start": {"expected_artifact_sha256": {"a": 1}}},
    )
    expect_bad_plan("mapping or None", path, {"warm_start": "scratch"})
    # Plan validation precedes comparison: a bad pin is a bad plan even
    # when the recorded config would also mismatch (a from-scratch run).
    scratch = _plan_run_config()
    del scratch["initialization"]
    scratch_path = tmp_path / "scratch_config.json"
    scratch_path.write_text(json.dumps(scratch))
    expect_bad_plan(
        "lowercase\\s+hex, 8 to 64 chars",
        scratch_path,
        {"warm_start": {"expected_artifact_sha256": {"best_model.zip": ""}}},
    )
    (tmp_path / "list.json").write_text("[]")
    expect_bad_plan("JSON object", tmp_path / "list.json", {"seed": 0})
