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
from gymnasium import Env
from gymnasium.spaces import Box

from courtside_dynamics.notebook_utils import (
    _rollout_wall_ball_seed,
    _summarize_wall_ball_episodes,
    _wall_ball_constructor_kwargs_match,
    check_run_artifacts,
    evaluate_best_wall_ball,
    print_stage_summary,
)
from courtside_dynamics.training.artifacts import EXPECTED_ARTIFACTS


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
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "best_model.zip").write_bytes(b"x" * 2048)
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (monitor / "0.monitor.csv").write_text("#header\n")

    missing = check_run_artifacts(tmp_path)
    assert "config.json" not in missing
    assert "best_model.zip" not in missing
    assert "monitor" not in missing
    assert "final_model.zip" in missing

    out = capsys.readouterr().out
    assert "2.0 KB" in out  # file size rendered human-readably
    assert "1 file(s)" in out  # directories report their file count


def test_check_run_artifacts_all_present(tmp_path, capsys):
    for _, rel in EXPECTED_ARTIFACTS:
        full = tmp_path / rel
        if rel in ("monitor", "checkpoints", "videos"):
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
    one_bounce_recoveries: int = 0,
    one_bounce_returns: int = 0,
    style_violation_reason: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_reward": reward,
        "episode_length": length,
        "completed_returns": returns,
        "paddle_hit_count": returns + 1,
        "legal_paddle_hit_count": (
            returns + 1 if legal_paddle_hits is None else legal_paddle_hits
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
            one_bounce_recoveries=2,
            one_bounce_returns=2,
        ),
    ]

    summary = _summarize_wall_ball_episodes(rows, survival_steps=(100, 500))

    assert summary["metrics"]["legal_paddle_hit_count"]["mean"] == 1.0
    assert summary["metrics"]["one_bounce_recovery_count"]["mean"] == 1.0
    assert summary["metrics"]["one_bounce_return_count"]["mean"] == 1.0
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
    assert payload["schema_version"] == 2
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

    summary_path = tmp_path / "best_model_long_horizon_eval.json"
    episodes_path = tmp_path / "best_model_long_horizon_episodes.csv"
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
