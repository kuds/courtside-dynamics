"""Tests for the notebook-facing run-report and artifact-audit helpers.

``print_stage_summary`` and ``check_run_artifacts`` are the notebook's
post-training troubleshooting surface: the first replays the end-of-run
report inline, the second audits the run directory against the shared
``EXPECTED_ARTIFACTS`` registry and explains what a missing artifact
usually means. Both must degrade gracefully on partial/crashed runs.
"""
from __future__ import annotations

from courtside_dynamics.notebook_utils import (
    check_run_artifacts,
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
