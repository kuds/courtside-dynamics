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
