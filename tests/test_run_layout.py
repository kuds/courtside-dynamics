"""Tests for the shared run-directory layout registry.

``RUN_LAYOUT`` is the single source of truth for where every artifact
lives inside a run directory: writers resolve through ``artifact_path``
(always the new layout), readers through ``locate_artifact`` (new layout
first, legacy flat root as fallback), and ``EXPECTED_ARTIFACTS`` -- the
audit/report registry -- derives from it. These tests pin the contract
that makes every pre-0.14 run in Drive keep working unmodified.
"""

from __future__ import annotations

import os

import pytest

from courtside_dynamics.training.artifacts import (
    EXPECTED_ARTIFACTS,
    RUN_LAYOUT,
    artifact_path,
    locate_artifact,
)

_DIRECTORY_NAMES = {
    "monitor_dir",
    "tensorboard_dir",
    "videos_dir",
    "checkpoints_dir",
    "stage_bests_dir",
}


def test_layout_keeps_human_files_at_root_and_sorts_the_rest():
    """Root keeps exactly the three files a human opens first; every
    other artifact is one folder deep by audience, and progress.csv is
    directly in metrics/ (pandas metrics, not TB event data)."""
    at_root = {name for name, rel in RUN_LAYOUT.items() if os.sep not in rel}
    assert at_root == {"config", "stage_summary", "run_config_toml", "checkpoints_dir"}
    assert RUN_LAYOUT["config"] == "config.json"
    assert RUN_LAYOUT["stage_summary"] == "stage_summary.txt"
    assert RUN_LAYOUT["run_config_toml"] == "run_config.toml"
    assert RUN_LAYOUT["checkpoints_dir"] == "checkpoints"
    assert RUN_LAYOUT["progress_csv"] == "metrics/progress.csv"
    assert RUN_LAYOUT["best_model"] == "model/best_model.zip"
    assert RUN_LAYOUT["evaluations"] == "metrics/evaluations.npz"
    assert RUN_LAYOUT["best_model_video"] == "media/best_model.mp4"
    assert RUN_LAYOUT["videos_dir"] == "media/videos"
    # Only the directories are placed (filenames themselves never move),
    # so cross-run artifact identity stays basename-stable.
    basenames = [os.path.basename(rel) for rel in RUN_LAYOUT.values()]
    assert len(set(basenames)) == len(basenames)


def test_artifact_path_locate_artifact_round_trip(tmp_path):
    """Whatever ``artifact_path`` hands a writer, ``locate_artifact``
    finds again -- for every registered artifact."""
    for name in RUN_LAYOUT:
        written = artifact_path(tmp_path, name)
        assert written == os.path.join(str(tmp_path), RUN_LAYOUT[name])
        if name in _DIRECTORY_NAMES:
            # Directory artifacts are created by the resolver itself.
            assert os.path.isdir(written)
        else:
            # File artifacts get their parent directory created.
            assert os.path.isdir(os.path.dirname(written))
            with open(written, "w") as f:
                f.write("x")
        assert locate_artifact(tmp_path, name) == written


def test_locate_artifact_falls_back_to_legacy_flat_layout(tmp_path):
    """A pre-0.14 run (everything flat at the root, progress.csv inside
    tensorboard/) resolves without modification."""
    (tmp_path / "best_model.zip").write_bytes(b"x")
    (tmp_path / "eval_info.csv").write_text("timestep,metric,value\n")
    (tmp_path / "monitor").mkdir()
    (tmp_path / "tensorboard").mkdir()
    (tmp_path / "tensorboard" / "progress.csv").write_text("train/loss\n")

    assert locate_artifact(tmp_path, "best_model") == str(
        tmp_path / "best_model.zip"
    )
    assert locate_artifact(tmp_path, "eval_info_csv") == str(
        tmp_path / "eval_info.csv"
    )
    assert locate_artifact(tmp_path, "monitor_dir") == str(tmp_path / "monitor")
    # The one legacy location that is not "basename at the root".
    assert locate_artifact(tmp_path, "progress_csv") == str(
        tmp_path / "tensorboard" / "progress.csv"
    )
    assert locate_artifact(tmp_path, "evaluations") is None


def test_locate_artifact_prefers_new_layout_over_legacy(tmp_path):
    new_path = artifact_path(tmp_path, "best_model")
    with open(new_path, "w") as f:
        f.write("new")
    (tmp_path / "best_model.zip").write_text("legacy")
    assert locate_artifact(tmp_path, "best_model") == new_path


def test_unknown_artifact_name_raises():
    with pytest.raises(KeyError):
        artifact_path("/does/not/matter", "no_such_artifact")
    with pytest.raises(KeyError):
        locate_artifact("/does/not/matter", "no_such_artifact")


def test_expected_artifacts_derive_from_run_layout():
    """The audit registry is a subset of RUN_LAYOUT (no drift possible),
    covering every unconditional artifact including the four pages that
    replaced the eval_info.png mega-plot."""
    layout_items = set(RUN_LAYOUT.items())
    assert set(EXPECTED_ARTIFACTS) <= layout_items
    names = {name for name, _rel in EXPECTED_ARTIFACTS}
    assert {
        "config",
        "stage_summary",
        "best_model",
        "best_vec_normalize",
        "final_model",
        "vec_normalize",
        "monitor_dir",
        "progress_csv",
        "evaluations",
        "eval_info_csv",
        "checkpoints_dir",
        "videos_dir",
        "learning_curve",
        "training_health_plot",
        "eval_headline",
        "eval_terminations",
        "eval_rewards",
        "eval_diagnostics",
        "best_model_video",
    } <= names
    # Conditional extras stay out of the default expectations: TOML
    # copies and WallBall long-horizon outputs are recipe-dependent.
    assert "run_config_toml" not in names
    assert "best_long_eval" not in names
    assert "best_long_eval_episodes" not in names
    # The retired single mega-plot is gone from the registry entirely.
    assert "eval_info.png" not in {rel for _name, rel in EXPECTED_ARTIFACTS}
