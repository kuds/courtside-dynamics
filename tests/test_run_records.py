"""Run records, sync manifests and run-directory trust (design L1).

``docs/design_campaign_structure_20261016.md`` step L1. Every
``train()`` run directory carries ``run_record.json`` (written as
``"running"`` at start, rewritten on every terminal path) and, written
last, ``sync_manifest.json``; a directory without a manifest that
verifies is incomplete by definition. These tests pin the record and
manifest formats, ``verify_run_dir`` and its CLI, the local-then-sync
mirror, and a real ``train()`` run down every terminal path -- including
a SIGKILL, which must leave a directory that verifies as incomplete.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest
from stable_baselines3.common.callbacks import BaseCallback

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.experiments import records, sync
from courtside_dynamics.experiments.records import (
    RUN_RECORD_NAME,
    SYNC_MANIFEST_NAME,
    build_run_record,
    canonical_sha256,
    verify_run_dir,
    write_json_atomic,
    write_run_record,
    write_sync_manifest,
)
from courtside_dynamics.experiments.sync import (
    RunDirMirror,
    SyncError,
    sync_run_dir,
    validate_sync_target,
)
from courtside_dynamics.experiments.verify import main as verify_main
from courtside_dynamics.training.train import TrainConfig, train

REPOSITORY_ROOT = Path(__file__).parents[1]


# -- records and manifests ----------------------------------------------------


def test_experiments_package_imports_without_sb3_or_torch():
    """The record, manifest, mirror and verify modules are read on
    machines (and in index jobs) without the train extra."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["stable_baselines3"] = None
        sys.modules["torch"] = None
        import courtside_dynamics.experiments
        import courtside_dynamics.experiments.records
        import courtside_dynamics.experiments.sync
        import courtside_dynamics.experiments.verify
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_write_json_atomic_never_leaves_a_torn_file(tmp_path):
    target = tmp_path / "record.json"
    write_json_atomic(target, {"status": "running"})
    assert json.loads(target.read_text()) == {"status": "running"}

    class Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("serialization failed mid-write")

    with pytest.raises(RuntimeError, match="mid-write"):
        write_json_atomic(target, {"status": "completed", "bad": Unprintable()})
    # The old file stands, and no temporary file is left behind.
    assert json.loads(target.read_text()) == {"status": "running"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["record.json"]


_CONFIG: dict[str, Any] = {
    "git_sha": "a" * 40,
    "versions": {"python": "3.13.0", "stable_baselines3": "2.9.0"},
    "recipe_name": "PaddleTennis",
    "algorithm": {"name": "SAC", "family": "SAC", "implementation": "sb3==2.9.0"},
    "comparability_key": {
        "task": "PaddleTennis",
        "eval_protocol": None,
        "algo_family": "SAC",
    },
    "evaluation_seeding": {"paired": True, "eval_seed": 1_000_000},
    "env": {
        "class": "PaddleTennisEnv",
        "constructor_kwargs": {"episode_len": 1500, "contact_shaping": 0.25},
        "observation_names_sha256": "b" * 64,
    },
    "evaluation_env": {
        "class": "PaddleTennisEnv",
        "constructor_kwargs": {"episode_len": 1500},
    },
    "train_config": {
        "total_timesteps": 2_000_000,
        "seed": 0,
        "warm_start": {"source_run_dir": "/mnt/runs/PaddleTennis/sac/20260809_211147"},
    },
    "initialization": {
        "source_run_dir": "/resolved/PaddleTennis/sac/20260809_211147",
        "source_artifacts": {
            "best_model.zip": {"path": "x", "sha256": "c" * 64},
            "config.json": {"path": "y", "sha256": "d" * 64},
        },
    },
}


def test_run_record_is_built_from_config_json_and_the_outcome():
    meta = {
        "timestep": 450_000,
        "selection_keys": ["episode_rally_returns_a_ep_mean", "success_rate"],
        "selection_values": {"episode_rally_returns_a_ep_mean": 0.4},
        "n_updates": 120,
        "metrics": {"ignored": 1.0},
    }
    record = build_run_record(
        _CONFIG,
        run_id="20261017_101010",
        created_at="2026-10-17T10:10:10+00:00",
        status=records.EARLY_STOP,
        finished_at="2026-10-17T12:00:00+00:00",
        stop_reason="degenerate_signal: flat",
        trained_timesteps=500_000,
        n_updates=490_000,
        git_dirty=False,
        best_model_meta=meta,
        wall_clock={"total_seconds": 10.0},
        preflight_warnings=("w",),
        artifacts=({"name": "config", "path": "config.json"},),
    )
    assert record["schema"] == records.RUN_RECORD_SCHEMA == "courtside-run-record/1"
    assert record["run_id"] == "20261017_101010"
    assert record["status"] == "early_stop"
    assert record["budget"] == {
        "total_timesteps": 2_000_000,
        "trained_timesteps": 500_000,
        "n_updates": 490_000,
    }
    assert record["algo"] == {"name": "SAC", "family": "SAC", "impl": "sb3==2.9.0"}
    assert record["recipe"] == "PaddleTennis"
    assert record["task"] == {
        "env_class": "PaddleTennisEnv",
        "constructor_kwargs_sha256": canonical_sha256(
            {"contact_shaping": 0.25, "episode_len": 1500}
        ),
        "evaluation_env_class": "PaddleTennisEnv",
        "evaluation_constructor_kwargs_sha256": canonical_sha256(
            {"episode_len": 1500}
        ),
        "observation_names_sha256": "b" * 64,
        "eval_protocol": None,
    }
    assert record["comparability_key"] == _CONFIG["comparability_key"]
    assert record["seeds"] == {
        "seed": 0,
        "evaluation_seeding": _CONFIG["evaluation_seeding"],
    }
    assert record["git"] == {"sha": "a" * 40, "dirty": False}
    # The source run's directory name (never its full, machine-specific
    # path) and the digests the initialization block verified.
    assert record["warm_start"] == {
        "source_run_dir_name": "20260809_211147",
        "artifact_sha256": {"best_model.zip": "c" * 64, "config.json": "d" * 64},
    }
    # Without the meta's directory its sha256 bindings go unchecked.
    assert record["selection"] == {
        "timestep": 450_000,
        "keys": ["episode_rally_returns_a_ep_mean", "success_rate"],
        "values": {"episode_rally_returns_a_ep_mean": 0.4},
        "n_updates": 120,
        "bindings": "not_checked",
        "unbound_reason": None,
    }
    assert record["experiment"] == {
        "id": None,
        "parent": None,
        "lever": None,
        "kind": None,
    }
    assert record["preflight_warnings"] == ["w"]
    assert record["artifacts"] == [{"name": "config", "path": "config.json"}]
    with pytest.raises(ValueError, match="run status must be one of"):
        build_run_record(_CONFIG, run_id="r", created_at="t", status="done")


def _sealed_run_dir(path: Path, *, status: str = records.COMPLETED) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n")
    (path / "model").mkdir()
    (path / "model" / "final_model.zip").write_bytes(b"weights")
    write_run_record(
        path,
        build_run_record({}, run_id=path.name, created_at="t", status=status),
    )
    write_sync_manifest(path)
    return path


def test_a_sealed_run_dir_verifies_complete(tmp_path):
    run_dir = _sealed_run_dir(tmp_path / "run")
    manifest = json.loads((run_dir / SYNC_MANIFEST_NAME).read_text())
    assert manifest["schema"] == records.SYNC_MANIFEST_SCHEMA
    assert manifest["run_id"] == "run"
    assert [entry["path"] for entry in manifest["files"]] == [
        "config.json",
        "model/final_model.zip",
        RUN_RECORD_NAME,
    ]
    assert manifest["file_count"] == 3
    result = verify_run_dir(run_dir)
    assert result.complete
    assert result.manifest == "verified"
    assert result.status == "completed"
    assert result.files_checked == 3
    assert result.problems == ()


def test_a_run_dir_without_a_manifest_is_incomplete(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_record(
        run_dir,
        build_run_record({}, run_id="run", created_at="t", status=records.RUNNING),
    )
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert result.manifest == "missing"
    assert result.status == "running"
    assert "never reached a terminal path" in result.problems[0]
    assert not verify_run_dir(tmp_path / "absent").complete


def test_a_terminal_record_without_a_manifest_was_never_sealed(tmp_path):
    """Verifier r2: a mirror whose only (final) pass died before its
    manifest quoted the record's terminal status, then said the run never
    reached a terminal path. It was never sealed; the problem says so and
    names the re-sync."""
    run_dir = tmp_path / "mirror"
    run_dir.mkdir()
    write_run_record(
        run_dir,
        build_run_record({}, run_id="run", created_at="t", status=records.COMPLETED),
    )
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert result.manifest == "missing"
    assert result.status == "completed"
    (problem,) = result.problems
    assert problem.startswith(
        "no sync_manifest.json, though its run record says 'completed': the "
        "directory was never sealed"
    )
    assert "experiments.sync.sync_run_dir" in problem
    assert "never reached a terminal path" not in problem


def test_replace_together_renames_all_or_leaves_the_old_files(tmp_path):
    first, second = tmp_path / "a.zip", tmp_path / "b.pkl"
    first.write_bytes(b"old a")
    second.write_bytes(b"old b")
    with pytest.raises(KeyboardInterrupt):
        with records.ReplaceTogether() as staged:
            Path(staged.stage(first)).write_bytes(b"new a")
            Path(staged.stage(second)).write_bytes(b"ne")
            raise KeyboardInterrupt  # inside the second write
    assert (first.read_bytes(), second.read_bytes()) == (b"old a", b"old b")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.zip", "b.pkl"]
    with records.ReplaceTogether() as staged:
        temporary = staged.stage(first)
        assert records.is_temporary_path(Path(temporary).name)
        Path(temporary).write_bytes(b"new a")
        Path(staged.stage(second)).write_bytes(b"new b")
        staged.commit()
    assert (first.read_bytes(), second.read_bytes()) == (b"new a", b"new b")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.zip", "b.pkl"]


def test_best_model_meta_bindings_are_checked_against_the_files(tmp_path):
    (tmp_path / "best_model.zip").write_bytes(b"model")
    (tmp_path / "best_vec_normalize.pkl").write_bytes(b"stats")
    meta = {
        "timestep": 200,
        "selection_keys": ["k"],
        "selection_values": {"k": 1.0},
        "artifacts": {
            "best_model.zip": {"sha256": records.sha256_file(tmp_path / "best_model.zip")},
            "best_vec_normalize.pkl": {
                "sha256": records.sha256_file(tmp_path / "best_vec_normalize.pkl")
            },
        },
    }
    assert records.best_model_meta_unbound(meta, tmp_path) is None
    selection = records.selection_from_best_model_meta(meta, directory=tmp_path)
    assert selection is not None
    assert (selection["timestep"], selection["bindings"]) == (200, "verified")
    for damage, reason in (
        (lambda: (tmp_path / "best_vec_normalize.pkl").write_bytes(b"older"),
         "best_vec_normalize.pkl (sha256 differs)"),
        (lambda: (tmp_path / "best_vec_normalize.pkl").unlink(),
         "best_vec_normalize.pkl (missing)"),
    ):
        damage()
        assert reason in records.best_model_meta_unbound(meta, tmp_path)
        selection = records.selection_from_best_model_meta(meta, directory=tmp_path)
        assert selection == {
            "timestep": None,
            "keys": ["k"],
            "values": None,
            "n_updates": None,
            "bindings": "unbound",
            "unbound_reason": records.best_model_meta_unbound(meta, tmp_path),
        }
    # A meta from before metas bound their files: nothing to check, as
    # the loaders accept it.
    legacy = {key: value for key, value in meta.items() if key != "artifacts"}
    assert records.best_model_meta_unbound(legacy, tmp_path) is None
    selection = records.selection_from_best_model_meta(legacy, directory=tmp_path)
    assert selection is not None
    assert (selection["timestep"], selection["bindings"]) == (200, "not_checked")
    for artifacts, reason in (
        ({}, "binds no files"),
        ({"best_vec_normalize.pkl": {"sha256": "0" * 64}}, "does not bind best_model.zip"),
        ({"best_model.zip": {"sha256": "0" * 64}, "../x.pkl": {}}, "'../x.pkl' (not a file name)"),
    ):
        assert reason in records.best_model_meta_unbound(
            {**meta, "artifacts": artifacts}, tmp_path
        )
    # An unbound selection leaves the triple out of the record's artifacts.
    record = build_run_record(
        {},
        run_id="r",
        created_at="t",
        status=records.INTERRUPTED,
        best_model_meta=meta,
        best_model_dir=tmp_path,
        artifacts=(
            {"name": "best_model", "path": "model/best_model.zip"},
            {"name": "best_model_meta", "path": "model/best_model_meta.json"},
            {"name": "final_model", "path": "model/final_model.zip"},
        ),
    )
    assert record["selection"]["bindings"] == "unbound"
    assert record["artifacts"] == [
        {"name": "final_model", "path": "model/final_model.zip"}
    ]


def test_the_best_model_triple_names_are_run_layout_names():
    from courtside_dynamics.training.artifacts import RUN_LAYOUT

    assert records.BEST_MODEL_ARTIFACTS <= set(RUN_LAYOUT)
    assert {
        RUN_LAYOUT[name].rsplit("/", 1)[-1] for name in records.BEST_MODEL_ARTIFACTS
    } == {"best_model.zip", "best_vec_normalize.pkl", "best_model_meta.json"}


@pytest.mark.parametrize(
    "damage, problem",
    [
        (lambda d: (d / "config.json").write_text("{\"edited\": 1}\n"), "bytes"),
        (lambda d: (d / "config.json").write_text("[]\n"), "sha256"),
        (lambda d: (d / "model" / "final_model.zip").unlink(), "missing"),
    ],
)
def test_a_listed_file_that_changed_makes_the_run_dir_incomplete(
    tmp_path, damage, problem
):
    run_dir = _sealed_run_dir(tmp_path / "run")
    damage(run_dir)
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert result.manifest == "mismatch"
    assert [m.problem for m in result.mismatches] == [problem]


def test_unlisted_files_are_reported_and_fail_only_strict(tmp_path):
    """A notebook's post-training plots and replay video land beside a
    sealed run: they are listed as unlisted, and only ``strict``
    refuses them."""
    run_dir = _sealed_run_dir(tmp_path / "run")
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "learning_curve.png").write_bytes(b"png")
    (run_dir / "reports" / "eval_headline_2.png").write_bytes(b"png")
    (run_dir / "media").mkdir()
    (run_dir / "media" / "best_model.mp4").write_bytes(b"mp4")
    lenient = verify_run_dir(run_dir)
    assert lenient.complete
    assert lenient.unlisted == (
        "media/best_model.mp4",
        "reports/eval_headline_2.png",
        "reports/learning_curve.png",
    )
    strict = verify_run_dir(run_dir, strict=True)
    assert not strict.complete
    assert "strict" in strict.problems[0]


@pytest.mark.parametrize(
    "stray",
    [
        "stray.txt",
        "model/stray.zip",
        ".hidden_tmp",
        "metrics/x/deep.bin",
        "media/videos/late.mp4",
    ],
)
def test_an_unlisted_file_outside_the_post_training_locations_fails(
    tmp_path, stray
):
    """Verifier r1: by default every stray file passed with exit code 0,
    an interrupted write's hidden leftover included. Only the
    post-training locations are tolerated now."""
    run_dir = _sealed_run_dir(tmp_path / "run")
    path = run_dir / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert result.manifest == "verified"
    assert result.unlisted == (stray,)
    assert stray in result.problems[0]
    assert "outside the post-training locations" in result.problems[0]
    assert verify_main([str(run_dir)]) == 1


@pytest.mark.parametrize("listed", [False, True], ids=["unlisted", "listed"])
@pytest.mark.parametrize(
    "leftover",
    [
        ".run_record.json.tmp-4242",
        "model/.best_model.zip.sync-tmp",
        "model/.best_model.zip.tmp-4242",
        "metrics/evaluations.npz.tmp.npz",
        # Even where plots may land.
        "reports/.learning_curve.png.tmp-17",
        "reports/tmpk2j3h4.tmp",
    ],
)
def test_a_temporary_leftover_fails_whether_or_not_the_manifest_lists_it(
    tmp_path, leftover, listed
):
    """Verifier r2 (major): the leftover rule applied only to unlisted
    files, so a leftover present when train() sealed the run -- listed
    by its own manifest -- verified complete under --strict. A leftover
    is an interrupted write wherever it sits and whoever listed it."""
    run_dir = _sealed_run_dir(tmp_path / "run")
    path = run_dir / leftover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if listed:
        write_sync_manifest(run_dir)
    for strict in (False, True):
        result = verify_run_dir(run_dir, strict=strict)
        assert not result.complete
        assert result.manifest == "verified"
        assert result.unlisted == (() if listed else (leftover,))
        assert result.problems == (
            f"1 temporary file(s) left by an interrupted write, listed in "
            f"{SYNC_MANIFEST_NAME} or not: {leftover}",
        )
    assert verify_main([str(run_dir)]) == 1


def test_the_post_training_locations_are_the_notebooks_layout():
    """What a notebook writes after train() (RUN_LAYOUT's plots, audits
    and replay video) is tolerated; nothing train() itself writes is."""
    from courtside_dynamics.training.artifacts import RUN_LAYOUT

    post_training = {
        "learning_curve",
        "training_health_plot",
        "eval_headline",
        "eval_terminations",
        "eval_rewards",
        "eval_diagnostics",
        "best_long_eval",
        "best_long_eval_episodes",
        "best_model_video",
    }
    for name, relative in RUN_LAYOUT.items():
        expected = name in post_training or (
            # Written during training, but under reports/ like the plots.
            name in ("curriculum_stages", "ladder_certification", "diagnosis_dir")
        )
        probe = relative if os.path.splitext(relative)[1] else f"{relative}/x.json"
        assert records.is_post_training_path(probe) is expected, name


def test_an_intact_snapshot_of_an_unfinished_run_is_incomplete(tmp_path):
    """A mid-run mirror: its manifest verifies, but the record it lists
    says the run was still training."""
    run_dir = _sealed_run_dir(tmp_path / "run", status=records.RUNNING)
    result = verify_run_dir(run_dir)
    assert result.manifest == "verified"
    assert result.status == "running"
    assert not result.complete
    assert "had not finished" in result.problems[0]


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "../outside", "bytes": 1, "sha256": "0" * 64}],
        [{"path": "/etc/passwd", "bytes": 1, "sha256": "0" * 64}],
        [{"path": "a", "bytes": -1, "sha256": "0" * 64}],
        [{"path": "a", "bytes": 1, "sha256": "short"}],
        "not-a-list",
    ],
)
def test_a_malformed_manifest_is_unreadable(tmp_path, files):
    run_dir = _sealed_run_dir(tmp_path / "run")
    manifest = json.loads((run_dir / SYNC_MANIFEST_NAME).read_text())
    manifest["files"] = files
    (run_dir / SYNC_MANIFEST_NAME).write_text(json.dumps(manifest))
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert result.manifest == "unreadable"


def test_a_manifest_that_omits_the_run_record_is_incomplete(tmp_path):
    run_dir = _sealed_run_dir(tmp_path / "run")
    manifest = json.loads((run_dir / SYNC_MANIFEST_NAME).read_text())
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != RUN_RECORD_NAME
    ]
    (run_dir / SYNC_MANIFEST_NAME).write_text(json.dumps(manifest))
    result = verify_run_dir(run_dir)
    assert not result.complete
    assert any("does not list run_record.json" in p for p in result.problems)


def test_verify_cli_exits_zero_only_when_every_run_dir_is_complete(
    tmp_path, capsys
):
    good = _sealed_run_dir(tmp_path / "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    assert verify_main([str(good)]) == 0
    assert "complete (status completed, 3 file(s) verified)" in capsys.readouterr().out
    assert verify_main([str(good), str(bad)]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out
    assert verify_main(["--json", str(bad)]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["complete"] is False and verdict["manifest"] == "missing"
    # The documented module entry point.
    result = subprocess.run(
        [sys.executable, "-m", "courtside_dynamics.experiments.verify", str(good)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# -- the local-then-sync mirror ------------------------------------------------


def test_validate_sync_target_refuses_before_any_output(tmp_path):
    log_dir = tmp_path / "local" / "run"
    assert validate_sync_target(log_dir, None) is None
    assert validate_sync_target(log_dir, tmp_path / "mirror") == (
        tmp_path / "mirror"
    ).resolve()
    (tmp_path / "empty").mkdir()
    assert validate_sync_target(log_dir, tmp_path / "empty") is not None
    with pytest.raises(ValueError, match="separate directory"):
        validate_sync_target(log_dir, log_dir)
    with pytest.raises(ValueError, match="separate directory"):
        validate_sync_target(log_dir, log_dir / "mirror")
    with pytest.raises(ValueError, match="separate directory"):
        validate_sync_target(log_dir, tmp_path / "local")
    with pytest.raises(ValueError, match="parent .* does not exist"):
        validate_sync_target(log_dir, tmp_path / "unmounted" / "drive" / "run")
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "other_run.txt").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        validate_sync_target(log_dir, tmp_path / "used")
    with pytest.raises(TypeError, match="sync_dir must be"):
        validate_sync_target(log_dir, 3)


def test_mirror_copies_changes_only_and_seals_each_pass(tmp_path, monkeypatch):
    source = _sealed_run_dir(tmp_path / "run", status=records.RUNNING)
    (source / "sync_manifest.json").unlink()  # mid-run: no local manifest yet
    mirror_dir = tmp_path / "mirror" / "run"
    mirror = RunDirMirror(source, mirror_dir)
    copied: list[str] = []
    copy_and_hash = sync._copy_and_hash

    def spy(src: Path, dst: Path):
        copied.append(src.relative_to(source).as_posix())
        return copy_and_hash(src, dst)

    monkeypatch.setattr(sync, "_copy_and_hash", spy)
    mirror.sync()
    assert copied == ["config.json", "model/final_model.zip", RUN_RECORD_NAME]
    first = verify_run_dir(mirror_dir)
    assert first.manifest == "verified" and first.status == "running"

    copied.clear()
    time.sleep(0.01)
    (source / "model" / "best_model.zip").write_bytes(b"best")
    write_run_record(
        source,
        build_run_record({}, run_id="run", created_at="t", status=records.COMPLETED),
    )
    write_sync_manifest(source)
    leftover = mirror_dir / "model" / ".best_model.zip.sync-tmp"
    leftover.write_text("left by a copy a failed pass cut short")
    mirror.sync()
    # Only what changed, and the source's own manifest is never copied.
    assert copied == ["model/best_model.zip", RUN_RECORD_NAME]
    assert not leftover.exists()
    final = verify_run_dir(mirror_dir, strict=True)
    assert final.complete and final.status == "completed"
    assert mirror.passes == 2 and mirror.failures == 0


def test_a_failed_mirror_pass_keeps_the_previous_manifest_and_never_verifies(
    tmp_path, monkeypatch
):
    """The final pass dies after copying some files (a Drive error): the
    mirror keeps the checkpoint pass's manifest, which no longer matches
    what was overwritten -- incomplete, never 'complete'."""
    source = _sealed_run_dir(tmp_path / "run", status=records.RUNNING)
    (source / SYNC_MANIFEST_NAME).unlink()
    mirror_dir = tmp_path / "mirror"
    mirror = RunDirMirror(source, mirror_dir)
    mirror.sync()
    checkpoint_manifest = (mirror_dir / SYNC_MANIFEST_NAME).read_text()

    (source / "config.json").write_text('{"final": true}\n')
    write_run_record(
        source,
        build_run_record({}, run_id="run", created_at="t", status=records.COMPLETED),
    )
    copy_and_hash = sync._copy_and_hash

    def drive_error(src: Path, dst: Path):
        if src.name == RUN_RECORD_NAME:
            raise OSError("Transport endpoint is not connected")
        return copy_and_hash(src, dst)

    monkeypatch.setattr(sync, "_copy_and_hash", drive_error)
    with pytest.raises(OSError, match="Transport endpoint"):
        mirror.sync()
    assert mirror.failures == 1
    assert (mirror_dir / SYNC_MANIFEST_NAME).read_text() == checkpoint_manifest
    result = verify_run_dir(mirror_dir)
    assert not result.complete
    assert result.manifest == "mismatch"
    assert [m.path for m in result.mismatches] == ["config.json"]

    # The next pass recopies what failed and seals the mirror.
    monkeypatch.setattr(sync, "_copy_and_hash", copy_and_hash)
    mirror.sync()
    assert verify_run_dir(mirror_dir).complete


def test_a_mirror_never_deletes_files_it_did_not_write(tmp_path):
    """Verifier r1: sync_run_dir (the public manual re-sync) deleted
    every file under its destination that the source lacked, so one
    slip in the path -- the Drive root -- wiped unrelated files without
    a word. A pass now refuses, before writing anything."""
    source = _sealed_run_dir(tmp_path / "local" / "20261017_000001")
    drive = tmp_path / "MyDrive"
    unrelated = [
        drive / "other_project" / "notes.txt",
        drive / "photo.jpg",
        drive / "training_runs" / "PaddleTennis" / "sac" / "20260810_211754"
        / "model" / "best_model.zip",
    ]
    for path in unrelated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep me")
    before = sorted(records.run_files(drive))
    with pytest.raises(SyncError, match=r"3 file\(s\) this run's mirror did not write"):
        sync_run_dir(source, drive)
    assert sorted(records.run_files(drive)) == before
    assert not (drive / SYNC_MANIFEST_NAME).exists()
    for path in unrelated:
        assert path.read_bytes() == b"keep me"

    # A file of the same name that differs from the source's is not
    # overwritten either.
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text('{"someone": "else"}\n')
    with pytest.raises(SyncError, match="config.json"):
        sync_run_dir(source, other)
    assert (other / "config.json").read_text() == '{"someone": "else"}\n'


def test_a_mirror_refuses_another_runs_mirror(tmp_path):
    first = _sealed_run_dir(tmp_path / "local" / "a")
    second = _sealed_run_dir(tmp_path / "local" / "b")
    mirror_dir = tmp_path / "drive" / "a"
    sync_run_dir(first, mirror_dir)
    sealed = (mirror_dir / SYNC_MANIFEST_NAME).read_text()
    with pytest.raises(SyncError, match="the mirror of run 'a', not 'b'"):
        sync_run_dir(second, mirror_dir)
    assert (mirror_dir / SYNC_MANIFEST_NAME).read_text() == sealed
    (mirror_dir / SYNC_MANIFEST_NAME).write_text("{")
    with pytest.raises(SyncError, match="cannot read"):
        sync_run_dir(first, mirror_dir)
    with pytest.raises(ValueError, match="neither may contain the other"):
        RunDirMirror(first, first / "mirror")
    with pytest.raises(ValueError, match="neither may contain the other"):
        RunDirMirror(first, first.parent)


def test_a_manual_resync_finishes_a_mirror_a_failed_pass_left_unsealed(
    tmp_path, monkeypatch
):
    """The final mirror failed before any manifest existed (no
    checkpoint pass, a Drive error mid-copy): the copies it left are
    byte-identical to the source, so the re-sync the error message
    names may overwrite them and completes the mirror."""
    source = _sealed_run_dir(tmp_path / "run")
    mirror_dir = tmp_path / "drive" / "run"
    copy_and_hash = sync._copy_and_hash

    def dies_on_the_record(src: Path, dst: Path):
        if src.name == RUN_RECORD_NAME:
            raise OSError("Transport endpoint is not connected")
        return copy_and_hash(src, dst)

    monkeypatch.setattr(sync, "_copy_and_hash", dies_on_the_record)
    with pytest.raises(OSError, match="Transport endpoint"):
        RunDirMirror(source, mirror_dir).sync()
    assert (mirror_dir / "config.json").is_file()
    assert not (mirror_dir / SYNC_MANIFEST_NAME).exists()
    monkeypatch.setattr(sync, "_copy_and_hash", copy_and_hash)
    sync_run_dir(source, mirror_dir)
    assert verify_run_dir(mirror_dir, strict=True).complete


def test_the_final_pass_redoes_a_mirror_its_change_cache_left_stale(tmp_path):
    """Verifier r1: a file rewritten with the same size and mtime_ns
    kept its stale copy, and the mirror's manifest repeated the cached
    digest, so the stale mirror verified complete. train()'s final pass
    checks the mirror's file list against the sealed local manifest and
    redoes the pass without the cache when they differ."""
    import importlib

    train_module = importlib.import_module("courtside_dynamics.training.train")
    source = tmp_path / "run"
    source.mkdir()
    (source / "a.bin").write_bytes(b"AAAA")
    mirror = RunDirMirror(source, tmp_path / "drive" / "run")
    mirror.sync()
    stat = (source / "a.bin").stat()
    (source / "a.bin").write_bytes(b"BBBB")
    os.utime(source / "a.bin", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    write_run_record(
        source,
        build_run_record({}, run_id="run", created_at="t", status=records.COMPLETED),
    )
    write_sync_manifest(source)
    # The bare cached pass keeps the stale bytes ...
    assert mirror.sync()["files"] != json.loads(
        (source / SYNC_MANIFEST_NAME).read_text()
    )["files"]
    assert (mirror.destination / "a.bin").read_bytes() == b"AAAA"
    # ... the final pass does not.
    lifecycle = object.__new__(train_module._RunLifecycle)
    lifecycle.log_dir = source
    lifecycle.mirror = mirror
    lifecycle._final_pass()
    assert (mirror.destination / "a.bin").read_bytes() == b"BBBB"
    assert verify_run_dir(mirror.destination, strict=True).complete


def test_a_copy_that_does_not_rehash_fails_the_pass(tmp_path, monkeypatch):
    source = _sealed_run_dir(tmp_path / "run")
    real_sha256_file = sync.sha256_file

    def corrupted(path):
        digest = real_sha256_file(path)
        return "0" * 64 if Path(path).name == "config.json" else digest

    monkeypatch.setattr(sync, "sha256_file", corrupted)
    with pytest.raises(SyncError, match="does not re-hash"):
        sync_run_dir(source, tmp_path / "mirror")
    assert not (tmp_path / "mirror" / SYNC_MANIFEST_NAME).exists()
    assert not verify_run_dir(tmp_path / "mirror").complete


# -- train() down every terminal path ---------------------------------------------


class _StepsAlive(gym.Wrapper):
    """BallBalance plus a ``steps_alive`` counter for headline selection."""

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._alive = 0
        return observation, {**info, "steps_alive": 0}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._alive += 1
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "steps_alive": self._alive},
        )


class _ZeroAlive(_StepsAlive):
    """A ``steps_alive`` that never moves: a provably dead signal."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, {**info, "steps_alive": 0}


def _steps_alive_env() -> gym.Env:
    return _StepsAlive(BallBalanceEnv(episode_len=12))


class _RaiseAt(BaseCallback):
    def __init__(self, exc: BaseException, at: int = 64) -> None:
        super().__init__()
        self.exc = exc
        self.at = at

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.at:
            raise self.exc
        return True


def _cfg(log_dir: Path, **overrides: Any) -> TrainConfig:
    """A tiny SAC run with headline selection (steps_alive saturates at
    the 12-step episode, so its selection score is flat)."""
    fields: dict[str, Any] = {
        "env_fn": _steps_alive_env,
        "algo": "SAC",
        "total_timesteps": 400,
        "log_dir": str(log_dir),
        "n_envs": 1,
        "seed": 0,
        "eval_freq": 100,
        "checkpoint_freq": 0,
        "video_freq": 0,
        "record_video": False,
        "normalize_obs": True,
        "n_eval_episodes": 2,
        "headline_key": "steps_alive",
        "model_kwargs": {"learning_starts": 16, "buffer_size": 500, "batch_size": 32},
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def _record(log_dir: Path) -> dict[str, Any]:
    return json.loads((log_dir / RUN_RECORD_NAME).read_text())


def _assert_sealed(log_dir: Path, status: str) -> dict[str, Any]:
    result = verify_run_dir(log_dir, strict=True)
    assert result.complete, result.problems
    assert result.status == status
    record = _record(log_dir)
    assert record["status"] == status
    assert record["finished_at"] is not None
    assert record["run_id"] == log_dir.name
    # The record and the config it summarizes agree.
    config = json.loads((log_dir / "config.json").read_text())
    assert record["comparability_key"] == config["comparability_key"]
    assert record["algo"]["impl"] == config["algorithm"]["implementation"]
    assert record["task"]["env_class"] == config["env"]["class"]
    assert record["git"]["sha"] == config["git_sha"]
    assert record["experiment"] == {
        "id": None,
        "parent": None,
        "lever": None,
        "kind": None,
    }
    return record


def test_completed_run_is_recorded_and_sealed(tmp_path):
    log_dir = tmp_path / "20261017_000001"
    model = train(_cfg(log_dir))
    record = _assert_sealed(log_dir, "completed")
    assert record["stop_reason"] is None
    assert record["budget"]["total_timesteps"] == 400
    assert record["budget"]["trained_timesteps"] == model.num_timesteps == 400
    assert record["budget"]["n_updates"] == model._n_updates
    assert record["selection"]["keys"][0] == "steps_alive_ep_mean"
    assert record["selection"]["timestep"] == 100
    wall_clock = record["wall_clock"]
    assert wall_clock["total_seconds"] > 0
    assert wall_clock["periodic_eval_seconds"] > 0
    assert wall_clock["closing_eval_seconds"] > 0
    assert 0 < wall_clock["eval_share"] < 1
    names = {artifact["name"] for artifact in record["artifacts"]}
    assert {"config", "stage_summary", "final_model", "best_model"} <= names
    assert record["sync"] is None
    assert record["preflight_warnings"] == []


def test_patience_early_stop_is_recorded(tmp_path):
    log_dir = tmp_path / "patience"
    model = train(
        _cfg(
            log_dir,
            total_timesteps=2_000,
            early_stop_patience=1,
            best_metric_keys=("steps_alive_ep_mean",),
        )
    )
    assert model.num_timesteps < 2_000
    record = _assert_sealed(log_dir, "early_stop")
    assert record["stop_reason"].startswith("early_stop_patience: no improvement")
    assert record["budget"]["trained_timesteps"] == model.num_timesteps


def test_degenerate_stop_is_recorded(tmp_path):
    log_dir = tmp_path / "degenerate"
    model = train(
        _cfg(
            log_dir,
            total_timesteps=2_000,
            best_metric_keys=("steps_alive_ep_mean",),
            early_stop_degenerate_evals=1,
            degenerate_guard_keys=("steps_alive_ep_mean",),
            env_fn=lambda: _ZeroAlive(BallBalanceEnv(episode_len=12)),
        )
    )
    assert model.num_timesteps < 2_000
    record = _assert_sealed(log_dir, "early_stop")
    assert record["stop_reason"].startswith("degenerate_signal:")


def test_keyboard_interrupt_is_recorded_and_sealed(tmp_path):
    log_dir = tmp_path / "interrupted"
    model = train(_cfg(log_dir, extra_callbacks=(_RaiseAt(KeyboardInterrupt()),)))
    record = _assert_sealed(log_dir, "interrupted")
    assert record["stop_reason"] == (
        f"interrupted: KeyboardInterrupt at {model.num_timesteps:,} of 400 steps"
    )
    assert record["budget"]["trained_timesteps"] == model.num_timesteps < 400
    summary = (log_dir / "stage_summary.txt").read_text()
    assert "Status:         interrupted" in summary
    assert "interrupted: KeyboardInterrupt" in summary


def test_crash_is_recorded_and_sealed_and_still_raises(tmp_path):
    log_dir = tmp_path / "crashed"
    with pytest.raises(RuntimeError, match="synthetic crash"):
        train(_cfg(log_dir, extra_callbacks=(_RaiseAt(RuntimeError("synthetic crash")),)))
    record = _assert_sealed(log_dir, "crashed")
    assert "synthetic crash" in record["stop_reason"]
    assert record["stop_reason"].startswith("crashed: RuntimeError(")
    assert 64 <= record["budget"]["trained_timesteps"] < 400


def test_a_failure_after_config_json_but_before_learn_is_recorded(tmp_path):
    """The selection-key probe runs once the evaluators exist, after
    config.json: its refusal is a terminal path too."""
    log_dir = tmp_path / "refused"
    with pytest.raises(ValueError, match="steps_alvie"):
        train(_cfg(log_dir, headline_key="steps_alvie"))
    record = _assert_sealed(log_dir, "crashed")
    assert record["stop_reason"].startswith("crashed: ValueError(")
    assert record["budget"]["trained_timesteps"] is None


@pytest.mark.parametrize("closing", ["sb3_logger", "train_env"])
def test_a_failure_while_closing_an_ended_run_is_recorded(
    tmp_path, monkeypatch, closing
):
    """Verifier r1 (blocker): an exception after the run reported its
    outcome -- an env close in the inner finally, the SB3 logger's
    TensorBoard and progress.csv flush in the outer one, straight onto
    the Drive mount by default -- escaped train() while the record
    said completed with no stop_reason, and the directory verified
    complete. The record now says crashed, quoting the outcome."""
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.vec_env import VecNormalize

    owner, real_close = (
        (Logger, Logger.close) if closing == "sb3_logger" else (VecNormalize, VecNormalize.close)
    )
    failed = {"n": 0}

    def close_then_fail(self):
        real_close(self)
        if not failed["n"]:
            failed["n"] += 1
            raise OSError("Transport endpoint is not connected")

    monkeypatch.setattr(owner, "close", close_then_fail)
    log_dir = tmp_path / closing
    with pytest.raises(OSError, match="Transport endpoint"):
        train(_cfg(log_dir))
    record = _assert_sealed(log_dir, "crashed")
    assert record["stop_reason"].startswith(
        "crashed after the run ended as completed: OSError("
    )
    assert "Transport endpoint is not connected" in record["stop_reason"]
    assert record["budget"]["trained_timesteps"] == 400


def test_a_reused_log_dir_that_crashes_early_records_only_its_own_attempt(
    tmp_path,
):
    """Verifier r1: reuse_log_dir rotated the CSVs and logs but not the
    model files, so an attempt that crashed before its first
    evaluation recorded the previous attempt's selection (timestep 100
    for a run that trained 50 steps) and its best and final models as
    its own artifacts."""
    log_dir = tmp_path / "run"
    train(_cfg(log_dir, total_timesteps=300))
    assert _record(log_dir)["selection"]["timestep"] == 100
    with pytest.raises(RuntimeError, match="early crash"):
        train(
            _cfg(
                log_dir,
                total_timesteps=300,
                reuse_log_dir=True,
                extra_callbacks=(_RaiseAt(RuntimeError("early crash"), at=50),),
            )
        )
    record = _assert_sealed(log_dir, "crashed")
    assert record["budget"]["trained_timesteps"] < 100
    assert record["selection"] is None
    names = {artifact["name"] for artifact in record["artifacts"]}
    assert not names & {"best_model", "best_model_meta", "best_vec_normalize"}
    # The crash salvage's own model and normalizer are this attempt's.
    assert {"final_model", "vec_normalize"} <= names
    assert len(list((log_dir / "model").glob("best_model.attempt_*.zip"))) == 1


class _Improving(gym.Wrapper):
    """BallBalance whose ``steps_alive`` (and, for reward selection, whose
    reward) is the env's reset count: every evaluation scores higher than
    the last, so every evaluation saves a new best."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._resets = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._resets += 1
        return observation, {**info, "steps_alive": self._resets}

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            float(self._resets),
            terminated,
            truncated,
            {**info, "steps_alive": self._resets},
        )


def _improving_env() -> gym.Env:
    return _Improving(BallBalanceEnv(episode_len=12))


def _interrupt_best_save(monkeypatch, boundary: str, *, at_save: int = 2) -> None:
    """Raise KeyboardInterrupt at ``boundary`` of the ``at_save``-th best save.

    Inside a write (``model``, ``normalizer``, ``meta``) the temporary is
    left torn -- half its bytes -- as a real interrupt would; between the
    renames (``model_renamed``: the zip is in place, the normalizer and
    meta are not; ``normalizer_renamed``: only the meta is not) and after
    the last one (``meta_renamed``).
    """
    import stable_baselines3
    from stable_baselines3.common.vec_env import VecNormalize

    calls: dict[str, int] = {}

    def due(name: str) -> bool:
        calls[name] = calls.get(name, 0) + 1
        return calls[name] == at_save

    def tear(path: str) -> None:
        size = os.path.getsize(path)
        with open(path, "r+b") as handle:
            handle.truncate(size // 2)
        raise KeyboardInterrupt

    def temporary_of(path: Any, name: str) -> bool:
        return os.path.basename(str(path)).startswith(f".{name}.tmp-")

    for owner in (stable_baselines3.SAC, VecNormalize):
        written = "best_model.zip" if owner is not VecNormalize else (
            "best_vec_normalize.pkl"
        )
        real_save = owner.save

        def save(self, path, *args, _real=real_save, _name=written, **kwargs):
            _real(self, path, *args, **kwargs)
            if temporary_of(path, _name) and due(_name) and boundary == (
                "model" if _name == "best_model.zip" else "normalizer"
            ):
                tear(str(path))

        monkeypatch.setattr(owner, "save", save)

    real_dump = json.dump

    def dump(payload, stream, *args, **kwargs):
        real_dump(payload, stream, *args, **kwargs)
        name = getattr(stream, "name", "")
        if temporary_of(name, "best_model_meta.json") and due("meta") and (
            boundary == "meta"
        ):
            stream.flush()
            tear(name)

    monkeypatch.setattr(json, "dump", dump)

    real_replace = os.replace
    before = {
        "model_renamed": "best_vec_normalize.pkl",
        "normalizer_renamed": "best_model_meta.json",
    }

    def replace(source, target, *args, **kwargs):
        if boundary in before and temporary_of(source, before[boundary]):
            if due(f"rename {before[boundary]}"):
                raise KeyboardInterrupt
        real_replace(source, target, *args, **kwargs)
        if boundary == "meta_renamed" and temporary_of(
            source, "best_model_meta.json"
        ):
            if due("renamed meta"):
                raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace)


def _temporaries(log_dir: Path) -> list[str]:
    return [
        relative
        for relative in records.run_files(log_dir)
        if records.is_temporary_path(relative)
    ]


@pytest.mark.parametrize(
    "boundary",
    [
        "model",
        "normalizer",
        "meta",
        "model_renamed",
        "normalizer_renamed",
        "meta_renamed",
    ],
)
def test_an_interrupt_at_each_best_save_boundary_never_seals_a_torn_best(
    tmp_path, monkeypatch, boundary
):
    """Verifier r2 (majors 1 and 2): a KeyboardInterrupt inside the
    best-model save left a torn best_model.zip that the record listed and
    the manifest sealed complete, and one between the triple's writes
    left a record whose selection described the previous best beside the
    new model. Every write now goes to a temporary that is renamed into
    place, the meta last: an interrupt inside a write leaves the previous
    best whole; one between the renames leaves a meta that no longer
    binds the files, so the record's selection is unbound (never the
    previous best's) and the triple is left out of its artifacts."""
    from stable_baselines3 import SAC

    from courtside_dynamics.notebook_utils import verify_best_checkpoint_pair

    _interrupt_best_save(monkeypatch, boundary)
    log_dir = tmp_path / boundary
    model = train(_cfg(log_dir, env_fn=_improving_env))
    assert model.num_timesteps == 200
    record = _assert_sealed(log_dir, "interrupted")
    assert _temporaries(log_dir) == []
    selection = record["selection"]
    names = {artifact["name"] for artifact in record["artifacts"]}
    summary = (log_dir / "stage_summary.txt").read_text()
    if boundary in ("model_renamed", "normalizer_renamed"):
        assert selection["bindings"] == "unbound"
        assert selection["timestep"] is None
        assert selection["values"] is None
        assert "best_model.zip (sha256 differs)" in selection["unbound_reason"]
        assert not names & {"best_model", "best_vec_normalize", "best_model_meta"}
        assert "Best model:     unbound: best_model_meta.json does not bind" in (
            summary
        )
        with pytest.raises(ValueError, match="does not match the SHA-256"):
            verify_best_checkpoint_pair(log_dir)
        return
    # The previous best (step 100) stands whole, or the new one (step
    # 200) when every rename had landed.
    expected = 200 if boundary == "meta_renamed" else 100
    assert selection["bindings"] == "verified"
    assert selection["unbound_reason"] is None
    assert selection["timestep"] == expected
    assert {"best_model", "best_vec_normalize", "best_model_meta"} <= names
    assert f"Best model:     step {expected:,}" in summary
    verify_best_checkpoint_pair(log_dir)
    SAC.load(log_dir / "model" / "best_model.zip", device="cpu")


@pytest.mark.parametrize("boundary", ["model", "normalizer"])
def test_an_interrupted_reward_path_best_save_leaves_the_previous_pair(
    tmp_path, monkeypatch, boundary
):
    """Reward selection: SB3's EvalCallback wrote best_model.zip in place,
    and a real SIGINT left a 16 kB zip that no loader could open, sealed
    complete (verifier r2). The pair is now written atomically too."""
    from stable_baselines3 import SAC

    _interrupt_best_save(monkeypatch, boundary)
    log_dir = tmp_path / boundary
    model = train(
        _cfg(
            log_dir,
            env_fn=_improving_env,
            info_dict_eval=False,
            headline_key=None,
        )
    )
    assert model.num_timesteps == 200
    record = _assert_sealed(log_dir, "interrupted")
    assert _temporaries(log_dir) == []
    assert record["selection"] is None
    names = {artifact["name"] for artifact in record["artifacts"]}
    assert {"best_model", "best_vec_normalize"} <= names
    best = SAC.load(log_dir / "model" / "best_model.zip", device="cpu")
    first = SAC.load(log_dir / "model" / "final_model.zip", device="cpu")
    # The step-100 best, not the step-200 policy the salvage saved.
    assert best.num_timesteps == 100 < first.num_timesteps


def test_an_interrupted_merged_stream_npz_write_leaves_no_temporary(
    tmp_path, monkeypatch
):
    """Verifier r2 (major 3): the merged-stream evaluations.npz writer
    caught only OSError, so a KeyboardInterrupt inside np.savez left
    metrics/evaluations.npz.tmp.npz, which the manifest listed and
    verification passed. It is removed now, and a listed leftover would
    fail verification anyway."""
    import numpy as np

    real_savez = np.savez
    calls = {"n": 0}

    def savez(file, *args, **kwargs):
        if str(file).endswith(".tmp.npz"):
            calls["n"] += 1
            if calls["n"] == 2:
                with open(file, "wb") as handle:
                    handle.write(b"PK\x03\x04 torn")
                raise KeyboardInterrupt
        return real_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez", savez)
    log_dir = tmp_path / "merged"
    model = train(_cfg(log_dir, final_info_eval=True))
    assert calls["n"] == 2
    assert model.num_timesteps == 200
    _assert_sealed(log_dir, "interrupted")
    assert _temporaries(log_dir) == []
    with np.load(log_dir / "metrics" / "evaluations.npz") as data:
        assert list(data["timesteps"]) == [100]


def test_a_sigkilled_run_leaves_an_incomplete_run_dir(tmp_path):
    """No terminal path runs under SIGKILL (a dead Colab VM): the record
    still says "running" and there is no manifest."""
    log_dir = tmp_path / "killed"
    script = textwrap.dedent(
        f"""
        from courtside_dynamics.envs import BallBalanceEnv
        from courtside_dynamics.training import TrainConfig, train

        train(TrainConfig(
            env_fn=lambda: BallBalanceEnv(episode_len=12),
            algo="SAC",
            total_timesteps=10_000_000,
            log_dir={str(log_dir)!r},
            n_envs=1,
            seed=0,
            eval_freq=10_000_000,
            checkpoint_freq=0,
            video_freq=0,
            record_video=False,
            info_dict_eval=False,
            n_eval_episodes=1,
            model_kwargs={{"learning_starts": 16, "buffer_size": 500, "batch_size": 32}},
        ))
        """
    )
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        monitor = log_dir / "metrics" / "monitor" / "0.monitor.csv"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.stderr.read().decode())
            # A finished episode row: training is under way.
            if monitor.is_file() and len(monitor.read_text().splitlines()) > 3:
                break
            time.sleep(0.2)
        else:
            raise AssertionError("the training subprocess never started an episode")
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        process.stderr.close()
    assert process.returncode == -signal.SIGKILL
    result = verify_run_dir(log_dir)
    assert not result.complete
    assert result.manifest == "missing"
    assert result.status == "running"
    assert verify_main([str(log_dir)]) == 1


# -- train() with sync_dir ------------------------------------------------------


def test_sync_dir_mirrors_at_checkpoints_and_seals_the_mirror(tmp_path):
    log_dir = tmp_path / "local" / "run"
    sync_dir = tmp_path / "drive" / "run"
    (tmp_path / "drive").mkdir()
    train(_cfg(log_dir, checkpoint_freq=100, sync_dir=str(sync_dir)))
    record = _assert_sealed(log_dir, "completed")
    # Four checkpoint passes; the final pass comes after the record.
    assert record["sync"] == {
        "dir": str(sync_dir),
        "checkpoint_mirrors": 4,
        "failed_mirrors": 0,
    }
    mirrored = verify_run_dir(sync_dir, strict=True)
    assert mirrored.complete and mirrored.status == "completed"
    local = json.loads((log_dir / SYNC_MANIFEST_NAME).read_text())
    remote = json.loads((sync_dir / SYNC_MANIFEST_NAME).read_text())
    assert remote["files"] == local["files"]
    config = json.loads((log_dir / "config.json").read_text())
    assert config["train_config"]["sync_dir"] == str(sync_dir)


def test_a_checkpoint_mirror_failure_never_stops_training(
    tmp_path, monkeypatch, capsys
):
    calls = {"n": 0}
    real_sync = RunDirMirror.sync

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("Drive hiccup")
        return real_sync(self)

    monkeypatch.setattr(RunDirMirror, "sync", flaky)
    log_dir = tmp_path / "local" / "run"
    (tmp_path / "drive").mkdir()
    model = train(
        _cfg(log_dir, checkpoint_freq=100, sync_dir=str(tmp_path / "drive" / "run"))
    )
    assert model.num_timesteps == 400
    assert "failed at 200 steps (OSError('Drive hiccup'))" in capsys.readouterr().out
    assert verify_run_dir(tmp_path / "drive" / "run").complete


def test_a_final_mirror_that_keeps_failing_raises_after_sealing_locally(
    tmp_path, monkeypatch
):
    import importlib

    train_module = importlib.import_module("courtside_dynamics.training.train")
    monkeypatch.setattr(train_module, "_FINAL_SYNC_RETRY_SECONDS", (0.0,))
    calls = {"n": 0}
    real_sync = RunDirMirror.sync

    def dies_at_the_end(self):
        calls["n"] += 1
        if calls["n"] > 2:  # both checkpoint passes succeed
            raise OSError("Transport endpoint is not connected")
        return real_sync(self)

    monkeypatch.setattr(RunDirMirror, "sync", dies_at_the_end)
    log_dir = tmp_path / "local" / "run"
    sync_dir = tmp_path / "drive" / "run"
    (tmp_path / "drive").mkdir()
    with pytest.raises(RuntimeError, match=r"failed 2 times .*sync_run_dir\("):
        train(
            _cfg(
                log_dir,
                total_timesteps=200,
                checkpoint_freq=100,
                sync_dir=str(sync_dir),
            )
        )
    _assert_sealed(log_dir, "completed")
    remote = verify_run_dir(sync_dir)
    assert not remote.complete
    # The manual re-sync the error names completes the mirror.
    monkeypatch.setattr(RunDirMirror, "sync", real_sync)
    sync_run_dir(log_dir, sync_dir)
    assert verify_run_dir(sync_dir, strict=True).complete


def test_an_interrupted_final_mirror_names_the_resync_and_reraises(
    tmp_path, monkeypatch, capsys
):
    """Verifier r2: a KeyboardInterrupt during the final mirror propagated
    with the local record sealed completed (the local-then-sync
    contract), but without the sync_run_dir hint the failure path
    prints."""
    calls = {"n": 0}
    real_sync = RunDirMirror.sync

    def interrupted_at_the_end(self, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:  # both checkpoint passes succeed
            raise KeyboardInterrupt
        return real_sync(self, **kwargs)

    monkeypatch.setattr(RunDirMirror, "sync", interrupted_at_the_end)
    log_dir = tmp_path / "local" / "run"
    sync_dir = tmp_path / "drive" / "run"
    (tmp_path / "drive").mkdir()
    with pytest.raises(KeyboardInterrupt):
        train(
            _cfg(
                log_dir,
                total_timesteps=200,
                checkpoint_freq=100,
                sync_dir=str(sync_dir),
            )
        )
    _assert_sealed(log_dir, "completed")
    assert not verify_run_dir(sync_dir).complete
    out = capsys.readouterr().out
    assert "was interrupted: " in out
    assert f"sync_run_dir({str(log_dir)!r}, {str(sync_dir.resolve())!r})" in out


def test_a_tilde_sync_dir_is_mirrored_where_it_was_validated(
    tmp_path, monkeypatch
):
    """Verifier r1: validate_sync_target checked '~/Drive/run' against
    $HOME, but the mirror was built from the raw string, so the run was
    mirrored into a literal './~/Drive/run' under the working directory
    (local disk on Colab) and said it had succeeded."""
    home = tmp_path / "home"
    (home / "Drive").mkdir(parents=True)
    working = tmp_path / "cwd"
    working.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(working)
    log_dir = tmp_path / "local" / "run"
    train(_cfg(log_dir, total_timesteps=200, sync_dir="~/Drive/run"))
    target = (home / "Drive" / "run").resolve()
    assert verify_run_dir(target, strict=True).complete
    assert not (working / "~").exists()
    record = _assert_sealed(log_dir, "completed")
    assert record["sync"]["dir"] == str(target)
    # config.json keeps the value as configured.
    config = json.loads((log_dir / "config.json").read_text())
    assert config["train_config"]["sync_dir"] == "~/Drive/run"


def test_a_used_sync_dir_is_refused_before_any_output(tmp_path):
    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "someone_elses_file").write_text("x")

    def unbuildable():
        raise AssertionError("train() built an env before validating sync_dir")

    log_dir = tmp_path / "local"
    with pytest.raises(ValueError, match="not an empty directory"):
        train(_cfg(log_dir, env_fn=unbuildable, sync_dir=str(tmp_path / "drive")))
    assert not log_dir.exists()
