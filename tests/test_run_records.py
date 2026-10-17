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
    assert record["selection"] == {
        "timestep": 450_000,
        "keys": ["episode_rally_returns_a_ep_mean", "success_rate"],
        "values": {"episode_rally_returns_a_ep_mean": 0.4},
        "n_updates": 120,
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
    """A notebook's post-training plots land beside a sealed run: they
    are listed as unlisted, and only ``strict`` refuses them."""
    run_dir = _sealed_run_dir(tmp_path / "run")
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "learning_curve.png").write_bytes(b"png")
    lenient = verify_run_dir(run_dir)
    assert lenient.complete
    assert lenient.unlisted == ("reports/learning_curve.png",)
    strict = verify_run_dir(run_dir, strict=True)
    assert not strict.complete
    assert "strict" in strict.problems[0]


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
    (mirror_dir / "stale.tmp").write_text("left by a failed pass")
    mirror.sync()
    # Only what changed, and the source's own manifest is never copied.
    assert copied == ["model/best_model.zip", RUN_RECORD_NAME]
    assert not (mirror_dir / "stale.tmp").exists()
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


def test_a_used_sync_dir_is_refused_before_any_output(tmp_path):
    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "someone_elses_file").write_text("x")

    def unbuildable():
        raise AssertionError("train() built an env before validating sync_dir")

    log_dir = tmp_path / "local"
    with pytest.raises(ValueError, match="not an empty directory"):
        train(_cfg(log_dir, env_fn=unbuildable, sync_dir=str(tmp_path / "drive")))
    assert not log_dir.exists()
