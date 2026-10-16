"""tools/trainer_digest.py: the hashing and comparison it gates SAC on.

The tool's train() jobs take minutes and run by hand before a merge
(``--base <ref>``); these tests pin, in milliseconds, that its hashes
ignore exactly the wall clock and nothing else, that its comparison is
loud about failures, and that every job still resolves its config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import trainer_digest


def _write_monitor(run: Path, name: str, rows: list[tuple[float, int, float]], t0: float):
    monitor = run / "metrics" / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    lines = ["#" + json.dumps({"t_start": t0, "env_id": "None"}), "r,l,t"]
    lines += [f"{r},{length},{t}" for r, length, t in rows]
    (monitor / name).write_text("\n".join(lines) + "\n")


def test_monitor_hash_ignores_only_the_wall_clock(tmp_path):
    a, b, c = (tmp_path / x for x in "abc")
    _write_monitor(a, "0.monitor.csv", [(1.0, 10, 0.5), (2.0, 12, 0.9)], t0=100.0)
    _write_monitor(b, "0.monitor.csv", [(1.0, 10, 7.5), (2.0, 12, 8.1)], t0=999.0)
    _write_monitor(c, "0.monitor.csv", [(1.0, 10, 0.5), (2.5, 12, 0.9)], t0=100.0)
    hash_a = trainer_digest._hash_monitors(str(a))
    assert hash_a == trainer_digest._hash_monitors(str(b))
    assert hash_a != trainer_digest._hash_monitors(str(c))
    assert trainer_digest._hash_monitors(str(tmp_path / "missing")) == "absent"


def test_eval_info_hash_keeps_every_task_metric(tmp_path):
    """No substring filter: term_timeout and *_second_bounce are metrics."""

    def hashed(value: float) -> str:
        run = tmp_path / str(value)
        (run / "metrics").mkdir(parents=True)
        (run / "metrics" / "eval_info.csv").write_text(
            "timestep,metric,value\n"
            f"500,term_timeout_ep_mean,{value}\n"
            "500,point_end_second_bounce_ep_mean,0.0\n"
        )
        return trainer_digest._hash_eval_info(str(run))

    assert hashed(0.0) != hashed(1.0)
    assert not trainer_digest._WALL_CLOCK_METRICS


def test_best_meta_hash_drops_only_the_artifact_shas(tmp_path):
    def hashed(meta: dict) -> str:
        run = tmp_path / str(len(list(tmp_path.iterdir())))
        (run / "model").mkdir(parents=True)
        (run / "model" / "best_model_meta.json").write_text(json.dumps(meta))
        return trainer_digest._hash_best_meta(str(run))

    meta = {"timestep": 768, "selection_values": {"success_rate": 0.5}}
    first = hashed({**meta, "artifacts": {"best_model.zip": {"sha256": "a"}}})
    assert first == hashed({**meta, "artifacts": {"best_model.zip": {"sha256": "b"}}})
    assert first != hashed({**meta, "timestep": 1536})


def test_compare_runs_is_loud_about_mismatches_and_failures():
    good = {"components": {"policy_state_dict": "abc", "torch_rng": "def"}}
    moved = {"components": {"policy_state_dict": "abc", "torch_rng": "xyz"}}
    rows, ok = trainer_digest.compare_runs(["j"], {"j": good}, {"j": good})
    assert ok and {row[4] for row in rows} == {"match"}
    rows, ok = trainer_digest.compare_runs(["j"], {"j": good}, {"j": moved})
    assert not ok
    assert ("j", "torch_rng", "def", "xyz", "MISMATCH") in rows
    rows, ok = trainer_digest.compare_runs(["j"], {"j": good}, {"j": {"error": ["boom"]}})
    assert not ok and rows == [("j", "base job FAILED", "", "", "ERROR")]
    failed = {"components": {"diagnosis_reports": "FAILED: probe crashed"}}
    _, ok = trainer_digest.compare_runs(["j"], {"j": failed}, {"j": failed})
    assert not ok  # a failed diagnosis never counts as a match


@pytest.mark.parametrize("job", sorted(trainer_digest.JOBS))
def test_every_job_resolves_its_config(job, tmp_path):
    cfg = trainer_digest.JOBS[job](str(tmp_path))
    assert cfg.seed == 0 and cfg.n_envs == 2
    assert cfg.record_video is False
    expected_algo = {
        "sac_ball_balance": "SAC",
        "demosac_off_ball_balance": "DEMOSAC",
        "sac_paddle_tennis_quick": "SAC",
        "ppo_ball_balance": "PPO",
    }[job]
    assert cfg.algo == expected_algo
    if job == "sac_paddle_tennis_quick":
        # The recipe's whole SAC bundle survives the buffer shrink.
        assert cfg.model_kwargs["use_sde"] is True
        assert cfg.model_kwargs["train_freq"] == (64, "step")
        assert cfg.normalize_obs_excluded_indices
        assert cfg.headline_key == "episode_rally_returns_a"
        assert cfg.checkpoint_diagnosis == {"episodes": 1, "seed_start": 5200}


def test_sides_run_under_different_hash_seeds(tmp_path):
    """Real runs draw a random hash seed; a str-hash-order dependence in
    training must show up as a mismatch, not match under a shared seed."""
    head = trainer_digest._job_env("head", tmp_path / "head" / "src")
    base = trainer_digest._job_env("base", tmp_path / "base" / "src")
    assert head["PYTHONHASHSEED"] != base["PYTHONHASHSEED"]
    for side, env in (("head", head), ("base", base)):
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path / side / "src")
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
