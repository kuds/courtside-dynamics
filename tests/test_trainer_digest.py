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
    # Which artifacts the meta binds is compared; only their sha256s are not.
    both = {"best_model.zip": {"sha256": "a"}, "best_vec_normalize.pkl": {"sha256": "c"}}
    assert first != hashed({**meta, "artifacts": both})


def _write_best_pair(run: Path, *, bind: dict[str, bytes] | None = None) -> Path:
    model = run / "model"
    model.mkdir(parents=True)
    files = {"best_model.zip": b"zip", "best_vec_normalize.pkl": b"pkl"}
    for name, content in files.items():
        (model / name).write_bytes(content)
    bound = files if bind is None else bind
    meta = {
        "timestep": 768,
        "artifacts": {
            name: {"sha256": trainer_digest._sha(content)}
            for name, content in bound.items()
        },
    }
    (model / "best_model_meta.json").write_text(json.dumps(meta))
    return model


def test_best_meta_bindings_are_checked_against_the_files(tmp_path):
    """A pair the meta does not vouch for is FAILED, which never matches."""
    check = trainer_digest._check_best_meta_bindings
    assert check(str(tmp_path / "none")) == "absent"
    _write_best_pair(tmp_path / "ok")
    assert check(str(tmp_path / "ok")) == (
        "verified: best_model.zip, best_vec_normalize.pkl"
    )
    _write_best_pair(
        tmp_path / "stale", bind={"best_model.zip": b"zip", "best_vec_normalize.pkl": b"x"}
    )
    assert check(str(tmp_path / "stale")) == (
        "FAILED: best_vec_normalize.pkl not the file(s) best_model_meta.json binds"
    )
    _write_best_pair(tmp_path / "unbound", bind={})
    assert check(str(tmp_path / "unbound")) == (
        "FAILED: best_model_meta.json binds no artifacts"
    )
    _write_best_pair(tmp_path / "missing")
    (tmp_path / "missing" / "model" / "best_model.zip").unlink()
    assert check(str(tmp_path / "missing")).startswith("FAILED: best_model.zip")


def test_layout_and_checkpoint_listings_name_what_exists(tmp_path):
    run = tmp_path / "run"
    assert trainer_digest._existing_layout_artifacts(str(run)) == "none"
    assert trainer_digest._checkpoint_files(str(run)) == "absent"
    _write_best_pair(run)
    (run / "config.json").write_text("{}")
    (run / "checkpoints").mkdir()
    for name in ("x_1536_steps.zip", "x_768_steps.zip"):
        (run / "checkpoints" / name).write_bytes(b"")
    assert trainer_digest._existing_layout_artifacts(str(run)) == (
        "best_model, best_model_meta, best_vec_normalize, checkpoints_dir, config"
    )
    assert trainer_digest._checkpoint_files(str(run)) == (
        "x_1536_steps.zip, x_768_steps.zip"
    )


def test_best_model_hash_reads_the_weights_not_the_zip_bytes(tmp_path):
    """Two saves of one model hash alike (SB3 zips embed timestamps); a
    weight or temperature change, or another algorithm, does not."""
    import torch
    from stable_baselines3 import PPO, SAC

    sac = SAC("MlpPolicy", "Pendulum-v1", buffer_size=64, seed=0, device="cpu")
    sac.save(tmp_path / "a")
    sac.save(tmp_path / "b")
    hashed = trainer_digest._hash_best_model
    first = hashed(tmp_path / "a.zip")
    assert first == hashed(tmp_path / "b.zip")
    assert first["best_model_variables"] not in ("none", "absent")
    with torch.no_grad():
        sac.log_ent_coef.add_(1.0)
    sac.save(tmp_path / "c")
    moved = hashed(tmp_path / "c.zip")
    assert moved["best_model_policy"] == first["best_model_policy"]
    assert moved["best_model_variables"] != first["best_model_variables"]
    with torch.no_grad():
        next(iter(sac.policy.parameters())).add_(1e-3)
    sac.save(tmp_path / "d")
    assert hashed(tmp_path / "d.zip")["best_model_policy"] != first["best_model_policy"]

    ppo = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, seed=0, device="cpu")
    ppo.save(tmp_path / "ppo")
    assert hashed(tmp_path / "ppo.zip")["best_model_variables"] == "none"
    assert hashed(None) == {"best_model_policy": "absent", "best_model_variables": "absent"}


def test_log_ent_coef_accessor_follows_the_algorithm_family():
    from stable_baselines3 import PPO, SAC

    from courtside_dynamics.training.demo_sac import DemoSAC

    auto = SAC("MlpPolicy", "Pendulum-v1", buffer_size=64, seed=0, device="cpu")
    assert trainer_digest._log_ent_coef(auto) is auto.log_ent_coef is not None
    demo = DemoSAC("MlpPolicy", "Pendulum-v1", buffer_size=64, seed=0, device="cpu")
    assert trainer_digest._log_ent_coef(demo) is demo.log_ent_coef is not None
    fixed = SAC(
        "MlpPolicy", "Pendulum-v1", buffer_size=64, ent_coef=0.1, seed=0, device="cpu"
    )
    assert trainer_digest._log_ent_coef(fixed) is None
    ppo = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, seed=0, device="cpu")
    assert trainer_digest._log_ent_coef(ppo) is None


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


def test_readable_mismatches_print_what_the_table_truncates():
    sha_a, sha_b = "a" * 64, "b" * 64
    head = {"j": {"components": {
        "policy_state_dict": sha_a,
        "run_layout_artifacts": "best_model, config, monitor_dir",
        "best_model_meta_bindings": "FAILED: best_model.zip not bound",
    }}}
    base = {"j": {"components": {
        "policy_state_dict": sha_b,
        "run_layout_artifacts": "best_model, best_model_meta, config, monitor_dir",
        "best_model_meta_bindings": "FAILED: best_model.zip not bound",
    }}}
    lines = trainer_digest.readable_mismatches(["j"], head, base)
    assert len(lines) == 2  # the two sha256s stay in the table only
    assert "HEAD: best_model, config, monitor_dir" in lines[1]
    assert "base: best_model, best_model_meta, config, monitor_dir" in lines[1]
    assert lines[0].startswith("j / best_model_meta_bindings:")


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
