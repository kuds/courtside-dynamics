"""Per-run artifacts: ``run_config.json`` and ``run_summary.txt``.

Both files are written by :func:`courtside_dynamics.training.train` so
every ``LOG_DIR`` is self-describing -- you can answer "how was this
``best_model.zip`` produced?" from disk alone, even after the Colab
runtime that produced it is gone.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from courtside_dynamics.training.train import TrainConfig


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in (
        "numpy",
        "stable_baselines3",
        "gymnasium",
        "mujoco",
        "torch",
        "courtside_dynamics",
    ):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        version = getattr(mod, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _probe_env(cfg: TrainConfig) -> dict[str, Any]:
    """Construct the env once to capture class + space metadata."""
    info: dict[str, Any] = {"class": None, "observation_shape": None, "action_shape": None}
    try:
        env = cfg.env_fn()
    except Exception:
        return info
    try:
        info["class"] = type(env).__name__
        obs_shape = getattr(env.observation_space, "shape", None)
        act_shape = getattr(env.action_space, "shape", None)
        if obs_shape is not None:
            info["observation_shape"] = list(obs_shape)
        if act_shape is not None:
            info["action_shape"] = list(act_shape)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return info


def write_run_config(cfg: TrainConfig, log_dir: str) -> str:
    """Snapshot the resolved cfg + provenance to ``log_dir/run_config.json``."""
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "env": _probe_env(cfg),
        "train_config": {
            "algo": cfg.algo,
            "total_timesteps": cfg.total_timesteps,
            "log_dir": cfg.log_dir,
            "name_prefix": cfg.name_prefix,
            "n_envs": cfg.n_envs,
            "eval_freq": cfg.eval_freq,
            "n_eval_episodes": cfg.n_eval_episodes,
            "video_length": cfg.video_length,
            "record_video": cfg.record_video,
            "policy": cfg.policy,
            "model_kwargs": cfg.model_kwargs,
            "csv_header": list(cfg.csv_header) if cfg.csv_header else None,
            "info_dict_eval": cfg.info_dict_eval,
            "phase_key": cfg.phase_key,
            "phase_labels": (
                {str(k): v for k, v in cfg.phase_labels.items()}
                if cfg.phase_labels
                else None
            ),
        },
    }
    out = os.path.join(log_dir, "run_config.json")
    with open(out, "w") as f:
        # default=repr so any callable / non-JSON value in model_kwargs
        # round-trips as a readable string instead of crashing the dump.
        json.dump(payload, f, indent=2, default=repr)
        f.write("\n")
    return out


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _read_monitor(log_dir: str) -> tuple[list[float], list[int]]:
    rewards: list[float] = []
    lengths: list[int] = []
    monitor_dir = os.path.join(log_dir, "monitor")
    if not os.path.isdir(monitor_dir):
        return rewards, lengths
    for entry in sorted(os.listdir(monitor_dir)):
        if not entry.endswith("monitor.csv"):
            continue
        with open(os.path.join(monitor_dir, entry)) as f:
            reader = csv.reader(f)
            next(reader, None)  # SB3 metadata comment
            header = next(reader, None)
            if not header or "r" not in header or "l" not in header:
                continue
            r_idx = header.index("r")
            l_idx = header.index("l")
            for row in reader:
                if not row:
                    continue
                rewards.append(float(row[r_idx]))
                lengths.append(int(row[l_idx]))
    return rewards, lengths


def write_run_summary(
    cfg: TrainConfig,
    log_dir: str,
    *,
    final_mean_reward: float,
    final_std_reward: float,
    duration_seconds: float,
) -> str:
    """Write a human-readable end-of-run report to ``log_dir/run_summary.txt``."""
    lines: list[str] = []
    lines.append(f"Run: {os.path.basename(os.path.normpath(log_dir))}")
    lines.append(f"Algo: {cfg.algo}")
    lines.append(f"Total timesteps: {cfg.total_timesteps:,}")
    lines.append(f"Wall-clock: {_format_duration(duration_seconds)}")
    lines.append("")
    lines.append("Final evaluation:")
    lines.append(
        f"  mean_reward = {final_mean_reward:.3f} +/- {final_std_reward:.3f}"
        f"  ({cfg.n_eval_episodes} episodes)"
    )

    eval_npz = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(eval_npz):
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        if results.size:
            mean_per_eval = results.mean(axis=1)
            std_per_eval = results.std(axis=1)
            best_idx = int(mean_per_eval.argmax())
            lines.append("")
            lines.append("Best eval (from evaluations.npz):")
            lines.append(
                f"  mean_reward = {mean_per_eval[best_idx]:.3f} +/- "
                f"{std_per_eval[best_idx]:.3f} at step "
                f"{int(timesteps[best_idx]):,}"
            )

    train_rewards, train_lengths = _read_monitor(log_dir)
    if train_rewards:
        last_n = min(100, len(train_rewards))
        lines.append("")
        lines.append("Training (per-episode):")
        lines.append(f"  episodes: {len(train_rewards):,}")
        lines.append(
            f"  reward mean (last {last_n}): "
            f"{statistics.mean(train_rewards[-last_n:]):.3f}"
        )
        lines.append(
            f"  episode length mean (last {last_n}): "
            f"{statistics.mean(train_lengths[-last_n:]):.1f}"
        )

    lines.append("")
    lines.append("Artifacts:")
    for label, path in [
        ("best_model", "best_model.zip"),
        ("final_model", "final_model.zip"),
        ("evaluations", "evaluations.npz"),
        ("run_config", "run_config.json"),
        ("learning_curve", "learning_curve.png"),
        ("best_model_video", "best_model.mp4"),
    ]:
        if os.path.exists(os.path.join(log_dir, path)):
            lines.append(f"  {label}: {path}")

    out = os.path.join(log_dir, "run_summary.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out
