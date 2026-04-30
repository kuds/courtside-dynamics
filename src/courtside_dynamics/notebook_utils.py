"""Notebook helpers: Drive mount, learning-curve plots, best-model replay.

The notebooks in this repo are intentionally slim Colab drivers. Anything
beyond the core training loop -- mounting Drive so checkpoints survive a
runtime restart, plotting reward curves, replaying the best policy --
lives here so each notebook stays a few cells of glue.

All helpers degrade gracefully outside of Colab so the same calls work
in a local Jupyter session.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np


def _in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def mount_drive(mount_point: str = "/content/drive") -> str:
    """Mount Google Drive in Colab; no-op (returns ``""``) elsewhere.

    Returns the ``MyDrive`` root on success so callers can build paths
    like ``os.path.join(mount_drive(), "courtside-dynamics", "logs")``.
    """
    if not _in_colab():
        return ""
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return os.path.join(mount_point, "MyDrive")


def resolve_run_dir(
    env: str,
    algo: str,
    *,
    use_drive: bool = False,
    drive_subdir: str = "courtside-dynamics",
    local_root: str = "./logs",
    timestamp: bool = True,
) -> str:
    """Pick a fresh run directory, optionally rooted under mounted Drive.

    Layout, modeled on mesozoic-labs::

        <root>/<env>/<algo lowercased>/<YYYYMMDD_HHMMSS>/

    With ``use_drive=True`` and Drive mounted at ``/content/drive``, the
    root is ``/content/drive/MyDrive/<drive_subdir>``. Otherwise it
    falls back to ``local_root``. Set ``timestamp=False`` to drop the
    timestamp leaf (re-uses the same dir across runs, which is handy
    when iterating but will mix artifacts).
    """
    if use_drive:
        my_drive = "/content/drive/MyDrive"
        if os.path.isdir(my_drive):
            root = os.path.join(my_drive, drive_subdir)
        else:
            print(
                "[notebook_utils] use_drive=True but /content/drive/MyDrive "
                "is not mounted; falling back to local logs."
            )
            root = local_root
    else:
        root = local_root

    parts = [root, env, algo.lower()]
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def disconnect_runtime(delay_seconds: int = 5) -> None:
    """Free the Colab GPU by tearing down the runtime. No-op locally.

    The optional delay gives any in-flight cell output a moment to
    render before the runtime disconnects.
    """
    if not _in_colab():
        return
    print(f"Disconnecting Colab runtime in {delay_seconds}s...")
    time.sleep(delay_seconds)
    from google.colab import runtime  # type: ignore

    runtime.unassign()


def _read_monitor_logs(monitor_dir: str) -> tuple[np.ndarray, np.ndarray]:
    rewards: list[float] = []
    lengths: list[int] = []
    if not os.path.isdir(monitor_dir):
        return np.array([]), np.array([])
    for entry in sorted(os.listdir(monitor_dir)):
        if not entry.endswith("monitor.csv"):
            continue
        with open(os.path.join(monitor_dir, entry)) as f:
            reader = csv.reader(f)
            next(reader, None)  # SB3 metadata comment
            header = next(reader, None)
            if not header or "r" not in header:
                continue
            r_idx = header.index("r")
            l_idx = header.index("l")
            for row in reader:
                if not row:
                    continue
                rewards.append(float(row[r_idx]))
                lengths.append(int(row[l_idx]))
    return np.array(rewards), np.array(lengths)


def plot_learning_curve(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    smoothing: int = 25,
    show: bool = True,
):
    """Two-panel reward plot from the artifacts written by ``train``.

    Left: per-episode training returns from ``log_dir/monitor/*.monitor.csv``
    with a rolling mean overlay. Right: deterministic eval rewards (mean
    +/- std) read from ``log_dir/evaluations.npz`` (written by SB3's
    ``EvalCallback``).
    """
    import matplotlib.pyplot as plt

    log_dir = str(log_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_rewards, _ = _read_monitor_logs(os.path.join(log_dir, "monitor"))
    if train_rewards.size:
        axes[0].plot(train_rewards, alpha=0.3, label="episode reward")
        if smoothing > 1 and train_rewards.size >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smooth = np.convolve(train_rewards, kernel, mode="valid")
            axes[0].plot(
                np.arange(smooth.size) + smoothing - 1,
                smooth,
                label=f"rolling mean ({smoothing})",
            )
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "no monitor logs", ha="center", va="center")
    axes[0].set_title("Training rewards (per episode)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")

    eval_npz = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(eval_npz):
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        mean = results.mean(axis=1)
        std = results.std(axis=1)
        axes[1].plot(timesteps, mean, label="eval mean")
        axes[1].fill_between(
            timesteps, mean - std, mean + std, alpha=0.25, label="+/-1 std"
        )
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "no evaluations.npz", ha="center", va="center")
    axes[1].set_title("Evaluation rewards")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Return")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def _load_best_model(log_dir: str, algo: str):
    from stable_baselines3 import PPO, SAC

    cls = {"SAC": SAC, "PPO": PPO}[algo.upper()]
    candidate = os.path.join(log_dir, "best_model.zip")
    if not os.path.exists(candidate):
        candidate = os.path.join(log_dir, "best_model")
    return cls.load(candidate)


def record_best_model_video(
    log_dir: str | Path,
    env_fn: Callable,
    *,
    algo: str = "SAC",
    video_length: int = 750,
    deterministic: bool = True,
    fps: int = 60,
    out_path: str | None = None,
) -> str:
    """Roll out ``best_model.zip`` against a fresh env and save MP4.

    The env returned by ``env_fn`` must be constructed with
    ``render_mode='rgb_array'`` so frames can be captured.
    """
    import imageio.v2 as imageio  # type: ignore

    model = _load_best_model(str(log_dir), algo)
    env = env_fn()
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError(
            "env_fn must construct the env with render_mode='rgb_array' "
            "so frames can be captured."
        )

    out_path = out_path or os.path.join(str(log_dir), "best_model.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frames: list[np.ndarray] = []
    total_reward = 0.0
    obs, _ = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if terminated or truncated:
            break
    env.close()

    if not frames:
        raise RuntimeError("No frames captured; check render_mode='rgb_array'.")

    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(
        f"Saved best-model rollout: {out_path}  "
        f"(return={total_reward:.2f}, frames={len(frames)})"
    )
    return out_path


def display_video(path: str, *, width: int = 480):
    """Embed an MP4 in a Jupyter / Colab cell."""
    from IPython.display import Video  # type: ignore

    return Video(path, embed=True, width=width)
