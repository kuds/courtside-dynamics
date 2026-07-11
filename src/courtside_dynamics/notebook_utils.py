"""Notebook helpers: Drive mount, learning-curve plots, best-model replay.

The notebooks in this repo are intentionally slim Colab drivers. Anything
beyond the core training loop -- mounting Drive so checkpoints survive a
runtime restart, plotting reward curves, replaying the best policy --
lives here so each notebook stays a few cells of glue.

All helpers degrade gracefully outside of Colab so the same calls work
in a local Jupyter session.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

# Single source of truth for the Colab check (also used by colab_setup).
from courtside_dynamics.colab_setup import _in_colab


def mount_drive(mount_point: str = "/content/drive") -> str:
    """Mount Google Drive in Colab; no-op (returns ``""``) elsewhere.

    Returns the ``MyDrive`` root on success so callers can build paths
    like ``os.path.join(mount_drive(), "Finding Theta", "courtside-dynamics")``.
    """
    if not _in_colab():
        return ""
    from google.colab import drive

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

    Layout::

        <root>/<env>/<algo lowercased>/<YYYYMMDD_HHMMSS>/

    With ``use_drive=True`` and Drive mounted at ``/content/drive``, the
    root is ``/content/drive/MyDrive/Finding Theta/<drive_subdir>/training_runs``.
    Otherwise it falls back to ``local_root``. Set ``timestamp=False``
    to drop the timestamp leaf (re-uses the same dir across runs, which
    is handy when iterating but will mix artifacts).
    """
    if use_drive:
        my_drive = "/content/drive/MyDrive"
        if os.path.isdir(my_drive):
            root = os.path.join(
                my_drive, "Finding Theta", drive_subdir, "training_runs"
            )
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
    from google.colab import runtime

    runtime.unassign()


def _read_monitor_logs(monitor_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rewards, lengths)`` arrays in wall-clock order across workers.

    Uses the shared ``read_monitor_rewards_lengths`` helper, which
    interleaves the per-worker ``*.monitor.csv`` files by absolute
    wall-clock time. Reading them naively in file order instead produces a
    fake "learn / collapse / re-learn" sawtooth, one cycle per worker.
    Returns empty arrays when no monitor logs exist yet.
    """
    from courtside_dynamics.training.monitor_log import (
        read_monitor_rewards_lengths,
    )

    rewards, lengths = read_monitor_rewards_lengths(monitor_dir)
    return np.array(rewards, dtype=float), np.array(lengths, dtype=int)


def plot_learning_curve(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    smoothing: int = 25,
    show: bool = True,
):
    """Four-panel learning curve from the artifacts written by ``train``.

    Top row: per-episode training returns and episode lengths from
    ``log_dir/monitor/*.monitor.csv`` with a rolling-mean overlay.
    Bottom row: deterministic eval returns and episode lengths (mean
    +/- std) from ``log_dir/evaluations.npz`` (written by SB3's
    ``EvalCallback``).
    """
    import matplotlib.pyplot as plt

    log_dir = str(log_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    train_rewards, train_lengths = _read_monitor_logs(
        os.path.join(log_dir, "monitor")
    )

    def _plot_train(ax, series, title, ylabel):
        if series.size:
            ax.plot(series, alpha=0.3, label="per-episode")
            if smoothing > 1 and series.size >= smoothing:
                kernel = np.ones(smoothing) / smoothing
                smooth = np.convolve(series, kernel, mode="valid")
                ax.plot(
                    np.arange(smooth.size) + smoothing - 1,
                    smooth,
                    label=f"rolling mean ({smoothing})",
                )
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no monitor logs", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)

    _plot_train(axes[0, 0], train_rewards, "Training rewards (per episode)", "Return")
    _plot_train(
        axes[0, 1],
        train_lengths.astype(float),
        "Training episode lengths",
        "Steps",
    )

    eval_npz = os.path.join(log_dir, "evaluations.npz")
    eval_data = np.load(eval_npz) if os.path.exists(eval_npz) else None

    # Best-checkpoint step (what EvalCallback saved as best_model.zip).
    # Marked on the eval panels so a post-best collapse -- curve falling
    # away right of the marker -- is visible at a glance.
    best_step = None
    if eval_data is not None and "results" in eval_data:
        results = eval_data["results"]
        if results.size:
            best_step = int(
                eval_data["timesteps"][int(results.mean(axis=1).argmax())]
            )

    def _plot_eval(ax, key, title, ylabel):
        if eval_data is None:
            ax.text(0.5, 0.5, "no evaluations.npz", ha="center", va="center")
        elif key not in eval_data:
            ax.text(0.5, 0.5, f"no '{key}' in evaluations.npz", ha="center", va="center")
        else:
            timesteps = eval_data["timesteps"]
            results = eval_data[key]
            mean = results.mean(axis=1)
            std = results.std(axis=1)
            ax.plot(timesteps, mean, label="eval mean")
            ax.fill_between(
                timesteps, mean - std, mean + std, alpha=0.25, label="+/-1 std"
            )
            if best_step is not None:
                ax.axvline(
                    best_step,
                    color="tab:green",
                    linestyle="--",
                    alpha=0.8,
                    label="best checkpoint",
                )
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)

    _plot_eval(axes[1, 0], "results", "Evaluation rewards", "Return")
    _plot_eval(axes[1, 1], "ep_lengths", "Evaluation episode lengths", "Steps")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def print_stage_summary(log_dir: str | Path) -> None:
    """Print the end-of-run report ``train()`` wrote to ``stage_summary.txt``.

    The report holds the final/best eval numbers, wall-clock duration,
    throughput, device info, and the training-health finals -- the first
    place to look when a run underperforms. ``train()`` writes it even for
    interrupted runs (marked ``status: interrupted``).
    """
    path = os.path.join(str(log_dir), "stage_summary.txt")
    try:
        with open(path) as f:
            print(f.read())
    except FileNotFoundError:
        print(
            f"[notebook_utils] no stage_summary.txt at {path} -- it is "
            "written when train() finishes (or is interrupted); its absence "
            "means the run crashed before the save/eval epilogue."
        )


#: Most common cause for each artifact being absent, shown by
#: ``check_run_artifacts`` next to MISSING entries.
_ARTIFACT_HINTS: dict[str, str] = {
    "config.json": "written at the very start of train(); absent means train() never ran on this dir",
    "monitor": "monitor CSVs appear once training starts; rows once the first episodes finish",
    "tensorboard/progress.csv": "written by the CSV logger train() wires up; appears after SB3's first metric dump",
    "evaluations.npz": "written by EvalCallback; check eval_freq <= total_timesteps",
    "eval_info.csv": "written by InfoDictEvalCallback each eval; requires info_dict_eval=True",
    "best_model.zip": "saved by EvalCallback on its first completed evaluation",
    "best_vec_normalize.pkl": "needs VecNormalize enabled plus at least one new-best eval",
    "checkpoints": "requires checkpoint_freq > 0 and a run long enough to reach the first checkpoint",
    "videos": "requires record_video=True, video_freq > 0, and moviepy installed",
    "final_model.zip": "written when learn() finishes or is interrupted; absent means the run crashed",
    "vec_normalize.pkl": "only written when VecNormalize is enabled (normalize_obs / normalize_reward)",
    "stage_summary.txt": "written by train()'s epilogue; absent means the run crashed before it",
    "learning_curve.png": "saved by the plot_learning_curve cell -- did it run with save_path set?",
    "eval_info.png": "saved by the plot_eval_info cell -- did it run with save_path set?",
    "training_health.png": "saved by the plot_training_health cell -- did it run with save_path set?",
    "best_model.mp4": "saved by record_best_model_video -- needs best_model.zip and rgb_array rendering",
}


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(num_bytes)} {unit}"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    raise AssertionError("unreachable")


def check_run_artifacts(log_dir: str | Path) -> list[str]:
    """Audit ``log_dir`` against every artifact a run should have produced.

    Prints one line per artifact -- present ones with their size (file
    count for directories), missing ones with the most common cause --
    and returns the missing relative paths. Run it at the end of a
    notebook session, after the plotting and replay cells: it catches
    silently-skipped outputs (no video because moviepy failed, no
    best_model because eval never fired) while the Colab runtime still
    exists to do something about it.
    """
    from courtside_dynamics.training.artifacts import EXPECTED_ARTIFACTS

    log_dir = str(log_dir)
    missing: list[str] = []
    label_width = max(len(label) for label, _ in EXPECTED_ARTIFACTS) + 2
    for label, rel in EXPECTED_ARTIFACTS:
        full = os.path.join(log_dir, rel)
        if os.path.isdir(full):
            n_files = 0
            total = 0
            for root, _dirs, files in os.walk(full):
                n_files += len(files)
                total += sum(
                    os.path.getsize(os.path.join(root, f)) for f in files
                )
            status = f"ok       {n_files} file(s), {_human_size(total)}"
        elif os.path.isfile(full):
            status = f"ok       {_human_size(os.path.getsize(full))}"
        else:
            missing.append(rel)
            hint = _ARTIFACT_HINTS.get(rel)
            status = "MISSING" + (f"  -- {hint}" if hint else "")
        print(f"{(label + ':').ljust(label_width)}{status}")
    if not missing:
        print("\nAll expected artifacts present.")
    return missing


def _split_eval_metric(name: str) -> tuple[str, str]:
    """Split an eval-info metric name into ``(stem, variant)``.

    ``rally_count_mean`` -> ``("rally_count", "mean")``,
    ``rally_count_ep_mean`` -> ``("rally_count", "ep_mean")``,
    ``phase_frac_<label>`` -> ``("phase_frac", "<label>")``,
    standalone names like ``episode_length`` -> ``("episode_length", "")``.
    """
    if name.startswith("phase_frac_"):
        return "phase_frac", name[len("phase_frac_") :]
    # ``_ep_mean`` must be checked before ``_mean`` (its own suffix) so the
    # per-episode variant overlays on the same panel as ``_mean``/``_max``.
    for suffix in ("_ep_mean", "_mean", "_final", "_max"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, ""


def plot_eval_info(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    show: bool = True,
    max_cols: int = 3,
):
    """Per-metric time-series grid from ``log_dir/eval_info.csv``.

    The CSV is written by ``InfoDictEvalCallback`` in long format
    ``(timestep, metric, value)``. This function pivots it into a grid
    with one panel per metric stem (e.g. ``rally_count``); ``_mean``,
    ``_final``, and ``_max`` variants are overlaid as separate lines
    on the same axes. ``phase_frac_<label>`` metrics share one panel.

    Returns ``None`` if the CSV is missing or empty.
    """
    import csv as _csv

    import matplotlib.pyplot as plt

    csv_path = os.path.join(str(log_dir), "eval_info.csv")
    if not os.path.exists(csv_path):
        print(f"[notebook_utils] no eval_info.csv at {csv_path}")
        return None

    series: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestep"])
                val = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            stem, variant = _split_eval_metric(row["metric"])
            xs, ys = series.setdefault(stem, {}).setdefault(variant, ([], []))
            xs.append(ts)
            ys.append(val)

    if not series:
        print(f"[notebook_utils] eval_info.csv is empty: {csv_path}")
        return None

    stems = sorted(series)
    n = len(stems)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False
    )

    for i, stem in enumerate(stems):
        ax = axes[i // cols][i % cols]
        for variant in sorted(series[stem]):
            xs, ys = series[stem][variant]
            ax.plot(xs, ys, marker=".", markersize=3, label=variant or "value")
        ax.set_title(stem)
        ax.set_xlabel("Timestep")
        if any(v for v in series[stem]):
            ax.legend(fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_training_health(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    show: bool = True,
    max_cols: int = 3,
):
    """Per-metric grid of SB3's ``train/*`` diagnostics over training.

    Reads ``LOG_DIR/tensorboard/progress.csv`` (written by the CSV logger
    wired up in ``train``). One panel per ``train/*`` column -- for SAC
    that's ``ent_coef`` (the entropy temperature), ``actor_loss``,
    ``critic_loss``, ``ent_coef_loss``; for PPO ``explained_variance``,
    ``approx_kl``, ``clip_fraction``, etc. A collapsing ``ent_coef`` or a
    diverging ``critic_loss`` explains a stalled SAC run that the reward
    curve alone won't.

    Returns ``None`` if ``progress.csv`` is missing or has no ``train/``
    columns yet.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = os.path.join(str(log_dir), "tensorboard", "progress.csv")
    if not os.path.exists(csv_path):
        print(f"[notebook_utils] no progress.csv at {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    metric_cols = sorted(c for c in df.columns if c.startswith("train/"))
    if not metric_cols:
        print("[notebook_utils] progress.csv has no train/* metrics yet")
        return None

    xcol = (
        "time/total_timesteps"
        if "time/total_timesteps" in df.columns
        else None
    )
    n = len(metric_cols)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False
    )

    for i, col in enumerate(metric_cols):
        ax = axes[i // cols][i % cols]
        series = df[col]
        mask = series.notna()
        if xcol is not None:
            ax.plot(df[xcol][mask], series[mask], marker=".", markersize=3)
            ax.set_xlabel("Timestep")
        else:
            xs = [j for j, keep in enumerate(mask) if keep]
            ax.plot(xs, series[mask], marker=".", markersize=3)
            ax.set_xlabel("Log step")
        ax.set_title(col[len("train/") :])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def _load_best_model(log_dir: str, algo: str):
    from courtside_dynamics.training.algos import resolve_algo

    cls = resolve_algo(algo)
    candidate = os.path.join(log_dir, "best_model.zip")
    if not os.path.exists(candidate):
        candidate = os.path.join(log_dir, "best_model")
    return cls.load(candidate)


def _load_obs_normalizer(log_dir: str, env_fn: Callable):
    """Return a callable ``obs -> normalized_obs`` paired with ``best_model.zip``.

    Prefers ``best_vec_normalize.pkl`` (snapshot taken at the moment the
    best model was saved) and falls back to ``vec_normalize.pkl`` (saved
    at end of training, may not match best_model's training-time stats)
    for older runs. Returns identity if neither file exists. Builds a
    throwaway ``DummyVecEnv`` only because ``VecNormalize.load`` requires
    a venv; we never step it.
    """
    path = os.path.join(log_dir, "best_vec_normalize.pkl")
    if not os.path.exists(path):
        path = os.path.join(log_dir, "vec_normalize.pkl")
    if not os.path.exists(path):
        return lambda obs: obs
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dummy = DummyVecEnv([env_fn])
    try:
        vec_norm = VecNormalize.load(path, dummy)
    except Exception:
        return lambda obs: obs
    finally:
        # ``normalize_obs`` only reads ``obs_rms``; the venv exists solely
        # to satisfy VecNormalize.load's signature and would otherwise leak
        # a full MuJoCo env for the rest of the notebook session.
        dummy.close()
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm.normalize_obs


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
    ``render_mode='rgb_array'`` so frames can be captured. If
    ``vec_normalize.pkl`` is present in ``log_dir`` the saved
    obs-normalizer is applied to each observation before
    ``model.predict``, matching the training-time wrapper.
    """
    import imageio.v2 as imageio

    model = _load_best_model(str(log_dir), algo)
    normalize_obs = _load_obs_normalizer(str(log_dir), env_fn)
    env = env_fn()
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError(
            "env_fn must construct the env with render_mode='rgb_array' "
            "so frames can be captured."
        )

    out_path = out_path or os.path.join(str(log_dir), "best_model.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frames: list[ArrayLike] = []
    total_reward = 0.0
    obs, _ = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(normalize_obs(obs), deterministic=deterministic)
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
    from IPython.display import Video

    return Video(path, embed=True, width=width)
