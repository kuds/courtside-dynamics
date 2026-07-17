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
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    "best_model_long_horizon_eval.json": "saved by the WallBall long-horizon best-model evaluation cell",
    "best_model_long_horizon_episodes.csv": "saved with one row per held-out WallBall evaluation seed",
}


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(num_bytes)} {unit}"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    raise AssertionError("unreachable")


def check_run_artifacts(
    log_dir: str | Path,
    *,
    extra_artifacts: Sequence[tuple[str, str]] = (),
) -> list[str]:
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
    artifacts = (*EXPECTED_ARTIFACTS, *extra_artifacts)
    missing: list[str] = []
    label_width = max(len(label) for label, _ in artifacts) + 2
    for label, rel in artifacts:
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

    # The recipe's headline metric (config.json: train_config.headline_key)
    # leads the grid, so the panel that answers "is the task improving?"
    # is the first thing on screen.
    headline_stem = None
    config_path = os.path.join(str(log_dir), "config.json")
    if os.path.exists(config_path):
        import json as _json

        try:
            with open(config_path) as f:
                headline_stem = (
                    _json.load(f).get("train_config") or {}
                ).get("headline_key")
        except (OSError, ValueError):
            headline_stem = None

    stems = sorted(series)
    if headline_stem in series:
        stems.remove(headline_stem)
        stems.insert(0, headline_stem)
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
        if stem == headline_stem:
            ax.set_title(f"{stem} (headline)", fontweight="bold")
        else:
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


WALL_BALL_LONG_HORIZON_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("best_long_eval", "best_model_long_horizon_eval.json"),
    ("best_long_eval_episodes", "best_model_long_horizon_episodes.csv"),
)

# Always report the practical rally gates, including explicit zeroes above
# the observed maximum. The first WallBallBaseline run reached exactly one
# return on every held-out seed; the former observed-max-only curve therefore
# omitted the most important result: 0% survived a second return.
WALL_BALL_RETURN_SURVIVAL_THRESHOLDS: tuple[int, ...] = (1, 2, 3, 5)


_WALL_BALL_REWARD_COMPONENTS: tuple[str, ...] = (
    "rew_wall",
    "rew_paddle",
    "rew_shaping",
    "rew_oob",
    "rew_double_bounce",
    "rew_stall",
    "rew_style_violation",
    "rew_recoverable_bounce",
    "rew_early_touch",
    "rew_weak_return",
    "rew_first_hit",
)

_WALL_BALL_TERMINATION_REASONS: tuple[str, ...] = (
    "out_of_bounds",
    "double_bounce",
    "stall",
    "style_violation",
    "nonfinite",
    "timeout",
    "evaluation_cutoff",
    "terminated_unknown",
    "unknown",
)

_WALL_BALL_REQUIRED_INFO_KEYS = frozenset(
    {
        "bounce_count",
        "paddle_hit_count",
        "wall_contact_count",
        "floor_bounce_total",
        "floor_bounce_count",
        "legal_paddle_hit_count",
        "one_bounce_recovery_count",
        "one_bounce_return_count",
        "style_violation_reason",
        *_WALL_BALL_REWARD_COMPONENTS,
        "term_oob",
        "term_double_bounce",
        "term_stall",
        "term_style_violation",
        "term_timeout",
        "term_nonfinite",
    }
)

_LEGACY_WALL_BALL_CONSTRUCTOR_DEFAULTS: dict[str, Any] = {
    "style_violation_penalty": 1.0,
    "rally_style": "open",
    "paddle_home_x": -1.7,
    "paddle_x_target_range": None,
    "recovery_reset_probability": 0.0,
    "post_bounce_reset_fraction": 0.5,
    "recoverable_bounce_bonus": 0.0,
    "recoverable_bounce_lateral_limit": 0.0,
    # Added in 0.10.0: absent from pre-0.10 run configs, whose behavior
    # matches these defaults exactly (terminal early touch, XML damping).
    "early_touch_penalty": None,
    "paddle_joint_damping": None,
    "weak_return_penalty": None,
    "first_hit_bonus": 0.0,
    "serve_start_x": 1.0,
    "paddle_start_x": None,
    "paddle_x_fence": None,
}


def _canonicalize_constructor_value(value: Any) -> Any:
    """Normalize JSON arrays and runtime tuples for config comparison."""
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_constructor_value(child) for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_constructor_value(child) for child in value]
    return value


def _wall_ball_constructor_kwargs_match(
    training_kwargs: Mapping[str, Any],
    evaluated_kwargs: Mapping[str, Any],
) -> bool:
    """Compare constructor settings while accepting new defaults in old runs."""
    training_comparable = _canonicalize_constructor_value(
        {key: value for key, value in training_kwargs.items() if key != "episode_len"}
    )
    evaluated_comparable = _canonicalize_constructor_value(
        {key: value for key, value in evaluated_kwargs.items() if key != "episode_len"}
    )
    missing = object()
    for key, default in _LEGACY_WALL_BALL_CONSTRUCTOR_DEFAULTS.items():
        if (
            key not in training_comparable
            and evaluated_comparable.get(key, missing)
            == _canonicalize_constructor_value(default)
        ):
            del evaluated_comparable[key]
    return training_comparable == evaluated_comparable


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return JSON-native population statistics for one metric."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
    if not bool(np.isfinite(array).all()):
        raise ValueError("evaluation metrics must be finite")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _termination_reason(
    info: dict[str, Any],
    *,
    terminated: bool,
    truncated: bool,
    evaluation_cutoff: bool,
) -> str:
    flags = (
        ("out_of_bounds", "term_oob"),
        ("double_bounce", "term_double_bounce"),
        ("stall", "term_stall"),
        ("style_violation", "term_style_violation"),
        ("nonfinite", "term_nonfinite"),
        ("timeout", "term_timeout"),
    )
    active = [name for name, key in flags if bool(info.get(key, False))]
    if len(active) > 1:
        raise ValueError(f"multiple WallBall termination flags are active: {active}")
    if active:
        return active[0]
    if evaluation_cutoff:
        return "evaluation_cutoff"
    if truncated:
        return "timeout"
    if terminated:
        return "terminated_unknown"
    return "unknown"


def _rollout_wall_ball_seed(
    env,
    model,
    normalize_obs: Callable,
    *,
    seed: int,
    episode_len: int,
    deterministic: bool,
) -> dict[str, Any]:
    """Roll out one held-out WallBall seed and return an auditable row."""
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    reward_component_totals = {key: 0.0 for key in _WALL_BALL_REWARD_COMPONENTS}
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    episode_steps = 0
    previous_counts = {
        "completed_returns": 0,
        "paddle_hits": 0,
        "legal_paddle_hits": 0,
        "wall_contacts": 0,
        "floor_bounces": 0,
        "one_bounce_recoveries": 0,
        "one_bounce_returns": 0,
    }
    floor_bounces_waiting_for_paddle = 0
    recovered_cycles_waiting_for_wall = 0
    post_floor_bounce_paddle_recoveries = 0
    post_floor_bounce_completed_returns = 0

    for _ in range(episode_len):
        episode_steps += 1
        action, _ = model.predict(normalize_obs(obs), deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        final_info = dict(info)
        missing_info = _WALL_BALL_REQUIRED_INFO_KEYS - final_info.keys()
        if missing_info:
            raise KeyError(
                "WallBall long-horizon evaluation requires info keys: "
                f"{sorted(missing_info)}"
            )
        for key in _WALL_BALL_REWARD_COMPONENTS:
            reward_component_totals[key] += float(info[key])

        current_counts = {
            "completed_returns": int(info["bounce_count"]),
            "paddle_hits": int(info["paddle_hit_count"]),
            "legal_paddle_hits": int(info["legal_paddle_hit_count"]),
            "wall_contacts": int(info["wall_contact_count"]),
            "floor_bounces": int(info["floor_bounce_total"]),
            "one_bounce_recoveries": int(info["one_bounce_recovery_count"]),
            "one_bounce_returns": int(info["one_bounce_return_count"]),
        }
        if any(
            current_counts[key] < previous_counts[key]
            for key in current_counts
        ):
            raise ValueError("WallBall episode counters must be monotone")
        deltas = {
            key: current_counts[key] - previous_counts[key]
            for key in current_counts
        }

        # Event-order audit at the policy/control-step resolution. A floor
        # contact is considered recovered only when a later counter delta
        # shows a paddle hit; that recovery becomes a completed return only
        # when a later gated wall return increments bounce_count.
        floor_bounces_waiting_for_paddle += deltas["floor_bounces"]
        if deltas["paddle_hits"] and floor_bounces_waiting_for_paddle:
            recovered = min(
                deltas["paddle_hits"], floor_bounces_waiting_for_paddle
            )
            post_floor_bounce_paddle_recoveries += recovered
            recovered_cycles_waiting_for_wall += recovered
            floor_bounces_waiting_for_paddle = 0
        if deltas["completed_returns"] and recovered_cycles_waiting_for_wall:
            completed = min(
                deltas["completed_returns"], recovered_cycles_waiting_for_wall
            )
            post_floor_bounce_completed_returns += completed
        if deltas["wall_contacts"]:
            floor_bounces_waiting_for_paddle = 0
            recovered_cycles_waiting_for_wall = 0
        previous_counts = current_counts
        if terminated or truncated:
            break

    evaluation_cutoff = not (terminated or truncated)
    row: dict[str, Any] = {
        "seed": int(seed),
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_steps),
        "completed_returns": int(final_info["bounce_count"]),
        "paddle_hit_count": int(final_info["paddle_hit_count"]),
        "legal_paddle_hit_count": int(final_info["legal_paddle_hit_count"]),
        "wall_contact_count": int(final_info["wall_contact_count"]),
        "floor_bounce_total": int(final_info["floor_bounce_total"]),
        "terminal_floor_bounce_count": int(final_info["floor_bounce_count"]),
        "one_bounce_recovery_count": int(final_info["one_bounce_recovery_count"]),
        "one_bounce_return_count": int(final_info["one_bounce_return_count"]),
        "post_floor_bounce_paddle_recoveries": int(
            post_floor_bounce_paddle_recoveries
        ),
        "post_floor_bounce_completed_returns": int(
            post_floor_bounce_completed_returns
        ),
        "unrecovered_floor_bounces": int(
            int(final_info["floor_bounce_total"])
            - post_floor_bounce_paddle_recoveries
        ),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "evaluation_cutoff": bool(evaluation_cutoff),
        "style_violation_reason": final_info["style_violation_reason"],
    }
    for key in (
        "term_oob",
        "term_double_bounce",
        "term_stall",
        "term_style_violation",
        "term_timeout",
        "term_nonfinite",
    ):
        row[key] = bool(final_info[key])
    row["termination_reason"] = _termination_reason(
        final_info,
        terminated=bool(terminated),
        truncated=bool(truncated),
        evaluation_cutoff=evaluation_cutoff,
    )
    for key, value in reward_component_totals.items():
        row[f"{key}_total"] = float(value)
    return row


def _summarize_wall_ball_episodes(
    rows: Sequence[dict[str, Any]],
    *,
    survival_steps: Sequence[int],
) -> dict[str, Any]:
    """Aggregate per-seed WallBall endurance and bounce-recovery metrics."""
    if not rows:
        raise ValueError("at least one WallBall evaluation episode is required")

    metrics = {
        name: _distribution_summary([float(row[name]) for row in rows])
        for name in (
            "episode_reward",
            "episode_length",
            "completed_returns",
            "paddle_hit_count",
            "legal_paddle_hit_count",
            "wall_contact_count",
            "floor_bounce_total",
            "terminal_floor_bounce_count",
            "one_bounce_recovery_count",
            "one_bounce_return_count",
            "post_floor_bounce_paddle_recoveries",
            "post_floor_bounce_completed_returns",
            "unrecovered_floor_bounces",
        )
    }
    reward_components = {
        key: _distribution_summary([float(row[f"{key}_total"]) for row in rows])
        for key in _WALL_BALL_REWARD_COMPONENTS
    }

    n = len(rows)
    step_survival = {
        str(threshold): {
            "count": sum(int(row["episode_length"]) >= threshold for row in rows),
            "rate": sum(int(row["episode_length"]) >= threshold for row in rows) / n,
        }
        for threshold in survival_steps
    }
    max_returns = max(1, max(int(row["completed_returns"]) for row in rows))
    return_thresholds = sorted(
        set(range(1, max_returns + 1))
        | set(WALL_BALL_RETURN_SURVIVAL_THRESHOLDS)
    )
    return_survival = {
        str(threshold): {
            "count": sum(int(row["completed_returns"]) >= threshold for row in rows),
            "rate": sum(int(row["completed_returns"]) >= threshold for row in rows) / n,
        }
        for threshold in return_thresholds
    }

    terminations = {
        reason: {
            "count": sum(row["termination_reason"] == reason for row in rows),
            "rate": sum(row["termination_reason"] == reason for row in rows) / n,
        }
        for reason in _WALL_BALL_TERMINATION_REASONS
    }
    style_violation_reasons = {
        reason: {
            "count": sum(
                bool(row["term_style_violation"])
                and str(row["style_violation_reason"] or "unspecified") == reason
                for row in rows
            ),
            "rate": sum(
                bool(row["term_style_violation"])
                and str(row["style_violation_reason"] or "unspecified") == reason
                for row in rows
            )
            / n,
        }
        for reason in sorted(
            {
                str(row["style_violation_reason"] or "unspecified")
                for row in rows
                if bool(row["term_style_violation"])
            }
        )
    }
    floor_totals = [int(row["floor_bounce_total"]) for row in rows]
    terminal_floor = [int(row["terminal_floor_bounce_count"]) for row in rows]
    floor_diagnostics = {
        "total_contacts": int(sum(floor_totals)),
        "episodes_with_any": int(sum(value > 0 for value in floor_totals)),
        "episodes_with_any_rate": float(sum(value > 0 for value in floor_totals) / n),
        "post_floor_bounce_paddle_recoveries": int(
            sum(int(row["post_floor_bounce_paddle_recoveries"]) for row in rows)
        ),
        "episodes_with_paddle_recovery": int(
            sum(int(row["post_floor_bounce_paddle_recoveries"]) > 0 for row in rows)
        ),
        "episodes_with_paddle_recovery_rate": float(
            sum(int(row["post_floor_bounce_paddle_recoveries"]) > 0 for row in rows)
            / n
        ),
        "post_floor_bounce_completed_returns": int(
            sum(int(row["post_floor_bounce_completed_returns"]) for row in rows)
        ),
        "episodes_with_completed_return": int(
            sum(int(row["post_floor_bounce_completed_returns"]) > 0 for row in rows)
        ),
        "episodes_with_completed_return_rate": float(
            sum(int(row["post_floor_bounce_completed_returns"]) > 0 for row in rows)
            / n
        ),
        # A paddle OR wall contact resets the consecutive counter, so this
        # is only an upper bound on recovered bounces, not proof that a
        # successful one-bounce return occurred.
        "contacts_reset_before_terminal_upper_bound": int(
            sum(
                total - final
                for total, final in zip(floor_totals, terminal_floor, strict=True)
            )
        ),
    }
    return {
        "metrics": metrics,
        "reward_components": reward_components,
        "step_survival": step_survival,
        "return_survival_curve": return_survival,
        "terminations": terminations,
        "style_violation_reasons": style_violation_reasons,
        "floor_bounce_diagnostics": floor_diagnostics,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, suffix=".tmp", delete=False
        ) as handle:
            tmp_path = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _atomic_write_csv(
    path: str,
    rows: Sequence[dict[str, Any]],
    *,
    evaluation_id: str,
) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, suffix=".tmp", newline="", delete=False
        ) as handle:
            tmp_path = handle.name
            fieldnames = ["evaluation_id", "episode_index", *rows[0].keys()]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow(
                    {
                        "evaluation_id": evaluation_id,
                        "episode_index": index,
                        **row,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def evaluate_best_wall_ball(
    log_dir: str | Path,
    env_fn: Callable,
    *,
    algo: str | None = None,
    episode_len: int = 5_000,
    seeds: Sequence[int] = tuple(range(10_000, 10_050)),
    deterministic: bool = True,
    device: str = "cpu",
    summary_path: str | None = None,
    episodes_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate the task-selected WallBall model on held-out long episodes.

    The helper is intentionally strict: it evaluates ``best_model.zip``
    with the exact paired ``best_vec_normalize.pkl`` and verifies that
    ``best_model_meta.json`` selected the checkpoint by completed returns.
    Summary JSON and one-row-per-seed CSV files are written atomically under
    ``log_dir`` (which is the mounted Google Drive run folder in Colab).
    """
    log_dir = str(log_dir)
    if (
        isinstance(episode_len, bool)
        or not isinstance(episode_len, int)
        or episode_len <= 0
    ):
        raise ValueError("episode_len must be a positive integer")
    resolved_seeds = tuple(seeds)
    if not resolved_seeds:
        raise ValueError("seeds must contain at least one held-out seed")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in resolved_seeds
    ):
        raise ValueError("seeds must be unique non-negative integers")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be unique non-negative integers")

    config_path = os.path.join(log_dir, "config.json")
    meta_path = os.path.join(log_dir, "best_model_meta.json")
    model_path = os.path.join(log_dir, "best_model.zip")
    normalizer_path = os.path.join(log_dir, "best_vec_normalize.pkl")
    for path in (config_path, meta_path, model_path, normalizer_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"required best-model evaluation artifact is missing: {path}"
            )

    with open(config_path) as handle:
        config = json.load(handle)
    with open(meta_path) as handle:
        best_meta = json.load(handle)
    train_config = config.get("train_config") or {}
    training_env_config = config.get("env") or {}
    configured_evaluation_env = config.get("evaluation_env")
    if isinstance(configured_evaluation_env, Mapping):
        evaluation_env_config = configured_evaluation_env
        constructor_profile = "evaluation"
    else:
        # Runs produced before separate evaluation profiles recorded only the
        # training constructor. Preserve their verification path.
        evaluation_env_config = training_env_config
        constructor_profile = "training"
    selection_keys = best_meta.get("selection_keys") or []
    if evaluation_env_config.get("class") != "WallBallEnv":
        raise ValueError("long-horizon helper requires a WallBallEnv run")
    if train_config.get("headline_key") != "bounce_count":
        raise ValueError(
            "WallBall best model was not configured for bounce_count task selection"
        )
    if not selection_keys or selection_keys[0] != "bounce_count_ep_mean":
        raise ValueError(
            "best_model_meta.json does not prove bounce_count_ep_mean selection"
        )
    selected_timestep = best_meta.get("timestep")
    if (
        isinstance(selected_timestep, bool)
        or not isinstance(selected_timestep, int)
        or selected_timestep < 0
    ):
        raise ValueError("best_model_meta.json must contain a non-negative timestep")
    recorded_algo = str(train_config.get("algo", "")).upper()
    if not recorded_algo:
        raise ValueError("config.json does not record the training algorithm")
    if algo is not None and algo.upper() != recorded_algo:
        raise ValueError(
            f"requested algorithm {algo.upper()} does not match the run's "
            f"recorded algorithm {recorded_algo}"
        )

    model_sha256 = _sha256(model_path)
    normalizer_sha256 = _sha256(normalizer_path)
    meta_artifacts = best_meta.get("artifacts")
    if meta_artifacts is None:
        pair_verification = "same_run_directory_legacy_metadata"
    else:
        if not isinstance(meta_artifacts, Mapping):
            raise ValueError("best_model_meta.json artifacts must be an object")
        actual_hashes = {
            "best_model.zip": model_sha256,
            "best_vec_normalize.pkl": normalizer_sha256,
        }
        for filename, actual_hash in actual_hashes.items():
            entry = meta_artifacts.get(filename)
            expected_hash = entry.get("sha256") if isinstance(entry, Mapping) else None
            if expected_hash != actual_hash:
                raise ValueError(
                    f"{filename} does not match the SHA-256 bound into "
                    "best_model_meta.json"
                )
        pair_verification = "verified_by_best_model_meta_sha256"

    from stable_baselines3.common.utils import check_for_correct_spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from courtside_dynamics.training.algos import resolve_algo

    model = resolve_algo(recorded_algo).load(model_path, device=device)
    model_timestep = getattr(model, "num_timesteps", None)
    if model_timestep != selected_timestep:
        raise ValueError(
            "best_model.zip timestep does not match best_model_meta.json: "
            f"{model_timestep!r} != {selected_timestep}"
        )
    dummy = DummyVecEnv([env_fn])
    try:
        try:
            vec_norm = VecNormalize.load(normalizer_path, dummy)
        except Exception as exc:
            raise RuntimeError(
                "could not load best_vec_normalize.pkl with the long-horizon "
                "WallBall environment"
            ) from exc
    finally:
        dummy.close()
    loaded_norm_obs = bool(getattr(vec_norm, "norm_obs", False))
    loaded_clip_obs = float(getattr(vec_norm, "clip_obs", np.nan))
    loaded_exclusions = tuple(
        int(index)
        for index in getattr(vec_norm, "normalize_obs_excluded_indices", ())
    )
    expected_norm_obs = bool(train_config.get("normalize_obs", False))
    expected_clip_obs = float(train_config.get("clip_obs", np.nan))
    expected_exclusions = tuple(
        int(index)
        for index in train_config.get("normalize_obs_excluded_indices", ())
    )
    if loaded_norm_obs != expected_norm_obs:
        raise ValueError(
            "best_vec_normalize.pkl norm_obs does not match config.json"
        )
    if expected_norm_obs and not np.isclose(loaded_clip_obs, expected_clip_obs):
        raise ValueError(
            "best_vec_normalize.pkl clip_obs does not match config.json"
        )
    if loaded_exclusions != expected_exclusions:
        raise ValueError(
            "best_vec_normalize.pkl excluded observation indices do not "
            "match config.json"
        )
    vec_norm.training = False
    vec_norm.norm_reward = False

    env = env_fn()
    try:
        unwrapped = getattr(env, "unwrapped", env)
        actual_episode_len = getattr(unwrapped, "episode_len", None)
        if actual_episode_len != episode_len:
            raise ValueError(
                "long-horizon env factory must construct WallBallEnv with "
                f"episode_len={episode_len}; got {actual_episode_len!r}"
            )
        wrapper_limit = getattr(env, "_max_episode_steps", None)
        if wrapper_limit is not None and wrapper_limit < episode_len:
            raise ValueError(
                "an outer TimeLimit truncates before the requested long horizon"
            )
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        rows = [
            _rollout_wall_ball_seed(
                env,
                model,
                vec_norm.normalize_obs,
                seed=seed,
                episode_len=episode_len,
                deterministic=deterministic,
            )
            for seed in resolved_seeds
        ]
        observation_shape = list(env.observation_space.shape)
        action_shape = list(env.action_space.shape)
        evaluated_constructor_kwargs = _canonicalize_constructor_value(
            dict(getattr(unwrapped, "_ezpickle_kwargs", {}))
        )
        recovery_reset_probability = getattr(
            unwrapped, "recovery_reset_probability", None
        )
        if recovery_reset_probability is not None and not np.isclose(
            float(recovery_reset_probability), 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "long-horizon WallBall evaluation requires normal-only "
                "recovery_reset_probability=0"
            )
    finally:
        env.close()

    training_constructor_kwargs = training_env_config.get("constructor_kwargs")
    evaluation_constructor_kwargs = evaluation_env_config.get(
        "constructor_kwargs"
    )
    if isinstance(evaluation_constructor_kwargs, Mapping):
        if not _wall_ball_constructor_kwargs_match(
            evaluation_constructor_kwargs, evaluated_constructor_kwargs
        ):
            raise ValueError(
                "long-horizon WallBall constructor settings differ from the "
                f"recorded {constructor_profile} profile (apart from the "
                "intentional episode_len override)"
            )
        environment_verification = (
            f"verified_against_{constructor_profile}_constructor"
        )
    else:
        environment_verification = "legacy_config_without_constructor_kwargs"

    survival_steps = tuple(
        sorted(
            step
            for step in {750, 1_500, 3_000, 5_000, episode_len}
            if step <= episode_len
        )
    )
    summary = _summarize_wall_ball_episodes(rows, survival_steps=survival_steps)
    summary_path = summary_path or os.path.join(
        log_dir, "best_model_long_horizon_eval.json"
    )
    episodes_path = episodes_path or os.path.join(
        log_dir, "best_model_long_horizon_episodes.csv"
    )
    log_dir_real = os.path.realpath(os.path.abspath(log_dir))
    summary_path = os.path.realpath(os.path.abspath(summary_path))
    episodes_path = os.path.realpath(os.path.abspath(episodes_path))
    if summary_path == episodes_path:
        raise ValueError("summary_path and episodes_path must be different files")
    for output_path in (summary_path, episodes_path):
        if os.path.dirname(output_path) != log_dir_real:
            raise ValueError("long-horizon outputs must be direct children of log_dir")

    evaluation_id = uuid4().hex
    _atomic_write_csv(episodes_path, rows, evaluation_id=evaluation_id)
    episodes_sha256 = _sha256(episodes_path)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_run": {
            "config_artifact": os.path.basename(config_path),
            "config_sha256": _sha256(config_path),
            "timestamp_utc": config.get("timestamp_utc"),
            "git_sha": config.get("git_sha"),
            "versions": config.get("versions"),
        },
        "policy": {
            "algorithm": recorded_algo,
            "device": device,
            "deterministic": bool(deterministic),
            "artifact": os.path.basename(model_path),
            "sha256": model_sha256,
            "pair_verification": pair_verification,
            "best_model_meta": best_meta,
        },
        "normalization": {
            "artifact": os.path.basename(normalizer_path),
            "sha256": normalizer_sha256,
            "class": type(vec_norm).__name__,
            "norm_obs": loaded_norm_obs,
            "clip_obs": loaded_clip_obs,
            "normalize_obs_excluded_indices": list(loaded_exclusions),
            "training": False,
            "norm_reward": False,
        },
        "environment": {
            "class": "WallBallEnv",
            "episode_len": int(episode_len),
            "observation_shape": observation_shape,
            "action_shape": action_shape,
            "verification": environment_verification,
            "training_constructor_kwargs": training_constructor_kwargs,
            "evaluation_constructor_kwargs": evaluation_constructor_kwargs,
            "evaluated_constructor_kwargs": evaluated_constructor_kwargs,
        },
        "evaluation": {
            "episode_count": len(resolved_seeds),
            "seeds": list(resolved_seeds),
            "event_order_resolution": "policy_control_step",
            "return_survival_thresholds": list(
                WALL_BALL_RETURN_SURVIVAL_THRESHOLDS
            ),
        },
        "outputs": {
            "episodes_csv": {
                "artifact": os.path.basename(episodes_path),
                "sha256": episodes_sha256,
                "evaluation_id_column": "evaluation_id",
            },
            "summary_json": {
                "artifact": os.path.basename(summary_path),
                "commit_marker": True,
            },
        },
        **summary,
    }
    _atomic_write_json(summary_path, payload)
    print(f"Saved long-horizon episode metrics: {episodes_path}")
    print(f"Saved long-horizon evaluation summary: {summary_path}")
    returns = payload["metrics"]["completed_returns"]
    lengths = payload["metrics"]["episode_length"]
    print(
        "Best WallBall long-horizon evaluation: "
        f"returns={returns['mean']:.2f} (p50={returns['p50']:.0f}, "
        f"p90={returns['p90']:.0f}), mean_steps={lengths['mean']:.1f}"
    )
    survival_text = ", ".join(
        f">={threshold}: "
        f"{payload['return_survival_curve'][str(threshold)]['rate']:.1%}"
        for threshold in WALL_BALL_RETURN_SURVIVAL_THRESHOLDS[1:]
    )
    print(f"Return survival: {survival_text}")
    return payload


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
