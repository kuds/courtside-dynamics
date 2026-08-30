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

# Shared run-directory layout registry: writers resolve output paths via
# ``artifact_path`` (always the new layout), readers via
# ``locate_artifact`` (new layout first, legacy flat root as fallback),
# so every pre-0.14 run in Drive keeps working unmodified.
from courtside_dynamics.training.artifacts import (
    EXPECTED_ARTIFACTS,
    RUN_LAYOUT,
    artifact_path,
    locate_artifact,
)


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


def resolve_run_config_file(
    env_name: str,
    *,
    use_drive: bool = False,
    drive_subdir: str = "courtside-dynamics",
    local_root: str = "./configs",
) -> Path | None:
    """Return the experiment TOML for ``env_name``, creating it on first use.

    Root: ``<MyDrive>/Finding Theta/<drive_subdir>/configs/`` when
    ``use_drive=True`` and Drive is mounted (the same root convention as
    :func:`resolve_run_dir`), else ``local_root``. An existing
    ``<root>/<name_prefix>.toml`` is returned untouched -- the user's
    edits between runs are the whole point -- otherwise the packaged
    starter is copied there via
    :func:`courtside_dynamics.run_config.copy_starter_config`.

    Always prints the resolved path, its sha256, and whether it was
    created or reused, so the choice stays visible; the run additionally
    records both in ``config.json`` and copies the file into the run
    directory. Returns None (with a printed reason) for recipes without
    a packaged starter, so a default-config run is never blocked.
    """
    from courtside_dynamics.run_config import (
        available_run_configs,
        copy_starter_config,
    )

    if use_drive:
        my_drive = "/content/drive/MyDrive"
        if os.path.isdir(my_drive):
            root = os.path.join(
                my_drive, "Finding Theta", drive_subdir, "configs"
            )
        else:
            print(
                "[notebook_utils] use_drive=True but /content/drive/MyDrive "
                "is not mounted; resolving the run config under "
                f"{local_root} (ephemeral -- edits will not survive this "
                "runtime)."
            )
            root = local_root
    else:
        root = local_root

    catalog = available_run_configs()
    if env_name not in catalog:
        print(
            f"[notebook_utils] no packaged starter run config for "
            f"{env_name!r}; continuing without a config file."
        )
        return None

    destination = Path(root) / catalog[env_name].name
    created = not destination.exists()
    if created:
        destination = copy_starter_config(env_name, root)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(
        f"[notebook_utils] run config: {destination} "
        f"({'created from packaged starter' if created else 'reusing existing copy'}, "
        f"sha256 {digest[:12]})"
    )
    return destination


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

    Top row: per-episode training returns and episode lengths from the
    run's ``metrics/monitor/*.monitor.csv`` files with a rolling-mean
    overlay. Bottom row: deterministic eval returns and episode lengths
    (mean +/- std) from ``metrics/evaluations.npz`` (written by SB3's
    ``EvalCallback``). Both are resolved through ``locate_artifact`` so
    legacy flat-layout runs plot identically.
    """
    import matplotlib.pyplot as plt

    log_dir = str(log_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    monitor_dir = locate_artifact(log_dir, "monitor_dir") or os.path.join(
        log_dir, RUN_LAYOUT["monitor_dir"]
    )
    train_rewards, train_lengths = _read_monitor_logs(monitor_dir)

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

    eval_npz = locate_artifact(log_dir, "evaluations")
    eval_data = np.load(eval_npz) if eval_npz is not None else None

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
    path = locate_artifact(log_dir, "stage_summary") or os.path.join(
        str(log_dir), RUN_LAYOUT["stage_summary"]
    )
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
#: ``check_run_artifacts`` next to MISSING entries. Keyed by the
#: RUN_LAYOUT (new-layout) relative path.
_ARTIFACT_HINTS: dict[str, str] = {
    RUN_LAYOUT["config"]: "written at the very start of train(); absent means train() never ran on this dir",
    RUN_LAYOUT["monitor_dir"]: "monitor CSVs appear once training starts; rows once the first episodes finish",
    RUN_LAYOUT["progress_csv"]: "written by the CSV logger train() wires up; appears after SB3's first metric dump",
    RUN_LAYOUT["evaluations"]: "written by the reward eval stream; check eval_freq <= total_timesteps",
    RUN_LAYOUT["eval_info_csv"]: "written by InfoDictEvalCallback each eval; requires info_dict_eval=True",
    RUN_LAYOUT["best_model"]: "saved by EvalCallback on its first completed evaluation",
    RUN_LAYOUT["best_vec_normalize"]: "needs VecNormalize enabled plus at least one new-best eval",
    RUN_LAYOUT["checkpoints_dir"]: "requires checkpoint_freq > 0 and a run long enough to reach the first checkpoint",
    RUN_LAYOUT["videos_dir"]: "requires record_video=True, video_freq > 0, and moviepy installed",
    RUN_LAYOUT["final_model"]: "written when learn() finishes or is interrupted; absent means the run crashed",
    RUN_LAYOUT["vec_normalize"]: "only written when VecNormalize is enabled (normalize_obs / normalize_reward)",
    RUN_LAYOUT["stage_summary"]: "written by train()'s epilogue; absent means the run crashed before it",
    RUN_LAYOUT["learning_curve"]: "saved by the plot_learning_curve cell -- did it run with save_path set?",
    RUN_LAYOUT["eval_headline"]: "saved by the plot_eval_info cell -- did it run with save_path set?",
    RUN_LAYOUT["eval_terminations"]: "saved by the plot_eval_info cell -- did it run with save_path set?",
    RUN_LAYOUT["eval_rewards"]: "saved by the plot_eval_info cell -- did it run with save_path set?",
    RUN_LAYOUT["eval_diagnostics"]: "saved by the plot_eval_info cell -- did it run with save_path set?",
    RUN_LAYOUT["training_health_plot"]: "saved by the plot_training_health cell -- did it run with save_path set?",
    RUN_LAYOUT["best_model_video"]: "saved by record_best_model_video -- needs best_model.zip and rgb_array rendering",
    RUN_LAYOUT["best_long_eval"]: "saved by the WallBall long-horizon best-model evaluation cell",
    RUN_LAYOUT["best_long_eval_episodes"]: "saved with one row per held-out WallBall evaluation seed",
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
    and returns the missing relative paths (as recorded in
    ``RUN_LAYOUT``). Run it at the end of a notebook session, after the
    plotting and replay cells: it catches silently-skipped outputs (no
    video because moviepy failed, no best_model because eval never
    fired) while the Colab runtime still exists to do something about
    it. Artifacts whose label matches a ``RUN_LAYOUT`` name (all of
    ``EXPECTED_ARTIFACTS``, plus registry-named extras such as
    ``WALL_BALL_LONG_HORIZON_ARTIFACTS``) are resolved through
    ``locate_artifact``, so legacy flat-layout runs audit clean too.
    """
    log_dir = str(log_dir)
    artifacts = (*EXPECTED_ARTIFACTS, *extra_artifacts)
    missing: list[str] = []
    label_width = max(len(label) for label, _ in artifacts) + 2
    for label, rel in artifacts:
        if label in RUN_LAYOUT:
            full = locate_artifact(log_dir, label) or os.path.join(
                log_dir, RUN_LAYOUT[label]
            )
        else:
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
    ``rew_hold_ep_sum_mean`` -> ``("rew_hold", "ep_sum_mean")``,
    ``phase_frac_<label>`` -> ``("phase_frac", "<label>")``,
    standalone names like ``episode_length`` -> ``("episode_length", "")``.
    """
    if name.startswith("phase_frac_"):
        return "phase_frac", name[len("phase_frac_") :]
    # ``_ep_sum_mean`` and ``_ep_mean`` must be checked before ``_mean``
    # (their own suffix) so the per-episode variants overlay on the same
    # panel as ``_mean``/``_max``.
    for suffix in ("_ep_sum_mean", "_ep_mean", "_mean", "_final", "_max"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, ""


#: Metric-stem prefixes that make up the ``eval_headline.png`` page:
#: the counters runs are compared on plus reward/length context.
_EVAL_HEADLINE_STEM_PREFIXES: tuple[str, ...] = (
    "bounce_count",
    "episode_reward",
    "episode_length",
    "paddle_hit",
    "return_count",
    "legal_paddle",
)

#: Themed pages of the eval-info report, in render order. Values are the
#: RUN_LAYOUT names of the pages (``reports/<page>.png``).
_EVAL_INFO_PAGES: tuple[str, ...] = (
    "eval_headline",
    "eval_terminations",
    "eval_rewards",
    "eval_diagnostics",
)


def _eval_info_page_for_stem(stem: str) -> str:
    """Assign one metric stem to its themed ``plot_eval_info`` page."""
    if any(stem.startswith(prefix) for prefix in _EVAL_HEADLINE_STEM_PREFIXES):
        return "eval_headline"
    if stem.startswith("term_") or stem == "phase_frac":
        return "eval_terminations"
    if stem.startswith("rew_"):
        return "eval_rewards"
    return "eval_diagnostics"


def plot_eval_info(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    show: bool = True,
    max_cols: int = 3,
    max_panels_per_page: int = 24,
):
    """Themed per-metric time-series pages from the run's ``eval_info.csv``.

    The CSV is written by ``InfoDictEvalCallback`` in long format
    ``(timestep, metric, value)``. This function pivots it into panels
    with one panel per metric stem (e.g. ``rally_count``); ``_mean``,
    ``_final``, and ``_max`` variants are overlaid as separate lines on
    the same axes. ``phase_frac_<label>`` metrics share one panel.

    Instead of one ~170-panel mega-grid, panels are grouped into four
    themed pages -- ``eval_headline`` (task counters, reward, length),
    ``eval_terminations`` (``term_*`` rates + phase fractions),
    ``eval_rewards`` (``rew_*`` components), and ``eval_diagnostics``
    (everything else). Each page holds at most ``max_panels_per_page``
    panels; spillover flows to numbered continuation pages
    (``eval_diagnostics_2.png``, ...), keeping every file readable and
    small. A page whose metric group is empty for this env still renders
    (with a note saying so), so ``check_run_artifacts`` audits the same
    four pages for every recipe.

    ``save_path`` (optional) enables saving: a directory is used as the
    pages' target directory; a file path (e.g. the historical
    ``.../eval_info.png``, or ``artifact_path(log_dir,
    "eval_headline")``) saves the pages next to it under the standard
    page names. Returns the list of page figures, or ``None`` if the
    CSV is missing or empty.
    """
    import csv as _csv

    import matplotlib.pyplot as plt

    csv_path = locate_artifact(log_dir, "eval_info_csv")
    if csv_path is None:
        print(f"[notebook_utils] no eval_info.csv under {log_dir}")
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
    # leads its page, so the panel that answers "is the task improving?"
    # is the first thing on screen.
    headline_stem = None
    config_path = locate_artifact(log_dir, "config")
    if config_path is not None:
        import json as _json

        try:
            with open(config_path) as f:
                headline_stem = (
                    _json.load(f).get("train_config") or {}
                ).get("headline_key")
        except (OSError, ValueError):
            headline_stem = None

    page_stems: dict[str, list[str]] = {page: [] for page in _EVAL_INFO_PAGES}
    for stem in sorted(series):
        page_stems[_eval_info_page_for_stem(stem)].append(stem)
    for stems in page_stems.values():
        if headline_stem in stems:
            stems.remove(headline_stem)
            stems.insert(0, headline_stem)

    pages_dir: str | None = None
    if save_path:
        save_path = str(save_path)
        if os.path.splitext(save_path)[1]:
            pages_dir = os.path.dirname(save_path) or "."
        else:
            pages_dir = save_path
        os.makedirs(pages_dir, exist_ok=True)

    figures = []
    for page in _EVAL_INFO_PAGES:
        stems = page_stems[page]
        if not stems:
            # Keep the four-page contract even when this env logs no
            # metrics in the group, so the artifact audit stays uniform.
            fig = plt.figure(figsize=(5, 3.5))
            fig.text(
                0.5,
                0.5,
                f"no {page} metrics in eval_info.csv",
                ha="center",
                va="center",
            )
            if pages_dir is not None:
                fig.savefig(os.path.join(pages_dir, f"{page}.png"), dpi=90)
            figures.append(fig)
            continue
        for chunk_index in range(0, len(stems), max_panels_per_page):
            chunk = stems[chunk_index : chunk_index + max_panels_per_page]
            n = len(chunk)
            cols = min(max_cols, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(
                rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False
            )

            for i, stem in enumerate(chunk):
                ax = axes[i // cols][i % cols]
                for variant in sorted(series[stem]):
                    xs, ys = series[stem][variant]
                    ax.plot(
                        xs, ys, marker=".", markersize=3,
                        label=variant or "value",
                    )
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
            if pages_dir is not None:
                page_number = chunk_index // max_panels_per_page + 1
                filename = (
                    f"{page}.png"
                    if page_number == 1
                    else f"{page}_{page_number}.png"
                )
                # dpi tuned so a full 24-panel page stays well under
                # ~400 KB -- readable and casually downloadable.
                fig.savefig(os.path.join(pages_dir, filename), dpi=90)
            figures.append(fig)

    if show:
        plt.show()
    return figures


def plot_training_health(
    log_dir: str | Path,
    *,
    save_path: str | None = None,
    show: bool = True,
    max_cols: int = 3,
):
    """Per-metric grid of SB3's ``train/*`` diagnostics over training.

    Reads the run's ``metrics/progress.csv`` (written by the CSV logger
    wired up in ``train``; legacy runs keep it under ``tensorboard/``,
    which ``locate_artifact`` resolves transparently). One panel per
    ``train/*`` column -- for SAC that's ``ent_coef`` (the entropy
    temperature), ``actor_loss``, ``critic_loss``, ``ent_coef_loss``;
    for PPO ``explained_variance``, ``approx_kl``, ``clip_fraction``,
    etc. A collapsing ``ent_coef`` or a diverging ``critic_loss``
    explains a stalled SAC run that the reward curve alone won't.

    Returns ``None`` if ``progress.csv`` is missing or has no ``train/``
    columns yet.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = locate_artifact(log_dir, "progress_csv")
    if csv_path is None:
        print(f"[notebook_utils] no progress.csv under {log_dir}")
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
        rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False
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
    candidate = locate_artifact(log_dir, "best_model")
    if candidate is None:
        # Ancient runs saved SB3's extension-less "best_model" stem.
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
    path = locate_artifact(log_dir, "best_vec_normalize") or locate_artifact(
        log_dir, "vec_normalize"
    )
    if path is None:
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


#: Labels double as RUN_LAYOUT names so ``check_run_artifacts`` resolves
#: these extras through ``locate_artifact`` (new reports/ location with
#: the legacy flat-root fallback).
WALL_BALL_LONG_HORIZON_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("best_long_eval", RUN_LAYOUT["best_long_eval"]),
    ("best_long_eval_episodes", RUN_LAYOUT["best_long_eval_episodes"]),
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
        "pre_bounce_legal_paddle_hit_count",
        "post_bounce_legal_paddle_hit_count",
        "opening_volley_count",
        "post_bounce_completed_return_count",
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
    # Added in 0.13.0: render-only floor styling; any value is
    # physics-identical to the pre-0.13 default.
    "court_style": "diagnostic",
    # Added in 0.14.0: 0.0 reproduces the flat +1 wall reward exactly.
    "wall_reward_increment": 0.0,
    # Added in 0.25.0: -6.0 reproduces the historical in-play volume
    # exactly (the true-baseline era widens it per-task).
    "ball_in_play_min_x": -6.0,
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
        "pre_bounce_legal_hits": 0,
        "post_bounce_legal_hits": 0,
        "opening_volleys": 0,
        "post_bounce_completed_returns": 0,
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
            "pre_bounce_legal_hits": int(
                info["pre_bounce_legal_paddle_hit_count"]
            ),
            "post_bounce_legal_hits": int(
                info["post_bounce_legal_paddle_hit_count"]
            ),
            "opening_volleys": int(info["opening_volley_count"]),
            "post_bounce_completed_returns": int(
                info["post_bounce_completed_return_count"]
            ),
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
        if (
            current_counts["pre_bounce_legal_hits"]
            + current_counts["post_bounce_legal_hits"]
            != current_counts["legal_paddle_hits"]
        ):
            raise ValueError(
                "WallBall pre/post-bounce legal hits must partition legal hits"
            )
        if (
            current_counts["opening_volleys"]
            > current_counts["pre_bounce_legal_hits"]
            or current_counts["opening_volleys"] > 1
        ):
            raise ValueError(
                "WallBall opening volleys must be a zero-or-one subset "
                "of pre-bounce legal hits"
            )
        if current_counts["post_bounce_completed_returns"] > min(
            current_counts["post_bounce_legal_hits"],
            current_counts["completed_returns"],
        ):
            raise ValueError(
                "WallBall post-bounce completed returns must be backed by "
                "post-bounce legal hits and completed returns"
            )
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
        "pre_bounce_legal_paddle_hit_count": int(
            final_info["pre_bounce_legal_paddle_hit_count"]
        ),
        "post_bounce_legal_paddle_hit_count": int(
            final_info["post_bounce_legal_paddle_hit_count"]
        ),
        "opening_volley_count": int(final_info["opening_volley_count"]),
        "post_bounce_completed_return_count": int(
            final_info["post_bounce_completed_return_count"]
        ),
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
            "pre_bounce_legal_paddle_hit_count",
            "post_bounce_legal_paddle_hit_count",
            "opening_volley_count",
            "post_bounce_completed_return_count",
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
    legal_hit_total = sum(
        int(row["legal_paddle_hit_count"]) for row in rows
    )
    post_bounce_hit_total = sum(
        int(row["post_bounce_legal_paddle_hit_count"]) for row in rows
    )
    completed_return_total = sum(
        int(row["completed_returns"]) for row in rows
    )
    post_bounce_return_total = sum(
        int(row["post_bounce_completed_return_count"]) for row in rows
    )
    opening_volley_total = sum(
        int(row["opening_volley_count"]) for row in rows
    )
    contact_sequence_diagnostics = {
        "legal_hit_total": int(legal_hit_total),
        "pre_bounce_legal_hit_total": int(
            sum(
                int(row["pre_bounce_legal_paddle_hit_count"])
                for row in rows
            )
        ),
        "post_bounce_legal_hit_total": int(post_bounce_hit_total),
        "post_bounce_legal_hit_rate": (
            float(post_bounce_hit_total / legal_hit_total)
            if legal_hit_total
            else 0.0
        ),
        "opening_volley_total": int(opening_volley_total),
        "episodes_with_opening_volley": int(
            sum(int(row["opening_volley_count"]) > 0 for row in rows)
        ),
        "opening_volley_episode_rate": float(
            sum(int(row["opening_volley_count"]) > 0 for row in rows) / n
        ),
        "completed_return_total": int(completed_return_total),
        "post_bounce_completed_return_total": int(
            post_bounce_return_total
        ),
        "post_bounce_completed_return_rate": (
            float(post_bounce_return_total / completed_return_total)
            if completed_return_total
            else 0.0
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
        "contact_sequence_diagnostics": contact_sequence_diagnostics,
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

    def _required_artifact(name: str) -> str:
        # New layout first, legacy flat root as fallback; when neither
        # exists, report the canonical new path so the error is concrete.
        located = locate_artifact(log_dir, name)
        return located or os.path.join(log_dir, RUN_LAYOUT[name])

    config_path = _required_artifact("config")
    meta_path = _required_artifact("best_model_meta")
    model_path = _required_artifact("best_model")
    normalizer_path = _required_artifact("best_vec_normalize")
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
    summary_path = summary_path or artifact_path(log_dir, "best_long_eval")
    episodes_path = episodes_path or artifact_path(
        log_dir, "best_long_eval_episodes"
    )
    log_dir_real = os.path.realpath(os.path.abspath(log_dir))
    # The reports/ folder of the run (where the registry places the
    # long-horizon outputs); the run root remains accepted for callers
    # that pass explicit legacy-layout paths.
    reports_dir_real = os.path.realpath(
        os.path.dirname(
            os.path.join(log_dir_real, RUN_LAYOUT["best_long_eval"])
        )
    )
    summary_path = os.path.realpath(os.path.abspath(summary_path))
    episodes_path = os.path.realpath(os.path.abspath(episodes_path))
    if summary_path == episodes_path:
        raise ValueError("summary_path and episodes_path must be different files")
    for output_path in (summary_path, episodes_path):
        if os.path.dirname(output_path) not in (log_dir_real, reports_dir_real):
            raise ValueError(
                "long-horizon outputs must live in log_dir or its "
                "reports/ folder"
            )

    evaluation_id = uuid4().hex
    _atomic_write_csv(episodes_path, rows, evaluation_id=evaluation_id)
    episodes_sha256 = _sha256(episodes_path)
    payload: dict[str, Any] = {
        "schema_version": 3,
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

    out_path = out_path or artifact_path(log_dir, "best_model_video")
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


# ---------------------------------------------------------------------------
# PaddleTennis staged campaign (notebooks/paddle_tennis_campaign.ipynb)
# ---------------------------------------------------------------------------
#
# The campaign notebook chains PaddleTennis training legs in one
# execution -- a from-scratch gate leg scored by the behavioral
# diagnosis instrument, then a main leg warm-started from whichever
# lineage the gate verdict selects -- and records every decision in a
# Drive-side manifest so a disconnected session resumes instead of
# retraining. The helpers here are deliberately engine-free: they read
# artifacts a finished ``train()`` run already wrote, score them
# against bars the notebook froze before launch, and never touch a run
# directory beyond adding a report under ``reports/``.

PADDLE_CAMPAIGN_MANIFEST_NAME = "campaign_manifest.json"
PADDLE_GATE_REPORT_NAME = "campaign_gate.json"

_BAR_SPEC_KEYS = frozenset(
    {"metric", "pass_at", "fail_at", "higher_is_better", "gating"}
)


def load_campaign_manifest(campaign_root: str | Path) -> dict[str, Any] | None:
    """Return the campaign's manifest, or ``None`` for a fresh root.

    The manifest is the campaign's resume point: which legs completed,
    into which run directories, what each report's verdict was, and
    which warm-start branch was decided. It is written atomically by
    :func:`write_campaign_manifest` after every state change, so a
    Colab disconnect can land anywhere and the next execution still
    sees a consistent picture.
    """
    path = os.path.join(str(campaign_root), PADDLE_CAMPAIGN_MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"campaign manifest must be a JSON object: {path}")
    return manifest


def write_campaign_manifest(
    campaign_root: str | Path, manifest: Mapping[str, Any]
) -> str:
    """Atomically persist the campaign manifest; returns its path."""
    path = os.path.join(str(campaign_root), PADDLE_CAMPAIGN_MANIFEST_NAME)
    _atomic_write_json(path, dict(manifest))
    return path


def _json_normalized(value: Any) -> Any:
    """Round-trip ``value`` through JSON so tuples compare as lists."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def require_campaign_fingerprint(
    manifest: Mapping[str, Any], fingerprint: Mapping[str, Any]
) -> None:
    """Refuse to resume a campaign whose frozen settings changed.

    The campaign notebook's settings cell is the campaign's frozen
    protocol (budgets, bars, seeds, the fallback lineage). Resuming
    after a disconnect must mean *continuing the same protocol*, so
    the manifest records a fingerprint of those settings at creation
    and every later execution has to match it -- the alternative is a
    campaign whose main leg silently runs under different rules than
    the gate leg it chains from (the standing "changes after start
    void the run" doctrine, enforced in code).
    """
    recorded = manifest.get("fingerprint")
    if not isinstance(recorded, Mapping):
        raise ValueError(
            "campaign manifest has no fingerprint; this root was not "
            "created by the campaign notebook -- choose a fresh "
            "CAMPAIGN_ID instead of resuming it"
        )
    recorded_normalized = _json_normalized(dict(recorded))
    current_normalized = _json_normalized(dict(fingerprint))
    if recorded_normalized == current_normalized:
        return
    missing = object()
    changed = sorted(
        key
        for key in set(recorded_normalized) | set(current_normalized)
        if recorded_normalized.get(key, missing)
        != current_normalized.get(key, missing)
    )
    raise ValueError(
        "campaign settings changed since this campaign was created "
        f"(differing keys: {changed}); a resumed campaign must run the "
        "protocol it froze. Restore the recorded settings, or start a "
        "new campaign under a fresh CAMPAIGN_ID."
    )


def next_stage_attempt_dir(campaign_root: str | Path, stage_name: str) -> str:
    """Create and return ``<root>/<stage>/attempt_NN`` for a fresh leg.

    Every training attempt gets a directory ``train()`` has never
    touched: a leg that crashed or was interrupted keeps its partial
    artifacts in place as evidence, and the retry starts clean in the
    next-numbered attempt instead of mixing two runs' artifacts in one
    directory.
    """
    if (
        not stage_name
        or os.sep in stage_name
        or stage_name != stage_name.strip()
    ):
        raise ValueError(
            f"stage_name must be a bare directory name: {stage_name!r}"
        )
    stage_root = os.path.join(str(campaign_root), stage_name)
    os.makedirs(stage_root, exist_ok=True)
    numbers = []
    for entry in os.listdir(stage_root):
        prefix, _, suffix = entry.partition("_")
        if prefix == "attempt" and suffix.isdigit():
            numbers.append(int(suffix))
    path = os.path.join(
        stage_root, f"attempt_{max(numbers, default=0) + 1:02d}"
    )
    os.makedirs(path)
    return path


def paddle_campaign_metrics(
    traces: Sequence[Any], interpoint_travels: Sequence[float] | None = None
) -> dict[str, Any]:
    """Aggregate diagnosis traces into the campaign's scalar metrics.

    ``traces`` are
    :class:`courtside_dynamics.training.paddle_diagnosis.EpisodeTrace`
    rows from ``run_player`` (accepted duck-typed so tests can build
    lightweight stand-ins). The aggregation mirrors the instrument's
    ``report()`` exactly -- a bar must read the same number a human
    reads in the run's own ``diagnosis_probe_*.txt``:

    - ``touched_after_bounce_rate`` pools the per-bounce touch flags;
    - ``k{1,2,3}_{receiving,serving}_survival`` is the fraction of the
      split's points with at least k policy hits (``None`` when the
      split is empty);
    - ``k{2,3}_either_survival`` takes the better parity (the
      registered RK1/RK2 criteria are "either parity");
    - ``ready_error_mean`` pools the bounce-time positional errors;
    - ``recovery_hold_travel_{mean,p90}`` pool the per-hit hold-window
      travels (the post-swing wander, the campaign's k=2 blocker
      metric);
    - ``interpoint_travel_mean`` is the R2 watch metric; when the
      caller has no travels list (``None``) it and
      ``interpoint_boundaries`` read ``None`` rather than a fabricated
      zero;
    - ``unsafe_terminations`` counts forced non-finite endings.
    """
    if not traces:
        raise ValueError("at least one diagnosis trace is required")

    receiving = [t for t in traces if not t.serve_side_is_policy]
    serving = [t for t in traces if t.serve_side_is_policy]

    def _survival(split: Sequence[Any], k: int) -> float | None:
        if not split:
            return None
        return float(sum(t.policy_hits >= k for t in split) / len(split))

    touched = [bool(flag) for t in traces for flag in t.touched_after_bounce]
    ready = [float(x) for t in traces for x in t.ready_errors]
    recovery = [float(x) for t in traces for x in t.recovery_travel]
    travels = (
        None
        if interpoint_travels is None
        else [float(x) for x in interpoint_travels]
    )

    metrics: dict[str, Any] = {
        "points": len(traces),
        "receiving_points": len(receiving),
        "serving_points": len(serving),
        "touched_after_bounce_rate": (
            float(np.mean(touched)) if touched else None
        ),
        "touched_after_bounce_samples": len(touched),
        "ready_error_mean": float(np.mean(ready)) if ready else None,
        "ready_error_p90": (
            float(np.percentile(ready, 90)) if ready else None
        ),
        "ready_error_samples": len(ready),
        "recovery_hold_travel_mean": (
            float(np.mean(recovery)) if recovery else None
        ),
        "recovery_hold_travel_p90": (
            float(np.percentile(recovery, 90)) if recovery else None
        ),
        "recovery_hold_samples": len(recovery),
        "crossings_mean": float(np.mean([t.crossings for t in traces])),
        "unsafe_terminations": int(
            sum(t.termination == "nonfinite_state" for t in traces)
        ),
        "interpoint_travel_mean": (
            float(np.mean(travels)) if travels else None
        ),
        "interpoint_boundaries": (
            len(travels) if travels is not None else None
        ),
    }
    for split_label, split in (("receiving", receiving), ("serving", serving)):
        for k in (1, 2, 3):
            metrics[f"k{k}_{split_label}_survival"] = _survival(split, k)
    for k in (2, 3):
        parities = [
            value
            for value in (
                metrics[f"k{k}_receiving_survival"],
                metrics[f"k{k}_serving_survival"],
            )
            if value is not None
        ]
        metrics[f"k{k}_either_survival"] = max(parities) if parities else None
    return metrics


def score_campaign_bars(
    metrics: Mapping[str, Any], bars: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Score aggregated metrics against a frozen bar table.

    Each bar spec is ``{"metric", "pass_at", "fail_at",
    "higher_is_better", "gating"}``. A bar reads PASS at its inclusive
    ``pass_at`` edge, FAIL strictly beyond ``fail_at``, and MIDDLE in
    between -- a reading exactly on the fail line lands MIDDLE, so any
    edge ambiguity in a pre-registration's bands routes to the
    maintainer rather than into an automated branch. A ``None`` metric
    scores NO_DATA (the instrument saw no qualifying events), which is
    never a PASS.

    The overall ``verdict`` is computed over gating bars only: FAIL if
    any gating bar FAILs, PASS if every gating bar PASSes, otherwise
    MIDDLE. A table with no gating bars (a record-only report) scores
    ``"RECORDED"``.
    """
    if not bars:
        raise ValueError("at least one bar is required")
    scored: dict[str, Any] = {}
    for name, spec in bars.items():
        unknown = set(spec) - _BAR_SPEC_KEYS
        absent = _BAR_SPEC_KEYS - set(spec)
        if unknown or absent:
            raise ValueError(
                f"bar {name!r} spec keys mismatch: missing "
                f"{sorted(absent)}, unknown {sorted(unknown)}"
            )
        if spec["metric"] not in metrics:
            raise KeyError(
                f"bar {name!r} reads unknown metric {spec['metric']!r}; "
                f"known: {sorted(metrics)}"
            )
        pass_at = float(spec["pass_at"])
        fail_at = float(spec["fail_at"])
        if spec["higher_is_better"]:
            if not fail_at <= pass_at:
                raise ValueError(
                    f"bar {name!r}: higher-is-better needs "
                    "fail_at <= pass_at"
                )
        elif not pass_at <= fail_at:
            raise ValueError(
                f"bar {name!r}: lower-is-better needs pass_at <= fail_at"
            )
        value = metrics[spec["metric"]]
        if value is None:
            verdict = "NO_DATA"
        elif spec["higher_is_better"]:
            if value >= pass_at:
                verdict = "PASS"
            elif value < fail_at:
                verdict = "FAIL"
            else:
                verdict = "MIDDLE"
        elif value <= pass_at:
            verdict = "PASS"
        elif value > fail_at:
            verdict = "FAIL"
        else:
            verdict = "MIDDLE"
        scored[name] = {
            "metric": spec["metric"],
            "value": value,
            "pass_at": pass_at,
            "fail_at": fail_at,
            "higher_is_better": bool(spec["higher_is_better"]),
            "gating": bool(spec["gating"]),
            "verdict": verdict,
        }
    gating = [bar for bar in scored.values() if bar["gating"]]
    if not gating:
        overall = "RECORDED"
    elif any(bar["verdict"] == "FAIL" for bar in gating):
        overall = "FAIL"
    elif all(bar["verdict"] == "PASS" for bar in gating):
        overall = "PASS"
    else:
        overall = "MIDDLE"
    return {"bars": scored, "verdict": overall}


def score_paddle_stage(
    run_dir: str | Path,
    *,
    bars: Mapping[str, Mapping[str, Any]],
    episodes: int = 30,
    seed_start: int = 5200,
    report_name: str = PADDLE_GATE_REPORT_NAME,
) -> dict[str, Any]:
    """Score a finished PaddleTennis leg's best checkpoint against bars.

    Loads the leg's protected ``best_model.zip`` with its paired
    ``best_vec_normalize.pkl``, rebuilds the evaluation environment
    from the run's own recorded constructor kwargs (the instrument
    must measure the task the leg actually trained on -- the
    ``DiagnosisProbeCallback`` contract), replays the behavioral
    diagnosis instrument on the calibration block, aggregates the
    traces via :func:`paddle_campaign_metrics`, and scores ``bars``
    via :func:`score_campaign_bars`. The defaults mirror the recipe's
    in-run ``checkpoint_diagnosis`` (30 episodes, seeds 5200+), so
    this report and the run's own ``diagnosis_probe_*.txt`` rows read
    the same instrument on the same seeds.

    This is a **single-checkpoint reading**: a pre-registered
    "at some checkpoint" bar is under-approximated here (the best
    checkpoint is one of them), so a PASS is sound while a FAIL may
    under-credit a run whose peak on that metric came elsewhere.

    The payload is written atomically to ``reports/<report_name>``
    under ``run_dir`` and returned.
    """
    from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
    from courtside_dynamics.training.paddle_diagnosis import (
        native_checkpoint_policy,
        run_player,
    )

    run_dir = str(run_dir)
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes <= 0
    ):
        raise ValueError("episodes must be a positive integer")

    located = {
        name: locate_artifact(run_dir, name)
        for name in ("config", "best_model", "best_vec_normalize")
    }
    missing = [
        os.path.join(run_dir, RUN_LAYOUT[name])
        for name, path in located.items()
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "stage scoring requires a finished run with a protected best "
            f"checkpoint; missing: {missing}"
        )
    config_path = located["config"]
    model_path = located["best_model"]
    normalizer_path = located["best_vec_normalize"]
    assert config_path and model_path and normalizer_path

    with open(config_path) as handle:
        config = json.load(handle)
    env_config = config.get("evaluation_env") or config.get("env") or {}
    if env_config.get("class") != "PaddleTennisEnv":
        raise ValueError(
            "stage scoring requires a PaddleTennisEnv run; config.json "
            f"records {env_config.get('class')!r}"
        )
    constructor_kwargs = env_config.get("constructor_kwargs")
    if not isinstance(constructor_kwargs, Mapping):
        raise ValueError(
            "config.json records no evaluation constructor kwargs; the "
            "instrument cannot rebuild the leg's environment"
        )
    env_kwargs = dict(constructor_kwargs)

    policy = native_checkpoint_policy(model_path, normalizer_path)
    traces, travels = run_player(
        policy,
        episodes=episodes,
        seed_start=seed_start,
        env_fn=lambda: PaddleTennisEnv(**env_kwargs),
    )
    metrics = paddle_campaign_metrics(traces, travels)
    scored = score_campaign_bars(metrics, bars)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "policy": {
            "artifact": os.path.basename(model_path),
            "sha256": _sha256(model_path),
        },
        "normalization": {
            "artifact": os.path.basename(normalizer_path),
            "sha256": _sha256(normalizer_path),
        },
        "environment": {
            "class": "PaddleTennisEnv",
            "constructor_kwargs": _canonicalize_constructor_value(env_kwargs),
        },
        "evaluation": {
            "instrument": "paddle_diagnosis.run_player",
            "checkpoint": "best",
            "reading": "single_checkpoint",
            "episodes": int(episodes),
            "seed_start": int(seed_start),
        },
        "metrics": metrics,
        "bars": scored["bars"],
        "verdict": scored["verdict"],
    }
    report_path = os.path.join(run_dir, "reports", report_name)
    _atomic_write_json(report_path, payload)
    bar_text = ", ".join(
        f"{name}={bar['verdict']}" for name, bar in scored["bars"].items()
    )
    print(
        f"[campaign] {report_name}: verdict {scored['verdict']} ({bar_text})"
    )
    print(f"[campaign] wrote {report_path}")
    return payload


def resolve_warm_start_branch(
    verdict: str,
    *,
    stage_run_dir: str | Path,
    fallback_run_dir: str | Path | None,
    middle_action: str = "stop",
) -> tuple[str, str | None]:
    """Map a gate verdict to the main leg's warm-start source.

    Returns ``(branch, source_run_dir)``:

    - ``PASS`` -> ``("continue", stage_run_dir)``: the from-scratch
      lineage keeps going from its own best checkpoint.
    - ``FAIL`` -> ``("fallback", fallback_run_dir)`` when a fallback
      lineage is configured, else ``("stop", None)``: the from-scratch
      attempt is booked FAIL, and with nowhere frozen to branch to the
      campaign stops rather than improvising.
    - ``MIDDLE`` / ``NO_DATA`` follow ``middle_action``: ``"stop"``
      (the default -- a declared middle is a maintainer's call by
      doctrine), ``"continue"``, or ``"fallback"`` (which also stops
      when no fallback lineage is configured).
    """
    if middle_action not in ("stop", "continue", "fallback"):
        raise ValueError(
            f"middle_action must be stop|continue|fallback: {middle_action!r}"
        )
    if verdict == "PASS":
        return "continue", str(stage_run_dir)
    if verdict == "FAIL":
        if fallback_run_dir is None:
            return "stop", None
        return "fallback", str(fallback_run_dir)
    if verdict in ("MIDDLE", "NO_DATA"):
        if middle_action == "continue":
            return "continue", str(stage_run_dir)
        if middle_action == "fallback" and fallback_run_dir is not None:
            return "fallback", str(fallback_run_dir)
        return "stop", None
    raise ValueError(f"unknown gate verdict: {verdict!r}")


_PLAN_TRAIN_CONFIG_KEYS = (
    "seed",
    "total_timesteps",
    "n_envs",
    "eval_freq",
    "checkpoint_freq",
)
_PLAN_KEYS = frozenset(
    {
        *_PLAN_TRAIN_CONFIG_KEYS,
        "env_class",
        "env_kwargs",
        "eval_env_kwargs",
        "drill_library_sha256",
        "warm_start",
    }
)
_PLAN_WARM_START_KEYS = frozenset(
    {
        "source_run_dir_suffix",
        "transfer_log_ent_coef",
        "expected_artifact_sha256",
    }
)
_PLAN_MISSING = object()


def _plan_values_match(recorded: Any, wanted: Any) -> bool:
    """Compare a recorded ``config.json`` value against a plan value.

    ``config.json`` holds JSON-native values while the plan comes from
    live notebook settings, so a planned tuple must compare equal to
    its recorded list form. Values JSON cannot round-trip fall back to
    plain equality.
    """
    try:
        return bool(_json_normalized(recorded) == _json_normalized(wanted))
    except (TypeError, ValueError):
        return bool(recorded == wanted)


class RunConfigPlanMismatch(ValueError):
    """A run's ``config.json`` diverges from its frozen plan.

    Raised by :func:`validate_run_config_against_plan` for genuine
    config drift only. A malformed plan or ``config.json`` raises plain
    :class:`ValueError` (note ``json.JSONDecodeError`` subclasses it),
    so a caller booking drift -- the campaign notebook's manifest --
    does not misattribute instrument errors as drift.
    """


def validate_run_config_against_plan(
    config_json_path: str | Path, expected: Mapping[str, Any]
) -> None:
    """Validate a run's ``config.json`` against a frozen plan.

    The pre-registration launch checklists
    (``docs/paddle_tennis_registered_run_prereg_20260816.md`` section 6,
    ``docs/paddle_tennis_lt1_prereg_20260823.md`` section 6) require the
    run's own ``config.json`` to be checked against the frozen plan
    before anything downstream consumes the run -- a drifted run must
    stop the campaign, not be gated and branched from. This is that
    check in code. Every divergence is collected and a single
    :class:`RunConfigPlanMismatch` lists them all, so the maintainer
    reads the full drift in one message instead of one mismatch per
    re-run; a malformed plan or ``config.json`` raises plain
    :class:`ValueError` instead (an instrument error, not drift).
    ``train()`` writes ``config.json`` at run start, so the check can
    also be run manually before the first checkpoint lands (the
    notebook automates it post-run as the backstop).

    ``expected`` maps plan keys to frozen values; only supplied keys are
    checked, and an unknown key is rejected loudly rather than silently
    validating nothing:

    - ``"seed"``, ``"total_timesteps"``, ``"n_envs"``, ``"eval_freq"``,
      ``"checkpoint_freq"``: exact matches against the recorded
      ``train_config`` block;
    - ``"env_class"``: the training environment's recorded class;
    - ``"env_kwargs"``: subset match against the training environment's
      recorded constructor kwargs (the plan pins the kwargs it cares
      about; recipe defaults it does not name stay unchecked);
    - ``"eval_env_kwargs"``: the same subset match against the recorded
      ``evaluation_env`` block's constructor kwargs — the eval-side pin
      the k=2 drill design's D6 needs (a run config's ``[eval_env]``
      table layers above the recipe's eval overrides, so the frozen
      plan asserts the drill kwargs are off on the evaluation env
      specifically);
    - ``"drill_library_sha256"``: full sha256 or prefix — lowercase
      hex, 8 to 64 chars, a malformed pin is a bad plan, not a
      mismatch — matched against the training environment's recorded
      construction-time drill-library digest (``env.drill_library_sha256``,
      banked into ``config.json`` by the env probe), i.e. what the run
      actually loaded, not what the file at the path contains at audit
      time;
    - ``"warm_start"``: ``None`` demands a from-scratch run (no
      ``initialization`` block, no ``train_config.warm_start``). A
      mapping demands a warm-started run -- ``train()`` binds the
      resolved ``initialization`` provenance into ``config.json``, so
      call this after training -- and may pin any of:

      - ``"source_run_dir_suffix"``: the recorded
        ``initialization.source_run_dir`` must equal it or end with
        ``"/" + suffix`` -- whole path components only, so a pin
        cannot match an evil sibling directory (mount points move
        between machines; the run-dir leaf does not);
      - ``"transfer_log_ent_coef"``: the recorded flag must match
        verbatim. Temperature membership is enforced only where
        ``train()`` records it: under ``True``,
        ``initialization.reset`` must not list ``"log_ent_coef"``, and
        ``initialization.transferred`` must list it when
        ``transferred_ent_coef`` is recorded (the evidence a
        temperature actually moved); under ``False``, neither
        ``transferred`` membership nor a recorded
        ``transferred_ent_coef`` is allowed, and ``reset`` membership
        is not required -- PPO and fixed-``ent_coef`` warm starts have
        no temperature to move, so they record the flag alone (the
        documented no-op) and pass on the flag comparison;
      - ``"expected_artifact_sha256"``: mapping of source artifact name
        to the full sha256 or a prefix -- lowercase hex, 8 to 64 chars,
        validated like
        :class:`~courtside_dynamics.training.WarmStartConfig` (a
        malformed pin is a bad plan, not a mismatch) -- matched against
        ``initialization.source_artifacts`` (``None`` skips the check,
        mirroring ``WarmStartConfig``).
    """
    if not expected:
        raise ValueError("an expected plan with at least one key is required")
    unknown = set(expected) - _PLAN_KEYS
    if unknown:
        raise ValueError(
            f"unknown expected-plan keys: {sorted(unknown)}; "
            f"known: {sorted(_PLAN_KEYS)}"
        )
    with open(config_json_path) as handle:
        config = json.load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(
            f"config.json must be a JSON object: {config_json_path}"
        )

    mismatches: list[str] = []
    train_config = config.get("train_config")
    if not isinstance(train_config, Mapping):
        train_config = {}
    env_info = config.get("env")
    if not isinstance(env_info, Mapping):
        env_info = {}

    for key in _PLAN_TRAIN_CONFIG_KEYS:
        if key not in expected:
            continue
        recorded = train_config.get(key, _PLAN_MISSING)
        if recorded is _PLAN_MISSING:
            mismatches.append(
                f"train_config.{key}: expected {expected[key]!r}, "
                "config.json records nothing"
            )
        elif not _plan_values_match(recorded, expected[key]):
            mismatches.append(
                f"train_config.{key}: expected {expected[key]!r}, "
                f"config.json records {recorded!r}"
            )

    if "env_class" in expected:
        recorded_class = env_info.get("class")
        if recorded_class != expected["env_class"]:
            mismatches.append(
                f"env.class: expected {expected['env_class']!r}, "
                f"config.json records {recorded_class!r}"
            )

    if "env_kwargs" in expected:
        mismatches.extend(
            _env_kwargs_plan_mismatches(
                expected["env_kwargs"],
                env_info,
                block="env",
                key="env_kwargs",
            )
        )

    if "eval_env_kwargs" in expected:
        eval_env_info = config.get("evaluation_env")
        if not isinstance(eval_env_info, Mapping):
            eval_env_info = {}
        mismatches.extend(
            _env_kwargs_plan_mismatches(
                expected["eval_env_kwargs"],
                eval_env_info,
                block="evaluation_env",
                key="eval_env_kwargs",
            )
        )

    if "drill_library_sha256" in expected:
        wanted_sha = expected["drill_library_sha256"]
        if (
            not isinstance(wanted_sha, str)
            or not 8 <= len(wanted_sha) <= 64
            or any(c not in "0123456789abcdef" for c in wanted_sha)
        ):
            raise ValueError(
                "expected drill_library_sha256 must be lowercase hex, "
                "8 to 64 chars"
            )
        recorded_sha = env_info.get("drill_library_sha256")
        if not isinstance(recorded_sha, str):
            mismatches.append(
                f"env.drill_library_sha256: expected {wanted_sha!r}, "
                "config.json records no consumed drill-library digest"
            )
        elif not recorded_sha.startswith(wanted_sha):
            mismatches.append(
                f"env.drill_library_sha256: expected {wanted_sha!r}, "
                f"config.json records {recorded_sha!r}"
            )

    if "warm_start" in expected:
        mismatches.extend(
            _warm_start_plan_mismatches(
                expected["warm_start"],
                initialization=config.get("initialization"),
                recorded_warm_start=train_config.get("warm_start"),
            )
        )

    if mismatches:
        raise RunConfigPlanMismatch(
            f"config.json diverges from the frozen plan in "
            f"{len(mismatches)} place(s) ({config_json_path}):\n- "
            + "\n- ".join(mismatches)
        )


def _env_kwargs_plan_mismatches(
    wanted_kwargs: Any,
    recorded_env_info: Mapping[str, Any],
    *,
    block: str,
    key: str,
) -> list[str]:
    """Subset-match a plan's kwargs pin against one recorded env block
    (``env`` or ``evaluation_env``) for
    :func:`validate_run_config_against_plan`."""
    if not isinstance(wanted_kwargs, Mapping):
        raise ValueError(f"expected {key} must be a mapping")
    mismatches: list[str] = []
    recorded_kwargs = recorded_env_info.get("constructor_kwargs")
    if not isinstance(recorded_kwargs, Mapping):
        if wanted_kwargs:
            mismatches.append(
                f"{block}.constructor_kwargs: expected "
                f"{sorted(wanted_kwargs)}, config.json records no "
                "constructor kwargs"
            )
        return mismatches
    for name in sorted(wanted_kwargs):
        wanted_value = wanted_kwargs[name]
        if name not in recorded_kwargs:
            mismatches.append(
                f"{block} kwarg {name!r}: expected {wanted_value!r}, "
                "config.json records no such kwarg"
            )
        elif not _plan_values_match(recorded_kwargs[name], wanted_value):
            mismatches.append(
                f"{block} kwarg {name!r}: expected {wanted_value!r}, "
                f"config.json records {recorded_kwargs[name]!r}"
            )
    return mismatches


def _warm_start_plan_mismatches(
    wanted: Mapping[str, Any] | None,
    *,
    initialization: Any,
    recorded_warm_start: Any,
) -> list[str]:
    """Collect the warm-start mismatches for
    :func:`validate_run_config_against_plan`."""
    if wanted is None:
        mismatches = []
        if initialization is not None:
            mismatches.append(
                "warm start: expected a from-scratch run, config.json "
                "records an initialization block"
            )
        if recorded_warm_start is not None:
            mismatches.append(
                "train_config.warm_start: expected None (from scratch), "
                f"config.json records {recorded_warm_start!r}"
            )
        return mismatches
    if not isinstance(wanted, Mapping):
        raise ValueError("expected warm_start must be a mapping or None")
    unknown = set(wanted) - _PLAN_WARM_START_KEYS
    if unknown:
        raise ValueError(
            f"unknown expected warm_start keys: {sorted(unknown)}; "
            f"known: {sorted(_PLAN_WARM_START_KEYS)}"
        )
    # Validate the whole plan before comparing anything, so a bad plan
    # is rejected as a bad plan even when the recorded config would
    # also mismatch (e.g. a from-scratch run).
    if "transfer_log_ent_coef" in wanted and not isinstance(
        wanted["transfer_log_ent_coef"], bool
    ):
        raise ValueError("expected transfer_log_ent_coef must be a bool")
    pins = wanted.get("expected_artifact_sha256")
    if pins is not None:
        if not isinstance(pins, Mapping) or not all(
            isinstance(value, str) for value in pins.values()
        ):
            raise ValueError(
                "expected expected_artifact_sha256 must be a mapping of "
                "artifact name to a sha256 string (or None)"
            )
        for value in pins.values():
            # WarmStartConfig's pin rule, mirrored: an empty pin would
            # match every digest, and uppercase or short pins silently
            # weaken the check.
            if not 8 <= len(value) <= 64 or any(
                c not in "0123456789abcdef" for c in value
            ):
                raise ValueError(
                    "expected_artifact_sha256 values must be lowercase "
                    "hex, 8 to 64 chars"
                )
    if not isinstance(initialization, Mapping):
        return [
            "warm start: expected a warm-started run, config.json records "
            "no initialization block (from scratch, or validated before "
            "train() bound the provenance)"
        ]

    mismatches = []
    if "source_run_dir_suffix" in wanted:
        # Whole path components only: the pin "leg1/attempt_01" must
        # not match an evil sibling like ".../evilleg1/attempt_01".
        suffix = str(wanted["source_run_dir_suffix"]).strip("/")
        recorded_source = str(initialization.get("source_run_dir", ""))
        trimmed_source = recorded_source.rstrip("/")
        if trimmed_source != suffix and not trimmed_source.endswith(
            "/" + suffix
        ):
            mismatches.append(
                "initialization.source_run_dir: expected a path ending "
                f"with {suffix!r}, config.json records {recorded_source!r}"
            )

    if "transfer_log_ent_coef" in wanted:
        wanted_flag = wanted["transfer_log_ent_coef"]
        recorded_flag = initialization.get(
            "transfer_log_ent_coef", _PLAN_MISSING
        )
        if recorded_flag is _PLAN_MISSING:
            mismatches.append(
                "initialization.transfer_log_ent_coef: expected "
                f"{wanted_flag}, config.json records nothing"
            )
        elif recorded_flag is not wanted_flag:
            mismatches.append(
                "initialization.transfer_log_ent_coef: expected "
                f"{wanted_flag}, config.json records {recorded_flag!r}"
            )
        # Membership is evidence-gated: ``train()`` only files
        # ``log_ent_coef`` under ``transferred``/``reset`` when both
        # sides carry a temperature, and records ``transferred_ent_coef``
        # exactly when one actually moved. PPO and fixed-``ent_coef``
        # warm starts record the flag alone (the documented no-op), so
        # only the verbatim flag comparison may trip on them.
        transferred = initialization.get("transferred") or ()
        reset = initialization.get("reset") or ()
        in_transferred = "log_ent_coef" in transferred
        in_reset = "log_ent_coef" in reset
        temperature_moved = "transferred_ent_coef" in initialization
        if wanted_flag:
            if temperature_moved and not in_transferred:
                mismatches.append(
                    "initialization.transferred: expected 'log_ent_coef' "
                    "(temperature transferred), config.json records it "
                    "absent"
                )
            if in_reset:
                mismatches.append(
                    "initialization.reset: expected no 'log_ent_coef' "
                    "(temperature transferred), config.json records it "
                    "reset"
                )
        else:
            if in_transferred:
                mismatches.append(
                    "initialization.transferred: expected no "
                    "'log_ent_coef' (temperature skip), config.json "
                    "records it transferred"
                )
            if temperature_moved:
                mismatches.append(
                    "initialization.transferred_ent_coef: expected no "
                    "moved temperature (temperature skip), config.json "
                    f"records {initialization['transferred_ent_coef']!r}"
                )

    if pins is not None:
        source_artifacts = initialization.get("source_artifacts")
        if not isinstance(source_artifacts, Mapping):
            mismatches.append(
                "initialization.source_artifacts: expected recorded "
                "source digests, config.json records none"
            )
        else:
            for name in sorted(pins):
                entry = source_artifacts.get(name)
                recorded_sha = (
                    entry.get("sha256") if isinstance(entry, Mapping) else None
                )
                if not isinstance(recorded_sha, str):
                    mismatches.append(
                        f"initialization.source_artifacts[{name!r}]: "
                        f"expected sha256 {pins[name]!r}, config.json "
                        "records none"
                    )
                elif not recorded_sha.startswith(pins[name]):
                    mismatches.append(
                        f"initialization.source_artifacts[{name!r}]: "
                        f"expected sha256 starting {pins[name]!r}, "
                        f"config.json records {recorded_sha!r}"
                    )
    return mismatches
