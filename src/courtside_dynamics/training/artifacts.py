"""Per-run artifacts: ``config.json`` and ``stage_summary.txt``.

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
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from courtside_dynamics.training.train import TrainConfig


def _git_sha_from_checkout() -> str | None:
    # Resolve against the package's own location, not the caller's cwd: a
    # bare ``git rev-parse`` records whatever repository the notebook
    # happens to run from, and returned null on every Colab run (both
    # 20260714 WallBall baselines have ``"git_sha": null``). Git discovers
    # repositories by walking up parent directories, so a site-packages
    # install nested inside an unrelated checkout (a project venv, a
    # dotfiles-managed home) would still find *that* repository -- the
    # ``ls-files`` gate below only accepts a SHA from a repository that
    # actually tracks this file (i.e. an editable install of the
    # courtside-dynamics checkout); anything else yields None.
    this_file = os.path.abspath(__file__)
    package_dir = os.path.dirname(this_file)
    try:
        tracked = subprocess.run(
            ["git", "-C", package_dir, "ls-files", "--error-unmatch",
             this_file],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if tracked.returncode != 0:
            return None
        out = subprocess.run(
            ["git", "-C", package_dir, "rev-parse", "HEAD"],
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


def _git_sha_from_install_metadata() -> str | None:
    """Commit recorded by pip for a VCS install (PEP 610).

    ``pip install "courtside-dynamics @ git+https://..."`` -- the Colab
    flow -- leaves no git checkout at runtime, but pip records the
    resolved commit in the distribution's ``direct_url.json``.
    """
    try:
        from importlib import metadata

        dist = metadata.distribution("courtside-dynamics")
        raw = dist.read_text("direct_url.json")
        if not raw:
            return None
        commit = (
            json.loads(raw).get("vcs_info", {}).get("commit_id")
        )
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    except Exception:
        return None
    return None


def _git_sha() -> str | None:
    # Provenance chain, most authoritative first: an explicit override
    # (for install flows neither probe can see), the editable-checkout
    # probe, then pip's recorded VCS commit.
    override = os.environ.get("COURTSIDE_DYNAMICS_GIT_SHA", "").strip()
    if override:
        return override
    return _git_sha_from_checkout() or _git_sha_from_install_metadata()


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


def _gpu_info() -> dict[str, Any]:
    """Capture CUDA / GPU details available before training starts."""
    info: dict[str, Any] = {"available": False}
    try:
        import torch
    except ImportError:
        return info
    info["cuda_version"] = torch.version.cuda
    info["available"] = bool(torch.cuda.is_available())
    if not info["available"]:
        return info
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    count = torch.cuda.device_count()
    info["device_count"] = count
    devices: list[dict[str, Any]] = []
    for i in range(count):
        try:
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        except Exception:
            devices.append({"index": i, "name": torch.cuda.get_device_name(i)})
    info["devices"] = devices
    return info


def _probe_env_fn(env_fn: Any) -> dict[str, Any]:
    """Construct one factory's env to capture class + space metadata."""
    info: dict[str, Any] = {
        "class": None,
        "observation_shape": None,
        "action_shape": None,
    }
    try:
        env = env_fn()
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
        constructor_kwargs = getattr(env, "_ezpickle_kwargs", None)
        if isinstance(constructor_kwargs, dict):
            # EzPickle captures every resolved constructor default, including
            # environment physics/reward settings that observation/action
            # spaces alone cannot identify. The long-horizon evaluator uses
            # this to reject recipe drift apart from its intentional horizon.
            info["constructor_kwargs"] = dict(constructor_kwargs)
        curriculum_metadata = getattr(env, "curriculum_metadata", None)
        if curriculum_metadata is not None:
            info["curriculum"] = dict(curriculum_metadata)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return info


def _probe_env(cfg: TrainConfig) -> dict[str, Any]:
    """Capture the training environment profile (legacy helper name)."""
    return _probe_env_fn(cfg.env_fn)


def write_run_config(cfg: TrainConfig, log_dir: str) -> str:
    """Snapshot the resolved cfg + provenance to ``log_dir/config.json``.

    When the config was built from a TOML run-configuration file, the
    file's byte-exact content is also copied to
    ``log_dir/run_config.toml`` so the run directory is self-contained
    even if the original file is later edited or deleted.
    """
    run_file = cfg.run_config_file
    if run_file is not None:
        copy_path = artifact_path(log_dir, "run_config_toml")
        with open(copy_path, "w", encoding="utf-8") as f:
            f.write(run_file.text)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "gpu": _gpu_info(),
        "run_config_file": (
            {
                "path": run_file.path,
                "sha256": run_file.sha256,
                "content": run_file.raw,
            }
            if run_file is not None
            else None
        ),
        "recipe_name": cfg.recipe_name,
        "env": _probe_env(cfg),
        # Keep ``env`` as the training profile for backwards compatibility.
        # Evaluation may intentionally disable curriculum reset modes while
        # preserving spaces and physics, so record that constructor too.
        "evaluation_env": _probe_env_fn(cfg.eval_env_fn or cfg.env_fn),
        "train_config": {
            "algo": cfg.algo,
            "total_timesteps": cfg.total_timesteps,
            "log_dir": cfg.log_dir,
            "name_prefix": cfg.name_prefix,
            "n_envs": cfg.n_envs,
            "seed": cfg.seed,
            "verbose": cfg.verbose,
            "eval_freq": cfg.eval_freq,
            "checkpoint_freq": cfg.checkpoint_freq,
            "video_freq": cfg.video_freq,
            "n_eval_episodes": cfg.n_eval_episodes,
            "early_stop_patience": cfg.early_stop_patience,
            "video_length": cfg.video_length,
            "record_video": cfg.record_video,
            "normalize_obs": cfg.normalize_obs,
            "normalize_reward": cfg.normalize_reward,
            "clip_obs": cfg.clip_obs,
            "clip_reward": cfg.clip_reward,
            "normalize_obs_excluded_indices": list(
                cfg.normalize_obs_excluded_indices
            ),
            "policy": cfg.policy,
            "model_kwargs": cfg.model_kwargs,
            "warm_start": (
                {
                    "source_run_dir": str(cfg.warm_start.source_run_dir),
                    "reset_observation_indices": list(
                        cfg.warm_start.reset_observation_indices
                    ),
                }
                if cfg.warm_start is not None
                else None
            ),
            "csv_header": list(cfg.csv_header) if cfg.csv_header else None,
            "info_dict_eval": cfg.info_dict_eval,
            "info_eval_keys": (
                list(cfg.info_eval_keys)
                if cfg.info_eval_keys is not None
                else None
            ),
            "info_eval_terminal_keys": list(cfg.info_eval_terminal_keys),
            "info_eval_distribution_keys": list(
                cfg.info_eval_distribution_keys
            ),
            "info_eval_survival_thresholds": {
                key: list(thresholds)
                for key, thresholds in cfg.info_eval_survival_thresholds.items()
            },
            "success_key": cfg.success_key,
            "success_threshold": cfg.success_threshold,
            "headline_key": cfg.headline_key,
            "phase_key": cfg.phase_key,
            "phase_labels": (
                {str(k): v for k, v in cfg.phase_labels.items()}
                if cfg.phase_labels
                else None
            ),
            "env_attr_schedules": [
                dict(schedule) for schedule in cfg.env_attr_schedules
            ],
            "best_metric_keys": (
                list(cfg.best_metric_keys)
                if cfg.best_metric_keys is not None
                else None
            ),
            "best_metric_min_delta": cfg.best_metric_min_delta,
            "confirm_best_eval": cfg.confirm_best_eval,
            "early_stop_degenerate_evals": cfg.early_stop_degenerate_evals,
            "degenerate_guard_keys": list(cfg.degenerate_guard_keys),
            "degenerate_min_evals": cfg.degenerate_min_evals,
            "performance_gate": (
                {
                    "metric_key": cfg.performance_gate["metric_key"],
                    "threshold": cfg.performance_gate["threshold"],
                    "sustain_evals": cfg.performance_gate["sustain_evals"],
                    # Mirror train()'s resolution defaults so the block
                    # records the gate as actually run whether the keys
                    # came from a recipe, a TOML, or were left unset.
                    "promotion_rule": cfg.performance_gate.get(
                        "promotion_rule", "consecutive"
                    ),
                    "advance_update_pause_steps": cfg.performance_gate.get(
                        "advance_update_pause_steps", 0
                    ),
                    "clear_replay_buffer_on_advance": (
                        cfg.performance_gate.get(
                            "clear_replay_buffer_on_advance", False
                        )
                    ),
                    "stages": [
                        dict(stage)
                        for stage in cfg.performance_gate["stages"]
                    ],
                }
                if cfg.performance_gate is not None
                else None
            ),
            "final_info_eval": cfg.final_info_eval,
        },
    }
    out = artifact_path(log_dir, "config")
    with open(out, "w") as f:
        # default=repr so any callable / non-JSON value in model_kwargs
        # round-trips as a readable string instead of crashing the dump.
        json.dump(payload, f, indent=2, default=repr)
        f.write("\n")
    return out


def _scalar_or_initial(value: Any) -> Any:
    """Return scalars unchanged; resolve schedule callables at progress=1.0."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if callable(value):
        try:
            return float(value(1.0))
        except Exception:
            return repr(value)
    return value


def _model_info(model: Any) -> dict[str, Any]:
    """Snapshot the SB3-resolved model + policy hyperparameters.

    Captures both what the user passed and what SB3 filled in from its
    defaults (e.g. ``net_arch=[256, 256]`` for ``MlpPolicy``), so the
    ``config.json`` is enough to reconstruct the exact training setup.
    """
    info: dict[str, Any] = {}
    info["algo_class"] = type(model).__name__
    info["device"] = str(getattr(model, "device", ""))

    policy = getattr(model, "policy", None)
    if policy is not None:
        info["policy_class"] = type(policy).__name__
        net_arch = getattr(policy, "net_arch", None)
        if net_arch is not None:
            info["net_arch"] = net_arch
        activation_fn = getattr(policy, "activation_fn", None)
        if activation_fn is not None:
            info["activation_fn"] = getattr(
                activation_fn, "__name__", repr(activation_fn)
            )
        try:
            num_params = sum(p.numel() for p in policy.parameters())
            info["policy_num_params"] = int(num_params)
        except Exception:
            pass

    # Hyperparameters common across SAC/PPO. Missing attrs are silently
    # skipped so this stays algo-agnostic.
    hyperparam_keys = (
        # Shared
        "learning_rate", "gamma", "batch_size", "max_grad_norm", "seed",
        # Off-policy (SAC, TD3, DQN)
        "tau", "buffer_size", "learning_starts", "train_freq",
        "gradient_steps", "ent_coef", "target_entropy",
        "target_update_interval",
        # On-policy (PPO, A2C)
        "n_steps", "n_epochs", "gae_lambda", "clip_range",
        "clip_range_vf", "vf_coef", "normalize_advantage",
    )
    hyperparams: dict[str, Any] = {}
    for attr in hyperparam_keys:
        if not hasattr(model, attr):
            continue
        value = getattr(model, attr)
        if attr in ("learning_rate", "clip_range", "clip_range_vf"):
            value = _scalar_or_initial(value)
        elif attr == "train_freq":
            value = repr(value)
        hyperparams[attr] = value
    info["hyperparameters"] = hyperparams
    return info


def update_run_config_with_model(model: Any, log_dir: str) -> str | None:
    """Augment ``log_dir/config.json`` with resolved model details.

    Called after the SB3 algorithm is constructed so SB3-default values
    (``net_arch=[256, 256]``, ``learning_rate=3e-4``, ``buffer_size=1e6``,
    etc.) end up on disk even when the user passes an empty
    ``model_kwargs``. No-op if ``config.json`` doesn't exist yet.
    """
    path = locate_artifact(log_dir, "config")
    if path is None:
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    payload["resolved_model"] = _model_info(model)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=repr)
        f.write("\n")
    return path


def update_run_config_with_initialization(
    initialization: dict[str, Any],
    log_dir: str,
) -> str | None:
    """Bind resolved warm-start provenance to ``config.json``."""
    path = locate_artifact(log_dir, "config")
    if path is None:
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    payload["initialization"] = initialization
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=repr)
        f.write("\n")
    return path


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _read_monitor(log_dir: str) -> tuple[list[float], list[int]]:
    """Return ``(rewards, lengths)`` in wall-clock order across workers.

    Uses the shared ``read_monitor_rewards_lengths`` helper so the "Recent
    train" stat reflects the genuinely most-recent episodes (interleaved by
    wall-clock time), not whatever the last per-worker file happened to hold.
    """
    from courtside_dynamics.training.monitor_log import (
        read_monitor_rewards_lengths,
    )

    monitor_dir = locate_artifact(log_dir, "monitor_dir")
    if monitor_dir is None:
        return [], []
    return read_monitor_rewards_lengths(monitor_dir)


def _read_training_health(log_dir: str) -> dict[str, float]:
    """Final value of every ``train/*`` metric from ``progress.csv``.

    SB3's CSV logger writes ``LOG_DIR/metrics/progress.csv`` (pre-0.14:
    ``LOG_DIR/tensorboard/progress.csv``) with one column per metric;
    cells are blank when a metric wasn't logged on that row. We discover
    the ``train/*`` columns from the header (so the same code covers SAC
    and PPO, and picks up any metric SB3 adds) and keep the last
    non-blank float per key, i.e. its end-of-run value. This matches
    ``plot_training_health``'s generic ``train/*`` discovery so the static
    report and the plot never disagree on which metrics matter. Returns
    ``{}`` if the file is absent (e.g. CSV logging off).
    """
    path = locate_artifact(log_dir, "progress_csv")
    if path is None:
        return {}
    finals: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        health_keys = [
            c for c in (reader.fieldnames or []) if c.startswith("train/")
        ]
        for row in reader:
            for key in health_keys:
                value = row.get(key, "")
                if value in ("", None):
                    continue
                try:
                    finals[key] = float(value)
                except (TypeError, ValueError):
                    continue
    return finals


def _read_eval_info_at_step(log_dir: str, target_step: int) -> dict[str, float]:
    """Pull metrics logged by ``InfoDictEvalCallback`` at ``target_step``.

    The callback writes a long-format ``timestep,metric,value`` CSV at
    every evaluation. Since it shares ``eval_freq`` with ``EvalCallback``,
    the row at the best-checkpoint step describes the same model snapshot
    that was saved as ``best_model.zip``.
    """
    path = locate_artifact(log_dir, "eval_info_csv")
    if path is None:
        return {}
    metrics: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row["timestep"])
            except (KeyError, ValueError):
                continue
            if ts != target_step:
                continue
            try:
                metrics[row["metric"]] = float(row["value"])
            except (KeyError, ValueError):
                continue
    return metrics


def _read_eval_info_series(
    log_dir: str, metric: str
) -> list[tuple[int, float]]:
    """All ``(timestep, value)`` points for one metric from ``eval_info.csv``."""
    path = locate_artifact(log_dir, "eval_info_csv")
    if path is None:
        return []
    points: list[tuple[int, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("metric") != metric:
                continue
            try:
                points.append((int(row["timestep"]), float(row["value"])))
            except (KeyError, TypeError, ValueError):
                continue
    points.sort(key=lambda p: p[0])
    return points


def _read_best_model_meta(log_dir: str) -> dict[str, Any] | None:
    """Load ``best_model_meta.json`` (task-metric selection provenance).

    Written by ``InfoDictEvalCallback`` when a headline metric owns
    best-model selection; absent on reward-selected runs.
    """
    path = locate_artifact(log_dir, "best_model_meta")
    if path is None:
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


_PROJECT_NAME = "courtside-dynamics"

#: Single source of truth for where every artifact lives inside a run
#: directory (the 0.14.0 layout). Keys are stable artifact names used by
#: writers (:func:`artifact_path`) and readers (:func:`locate_artifact`);
#: values are run-dir-relative paths. Only the three files a human opens
#: first stay at the root; everything else is one folder deep by
#: audience (model artifacts, machine metrics, human reports, media).
#: Entries whose relative path has no file extension are directories.
RUN_LAYOUT: dict[str, str] = {
    # Root: identity & provenance.
    "config": "config.json",
    "stage_summary": "stage_summary.txt",
    "run_config_toml": "run_config.toml",
    # model/: everything needed to reload a policy.
    "best_model": "model/best_model.zip",
    "best_vec_normalize": "model/best_vec_normalize.pkl",
    "best_model_meta": "model/best_model_meta.json",
    "final_model": "model/final_model.zip",
    "vec_normalize": "model/vec_normalize.pkl",
    # Per-stage champions archived by the performance gate before each
    # advance resets selection state (stage_bests/stage_NN/ holds the
    # same best_model/best_vec_normalize/best_model_meta triple).
    "stage_bests_dir": "model/stage_bests",
    # metrics/: machine-readable training/eval series. ``progress.csv``
    # lives directly in metrics/ (it is pandas-readable metrics, not TB
    # event data), no longer inside the TensorBoard folder.
    "monitor_dir": "metrics/monitor",
    "tensorboard_dir": "metrics/tensorboard",
    "progress_csv": "metrics/progress.csv",
    "evaluations": "metrics/evaluations.npz",
    "eval_info_csv": "metrics/eval_info.csv",
    "eval_info_final_csv": "metrics/eval_info_final.csv",
    # reports/: human-facing plots and post-training audits.
    "learning_curve": "reports/learning_curve.png",
    "training_health_plot": "reports/training_health.png",
    "eval_headline": "reports/eval_headline.png",
    "eval_terminations": "reports/eval_terminations.png",
    "eval_rewards": "reports/eval_rewards.png",
    "eval_diagnostics": "reports/eval_diagnostics.png",
    "best_long_eval": "reports/best_model_long_horizon_eval.json",
    "best_long_eval_episodes": "reports/best_model_long_horizon_episodes.csv",
    "curriculum_stages": "reports/curriculum_stages.json",
    # media/: replay footage.
    "best_model_video": "media/best_model.mp4",
    "videos_dir": "media/videos",
    # checkpoints/: periodic full-state snapshots (unchanged location).
    "checkpoints_dir": "checkpoints",
}

#: Pre-0.14 locations that differ from "same basename at the run root".
#: ``progress.csv`` used to be written inside the TensorBoard folder.
_LEGACY_LAYOUT_OVERRIDES: dict[str, str] = {
    "progress_csv": os.path.join("tensorboard", "progress.csv"),
}


def _legacy_artifact_relpath(name: str) -> str:
    """Pre-0.14 flat location of artifact ``name``, relative to the run dir."""
    return _LEGACY_LAYOUT_OVERRIDES.get(
        name, os.path.basename(RUN_LAYOUT[name])
    )


def artifact_path(run_dir: str | os.PathLike[str], name: str) -> str:
    """Writer-side resolver: the canonical (new-layout) path for ``name``.

    Creates the destination directory (the parent for file artifacts,
    the directory itself for directory artifacts) so writers can open
    the returned path directly. Always returns the new location --
    writers never produce the legacy flat layout.
    """
    rel = RUN_LAYOUT[name]
    path = os.path.join(os.fspath(run_dir), rel)
    if os.path.splitext(rel)[1]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
    return path


def locate_artifact(run_dir: str | os.PathLike[str], name: str) -> str | None:
    """Reader-side resolver: find artifact ``name`` under ``run_dir``.

    Returns the new-layout path if it exists, else the legacy flat path
    (pre-0.14 runs, e.g. ``best_model.zip`` at the root or
    ``tensorboard/progress.csv``) if that exists, else ``None``. Every
    existing run directory therefore keeps working unmodified.
    """
    root = os.fspath(run_dir)
    new = os.path.join(root, RUN_LAYOUT[name])
    if os.path.exists(new):
        return new
    legacy = os.path.join(root, _legacy_artifact_relpath(name))
    if os.path.exists(legacy):
        return legacy
    return None


#: Artifacts that only exist for specific configurations (TOML-driven
#: runs, ``final_info_eval`` recipes, WallBall post-training audits) or
#: that nothing audits directly (the TensorBoard event dir, whose useful
#: scalars are mirrored to ``progress.csv``). Excluded from the default
#: expectations below; the notebook passes the WallBall long-horizon
#: extras to ``check_run_artifacts`` explicitly.
_CONDITIONAL_ARTIFACTS: frozenset[str] = frozenset(
    {
        "run_config_toml",
        "tensorboard_dir",
        "eval_info_final_csv",
        "best_long_eval",
        "best_long_eval_episodes",
        # Written only by performance-gated curriculum runs.
        "stage_bests_dir",
        "curriculum_stages",
    }
)

#: ``(name, path relative to log_dir)`` for every artifact a training
#: run (plus the notebook's plotting/replay cells) should produce.
#: Derived from :data:`RUN_LAYOUT` so ``write_run_summary``'s
#: "Artifacts" section and the notebook-facing ``check_run_artifacts``
#: audit helper never drift from the layout registry.
EXPECTED_ARTIFACTS: tuple[tuple[str, str], ...] = tuple(
    (name, rel)
    for name, rel in RUN_LAYOUT.items()
    if name not in _CONDITIONAL_ARTIFACTS
)


def _env_display_name(cfg: TrainConfig) -> str:
    """Friendly env name (e.g. ``WallBallEnv`` -> ``WallBall``)."""
    if cfg.recipe_name:
        return cfg.recipe_name
    info = _probe_env(cfg)
    cls = info.get("class") or ""
    return cls[:-3] if cls.endswith("Env") else (cls or "Unknown")


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _kv(label: str, value: str, label_width: int = 16) -> str:
    return f"{(label + ':').ljust(label_width)}{value}"


def _section(title: str) -> list[str]:
    return ["", title, "-" * 40]


def write_run_summary(
    cfg: TrainConfig,
    log_dir: str,
    *,
    final_mean_reward: float,
    final_std_reward: float,
    duration_seconds: float,
    device: str | None = None,
    status: str = "completed",
    actual_timesteps: int | None = None,
) -> str:
    """Write a human-readable end-of-run report to ``log_dir/stage_summary.txt``.

    ``status`` distinguishes a run that reached its full budget
    (``"completed"``) from one cut short by KeyboardInterrupt
    (``"interrupted"``); the eval numbers are real either way.
    ``actual_timesteps`` is the number of env steps actually trained --
    it differs from ``cfg.total_timesteps`` when early stopping or an
    interrupt cut the run short, and it is what the throughput figure
    should be computed from.
    """
    env_name = _env_display_name(cfg)
    git_sha = _git_sha()
    short_sha = git_sha[:7] if git_sha else "unknown"
    duration_str = _format_duration(duration_seconds)
    steps_trained = (
        actual_timesteps if actual_timesteps is not None else cfg.total_timesteps
    )
    throughput_fps = (
        int(steps_trained / duration_seconds) if duration_seconds > 0 else 0
    )

    lines: list[str] = []
    title = f"{_PROJECT_NAME}: {cfg.algo} on {env_name}"
    lines.append(title)
    lines.append("=" * 50)
    lines.append("")

    lines.append(_kv("Project", _PROJECT_NAME))
    lines.append(_kv("Environment", env_name))
    lines.append(_kv("Algorithm", cfg.algo))
    seed = cfg.seed
    if seed is None and cfg.model_kwargs:
        seed = cfg.model_kwargs.get("seed")
    if seed is not None:
        lines.append(_kv("Seed", str(seed)))
    lines.append(
        _kv("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    lines.append(_kv("Status", status))
    lines.append(_kv("Git SHA", short_sha))
    if cfg.run_config_file is not None:
        lines.append(
            _kv(
                "Run config",
                f"{os.path.basename(cfg.run_config_file.path)} "
                f"({cfg.run_config_file.sha256[:12]})",
            )
        )
    if steps_trained < cfg.total_timesteps:
        lines.append(
            _kv(
                "Timesteps",
                f"{steps_trained:,} of {cfg.total_timesteps:,} budget "
                "(stopped early)",
            )
        )
    else:
        lines.append(_kv("Timesteps", f"{cfg.total_timesteps:,}"))
    lines.append(_kv("Duration", duration_str))
    lines.append(_kv("Throughput", f"{throughput_fps} FPS"))
    lines.append(
        _kv("Final eval", f"{final_mean_reward:.3f} +/- {final_std_reward:.3f}")
    )

    train_rewards, train_lengths = _read_monitor(log_dir)
    if train_lengths:
        last_n = min(100, len(train_lengths))
        ep_mean, ep_std = _mean_std(train_lengths[-last_n:])
        lines.append(
            _kv("Avg ep length", f"{ep_mean:.1f} +/- {ep_std:.1f} steps")
        )

    eval_npz = locate_artifact(log_dir, "evaluations")
    best_step: int | None = None
    best_mean: float | None = None
    best_std: float | None = None
    eval_timesteps: np.ndarray | None = None
    eval_means: np.ndarray | None = None
    eval_stds: np.ndarray | None = None
    if eval_npz is not None:
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        if results.size:
            eval_timesteps = timesteps
            eval_means = results.mean(axis=1)
            eval_stds = results.std(axis=1)
            best_idx = int(eval_means.argmax())
            best_step = int(timesteps[best_idx])
            best_mean = float(eval_means[best_idx])
            best_std = float(eval_stds[best_idx])
            lines.append(
                _kv(
                    "Best eval",
                    f"{best_mean:.3f} +/- {best_std:.3f} (at {best_step:,} steps)",
                )
            )
            # Make post-best collapse impossible to miss: the first
            # real WallBall run ended 7 points below its best and the
            # operator had to diff two lines to notice. Flag a drop
            # bigger than 2 eval-stds AND 20% of the best mean.
            delta = final_mean_reward - best_mean
            regressed = delta < -max(2 * best_std, 0.2 * abs(best_mean))
            note = (
                "  <-- policy regressed after best; deploy best_model.zip"
                if regressed
                else ""
            )
            lines.append(_kv("Final vs best", f"{delta:+.3f}{note}"))

    # Task-metric selection provenance: when best_model.zip was chosen
    # by the headline metric rather than reward, say so -- the reward
    # lines above describe a *different* checkpoint in that case.
    best_meta = _read_best_model_meta(log_dir)
    selected_step: int | None = None
    if best_meta is not None:
        try:
            selected_step = int(best_meta["timestep"])
        except (KeyError, TypeError, ValueError):
            selected_step = None
        if selected_step is not None:
            keys = best_meta.get("selection_keys") or []
            values = best_meta.get("selection_values") or {}
            primary = keys[0] if keys else None
            desc = f"step {selected_step:,}"
            if primary is not None and primary in values:
                desc += f" ({primary} {float(values[primary]):.2f})"
            # Curriculum context: a matched-stage score is meaningless
            # without its stage (run 20260721_004722's "best 3.80" took
            # a progress.csv join to attribute to stage-2 geometry).
            context = best_meta.get("context") or {}
            stage = context.get("curriculum_stage_index")
            if isinstance(stage, (int, float)):
                desc += f" [at curriculum stage {int(stage)}]"
            desc += "  [task-metric selection]"
            lines.append(_kv("Best model", desc))

    # Headline task metric. Eval reward can be a poor progress measure
    # (WallBall's is dominated by tracking shaping, and success_rate
    # saturates once every episode completes one exchange), so when the
    # recipe names a headline counter, surface its mean-over-eval-
    # episodes terminal value alongside the reward lines. The best is
    # over the metric's own series -- it need not coincide with the
    # best-reward checkpoint.
    if cfg.headline_key:
        metric = f"{cfg.headline_key}_ep_mean"
        headline = _read_eval_info_series(log_dir, metric)
        if headline:
            final_val = headline[-1][1]
            head_step, head_val = max(headline, key=lambda p: p[1])
            lines.append(_kv("Headline", metric))
            lines.append(_kv("Headline final", f"{final_val:.2f}"))
            lines.append(
                _kv(
                    "Headline best",
                    f"{head_val:.2f} (at {head_step:,} steps)",
                )
            )
        survival_parts: list[str] = []
        for threshold in cfg.info_eval_survival_thresholds.get(
            cfg.headline_key, ()
        ):
            survival_metric = (
                f"{cfg.headline_key}_ep_ge_{threshold}_rate"
            )
            survival_series = _read_eval_info_series(
                log_dir, survival_metric
            )
            if survival_series:
                survival_parts.append(
                    f">={threshold} {survival_series[-1][1] * 100:.1f}%"
                )
        if survival_parts:
            lines.append(_kv("Survival final", "  ".join(survival_parts)))

    if train_rewards:
        last_n = min(100, len(train_rewards))
        r_mean, r_std = _mean_std(train_rewards[-last_n:])
        lines.append(
            _kv(
                "Recent train",
                f"{r_mean:.3f} +/- {r_std:.3f} (last {last_n} episodes)",
            )
        )

    gpu = _gpu_info()
    versions = _versions()
    lines.extend(_section("Device"))
    lines.append(f"  {_kv('Device', device or 'cpu')}")
    if gpu.get("available"):
        devices = gpu.get("devices") or []
        if devices:
            primary = devices[0]
            lines.append(f"  {_kv('GPU', str(primary.get('name', 'unknown')))}")
            total_mem = primary.get("total_memory_bytes")
            if total_mem:
                lines.append(
                    f"  {_kv('VRAM', f'{int(total_mem) // (1024 * 1024)} MB')}"
                )
            cap = primary.get("capability")
            if cap:
                lines.append(f"  {_kv('Compute cap', str(cap))}")
        cuda_version = gpu.get("cuda_version")
        if cuda_version:
            lines.append(f"  {_kv('CUDA', str(cuda_version))}")
        cudnn_version = gpu.get("cudnn_version")
        if cudnn_version:
            lines.append(f"  {_kv('cuDNN', str(cudnn_version))}")
    if "torch" in versions:
        lines.append(f"  {_kv('PyTorch', versions['torch'])}")

    lines.extend(_section("Hyperparameters"))
    hp_items: list[tuple[str, Any]] = [
        ("total_timesteps", cfg.total_timesteps),
        ("n_envs", cfg.n_envs),
    ]
    for key, value in (cfg.model_kwargs or {}).items():
        hp_items.append((key, value))
    key_width = max((len(k) for k, _ in hp_items), default=0) + 4
    for key, value in hp_items:
        lines.append(f"  {key.ljust(key_width)}{value}")

    # The section describes the checkpoint that best_model.zip actually
    # holds: the task-metric-selected step when meta exists, else the
    # reward-best step.
    section_step = selected_step if selected_step is not None else best_step
    if section_step is not None:
        lines.extend(
            _section(f"Best Checkpoint Evaluation (step {section_step:,})")
        )
        section_mean: float | None = None
        section_std: float | None = None
        if section_step == best_step:
            section_mean, section_std = best_mean, best_std
        elif eval_timesteps is not None:
            matches = np.flatnonzero(eval_timesteps == section_step)
            if matches.size:
                assert eval_means is not None and eval_stds is not None
                section_mean = float(eval_means[matches[0]])
                section_std = float(eval_stds[matches[0]])
        if section_mean is not None and section_std is not None:
            lines.append(
                f"  {_kv('Reward', f'{section_mean:.3f} +/- {section_std:.3f}')}"
            )
        eval_info = _read_eval_info_at_step(log_dir, section_step)
        if eval_info:
            ep_len = eval_info.get("episode_length")
            if ep_len is not None:
                lines.append(f"  {_kv('Episode length', f'{ep_len:.1f}')}")
            # Counter-style keys: render `<key>: final X  max Y`.
            counter_finals = sorted(
                k[: -len("_final")] for k in eval_info if k.endswith("_final")
            )
            if counter_finals:
                key_width = max(len(k) for k in counter_finals) + 2
                for base in counter_finals:
                    final = eval_info.get(f"{base}_final")
                    peak = eval_info.get(f"{base}_max")
                    parts = [f"final {final:.2f}" if final is not None else ""]
                    if peak is not None:
                        parts.append(f"max {peak:.2f}")
                    lines.append(
                        f"  {(base + ':').ljust(key_width)}{'  '.join(p for p in parts if p)}"
                    )
            if cfg.headline_key:
                survival_parts = []
                for threshold in cfg.info_eval_survival_thresholds.get(
                    cfg.headline_key, ()
                ):
                    metric = (
                        f"{cfg.headline_key}_ep_ge_{threshold}_rate"
                    )
                    if metric in eval_info:
                        survival_parts.append(
                            f">={threshold} {eval_info[metric] * 100:.1f}%"
                        )
                if survival_parts:
                    lines.append(
                        f"  {_kv('Return survival', '  '.join(survival_parts))}"
                    )
            phase_keys = sorted(
                k for k in eval_info if k.startswith("phase_frac_")
            )
            if phase_keys:
                lines.append("  Phase fractions:")
                pf_width = (
                    max(len(k[len("phase_frac_"):]) for k in phase_keys) + 4
                )
                for k in phase_keys:
                    label = k[len("phase_frac_"):]
                    pct = eval_info[k] * 100
                    lines.append(f"    {label.ljust(pf_width)}{pct:.1f}%")

    health = _read_training_health(log_dir)
    if health:
        lines.extend(_section("Training Health (final)"))
        hw = max(len(k) for k in health) + 2
        for key in sorted(health):
            lines.append(f"  {(key + ':').ljust(hw)}{health[key]:.4g}")

    artifact_lines: list[str] = []
    for label, _path in EXPECTED_ARTIFACTS:
        if label == "stage_summary":
            continue  # this report itself; listing it would be circular
        full = locate_artifact(log_dir, label)
        if full is not None:
            artifact_lines.append(
                f"  {_kv(label, os.path.relpath(full, log_dir))}"
            )
    if artifact_lines:
        lines.extend(_section("Artifacts"))
        lines.extend(artifact_lines)

    out = artifact_path(log_dir, "stage_summary")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out
