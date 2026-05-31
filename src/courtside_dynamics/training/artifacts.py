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
    """Snapshot the resolved cfg + provenance to ``log_dir/config.json``."""
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "versions": _versions(),
        "gpu": _gpu_info(),
        "env": _probe_env(cfg),
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
            "video_length": cfg.video_length,
            "record_video": cfg.record_video,
            "normalize_obs": cfg.normalize_obs,
            "normalize_reward": cfg.normalize_reward,
            "clip_obs": cfg.clip_obs,
            "clip_reward": cfg.clip_reward,
            "policy": cfg.policy,
            "model_kwargs": cfg.model_kwargs,
            "csv_header": list(cfg.csv_header) if cfg.csv_header else None,
            "info_dict_eval": cfg.info_dict_eval,
            "success_key": cfg.success_key,
            "success_threshold": cfg.success_threshold,
            "phase_key": cfg.phase_key,
            "phase_labels": (
                {str(k): v for k, v in cfg.phase_labels.items()}
                if cfg.phase_labels
                else None
            ),
        },
    }
    out = os.path.join(log_dir, "config.json")
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
    path = os.path.join(log_dir, "config.json")
    if not os.path.exists(path):
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


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _read_monitor(log_dir: str) -> tuple[list[float], list[int]]:
    """Return ``(rewards, lengths)`` in wall-clock order across workers.

    Delegates to ``load_monitor_episodes`` so the "Recent train" stat
    below reflects the genuinely most-recent episodes (interleaved by
    wall-clock time), not whatever the last per-worker file happened to
    hold. Returns empty lists when no monitor logs exist.
    """
    from courtside_dynamics.training.monitor_log import load_monitor_episodes

    try:
        bundle = load_monitor_episodes(os.path.join(log_dir, "monitor"))
    except FileNotFoundError:
        return [], []
    df = bundle.episodes
    rewards = [float(x) for x in df["r"]] if "r" in df else []
    lengths = [int(x) for x in df["l"]] if "l" in df else []
    return rewards, lengths


#: SB3 ``train/*`` diagnostics worth surfacing in the static run report.
#: Missing keys are skipped, so the same list covers SAC and PPO.
_TRAINING_HEALTH_KEYS = (
    # SAC / off-policy
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/actor_loss",
    "train/critic_loss",
    # PPO / on-policy
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    # shared
    "train/learning_rate",
    "train/loss",
)


def _read_training_health(log_dir: str) -> dict[str, float]:
    """Final value of each known ``train/*`` metric from ``progress.csv``.

    SB3's CSV logger writes ``LOG_DIR/tensorboard/progress.csv`` with one
    column per metric; cells are blank when a metric wasn't logged on that
    row. We keep the last non-blank float per key, i.e. its end-of-run
    value. Returns ``{}`` if the file is absent (e.g. CSV logging off).
    """
    path = os.path.join(log_dir, "tensorboard", "progress.csv")
    if not os.path.exists(path):
        return {}
    finals: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in _TRAINING_HEALTH_KEYS:
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
    path = os.path.join(log_dir, "eval_info.csv")
    if not os.path.exists(path):
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


_PROJECT_NAME = "courtside-dynamics"


def _env_display_name(cfg: TrainConfig) -> str:
    """Friendly env name (e.g. ``WallBallEnv`` -> ``WallBall``)."""
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
) -> str:
    """Write a human-readable end-of-run report to ``log_dir/stage_summary.txt``."""
    env_name = _env_display_name(cfg)
    git_sha = _git_sha()
    short_sha = git_sha[:7] if git_sha else "unknown"
    duration_str = _format_duration(duration_seconds)
    throughput_fps = (
        int(cfg.total_timesteps / duration_seconds) if duration_seconds > 0 else 0
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
    lines.append(_kv("Status", "completed"))
    lines.append(_kv("Git SHA", short_sha))
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

    eval_npz = os.path.join(log_dir, "evaluations.npz")
    best_step: int | None = None
    best_mean: float | None = None
    best_std: float | None = None
    if os.path.exists(eval_npz):
        data = np.load(eval_npz)
        timesteps = data["timesteps"]
        results = data["results"]
        if results.size:
            mean_per_eval = results.mean(axis=1)
            std_per_eval = results.std(axis=1)
            best_idx = int(mean_per_eval.argmax())
            best_step = int(timesteps[best_idx])
            best_mean = float(mean_per_eval[best_idx])
            best_std = float(std_per_eval[best_idx])
            lines.append(
                _kv(
                    "Best eval",
                    f"{best_mean:.3f} +/- {best_std:.3f} (at {best_step:,} steps)",
                )
            )

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

    if best_step is not None and best_mean is not None and best_std is not None:
        lines.extend(
            _section(f"Best Checkpoint Evaluation (step {best_step:,})")
        )
        lines.append(f"  {_kv('Reward', f'{best_mean:.3f} +/- {best_std:.3f}')}")
        eval_info = _read_eval_info_at_step(log_dir, best_step)
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
    for label, path in [
        ("best_model", "best_model.zip"),
        ("final_model", "final_model.zip"),
        ("evaluations", "evaluations.npz"),
        ("config", "config.json"),
        ("learning_curve", "learning_curve.png"),
        ("eval_info_csv", "eval_info.csv"),
        ("eval_info_plot", "eval_info.png"),
        ("training_health_plot", "training_health.png"),
        ("progress_csv", "tensorboard/progress.csv"),
        ("checkpoints_dir", "checkpoints"),
        ("vec_normalize", "vec_normalize.pkl"),
        ("best_vec_normalize", "best_vec_normalize.pkl"),
        ("best_model_video", "best_model.mp4"),
    ]:
        full = os.path.join(log_dir, path)
        if os.path.exists(full):
            artifact_lines.append(f"  {_kv(label, path)}")
    if artifact_lines:
        lines.extend(_section("Artifacts"))
        lines.extend(artifact_lines)

    out = os.path.join(log_dir, "stage_summary.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out
