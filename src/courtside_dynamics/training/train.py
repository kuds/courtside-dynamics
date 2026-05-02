"""High-level training entry point shared by every environment notebook.

A typical notebook cell boils down to::

    from courtside_dynamics.training import TrainConfig, train
    from courtside_dynamics.envs import BallBounceEnv

    cfg = TrainConfig(
        env_fn=lambda: BallBounceEnv(render_mode="rgb_array", min_force=100.0),
        algo="SAC",
        total_timesteps=1_500_000,
        log_dir="./logs/BallBounce",
        name_prefix="ball_bounce",
    )
    model = train(cfg)

The helper builds vectorized train / eval envs, wires ``EvalCallback`` and
``VideoRecordCallback``, and runs ``model.learn`` end-to-end.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
from courtside_dynamics.callbacks.video_record import (
    InfoRowFn,
    VideoRecordCallback,
)
from courtside_dynamics.training.artifacts import (
    write_run_config,
    write_run_summary,
)

_ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
}


@dataclass
class TrainConfig:
    """All knobs for a single training run.

    Attributes
    ----------
    env_fn:
        Zero-arg factory returning a fresh Gymnasium env. The helper wraps
        it with ``check_env`` and replicates it across ``n_envs`` workers.
    algo:
        ``"SAC"`` or ``"PPO"``.
    total_timesteps:
        Total env steps to train for (summed across workers).
    log_dir:
        Directory for monitor logs, TensorBoard, evaluation results,
        videos, and the final/best model checkpoints.
    name_prefix:
        Filename prefix used by the video recording callback.
    n_envs:
        Number of parallel training workers.
    eval_freq:
        Run ``EvalCallback`` every ``eval_freq`` training steps (summed
        across workers).
    n_eval_episodes:
        Episodes per ``EvalCallback`` evaluation.
    video_length:
        Max steps per recorded rollout from ``VideoRecordCallback``.
    record_video:
        Whether to attach the video recording callback.
    policy:
        SB3 policy name, usually ``"MlpPolicy"``.
    model_kwargs:
        Extra kwargs forwarded to the SB3 algorithm constructor.
    csv_header / info_row_fn:
        Passed through to ``VideoRecordCallback`` so envs can log their
        custom info rows.
    info_dict_eval:
        When ``True`` (default), attaches an ``InfoDictEvalCallback`` that
        logs per-episode aggregates of every scalar ``info`` key from
        eval rollouts to TensorBoard. Set ``False`` to skip this pass
        (e.g. for envs that emit no interesting ``info`` scalars).
    phase_key / phase_labels:
        Forwarded to ``InfoDictEvalCallback`` so envs with a state
        machine (e.g. TennisWall) get per-phase time-fraction logs.
    extra_callbacks:
        Additional callbacks to run alongside eval / video recording.
    """

    env_fn: Callable
    algo: str = "SAC"
    total_timesteps: int = 1_000_000
    log_dir: str = "./logs/run"
    name_prefix: str = "rl_model"
    n_envs: int = 4
    eval_freq: int = 25_000
    n_eval_episodes: int = 30
    video_length: int = 10_000
    record_video: bool = True
    policy: str = "MlpPolicy"
    model_kwargs: dict = field(default_factory=dict)
    csv_header: Sequence[str] | None = None
    info_row_fn: InfoRowFn | None = None
    info_dict_eval: bool = True
    phase_key: str | None = None
    phase_labels: dict[int, str] | None = None
    extra_callbacks: Iterable[BaseCallback] = field(default_factory=tuple)


def _build_algo(name: str, env, log_dir: str, **model_kwargs) -> BaseAlgorithm:
    try:
        cls = _ALGOS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algo '{name}'. Expected one of {sorted(_ALGOS)}."
        ) from exc
    return cls(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **model_kwargs,
    )


def train(cfg: TrainConfig) -> BaseAlgorithm:
    """Run an end-to-end training loop for one ``TrainConfig``.

    Returns the trained model. Side effects: writes monitor logs,
    TensorBoard events, evaluation npz, video MP4s, ``best_model`` /
    ``final_model`` checkpoints, plus ``config.json`` (provenance
    snapshot at start) and ``stage_summary.txt`` (eval / wall-clock
    report at end) under ``cfg.log_dir``.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    write_run_config(cfg, cfg.log_dir)
    start_time = time.monotonic()

    def checked_env_fn():
        env = cfg.env_fn()
        check_env(env)
        return env

    train_env = make_vec_env(
        checked_env_fn,
        n_envs=cfg.n_envs,
        monitor_dir=os.path.join(cfg.log_dir, "monitor"),
    )
    eval_env = make_vec_env(checked_env_fn, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.log_dir,
        log_path=cfg.log_dir,
        render=False,
        deterministic=True,
        n_eval_episodes=cfg.n_eval_episodes,
        eval_freq=cfg.eval_freq,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if cfg.record_video:
        callbacks.append(
            VideoRecordCallback(
                env_fn=cfg.env_fn,
                save_path=os.path.join(cfg.log_dir, "videos"),
                video_length=cfg.video_length,
                save_freq=cfg.eval_freq,
                name_prefix=cfg.name_prefix,
                csv_header=cfg.csv_header,
                info_row_fn=cfg.info_row_fn,
            )
        )
    if cfg.info_dict_eval:
        info_eval_env = make_vec_env(checked_env_fn, n_envs=1)
        callbacks.append(
            InfoDictEvalCallback(
                eval_env=info_eval_env,
                n_eval_episodes=max(1, cfg.n_eval_episodes // 4),
                eval_freq=cfg.eval_freq,
                phase_key=cfg.phase_key,
                phase_labels=cfg.phase_labels,
            )
        )
    callbacks.extend(cfg.extra_callbacks)

    model = _build_algo(cfg.algo, train_env, cfg.log_dir, **cfg.model_kwargs)

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        model.save(os.path.join(cfg.log_dir, "final_model"))

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=cfg.n_eval_episodes
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        write_run_summary(
            cfg,
            cfg.log_dir,
            final_mean_reward=float(mean_reward),
            final_std_reward=float(std_reward),
            duration_seconds=time.monotonic() - start_time,
            device=str(getattr(model, "device", "")) or None,
        )
    finally:
        train_env.close()
        eval_env.close()

    return model
