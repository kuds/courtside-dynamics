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

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import VecNormalize

from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
from courtside_dynamics.callbacks.video_record import (
    InfoRowFn,
    VideoRecordCallback,
)
from courtside_dynamics.training.algos import (
    ALGOS as _ALGOS,
)
from courtside_dynamics.training.algos import (
    OFF_POLICY_ALGOS as _OFF_POLICY_ALGOS,
)
from courtside_dynamics.training.artifacts import (
    update_run_config_with_model,
    write_run_config,
    write_run_summary,
)


class _SaveVecNormalizeOnNewBest(BaseCallback):
    """Snapshot the model's ``VecNormalize`` stats when EvalCallback saves a new best.

    Wired in via ``EvalCallback(..., callback_on_new_best=...)``. Without
    this, ``vec_normalize.pkl`` is only written at the very end of
    training, so its running stats reflect the *final* train_env state —
    not the moment ``best_model.zip`` was saved. Replaying ``best_model``
    against final-step stats is a silent obs-distribution mismatch.
    """

    def __init__(self, save_path: str) -> None:
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        get_vec_norm = getattr(self.model, "get_vec_normalize_env", None)
        if get_vec_norm is None:
            return True
        vec_norm = get_vec_norm()
        if vec_norm is None:
            return True
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        vec_norm.save(self.save_path)
        return True


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
    seed:
        Master RNG seed. When set it is forwarded to the SB3 algorithm
        (seeding policy init, action sampling, and the training env) and
        used to derive distinct, deterministic seeds for the eval,
        info-eval, and video-recording envs, making a run reproducible
        end-to-end. Leave ``None`` (default) for nondeterministic runs.
        An explicit ``model_kwargs['seed']`` takes precedence.
    eval_freq:
        Run ``EvalCallback`` every ``eval_freq`` *environment* steps
        (summed across workers). The helper converts this to SB3's
        per-vec-step ``n_calls`` semantics by dividing by ``n_envs``,
        so the cadence is independent of how many workers are used.
    checkpoint_freq:
        Save a full ``CheckpointCallback`` snapshot every
        ``checkpoint_freq`` environment steps. Same n_envs-independent
        semantics as ``eval_freq``. Set ``<= 0`` to skip checkpointing.
    video_freq:
        Record a rollout video every ``video_freq`` environment steps.
        Same n_envs-independent semantics. Set ``<= 0`` to skip videos
        (or use ``record_video=False``).
    n_eval_episodes:
        Episodes per ``EvalCallback`` evaluation.
    video_length:
        Max steps per recorded rollout from ``VideoRecordCallback``.
    record_video:
        Whether to attach the video recording callback.
    normalize_obs:
        When ``True`` (default), wrap envs in ``VecNormalize`` with a
        running observation mean/std so the policy sees inputs on a
        consistent scale. Stats are saved to ``LOG_DIR/vec_normalize.pkl``
        and alongside each ``CheckpointCallback`` snapshot.
    normalize_reward:
        Tri-state. ``True`` / ``False`` force VecNormalize's return
        normalization on/off. ``None`` (default) picks the per-algo
        default: on for PPO (stabilizes value-function learning), off
        for SAC (interacts poorly with auto-tuned entropy temperature).
    clip_obs / clip_reward:
        Forwarded to ``VecNormalize``. Defaults match SB3 (10.0 each).
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
    success_key / success_threshold:
        Forwarded to ``InfoDictEvalCallback``. When ``success_key`` is
        set, the fraction of eval episodes whose terminal
        ``info[success_key] >= success_threshold`` is logged as
        ``eval_info/success_rate`` -- the real task metric for sparse
        objectives (e.g. ``"bounce_count"`` for WallBall).
    verbose:
        SB3 ``verbose`` level for the algorithm; when non-zero, the
        per-rollout training table is also streamed to stdout (useful on
        long Colab runs). Independent of the CSV/TensorBoard logging that
        is always on.
    phase_key / phase_labels:
        Forwarded to ``InfoDictEvalCallback`` so envs with a state
        machine get per-phase time-fraction logs.
    extra_callbacks:
        Additional callbacks to run alongside eval / video recording.
    """

    env_fn: Callable
    algo: str = "SAC"
    total_timesteps: int = 1_000_000
    log_dir: str = "./logs/run"
    name_prefix: str = "rl_model"
    n_envs: int = 4
    seed: int | None = None
    eval_freq: int = 25_000
    checkpoint_freq: int = 250_000
    video_freq: int = 250_000
    n_eval_episodes: int = 30
    video_length: int = 10_000
    record_video: bool = True
    normalize_obs: bool = True
    normalize_reward: bool | None = None
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    policy: str = "MlpPolicy"
    verbose: int = 0
    model_kwargs: dict = field(default_factory=dict)
    csv_header: Sequence[str] | None = None
    info_row_fn: InfoRowFn | None = None
    info_dict_eval: bool = True
    success_key: str | None = None
    success_threshold: float = 1.0
    phase_key: str | None = None
    phase_labels: dict[int, str] | None = None
    extra_callbacks: Iterable[BaseCallback] = field(default_factory=tuple)


def _offset_seed(seed: int | None, offset: int) -> int | None:
    """Derive a distinct child seed, or ``None`` when seeding is disabled.

    Each helper env (eval, info-eval, video) gets its own offset so they
    don't all replay the identical stream the training env saw, while the
    whole run still reproduces exactly from a single master ``seed``.
    """
    return None if seed is None else seed + offset


def _build_algo(
    name: str,
    env,
    log_dir: str,
    *,
    policy: str = "MlpPolicy",
    verbose: int = 0,
    **model_kwargs,
) -> BaseAlgorithm:
    try:
        cls = _ALGOS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algo '{name}'. Expected one of {sorted(_ALGOS)}."
        ) from exc

    # SAC defaults to ``gradient_steps=1`` with ``train_freq=1``: each
    # rollout adds ``n_envs`` transitions to the replay buffer but runs only
    # ONE gradient update -- an update:data ratio of 1:n_envs that starves
    # the policy as ``n_envs`` grows. ``gradient_steps=-1`` runs as many
    # updates as steps collected, restoring a 1:1 ratio independent of
    # ``n_envs``. Applied only when the caller didn't pin it explicitly.
    if name.upper() in _OFF_POLICY_ALGOS:
        model_kwargs.setdefault("gradient_steps", -1)

    return cls(
        policy,
        env,
        verbose=verbose,
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
        seed=cfg.seed,
        monitor_dir=os.path.join(cfg.log_dir, "monitor"),
    )
    eval_env = make_vec_env(
        checked_env_fn, n_envs=1, seed=_offset_seed(cfg.seed, 1)
    )

    # PPO benefits from return normalization; SAC's auto-tuned alpha
    # interacts poorly with it. ``normalize_reward=None`` picks the
    # per-algo default; True/False forces the choice.
    norm_reward = (
        cfg.normalize_reward
        if cfg.normalize_reward is not None
        else cfg.algo.upper() == "PPO"
    )
    use_vec_normalize = cfg.normalize_obs or norm_reward
    if use_vec_normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=norm_reward,
            clip_obs=cfg.clip_obs,
            clip_reward=cfg.clip_reward,
        )
        # Eval envs always read the env's true reward (norm_reward=False)
        # and freeze the running stats (training=False); SB3's EvalCallback
        # syncs obs_rms/ret_rms from train_env before each eval.
        eval_env = VecNormalize(
            eval_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=False,
            clip_obs=cfg.clip_obs,
            training=False,
        )

    # SB3 callbacks fire on n_calls (per vec-env step), so an env-step
    # value of N means n_calls of max(N // n_envs, 1). This keeps the
    # cadence independent of n_envs.
    def _calls(env_steps: int) -> int:
        return max(env_steps // max(cfg.n_envs, 1), 1)

    on_new_best: BaseCallback | None = None
    if use_vec_normalize:
        on_new_best = _SaveVecNormalizeOnNewBest(
            os.path.join(cfg.log_dir, "best_vec_normalize.pkl")
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.log_dir,
        log_path=cfg.log_dir,
        render=False,
        deterministic=True,
        n_eval_episodes=cfg.n_eval_episodes,
        eval_freq=_calls(cfg.eval_freq),
        callback_on_new_best=on_new_best,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if cfg.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=_calls(cfg.checkpoint_freq),
                save_path=os.path.join(cfg.log_dir, "checkpoints"),
                name_prefix=cfg.name_prefix,
                save_vecnormalize=use_vec_normalize,
            )
        )
    if cfg.record_video and cfg.video_freq > 0:
        callbacks.append(
            VideoRecordCallback(
                env_fn=cfg.env_fn,
                save_path=os.path.join(cfg.log_dir, "videos"),
                video_length=cfg.video_length,
                save_freq=_calls(cfg.video_freq),
                name_prefix=cfg.name_prefix,
                csv_header=cfg.csv_header,
                info_row_fn=cfg.info_row_fn,
                seed=_offset_seed(cfg.seed, 3),
            )
        )
    if cfg.info_dict_eval:
        info_eval_env = make_vec_env(
            checked_env_fn, n_envs=1, seed=_offset_seed(cfg.seed, 2)
        )
        if use_vec_normalize:
            info_eval_env = VecNormalize(
                info_eval_env,
                norm_obs=cfg.normalize_obs,
                norm_reward=False,
                clip_obs=cfg.clip_obs,
                training=False,
            )
        callbacks.append(
            InfoDictEvalCallback(
                eval_env=info_eval_env,
                n_eval_episodes=max(1, cfg.n_eval_episodes // 4),
                eval_freq=_calls(cfg.eval_freq),
                phase_key=cfg.phase_key,
                phase_labels=cfg.phase_labels,
                success_key=cfg.success_key,
                success_threshold=cfg.success_threshold,
                csv_path=os.path.join(cfg.log_dir, "eval_info.csv"),
            )
        )
    callbacks.extend(cfg.extra_callbacks)

    # cfg.seed is the first-class knob; an explicit model_kwargs['seed']
    # still wins so power users can override per-algo if they want.
    model_kwargs = dict(cfg.model_kwargs)
    if cfg.seed is not None:
        model_kwargs.setdefault("seed", cfg.seed)
    model = _build_algo(
        cfg.algo,
        train_env,
        cfg.log_dir,
        policy=cfg.policy,
        verbose=cfg.verbose,
        **model_kwargs,
    )

    # Route SB3's own diagnostics (SAC ent_coef/actor/critic losses, PPO
    # explained_variance/approx_kl, ...) to a CSV alongside TensorBoard so
    # the run directory is self-diagnosing after the Colab runtime is gone.
    # ``progress.csv`` is read back by stage_summary + plot_training_health.
    log_formats = ["csv", "tensorboard"]
    if cfg.verbose:
        log_formats.append("stdout")
    model.set_logger(
        configure_logger(os.path.join(cfg.log_dir, "tensorboard"), log_formats)
    )
    update_run_config_with_model(model, cfg.log_dir)

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        model.save(os.path.join(cfg.log_dir, "final_model"))
        if use_vec_normalize:
            train_env.save(os.path.join(cfg.log_dir, "vec_normalize.pkl"))

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
