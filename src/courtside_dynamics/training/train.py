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

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from courtside_dynamics.callbacks.env_attr_schedule import (
    LinearEnvAttrScheduleCallback,
)
from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
from courtside_dynamics.callbacks.performance_gate import (
    PerformanceGatedEnvStagesCallback,
)
from courtside_dynamics.callbacks.video_record import (
    InfoRowFn,
    VideoRecordCallback,
)
from courtside_dynamics.training.algos import (
    OFF_POLICY_ALGOS as _OFF_POLICY_ALGOS,
)
from courtside_dynamics.training.algos import (
    resolve_algo as _resolve_algo,
)
from courtside_dynamics.training.artifacts import (
    RUN_LAYOUT,
    artifact_path,
    locate_artifact,
    update_run_config_with_initialization,
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


@dataclass(frozen=True, slots=True)
class WarmStartConfig:
    """Policy-only initialization from one canonical prior best run.

    The source directory must contain the paired ``best_model.zip`` and
    ``best_vec_normalize.pkl`` artifacts plus that run's ``config.json``.
    The source and target must run the same algorithm (PPO or SAC). Only
    policy tensors and observation running statistics transfer -- for SAC
    that includes actor, critics, and critic targets via the policy
    state dict, plus the auto-tuned entropy temperature when both sides
    use auto entropy (a fresh ``"auto"`` restarts at ent_coef 1.0, which
    would churn a converged policy). The new run deliberately starts
    with fresh optimizer, reward-normalization, schedule, timestep,
    buffer, logger, and callback state.

    ``reset_observation_indices`` identifies deterministic observation fields
    whose semantics changed between fixed stages (notably newly active mask
    entries). Their carried mean is replaced with the target environment's
    seeded reset value and their variance with one, so the initial normalized
    value is zero instead of clipping against stale near-zero variance.
    """

    source_run_dir: str | Path
    reset_observation_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_run_dir, (str, os.PathLike)):
            raise TypeError("source_run_dir must be a path")
        if not str(self.source_run_dir).strip():
            raise ValueError("source_run_dir must not be empty")
        indices = tuple(self.reset_observation_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("reset_observation_indices must contain integers")
        if any(index < 0 for index in indices):
            raise ValueError("reset_observation_indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("reset_observation_indices must be unique")
        object.__setattr__(self, "reset_observation_indices", tuple(sorted(indices)))


@dataclass(frozen=True, slots=True)
class _WarmStartArtifacts:
    source_run_dir: Path
    model_path: Path
    normalizer_path: Path
    config_path: Path
    source_config: dict[str, Any]
    source_hashes: dict[str, str]
    target_env_class: str
    target_curriculum: dict[str, Any] | None
    reset_observation_values: tuple[float, ...]


class SelectiveVecNormalize(VecNormalize):
    """VecNormalize that leaves selected one-dimensional observations raw.

    Humanoid-tennis' semantic/discrete tail is already bounded and should not
    inherit stale running statistics across stages.  Exclusions are symmetric:
    ``normalize_obs`` restores raw values and ``unnormalize_obs`` restores the
    corresponding normalized inputs, so both methods remain inverses.

    The exclusion tuple is ordinary pickle state.  ``VecNormalize.save/load``
    therefore preserves this subclass and its exact normalization contract.
    """

    def __init__(
        self,
        venv: VecEnv,
        *,
        normalize_obs_excluded_indices: Sequence[int] = (),
        **kwargs: Any,
    ) -> None:
        self.normalize_obs_excluded_indices = _validate_observation_indices(
            normalize_obs_excluded_indices,
            name="normalize_obs_excluded_indices",
        )
        super().__init__(venv, **kwargs)
        if self.normalize_obs_excluded_indices:
            shape = self.observation_space.shape
            if shape is None or len(shape) != 1:
                raise ValueError(
                    "normalize_obs_excluded_indices require a one-dimensional "
                    "Box observation"
                )
            if self.normalize_obs_excluded_indices[-1] >= shape[0]:
                raise ValueError(
                    "normalize_obs_excluded_indices exceed the observation size"
                )

    def normalize_obs(self, obs: Any) -> Any:
        normalized = super().normalize_obs(obs)
        if not self.norm_obs or not self.normalize_obs_excluded_indices:
            return normalized
        if isinstance(normalized, dict):
            raise TypeError("selective normalization requires a Box observation")
        result = np.asarray(normalized).copy()
        raw = np.asarray(obs)
        result[..., self.normalize_obs_excluded_indices] = raw[
            ..., self.normalize_obs_excluded_indices
        ]
        return result

    def unnormalize_obs(self, obs: Any) -> Any:
        unnormalized = super().unnormalize_obs(obs)
        if not self.norm_obs or not self.normalize_obs_excluded_indices:
            return unnormalized
        if isinstance(unnormalized, dict):
            raise TypeError("selective normalization requires a Box observation")
        result = np.asarray(unnormalized).copy()
        normalized = np.asarray(obs)
        result[..., self.normalize_obs_excluded_indices] = normalized[
            ..., self.normalize_obs_excluded_indices
        ]
        return result


@dataclass
class TrainConfig:
    """All knobs for a single training run.

    Attributes
    ----------
    env_fn:
        Zero-arg factory returning a fresh Gymnasium env. The helper runs
        ``check_env`` on the first instance it builds and replicates the
        factory across ``n_envs`` workers.
    eval_env_fn:
        Optional zero-arg factory used exclusively for evaluation, final
        scoring, and milestone videos. When omitted, ``env_fn`` is reused.
        Curriculum recipes should provide a factory with curriculum resets
        disabled so checkpoint selection always measures complete episodes
        from the canonical starting state.
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
    early_stop_patience:
        When set to N, stop training after N consecutive evaluations
        without a new best. "Best" follows the same metric as model
        selection: the headline task metric when ``headline_key`` is
        set (see below), otherwise mean eval reward via SB3's
        ``StopTrainingOnNoModelImprovement``. The same count is used
        as the warm-up (``min_evals``) and warm-up evaluations never
        count against the patience, so a run gets at least 2N
        evaluations before it can stop. When ``env_attr_schedules``
        are configured, the warm-up is extended to cover the longest
        schedule so the run always trains for a full patience window
        at the final scheduled distribution (run 20260714_211111's
        schedule endpoint coincided exactly with its earliest
        reachable stop, guaranteeing zero time at the target
        distribution). ``None`` (default) trains for the full budget.
        Motivation: the first real WallBall run kept training for 10+
        GPU-hours after its best checkpoint while the policy
        collapsed; run 20260712_190054 then spent 525k steps
        confirming a reward plateau of a degenerate policy -- both the
        selection and the stop must watch the metric that matters.
    video_length:
        Max steps per recorded rollout from ``VideoRecordCallback``.
    record_video:
        Whether to attach the video recording callback.
    normalize_obs:
        When ``True`` (default), wrap envs in ``VecNormalize`` with a
        running observation mean/std so the policy sees inputs on a
        consistent scale. Stats are saved to
        ``LOG_DIR/model/vec_normalize.pkl`` and alongside each
        ``CheckpointCallback`` snapshot.
    normalize_reward:
        Tri-state. ``True`` / ``False`` force VecNormalize's return
        normalization on/off. ``None`` (default) picks the per-algo
        default: on for PPO (stabilizes value-function learning), off
        for SAC (interacts poorly with auto-tuned entropy temperature).
    clip_obs / clip_reward:
        Forwarded to ``VecNormalize``. Defaults match SB3 (10.0 each).
    normalize_obs_excluded_indices:
        Indices in a one-dimensional observation that remain raw while all
        other coordinates use VecNormalize. The same exclusion tuple must be
        used by a source and target warm-start run.
    policy:
        SB3 policy name, usually ``"MlpPolicy"``.
    model_kwargs:
        Extra kwargs forwarded to the SB3 algorithm constructor.
    warm_start:
        Optional policy-only initialization from a prior run directory of
        the same algorithm (PPO or SAC). The canonical best checkpoint
        and matching VecNormalize snapshot are validated and their
        policy/observation state is transferred into a fresh target
        model (for SAC, also the auto-tuned entropy temperature). This
        is intentionally not a training resume: optimizers, buffers,
        timesteps, and callback state start fresh, so an off-policy
        continuation should pair this with a raised ``learning_starts``
        (fresh data before the first update).
    csv_header / info_row_fn:
        Passed through to ``VideoRecordCallback`` so envs can log their
        custom info rows.
    info_dict_eval:
        When ``True`` (default), attaches an ``InfoDictEvalCallback`` that
        logs per-episode aggregates of every scalar ``info`` key from
        eval rollouts to TensorBoard. Set ``False`` to skip this pass
        (e.g. for envs that emit no interesting ``info`` scalars).
    info_eval_keys / info_eval_terminal_keys:
        Optional compact evaluation schema. ``info_eval_keys`` limits
        per-step aggregation to an allowlist; ``None`` keeps the historical
        all-scalars behaviour. ``info_eval_terminal_keys`` records selected
        terminal values only, such as mutually exclusive fault flags.
    info_eval_distribution_keys:
        Terminal counters for which the evaluator reports episode
        min/median/90th-percentile/max values.
    info_eval_survival_thresholds:
        Mapping from terminal counters to positive integer thresholds. The
        info evaluator reports the fraction of episodes reaching every
        threshold as ``<key>_ep_ge_<threshold>_rate``.
    success_key / success_threshold:
        Forwarded to ``InfoDictEvalCallback``. When ``success_key`` is
        set, the fraction of eval episodes whose terminal
        ``info[success_key]`` >= ``success_threshold`` is logged as
        ``eval_info/success_rate`` -- the real task metric for sparse
        objectives (e.g. ``"bounce_count"`` for WallBall).
    headline_key:
        Info counter treated as the run's headline task metric. The
        stage summary surfaces ``<headline_key>_ep_mean`` (the
        mean-over-eval-episodes terminal value logged by
        ``InfoDictEvalCallback``) next to the reward lines, and
        ``plot_eval_info`` puts its panel first. Use this when eval
        reward is a poor progress measure -- WallBall's reward is
        dominated by tracking shaping, and its success_rate saturates
        as soon as every episode completes one exchange, so runs are
        compared on ``bounce_count_ep_mean`` (rally exchanges per
        episode) instead.

        When set (and ``info_dict_eval`` is on), the headline metric
        also *owns model selection and early stopping*:
        ``best_model.zip`` / ``best_vec_normalize.pkl`` are saved by
        ``InfoDictEvalCallback`` on a new best
        ``(<headline_key>_ep_mean, success_rate,
        episode_reward_mean)`` lexicographic score (reward is only the
        final tie-break), a ``best_model_meta.json`` records which
        step won and why, and ``early_stop_patience`` counts
        evaluations without improvement of that score. Run
        20260712_190054 is the cautionary tale: reward-based selection
        crowned a catch-and-stall policy with zero completed rallies
        as ``best_model.zip`` while ``bounce_count_ep_mean`` sat at 0.
        The info-dict evaluator then runs the full
        ``n_eval_episodes`` (not the /4 reporting sample) so
        selection isn't keyed to a noisy 7-episode estimate.
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
    env_attr_schedules:
        Declarative linear schedules for numeric attributes on every training
        environment. Each mapping is forwarded to
        ``LinearEnvAttrScheduleCallback`` and is driven by SB3's global
        timestep count. Evaluation environments are never modified.
    best_metric_keys:
        Optional override of the lexicographic selection score used when
        ``headline_key`` owns model selection. ``None`` (default) keeps
        the historical ``(<headline_key>_ep_mean, success_rate,
        episode_reward_mean)`` composition. WallBall runs
        20260714_050506/211111 showed why a recipe may want to drop the
        reward tie-break: eval reward there was ~88% tracking shaping
        in one run and pure penalty float-noise in the other, and the
        noise crowned a best model.
    best_metric_min_delta:
        Minimum per-key improvement for the selection score (see
        ``InfoDictEvalCallback.best_metric_min_delta``). The default
        ``0.0`` keeps the historical strict-``>`` semantics, which is
        right for selection keys that include a continuous reward
        tie-break. Recipes whose keys are all episode means/rates over
        ``n`` episodes should set roughly half that granularity
        (``0.5 / n``) so a real one-episode change registers while
        float noise (run 20260714_211111's 1e-8 reward "improvement")
        cannot crown a best model or reset the early-stop patience.
    confirm_best_eval:
        When ``True``, a candidate best must also win an independent
        second eval batch before ``best_model.zip`` is overwritten
        (see ``InfoDictEvalCallback.confirm_best``). Default ``False``.
    performance_gate:
        Optional performance-gated curriculum, a mapping with keys
        ``stages`` (ordered sequence of {env attribute: value} dicts),
        ``metric_key``, ``threshold``, and ``sustain_evals``, forwarded
        to ``PerformanceGatedEnvStagesCallback``. Stage 0 is applied at
        training start to the training env AND the info-eval env (so
        selection/guard metrics stay matched to the training stage); the
        ladder advances one stage each time the metric holds the
        threshold for the sustained number of consecutive evaluations.
        Requires ``info_dict_eval``. Unlike ``env_attr_schedules`` this
        is earned progression, not a timestep clock.
    reward_eval_episodes:
        Episode count for the reward-only ``EvalCallback`` stream. Only
        settable when a headline metric owns selection (otherwise that
        stream IS selection and must keep the full ``n_eval_episodes``).
        Under headline selection the stream is reporting-only
        (``evaluations.npz`` + ``eval/mean_reward``), yet historically
        still rolled the full 30 episodes every ``eval_freq`` — in run
        20260721_004722 the three 30-episode eval streams together cost
        about as many env steps as the 3M training steps themselves.
        ``None`` (default) keeps ``n_eval_episodes``.

        Ignored when ``final_info_eval`` is also on: the two streams roll
        the *same* distribution, so the reward evaluator is retired
        entirely and ``final_eval_episodes`` sizes the surviving stream.
    final_eval_episodes:
        Episode count for the ``final_info_eval`` stream. ``None``
        (default) resolves to the full ``n_eval_episodes`` when the reward
        evaluator was merged into this stream, and to the historical
        ``n_eval_episodes // 2`` reporting sample when both streams still
        run.

        The full count is the right default for a merged stream because
        it is then the *only* stream scoring the campaign's goal task
        during training. At run 20260721_004722's per-episode spread the
        old 5-episode reward stream had a standard error of ~0.8-1.1
        bounces -- it could not resolve the transfer curve
        (0.30 -> 0.98 -> 1.76) that the depth campaign exists to buy.
        Set this explicitly to trade that resolution back for wall clock.
    final_info_eval:
        When ``True``, attach a second ``InfoDictEvalCallback`` running
        on the *unmodified* evaluation env (the recipe's
        ``eval_env_overrides`` configuration -- for a curriculum, the
        final stage) under the ``eval_info_final`` prefix and
        ``eval_info_final.csv``. With a performance gate active, the
        matched stream drives selection while this stream is the honest
        final-task progress metric; the gap between them is the
        transfer deficit, visible per evaluation instead of
        post-mortem. Requires ``info_dict_eval``.
    early_stop_degenerate_evals / degenerate_guard_keys /
    degenerate_min_evals:
        Enable ``InfoDictEvalCallback``'s degenerate-signal stop: end
        the run once ``early_stop_degenerate_evals`` consecutive
        evaluations produce a flat selection score (within
        ``best_metric_min_delta``) while every
        ``degenerate_guard_keys`` metric is exactly zero. Kills runs
        whose eval signal is provably dead (zero task competence, flat
        score) within a handful of evaluations instead of a full
        patience window. ``degenerate_min_evals`` warms the guard up;
        when ``None`` (default) and ``env_attr_schedules`` exist, it
        is derived to cover the longest schedule's hold phase, where a
        dead full-difficulty eval is still expected by design.
        Defaults (0 / empty) disable the guard.
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
    early_stop_patience: int | None = None
    video_length: int = 10_000
    record_video: bool = True
    normalize_obs: bool = True
    normalize_reward: bool | None = None
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    normalize_obs_excluded_indices: tuple[int, ...] = ()
    policy: str = "MlpPolicy"
    verbose: int = 0
    model_kwargs: dict = field(default_factory=dict)
    warm_start: WarmStartConfig | None = None
    csv_header: Sequence[str] | None = None
    info_row_fn: InfoRowFn | None = None
    info_dict_eval: bool = True
    info_eval_keys: Sequence[str] | None = None
    info_eval_terminal_keys: Sequence[str] = field(default_factory=tuple)
    info_eval_distribution_keys: Sequence[str] = field(default_factory=tuple)
    success_key: str | None = None
    success_threshold: float = 1.0
    headline_key: str | None = None
    phase_key: str | None = None
    phase_labels: dict[int, str] | None = None
    extra_callbacks: Iterable[BaseCallback] = field(default_factory=tuple)
    # New fields stay after the original public dataclass fields so existing
    # positional TrainConfig construction keeps its historical meaning.
    eval_env_fn: Callable | None = None
    recipe_name: str | None = None
    info_eval_survival_thresholds: Mapping[str, Sequence[int]] = field(
        default_factory=dict
    )
    env_attr_schedules: Sequence[Mapping[str, Any]] = field(
        default_factory=tuple
    )
    best_metric_keys: Sequence[str] | None = None
    best_metric_min_delta: float = 0.0
    confirm_best_eval: bool = False
    early_stop_degenerate_evals: int = 0
    degenerate_guard_keys: Sequence[str] = field(default_factory=tuple)
    degenerate_min_evals: int | None = None
    performance_gate: Mapping[str, Any] | None = None
    final_info_eval: bool = False
    reward_eval_episodes: int | None = None
    final_eval_episodes: int | None = None
    # Parsed TOML run-configuration file (courtside_dynamics.run_config
    # .RunFileConfig), attached by build_train_config(config_file=...)
    # so artifacts can record its provenance and copy it into the run
    # directory. ``Any`` to keep this module free of a run_config import.
    run_config_file: Any = None


def _offset_seed(seed: int | None, offset: int) -> int | None:
    """Derive a distinct child seed, or ``None`` when seeding is disabled.

    Each helper env (eval, info-eval, video) gets its own offset so they
    don't all replay the identical stream the training env saw, while the
    whole run still reproduces exactly from a single master ``seed``.

    Callers must keep helper offsets *past the training-worker block*:
    ``make_vec_env(seed=s, n_envs=n)`` seeds worker ``i`` with ``s + i``,
    so any offset below ``n_envs`` would hand a helper env the exact seed
    (and reset stream) of one of the training workers -- the eval env
    would then score the policy on state sequences it trained on.
    """
    return None if seed is None else seed + offset


def _env_steps_to_calls(env_steps: int, n_envs: int) -> int:
    """Convert an env-step cadence into SB3's per-vec-step ``n_calls``.

    SB3 callbacks fire on ``n_calls`` (one per vec-env step, i.e. per
    ``n_envs`` environment steps), so a cadence of N environment steps
    means ``max(N // n_envs, 1)`` calls. This keeps eval/checkpoint/video
    frequency independent of how many workers a run uses.
    """
    return max(env_steps // max(n_envs, 1), 1)


def _build_algo(
    name: str,
    env,
    log_dir: str,
    *,
    policy: str = "MlpPolicy",
    **model_kwargs,
) -> BaseAlgorithm:
    cls = _resolve_algo(name)

    # SAC defaults to ``gradient_steps=1`` with ``train_freq=1``: each
    # rollout adds ``n_envs`` transitions to the replay buffer but runs only
    # ONE gradient update -- an update:data ratio of 1:n_envs that starves
    # the policy as ``n_envs`` grows. ``gradient_steps=-1`` runs as many
    # updates as steps collected, restoring a 1:1 ratio independent of
    # ``n_envs``. Applied only when the caller didn't pin it explicitly.
    if name.upper() in _OFF_POLICY_ALGOS:
        model_kwargs.setdefault("gradient_steps", -1)

    # ``verbose`` flows through ``model_kwargs`` (not an explicit param) so a
    # caller-supplied ``model_kwargs["verbose"]`` can't collide with a second
    # ``verbose=`` keyword and raise TypeError at construction.
    return cls(
        policy,
        env,
        tensorboard_log=artifact_path(log_dir, "tensorboard_dir"),
        **model_kwargs,
    )


def _validate_observation_indices(
    indices: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    resolved = tuple(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in resolved):
        raise TypeError(f"{name} must contain integers")
    if any(index < 0 for index in resolved):
        raise ValueError(f"{name} must be non-negative")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(resolved))


def _resolved_norm_reward(cfg: TrainConfig) -> bool:
    return (
        cfg.normalize_reward
        if cfg.normalize_reward is not None
        else cfg.algo.upper() == "PPO"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_norm_reward(train_config: dict[str, Any]) -> bool:
    configured = train_config.get("normalize_reward")
    if configured is None:
        return str(train_config.get("algo", "")).upper() == "PPO"
    if not isinstance(configured, bool):
        raise ValueError("source normalize_reward must be bool or null")
    return configured


def _prepare_warm_start(cfg: TrainConfig) -> _WarmStartArtifacts | None:
    """Resolve and cheaply validate a canonical prior run before writing output."""
    warm_start = cfg.warm_start
    if warm_start is None:
        return None
    algo = cfg.algo.upper()
    if algo not in ("PPO", "SAC"):
        raise ValueError(
            "policy-only warm start currently supports PPO and SAC only"
        )
    if not cfg.normalize_obs:
        raise ValueError("policy-only warm start requires normalize_obs=True")

    source_dir = Path(warm_start.source_run_dir).expanduser().resolve()
    target_dir = Path(cfg.log_dir).expanduser().resolve()
    if source_dir == target_dir:
        raise ValueError("warm-start source and target log directories must differ")
    if not source_dir.is_dir():
        raise ValueError(f"warm-start source run does not exist: {source_dir}")

    # ``locate_artifact`` resolves the 0.14.0 layout (``model/best_model.zip``)
    # with a fallback to the legacy flat root, so a warm start can source
    # both old and new run directories. When neither location exists, fall
    # back to the canonical new path so the error names a concrete file.
    def _source_artifact(name: str) -> Path:
        located = locate_artifact(source_dir, name)
        return Path(located) if located else source_dir / RUN_LAYOUT[name]

    model_path = _source_artifact("best_model")
    normalizer_path = _source_artifact("best_vec_normalize")
    config_path = _source_artifact("config")
    for label, path in (
        ("best model", model_path),
        ("best VecNormalize", normalizer_path),
        ("run config", config_path),
    ):
        if not path.is_file():
            raise ValueError(f"warm-start {label} is missing: {path}")

    try:
        source_config_value = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"warm-start config is not valid JSON: {config_path}"
        ) from error
    if not isinstance(source_config_value, dict):
        raise ValueError("warm-start config must contain a JSON object")
    source_config: dict[str, Any] = source_config_value
    source_train = source_config.get("train_config")
    source_env = source_config.get("env")
    if not isinstance(source_train, dict) or not isinstance(source_env, dict):
        raise ValueError("warm-start config lacks train_config/env provenance")
    if str(source_train.get("algo", "")).upper() != algo:
        raise ValueError(
            f"warm-start source algo must match the target "
            f"({algo}), got {source_train.get('algo')!r}"
        )
    if source_train.get("normalize_obs") is not True:
        raise ValueError("warm-start source must have normalize_obs=True")
    try:
        source_exclusions = _validate_observation_indices(
            source_train.get("normalize_obs_excluded_indices", ()),
            name="source normalize_obs_excluded_indices",
        )
    except TypeError as error:
        raise ValueError(str(error)) from error
    target_exclusions = _validate_observation_indices(
        cfg.normalize_obs_excluded_indices,
        name="normalize_obs_excluded_indices",
    )
    if source_exclusions != target_exclusions:
        raise ValueError("source and target normalize_obs_excluded_indices differ")
    if _config_norm_reward(source_train) != _resolved_norm_reward(cfg):
        raise ValueError("source and target reward-normalization settings differ")
    for name, target in (("clip_obs", cfg.clip_obs), ("clip_reward", cfg.clip_reward)):
        try:
            source = float(source_train[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"warm-start source config lacks valid {name}") from error
        if not math.isclose(source, float(target), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"source and target {name} settings differ")

    target_env = cfg.env_fn()
    try:
        target_class = type(target_env).__name__
        target_observation_shape = list(target_env.observation_space.shape or ())
        target_action_shape = list(target_env.action_space.shape or ())
        target_curriculum_value = getattr(target_env, "curriculum_metadata", None)
        target_curriculum = (
            dict(target_curriculum_value)
            if isinstance(target_curriculum_value, Mapping)
            else None
        )
        reset_values: tuple[float, ...] = ()
        if warm_start.reset_observation_indices:
            result = target_env.reset(seed=cfg.seed)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("target reset must return (observation, info)")
            observation = np.asarray(result[0])
            if observation.ndim != 1:
                raise ValueError(
                    "reset_observation_indices require a one-dimensional observation"
                )
            if warm_start.reset_observation_indices[-1] >= observation.size:
                raise ValueError(
                    "reset_observation_indices exceed the observation size"
                )
            reset_values = tuple(
                float(observation[index])
                for index in warm_start.reset_observation_indices
            )
            if not all(math.isfinite(value) for value in reset_values):
                raise ValueError("target reset observation values must be finite")
    finally:
        target_env.close()

    if source_env.get("class") != target_class:
        raise ValueError("source and target environment classes differ")
    if source_env.get("observation_shape") != target_observation_shape:
        raise ValueError("source and target observation spaces differ")
    if source_env.get("action_shape") != target_action_shape:
        raise ValueError("source and target action spaces differ")

    return _WarmStartArtifacts(
        source_run_dir=source_dir,
        model_path=model_path,
        normalizer_path=normalizer_path,
        config_path=config_path,
        source_config=source_config,
        source_hashes={
            "best_model.zip": _sha256_file(model_path),
            "best_vec_normalize.pkl": _sha256_file(normalizer_path),
            "config.json": _sha256_file(config_path),
        },
        target_env_class=target_class,
        target_curriculum=target_curriculum,
        reset_observation_values=reset_values,
    )


def _load_warm_start_normalizer(
    base_env: VecEnv,
    artifacts: _WarmStartArtifacts,
    cfg: TrainConfig,
    *,
    norm_reward: bool,
) -> VecNormalize:
    """Attach source observation statistics and reset target-task state."""
    normalizer = VecNormalize.load(str(artifacts.normalizer_path), base_env)
    if normalizer.norm_obs is not True:
        raise ValueError("warm-start normalizer must have norm_obs=True")
    source_exclusions = _validate_observation_indices(
        getattr(normalizer, "normalize_obs_excluded_indices", ()),
        name="source normalize_obs_excluded_indices",
    )
    target_exclusions = _validate_observation_indices(
        cfg.normalize_obs_excluded_indices,
        name="normalize_obs_excluded_indices",
    )
    if source_exclusions != target_exclusions:
        raise ValueError("source and target normalizer observation exclusions differ")
    if normalizer.norm_reward != norm_reward:
        raise ValueError("source and target normalizer norm_reward settings differ")
    for name, target in (("clip_obs", cfg.clip_obs), ("clip_reward", cfg.clip_reward)):
        source = float(getattr(normalizer, name))
        if not math.isclose(source, float(target), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"source and target normalizer {name} settings differ")
    if isinstance(normalizer.obs_rms, dict):
        raise ValueError("policy-only warm start currently requires a Box observation")
    mean = np.asarray(normalizer.obs_rms.mean)
    variance = np.asarray(normalizer.obs_rms.var)
    expected_shape = tuple(normalizer.observation_space.shape or ())
    if mean.shape != expected_shape or variance.shape != expected_shape:
        raise ValueError("warm-start observation statistics have the wrong shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("warm-start observation statistics must be finite")
    if np.any(variance < 0.0):
        raise ValueError("warm-start observation variance must be non-negative")

    indices = cfg.warm_start.reset_observation_indices if cfg.warm_start else ()
    if indices:
        index_array = np.asarray(indices, dtype=np.intp)
        mean[index_array] = artifacts.reset_observation_values
        variance[index_array] = 1.0

    # Observation scale is the only running state intentionally transferred.
    # A new fixed stage gets a fresh return distribution and rollout state.
    normalizer.ret_rms = RunningMeanStd(shape=())
    normalizer.returns = np.zeros(normalizer.num_envs)
    normalizer.old_reward = np.array([])
    normalizer.old_obs = np.array([])
    normalizer.training = True
    return normalizer


def train(cfg: TrainConfig) -> BaseAlgorithm:
    """Run an end-to-end training loop for one ``TrainConfig``.

    Returns the trained model. Side effects: writes monitor logs,
    TensorBoard events, evaluation npz, video MP4s, ``best_model`` /
    ``final_model`` checkpoints, plus ``config.json`` (provenance
    snapshot at start) and ``stage_summary.txt`` (eval / wall-clock
    report at end) under ``cfg.log_dir``.
    """
    # Validate the algo name up front, before any envs are built or
    # artifacts written -- a typo'd ``algo`` should fail in milliseconds,
    # not after constructing (and checking) a fleet of MuJoCo envs.
    _resolve_algo(cfg.algo)
    env_attr_schedule_callbacks = tuple(
        LinearEnvAttrScheduleCallback(**dict(schedule))
        for schedule in cfg.env_attr_schedules
    )
    cfg.normalize_obs_excluded_indices = _validate_observation_indices(
        cfg.normalize_obs_excluded_indices,
        name="normalize_obs_excluded_indices",
    )
    if cfg.normalize_obs_excluded_indices and not cfg.normalize_obs:
        raise ValueError("normalize_obs_excluded_indices require normalize_obs=True")
    warm_start_artifacts = _prepare_warm_start(cfg)

    os.makedirs(cfg.log_dir, exist_ok=True)
    write_run_config(cfg, cfg.log_dir)
    start_time = time.monotonic()

    # ``check_env`` replays several reset/step cycles; running it on every
    # worker repeats the same verdict. Training and evaluation factories may
    # intentionally differ in reset distribution, though, so check the first
    # instance from each profile independently.
    train_env_checked = False
    eval_env_checked = False
    resolved_eval_env_fn = cfg.eval_env_fn or cfg.env_fn

    def checked_train_env_fn():
        nonlocal train_env_checked
        env = cfg.env_fn()
        if not train_env_checked:
            check_env(env)
            train_env_checked = True
        return env

    def checked_eval_env_fn():
        nonlocal eval_env_checked
        env = resolved_eval_env_fn()
        if not eval_env_checked:
            check_env(env)
            eval_env_checked = True
        return env

    # Helper envs (eval, info-eval, video) get seeds *past* the training
    # worker block: make_vec_env seeds worker i with ``seed + i``, so
    # offsets 1..n_envs-1 would replay a training worker's reset stream.
    eval_seed_offset = cfg.n_envs

    opened_envs: list[VecEnv] = []
    try:
        train_env = make_vec_env(
            checked_train_env_fn,
            n_envs=cfg.n_envs,
            seed=cfg.seed,
            monitor_dir=artifact_path(cfg.log_dir, "monitor_dir"),
        )
        opened_envs.append(train_env)
        eval_env = make_vec_env(
            checked_eval_env_fn,
            n_envs=1,
            seed=_offset_seed(cfg.seed, eval_seed_offset),
        )
        opened_envs.append(eval_env)

        # PPO benefits from return normalization; SAC's auto-tuned alpha
        # interacts poorly with it. ``normalize_reward=None`` picks the
        # per-algo default; True/False forces the choice.
        norm_reward = _resolved_norm_reward(cfg)
        use_vec_normalize = cfg.normalize_obs or norm_reward
        if use_vec_normalize:
            if warm_start_artifacts is not None:
                train_env = _load_warm_start_normalizer(
                    train_env,
                    warm_start_artifacts,
                    cfg,
                    norm_reward=norm_reward,
                )
            else:
                train_env = SelectiveVecNormalize(
                    train_env,
                    norm_obs=cfg.normalize_obs,
                    norm_reward=norm_reward,
                    clip_obs=cfg.clip_obs,
                    clip_reward=cfg.clip_reward,
                    normalize_obs_excluded_indices=(cfg.normalize_obs_excluded_indices),
                )
            # Eval envs always read the env's true reward (norm_reward=False)
            # and freeze the running stats (training=False); SB3's EvalCallback
            # syncs obs_rms/ret_rms from train_env before each eval.
            eval_env = SelectiveVecNormalize(
                eval_env,
                norm_obs=cfg.normalize_obs,
                norm_reward=False,
                clip_obs=cfg.clip_obs,
                training=False,
                normalize_obs_excluded_indices=(cfg.normalize_obs_excluded_indices),
            )

        def _calls(env_steps: int) -> int:
            return _env_steps_to_calls(env_steps, cfg.n_envs)

        # When a headline task metric is declared (and the info-dict
        # evaluator is on), InfoDictEvalCallback owns best-model
        # selection and early stopping; EvalCallback is demoted to
        # reward reporting (evaluations.npz + TensorBoard). Otherwise
        # the historical reward-based selection stands. Rationale: run
        # 20260712_190054's reward-selected "best" model completed zero
        # rallies -- reward and the task metric can diverge, and the
        # artifacts must follow the task metric when the recipe names
        # one.
        headline_selection = bool(cfg.info_dict_eval and cfg.headline_key)

        reward_eval_episodes = cfg.n_eval_episodes
        if cfg.reward_eval_episodes is not None:
            if (
                isinstance(cfg.reward_eval_episodes, bool)
                or not isinstance(cfg.reward_eval_episodes, int)
                or cfg.reward_eval_episodes < 1
            ):
                raise ValueError(
                    "reward_eval_episodes must be a positive integer"
                )
            if not headline_selection:
                raise ValueError(
                    "reward_eval_episodes requires headline-metric "
                    "selection (info_dict_eval + headline_key): without "
                    "it the reward eval stream owns best-model selection "
                    "and must keep the full n_eval_episodes"
                )
            reward_eval_episodes = cfg.reward_eval_episodes

        if cfg.final_eval_episodes is not None:
            if (
                isinstance(cfg.final_eval_episodes, bool)
                or not isinstance(cfg.final_eval_episodes, int)
                or cfg.final_eval_episodes < 1
            ):
                raise ValueError(
                    "final_eval_episodes must be a positive integer"
                )
            if not (cfg.info_dict_eval and cfg.final_info_eval):
                raise ValueError(
                    "final_eval_episodes requires info_dict_eval and "
                    "final_info_eval: it sizes the final-config eval "
                    "stream, which only exists when both are on"
                )

        # The reward EvalCallback and the final-config info-eval stream
        # roll the SAME distribution (the recipe's eval_env_overrides --
        # for a curriculum, the ladder's final stage): the gate re-syncs
        # only the matched evaluator. Under headline selection the reward
        # stream is reporting-only, and InfoDictEvalCallback already
        # collects per-episode returns for its reward tie-break, so it is
        # a strict superset. Retire the duplicate pass and hand it
        # evaluations.npz -- one env and one rollout fewer per eval, and
        # the goal-task curve stops being a 5-episode estimate.
        merge_reward_eval_into_final = bool(
            headline_selection and cfg.info_dict_eval and cfg.final_info_eval
        )
        final_eval_episodes = cfg.final_eval_episodes
        if final_eval_episodes is None:
            if merge_reward_eval_into_final:
                # One stream now scores the goal task, so give it the full
                # episode budget rather than the // 2 reporting sample the
                # split streams used. The old reward stream's 5 episodes
                # had a standard error of ~0.8-1.1 bounces at run
                # 20260721_004722's per-episode spread -- it could not
                # resolve the 0.30 -> 0.98 -> 1.76 transfer curve at all.
                # Costs ~10 episodes per evaluation over the two streams
                # it replaces, for a 6x larger sample on the campaign's
                # actual target metric.
                final_eval_episodes = cfg.n_eval_episodes
            else:
                # Two streams still split the work: keep the historical
                # reporting-sample size so unmerged runs are unchanged.
                final_eval_episodes = max(1, cfg.n_eval_episodes // 2)

        on_new_best: BaseCallback | None = None
        if use_vec_normalize and not headline_selection:
            on_new_best = _SaveVecNormalizeOnNewBest(
                artifact_path(cfg.log_dir, "best_vec_normalize")
            )

        after_eval: BaseCallback | None = None
        if (
            not headline_selection
            and cfg.early_stop_patience is not None
            and cfg.early_stop_patience > 0
        ):
            # min_evals = patience gives the run a warm-up of the same
            # length as the patience window, so at least 2N evaluations
            # happen before the stop can fire. verbose=1 so the stop
            # reason is visible in the Colab cell output.
            after_eval = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=cfg.early_stop_patience,
                min_evals=cfg.early_stop_patience,
                verbose=1,
            )

        # SB3's EvalCallback/InfoDictEvalCallback join ``best_model.zip``
        # onto their save directory themselves, so both are pointed at the
        # layout's model/ folder; EvalCallback likewise writes
        # ``evaluations.npz`` inside ``log_path``, i.e. metrics/.
        best_model_dir = os.path.dirname(artifact_path(cfg.log_dir, "best_model"))
        eval_callback: EvalCallback | None = None
        if not merge_reward_eval_into_final:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=(
                    None if headline_selection else best_model_dir
                ),
                log_path=os.path.dirname(
                    artifact_path(cfg.log_dir, "evaluations")
                ),
                render=False,
                deterministic=True,
                n_eval_episodes=reward_eval_episodes,
                eval_freq=_calls(cfg.eval_freq),
                callback_on_new_best=on_new_best,
                callback_after_eval=after_eval,
            )

        callbacks: list[BaseCallback] = (
            [] if eval_callback is None else [eval_callback]
        )
        # Bound outside the gated-construction branch so the interrupt
        # salvage path can finalize the gate's stage history (or skip
        # cleanly for non-gated runs).
        gate_callback: PerformanceGatedEnvStagesCallback | None = None
        if cfg.checkpoint_freq > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=_calls(cfg.checkpoint_freq),
                    save_path=artifact_path(cfg.log_dir, "checkpoints_dir"),
                    name_prefix=cfg.name_prefix,
                    save_vecnormalize=use_vec_normalize,
                )
            )
        if cfg.record_video and cfg.video_freq > 0:
            callbacks.append(
                VideoRecordCallback(
                    env_fn=resolved_eval_env_fn,
                    save_path=artifact_path(cfg.log_dir, "videos_dir"),
                    video_length=cfg.video_length,
                    save_freq=_calls(cfg.video_freq),
                    name_prefix=cfg.name_prefix,
                    csv_header=cfg.csv_header,
                    info_row_fn=cfg.info_row_fn,
                    seed=_offset_seed(cfg.seed, eval_seed_offset + 2),
                )
            )
        info_eval_env = None
        if cfg.info_dict_eval:
            info_eval_env = make_vec_env(
                checked_eval_env_fn,
                n_envs=1,
                seed=_offset_seed(cfg.seed, eval_seed_offset + 1),
            )
            opened_envs.append(info_eval_env)
            if use_vec_normalize:
                info_eval_env = SelectiveVecNormalize(
                    info_eval_env,
                    norm_obs=cfg.normalize_obs,
                    norm_reward=False,
                    clip_obs=cfg.clip_obs,
                    training=False,
                    normalize_obs_excluded_indices=(cfg.normalize_obs_excluded_indices),
                )
            if headline_selection:
                # Selection duty: full episode count (a //4 sample
                # quantizes bounce_count_ep_mean into sevenths at the
                # default 30 episodes -- too noisy to pick checkpoints
                # by), plus the lexicographic best score with reward
                # as the last tie-break unless the recipe overrides
                # the keys.
                info_eval_episodes = cfg.n_eval_episodes
                if cfg.best_metric_keys:
                    best_metric_keys = tuple(cfg.best_metric_keys)
                else:
                    best_metric_keys = (
                        f"{cfg.headline_key}_ep_mean",
                        *(("success_rate",) if cfg.success_key else ()),
                        "episode_reward_mean",
                    )
                selection_kwargs: dict[str, Any] = {
                    "best_metric_keys": best_metric_keys,
                    "best_model_save_path": best_model_dir,
                    "best_metric_min_delta": cfg.best_metric_min_delta,
                    "confirm_best": cfg.confirm_best_eval,
                }
                if cfg.early_stop_degenerate_evals > 0:
                    selection_kwargs["degenerate_stop_evals"] = (
                        cfg.early_stop_degenerate_evals
                    )
                    selection_kwargs["degenerate_guard_keys"] = tuple(
                        cfg.degenerate_guard_keys
                    )
                    degenerate_min_evals = cfg.degenerate_min_evals
                    if (
                        degenerate_min_evals is None
                        and env_attr_schedule_callbacks
                    ):
                        # While a reset curriculum still holds its start
                        # distribution, a dead full-difficulty eval is
                        # expected by design -- arm the guard only once
                        # every schedule has begun tapering.
                        degenerate_min_evals = math.ceil(
                            max(
                                callback.hold_until_timesteps
                                for callback in env_attr_schedule_callbacks
                            )
                            / cfg.eval_freq
                        )
                    selection_kwargs["degenerate_min_evals"] = (
                        degenerate_min_evals or 0
                    )
                if (
                    cfg.early_stop_patience is not None
                    and cfg.early_stop_patience > 0
                ):
                    selection_kwargs["early_stop_patience"] = (
                        cfg.early_stop_patience
                    )
                    min_evals = cfg.early_stop_patience
                    if env_attr_schedule_callbacks:
                        # Guarantee a full patience window of training
                        # at the final scheduled distribution: without
                        # this, a schedule ending at the earliest
                        # reachable stop gets zero post-taper training
                        # (run 20260714_211111's design flaw).
                        schedule_end = max(
                            callback.end_timesteps
                            for callback in env_attr_schedule_callbacks
                        )
                        min_evals = max(
                            min_evals,
                            math.ceil(schedule_end / cfg.eval_freq),
                        )
                    selection_kwargs["early_stop_min_evals"] = min_evals
            else:
                # Reporting only: a quarter of the eval budget keeps
                # the extra rollout pass cheap.
                info_eval_episodes = max(1, cfg.n_eval_episodes // 4)
                selection_kwargs = {}
            info_eval_callback = InfoDictEvalCallback(
                eval_env=info_eval_env,
                n_eval_episodes=info_eval_episodes,
                eval_freq=_calls(cfg.eval_freq),
                phase_key=cfg.phase_key,
                phase_labels=cfg.phase_labels,
                success_key=cfg.success_key,
                success_threshold=cfg.success_threshold,
                info_keys=cfg.info_eval_keys,
                terminal_info_keys=cfg.info_eval_terminal_keys,
                episode_distribution_keys=cfg.info_eval_distribution_keys,
                episode_survival_thresholds=(
                    cfg.info_eval_survival_thresholds
                ),
                csv_path=artifact_path(cfg.log_dir, "eval_info_csv"),
                **selection_kwargs,
            )
            callbacks.append(info_eval_callback)
            if cfg.performance_gate is not None:
                # Ordered after the info-eval callback so a trigger sees
                # that same trigger's fresh metrics.
                gate_spec = dict(cfg.performance_gate)
                unknown_gate_keys = sorted(
                    set(gate_spec)
                    - {
                        "stages",
                        "metric_key",
                        "threshold",
                        "sustain_evals",
                        "promotion_rule",
                        "advance_update_pause_steps",
                        "clear_replay_buffer_on_advance",
                        "reset_entropy_on_advance",
                        "entropy_reset_value",
                        "pool_confirmation_samples",
                    }
                )
                if unknown_gate_keys:
                    # A typo'd gate key was previously a silent no-op --
                    # exactly the failure class this repo bans.
                    raise ValueError(
                        f"unknown performance_gate key(s) "
                        f"{unknown_gate_keys}"
                    )
                gate_callback = PerformanceGatedEnvStagesCallback(
                    stages=gate_spec["stages"],
                    metric_key=gate_spec["metric_key"],
                    threshold=gate_spec["threshold"],
                    sustain_evals=gate_spec["sustain_evals"],
                    promotion_rule=gate_spec.get(
                        "promotion_rule", "consecutive"
                    ),
                    advance_update_pause_steps=gate_spec.get(
                        "advance_update_pause_steps", 0
                    ),
                    clear_replay_buffer_on_advance=gate_spec.get(
                        "clear_replay_buffer_on_advance", False
                    ),
                    reset_entropy_on_advance=gate_spec.get(
                        "reset_entropy_on_advance", False
                    ),
                    entropy_reset_value=gate_spec.get(
                        "entropy_reset_value"
                    ),
                    pool_confirmation_samples=gate_spec.get(
                        "pool_confirmation_samples", False
                    ),
                    info_eval=info_eval_callback,
                    # Per-stage champion archive + history report:
                    # without these, every advance's selection reset
                    # lets the next stage's first eval overwrite the
                    # departing stage's best_model.zip. config.json is
                    # copied alongside each archived triple so a stage
                    # dir is a valid warm-start source.
                    stage_bests_dir=artifact_path(
                        cfg.log_dir, "stage_bests_dir"
                    ),
                    stage_history_path=artifact_path(
                        cfg.log_dir, "curriculum_stages"
                    ),
                    run_config_path=artifact_path(cfg.log_dir, "config"),
                    verbose=cfg.verbose,
                )
                callbacks.append(gate_callback)
            if cfg.final_info_eval:
                # Honest final-task metrics on the recipe's unmodified
                # eval configuration; deliberately NOT registered with
                # the performance gate.
                final_info_env = make_vec_env(
                    checked_eval_env_fn,
                    n_envs=1,
                    seed=_offset_seed(cfg.seed, eval_seed_offset + 3),
                )
                opened_envs.append(final_info_env)
                if use_vec_normalize:
                    final_info_env = SelectiveVecNormalize(
                        final_info_env,
                        norm_obs=cfg.normalize_obs,
                        norm_reward=False,
                        clip_obs=cfg.clip_obs,
                        training=False,
                        normalize_obs_excluded_indices=(
                            cfg.normalize_obs_excluded_indices
                        ),
                    )
                callbacks.append(
                    InfoDictEvalCallback(
                        eval_env=final_info_env,
                        n_eval_episodes=final_eval_episodes,
                        eval_freq=_calls(cfg.eval_freq),
                        log_prefix="eval_info_final",
                        # Owns evaluations.npz when the reward evaluator
                        # was retired into this stream.
                        evaluations_npz_path=(
                            artifact_path(cfg.log_dir, "evaluations")
                            if merge_reward_eval_into_final
                            else None
                        ),
                        phase_key=cfg.phase_key,
                        phase_labels=cfg.phase_labels,
                        success_key=cfg.success_key,
                        success_threshold=cfg.success_threshold,
                        info_keys=cfg.info_eval_keys,
                        terminal_info_keys=cfg.info_eval_terminal_keys,
                        episode_distribution_keys=(
                            cfg.info_eval_distribution_keys
                        ),
                        episode_survival_thresholds=(
                            cfg.info_eval_survival_thresholds
                        ),
                        csv_path=artifact_path(
                            cfg.log_dir, "eval_info_final_csv"
                        ),
                    )
                )
        elif cfg.performance_gate is not None or cfg.final_info_eval:
            raise ValueError(
                "performance_gate and final_info_eval require info_dict_eval"
            )
        callbacks.extend(env_attr_schedule_callbacks)
        callbacks.extend(cfg.extra_callbacks)

        # cfg.seed / cfg.verbose are the first-class knobs; an explicit value in
        # model_kwargs still wins (setdefault), and routing both through
        # model_kwargs avoids a duplicate-keyword TypeError at construction.
        model_kwargs = dict(cfg.model_kwargs)
        if cfg.seed is not None:
            model_kwargs.setdefault("seed", cfg.seed)
        model_kwargs.setdefault("verbose", cfg.verbose)
        effective_verbose = model_kwargs["verbose"]

        # Load the source before constructing the target. SB3 restores the saved
        # source seed while loading; constructing the fresh target afterwards
        # re-establishes cfg.seed for the new run's stochastic sequence.
        source_model: BaseAlgorithm | None = None
        if warm_start_artifacts is not None:
            source_model = _resolve_algo(cfg.algo).load(
                str(warm_start_artifacts.model_path),
                device="cpu",
            )
            check_for_correct_spaces(
                train_env,
                source_model.observation_space,
                source_model.action_space,
            )

        model = _build_algo(
            cfg.algo,
            train_env,
            cfg.log_dir,
            policy=cfg.policy,
            **model_kwargs,
        )
        transferred_log_ent_coef: float | None = None
        if source_model is not None:
            if type(source_model.policy) is not type(model.policy):
                raise ValueError("source and target policy classes differ")
            # PPO policies own a single optimizer; SAC splits actor and
            # critic. Either way the fresh target's optimizers must be
            # stateless before the weight transfer -- optimizer moments
            # are deliberately NOT carried over. (``Any``-typed access:
            # nn.Module.__getattr__ erases the submodule types.)
            policy_any: Any = model.policy
            fresh_optimizers = (
                (policy_any.actor.optimizer, policy_any.critic.optimizer)
                if hasattr(model.policy, "actor")
                else (policy_any.optimizer,)
            )
            if any(optimizer.state for optimizer in fresh_optimizers):
                raise RuntimeError(
                    "fresh target optimizer unexpectedly has state"
                )
            # For SAC, policy.state_dict() covers actor, critic, AND the
            # critic target networks, so the transfer resumes from a
            # self-consistent TD state rather than fresh random targets.
            model.policy.load_state_dict(source_model.policy.state_dict(), strict=True)
            # SAC's auto-tuned entropy temperature lives on the algorithm,
            # not the policy. A fresh "auto" restarts at ent_coef=1.0 --
            # a huge entropy bonus that would churn a converged policy
            # (run 20260721_004722's coefficient sat at ~0.0009) -- so a
            # matching auto source hands its temperature over. Mixing
            # auto and fixed entropy across the boundary is a config
            # contradiction; fail rather than guess.
            source_log_ent = getattr(source_model, "log_ent_coef", None)
            target_log_ent = getattr(model, "log_ent_coef", None)
            if (source_log_ent is None) != (target_log_ent is None):
                raise ValueError(
                    "warm-start source and target entropy configurations "
                    "differ (auto-tuned vs fixed ent_coef)"
                )
            if source_log_ent is not None and target_log_ent is not None:
                target_log_ent.data.copy_(
                    source_log_ent.data.to(target_log_ent.device)
                )
                transferred_log_ent_coef = float(
                    target_log_ent.detach().exp().item()
                )

            assert warm_start_artifacts is not None
            assert isinstance(train_env, VecNormalize)
            source_env = warm_start_artifacts.source_config["env"]
            source_curriculum = source_env.get("curriculum")
            assert not isinstance(train_env.obs_rms, dict)
            initialization = {
                "mode": "policy_and_observation_stats",
                "source_run_dir": str(warm_start_artifacts.source_run_dir),
                "source_artifacts": {
                    name: {
                        # The located path: model/ for new-layout sources,
                        # the flat root for legacy runs.
                        "path": str(path),
                        "sha256": warm_start_artifacts.source_hashes[name],
                    }
                    for name, path in (
                        ("best_model.zip", warm_start_artifacts.model_path),
                        (
                            "best_vec_normalize.pkl",
                            warm_start_artifacts.normalizer_path,
                        ),
                        ("config.json", warm_start_artifacts.config_path),
                    )
                },
                "source": {
                    "algo": cfg.algo.upper(),
                    "environment_class": source_env.get("class"),
                    "curriculum": source_curriculum,
                    "model_num_timesteps": int(source_model.num_timesteps),
                    "policy_class": type(source_model.policy).__name__,
                },
                "target": {
                    "algo": cfg.algo.upper(),
                    "environment_class": warm_start_artifacts.target_env_class,
                    "curriculum": warm_start_artifacts.target_curriculum,
                    "policy_class": type(model.policy).__name__,
                },
                "transferred": [
                    "policy.state_dict",
                    "vec_normalize.obs_rms",
                    *(
                        ["log_ent_coef"]
                        if transferred_log_ent_coef is not None
                        else []
                    ),
                ],
                "reset": [
                    "policy.optimizer_state",
                    "learning_rate_and_progress_schedule",
                    "num_timesteps",
                    (
                        "replay_buffer"
                        if cfg.algo.upper() in _OFF_POLICY_ALGOS
                        else "rollout_buffer"
                    ),
                    "vec_normalize.ret_rms",
                    "vec_normalize.returns",
                    "logger_and_callback_state",
                ],
                **(
                    {"transferred_ent_coef": transferred_log_ent_coef}
                    if transferred_log_ent_coef is not None
                    else {}
                ),
                "normalize_obs_excluded_indices": list(
                    cfg.normalize_obs_excluded_indices
                ),
                "reset_observation_indices": list(
                    cfg.warm_start.reset_observation_indices
                    if cfg.warm_start is not None
                    else ()
                ),
                "reset_observation_values": list(
                    warm_start_artifacts.reset_observation_values
                ),
                "observation_rms_count": float(train_env.obs_rms.count),
                "reward_statistics_reset": True,
                "optimizer_state_transferred": False,
            }
            update_run_config_with_initialization(initialization, cfg.log_dir)
            del source_model

        # Route SB3's own diagnostics (SAC ent_coef/actor/critic losses, PPO
        # explained_variance/approx_kl, ...) to a CSV alongside TensorBoard so
        # the run directory is self-diagnosing after the Colab runtime is gone.
        # ``progress.csv`` is read back by stage_summary + plot_training_health;
        # it lives directly in metrics/ (pandas-readable metrics, not TB event
        # data) while the event files go to metrics/tensorboard, so the two
        # formats get their own directories instead of SB3's shared folder.
        # set_logger marks the logger custom, so SB3's learn() leaves it intact
        # instead of resetting to its default (TensorBoard-only) configuration.
        tensorboard_dir = artifact_path(cfg.log_dir, "tensorboard_dir")
        output_formats = [
            make_output_format(
                "csv",
                os.path.dirname(artifact_path(cfg.log_dir, "progress_csv")),
            ),
            make_output_format("tensorboard", tensorboard_dir),
        ]
        if effective_verbose:
            output_formats.append(make_output_format("stdout", tensorboard_dir))
        model.set_logger(
            Logger(folder=tensorboard_dir, output_formats=output_formats)
        )
        update_run_config_with_model(model, cfg.log_dir)

        interrupted = False
        try:
            try:
                model.learn(
                    total_timesteps=cfg.total_timesteps,
                    callback=CallbackList(callbacks),
                    reset_num_timesteps=True,
                    progress_bar=False,
                )
            except KeyboardInterrupt:
                # A stopped Colab cell / ^C shouldn't vaporize hours of
                # training. Salvage the run: fall through to save the model,
                # normalization stats, final eval, and summary (marked
                # "interrupted") so the run dir stays self-describing.
                interrupted = True
                print(
                    "Training interrupted -- saving final_model and summary "
                    "for the partial run."
                )
                # SB3 only calls on_training_end after a clean training
                # loop, so an interrupt would otherwise lose the gate's
                # in-flight stage row and final-stage archive; finalize
                # is idempotent, so a duplicate close cannot occur.
                if gate_callback is not None:
                    gate_callback.finalize()
            model.save(artifact_path(cfg.log_dir, "final_model"))
            if use_vec_normalize:
                assert isinstance(train_env, VecNormalize)
                train_env.save(artifact_path(cfg.log_dir, "vec_normalize"))
                # EvalCallback last synced eval_env's running stats up to
                # eval_freq steps ago; re-sync so the final eval normalizes
                # observations with the end-of-training statistics.
                sync_envs_normalization(train_env, eval_env)

            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=cfg.n_eval_episodes
            )
            mean_reward_value = (
                sum(mean_reward) / len(mean_reward)
                if isinstance(mean_reward, list)
                else float(mean_reward)
            )
            std_reward_value = (
                sum(std_reward) / len(std_reward)
                if isinstance(std_reward, list)
                else float(std_reward)
            )
            print(f"Mean reward: {mean_reward_value:.2f} +/- {std_reward_value:.2f}")
            write_run_summary(
                cfg,
                cfg.log_dir,
                final_mean_reward=mean_reward_value,
                final_std_reward=std_reward_value,
                duration_seconds=time.monotonic() - start_time,
                device=str(getattr(model, "device", "")) or None,
                status="interrupted" if interrupted else "completed",
                actual_timesteps=int(model.num_timesteps),
            )
        finally:
            train_env.close()
            eval_env.close()
            if info_eval_env is not None:
                info_eval_env.close()
            opened_envs.clear()

        return model
    finally:
        for opened_env in reversed(opened_envs):
            opened_env.close()
