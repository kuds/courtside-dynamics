"""Curated env+algo recipes for the consolidated training notebook.

Each :class:`Recipe` knows everything about an environment that the
notebook would otherwise have to spell out: the env class, its
constructor kwargs, the per-env extras that go into ``TrainConfig``
(custom CSV rows, phase labels, ...), and a sane default budget for
training. The notebook only has to pick a name (``"WallBall"``) and
an algorithm (``"SAC"``) and call :func:`build_train_config`.

The notebook picks a recipe name and may override its algorithm explicitly;
otherwise each recipe's default algorithm and worker count survive. Adding a
new env is one entry in :data:`RECIPES`; the notebook needs no edits.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from courtside_dynamics.envs import (
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    BallBalanceEnv,
    BallBounceEnv,
    HumanoidTennisCoopEnv,
    WallBallEnv,
)
from courtside_dynamics.training import TrainConfig


@dataclass(frozen=True)
class Recipe:
    """Static description of an env+training setup."""

    env_cls: type
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    default_total_timesteps: int = 1_000_000
    default_algo: str = "SAC"
    name_prefix: str = "rl_model"
    extra_cfg: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # Kept after the original dataclass fields for positional compatibility.
    # Applied on top of ``env_kwargs`` only for evaluation, final scoring,
    # videos, and post-training endurance audits. Curriculum recipes use this
    # to force canonical full-episode resets without changing training.
    eval_env_overrides: dict[str, Any] = field(default_factory=dict)


def _ball_bounce_info_row(
    info: dict, reward: float, total_reward: float, done: bool
) -> Sequence[object]:
    return [
        info["ball_velocity"],
        info["ball_accelerometer"],
        info["ball_to_paddle"],
        info["bounce_count"],
        info["touch_sensor"],
        reward,
        total_reward,
        done,
    ]


_BALL_BOUNCE_CSV_HEADER = [
    "ball_velocity_x", "ball_velocity_y", "ball_velocity_z",
    "ball_accelerometer_x", "ball_accelerometer_y", "ball_accelerometer_z",
    "from_to_x1", "from_to_y1", "from_to_z1",
    "from_to_x2", "from_to_y2", "from_to_z2",
    "bounce_count", "touch_sensor", "reward", "total_reward", "done",
]


# Phase 3 exposes a compact recording schema. The full tennis ``info`` dict
# intentionally retains detailed rule/contact traces for correctness audits;
# flattening every scalar would create more than a hundred CSV columns.
_HUMANOID_TENNIS_CSV_KEYS = (
    "episode_step",
    "serve_side",
    "initial_ball_x",
    "initial_ball_y",
    "initial_ball_z",
    "initial_ball_vx",
    "initial_ball_vy",
    "initial_ball_vz",
    "ball_side",
    "expected_returner",
    "rally_phase",
    "rally_count",
    "legal_hit_count",
    "bounce_count",
    "net_crossing_count",
    "valid_return_rate",
    "event_valid_racket_hit",
    "event_valid_return",
    "event_first_bounce",
    "termination_reason",
    "termination_reason_name",
    "term_timeout",
    "rew_valid_return",
    "rew_shaping",
    "rew_shaping_clawback",
    "rew_fault",
)
_HUMANOID_TENNIS_CSV_HEADER = [
    *_HUMANOID_TENNIS_CSV_KEYS,
    "reward", "total_reward", "done",
]

_HUMANOID_TENNIS_CURRICULUM_CSV_KEYS = (
    "curriculum_stage",
    "curriculum_stage_name",
    "curriculum_objective",
    "active_action_count",
    "active_action_count_a",
    "active_action_count_b",
    "stage_success",
    "stage_attempt_complete",
    "target_hit",
    "target_miss",
    "term_stage_success",
    "term_target_miss",
)
_HUMANOID_TENNIS_CURRICULUM_CSV_HEADER = [
    *_HUMANOID_TENNIS_CSV_KEYS,
    *_HUMANOID_TENNIS_CURRICULUM_CSV_KEYS,
    "reward", "total_reward", "done",
]


def _humanoid_tennis_info_row(
    info: dict, reward: float, total_reward: float, done: bool
) -> Sequence[object]:
    return [
        *(info[key] for key in _HUMANOID_TENNIS_CSV_KEYS),
        reward,
        total_reward,
        done,
    ]


def _humanoid_tennis_curriculum_info_row(
    info: dict, reward: float, total_reward: float, done: bool
) -> Sequence[object]:
    return [
        *(info[key] for key in _HUMANOID_TENNIS_CSV_KEYS),
        *(info[key] for key in _HUMANOID_TENNIS_CURRICULUM_CSV_KEYS),
        reward,
        total_reward,
        done,
    ]


_HUMANOID_TENNIS_EVAL_KEYS = (
    "rally_count",
    "valid_return_rate",
    "legal_hit_count",
)
_HUMANOID_TENNIS_TERMINAL_EVAL_KEYS = (
    "duplicate_contact_suppressed_count",
    "duplicate_event_suppressed_count",
    "stale_event_suppressed_count",
    # The episode means of these mutually exclusive flags form the fault
    # distribution and remain comparable across future curriculum stages.
    "term_unsafe",
    "term_humanoid_net",
    "term_racket_net",
    "term_ball_humanoid_a",
    "term_ball_humanoid_b",
    "term_ball_net",
    "term_out_of_bounds",
    "term_second_bounce",
    "term_failed_to_cross",
    "term_reverse_crossing",
    "term_wrong_hitter",
    "term_double_hit",
    "term_premature_hit",
    "term_simultaneous_racket_contact",
    "term_timeout",
)

_HUMANOID_TENNIS_CURRICULUM_TERMINAL_EVAL_KEYS = (
    *_HUMANOID_TENNIS_TERMINAL_EVAL_KEYS,
    "stage_success",
    "target_hit",
    "target_miss",
    "term_stage_success",
    "term_target_miss",
)

# VecNormalize should learn scales for continuous physical state only. The
# discrete/bounded rally state, contact memory, and active-action mask must stay
# raw so newly reachable curriculum bits do not inherit near-zero variance and
# clip at a stage boundary.
_HUMANOID_TENNIS_NORMALIZATION_EXCLUSIONS = tuple(
    range(
        HUMANOID_TENNIS_OBSERVATION_LAYOUT.rally_state.start,
        HUMANOID_TENNIS_OBSERVATION_LAYOUT.total_size,
    )
)


def _tennis_curriculum_extra_cfg(*, video_length: int) -> dict[str, Any]:
    """Shared fixed-stage recording/evaluation contract for Stages 0–2."""
    return {
        "n_envs": 1,
        "normalize_obs_excluded_indices": (
            _HUMANOID_TENNIS_NORMALIZATION_EXCLUSIONS
        ),
        "eval_freq": 25_000,
        "checkpoint_freq": 100_000,
        "video_freq": 100_000,
        "n_eval_episodes": 20,
        "video_length": video_length,
        "csv_header": _HUMANOID_TENNIS_CURRICULUM_CSV_HEADER,
        "info_row_fn": _humanoid_tennis_curriculum_info_row,
        "info_eval_keys": (
            "stage_success",
            "target_hit",
            "rally_count",
            "valid_return_rate",
            "legal_hit_count",
        ),
        "info_eval_terminal_keys": (
            _HUMANOID_TENNIS_CURRICULUM_TERMINAL_EVAL_KEYS
        ),
        "info_eval_distribution_keys": ("rally_count",),
        "success_key": "stage_success",
        "success_threshold": 1.0,
        "phase_key": "rally_phase",
        "phase_labels": {
            0: "initial_feed",
            1: "awaiting_return",
            2: "return_in_flight",
            3: "terminal",
        },
    }


RECIPES: dict[str, Recipe] = {
    "BallBalance": Recipe(
        env_cls=BallBalanceEnv,
        env_kwargs={"render_mode": "rgb_array"},
        default_total_timesteps=1_000_000,
        name_prefix="ball_balance",
        description="Keep a ball on a 6-DOF tray.",
    ),
    "BallBounce": Recipe(
        env_cls=BallBounceEnv,
        env_kwargs={"render_mode": "rgb_array", "min_force": 100.0},
        default_total_timesteps=1_500_000,
        name_prefix="ball_bounce",
        extra_cfg={
            "csv_header": _BALL_BOUNCE_CSV_HEADER,
            "info_row_fn": _ball_bounce_info_row,
            # A single passive contact is not sustained juggling. The
            # environment counts only deliberate top-face rebounds, and the
            # recipe requires ten within one episode before reporting success.
            "success_key": "bounce_count",
            "success_threshold": 10.0,
            "headline_key": "bounce_count",
            "info_eval_keys": (
                "bounce_count",
                "contact_episode_count",
                "touch_sensor",
                "valid_bounce",
            ),
            "info_eval_terminal_keys": (
                "term_ball_dropped",
                "term_nonfinite",
                "term_timeout",
            ),
            "info_eval_distribution_keys": ("bounce_count",),
        },
        description=(
            "Deliberately juggle a ball from the top face of a 6-DOF paddle."
        ),
    ),
    "WallBall": Recipe(
        env_cls=WallBallEnv,
        # serve_vy_max widened from the 1.8 default: rally forensics on
        # the 20260712 best model showed a skill cliff at exchange 3
        # (survival 100%/100%/37%/7%/0%) because angled receive states
        # only ever arise from the agent's own imprecise returns --
        # off-distribution until it is too late. Serving up to ~23
        # degrees off-axis (vy 2.6 vs vx ~6) puts those states in the
        # training distribution from the first step of every episode.
        env_kwargs={
            "render_mode": "rgb_array",
            "min_force": 20.0,
            "serve_vy_max": 2.6,
        },
        default_total_timesteps=1_500_000,
        name_prefix="wall_ball",
        extra_cfg={
            # An eval episode "succeeds" once it completes a full rally
            # cycle (a gated wall hit). Surfaces eval_info/success_rate.
            "success_key": "bounce_count",
            "success_threshold": 1.0,
            # Headline metric: rally exchanges per eval episode
            # (bounce_count_ep_mean). success_rate saturates at 1.0
            # once every episode completes a single exchange (the
            # 20260712 run hit that by 250k steps), and eval reward is
            # dominated by tracking shaping -- the exchange count is
            # the task-native measure of rally quality. Setting it also
            # hands best-model selection and early stopping to this
            # metric (see TrainConfig.headline_key): the 20260712 run's
            # reward-selected best_model.zip completed zero rallies.
            "headline_key": "bounce_count",
            # Episode min/median/p90/max of the exchange count, so the
            # eval CSV shows whether a mean of 1.0 is "every episode
            # rallies once" or "one episode rallied ten times".
            "info_eval_distribution_keys": ("bounce_count",),
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "await_bounce",
                1: "await_paddle",
                2: "await_wall",
            },
        },
        description=(
            "Rally a ball against a wall with a face-only paddle at a fixed "
            "10-degree upward pitch, three target-controlled DOFs, and a "
            "strict gated wall-hit reward."
        ),
    ),
    "WallBallVolley": Recipe(
        env_cls=WallBallEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "min_force": 20.0,
            "serve_vy_max": 2.6,
            "rally_style": "volley",
            # World-space values that exactly preserve WallBall's current
            # XML home pose and full x workspace. This preset changes the
            # rally rule, not the paddle geometry.
            "paddle_home_x": -1.7,
            "paddle_x_target_range": (-4.7, 0.3),
        },
        default_total_timesteps=1_500_000,
        name_prefix="wall_ball_volley",
        extra_cfg={
            "success_key": "bounce_count",
            "success_threshold": 1.0,
            "headline_key": "bounce_count",
            "info_eval_distribution_keys": ("bounce_count",),
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "await_bounce",
                1: "await_paddle",
                2: "await_wall",
            },
        },
        description=(
            "Volley against the wall without letting the ball touch the "
            "floor, using the full face-only paddle workspace."
        ),
    ),
    "WallBallBaseline": Recipe(
        env_cls=WallBallEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "min_force": 20.0,
            "rally_style": "one_bounce",
            # Calibrated in world space: the lower, slower serve bounces
            # around x=-1.72 before reaching a paddle kept near x=-2.7.
            # The measured bounce -> paddle -> wall sequence succeeded in
            # 500/500 scripted trials; a parked paddle scored 0/1,000.
            "paddle_home_x": -2.7,
            # A calibration sweep found -2.1 to be the safest modest forward
            # extension: -1.8 created more out-of-bounds and style faults.
            "paddle_x_target_range": (-3.2, -2.1),
            "serve_speed": 5.5,
            "serve_speed_jitter": 0.5,
            "serve_lob": 0.0,
            "serve_vy_min": 0.8,
            "serve_vy_max": 2.0,
            # Early training is 40% normal serves, 30% incoming-wall
            # fragments, and 30% post-bounce fragments. The global-step
            # schedule below tapers the combined recovery share to 15%.
            "recovery_reset_probability": 0.6,
            "post_bounce_reset_fraction": 0.5,
            # Reward the policy's outgoing shot only when its next bounce
            # projects inside the paddle's lateral recovery corridor.
            "recoverable_bounce_bonus": 0.25,
            "recoverable_bounce_lateral_limit": 2.0,
        },
        eval_env_overrides={
            # Evaluation must never sample recovery-curriculum starts: every
            # score represents a complete baseline episode from the serve.
            "recovery_reset_probability": 0.0,
        },
        default_total_timesteps=1_500_000,
        name_prefix="wall_ball_baseline",
        extra_cfg={
            "success_key": "bounce_count",
            # One return was solved by every held-out seed in the first
            # baseline run; success now means surviving the actual skill
            # cliff and completing a second return.
            "success_threshold": 2.0,
            "headline_key": "bounce_count",
            # Selection ignores raw eval reward: it was ~88% tracking
            # shaping in run 20260714_050506 and frozen at the -1.0
            # penalty in 20260714_211111, where a ~1e-8 reward
            # difference crowned the best model and reset the early-stop
            # patience. The survival rate at the success bar breaks ties
            # between equal exchange means instead.
            "best_metric_keys": (
                "bounce_count_ep_mean",
                "bounce_count_ep_ge_2_rate",
            ),
            # Half the granularity of a 30-episode mean/rate: a real
            # one-episode change (1/30) registers as an improvement,
            # float noise never does. Both selection keys above are
            # episode aggregates, so no continuous reward key is blunted
            # by the threshold.
            "best_metric_min_delta": 0.5 / 30,
            # A candidate best must hold on a second, independent eval
            # batch: run 20260714_050506's best checkpoint beat its
            # plateau by exactly one lucky 2-bounce episode in 30.
            "confirm_best_eval": True,
            # Kill a run whose eval signal is provably dead -- flat
            # selection score with zero paddle contact. Run
            # 20260714_211111 spent 750k steps at exactly -1.0 reward
            # with no ball contact from its second evaluation onward.
            "early_stop_degenerate_evals": 5,
            "degenerate_guard_keys": ("paddle_hit_count_ep_mean",),
            "info_eval_distribution_keys": ("bounce_count",),
            "info_eval_survival_thresholds": {
                "bounce_count": (2, 3, 5),
            },
            "env_attr_schedules": (
                {
                    "attr_name": "recovery_reset_probability",
                    "start_value": 0.6,
                    "end_value": 0.15,
                    "hold_until_timesteps": 250_000,
                    "end_timesteps": 750_000,
                },
            ),
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "await_bounce",
                1: "await_paddle",
                2: "await_wall",
            },
        },
        description=(
            "Play one-bounce baseline rallies from a rear paddle zone with "
            "tapered post-wall recovery practice and normal-serve evaluation."
        ),
    ),
    "HumanoidTennisStage0Intercept": Recipe(
        env_cls=HumanoidTennisCoopEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "episode_len": 150,
            "curriculum_config": 0,
        },
        default_total_timesteps=500_000,
        default_algo="PPO",
        name_prefix="humanoid_tennis_stage0_intercept",
        extra_cfg=_tennis_curriculum_extra_cfg(video_length=150),
        description=(
            "Experimental fixed-pelvis, two-shoulder physical intercept task. "
            "Success is the first legal racket contact; convergence is not "
            "claimed."
        ),
    ),
    "HumanoidTennisStage1AnchoredReturn": Recipe(
        env_cls=HumanoidTennisCoopEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "episode_len": 300,
            "curriculum_config": 1,
        },
        default_total_timesteps=1_000_000,
        default_algo="PPO",
        name_prefix="humanoid_tennis_stage1_anchored_return",
        extra_cfg=_tennis_curriculum_extra_cfg(video_length=300),
        description=(
            "Experimental fixed-pelvis right-arm return into a generous "
            "physical target region; convergence is not claimed."
        ),
    ),
    "HumanoidTennisStage2RandomizedReturn": Recipe(
        env_cls=HumanoidTennisCoopEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "episode_len": 300,
            "curriculum_config": 2,
        },
        default_total_timesteps=2_000_000,
        default_algo="PPO",
        name_prefix="humanoid_tennis_stage2_randomized_return",
        extra_cfg=_tennis_curriculum_extra_cfg(video_length=300),
        description=(
            "Experimental fixed-pelvis target return with bounded seeded "
            "launch randomization; convergence is not claimed."
        ),
    ),
    "HumanoidTennisCoopSmoke": Recipe(
        env_cls=HumanoidTennisCoopEnv,
        env_kwargs={"render_mode": "rgb_array", "episode_len": 250},
        # This short run exercises Gymnasium/SB3 integration, evaluation,
        # recording, and artifacts. Phase 4 owns constrained curriculum
        # presets; this is not a viable free-standing tennis training claim.
        default_total_timesteps=10_000,
        name_prefix="humanoid_tennis_coop_smoke",
        extra_cfg={
            "n_envs": 1,
            "normalize_obs_excluded_indices": (
                _HUMANOID_TENNIS_NORMALIZATION_EXCLUSIONS
            ),
            "eval_freq": 2_500,
            "checkpoint_freq": 5_000,
            "video_freq": 5_000,
            "n_eval_episodes": 4,
            "video_length": 250,
            "csv_header": _HUMANOID_TENNIS_CSV_HEADER,
            "info_row_fn": _humanoid_tennis_info_row,
            "info_eval_keys": _HUMANOID_TENNIS_EVAL_KEYS,
            "info_eval_terminal_keys": _HUMANOID_TENNIS_TERMINAL_EVAL_KEYS,
            "info_eval_distribution_keys": ("rally_count",),
            "success_key": "rally_target_reached",
            "success_threshold": 1.0,
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "initial_feed",
                1: "awaiting_return",
                2: "return_in_flight",
                3: "terminal",
            },
        },
        description=(
            "Pipeline and recording smoke run for the centralized two-G1 "
            "tennis environment; not a full-tennis training preset."
        ),
    ),
}


# Quick-test overrides applied on top of the recipe defaults so a notebook
# can verify the whole pipeline (callbacks, video, plotting) end-to-end in
# a couple of minutes instead of hours.
_QUICK_TEST_OVERRIDES: dict[str, Any] = {
    "total_timesteps": 25_000,
    "eval_freq": 5_000,
    "checkpoint_freq": 10_000,
    "video_freq": 10_000,
    "n_eval_episodes": 3,
    "video_length": 750,
}


def make_env_fn(
    env_name: str,
    *,
    env_overrides: Mapping[str, Any] | None = None,
):
    """Return a zero-arg env factory for ``env_name``.

    Useful for tools (replay video, custom evaluation) that need a fresh
    env outside of :func:`train`. ``env_overrides`` is copied into this
    factory only; it never mutates the recipe or a training config that was
    already built. This keeps post-training long-horizon evaluation isolated
    from the environment used to train and select the checkpoint.
    """
    recipe = RECIPES[env_name]
    kwargs = dict(recipe.env_kwargs)
    if env_overrides:
        kwargs.update(env_overrides)

    def _factory():
        return recipe.env_cls(**kwargs)

    return _factory


def make_eval_env_fn(
    env_name: str,
    *,
    env_overrides: Mapping[str, Any] | None = None,
):
    """Return the canonical evaluation factory for ``env_name``.

    The recipe's evaluation overrides are layered over its training kwargs,
    then one-off caller overrides (such as a longer episode horizon) are
    applied last. This keeps checkpoint selection and post-training audits on
    standard full episodes even when the training factory samples curriculum
    reset states.
    """
    recipe = RECIPES[env_name]
    kwargs = dict(recipe.env_kwargs)
    kwargs.update(recipe.eval_env_overrides)
    if env_overrides:
        kwargs.update(env_overrides)

    def _factory():
        return recipe.env_cls(**kwargs)

    return _factory


def build_train_config(
    env_name: str,
    *,
    algo: str | None = None,
    log_dir: str,
    total_timesteps: int | None = None,
    quick_test: bool = False,
    **overrides: Any,
) -> TrainConfig:
    """Materialize a :class:`TrainConfig` from a registered recipe.

    Parameters
    ----------
    env_name:
        Key into :data:`RECIPES` (e.g. ``"WallBall"``).
    algo:
        ``"SAC"`` or ``"PPO"``. When omitted, uses the recipe's default;
        the masked humanoid-tennis curricula default to PPO so SAC does not
        auto-tune entropy against 51–56 inactive action dimensions.
    log_dir:
        Directory for monitor logs, TensorBoard, checkpoints, videos.
    total_timesteps:
        Override the recipe's default training budget. Like every other
        explicit override, it wins over the ``quick_test`` presets.
    quick_test:
        Apply :data:`_QUICK_TEST_OVERRIDES` so the whole pipeline runs
        end-to-end in a couple of minutes -- handy for smoke-testing on
        a new Colab runtime.
    **overrides:
        Any other ``TrainConfig`` field (``eval_freq``, ``n_envs``,
        ``model_kwargs``, ...).
    """
    if env_name not in RECIPES:
        raise KeyError(
            f"Unknown env '{env_name}'. Choose one of {sorted(RECIPES)}."
        )
    recipe = RECIPES[env_name]
    resolved_algo = recipe.default_algo if algo is None else algo

    cfg_kwargs: dict[str, Any] = {
        "env_fn": make_env_fn(env_name),
        "eval_env_fn": make_eval_env_fn(env_name),
        "recipe_name": env_name,
        "algo": resolved_algo,
        "log_dir": log_dir,
        "name_prefix": f"{recipe.name_prefix}_{resolved_algo.lower()}",
        "total_timesteps": recipe.default_total_timesteps,
    }
    cfg_kwargs.update(recipe.extra_cfg)

    if quick_test:
        cfg_kwargs.update(_QUICK_TEST_OVERRIDES)

    # Explicit caller choices are applied last so they always win --
    # including over the quick-test presets. (``total_timesteps`` used to
    # be silently discarded under ``quick_test=True``, unlike every other
    # override, which made "quick test but a bit longer" impossible.)
    if total_timesteps is not None:
        cfg_kwargs["total_timesteps"] = total_timesteps
    cfg_kwargs.update(overrides)

    return TrainConfig(**cfg_kwargs)
