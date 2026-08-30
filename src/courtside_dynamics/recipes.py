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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from courtside_dynamics.envs import (
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    PADDLE_TENNIS_NORMALIZED_SLICE,
    PADDLE_TENNIS_OBSERVATION_NAMES,
    BallBalanceEnv,
    BallBounceEnv,
    HumanoidTennisCoopEnv,
    PaddleTennisEnv,
    TennisRewardConfig,
    WallBallEnv,
)
from courtside_dynamics.training import TrainConfig
from courtside_dynamics.training.algos import resolve_algo, validate_model_kwargs


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
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "ball_accelerometer_x",
    "ball_accelerometer_y",
    "ball_accelerometer_z",
    "from_to_x1",
    "from_to_y1",
    "from_to_z1",
    "from_to_x2",
    "from_to_y2",
    "from_to_z2",
    "bounce_count",
    "touch_sensor",
    "reward",
    "total_reward",
    "done",
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
    "reward",
    "total_reward",
    "done",
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
    "reward",
    "total_reward",
    "done",
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


# Every key exists in every step info dict (the env routes even forced
# nonfinite terminations through the rules machine's full schema).
_PADDLE_TENNIS_CSV_KEYS = (
    "serve_side_name",
    "serve_side_is_policy",
    "ball_side",
    "expected_returner",
    "rally_phase",
    "rally_count",
    "legal_hit_count",
    "bounce_count",
    "net_crossing_count",
    "crossings",
    "event_valid_racket_hit",
    "event_valid_return",
    "event_first_bounce",
    "termination_reason",
    "termination_reason_name",
    "term_timeout",
    "rew_return",
    "rew_fault",
    "rew_unsafe",
    "rew_shaping",
    "rew_shaping_clawback",
    "rew_reach",
    "rew_reach_clawback",
    "rew_hold",
    "rew_hold_clawback",
    "points_played",
    "completed_point_crossings",
    "point_serve_nudged",
)
_PADDLE_TENNIS_CSV_HEADER = [
    *_PADDLE_TENNIS_CSV_KEYS,
    "reward",
    "total_reward",
    "done",
]


def _paddle_tennis_info_row(
    info: dict, reward: float, total_reward: float, done: bool
) -> Sequence[object]:
    return [
        *(info[key] for key in _PADDLE_TENNIS_CSV_KEYS),
        reward,
        total_reward,
        done,
    ]


_PADDLE_TENNIS_TERMINAL_EVAL_KEYS = (
    "duplicate_contact_suppressed_count",
    "duplicate_event_suppressed_count",
    "stale_event_suppressed_count",
    # Mean = fraction of eval episodes where the policy served; with
    # strict alternation this should sit near 0.5 in every batch.
    "serve_side_is_policy",
    # The grouped fault distribution (mutually exclusive flags; the
    # env's term_* convention, one is 1.0 on each terminal step).
    "term_out_of_bounds",
    "term_ball_net",
    "term_second_bounce",
    "term_failed_to_cross",
    "term_illegal_hit",
    "term_net_touch",
    "term_volley",
    "term_nonfinite",
    "term_timeout",
    # n-point era counters (cumulative per-episode, so the terminal
    # read gives episode totals; under the frozen one-point default a
    # fault-ended episode records its single completed point and a
    # truncation-cut one records zero).
    "points_played",
    "completed_point_crossings",
    "point_serve_nudged",
    "point_end_out_of_bounds",
    "point_end_ball_net",
    "point_end_second_bounce",
    "point_end_failed_to_cross",
    "point_end_illegal_hit",
    "point_end_net_touch",
    "point_end_volley",
)

# The env's nine-component reward decomposition: per-step increments
# that sum to each step's reward (envs/paddle_tennis.py). Collected at
# eval so every component's per-episode delivered dose lands in
# eval_info.csv as ``<key>_ep_sum_mean`` -- the ``rew_`` prefix is what
# opts a collected key into the eval callback's episode-sum aggregation
# (review doc §1.6: the milestone-video CSVs were the only artifact
# carrying the decomposition, so a shaping escrow's dose was not
# auditable from the metrics artifacts).
_PADDLE_TENNIS_REWARD_COMPONENT_KEYS = (
    "rew_return",
    "rew_fault",
    "rew_unsafe",
    "rew_shaping",
    "rew_shaping_clawback",
    "rew_reach",
    "rew_reach_clawback",
    "rew_hold",
    "rew_hold_clawback",
)

# Same rationale as the humanoid exclusions above: normalize only the
# continuous physical block; the bounded rally/contact tail stays raw.
_PADDLE_TENNIS_NORMALIZATION_EXCLUSIONS = tuple(
    range(
        PADDLE_TENNIS_NORMALIZED_SLICE.stop,
        len(PADDLE_TENNIS_OBSERVATION_NAMES),
    )
)


def _tennis_curriculum_extra_cfg(
    *, video_length: int, early_stop_patience: int
) -> dict[str, Any]:
    """Shared fixed-stage recording/evaluation contract for Stages 0–2.

    ``early_stop_patience`` is per-stage because the earliest reachable
    stop is evaluation ``2 * patience`` (min_evals equals patience), so
    the value must be scaled to each stage's eval budget
    (total_timesteps / eval_freq = 20 / 40 / 80 evals) or the stop can
    never fire — the WallBall-era 20 was mathematically inert for
    Stages 0–1 (humanoid_env_review.md finding N-F1).
    """
    return {
        "n_envs": 1,
        "early_stop_patience": early_stop_patience,
        "normalize_obs_excluded_indices": (_HUMANOID_TENNIS_NORMALIZATION_EXCLUSIONS),
        # The stage reward is hand-scaled and sparse: terminal outcomes
        # in {-2, -1, 0, +1}, plus (stages 1-2) an escrowed +/-0.25
        # contact pulse that commits on success and claws back otherwise.
        # The PPO default (VecNormalize returns) would amplify the first
        # rare success toward the clip_reward ceiling exactly when
        # learning begins (humanoid_env_review.md I-F5).
        "normalize_reward": False,
        # Stock iid per-step Gaussian exploration at 100 Hz averages to a
        # near-still arm (0 contacts in 264 random episodes, while a
        # *constant* random action held all episode succeeds ~15% —
        # humanoid_env_review.md C2 / DECISIONS.md). gSDE gives temporally
        # correlated exploration; the small entropy floor guards against
        # std collapse onto the repeatable "safe non-swing" −1 while the
        # success reward is still unsampled.
        "model_kwargs": {
            "use_sde": True,
            "ent_coef": 0.01,
        },
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
        "info_eval_terminal_keys": (_HUMANOID_TENNIS_CURRICULUM_TERMINAL_EVAL_KEYS),
        "info_eval_distribution_keys": ("rally_count",),
        "success_key": "stage_success",
        "success_threshold": 1.0,
        # Best-model selection and early stop follow the task metric, not
        # the shaped eval reward (run 20260712_190054's reward-selected
        # "best" completed zero rallies). legal_hit_count_ep_mean sits
        # between the success keys and the reward tie-break so
        # pre-success contact progress (the regime the escrowed shaping
        # creates) still resets patience and updates best_model instead
        # of presenting a bit-flat score until the first target return.
        "headline_key": "stage_success",
        "best_metric_keys": (
            "stage_success_ep_mean",
            "success_rate",
            "legal_hit_count_ep_mean",
            "episode_reward_mean",
        ),
        # Note: eval is deterministic for stages 0-1 (scripted launch,
        # deterministic policy, frozen normalizer), so the confirmation
        # batch replays the candidate exactly there; it only guards
        # against single-batch flukes under stage 2's randomized feeds
        # and is kept uniform as cheap future-proofing.
        "confirm_best_eval": True,
        # With headline selection the reward EvalCallback stream is
        # reporting-only; 5 episodes keep evaluations.npz alive while the
        # info-dict stream takes the full 20 for selection.
        "reward_eval_episodes": 5,
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
        extra_cfg={
            # Colab-calibrated SAC worker count (see the notebook's
            # n_envs table): off-policy, so 8 rollout workers keep
            # the replay buffer fed without starving the ~8 vCPUs.
            "n_envs": 8,
            "early_stop_patience": 20,
        },
        description="Keep a ball on a 6-DOF tray.",
    ),
    "BallBounce": Recipe(
        env_cls=BallBounceEnv,
        env_kwargs={"render_mode": "rgb_array", "min_force": 100.0},
        default_total_timesteps=1_500_000,
        name_prefix="ball_bounce",
        extra_cfg={
            "n_envs": 8,
            "early_stop_patience": 20,
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
        description=("Deliberately juggle a ball from the top face of a 6-DOF paddle."),
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
            "n_envs": 8,
            "early_stop_patience": 20,
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
            "n_envs": 8,
            "early_stop_patience": 20,
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
            # 500/500 scripted trials; a parked paddle scored 0/200.
            "paddle_home_x": -2.7,
            # Lane front -1.6 was chosen by the 2026-07-16 damping x lane
            # sweep (paired with paddle_joint_damping 8 below -- the
            # two only work together: (-1.6, damping 5) collapses to 51%
            # oracle second returns from OOB/style faults, and
            # (-2.1, damping 8) to 49% from a too-slow paddle in a narrow
            # lane). At (-1.6, damping 8) the scripted oracle completes
            # >=2 returns from 92% of standard serves (n=500), and --
            # decisive for learnability -- a placement-blind full-swing
            # tracker recovers a second exchange in 70% of episodes vs
            # exactly 0% at the old (-3.2, -2.1)/damping-5 geometry,
            # which is why run 20260714_050506 plateaued at one return.
            # NOTE: the recoverable_bounce_score projection plane derives
            # from this front edge, so widening the lane also moved the
            # placement target forward from -2.1 to -1.6.
            "paddle_x_target_range": (-3.2, -1.6),
            # The serve's first bounce lands at x in [-2.01, -1.50], so
            # most of it is now inside the widened lane and a forward
            # paddle can touch the ball pre-bounce. Terminal
            # paddle_before_bounce made that trap as costly as total
            # passivity (a naive front-camper faults 39% of episodes at
            # this lane front); the softened fine keeps the gate shut
            # but lets the rally -- and learning -- continue.
            "early_touch_penalty": 0.25,
            # weak_return_penalty deliberately UNSET (terminal rule).
            # The 0.13.0 fined retry (0.1) was falsified by run
            # 20260718_213222: terminal weak returns and terminal OOB
            # are symmetric (-1 each) and push the policy toward
            # committed swings; the retry made undershooting 10x
            # cheaper than overshooting and kept the episode alive, so
            # SAC converged to soft, unchainable returns (82% double
            # bounce, OOB ~0, recoverable_bounce_score peak 0.44 vs
            # ~0.99, plateau at one exchange, 1.33 vs 3.23 bounces).
            # Any future retry must keep the fine near-symmetric with
            # out_of_bounds_penalty.
            # Cap saturated swings at 100 N / 8 = 12.5 m/s (~2.3x serve
            # speed) instead of the shared XML's 20 m/s: bang-bang
            # full-power returns then rebound shallow enough to recover.
            # Baseline-only -- open/volley keep the XML's damping 5 and
            # their existing calibration.
            "paddle_joint_damping": 8.0,
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
            "n_envs": 8,
            "early_stop_patience": 20,
            # model_kwargs deliberately empty: SB3 defaults (auto
            # entropy, gamma 0.99) are the configuration both reference
            # runs (20260717_165358, 20260718_023737) learned with. The
            # 0.13.0 gamma 0.995 shipped bundled with the weak-return
            # retry and regressed (run 20260718_213222); it has not
            # been exonerated in isolation and stays out until it is.
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
                # ge_2 saturated at ~100% in runs 20260717_165358 and
                # 20260718_023737 (a dead tiebreaker); ge_5 is the tail
                # that actually improved (8% -> 22% long-horizon) while
                # the 750-step eval cap compresses the mean.
                "bounce_count_ep_ge_5_rate",
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
    "WallBallBootstrap": Recipe(
        env_cls=WallBallEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "min_force": 20.0,
            "rally_style": "one_bounce",
            "paddle_home_x": -2.7,
            "paddle_x_target_range": (-3.2, -1.6),
            "serve_speed": 5.5,
            "serve_lob": 0.0,
            "serve_vy_min": 0.8,
            # Stage-0 serve shape: a narrow lateral corridor and reduced
            # speed jitter shrink the tracking burden that run
            # 20260717_040824's SAC never crossed (zero ball contact in
            # 125k steps while a scripted y-tracker contacts 100% of
            # serves). The performance gate widens both back to the
            # baseline serve distribution as competence is demonstrated;
            # every stage is a subset of the final distribution, so the
            # skill transfers upward by construction. serve_vy_min stays
            # 0.8 at every stage -- the no-op invariant (a parked paddle
            # can never be hit by the serve) is not a curriculum knob.
            "serve_vy_max": 1.1,
            "serve_speed_jitter": 0.2,
            # Bootstrap reward package, calibrated 2026-07-17: it makes
            # the competence ladder strictly monotone at episode level
            # for the first time (stage-0 serve, n=120: parked -1.00 <
            # weak-swing tracker -0.85 < placement-blind full swing
            # +7.63 < oracle +12.07; touch-then-deaden possession also
            # measures -0.85 -- repeat paddle taps no longer reset the
            # stall clock, so held-ball rides to truncation are dead).
            # Depth and serve-pace ladders were swept and REJECTED:
            # close-court play collapses to ~50% oracle second returns
            # (wall rebounds fly out) and slow serves underpower returns
            # (oracle first returns drop from 100% to 12% at speed 3.5).
            "early_touch_penalty": 0.25,
            "weak_return_penalty": 0.1,
            "first_hit_bonus": 0.25,
            "paddle_joint_damping": 8.0,
            # Whole-task episodes only: runs 20260714_211111 and
            # 20260717_025611 showed fragment-heavy training masters the
            # fragments and never learns the serve.
            "recovery_reset_probability": 0.0,
            "recoverable_bounce_bonus": 0.25,
            "recoverable_bounce_lateral_limit": 2.0,
        },
        eval_env_overrides={
            # Canonical evaluation: the full baseline serve distribution.
            "serve_vy_max": 2.0,
            "serve_speed_jitter": 0.5,
        },
        default_total_timesteps=1_500_000,
        name_prefix="wall_ball_bootstrap",
        extra_cfg={
            "n_envs": 8,
            "early_stop_patience": 20,
            "success_key": "bounce_count",
            "success_threshold": 2.0,
            "headline_key": "bounce_count",
            "best_metric_keys": (
                "bounce_count_ep_mean",
                "bounce_count_ep_ge_2_rate",
            ),
            "best_metric_min_delta": 0.5 / 30,
            "confirm_best_eval": True,
            "early_stop_degenerate_evals": 5,
            "degenerate_guard_keys": ("paddle_hit_count_ep_mean",),
            "info_eval_distribution_keys": ("bounce_count",),
            "info_eval_survival_thresholds": {
                "bounce_count": (2, 3, 5),
            },
            # Earned progression through nested serve distributions; the
            # matched (training-stage) eval drives the gate and model
            # selection, eval_info_final.csv tracks the canonical serve.
            "performance_gate": {
                "metric_key": "bounce_count_ep_mean",
                "threshold": 1.3,
                "sustain_evals": 2,
                "stages": (
                    {"serve_vy_max": 1.1, "serve_speed_jitter": 0.2},
                    {"serve_vy_max": 1.4, "serve_speed_jitter": 0.3},
                    {"serve_vy_max": 1.7, "serve_speed_jitter": 0.4},
                    {"serve_vy_max": 2.0, "serve_speed_jitter": 0.5},
                ),
            },
            "final_info_eval": True,
            # Exploration package the failed runs never tried: entropy
            # auto-tuning from a safe floor (the historical collapse to
            # 0.0005 was measured on the legacy 5-action env), real
            # warm-up before gradient steps, and a buffer small enough
            # to evict stale early data within a stage's lifetime.
            "model_kwargs": {
                "ent_coef": "auto_0.02",
                "target_entropy": -1.5,
                "learning_starts": 10_000,
                "buffer_size": 500_000,
                "gamma": 0.995,
            },
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "await_bounce",
                1: "await_paddle",
                2: "await_wall",
            },
        },
        description=(
            "HISTORICAL: the cold-start problem this recipe targets was "
            "solved by auto-entropy before it ever trained "
            "(lessons_learned.md lesson 5), and its reward package "
            "bundles the since-falsified weak-return retry. Its "
            "original successor, WallBallDepthCurriculum, was itself "
            "retired by the 0.24.0 diagnosis. Kept for the record; for "
            "new work use WallBallGoalRally or WallBallTrueBaseline."
        ),
    ),
    "WallBallDepthCurriculum": Recipe(
        env_cls=WallBallEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "min_force": 20.0,
            # Open scoring, emergent style (design doc
            # design_wall_ball_depth_curriculum.md): any paddle hit
            # opens the return gate, volleys are legal paid returns, and
            # there is no early-touch fine — near the wall the natural
            # game is volleying, while the sliding deep stages are
            # intended to make one-bounce baseline play emerge from
            # geometry instead of a fault taxonomy.
            # None of the one_bounce-era shaping (first_hit_bonus,
            # weak_return_penalty, recovery fragments,
            # recoverable_bounce_*) carries over.
            "rally_style": "open",
            # The action mapping stays pinned to the full physical
            # workspace for the entire run so action *scale* never
            # drifts across stages; each stage confines position with
            # paddle_x_fence instead. What each stage does move is the
            # map's pivot (paddle_home_x), because a fixed pivot outside
            # a receded fence throws most of the action range away: run
            # 20260727_004014 finished with home -1.7 against a
            # (-4.7, -3.0) fence, where 71.7% of the x range -- all of
            # action 0 and the entire positive half -- clamped onto the
            # fence's front edge. Holding the pivot at each fence's
            # midpoint keeps the usable share roughly flat (0.49, 0.43,
            # 0.42, 0.47, 0.63) instead of collapsing 0.67 -> 0.28.
            "paddle_home_x": -1.25,
            "paddle_x_target_range": (-4.7, 0.3),
            # Dense credit on the outgoing leg; see WallBallEnv's
            # return_shaping_scale. Deliberately much smaller than
            # track_shaping_scale (0.5): the outgoing gap is metres wide,
            # and the 20260727 audit found incoming shaping already
            # crowding out the flat +1 wall payment (57% -> 73% of the
            # per-cycle reward as the fence recedes). 0.15 adds about
            # +1.0 per completed return -- comparable to the wall, not
            # dominant.
            "return_shaping_scale": 0.15,
            # Stage-0 geometry — volley range, where striking is easy.
            # Run 20260722_124613 exposed a flaw in the original ladder:
            # every stage shared a front-court interval, so the policy
            # could advance while keeping its contacts near the same
            # shallow x position. The replacement ladder below slides the
            # entire fence backward, preserving adjacent overlap but
            # removing the all-stage common refuge. This fixed-origin
            # production baseline passed its 200-episode-per-cell sweep
            # on 2026-07-23. A separately named aligned recipe preserves
            # this control while the serve-origin hypothesis is tested.
            #
            # 0.22.0 holds every fence 2.1 m wide instead of letting it
            # narrow 3.0 -> 1.7 with depth. Return pace is set by the
            # paddle's forward speed at contact, which needs runway: a
            # swept probe on the old goal fence scored 0 completed
            # returns from 0.0-0.4 m of travel, 1 from 0.6-0.9 m, and
            # 2-3 from 1.2-1.6 m. The old ladder left exactly 0.9 m at
            # the goal, so 100% of audited episodes returned the serve
            # and then had no runway to re-load -- the measured cause of
            # the 1.14-return plateau. Every stage now clears >= 1.3 m.
            # 2.1 m is the widest constant width that still leaves the
            # ladder without an all-stage refuge: the fence travels 2.4 m
            # back (-2.3 -> -4.7), so any width >= 2.4 would hand every
            # stage a shared front-court interval again -- the exact flaw
            # run 20260722_124613 exposed. 2.1 keeps a 0.3 m separation,
            # matching the ladder it replaces.
            "paddle_x_fence": (-2.3, -0.2),
            "paddle_start_x": -1.6,
            # Serve energy co-moves with depth via the gate (5.2 -> 7.0)
            # so the ball reliably reaches the deeper paddle. This
            # production baseline keeps its serve origin fixed at 1.0.
            # The flat serve (lob 0) keeps contact at face height —
            # lofted serves sail over the fixed-height face (sweep
            # iteration 1).
            "serve_start_x": 1.0,
            "serve_speed": 5.2,
            "serve_lob": 0.0,
            "serve_speed_jitter": 0.5,
            "serve_vy_min": 0.8,
            "serve_vy_max": 2.0,
            # Baseline-calibrated servo damping (see WallBallBaseline):
            # caps saturated swings at ~12.5 m/s so full-power returns
            # rebound shallow enough to recover.
            "paddle_joint_damping": 8.0,
        },
        eval_env_overrides={
            # The canonical scoring task is the ladder's FINAL stage —
            # the campaign goal. The gate re-applies the current
            # training stage to the matched evaluator (selection and
            # early stopping stay matched-stage), but the final-config
            # evaluator (eval_info_final.csv), milestone videos, the
            # reward eval stream, and the post-training long-horizon
            # audit all use this factory unsynced; leaving it at the
            # stage-0 defaults would score the easiest geometry for the
            # whole run and silently inflate apparent competence.
            # "Deepest stage reached" is not expressible with the
            # final-eval machinery. Within one recipe this fixed goal
            # task is comparable across runs; cross-recipe comparison
            # requires evaluation on both serve geometries. Must equal
            # the last performance_gate stage (pinned by test).
            "paddle_x_fence": (-4.7, -2.6),
            "paddle_start_x": -3.9,
            "paddle_home_x": -3.65,
            "serve_start_x": 1.0,
            "serve_speed": 7.0,
        },
        default_total_timesteps=6_000_000,
        name_prefix="wall_ball_depth_curriculum",
        extra_cfg={
            "n_envs": 8,
            "early_stop_patience": 20,
            # gamma 0.995, and ONLY gamma: SB3's auto entropy stays.
            # The exchange cadence in this task is 117-135 env steps
            # (measured on the calibrated oracle), and at the SB3
            # default 0.99 the next return is worth 0.99^130 ~ 0.27 of
            # the one in hand and the third ~0.07 -- the agent cannot
            # see across an exchange, so cashing out at exchange 2 is
            # optimal under the discount it is given. The only run that
            # ever produced a long rally in this project
            # (WallBall 20260713_192636, bounce_count_ep_mean 12.30 --
            # absent from every doc, see wall_ball_aligned_patience_review)
            # used gamma 0.995. The one prior attempt shipped it bundled
            # with the falsified weak-return retry and was reverted with
            # it (CHANGELOG 0.13.0), so it has never been tested alone.
            "model_kwargs": {"gamma": 0.995},
            "success_key": "bounce_count",
            # Success mirrors the gate bar: a sustained three-exchange
            # rally is what earns the next depth stage.
            "success_threshold": 3.0,
            "headline_key": "bounce_count",
            "best_metric_keys": (
                "bounce_count_ep_mean",
                "bounce_count_ep_ge_5_rate",
            ),
            # Promotion is a threshold crossing on a noisy statistic, so
            # the gate's precision is what decides it. Per-episode
            # bounce_count sd is ~0.87, giving a 30-episode batch sd of
            # ~0.16 and a 3-eval window sd of ~0.092 -- against which
            # runs 20260724_152530 / 20260725_171747 cleared the 3.0 bar
            # at stage 1 by 0.011, about a tenth of a standard error. 60
            # episodes cut the window sd to ~0.065. Deliberately NOT a
            # lower threshold: 3.0 is demonstrably reachable (a
            # predictive controller on these same actuators scores 4.39
            # at stage 0), so the defect was the measurement, not the bar.
            "n_eval_episodes": 60,
            # Pin the goal-task stream at its 0.20.0 size. It follows
            # n_eval_episodes under the merge, and letting it double
            # would be paid on a run whose eval streams already cost
            # about as many env steps as training itself.
            "final_eval_episodes": 30,
            # Half the 1/60 granularity, per best_metric_min_delta's docs.
            "best_metric_min_delta": 0.5 / 60,
            "confirm_best_eval": True,
            "early_stop_degenerate_evals": 5,
            "degenerate_guard_keys": ("paddle_hit_count_ep_mean",),
            "info_eval_distribution_keys": ("bounce_count",),
            "info_eval_survival_thresholds": {
                "bounce_count": (2, 3, 5),
            },
            # Earned progression from volley range back toward the
            # workspace baseline. The matched (training-stage) eval
            # drives the gate and model selection; eval_info_final.csv
            # and the final long-horizon audit score the ladder's final
            # stage (see eval_env_overrides). curriculum/stage_index in
            # TensorBoard is the campaign's real headline: how far back
            # did it earn its way?
            #
            # Run 20260721_004722 (the first campaign run) sharpened
            # three gate settings without touching the bar itself:
            #
            # - window_mean promotion: with 30-episode batches the
            #   per-eval SE near the bar is ~0.4 bounces, so "two
            #   consecutive >= 3.0" is a coin flip for a true-3.0
            #   policy -- stage 2 cleared 3.0 in four separate evals
            #   without ever promoting. The 2-eval mean estimates the
            #   same quantity at half the variance.
            # - Promotion warm-up (pause + clear): each advance
            #   previously crashed the matched eval (3.37 -> 1.43,
            #   3.10 -> 1.70; ~0.5-1M steps of recovery) while SAC kept
            #   updating 1:1 against a buffer that was ~100%
            #   previous-stage physics (the fence participates in
            #   dynamics via target clamping). Now an advance drops the
            #   buffer and holds gradient updates for 50k steps of
            #   fresh frontier-stage data.
            #
            # Run 20260721_142121 (the second campaign run, first with
            # window_mean) widened the window from 2 to 3 evals: its
            # stage-0 exit cleared on a 2-eval mean of 3.08 -- ~0.3 SE
            # above the bar, a variance-driven promotion by the same SE
            # arithmetic above -- while its stage-1 exit (3.37) would
            # have cleared any reasonable window. The bar itself stays
            # 3.0: observed skill ceilings at the easy stages (eval
            # maxima 3.27/3.47) leave no headroom to raise it without
            # gating the whole run, and the stage-2 stall that ended
            # the run was a post-advance recovery failure, not a
            # premature promotion.
            "performance_gate": {
                "metric_key": "bounce_count_ep_mean",
                "threshold": 3.0,
                "sustain_evals": 3,
                "promotion_rule": "window_mean",
                "advance_update_pause_steps": 50_000,
                "clear_replay_buffer_on_advance": True,
                # Run 20260727_004014 promoted three times and then sat
                # on stage 3 for 97 evaluations with train/ent_coef at
                # 0.0011 -- effectively deterministic. Every advance was
                # dropping a policy with no exploration budget into new
                # geometry. This lever was added in 0.20.0 for exactly
                # that failure and had never been exercised; it is legal
                # here because model_kwargs pins only gamma, leaving
                # SB3's ent_coef on "auto".
                "reset_entropy_on_advance": True,
                # Each stage: fence 2.1 m wide (constant), paddle_home_x
                # on the fence midpoint, and >= 1.3 m of runway between
                # paddle_start_x and the fence's front edge.
                "stages": (
                    {
                        "paddle_x_fence": (-2.3, -0.2),
                        "paddle_start_x": -1.6,
                        "paddle_home_x": -1.25,
                        "serve_start_x": 1.0,
                        "serve_speed": 5.2,
                    },
                    {
                        "paddle_x_fence": (-2.9, -0.8),
                        "paddle_start_x": -2.1,
                        "paddle_home_x": -1.85,
                        "serve_start_x": 1.0,
                        "serve_speed": 5.5,
                    },
                    {
                        "paddle_x_fence": (-3.5, -1.4),
                        "paddle_start_x": -2.7,
                        "paddle_home_x": -2.45,
                        "serve_start_x": 1.0,
                        "serve_speed": 6.0,
                    },
                    {
                        "paddle_x_fence": (-4.1, -2.0),
                        "paddle_start_x": -3.3,
                        "paddle_home_x": -3.05,
                        "serve_start_x": 1.0,
                        "serve_speed": 6.5,
                    },
                    {
                        "paddle_x_fence": (-4.7, -2.6),
                        "paddle_start_x": -3.9,
                        "paddle_home_x": -3.65,
                        "serve_start_x": 1.0,
                        "serve_speed": 7.0,
                    },
                ),
            },
            # Startup ladder certification (0.23.0): every run sweeps
            # its own gate stages with the scripted reference cells
            # before training and writes the verdict to
            # reports/ladder_certification.json. Derived from the live
            # performance_gate spec and env_fn, so there is no second
            # stage table to go stale — the 0.22.0 ladder shipped
            # uncertified because tools/depth_stage_sweep.py still
            # encoded the retired geometry (review 20260727_233859).
            # Probes recalibrated for the constant-width geometry
            # (0.24.0): a lead_charge gap grid on calibration seeds
            # 0-99 (100 eps/cell), then a refinement pass at 200
            # eps/cell on seeds 0-199 choosing per stage by >=2-return
            # rate with margin over the 0.90 floor rather than the raw
            # argmax (calibration argmaxes routinely shed 3-5 points
            # held-out; see DECISIONS.md). Frozen at n=200:
            # 92.5%/2.26, 97.0%/2.44, 98.0%/2.56, 96.5%/2.58,
            # 95.0%/2.56 for stages 0-4. The stock 0.21-era probes
            # ({run_up 1.1}, {charge_gap 1.0/1.0/1.8/1.7}) measured
            # 36%/93%/94%/70%/93% on seeds 1000-1099 and fail
            # feasibility at stages 0 and 3 plus monotonicity and the
            # inversion detector at stage 0/1. Held-out verdict for the
            # frozen set lives in the 2026-07-28 review doc (seeds
            # 3000-3099, 100 eps/cell, run through this exact code
            # path). 30 episodes/cell at startup (~0.2M env steps,
            # under a minute) because the 0.90 feasibility floor sits
            # close to the probes' true 92-98% rates: at 10 episodes a
            # single unlucky seed flips the verdict.
            "ladder_certification": {
                "episodes": 30,
                "seed_start": 30_000,
                "oracle_probes": (
                    {"lead_charge": 3.0},
                    {"lead_charge": 0.8},
                    {"lead_charge": 1.0},
                    {"lead_charge": 1.2},
                    {"lead_charge": 3.0},
                ),
            },
            "final_info_eval": True,
            # The reward EvalCallback stream is reporting-only under
            # headline selection yet historically rolled the full 30
            # episodes every 25k steps; across run 20260721_004722 the
            # three eval streams together cost about as many env steps
            # as the 3M training steps. 5 episodes keep evaluations.npz
            # alive and return the wall clock to training.
            "reward_eval_episodes": 5,
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "await_bounce",
                1: "await_paddle",
                2: "await_wall",
            },
        },
        description=(
            "HISTORICAL (the sliding-fence ladder was retired by the "
            "0.24.0 campaign diagnosis: three ladder generations "
            "stalled below the flat 3.0 bar, and the pre-registered "
            "paired A/B showed direct goal-task training beats every "
            "curriculum variant -- use WallBallGoalRally or "
            "WallBallTrueBaseline). Kept for reproducing the ladder "
            "era: open-scoring wall rallies with a performance-gated "
            "depth ladder that walks the paddle fence from volley "
            "range to the workspace baseline each time a sustained "
            "three-exchange rally is demonstrated."
        ),
    ),
    "PaddleTennis": Recipe(
        env_cls=PaddleTennisEnv,
        # The adopted era (LR1 ADOPT verdict,
        # docs/design_paddle_tennis_reach_shaping.md §4a): continuous
        # n-point play with position carryover, the contact escrow,
        # and the reach escrow. Each shipped default-off first and was
        # certified bit-identical + farm-proof by its own battery
        # (NP0-NP3, S1/S2, RS0-RS2) before adoption; the pilots that
        # justified the flip are L2W (20260815_180815) and LR1
        # (20260816_135919 — k=2 above zero at five checkpoints, the
        # era headline's first movement).
        env_kwargs={
            "render_mode": "rgb_array",
            "points_per_episode": None,
            "contact_shaping": 0.25,
            "reach_shaping": 0.25,
        },
        default_total_timesteps=2_000_000,
        name_prefix="paddle_tennis",
        extra_cfg={
            # Colab-calibrated SAC worker count (the wall-ball value;
            # this env steps the same physics at the same rates).
            "n_envs": 8,
            "early_stop_patience": 20,
            # Exploration package, from the ground-era pilot diagnosis
            # (docs/paddle_tennis_diagnosis_20260808.md; provenance
            # and pre-registered pilot in
            # docs/paddle_tennis_exploration_20260808.md). The raised
            # target is the actual entropy floor: SB3's dual update
            # anneals the multiplier low when entropy meets the
            # target and raises it when entropy sags below -- the
            # stock run's 5e-5 coefficient was the tuner resting at
            # the too-low default target (-3.0), so the fix is the
            # target, not the coefficient. auto_0.02/-1.5 is the
            # WallBall bootstrap *design* (DECISIONS.md 0.11.0 --
            # never trained; adopted as design precedent, and pinning
            # a float instead is a measured poison per
            # lessons_learned.md's ent_coef x episode_len budget).
            # gSDE (humanoid C2, measured under PPO) needs train_freq
            # here to work at all under SAC: the off-policy collector
            # resets the gSDE noise matrix every rollout, and at the
            # default train_freq=1 that is every single step -- iid
            # noise in a gSDE costume. 64 vec-steps holds one matrix
            # through a coherent approach (half the 126-step exchange
            # cadence); gradient_steps=-1 (set by train()) keeps the
            # 1:1 update-per-transition ratio.
            "model_kwargs": {
                "use_sde": True,
                "ent_coef": "auto_0.02",
                "target_entropy": -1.5,
                "train_freq": (64, "step"),
            },
            "normalize_obs_excluded_indices": (_PADDLE_TENNIS_NORMALIZATION_EXCLUSIONS),
            "csv_header": _PADDLE_TENNIS_CSV_HEADER,
            "info_row_fn": _paddle_tennis_info_row,
            # Headline selection stays on crossings (the era's bridge
            # metric), but success moves to the side-A-only hit count:
            # under continuous play the opponent's serve-returns alone
            # clear any crossings threshold (success_rate read 1.000
            # at all 80 evals of the failed L2 pilot), and the plain
            # legal_hit_count counts both sides (~0.37 nonzero mean on
            # a zero-contact run). legal_hit_count_a is the policy's
            # own final-point hit count — exactly zero for a dead
            # policy, which is also what arms the degenerate guard
            # below (review doc §4c; the L2W-hardened guard set).
            "success_key": "legal_hit_count_a",
            "success_threshold": 1.0,
            "headline_key": "crossings",
            "info_eval_keys": (
                "crossings",
                "rally_count",
                "legal_hit_count",
                "legal_hit_count_a",
                "bounce_count",
                # Reward decomposition -> per-episode dose audit in
                # eval_info.csv (see the constant's comment).
                *_PADDLE_TENNIS_REWARD_COMPONENT_KEYS,
            ),
            # Selection/stop hygiene, measured in on the pilots: the
            # min-delta sits above the ±0.2 opponent-crossings noise
            # that re-crowned best_model for 80 straight evals on the
            # L2 run; confirm_best_eval is the WallBall precedent; the
            # degenerate guard kills a zero-contact run in ~5 evals
            # (~125k steps) instead of a full budget.
            "best_metric_min_delta": 0.25,
            "confirm_best_eval": True,
            "early_stop_degenerate_evals": 5,
            "degenerate_guard_keys": ("legal_hit_count_a_ep_mean",),
            "info_eval_terminal_keys": _PADDLE_TENNIS_TERMINAL_EVAL_KEYS,
            "info_eval_distribution_keys": ("crossings",),
            # Behavioral diagnosis at every checkpoint save: the
            # exchange/positioning instrument that separated the
            # ground-era pilot's plateau (one memorized serve-return
            # macro, no general ball-reaching) from the stroke- and
            # opponent-side explanations. 30 episodes on the
            # diagnosis calibration block; ~90 s of CPU per
            # checkpoint while the GPU idles at the save boundary.
            "checkpoint_diagnosis": {
                "episodes": 30,
                "seed_start": 5200,
            },
            "phase_key": "rally_phase",
            "phase_labels": {
                0: "initial_feed",
                1: "awaiting_return",
                2: "return_in_flight",
                3: "terminal",
            },
        },
        description=(
            "Two-sided paddle rally on the probe-frozen 13 m court "
            "under ground rules (pre-bounce returns fault; "
            "docs/paddle_tennis_ground_rules_20260803.md): the policy "
            "plays side A against the bounce-waiting ground oracle "
            "through the P4 mirror in continuous n-point play "
            "(1500-step episodes, alternating serves, paddle "
            "positions carried over between points), cooperative +1 "
            "per confirmed return by either side plus the escrowed "
            "contact and reach shapings (0.25 each; farm-proof, paid "
            "only when the milestone is taken). The era certified "
            "held-out through the n-point probes harness (NP3) and "
            "adopted on the LR1 pilot verdict."
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
        # patience 8 of a 20-eval budget: earliest stop at eval 16.
        extra_cfg=_tennis_curriculum_extra_cfg(video_length=150, early_stop_patience=8),
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
            # Target-return stages otherwise pay a flat −1 for whiff,
            # off-target hit, and net fault alike — zero differential
            # signal between hitting and never moving (30/30 random
            # episodes at exactly −1.000, humanoid_env_review.md). The
            # escrowed shaping pays +0.25 at racket contact and claws it
            # back at episode end unless the hit becomes a target return,
            # so the discounted contact-time signal survives while
            # episode-total farming of non-target hits stays impossible.
            "reward_config": TennisRewardConfig(valid_hit_shaping=0.25),
        },
        default_total_timesteps=1_000_000,
        default_algo="PPO",
        name_prefix="humanoid_tennis_stage1_anchored_return",
        # patience 12 of a 40-eval budget: earliest stop at eval 24.
        extra_cfg=_tennis_curriculum_extra_cfg(
            video_length=300, early_stop_patience=12
        ),
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
            # Same escrowed contact shaping as Stage 1 (see that recipe).
            "reward_config": TennisRewardConfig(valid_hit_shaping=0.25),
        },
        default_total_timesteps=2_000_000,
        default_algo="PPO",
        name_prefix="humanoid_tennis_stage2_randomized_return",
        # patience 20 of an 80-eval budget: earliest stop at eval 40.
        extra_cfg=_tennis_curriculum_extra_cfg(
            video_length=300, early_stop_patience=20
        ),
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
            "early_stop_patience": 20,
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


_ALIGNED_DEPTH_SERVE_ORIGINS = (1.0, 0.69, 0.34, -0.01, -0.35)


def _make_aligned_depth_curriculum(base: Recipe) -> Recipe:
    """Build the experimental serve-aligned arm without mutating its control."""
    env_kwargs = deepcopy(base.env_kwargs)
    extra_cfg = deepcopy(base.extra_cfg)
    gate = extra_cfg["performance_gate"]
    base_stages = gate["stages"]
    if len(base_stages) != len(_ALIGNED_DEPTH_SERVE_ORIGINS):
        raise ValueError(
            "aligned serve-origin schedule must match the baseline stage count"
        )
    gate["stages"] = tuple(
        {**stage, "serve_start_x": serve_start_x}
        for stage, serve_start_x in zip(
            base_stages,
            _ALIGNED_DEPTH_SERVE_ORIGINS,
            strict=True,
        )
    )
    eval_env_overrides = {
        **deepcopy(base.eval_env_overrides),
        "serve_start_x": _ALIGNED_DEPTH_SERVE_ORIGINS[-1],
    }
    return replace(
        base,
        env_kwargs=env_kwargs,
        eval_env_overrides=eval_env_overrides,
        name_prefix="wall_ball_depth_curriculum_aligned",
        extra_cfg=extra_cfg,
        description=(
            "HISTORICAL (closed permanently on 2026-07-28: Phase D's "
            "pre-registered no-go fired -- alignment falsified at both "
            "test sites; see plan_wall_ball_aligned_deep_stages.md). "
            "The serve-origin treatment arm of the retired depth "
            "ladder: the same open-scoring ladder with stage-specific "
            "serve origins aligning first-bounce position to paddle "
            "start, with WallBallDepthCurriculum as the fixed-origin "
            "control. Kept for reproducing the A/B."
        ),
    )


RECIPES["WallBallDepthCurriculumAligned"] = _make_aligned_depth_curriculum(
    RECIPES["WallBallDepthCurriculum"]
)


def _make_goal_rally(base: Recipe) -> Recipe:
    """Build the direct goal-task recipe (0.24.0) — no curriculum at all.

    The 2026-07-28 diagnosis review's structural replacement for the
    sliding-fence depth ladder. Three runs of that ladder (0.19-0.22)
    stalled below a flat 3.0 bar that no scripted reference reaches at
    any rung, and every promotion paid a measured tax: a +0.35 m
    serve-landing jump plus -- because a fence slide changes the
    *dynamics* via target clamping -- the replay-wipe/entropy-reset/
    action-map machinery whose one exercised instance cost run
    20260727_233859 ~4.3M of 6M steps. Meanwhile the 0.22.0
    constant-width goal fence had already made the goal task itself
    learnable (crude placement-blind two-return rate 80-84%, vs
    9.5-16% on the retired fence) -- and no run ever trained there,
    because the ladder was in the way.

    The pre-registered goal-fence structure A/B/C (diagnosis review
    S2: direct vs aligned-serve vs serve-origin mixture, paired
    common-seed local SAC) answered the remaining question: the true
    goal serve is directly learnable from a cold start, fastest of the
    three conditions (held-out 100-seed cross-eval: direct 2.03/2.71
    completed returns at 500k local steps vs the campaign's all-time
    Colab best of 1.14; mixture 2.27 -- no better; aligned-only
    1.09-1.51 -- worse). Per the pre-registered decision rules, no
    serve curriculum earns its complexity: this recipe trains the
    canonical goal task from the first step, and evaluation equals
    training.

    The single-stage gate never promotes; it exists so the run keeps
    the campaign's artifact contract (stage-context stamping,
    curriculum_stages.json, and startup certification of the training
    geometry -- certified held-out twice: 96% oracle two-return rate on
    seeds 3000-3099, 91% on 3100-3199). The 3.0 threshold is purely
    informational: curriculum/gate_window_mean crossing it in
    TensorBoard IS the campaign goal being met.
    """
    env_kwargs = deepcopy(base.env_kwargs)
    extra_cfg = deepcopy(base.extra_cfg)
    goal = dict(base.extra_cfg["performance_gate"]["stages"][-1])
    # Train ON the campaign-goal task: the depth ladder's final stage,
    # which is also this recipe's (identical) evaluation task.
    env_kwargs.update(goal)
    stages = (dict(goal),)
    extra_cfg["performance_gate"] = {
        "metric_key": "bounce_count_ep_mean",
        "threshold": 3.0,
        "sustain_evals": 3,
        "promotion_rule": "window_mean",
        "stages": stages,
    }
    extra_cfg["ladder_certification"] = {
        "episodes": 30,
        "seed_start": 30_000,
        "oracle_probes": ({"lead_charge": 3.0},),
    }
    # Training and evaluation are the same distribution, so the second
    # info-eval stream would duplicate the matched one; the reward
    # EvalCallback (5 episodes) keeps evaluations.npz alive instead.
    extra_cfg["final_info_eval"] = False
    extra_cfg["final_eval_episodes"] = None
    eval_env_overrides = dict(goal)
    return replace(
        base,
        env_kwargs=env_kwargs,
        eval_env_overrides=eval_env_overrides,
        name_prefix="wall_ball_goal_rally",
        extra_cfg=extra_cfg,
        description=(
            "Train the campaign-goal task directly: open-scoring rallies "
            "at the constant goal fence (-4.7, -2.6), paddle start -3.9, "
            "serve origin 1.0, serve speed 7.0 — the depth ladder's final "
            "stage as the whole task, replacing the retired sliding-fence "
            "curriculum. Paired local A/B evidence (diagnosis review "
            "2026-07-28): direct training here beats every curriculum "
            "variant tested and the ladder's all-time goal transfer."
        ),
    )


RECIPES["WallBallGoalRally"] = _make_goal_rally(RECIPES["WallBallDepthCurriculum"])


def _make_true_baseline(base: Recipe) -> Recipe:
    """Build the 0.25.0 true-baseline era recipe from the goal-rally one.

    The goal-rally era proved deep receive-and-recover play at the
    workspace edge (-4.7): 2 of 3 seeds sustained >= 3.3 completed
    returns, all contacting at -3.4..-3.7 with the fixed-pitch face
    (replication review 2026-07-30). This era extends the same direct
    task to the ITF baseline: the paddle workspace reaches -8.2 (just
    behind the baseline the tennis style draws at -7.985), the paddle
    starts at -7.9, and an 11 m/s serve forces a genuinely deep first
    contact.

    Every geometry and mechanics number here was verified by scripted
    probes before this recipe existed (design doc
    design_wall_ball_true_baseline.md):

    - Serve: from origin 1.0 at speed 11.0, the serve's post-bounce leg
      crosses x=-7.9 in 100% of probe episodes at 0.54 m height and
      9.4 m/s -- reachable by the face (its centre spans 0.05-2.0 m
      when addressing) and forced deep: oracle first contact averages
      -7.0 (p90 -6.99). Speeds below 10 die before the baseline; the
      pre-extension serve (7.0) is unreturnable from -7.9 only because
      it never arrives, not because of paddle mechanics.
    - Fence (-8.2, -2.6): a baseline-only fence is unsustainable by
      geometry -- measured rebounds off deep returns land at mean
      x=-1.0, so the rally necessarily travels forward. The fence keeps
      the goal-era back edge -2.6, adding the deep band rather than
      relocating play; the deep *receive* is forced by the serve, and
      forward recovery stays legal. paddle_home_x -5.4 is the fence
      midpoint (the 0.22.0 usable-share rule).
    - In-play bound: legal deep serves and rebounds legitimately hop
      past x=-9 behind the -8.2 workspace, so this recipe widens the
      env's deep out-of-bounds edge to -10.0 (a per-task kwarg added in
      0.25.0; every earlier recipe keeps the historical -6.0).
    - Reference band (calibrated lead-charge 2.6 oracle, seeds 0-59):
      mean 1.98 completed returns, >=2 rate 67%, >=3 rate 22%, exchange
      cadence ~156 steps (vs ~125 at the goal task). The certification
      feasibility floor is set to 0.50 -- the measured band minus two
      30-episode sampling standard deviations -- because the stock 0.90
      floor encodes "near-certain two exchanges", which no scripted
      reference achieves on an 11.8 m court. The resolved floor is
      stamped into the certification report.

    The single-stage gate again never promotes; the 3.0 threshold stays
    the campaign-wide informational mark (NOT a promotion bar -- there
    is nothing to promote to). Era-specific success criteria are
    pre-registered in the design doc, calibrated against the reference
    band above rather than the goal-era's.
    """
    env_kwargs = deepcopy(base.env_kwargs)
    extra_cfg = deepcopy(base.extra_cfg)
    task = {
        "paddle_x_fence": (-8.2, -2.6),
        "paddle_start_x": -7.9,
        "paddle_home_x": -5.4,
        "serve_start_x": 1.0,
        "serve_speed": 11.0,
    }
    env_kwargs.update(task)
    # The action mapping opts into the extended physical workspace and
    # stays pinned there for the whole run (mapping never moves; the
    # fence confines position). The bare-env default mapping remains
    # frozen at (-4.7, 0.3), so this range must be explicit.
    env_kwargs["paddle_x_target_range"] = (-8.2, 0.3)
    env_kwargs["ball_in_play_min_x"] = -10.0
    stages = (dict(task),)
    extra_cfg["performance_gate"] = {
        "metric_key": "bounce_count_ep_mean",
        "threshold": 3.0,
        "sustain_evals": 3,
        "promotion_rule": "window_mean",
        "stages": stages,
    }
    extra_cfg["ladder_certification"] = {
        "episodes": 30,
        "seed_start": 30_000,
        "oracle_probes": ({"lead_charge": 2.6},),
        "feasibility_ge2_floor": 0.50,
    }
    extra_cfg["final_info_eval"] = False
    extra_cfg["final_eval_episodes"] = None
    # Patience 60, not the goal era's 20: the replication's record seed
    # set its best window at 3.2M steps under patience 60 after long
    # flat stretches, and this era's exchange cadence is ~25% longer
    # still, so improvement arrives in fewer evals per step.
    extra_cfg["early_stop_patience"] = 60
    eval_env_overrides = dict(task)
    return replace(
        base,
        env_kwargs=env_kwargs,
        eval_env_overrides=eval_env_overrides,
        name_prefix="wall_ball_true_baseline",
        extra_cfg=extra_cfg,
        description=(
            "Extend the proven direct-task recipe to the ITF baseline: "
            "workspace and fence reach -8.2, paddle starts at -7.9, and "
            "an 11 m/s serve forces first contact near -7.0 before the "
            "rally recovers forward (rebounds land near -1.0). Probe-"
            "verified geometry throughout; certification floor 0.50 per "
            "the measured oracle band (67% two-return rate) on a task "
            "harder than any scripted reference plays."
        ),
    )


RECIPES["WallBallTrueBaseline"] = _make_true_baseline(RECIPES["WallBallGoalRally"])


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
    base_env_overrides: Mapping[str, Any] | None = None,
):
    """Return the canonical evaluation factory for ``env_name``.

    The recipe's evaluation overrides are layered over its training kwargs,
    then one-off caller overrides (such as a longer episode horizon) are
    applied last. This keeps checkpoint selection and post-training audits on
    standard full episodes even when the training factory samples curriculum
    reset states.

    ``base_env_overrides`` sit *below* the recipe's evaluation overrides:
    a run-config file's ``[env]`` table lands there, so it reaches both
    the training and evaluation environments (a physics tweak must not
    silently split the two) while the recipe's canonical evaluation
    settings -- and then the file's ``[eval_env]`` table via
    ``env_overrides`` -- still win for evaluation.
    """
    recipe = RECIPES[env_name]
    kwargs = dict(recipe.env_kwargs)
    if base_env_overrides:
        kwargs.update(base_env_overrides)
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
    config_file: str | Path | None = None,
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
    config_file:
        Optional path to a TOML run-configuration file
        (docs/run_config_file_spec.md). Its ``[train]`` table sits
        between the recipe and the ``quick_test`` presets (mapping
        fields like ``model_kwargs`` deep-merge one level; everything
        else replaces); its ``[env]`` table reaches both the training
        and evaluation environments below the recipe's evaluation
        overrides; its ``[eval_env]`` table wins last for evaluation.
        Always explicit -- nothing is auto-discovered -- and the parsed
        file (path, sha256, content) rides on the returned config so
        the training artifacts can record and copy it. Supplying a file
        also eagerly constructs and closes one training and one
        evaluation environment so a typo'd env kwarg fails here, in
        seconds, instead of mid-``train()``.
    **overrides:
        Any other ``TrainConfig`` field (``eval_freq``, ``n_envs``,
        ``model_kwargs``, ...). Explicit overrides replace wholesale
        and win over every other layer.
    """
    if env_name not in RECIPES:
        raise KeyError(f"Unknown env '{env_name}'. Choose one of {sorted(RECIPES)}.")
    if "run_config_file" in overrides:
        raise ValueError(
            "run_config_file is set by build_train_config itself and "
            "records provenance; pass config_file=<path> instead"
        )
    if config_file is not None:
        clashing = sorted({"env_fn", "eval_env_fn"} & set(overrides))
        if clashing:
            raise ValueError(
                f"config_file cannot be combined with explicit "
                f"{clashing} overrides: the factories would silently "
                f"discard the file's [env]/[eval_env] tables while "
                f"config.json records the file as applied. Drop the "
                f"factory override or the config_file."
            )
    recipe = RECIPES[env_name]
    resolved_algo = recipe.default_algo if algo is None else algo
    # Fail on a typo'd algo name now, before any env or artifact work
    # (train() would otherwise only hit the registry after the eager
    # env probe below has built and torn down real environments).
    resolve_algo(resolved_algo)

    file_config = None
    if config_file is not None:
        from courtside_dynamics.run_config import load_run_config

        file_config = load_run_config(config_file)

    cfg_kwargs: dict[str, Any] = {
        "env_fn": make_env_fn(
            env_name,
            env_overrides=(file_config.env if file_config else None),
        ),
        "eval_env_fn": make_eval_env_fn(
            env_name,
            base_env_overrides=(file_config.env if file_config else None),
            env_overrides=(file_config.eval_env if file_config else None),
        ),
        "recipe_name": env_name,
        "algo": resolved_algo,
        "log_dir": log_dir,
        "name_prefix": f"{recipe.name_prefix}_{resolved_algo.lower()}",
        "total_timesteps": recipe.default_total_timesteps,
    }
    cfg_kwargs.update(recipe.extra_cfg)

    if file_config is not None:
        from courtside_dynamics.run_config import merge_train_overrides

        cfg_kwargs = merge_train_overrides(cfg_kwargs, file_config.train)
        cfg_kwargs["run_config_file"] = file_config

    if quick_test:
        cfg_kwargs.update(_QUICK_TEST_OVERRIDES)

    # Explicit caller choices are applied last so they always win --
    # including over the quick-test presets and a config file.
    # (``total_timesteps`` used to be silently discarded under
    # ``quick_test=True``, unlike every other override, which made
    # "quick test but a bit longer" impossible.)
    if total_timesteps is not None:
        cfg_kwargs["total_timesteps"] = total_timesteps
    cfg_kwargs.update(overrides)

    # Validate the merged model_kwargs (recipe extra_cfg, TOML deep-merge,
    # and explicit overrides alike) against the resolved algorithm's
    # constructor before the env probe: a cross-algorithm key would
    # otherwise TypeError only inside train() after the vectorised env
    # fleet is built, and a string ent_coef on PPO even survives
    # construction, crashing a full rollout into the run.
    merged_model_kwargs = cfg_kwargs.get("model_kwargs")
    if merged_model_kwargs:
        validate_model_kwargs(resolved_algo, merged_model_kwargs)

    cfg = TrainConfig(**cfg_kwargs)
    if file_config is not None:
        # Fail on a typo'd [env]/[eval_env] kwarg now, in seconds,
        # rather than mid-train() after loggers and callbacks spin up.
        for factory in (cfg.env_fn, cfg.eval_env_fn):
            if factory is not None:
                factory().close()
    return cfg
