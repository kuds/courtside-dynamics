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
from pathlib import Path
from typing import Any

from courtside_dynamics.envs import (
    HUMANOID_TENNIS_OBSERVATION_LAYOUT,
    BallBalanceEnv,
    BallBounceEnv,
    HumanoidTennisCoopEnv,
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
        "normalize_obs_excluded_indices": (
            _HUMANOID_TENNIS_NORMALIZATION_EXCLUSIONS
        ),
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
        "info_eval_terminal_keys": (
            _HUMANOID_TENNIS_CURRICULUM_TERMINAL_EVAL_KEYS
        ),
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
            "HISTORICAL (superseded by WallBallDepthCurriculum): the "
            "cold-start problem this recipe targets was solved by "
            "auto-entropy before it ever trained (lessons_learned.md "
            "lesson 5), and its reward package bundles the since-"
            "falsified weak-return retry. Kept for the record; use "
            "WallBallDepthCurriculum for gated-curriculum training."
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
            # game is volleying, while from the deep stages every ball
            # has bounced long before it arrives, so one-bounce baseline
            # play emerges from geometry instead of a fault taxonomy.
            # None of the one_bounce-era shaping (first_hit_bonus,
            # weak_return_penalty, recovery fragments,
            # recoverable_bounce_*) carries over.
            "rally_style": "open",
            # The action mapping stays pinned to the full physical
            # workspace for the entire run so action semantics never
            # drift across stages; each stage confines *position* with
            # paddle_x_fence instead.
            "paddle_home_x": -1.7,
            "paddle_x_target_range": (-4.7, 0.3),
            # Stage-0 geometry — volley range, where striking is easy.
            # The full five-stage ladder below was calibrated by
            # tools/depth_stage_sweep.py (2026-07-20, 200 episodes per
            # cell): at every stage parked < crude full-swing <
            # charge-and-lead oracle strictly, the oracle completes >=2
            # returns from 93-97% of serves, and the placement-blind
            # crude policy still completes a second exchange in 66-83%
            # of episodes — the learnability bar the proven baseline
            # geometry set (lesson 9).
            "paddle_x_fence": (-2.7, 0.3),
            "paddle_start_x": -1.6,
            # Serve energy co-moves with depth via the gate (5.2 -> 7.0)
            # so the ball reliably reaches the deeper paddle; the flat
            # serve (lob 0) keeps contact at face height — lofted serves
            # sail over the fixed-height face (sweep iteration 1).
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
            # final-eval machinery; the fixed goal task is also the
            # yardstick that stays comparable across runs. Must equal
            # the last performance_gate stage (pinned by test).
            "paddle_x_fence": (-4.7, -1.2),
            "paddle_start_x": -3.9,
            "serve_speed": 7.0,
        },
        default_total_timesteps=3_000_000,
        name_prefix="wall_ball_depth_curriculum",
        extra_cfg={
            "n_envs": 8,
            "early_stop_patience": 20,
            # model_kwargs deliberately empty: SB3 defaults (auto
            # entropy, gamma 0.99) are what every run that learned the
            # task used (lesson 5); the campaign's capacity and reward
            # probes both came back null, so nothing else is earned.
            "success_key": "bounce_count",
            # Success mirrors the gate bar: a sustained three-exchange
            # rally is what earns the next depth stage.
            "success_threshold": 3.0,
            "headline_key": "bounce_count",
            "best_metric_keys": (
                "bounce_count_ep_mean",
                "bounce_count_ep_ge_5_rate",
            ),
            "best_metric_min_delta": 0.5 / 30,
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
                "stages": (
                    {
                        "paddle_x_fence": (-2.7, 0.3),
                        "paddle_start_x": -1.6,
                        "serve_speed": 5.2,
                    },
                    {
                        "paddle_x_fence": (-3.2, -0.6),
                        "paddle_start_x": -2.1,
                        "serve_speed": 5.5,
                    },
                    {
                        "paddle_x_fence": (-3.7, -1.0),
                        "paddle_start_x": -2.7,
                        "serve_speed": 6.0,
                    },
                    {
                        "paddle_x_fence": (-4.2, -1.2),
                        "paddle_start_x": -3.3,
                        "serve_speed": 6.5,
                    },
                    {
                        "paddle_x_fence": (-4.7, -1.2),
                        "paddle_start_x": -3.9,
                        "serve_speed": 7.0,
                    },
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
            "Earn your way back to the baseline: open-scoring wall "
            "rallies with a performance-gated depth ladder that walks "
            "the paddle fence from volley range to the workspace "
            "baseline each time a sustained three-exchange rally is "
            "demonstrated."
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
        extra_cfg=_tennis_curriculum_extra_cfg(
            video_length=150, early_stop_patience=8
        ),
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
        raise KeyError(
            f"Unknown env '{env_name}'. Choose one of {sorted(RECIPES)}."
        )
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
