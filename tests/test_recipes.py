"""Tests for the recipe registry that drives the consolidated notebook.

``build_train_config`` is the single entry point the training notebook
uses, so a broken recipe (bad env kwargs, an ``extra_cfg`` key that
``TrainConfig`` doesn't accept, a stale ``info_row_fn``) breaks every
run of that env. These tests materialize each recipe for real.
"""
from __future__ import annotations

import numpy as np
import pytest

from courtside_dynamics.recipes import (
    RECIPES,
    build_train_config,
    make_env_fn,
    make_eval_env_fn,
)


@pytest.mark.parametrize("env_name", sorted(RECIPES))
def test_build_train_config_materializes(env_name, tmp_path):
    """Every recipe must produce a valid TrainConfig whose env factory
    builds an env of the recipe's class."""
    cfg = build_train_config(env_name, algo="SAC", log_dir=str(tmp_path))
    assert cfg.algo == "SAC"
    assert cfg.log_dir == str(tmp_path)
    assert cfg.total_timesteps == RECIPES[env_name].default_total_timesteps
    assert cfg.name_prefix.endswith("_sac")
    assert cfg.recipe_name == env_name
    assert cfg.eval_env_fn is not None

    env = cfg.env_fn()
    try:
        assert isinstance(env, RECIPES[env_name].env_cls)
        assert env.render_mode == "rgb_array"
    finally:
        env.close()


def test_quick_test_overrides_shrink_budget(tmp_path):
    cfg = build_train_config(
        "BallBalance", log_dir=str(tmp_path), quick_test=True
    )
    assert cfg.total_timesteps == 25_000
    assert cfg.eval_freq == 5_000


def test_explicit_overrides_win_over_quick_test(tmp_path):
    """Caller overrides are applied last, on top of quick-test values."""
    cfg = build_train_config(
        "BallBalance",
        log_dir=str(tmp_path),
        quick_test=True,
        total_timesteps=123,
        n_envs=2,
    )
    assert cfg.total_timesteps == 123
    assert cfg.n_envs == 2


def test_unknown_env_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        build_train_config("NoSuchEnv", log_dir=str(tmp_path))


def test_make_env_fn_returns_fresh_instances():
    factory = make_env_fn("BallBalance")
    a, b = factory(), factory()
    try:
        assert a is not b
        assert type(a) is type(b)
    finally:
        a.close()
        b.close()


def test_make_env_fn_override_isolated_from_training_factory_and_recipe():
    """A post-training horizon override must not alter training semantics."""
    recipe_kwargs = dict(RECIPES["WallBall"].env_kwargs)
    training_factory = make_env_fn("WallBall")
    long_factory = make_env_fn("WallBall", env_overrides={"episode_len": 5_000})

    training_env = training_factory()
    long_env = long_factory()
    try:
        assert training_env.episode_len == 750
        assert long_env.episode_len == 5_000
        assert training_env.observation_space == long_env.observation_space
        assert training_env.action_space == long_env.action_space
        assert RECIPES["WallBall"].env_kwargs == recipe_kwargs
    finally:
        training_env.close()
        long_env.close()


def test_make_eval_env_fn_layers_recipe_and_caller_overrides():
    recipe_eval_overrides = dict(
        RECIPES["WallBallBaseline"].eval_env_overrides
    )
    factory = make_eval_env_fn(
        "WallBallBaseline", env_overrides={"episode_len": 5_000}
    )

    env = factory()
    try:
        assert env.episode_len == 5_000
        assert env.recovery_reset_probability == 0.0
        assert RECIPES["WallBallBaseline"].eval_env_overrides == (
            recipe_eval_overrides
        )
    finally:
        env.close()


def test_wall_ball_run_config_records_constructor_settings(tmp_path):
    """Metrics can detect physics/reward recipe drift after training."""
    import json

    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config("WallBall", log_dir=str(tmp_path))
    config_path = write_run_config(cfg, str(tmp_path))
    with open(config_path) as handle:
        payload = json.load(handle)
    kwargs = payload["env"]["constructor_kwargs"]

    assert payload["recipe_name"] == "WallBall"
    assert payload["evaluation_env"]["constructor_kwargs"] == kwargs
    assert kwargs["episode_len"] == 750
    assert kwargs["min_force"] == 20.0
    assert kwargs["serve_vy_max"] == 2.6
    assert kwargs["render_mode"] == "rgb_array"


def test_wall_ball_baseline_run_config_records_curriculum_schedule(tmp_path):
    import json

    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config("WallBallBaseline", log_dir=str(tmp_path))
    config_path = write_run_config(cfg, str(tmp_path))
    with open(config_path) as handle:
        payload = json.load(handle)

    assert payload["env"]["constructor_kwargs"][
        "recovery_reset_probability"
    ] == 0.6
    assert payload["evaluation_env"]["constructor_kwargs"][
        "recovery_reset_probability"
    ] == 0.0
    assert payload["train_config"]["env_attr_schedules"] == [
        {
            "attr_name": "recovery_reset_probability",
            "start_value": 0.6,
            "end_value": 0.15,
            "hold_until_timesteps": 250_000,
            "end_timesteps": 750_000,
        }
    ]


@pytest.mark.parametrize(
    "env_name",
    [
        "BallBounce",
        "WallBall",
        "WallBallVolley",
        "WallBallBaseline",
        "WallBallDepthCurriculum",
        "WallBallDepthCurriculumAligned",
    ],
)
def test_contact_envs_wire_a_success_metric(env_name, tmp_path):
    """Both contact-driven envs define ``success_key`` so eval runs log
    ``eval_info/success_rate``, and the key must exist in the env's
    ``info`` dict (a typo'd key silently omits the metric)."""
    import numpy as np

    cfg = build_train_config(env_name, log_dir=str(tmp_path))
    assert cfg.success_key == "bounce_count"

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert cfg.success_key in info
    finally:
        env.close()


def test_ball_bounce_recipe_requires_sustained_valid_bounces(tmp_path):
    cfg = build_train_config("BallBounce", log_dir=str(tmp_path))

    assert cfg.success_key == "bounce_count"
    assert cfg.success_threshold == 10.0
    assert cfg.headline_key == "bounce_count"
    assert cfg.info_eval_keys == (
        "bounce_count",
        "contact_episode_count",
        "touch_sensor",
        "valid_bounce",
    )
    assert cfg.info_eval_terminal_keys == (
        "term_ball_dropped",
        "term_nonfinite",
        "term_timeout",
    )
    assert cfg.info_eval_distribution_keys == ("bounce_count",)


def test_ball_bounce_info_row_matches_header(tmp_path):
    """The custom CSV row must flatten to exactly the header width."""
    import numpy as np

    from courtside_dynamics.callbacks.video_record import _flatten_row

    cfg = build_train_config("BallBounce", log_dir=str(tmp_path))
    assert cfg.csv_header is not None and cfg.info_row_fn is not None

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, reward, _, done, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        row = _flatten_row(
            cfg.info_row_fn(info, float(reward), float(reward), bool(done))
        )
        assert len(row) == len(list(cfg.csv_header))
    finally:
        env.close()


def test_paddle_tennis_recipe_carries_reward_decomposition_at_eval(tmp_path):
    """Review 20260823 §1.6: the nine ``rew_*`` components must reach
    eval_info.csv. They ride ``info_eval_keys``, where the eval
    callback's ``rew_`` prefix convention sums each one per episode
    into ``<key>_ep_sum_mean``; a typo'd key silently vanishes from the
    artifacts, so each key must also exist in the env's ``info`` dict."""
    import numpy as np

    components = (
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
    cfg = build_train_config("PaddleTennis", log_dir=str(tmp_path))
    assert cfg.info_eval_keys is not None
    for key in components:
        assert key in cfg.info_eval_keys

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        for key in components:
            assert key in info
    finally:
        env.close()


def test_humanoid_tennis_smoke_recipe_has_compact_recording_schema(tmp_path):
    """Phase 3 records rally diagnostics without flattening every rule key."""
    import numpy as np

    from courtside_dynamics.callbacks._info import _scalar_info_keys
    from courtside_dynamics.callbacks.video_record import _flatten_row

    recipe_name = "HumanoidTennisCoopSmoke"
    recipe = RECIPES[recipe_name]
    cfg = build_train_config(recipe_name, log_dir=str(tmp_path))

    assert "smoke" in recipe.description.lower()
    assert "not a full-tennis training preset" in recipe.description.lower()
    assert cfg.n_envs == 1
    assert cfg.total_timesteps == 10_000
    assert cfg.success_key == "rally_target_reached"
    assert cfg.phase_key == "rally_phase"
    assert cfg.normalize_obs_excluded_indices == tuple(range(193, 299))
    assert cfg.info_eval_keys == (
        "rally_count",
        "valid_return_rate",
        "legal_hit_count",
    )
    assert cfg.info_eval_distribution_keys == ("rally_count",)
    assert "term_ball_net" in cfg.info_eval_terminal_keys
    assert "term_second_bounce" in cfg.info_eval_terminal_keys
    assert "term_out_of_bounds" in cfg.info_eval_terminal_keys

    assert cfg.csv_header is not None and cfg.info_row_fn is not None
    header = list(cfg.csv_header)
    assert len(header) < 30
    assert header[-3:] == ["reward", "total_reward", "done"]
    assert {
        "initial_ball_x",
        "initial_ball_y",
        "initial_ball_z",
        "initial_ball_vx",
        "initial_ball_vy",
        "initial_ball_vz",
    } <= set(header)

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        row = _flatten_row(
            cfg.info_row_fn(
                info,
                float(reward),
                float(reward),
                bool(terminated or truncated),
            )
        )
        assert len(row) == len(header)
        assert len(_scalar_info_keys(info)) > len(header)
    finally:
        env.close()


@pytest.mark.parametrize(
    (
        "recipe_name",
        "stage",
        "active_count",
        "episode_len",
        "patience",
        "hit_shaping",
    ),
    [
        ("HumanoidTennisStage0Intercept", 0, 2, 150, 8, 0.0),
        ("HumanoidTennisStage1AnchoredReturn", 1, 7, 300, 12, 0.25),
        ("HumanoidTennisStage2RandomizedReturn", 2, 7, 300, 20, 0.25),
    ],
)
def test_humanoid_tennis_curriculum_recipes_are_fixed_stage_and_recordable(
    recipe_name,
    stage,
    active_count,
    episode_len,
    patience,
    hit_shaping,
    tmp_path,
):
    import numpy as np

    from courtside_dynamics.callbacks.video_record import _flatten_row

    recipe = RECIPES[recipe_name]
    cfg = build_train_config(recipe_name, log_dir=str(tmp_path))
    assert cfg.n_envs == 1
    assert cfg.algo == "PPO"
    assert cfg.success_key == "stage_success"
    assert cfg.normalize_obs_excluded_indices == tuple(range(193, 299))
    # Task-metric selection, no return normalization on the sparse
    # hand-scaled reward, gSDE exploration with a small entropy floor,
    # and per-stage patience scaled to each eval budget (earliest stop
    # is eval 2 * patience).
    assert cfg.headline_key == "stage_success"
    assert cfg.best_metric_keys == (
        "stage_success_ep_mean",
        "success_rate",
        "legal_hit_count_ep_mean",
        "episode_reward_mean",
    )
    assert cfg.confirm_best_eval is True
    assert cfg.reward_eval_episodes == 5
    assert cfg.normalize_reward is False
    assert cfg.model_kwargs == {"use_sde": True, "ent_coef": 0.01}
    assert cfg.early_stop_patience == patience
    assert "experimental" in recipe.description.lower()
    assert "convergence is not claimed" in recipe.description.lower()
    assert cfg.csv_header is not None and cfg.info_row_fn is not None

    env = cfg.env_fn()
    try:
        _, reset_info = env.reset(seed=0)
        assert env.episode_len == episode_len
        assert env.reward_config.valid_hit_shaping == pytest.approx(
            hit_shaping
        )
        assert reset_info["curriculum_stage"] == stage
        assert reset_info["active_action_count"] == active_count
        _, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        row = _flatten_row(
            cfg.info_row_fn(
                info,
                float(reward),
                float(reward),
                bool(terminated or truncated),
            )
        )
        assert len(row) == len(list(cfg.csv_header))
        assert {
            "curriculum_stage",
            "stage_success",
            "target_hit",
            "term_target_miss",
        } <= set(cfg.csv_header)
    finally:
        env.close()


def test_curriculum_recipe_allows_explicit_sac_despite_ppo_default(tmp_path):
    cfg = build_train_config(
        "HumanoidTennisStage0Intercept",
        algo="SAC",
        log_dir=str(tmp_path),
    )
    assert cfg.algo == "SAC"
    assert cfg.name_prefix.endswith("_sac")


def test_build_train_config_rejects_unknown_algo_before_env_work(tmp_path):
    with pytest.raises(ValueError, match="Unknown algo"):
        build_train_config("BallBalance", algo="PPPO", log_dir=str(tmp_path))


def test_build_train_config_rejects_cross_algo_model_kwargs(tmp_path):
    # WallBallBootstrap's recipe bundle carries SAC-only keys
    # (learning_starts, buffer_size, target_entropy); flipping the algo
    # must fail here, not inside train() after the env fleet is built.
    with pytest.raises(ValueError, match="not accepted by PPO"):
        build_train_config(
            "WallBallBootstrap", algo="PPO", log_dir=str(tmp_path)
        )


def test_build_train_config_rejects_string_ent_coef_for_ppo(tmp_path):
    # A string ent_coef survives PPO *construction* and only crashes at
    # the first gradient update, a full rollout into the run.
    with pytest.raises(ValueError, match="numeric ent_coef"):
        build_train_config(
            "BallBalance",
            algo="PPO",
            log_dir=str(tmp_path),
            model_kwargs={"ent_coef": "auto_0.02"},
        )


def test_build_train_config_model_kwargs_error_suggests_close_match(tmp_path):
    with pytest.raises(ValueError, match="did you mean 'learning_rate'"):
        build_train_config(
            "BallBalance",
            log_dir=str(tmp_path),
            model_kwargs={"learning_rte": 3e-4},
        )


def test_curriculum_recipe_writes_stage_provenance(tmp_path):
    import json

    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config(
        "HumanoidTennisStage0Intercept",
        log_dir=str(tmp_path),
    )
    path = write_run_config(cfg, str(tmp_path))
    with open(path) as config_file:
        payload = json.load(config_file)
    curriculum = payload["env"]["curriculum"]
    assert curriculum["curriculum_stage"] == 0
    assert curriculum["curriculum_stage_name"] == "anchored_racket_intercept"
    assert curriculum["active_joint_names"] == [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
    ]
    assert curriculum["launch_position_noise"] == [0.0, 0.0, 0.0]
    assert curriculum["launch_speed"] > 0.0
    assert curriculum["receiver_distance_from_net"] == 3.0
    assert curriculum["fault_strictness"] == "regulation"
    assert curriculum["curriculum_launch_overridden"] is False


def test_training_notebook_preserves_curriculum_recipe_defaults():
    import json
    from pathlib import Path

    notebook_path = Path(__file__).parents[1] / "notebooks" / "sb3_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", ())
    )

    assert 'ALGO = None' in source
    assert 'TOTAL_TIMESTEPS = None' in source
    assert 'REPO_REF = "main"' in source
    assert "github.com/kuds/courtside-dynamics@{REPO_REF}" in source
    assert 'ALGO = ALGO or RECIPES[ENV].default_algo' in source
    # 0.12.0 moved worker-count resolution into the recipes: the
    # notebook must not hand-resolve n_envs (that explicit kwarg would
    # silently beat a TOML config file's [train] table).
    assert "recipe_n_envs" not in source
    assert "if N_ENVS is not None:" in source
    assert "config_file=CONFIG_FILE" in source
    assert "cfg.checkpoint_freq =" not in source
    assert "cfg.video_freq =" not in source
    assert "HumanoidTennisStage0Intercept" in source
    assert "HumanoidTennisStage2RandomizedReturn" in source


@pytest.mark.parametrize(
    "env_name",
    [
        "WallBall",
        "WallBallVolley",
        "WallBallBaseline",
        "WallBallDepthCurriculum",
        "WallBallDepthCurriculumAligned",
        "WallBallGoalRally",
    ],
)
def test_wall_ball_headline_metric_is_rally_count(env_name, tmp_path):
    """WallBall names ``bounce_count`` as its headline metric:
    success_rate saturates once every eval episode completes a single
    exchange, and eval reward is dominated by tracking shaping, so runs
    are compared on rally exchanges per episode instead."""
    cfg = build_train_config(env_name, log_dir=str(tmp_path))
    assert cfg.headline_key == "bounce_count"
    assert cfg.phase_key == "rally_phase"
    assert cfg.phase_labels == {
        0: "await_bounce",
        1: "await_paddle",
        2: "await_wall",
    }


@pytest.mark.parametrize(
    "env_name",
    [
        "WallBall",
        "WallBallVolley",
        "WallBallBaseline",
        "WallBallDepthCurriculum",
        "WallBallDepthCurriculumAligned",
        "WallBallGoalRally",
    ],
)
def test_wall_ball_recipe_uses_simplified_paddle_interface(env_name, tmp_path):
    cfg = build_train_config(env_name, log_dir=str(tmp_path))
    env = cfg.env_fn()
    try:
        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (23,)
        assert env.model.nu == 3
    finally:
        env.close()


def test_wall_ball_recipe_preserves_original_open_setup():
    kwargs = RECIPES["WallBall"].env_kwargs
    assert "rally_style" not in kwargs
    assert "paddle_home_x" not in kwargs
    assert "paddle_x_target_range" not in kwargs
    assert "face-only" in RECIPES["WallBall"].description
    assert "fixed 10-degree upward pitch" in RECIPES["WallBall"].description


@pytest.mark.parametrize(
    ("env_name", "rally_style", "home_x", "target_range"),
    [
        ("WallBallVolley", "volley", -1.7, (-4.7, 0.3)),
        ("WallBallBaseline", "one_bounce", -2.7, (-3.2, -1.6)),
        # The depth recipes re-centre the pivot per stage (0.22.0), so
        # env_kwargs carries stage 0's midpoint rather than the shared
        # -1.7; the mapping range stays pinned to the full workspace.
        ("WallBallDepthCurriculum", "open", -1.25, (-4.7, 0.3)),
        (
            "WallBallDepthCurriculumAligned",
            "open",
            -1.25,
            (-4.7, 0.3),
        ),
    ],
)
def test_wall_ball_style_recipes_record_world_space_paddle_setup(
    env_name, rally_style, home_x, target_range
):
    kwargs = RECIPES[env_name].env_kwargs
    assert kwargs["rally_style"] == rally_style
    assert kwargs["paddle_home_x"] == home_x
    assert kwargs["paddle_x_target_range"] == target_range


def test_wall_ball_baseline_softens_early_touch():
    """The widened lane puts most of the serve's bounce footprint inside
    the paddle's reach, so the baseline recipe must soften the
    paddle_before_bounce trap into a non-terminal fine."""
    kwargs = RECIPES["WallBallBaseline"].env_kwargs
    assert kwargs["early_touch_penalty"] == 0.25


def test_wall_ball_baseline_slows_the_paddle():
    """The widened lane only works with the slower swing cap (and vice
    versa): the 2026-07-16 sweep measured either change alone collapsing
    oracle second returns to ~50%. Only the baseline overrides damping;
    open/volley keep the shared XML calibration."""
    assert RECIPES["WallBallBaseline"].env_kwargs["paddle_joint_damping"] == 8.0
    assert "paddle_joint_damping" not in RECIPES["WallBall"].env_kwargs
    assert "paddle_joint_damping" not in RECIPES["WallBallVolley"].env_kwargs


def test_wall_ball_bootstrap_recipe_gates_nested_serve_distributions():
    """The bootstrap ladder must be nested (each stage a subset of the
    final serve distribution), start at the constructor's stage-0 values,
    end exactly at the canonical eval serve, and never touch the no-op
    invariant's serve_vy_min."""
    recipe = RECIPES["WallBallBootstrap"]
    gate = recipe.extra_cfg["performance_gate"]
    stages = gate["stages"]

    vy = [stage["serve_vy_max"] for stage in stages]
    jitter = [stage["serve_speed_jitter"] for stage in stages]
    assert vy == sorted(vy) and jitter == sorted(jitter)
    assert stages[0]["serve_vy_max"] == recipe.env_kwargs["serve_vy_max"]
    assert (
        stages[0]["serve_speed_jitter"]
        == recipe.env_kwargs["serve_speed_jitter"]
    )
    assert stages[-1]["serve_vy_max"] == (
        recipe.eval_env_overrides["serve_vy_max"]
    )
    assert stages[-1]["serve_speed_jitter"] == (
        recipe.eval_env_overrides["serve_speed_jitter"]
    )
    assert all("serve_vy_min" not in stage for stage in stages)
    assert recipe.env_kwargs["serve_vy_min"] == 0.8

    # Bootstrap reward package + whole-task resets.
    assert recipe.env_kwargs["first_hit_bonus"] == 0.25
    assert recipe.env_kwargs["weak_return_penalty"] == 0.1
    assert recipe.env_kwargs["early_touch_penalty"] == 0.25
    assert recipe.env_kwargs["recovery_reset_probability"] == 0.0
    assert recipe.extra_cfg["final_info_eval"] is True
    assert recipe.extra_cfg["model_kwargs"]["ent_coef"] == "auto_0.02"


@pytest.mark.parametrize(
    "recipe_name",
    ("WallBallDepthCurriculum", "WallBallDepthCurriculumAligned"),
)
def test_depth_curriculum_config_json_records_full_gate(
    recipe_name, tmp_path
):
    """config.json's structured gate block must record the gate as run:
    run 20260721_142121's promotion_rule/pause/clear were active (recipe
    defaults) but provably absent from its artifact."""
    import json

    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config(recipe_name, log_dir=str(tmp_path))
    path = write_run_config(cfg, str(tmp_path))
    with open(path) as handle:
        gate = json.load(handle)["train_config"]["performance_gate"]
    assert gate["sustain_evals"] == 3
    assert gate["promotion_rule"] == "window_mean"
    assert gate["advance_update_pause_steps"] == 50_000
    assert gate["clear_replay_buffer_on_advance"] is True
    assert len(gate["stages"]) == 5


def test_wall_ball_bootstrap_config_builds_and_records_gate(tmp_path):
    cfg = build_train_config("WallBallBootstrap", log_dir=str(tmp_path))
    assert cfg.performance_gate is not None
    assert cfg.performance_gate["metric_key"] == "bounce_count_ep_mean"
    assert cfg.final_info_eval is True
    env = cfg.env_fn()
    try:
        assert env.serve_vy_max == 1.1
        assert env.first_hit_bonus == 0.25
    finally:
        env.close()
    eval_env = cfg.eval_env_fn()
    try:
        assert eval_env.serve_vy_max == 2.0
    finally:
        eval_env.close()


def test_baseline_oracle_defaults_track_recipe_lane():
    """wall_ball_baseline_oracle_action plans within -- and inverts the
    action mapping of -- the lane its defaults describe; a recipe lane
    change that forgets the oracle would silently mis-calibrate every
    solvability gate."""
    import inspect

    from courtside_dynamics.scripted_policies import (
        wall_ball_baseline_oracle_action,
    )

    signature = inspect.signature(wall_ball_baseline_oracle_action)
    kwargs = RECIPES["WallBallBaseline"].env_kwargs
    assert (
        signature.parameters["paddle_x_target_range"].default
        == kwargs["paddle_x_target_range"]
    )
    assert (
        signature.parameters["paddle_home_x"].default
        == kwargs["paddle_home_x"]
    )


def test_wall_ball_baseline_recipe_uses_calibrated_bounce_first_serve():
    kwargs = RECIPES["WallBallBaseline"].env_kwargs
    assert kwargs["serve_speed"] == 5.5
    assert kwargs["serve_speed_jitter"] == 0.5
    assert kwargs["serve_lob"] == 0.0
    assert kwargs["serve_vy_min"] == 0.8
    assert kwargs["serve_vy_max"] == 2.0


def test_wall_ball_baseline_reports_multi_return_survival(tmp_path):
    cfg = build_train_config("WallBallBaseline", log_dir=str(tmp_path))

    assert cfg.success_threshold == 2.0
    assert cfg.info_eval_survival_thresholds == {
        "bounce_count": (2, 3, 5),
    }
    assert cfg.env_attr_schedules == (
        {
            "attr_name": "recovery_reset_probability",
            "start_value": 0.6,
            "end_value": 0.15,
            "hold_until_timesteps": 250_000,
            "end_timesteps": 750_000,
        },
    )


def test_wall_ball_baseline_combines_recovery_curriculum_and_reward():
    kwargs = RECIPES["WallBallBaseline"].env_kwargs

    assert kwargs["recovery_reset_probability"] == 0.6
    assert kwargs["post_bounce_reset_fraction"] == 0.5
    assert kwargs["recoverable_bounce_bonus"] == 0.25
    assert kwargs["recoverable_bounce_lateral_limit"] == 2.0


def test_wall_ball_baseline_reverted_to_proven_configuration():
    """Run 20260718_213222 falsified the 0.13.0 bundle: the asymmetric
    weak-return retry taught soft, unchainable returns (1.33 vs 3.23
    bounces). The recipe pins the configuration both reference runs
    learned with -- terminal weak returns (symmetric with OOB) and SB3
    default gamma -- keeping only the ge_5 selection tiebreak."""
    recipe = RECIPES["WallBallBaseline"]
    assert "weak_return_penalty" not in recipe.env_kwargs
    assert "weak_return_penalty" not in recipe.eval_env_overrides
    assert "model_kwargs" not in recipe.extra_cfg
    assert recipe.extra_cfg["best_metric_keys"] == (
        "bounce_count_ep_mean",
        "bounce_count_ep_ge_5_rate",
    )

    for factory in (make_env_fn("WallBallBaseline"),
                    make_eval_env_fn("WallBallBaseline")):
        env = factory()
        try:
            assert env.weak_return_penalty is None
        finally:
            env.close()


def test_selection_survival_keys_are_backed_by_thresholds():
    """A best_metric_keys entry like bounce_count_ep_ge_5_rate only
    exists because 5 is in info_eval_survival_thresholds; if the two
    ever decouple, InfoDictEvalCallback scores the missing key as -inf
    and the tiebreaker silently dies. Enforce the coupling for every
    recipe."""
    import re

    for name, recipe in RECIPES.items():
        keys = recipe.extra_cfg.get("best_metric_keys", ())
        thresholds = recipe.extra_cfg.get("info_eval_survival_thresholds", {})
        for key in keys:
            match = re.fullmatch(r"(.+)_ep_ge_(\d+)_rate", key)
            if match is None:
                continue
            base, bar = match.group(1), int(match.group(2))
            assert bar in tuple(thresholds.get(base, ())), (
                f"{name}: selection key {key!r} needs {bar} in "
                f"info_eval_survival_thresholds[{base!r}]"
            )


def test_wall_ball_depth_curriculum_walks_the_fence_back():
    """The calibrated ladder slides toward baseline without a common
    front-court refuge while adjacent stages remain connected."""
    recipe = RECIPES["WallBallDepthCurriculum"]
    gate = recipe.extra_cfg["performance_gate"]
    stages = gate["stages"]

    assert len(stages) == 5
    for key, value in stages[0].items():
        assert recipe.env_kwargs[key] == value

    fences = [stage["paddle_x_fence"] for stage in stages]
    assert fences == [
        (-2.3, -0.2),
        (-2.9, -0.8),
        (-3.5, -1.4),
        (-4.1, -2.0),
        (-4.7, -2.6),
    ]
    backs = [back for back, _ in fences]
    fronts = [front for _, front in fences]
    starts = [stage["paddle_start_x"] for stage in stages]
    serve_origins = [
        stage.get("serve_start_x", recipe.env_kwargs["serve_start_x"])
        for stage in stages
    ]
    speeds = [stage["serve_speed"] for stage in stages]
    assert backs == sorted(backs, reverse=True)
    assert fronts == sorted(fronts, reverse=True)
    assert starts == [-1.6, -2.1, -2.7, -3.3, -3.9]
    assert serve_origins == [1.0] * 5
    assert speeds == [5.2, 5.5, 6.0, 6.5, 7.0]
    # Interval intersection is non-empty iff max(lower) <= min(upper).
    assert max(backs) > min(fronts)
    for current, following in zip(fences[:-1], fences[1:], strict=True):
        assert max(current[0], following[0]) < min(
            current[1], following[1]
        )
    for stage in stages:
        back, front = stage["paddle_x_fence"]
        assert back <= stage["paddle_start_x"] <= front

    mapping = recipe.env_kwargs["paddle_x_target_range"]
    assert mapping == (-4.7, 0.3)
    assert stages[-1]["paddle_x_fence"][0] == mapping[0]

    assert gate["metric_key"] == "bounce_count_ep_mean"
    assert gate["threshold"] == 3.0
    assert gate["sustain_evals"] == 3
    # Run-1 gate refinements (see the recipe comment): the bar itself is
    # untouched, but promotion reads the 2-eval mean (stage 2 cleared
    # 3.0 four separate times without two-in-a-row), and every advance
    # runs the warm-up package -- clear the stale-stage buffer, pause
    # updates for 50k fresh frontier steps.
    assert gate["promotion_rule"] == "window_mean"
    assert gate["advance_update_pause_steps"] == 50_000
    assert gate["clear_replay_buffer_on_advance"] is True
    # Run 20260727_004014 stalled on stage 3 with ent_coef at 0.0011;
    # every advance had been handing new geometry a deterministic policy.
    assert gate["reset_entropy_on_advance"] is True
    assert recipe.extra_cfg["final_info_eval"] is True
    assert recipe.extra_cfg["reward_eval_episodes"] == 5


def test_wall_ball_depth_curriculum_keeps_runway_and_pivot_at_every_stage():
    """Return pace comes from paddle speed at contact, and speed needs
    runway. The pre-0.22.0 ladder narrowed the fence 3.0 -> 1.7 m while
    receding, leaving 0.9 m of travel at the goal -- a probe sweep scored
    0 completed returns below 0.4 m, 1 at 0.6-0.9 m and 2-3 at 1.2-1.6 m,
    and run 20260727_004014 duly returned the serve in 100% of audited
    episodes and a second ball in 10%. Hold the width constant, keep
    every stage above the 1.2 m knee, and keep the action map's pivot on
    the fence midpoint so the usable action share stops collapsing.
    """
    for name in ("WallBallDepthCurriculum", "WallBallDepthCurriculumAligned"):
        recipe = RECIPES[name]
        stages = recipe.extra_cfg["performance_gate"]["stages"]
        widths = {
            round(front - back, 6)
            for back, front in (s["paddle_x_fence"] for s in stages)
        }
        assert widths == {2.1}, f"{name} fence width drifts: {widths}"
        for index, stage in enumerate(stages):
            back, front = stage["paddle_x_fence"]
            runway = front - stage["paddle_start_x"]
            assert runway >= 1.2, f"{name} stage {index} runway {runway}"
            midpoint = (back + front) / 2.0
            assert stage["paddle_home_x"] == pytest.approx(midpoint), (
                f"{name} stage {index} pivot off the fence midpoint"
            )
        # The unsynced goal evaluator must stay equal to the last stage,
        # pivot included, or it scores a geometry no stage ever trained.
        last = stages[-1]
        overrides = recipe.eval_env_overrides
        for key in ("paddle_x_fence", "paddle_start_x", "paddle_home_x"):
            assert overrides[key] == last[key], f"{name} eval drift on {key}"


@pytest.mark.parametrize(
    "recipe_name",
    ("WallBallDepthCurriculum", "WallBallDepthCurriculumAligned"),
)
def test_wall_ball_depth_curriculum_uses_open_scoring_and_defaults(
    recipe_name,
):
    """Open scoring, emergent style: no one_bounce fault taxonomy or
    bootstrap shaping may leak in, and model_kwargs pins gamma ONLY.

    model_kwargs was empty until the exchange cadence was measured at
    117-135 env steps, which prices the next return at 0.99^130 ~ 0.27
    under SB3's default gamma. gamma 0.995 is now pinned; entropy must
    stay on SB3 auto, because the 20260717 A/B showed a fixed
    ent_coef 0.02 leaves the policy never touching the ball.
    """
    recipe = RECIPES[recipe_name]
    kwargs = recipe.env_kwargs

    assert kwargs["rally_style"] == "open"
    for banned in (
        "early_touch_penalty",
        "weak_return_penalty",
        "first_hit_bonus",
        "recovery_reset_probability",
        "recoverable_bounce_bonus",
        "recoverable_bounce_lateral_limit",
        "wall_reward_increment",
    ):
        assert banned not in kwargs, banned
    # gamma is pinned; nothing else may be. In particular ent_coef must
    # stay absent so SB3's auto entropy tuning survives.
    assert recipe.extra_cfg["model_kwargs"] == {"gamma": 0.995}
    # The gate decides promotion by crossing a threshold on a noisy
    # statistic, so its sample size is load-bearing: 30 episodes gave a
    # 3-eval window sd of ~0.092 against a bar these runs cleared by
    # 0.011. The goal-task stream stays pinned at 30 so eval cost does
    # not double with it.
    assert recipe.extra_cfg["n_eval_episodes"] == 60
    assert recipe.extra_cfg["final_eval_episodes"] == 30
    assert recipe.extra_cfg["best_metric_min_delta"] == pytest.approx(0.5 / 60)
    # Canonical scoring pins the ladder's FINAL stage: the final-config
    # evaluator, milestone videos, and the long-horizon audit build from
    # eval_env_overrides unsynced by the gate, so an empty dict would
    # score the easiest (stage-0) geometry for the whole run. The
    # matched evaluator is re-synced per stage by the gate regardless.
    last_stage = recipe.extra_cfg["performance_gate"]["stages"][-1]
    assert recipe.eval_env_overrides == dict(last_stage)
    assert recipe.extra_cfg["success_threshold"] == 3.0
    assert recipe.extra_cfg["best_metric_keys"] == (
        "bounce_count_ep_mean",
        "bounce_count_ep_ge_5_rate",
    )


@pytest.mark.parametrize(
    ("recipe_name", "final_serve_start_x"),
    (
        ("WallBallDepthCurriculum", 1.0),
        ("WallBallDepthCurriculumAligned", -0.35),
    ),
)
def test_wall_ball_depth_curriculum_config_builds_and_stages_apply(
    recipe_name, final_serve_start_x, tmp_path
):
    """The gate's stage dicts must be reachable, settable env attributes
    (a typo'd stage key would otherwise die mid-run), and the built
    config must carry the gate and the 6M budget."""
    cfg = build_train_config(recipe_name, log_dir=str(tmp_path))
    assert cfg.performance_gate is not None
    assert cfg.performance_gate["metric_key"] == "bounce_count_ep_mean"
    assert cfg.total_timesteps == 6_000_000
    assert cfg.final_info_eval is True

    env = cfg.env_fn()
    eval_env = cfg.eval_env_fn()
    try:
        assert env.rally_style == "open"
        assert env.paddle_x_fence == (-2.3, -0.2)
        assert env.serve_start_x == 1.0
        assert env.serve_speed == 5.2
        assert eval_env.serve_start_x == final_serve_start_x
        for key, value in cfg.performance_gate["stages"][0].items():
            setattr(eval_env, key, value)
        eval_env.reset(seed=0)
        assert eval_env.serve_start_x == 1.0
        for stage in cfg.performance_gate["stages"]:
            for key, value in stage.items():
                assert hasattr(env, key)
                setattr(env, key, value)
        env.reset(seed=0)
        assert env.paddle_x_fence == (-4.7, -2.6)
        assert env.serve_start_x == final_serve_start_x
        # Applying a stage must move the action map's pivot, not just
        # the public attribute: until 0.22.0 paddle_home_x was a plain
        # attribute, so a stage that set it left _control_home stale and
        # the gate advanced against an unchanged mapping.
        assert env.paddle_home_x == -3.65
        zero_action_x = (
            env._action_to_controls(np.zeros(3, dtype=np.float32))[0]
            + env._paddle_x_origin
        )
        assert zero_action_x == pytest.approx(-3.65)
    finally:
        env.close()
        eval_env.close()


def test_aligned_depth_curriculum_is_an_isolated_paired_treatment():
    baseline = RECIPES["WallBallDepthCurriculum"]
    aligned = RECIPES["WallBallDepthCurriculumAligned"]
    baseline_gate = baseline.extra_cfg["performance_gate"]
    aligned_gate = aligned.extra_cfg["performance_gate"]
    baseline_stages = baseline_gate["stages"]
    aligned_stages = aligned_gate["stages"]

    assert baseline.name_prefix == "wall_ball_depth_curriculum"
    assert aligned.name_prefix == "wall_ball_depth_curriculum_aligned"
    # Status marker updated 2026-08-02: the arm closed on the Phase D
    # no-go (was EXPERIMENTAL while the A/B was pending).
    assert "HISTORICAL" in aligned.description
    assert baseline.env_kwargs == aligned.env_kwargs
    assert baseline.env_kwargs is not aligned.env_kwargs
    assert baseline.extra_cfg is not aligned.extra_cfg
    assert baseline_gate is not aligned_gate
    assert baseline_stages is not aligned_stages
    assert baseline_stages[0] == aligned_stages[0]
    assert baseline_stages[0] is not aligned_stages[0]

    assert {
        key: value
        for key, value in baseline_gate.items()
        if key != "stages"
    } == {
        key: value
        for key, value in aligned_gate.items()
        if key != "stages"
    }
    for stage_index, (baseline_stage, aligned_stage) in enumerate(
        zip(baseline_stages, aligned_stages, strict=True)
    ):
        assert {
            key: value
            for key, value in aligned_stage.items()
            if key != "serve_start_x"
        } == {
            key: value
            for key, value in baseline_stage.items()
            if key != "serve_start_x"
        }
        assert baseline_stage["serve_start_x"] == 1.0
        if stage_index == 0:
            assert baseline_stage == aligned_stage

    baseline_origins = [
        stage.get("serve_start_x", baseline.env_kwargs["serve_start_x"])
        for stage in baseline_stages
    ]
    aligned_origins = [
        stage.get("serve_start_x", aligned.env_kwargs["serve_start_x"])
        for stage in aligned_stages
    ]
    assert baseline_origins == [1.0] * 5
    assert aligned_origins == [1.0, 0.69, 0.34, -0.01, -0.35]
    assert baseline.eval_env_overrides == dict(baseline_stages[-1])
    assert aligned.eval_env_overrides == dict(aligned_stages[-1])


def test_wall_ball_bootstrap_is_marked_historical():
    """Bootstrap is kept for the record but must say so: its cold-start
    problem was solved by auto-entropy before it ever ran, and its
    reward package bundles the falsified weak-return retry. The
    supersession pointer must name the CURRENT era recipes -- its
    original successor was itself retired."""
    description = RECIPES["WallBallBootstrap"].description
    assert "HISTORICAL" in description
    assert "WallBallGoalRally" in description
    assert "WallBallTrueBaseline" in description


def test_retired_depth_curriculum_recipes_are_marked_historical():
    """The 0.24.0 diagnosis retired the sliding-fence ladder and the
    Phase D no-go closed the aligned arm permanently, yet both
    descriptions read as live presets for months. Recipe descriptions
    are the browsing surface for the next campaign's template hunt --
    a retired preset must say so and point at the current era."""
    ladder = RECIPES["WallBallDepthCurriculum"].description
    assert "HISTORICAL" in ladder
    assert "WallBallGoalRally" in ladder

    aligned = RECIPES["WallBallDepthCurriculumAligned"].description
    assert "HISTORICAL" in aligned
    assert "closed" in aligned.lower()


def test_readme_lists_every_recipe():
    """The README's recipe list went two releases stale: 0.24.0's
    WallBallGoalRally and 0.25.0's WallBallTrueBaseline -- the recipes
    behind the campaign's headline results -- appeared nowhere in it.
    Pin the list so the next recipe cannot drift the same way."""
    from pathlib import Path

    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if not readme_path.is_file():
        pytest.skip("README.md not present (installed-package test run)")
    readme = readme_path.read_text(encoding="utf-8")
    missing = [name for name in RECIPES if f"`{name}`" not in readme]
    assert not missing, (
        f"recipes missing from README.md's recipe list: {missing}"
    )


def test_goal_rally_trains_the_goal_task_directly():
    """The 0.24.0 structural replacement for the depth ladder: the
    campaign-goal task IS the whole task. No curriculum, no advance
    package; the single-stage gate exists for artifact/certification
    parity with the gated corpus and can never promote."""
    recipe = RECIPES["WallBallGoalRally"]
    depth = RECIPES["WallBallDepthCurriculum"]
    goal = dict(depth.extra_cfg["performance_gate"]["stages"][-1])

    # Training geometry and serve ARE the campaign goal.
    for key, value in goal.items():
        assert recipe.env_kwargs[key] == value, key
    assert recipe.env_kwargs["serve_start_x"] == 1.0

    gate = recipe.extra_cfg["performance_gate"]
    stages = gate["stages"]
    assert [dict(s) for s in stages] == [goal]
    # Evaluation equals training equals the depth ladder's goal task,
    # so every historical goal number remains directly comparable.
    assert recipe.eval_env_overrides == dict(goal)
    assert recipe.eval_env_overrides == depth.eval_env_overrides

    # The 3.0 bar is informational on a single-stage gate (nothing to
    # promote to); it marks the campaign goal in gate_window_mean.
    assert gate["threshold"] == 3.0
    assert gate["promotion_rule"] == "window_mean"
    assert "clear_replay_buffer_on_advance" not in gate
    assert "advance_update_pause_steps" not in gate
    assert "reset_entropy_on_advance" not in gate
    assert "stage_eval_budget" not in gate

    # Startup certification sweeps the (one) training geometry with the
    # recalibrated lead-charge probe from the reserved 30000+ block.
    cert = recipe.extra_cfg["ladder_certification"]
    assert list(cert["oracle_probes"]) == [{"lead_charge": 3.0}]
    assert cert["seed_start"] >= 30_000
    assert cert["episodes"] == 30

    # Train == eval distribution: the duplicate final info stream is
    # off, and the 5-episode reward evaluator owns evaluations.npz.
    assert recipe.extra_cfg["final_info_eval"] is False
    assert recipe.extra_cfg["final_eval_episodes"] is None
    assert recipe.extra_cfg["reward_eval_episodes"] == 5

    assert recipe.extra_cfg["n_eval_episodes"] == 60
    assert recipe.extra_cfg["success_threshold"] == 3.0
    assert recipe.extra_cfg["model_kwargs"] == {"gamma": 0.995}


def test_goal_rally_config_builds_and_gate_stage_applies(tmp_path):
    cfg = build_train_config("WallBallGoalRally", log_dir=str(tmp_path))
    env = cfg.env_fn()
    eval_env = cfg.eval_env_fn()
    try:
        for e in (env, eval_env):
            assert e.unwrapped.paddle_x_fence == (-4.7, -2.6)
            assert e.unwrapped.paddle_start_x == -3.9
            assert e.unwrapped.paddle_home_x == -3.65
            assert e.unwrapped.serve_start_x == 1.0
            assert e.unwrapped.serve_speed == 7.0
        # Action mapping stays pinned to the full workspace, pivoted on
        # the goal fence midpoint (the 0.22.0 usable-share fix).
        assert env.unwrapped.paddle_x_target_range == (-4.7, 0.3)
    finally:
        env.close()
        eval_env.close()


def test_true_baseline_extends_the_goal_recipe_to_the_itf_baseline():
    """The 0.25.0 era: the goal-rally structure, opted into the extended
    workspace, with probe-derived serve energy and in-play bound. The
    era is a delta on the proven recipe, not a new design."""
    recipe = RECIPES["WallBallTrueBaseline"]
    goal = RECIPES["WallBallGoalRally"]

    task = {
        "paddle_x_fence": (-8.2, -2.6),
        "paddle_start_x": -7.9,
        "paddle_home_x": -5.4,
        "serve_start_x": 1.0,
        "serve_speed": 11.0,
    }
    for key, value in task.items():
        assert recipe.env_kwargs[key] == value, key
    # The extended action mapping is explicit opt-in (the bare-env
    # default stays frozen at (-4.7, 0.3)), pivoted on the fence
    # midpoint per the 0.22.0 usable-share rule.
    assert recipe.env_kwargs["paddle_x_target_range"] == (-8.2, 0.3)
    assert recipe.env_kwargs["paddle_home_x"] == pytest.approx(
        sum(recipe.env_kwargs["paddle_x_fence"]) / 2
    )
    # Deep serves/rebounds legitimately hop past -9; only this era
    # widens the in-play bound, and only via the per-task kwarg.
    assert recipe.env_kwargs["ball_in_play_min_x"] == -10.0
    assert "ball_in_play_min_x" not in goal.env_kwargs

    # Same single-stage informational gate contract as the goal era:
    # nothing to promote to, no advance package.
    gate = recipe.extra_cfg["performance_gate"]
    assert [dict(s) for s in gate["stages"]] == [task]
    assert gate["threshold"] == 3.0
    assert gate["promotion_rule"] == "window_mean"
    assert "clear_replay_buffer_on_advance" not in gate
    assert "advance_update_pause_steps" not in gate
    assert "reset_entropy_on_advance" not in gate
    assert recipe.eval_env_overrides == task

    # Certification: calibrated lead-charge 2.6 probe from the reserved
    # 30000+ block, feasibility floor 0.50 per the measured oracle band
    # (67% two-return rate on a task no scripted reference dominates).
    cert = recipe.extra_cfg["ladder_certification"]
    assert list(cert["oracle_probes"]) == [{"lead_charge": 2.6}]
    assert cert["seed_start"] >= 30_000
    assert cert["episodes"] == 30
    assert cert["feasibility_ge2_floor"] == 0.50
    # The stock floor is untouched for every other recipe.
    assert "feasibility_ge2_floor" not in goal.extra_cfg[
        "ladder_certification"
    ]

    assert recipe.extra_cfg["final_info_eval"] is False
    assert recipe.extra_cfg["final_eval_episodes"] is None
    assert recipe.extra_cfg["early_stop_patience"] == 60
    assert recipe.extra_cfg["model_kwargs"] == {"gamma": 0.995}


def test_true_baseline_config_builds_and_gate_stage_applies(tmp_path):
    cfg = build_train_config("WallBallTrueBaseline", log_dir=str(tmp_path))
    env = cfg.env_fn()
    eval_env = cfg.eval_env_fn()
    try:
        for e in (env, eval_env):
            assert e.unwrapped.paddle_x_fence == (-8.2, -2.6)
            assert e.unwrapped.paddle_start_x == -7.9
            assert e.unwrapped.paddle_home_x == -5.4
            assert e.unwrapped.serve_start_x == 1.0
            assert e.unwrapped.serve_speed == 11.0
            assert e.unwrapped.paddle_x_target_range == (-8.2, 0.3)
            assert e.unwrapped.ball_in_play_min_x == -10.0
    finally:
        env.close()
        eval_env.close()
