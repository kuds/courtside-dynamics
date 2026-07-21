"""Tests for the recipe registry that drives the consolidated notebook.

``build_train_config`` is the single entry point the training notebook
uses, so a broken recipe (bad env kwargs, an ``extra_cfg`` key that
``TrainConfig`` doesn't accept, a stale ``info_row_fn``) breaks every
run of that env. These tests materialize each recipe for real.
"""
from __future__ import annotations

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
    ("recipe_name", "stage", "active_count", "episode_len"),
    [
        ("HumanoidTennisStage0Intercept", 0, 2, 150),
        ("HumanoidTennisStage1AnchoredReturn", 1, 7, 300),
        ("HumanoidTennisStage2RandomizedReturn", 2, 7, 300),
    ],
)
def test_humanoid_tennis_curriculum_recipes_are_fixed_stage_and_recordable(
    recipe_name,
    stage,
    active_count,
    episode_len,
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
    assert "experimental" in recipe.description.lower()
    assert "convergence is not claimed" in recipe.description.lower()
    assert cfg.csv_header is not None and cfg.info_row_fn is not None

    env = cfg.env_fn()
    try:
        _, reset_info = env.reset(seed=0)
        assert env.episode_len == episode_len
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
        ("WallBallDepthCurriculum", "open", -1.7, (-4.7, 0.3)),
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
    """The depth ladder must walk monotonically deeper with serve energy
    co-moving, start at the constructor's stage-0 geometry, keep every
    fence generous (deep-narrow fences die on unreachable mid-court
    rebounds), and end at the physical workspace limit -- all under a
    pinned full-workspace action mapping so action semantics never
    drift. Stage values were calibrated by tools/depth_stage_sweep.py;
    edits here must re-run that sweep."""
    recipe = RECIPES["WallBallDepthCurriculum"]
    gate = recipe.extra_cfg["performance_gate"]
    stages = gate["stages"]

    assert len(stages) == 5
    for key, value in stages[0].items():
        assert recipe.env_kwargs[key] == value

    backs = [stage["paddle_x_fence"][0] for stage in stages]
    fronts = [stage["paddle_x_fence"][1] for stage in stages]
    starts = [stage["paddle_start_x"] for stage in stages]
    speeds = [stage["serve_speed"] for stage in stages]
    assert backs == sorted(backs, reverse=True)
    assert fronts == sorted(fronts, reverse=True)
    assert starts == sorted(starts, reverse=True)
    assert speeds == sorted(speeds)
    assert all(
        front - back >= 2.5
        for back, front in zip(backs, fronts, strict=True)
    )
    for stage in stages:
        back, front = stage["paddle_x_fence"]
        assert back <= stage["paddle_start_x"] <= front

    mapping = recipe.env_kwargs["paddle_x_target_range"]
    assert mapping == (-4.7, 0.3)
    assert stages[-1]["paddle_x_fence"][0] == mapping[0]

    assert gate["metric_key"] == "bounce_count_ep_mean"
    assert gate["threshold"] == 3.0
    assert gate["sustain_evals"] == 2
    # Run-1 gate refinements (see the recipe comment): the bar itself is
    # untouched, but promotion reads the 2-eval mean (stage 2 cleared
    # 3.0 four separate times without two-in-a-row), and every advance
    # runs the warm-up package -- clear the stale-stage buffer, pause
    # updates for 50k fresh frontier steps.
    assert gate["promotion_rule"] == "window_mean"
    assert gate["advance_update_pause_steps"] == 50_000
    assert gate["clear_replay_buffer_on_advance"] is True
    assert recipe.extra_cfg["final_info_eval"] is True
    assert recipe.extra_cfg["reward_eval_episodes"] == 5


def test_wall_ball_depth_curriculum_uses_open_scoring_and_defaults():
    """Open scoring, emergent style: no one_bounce fault taxonomy or
    bootstrap shaping may leak in, and model_kwargs stays empty (SB3
    auto entropy -- lesson 5; capacity and reward probes were null)."""
    recipe = RECIPES["WallBallDepthCurriculum"]
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
    assert "model_kwargs" not in recipe.extra_cfg
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


def test_wall_ball_depth_curriculum_config_builds_and_stages_apply(tmp_path):
    """The gate's stage dicts must be reachable, settable env attributes
    (a typo'd stage key would otherwise die mid-run), and the built
    config must carry the gate and the 3M budget."""
    cfg = build_train_config("WallBallDepthCurriculum", log_dir=str(tmp_path))
    assert cfg.performance_gate is not None
    assert cfg.performance_gate["metric_key"] == "bounce_count_ep_mean"
    assert cfg.total_timesteps == 3_000_000
    assert cfg.final_info_eval is True

    env = cfg.env_fn()
    try:
        assert env.rally_style == "open"
        assert env.paddle_x_fence == (-2.7, 0.3)
        assert env.serve_speed == 5.2
        for stage in cfg.performance_gate["stages"]:
            for key, value in stage.items():
                assert hasattr(env, key)
                setattr(env, key, value)
        env.reset(seed=0)
        assert env.paddle_x_fence == (-4.7, -1.2)
    finally:
        env.close()


def test_wall_ball_bootstrap_is_marked_historical():
    """Bootstrap is kept for the record but must say so: its cold-start
    problem was solved by auto-entropy before it ever ran, and its
    reward package bundles the falsified weak-return retry."""
    description = RECIPES["WallBallBootstrap"].description
    assert "HISTORICAL" in description
    assert "WallBallDepthCurriculum" in description
