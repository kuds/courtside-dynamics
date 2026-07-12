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


@pytest.mark.parametrize("env_name", ["BallBounce", "WallBall"])
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
    assert 'ALGO = ALGO or RECIPES[ENV].default_algo' in source
    assert 'recipe_n_envs = RECIPES[ENV].extra_cfg.get("n_envs")' in source
    assert "cfg.checkpoint_freq =" not in source
    assert "cfg.video_freq =" not in source
    assert "HumanoidTennisStage0Intercept" in source
    assert "HumanoidTennisStage2RandomizedReturn" in source
