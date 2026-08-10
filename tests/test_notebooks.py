"""Static contracts for executable repository notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[1]
SB3_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb"
HUMANOID_NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "humanoid_tennis_training.ipynb"
)


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _load_humanoid_notebook() -> dict[str, Any]:
    return json.loads(HUMANOID_NOTEBOOK.read_text())


def _load_sb3_notebook() -> dict[str, Any]:
    return json.loads(SB3_NOTEBOOK.read_text())


def test_humanoid_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = _load_humanoid_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    cells = notebook["cells"]
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = _source(cell)
        if any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ):
            continue
        compile(source, f"{HUMANOID_NOTEBOOK}:{cell['id']}", "exec")


def test_humanoid_notebook_bootstraps_mujoco_before_environment_imports() -> None:
    code_cells = [
        _source(cell)
        for cell in _load_humanoid_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]
    setup_index = next(
        index for index, source in enumerate(code_cells) if "setup_colab()" in source
    )
    recipe_index = next(
        index
        for index, source in enumerate(code_cells)
        if "from courtside_dynamics.recipes import" in source
    )
    assert setup_index < recipe_index


def test_humanoid_notebook_preserves_curriculum_recipe_contracts() -> None:
    notebook_source = "\n".join(
        _source(cell) for cell in _load_humanoid_notebook()["cells"]
    )
    assert "STAGES = (0, 1, 2)" in notebook_source
    assert '0: "HumanoidTennisStage0Intercept"' in notebook_source
    assert '1: "HumanoidTennisStage1AnchoredReturn"' in notebook_source
    assert '2: "HumanoidTennisStage2RandomizedReturn"' in notebook_source
    assert "AUTO_ADVANCE = True" in notebook_source
    assert 'ALGO = None' in notebook_source
    assert 'TOTAL_TIMESTEPS = None' in notebook_source
    # None-sentinel knobs: an unconditional early_stop_patience=... or
    # model_kwargs={} override would replace the recipes' calibrated
    # per-stage patience (8/12/20) and exploration bundle wholesale.
    assert "EARLY_STOP_PATIENCE = None" in notebook_source
    assert "MODEL_KWARGS = None" in notebook_source
    assert "early_stop_patience=EARLY_STOP_PATIENCE" not in notebook_source
    assert "model_kwargs=MODEL_KWARGS" not in notebook_source
    assert 'overrides["early_stop_patience"] = EARLY_STOP_PATIENCE' in notebook_source
    assert 'overrides["model_kwargs"] = MODEL_KWARGS' in notebook_source
    assert 'recipe_n_envs = recipe.extra_cfg.get("n_envs")' in notebook_source
    assert "cfg.checkpoint_freq =" not in notebook_source
    assert "cfg.video_freq =" not in notebook_source
    assert "active={reset_info['active_action_count']}/58" in notebook_source
    assert "does not demonstrate two free-standing humanoids" in notebook_source
    assert "policy parameters and matching observation-normalization" in notebook_source
    assert "fresh optimizer" in notebook_source
    assert "QUICK_TEST and AUTO_ADVANCE" in notebook_source
    assert 'RESOLVED_ALGO != "PPO"' in notebook_source


def test_humanoid_notebook_warm_starts_only_after_canonical_promotion() -> None:
    notebook_source = "\n".join(
        _source(cell) for cell in _load_humanoid_notebook()["cells"]
    )
    assert "for stage in STAGES_TO_RUN:" in notebook_source
    # Best-model artifacts resolve through the shared layout registry so
    # both new (model/best_model.zip) and legacy flat stage dirs work.
    assert (
        'located_model = locate_artifact(stage_dir, "best_model")'
        in notebook_source
    )
    assert (
        'located_normalizer = locate_artifact(stage_dir, "best_vec_normalize")'
        in notebook_source
    )
    assert "best_model_path = Path(located_model)" in notebook_source
    assert "best_normalizer_path = Path(located_normalizer)" in notebook_source
    assert "resolve_algo(cfg.algo).load" in notebook_source
    assert "for eval_stage in range(stage + 1)" in notebook_source
    assert "evaluate_curriculum_stage(" in notebook_source
    assert "policy_artifact_path=best_model_path" in notebook_source
    assert "normalization_artifact_path=best_normalizer_path" in notebook_source
    assert "make_env_fn(STAGE_RECIPES[eval_stage])" in notebook_source
    assert "suite=" not in notebook_source
    assert "assess_curriculum_promotion(" in notebook_source
    assert "report.to_dict()" in notebook_source
    assert 'stage_dir / "promotion_report.json"' in notebook_source
    assert "report.eligible_for_manual_promotion and not extra_failures" in notebook_source
    assert "if not automatic_eligible:" in notebook_source
    assert 'warm_start = WarmStartConfig(source_run_dir=str(stage_dir))' in notebook_source
    assert "while " not in notebook_source
    assert "one_shot_canonical_evaluation" in notebook_source
    assert "Interrupted training was saved for recovery but cannot auto-promote" in (
        notebook_source
    )


def test_humanoid_notebook_reports_optional_metrics_and_disconnects() -> None:
    notebook_source = "\n".join(
        _source(cell) for cell in _load_humanoid_notebook()["cells"]
    )
    assert "EXTRA_STAGE_CRITERIA" in notebook_source
    for metric in (
        "mean_reward",
        "mean_episode_steps",
        "mean_valid_returns",
        "mean_rally_count",
    ):
        assert metric in notebook_source
    assert "stage_success" in notebook_source
    assert "per_side_success_rate" in notebook_source
    assert "curriculum_manifest.json" in notebook_source
    assert "DISCONNECT_WHEN_DONE = True" in notebook_source
    assert "disconnect_runtime(delay_seconds=30)" in notebook_source


def test_sb3_notebook_runs_long_horizon_eval_after_training() -> None:
    notebook = _load_sb3_notebook()
    source = "\n".join(_source(cell) for cell in notebook["cells"])

    train_index = source.index("model = train(cfg)")
    long_eval_index = source.index("evaluate_best_wall_ball(")
    video_index = source.index("record_best_model_video(")
    audit_index = source.index("missing = check_run_artifacts(")
    assert train_index < long_eval_index < video_index < audit_index

    assert "RUN_LONG_HORIZON_EVAL = True" in source
    assert "LONG_HORIZON_EPISODE_LEN = 5_000" in source
    assert "LONG_HORIZON_N_EPISODES = 50" in source
    assert "LONG_HORIZON_SEED_START = 10_000" in source
    assert "WallBallVolley" in source
    assert "WallBallBaseline" in source
    assert "issubclass(RECIPES[ENV].env_cls, WallBallEnv)" in source
    assert "if is_wall_ball_recipe and RUN_LONG_HORIZON_EVAL:" in source
    assert 'if ENV == "WallBall" and RUN_LONG_HORIZON_EVAL:' not in source
    # The audit env layers the run config's [env]/[eval_env] tables so
    # an edited Drive TOML does not make the recorded-vs-audited
    # constructor-profile check raise on the default workflow.
    assert "base_env_overrides=(file_cfg.env if file_cfg else None)" in source
    assert "**(file_cfg.eval_env if file_cfg else {})," in source
    assert '"episode_len": long_episode_len,' in source
    assert "make_eval_env_fn(" in source
    assert '"return_survival_curve": long_eval["return_survival_curve"]' in source
    assert "cfg.eval_env_fn or cfg.env_fn" in source
    assert "WALL_BALL_LONG_HORIZON_ARTIFACTS" in source
    artifacts_index = source.index(
        "long_horizon_artifacts = WALL_BALL_LONG_HORIZON_ARTIFACTS"
    )
    assert artifacts_index < long_eval_index
    assert "best_model_long_horizon_eval.json" in source
    assert "best_model_long_horizon_episodes.csv" in source
    assert "cfg.env_fn =" not in source
    assert "RECIPES[ENV].env_kwargs[" not in source


def test_sb3_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = _load_sb3_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = _source(cell)
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        compile(source, f"{SB3_NOTEBOOK}:{cell.get('id', 'unknown')}", "exec")


def test_sb3_notebook_uses_run_config_files_not_pinned_model_kwargs() -> None:
    source = "\n".join(
        _source(cell) for cell in _load_sb3_notebook()["cells"]
    )
    # The 20260717 A/B (run 025611 vs 165358) showed the old pinned
    # bundle prevents WallBall from learning at all; the notebook must
    # default to recipe/SB3 behavior and route tweaks through TOML files.
    assert "MODEL_KWARGS = None" in source
    assert 'MODEL_KWARGS = {"ent_coef": 0.02' not in source
    # CONFIG_FILE is a mode switch: "auto" resolves the recipe's TOML
    # (created from the packaged starter on first use) after the Drive
    # mount; None and explicit paths remain available.
    assert 'CONFIG_FILE = "auto"' in source
    assert 'if CONFIG_FILE == "auto":' in source
    assert "resolve_run_config_file(ENV, use_drive=USE_DRIVE)" in source
    assert "config_file=CONFIG_FILE" in source
    # Notebook variables are layered: only explicitly-set values are
    # passed, so a TOML's [train] table is not silently overridden.
    assert 'if MODEL_KWARGS is not None:' in source
    assert 'if N_ENVS is not None:' in source
    assert 'if EARLY_STOP_PATIENCE is not None:' in source
    # Recipes carry the calibrated patience; the notebook defers.
    assert "EARLY_STOP_PATIENCE = None" in source
    assert "EARLY_STOP_PATIENCE = 20" not in source
    # Overriding a recipe's default algorithm re-applies the per-algo
    # worker-count suggestion (a recipe's n_envs is calibrated for its
    # default algo only).
    assert 'ALGO.upper() != RECIPES[ENV].default_algo.upper()' in source
    assert "model_kwargs=MODEL_KWARGS" not in source
    # Discovery cell: recipes and packaged starters are listable, and
    # starters are copied (never edited in site-packages).
    assert "available_run_configs" in source
    assert "copy_starter_config" in source
    assert "WallBallBootstrap" in source
    assert "cfg.run_config_file" in source


def test_sb3_notebook_replay_uses_tennis_court_style() -> None:
    source = "\n".join(
        _source(cell) for cell in _load_sb3_notebook()["cells"]
    )
    assert 'REPLAY_COURT_STYLE = "tennis"' in source
    # The replay env derives from the run's own factory (so a TOML's
    # [env]/[eval_env] overrides survive), with only the render-only
    # style applied on top -- on every court that supports styles
    # (WallBall and PaddleTennis both expose ``court_style``; envs
    # without it pass through untouched).
    assert "base_replay_env_fn = cfg.eval_env_fn or cfg.env_fn" in source
    assert 'if hasattr(env, "court_style"):' in source
    assert "env.court_style = REPLAY_COURT_STYLE" in source
    # Metrics paths must not restyle their envs.
    assert "court_style" not in source.split("REPLAY_COURT_STYLE")[0]


def test_sb3_notebook_lists_every_recipe() -> None:
    """The notebook's inline recipe menus (the ENV cell comment and the
    intro markdown) went two releases stale exactly like the README's
    list -- 0.24.0's WallBallGoalRally and 0.25.0's WallBallTrueBaseline
    appeared nowhere in the generic driver notebook. Pin every recipe
    name into the notebook source so the next recipe cannot drift."""
    from courtside_dynamics.recipes import RECIPES

    source = "\n".join(
        _source(cell) for cell in _load_sb3_notebook()["cells"]
    )
    missing = [name for name in RECIPES if name not in source]
    assert not missing, (
        f"recipes missing from the sb3 notebook's menus: {missing}"
    )
