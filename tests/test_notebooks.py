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
    assert 'best_model_path = stage_dir / "best_model.zip"' in notebook_source
    assert (
        'best_normalizer_path = stage_dir / "best_vec_normalize.pkl"'
        in notebook_source
    )
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
    assert 'if ENV == "WallBall" and RUN_LONG_HORIZON_EVAL:' in source
    assert 'env_overrides={"episode_len": long_episode_len}' in source
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
