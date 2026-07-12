"""Static contracts for executable repository notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[1]
HUMANOID_NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "humanoid_tennis_training.ipynb"
)


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _load_humanoid_notebook() -> dict[str, Any]:
    return json.loads(HUMANOID_NOTEBOOK.read_text())


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


def test_humanoid_notebook_preserves_fixed_stage_recipe_contracts() -> None:
    notebook_source = "\n".join(
        _source(cell) for cell in _load_humanoid_notebook()["cells"]
    )
    assert 'STAGE = 0' in notebook_source
    assert '0: "HumanoidTennisStage0Intercept"' in notebook_source
    assert '1: "HumanoidTennisStage1AnchoredReturn"' in notebook_source
    assert '2: "HumanoidTennisStage2RandomizedReturn"' in notebook_source
    assert 'ALGO = None' in notebook_source
    assert 'TOTAL_TIMESTEPS = None' in notebook_source
    assert 'recipe_n_envs = recipe.extra_cfg.get("n_envs")' in notebook_source
    assert "cfg.checkpoint_freq =" not in notebook_source
    assert "cfg.video_freq =" not in notebook_source
    assert "full 58-value centralized action API" in notebook_source
    assert "does not demonstrate two free-standing humanoids" in notebook_source
    assert "automatic checkpoint transfer" in notebook_source


def test_humanoid_notebook_uses_best_artifacts_and_canonical_promotion() -> None:
    notebook_source = "\n".join(
        _source(cell) for cell in _load_humanoid_notebook()["cells"]
    )
    assert 'best_model_path = run_path / "best_model.zip"' in notebook_source
    assert (
        'best_normalizer_path = run_path / "best_vec_normalize.pkl"'
        in notebook_source
    )
    assert "resolve_algo(cfg.algo).load" in notebook_source
    assert "for eval_stage in range(STAGE + 1)" in notebook_source
    assert "evaluate_curriculum_stage(" in notebook_source
    assert "policy_artifact_path=best_model_path" in notebook_source
    assert "normalization_artifact_path=normalization_path" in notebook_source
    assert "make_env_fn(STAGE_RECIPES[eval_stage])" in notebook_source
    assert "suite=" not in notebook_source
    assert "assess_curriculum_promotion(" in notebook_source
    assert "promotion_report.to_dict()" in notebook_source
    assert 'run_path / "promotion_report.json"' in notebook_source
    assert "Eligible for manual promotion" in notebook_source
    assert "never advances the curriculum" in notebook_source
