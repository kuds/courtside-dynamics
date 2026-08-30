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
CAMPAIGN_NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "paddle_tennis_campaign.ipynb"
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


def _load_campaign_notebook() -> dict[str, Any]:
    return json.loads(CAMPAIGN_NOTEBOOK.read_text())


def test_campaign_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = _load_campaign_notebook()
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
        compile(source, f"{CAMPAIGN_NOTEBOOK}:{cell['id']}", "exec")


def test_campaign_notebook_bootstraps_mujoco_before_environment_imports() -> None:
    code_cells = [
        _source(cell)
        for cell in _load_campaign_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]
    setup_index = next(
        index
        for index, source in enumerate(code_cells)
        if "setup_colab()" in source
    )
    recipe_index = next(
        index
        for index, source in enumerate(code_cells)
        if "from courtside_dynamics.recipes import" in source
    )
    assert setup_index < recipe_index


def test_campaign_notebook_freezes_the_preregistered_plan() -> None:
    """The settings cell is the campaign's frozen protocol: every number
    ships pinned to the pre-registration
    (docs/paddle_tennis_registered_run_prereg_20260816.md §1a/§1b/§2),
    and the manifest fingerprint refuses a silent mid-campaign edit."""
    source = "\n".join(
        _source(cell) for cell in _load_campaign_notebook()["cells"]
    )
    # The LS1 gate-leg shape and both branch budgets.
    assert "GATE_TIMESTEPS = 1_000_000" in source
    assert "MAIN_TIMESTEPS_ON_PASS = 2_000_000" in source
    assert "MAIN_TIMESTEPS_ON_FALLBACK = 3_000_000" in source
    assert "N_ENVS = 4" in source
    assert "EVAL_FREQ = 25_000" in source
    assert "CHECKPOINT_FREQ = 100_000" in source
    assert "WARM_START_LEARNING_STARTS = 25_000" in source
    # The warm-start and env-override knobs ship at the
    # backward-compatible defaults: leg 1 from scratch, temperatures
    # transferred, no sha pins, recipe task exactly.
    assert "LEG1_WARM_START_RUN_DIR = None" in source
    assert "LEG1_TRANSFER_LOG_ENT_COEF = True" in source
    assert "LEG1_EXPECTED_ARTIFACT_SHA256 = None" in source
    assert "LEG2_TRANSFER_LOG_ENT_COEF = True" in source
    assert "LEG2_EXPECTED_ARTIFACT_SHA256 = None" in source
    assert "ENV_KWARGS = {}" in source
    # The compatibility and scope decisions are stated where the knobs
    # live: pre-knob campaigns cannot resume under the grown
    # fingerprint, and the campaign template deliberately has no
    # CONFIG_FILE/TOML knob (ENV_KWARGS is its [env] table).
    assert "cannot resume under the updated notebook" in source
    assert "deliberately out of scope for the campaign template" in source
    # The gate reads the recipe's own diagnosis instrument settings.
    assert "GATE_EPISODES = 30" in source
    assert "GATE_SEED_START = 5200" in source
    # LS1 bars: LS-C and LS-K1 gate; LS-G is informational.
    assert '"LS-C"' in source
    assert '"LS-K1"' in source
    assert '"LS-G"' in source
    assert '"metric": "touched_after_bounce_rate"' in source
    assert '"metric": "k1_receiving_survival"' in source
    assert '"metric": "ready_error_mean"' in source
    # The frozen fallback lineage both pilots warm-started from.
    assert "20260809_211147" in source
    # Registered §3 bands appear as record-only readings.
    for band in ('"RK1"', '"RE1"', '"RE3"', '"RS2"'):
        assert band in source
    assert '"metric": "k2_either_survival"' in source
    # Decision knobs are validated, and QUICK_TEST cannot silently
    # auto-branch on non-evidence gate numbers.
    assert "QUICK_TEST and FORCE_BRANCH is None" in source
    assert 'GATE_MIDDLE_ACTION = "stop"' in source
    assert "FORCE_BRANCH = None" in source


def test_campaign_notebook_resumes_and_branches_via_helpers() -> None:
    source = "\n".join(
        _source(cell) for cell in _load_campaign_notebook()["cells"]
    )
    # Manifest-driven resume on a stable (non-timestamped) root.
    assert '"PaddleTennisCampaign", "sac", use_drive=USE_DRIVE, timestamp=False' in (
        source
    )
    assert "load_campaign_manifest" in source
    assert "write_campaign_manifest" in source
    assert "require_campaign_fingerprint(manifest, FINGERPRINT)" in source
    assert "next_stage_attempt_dir(CAMPAIGN_ROOT, stage_name)" in source
    # The gate and final report use the shared scorer.
    assert "score_paddle_stage(" in source
    assert "bars=GATE_BARS" in source
    assert "bars=FINAL_REPORT_BARS" in source
    assert 'report_name="campaign_final_report.json"' in source
    # Branching goes through the tested helper, and the main leg
    # warm-starts from the decided lineage with the frozen transfer
    # flags and sha pins.
    assert "resolve_warm_start_branch(" in source
    assert "source_run_dir=str(source)," in source
    assert "transfer_log_ent_coef=LEG2_TRANSFER_LOG_ENT_COEF," in source
    # The leg-2 sha pins bind the fallback lineage only: the continue
    # branch's source is the gate leg's own best, whose digests are
    # unknowable at fingerprint time -- a pin applied there would wedge
    # every passing campaign.
    assert "expected_artifact_sha256=leg2_expected_sha256," in source
    assert "LEG2_EXPECTED_ARTIFACT_SHA256\n" in source
    assert 'if decision["branch"] == "fallback"' in source
    assert "expected_artifact_sha256=LEG2_EXPECTED_ARTIFACT_SHA256," not in (
        source
    )
    assert "applies to the fallback lineage only" in source
    # Leg 1 stays from scratch unless the settings pin a lineage (the
    # warm-started leg-1 path, e.g. an LT1-shape pilot).
    assert "if LEG1_WARM_START_RUN_DIR is not None:" in source
    assert "source_run_dir=str(LEG1_WARM_START_RUN_DIR)," in source
    assert "transfer_log_ent_coef=LEG1_TRANSFER_LOG_ENT_COEF," in source
    assert "expected_artifact_sha256=LEG1_EXPECTED_ARTIFACT_SHA256," in source
    assert "warm_start=leg1_warm_start," in source
    assert 'seed=SEED + 10_000' in source
    # The warm-started legs merge learning_starts INTO the recipe's
    # calibrated SAC bundle -- a bare {"learning_starts": ...} override
    # would silently drop use_sde / auto-entropy / train_freq.
    assert '**RECIPES[RECIPE].extra_cfg["model_kwargs"],' in source
    assert '"learning_starts": WARM_START_LEARNING_STARTS,' in source
    # Interrupted legs cannot gate; the retry lands in a fresh attempt.
    assert "cannot be gated or branched" in source
    assert "del model" in source
    # No post-build config mutation (the humanoid notebook's contract).
    assert "cfg.checkpoint_freq =" not in source
    assert "cfg.video_freq =" not in source
    assert "cfg.env_fn =" not in source


def test_campaign_notebook_validates_config_and_plumbs_env_kwargs() -> None:
    """The prereg §6 gap, closed: every leg's config.json is validated
    against the frozen plan right after train() returns, the verdict is
    recorded in the manifest before any gate scoring, and a mismatch
    stops the campaign. ENV_KWARGS rides the run-config [env]-table
    route so both legs' training and evaluation envs get the same
    overrides."""
    source = "\n".join(
        _source(cell) for cell in _load_campaign_notebook()["cells"]
    )
    # The expected mapping derives from the frozen settings; QUICK_TEST
    # hands budget/cadence to the quick-test presets.
    assert "def leg_expected_plan(*, seed, total_timesteps, warm_start):" in (
        source
    )
    assert '"env_class": "PaddleTennisEnv",' in source
    assert '"env_kwargs": dict(ENV_KWARGS),' in source
    assert '"source_run_dir_suffix": "/".join(parts).lstrip("/"),' in source
    assert '"transfer_log_ent_coef": warm_start.transfer_log_ent_coef,' in (
        source
    )
    assert (
        '"expected_artifact_sha256": warm_start.expected_artifact_sha256,'
        in source
    )
    # Validation runs after train() and before any scoring; the verdict
    # (ok or the full error text) lands in the manifest either way.
    # Only RunConfigPlanMismatch is booked as config drift -- a missing
    # config.json, a JSONDecodeError (a ValueError subclass), or the
    # validator's own bad-plan errors propagate as instrument failures.
    assert "validate_run_config_against_plan(config_path, expected)" in source
    assert '"config_validation": validation,' in source
    assert "RunConfigPlanMismatch," in source
    assert "except RunConfigPlanMismatch as err:" in source
    assert "except ValueError" not in source
    assert 'validation = {"verdict": "mismatch", "error": str(err)}' in source
    train_index = source.index("model = train(cfg)")
    validate_index = source.index("validate_run_config_against_plan(")
    record_index = source.index('"config_validation": validation,')
    gate_score_index = source.index("gate_report = score_paddle_stage(")
    assert train_index < validate_index < record_index < gate_score_index
    assert 'if validation["verdict"] != "ok":' in source
    assert "does not match the" in source
    # record_stage merges over run_leg's record instead of dropping the
    # recorded validation verdict.
    assert 'record = dict(manifest["stages"].get(stage_name) or {})' in source
    # ENV_KWARGS goes through the supported factory route ([env]-table
    # semantics: training env + below the recipe's eval overrides), and
    # an empty dict leaves the recipe factories untouched.
    assert "if ENV_KWARGS:" in source
    assert (
        'overrides["env_fn"] = make_env_fn(RECIPE, env_overrides=ENV_KWARGS)'
        in source
    )
    assert "base_env_overrides=ENV_KWARGS" in source
    # Every new knob is fingerprinted so a resumed campaign cannot
    # silently run under different warm-start/task settings.
    for fingerprint_entry in (
        '"leg1_warm_start_run_dir": LEG1_WARM_START_RUN_DIR,',
        '"leg1_transfer_log_ent_coef": LEG1_TRANSFER_LOG_ENT_COEF,',
        '"leg1_expected_artifact_sha256": LEG1_EXPECTED_ARTIFACT_SHA256,',
        '"leg2_transfer_log_ent_coef": LEG2_TRANSFER_LOG_ENT_COEF,',
        '"leg2_expected_artifact_sha256": LEG2_EXPECTED_ARTIFACT_SHA256,',
        '"env_kwargs": ENV_KWARGS,',
    ):
        assert fingerprint_entry in source
