"""Startup ladder certification: the guard against uncertified geometry.

Run 20260727_233859 trained 17 hours on a five-stage ladder whose
standalone sweep tool still encoded the previous release's stage table.
These tests pin the two properties that prevent a recurrence: the
certifier derives everything from the live gate spec (no second stage
table), and stage application is verified attribute-by-attribute on the
real env (the pre-0.22.0 ``paddle_home_x`` silent no-op class).
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from courtside_dynamics.envs import WallBallEnv
from courtside_dynamics.training import ladder_certification as lc

# The real recipe's first two rungs, so unit runs exercise the exact
# attribute names and value shapes the gate applies in production.
STAGE_0 = {
    "paddle_x_fence": (-2.3, -0.2),
    "paddle_start_x": -1.6,
    "paddle_home_x": -1.25,
    "serve_start_x": 1.0,
    "serve_speed": 5.2,
}
STAGE_1 = {
    "paddle_x_fence": (-2.9, -0.8),
    "paddle_start_x": -2.1,
    "paddle_home_x": -1.85,
    "serve_start_x": 1.0,
    "serve_speed": 5.5,
}


def _tiny_env_fn():
    # Short episodes keep every certification cell to a few dozen steps;
    # the wide mapping mirrors the recipe (stage fences must lie inside
    # the action-map range for the fence clamp to behave).
    return WallBallEnv(
        episode_len=120,
        rally_style="open",
        min_force=20.0,
        paddle_x_target_range=(-4.7, 0.3),
        paddle_home_x=-1.25,
    )


def test_validate_ladder_geometry_flags_the_known_failure_modes():
    # The live 5-stage recipe ladder is the canonical pass: adjacent
    # overlap everywhere, no interval shared by every stage.
    from courtside_dynamics.recipes import RECIPES

    live = RECIPES["WallBallDepthCurriculum"].extra_cfg["performance_gate"][
        "stages"
    ]
    assert lc.validate_ladder_geometry(live) == []
    assert lc.validate_ladder_geometry([STAGE_0, STAGE_1]) == []

    start_outside = [{**STAGE_0, "paddle_start_x": 5.0}]
    assert any(
        "outside fence" in f
        for f in lc.validate_ladder_geometry(start_outside)
    )

    disjoint = [STAGE_0, {**STAGE_1, "paddle_x_fence": (-9.0, -8.0)}]
    assert any(
        "do not overlap" in f for f in lc.validate_ladder_geometry(disjoint)
    )

    # The all-stage refuge run 20260722_124613 exposed (three stages —
    # at two, adjacent overlap IS the shared interval, and is required).
    refuge = [
        STAGE_0,
        STAGE_1,
        {**STAGE_1, "paddle_x_fence": (-2.6, -0.5), "paddle_start_x": -1.9},
    ]
    assert any(
        "share a reachable x interval" in f
        for f in lc.validate_ladder_geometry(refuge)
    )

    assert lc.validate_ladder_geometry([]) == ["ladder has no stages"]


def test_validate_ladder_geometry_skips_geometry_free_stages():
    """Gate stages are attribute-arbitrary (serve-only ladders exist)."""
    serve_only = [{"serve_speed": 4.0}, {"serve_speed": 5.0}]
    assert lc.validate_ladder_geometry(serve_only) == []
    # A mixed ladder still checks the stages that declare geometry.
    mixed = [{"serve_speed": 4.0}, {**STAGE_0, "paddle_start_x": 5.0}]
    assert any(
        "outside fence" in f for f in lc.validate_ladder_geometry(mixed)
    )


def test_apply_stage_records_setter_rejection_instead_of_raising():
    """A rejecting property setter is a finding, not an abort."""
    env = _tiny_env_fn()
    try:
        applied, failures = lc.apply_stage(
            env, {"paddle_x_fence": (-9.0, -8.0)}
        )
        assert len(failures) == 1
        assert "rejected value" in failures[0]
        assert "paddle_x_fence" not in applied
    finally:
        env.close()


def test_apply_stage_verifies_existence_and_round_trip():
    env = _tiny_env_fn()
    try:
        applied, failures = lc.apply_stage(env, STAGE_1)
        assert failures == []
        # paddle_home_x is the property whose pre-0.22.0 silent no-op
        # motivated this module: the round-trip must see the new pivot.
        assert applied["paddle_home_x"] == pytest.approx(-1.85)
        assert env.unwrapped.paddle_home_x == pytest.approx(-1.85)

        _, bad = lc.apply_stage(env, {"definitely_not_an_attr": 1.0})
        assert len(bad) == 1
        assert "definitely_not_an_attr" in bad[0]
    finally:
        env.close()


def test_certify_ladder_end_to_end_writes_a_complete_report():
    report = lc.certify_ladder(
        _tiny_env_fn,
        [STAGE_0, STAGE_1],
        episodes=1,
        seed_start=lc.DEFAULT_SEED_START,
        oracle_probes=[{"run_up": 1.1}, {"charge_gap": 1.0}],
        gate_metric_key="bounce_count_ep_mean",
        gate_threshold=3.0,
        max_episode_steps=60,
    )

    assert report["schema"] == 1
    assert report["gate"]["threshold"] == 3.0
    assert len(report["stages"]) == 2
    for stage in report["stages"]:
        assert set(stage["cells"]) == {"parked", "crude", "oracle"}
        for cell in stage["cells"].values():
            assert cell["episodes"] == 1
            assert isinstance(cell["bounce_mean"], float)
        assert stage["applied"]["paddle_start_x"] == stage["geometry"][
            "paddle_start_x"
        ]
        assert stage["landing"]["episodes"] == 1
    assert isinstance(report["passed"], bool)
    assert report["passed"] == (not report["failures"])
    # The report is the run artifact: it must serialize as-is.
    json.dumps(report)
    # And the human summary must render from the same payload.
    assert "LADDER CERTIFICATION" in lc.format_report(report)


def test_certify_ladder_rejects_malformed_probes():
    with pytest.raises(ValueError, match="cover every stage"):
        lc.certify_ladder(
            _tiny_env_fn,
            [STAGE_0, STAGE_1],
            oracle_probes=[{"run_up": 1.1}],
        )
    with pytest.raises(ValueError, match="exactly one"):
        lc.certify_ladder(
            _tiny_env_fn,
            [STAGE_0],
            oracle_probes=[{"run_up": 1.1, "charge_gap": 1.0}],
        )
    with pytest.raises(ValueError, match="positive integer"):
        lc.certify_ladder(
            _tiny_env_fn,
            [STAGE_0],
            episodes=0,
            oracle_probes=[{"run_up": 1.1}],
        )


def _fake_report(passed=True):
    return {
        "schema": 1,
        "provenance": {"episodes_per_cell": 1, "seed_start": 30_000},
        "gate": {"metric_key": "bounce_count_ep_mean", "threshold": 3.0},
        "stages": [],
        "failures": [] if passed else ["stage 0: synthetic failure"],
        "warnings": [],
        "passed": passed,
    }


def test_run_startup_certification_validates_spec_and_writes(tmp_path,
                                                             monkeypatch):
    seen = {}

    def fake_certify(env_fn, stages, **kwargs):
        seen["stages"] = stages
        seen["kwargs"] = kwargs
        return _fake_report()

    monkeypatch.setattr(lc, "certify_ladder", fake_certify)
    cfg = SimpleNamespace(
        env_fn=_tiny_env_fn,
        performance_gate={
            "metric_key": "bounce_count_ep_mean",
            "threshold": 3.0,
            "sustain_evals": 3,
            "stages": (STAGE_0, STAGE_1),
        },
        ladder_certification={
            "episodes": 1,
            "oracle_probes": ({"run_up": 1.1}, {"charge_gap": 1.0}),
        },
    )
    out = tmp_path / "reports" / "ladder_certification.json"
    report = lc.run_startup_certification(cfg, str(out))
    assert report["passed"]
    assert seen["stages"] == [dict(STAGE_0), dict(STAGE_1)]
    assert seen["kwargs"]["gate_threshold"] == 3.0
    assert json.loads(out.read_text())["passed"] is True


def test_run_startup_certification_rejects_bad_specs(tmp_path):
    base = dict(
        env_fn=_tiny_env_fn,
        performance_gate={
            "metric_key": "m", "threshold": 1.0, "sustain_evals": 1,
            "stages": (STAGE_0,),
        },
    )
    with pytest.raises(ValueError, match="unknown ladder_certification"):
        lc.run_startup_certification(
            SimpleNamespace(
                **base,
                ladder_certification={"orackle_probes": ()},
            ),
            str(tmp_path / "r.json"),
        )
    with pytest.raises(ValueError, match="performance_gate with stages"):
        lc.run_startup_certification(
            SimpleNamespace(
                env_fn=_tiny_env_fn,
                performance_gate=None,
                ladder_certification={"oracle_probes": ()},
            ),
            str(tmp_path / "r.json"),
        )


def test_probe_stage_count_mismatch_records_a_skip_not_a_crash(tmp_path):
    """The documented continuation workflow shrinks the gate wholesale
    while the recipe's full probe table survives the merge — advisory
    certification must skip with a report, not refuse to train."""
    cfg = SimpleNamespace(
        env_fn=_tiny_env_fn,
        performance_gate={
            "metric_key": "bounce_count_ep_mean",
            "threshold": 3.0,
            "sustain_evals": 3,
            "stages": (STAGE_0, STAGE_1),
        },
        ladder_certification={
            "oracle_probes": (
                {"run_up": 1.1},
                {"charge_gap": 1.0},
                {"charge_gap": 1.0},
                {"charge_gap": 1.8},
                {"charge_gap": 1.7},
            ),
        },
    )
    out = tmp_path / "r.json"
    report = lc.run_startup_certification(cfg, str(out))
    assert report["skipped"] is True
    assert report["passed"] is False
    assert "5 oracle probe(s) for 2 gate stage(s)" in report["failures"][0]
    assert json.loads(out.read_text())["skipped"] is True
    # ...but enforce still turns the skip into a hard stop.
    cfg.ladder_certification = {
        **cfg.ladder_certification, "enforce": True,
    }
    with pytest.raises(RuntimeError, match="certification skipped"):
        lc.run_startup_certification(cfg, str(out))


def test_missing_oracle_probes_is_a_contextual_error(tmp_path):
    cfg = SimpleNamespace(
        env_fn=_tiny_env_fn,
        performance_gate={
            "metric_key": "m", "threshold": 1.0, "sustain_evals": 1,
            "stages": (STAGE_0,),
        },
        ladder_certification={"episodes": 5},
    )
    with pytest.raises(ValueError, match="oracle_probes"):
        lc.run_startup_certification(cfg, str(tmp_path / "r.json"))


def test_toml_layer_validates_ladder_certification_at_load(tmp_path):
    """Broken specs fail at load with the file named (repo cardinal
    rule: no silent no-ops, no bare startup KeyErrors)."""
    from courtside_dynamics.run_config import load_run_config

    def write(body):
        path = tmp_path / "run.toml"
        path.write_text(body)
        return path

    good = write(
        "[train.ladder_certification]\n"
        "episodes = 5\n"
        "oracle_probes = [{ run_up = 1.1 }, { charge_gap = 1.0 }]\n"
    )
    loaded = load_run_config(good)
    assert loaded.train["ladder_certification"]["episodes"] == 5

    disabled = write('[train]\nladder_certification = "none"\n')
    assert load_run_config(disabled).train["ladder_certification"] is None

    with pytest.raises(ValueError, match="did you mean 'oracle_probes'"):
        load_run_config(write(
            "[train.ladder_certification]\n"
            "orace_probes = [{ run_up = 1.1 }]\n"
        ))
    with pytest.raises(ValueError, match="must define oracle_probes"):
        load_run_config(write(
            "[train.ladder_certification]\nepisodes = 5\n"
        ))
    with pytest.raises(ValueError, match="exactly one"):
        load_run_config(write(
            "[train.ladder_certification]\n"
            "oracle_probes = [{ run_up = 1.1, charge_gap = 1.0 }]\n"
        ))
    with pytest.raises(ValueError, match="episodes must be an integer"):
        load_run_config(write(
            "[train.ladder_certification]\n"
            "episodes = 0\n"
            "oracle_probes = [{ run_up = 1.1 }]\n"
        ))


def test_run_startup_certification_enforce_raises_on_failure(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(
        lc, "certify_ladder", lambda *a, **k: _fake_report(passed=False)
    )
    cfg = SimpleNamespace(
        env_fn=_tiny_env_fn,
        performance_gate={
            "metric_key": "m", "threshold": 1.0, "sustain_evals": 1,
            "stages": (STAGE_0,),
        },
        ladder_certification={
            "enforce": True,
            "oracle_probes": ({"run_up": 1.1},),
        },
    )
    out = tmp_path / "r.json"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        lc.run_startup_certification(cfg, str(out))
    # Advisory-by-default is the policy; enforcement must still leave
    # the evidence on disk for the post-mortem.
    assert json.loads(out.read_text())["passed"] is False


def test_recipe_spec_exists_covers_every_stage_and_serializes():
    """The recipe's spec is the anti-staleness contract this PR ships."""
    from courtside_dynamics.recipes import RECIPES

    for name in ("WallBallDepthCurriculum", "WallBallDepthCurriculumAligned"):
        recipe = RECIPES[name]
        spec = recipe.extra_cfg["ladder_certification"]
        stages = recipe.extra_cfg["performance_gate"]["stages"]
        assert len(spec["oracle_probes"]) == len(stages)
        for probe in spec["oracle_probes"]:
            assert set(probe) in (
                {"run_up"}, {"charge_gap"}, {"lead_charge"}
            )
        # Reserved probe block, disjoint from every burned ledger range.
        assert spec["seed_start"] >= 30_000
        json.dumps(spec)


def test_release_cli_certifies_the_live_recipe_stages(monkeypatch, tmp_path):
    """`--ladder release` must read stages from the recipe at call time."""
    from courtside_dynamics.recipes import RECIPES
    from tools import depth_stage_sweep

    seen = {}

    def fake_certify(env_fn, stages, **kwargs):
        seen["stages"] = stages
        env = env_fn()
        try:
            assert isinstance(env.unwrapped, WallBallEnv)
        finally:
            env.close()
        return _fake_report()

    monkeypatch.setattr(lc, "certify_ladder", fake_certify)
    args = argparse.Namespace(
        episodes=1,
        json_output=str(tmp_path / "release.json"),
        seed_start=None,
        ladder="release",
        serve_origin_blend=None,
        oracle_probe=None,
        calibration_seed_start=None,
        calibration_episodes=None,
        calibration_artifact=None,
    )
    assert depth_stage_sweep._run_release_certification(args) == 0

    expected = [
        dict(stage)
        for stage in RECIPES["WallBallDepthCurriculum"].extra_cfg[
            "performance_gate"
        ]["stages"]
    ]
    assert seen["stages"] == expected
    assert json.loads((tmp_path / "release.json").read_text())["passed"]


@pytest.mark.parametrize(
    "override",
    [
        {"serve_origin_blend": 0.5},
        {"oracle_probe": ["0=run_up:1.0"]},
        {"calibration_seed_start": 100},
        {"calibration_episodes": 50},
        {"calibration_artifact": "x.json"},
    ],
)
def test_release_cli_rejects_candidate_only_flags(override):
    from tools import depth_stage_sweep

    args = argparse.Namespace(
        episodes=1,
        json_output=None,
        seed_start=None,
        ladder="release",
        serve_origin_blend=None,
        oracle_probe=None,
        calibration_seed_start=None,
        calibration_episodes=None,
        calibration_artifact=None,
    )
    for key, value in override.items():
        setattr(args, key, value)
    with pytest.raises(SystemExit):
        depth_stage_sweep._run_release_certification(args)


@pytest.mark.timeout(240)
def test_train_certifies_at_startup_and_records_the_artifact(tmp_path,
                                                             capsys):
    """End to end: a gated run certifies itself before training."""
    from courtside_dynamics.training.train import TrainConfig, train

    cfg = TrainConfig(
        env_fn=lambda: WallBallEnv(
            episode_len=60,
            rally_style="open",
            min_force=20.0,
            paddle_x_target_range=(-4.7, 0.3),
            paddle_home_x=-1.25,
        ),
        algo="SAC",
        total_timesteps=300,
        log_dir=str(tmp_path),
        n_envs=1,
        seed=0,
        eval_freq=150,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        normalize_obs=False,
        n_eval_episodes=1,
        info_dict_eval=True,
        headline_key="bounce_count",
        model_kwargs={"learning_starts": 16, "buffer_size": 400},
        performance_gate={
            "metric_key": "bounce_count_ep_mean",
            "threshold": 3.0,
            "sustain_evals": 3,
            "stages": (STAGE_0, STAGE_1),
        },
        ladder_certification={
            "episodes": 1,
            "max_episode_steps": 40,
            "oracle_probes": ({"run_up": 1.1}, {"charge_gap": 1.0}),
        },
    )
    train(cfg)

    report = json.loads(
        (tmp_path / "reports" / "ladder_certification.json").read_text()
    )
    assert report["schema"] == 1
    assert len(report["stages"]) == 2
    assert "LADDER CERTIFICATION" in capsys.readouterr().out

    config = json.loads((tmp_path / "config.json").read_text())
    recorded = config["train_config"]["ladder_certification"]
    assert recorded["episodes"] == 1


def test_lead_charge_probe_mode_certifies_and_serializes():
    """The recalibrated probe family: charge-gap trigger + ballistic
    lead while charging. Review 20260727_233859 measured the stock
    y/z-tracking charge failing the 90% feasibility floor at stages 0
    and 3 of the constant-width ladder; the lead variant is what the
    recipe's oracle_probes now use, so the certifier must accept it as
    a first-class mode."""
    report = lc.certify_ladder(
        _tiny_env_fn,
        [STAGE_0, STAGE_1],
        episodes=1,
        seed_start=lc.DEFAULT_SEED_START,
        oracle_probes=[{"lead_charge": 3.0}, {"lead_charge": 0.8}],
        gate_metric_key="bounce_count_ep_mean",
        gate_threshold=3.0,
        max_episode_steps=60,
    )
    assert len(report["stages"]) == 2
    for stage in report["stages"]:
        assert set(stage["oracle_probe"]) == {"lead_charge"}
    json.dumps(report)


def test_lead_charge_is_mutually_exclusive_with_other_modes():
    with pytest.raises(ValueError, match="exactly one"):
        lc.certify_ladder(
            _tiny_env_fn,
            [STAGE_0],
            oracle_probes=[{"lead_charge": 3.0, "charge_gap": 1.0}],
        )
    with pytest.raises(ValueError, match="exactly one of run_up"):
        lc._oracle_action(
            __import__("numpy").zeros(23),
            (-2.3, -0.2),
            -1.25,
            (-4.7, 0.3),
            None,
            1.0,
            2.0,
        )


def test_lead_charge_leads_the_intercept_while_charging():
    """While charging, the lead controller must aim at the ballistic
    intercept, not the ball's current y — that lateral scramble is the
    measured failure of the stock charge on the constant-width fences."""
    import numpy as np

    obs = np.zeros(23)
    obs[0:3] = (-1.0, 0.6, 1.0)   # ball at x=-1.0, y=0.6, z=1.0
    obs[3:6] = (-3.0, 1.0, 0.5)   # incoming, drifting +y
    obs[13] = 1.0                 # post-bounce
    obs[14] = -1.0 - (-2.1)       # paddle at start (dx = ball - paddle)
    fence, home, mapping = (-2.9, -0.8), -1.85, (-4.7, 0.3)
    lead = lc._oracle_action(obs, fence, home, mapping, None, None, 3.0)
    track = lc._oracle_action(obs, fence, home, mapping, None, 3.0, None)
    # Same commit decision (x identical), different lateral aim.
    assert lead[0] == track[0]
    assert lead[1] > track[1]  # leads the +y drift instead of tracking
