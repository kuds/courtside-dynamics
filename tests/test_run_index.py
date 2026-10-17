"""The regenerated run index and the legacy backfill (design L1, second half).

``docs/design_campaign_structure_20261016.md`` step L1. Per-run records
are the source of truth and ``<root>/index/runs.jsonl`` is derived from
them, so these tests pin: every kind of run directory becomes one
normalized row (a run record as written plus its manifest verdict; a
pre-record directory backfilled from its ``config.json`` and
``stage_summary.txt``; an aborted launch as a stub), nothing else is
ever parsed as a run, regeneration is byte-identical, no machine path
reaches a backfilled row, and the curated campaign map joins by run id.

``tests/fixtures/run_index/training_runs`` holds five real archived run
directories (text files copied from the Drive archive with machine
paths scrubbed; model and checkpoint files as empty placeholders).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from courtside_dynamics.experiments import index as run_index
from courtside_dynamics.experiments import records
from courtside_dynamics.experiments.index import (
    ABORTED,
    RUN_INDEX_SCHEMA,
    build_index,
    load_legacy_levers,
    main,
    parse_stage_summary,
)
from courtside_dynamics.experiments.records import (
    RUN_RECORD_NAME,
    SYNC_MANIFEST_NAME,
    build_run_record,
    write_run_record,
    write_sync_manifest,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "run_index" / "training_runs"

L2_PILOT = "PaddleTennis/sac/20260810_211754"
L2W = "PaddleTennis/sac/20260815_180815"
CPU_FALSE_START = "PaddleTennis/sac/20260828_114038"
WALL_BALL_RUN_1 = "WallBallBaseline/sac/20260714_050506"
GOAL_RALLY_FIRST_LAUNCH = "WallBallGoalRally/sac/20260728_224706"


def _rows_by_path(result: run_index.RunIndex) -> dict[str, dict]:
    return {row["index"]["path"]: row for row in result.rows}


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "training_runs"
    shutil.copytree(FIXTURE_ROOT, root)
    return root


# -- module contracts ----------------------------------------------------------


def test_index_imports_without_sb3_or_torch():
    """The index runs on a Colab runtime or laptop without the train extra."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["stable_baselines3"] = None
        sys.modules["torch"] = None
        import courtside_dynamics.experiments.index as index
        assert index.load_legacy_levers()
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_index_layout_and_families_match_the_training_package():
    """The index keeps local copies (importing training would import SB3)."""
    from courtside_dynamics.training import artifacts
    from courtside_dynamics.training.algos import algo_family, resolve_algo

    for name, (new, legacy) in run_index._LAYOUT.items():
        assert new == artifacts.RUN_LAYOUT[name], name
        assert legacy == artifacts._legacy_artifact_relpath(name), name
    assert artifacts.RUN_LAYOUT["run_record"] == RUN_RECORD_NAME
    assert artifacts.RUN_LAYOUT["sync_manifest"] == SYNC_MANIFEST_NAME
    for name, family in run_index._ALGO_FAMILIES.items():
        assert resolve_algo(name).__name__ == name
        assert algo_family(name) == family


def test_the_campaign_map_ships_as_package_data():
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        package_data = tomllib.load(handle)["tool"]["setuptools"]["package-data"]
    assert "*.toml" in package_data["courtside_dynamics.experiments"]


# -- the legacy backfill over real archived run directories -------------------


def test_fixture_runs_are_backfilled_into_the_record_shape():
    result = build_index(FIXTURE_ROOT)
    rows = _rows_by_path(result)
    assert sorted(rows) == sorted(
        [L2_PILOT, L2W, CPU_FALSE_START, WALL_BALL_RUN_1, GOAL_RALLY_FIRST_LAUNCH]
    )
    assert result.skipped == ()
    record_keys = set(
        build_run_record({}, run_id="r", created_at="t", status=records.RUNNING)
    )
    for row in rows.values():
        assert set(row) == record_keys | {"index", "campaign"}
        assert row["schema"] == records.RUN_RECORD_SCHEMA
        assert row["index"]["schema"] == RUN_INDEX_SCHEMA
        assert row["index"]["source"] == "legacy"
        assert row["index"]["version_inferred"] is True
        assert row["index"]["verification"] is None
        assert row["index"]["problems"] == []
        assert row["algo"] == {"name": "SAC", "family": "SAC", "impl": "sb3==2.9.0"}
        assert row["experiment"] == {
            "id": None,
            "parent": None,
            "lever": None,
            "kind": None,
        }

    # The n-point L2 pilot: an early stop the summary labels "completed";
    # its "X of Y budget" line tells them apart.
    pilot = rows[L2_PILOT]
    assert pilot["status"] == records.EARLY_STOP
    assert pilot["stop_reason"].startswith("early_stop_patience: no improvement")
    assert pilot["budget"] == {
        "total_timesteps": 2_000_000,
        "trained_timesteps": 1_125_000,
        "n_updates": None,
    }
    assert pilot["git"] == {
        "sha": "2e597b1f3bd2bb9dfaec08d92f4195368e91d398",
        "dirty": None,
    }
    assert pilot["recipe"] == "PaddleTennis"
    assert pilot["selection"]["timestep"] == 625_000
    assert pilot["index"]["summary"]["best_model_step"] == 625_000
    assert pilot["index"]["summary"]["headline_best"] == 3.3
    assert pilot["index"]["files"]["latest_checkpoint_step"] == 1_100_000
    assert pilot["index"]["run_config"] == {
        "name": "paddle_tennis_npoint_pilot.toml",
        "sha256": "79f6f3a67a9ca05a38b567274b2ffe1459b9b018b2d31e5b31888882596066b9",
    }
    assert pilot["wall_clock"]["total_seconds"] == 7 * 3600 + 49 * 60 + 36
    assert pilot["index"]["inferred"] == [
        "algo",
        "budget.trained_timesteps",
        "status",
        "stop_reason",
        "wall_clock",
    ]

    # L2W: a stop reason on a full-budget run is the train-3 label; the
    # row reads it as train() records it now.
    l2w = rows[L2W]
    assert l2w["status"] == records.COMPLETED
    assert l2w["budget"]["trained_timesteps"] == 1_000_000
    assert l2w["stop_reason"].startswith(
        "full_budget: trained all 1,000,000 steps; the stop condition also "
        "held at the final evaluation (early_stop_patience: "
    )
    assert l2w["warm_start"]["source_run_dir_name"] == "20260809_211147"
    assert set(l2w["warm_start"]["artifact_sha256"]) == {
        "best_model.zip",
        "best_vec_normalize.pkl",
        "config.json",
    }

    # A CPU false start: config.json and a first-evaluation best model,
    # no summary, no final model.
    stub = rows[CPU_FALSE_START]
    assert stub["status"] == ABORTED
    assert stub["stop_reason"] is None
    assert stub["index"]["stub"] is True
    assert stub["budget"]["trained_timesteps"] is None
    assert stub["index"]["summary"] is None
    assert stub["selection"]["timestep"] == 25_000
    assert stub["index"]["files"]["best_model"] is True
    assert stub["index"]["files"]["final_model"] is False

    # A salvaged launch (final model saved, no summary) is aborted but
    # not a stub.
    salvaged = rows[GOAL_RALLY_FIRST_LAUNCH]
    assert salvaged["status"] == ABORTED
    assert salvaged["index"]["stub"] is False
    assert salvaged["budget"]["total_timesteps"] == 3_000_000

    # The first WallBall run: pre-0.14 flat layout, no git sha, no
    # recipe name, an early stop whose summary names no reason.
    wall_ball = rows[WALL_BALL_RUN_1]
    assert wall_ball["status"] == records.EARLY_STOP
    assert wall_ball["stop_reason"] == (
        "stopped_early: trained 1,100,000 of 1,500,000 steps; the summary "
        "records no stop reason"
    )
    assert wall_ball["git"]["sha"] is None
    assert wall_ball["index"]["summary"]["git_sha"] is None
    assert wall_ball["recipe"] is None
    assert wall_ball["index"]["run_config"] is None
    assert wall_ball["selection"]["timestep"] == 600_000
    files = wall_ball["index"]["files"]
    assert files["best_model"] and files["final_model"] and files["best_model_meta"]
    assert files["checkpoints"] == 1
    assert files["latest_checkpoint_step"] == 1_000_000

    # Every row joins the campaign map by run id.
    assert rows[L2W]["campaign"]["lever"] == "npoint"
    assert rows[CPU_FALSE_START]["campaign"]["lever"] == "temperature_skip"
    assert rows[WALL_BALL_RUN_1]["campaign"]["lever"] == "era/wall_ball_baseline"


def test_no_machine_path_reaches_a_backfilled_row():
    """The fixture configs hold (scrubbed) absolute paths in log_dir, the
    run config file and the warm start; none of them is copied."""
    text = "\n".join(build_index(FIXTURE_ROOT).lines())
    configs = "".join(path.read_text() for path in FIXTURE_ROOT.rglob("config.json"))
    assert "/scrubbed/" in configs
    assert "/scrubbed/" not in text
    assert str(FIXTURE_ROOT) not in text


def test_regenerating_the_index_is_byte_identical(archive, capsys):
    assert main(["--root", str(archive)]) == 0
    out = archive / "index" / "runs.jsonl"
    first = out.read_bytes()
    # The index directory now sits under the root: it is never scanned.
    assert main(["--root", str(archive)]) == 0
    assert out.read_bytes() == first
    other = archive.parent / "elsewhere.jsonl"
    assert main(["--root", str(archive), "--out", str(other)]) == 0
    assert other.read_bytes() == first
    lines = first.decode("utf-8").splitlines()
    assert len(lines) == 5
    paths = [json.loads(line)["index"]["path"] for line in lines]
    assert paths == sorted(paths)
    for line in lines:
        assert line == json.dumps(json.loads(line), sort_keys=True)
    printed = capsys.readouterr().out
    assert "5 run(s), 0 from run records and 5 backfilled (2 aborted, 1 of them" in printed
    assert "5 joined to the campaign map" in printed
    assert "0 directories skipped" in printed


# -- what is and is not a run ---------------------------------------------------


def test_an_empty_run_named_directory_is_an_aborted_stub(tmp_path):
    (tmp_path / "PaddleTennis" / "sac" / "20260828_112830").mkdir(parents=True)
    (tmp_path / "PaddleTennis" / "sac" / "20260815_180638" / "media").mkdir(
        parents=True
    )
    rows = _rows_by_path(build_index(tmp_path))
    assert sorted(rows) == [
        "PaddleTennis/sac/20260815_180638",
        "PaddleTennis/sac/20260828_112830",
    ]
    for row in rows.values():
        assert row["status"] == ABORTED
        assert row["index"]["stub"] is True
        assert row["index"]["files"]["config"] is False
        assert row["algo"] == {"name": None, "family": None, "impl": None}
    assert rows["PaddleTennis/sac/20260828_112830"]["campaign"]["lever"] == (
        "temperature_skip"
    )


def test_directories_that_are_not_courtside_runs_are_skipped_never_parsed(
    tmp_path,
):
    shutil.copytree(FIXTURE_ROOT / L2W, tmp_path / "campaign" / "gate" / "a01")
    (tmp_path / "campaign" / "campaign_manifest.json").write_text("{}")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "lt1.toml").write_text("[train]\n")
    (tmp_path / "other_project").mkdir()
    (tmp_path / "other_project" / "config.json").write_text('{"model": "bert"}')
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{")
    monitor = tmp_path / "PaddleTennis" / "sac" / "20260101_000000" / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "0.monitor.csv").write_text("#\n")
    (tmp_path / "README.md").write_text("a note at the root is not a skip\n")

    result = build_index(tmp_path)
    assert [row["index"]["path"] for row in result.rows] == ["campaign/gate/a01"]
    reasons = {skipped.path: skipped.reason for skipped in result.skipped}
    assert sorted(reasons) == [
        "PaddleTennis/sac/20260101_000000",
        "broken",
        "campaign",
        "configs",
        "other_project",
    ]
    assert reasons["other_project"].startswith(
        "config.json is not a courtside run config"
    )
    assert reasons["broken"].startswith("config.json is unreadable")
    assert reasons["PaddleTennis/sac/20260101_000000"].startswith(
        "named like a run but holds no config.json"
    )
    assert reasons["configs"] == "holds files but is not a run directory"


def test_videos_and_checkpoints_are_counted_in_either_layout(tmp_path):
    new = tmp_path / "Env" / "sac" / "20260101_000001"
    shutil.copytree(FIXTURE_ROOT / L2W, new)
    (new / "media" / "videos").mkdir(parents=True)
    (new / "media" / "videos" / "a-episode-0.mp4").write_bytes(b"")
    (new / "media" / "videos" / "a-episode-0.meta.json").write_text("{}")
    flat = tmp_path / "Env" / "sac" / "20260101_000002"
    shutil.copytree(FIXTURE_ROOT / WALL_BALL_RUN_1, flat)
    (flat / "videos").mkdir()
    for step in (250_000, 1_000_000):
        (flat / "videos" / f"step-{step}.mp4").write_bytes(b"")
    (flat / "checkpoints" / "wall_ball_baseline_sac_250000_steps.zip").write_bytes(
        b""
    )
    rows = _rows_by_path(build_index(tmp_path))
    assert rows["Env/sac/20260101_000001"]["index"]["files"]["videos"] == 1
    flat_files = rows["Env/sac/20260101_000002"]["index"]["files"]
    assert flat_files["videos"] == 2
    assert flat_files["checkpoints"] == 2
    assert flat_files["latest_checkpoint_step"] == 1_000_000


# -- run records ----------------------------------------------------------------


def _record_run(path: Path) -> dict:
    shutil.copytree(FIXTURE_ROOT / L2W, path)
    config = json.loads((path / "config.json").read_text())
    record = build_run_record(
        config,
        run_id=path.name,
        created_at="2026-10-17T10:00:00+00:00",
        status=records.COMPLETED,
        finished_at="2026-10-17T12:00:00+00:00",
        trained_timesteps=1_000_000,
    )
    write_run_record(path, record)
    write_sync_manifest(path)
    return json.loads((path / RUN_RECORD_NAME).read_text())


def test_a_run_record_is_taken_as_written_with_its_manifest_verdict(tmp_path):
    run_dir = tmp_path / "PaddleTennis" / "sac" / "20261017_100000"
    record = _record_run(run_dir)
    row = build_index(tmp_path).rows[0]
    assert {key: row[key] for key in record} == record
    assert row["index"]["source"] == "run_record"
    assert row["index"]["version_inferred"] is False
    assert row["index"]["inferred"] == []
    assert row["index"]["verification"] == {
        "manifest": "verified",
        "complete": True,
        "files_checked": len(records.run_files(run_dir)),
        "problems": [],
        "unlisted": [],
    }
    assert row["index"]["files"]["run_record"] is True
    assert row["index"]["files"]["sync_manifest"] is True
    assert row["campaign"] is None

    (run_dir / "model" / "final_model.zip").write_bytes(b"changed")
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "learning_curve.png").write_bytes(b"png")
    verification = build_index(tmp_path).rows[0]["index"]["verification"]
    assert verification["manifest"] == "mismatch"
    assert verification["complete"] is False
    assert verification["unlisted"] == ["reports/learning_curve.png"]

    skipped = build_index(tmp_path, verify=False).rows[0]["index"]["verification"]
    assert skipped["manifest"] == "not_checked"
    assert skipped["complete"] is None


def test_an_unreadable_run_record_falls_back_to_the_backfill_and_says_so(tmp_path):
    run_dir = tmp_path / "PaddleTennis" / "sac" / "20260815_180815"
    _record_run(run_dir)
    (run_dir / RUN_RECORD_NAME).write_text("{")
    row = build_index(tmp_path).rows[0]
    assert row["index"]["source"] == "legacy"
    assert row["status"] == records.COMPLETED
    assert row["campaign"]["lever"] == "npoint"
    assert len(row["index"]["problems"]) == 1
    assert "backfilled from config.json instead" in row["index"]["problems"][0]


# -- stage_summary.txt ----------------------------------------------------------


_SUMMARY = """courtside-dynamics: PPO on PaddleTennis
==================================================

Project:        courtside-dynamics
Status:         {status}
Git SHA:        abc1234
Timesteps:      {timesteps}
SB3 n_updates:  1,230 (optimization epochs over all rollouts)
Duration:       0h 02m 05s
Best model:     step 4,096 (crossings_ep_mean 1.50)  [task-metric selection]
Headline best:  -1.5e-03 (at 4,096 steps)  [eval_info series]

Device
----------------------------------------
Status:         ignored, a section line
"""


def test_parse_stage_summary_reads_the_head_block():
    parsed = parse_stage_summary(
        _SUMMARY.format(status="completed", timesteps="8,192 of 8,000 budget (+192)")
    )
    assert parsed["status"] == "completed"
    assert parsed["git_sha"] == "abc1234"
    assert parsed["trained_timesteps"] == 8_192
    assert parsed["budget_timesteps"] == 8_000
    assert parsed["n_updates"] == 1_230
    assert parsed["duration_seconds"] == 125
    assert parsed["best_model_step"] == 4_096
    assert parsed["headline_best"] == -1.5e-3
    assert parsed["headline_best_step"] == 4_096
    assert parsed["stop_reason"] is None
    assert parsed["best_eval_step"] is None


@pytest.mark.parametrize(
    ("status", "timesteps", "expected", "problem"),
    [
        ("interrupted", "4,096 of 8,000 budget (stopped early)", "interrupted", None),
        ("crashed", "4,096 of 8,000 budget (stopped early)", "crashed", None),
        ("finished", "8,000", None, "no known Status ('finished')"),
        (
            "completed",
            "7,000 of 9,000 budget (stopped early)",
            "early_stop",
            "differs from config",
        ),
    ],
)
def test_legacy_outcomes_follow_the_summary(
    tmp_path, status, timesteps, expected, problem
):
    run_dir = tmp_path / "Env" / "ppo" / "20260101_000000"
    run_dir.mkdir(parents=True)
    config = {"train_config": {"algo": "PPO", "total_timesteps": 8_000}}
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "stage_summary.txt").write_text(
        _SUMMARY.format(status=status, timesteps=timesteps)
    )
    row = build_index(tmp_path, legacy_levers={}).rows[0]
    assert row["status"] == expected
    assert row["algo"] == {"name": "PPO", "family": "PPO", "impl": None}
    assert row["budget"]["n_updates"] == 1_230
    if problem is None:
        assert row["index"]["problems"] == []
    else:
        assert any(problem in text for text in row["index"]["problems"])


# -- the campaign map -------------------------------------------------------------


#: Every PaddleTennis run directory in the Drive archive: 15 completed
#: runs and 5 aborted launches.
_PADDLE_TENNIS_RUNS = {
    "20260803_004559": "baseline",
    "20260808_022106": "baseline",
    "20260809_005951": "exploration_package",
    "20260809_161704": "contact_shaping",
    "20260809_211147": "contact_shaping",
    "20260810_211754": "npoint",
    "20260815_015143": "npoint",
    "20260815_180815": "npoint",
    "20260816_135919": "reach_shaping",
    "20260816_235141": "registered_run",
    "20260818_210727": "hold_escrow",
    "20260820_013545": "hold_escrow",
    "20260821_013700": "hold_escrow",
    "20260828_113136": "temperature_skip",
    "20260828_121324": "temperature_skip",
    "20260815_180638": "infra",
    "20260816_135115": "infra",
    "20260828_112830": "temperature_skip",
    "20260828_114038": "temperature_skip",
    "20260828_114815": "temperature_skip",
}


def test_every_paddle_tennis_run_maps_to_exactly_one_lever():
    campaign_map = load_legacy_levers()
    paddle = {
        run_id: entry["lever"]
        for run_id, entry in campaign_map.items()
        if entry["campaign"] == "paddle_tennis"
    }
    assert paddle == _PADDLE_TENNIS_RUNS
    wall_ball = [e for e in campaign_map.values() if e["campaign"] == "wall_ball"]
    assert len(wall_ball) == 27
    assert all(entry["lever"].startswith("era/") for entry in wall_ball)
    assert len(campaign_map) == len(paddle) + len(wall_ball)
    for run_id, entry in campaign_map.items():
        assert entry["verdict"]
        if "doc" in entry:
            assert (REPOSITORY_ROOT / entry["doc"]).is_file(), (run_id, entry["doc"])
        else:
            assert entry["note"], run_id


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('[runs."r"]\ncampaign = "c"\nlever = "l"\nverdict = "v"\nwho = "x"\n', "unknown key"),
        ('[runs."r"]\ncampaign = "c"\nverdict = "v"\ndoc = "d"\n', "missing key"),
        ('[runs."r"]\ncampaign = "c"\nlever = "l"\nverdict = "v"\n', "a note must say"),
        ('[runs."r"]\ncampaign = "c"\nlever = ""\nverdict = "v"\ndoc = "d"\n', "non-empty"),
        ('title = "t"\n', "unknown top-level"),
        ("[runs\n", "not valid TOML"),
    ],
)
def test_the_campaign_map_is_strict(tmp_path, text, match):
    path = tmp_path / "levers.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_legacy_levers(path)


def test_the_cli_joins_a_given_campaign_map(archive, tmp_path):
    levers = tmp_path / "levers.toml"
    levers.write_text(
        '[runs."20260714_050506"]\ncampaign = "wall_ball"\nlever = "era/x"\n'
        'verdict = "v"\nnote = "a custom map"\n'
    )
    out = tmp_path / "runs.jsonl"
    assert main(
        ["--root", str(archive), "--out", str(out), "--legacy-levers", str(levers)]
    ) == 0
    rows = {
        json.loads(line)["run_id"]: json.loads(line)
        for line in out.read_text().splitlines()
    }
    assert rows["20260714_050506"]["campaign"] == {
        "campaign": "wall_ball",
        "lever": "era/x",
        "verdict": "v",
        "note": "a custom map",
    }
    assert rows["20260815_180815"]["campaign"] is None
