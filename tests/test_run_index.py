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


def test_no_machine_path_reaches_a_backfilled_row(archive):
    """The fixture configs hold (scrubbed) absolute paths in log_dir, the
    run config file and the warm start; none of them is copied. Nor do
    problem messages carry one (verifier r1: a malformed meta file put
    its absolute path into the row), however the root is spelled."""
    text = "\n".join(build_index(FIXTURE_ROOT).lines())
    configs = "".join(path.read_text() for path in FIXTURE_ROOT.rglob("config.json"))
    assert "/scrubbed/" in configs
    assert "/scrubbed/" not in text
    assert str(FIXTURE_ROOT) not in text

    (archive / L2W / "model" / "best_model_meta.json").write_text('{"timestep": ')
    (archive / L2_PILOT / "stage_summary.txt").write_bytes(b"\xff\xfe")
    (archive / "PaddleTennis" / "sac" / "20260101_000000").mkdir()
    (archive / "PaddleTennis" / "sac" / "20260101_000000" / "config.json").write_text("")
    damaged = build_index(archive.resolve())
    text = "\n".join(damaged.lines())
    problems = [p for row in damaged.rows for p in row["index"]["problems"]]
    assert len(problems) == 3
    assert any(
        p.startswith(
            "the selection is omitted: model/best_model_meta.json is unreadable: "
            "JSONDecodeError"
        )
        for p in problems
    )
    assert str(archive.resolve()) not in text
    assert str(archive.parent) not in text


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
    assert (
        "5 run(s): 0 from run records (0 still say running: training, or "
        "dead), 5 backfilled (2 aborted); 1 stub(s); 5 joined to the "
        "campaign map"
    ) in printed
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


def test_a_stray_config_json_never_hides_the_runs_beneath_it(tmp_path):
    """Verifier r1: the scan stopped at any directory with config.json,
    so a run's config.json copied into its recipe folder made that
    folder an aborted stub row ("sac") and hid the real run beneath it,
    with no skip and no problem; a foreign config.json higher up hid
    the whole subtree."""
    run = tmp_path / "R" / "sac" / "20260101_000000"
    shutil.copytree(FIXTURE_ROOT / L2W, run)
    shutil.copy(run / "config.json", tmp_path / "R" / "sac" / "config.json")
    result = build_index(tmp_path, legacy_levers={})
    rows = _rows_by_path(result)
    assert sorted(rows) == ["R/sac", "R/sac/20260101_000000"]
    assert rows["R/sac/20260101_000000"]["status"] == records.COMPLETED
    assert rows["R/sac/20260101_000000"]["index"]["problems"] == []
    assert rows["R/sac"]["index"]["problems"] == [
        "holds nested run directories (20260101_000000), indexed as rows of "
        "their own"
    ]

    (tmp_path / "R" / "sac" / "config.json").unlink()
    (tmp_path / "R" / "config.json").write_text('{"model": "bert"}')
    result = build_index(tmp_path, legacy_levers={})
    assert [row["index"]["path"] for row in result.rows] == ["R/sac/20260101_000000"]
    (skipped,) = result.skipped
    assert skipped.path == "R"
    assert skipped.reason.startswith("config.json is not a courtside run config")
    assert skipped.reason.endswith(
        "it holds nested run directories (sac), indexed as rows of their own"
    )


@pytest.mark.parametrize(
    ("config_text", "problem"),
    [
        ("", "config.json is unreadable: JSONDecodeError: Expecting value"),
        ('{"model": "bert"}', "config.json is not a courtside run config"),
    ],
)
def test_a_run_named_directory_with_a_broken_config_is_a_row(
    tmp_path, config_text, problem
):
    """Verifier r1: a zero-byte config.json (a partial Drive sync) in a
    run-named directory was only listed as skipped, so the dead launch
    dropped out of status queries. It is an aborted row now, carrying
    the config problem; a stub without a final model."""
    run = tmp_path / "PaddleTennis" / "sac" / "20260828_114038"
    run.mkdir(parents=True)
    (run / "config.json").write_text(config_text)
    result = build_index(tmp_path)
    assert result.skipped == ()
    (row,) = result.rows
    assert row["status"] == ABORTED
    assert row["index"]["stub"] is True
    assert row["campaign"]["lever"] == "temperature_skip"
    (message,) = row["index"]["problems"]
    assert message.startswith(problem)
    assert message.endswith("; the row has none of its fields")


@pytest.mark.parametrize(
    "marker", ["run_config.toml", "metrics/monitor/0.monitor.csv"]
)
def test_a_launch_that_died_before_config_json_is_an_aborted_stub(
    tmp_path, capsys, marker
):
    """Verifier r2: train() writes run_config.toml a fraction of a second
    before config.json (Drive shows 0.1-0.2 s between them), so a launch
    that died in between held files but no config.json and was skipped:
    missing from the aborted rows and the stub count. Another project's
    run-named layout (TennisWall: run_config.json and a root monitor/)
    is still skipped, never parsed."""
    run = tmp_path / "PaddleTennis" / "sac" / "20260828_114038"
    (run / marker).parent.mkdir(parents=True, exist_ok=True)
    (run / marker).write_text("[train]\n")
    tennis_wall = tmp_path / "TennisWall" / "sac" / "20260502_025419"
    (tennis_wall / "monitor").mkdir(parents=True)
    (tennis_wall / "monitor" / "0.monitor.csv").write_text("#\n")
    (tennis_wall / "run_config.json").write_text("{}")
    result = build_index(tmp_path)
    (row,) = result.rows
    assert row["index"]["path"] == "PaddleTennis/sac/20260828_114038"
    assert row["status"] == ABORTED
    assert row["index"]["stub"] is True
    assert row["campaign"]["lever"] == "temperature_skip"
    assert row["index"]["problems"] == [run_index._NO_CONFIG]
    (skipped,) = result.skipped
    assert skipped.path == "TennisWall/sac/20260502_025419"
    assert skipped.reason.startswith("named like a run but holds no config.json")
    assert main(["--root", str(tmp_path)]) == 0
    assert "(1 aborted); 1 stub(s)" in capsys.readouterr().out


def _set(path: tuple[str, ...], value):
    def edit(document: dict) -> None:
        for key in path[:-1]:
            document = document[key]
        document[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    ("config_edit", "meta_edit", "problem"),
    [
        (
            _set(("train_config", "warm_start"), "20260809_211147"),
            None,
            "config.json train_config.warm_start is a string, not an object; "
            "read as missing",
        ),
        (
            _set(("env",), ["PaddleTennisEnv"]),
            None,
            "config.json env is a list, not an object; read as missing",
        ),
        (
            _set(("versions",), "sb3 2.9.0"),
            None,
            "config.json versions is a string, not an object; read as missing",
        ),
        (
            _set(("algorithm",), "SAC"),
            None,
            "config.json algorithm is a string, not an object; read as missing",
        ),
        (
            None,
            _set(("selection_keys",), 3),
            "best_model_meta.json selection_keys is a number, not a list; "
            "read as missing",
        ),
        (
            None,
            _set(("selection_values",), [0.5]),
            "best_model_meta.json selection_values is a list, not an object; "
            "read as missing",
        ),
    ],
)
def test_a_field_of_the_wrong_type_is_a_row_problem_never_a_crash(
    tmp_path, config_edit, meta_edit, problem
):
    """Verifier r2: one config.json or best_model_meta.json field of the
    wrong type made the whole index build fail with a bare AttributeError
    or TypeError naming no file. The field is read as missing and the
    row says which file and field."""
    run = tmp_path / "Env" / "sac" / "20260101_000001"
    shutil.copytree(FIXTURE_ROOT / L2W, run)
    for edit, relative in (
        (config_edit, "config.json"),
        (meta_edit, "model/best_model_meta.json"),
    ):
        if edit is not None:
            document = json.loads((run / relative).read_text())
            edit(document)
            (run / relative).write_text(json.dumps(document))
    (row,) = build_index(tmp_path, legacy_levers={}).rows
    assert row["index"]["problems"] == [problem]
    assert row["status"] == records.COMPLETED
    assert row["algo"]["name"] == "SAC"
    if meta_edit is not None:
        assert row["selection"]["timestep"] is not None


def test_a_record_row_of_unexpected_types_is_flagged_never_a_crash(tmp_path):
    """A run record whose status is not a string, or which has none, is a
    row with a problem; neither fails the index or its CLI line."""
    for name, status in (("20260101_000001", ["completed"]), ("20260101_000002", None)):
        run = tmp_path / "Env" / "sac" / name
        run.mkdir(parents=True)
        record = build_run_record({}, run_id=name, created_at="t", status=records.RUNNING)
        if status is None:
            del record["status"]
        else:
            record["status"] = status
        write_run_record(run, record)
    result = build_index(tmp_path, legacy_levers={}, verify=False)
    for row in result.rows:
        assert row["index"]["problems"][-1].startswith(
            f"{RUN_RECORD_NAME} has no known status"
        )
        assert row["index"]["stub"] is True
    assert main(["--root", str(tmp_path), "--skip-verify"]) == 0


def test_an_unbound_selection_in_a_run_record_is_a_row_problem(tmp_path):
    """A record whose best-model meta no longer binds its files (an
    interrupt between the triple's renames) says so in the index."""
    run = tmp_path / "Env" / "sac" / "20260101_000001"
    (run / "model").mkdir(parents=True)
    (run / "model" / "best_model.zip").write_bytes(b"new")
    meta = {
        "timestep": 100,
        "selection_keys": ["steps_alive_ep_mean"],
        "artifacts": {"best_model.zip": {"sha256": "0" * 64}},
    }
    record = build_run_record(
        {},
        run_id=run.name,
        created_at="t",
        status=records.INTERRUPTED,
        best_model_meta=meta,
        best_model_dir=run / "model",
    )
    assert record["selection"]["bindings"] == "unbound"
    write_run_record(run, record)
    write_sync_manifest(run)
    (row,) = build_index(tmp_path, legacy_levers={}).rows
    assert row["index"]["verification"]["complete"] is True
    (problem,) = row["index"]["problems"]
    assert problem.startswith(
        "the best model is unbound (best_model_meta.json does not bind the "
        "files beside it: best_model.zip (sha256 differs)"
    )


def test_a_bom_or_an_integral_float_budget_is_a_courtside_config(tmp_path):
    """Verifier r1: a UTF-8 byte-order mark or ``total_timesteps`` 1e6
    written as a float made a run's config.json "not courtside"."""
    bom = tmp_path / "Env" / "sac" / "20260101_000001"
    shutil.copytree(FIXTURE_ROOT / L2W, bom)
    text = (bom / "config.json").read_text()
    (bom / "config.json").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    floated = tmp_path / "Env" / "sac" / "20260101_000002"
    shutil.copytree(FIXTURE_ROOT / L2W, floated)
    config = json.loads(text)
    config["train_config"]["total_timesteps"] = 1e6
    (floated / "config.json").write_text(json.dumps(config))
    result = build_index(tmp_path, legacy_levers={})
    assert result.skipped == ()
    for row in result.rows:
        assert row["status"] == records.COMPLETED
        assert row["budget"]["trained_timesteps"] == 1_000_000
        assert row["index"]["problems"] == []


def test_an_unreadable_summary_is_not_an_aborted_run(tmp_path):
    """Verifier r1: a summary that exists but is not UTF-8 (written
    under another locale) made the run "aborted", and its problem
    embedded the whole file. Only a missing summary means no terminal
    path was reached."""
    run = tmp_path / "PaddleTennis" / "sac" / "20260815_180815"
    shutil.copytree(FIXTURE_ROOT / L2W, run)
    summary = (run / "stage_summary.txt").read_text()
    (run / "stage_summary.txt").write_bytes(
        summary.replace("Git SHA", "Git S\u00c9A").encode("latin-1")
    )
    (row,) = build_index(tmp_path).rows
    assert row["status"] is None
    assert row["index"]["stub"] is False
    assert row["index"]["files"]["stage_summary"] is True
    (problem,) = row["index"]["problems"]
    assert problem.startswith(
        "stage_summary.txt is unreadable (UnicodeDecodeError: 'utf-8' codec "
        "can't decode byte 0xc9 in position "
    )
    assert "courtside-dynamics" not in problem
    assert len(problem) < 200


def test_videos_and_checkpoints_are_counted_in_either_layout(tmp_path):
    new = tmp_path / "Env" / "sac" / "20260101_000001"
    shutil.copytree(FIXTURE_ROOT / L2W, new)
    (new / "media" / "videos").mkdir(parents=True)
    (new / "media" / "videos" / "a-episode-0.mp4").write_bytes(b"")
    (new / "media" / "videos" / "a-episode-0.meta.json").write_text("{}")
    flat = tmp_path / "Env" / "sac" / "20260101_000002"
    shutil.copytree(FIXTURE_ROOT / WALL_BALL_RUN_1, flat)
    # Pre-0.14 runs keep the replay video at their root (verifier r1:
    # every flat-layout run with one read "videos 0", as if none).
    (flat / "best_model.mp4").write_bytes(b"")
    (new / "media" / "best_model.mp4").parent.mkdir(parents=True, exist_ok=True)
    (new / "media" / "best_model.mp4").write_bytes(b"")
    (flat / "videos").mkdir()
    for step in (250_000, 1_000_000):
        (flat / "videos" / f"step-{step}.mp4").write_bytes(b"")
    (flat / "checkpoints" / "wall_ball_baseline_sac_250000_steps.zip").write_bytes(
        b""
    )
    rows = _rows_by_path(build_index(tmp_path))
    assert rows["Env/sac/20260101_000001"]["index"]["files"]["videos"] == 1
    assert rows["Env/sac/20260101_000001"]["index"]["files"]["best_model_video"]
    flat_files = rows["Env/sac/20260101_000002"]["index"]["files"]
    assert flat_files["best_model_video"] is True
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


@pytest.mark.parametrize("verify", [True, False])
def test_a_run_record_that_still_says_running_is_flagged(tmp_path, capsys, verify):
    """Verifier r1: a run killed after its first record (status still
    "running", no manifest, no summary, no final model) became a row
    with no problem and no stub -- under --skip-verify, the Colab
    default, not even a manifest verdict: a dead run that looked live,
    the R3 failure L1 exists to fix."""
    run_dir = tmp_path / "PaddleTennis" / "sac" / "20260828_114038"
    shutil.copytree(FIXTURE_ROOT / CPU_FALSE_START, run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    write_run_record(
        run_dir,
        build_run_record(
            config, run_id=run_dir.name, created_at="t", status=records.RUNNING
        ),
    )
    (row,) = build_index(tmp_path, verify=verify).rows
    assert row["status"] == records.RUNNING
    assert row["index"]["source"] == "run_record"
    assert row["index"]["stub"] is True
    assert row["index"]["problems"] == [
        "run_record.json says 'running' and there is no sync_manifest.json: the "
        "run is still training, or died without reaching a terminal path"
    ]
    assert main(["--root", str(tmp_path)] + ([] if verify else ["--skip-verify"])) == 0
    captured = capsys.readouterr()
    assert "1 from run records (1 still say running: training, or dead)" in captured.out
    assert "1 stub(s)" in captured.out
    assert "WARNING PaddleTennis/sac/20260828_114038: run_record.json says" in (
        captured.err
    )

    # A mid-run mirror: its manifest lists the running record.
    write_sync_manifest(run_dir)
    (row,) = build_index(tmp_path, verify=verify).rows
    assert row["index"]["problems"][0].endswith(
        "lists it: a mid-run mirror snapshot, not a finished run"
    )
    # A final model makes it no stub, as for a backfilled launch.
    (run_dir / "model" / "final_model.zip").write_bytes(b"")
    (row,) = build_index(tmp_path, verify=verify).rows
    assert row["index"]["stub"] is False


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
Best eval:      438.000 +/- 0.000 (at 2,048 steps, SB3 n_updates 0)  [periodic eval series]
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
    # The on-policy form names the update count after the steps.
    assert parsed["best_eval_step"] == 2_048


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
