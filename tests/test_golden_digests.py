"""The golden env-stream digests: the gate every env refactor must pass.

``tests/golden/env_digests.json`` holds a digest of every CI case in
:mod:`courtside_dynamics.digests` (obs, reward, terminated, truncated,
qpos and qvel at every reset and step, plus the scalar info items),
recorded by ``tools/golden_digest.py --write``. A failure here is an era
break or physics drift: declare it (new task version or a
CHANGELOG/DECISIONS entry) before re-recording, never re-record to make
the test pass.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from courtside_dynamics.digests import (
    CI_CASES,
    DECLARED_DRIFT,
    FULL_CASES,
    MISMATCH_EXPLANATION,
    PADDLE_TENNIS_REWARD_INFO_KEYS,
    StreamCase,
    digest_mismatches,
    expected_digest,
    format_mismatch,
    iter_stream,
    stream_digest,
    version_info,
)
from courtside_dynamics.envs import BallBalanceEnv
from tools import golden_digest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "env_digests.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return golden_digest.load_golden(GOLDEN_PATH)


def test_golden_file_is_the_tools_file():
    assert golden_digest.GOLDEN_PATH == GOLDEN_PATH


@pytest.mark.parametrize("set_name", ["ci", "full"])
def test_golden_records_exactly_the_defined_cases(golden, set_name):
    """No stale entry, no unrecorded case: either drifts silently."""
    cases = {"ci": CI_CASES, "full": FULL_CASES}[set_name]
    assert sorted(golden["sets"][set_name]) == sorted(case.name for case in cases)
    for name, entry in golden["sets"][set_name].items():
        assert "stream_sha256" in entry, f"{name} has no primary digest"
        assert entry["versions"].keys() == version_info().keys()
        for label, slot in entry.get("declared_drift", {}).items():
            assert name in DECLARED_DRIFT[label], (name, label)
            assert slot["reason"] == DECLARED_DRIFT[label][name]


@pytest.mark.parametrize("case", CI_CASES, ids=lambda case: case.name)
def test_stream_digest_matches_golden(golden, case):
    live = version_info()
    expected, recorded, drift = expected_digest(
        golden["sets"]["ci"][case.name], case.name, live
    )
    lines = digest_mismatches(expected, stream_digest(case))
    assert not lines, format_mismatch("ci", case.name, lines, recorded, live, drift)


def test_paddle_cases_cover_all_nine_reward_terms():
    paddle = [case for case in CI_CASES if case.recipe == "PaddleTennis"]
    assert len(paddle) == 8
    for case in paddle:
        assert case.required_info_keys == PADDLE_TENNIS_REWARD_INFO_KEYS
        assert case.info_keys is None  # every scalar item is digested


@pytest.mark.parametrize(
    "name, clawbacks",
    [
        (
            "paddle_tennis/recipe/oracle_nan_contact_escrow",
            {"rew_shaping_clawback": -0.25},
        ),
        (
            "paddle_tennis/hold/oracle_nan_reach_hold_escrow",
            {
                "rew_reach_clawback": -0.17534702856592194,
                "rew_hold_clawback": -0.009889609459489718,
            },
        ),
    ],
)
def test_nan_witness_lands_on_a_pending_escrow(name, clawbacks):
    """The NaN action must hit while the named escrows hold a payment.

    Re-proves the witness calibration from the stream itself: the NaN
    step ends the episode through the nonfinite guard, pays the unsafe
    penalty, and claws back exactly what was pending.
    """
    (case,) = [case for case in CI_CASES if case.name == name]
    nan_step = next(
        record for record in iter_stream(case) if record.step == case.nan_action_step
    )
    assert nan_step.action is not None and np.isnan(nan_step.action).all()
    assert nan_step.terminated and not nan_step.truncated
    info = nan_step.info
    assert info["term_nonfinite"] == 1.0
    for key in ("rew_shaping_clawback", "rew_reach_clawback", "rew_hold_clawback"):
        assert info[key] == clawbacks.get(key, 0.0), key
    assert nan_step.reward == -2.0 + sum(clawbacks.values())


def test_stream_digest_is_sensitive_to_the_nan_injection():
    clean = StreamCase(
        name="probe", recipe="BallBalance", policy="zero", seeds=(0,), steps=40
    )
    poisoned = StreamCase(
        name="probe",
        recipe="BallBalance",
        policy="zero",
        seeds=(0,),
        steps=40,
        nan_action_step=10,
    )
    a, b = stream_digest(clean), stream_digest(poisoned)
    assert a["stream_sha256"] != b["stream_sha256"]
    assert a == stream_digest(clean)  # and deterministic in-process


def test_case_validation_is_loud():
    with pytest.raises(ValueError, match="exactly one"):
        StreamCase(
            name="x",
            recipe="BallBalance",
            env_cls=BallBalanceEnv,
            policy="zero",
            seeds=(0,),
            steps=1,
        )
    with pytest.raises(ValueError, match="unknown recipe"):
        StreamCase(name="x", recipe="Nope", policy="zero", seeds=(0,), steps=1)
    with pytest.raises(ValueError, match="policy must be one of"):
        StreamCase(name="x", recipe="BallBalance", policy="best", seeds=(0,), steps=1)
    with pytest.raises(ValueError, match="nan_action_step"):
        StreamCase(
            name="x",
            recipe="BallBalance",
            policy="zero",
            seeds=(0,),
            steps=5,
            nan_action_step=5,
        )


def test_policies_without_a_meaning_refuse_loudly():
    for policy in ("oracle", "camper"):
        case = StreamCase(
            name="x", env_cls=BallBalanceEnv, policy=policy, seeds=(0,), steps=2
        )
        with pytest.raises(ValueError, match="no (scripted oracle|camper)"):
            stream_digest(case)


def test_missing_required_info_key_raises():
    case = StreamCase(
        name="x",
        env_cls=BallBalanceEnv,
        policy="zero",
        seeds=(0,),
        steps=3,
        required_info_keys=("rew_return",),
    )
    with pytest.raises(KeyError, match="rew_return"):
        stream_digest(case)


def test_check_reports_a_mismatch_with_versions_and_explanation(tmp_path, capsys):
    golden = json.loads(GOLDEN_PATH.read_text())
    entry = golden["sets"]["ci"]["ball_balance/zero"]
    entry["stream_sha256"] = "0" * 64
    tampered = tmp_path / "env_digests.json"
    tampered.write_text(json.dumps(golden))
    code = golden_digest.main(
        ["--check", "--golden", str(tampered), "--case", "ball_balance/zero"]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "GOLDEN DIGEST MISMATCH: ci/ball_balance/zero" in err
    assert f"expected '{'0' * 64}'" in err
    assert f"recorded under: {entry['versions']}" in err
    assert f"running under:  {version_info()}" in err
    assert MISMATCH_EXPLANATION in err


def test_write_reports_changes_and_keeps_unchanged_provenance(tmp_path, capsys):
    path = tmp_path / "env_digests.json"
    path.write_text(GOLDEN_PATH.read_text())
    golden = json.loads(path.read_text())
    golden["sets"]["ci"]["ball_balance/random"]["info_sha256"] = "f" * 64
    path.write_text(json.dumps(golden))
    cases = ["ball_balance/zero", "ball_balance/random"]
    argv = ["--write", "--golden", str(path)]
    for name in cases:
        argv += ["--case", name]
    assert golden_digest.main(argv) == 0
    out = capsys.readouterr().out
    assert "unchanged                        ball_balance/zero" in out
    assert "CHANGED (info_sha256)" in out
    written = json.loads(path.read_text())["sets"]["ci"]
    original = json.loads(GOLDEN_PATH.read_text())["sets"]["ci"]
    # Unchanged entries keep their recording versions verbatim.
    assert written["ball_balance/zero"] == original["ball_balance/zero"]
    assert written["ball_balance/random"]["versions"] == version_info()
    assert written.keys() == original.keys()  # a --case write keeps the rest


def test_check_is_deterministic_across_processes():
    """Another interpreter, another hash seed: the same digests."""
    env = {
        **os.environ,
        "PYTHONHASHSEED": "1234",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "golden_digest.py"),
            "--case",
            "paddle_tennis/recipe/oracle_nan_contact_escrow",
            "--case",
            "ball_bounce/oracle",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2/2 cases match" in result.stdout


def test_digests_never_import_sb3_or_torch():
    """The digest reads the raw env: no training stack in the process."""
    code = (
        "import sys\n"
        "from courtside_dynamics.digests import CI_CASES, stream_digest\n"
        "stream_digest(next(c for c in CI_CASES if c.recipe == 'PaddleTennis'"
        " and c.nan_action_step is not None))\n"
        "leaked = sorted(m for m in ('torch', 'stable_baselines3') if m in sys.modules)\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
