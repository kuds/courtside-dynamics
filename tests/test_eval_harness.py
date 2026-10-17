"""Tests for evaluation snapshots, protocols, the offline harness, rescore
and the probe CLI (design M3 + L2).

The gate: on a fresh CPU-trained tiny run (SAC and PPO, PaddleTennis
with the recipe's paired evaluation), ``harness.evaluate`` on every
saved snapshot equals that evaluation's ``eval_info.csv`` row bit for
bit, and ``select()`` over the harness rows picks the in-loop best
unless a confirmation disagreement occurs (where ``in_loop_select()``
reproduces the in-loop pick instead).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from courtside_dynamics.experiments import harness, protocols, records, snapshots
from courtside_dynamics.experiments.protocols import (
    INLOOP,
    RECIPE_PROTOCOLS,
    EvalProtocol,
    from_run_config,
    from_train_config,
    get_protocol,
    protocol_for_run,
    registered_protocols,
    resolve_seed_block,
)
from courtside_dynamics.experiments.select import (
    CONFIRM_REJECTED,
    IMPROVED,
    in_loop_select,
    select,
)
from courtside_dynamics.recipes import (
    RECIPES,
    build_train_config,
    make_env_fn,
    make_eval_env_fn,
)
from courtside_dynamics.training.train import TrainConfig, train
from tests._helpers import (
    TINY_BUDGET,
    TINY_EPISODE_LEN,
    TINY_EVAL_FREQ,
)
from tests._helpers import tiny_paddle_config as _tiny_paddle_config

REPOSITORY_ROOT = Path(__file__).parents[1]
_AUGUST_RUN = (
    REPOSITORY_ROOT
    / "tests/fixtures/run_index/training_runs/PaddleTennis/sac/20260815_180815"
)

#: The tiny runs' episode length and evaluation cadence (tests/_helpers.py).
_EPISODE_LEN = TINY_EPISODE_LEN
_EVAL_FREQ = TINY_EVAL_FREQ
_BUDGET = TINY_BUDGET


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One fresh CPU-trained tiny run per algorithm family."""
    root = tmp_path_factory.mktemp("tiny_runs")
    runs = {}
    for algo in ("SAC", "PPO"):
        run_dir = root / algo.lower()
        cfg = _tiny_paddle_config(algo, run_dir)
        assert cfg.snapshot_evals is True  # the recipe's default
        train(cfg)
        runs[algo] = run_dir
    return runs


@pytest.fixture(scope="module")
def inloop_rows(tiny_runs: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Each tiny run's snapshots rescored under its own in-loop protocol."""
    rows = {}
    for algo, run_dir in tiny_runs.items():
        [path] = harness.rescore([run_dir], INLOOP, which=("snapshots",))
        assert path == run_dir / "scores" / "inloop.jsonl"
        rows[algo] = harness.read_scores(path)
    return rows


def _eval_info_rows(run_dir: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = defaultdict(dict)
    with open(run_dir / "metrics" / "eval_info.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[int(row["timestep"])][row["metric"]] = float(row["value"])
    return dict(rows)


# -- the gate -------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["SAC", "PPO"])
def test_harness_equals_every_in_loop_evaluation_bit_for_bit(tiny_runs, algo):
    run_dir = tiny_runs[algo]
    in_loop = _eval_info_rows(run_dir)
    protocol = protocol_for_run(run_dir)
    assert protocol.paired and protocol.confirm and protocol.n_episodes == 2
    # The run's own protocol is the recipe's, at this run's episode
    # length and episode count.
    recipe_protocol = get_protocol("paddle-npoint@v3").derive(
        "tiny@v1", env_overrides={"episode_len": _EPISODE_LEN}, n_episodes=2,
        min_deltas=(0.0, 0.0, 0.0),
    )
    assert protocol.fingerprint == recipe_protocol.fingerprint
    index = snapshots.read_snapshot_index(run_dir / "snapshots")
    assert [entry.timestep for entry in index.entries] == sorted(in_loop)
    assert sorted(in_loop) == list(range(_EVAL_FREQ, _BUDGET + 1, _EVAL_FREQ))
    for entry in index.entries:
        metrics = harness.evaluate(
            run_dir / "snapshots" / entry.model,
            run_dir / "snapshots" / entry.normalizer,
            protocol,
        )
        expected = dict(in_loop[entry.timestep])
        n_updates = expected.pop("n_updates", None)
        # Every metric column, compared as floats parsed from the CSV
        # (which writes repr, so a float round-trips exactly).
        assert metrics == expected, entry.timestep
        if algo == "PPO":
            assert n_updates == float(entry.n_updates)
        else:
            assert n_updates is None
            assert isinstance(entry.n_updates, int)


@pytest.mark.parametrize("algo", ["SAC", "PPO"])
def test_select_over_harness_rows_picks_the_in_loop_best(tiny_runs, inloop_rows, algo):
    run_dir = tiny_runs[algo]
    rows = inloop_rows[algo]
    in_loop = _eval_info_rows(run_dir)
    for row in rows:
        assert row["artifact"] == "snapshot"
        assert row["protocol"] == f"{INLOOP}:{run_dir.name}"
        assert row["seed_source"] == "run_paired_block"
        assert row["seed_start"] == 1_000_000
        assert row["metrics"] == {
            k: v for k, v in in_loop[row["step"]].items() if k != "n_updates"
        }
        assert row["confirm_metrics"] is not None
    meta = json.loads((run_dir / "model" / "best_model_meta.json").read_text())
    rule = protocol_for_run(run_dir).selection_rule()
    within = select(rows, rule)
    replay = in_loop_select(rows, rule)
    # The replay of the in-loop rule always reproduces the in-loop pick.
    assert replay.step == meta["timestep"]
    ours = [decision.decision for decision in within.decisions]
    theirs = [decision.decision for decision in replay.decisions]
    if ours == theirs:
        assert within.step == meta["timestep"]
    else:
        # A confirmation disagreement: the rules part only where the
        # in-loop one compared a confirmation with the first best's
        # standard-block score.
        assert within.step != meta["timestep"]
        for mine, its in zip(ours, theirs, strict=True):
            if mine != its:
                assert {mine, its} == {IMPROVED, CONFIRM_REJECTED}
    # A best accepted on a confirmation recorded the confirmation-block
    # values the harness reproduces.
    if "confirmation_values" in meta:
        chosen = next(row for row in rows if row["step"] == meta["timestep"])
        for key, value in meta["confirmation_values"].items():
            assert chosen["confirm_metrics"][key] == value


def test_snapshots_are_recorded_sealed_and_survive_rescoring(tiny_runs, inloop_rows):
    run_dir = tiny_runs["SAC"]
    record = json.loads((run_dir / "run_record.json").read_text())
    assert record["snapshots"] == {
        "index": "snapshots/index.json",
        "count": 4,
        "timesteps": [256, 512, 768, 1024],
        "failures": [],
    }
    assert {"name": "snapshots_dir", "path": "snapshots"} in record["artifacts"]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train_config"]["snapshot_evals"] is True
    manifest = json.loads((run_dir / "sync_manifest.json").read_text())
    listed = {entry["path"] for entry in manifest["files"]}
    assert "snapshots/index.json" in listed
    assert "snapshots/snapshot_256_steps.zip" in listed
    assert "snapshots/snapshot_vecnormalize_1024_steps.pkl" in listed
    # The rescore wrote scores/ after the run sealed itself: tolerated,
    # like the notebooks' reports/.
    result = records.verify_run_dir(run_dir)
    assert result.complete, result.problems
    assert "scores/inloop.jsonl" in result.unlisted
    assert not records.verify_run_dir(run_dir, strict=True).complete
    # A snapshot loads through the run's recorded algorithm.
    from courtside_dynamics.training.algos import locate_run_config

    assert locate_run_config(run_dir / "snapshots" / "snapshot_256_steps.zip") == str(
        run_dir / "config.json"
    )


# -- rescoring ------------------------------------------------------------------


def _legacy_run(tmp_path: Path, source: Path) -> Path:
    """An August run directory (20260815_180815's config.json, summary and
    meta, byte for byte) holding a tiny run's real model files in the
    legacy names: checkpoints/paddle_tennis_sac_<N>_steps.zip with
    paddle_tennis_sac_vecnormalize_<N>_steps.pkl."""
    run_dir = tmp_path / "20260815_180815"
    shutil.copytree(_AUGUST_RUN, run_dir)
    for name in ("best_model.zip", "best_vec_normalize.pkl", "final_model.zip", "vec_normalize.pkl"):
        shutil.copyfile(source / "model" / name, run_dir / "model" / name)
    (run_dir / "checkpoints" / "paddle_tennis_sac_1000000_steps.zip").unlink()
    shutil.copyfile(
        source / "checkpoints" / "paddle_tennis_sac_512_steps.zip",
        run_dir / "checkpoints" / "paddle_tennis_sac_512_steps.zip",
    )
    shutil.copyfile(
        source / "checkpoints" / "paddle_tennis_sac_vecnormalize_512_steps.pkl",
        run_dir / "checkpoints" / "paddle_tennis_sac_vecnormalize_512_steps.pkl",
    )
    # Bind the fixture's meta to the files now beside it.
    meta_path = run_dir / "model" / "best_model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["artifacts"] = {
        name: {"sha256": records.sha256_file(run_dir / "model" / name)}
        for name in ("best_model.zip", "best_vec_normalize.pkl")
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return run_dir


def _legacy_protocol() -> EvalProtocol:
    return get_protocol("paddle-npoint@v2").derive(
        "paddle-npoint-short@v1",
        env_overrides={"episode_len": 150},
        n_episodes=1,
    )


def test_rescore_reads_the_august_runs_legacy_layout(tmp_path, tiny_runs):
    run_dir = _legacy_run(tmp_path, tiny_runs["SAC"])
    config = json.loads((run_dir / "config.json").read_text())
    # What the August records hold: an algorithm, env kwargs, no paired
    # block (eval_seed null, no evaluation_seeding).
    assert config["train_config"]["algo"] == "SAC"
    assert config["train_config"].get("eval_seed") is None
    assert "evaluation_seeding" not in config
    protocol = _legacy_protocol()
    [path] = harness.rescore([run_dir], protocol, blocks=("standard",))
    assert path == run_dir / "scores" / "paddle-npoint-short@v1.jsonl"
    rows = harness.read_scores(path)
    assert [(row["artifact"], row["step"]) for row in rows] == [
        ("best", 25_000),
        ("checkpoint", 512),
    ]
    for row in rows:
        assert row["algo"] == "SAC"
        assert row["protocol_fingerprint"] == protocol.fingerprint
        # Today's derivation of the paired block from the run's seed 0.
        assert (row["seed_start"], row["seed_source"]) == (1_000_000, "run_seed_offset")
        assert row["confirm_metrics"] is None
        assert row["model"]["sha256"] == records.sha256_file(run_dir / row["model"]["path"])
    checkpoint = rows[1]
    assert checkpoint["model"]["path"] == "checkpoints/paddle_tennis_sac_512_steps.zip"
    assert (
        checkpoint["normalizer"]["path"]
        == "checkpoints/paddle_tennis_sac_vecnormalize_512_steps.pkl"
    )
    assert checkpoint["metrics"] == harness.evaluate(
        run_dir / checkpoint["model"]["path"],
        run_dir / checkpoint["normalizer"]["path"],
        protocol,
        seed_start=1_000_000,
    )
    # One combined file under out_dir, every run's rows in it.
    [combined] = harness.rescore(
        [run_dir], protocol, which=("checkpoints",), out_dir=tmp_path / "out"
    )
    assert combined == tmp_path / "out" / "scores" / "paddle-npoint-short@v1.jsonl"
    assert [row["metrics"] for row in harness.read_scores(combined)] == [
        checkpoint["metrics"]
    ]


def test_rescore_scores_only_the_steps_asked_for(tmp_path, tiny_runs, capsys):
    """``steps`` (the notebook's QUICK, the CLI's --steps) keeps the
    snapshots and checkpoints at those steps, besides the best model,
    and refuses a step the run does not have."""
    from courtside_dynamics.experiments.rescore import main

    run_dir = tmp_path / "run"
    shutil.copytree(tiny_runs["SAC"], run_dir)
    shutil.rmtree(run_dir / "scores", ignore_errors=True)
    assert harness.series_steps(run_dir, "snapshots") == [256, 512, 768, 1024]
    assert harness.series_steps(run_dir, "checkpoints") == [512, 1024]
    assert harness.series_steps(tmp_path, "checkpoints") == []
    with pytest.raises(ValueError, match="'snapshots' or 'checkpoints'"):
        harness.series_steps(run_dir, "best")
    best_step = json.loads((run_dir / "model" / "best_model_meta.json").read_text())[
        "timestep"
    ]
    artifacts, problems = harness.run_artifacts(
        run_dir, ("best", "snapshots"), steps=[512, 1024]
    )
    assert problems == []
    assert [(a.kind, a.step) for a in artifacts] == [
        ("best", best_step),
        ("snapshot", 512),
        ("snapshot", 1024),
    ]
    protocol = _legacy_protocol()
    [path] = harness.rescore(
        [run_dir], protocol, which=("best", "checkpoints"), steps=(1024,),
        blocks=("standard",), out_dir=tmp_path / "out",
    )
    assert [(r["artifact"], r["step"]) for r in harness.read_scores(path)] == [
        ("best", best_step),
        ("checkpoint", 1024),
    ]
    with pytest.raises(ValueError, match=r"no checkpoints artifact at step\(s\) \[300\]"):
        harness.rescore(
            [run_dir], protocol, which=("checkpoints",), steps=(1024, 300),
            out_dir=tmp_path / "refused",
        )
    assert not (tmp_path / "refused").exists()
    with pytest.raises(ValueError, match="steps select snapshots or checkpoints"):
        harness.run_artifacts(run_dir, ("best",), steps=(512,))
    assert main(
        ["--protocol", INLOOP, "--which", "snapshots", "--steps", "256,1024", str(run_dir)]
    ) == 0
    capsys.readouterr()
    rows = harness.read_scores(run_dir / "scores" / "inloop.jsonl")
    assert [r["step"] for r in rows] == [256, 1024]
    with pytest.raises(SystemExit):
        main(["--protocol", INLOOP, "--steps", "x", str(run_dir)])
    assert "--steps takes" in capsys.readouterr().err


def test_rescore_refuses_before_scoring_anything(tmp_path, tiny_runs):
    run_dir = _legacy_run(tmp_path, tiny_runs["SAC"])
    protocol = _legacy_protocol()
    # A best model the meta does not bind (a save split between renames).
    (run_dir / "model" / "best_vec_normalize.pkl").write_bytes(b"other")
    # A checkpoint without its normalizer, in a run that normalized.
    (run_dir / "checkpoints" / "paddle_tennis_sac_vecnormalize_512_steps.pkl").unlink()
    with pytest.raises(ValueError) as refused:
        harness.rescore([run_dir], protocol)
    message = str(refused.value)
    assert "best_vec_normalize.pkl (sha256 differs)" in message
    assert "paddle_tennis_sac_512_steps.zip has no normalizer" in message
    assert not (run_dir / "scores").exists()
    # A tampered snapshot is refused the same way.
    snapshot_run = tmp_path / "snap"
    shutil.copytree(tiny_runs["PPO"], snapshot_run)
    (snapshot_run / "snapshots" / "snapshot_512_steps.zip").write_bytes(b"torn")
    with pytest.raises(ValueError, match="snapshot_512_steps.zip \\(sha256 differs\\)"):
        harness.rescore([snapshot_run], INLOOP, which=("snapshots",))
    with pytest.raises(ValueError, match="must hold 'standard'"):
        harness.rescore([tiny_runs["PPO"]], protocol, blocks=("confirm",))
    with pytest.raises(ValueError, match="only blocks of protocol"):
        harness.rescore(
            [tiny_runs["PPO"]],
            protocol.derive("unconfirmed@v1", confirm=False),
            blocks=("standard", "confirm"),
        )
    with pytest.raises(ValueError, match="no best/snapshots/checkpoints"):
        empty = tmp_path / "empty"
        empty.mkdir()
        shutil.copyfile(run_dir / "config.json", empty / "config.json")
        harness.rescore([empty], protocol)


def test_evaluate_refuses_what_it_cannot_score_faithfully(tiny_runs):
    run_dir = tiny_runs["SAC"]
    model = run_dir / "snapshots" / "snapshot_256_steps.zip"
    protocol = protocol_for_run(run_dir)
    with pytest.raises(ValueError, match="normalize_obs=True"):
        harness.evaluate(model, None, protocol)
    with pytest.raises(ValueError, match="not this protocol's"):
        harness.evaluate(
            model, run_dir / "snapshots" / "snapshot_vecnormalize_256_steps.pkl",
            protocol, device="cuda",
        )
    with pytest.raises(ValueError, match="block"):
        harness.evaluate(
            model, run_dir / "snapshots" / "snapshot_vecnormalize_256_steps.pkl",
            protocol, block="held_out",
        )


def test_rescore_cli(tiny_runs, capsys, tmp_path):
    from courtside_dynamics.experiments.rescore import main

    run_dir = tmp_path / "run"
    shutil.copytree(tiny_runs["SAC"], run_dir)
    shutil.rmtree(run_dir / "scores", ignore_errors=True)
    assert main(["--protocol", INLOOP, "--which", "snapshots", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "select() over the snapshots: step" in out
    assert "in-loop best_model: step" in out
    assert (run_dir / "scores" / "inloop.jsonl").is_file()
    assert main(["--protocol", "no-such@v1", str(run_dir)]) == 2
    assert "unknown evaluation protocol" in capsys.readouterr().err


# -- protocols ------------------------------------------------------------------


def test_paddle_npoint_v3_is_todays_in_loop_selection_evaluation():
    """The literal protocol equals what train() wires for the recipe."""
    from courtside_dynamics.callbacks.info_dict_eval import CONFIRMATION_SEED_OFFSET
    from courtside_dynamics.training.train import EVAL_SEED_OFFSET

    v3 = get_protocol("paddle-npoint@v3")
    assert RECIPE_PROTOCOLS["PaddleTennis"] == "paddle-npoint@v3"
    for algo in ("SAC", "PPO"):
        cfg = build_train_config("PaddleTennis", algo=algo, log_dir="x", seed=0)
        derived = from_train_config(cfg, id="derived")
        assert derived.fingerprint == v3.fingerprint, algo
    assert v3.n_episodes == 30 and v3.paired and v3.confirm
    assert v3.seed_start is None  # the run's own block
    assert v3.confirm_seed_offset == CONFIRMATION_SEED_OFFSET
    assert v3.run_seed_offset == EVAL_SEED_OFFSET
    assert [dict(option) for option in v3.reset_options or ()] == [
        {"serve_side": "a"},
        {"serve_side": "b"},
    ]
    assert v3.selection_rule().min_deltas == (1 / 60, 1 / 60, 0.25)
    assert v3.env_kwargs["drill_fraction"] == 0.0
    assert v3.device == "cpu"
    # Training keeps the aids; only the evaluation env strips them.
    recipe = RECIPES["PaddleTennis"]
    assert recipe.env_kwargs["contact_shaping"] == 0.25
    assert recipe.env_kwargs["reach_shaping"] == 0.25
    for name in ("contact_shaping", "reach_shaping", "hold_shaping"):
        assert recipe.eval_env_overrides[name] == 0.0


#: The two eras' fingerprints, pinned: @v2 is unchanged by L3 (its
#: value before @v3 existed), and either changing is a new version.
_V2_FINGERPRINT = "30526bfdad6dd23dbc5c8dc5bac884064baefb4a2cc95a450a708c1a6b018a66"
_V3_FINGERPRINT = "863bb3843aa9ec0fbd6e5ca2f8f2d16a6faaba70e4a522c5b4f165cbbb766efa"

#: The escrow channels @v3 zeroes (rew_* info keys).
_SHAPING_CHANNELS = (
    "rew_shaping",
    "rew_shaping_clawback",
    "rew_reach",
    "rew_reach_clawback",
    "rew_hold",
    "rew_hold_clawback",
)


def test_paddle_npoint_v3_is_v2_without_the_training_aids():
    v2 = get_protocol("paddle-npoint@v2")
    v3 = get_protocol("paddle-npoint@v3")
    assert (v2.fingerprint, v3.fingerprint) == (_V2_FINGERPRINT, _V3_FINGERPRINT)
    assert {"paddle-npoint@v2", "paddle-npoint@v3"} <= set(registered_protocols())
    # Every field but the env's kwargs is @v2's ...
    content_v2, content_v3 = v2.canonical(), v3.canonical()
    assert content_v2.pop("env_kwargs") != content_v3.pop("env_kwargs")
    assert content_v2 == content_v3
    # ... and the kwargs differ exactly in the shaping scales @v2 had on.
    changed = {
        key: (v2.env_kwargs[key], v3.env_kwargs[key])
        for key in v2.env_kwargs
        if v2.env_kwargs[key] != v3.env_kwargs[key]
    }
    assert changed == {"contact_shaping": (0.25, 0.0), "reach_shaping": (0.25, 0.0)}
    assert set(v2.env_kwargs) == set(v3.env_kwargs)
    assert v3.env_kwargs["hold_shaping"] == 0.0 == v2.env_kwargs["hold_shaping"]
    assert dict(protocols.PADDLE_TRAINING_AIDS_OFF) == {
        "contact_shaping": 0.0,
        "reach_shaping": 0.0,
        "hold_shaping": 0.0,
    }
    restored = v3.derive(
        "restored@v1", env_overrides={"contact_shaping": 0.25, "reach_shaping": 0.25}
    )
    assert restored.fingerprint == v2.fingerprint


def test_eval_protocol_record_names_the_registered_protocol_or_none():
    from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv

    cfg = build_train_config("PaddleTennis", log_dir="x", seed=0)
    assert protocols.eval_protocol_record(cfg) == {
        "id": "paddle-npoint@v3",
        "fingerprint": _V3_FINGERPRINT,
    }
    # An env whose constructor kwargs cannot be read: no record, said so.
    with pytest.warns(UserWarning, match="no evaluation protocol recorded"):
        assert (
            protocols.eval_protocol_record(cfg, env_cls=PaddleTennisEnv, env_kwargs=None)
            is None
        )
    with pytest.raises(ValueError, match="pass its env_cls"):
        from_train_config(cfg, id="x", env_kwargs={"episode_len": 1500})
    cfg.info_dict_eval = False
    assert protocols.eval_protocol_record(cfg) is None


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    return bool(left == right)


@pytest.mark.parametrize(("policy", "seed", "side"), [
    ("oracle", 5500, "a"), ("oracle", 5501, "b"), ("random", 5502, "a"),
])
def test_paddle_tennis_observations_carry_no_escrow_state(policy, seed, side):
    """@v3's premise. The observation is physical state, the rules'
    snapshot, the clock and the contact latches: no escrow balance, no
    hold window. So with the escrows on (@v2's eval env) and off
    (@v3's), the same seeds and actions give bit-identical observations,
    endings and non-reward info at every step while an escrow is
    pending, and the reward differs by exactly the escrow channels."""
    from courtside_dynamics.digests import scripted_policy
    from courtside_dynamics.training.artifacts import env_observation_names

    v2 = get_protocol("paddle-npoint@v2")
    v3 = get_protocol("paddle-npoint@v3")
    shaped, bare = v2.make_env(), v3.make_env()
    names = env_observation_names(shaped)
    assert names == env_observation_names(bare)
    assert names is not None and len(names) == shaped.observation_space.shape[0]
    for fragment in ("shaping", "escrow", "reach", "hold", "clawback", "rew"):
        assert not any(fragment in name for name in names), fragment
    act_shaped = scripted_policy(policy, shaped, seed)
    act_bare = scripted_policy(policy, bare, seed)
    options = {"serve_side": side}
    obs_shaped, _ = shaped.reset(seed=seed, options=options)
    obs_bare, _ = bare.reset(seed=seed, options=options)
    paid = pending_steps = 0.0
    for _ in range(v3.env_kwargs["episode_len"]):
        assert obs_shaped.tobytes() == obs_bare.tobytes()
        action = np.asarray(act_shaped(obs_shaped))
        assert np.array_equal(action, np.asarray(act_bare(obs_bare)))
        obs_shaped, reward_shaped, term_shaped, trunc_shaped, info_shaped = shaped.step(action)
        obs_bare, reward_bare, term_bare, trunc_bare, info_bare = bare.step(action)
        assert (term_shaped, trunc_shaped) == (term_bare, trunc_bare)
        escrow = sum(info_shaped[key] for key in _SHAPING_CHANNELS)
        assert all(info_bare[key] == 0.0 for key in _SHAPING_CHANNELS)
        assert reward_shaped - reward_bare == pytest.approx(escrow, abs=1e-12)
        assert info_bare.keys() == info_shaped.keys()
        for key in info_shaped:
            if key not in _SHAPING_CHANNELS:
                assert _same_value(info_shaped[key], info_bare[key]), key
        paid += abs(escrow)
        pending_steps += shaped._pending_shaping > 0 or shaped._pending_reach > 0
        if term_shaped or trunc_shaped:
            break
    assert obs_shaped.tobytes() == obs_bare.tobytes()
    if policy == "oracle":
        # Not vacuous: the oracle collects shaping and holds an escrow
        # balance the observation never shows.
        assert paid > 0.5 and pending_steps > 50


#: The metrics the escrow channels feed: everything else must agree.
def _reward_metrics(metrics: dict[str, float]) -> set[str]:
    return {"episode_reward_mean"} | {
        key for key in metrics if key.startswith(("rew_shaping", "rew_reach", "rew_hold"))
    }


def _assert_v2_v3_agree(v2_metrics: dict[str, float], v3_metrics: dict[str, float]) -> float:
    """Every non-reward metric equal bit for bit; the reward apart by the
    escrows. Returns the escrow the batch collected under @v2."""
    assert v2_metrics.keys() == v3_metrics.keys()
    reward_keys = _reward_metrics(v2_metrics)
    for key in v2_metrics.keys() - reward_keys:
        assert v2_metrics[key] == v3_metrics[key], key
    for key in RECIPE_SELECTION_KEYS[:2]:
        assert v2_metrics[key] == v3_metrics[key], key
    escrow = sum(
        v2_metrics.get(f"{channel}_ep_sum_mean", 0.0) for channel in _SHAPING_CHANNELS
    )
    for channel in _SHAPING_CHANNELS:
        assert v3_metrics.get(f"{channel}_ep_sum_mean", 0.0) == 0.0
    # The VecEnv stores each step's reward as float32, so the @v2 sum
    # carries that rounding; the task reward itself is identical.
    assert v2_metrics["episode_reward_mean"] - v3_metrics[
        "episode_reward_mean"
    ] == pytest.approx(escrow, abs=1e-5)
    for channel in ("rew_return", "rew_fault", "rew_unsafe"):
        key = f"{channel}_ep_sum_mean"
        assert v2_metrics[key] == v3_metrics[key], key
    return escrow


#: The PaddleTennis selection keys, in order (headline, success, reward).
RECIPE_SELECTION_KEYS = (
    "episode_rally_returns_a_ep_mean",
    "success_rate",
    "episode_reward_mean",
)


def test_v2_and_v3_differ_only_in_the_shaping_reward(tiny_runs, tmp_path):
    """Design L3's gate: for the same policy, the k>=2 headline and the
    other non-reward selection key (indeed every non-reward metric) are
    identical under @v2 and @v3, on both blocks, while
    episode_reward_mean differs exactly by the shaping collected. The
    scripted oracle (which converts k>=2 and collects shaping) on full
    1,500-step episodes, and the tiny trained run's snapshots and best
    model on the 300-step episodes it trained on. Two episodes a block."""
    from courtside_dynamics.experiments.probe import score_scripted

    v2 = get_protocol("paddle-npoint@v2").derive("v2-gate@v1", n_episodes=2)
    v3 = get_protocol("paddle-npoint@v3").derive("v3-gate@v1", n_episodes=2)
    assert v2.selection_rule() == v3.selection_rule()
    assert v3.selection_rule().keys == RECIPE_SELECTION_KEYS
    for block in harness.BLOCKS:
        oracle_v2 = score_scripted("oracle", v2, seed_start=5500, block=block)
        oracle_v3 = score_scripted("oracle", v3, seed_start=5500, block=block)
        escrow = _assert_v2_v3_agree(oracle_v2, oracle_v3)
        # The oracle converts k>=2 and its @v2 tie-break carried shaping.
        assert oracle_v2["episode_rally_returns_a_ep_mean"] > 0.0
        assert escrow > 1.0
    run_dir = tiny_runs["SAC"]
    rows: dict[str, list[dict[str, Any]]] = {}
    short = {"episode_len": _EPISODE_LEN}
    for protocol in (
        v2.derive("v2-gate-short@v1", env_overrides=short),
        v3.derive("v3-gate-short@v1", env_overrides=short),
    ):
        [path] = harness.rescore(
            [run_dir], protocol, which=("best", "snapshots"), out_dir=tmp_path / protocol.id
        )
        rows[protocol.id] = harness.read_scores(path)
    rows_v2, rows_v3 = rows["v2-gate-short@v1"], rows["v3-gate-short@v1"]
    assert [(r["artifact"], r["step"]) for r in rows_v2] == [
        (r["artifact"], r["step"]) for r in rows_v3
    ]
    assert len(rows_v2) == 5  # the best model and four snapshots
    for row_v2, row_v3 in zip(rows_v2, rows_v3, strict=True):
        assert row_v2["seed_start"] == row_v3["seed_start"] == 1_000_000
        _assert_v2_v3_agree(row_v2["metrics"], row_v3["metrics"])
        _assert_v2_v3_agree(row_v2["confirm_metrics"], row_v3["confirm_metrics"])
    # The rule's first two keys never move, so select() can differ only
    # through the reward tie-break: the era boundary, and nothing else.
    snapshots_v2 = [r for r in rows_v2 if r["artifact"] == "snapshot"]
    snapshots_v3 = [r for r in rows_v3 if r["artifact"] == "snapshot"]
    for chosen in (select(snapshots_v2, v2), select(snapshots_v3, v3)):
        assert chosen.step in {r["step"] for r in snapshots_v2}


def test_every_registered_protocol_is_its_recipes_in_loop_evaluation():
    assert set(RECIPE_PROTOCOLS) == set(RECIPES)
    # Every recipe's in-loop protocol, plus the earlier PaddleTennis era.
    assert set(registered_protocols()) == {
        *RECIPE_PROTOCOLS.values(),
        "paddle-npoint@v2",
    }
    for recipe, protocol_id in RECIPE_PROTOCOLS.items():
        assert protocols.PROTOCOL_ID.match(protocol_id), protocol_id
        protocol = get_protocol(protocol_id)
        cfg = build_train_config(recipe, log_dir="x", seed=0)
        assert from_train_config(cfg, id="x").fingerprint == protocol.fingerprint, recipe
        assert protocol.recipe == recipe
        assert get_protocol(protocol_id) is protocol  # cached


def test_a_recipe_change_refuses_its_pinned_protocol(monkeypatch):
    monkeypatch.setattr(protocols, "_CACHE", {})
    monkeypatch.setitem(
        protocols._PINNED_FINGERPRINTS, "ball-balance-inloop@v1", "0" * 64
    )
    with pytest.raises(ValueError, match="never re-pin"):
        get_protocol("ball-balance-inloop@v1")
    with pytest.raises(KeyError, match="paddle-npoint@v2"):
        get_protocol("paddle-npoint@v9")


def test_the_fingerprint_is_the_content():
    v2 = get_protocol("paddle-npoint@v2")
    renamed = v2.derive("another@v1", description="same content")
    assert renamed.fingerprint == v2.fingerprint
    for changes in (
        {"n_episodes": 31},
        {"seed_start": 5},
        {"confirm": False},
        {"min_deltas": (0.0, 1 / 60, 0.25)},
        {"device": "cuda"},
        {"reset_options": ({"serve_side": "b"}, {"serve_side": "a"})},
        {"info_keys": v2.info_keys[:-1]},
    ):
        assert v2.derive("x@v1", **changes).fingerprint != v2.fingerprint, changes
    assert (
        v2.derive("x@v1", env_overrides={"reach_shaping": 0.0}).fingerprint
        != v2.fingerprint
    )
    with pytest.raises(ValueError, match="takes no"):
        v2.derive("x@v1", env_overrides={"no_such_kwarg": 1})
    with pytest.raises(ValueError, match="paired"):
        v2.derive("x@v1", paired=False)
    with pytest.raises(TypeError):
        v2.env_kwargs["episode_len"] = 1  # type: ignore[index]


@pytest.mark.parametrize("recipe", ["PaddleTennis", "WallBallTrueBaseline", "BallBalance"])
def test_a_runs_config_json_rebuilds_its_in_loop_protocol(tmp_path, recipe):
    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config(recipe, log_dir=str(tmp_path), seed=3)
    write_run_config(cfg, str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    rebuilt = from_run_config(config, id="from-json")
    assert rebuilt.fingerprint == from_train_config(cfg, id="x").fingerprint
    assert rebuilt.fingerprint == get_protocol(RECIPE_PROTOCOLS[recipe]).fingerprint


def test_a_record_holding_object_reprs_cannot_rebuild_its_env(tmp_path):
    from courtside_dynamics.training.artifacts import write_run_config

    cfg = build_train_config(
        "HumanoidTennisStage1AnchoredReturn", log_dir=str(tmp_path), seed=0
    )
    write_run_config(cfg, str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    with pytest.raises(ValueError, match="object's repr"):
        from_run_config(config, id="x")


def test_seed_blocks_resolve_from_the_run_or_refuse():
    paired = get_protocol("paddle-npoint@v2")
    unpaired = get_protocol("ball-balance-inloop@v1")
    assert not unpaired.paired
    explicit = resolve_seed_block(paired, seed_start=5500)
    assert (explicit.start, explicit.source) == (5500, "explicit")
    assert explicit.for_block(paired, "confirm") == 105_500
    assert resolve_seed_block(paired.derive("x@v1", seed_start=7)).source == "protocol"
    recorded = {
        "train_config": {"seed": 4, "eval_seed": None, "n_envs": 8},
        "evaluation_seeding": {"paired": True, "selection_batch_seed_start": 77},
    }
    assert resolve_seed_block(paired, run_config=recorded).start == 77
    configured = {"train_config": {"seed": 4, "eval_seed": 9, "n_envs": 8}}
    assert resolve_seed_block(paired, run_config=configured).source == "run_eval_seed"
    august = {"train_config": {"seed": 4, "eval_seed": None, "n_envs": 8}}
    block = resolve_seed_block(paired, run_config=august)
    assert (block.start, block.source) == (1_000_004, "run_seed_offset")
    block = resolve_seed_block(unpaired, run_config=august)
    assert (block.start, block.source) == (13, "run_info_eval_env")
    with pytest.raises(ValueError, match="pass seed_start"):
        resolve_seed_block(paired)
    with pytest.raises(ValueError, match="unseeded"):
        resolve_seed_block(paired, run_config={"train_config": {"seed": None}})
    with pytest.raises(ValueError, match="block must be"):
        explicit.for_block(paired, "held_out")


def test_an_unpaired_protocol_reproduces_the_first_in_loop_evaluation(tmp_path):
    """An unpaired run's stream continues across evaluations, so the
    harness (a fresh env per score) reproduces only its first row; the
    protocol's block is the info evaluator's env seed."""
    cfg = build_train_config(
        "BallBalance",
        log_dir=str(tmp_path / "run"),
        seed=0,
        n_envs=2,
        total_timesteps=400,
        eval_freq=200,
        n_eval_episodes=8,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        snapshot_evals=True,
        env_fn=make_env_fn("BallBalance", env_overrides={"episode_len": 50}),
        eval_env_fn=make_eval_env_fn("BallBalance", env_overrides={"episode_len": 50}),
        model_kwargs={"buffer_size": 1_000, "learning_starts": 50, "batch_size": 32},
    )
    train(cfg)
    run_dir = tmp_path / "run"
    protocol = protocol_for_run(run_dir)
    assert not protocol.paired and not protocol.selection_keys
    assert protocol.n_episodes == 2  # the reporting stream's quarter
    first = snapshots.read_snapshot_index(run_dir / "snapshots").entries[0]
    metrics = harness.evaluate(
        run_dir / "snapshots" / first.model,
        run_dir / "snapshots" / first.normalizer,
        protocol,
    )
    assert metrics == _eval_info_rows(run_dir)[first.timestep]
    with pytest.raises(ValueError, match="no selection keys"):
        protocol.selection_rule()


# -- snapshots ------------------------------------------------------------------


def _writer_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_snapshot_writer_binds_each_pair_atomically(tmp_path):
    directory = tmp_path / "snapshots"
    writer = snapshots.SnapshotWriter(directory, run_id="run")
    writer.open()

    def save(content: bytes):
        return lambda path: Path(path).write_bytes(content)

    writer.write(100, save_model=save(b"m100"), save_normalizer=save(b"n100"), n_updates=3)
    writer.write(200, save_model=save(b"m200"), n_updates=None)
    assert _writer_files(directory) == [
        "index.json",
        "snapshot_100_steps.zip",
        "snapshot_200_steps.zip",
        "snapshot_vecnormalize_100_steps.pkl",
    ]
    index = snapshots.read_snapshot_index(directory)
    assert [entry.timestep for entry in index.entries] == [100, 200]
    assert index.entries[0].normalizer == "snapshot_vecnormalize_100_steps.pkl"
    assert index.entries[1].normalizer is None
    assert index.entries[0].n_updates == 3
    assert index.entries[0].model_sha256 == records.sha256_file(
        directory / "snapshot_100_steps.zip"
    )
    assert all(snapshots.snapshot_unbound(directory, e) is None for e in index.entries)
    # A failing save leaves the previous index whole and no temporary.
    before = (directory / "index.json").read_bytes()

    def broken(path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        writer.write(300, save_model=broken)
    assert (directory / "index.json").read_bytes() == before
    assert _writer_files(directory) == [
        "index.json",
        "snapshot_100_steps.zip",
        "snapshot_200_steps.zip",
        "snapshot_vecnormalize_100_steps.pkl",
    ]
    writer.note_failure(300, OSError(28, "No space left on device"))
    index = snapshots.read_snapshot_index(directory)
    assert index.failures == ({"timestep": 300, "error": "OSError: No space left on device"},)
    assert snapshots.snapshot_summary(tmp_path) == {
        "index": "snapshots/index.json",
        "count": 2,
        "timesteps": [100, 200],
        "failures": [300],
    }
    with pytest.raises(ValueError, match="already written"):
        writer.write(100, save_model=save(b"again"))
    # A tampered file no longer binds.
    (directory / "snapshot_200_steps.zip").write_bytes(b"other")
    assert "sha256 differs" in str(snapshots.snapshot_unbound(directory, index.entries[1]))
    # A second writer refuses the directory: its entries are not this run's.
    with pytest.raises(FileExistsError, match="previous attempt"):
        snapshots.SnapshotWriter(directory, run_id="again").open()


def test_snapshot_index_parsing_refuses_malformed_entries(tmp_path):
    directory = tmp_path / "snapshots"
    directory.mkdir()
    assert snapshots.snapshot_summary(tmp_path) is None
    good = {"path": "snapshot_1_steps.zip", "sha256": "a" * 64}
    for entry, match in (
        ({"timestep": -1, "model": good}, "timestep"),
        ({"timestep": 1, "model": {"path": "../x.zip", "sha256": "a" * 64}}, "file name"),
        ({"timestep": 1, "model": {"path": "x.zip"}}, "sha256"),
        ({"timestep": 1, "n_updates": "3", "model": good}, "n_updates"),
    ):
        (directory / "index.json").write_text(
            json.dumps({"schema": snapshots.SNAPSHOT_INDEX_SCHEMA, "snapshots": [entry]})
        )
        with pytest.raises(ValueError, match=match):
            snapshots.read_snapshot_index(directory)
    (directory / "index.json").write_text(json.dumps({"schema": "other/1"}))
    with pytest.raises(ValueError, match="schema"):
        snapshots.read_snapshot_index(directory)
    assert "schema" in snapshots.snapshot_summary(tmp_path)["problem"]


def _ball_balance_config(log_dir: Path, **overrides: Any) -> TrainConfig:
    return build_train_config(
        "BallBalance",
        log_dir=str(log_dir),
        seed=0,
        n_envs=2,
        total_timesteps=600,
        eval_freq=200,
        n_eval_episodes=2,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        env_fn=make_env_fn("BallBalance", env_overrides={"episode_len": 50}),
        eval_env_fn=make_eval_env_fn("BallBalance", env_overrides={"episode_len": 50}),
        model_kwargs={"buffer_size": 1_000, "learning_starts": 50, "batch_size": 32},
        **overrides,
    )


def test_snapshots_change_no_training_stream(tmp_path):
    """The snapshot callback only reads the model: the same seeded run with
    and without it trains the same weights, evaluates the same rows and
    leaves the global RNGs in the same state."""
    results = {}
    for flag in (False, True):
        model = train(_ball_balance_config(tmp_path / str(flag), snapshot_evals=flag))
        results[flag] = (
            {k: v.detach().clone() for k, v in model.policy.state_dict().items()},
            (tmp_path / str(flag) / "metrics" / "eval_info.csv").read_text(),
            np.random.get_state()[1].copy(),
            torch.get_rng_state().clone(),
        )
    (weights_off, csv_off, np_off, torch_off), (weights_on, csv_on, np_on, torch_on) = (
        results[False],
        results[True],
    )
    assert weights_off.keys() == weights_on.keys()
    assert all(torch.equal(weights_off[k], weights_on[k]) for k in weights_off)
    assert csv_off == csv_on
    assert np.array_equal(np_off, np_on)
    assert torch.equal(torch_off, torch_on)
    assert not (tmp_path / "False" / "snapshots").exists()
    assert json.loads((tmp_path / "False" / "run_record.json").read_text())["snapshots"] is None
    entries = snapshots.read_snapshot_index(tmp_path / "True" / "snapshots").entries
    assert [entry.timestep for entry in entries] == [200, 400, 600]


def test_snapshot_evals_is_validated_and_reuse_sees_previous_snapshots(tmp_path):
    with pytest.raises(TypeError, match="snapshot_evals must be a bool"):
        train(_ball_balance_config(tmp_path / "bad", snapshot_evals="yes"))
    assert not (tmp_path / "bad").exists()
    stale = tmp_path / "stale"
    (stale / "snapshots").mkdir(parents=True)
    (stale / "snapshots" / "index.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshots/index.json"):
        train(_ball_balance_config(stale, snapshot_evals=True))


def test_a_failed_snapshot_is_listed_and_training_continues(tmp_path, monkeypatch, capsys):
    real_write = snapshots.SnapshotWriter.write

    def flaky(self, timestep, **kwargs):
        if timestep == 400:
            raise OSError(5, "Input/output error")
        return real_write(self, timestep, **kwargs)

    monkeypatch.setattr(snapshots.SnapshotWriter, "write", flaky)
    train(_ball_balance_config(tmp_path / "run", snapshot_evals=True))
    assert "could not save the evaluation snapshot at 400 steps" in capsys.readouterr().out
    record = json.loads((tmp_path / "run" / "run_record.json").read_text())
    assert record["status"] == "completed"
    assert record["snapshots"]["timesteps"] == [200, 600]
    assert record["snapshots"]["failures"] == [400]


def test_scores_are_a_post_training_location():
    assert records.is_post_training_path("scores/paddle-npoint@v2.jsonl")
    assert not records.is_post_training_path("snapshots/index.json")


# -- the probe CLI --------------------------------------------------------------


def test_probe_scores_a_witness_under_a_derived_protocol(capsys):
    from courtside_dynamics.experiments.probe import (
        main,
        probe_protocol,
        score_scripted,
    )

    argv = [
        "--recipe", "PaddleTennis", "--policy", "oracle",
        "--protocol", "paddle-npoint@v2",
        "--env", "episode_len=150", "--episodes", "2", "--seed-start", "5500",
    ]
    assert main([*argv, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    derived = probe_protocol(
        get_protocol("paddle-npoint@v2"), env_overrides={"episode_len": 150}, episodes=2
    )
    assert report["protocol"] == derived.id
    assert report["protocol_fingerprint"] == derived.fingerprint
    assert report["seed_start"] == 5500
    assert report["metrics"] == score_scripted("oracle", derived, seed_start=5500)
    assert report["metrics"]["serve_side_is_policy_ep_mean"] == 0.5
    assert main(argv) == 0
    assert "episode_reward_mean" in capsys.readouterr().out
    # Refusals exit 2 with the reason.
    assert main([*argv[:6], "--episodes", "1"]) == 2
    assert "pass seed_start" in capsys.readouterr().err
    assert main(["--recipe", "WallBall", *argv[2:]]) == 2
    assert "evaluates PaddleTennisEnv" in capsys.readouterr().err
    assert main([*argv, "--normalizer", "x.pkl"]) == 2
    assert "raw observations" in capsys.readouterr().err


def test_probe_scores_a_snapshot_like_the_harness(tiny_runs, capsys):
    from courtside_dynamics.experiments.probe import main

    run_dir = tiny_runs["PPO"]
    model = run_dir / "snapshots" / "snapshot_512_steps.zip"
    normalizer = run_dir / "snapshots" / "snapshot_vecnormalize_512_steps.pkl"
    assert (
        main(
            [
                "--recipe", "PaddleTennis", "--policy", str(model),
                "--normalizer", str(normalizer), "--protocol", "paddle-npoint@v3",
                "--env", f"episode_len={_EPISODE_LEN}", "--episodes", "2", "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    # The run's own paired block, read from its config.json.
    assert report["seed_start"] == 1_000_000
    assert report["metrics"] == {
        k: v for k, v in _eval_info_rows(run_dir)[512].items() if k != "n_updates"
    }


def test_experiments_modules_import_without_the_training_stack():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules["stable_baselines3"] = None
        sys.modules["torch"] = None
        import courtside_dynamics.experiments.select
        import courtside_dynamics.experiments.snapshots
        import courtside_dynamics.experiments.protocols
        import courtside_dynamics.experiments.harness
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
