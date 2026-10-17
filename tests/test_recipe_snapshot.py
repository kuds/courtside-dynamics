"""The recipe snapshot: part 1 of the SAC freeze.

``tests/golden/recipe_snapshot.json`` records the ``TrainConfig`` that
``build_train_config`` resolves for every recipe x {default algo, SAC,
DEMOSAC, PPO} x quick_test, canonically serialized by
``tools/recipe_snapshot.py`` (env kwargs recovered from the factories,
callables by qualified name). A diff here means a recipe's training path
changed: declare it and re-record with ``--write`` in the same commit,
never to make the test pass.
"""
from __future__ import annotations

import dataclasses
import functools
import json
from enum import IntEnum

import numpy as np
import pytest

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.recipes import RECIPES, build_train_config, make_env_fn
from tools import recipe_snapshot


def test_live_resolutions_match_the_recorded_snapshot(monkeypatch):
    # The snapshot must never build an env: refuse any construction.
    from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

    def no_envs(*_args, **_kwargs):
        raise AssertionError("the recipe snapshot constructed an env")

    monkeypatch.setattr(MujocoEnv, "__init__", no_envs)
    recorded = recipe_snapshot.load_snapshot(recipe_snapshot.SNAPSHOT_PATH)
    problems = recipe_snapshot.compare(
        recorded["entries"], recipe_snapshot.build_snapshot()
    )
    assert not problems, "\n".join([*problems, recipe_snapshot.DIFF_EXPLANATION])


def test_snapshot_spans_the_matrix():
    entries = recipe_snapshot.load_snapshot(recipe_snapshot.SNAPSHOT_PATH)["entries"]
    for name, recipe in RECIPES.items():
        algos = {recipe.default_algo.upper(), "SAC", "DEMOSAC", "PPO"}
        for algo in algos:
            for quick_test in (False, True):
                assert f"{name}|{algo}|quick_test={quick_test}" in entries
    # Every recipe records its PPO resolution, the PPO-default ones too;
    # the depth ladders record their refusal (SAC-only gate levers).
    ppo_default = [n for n, r in RECIPES.items() if r.default_algo.upper() == "PPO"]
    assert ppo_default
    refused = {"WallBallDepthCurriculum", "WallBallDepthCurriculumAligned"}
    for name in RECIPES:
        entry = entries[f"{name}|PPO|quick_test=False"]
        if name in refused:
            assert "clear_replay_buffer_on_advance" in entry["refused"]
        else:
            assert entry["train_config"]["algo"] == "PPO"
    # Paired evaluation survives only a seeded build, so it is recorded.
    seeded = entries["PaddleTennis|SAC|quick_test=False|seed=0"]["train_config"]
    assert seeded["eval_reset_options"] == [{"serve_side": "a"}, {"serve_side": "b"}]
    assert "log_dir" not in seeded


def test_env_factories_are_recorded_by_their_constructor_kwargs():
    cfg = build_train_config("PaddleTennis", log_dir="x", seed=0)
    out = recipe_snapshot.canonical_train_config(cfg)
    assert out["env_fn"] == {
        "__env_factory__": "courtside_dynamics.recipes.make_env_fn.<locals>._factory",
        "env_cls": "courtside_dynamics.envs.paddle_tennis.PaddleTennisEnv",
        "kwargs": recipe_snapshot.canonical(
            dict(RECIPES["PaddleTennis"].env_kwargs)
        ),
    }
    assert out["eval_env_fn"]["kwargs"]["drill_fraction"] == 0.0
    assert out["info_row_fn"] == {
        "__callable__": "courtside_dynamics.recipes._paddle_tennis_info_row"
    }
    # train_freq is a tuple in SB3's contract; a list would be a diff.
    assert out["model_kwargs"]["train_freq"] == [64, "step"]


def test_foreign_env_factories_refuse_to_serialize():
    cfg = build_train_config(
        "BallBalance", log_dir="x", env_fn=lambda: BallBalanceEnv(episode_len=5)
    )
    with pytest.raises(TypeError, match="not a recipe env factory"):
        recipe_snapshot.canonical_train_config(cfg)
    # A recipe factory with overrides is fine and carries them.
    factory = make_env_fn("BallBalance", env_overrides={"episode_len": 5})
    assert recipe_snapshot._env_factory(factory, "$")["kwargs"]["episode_len"] == 5


def test_canonical_form_keeps_types_apart():
    class Stage(IntEnum):
        ONE = 1

    @dataclasses.dataclass
    class Box:
        x: tuple[int, ...]

    canonical = recipe_snapshot.canonical
    assert canonical((1, 2)) == [1, 2]
    assert canonical([1, 2]) == {"__list__": [1, 2]}
    assert canonical(Stage.ONE) != canonical(1)
    assert canonical({0: "a"}) == {"__items__": [[0, "a"]]}
    assert canonical(np.float32(0.5)) == {"__numpy__": "<f4", "value": 0.5}
    assert canonical(Box((1,)))["fields"] == {"x": [1]}
    assert canonical(float("nan")) == {"__float__": "nan"}
    assert canonical(functools.partial(max, 1))["__partial__"] == {
        "__callable__": "builtins.max"
    }
    assert canonical(BallBalanceEnv) == {
        "__class__": "courtside_dynamics.envs.ball_balance.BallBalanceEnv"
    }
    with pytest.raises(ValueError, match="absolute path"):
        canonical({"library": "/home/someone/lib.pkl"})
    with pytest.raises(TypeError, match="cannot canonicalize"):
        canonical(object())
    json.dumps(canonical({"a": (1, [2], {3})}))  # always JSON-ready


def test_check_reports_a_diff_loudly(tmp_path, capsys):
    snapshot = recipe_snapshot.load_snapshot(recipe_snapshot.SNAPSHOT_PATH)
    key = "BallBalance|SAC|quick_test=False"
    snapshot["entries"][key]["train_config"]["n_envs"] = 99
    path = tmp_path / "recipe_snapshot.json"
    path.write_text(json.dumps(snapshot))
    assert recipe_snapshot.main(["--snapshot", str(path)]) == 1
    err = capsys.readouterr().err
    assert f"{key}:\n  $.train_config.n_envs: expected 99, got 8" in err
    assert recipe_snapshot.DIFF_EXPLANATION in err


@pytest.mark.parametrize(
    "recorded, live",
    [(8, 8.0), (True, 1), (0.0, 0), (0.0, -0.0), ([64, "step"], [64.0, "step"])],
)
def test_compare_keeps_types_apart(recorded, live):
    """Python's == calls these equal; the canonical JSON text does not."""
    key = "BallBalance|SAC|quick_test=False"
    problems = recipe_snapshot.compare(
        {key: {"train_config": {"n_envs": recorded}}},
        {key: {"train_config": {"n_envs": live}}},
    )
    assert problems == [
        f"{key}:\n  $.train_config.n_envs: expected {recorded!r}, got {live!r}"
    ]


def test_check_refuses_another_checkouts_package(tmp_path, monkeypatch):
    """In a worktree the editable install imports the main checkout's code."""
    monkeypatch.setattr(recipe_snapshot, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="another checkout's code"):
        recipe_snapshot.main([])
