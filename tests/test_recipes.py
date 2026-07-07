"""Tests for the recipe registry that drives the consolidated notebook.

``build_train_config`` is the single entry point the training notebook
uses, so a broken recipe (bad env kwargs, an ``extra_cfg`` key that
``TrainConfig`` doesn't accept, a stale ``info_row_fn``) breaks every
run of that env. These tests materialize each recipe for real.
"""
from __future__ import annotations

import pytest

from courtside_dynamics.recipes import (
    RECIPES,
    build_train_config,
    make_env_fn,
)


@pytest.mark.parametrize("env_name", sorted(RECIPES))
def test_build_train_config_materializes(env_name, tmp_path):
    """Every recipe must produce a valid TrainConfig whose env factory
    builds an env of the recipe's class."""
    cfg = build_train_config(env_name, algo="SAC", log_dir=str(tmp_path))
    assert cfg.algo == "SAC"
    assert cfg.log_dir == str(tmp_path)
    assert cfg.total_timesteps == RECIPES[env_name].default_total_timesteps
    assert cfg.name_prefix.endswith("_sac")

    env = cfg.env_fn()
    try:
        assert isinstance(env, RECIPES[env_name].env_cls)
        assert env.render_mode == "rgb_array"
    finally:
        env.close()


def test_quick_test_overrides_shrink_budget(tmp_path):
    cfg = build_train_config(
        "BallBalance", log_dir=str(tmp_path), quick_test=True
    )
    assert cfg.total_timesteps == 25_000
    assert cfg.eval_freq == 5_000


def test_explicit_overrides_win_over_quick_test(tmp_path):
    """Caller overrides are applied last, on top of quick-test values."""
    cfg = build_train_config(
        "BallBalance",
        log_dir=str(tmp_path),
        quick_test=True,
        total_timesteps=123,
        n_envs=2,
    )
    assert cfg.total_timesteps == 123
    assert cfg.n_envs == 2


def test_unknown_env_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        build_train_config("NoSuchEnv", log_dir=str(tmp_path))


def test_make_env_fn_returns_fresh_instances():
    factory = make_env_fn("BallBalance")
    a, b = factory(), factory()
    try:
        assert a is not b
        assert type(a) is type(b)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("env_name", ["BallBounce", "WallBall"])
def test_contact_envs_wire_a_success_metric(env_name, tmp_path):
    """Both contact-driven envs define ``success_key`` so eval runs log
    ``eval_info/success_rate``, and the key must exist in the env's
    ``info`` dict (a typo'd key silently omits the metric)."""
    import numpy as np

    cfg = build_train_config(env_name, log_dir=str(tmp_path))
    assert cfg.success_key == "bounce_count"

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert cfg.success_key in info
    finally:
        env.close()


def test_ball_bounce_info_row_matches_header(tmp_path):
    """The custom CSV row must flatten to exactly the header width."""
    import numpy as np

    from courtside_dynamics.callbacks.video_record import _flatten_row

    cfg = build_train_config("BallBounce", log_dir=str(tmp_path))
    assert cfg.csv_header is not None and cfg.info_row_fn is not None

    env = cfg.env_fn()
    try:
        env.reset(seed=0)
        _, reward, _, done, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        row = _flatten_row(
            cfg.info_row_fn(info, float(reward), float(reward), bool(done))
        )
        assert len(row) == len(list(cfg.csv_header))
    finally:
        env.close()
