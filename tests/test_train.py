"""Tests for the shared training entry point's model construction.

These pin down three pipeline-level guarantees that are easy to regress
and hard to notice from a training curve alone:

1. SAC runs as many gradient updates as transitions it collects per
   rollout (``gradient_steps=-1``). With SB3's default of 1, a vectorised
   training env quietly performs only ``1/n_envs`` of the updates it
   should, starving the policy of learning as ``n_envs`` grows.
2. ``seed`` is forwarded to the algorithm so runs are reproducible.
3. ``policy`` is actually used -- it used to be hardcoded to
   ``"MlpPolicy"`` so the configured value was silently ignored.
"""
from __future__ import annotations

import pytest
from stable_baselines3.common.env_util import make_vec_env

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.training.train import _build_algo, _offset_seed


@pytest.fixture
def env():
    venv = make_vec_env(lambda: BallBalanceEnv(), n_envs=1)
    try:
        yield venv
    finally:
        venv.close()


def test_sac_defaults_to_full_gradient_steps(env, tmp_path):
    """SAC should match gradient updates to steps collected (``-1``)."""
    model = _build_algo("SAC", env, str(tmp_path))
    assert model.gradient_steps == -1


def test_sac_respects_explicit_gradient_steps(env, tmp_path):
    """An explicit ``gradient_steps`` still wins over the off-policy default."""
    model = _build_algo("SAC", env, str(tmp_path), gradient_steps=2)
    assert model.gradient_steps == 2


def test_ppo_does_not_receive_gradient_steps(env, tmp_path):
    """PPO is on-policy and has no ``gradient_steps``; building must not
    error from the SAC-only default leaking through."""
    model = _build_algo("PPO", env, str(tmp_path))
    assert not hasattr(model, "gradient_steps")


def test_seed_is_forwarded_to_model(env, tmp_path):
    model = _build_algo("SAC", env, str(tmp_path), seed=1234)
    assert model.seed == 1234


def test_policy_argument_is_forwarded(env, tmp_path):
    """A bogus policy name must reach SB3 and raise -- proving ``policy``
    isn't silently dropped in favour of a hardcoded ``MlpPolicy``."""
    with pytest.raises((ValueError, KeyError)):
        _build_algo("SAC", env, str(tmp_path), policy="NoSuchPolicy")


def test_unknown_algo_raises(env, tmp_path):
    with pytest.raises(ValueError):
        _build_algo("DDPG", env, str(tmp_path))


def test_offset_seed_passes_through_none():
    assert _offset_seed(None, 1) is None


def test_offset_seed_is_distinct_per_offset():
    base = 100
    assert _offset_seed(base, 0) == 100
    assert _offset_seed(base, 1) == 101
    assert _offset_seed(base, 2) == 102


def test_algo_registry_has_sac_and_ppo():
    from courtside_dynamics.training.algos import ALGOS, OFF_POLICY_ALGOS

    assert set(ALGOS) == {"SAC", "PPO"}
    # SAC is off-policy (gets gradient_steps=-1); PPO is not.
    assert "SAC" in OFF_POLICY_ALGOS
    assert "PPO" not in OFF_POLICY_ALGOS


def test_scalar_info_keys_reexported_from_video_record():
    """The helper moved to ``callbacks._info`` but must stay importable
    from ``video_record`` (existing code and tests import it there)."""
    from courtside_dynamics.callbacks._info import _scalar_info_keys as canonical
    from courtside_dynamics.callbacks.video_record import (
        _scalar_info_keys as reexported,
    )

    assert canonical is reexported


def test_verbose_forwarded_to_model(env, tmp_path):
    model = _build_algo("SAC", env, str(tmp_path), verbose=2)
    assert model.verbose == 2
