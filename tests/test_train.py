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
from courtside_dynamics.training.train import (
    _build_algo,
    _env_steps_to_calls,
    _offset_seed,
)


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


def test_algo_name_is_case_insensitive(env, tmp_path):
    """``algo="sac"`` must resolve like ``"SAC"`` -- every other algo
    comparison in the project uses ``.upper()``, so the registry lookup
    can't be the one place that is case-sensitive."""
    model = _build_algo("sac", env, str(tmp_path))
    # The off-policy gradient_steps default must apply to "sac" too.
    assert model.gradient_steps == -1


def test_train_rejects_unknown_algo_before_any_setup(tmp_path):
    """A typo'd algo must fail fast, before envs are built or artifacts
    written -- the log dir should not even exist afterwards."""
    import os

    from courtside_dynamics.training import TrainConfig, train

    log_dir = os.path.join(str(tmp_path), "run")
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="DDPG",
        log_dir=log_dir,
    )
    with pytest.raises(ValueError):
        train(cfg)
    assert not os.path.exists(log_dir), (
        "train() built artifacts before validating the algo name"
    )


def test_offset_seed_passes_through_none():
    assert _offset_seed(None, 1) is None


def test_offset_seed_is_distinct_per_offset():
    base = 100
    assert _offset_seed(base, 0) == 100
    assert _offset_seed(base, 1) == 101
    assert _offset_seed(base, 2) == 102


def test_env_steps_to_calls_scales_with_n_envs():
    """An env-step cadence of N means N // n_envs vec-steps, floored at 1,
    so the wall-clock eval/checkpoint cadence is independent of n_envs."""
    assert _env_steps_to_calls(25_000, 1) == 25_000
    assert _env_steps_to_calls(25_000, 4) == 6_250
    # Cadence smaller than one vec step still fires every call.
    assert _env_steps_to_calls(2, 4) == 1
    # Degenerate n_envs values must not divide by zero.
    assert _env_steps_to_calls(100, 0) == 100


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


def test_verbose_in_model_kwargs_does_not_collide(env, tmp_path):
    """train() routes verbose through model_kwargs; a user-supplied
    model_kwargs['verbose'] must not raise a duplicate-keyword TypeError."""
    model_kwargs = {"verbose": 1}
    model = _build_algo("SAC", env, str(tmp_path), policy="MlpPolicy", **model_kwargs)
    assert model.verbose == 1


def test_early_stop_patience_cuts_training_short(tmp_path):
    """With ``early_stop_patience`` set and a flat eval reward (BallBalance
    caps at +1/step), training must stop well before the full budget --
    and the summary must record the shortfall. Regression target: the
    first WallBall run trained 3.8M steps past its best checkpoint."""
    import os

    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig, train

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=40),
        algo="SAC",
        total_timesteps=6_000,
        log_dir=str(tmp_path),
        n_envs=1,
        seed=0,
        eval_freq=200,
        early_stop_patience=1,  # warm-up 1 eval + 1 non-improving eval
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        normalize_obs=False,
        n_eval_episodes=1,
        model_kwargs={"learning_starts": 16, "buffer_size": 500},
    )
    model = train(cfg)
    assert model.num_timesteps < cfg.total_timesteps, (
        f"early stop never fired: trained {model.num_timesteps} of "
        f"{cfg.total_timesteps}"
    )
    summary = open(os.path.join(str(tmp_path), "stage_summary.txt")).read()
    assert "stopped early" in summary
    # The knob is part of the run's provenance snapshot.
    import json

    config = json.load(open(os.path.join(str(tmp_path), "config.json")))
    assert config["train_config"]["early_stop_patience"] == 1


def test_csv_logger_survives_learn(tmp_path):
    """The logger configured in train() must persist across model.learn so
    progress.csv is actually written -- SB3 only resets the logger when it
    wasn't explicitly set, so set_logger must make it stick."""
    import os

    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import TrainConfig, train

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="SAC",
        total_timesteps=256,
        log_dir=str(tmp_path),
        n_envs=1,
        eval_freq=10_000,  # don't fire EvalCallback in this short run
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        normalize_obs=False,
        n_eval_episodes=1,  # keep the end-of-train final eval cheap
        model_kwargs={"learning_starts": 16, "buffer_size": 500},
    )
    train(cfg)
    assert os.path.exists(
        os.path.join(str(tmp_path), "tensorboard", "progress.csv")
    ), "CSV logger was reset by learn(); progress.csv not written"
