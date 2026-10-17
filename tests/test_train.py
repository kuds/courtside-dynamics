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

import csv
import hashlib
import importlib
import json
import os
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.training.artifacts import (
    update_run_config_with_model,
    write_run_config,
)
from courtside_dynamics.training.train import (
    SelectiveVecNormalize,
    TrainConfig,
    WarmStartConfig,
    _build_algo,
    _env_steps_to_calls,
    _load_warm_start_normalizer,
    _offset_seed,
    _prepare_warm_start,
    train,
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


def test_sac_train_freq_list_from_toml_is_coerced_to_tuple(env, tmp_path):
    """A TOML ``train_freq = [64, "step"]`` arrives as a list, which SB3
    rejects at construction; the chokepoint coerces so a run-config
    override cannot fail late inside ``train()``."""
    model = _build_algo("SAC", env, str(tmp_path), train_freq=[4, "step"])
    assert model.train_freq.frequency == 4
    assert model.train_freq.unit.value == "step"


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


def test_algo_registry_has_sac_ppo_and_demosac():
    """The registry is the single source of truth, and off-policy
    membership is the silent trap: an off-policy algo missing from
    OFF_POLICY_ALGOS keeps SB3's gradient_steps=1 against a vectorised
    train_freq and trains 1 update per n_envs*train_freq transitions."""
    from courtside_dynamics.training.algos import ALGOS, OFF_POLICY_ALGOS
    from courtside_dynamics.training.demo_sac import DemoSAC

    assert set(ALGOS) == {"SAC", "PPO", "DEMOSAC"}
    assert ALGOS["DEMOSAC"] is DemoSAC
    assert "SAC" in OFF_POLICY_ALGOS
    assert "DEMOSAC" in OFF_POLICY_ALGOS
    assert "PPO" not in OFF_POLICY_ALGOS


def test_algo_families_are_derived_from_the_registered_classes():
    """PPO plan Step 1: the derived sets equal the hand-kept ones they
    replace, and every algorithm maps to exactly one config family."""
    from courtside_dynamics.training.algos import (
        ALGO_FAMILIES,
        FAMILIES,
        OFF_POLICY_ALGOS,
        ON_POLICY_ALGOS,
        algo_family,
    )

    assert OFF_POLICY_ALGOS == frozenset({"SAC", "DEMOSAC"})
    assert ON_POLICY_ALGOS == frozenset({"PPO"})
    assert ALGO_FAMILIES == {"SAC": "SAC", "DEMOSAC": "SAC", "PPO": "PPO"}
    assert FAMILIES == frozenset({"SAC", "PPO"})
    # Case-insensitive like the registry; unknown names fail loudly.
    assert algo_family("DemoSAC") == "SAC"
    assert algo_family("ppo") == "PPO"
    with pytest.raises(ValueError, match="Unknown algo"):
        algo_family("PPPO")
    # A family never mixes on- and off-policy members.
    for name, family in ALGO_FAMILIES.items():
        assert (name in OFF_POLICY_ALGOS) == (family in OFF_POLICY_ALGOS)


@pytest.mark.parametrize(
    "algo, expected", [("PPO", True), ("SAC", False), ("DEMOSAC", False), ("DemoSAC", False)]
)
def test_reward_normalization_default_follows_the_family(tmp_path, algo, expected):
    """normalize_reward=None resolves on for PPO, off for SAC and DemoSAC,
    in both resolvers (the live config and a recorded train_config)."""
    from courtside_dynamics.training.train import (
        TrainConfig,
        _config_norm_reward,
        _resolved_norm_reward,
    )

    cfg = TrainConfig(env_fn=BallBalanceEnv, algo=algo, log_dir=str(tmp_path))
    assert _resolved_norm_reward(cfg) is expected
    assert _config_norm_reward({"algo": algo, "normalize_reward": None}) is expected
    # An explicit choice wins either way.
    cfg.normalize_reward = not expected
    assert _resolved_norm_reward(cfg) is (not expected)
    assert _config_norm_reward({"algo": algo, "normalize_reward": not expected}) is (
        not expected
    )


def test_recorded_reward_normalization_of_an_unknown_algo_is_off():
    """config.json data is unvalidated: a missing or unknown algo resolves
    off, as the hand-written == "PPO" test did, instead of raising."""
    from courtside_dynamics.training.train import _config_norm_reward

    assert _config_norm_reward({}) is False
    assert _config_norm_reward({"algo": "TD3"}) is False
    with pytest.raises(ValueError, match="must be bool or null"):
        _config_norm_reward({"algo": "PPO", "normalize_reward": "yes"})


def test_validate_model_kwargs_accepts_and_rejects_per_algo():
    from courtside_dynamics.training.algos import validate_model_kwargs

    # Shared and per-algo keys pass for the algo that owns them.
    validate_model_kwargs("PPO", {"n_steps": 512, "ent_coef": 0.01})
    validate_model_kwargs("SAC", {"buffer_size": 1_000, "ent_coef": "auto"})
    validate_model_kwargs("sac", {"gradient_steps": -1})  # case-insensitive

    with pytest.raises(ValueError, match="not accepted by PPO"):
        validate_model_kwargs("PPO", {"buffer_size": 1_000})
    with pytest.raises(ValueError, match="numeric ent_coef"):
        validate_model_kwargs("PPO", {"ent_coef": "auto_0.02"})
    # Keys _build_algo supplies itself are rejected outright.
    with pytest.raises(ValueError, match="trainer supplies"):
        validate_model_kwargs("PPO", {"tensorboard_log": "/tmp/tb"})


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
    # The reward path (SB3's StopTrainingOnNoModelImprovement under
    # reward selection) now says why it stopped, in the summary and the
    # run record (repo_review_20261016 train-1); when it stops is SB3's
    # rule, unchanged: the count must exceed the patience.
    reason = (
        "early_stop_patience: no new best mean reward for the last 2 "
        "evaluations (reward selection: SB3 StopTrainingOnNoModelImprovement, "
        "patience 1)"
    )
    assert f"Stop reason:    {reason}" in summary
    record = json.load(open(os.path.join(str(tmp_path), "run_record.json")))
    assert record["status"] == "early_stop"
    assert record["stop_reason"] == reason
    assert model.num_timesteps == 3 * 200
    # The knob is part of the run's provenance snapshot.
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
    # progress.csv lives directly in metrics/ (pandas-readable metrics,
    # not TB event data), NOT inside the tensorboard folder.
    assert os.path.exists(os.path.join(str(tmp_path), "metrics", "progress.csv")), (
        "CSV logger was reset by learn(); progress.csv not written"
    )
    assert not os.path.exists(
        os.path.join(str(tmp_path), "metrics", "tensorboard", "progress.csv")
    )


def test_train_closes_its_sb3_logger(tmp_path, monkeypatch):
    """train() creates the SB3 Logger (progress.csv + TensorBoard) and
    must release it on every exit: a campaign notebook runs several legs
    in one process, and each leaked leg kept its CSV and event-file
    handles open (ResourceWarnings under pytest). Covers a clean return
    and an exception raised after the logger exists."""
    from stable_baselines3.common.logger import (
        CSVOutputFormat,
        TensorBoardOutputFormat,
    )

    from courtside_dynamics.training import TrainConfig, train

    # The package re-exports the train() function under the submodule's
    # name, so fetch the module itself to patch its globals.
    train_module = importlib.import_module("courtside_dynamics.training.train")
    created = []

    class _TrackedLogger(train_module.Logger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(train_module, "Logger", _TrackedLogger)

    def assert_closed(logger):
        kinds = {type(fmt) for fmt in logger.output_formats}
        assert {CSVOutputFormat, TensorBoardOutputFormat} <= kinds
        for fmt in logger.output_formats:
            if isinstance(fmt, CSVOutputFormat):
                assert fmt.file.closed
            if isinstance(fmt, TensorBoardOutputFormat):
                # torch's SummaryWriter.close() drops its event-file
                # writers (SB3 keeps the SummaryWriter object itself).
                assert fmt.writer.all_writers is None

    def make_cfg(log_dir):
        return TrainConfig(
            env_fn=lambda: BallBalanceEnv(),
            algo="SAC",
            total_timesteps=64,
            log_dir=str(log_dir),
            n_envs=1,
            eval_freq=10_000,
            checkpoint_freq=0,
            video_freq=0,
            record_video=False,
            info_dict_eval=False,
            normalize_obs=False,
            n_eval_episodes=1,
            model_kwargs={"learning_starts": 16, "buffer_size": 500},
        )

    model = train(make_cfg(tmp_path / "clean"))
    assert len(created) == 1
    assert model.logger is created[0]
    assert_closed(created[0])

    def fail_after_logger(*args, **kwargs):
        raise RuntimeError("boom after the logger exists")

    monkeypatch.setattr(train_module, "update_run_config_with_model", fail_after_logger)
    with pytest.raises(RuntimeError, match="boom after the logger exists"):
        train(make_cfg(tmp_path / "raised"))
    assert len(created) == 2
    assert_closed(created[1])


def test_warm_start_config_validates_and_canonicalizes_indices(tmp_path):
    config = WarmStartConfig(tmp_path, reset_observation_indices=(3, 1))
    assert config.reset_observation_indices == (1, 3)
    with pytest.raises(TypeError, match="integers"):
        WarmStartConfig(tmp_path, reset_observation_indices=(True,))
    with pytest.raises(ValueError, match="non-negative"):
        WarmStartConfig(tmp_path, reset_observation_indices=(-1,))
    with pytest.raises(ValueError, match="unique"):
        WarmStartConfig(tmp_path, reset_observation_indices=(1, 1))


def test_selective_vec_normalize_round_trip_and_pickle(tmp_path):
    base_env = make_vec_env(lambda: BallBalanceEnv(), n_envs=1)
    normalizer = SelectiveVecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=False,
        normalize_obs_excluded_indices=(0, 2),
    )
    try:
        shape = normalizer.observation_space.shape
        assert shape is not None
        normalizer.obs_rms.mean = np.linspace(1.0, 2.0, shape[0])
        normalizer.obs_rms.var = np.linspace(2.0, 3.0, shape[0])
        raw = np.linspace(-3.0, 3.0, shape[0], dtype=np.float64)[None, :]
        normalized = normalizer.normalize_obs(raw)
        np.testing.assert_allclose(normalized[..., (0, 2)], raw[..., (0, 2)])
        np.testing.assert_allclose(
            normalizer.unnormalize_obs(normalized),
            raw,
            atol=1e-7,
        )

        path = tmp_path / "selective.pkl"
        normalizer.save(str(path))
        loaded_base = make_vec_env(lambda: BallBalanceEnv(), n_envs=1)
        loaded = VecNormalize.load(str(path), loaded_base)
        try:
            assert isinstance(loaded, SelectiveVecNormalize)
            assert loaded.normalize_obs_excluded_indices == (0, 2)
            loaded_normalized = loaded.normalize_obs(raw)
            np.testing.assert_allclose(loaded_normalized, normalized)
        finally:
            loaded.close()
    finally:
        normalizer.close()


def _make_ppo_warm_start_source(tmp_path, *, excluded_indices=(0,)):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    def env_fn():
        return BallBalanceEnv(episode_len=12)

    source_cfg = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        log_dir=str(source_dir),
        n_envs=1,
        normalize_obs=True,
        normalize_reward=True,
        normalize_obs_excluded_indices=tuple(excluded_indices),
        model_kwargs={"n_steps": 8, "batch_size": 4, "n_epochs": 1},
    )
    raw_env = make_vec_env(env_fn, n_envs=1, seed=7)
    normalizer = SelectiveVecNormalize(
        raw_env,
        norm_obs=True,
        norm_reward=True,
        normalize_obs_excluded_indices=tuple(excluded_indices),
    )
    model = _build_algo(
        "PPO",
        normalizer,
        str(source_dir),
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        seed=7,
    )
    with torch.no_grad():
        first_parameter = next(model.policy.parameters())
        first_parameter.fill_(0.125)
    shape = normalizer.observation_space.shape
    assert shape is not None
    normalizer.obs_rms.mean = np.linspace(10.0, 20.0, shape[0])
    normalizer.obs_rms.var = np.linspace(2.0, 4.0, shape[0])
    normalizer.obs_rms.count = 123.0
    normalizer.ret_rms.mean = np.asarray(9.0)
    normalizer.ret_rms.var = np.asarray(4.0)
    normalizer.ret_rms.count = 55.0
    model.save(source_dir / "best_model.zip")
    normalizer.save(source_dir / "best_vec_normalize.pkl")
    write_run_config(source_cfg, str(source_dir))
    update_run_config_with_model(model, str(source_dir))
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in model.policy.state_dict().items()
    }
    normalizer.close()
    return source_dir, env_fn, source_state


def test_warm_start_normalizer_carries_obs_and_resets_target_state(tmp_path):
    source_dir, env_fn, _source_state = _make_ppo_warm_start_source(tmp_path)
    target_cfg = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        log_dir=str(tmp_path / "target"),
        n_envs=2,
        normalize_obs=True,
        normalize_reward=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(
            source_dir,
            reset_observation_indices=(1,),
        ),
    )
    artifacts = _prepare_warm_start(target_cfg)
    assert artifacts is not None
    base_env = make_vec_env(env_fn, n_envs=2, seed=3)
    loaded = _load_warm_start_normalizer(
        base_env,
        artifacts,
        target_cfg,
        norm_reward=True,
    )
    try:
        assert loaded.num_envs == 2
        assert loaded.normalize_obs_excluded_indices == (0,)
        assert loaded.obs_rms.mean[0] == pytest.approx(10.0)
        assert loaded.obs_rms.mean[1] == pytest.approx(
            artifacts.reset_observation_values[0]
        )
        assert loaded.obs_rms.var[1] == pytest.approx(1.0)
        assert loaded.obs_rms.count == pytest.approx(123.0)
        assert loaded.ret_rms.mean == pytest.approx(0.0)
        assert loaded.ret_rms.var == pytest.approx(1.0)
        assert loaded.ret_rms.count == pytest.approx(1e-4)
        np.testing.assert_array_equal(loaded.returns, np.zeros(2))
        assert loaded.training is True
    finally:
        loaded.close()


class _CaptureWarmStart(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.policy_state: dict[str, torch.Tensor] | None = None
        self.optimizer_state_count: int | None = None
        self.start_timestep: int | None = None

    def _on_training_start(self) -> None:
        self.policy_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.policy.state_dict().items()
        }
        self.optimizer_state_count = len(self.model.policy.optimizer.state)
        self.start_timestep = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        return True


def test_train_warm_starts_policy_only_and_records_provenance(tmp_path):
    source_dir, env_fn, source_state = _make_ppo_warm_start_source(tmp_path)
    target_dir = tmp_path / "target"
    capture = _CaptureWarmStart()
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        total_timesteps=8,
        log_dir=str(target_dir),
        n_envs=1,
        seed=13,
        eval_freq=10_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        n_eval_episodes=1,
        normalize_obs=True,
        normalize_reward=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
        model_kwargs={"n_steps": 8, "batch_size": 4, "n_epochs": 1},
        extra_callbacks=(capture,),
    )
    train(cfg)

    assert capture.policy_state is not None
    assert capture.policy_state.keys() == source_state.keys()
    for name, expected in source_state.items():
        torch.testing.assert_close(capture.policy_state[name], expected)
    assert capture.optimizer_state_count == 0
    assert capture.start_timestep == 0

    config = json.loads((target_dir / "config.json").read_text())
    initialization = config["initialization"]
    assert initialization["mode"] == "policy_and_observation_stats"
    assert initialization["optimizer_state_transferred"] is False
    assert initialization["reward_statistics_reset"] is True
    assert initialization["normalize_obs_excluded_indices"] == [0]
    assert "policy.optimizer_state" in initialization["reset"]
    for filename in ("best_model.zip", "best_vec_normalize.pkl", "config.json"):
        expected = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        assert initialization["source_artifacts"][filename]["sha256"] == expected


def _make_sac_warm_start_source(
    tmp_path, *, log_ent_coef=-6.5, model_kwargs=None
):
    """SAC sibling of the PPO helper: tiny model, marked policy weights,
    a deliberately collapsed entropy temperature, saved as a canonical
    best-run directory. ``model_kwargs`` (e.g. ``use_sde=True``) shape
    the source policy so a matching target can load its state dict."""
    source_dir = tmp_path / "sac_source"
    source_dir.mkdir()
    extra_kwargs = dict(model_kwargs or {})

    def env_fn():
        return BallBalanceEnv(episode_len=12)

    source_cfg = TrainConfig(
        env_fn=env_fn,
        algo="SAC",
        log_dir=str(source_dir),
        n_envs=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        model_kwargs={
            "buffer_size": 64,
            "learning_starts": 1_000,
            **extra_kwargs,
        },
    )
    raw_env = make_vec_env(env_fn, n_envs=1, seed=7)
    normalizer = SelectiveVecNormalize(
        raw_env,
        norm_obs=True,
        norm_reward=False,
        normalize_obs_excluded_indices=(0,),
    )
    model = _build_algo(
        "SAC",
        normalizer,
        str(source_dir),
        buffer_size=64,
        learning_starts=1_000,
        seed=7,
        **extra_kwargs,
    )
    with torch.no_grad():
        first_parameter = next(model.policy.parameters())
        first_parameter.fill_(0.125)
        model.log_ent_coef.fill_(log_ent_coef)
    model.save(source_dir / "best_model.zip")
    normalizer.save(source_dir / "best_vec_normalize.pkl")
    write_run_config(source_cfg, str(source_dir))
    update_run_config_with_model(model, str(source_dir))
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in model.policy.state_dict().items()
    }
    normalizer.close()
    return source_dir, env_fn, source_state


class _CaptureSacWarmStart(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.policy_state: dict[str, torch.Tensor] | None = None
        self.optimizer_state_counts: tuple[int, int] | None = None
        self.log_ent_coef: float | None = None
        self.start_timestep: int | None = None

    def _on_training_start(self) -> None:
        self.policy_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.policy.state_dict().items()
        }
        self.optimizer_state_counts = (
            len(self.model.policy.actor.optimizer.state),
            len(self.model.policy.critic.optimizer.state),
        )
        self.log_ent_coef = float(self.model.log_ent_coef.detach().item())
        self.start_timestep = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        return True


def test_train_warm_starts_sac_policy_and_entropy(tmp_path):
    source_dir, env_fn, source_state = _make_sac_warm_start_source(tmp_path)
    target_dir = tmp_path / "sac_target"
    capture = _CaptureSacWarmStart()
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="SAC",
        total_timesteps=8,
        log_dir=str(target_dir),
        n_envs=1,
        seed=13,
        eval_freq=10_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        n_eval_episodes=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
        model_kwargs={"buffer_size": 64, "learning_starts": 1_000},
        extra_callbacks=(capture,),
    )
    train(cfg)

    # The full SAC policy transferred: actor, critics, AND critic
    # targets -- a fresh random target network would put the TD
    # bootstrap far from the transferred critics.
    assert capture.policy_state is not None
    assert capture.policy_state.keys() == source_state.keys()
    assert any("critic_target" in name for name in source_state)
    for name, expected in source_state.items():
        torch.testing.assert_close(capture.policy_state[name], expected)
    # Optimizers start stateless; the timestep clock restarts.
    assert capture.optimizer_state_counts == (0, 0)
    assert capture.start_timestep == 0
    # The collapsed auto-entropy temperature carried over instead of
    # restarting at ent_coef=1.0.
    assert capture.log_ent_coef == pytest.approx(-6.5)

    config = json.loads((target_dir / "config.json").read_text())
    initialization = config["initialization"]
    assert initialization["mode"] == "policy_and_observation_stats"
    assert initialization["source"]["algo"] == "SAC"
    assert "log_ent_coef" in initialization["transferred"]
    assert initialization["transferred_ent_coef"] == pytest.approx(
        float(np.exp(-6.5))
    )
    assert "replay_buffer" in initialization["reset"]
    # The default pairing records that the temperature transfer was on.
    assert initialization["transfer_log_ent_coef"] is True
    assert "log_ent_coef" not in initialization["reset"]
    # Both sides carry observation_names: the fingerprint was compared.
    assert initialization["observation_fingerprint"] == "verified"


def test_train_warm_starts_demosac_from_plain_sac_source(tmp_path):
    """The LD1-prime pilot shape: a ``DEMOSAC`` target (injection OFF)
    warm-started from a plain-SAC lineage run. The policy container is
    shared across the SAC family, so the gates admit the pair and the
    transfer is the full SAC transfer (actor, critics, targets,
    log_ent_coef); provenance records the source algo verbatim."""
    source_dir, env_fn, source_state = _make_sac_warm_start_source(tmp_path)
    target_dir = tmp_path / "demosac_target"
    capture = _CaptureSacWarmStart()
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="DEMOSAC",
        total_timesteps=8,
        log_dir=str(target_dir),
        n_envs=1,
        seed=13,
        eval_freq=10_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        n_eval_episodes=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
        model_kwargs={"buffer_size": 64, "learning_starts": 1_000},
        extra_callbacks=(capture,),
    )
    train(cfg)

    assert capture.policy_state is not None
    assert capture.policy_state.keys() == source_state.keys()
    for name, expected in source_state.items():
        torch.testing.assert_close(capture.policy_state[name], expected)
    assert capture.optimizer_state_counts == (0, 0)
    assert capture.start_timestep == 0
    assert capture.log_ent_coef == pytest.approx(-6.5)

    config = json.loads((target_dir / "config.json").read_text())
    assert config["train_config"]["algo"] == "DEMOSAC"
    assert config["resolved_model"]["algo_class"] == "DemoSAC"
    assert config["resolved_model"]["hyperparameters"]["demo_fraction"] == 0.0
    assert "demo_library_sha256" not in config["resolved_model"]
    initialization = config["initialization"]
    assert initialization["mode"] == "policy_and_observation_stats"
    assert initialization["source"]["algo"] == "SAC"
    assert "log_ent_coef" in initialization["transferred"]


def test_prepare_warm_start_refuses_cross_family_source(tmp_path):
    """The family widening admits SAC<->DemoSAC only; a PPO source can
    never seed a SAC-family target (different policy container)."""
    source_dir, env_fn, _ = _make_ppo_warm_start_source(tmp_path)
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="DEMOSAC",
        log_dir=str(tmp_path / "target"),
        n_envs=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
    )
    with pytest.raises(ValueError, match="share its SAC family"):
        _prepare_warm_start(cfg)


def test_train_warm_start_skips_sac_entropy_when_flagged(tmp_path):
    """``transfer_log_ent_coef=False`` -- the temperature-skip warm start:
    the policy still transfers in full, but the target keeps its own
    fresh ``auto_0.02`` init instead of inheriting the source's
    collapsed temperature."""
    source_dir, env_fn, source_state = _make_sac_warm_start_source(tmp_path)
    target_dir = tmp_path / "sac_target_skip"
    capture = _CaptureSacWarmStart()
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="SAC",
        total_timesteps=8,
        log_dir=str(target_dir),
        n_envs=1,
        seed=13,
        eval_freq=10_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        n_eval_episodes=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir, transfer_log_ent_coef=False),
        model_kwargs={
            "buffer_size": 64,
            "learning_starts": 1_000,
            "ent_coef": "auto_0.02",
        },
        extra_callbacks=(capture,),
    )
    train(cfg)

    # Policy transfer is unaffected by the flag.
    assert capture.policy_state is not None
    for name, expected in source_state.items():
        torch.testing.assert_close(capture.policy_state[name], expected)
    # The source's collapsed -6.5 was NOT copied; auto_0.02's init stands.
    assert capture.log_ent_coef == pytest.approx(float(np.log(0.02)))

    config = json.loads((target_dir / "config.json").read_text())
    initialization = config["initialization"]
    assert "log_ent_coef" not in initialization["transferred"]
    assert "log_ent_coef" in initialization["reset"]
    assert initialization["transfer_log_ent_coef"] is False
    assert "transferred_ent_coef" not in initialization
    assert config["train_config"]["warm_start"]["transfer_log_ent_coef"] is False


def test_warm_start_enforces_expected_artifact_sha256(tmp_path):
    """A pinned source artifact is verified by content before any output:
    full digests and >=8-char lowercase prefixes both match; a mismatch
    aborts naming the artifact."""
    source_dir, env_fn, _ = _make_sac_warm_start_source(tmp_path)
    model_sha = hashlib.sha256(
        (source_dir / "best_model.zip").read_bytes()
    ).hexdigest()
    normalizer_sha = hashlib.sha256(
        (source_dir / "best_vec_normalize.pkl").read_bytes()
    ).hexdigest()

    def make_cfg(log_dir, pins):
        return TrainConfig(
            env_fn=env_fn,
            algo="SAC",
            total_timesteps=8,
            log_dir=str(log_dir),
            n_envs=1,
            seed=13,
            eval_freq=10_000,
            checkpoint_freq=0,
            video_freq=0,
            record_video=False,
            info_dict_eval=False,
            n_eval_episodes=1,
            normalize_obs=True,
            normalize_obs_excluded_indices=(0,),
            warm_start=WarmStartConfig(
                source_dir, expected_artifact_sha256=pins
            ),
            model_kwargs={"buffer_size": 64, "learning_starts": 1_000},
        )

    train(
        make_cfg(
            tmp_path / "sac_target_pinned",
            {
                "best_model.zip": model_sha,
                "best_vec_normalize.pkl": normalizer_sha[:12],
            },
        )
    )
    config = json.loads(
        (tmp_path / "sac_target_pinned" / "config.json").read_text()
    )
    recorded = config["train_config"]["warm_start"]["expected_artifact_sha256"]
    assert recorded == {
        "best_model.zip": model_sha,
        "best_vec_normalize.pkl": normalizer_sha[:12],
    }

    with pytest.raises(ValueError, match="does not match its pinned"):
        train(
            make_cfg(
                tmp_path / "sac_target_bad_pin",
                {"best_model.zip": "deadbeef" * 8},
            )
        )


def test_warm_start_config_validates_new_fields(tmp_path):
    with pytest.raises(TypeError, match="transfer_log_ent_coef"):
        WarmStartConfig("some_dir", transfer_log_ent_coef=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown artifact"):
        WarmStartConfig(
            "some_dir", expected_artifact_sha256={"final_model.zip": "a" * 64}
        )
    for bad in ("abc123", "A" * 64, "xyz" * 8, "a" * 65):
        with pytest.raises(ValueError, match="lowercase hex"):
            WarmStartConfig(
                "some_dir", expected_artifact_sha256={"best_model.zip": bad}
            )


def test_warm_start_rejects_algo_mismatch(tmp_path):
    """A SAC target must not silently ingest a PPO source (or vice
    versa) -- the policy classes differ and the transfer would be
    meaningless even where shapes happen to line up."""
    source_dir, env_fn, _source_state = _make_ppo_warm_start_source(
        tmp_path, excluded_indices=(0,)
    )
    cfg = TrainConfig(
        env_fn=env_fn,
        algo="SAC",
        log_dir=str(tmp_path / "mismatch_target"),
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
    )
    with pytest.raises(ValueError, match="source algo must match"):
        _prepare_warm_start(cfg)


def test_prepare_warm_start_resolves_new_layout_source(tmp_path):
    """A 0.14-layout source run (model/best_model.zip) must warm-start
    exactly like a legacy flat run -- the loader resolves both through
    ``locate_artifact``."""
    source_dir, env_fn, _source_state = _make_ppo_warm_start_source(tmp_path)
    model_subdir = source_dir / "model"
    model_subdir.mkdir()
    (source_dir / "best_model.zip").rename(model_subdir / "best_model.zip")
    (source_dir / "best_vec_normalize.pkl").rename(
        model_subdir / "best_vec_normalize.pkl"
    )

    cfg = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        log_dir=str(tmp_path / "target"),
        normalize_obs=True,
        normalize_reward=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
    )
    artifacts = _prepare_warm_start(cfg)
    assert artifacts is not None
    assert artifacts.model_path == (model_subdir / "best_model.zip").resolve()
    assert artifacts.normalizer_path == (
        model_subdir / "best_vec_normalize.pkl"
    ).resolve()
    assert artifacts.config_path == (source_dir / "config.json").resolve()


def test_warm_start_rejects_invalid_source_before_writing_target(tmp_path):
    target_dir = tmp_path / "target"
    # Cadences of whole rollouts (4 workers x SB3's 2,048 steps), which
    # the on-policy pre-flight requires before the warm-start checks.
    whole_rollouts = {"eval_freq": 8_192, "checkpoint_freq": 0}
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="PPO",
        log_dir=str(target_dir),
        warm_start=WarmStartConfig(tmp_path / "missing"),
        **whole_rollouts,
    )
    with pytest.raises(ValueError, match="does not exist"):
        train(cfg)
    assert not target_dir.exists()

    source_dir, env_fn, _source_state = _make_ppo_warm_start_source(tmp_path)
    mismatched_dir = tmp_path / "mismatched"
    mismatch = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        log_dir=str(mismatched_dir),
        normalize_obs_excluded_indices=(1,),
        warm_start=WarmStartConfig(source_dir),
        **whole_rollouts,
    )
    with pytest.raises(ValueError, match="excluded_indices differ"):
        train(mismatch)
    assert not mismatched_dir.exists()


def test_warm_start_setup_failure_closes_every_constructed_env(tmp_path):
    source_dir, _source_env_fn, _source_state = _make_ppo_warm_start_source(
        tmp_path
    )
    source_config_path = source_dir / "config.json"
    source_config = json.loads(source_config_path.read_text())
    # Let cheap config validation pass, while leaving the serialized
    # normalizer's real clip_obs=10 so loading fails after train/eval envs exist.
    source_config["train_config"]["clip_obs"] = 5.0
    source_config_path.write_text(json.dumps(source_config))

    constructed: set[int] = set()
    closed: set[int] = set()

    def tracked_env_fn():
        env = BallBalanceEnv(episode_len=12)
        identity = id(env)
        constructed.add(identity)
        original_close = env.close

        def tracked_close():
            closed.add(identity)
            original_close()

        env.close = tracked_close
        return env

    cfg = TrainConfig(
        env_fn=tracked_env_fn,
        algo="PPO",
        log_dir=str(tmp_path / "target"),
        n_envs=1,
        normalize_obs=True,
        normalize_reward=True,
        clip_obs=5.0,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
        # Whole rollouts of SB3's 2,048 steps (the on-policy pre-flight).
        eval_freq=2_048,
        checkpoint_freq=0,
    )
    with pytest.raises(ValueError, match="normalizer clip_obs settings differ"):
        train(cfg)

    assert constructed
    assert constructed <= closed


def test_warm_start_supports_ppo_and_sac_family_only(tmp_path):
    """The transfer path is written for PPO and the SAC family this
    project trains (SAC and its DemoSAC subclass share the policy
    container); any future algorithm outside them must extend it
    explicitly rather than falling through half-supported."""
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="TD3",
        log_dir=str(tmp_path / "target"),
        warm_start=WarmStartConfig(tmp_path),
    )
    with pytest.raises(ValueError, match="PPO and the SAC family"):
        _prepare_warm_start(cfg)


def test_reward_eval_episodes_requires_headline_selection(tmp_path):
    """Trimming the reward eval stream is only legal when it is
    reporting-only; without headline selection that stream owns
    best-model selection and must keep the full episode count."""
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=12),
        algo="PPO",
        total_timesteps=8,
        log_dir=str(tmp_path / "target"),
        info_dict_eval=False,
        reward_eval_episodes=5,
        model_kwargs={"n_steps": 8, "batch_size": 4, "n_epochs": 1},
    )
    with pytest.raises(ValueError, match="headline-metric selection"):
        train(cfg)


def test_run_summary_surfaces_headline_metric(tmp_path):
    """With ``headline_key`` set, the stage summary reports the metric's
    ``_ep_mean`` series: its last value and its own best (which need not
    coincide with the best-reward checkpoint). Eval reward is dominated
    by shaping on WallBall, so this is the line runs are compared on."""
    from courtside_dynamics.training.artifacts import write_run_summary

    (tmp_path / "eval_info.csv").write_text(
        "timestep,metric,value\n"
        "25000,bounce_count_ep_mean,0.0\n"
        "50000,bounce_count_ep_mean,2.86\n"
        "75000,bounce_count_ep_mean,2.14\n"
        "75000,bounce_count_ep_ge_2_rate,0.4\n"
        "75000,bounce_count_ep_ge_3_rate,0.1\n"
        "75000,bounce_count_ep_ge_5_rate,0.0\n"
    )

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(
        env_fn=env_fn,
        log_dir=str(tmp_path),
        headline_key="bounce_count",
        info_eval_survival_thresholds={"bounce_count": (2, 3, 5)},
    )
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    assert "bounce_count_ep_mean" in text
    assert "Headline final: 2.14" in text
    assert "2.86 (at 50,000 steps)" in text
    assert "Survival final:" in text
    assert ">=2 40.0%" in text
    assert ">=3 10.0%" in text


def test_run_summary_skips_headline_without_data(tmp_path):
    """A headline key with no matching eval_info rows (typo, or a run
    that died before the first eval) must not emit headline lines or
    crash the report."""
    from courtside_dynamics.training.artifacts import write_run_summary

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(
        env_fn=env_fn,
        log_dir=str(tmp_path),
        headline_key="bounce_count",
    )
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    assert "Headline" not in text


def test_run_summary_reports_task_metric_selected_best_model(tmp_path):
    """With ``best_model_meta.json`` present (task-metric selection), the
    summary names the selected step and keys the best-checkpoint section
    to it -- not to the reward-argmax step, which can be a different
    (and, per run 20260712_190054, degenerate) checkpoint."""
    import json

    import numpy as np

    from courtside_dynamics.training.artifacts import write_run_summary

    # Reward series peaks at 75k...
    np.savez(
        tmp_path / "evaluations.npz",
        timesteps=np.array([25_000, 50_000, 75_000]),
        results=np.array([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]),
        ep_lengths=np.array([[30, 30], [30, 30], [30, 30]]),
    )
    # ...but the task metric selected 50k.
    (tmp_path / "best_model_meta.json").write_text(
        json.dumps(
            {
                "timestep": 50_000,
                "selection_keys": [
                    "bounce_count_ep_mean",
                    "episode_reward_mean",
                ],
                "selection_values": {
                    "bounce_count_ep_mean": 3.2,
                    "episode_reward_mean": 1.0,
                },
            }
        )
    )
    (tmp_path / "eval_info.csv").write_text(
        "timestep,metric,value\n"
        "50000,bounce_count_ep_mean,3.2\n"
        "50000,bounce_count_final,3.0\n"
        "50000,episode_reward_mean,0.8\n"
        "50000,bounce_count_ep_ge_2_rate,0.8\n"
        "50000,bounce_count_ep_ge_3_rate,0.25\n"
        "50000,bounce_count_ep_ge_5_rate,0.0\n"
        "75000,bounce_count_ep_mean,0.0\n"
    )

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(
        env_fn=env_fn,
        log_dir=str(tmp_path),
        headline_key="bounce_count",
        info_eval_survival_thresholds={"bounce_count": (2, 3, 5)},
    )
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    assert "Best model" in text
    assert "step 50,000 (bounce_count_ep_mean 3.20)" in text
    assert "[task-metric selection]" in text
    # The best-checkpoint section describes the *selected* step. This
    # meta predates the ``metrics`` block, so the selecting batch is
    # the selecting evaluator's own eval_info.csv row at that step --
    # its reward (0.800), not the reward stream's 1.000 at 50k (other
    # episodes) nor the 75k argmax.
    assert "Best Checkpoint Evaluation (step 50,000)" in text
    best_block = text.split("Best Checkpoint Evaluation", 1)[1]
    assert "Reward:         0.800  [selecting batch]" in best_block
    assert "1.000 +/- 0.000" not in best_block
    assert "bounce_count: ep-mean 3.20  last-episode 3.00" in best_block
    assert "Return survival:" in text
    assert ">=2 80.0%" in text
    assert ">=3 25.0%" in text
    # The reward-series best line is still reported for context.
    assert "2.000 +/- 0.000 (at 75,000 steps)" in text


def test_run_summary_labels_each_evaluation_instrument(tmp_path):
    """Every evaluation line must name the instrument it reports: run
    20260821_013700 left three different "final eval" numbers (closing
    eval, evaluations.npz row, eval_info.csv row) that a reader could
    not tell apart."""
    from courtside_dynamics.training.artifacts import write_run_summary

    np.savez(
        tmp_path / "evaluations.npz",
        timesteps=np.array([25_000, 50_000]),
        results=np.array([[0.5, 0.5], [2.0, 2.0]]),
        ep_lengths=np.array([[30, 30], [30, 30]]),
    )
    (tmp_path / "eval_info.csv").write_text(
        "timestep,metric,value\n"
        "25000,bounce_count_ep_mean,2.86\n"
        "50000,bounce_count_ep_mean,2.14\n"
    )
    monitor_dir = tmp_path / "metrics" / "monitor"
    monitor_dir.mkdir(parents=True)
    (monitor_dir / "0.monitor.csv").write_text(
        '#{"t_start": 0.0}\nr,l,t\n1.0,10,1.0\n2.0,10,2.0\n'
    )

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(
        env_fn=env_fn,
        log_dir=str(tmp_path),
        headline_key="bounce_count",
    )
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=0.976,
        final_std_reward=1.626,
        duration_seconds=10.0,
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    # Closing eval: train()'s epilogue evaluate_policy pass, distinct
    # from both periodic series.
    assert (
        "Final eval:     0.976 +/- 1.626  [closing eval, fresh episodes]"
        in text
    )
    # EvalCallback's reward series (evaluations.npz).
    assert "2.000 +/- 0.000 (at 50,000 steps)  [periodic eval series]" in text
    # InfoDictEvalCallback's series (eval_info.csv), on both lines.
    assert "Headline final: 2.14  [eval_info series]" in text
    assert "2.86 (at 25,000 steps)  [eval_info series]" in text
    # Training-episode rewards from the train env's Monitor logs.
    assert "1.500 +/- 0.707 (last 2 episodes)  [train monitor logs]" in text


class _StepsAlive(gym.Wrapper):
    """BallBalance plus a ``steps_alive`` info counter.

    BallBalance's own ``info`` is empty, so a headline key on it names
    a metric no evaluation emits -- which ``train()`` now rejects
    (review §2.1). The counter gives the headline-selection tests a
    real task metric to key on.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps_alive = 0
        return obs, {**info, "steps_alive": 0}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps_alive += 1
        return (
            obs,
            reward,
            terminated,
            truncated,
            {**info, "steps_alive": self._steps_alive},
        )


def _steps_alive_env(episode_len: int = 12) -> gym.Env:
    return _StepsAlive(BallBalanceEnv(episode_len=episode_len))


def _merged_eval_cfg(tmp_path, **overrides):
    """A tiny headline-selection run with the final-config eval stream on."""
    cfg_kwargs = dict(
        env_fn=_steps_alive_env,
        algo="SAC",
        total_timesteps=600,
        log_dir=str(tmp_path),
        n_envs=1,
        seed=0,
        eval_freq=200,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        normalize_obs=False,
        n_eval_episodes=2,
        info_dict_eval=True,
        headline_key="steps_alive",
        final_info_eval=True,
        model_kwargs={"learning_starts": 16, "buffer_size": 500},
    )
    cfg_kwargs.update(overrides)
    return TrainConfig(**cfg_kwargs)


def test_stop_reason_lands_in_the_stage_summary(tmp_path):
    """End-to-end pin for the stop_reason glue inside train(): the
    callback ends training with a reason, and the summary must carry it.
    The two ends (callbacks setting stop_reason; write_run_summary
    rendering one) are pinned elsewhere -- this covers the extraction
    scan in between, whose silent failure would quietly return every
    stopped-early summary to a reason-less '(stopped early)'."""

    class _StopWithReason(BaseCallback):
        def __init__(self):
            super().__init__()
            self.stop_reason: str | None = None

        def _on_step(self) -> bool:
            if self.num_timesteps >= 64:
                self.stop_reason = (
                    "test_guard: synthetic stop for the glue test"
                )
                return False
            return True

    train(
        _merged_eval_cfg(
            tmp_path, extra_callbacks=(_StopWithReason(),)
        )
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    assert "Stop reason" in text
    assert "test_guard: synthetic stop for the glue test" in text


def test_completed_final_info_eval_run_closes_every_constructed_env(
    tmp_path,
):
    """Success-path sibling of the warm-start close test: the inner
    cleanup used to ``opened_envs.clear()`` before the outer finally
    ran, so the final_info_eval base env leaked its MuJoCo resources on
    every completed run."""
    constructed: set[int] = set()
    closed: set[int] = set()

    def tracked_env_fn():
        env = _steps_alive_env()
        identity = id(env)
        constructed.add(identity)
        original_close = env.close

        def tracked_close():
            closed.add(identity)
            original_close()

        env.close = tracked_close
        return env

    train(_merged_eval_cfg(tmp_path, env_fn=tracked_env_fn))
    assert constructed
    assert constructed <= closed


def test_final_info_eval_owns_evaluations_npz_when_reward_stream_retired(
    tmp_path,
):
    """One stream, one rollout, same artifact.

    The reward EvalCallback and the final-config info-eval roll the SAME
    distribution (the recipe's eval_env_overrides), and under headline
    selection the reward stream is reporting-only. Retiring it must not
    cost the ``evaluations.npz`` artifact every downstream reader expects.
    """
    import json

    import numpy as np

    train(_merged_eval_cfg(tmp_path))

    payload = np.load(tmp_path / "metrics" / "evaluations.npz")
    assert set(payload.files) >= {"timesteps", "results", "ep_lengths"}
    # Rectangular: one row per evaluation, one column per episode.
    assert payload["results"].ndim == 2
    assert payload["results"].shape[0] == payload["timesteps"].shape[0]
    assert payload["results"].shape == payload["ep_lengths"].shape
    assert payload["results"].shape[0] >= 1
    # A merged stream is the only one scoring the goal task, so it gets
    # the FULL n_eval_episodes -- not the // 2 reporting sample the split
    # streams used.
    assert payload["results"].shape[1] == 2
    assert (payload["ep_lengths"] > 0).all()

    # Both stream sizes are now part of the run's provenance snapshot.
    config = json.load(open(tmp_path / "config.json"))
    assert "reward_eval_episodes" in config["train_config"]
    assert "final_eval_episodes" in config["train_config"]


def test_final_eval_episodes_sizes_the_merged_stream(tmp_path):
    import numpy as np

    train(_merged_eval_cfg(tmp_path, final_eval_episodes=3))

    payload = np.load(tmp_path / "metrics" / "evaluations.npz")
    assert payload["results"].shape[1] == 3


def test_evaluations_npz_stays_readable_by_the_learning_plots(tmp_path):
    """notebook_utils reads this artifact by name; keep the contract."""
    from courtside_dynamics.notebook_utils import locate_artifact

    train(_merged_eval_cfg(tmp_path))
    assert locate_artifact(tmp_path, "evaluations") is not None


def test_final_eval_episodes_requires_the_final_info_eval_stream(tmp_path):
    cfg = _merged_eval_cfg(
        tmp_path, final_info_eval=False, final_eval_episodes=4
    )
    with pytest.raises(ValueError, match="requires info_dict_eval and"):
        train(cfg)


def test_final_eval_episodes_rejects_non_positive(tmp_path):
    cfg = _merged_eval_cfg(tmp_path, final_eval_episodes=0)
    with pytest.raises(
        ValueError, match="final_eval_episodes must be a positive integer"
    ):
        train(cfg)


def test_reward_eval_stream_survives_without_the_final_info_eval(tmp_path):
    """Nothing to merge into: EvalCallback keeps owning evaluations.npz."""
    import numpy as np

    train(
        _merged_eval_cfg(
            tmp_path, final_info_eval=False, reward_eval_episodes=1
        )
    )
    payload = np.load(tmp_path / "metrics" / "evaluations.npz")
    assert payload["results"].shape[1] == 1


def test_merged_stream_gets_the_full_episode_budget(tmp_path):
    """A merged stream is sized by n_eval_episodes, not the // 2 sample.

    Mirrors the depth recipe's shape (n_eval_episodes 30 with
    reward_eval_episodes 5): the retired reward stream's small budget must
    NOT cap the surviving stream, because that stream is then the only one
    scoring the campaign's goal task. The old split sizing would have
    given max(n // 2, reward_eval_episodes) = 2 here.
    """
    import numpy as np

    train(
        _merged_eval_cfg(
            tmp_path,
            n_eval_episodes=4,
            reward_eval_episodes=1,
            total_timesteps=400,
            eval_freq=200,
        )
    )

    payload = np.load(tmp_path / "metrics" / "evaluations.npz")
    assert payload["results"].shape[1] == 4


def test_eval_verbose_prints_one_line_per_evaluation(tmp_path, capsys):
    """`eval_verbose` restores periodic progress without SB3's table.

    Retiring the reward EvalCallback left a run silent between stage
    advances, so a multi-hour job showed nothing. The heartbeat must
    carry the numbers a watcher actually needs (reward, episode length)
    and must NOT drag in SB3's per-rollout table, which SAC emits every
    log_interval (4) episodes.
    """
    train(
        _merged_eval_cfg(
            tmp_path,
            total_timesteps=400,
            eval_freq=200,
            eval_verbose=1,
            verbose=0,
        )
    )

    captured = capsys.readouterr().out
    # Both streams report, and the log_prefix keeps them apart.
    assert "[eval_info]" in captured
    assert "[eval_info_final]" in captured
    assert "reward=" in captured
    assert "len=" in captured
    # One line per evaluation per stream, not per rollout.
    assert captured.count("[eval_info]") == 2
    # SB3's rollout table stays off: that is the whole point of the split.
    assert "rollout/" not in captured


def test_eval_verbose_defaults_to_the_algorithm_verbosity(tmp_path, capsys):
    """``None`` follows ``verbose``, so existing runs stay silent."""
    train(
        _merged_eval_cfg(
            tmp_path,
            total_timesteps=400,
            eval_freq=200,
            verbose=0,
        )
    )

    assert "[eval_info]" not in capsys.readouterr().out


def test_no_merge_without_headline_selection(tmp_path):
    """Without headline selection the reward stream owns selection.

    It must keep running and keep writing evaluations.npz at the full
    n_eval_episodes, even with the final-config stream also attached.
    """
    import numpy as np

    cfg = _merged_eval_cfg(
        tmp_path,
        n_eval_episodes=4,
        headline_key=None,  # no headline selection -> no merge
        total_timesteps=400,
        eval_freq=200,
    )
    train(cfg)

    payload = np.load(tmp_path / "metrics" / "evaluations.npz")
    assert payload["results"].shape[1] == 4


def test_config_json_records_every_train_config_data_field(tmp_path):
    """``config.json``'s ``train_config`` block is hand-maintained.

    A new ``TrainConfig`` field is therefore silently absent from every
    run's provenance snapshot until someone remembers to add it --
    ``reward_eval_episodes`` was missing for its whole life, so a run's
    artifacts could not say whether its reward stream rolled 5 episodes or
    30. Pin the coverage so the next field cannot drift the same way.
    """
    import dataclasses

    # The only fields deliberately absent from the block.
    code_valued = {"env_fn", "eval_env_fn", "extra_callbacks", "info_row_fn"}
    recorded_at_top_level = {"recipe_name", "run_config_file"}

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        log_dir=str(tmp_path),
    )
    write_run_config(cfg, str(tmp_path))
    payload = json.loads((tmp_path / "config.json").read_text())

    expected = {
        field.name
        for field in dataclasses.fields(TrainConfig)
    } - code_valued - recorded_at_top_level
    recorded = set(payload["train_config"])

    missing = expected - recorded
    assert not missing, (
        f"TrainConfig fields absent from config.json's train_config block: "
        f"{sorted(missing)}. Add them to artifacts.write_run_config, or to "
        f"this test's exclusion sets with a reason."
    )
    # Nothing derived should be smuggled in either -- the block should be
    # exactly the run's configuration.
    assert not recorded - expected, sorted(recorded - expected)
    # The excluded-but-real fields must still be recorded somewhere.
    for name in recorded_at_top_level:
        assert name in payload


def test_config_json_gate_block_records_every_gate_key(tmp_path):
    """The nested ``performance_gate`` block is hand-maintained too.

    The 0.24.0 ``stage_eval_budget`` pair was accepted by ``train()``
    but absent from this block for two releases, so a run stopped by
    the staleness guard had a ``config.json`` showing no guard was
    configured. Pin the block to ``PERFORMANCE_GATE_KEYS`` so the next
    gate lever cannot drift the same way.
    """
    from courtside_dynamics.training.train import PERFORMANCE_GATE_KEYS

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        log_dir=str(tmp_path),
        performance_gate={
            "metric_key": "m",
            "threshold": 1.0,
            "sustain_evals": 1,
            "stages": ({"episode_len": 8},),
        },
    )
    write_run_config(cfg, str(tmp_path))
    payload = json.loads((tmp_path / "config.json").read_text())
    gate_block = payload["train_config"]["performance_gate"]
    assert set(gate_block) == set(PERFORMANCE_GATE_KEYS)
    # Unset optional levers are recorded at train()'s resolution
    # defaults, so the block reads as the gate actually ran.
    assert gate_block["stage_eval_budget"] is None
    assert gate_block["stage_eval_budget_action"] == "stop"


# ---------------------------------------------------------------------------
# Robustness closures from docs/rl_pipeline_review_20260828.md (after-LT1)
# ---------------------------------------------------------------------------


def _unbuildable_env():
    raise AssertionError("train() built an env before validating its config")


class _RaiseAt(BaseCallback):
    """Raise ``exc`` out of ``model.learn()`` once ``at`` steps are done."""

    def __init__(self, exc: BaseException | type[BaseException], at: int = 64):
        super().__init__()
        self.exc = exc
        self.at = at

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.at:
            raise self.exc
        return True


def _salvage_cfg(tmp_path, **overrides) -> TrainConfig:
    """A short VecNormalize'd SAC run with no periodic evaluation."""
    cfg_kwargs = dict(
        env_fn=lambda: BallBalanceEnv(episode_len=12),
        algo="SAC",
        total_timesteps=2_000,
        log_dir=str(tmp_path),
        n_envs=1,
        seed=0,
        eval_freq=100_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        normalize_obs=True,
        n_eval_episodes=1,
        model_kwargs={"learning_starts": 16, "buffer_size": 500},
    )
    cfg_kwargs.update(overrides)
    return TrainConfig(**cfg_kwargs)


def _summary_field(tmp_path, label: str) -> str:
    text = (tmp_path / "stage_summary.txt").read_text()
    line = next(line for line in text.splitlines() if line.startswith(label))
    return line.split(":", 1)[1].strip()


def test_keyboard_interrupt_salvages_the_run(tmp_path):
    """Review §2.13: the salvage protecting ~20-hour Colab runs had no
    test. An interrupt mid-learn must still leave the final model, its
    normalizer, and a summary marked ``interrupted``, and return."""
    from stable_baselines3 import SAC

    from courtside_dynamics.training.artifacts import artifact_path

    cfg = _salvage_cfg(tmp_path, extra_callbacks=(_RaiseAt(KeyboardInterrupt),))
    try:
        model = train(cfg)
    except KeyboardInterrupt:  # pragma: no cover - the regression itself
        pytest.fail("KeyboardInterrupt escaped train()'s salvage path")

    assert 64 <= model.num_timesteps < cfg.total_timesteps
    final_model = artifact_path(str(tmp_path), "final_model")
    assert SAC.load(final_model).num_timesteps == model.num_timesteps
    venv = make_vec_env(lambda: BallBalanceEnv(episode_len=12), n_envs=1)
    try:
        VecNormalize.load(artifact_path(str(tmp_path), "vec_normalize"), venv)
    finally:
        venv.close()
    assert _summary_field(tmp_path, "Status") == "interrupted"
    # The closing evaluation still runs on an interrupt.
    assert "closing eval" in _summary_field(tmp_path, "Final eval")


def test_non_keyboard_interrupt_crash_salvages_then_reraises(tmp_path, capsys):
    """Review §2.2/§3: any other exception out of ``learn()`` used to skip
    the whole epilogue. It must now save the model and normalizer, mark
    the summary ``crashed`` with the error, and still re-raise."""
    from stable_baselines3 import SAC

    from courtside_dynamics.training.artifacts import artifact_path

    cfg = _salvage_cfg(
        tmp_path,
        extra_callbacks=(_RaiseAt(RuntimeError("synthetic crash")),),
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        train(cfg)

    assert "Training crashed" in capsys.readouterr().out
    loaded = SAC.load(artifact_path(str(tmp_path), "final_model"))
    assert 64 <= loaded.num_timesteps < cfg.total_timesteps
    venv = make_vec_env(lambda: BallBalanceEnv(episode_len=12), n_envs=1)
    try:
        VecNormalize.load(artifact_path(str(tmp_path), "vec_normalize"), venv)
    finally:
        venv.close()
    assert _summary_field(tmp_path, "Status") == "crashed"
    assert "synthetic crash" in _summary_field(tmp_path, "Stop reason")
    # No closing evaluation: the env may be what broke.
    assert _summary_field(tmp_path, "Final eval") == "not run (run crashed)"


def test_crash_salvage_failure_never_masks_the_original_error(
    tmp_path, monkeypatch, capsys
):
    """A salvage step that itself fails (the Drive mount that caused the
    crash, say) is printed and skipped; the caller still sees the
    original exception, and the earlier salvage steps still land."""
    import importlib

    from courtside_dynamics.training.artifacts import artifact_path

    train_module = importlib.import_module("courtside_dynamics.training.train")

    def _broken_summary(*args, **kwargs):
        raise OSError("drive went away")

    monkeypatch.setattr(train_module, "write_run_summary", _broken_summary)
    cfg = _salvage_cfg(
        tmp_path,
        extra_callbacks=(_RaiseAt(RuntimeError("synthetic crash")),),
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        train(cfg)
    assert "could not salvage stage_summary" in capsys.readouterr().out
    assert (tmp_path / "model" / "final_model.zip").is_file()
    assert os.path.isfile(artifact_path(str(tmp_path), "vec_normalize"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"headline_key": "steps_alvie"},
            r"headline_key 'steps_alvie': it is not a scalar info key .* "
            r"\(did you mean 'steps_alive'\?\)",
        ),
        (
            {"best_metric_keys": ("steps_alive_ep_maen",)},
            r"best_metric_keys 'steps_alive_ep_maen': no evaluation emits .*"
            r"\(did you mean 'steps_alive_ep_mean'\?\)",
        ),
        (
            {"success_key": "steps_alvie"},
            r"success_key 'steps_alvie': it is not a scalar info key",
        ),
        (
            {
                "early_stop_degenerate_evals": 2,
                "degenerate_guard_keys": ("paddle_hit_count_ep_mean",),
            },
            r"degenerate_guard_keys 'paddle_hit_count_ep_mean'",
        ),
        (
            {"info_eval_keys": ("unrelated",)},
            r"headline_key 'steps_alive': the eval env emits it, but "
            r"info_eval_keys filters it out",
        ),
    ],
)
def test_train_rejects_metric_keys_no_evaluation_produces(
    tmp_path, overrides, message
):
    """Review §2.1: a miskeyed selection metric scored -inf at every
    evaluation, so a typo trained normally while selection silently fell
    through to reward. train() must refuse it before any training step."""
    cfg = _merged_eval_cfg(tmp_path, final_info_eval=False, **overrides)
    with pytest.raises(ValueError, match=message):
        train(cfg)
    assert not (tmp_path / "model" / "final_model.zip").exists()
    assert not (tmp_path / "metrics" / "eval_info.csv").exists()


def test_train_accepts_the_gate_stage_index_as_a_selection_key(tmp_path):
    """Context metrics a performance gate stamps at training start are
    producible even though no env info key carries them."""
    cfg = _merged_eval_cfg(
        tmp_path,
        final_info_eval=False,
        total_timesteps=200,
        best_metric_keys=("steps_alive_ep_mean", "curriculum_stage_index"),
        performance_gate={
            "metric_key": "steps_alive_ep_mean",
            "threshold": 1e9,
            "sustain_evals": 1,
            "stages": ({"episode_len": 12},),
        },
    )
    train(cfg)
    meta = json.loads((tmp_path / "model" / "best_model_meta.json").read_text())
    assert "curriculum_stage_index" in meta["selection_values"]


def test_train_validates_model_kwargs_before_any_setup(tmp_path):
    """Review §3: only the recipe path validated model_kwargs, so a direct
    TrainConfig with a cross-algorithm key crashed inside SB3 after the
    env fleet was built and artifacts written."""
    log_dir = tmp_path / "run"
    cfg = TrainConfig(
        env_fn=_unbuildable_env,
        algo="PPO",
        log_dir=str(log_dir),
        model_kwargs={"buffer_size": 1_000},
    )
    with pytest.raises(ValueError, match="buffer_size"):
        train(cfg)
    assert not log_dir.exists()


def test_require_device_cuda_fails_fast_without_cuda(tmp_path, monkeypatch):
    """Review §2.3: SB3's device='auto' silently resolved to CPU on two LT1
    launches. With require_device='cuda' the run must stop before any env
    is built or the run directory created."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    log_dir = tmp_path / "run"
    cfg = TrainConfig(
        env_fn=_unbuildable_env, log_dir=str(log_dir), require_device="cuda"
    )
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\)"):
        train(cfg)
    assert not log_dir.exists()


def test_require_device_rejects_bad_values_and_contradictions(
    tmp_path, monkeypatch
):
    from courtside_dynamics.training.train import _check_required_device

    with pytest.raises(ValueError, match="require_device must be None or"):
        _check_required_device(
            TrainConfig(env_fn=_unbuildable_env, require_device="gpu")
        )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="contradicts require_device"):
        _check_required_device(
            TrainConfig(
                env_fn=_unbuildable_env,
                require_device="cuda",
                model_kwargs={"device": "cpu"},
            )
        )
    for device in ("auto", "cuda", "cuda:0"):
        _check_required_device(
            TrainConfig(
                env_fn=_unbuildable_env,
                require_device="cuda",
                model_kwargs={"device": device},
            )
        )
    # Opt-in: the default never touches torch.cuda.
    monkeypatch.setattr(torch.cuda, "is_available", _unbuildable_env)
    _check_required_device(TrainConfig(env_fn=_unbuildable_env))


def test_check_model_device_asserts_the_built_model_placement():
    from types import SimpleNamespace

    from courtside_dynamics.training.train import _check_model_device

    on_cpu: Any = SimpleNamespace(device=torch.device("cpu"))
    on_cuda: Any = SimpleNamespace(device=torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match="built on cpu"):
        _check_model_device(on_cpu, "cuda")
    _check_model_device(on_cuda, "cuda")
    _check_model_device(on_cpu, None)


def test_config_updaters_warn_loudly_on_an_unreadable_config(tmp_path, capsys):
    """Review §2.5: the provenance writers returned None with no output
    on an unreadable config.json, silently dropping the block."""
    from types import SimpleNamespace

    config = tmp_path / "config.json"
    config.write_text('{"train_config": ')  # truncated mid-write
    assert update_run_config_with_model(SimpleNamespace(), str(tmp_path)) is None
    out = capsys.readouterr().out
    assert "[artifacts]" in out
    assert str(config) in out
    assert "JSONDecodeError" in out
    assert "'resolved_model'" in out


def test_config_updaters_refuse_to_drop_pinned_provenance(tmp_path):
    """Pinned digests are what a frozen plan validates; losing them must
    stop the run at start, not surface post hoc as a plan mismatch."""
    from types import SimpleNamespace

    from courtside_dynamics.training.artifacts import (
        update_run_config_with_initialization,
    )

    (tmp_path / "config.json").write_text('{"train_config": ')
    demo_model = SimpleNamespace(demo_library_sha256="ab" * 32)
    with pytest.raises(RuntimeError, match="demo_library_sha256"):
        update_run_config_with_model(demo_model, str(tmp_path))
    with pytest.raises(RuntimeError, match="initialization.source_artifacts"):
        update_run_config_with_initialization({"mode": "x"}, str(tmp_path))


def test_unreadable_best_model_meta_is_reported(tmp_path, capsys):
    from courtside_dynamics.training.artifacts import _read_best_model_meta

    meta = tmp_path / "model" / "best_model_meta.json"
    meta.parent.mkdir()
    meta.write_text("{")
    assert _read_best_model_meta(str(tmp_path)) is None
    out = capsys.readouterr().out
    assert "[artifacts]" in out and str(meta) in out


# ---------------------------------------------------------------------------
# 2026-10-05 fix batch (docs/repo_review_20261005.md §5b, §7.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"best_metric_min_delta": {"success_rate": 0.05}},
            ValueError,
            "requires headline-metric selection",
        ),
        (
            {
                "headline_key": "steps_alive",
                "best_metric_min_delta": {"crossings_ep_mean": 0.25},
            },
            ValueError,
            r"\['crossings_ep_mean'\], which are not selection keys",
        ),
        (
            {
                "headline_key": "steps_alive",
                "best_metric_min_delta": {"steps_alive_ep_mean": True},
            },
            TypeError,
            "must be a number",
        ),
        (
            {"headline_key": "steps_alive", "best_metric_min_delta": -0.5},
            ValueError,
            "finite and nonnegative",
        ),
        (
            # No success_key: success_rate is not a resolved selection key.
            {
                "headline_key": "steps_alive",
                "degenerate_flat_keys": ("success_rate",),
            },
            ValueError,
            r"\['success_rate'\] are not selection keys",
        ),
        (
            {"degenerate_flat_keys": ("episode_reward_mean",)},
            ValueError,
            "requires headline-metric selection",
        ),
        (
            {
                "headline_key": "steps_alive",
                "degenerate_flat_keys": "episode_reward_mean",
            },
            TypeError,
            "sequence of strings",
        ),
        (
            {"headline_key": "steps_alive", "degenerate_flat_keys": ()},
            ValueError,
            "at least one",
        ),
        ({"eval_seed": True}, TypeError, "eval_seed must be an integer"),
        ({"eval_seed": 2.0}, TypeError, "eval_seed must be an integer"),
        ({"eval_seed": -3}, ValueError, "eval_seed must be nonnegative"),
        (
            # Unseeded and no eval_seed: nothing would apply the options.
            {"eval_reset_options": ({"serve_side": "a"},)},
            ValueError,
            "requires paired evaluation",
        ),
        (
            {"seed": 0, "eval_reset_options": {"serve_side": "a"}},
            TypeError,
            "sequence of option mappings",
        ),
        (
            {
                "seed": 0,
                "info_dict_eval": False,
                "eval_reset_options": ({"serve_side": "a"},),
            },
            ValueError,
            "requires info_dict_eval",
        ),
        ({"monitor_info_keywords": "steps_alive"}, TypeError, "strings"),
        ({"reuse_log_dir": 1}, TypeError, "reuse_log_dir must be a bool"),
    ],
)
def test_train_rejects_bad_evaluation_settings_before_any_setup(
    tmp_path, overrides, error, message
):
    """Contract C3 / T10: the new selection, pairing and monitor fields
    fail loudly -- before any env is built or the run dir created."""
    log_dir = tmp_path / "run"
    cfg = TrainConfig(env_fn=_unbuildable_env, log_dir=str(log_dir), **overrides)
    with pytest.raises(error, match=message):
        train(cfg)
    assert not log_dir.exists()


def test_train_wires_selection_and_paired_evaluation(tmp_path, monkeypatch):
    """The per-key delta, the flatness subset and the paired seeds reach
    the evaluators train() builds, and config.json records them: the
    configured values in ``train_config``, the resolved seeds in
    ``evaluation_seeding``."""
    from courtside_dynamics.callbacks.info_dict_eval import (
        CONFIRMATION_SEED_OFFSET,
        InfoDictEvalCallback,
    )

    # The package re-exports the train() function under the module's
    # name, so resolve the module itself.
    train_module = importlib.import_module("courtside_dynamics.training.train")
    built: list[InfoDictEvalCallback] = []

    class _Recording(InfoDictEvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(train_module, "InfoDictEvalCallback", _Recording)
    delta = {"steps_alive_ep_mean": 0.5, "episode_reward_mean": 0.25}
    options = ({"probe": "a"}, {"probe": "b"})
    cfg = _merged_eval_cfg(
        tmp_path,
        total_timesteps=200,
        # The real eval wrapper stack: the paired seeds and options pass
        # through SelectiveVecNormalize(training=False).
        normalize_obs=True,
        best_metric_min_delta=delta,
        degenerate_flat_keys=("steps_alive_ep_mean",),
        eval_reset_options=options,
    )
    train(cfg)

    selection, final = built
    assert selection.best_metric_keys == (
        "steps_alive_ep_mean",
        "episode_reward_mean",
    )
    assert selection._min_deltas == (0.5, 0.25)
    assert selection.degenerate_flat_keys == ("steps_alive_ep_mean",)
    # seed=0, eval_seed=None and reset options set: the derived block,
    # seed + EVAL_SEED_OFFSET. Literal values throughout, so a changed
    # offset constant cannot pass by construction.
    assert selection.eval_seed == 1_000_000
    assert selection.eval_reset_options == options
    # The final-config stream stays fresh-random: no seed block, and so
    # no reset options either (they only ride on paired resets).
    assert final.eval_seed is None
    assert final.eval_reset_options is None
    # The selection batch and its confirmation batch own disjoint seed
    # ranges.
    selection_block = set(
        range(selection.eval_seed, selection.eval_seed + selection.n_eval_episodes)
    )
    confirmation_start = selection.eval_seed + CONFIRMATION_SEED_OFFSET
    assert not selection_block & set(
        range(confirmation_start, confirmation_start + selection.n_eval_episodes)
    )

    config = json.loads((tmp_path / "config.json").read_text())
    recorded = config["train_config"]
    assert recorded["best_metric_min_delta"] == delta
    assert recorded["degenerate_flat_keys"] == ["steps_alive_ep_mean"]
    assert recorded["eval_seed"] is None
    assert recorded["eval_reset_options"] == [dict(o) for o in options]
    # Exactly the streams that ran: confirm_best is off (no confirmation
    # block) and the reward stream was merged into the final stream.
    assert config["evaluation_seeding"] == {
        "paired": True,
        "eval_seed": 1_000_000,
        "derived_from_seed": True,
        "selection_batch_seed_start": 1_000_000,
        "streams": {
            "eval_info": "paired",
            "eval_info_final": "unpaired",
            "closing_eval": "unpaired",
        },
    }
    assert selection.confirm_best is False
    meta = json.loads((tmp_path / "model" / "best_model_meta.json").read_text())
    assert meta["eval_seed"] == 1_000_000


def test_seeded_run_without_reset_options_stays_unpaired(tmp_path, monkeypatch):
    """Pairing is opt-in: a seeded run that sets neither eval_seed nor
    eval_reset_options hands both info-dict streams eval_seed=None (the
    de02d13 unpaired stream). Deriving for every seeded run made every
    paired episode of the humanoid curricula serve from side A (a
    seeded reset restarts the serve alternation) and replayed the
    WallBall long-horizon audit's held-out seeds."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    train_module = importlib.import_module("courtside_dynamics.training.train")
    built: list[InfoDictEvalCallback] = []

    class _Recording(InfoDictEvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(train_module, "InfoDictEvalCallback", _Recording)
    train(_merged_eval_cfg(tmp_path, total_timesteps=200))

    assert len(built) == 2
    assert [callback.eval_seed for callback in built] == [None, None]
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["evaluation_seeding"] == {
        "paired": False,
        "eval_seed": None,
        "streams": {
            "eval_info": "unpaired",
            "eval_info_final": "unpaired",
            "closing_eval": "unpaired",
        },
    }


class _InitEcho(_StepsAlive):
    """``_StepsAlive`` echoing the episode's noisy reset state as
    ``init_x``, so an eval row shows whether its episodes were replayed
    (paired) or freshly drawn (unpaired)."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._init_x = float(obs[0])
        return obs, {**info, "init_x": self._init_x}

    def step(self, action):
        *head, info = super().step(action)
        return (*head, {**info, "init_x": self._init_x})


def _eval_rows_by_step(path) -> dict[str, dict[str, str]]:
    import csv

    by_step: dict[str, dict[str, str]] = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            by_step.setdefault(row["timestep"], {})[row["metric"]] = row["value"]
    return by_step


def test_paired_evaluation_repeats_identical_metrics_end_to_end(tmp_path):
    """A policy that never updates (learning_starts beyond the budget),
    evaluated twice by a seeded run: paired evaluation replays the same
    feeds, so both eval_info.csv rows are identical metric for metric --
    including ``init_x``, the episode's noisy reset state, which the
    legacy unpaired stream re-draws at every evaluation."""
    cfg = _merged_eval_cfg(
        tmp_path,
        env_fn=lambda: _InitEcho(BallBalanceEnv(episode_len=12)),
        final_info_eval=False,
        total_timesteps=400,
        n_eval_episodes=3,
        model_kwargs={"learning_starts": 10_000, "buffer_size": 500},
        # Pairing is opt-in (an explicit seed, or reset options to
        # derive one for); this env takes no reset options.
        eval_seed=4_321,
    )
    train(cfg)
    by_step = _eval_rows_by_step(tmp_path / "metrics" / "eval_info.csv")
    assert sorted(by_step, key=int) == ["200", "400"]
    assert "init_x_ep_mean" in by_step["200"]
    assert by_step["200"] == by_step["400"]


def test_final_info_eval_stream_stays_fresh_random_under_pairing(tmp_path):
    """docs/DECISIONS.md ("Unpaired evaluation is the root of the gate
    noise") and review 20260828 section 2.8 pair the *matched* stream and
    keep the final-config stream fresh-random, the unbiased estimate.
    A paired run of a never-updating policy therefore replays its
    selection rows exactly while every final-stream evaluation draws new
    episodes (a different mean reset state). The final stream used to
    be paired on its own fixed block, replaying identical rows too."""
    cfg = _merged_eval_cfg(
        tmp_path,
        env_fn=lambda: _InitEcho(BallBalanceEnv(episode_len=12)),
        total_timesteps=400,
        n_eval_episodes=3,
        model_kwargs={"learning_starts": 10_000, "buffer_size": 500},
        eval_seed=4_321,
    )
    train(cfg)
    selection = _eval_rows_by_step(tmp_path / "metrics" / "eval_info.csv")
    final = _eval_rows_by_step(tmp_path / "metrics" / "eval_info_final.csv")
    assert sorted(selection, key=int) == sorted(final, key=int) == ["200", "400"]
    assert selection["200"] == selection["400"]
    assert final["200"]["init_x_ep_mean"] != final["400"]["init_x_ep_mean"]


def test_resolve_eval_seed_derivation():
    """Explicit wins; a seeded run with reset options derives
    seed + 1_000_000; a seeded run without them, and an unseeded run,
    stay unpaired (the legacy stream)."""
    from courtside_dynamics.training.artifacts import _evaluation_seeding
    from courtside_dynamics.training.train import resolve_eval_seed

    options = ({"serve_side": "a"}, {"serve_side": "b"})

    def cfg(**kwargs):
        return TrainConfig(env_fn=_unbuildable_env, **kwargs)

    assert resolve_eval_seed(cfg(seed=3, eval_reset_options=options)) == 1_000_003
    assert resolve_eval_seed(cfg(seed=3)) is None
    assert resolve_eval_seed(cfg(seed=3, eval_seed=5)) == 5
    assert resolve_eval_seed(cfg(eval_seed=5)) == 5
    assert resolve_eval_seed(cfg(eval_seed=5, eval_reset_options=options)) == 5
    assert resolve_eval_seed(cfg(eval_reset_options=options)) is None
    assert resolve_eval_seed(cfg()) is None
    for unpaired in (cfg(), cfg(seed=3)):
        seeding = _evaluation_seeding(unpaired)
        assert (seeding["paired"], seeding["eval_seed"]) == (False, None)
    assert _evaluation_seeding(cfg(eval_seed=5))["derived_from_seed"] is False


_ALL_UNPAIRED_SB3 = {"reward_eval": "unpaired", "closing_eval": "unpaired"}


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (
            # An eval_seed no stream applies: no info-dict evaluator runs.
            {"info_dict_eval": False, "eval_seed": 7},
            {"paired": False, "eval_seed": None, "streams": _ALL_UNPAIRED_SB3},
        ),
        (
            # confirm_best is only wired under headline selection.
            {"eval_seed": 7, "confirm_best_eval": True},
            {
                "paired": True,
                "eval_seed": 7,
                "derived_from_seed": False,
                "selection_batch_seed_start": 7,
                "streams": {"eval_info": "paired", **_ALL_UNPAIRED_SB3},
            },
        ),
        (
            # Headline selection without confirm_best: no confirmation.
            {"eval_seed": 7, "headline_key": "steps_alive"},
            {
                "paired": True,
                "eval_seed": 7,
                "derived_from_seed": False,
                "selection_batch_seed_start": 7,
                "streams": {"eval_info": "paired", **_ALL_UNPAIRED_SB3},
            },
        ),
        (
            # Every info-dict stream on; the reward stream merged away.
            {
                "seed": 3,
                "eval_reset_options": ({"serve_side": "a"},),
                "headline_key": "steps_alive",
                "confirm_best_eval": True,
                "final_info_eval": True,
            },
            {
                "paired": True,
                "eval_seed": 1_000_003,
                "derived_from_seed": True,
                "selection_batch_seed_start": 1_000_003,
                "confirmation_batch_seed_start": 1_100_003,
                "streams": {
                    "eval_info": "paired",
                    "eval_info_confirmation": "paired",
                    "eval_info_final": "unpaired",
                    "closing_eval": "unpaired",
                },
            },
        ),
        (
            # Unpaired run: the confirmation stream runs, unpaired.
            {
                "headline_key": "steps_alive",
                "confirm_best_eval": True,
                "final_info_eval": True,
            },
            {
                "paired": False,
                "eval_seed": None,
                "streams": {
                    "eval_info": "unpaired",
                    "eval_info_confirmation": "unpaired",
                    "eval_info_final": "unpaired",
                    "closing_eval": "unpaired",
                },
            },
        ),
    ],
)
def test_evaluation_seeding_records_only_the_streams_that_run(overrides, expected):
    """config.json's evaluation_seeding used to claim the selection,
    confirmation and final-stream seed blocks for every paired config,
    including streams train() never wires (info_dict_eval off, no
    headline selection or confirm_best, final_info_eval off). It now
    lists exactly the streams that run, and a seed block only for a
    paired stream among them."""
    from courtside_dynamics.training.artifacts import _evaluation_seeding

    cfg = TrainConfig(env_fn=_unbuildable_env, **overrides)
    assert _evaluation_seeding(cfg) == expected


def test_monitor_info_keywords_reach_the_training_monitor(tmp_path):
    """Contract C3: the training workers' Monitor appends the named info
    keys to every episode row, and every monitor reader (stage summary,
    learning plots) tolerates the extra column."""
    from courtside_dynamics.training import load_monitor_episodes
    from courtside_dynamics.training.monitor_log import (
        read_monitor_rewards_lengths,
    )

    cfg = _salvage_cfg(
        tmp_path,
        env_fn=_steps_alive_env,
        total_timesteps=60,
        monitor_info_keywords=("steps_alive",),
    )
    train(cfg)

    monitor_dir = tmp_path / "metrics" / "monitor"
    (monitor_csv,) = monitor_dir.glob("*.monitor.csv")
    assert monitor_csv.read_text().splitlines()[1].split(",") == [
        "r",
        "l",
        "t",
        "steps_alive",
    ]
    episodes = load_monitor_episodes(str(monitor_dir)).episodes
    assert len(episodes) >= 5
    # Read at each episode's final step: the counter equals the length.
    assert list(episodes["steps_alive"]) == list(episodes["l"])
    rewards, lengths = read_monitor_rewards_lengths(str(monitor_dir))
    assert lengths == list(episodes["l"]) and len(rewards) == len(lengths)
    assert "[train monitor logs]" in (tmp_path / "stage_summary.txt").read_text()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["train_config"]["monitor_info_keywords"] == ["steps_alive"]


def test_monitor_info_keywords_must_be_keys_the_env_emits(tmp_path):
    """Monitor raises KeyError at the first episode end for a key the env
    does not emit; train() probes the env and refuses it up front."""
    log_dir = tmp_path / "run"
    cfg = _salvage_cfg(
        log_dir, env_fn=_steps_alive_env, monitor_info_keywords=("steps_alvie",)
    )
    with pytest.raises(
        ValueError, match=r"'steps_alvie' \(did you mean 'steps_alive'\?\)"
    ):
        train(cfg)
    assert not log_dir.exists()


def test_config_json_records_the_observation_fingerprint(tmp_path):
    """Contract C3: the env probe banks the observation layout by name,
    also through a Gymnasium wrapper (read via get_wrapper_attr)."""
    from courtside_dynamics.training.artifacts import observation_names_sha256

    cfg = TrainConfig(env_fn=_steps_alive_env, log_dir=str(tmp_path))
    write_run_config(cfg, str(tmp_path))
    env_block = json.loads((tmp_path / "config.json").read_text())["env"]
    names = list(BallBalanceEnv.observation_names)
    assert env_block["observation_names"] == names
    assert env_block["observation_names_sha256"] == observation_names_sha256(
        names
    )


def _rewrite_source_observation_names(source_dir, names, *, keep_names=True):
    from courtside_dynamics.training.artifacts import observation_names_sha256

    config_path = source_dir / "config.json"
    config = json.loads(config_path.read_text())
    if names is None:
        config["env"].pop("observation_names", None)
        config["env"].pop("observation_names_sha256", None)
    else:
        config["env"]["observation_names_sha256"] = observation_names_sha256(
            names
        )
        if keep_names:
            config["env"]["observation_names"] = list(names)
        else:
            config["env"].pop("observation_names", None)
    config_path.write_text(json.dumps(config))


def _sac_warm_start_target(
    env_fn, source_dir, log_dir, *, algo="SAC", extra_callbacks=(), **model_kwargs
):
    """The SAC warm-start tests' 8-step target leg (warmup never ends)."""
    return TrainConfig(
        env_fn=env_fn,
        algo=algo,
        total_timesteps=8,
        log_dir=str(log_dir),
        n_envs=1,
        seed=13,
        eval_freq=10_000,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        info_dict_eval=False,
        n_eval_episodes=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
        model_kwargs={"buffer_size": 64, "learning_starts": 1_000, **model_kwargs},
        extra_callbacks=extra_callbacks,
    )


def test_warm_start_refuses_a_same_shape_observation_layout_change(tmp_path):
    """Review §7.2 (*new*): warm start compared shapes only, so a
    same-width meaning change (world-frame spin, scaled counters) would
    load every old checkpoint onto a different task. The recorded layout
    is now compared by name, naming the first differing index."""
    source_dir, env_fn, _ = _make_sac_warm_start_source(tmp_path)
    names = list(BallBalanceEnv.observation_names)
    renamed = [*names[:3], "ball_vx_world_frame", *names[4:]]
    _rewrite_source_observation_names(source_dir, renamed)
    cfg = _sac_warm_start_target(env_fn, source_dir, tmp_path / "target")
    with pytest.raises(
        ValueError,
        match=r"observation index 3 is 'ball_vx_world_frame' in the "
        r"recorded layout but 'ball_vx' in this env",
    ):
        _prepare_warm_start(cfg)

    # Only the digest survives: still refused, naming both digests.
    _rewrite_source_observation_names(source_dir, renamed, keep_names=False)
    with pytest.raises(ValueError, match="observation_names_sha256"):
        _prepare_warm_start(cfg)


def test_warm_start_from_a_pre_fingerprint_source_keeps_the_shape_check(
    tmp_path, capsys
):
    """Every run directory written before the fingerprint existed lacks
    it; those still warm-start on the shape check, with a notice."""
    source_dir, env_fn, _ = _make_sac_warm_start_source(tmp_path)
    _rewrite_source_observation_names(source_dir, None)
    target_dir = tmp_path / "target"
    train(_sac_warm_start_target(env_fn, source_dir, target_dir))
    assert "predates the observation fingerprint" in capsys.readouterr().out
    config = json.loads((target_dir / "config.json").read_text())
    initialization = config["initialization"]
    assert initialization["observation_fingerprint"] == "source_lacks_fingerprint"


class _CaptureWarmup(BaseCallback):
    """Record the warmup mode and count uniform-random action draws."""

    def __init__(self) -> None:
        super().__init__()
        self.use_sde_at_warmup: bool | None = None
        self.uniform_samples = 0

    def _on_training_start(self) -> None:
        self.use_sde_at_warmup = bool(self.model.use_sde_at_warmup)
        sample = self.model.action_space.sample

        def counting_sample(*args, **kwargs):
            self.uniform_samples += 1
            return sample(*args, **kwargs)

        self.model.action_space.sample = counting_sample

    def _on_training_end(self) -> None:
        # Drop the instance override (the closure holds this callback,
        # which SB3 cannot pickle into final_model.zip).
        del self.model.action_space.sample

    def _on_step(self) -> bool:
        return True


@pytest.mark.parametrize("algo", ["SAC", "DEMOSAC"])
def test_gsde_warm_start_refills_with_the_transferred_policy(tmp_path, algo):
    """Review §5b #7 (*confirmed*): a warm start's learning_starts refill
    sampled uniform-random actions (use_sde_at_warmup=False), so every
    warm-started leg spent its warmup on noise instead of the transferred
    policy. Under gSDE, train() now defaults use_sde_at_warmup=True."""
    source_dir, env_fn, _ = _make_sac_warm_start_source(
        tmp_path, model_kwargs={"use_sde": True}
    )
    capture = _CaptureWarmup()
    target_dir = tmp_path / "target"
    train(
        _sac_warm_start_target(
            env_fn,
            source_dir,
            target_dir,
            algo=algo,
            use_sde=True,
            extra_callbacks=(capture,),
        )
    )
    assert capture.use_sde_at_warmup is True
    assert capture.uniform_samples == 0
    config = json.loads((target_dir / "config.json").read_text())
    assert config["initialization"]["warmup"] == {
        "learning_starts": 1_000,
        "use_sde": True,
        "use_sde_at_warmup": True,
        "use_sde_at_warmup_source": "warm_start_default",
        "warmup_actions": "policy_with_gsde_noise",
    }
    assert config["resolved_model"]["hyperparameters"]["use_sde_at_warmup"] is True
    # The caller's model_kwargs are recorded as given, not rewritten.
    assert "use_sde_at_warmup" not in config["train_config"]["model_kwargs"]


def test_warm_start_respects_an_explicit_use_sde_at_warmup(tmp_path, capsys):
    source_dir, env_fn, _ = _make_sac_warm_start_source(
        tmp_path, model_kwargs={"use_sde": True}
    )
    capture = _CaptureWarmup()
    target_dir = tmp_path / "target"
    train(
        _sac_warm_start_target(
            env_fn,
            source_dir,
            target_dir,
            use_sde=True,
            use_sde_at_warmup=False,
            extra_callbacks=(capture,),
        )
    )
    assert capture.use_sde_at_warmup is False
    assert capture.uniform_samples == 8
    warmup = json.loads((target_dir / "config.json").read_text())[
        "initialization"
    ]["warmup"]
    assert warmup["use_sde_at_warmup_source"] == "model_kwargs"
    assert warmup["warmup_actions"] == "uniform_random"
    assert "take uniform-random actions" in capsys.readouterr().out


def test_warm_start_without_gsde_warns_that_warmup_is_uniform(tmp_path, capsys):
    """Without gSDE SB3 cannot act with the policy during warmup at all;
    the run must say so instead of implying the policy refills."""
    source_dir, env_fn, _ = _make_sac_warm_start_source(tmp_path)
    target_dir = tmp_path / "target"
    train(_sac_warm_start_target(env_fn, source_dir, target_dir))
    out = capsys.readouterr().out
    assert "first 1,000 steps (learning_starts) take uniform-random" in out
    warmup = json.loads((target_dir / "config.json").read_text())[
        "initialization"
    ]["warmup"]
    assert warmup["use_sde"] is False
    assert warmup["warmup_actions"] == "uniform_random"
    assert warmup["use_sde_at_warmup_source"] == "sb3_default"


def test_run_summary_reports_the_selecting_batch_from_best_model_meta(
    tmp_path,
):
    """Under headline selection the best-checkpoint block used to read
    its reward from evaluations.npz (another stream's episodes) and its
    counters from ``*_final`` (one episode). It now reports the batch
    that won selection, from best_model_meta.json's ``metrics``."""
    import json

    from courtside_dynamics.training.artifacts import write_run_summary

    np.savez(
        tmp_path / "evaluations.npz",
        timesteps=np.array([25_000, 50_000]),
        results=np.array([[0.5, 0.5], [1.0, 1.0]]),
        ep_lengths=np.array([[30, 30], [30, 30]]),
    )
    (tmp_path / "best_model_meta.json").write_text(
        json.dumps(
            {
                "timestep": 50_000,
                "selection_keys": ["bounce_count_ep_mean", "success_rate"],
                "selection_values": {
                    "bounce_count_ep_mean": 3.4,
                    "success_rate": 0.6,
                },
                "metrics": {
                    "episode_reward_mean": 4.25,
                    "episode_length": 812.0,
                    "success_rate": 0.6,
                    "bounce_count_ep_mean": 3.4,
                    "bounce_count_final": 1.0,
                    "bounce_count_max": 5.0,
                },
            }
        )
    )
    # A stale/mixed CSV row at the same step must not leak in.
    (tmp_path / "eval_info.csv").write_text(
        "timestep,metric,value\n"
        "50000,episode_reward_mean,9.9\n"
        "50000,bounce_count_ep_mean,9.9\n"
        "50000,bounce_count_final,9.0\n"
    )

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(env_fn=env_fn, log_dir=str(tmp_path), headline_key="bounce_count")
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    text = (tmp_path / "stage_summary.txt").read_text()
    best_block = text.split("Best Checkpoint Evaluation (step 50,000)", 1)[1]
    assert "Reward:         4.250  [selecting batch]" in best_block
    assert "1.000 +/- 0.000" not in best_block
    assert "Episode length: 812.0" in best_block
    assert "Success rate:   60.0%" in best_block
    assert "bounce_count: ep-mean 3.40  last-episode 1.00  max 5.00" in best_block
    assert "9.9" not in best_block


def test_run_summary_labels_final_counters_as_last_episode(tmp_path):
    """Reward-selected runs keep the reward stream's numbers, but a
    ``*_final`` counter is still one episode and is labeled so."""
    from courtside_dynamics.training.artifacts import write_run_summary

    np.savez(
        tmp_path / "evaluations.npz",
        timesteps=np.array([25_000]),
        results=np.array([[0.5, 1.5]]),
        ep_lengths=np.array([[30, 30]]),
    )
    (tmp_path / "eval_info.csv").write_text(
        "timestep,metric,value\n"
        "25000,rally_count_final,2.0\n"
        "25000,rally_count_max,3.0\n"
    )

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    write_run_summary(
        TrainConfig(env_fn=env_fn, log_dir=str(tmp_path)),
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    best_block = (tmp_path / "stage_summary.txt").read_text().split(
        "Best Checkpoint Evaluation (step 25,000)", 1
    )[1]
    assert "Reward:         1.000 +/- 0.500" in best_block
    assert "rally_count: last-episode 2.00  max 3.00" in best_block
    assert " final " not in best_block


def test_train_refuses_a_log_dir_holding_a_previous_attempts_eval_rows(
    tmp_path,
):
    """A second attempt in the same log_dir used to append its rows to
    the first attempt's eval_info.csv (and leave that attempt's best
    triple beside the new config.json). Refused before any output."""
    previous_config = '{"previous": "attempt"}\n'
    (tmp_path / "config.json").write_text(previous_config)
    csv_path = tmp_path / "metrics" / "eval_info.csv"
    csv_path.parent.mkdir()
    rows = "timestep,metric,value\n25000,steps_alive_ep_mean,3.0\n"
    csv_path.write_text(rows)
    cfg = TrainConfig(env_fn=_unbuildable_env, log_dir=str(tmp_path))
    with pytest.raises(ValueError, match="previous training attempt"):
        train(cfg)
    assert (tmp_path / "config.json").read_text() == previous_config
    assert csv_path.read_text() == rows


def test_reuse_log_dir_rotates_the_previous_attempts_eval_logs(
    tmp_path, capsys
):
    """``reuse_log_dir=True`` is the explicit opt-in: the old evaluation
    CSV moves aside, and the new one holds this attempt's rows only."""
    import csv

    def eval_rows(path):
        with open(path) as stream:
            return [
                (row["timestep"], row["metric"]) for row in csv.DictReader(stream)
            ]

    first = _merged_eval_cfg(tmp_path, final_info_eval=False, total_timesteps=400)
    train(first)
    csv_path = tmp_path / "metrics" / "eval_info.csv"
    first_rows = eval_rows(csv_path)
    assert first_rows

    second = _merged_eval_cfg(
        tmp_path, final_info_eval=False, total_timesteps=400, reuse_log_dir=True
    )
    train(second)
    out = capsys.readouterr().out
    assert "reuse_log_dir=True: rotated the previous attempt's" in out
    (rotated,) = (tmp_path / "metrics").glob("eval_info.attempt_*.csv")
    assert eval_rows(rotated) == first_rows
    second_rows = eval_rows(csv_path)
    # One row per (timestep, metric): no second attempt appended.
    assert len(second_rows) == len(set(second_rows)) == len(first_rows)


def test_reuse_log_dir_rotates_the_previous_attempts_records_and_logs(
    tmp_path, capsys
):
    """repo_review_20261016 train-5: SB3's Monitor overwrites only the
    CSVs of this attempt's workers (a 2-worker attempt's 1.monitor.csv
    outlived a 1-worker retry), and TensorBoard reads every event file
    in its folder, so both were read back into the new attempt's
    curves. Both directories rotate whole, beside the run record and
    the sync manifest, and the new attempt seals the directory."""
    from courtside_dynamics.experiments.records import verify_run_dir

    metrics = tmp_path / "metrics"
    small = {"learning_starts": 16, "buffer_size": 500, "batch_size": 32}
    train(
        _merged_eval_cfg(
            tmp_path,
            final_info_eval=False,
            total_timesteps=400,
            n_envs=2,
            model_kwargs=small,
        )
    )
    first_monitors = sorted(path.name for path in (metrics / "monitor").iterdir())
    assert first_monitors == ["0.monitor.csv", "1.monitor.csv"]
    first_events = sorted(path.name for path in (metrics / "tensorboard").iterdir())
    assert first_events
    first_record = (tmp_path / "run_record.json").read_text()
    first_manifest = (tmp_path / "sync_manifest.json").read_text()

    train(
        _merged_eval_cfg(
            tmp_path,
            final_info_eval=False,
            total_timesteps=400,
            reuse_log_dir=True,
            model_kwargs=small,
        )
    )
    out = capsys.readouterr().out
    assert out.count("reuse_log_dir=True: rotated the previous attempt's") == 5
    (old_monitor,) = metrics.glob("monitor.attempt_*")
    assert sorted(path.name for path in old_monitor.iterdir()) == first_monitors
    assert sorted(path.name for path in (metrics / "monitor").iterdir()) == [
        "0.monitor.csv"
    ]
    (old_events,) = metrics.glob("tensorboard.attempt_*")
    assert sorted(path.name for path in old_events.iterdir()) == first_events
    current_events = {path.name for path in (metrics / "tensorboard").iterdir()}
    assert current_events and not current_events & set(first_events)
    (old_record,) = tmp_path.glob("run_record.attempt_*.json")
    assert old_record.read_text() == first_record
    (old_manifest,) = tmp_path.glob("sync_manifest.attempt_*.json")
    assert old_manifest.read_text() == first_manifest
    assert verify_run_dir(tmp_path, strict=True).complete


def test_train_refuses_a_log_dir_holding_a_previous_attempts_run_record(
    tmp_path,
):
    """An attempt that died before its first evaluation leaves no eval
    rows, but every attempt writes its run record first: a retry in the
    same directory is refused before any output, like the rows."""
    record = '{"status": "running"}\n'
    (tmp_path / "run_record.json").write_text(record)
    cfg = TrainConfig(env_fn=_unbuildable_env, log_dir=str(tmp_path))
    with pytest.raises(ValueError, match=r"previous training attempt .*run_record"):
        train(cfg)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run_record.json"]
    assert (tmp_path / "run_record.json").read_text() == record


def test_interrupt_salvage_saves_the_model_before_a_failing_gate_finalize(
    tmp_path, monkeypatch, capsys
):
    """repo_review_20261016 train-2: the KeyboardInterrupt handler called
    the gate's finalize first and unguarded, so an OSError archiving the
    final stage's best (on the Drive mount) escaped train() before
    final_model, the normalizer or the summary existed. The salvage now
    saves the model and normalizer first and isolates finalize, like
    the crash salvage."""
    from stable_baselines3 import SAC

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
    from courtside_dynamics.experiments.records import verify_run_dir
    from courtside_dynamics.training.artifacts import artifact_path

    def archive_fails(self, destination_dir):
        raise OSError("Transport endpoint is not connected")

    monkeypatch.setattr(InfoDictEvalCallback, "archive_best", archive_fails)
    cfg = _family_e2e_cfg(
        tmp_path,
        "SAC",
        early_stop_patience=None,
        final_info_eval=False,
        confirm_best_eval=False,
        extra_callbacks=(_RaiseAt(KeyboardInterrupt, at=200),),
    )
    model = train(cfg)
    out = capsys.readouterr().out
    assert (
        "could not salvage performance-gate stage history: "
        "OSError('Transport endpoint is not connected')"
    ) in out
    assert 200 <= model.num_timesteps < cfg.total_timesteps
    final_model = artifact_path(str(tmp_path), "final_model")
    assert SAC.load(final_model).num_timesteps == model.num_timesteps
    assert os.path.isfile(artifact_path(str(tmp_path), "vec_normalize"))
    assert _summary_field(tmp_path, "Status") == "interrupted"
    assert "closing eval" in _summary_field(tmp_path, "Final eval")
    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["status"] == "interrupted"
    assert verify_run_dir(tmp_path, strict=True).complete


def test_a_guard_firing_at_the_final_evaluation_completes_the_budget(tmp_path):
    """repo_review_20261016 train-3: when the stop condition first holds
    at the last scheduled evaluation the run trained its full budget;
    both 2026-08-28 runs were labeled early_stop_patience anyway."""
    model = train(
        _merged_eval_cfg(
            tmp_path,
            final_info_eval=False,
            total_timesteps=400,
            eval_freq=100,
            early_stop_patience=2,
            best_metric_keys=("steps_alive_ep_mean",),
        )
    )
    assert model.num_timesteps == 400
    reason = _summary_field(tmp_path, "Stop reason")
    assert reason.startswith(
        "full_budget: trained all 400 steps; the stop condition also held at "
        "the final evaluation (early_stop_patience: no improvement in"
    )
    assert _summary_field(tmp_path, "Status") == "completed"
    assert _summary_field(tmp_path, "Timesteps") == "400"
    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["status"] == "completed"
    assert record["stop_reason"] == reason


def test_the_reward_path_stop_at_the_final_evaluation_completes_the_budget(
    tmp_path,
):
    cfg = _salvage_cfg(
        tmp_path,
        total_timesteps=600,
        eval_freq=200,
        early_stop_patience=1,
        normalize_obs=False,
        model_kwargs={"learning_starts": 16, "buffer_size": 500, "batch_size": 32},
    )
    model = train(cfg)
    assert model.num_timesteps == 600
    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["status"] == "completed"
    assert record["stop_reason"].startswith("full_budget: trained all 600 steps")
    assert "no new best mean reward" in record["stop_reason"]


def test_inert_early_stops_are_warned_before_output_and_recorded(
    tmp_path, capsys
):
    """repo_review_20261016 train-3: nothing said when a budget was too
    small for a stop to fire at all."""
    from courtside_dynamics.training.train import early_stop_reachability

    cfg = _merged_eval_cfg(
        tmp_path,
        final_info_eval=False,
        total_timesteps=200,
        eval_freq=100,
        early_stop_patience=2,
    )
    warning = (
        "early_stop_patience can never fire: its earliest stop is "
        "evaluation 4 (after a warm-up of 2), but total_timesteps 200 at "
        "eval_freq 100 runs 2 evaluation(s); this run trains its full "
        "budget whatever its evaluations read"
    )
    assert early_stop_reachability(cfg)["warnings"] == [warning]
    train(cfg)
    assert f"[train] warning: {warning}" in capsys.readouterr().out
    config = json.loads((tmp_path / "config.json").read_text())
    reachability = config["resolved_training"]["early_stop_reachability"]
    assert reachability["warnings"] == [warning]
    assert reachability["guards"]["early_stop_patience"]["inert"] is True
    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["preflight_warnings"] == [warning]


def test_recipe_early_stop_reachability_matches_the_review():
    """BallBalance's reward-path patience of 20 can never fire at its 1M
    budget (earliest stop 41 of 40 evaluations); PaddleTennis's headline
    patience can (40 of 80)."""
    from courtside_dynamics.recipes import build_train_config
    from courtside_dynamics.training.train import early_stop_reachability

    ball = early_stop_reachability(
        build_train_config("BallBalance", log_dir="x", seed=0)
    )
    assert ball["scheduled_evaluations"] == 40
    assert ball["guards"]["early_stop_patience"] == {
        "selection": "reward",
        "warmup_evals": 20,
        "earliest_stop_evaluation": 41,
        "inert": True,
    }
    assert len(ball["warnings"]) == 1
    paddle = early_stop_reachability(
        build_train_config("PaddleTennis", log_dir="x", seed=0)
    )
    assert paddle["scheduled_evaluations"] == 80
    assert paddle["guards"]["early_stop_patience"]["earliest_stop_evaluation"] == 40
    assert paddle["warnings"] == []


_GATE = {
    "stages": [{"x": 1.0}],
    "metric_key": "steps_alive_ep_mean",
    "threshold": 1.0,
    "sustain_evals": 1,
}


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"reward_eval_episodes": 0}, ValueError, "reward_eval_episodes must be"),
        ({"reward_eval_episodes": 5}, ValueError, "requires headline-metric"),
        (
            {"headline_key": "steps_alive", "final_eval_episodes": 0},
            ValueError,
            "final_eval_episodes must be",
        ),
        ({"final_eval_episodes": 4}, ValueError, "requires info_dict_eval and"),
        (
            {"checkpoint_diagnosis": {"episodez": 3}},
            ValueError,
            r"checkpoint_diagnosis has unknown keys \['episodez'\]",
        ),
        (
            {"checkpoint_diagnosis": {"episodes": 3}, "checkpoint_freq": 0},
            ValueError,
            "requires checkpoint_freq > 0",
        ),
        (
            {"checkpoint_diagnosis": {"episodes": 0}},
            ValueError,
            "episodes must be positive",
        ),
        ({"checkpoint_diagnosis": ("episodes",)}, TypeError, "must be a mapping"),
        (
            {"info_dict_eval": False, "final_info_eval": True},
            ValueError,
            "require info_dict_eval",
        ),
        (
            {"info_dict_eval": False, "performance_gate": _GATE},
            ValueError,
            "require info_dict_eval",
        ),
        (
            {"performance_gate": {**_GATE, "sustain_evalz": 2}},
            ValueError,
            r"unknown performance_gate key\(s\) \['sustain_evalz'\]",
        ),
        (
            {
                "performance_gate": {
                    key: value for key, value in _GATE.items() if key != "threshold"
                }
            },
            ValueError,
            r"performance_gate must set \['threshold'\]",
        ),
    ],
)
def test_misconfigured_reuse_retry_leaves_the_previous_attempt_untouched(
    tmp_path, overrides, error, message
):
    """The eval-stream, diagnosis and gate checks used to run inside the
    callback wiring -- after reuse_log_dir=True had already rotated the
    previous attempt's eval CSVs aside and overwritten its config.json.
    They are pre-flight checks now: the retry fails before any env is
    built and the earlier attempt's artifacts stay exactly where they
    were."""
    previous_config = '{"previous": "attempt"}\n'
    (tmp_path / "config.json").write_text(previous_config)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    rows = "timestep,metric,value\n25000,steps_alive_ep_mean,3.0\n"
    for name in ("eval_info.csv", "eval_info_final.csv"):
        (metrics / name).write_text(rows)
    cfg = TrainConfig(
        env_fn=_unbuildable_env,
        log_dir=str(tmp_path),
        reuse_log_dir=True,
        **overrides,
    )
    with pytest.raises(error, match=message):
        train(cfg)
    assert (tmp_path / "config.json").read_text() == previous_config
    assert sorted(path.name for path in metrics.iterdir()) == [
        "eval_info.csv",
        "eval_info_final.csv",
    ]
    for name in ("eval_info.csv", "eval_info_final.csv"):
        assert (metrics / name).read_text() == rows


@pytest.mark.parametrize(
    "bad_entry, message",
    [
        ({"serve_side": "c"}, "serve_side must be 'a', 'b'"),
        ({"serve_sdie": "b"}, r"unsupported reset options: \['serve_sdie'\]"),
    ],
)
def test_invalid_eval_reset_options_fail_before_any_output(
    tmp_path, bad_entry, message
):
    """eval_reset_options were only shape-checked up front, so an option
    the env's reset rejects passed pre-flight, wrote config.json and
    crashed model.learn() at the first paired evaluation. A throwaway
    evaluation env now replays the paired reset for every distinct
    entry before any output is written."""
    from courtside_dynamics.envs import PaddleTennisEnv

    log_dir = tmp_path / "run"
    cfg = TrainConfig(
        env_fn=PaddleTennisEnv,
        algo="SAC",
        total_timesteps=16,
        log_dir=str(log_dir),
        n_envs=1,
        seed=0,
        eval_freq=8,
        checkpoint_freq=0,
        record_video=False,
        n_eval_episodes=1,
        # First, so the unprobed run reached it at its first evaluation
        # (episode i uses entry i % len); later entries are probed too.
        eval_reset_options=(bad_entry, {"serve_side": "a"}),
    )
    with pytest.raises(ValueError, match="eval_reset_options entry") as raised:
        train(cfg)
    assert raised.match(message)
    assert not log_dir.exists()


def test_eval_reset_option_probe_resets_once_per_distinct_mapping():
    """The pre-flight probe uses the evaluation factory (not the training
    one), resets one throwaway instance once per distinct mapping with the
    resolved eval seed, and closes it."""
    from courtside_dynamics.training.train import _validate_eval_reset_options

    resets: list[tuple[int | None, dict[str, Any] | None]] = []
    closed: list[bool] = []

    class _Recorder(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            resets.append((seed, options))
            return self.env.reset(seed=seed, options=options)

        def close(self):
            closed.append(True)
            super().close()

    cfg = TrainConfig(
        env_fn=_unbuildable_env,
        eval_env_fn=lambda: _Recorder(BallBalanceEnv(episode_len=12)),
        seed=4,
        eval_reset_options=(
            {"serve_side": "a"},
            {"serve_side": "b"},
            {"serve_side": "a"},
        ),
    )
    _validate_eval_reset_options(cfg, 1_000_004)
    assert resets == [
        (1_000_004, {"serve_side": "a"}),
        (1_000_004, {"serve_side": "b"}),
    ]
    assert closed == [True]
    # Unpaired (no options): nothing to probe, no env built.
    _validate_eval_reset_options(
        TrainConfig(env_fn=_unbuildable_env, seed=4), None
    )


# -- PPO plan Step 4: the on-policy pre-flight --------------------------------


def _on_policy_cfg(log_dir, **overrides) -> TrainConfig:
    """A PPO config that passes the on-policy pre-flight (rollout 4 x 64
    = 256 env steps, minibatch 64, cadences of whole rollouts); each
    refusal test breaks exactly one thing. The env is unbuildable: every
    refusal must come before any env work."""
    model_kwargs = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
    model_kwargs.update(overrides.pop("model_kwargs", {}))
    fields: dict[str, Any] = {
        "env_fn": _unbuildable_env,
        "algo": "PPO",
        "log_dir": str(log_dir),
        "n_envs": 4,
        "total_timesteps": 1_024,
        "eval_freq": 512,
        "checkpoint_freq": 1_024,
        "model_kwargs": model_kwargs,
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def test_on_policy_preflight_accepts_whole_rollout_geometry(tmp_path):
    from courtside_dynamics.training.train import _validate_on_policy_config

    _validate_on_policy_config(_on_policy_cfg(tmp_path / "run"))
    # checkpoint_freq <= 0 disables checkpoints, so it has no cadence.
    _validate_on_policy_config(_on_policy_cfg(tmp_path / "run", checkpoint_freq=0))


@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param(
            {
                "performance_gate": {
                    "stages": ({},),
                    "metric_key": "m",
                    "threshold": 1.0,
                    "sustain_evals": 1,
                    "advance_update_pause_steps": 1_000,
                },
                "info_dict_eval": True,
            },
            r"\['advance_update_pause_steps'\]",
            id="gate-update-pause",
        ),
        pytest.param(
            {
                "performance_gate": {
                    "stages": ({},),
                    "metric_key": "m",
                    "threshold": 1.0,
                    "sustain_evals": 1,
                    "clear_replay_buffer_on_advance": True,
                },
            },
            r"\['clear_replay_buffer_on_advance'\]",
            id="gate-replay-wipe",
        ),
        pytest.param(
            {
                "performance_gate": {
                    "stages": ({},),
                    "metric_key": "m",
                    "threshold": 1.0,
                    "sustain_evals": 1,
                    "reset_entropy_on_advance": True,
                },
            },
            r"\['reset_entropy_on_advance'\]",
            id="gate-entropy-reset",
        ),
        pytest.param(
            {"model_kwargs": {"batch_size": 1}},
            "normalize_advantage",
            id="batch-size-one",
        ),
        pytest.param(
            {"n_envs": 1, "model_kwargs": {"n_steps": 1, "batch_size": 1}},
            r"n_steps \* n_envs = 1 \* 1 = 1",
            id="one-transition-rollout",
        ),
        pytest.param(
            {"model_kwargs": {"batch_size": 96}},
            "batch_size 96 does not divide the rollout",
            id="batch-does-not-divide",
        ),
        pytest.param(
            {"total_timesteps": 255, "eval_freq": 256, "checkpoint_freq": 0},
            "total_timesteps 255 is shorter than one rollout",
            id="budget-under-a-rollout",
        ),
        pytest.param(
            {"eval_freq": 128},
            "eval_freq 128 is 32 steps per worker",
            id="eval-inside-a-rollout",
        ),
        pytest.param(
            {"eval_freq": 640},
            r"eval_freq 640 is not a whole number of rollouts .* e\.g\. 512",
            id="eval-not-whole-rollouts",
        ),
        pytest.param(
            {"checkpoint_freq": 1_000},
            r"checkpoint_freq 1,000 is not a whole number of rollouts",
            id="checkpoint-not-whole-rollouts",
        ),
        pytest.param(
            {"model_kwargs": {"policy_kwargs": {"log_std_int": -1.0}}},
            "did you mean 'log_std_init'",
            id="unknown-policy-kwarg",
        ),
        pytest.param(
            {"model_kwargs": {"policy_kwargs": {"use_sde": True}}},
            r"policy_kwargs may not set \['use_sde'\]",
            id="reserved-policy-kwarg",
        ),
        pytest.param(
            {"model_kwargs": {"n_steps": 64.0}},
            "must be a positive integer",
            id="non-integer-n-steps",
        ),
        pytest.param(
            {"model_kwargs": {"gamma": "0.995"}},
            "gamma must be a number",
            id="non-numeric-gamma",
        ),
        pytest.param(
            {
                "model_kwargs": {
                    "policy_kwargs": {"net_arch": {"pi": [8], "qf": [8]}}
                }
            },
            r"net_arch keys \['qf'\] .* \('qf' is SAC's critic key",
            id="sac-net-arch-key",
        ),
        pytest.param(
            # The floor is ceil(20 * 256 / 512) = 10 evaluations.
            {
                "performance_gate": {
                    "stages": ({}, {}),
                    "metric_key": "m",
                    "threshold": 1.0,
                    "sustain_evals": 1,
                    "stage_eval_budget": 10,
                },
            },
            r"stage_eval_budget 10 would stop the run at evaluation 10, "
            r"before min_policy_updates_before_stop=20 .* at least 11",
            id="gate-budget-inside-the-update-floor",
        ),
    ],
)
def test_on_policy_preflight_refuses_before_any_output(tmp_path, overrides, match):
    """One refusal per case; each leaves the log_dir untouched (no
    config.json, no directory), since the check runs with the other
    config-only checks before any env is built or artifact written."""
    log_dir = tmp_path / "run"
    cfg = _on_policy_cfg(log_dir, **overrides)
    with pytest.raises(ValueError, match=match):
        train(cfg)
    assert not log_dir.exists()


def test_on_policy_preflight_is_inert_for_the_sac_family(tmp_path):
    """SAC and DemoSAC return at once: their cadences are not counted in
    rollouts, and the gate's SAC-only levers are theirs to use."""
    from courtside_dynamics.training.train import _validate_on_policy_config

    for algo in ("SAC", "DEMOSAC"):
        cfg = _on_policy_cfg(
            tmp_path / "run",
            algo=algo,
            eval_freq=100,
            checkpoint_freq=333,
            total_timesteps=1,
            model_kwargs={"n_steps": 1, "batch_size": 1},
            performance_gate={
                "stages": ({},),
                "metric_key": "m",
                "threshold": 1.0,
                "sustain_evals": 1,
                "clear_replay_buffer_on_advance": True,
                "reset_entropy_on_advance": True,
            },
        )
        _validate_on_policy_config(cfg)


def test_validate_policy_kwargs_resolves_the_policy_alias():
    from courtside_dynamics.training.algos import validate_policy_kwargs

    validate_policy_kwargs("PPO", "MlpPolicy", {"log_std_init": -2.0, "ortho_init": False})
    with pytest.raises(ValueError, match="not one of PPO's policies"):
        validate_policy_kwargs("PPO", "MlpPolicyy", {"ortho_init": False})
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_policy_kwargs("PPO", "MlpPolicy", [("ortho_init", False)])


def test_actor_critic_net_arch_names_only_pi_and_vf(tmp_path):
    """SB3's MlpExtractor reads "pi" and "vf" and ignores other keys, so
    SAC's net_arch dict built a PPO value network with no hidden layer;
    it is refused instead. SAC's own pi/qf dict stays valid for SAC."""
    from courtside_dynamics.recipes import build_train_config
    from courtside_dynamics.training.algos import validate_policy_kwargs

    validate_policy_kwargs("PPO", "MlpPolicy", {"net_arch": {"pi": [64], "vf": [32]}})
    validate_policy_kwargs("PPO", "MlpPolicy", {"net_arch": {"pi": [64]}})
    validate_policy_kwargs("PPO", "MlpPolicy", {"net_arch": [64, 64]})
    validate_policy_kwargs("SAC", "MlpPolicy", {"net_arch": {"pi": [64], "qf": [64]}})
    with pytest.raises(ValueError, match=r"\['critic'\] are not read by"):
        validate_policy_kwargs("PPO", "MlpPolicy", {"net_arch": {"critic": [64]}})
    # The audit's case: refused when the config is built.
    with pytest.raises(ValueError, match="'qf' is SAC's critic key"):
        build_train_config(
            "BallBalance",
            algo="PPO",
            log_dir=str(tmp_path),
            model_kwargs={
                "n_steps": 512,
                "batch_size": 256,
                "policy_kwargs": {
                    "log_std_init": -1.0,
                    "net_arch": {"pi": [256, 256], "qf": [256, 256]},
                },
            },
        )


def _budget_gate(budget: int, **extra: Any) -> dict[str, Any]:
    return {
        "stages": ({}, {}),
        "metric_key": "m",
        "threshold": 1.0,
        "sustain_evals": 1,
        "stage_eval_budget": budget,
        **extra,
    }


def test_on_policy_gate_budget_must_outlast_the_update_floor(tmp_path):
    """The budget stop has no warm-up for the floor to raise, so a
    budget at or below the floor (10 evaluations here) is refused; one
    past it, a forced advance, a one-stage gate (the final stage has no
    budget) and K = 0 are not. SAC keeps any budget."""
    from courtside_dynamics.training.train import _validate_on_policy_config

    def check(**overrides: Any) -> None:
        _validate_on_policy_config(_on_policy_cfg(tmp_path / "run", **overrides))

    with pytest.raises(ValueError, match="stage_eval_budget 1 would stop"):
        check(performance_gate=_budget_gate(1))
    check(performance_gate=_budget_gate(11))
    check(
        performance_gate=_budget_gate(1, stage_eval_budget_action="advance")
    )
    check(performance_gate={**_budget_gate(1), "stages": ({},)})
    check(performance_gate=_budget_gate(1), min_policy_updates_before_stop=0)
    check(
        algo="SAC",
        eval_freq=100,
        model_kwargs={"n_steps": 1, "batch_size": 1},
        performance_gate=_budget_gate(1),
    )


@pytest.mark.parametrize("algo", ["SAC", "PPO"])
@pytest.mark.parametrize(
    "value, error, match",
    [
        pytest.param(-1, ValueError, "must be nonnegative, got -1", id="negative"),
        pytest.param(2.5, TypeError, "None or an integer, got 2.5", id="float"),
        pytest.param(True, TypeError, "None or an integer, got True", id="bool"),
    ],
)
def test_malformed_degenerate_min_evals_is_refused_before_output(
    tmp_path, algo, value, error, match
):
    """An SAC run used to fail inside the update-floor print with a
    TypeError on None (after config.json), a PPO run to accept -1 as
    "raised from -1": both now refuse it with the other config-only
    checks, before the log_dir exists."""
    log_dir = tmp_path / "run"
    overrides: dict[str, Any] = {"degenerate_min_evals": value, "algo": algo}
    if algo == "SAC":
        overrides["model_kwargs"] = {"learning_starts": 4, "buffer_size": 64}
    with pytest.raises(error, match=match):
        train(_dead_signal_cfg(log_dir, **overrides))
    assert not log_dir.exists()


# -- PPO plan Step 5: zero policy updates are flagged -------------------------


def test_run_summary_flags_zero_policy_updates(tmp_path, capsys):
    from courtside_dynamics.training.artifacts import write_run_summary

    def env_fn():
        raise RuntimeError("no env needed; probe degrades gracefully")

    cfg = TrainConfig(env_fn=env_fn, algo="PPO", log_dir=str(tmp_path))
    for n_updates, flagged in ((3, False), (0, True)):
        write_run_summary(
            cfg,
            str(tmp_path),
            final_mean_reward=1.0,
            final_std_reward=0.5,
            duration_seconds=10.0,
            n_updates=n_updates,
        )
        text = (tmp_path / "stage_summary.txt").read_text()
        assert f"SB3 n_updates:  {n_updates} (optimization epochs" in text
        assert ("WARNING: no update ran" in text) is flagged
        assert ("ran no update" in capsys.readouterr().out) is flagged
    # None (the SAC family) leaves the summary as it was.
    write_run_summary(
        cfg,
        str(tmp_path),
        final_mean_reward=1.0,
        final_std_reward=0.5,
        duration_seconds=10.0,
    )
    assert "SB3 n_updates:" not in (tmp_path / "stage_summary.txt").read_text()


class _StopAtFirstStep(BaseCallback):
    def _on_step(self) -> bool:
        return False


def test_ppo_run_stopped_inside_its_first_rollout_reports_zero_updates(tmp_path):
    """A stop inside the first rollout used to leave a "completed" run
    whose saved models were the initial policy, with nothing to say so."""
    log_dir = tmp_path / "run"
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=12),
        algo="PPO",
        log_dir=str(log_dir),
        n_envs=1,
        seed=0,
        total_timesteps=64,
        eval_freq=64,
        checkpoint_freq=0,
        n_eval_episodes=1,
        record_video=False,
        info_dict_eval=False,
        model_kwargs={"n_steps": 64, "batch_size": 64, "n_epochs": 1},
        extra_callbacks=(_StopAtFirstStep(),),
    )
    model = train(cfg)
    assert model._n_updates == 0
    text = (log_dir / "stage_summary.txt").read_text()
    assert "SB3 n_updates:  0 (optimization epochs over all rollouts)  <-- WARNING" in text
    # A SAC run's summary has no update line (unchanged).
    sac_dir = tmp_path / "sac"
    sac = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=12),
        algo="SAC",
        log_dir=str(sac_dir),
        n_envs=1,
        seed=0,
        total_timesteps=8,
        eval_freq=8,
        checkpoint_freq=0,
        n_eval_episodes=1,
        record_video=False,
        info_dict_eval=False,
        model_kwargs={"buffer_size": 64, "learning_starts": 4},
    )
    train(sac)
    # (Its Training Health section still lists train/n_updates.)
    assert "SB3 n_updates:" not in (sac_dir / "stage_summary.txt").read_text()


# -- PPO plan Step 6: guards counted in policy updates ------------------------


class _Untouched(gym.Wrapper):
    """BallBalance plus a ``touches`` counter that never moves.

    A provably dead task metric: headline selection on it is flat and
    its guard key is always exactly zero, so an armed degenerate-signal
    stop fires at the first evaluation it is allowed to.
    """

    difficulty = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, {**info, "touches": 0}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, "touches": 0}


def _untouched_env() -> gym.Env:
    return _Untouched(BallBalanceEnv(episode_len=12))


def _dead_signal_cfg(log_dir, **overrides) -> TrainConfig:
    """A PPO run whose degenerate stop is armed to fire at once: one
    flat, guard-zero evaluation suffices, an evaluation ends every
    8-step rollout (one worker), and the budget allows 20 of them."""
    fields: dict[str, Any] = {
        "env_fn": _untouched_env,
        "algo": "PPO",
        "log_dir": str(log_dir),
        "n_envs": 1,
        "seed": 0,
        "total_timesteps": 160,
        "eval_freq": 8,
        "checkpoint_freq": 0,
        "n_eval_episodes": 1,
        "record_video": False,
        "headline_key": "touches",
        "best_metric_keys": ("touches_ep_mean",),
        "early_stop_degenerate_evals": 1,
        "degenerate_guard_keys": ("touches_ep_mean",),
        "model_kwargs": {"n_steps": 8, "batch_size": 8, "n_epochs": 1},
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def _eval_info_series(log_dir, metric) -> list[tuple[int, float]]:
    with open(os.path.join(log_dir, "metrics", "eval_info.csv")) as stream:
        rows = list(csv.DictReader(stream))
    return [
        (int(row["timestep"]), float(row["value"]))
        for row in rows
        if row["metric"] == metric
    ]


def test_policy_update_floor_counts_whole_rollouts(tmp_path):
    from courtside_dynamics.training.train import _policy_update_floor_evals

    # Rollout 4 x 64 = 256 env steps, an evaluation every 2 rollouts:
    # 20 updates take ceil(20 * 256 / 512) = 10 evaluations.
    assert _policy_update_floor_evals(_on_policy_cfg(tmp_path)) == 10
    assert (
        _policy_update_floor_evals(
            _on_policy_cfg(tmp_path, min_policy_updates_before_stop=3)
        )
        == 2
    )
    assert (
        _policy_update_floor_evals(
            _on_policy_cfg(tmp_path, min_policy_updates_before_stop=0)
        )
        == 0
    )
    # SB3's default rollout (2,048 per worker) when n_steps is not given.
    default_rollout = _on_policy_cfg(
        tmp_path, model_kwargs={}, eval_freq=8_192, n_envs=4
    )
    default_rollout.model_kwargs.pop("n_steps")
    assert _policy_update_floor_evals(default_rollout) == 20
    # The off-policy family has no rollout to count in.
    assert _policy_update_floor_evals(_on_policy_cfg(tmp_path, algo="SAC")) == 0


@pytest.mark.parametrize(
    "floor_updates, stop_eval, updates_at_stop",
    [
        # The audit's reproduction: with no floor the stop fires at the
        # first evaluation, before any update ran, and the run still
        # reports "completed".
        pytest.param(0, 1, 0, id="no-floor"),
        # Floored at 3 updates (3 evaluations at one rollout each): the
        # guard counts from evaluation 4, which follows the 3rd update.
        pytest.param(3, 4, 3, id="floor-3"),
    ],
)
def test_ppo_degenerate_guard_cannot_fire_before_k_policy_updates(
    tmp_path, capsys, floor_updates, stop_eval, updates_at_stop
):
    log_dir = tmp_path / "run"
    model = train(
        _dead_signal_cfg(log_dir, min_policy_updates_before_stop=floor_updates)
    )
    # n_epochs=1 without target_kl: SB3's count is one per rollout.
    assert model._n_updates == updates_at_stop >= floor_updates
    summary = (log_dir / "stage_summary.txt").read_text()
    assert "degenerate_signal" in summary
    # Every evaluation records the update count behind it; the last one
    # is the stopping evaluation.
    n_updates = _eval_info_series(log_dir, "n_updates")
    assert [step for step, _ in n_updates] == [
        8 * index for index in range(1, stop_eval + 1)
    ]
    assert [value for _, value in n_updates] == [
        float(index) for index in range(stop_eval)
    ]
    assert n_updates[-1][1] >= floor_updates
    config = json.loads((log_dir / "config.json").read_text())
    assert config["train_config"]["min_policy_updates_before_stop"] == floor_updates
    assert config["resolved_training"]["rollout_env_steps"] == 8
    assert config["resolved_training"]["policy_update_floor_evals"] == floor_updates
    raised = "degenerate_min_evals raised from 0 to 3 evaluations"
    assert (raised in capsys.readouterr().out) is (floor_updates > 0)


def _recorded_selection_kwargs(monkeypatch, cfg) -> dict[str, Any]:
    """The kwargs train() builds the selecting evaluator with (the run
    is stopped at its first step: only the wiring matters)."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    train_module = importlib.import_module("courtside_dynamics.training.train")
    built: list[dict[str, Any]] = []

    class _Recording(InfoDictEvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(kwargs)

    monkeypatch.setattr(train_module, "InfoDictEvalCallback", _Recording)
    train(cfg)
    return built[0]


def test_sac_stop_warmups_are_wired_as_before(tmp_path, monkeypatch):
    """The floor is on-policy only: an SAC run's warm-ups are today's,
    including the ones derived from a schedule, and its evaluations
    record no update count."""
    cfg = _dead_signal_cfg(
        tmp_path / "run",
        algo="SAC",
        eval_freq=100,
        total_timesteps=1_000,
        early_stop_degenerate_evals=2,
        early_stop_patience=3,
        env_attr_schedules=(
            {
                "attr_name": "difficulty",
                "start_value": 0.0,
                "end_value": 1.0,
                "hold_until_timesteps": 250,
                "end_timesteps": 650,
            },
        ),
        model_kwargs={"learning_starts": 4, "buffer_size": 64},
        extra_callbacks=(_StopAtFirstStep(),),
    )
    kwargs = _recorded_selection_kwargs(monkeypatch, cfg)
    assert kwargs["degenerate_stop_evals"] == 2
    # ceil(hold 250 / eval 100) = 3; patience max(3, ceil(650 / 100)) = 7.
    assert kwargs["degenerate_min_evals"] == 3
    assert kwargs["early_stop_patience"] == 3
    assert kwargs["early_stop_min_evals"] == 7
    assert kwargs["record_n_updates"] is False
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["resolved_training"]["policy_update_floor_evals"] is None
    # The reachability record reads the warm-ups the callbacks got.
    reachability = config["resolved_training"]["early_stop_reachability"]
    assert reachability["scheduled_evaluations"] == 10
    assert reachability["guards"] == {
        "early_stop_patience": {
            "selection": "headline",
            "warmup_evals": 7,
            "earliest_stop_evaluation": 10,
            "inert": False,
        },
        "degenerate_signal": {
            "warmup_evals": 3,
            "earliest_stop_evaluation": 5,
            "inert": False,
        },
    }
    assert reachability["warnings"] == []


def test_ppo_stop_warmups_are_floored_in_policy_updates(tmp_path, monkeypatch):
    # Rollout 8, an evaluation per rollout: 20 updates = 20 evaluations,
    # above both the schedule-derived and the configured warm-ups.
    cfg = _dead_signal_cfg(
        tmp_path / "run",
        early_stop_patience=3,
        env_attr_schedules=(
            {
                "attr_name": "difficulty",
                "start_value": 0.0,
                "end_value": 1.0,
                "hold_until_timesteps": 24,
                "end_timesteps": 64,
            },
        ),
        extra_callbacks=(_StopAtFirstStep(),),
    )
    kwargs = _recorded_selection_kwargs(monkeypatch, cfg)
    assert kwargs["degenerate_min_evals"] == 20
    assert kwargs["early_stop_min_evals"] == 20
    assert kwargs["record_n_updates"] is True
    # 160 steps at an evaluation per 8-step rollout run 20 evaluations,
    # so both floored stops are past the budget, and the record says so.
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    reachability = config["resolved_training"]["early_stop_reachability"]
    assert reachability["scheduled_evaluations"] == 20
    guards = reachability["guards"]
    assert guards["degenerate_signal"]["warmup_evals"] == 20
    assert guards["early_stop_patience"]["warmup_evals"] == 20
    assert guards["degenerate_signal"]["inert"] is True
    assert guards["early_stop_patience"]["inert"] is True
    assert len(reachability["warnings"]) == 2


def test_reward_selected_ppo_patience_warmup_is_floored(tmp_path, monkeypatch):
    """Without headline selection SB3's StopTrainingOnNoModelImprovement
    owns the patience; its min_evals is floored the same way, an SAC
    run's is the patience itself."""
    train_module = importlib.import_module("courtside_dynamics.training.train")
    built: list[dict[str, Any]] = []

    class _Recording(train_module._StopOnNoRewardImprovement):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(kwargs)

    monkeypatch.setattr(train_module, "_StopOnNoRewardImprovement", _Recording)
    for algo, model_kwargs in (
        ("PPO", {"n_steps": 8, "batch_size": 8, "n_epochs": 1}),
        ("SAC", {"learning_starts": 4, "buffer_size": 64}),
    ):
        train(
            _dead_signal_cfg(
                tmp_path / algo,
                algo=algo,
                headline_key=None,
                best_metric_keys=None,
                early_stop_degenerate_evals=0,
                degenerate_guard_keys=(),
                info_dict_eval=False,
                early_stop_patience=2,
                model_kwargs=model_kwargs,
                extra_callbacks=(_StopAtFirstStep(),),
            )
        )
    ppo, sac = built
    assert ppo["min_evals"] == 20
    assert sac["min_evals"] == 2
    assert ppo["max_no_improvement_evals"] == sac["max_no_improvement_evals"] == 2


def test_sac_eval_info_csv_has_no_update_rows(tmp_path):
    train(_merged_eval_cfg(tmp_path, total_timesteps=200))
    assert _eval_info_series(tmp_path, "steps_alive_ep_mean")
    assert not _eval_info_series(tmp_path, "n_updates")


@pytest.mark.parametrize(
    "overrides, match",
    [
        pytest.param(
            {"min_policy_updates_before_stop": -1},
            "must be a nonnegative integer",
            id="negative",
        ),
        pytest.param(
            {"min_policy_updates_before_stop": True},
            "must be a nonnegative integer",
            id="bool",
        ),
        pytest.param(
            {"algo": "SAC", "min_policy_updates_before_stop": 5},
            "applies to on-policy runs only",
            id="sac-non-default",
        ),
        pytest.param({"vec_env": "threads"}, "vec_env must be one of", id="vec-env"),
    ],
)
def test_worker_and_update_floor_settings_refuse_before_any_output(
    tmp_path, overrides, match
):
    log_dir = tmp_path / "run"
    cfg = _on_policy_cfg(log_dir, **overrides)
    if cfg.algo != "PPO":
        cfg.model_kwargs = {}
    with pytest.raises(ValueError, match=match):
        train(cfg)
    assert not log_dir.exists()


def test_schedules_move_only_at_ppo_rollout_starts(tmp_path):
    """End to end: every 8-step rollout trains at one scheduled value,
    and the values still follow the schedule's clock."""
    seen: list[tuple[int, float]] = []

    class _Recorder(BaseCallback):
        def _on_step(self) -> bool:
            value = self.training_env.env_method(
                "get_wrapper_attr", "difficulty"
            )[0]
            seen.append((self.num_timesteps, float(value)))
            return True

    log_dir = tmp_path / "run"
    train(
        _dead_signal_cfg(
            log_dir,
            total_timesteps=48,
            eval_freq=48,
            early_stop_degenerate_evals=0,
            degenerate_guard_keys=(),
            env_attr_schedules=(
                {
                    "attr_name": "difficulty",
                    "start_value": 0.0,
                    "end_value": 1.0,
                    "hold_until_timesteps": 0,
                    "end_timesteps": 40,
                },
            ),
            extra_callbacks=(_Recorder(),),
        )
    )
    rollouts = [seen[index : index + 8] for index in range(0, 48, 8)]
    values = [{value for _, value in rollout} for rollout in rollouts]
    assert all(len(rollout_values) == 1 for rollout_values in values)
    # The value at each rollout start: value_at(start timestep).
    assert [rollout_values.pop() for rollout_values in values] == [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]


# -- PPO plan Step 7: the reward normalizer discounts at the model's gamma ----


def _pickled_gamma(path) -> float:
    import pickle

    with open(path, "rb") as stream:
        return float(pickle.load(stream).gamma)


@pytest.mark.parametrize(
    "algo, normalize_reward, expected_gamma, source",
    [
        pytest.param("PPO", True, 0.995, "model_gamma", id="ppo-normalized"),
        # Return statistics unused: VecNormalize's default stands.
        pytest.param("PPO", False, 0.99, "vec_normalize_default", id="ppo-raw"),
        # SB3 updates ret_rms even without reward normalization, so the
        # SAC family's pickled normalizer must keep its 0.99.
        pytest.param("SAC", None, 0.99, "vec_normalize_default", id="sac"),
    ],
)
def test_reward_normalizer_gamma_follows_an_on_policy_model(
    tmp_path, algo, normalize_reward, expected_gamma, source
):
    model_kwargs = (
        {"n_steps": 8, "batch_size": 8, "n_epochs": 1, "gamma": 0.995}
        if algo == "PPO"
        else {"learning_starts": 4, "buffer_size": 64, "gamma": 0.995}
    )
    log_dir = tmp_path / "run"
    train(
        _dead_signal_cfg(
            log_dir,
            algo=algo,
            total_timesteps=16,
            eval_freq=8,
            early_stop_degenerate_evals=0,
            degenerate_guard_keys=(),
            normalize_reward=normalize_reward,
            model_kwargs=model_kwargs,
        )
    )
    assert _pickled_gamma(log_dir / "model" / "vec_normalize.pkl") == expected_gamma
    assert (
        _pickled_gamma(log_dir / "model" / "best_vec_normalize.pkl")
        == expected_gamma
    )
    resolved = json.loads((log_dir / "config.json").read_text())["resolved_training"]
    assert resolved["normalize_reward_resolved"] is (algo == "PPO" and normalize_reward)
    assert resolved["vec_normalize"] is True
    assert resolved["vec_normalize_gamma"] == expected_gamma
    assert resolved["vec_normalize_gamma_source"] == source


def test_warm_start_normalizer_takes_the_target_gamma(tmp_path):
    source_dir, env_fn, _ = _make_ppo_warm_start_source(tmp_path)
    target_cfg = TrainConfig(
        env_fn=env_fn,
        algo="PPO",
        log_dir=str(tmp_path / "target"),
        n_envs=1,
        normalize_reward=True,
        normalize_obs_excluded_indices=(0,),
        warm_start=WarmStartConfig(source_dir),
    )
    artifacts = _prepare_warm_start(target_cfg)
    assert artifacts is not None
    for gamma, expected in ((0.995, 0.995), (None, 0.99)):
        loaded = _load_warm_start_normalizer(
            make_vec_env(env_fn, n_envs=1, seed=3),
            artifacts,
            target_cfg,
            norm_reward=True,
            gamma=gamma,
        )
        try:
            # None keeps what the source pickled (VecNormalize's 0.99).
            assert loaded.gamma == expected
        finally:
            loaded.close()


# -- PPO plan Step 8: on-policy MLP runs train on the CPU ---------------------


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({}, ("cpu", "on_policy_mlp_cpu_default"), id="ppo-mlp"),
        pytest.param(
            {"policy": "CnnPolicy"}, ("auto", "sb3_default"), id="ppo-cnn"
        ),
        pytest.param(
            {"policy": "MultiInputPolicy"},
            ("auto", "sb3_default"),
            id="ppo-multi-input",
        ),
        pytest.param(
            {"require_device": "cuda"}, ("auto", "sb3_default"), id="ppo-require-cuda"
        ),
        pytest.param(
            {"model_kwargs": {"device": "cuda:1"}},
            ("cuda:1", "model_kwargs"),
            id="ppo-explicit",
        ),
        pytest.param({"algo": "SAC"}, ("auto", "sb3_default"), id="sac"),
        pytest.param({"algo": "DEMOSAC"}, ("auto", "sb3_default"), id="demosac"),
    ],
)
def test_device_request_defaults_on_policy_mlp_runs_to_cpu(
    tmp_path, overrides, expected
):
    from courtside_dynamics.training.train import _device_request

    assert _device_request(_on_policy_cfg(tmp_path, **overrides)) == expected


def test_ppo_trains_on_cpu_when_cuda_is_available(tmp_path, monkeypatch):
    """With CUDA reported available, SB3's "auto" would place the policy
    on the GPU (on this CPU-only build, it would crash moving there):
    the on-policy MLP default keeps PPO on the CPU, while an SAC run
    still asks SB3 for "auto"."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    train_module = importlib.import_module("courtside_dynamics.training.train")
    built_kwargs: dict[str, dict[str, Any]] = {}
    real_build = train_module._build_algo

    class _BuildStopped(Exception):
        pass

    def recording_build(name, env, log_dir, *, policy="MlpPolicy", **kwargs):
        built_kwargs[name.upper()] = dict(kwargs)
        if name.upper() == "SAC":
            # SB3's "auto" would now pick the (absent) GPU: stop here.
            raise _BuildStopped
        return real_build(name, env, log_dir, policy=policy, **kwargs)

    monkeypatch.setattr(train_module, "_build_algo", recording_build)
    log_dir = tmp_path / "ppo"
    model = train(
        _dead_signal_cfg(
            log_dir,
            total_timesteps=8,
            early_stop_degenerate_evals=0,
            degenerate_guard_keys=(),
        )
    )
    assert model.device.type == "cpu"
    assert built_kwargs["PPO"]["device"] == "cpu"
    config = json.loads((log_dir / "config.json").read_text())
    assert config["resolved_training"]["device_request"] == "cpu"
    assert (
        config["resolved_training"]["device_request_source"]
        == "on_policy_mlp_cpu_default"
    )
    assert config["resolved_model"]["device"] == "cpu"
    # The caller's model_kwargs are recorded as given.
    assert "device" not in config["train_config"]["model_kwargs"]

    with pytest.raises(_BuildStopped):
        train(
            _dead_signal_cfg(
                tmp_path / "sac",
                algo="SAC",
                total_timesteps=8,
                early_stop_degenerate_evals=0,
                degenerate_guard_keys=(),
                model_kwargs={"learning_starts": 4, "buffer_size": 64},
            )
        )
    assert "device" not in built_kwargs["SAC"]


# -- PPO plan Step 9: warm-start exploration ---------------------------------


def test_warm_start_config_validates_reset_log_std(tmp_path):
    assert WarmStartConfig(tmp_path).reset_log_std is False
    with pytest.raises(TypeError, match="reset_log_std must be a bool"):
        WarmStartConfig(tmp_path, reset_log_std=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("reset_log_std", [False, True])
def test_ppo_warm_start_transfers_or_resets_log_std(tmp_path, reset_log_std):
    source_dir, env_fn, source_state = _make_ppo_warm_start_source(tmp_path)
    # The fixture moved the policy's first parameter, log_std, to 0.125.
    assert torch.equal(
        source_state["log_std"], torch.full_like(source_state["log_std"], 0.125)
    )
    capture = _CaptureWarmStart()
    target_dir = tmp_path / "target"
    train(
        TrainConfig(
            env_fn=env_fn,
            algo="PPO",
            total_timesteps=8,
            log_dir=str(target_dir),
            n_envs=1,
            seed=13,
            eval_freq=8,
            checkpoint_freq=0,
            record_video=False,
            info_dict_eval=False,
            n_eval_episodes=1,
            normalize_reward=True,
            normalize_obs_excluded_indices=(0,),
            warm_start=WarmStartConfig(source_dir, reset_log_std=reset_log_std),
            model_kwargs={
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
                "policy_kwargs": {"log_std_init": -1.5},
            },
            extra_callbacks=(capture,),
        )
    )
    assert capture.policy_state is not None
    expected_log_std = (
        torch.full_like(source_state["log_std"], -1.5)
        if reset_log_std
        else source_state["log_std"]
    )
    torch.testing.assert_close(capture.policy_state["log_std"], expected_log_std)
    # Every other tensor is the source's.
    for name, expected in source_state.items():
        if name != "log_std":
            torch.testing.assert_close(capture.policy_state[name], expected)

    config = json.loads((target_dir / "config.json").read_text())
    assert config["train_config"]["warm_start"]["reset_log_std"] is reset_log_std
    initialization = config["initialization"]
    assert initialization["reset_log_std"] is reset_log_std
    assert ("policy.log_std" in initialization["reset"]) is reset_log_std
    assert ("policy.log_std" in initialization["transferred"]) is not reset_log_std
    if reset_log_std:
        assert initialization["log_std_reset_value"] == -1.5
    else:
        assert "log_std_reset_value" not in initialization


def test_ppo_warm_start_refuses_a_temperature_skip(tmp_path):
    """PPO has no temperature: transfer_log_ent_coef=False used to be a
    silent no-op; it is refused, pointing at reset_log_std."""
    log_dir = tmp_path / "run"
    cfg = _on_policy_cfg(
        log_dir,
        warm_start=WarmStartConfig(tmp_path / "missing", transfer_log_ent_coef=False),
    )
    with pytest.raises(ValueError, match=r"reset_log_std=True"):
        train(cfg)
    assert not log_dir.exists()


@pytest.mark.parametrize("algo", ["SAC", "DEMOSAC"])
def test_sac_family_warm_start_refuses_reset_log_std(tmp_path, algo):
    log_dir = tmp_path / "run"
    cfg = _on_policy_cfg(
        log_dir,
        algo=algo,
        warm_start=WarmStartConfig(tmp_path / "missing", reset_log_std=True),
    )
    cfg.model_kwargs = {}
    with pytest.raises(ValueError, match="reset_log_std applies to on-policy"):
        train(cfg)
    assert not log_dir.exists()


# -- PPO plan Step 13: opt-in subprocess training workers --------------------


def _rollout(vec_env, steps: int, seed: int) -> list[tuple[Any, ...]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[Any, ...]] = [(vec_env.reset(),)]
    for _ in range(steps):
        actions = rng.uniform(
            -1.0, 1.0, size=(vec_env.num_envs, *vec_env.action_space.shape)
        ).astype(np.float32)
        obs, rewards, dones, infos = vec_env.step(actions)
        terminal = [info.get("terminal_observation") for info in infos]
        out.append((obs, rewards, dones, terminal))
    return out


def test_subproc_workers_return_exactly_what_dummy_workers_return():
    """Observations, rewards (float32, as DummyVecEnv's buffers hold
    them), dones and terminal observations are identical step for step,
    across episode ends and auto-resets."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from courtside_dynamics.training.train import DummyDtypeSubprocVecEnv

    def env_fn():
        return BallBalanceEnv(episode_len=12)

    dummy = make_vec_env(env_fn, n_envs=2, seed=0)
    try:
        expected = _rollout(dummy, 40, seed=1)
    finally:
        dummy.close()
    subproc = make_vec_env(
        env_fn, n_envs=2, seed=0, vec_env_cls=DummyDtypeSubprocVecEnv
    )
    try:
        actual = _rollout(subproc, 40, seed=1)
    finally:
        subproc.close()
    assert sum(bool(np.any(step[2])) for step in expected[1:]) >= 3
    (reset_expected,), (reset_actual,) = expected[0], actual[0]
    assert reset_actual.dtype == reset_expected.dtype
    np.testing.assert_array_equal(reset_actual, reset_expected)
    for (obs, rewards, dones, terminal), (obs2, rewards2, dones2, terminal2) in zip(
        expected[1:], actual[1:], strict=True
    ):
        assert obs2.dtype == obs.dtype and rewards2.dtype == rewards.dtype
        np.testing.assert_array_equal(obs2, obs)
        np.testing.assert_array_equal(rewards2, rewards)
        np.testing.assert_array_equal(dones2, dones)
        for left, right in zip(terminal, terminal2, strict=True):
            assert (left is None) == (right is None)
            if left is not None:
                np.testing.assert_array_equal(right, left)
    # Why the subclass exists: stock SubprocVecEnv stacks the workers'
    # Python-float rewards as float64, which VecNormalize's return
    # statistics would then see.
    stock = make_vec_env(env_fn, n_envs=2, seed=0, vec_env_cls=SubprocVecEnv)
    try:
        stock_rewards = _rollout(stock, 1, seed=1)[1][1]
    finally:
        stock.close()
    assert stock_rewards.dtype == np.float64
    np.testing.assert_array_equal(stock_rewards.astype(np.float32), expected[1][1])


def test_ppo_trains_identically_on_subproc_workers(tmp_path):
    """A whole PPO run (reward normalization on) on subprocess workers
    equals the in-process run: policy, normalizer and monitor logs."""
    import pickle

    class _WorkerKind(BaseCallback):
        def _on_training_start(self) -> None:
            self.kind = type(self.training_env.unwrapped).__name__

        def _on_step(self) -> bool:
            return True

    runs = {}
    kinds = {}
    for vec_env in ("dummy", "subproc"):
        log_dir = tmp_path / vec_env
        worker_kind = _WorkerKind()
        model = train(
            _dead_signal_cfg(
                log_dir,
                n_envs=2,
                total_timesteps=64,
                eval_freq=32,
                early_stop_degenerate_evals=0,
                degenerate_guard_keys=(),
                vec_env=vec_env,
                model_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 2},
                extra_callbacks=(worker_kind,),
            )
        )
        kinds[vec_env] = worker_kind.kind
        with open(log_dir / "model" / "vec_normalize.pkl", "rb") as stream:
            normalizer = pickle.load(stream)
        monitors = {}
        for path in sorted((log_dir / "metrics" / "monitor").glob("*.monitor.csv")):
            with open(path) as stream:
                stream.readline()
                monitors[path.name] = [
                    {k: v for k, v in row.items() if k != "t"}
                    for row in csv.DictReader(stream)
                ]
        runs[vec_env] = (
            {k: v.detach().clone() for k, v in model.policy.state_dict().items()},
            normalizer,
            monitors,
            json.loads((log_dir / "config.json").read_text()),
        )
    dummy_state, dummy_norm, dummy_monitors, _ = runs["dummy"]
    sub_state, sub_norm, sub_monitors, config = runs["subproc"]
    assert dummy_state.keys() == sub_state.keys()
    for name in dummy_state:
        assert torch.equal(dummy_state[name], sub_state[name]), name
    for stats in ("obs_rms", "ret_rms"):
        left, right = getattr(dummy_norm, stats), getattr(sub_norm, stats)
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.var, right.var)
        assert left.count == right.count
    assert dummy_monitors and dummy_monitors == sub_monitors
    assert kinds == {"dummy": "DummyVecEnv", "subproc": "DummyDtypeSubprocVecEnv"}
    assert config["train_config"]["vec_env"] == "subproc"


# -- PPO plan Step 10: loaders follow the recorded algorithm ------------------


def _write_recorded_algo(run_dir, algo) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train_config": {"algo": algo}})
    )


def test_locate_run_config_follows_the_run_layout(tmp_path):
    """Every place the run layout puts a model resolves to its run's
    config.json, and nowhere else does."""
    from courtside_dynamics.training.algos import locate_run_config

    run = tmp_path / "run"
    _write_recorded_algo(run, "PPO")
    config = str(run / "config.json")
    for relative in (
        "best_model.zip",  # pre-0.14 flat layout
        "best_model",  # SB3's extension-less stem
        "model/best_model.zip",
        "model/final_model.zip",
        "checkpoints/rl_model_64_steps.zip",
        "model/stage_bests/stage_00/best_model.zip",
        "stage_bests/stage_01/best_model.zip",  # pre-0.14 stage archive
    ):
        assert locate_run_config(run / relative) == config, relative
    assert locate_run_config(run) == config
    # Outside the layout's model folders, a config.json above is not
    # the checkpoint's record.
    assert locate_run_config(run / "elsewhere" / "best_model.zip") is None
    assert locate_run_config(run / "metrics" / "monitor" / "x.zip") is None
    assert locate_run_config(tmp_path / "loose.zip") is None


def test_recorded_run_algo_refuses_a_record_that_says_nothing(tmp_path):
    from courtside_dynamics.training.algos import recorded_run_algo

    assert recorded_run_algo(tmp_path / "model" / "best_model.zip") is None
    _write_recorded_algo(tmp_path / "ok", "DemoSAC")
    assert recorded_run_algo(tmp_path / "ok" / "model" / "best_model.zip") == (
        "DEMOSAC"
    )
    blank = tmp_path / "blank"
    blank.mkdir()
    (blank / "config.json").write_text(json.dumps({"train_config": {}}))
    with pytest.raises(ValueError, match="records no train_config.algo"):
        recorded_run_algo(blank / "model" / "best_model.zip")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="could not read"):
        recorded_run_algo(broken / "model" / "best_model.zip")


def _saved_model(algo_cls, path, **kwargs):
    venv = make_vec_env(lambda: BallBalanceEnv(episode_len=8), n_envs=1)
    try:
        model = algo_cls("MlpPolicy", venv, seed=0, device="cpu", **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
    finally:
        venv.close()
    return model


def test_load_run_model_loads_each_family_through_its_root_class(
    tmp_path, monkeypatch
):
    """SAC and DemoSAC runs load through SAC.load (DemoSAC's inference
    path: no demo library needed), PPO runs through PPO.load; an
    explicit algorithm must agree with the record."""
    from stable_baselines3 import PPO, SAC

    from courtside_dynamics.training.algos import load_run_model
    from courtside_dynamics.training.demo_sac import DemoSAC

    sac_path = tmp_path / "sac" / "model" / "best_model.zip"
    _saved_model(SAC, sac_path, buffer_size=100, learning_starts=0)
    _write_recorded_algo(tmp_path / "sac", "SAC")
    demo_path = tmp_path / "demo" / "model" / "best_model.zip"
    _saved_model(DemoSAC, demo_path, buffer_size=100, learning_starts=0)
    _write_recorded_algo(tmp_path / "demo", "DemoSAC")
    ppo_path = tmp_path / "ppo" / "checkpoints" / "rl_model_8_steps.zip"
    saved_ppo = _saved_model(PPO, ppo_path, n_steps=8, batch_size=8)
    _write_recorded_algo(tmp_path / "ppo", "PPO")

    # SAC keeps exactly the call it always made.
    calls = []
    real_sac_load = SAC.load

    def spy(path, **kwargs):
        calls.append((path, kwargs))
        return real_sac_load(path, **kwargs)

    monkeypatch.setattr(SAC, "load", spy)
    assert type(load_run_model(sac_path, device="cpu")) is SAC
    assert type(load_run_model(demo_path, "demosac", device="cpu")) is SAC
    assert calls == [
        (str(sac_path), {"device": "cpu"}),
        (str(demo_path), {"device": "cpu"}),
    ]

    ppo = load_run_model(ppo_path, device="cpu")
    assert type(ppo) is PPO
    for name, tensor in saved_ppo.policy.state_dict().items():
        assert torch.equal(tensor, ppo.policy.state_dict()[name]), name

    with pytest.raises(ValueError, match="requested algorithm SAC does not match"):
        load_run_model(ppo_path, "SAC")
    # DemoSAC and SAC share a family but are different records.
    with pytest.raises(ValueError, match="does not match the algorithm DEMOSAC"):
        load_run_model(demo_path, "SAC")

    loose = tmp_path / "loose.zip"
    loose.write_bytes(ppo_path.read_bytes())
    with pytest.raises(FileNotFoundError, match="pass the algorithm"):
        load_run_model(loose)
    assert type(load_run_model(loose, "PPO", device="cpu")) is PPO
    # Outside any run, the zip's own policy class still refuses a
    # wrong family (SAC.load used to fail inside SB3 with an
    # AttributeError on 'actor'), with or without the .zip suffix.
    for path in (loose, str(loose)[: -len(".zip")]):
        with pytest.raises(
            ValueError,
            match=r"ActorCriticPolicy, a PPO-family policy, so it cannot load "
            r"as SAC \(family SAC\)",
        ):
            load_run_model(path, "SAC")
    loose_sac = tmp_path / "loose_sac.zip"
    loose_sac.write_bytes(sac_path.read_bytes())
    with pytest.raises(ValueError, match="a SAC-family policy, so it cannot load as PPO"):
        load_run_model(loose_sac, "PPO")
    assert type(load_run_model(loose_sac, "DemoSAC", device="cpu")) is SAC


def test_notebook_best_model_loader_follows_the_record(tmp_path):
    """record_best_model_video's loader: algo=None reads the record, a
    disagreeing algo raises (the old "SAC" default could not load a PPO
    run)."""
    import inspect as _inspect

    from stable_baselines3 import PPO

    from courtside_dynamics.notebook_utils import (
        _load_best_model,
        record_best_model_video,
    )

    _saved_model(PPO, tmp_path / "model" / "best_model.zip", n_steps=8, batch_size=8)
    _write_recorded_algo(tmp_path, "PPO")
    assert type(_load_best_model(str(tmp_path))) is PPO
    assert type(_load_best_model(str(tmp_path), "ppo")) is PPO
    with pytest.raises(ValueError, match="does not match"):
        _load_best_model(str(tmp_path), "SAC")
    default = _inspect.signature(record_best_model_video).parameters["algo"]
    assert default.default is None


# -- PPO plan Step 11: provenance and on-policy training health ---------------


def test_model_info_records_on_policy_stability_settings(tmp_path):
    """PPO's resolved_model gains target_kl, sde_sample_freq,
    log_std_init, ortho_init and, once training ends, its progress;
    SAC's record is untouched (no target_kl attribute, and the on-policy
    keys are gated)."""
    from stable_baselines3 import PPO, SAC

    from courtside_dynamics.training.artifacts import _model_info

    ppo = _saved_model(
        PPO,
        tmp_path / "ppo.zip",
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        target_kl=0.02,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"log_std_init": -2.0, "ortho_init": False},
    )
    info = _model_info(ppo)
    assert info["hyperparameters"]["target_kl"] == 0.02
    assert info["hyperparameters"]["sde_sample_freq"] == 4
    assert info["log_std_init"] == -2.0
    assert info["ortho_init"] is False
    assert "n_updates" not in info and "timesteps" not in info
    venv = make_vec_env(lambda: BallBalanceEnv(episode_len=8), n_envs=1)
    try:
        ppo.set_env(venv)
        ppo.learn(total_timesteps=12)
    finally:
        venv.close()
    progressed = _model_info(ppo, progress=True)
    assert progressed["n_updates"] == ppo._n_updates == 2
    assert progressed["timesteps"] == {"trained": 16, "budget": 12}

    sac = _saved_model(SAC, tmp_path / "sac.zip", buffer_size=100, use_sde=True)
    sac_info = _model_info(sac, progress=True)
    assert "target_kl" not in sac_info["hyperparameters"]
    assert "sde_sample_freq" not in sac_info["hyperparameters"]
    for key in ("log_std_init", "ortho_init", "n_updates", "timesteps"):
        assert key not in sac_info, key


def _humanoid_shaped_series(rows: int = 40) -> dict[str, list[float]]:
    """progress.csv rows as the 2026-10-16 humanoid PPO probe read: no
    target_kl cut, approx_kl 0.22-1.71, clip_fraction 0.72-0.83, a value
    head that ends worse than the mean, and a log std that drifted up
    from log_std_init -2 under the entropy bonus."""
    fraction = [index / (rows - 1) for index in range(rows)]
    return {
        "train/approx_kl": [0.22 + 1.49 * f for f in fraction],
        "train/clip_fraction": [0.72 + 0.11 * f for f in fraction],
        "train/explained_variance": [0.4 - 0.8 * f for f in fraction],
        "train/std": [float(np.exp(-2.0 + 1.6 * f)) for f in fraction],
    }


def test_on_policy_health_flags_fire_on_a_humanoid_shaped_run():
    from courtside_dynamics.training.artifacts import on_policy_health_flags

    flags = on_policy_health_flags(
        _humanoid_shaped_series(), target_kl=0.02, log_std_init=-2.0
    )
    assert [flag.split(" ", 1)[0] for flag in flags] == [
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "log",
    ]
    assert "> 0.05" in flags[0]
    assert "drifted +1.6 nats" in flags[3]
    # A healthy run (the BallBalance profile's measured approx_kl
    # 0.011-0.014) raises nothing; a looser target_kl raises the limit.
    healthy = {
        "train/approx_kl": [0.011, 0.014, 0.012, 0.013],
        "train/clip_fraction": [0.1, 0.12, 0.11, 0.09],
        "train/explained_variance": [-0.5, 0.2, 0.6, 0.8],
        "train/std": [0.36, 0.35, 0.33, 0.30],
    }
    assert on_policy_health_flags(healthy, target_kl=0.03, log_std_init=-1.0) == []
    assert on_policy_health_flags(
        {"train/approx_kl": [0.07]}, target_kl=0.05, log_std_init=None
    ) == []
    assert on_policy_health_flags(
        {"train/approx_kl": [0.07]}, target_kl=None, log_std_init=None
    )
    # Only the last quarter counts: an early spike does not flag.
    assert on_policy_health_flags(
        {"train/explained_variance": [-3.0, -2.0, -1.0, 0.5]},
        target_kl=None,
        log_std_init=None,
    ) == []
    # A non-finite window is flagged, never averaged away; an unknown
    # log_std_init skips the drift flag.
    nan_flags = on_policy_health_flags(
        {"train/approx_kl": [0.01, float("nan")], "train/std": [5.0]},
        target_kl=0.02,
        log_std_init=None,
    )
    assert nan_flags == ["approx_kl: non-finite values in the last quarter"]


def _write_progress_csv(log_dir, series) -> None:
    from courtside_dynamics.training.artifacts import artifact_path

    keys = sorted(series)
    rows = max(len(values) for values in series.values())
    with open(artifact_path(str(log_dir), "progress_csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time/total_timesteps", *keys])
        for index in range(rows):
            writer.writerow(
                [index * 2048]
                + [
                    series[key][index] if index < len(series[key]) else ""
                    for key in keys
                ]
            )


def test_run_summary_flags_on_policy_health_against_the_record(tmp_path, capsys):
    """The summary reads target_kl and log_std_init from the run's own
    resolved_model; the SAC family's summary has no flag block."""
    from courtside_dynamics.training.artifacts import write_run_summary

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        algo="PPO",
        log_dir=str(tmp_path),
        total_timesteps=10_000,
    )
    write_run_config(cfg, str(tmp_path))
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["resolved_model"] = {
        "hyperparameters": {"target_kl": 0.02},
        "log_std_init": -2.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    _write_progress_csv(tmp_path, _humanoid_shaped_series())

    def summary(**kwargs):
        write_run_summary(
            cfg,
            str(tmp_path),
            final_mean_reward=None,
            final_std_reward=None,
            duration_seconds=1.0,
            actual_timesteps=12_288,
            **kwargs,
        )
        return (tmp_path / "stage_summary.txt").read_text()

    text = summary(n_updates=60)
    assert "Timesteps:      12,288 of 10,000 budget (+2,288: the last rollout ran past it)" in text
    assert "On-policy flags (means over the last quarter):" in text
    assert text.count("    FLAG ") == 4
    assert "FLAG approx_kl" in text and "FLAG log std" in text
    printed = capsys.readouterr().out
    assert printed.count("[train] WARNING: on-policy training health:") == 4

    # The off-policy summary (n_updates None) keeps its old shape.
    text = summary(n_updates=None)
    assert "On-policy flags" not in text and "FLAG" not in text
    assert "Timesteps:      10,000\n" in text
    assert "train/approx_kl:" in text


def test_epochs_per_update_flag_catches_a_target_kl_cut_in_epoch_one():
    """The gSDE profiles' quick tests ended every update in its first
    epoch (SB3 n_updates one per rollout at n_epochs 10) while the mean
    approx_kl looked healthy; the per-update epochs say so."""
    from courtside_dynamics.training.artifacts import (
        epochs_per_update,
        on_policy_health_flags,
    )

    cut = {"train/n_updates": [1.0, 2.0, 3.0, 4.0], "train/approx_kl": [0.013] * 4}
    assert epochs_per_update(cut) == [1.0, 1.0, 1.0, 1.0]
    flags = on_policy_health_flags(cut, target_kl=0.02, log_std_init=None, n_epochs=10)
    assert flags == [
        "epochs per update 1 of n_epochs 10: target_kl 0.02 ends the updates "
        "early, so the configured epochs barely act (calibrate the learning "
        "rate against target_kl)"
    ]
    # Every epoch ran (BallBalance's 70 over 7 rollouts), or only early
    # updates were cut (the last quarter decides), or n_epochs unknown.
    full = {"train/n_updates": [10.0, 20.0, 30.0, 40.0]}
    assert on_policy_health_flags(full, target_kl=0.03, log_std_init=None, n_epochs=10) == []
    late = {"train/n_updates": [1.0, 2.0, 3.0, 13.0]}
    assert epochs_per_update(late) == [1.0, 1.0, 1.0, 10.0]
    assert on_policy_health_flags(late, target_kl=0.02, log_std_init=None, n_epochs=10) == []
    assert on_policy_health_flags(cut, target_kl=0.02, log_std_init=None) == []
    assert epochs_per_update({}) == []


def _write_eval_artifacts(log_dir, rows) -> None:
    """evaluations.npz and eval_info.csv for (step, mean reward, SB3
    n_updates) evaluations, as the reward and info streams write them."""
    from courtside_dynamics.training.artifacts import artifact_path

    npz_path = artifact_path(str(log_dir), "evaluations")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    results = np.array([[reward, reward] for _, reward, _ in rows])
    np.savez(
        npz_path,
        timesteps=np.array([step for step, _, _ in rows]),
        results=results,
        ep_lengths=np.ones_like(results),
    )
    with open(artifact_path(str(log_dir), "eval_info_csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "metric", "value"])
        for step, reward, updates in rows:
            writer.writerow([step, "episode_reward_mean", reward])
            writer.writerow([step, "n_updates", float(updates)])


def test_run_summary_flags_a_best_model_selected_before_any_update(
    tmp_path, capsys
):
    """An evaluation runs before its rollout's update, so a best chosen
    at SB3 n_updates 0 is the untrained initial policy: the BallBalance
    PPO quick test kept it (438 at 4,096 steps, then down to 206) and
    its summary advised deploying it."""
    from courtside_dynamics.training.artifacts import write_run_summary

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        algo="PPO",
        log_dir=str(tmp_path),
        total_timesteps=12_288,
    )
    warning = (
        "  <-- WARNING: selected before any update; best_model.zip is the "
        "untrained initial policy"
    )

    def summary(*, n_updates: int | None = 30) -> str:
        write_run_summary(
            cfg,
            str(tmp_path),
            final_mean_reward=206.0,
            final_std_reward=0.0,
            duration_seconds=1.0,
            actual_timesteps=12_288,
            n_updates=n_updates,
        )
        return (tmp_path / "stage_summary.txt").read_text()

    # Reward selection: best_model.zip is the reward stream's best.
    _write_eval_artifacts(
        tmp_path, [(4_096, 438.0, 0), (8_192, 312.3, 10), (12_288, 183.7, 20)]
    )
    text = summary()
    assert (
        "Best eval:      438.000 +/- 0.000 (at 4,096 steps, SB3 n_updates 0)"
        f"  [periodic eval series]{warning}\n"
    ) in text
    assert (
        "<-- policy regressed after best, but the best scored the untrained "
        "initial policy (SB3 n_updates 0)"
    ) in text
    assert "deploy best_model.zip" not in text
    assert (
        "[train] WARNING: best_model.zip was selected at 4,096 steps, before "
        "any PPO update (SB3 n_updates 0)"
    ) in capsys.readouterr().out

    # A trained best: its count, no warning.
    _write_eval_artifacts(
        tmp_path, [(4_096, 312.3, 0), (8_192, 438.0, 10), (12_288, 183.7, 20)]
    )
    text = summary()
    assert "(at 8,192 steps, SB3 n_updates 10)  [periodic eval series]\n" in text
    assert "WARNING: selected before" not in text
    assert "deploy best_model.zip" in text
    assert "best_model.zip was selected" not in capsys.readouterr().out

    # The SAC family (n_updates None) reads as before.
    _write_eval_artifacts(tmp_path, [(4_096, 438.0, 0), (8_192, 312.3, 10)])
    text = summary(n_updates=None)
    assert "(at 4,096 steps)  [periodic eval series]\n" in text
    assert "WARNING" not in text


def test_run_summary_reads_the_headline_best_from_its_meta(tmp_path, capsys):
    """Under headline selection best_model.zip is best_model_meta.json's
    checkpoint: its recorded n_updates (or, in an older meta, the
    eval_info.csv row at its step) decides the warning."""
    from courtside_dynamics.training.artifacts import artifact_path, write_run_summary

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        algo="PPO",
        log_dir=str(tmp_path),
        total_timesteps=12_288,
        headline_key="touches",
    )
    _write_eval_artifacts(tmp_path, [(4_096, 438.0, 0), (8_192, 312.3, 10)])
    meta_path = artifact_path(str(tmp_path), "best_model_meta")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    def summary(meta: dict[str, Any]) -> str:
        with open(meta_path, "w") as stream:
            json.dump(meta, stream)
        write_run_summary(
            cfg,
            str(tmp_path),
            final_mean_reward=300.0,
            final_std_reward=0.0,
            duration_seconds=1.0,
            actual_timesteps=12_288,
            n_updates=20,
        )
        return (tmp_path / "stage_summary.txt").read_text()

    selected = {
        "timestep": 4_096,
        "selection_keys": ["touches_ep_mean"],
        "selection_values": {"touches_ep_mean": 0.0},
    }
    text = summary({**selected, "n_updates": 0})
    assert (
        "Best model:     step 4,096, SB3 n_updates 0 (touches_ep_mean 0.00)"
        "  [task-metric selection]  <-- WARNING: selected before any update"
    ) in text
    # The reward stream's best is not best_model.zip here: no warning
    # on its line, one in all.
    assert text.count("WARNING") == 1
    assert "best_model.zip was selected at 4,096 steps" in capsys.readouterr().out

    # An older meta without the count falls back to eval_info.csv.
    text = summary({**selected, "timestep": 8_192})
    assert "Best model:     step 8,192, SB3 n_updates 10 (touches_ep_mean" in text
    assert "WARNING: selected before" not in text


def test_run_summary_states_the_resolved_reward_normalization(tmp_path):
    """static-9: PPO's normalize_reward=None resolves to on, discounted
    at the model's gamma; config.json recorded it, the summary now says
    it. The SAC summary has no such line."""
    from courtside_dynamics.training.artifacts import write_run_summary

    for algo, n_updates in (("PPO", 8), ("SAC", None)):
        log_dir = tmp_path / algo
        cfg = TrainConfig(
            env_fn=lambda: BallBalanceEnv(episode_len=8),
            algo=algo,
            log_dir=str(log_dir),
            model_kwargs={"gamma": 0.995},
        )
        write_run_config(cfg, str(log_dir))
        write_run_summary(
            cfg,
            str(log_dir),
            final_mean_reward=None,
            final_std_reward=None,
            duration_seconds=1.0,
            n_updates=n_updates,
        )
        text = (log_dir / "stage_summary.txt").read_text()
        line = (
            "reward_norm        on (normalize_reward=None), normalizer gamma "
            "0.995 (the model's gamma)"
        )
        assert (line in text) is (algo == "PPO"), text
        assert ("reward_norm" in text) is (algo == "PPO")


@pytest.mark.parametrize(
    ("algo", "family", "name"),
    [("SAC", "SAC", "SAC"), ("demosac", "SAC", "DemoSAC"), ("PPO", "PPO", "PPO")],
)
def test_config_json_records_algorithm_and_comparability_key(
    tmp_path, algo, family, name
):
    import stable_baselines3

    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        algo=algo,
        log_dir=str(tmp_path),
    )
    write_run_config(cfg, str(tmp_path))
    payload = json.loads((tmp_path / "config.json").read_text())
    assert payload["algorithm"] == {
        "name": name,
        "family": family,
        "implementation": f"sb3=={stable_baselines3.__version__}",
    }
    # Without a recipe the task is the env class; SAC and PPO runs of
    # one task never share a key.
    assert payload["comparability_key"] == {
        "task": "BallBalanceEnv",
        "eval_protocol": None,
        "algo_family": family,
    }
    recipe_cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(episode_len=8),
        algo=algo,
        log_dir=str(tmp_path / "recipe"),
        recipe_name="BallBalance",
    )
    write_run_config(recipe_cfg, str(tmp_path / "recipe"))
    recorded = json.loads((tmp_path / "recipe" / "config.json").read_text())
    assert recorded["comparability_key"]["task"] == "BallBalance"


# -- PPO plan Step 12: tests that cross the whole path ------------------------


def _family_e2e_cfg(log_dir, algo: str, **overrides) -> TrainConfig:
    """A tiny BallBalance run (episode_len 12, one worker) through the
    whole trainer: headline selection with confirm_best and a paired
    eval_seed, the final-config stream, VecNormalize checkpoints, a
    one-stage performance gate, a patience stop and the closing
    evaluation. Each family gets its own update geometry: SAC updates
    every step after 16, PPO every 8-step rollout."""
    if algo == "PPO":
        model_kwargs = {"n_steps": 8, "batch_size": 8, "n_epochs": 2}
    else:
        model_kwargs = {"learning_starts": 16, "buffer_size": 500, "batch_size": 16}
    fields: dict[str, Any] = {
        "env_fn": _steps_alive_env,
        "algo": algo,
        "log_dir": str(log_dir),
        "n_envs": 1,
        "seed": 0,
        "total_timesteps": 1_024,
        "eval_freq": 32,
        "checkpoint_freq": 64,
        "video_freq": 0,
        "record_video": False,
        "n_eval_episodes": 2,
        "info_dict_eval": True,
        "headline_key": "steps_alive",
        "confirm_best_eval": True,
        "eval_seed": 4_000,
        "final_info_eval": True,
        "early_stop_patience": 2,
        "performance_gate": {
            "metric_key": "steps_alive_ep_mean",
            "threshold": 1e9,
            "sustain_evals": 1,
            "stages": ({"episode_len": 12},),
        },
        "model_kwargs": model_kwargs,
    }
    fields.update(overrides)
    return TrainConfig(**fields)


@pytest.mark.parametrize("algo", ["SAC", "PPO"])
def test_train_runs_each_family_end_to_end(tmp_path, algo):
    """Both families through every stage of train(): selection,
    checkpoints, the gate, the stop and the closing evaluation, with
    the artifacts, records and loaders each family needs."""
    import pickle

    from courtside_dynamics.notebook_utils import _load_best_model
    from courtside_dynamics.training.algos import load_run_model, resolve_algo

    model = train(_family_e2e_cfg(tmp_path, algo))
    on_policy = algo == "PPO"

    # The headline stream plateaus at 12 steps alive (BallBalance
    # cannot drop the ball in 12 steps), so patience ends the run:
    # SAC after its 2-evaluation warm-up, PPO only once the update
    # floor's ceil(20 * 8 / 32) = 5 evaluations have passed.
    summary = (tmp_path / "stage_summary.txt").read_text()
    assert "Stop reason:    early_stop_patience: no improvement in" in summary
    assert model.num_timesteps < 1_024
    assert (
        f"Timesteps:      {model.num_timesteps:,} of 1,024 budget (stopped early)"
        in summary
    )
    assert "[closing eval, fresh episodes]" in summary
    assert "[task-metric selection]" in summary
    assert ("SB3 n_updates" in summary) is on_policy
    assert ("On-policy flags" in summary) is on_policy
    # Evaluation 1 sets the best and every later one ties it, so the
    # stop lands at warm-up + patience evaluations: 2 + 2 for SAC,
    # 5 + 2 for PPO.
    expected_stop = (5 + 2) * 32 if on_policy else (2 + 2) * 32
    assert model.num_timesteps == expected_stop

    for relative in (
        "config.json",
        "model/best_model.zip",
        "model/best_vec_normalize.pkl",
        "model/best_model_meta.json",
        "model/final_model.zip",
        "model/vec_normalize.pkl",
        "model/stage_bests/stage_00/best_model.zip",
        "metrics/evaluations.npz",
        "metrics/eval_info.csv",
        "reports/curriculum_stages.json",
    ):
        assert (tmp_path / relative).is_file(), relative
    for steps in range(64, model.num_timesteps + 1, 64):
        assert (tmp_path / "checkpoints" / f"rl_model_{steps}_steps.zip").is_file()
        assert (
            tmp_path / "checkpoints" / f"rl_model_vecnormalize_{steps}_steps.pkl"
        ).is_file()

    meta = json.loads((tmp_path / "model" / "best_model_meta.json").read_text())
    assert meta["selection_keys"][0] == "steps_alive_ep_mean"
    assert meta["eval_seed"] == 4_000
    eval_rows = (tmp_path / "metrics" / "eval_info.csv").read_text()
    assert (",n_updates," in eval_rows) is on_policy
    # The update count behind the selected policy (on-policy only):
    # evaluation 1, after 3 rollouts of 2 epochs.
    if on_policy:
        assert meta["n_updates"] == 6
        assert "Best model:     step 32, SB3 n_updates 6 (" in summary
    else:
        assert "n_updates" not in meta
    # progress.csv carries every update's train/* row, the last one
    # included (SB3 only dumps it with the next rollout's row).
    with open(tmp_path / "metrics" / "progress.csv") as stream:
        logged_updates = [
            float(row["train/n_updates"])
            for row in csv.DictReader(stream)
            if row.get("train/n_updates")
        ]
    if on_policy:
        assert logged_updates[-1] == model._n_updates
        assert "Epochs per update: 2.00 of n_epochs 2" in summary

    config = json.loads((tmp_path / "config.json").read_text())
    family = "PPO" if on_policy else "SAC"
    assert config["algorithm"]["family"] == family
    assert config["comparability_key"]["algo_family"] == family
    resolved = config["resolved_model"]
    assert resolved["algo_class"] == algo
    on_policy_keys = {"log_std_init", "ortho_init", "n_updates", "timesteps"}
    if on_policy:
        assert on_policy_keys <= set(resolved)
        assert {"target_kl", "sde_sample_freq", "n_steps"} <= set(
            resolved["hyperparameters"]
        )
        assert resolved["n_updates"] == model._n_updates > 0
        assert resolved["timesteps"] == {
            "trained": model.num_timesteps,
            "budget": 1_024,
        }
    else:
        assert not on_policy_keys & set(resolved)
        assert {"buffer_size", "learning_starts", "ent_coef"} <= set(
            resolved["hyperparameters"]
        )
        assert "target_kl" not in resolved["hyperparameters"]

    # Reward normalization follows the family (on for PPO, off for SAC),
    # in the record and in every pickled normalizer.
    assert config["resolved_training"]["normalize_reward_resolved"] is on_policy
    for pickled in (
        tmp_path / "model" / "vec_normalize.pkl",
        tmp_path / "model" / "best_vec_normalize.pkl",
        tmp_path / "checkpoints" / "rl_model_vecnormalize_64_steps.pkl",
    ):
        with open(pickled, "rb") as stream:
            normalizer = pickle.load(stream)
        assert normalizer.norm_obs is True
        assert normalizer.norm_reward is on_policy, pickled

    # Every saved model loads back through its own class.
    cls = resolve_algo(algo)
    for path in (
        tmp_path / "model" / "best_model.zip",
        tmp_path / "model" / "stage_bests" / "stage_00" / "best_model.zip",
        tmp_path / "checkpoints" / "rl_model_64_steps.zip",
    ):
        assert type(load_run_model(path, device="cpu")) is cls, path
    assert type(_load_best_model(str(tmp_path))) is cls


def test_ppo_training_is_deterministic(tmp_path):
    """The PPO counterpart of SD0: two seeded runs (reward normalization
    on, two workers) train identical parameters and normalizers."""
    import pickle

    states = []
    normalizers = []
    for name in ("first", "second"):
        model = train(
            _family_e2e_cfg(
                tmp_path / name,
                "PPO",
                n_envs=2,
                total_timesteps=128,
                eval_freq=32,
                checkpoint_freq=0,
                early_stop_patience=None,
                model_kwargs={"n_steps": 16, "batch_size": 16, "n_epochs": 2},
            )
        )
        states.append(model.policy.state_dict())
        with open(tmp_path / name / "model" / "vec_normalize.pkl", "rb") as f:
            normalizers.append(pickle.load(f))
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key
    for stats in ("obs_rms", "ret_rms"):
        left, right = (getattr(n, stats) for n in normalizers)
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.var, right.var)


class _Reach1D(gym.Env):
    """A point mass steered toward a goal on a line: the CI learning bar.

    Reward ``-|goal - x|`` every step, truncated at 40 steps and never
    terminated, so the value head must bootstrap every episode end. The
    zero action scores about -26.7; the optimum (full speed to the goal,
    then hold) averages about -3.3 over the start distribution.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.x = 0.0
        self.goal = 0.0
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.x, self.goal - self.x], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.uniform(-1.0, 1.0))
        self.goal = float(self.np_random.uniform(-1.0, 1.0))
        self.t = 0
        return self._obs(), {"distance": abs(self.goal - self.x)}

    def step(self, action):
        self.x += 0.1 * float(np.clip(action[0], -1.0, 1.0))
        self.t += 1
        distance = abs(self.goal - self.x)
        return self._obs(), -distance, False, self.t >= 40, {"distance": distance}


def test_ppo_learns_a_one_dimensional_reach_through_train(tmp_path):
    """PPO plan Step 12's CI learning bar: PPO through the whole trainer
    (4 workers x 256-step rollouts, reward normalization on) must learn
    a task SB3 PPO solves in seconds. Measured (seeds 0, 1, 2): the best
    paired batch reached -3.41, -3.22 and -3.23 within 24,576 steps
    (24 rollouts, SB3 n_updates 240; about 11 s on one thread), max
    approx_kl 0.014-0.022. A broken advantage, normalizer, gamma or
    rollout path fails the -6 bar long before it fails a unit test.
    Selection is on the paired evaluator's reward, so the reloaded best
    pair must reproduce the selecting batch exactly."""
    import pickle

    from courtside_dynamics.training.algos import load_run_model
    from courtside_dynamics.training.artifacts import _read_training_health_series

    eval_seed = 9_000
    cfg = TrainConfig(
        env_fn=_Reach1D,
        algo="PPO",
        log_dir=str(tmp_path),
        n_envs=4,
        seed=0,
        total_timesteps=24_576,
        eval_freq=4_096,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        n_eval_episodes=10,
        early_stop_patience=None,
        info_dict_eval=True,
        headline_key="distance",
        best_metric_keys=("episode_reward_mean",),
        eval_seed=eval_seed,
        model_kwargs={
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "learning_rate": 1e-3,
        },
    )
    model = train(cfg)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["resolved_training"]["normalize_reward_resolved"] is True
    assert model._n_updates == 24 * 10

    meta = json.loads((tmp_path / "model" / "best_model_meta.json").read_text())
    best = meta["metrics"]["episode_reward_mean"]
    assert best >= -6.0, best
    approx_kl = _read_training_health_series(str(tmp_path))["train/approx_kl"]
    assert approx_kl and max(approx_kl) < 0.1, approx_kl

    best_model = load_run_model(tmp_path / "model" / "best_model.zip", device="cpu")
    with open(tmp_path / "model" / "best_vec_normalize.pkl", "rb") as stream:
        normalizer = pickle.load(stream)
    returns = []
    for episode in range(cfg.n_eval_episodes):
        env = _Reach1D()
        obs, _ = env.reset(seed=eval_seed + episode)
        total, done = 0.0, False
        while not done:
            action, _ = best_model.predict(
                normalizer.normalize_obs(obs[None]), deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(action[0])
            # The evaluator sums the vec env's float32 reward buffer.
            total += float(np.float32(reward))
            done = terminated or truncated
        returns.append(total)
    assert sum(returns) / len(returns) == best
