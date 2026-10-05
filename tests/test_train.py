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
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="PPO",
        log_dir=str(target_dir),
        warm_start=WarmStartConfig(tmp_path / "missing"),
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
