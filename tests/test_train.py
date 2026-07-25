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
import json

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


def _make_sac_warm_start_source(tmp_path, *, log_ent_coef=-6.5):
    """SAC sibling of the PPO helper: tiny model, marked policy weights,
    a deliberately collapsed entropy temperature, saved as a canonical
    best-run directory."""
    source_dir = tmp_path / "sac_source"
    source_dir.mkdir()

    def env_fn():
        return BallBalanceEnv(episode_len=12)

    source_cfg = TrainConfig(
        env_fn=env_fn,
        algo="SAC",
        log_dir=str(source_dir),
        n_envs=1,
        normalize_obs=True,
        normalize_obs_excluded_indices=(0,),
        model_kwargs={"buffer_size": 64, "learning_starts": 1_000},
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


def test_warm_start_supports_ppo_and_sac_only(tmp_path):
    """The transfer path is written for exactly the two algorithms this
    project trains; any future third algorithm must extend it explicitly
    rather than falling through half-supported."""
    cfg = TrainConfig(
        env_fn=lambda: BallBalanceEnv(),
        algo="TD3",
        log_dir=str(tmp_path / "target"),
        warm_start=WarmStartConfig(tmp_path),
    )
    with pytest.raises(ValueError, match="supports PPO and SAC only"):
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
    # The best-checkpoint section describes the *selected* step, with
    # the reward looked up at that step (1.000), not the 75k argmax.
    assert "Best Checkpoint Evaluation (step 50,000)" in text
    assert "Reward:         1.000 +/- 0.000" in text
    assert "Return survival:" in text
    assert ">=2 80.0%" in text
    assert ">=3 25.0%" in text
    # The reward-series best line is still reported for context.
    assert "2.000 +/- 0.000 (at 75,000 steps)" in text


def _merged_eval_cfg(tmp_path, **overrides):
    """A tiny headline-selection run with the final-config eval stream on."""
    cfg_kwargs = dict(
        env_fn=lambda: BallBalanceEnv(episode_len=12),
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
