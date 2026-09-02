"""DemoSAC: the LD1′ demonstration-injection surface
(docs/design_paddle_tennis_demo_injection.md).

SD0 — default-off is bit-identical to stock SAC (parameter-stream
lockstep on a seeded run); loud validation of half-configured pairs
and malformed libraries; exact per-minibatch composition when on;
the BC term and its Q-filter; the held-out ordering measurement;
save/load round trips (including the plain ``SAC.load`` path the
diagnosis tools use); registry membership with the resolved
``gradient_steps`` (the silent 256x under-training trap); and the
provenance digest reaching the model probe.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.training.algos import (
    ALGOS,
    OFF_POLICY_ALGOS,
    validate_model_kwargs,
)
from courtside_dynamics.training.artifacts import _model_info
from courtside_dynamics.training.demo_sac import DEMO_LIBRARY_SCHEMA, DemoSAC
from courtside_dynamics.training.train import _build_algo

_SMALL = dict(
    learning_starts=64,
    train_freq=1,
    gradient_steps=1,
    batch_size=32,
    buffer_size=1_000,
    seed=0,
    verbose=0,
    device="cpu",
)


@pytest.fixture
def venv():
    env = make_vec_env(lambda: BallBalanceEnv(), n_envs=1, seed=0)
    try:
        yield env
    finally:
        env.close()


def _synthetic_library(path, venv, *, trajectories=6, steps=12, seed=0):
    """A schema-faithful demo library from random BallBalance play:
    the env-agnostic replay-tuple layout the tool emits, with the
    train/held-out split (every 5th entry held out)."""
    rng = np.random.default_rng(seed)
    obs_dim = int(np.prod(venv.observation_space.shape))
    act_dim = int(np.prod(venv.action_space.shape))
    trajs = []
    obs = venv.reset()
    for index in range(trajectories):
        rows = {k: [] for k in ("obs", "actions", "next_obs", "rewards", "terminated", "truncated")}
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=(1, act_dim)).astype(np.float32)
            next_obs, reward, done, infos = venv.step(action)
            rows["obs"].append(obs[0].astype(np.float64))
            rows["actions"].append(action[0].astype(np.float64))
            rows["next_obs"].append(
                (infos[0]["terminal_observation"] if done[0] and "terminal_observation" in infos[0] else next_obs[0]).astype(np.float64)
            )
            rows["rewards"].append(float(reward[0]))
            truncated = bool(infos[0].get("TimeLimit.truncated", False))
            rows["terminated"].append(bool(done[0]) and not truncated)
            rows["truncated"].append(truncated)
            obs = next_obs
        trajs.append(
            {
                "source": "synthetic",
                "entry": index,
                "split": "heldout" if index % 5 == 0 else "train",
                "obs": np.asarray(rows["obs"]),
                "actions": np.asarray(rows["actions"]),
                "next_obs": np.asarray(rows["next_obs"]),
                "rewards": np.asarray(rows["rewards"]),
                "terminated": np.asarray(rows["terminated"], dtype=bool),
                "truncated": np.asarray(rows["truncated"], dtype=bool),
                "hit_step": 3,
                "confirm_step": 6,
                "ender": "synthetic",
            }
        )
    assert obs_dim == trajs[0]["obs"].shape[1]
    library = {"schema": DEMO_LIBRARY_SCHEMA, "trajectories": trajs}
    with open(path, "wb") as f:
        pickle.dump(library, f)
    n_train = sum(steps for t in trajs if t["split"] == "train")
    return str(path), n_train


def _state(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


class TestRegistryAndValidation:
    def test_registered_as_off_policy_with_full_gradient_steps(self, venv, tmp_path):
        """The trap the design names: an off-policy algo missing from
        the registry silently runs SB3's gradient_steps=1 against
        train_freq=(64, 'step') — 1 update per 256 transitions."""
        assert ALGOS["DEMOSAC"] is DemoSAC
        assert "DEMOSAC" in OFF_POLICY_ALGOS
        model = _build_algo("DemoSAC", venv, str(tmp_path))
        assert model.gradient_steps == -1
        assert isinstance(model, DemoSAC)

    def test_model_kwargs_validation_stays_strict_through_the_subclass(self):
        validate_model_kwargs("DemoSAC", {"demo_fraction": 0.2, "buffer_size": 10})
        with pytest.raises(ValueError, match="not accepted by DemoSAC"):
            validate_model_kwargs("DemoSAC", {"demo_fractoin": 0.2})

    def test_half_configured_pairs_and_bounds_are_rejected(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        with pytest.raises(ValueError, match="enabled together"):
            DemoSAC("MlpPolicy", venv, demo_fraction=0.2, **_SMALL)
        with pytest.raises(ValueError, match="enabled together"):
            DemoSAC("MlpPolicy", venv, demo_library=path, **_SMALL)
        with pytest.raises(ValueError, match="demo_fraction"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=1.0, **_SMALL)
        with pytest.raises(ValueError, match="demo_bc_coef > 0 needs"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_bc_coef=0.1, **_SMALL)
        with pytest.raises(ValueError, match="demo_bc_filter"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.2, demo_bc_filter="x", **_SMALL)
        with pytest.raises(ValueError, match="demo_window"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.2, demo_window="x", **_SMALL)
        with pytest.raises(ValueError, match="n_steps"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.2, n_steps=3, **_SMALL)

    def test_malformed_library_refused_at_construction(self, venv, tmp_path):
        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"schema": "other", "trajectories": []}, f)
        with pytest.raises(ValueError, match="schema"):
            DemoSAC("MlpPolicy", venv, demo_library=str(bad), demo_fraction=0.2, **_SMALL)
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        with open(path, "rb") as f:
            library = pickle.load(f)
        library["trajectories"][1]["actions"] = library["trajectories"][1]["actions"][:, :1]
        shaped = tmp_path / "shaped.pkl"
        with open(shaped, "wb") as f:
            pickle.dump(library, f)
        with pytest.raises(ValueError, match="inconsistent array shapes"):
            DemoSAC("MlpPolicy", venv, demo_library=str(shaped), demo_fraction=0.2, **_SMALL)


class TestSD0BitIdentity:
    def test_off_locksteps_stock_sac_parameter_stream(self, tmp_path):
        """SD0: with the demo surface off, DemoSAC's gradient stream is
        stock SAC's — identical parameters after a seeded run (same
        replay draws from the global RNG, no extra forward passes)."""
        results = []
        for cls in (SAC, DemoSAC):
            env = make_vec_env(lambda: BallBalanceEnv(), n_envs=1, seed=0)
            try:
                model = cls("MlpPolicy", env, **_SMALL)
                model.learn(total_timesteps=256)
                results.append((_state(model), float(model.log_ent_coef.detach().item())))
            finally:
                env.close()
        (sac_state, sac_ent), (demo_state, demo_ent) = results
        assert sac_state.keys() == demo_state.keys()
        for key in sac_state:
            assert torch.equal(sac_state[key], demo_state[key]), key
        assert sac_ent == demo_ent


class TestInjection:
    def test_loader_builds_train_buffer_and_holdout(self, venv, tmp_path):
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        assert model.demo_transitions == n_train
        assert model.demo_buffer is not None and model.demo_buffer.full
        assert model.demo_holdout is not None
        holdout_obs, holdout_act = model.demo_holdout
        assert holdout_obs.shape[0] == holdout_act.shape[0] == 2 * 12  # entries 0 and 5
        assert isinstance(model.demo_library_sha256, str) and len(model.demo_library_sha256) == 64
        ordering = model.demo_q_ordering()
        assert ordering is not None and 0.0 <= ordering <= 1.0

    def test_to_confirm_window_truncates_trajectories(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC(
            "MlpPolicy", venv, demo_library=path, demo_fraction=0.25, demo_window="to_confirm", **_SMALL
        )
        # 4 train trajectories x (confirm_step 6 + 1) steps
        assert model.demo_transitions == 4 * 7

    def test_minibatch_composition_is_exact(self, venv, tmp_path, monkeypatch):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        sizes = {"live": [], "demo": []}
        live_sample = model.replay_buffer.sample
        demo_sample = model.demo_buffer.sample

        def spy_live(batch_size, env=None):
            sizes["live"].append(batch_size)
            return live_sample(batch_size, env=env)

        def spy_demo(batch_size, env=None):
            sizes["demo"].append(batch_size)
            return demo_sample(batch_size, env=env)

        monkeypatch.setattr(model.replay_buffer, "sample", spy_live)
        monkeypatch.setattr(model.demo_buffer, "sample", spy_demo)
        model.learn(total_timesteps=128)
        assert sizes["demo"] and set(sizes["demo"]) == {8}   # round(0.25 * 32)
        assert set(sizes["live"]) == {24}
        assert len(sizes["live"]) == len(sizes["demo"])

    def test_bc_term_and_q_filter_log(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        for bc_filter in ("none", "q"):
            model = DemoSAC(
                "MlpPolicy", venv, demo_library=path, demo_fraction=0.25,
                demo_bc_coef=0.5, demo_bc_filter=bc_filter, **_SMALL,
            )
            model.learn(total_timesteps=96)
            model.train(gradient_steps=1, batch_size=32)
            logged = model.logger.name_to_value
            assert logged["train/demo_fraction"] == 0.25
            assert "train/demo_bc_loss" in logged
            assert ("train/demo_q_filter_pass" in logged) == (bc_filter == "q")
            if bc_filter == "q":
                assert 0.0 <= logged["train/demo_q_filter_pass"] <= 1.0

    def test_save_and_load_round_trips(self, venv, tmp_path):
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        model.learn(total_timesteps=96)
        assert "demo_buffer" in model._excluded_save_params()
        checkpoint = tmp_path / "model.zip"
        model.save(str(checkpoint))
        # The diagnosis/harvest tools load checkpoints as plain SAC.
        plain = SAC.load(str(checkpoint), device="cpu")
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, plain.policy.state_dict()[key]), key
        # The subclass reloads its buffers from the recorded library path.
        again = DemoSAC.load(str(checkpoint), env=venv, device="cpu")
        assert again.demo_fraction == 0.25
        assert again.demo_transitions == n_train
        assert again.demo_library_sha256 == model.demo_library_sha256

    def test_model_probe_records_the_consumed_digest(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        info = _model_info(model)
        assert info["algo_class"] == "DemoSAC"
        assert info["demo_library_sha256"] == model.demo_library_sha256
        assert info["hyperparameters"]["demo_fraction"] == 0.25
        assert info["hyperparameters"]["demo_library"] == path
        plain = SAC("MlpPolicy", venv, **_SMALL)
        assert "demo_library_sha256" not in _model_info(plain)
