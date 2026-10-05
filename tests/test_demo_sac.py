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

import hashlib
import json
import os
import pickle

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from courtside_dynamics.envs import BallBalanceEnv
from courtside_dynamics.notebook_utils import (
    RunConfigPlanMismatch,
    validate_run_config_against_plan,
)
from courtside_dynamics.training.algos import (
    ALGOS,
    OFF_POLICY_ALGOS,
    validate_model_kwargs,
)
from courtside_dynamics.training.artifacts import (
    _model_info,
    observation_names_sha256,
)
from courtside_dynamics.training.demo_sac import DEMO_LIBRARY_SCHEMA, DemoSAC
from courtside_dynamics.training.train import (
    SelectiveVecNormalize,
    TrainConfig,
    _build_algo,
    train,
)

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

    def test_every_demo_kwarg_passes_model_kwargs_validation(self):
        """The whole LD1′ surface is suppliable through a run's
        model_kwargs (the campaign notebook's route) — the sha pin
        included — and none of it is accepted by plain SAC."""
        demo_kwargs = {
            "demo_library": "lib.pkl",
            "demo_library_sha256": "0123abcd",
            "demo_fraction": 0.1,
            "demo_bc_coef": 0.0,
            "demo_bc_filter": "none",
            "demo_window": "point",
        }
        validate_model_kwargs("DemoSAC", demo_kwargs)
        for key, value in demo_kwargs.items():
            with pytest.raises(ValueError, match="not accepted by SAC"):
                validate_model_kwargs("SAC", {key: value})

    def test_library_sha_pin_is_checked_at_construction(self, venv, tmp_path):
        """A moved or re-harvested library aborts the launch at
        construction instead of voiding the run after the fact; the
        banked attribute is always the full digest, whatever the pin's
        length."""
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        on = dict(demo_library=path, demo_fraction=0.25, **_SMALL)
        for pin in (digest, digest[:8]):
            model = DemoSAC("MlpPolicy", venv, demo_library_sha256=pin, **on)
            assert model.demo_library_sha256 == digest
        wrong = ("0" if digest[0] != "0" else "1") + digest[1:12]
        with pytest.raises(ValueError, match="does not match the pinned"):
            DemoSAC("MlpPolicy", venv, demo_library_sha256=wrong, **on)
        for bad in ("", "0123abc", "ABCDEF01", "g" * 8, digest + "0", 123):
            with pytest.raises(ValueError, match="lowercase hex, 8 to 64"):
                DemoSAC("MlpPolicy", venv, demo_library_sha256=bad, **on)
        with pytest.raises(ValueError, match="pins a demo_library"):
            DemoSAC("MlpPolicy", venv, demo_library_sha256=digest, **_SMALL)

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
        # A fraction that rounds to zero demo rows at this batch size
        # would be a silent no-op with the library recorded as in play.
        with pytest.raises(ValueError, match="rounds to 0 demo rows"):
            DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.01, **_SMALL)

    def test_malformed_library_refused_at_first_use(self, venv, tmp_path):
        """The buffer builds lazily (so inference loaders never need the
        file), but the first learn() still refuses a malformed
        library loudly."""
        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"schema": "other", "trajectories": []}, f)
        model = DemoSAC("MlpPolicy", venv, demo_library=str(bad), demo_fraction=0.2, **_SMALL)
        with pytest.raises(ValueError, match="schema"):
            model.learn(total_timesteps=8)
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        with open(path, "rb") as f:
            library = pickle.load(f)
        library["trajectories"][1]["actions"] = library["trajectories"][1]["actions"][:, :1]
        shaped = tmp_path / "shaped.pkl"
        with open(shaped, "wb") as f:
            pickle.dump(library, f)
        with pytest.raises(ValueError, match="inconsistent array shapes"):
            DemoSAC("MlpPolicy", venv, demo_library=str(shaped), demo_fraction=0.2, **_SMALL).learn(8)


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

    def test_off_locksteps_on_the_recipe_shape(self):
        """SD0 on the pilot's recipe shape — gSDE with the 64-step noise
        hold, SelectiveVecNormalize with an excluded index, n_envs 4 x
        train_freq (64, 'step') with gradient_steps=-1, batch 256, auto
        temperature with a target: parameters, log_ent_coef, the
        normalizer's running stats AND the global numpy/torch RNG
        states are identical after the run (the explicit global-RNG
        accounting the D-G certificate names)."""
        results = []
        for cls in (SAC, DemoSAC):
            raw = make_vec_env(lambda: BallBalanceEnv(), n_envs=4, seed=0)
            env = SelectiveVecNormalize(
                raw, norm_obs=True, norm_reward=False, normalize_obs_excluded_indices=(0,)
            )
            try:
                model = cls(
                    "MlpPolicy",
                    env,
                    use_sde=True,
                    sde_sample_freq=64,
                    ent_coef="auto_0.02",
                    target_entropy=-1.5,
                    train_freq=(64, "step"),
                    gradient_steps=-1,
                    batch_size=256,
                    buffer_size=4096,
                    learning_starts=256,
                    seed=0,
                    verbose=0,
                    device="cpu",
                )
                model.learn(total_timesteps=1024)
                assert model._n_updates == 768  # 3 trained rollouts x 256 updates
                results.append(
                    (
                        _state(model),
                        float(model.log_ent_coef.detach().item()),
                        np.array(env.obs_rms.mean, copy=True),
                        np.array(env.obs_rms.var, copy=True),
                        np.random.get_state()[1].copy(),
                        torch.get_rng_state().clone(),
                    )
                )
            finally:
                env.close()
        sac, demo = results
        assert sac[0].keys() == demo[0].keys()
        for key in sac[0]:
            assert torch.equal(sac[0][key], demo[0][key]), key
        assert sac[1] == demo[1]
        np.testing.assert_array_equal(sac[2], demo[2])
        np.testing.assert_array_equal(sac[3], demo[3])
        np.testing.assert_array_equal(sac[4], demo[4])
        assert torch.equal(sac[5], demo[5])


class TestInjection:
    def test_loader_builds_train_buffer_and_holdout(self, venv, tmp_path):
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        assert model.demo_buffer is None  # lazy: nothing loaded at construction
        # ...but the provenance digest is banked immediately (the
        # trainer writes config.json before learn()).
        with open(path, "rb") as f:
            assert model.demo_library_sha256 == hashlib.sha256(f.read()).hexdigest()
        model._ensure_demo_loaded()
        assert model.demo_transitions == n_train
        assert model.demo_buffer is not None and model.demo_buffer.full
        assert model.demo_holdout is not None
        holdout_obs, holdout_act = model.demo_holdout
        assert holdout_obs.shape[0] == holdout_act.shape[0] == 2 * 12  # entries 0 and 5
        # The launch-state population (G1's): one row per held-out trajectory.
        assert model.demo_holdout_launch is not None
        launch_obs, launch_act = model.demo_holdout_launch
        assert launch_obs.shape[0] == launch_act.shape[0] == 2
        np.testing.assert_array_equal(launch_obs[0], holdout_obs[0])
        np.testing.assert_array_equal(launch_obs[1], holdout_obs[12])
        ordering = model.demo_q_ordering()
        assert ordering is not None and 0.0 <= ordering <= 1.0
        launch_ordering = model.demo_q_ordering_launch()
        assert launch_ordering is not None and 0.0 <= launch_ordering <= 1.0

    def test_to_confirm_window_truncates_trajectories(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC(
            "MlpPolicy", venv, demo_library=path, demo_fraction=0.25, demo_window="to_confirm", **_SMALL
        )
        model._ensure_demo_loaded()
        # 4 train trajectories x (confirm_step 6 + 1) steps
        assert model.demo_transitions == 4 * 7

    def test_minibatch_composition_is_exact(self, venv, tmp_path, monkeypatch):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        model._ensure_demo_loaded()
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

    def test_ordering_series_starts_at_the_first_train_call(self, venv, tmp_path):
        """Both ordering series are logged after the first train() call's
        updates (then every 50th call) — the trend series; the step-0
        baseline is an explicit call before learn()."""
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        baseline = model.demo_q_ordering_launch()
        assert baseline is not None and 0.0 <= baseline <= 1.0
        model.learn(total_timesteps=_SMALL["learning_starts"] + 1)  # exactly one train() call
        assert model._n_updates == 1
        logged = model.logger.name_to_value
        assert 0.0 <= logged["train/demo_q_ordering"] <= 1.0
        assert 0.0 <= logged["train/demo_q_ordering_launch"] <= 1.0

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
        # The subclass reloads WITHOUT needing the library file (an
        # algo-resolving inference loader), and rebuilds the buffers
        # lazily from the recorded path when training resumes.
        again = DemoSAC.load(str(checkpoint), env=venv, device="cpu")
        assert again.demo_fraction == 0.25
        assert again.demo_buffer is None
        assert again.demo_library_sha256 == model.demo_library_sha256
        again._ensure_demo_loaded()
        assert again.demo_transitions == n_train
        os.rename(path, path + ".moved")
        try:
            absent = DemoSAC.load(str(checkpoint), env=venv, device="cpu")
            assert absent.demo_library_sha256 == model.demo_library_sha256
            with pytest.raises(FileNotFoundError):
                absent.learn(total_timesteps=8)
        finally:
            os.rename(path + ".moved", path)

    def test_load_overrides_are_revalidated_and_digest_follows_the_file(self, venv, tmp_path):
        """SB3's load() applies override kwargs straight onto __dict__:
        the pairing rules re-run there, a load that switches the surface
        off carries no digest, a load onto a different library re-derives
        the digest from THAT file, and a library that changed on disk
        under a banked digest is refused at first use."""
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        checkpoint = tmp_path / "model.zip"
        model.save(str(checkpoint))
        with pytest.raises(ValueError, match="enabled together"):
            DemoSAC.load(str(checkpoint), env=venv, device="cpu", demo_library=None)
        with pytest.raises(ValueError, match="enabled together"):
            DemoSAC.load(str(checkpoint), env=venv, device="cpu", demo_fraction=0.0)
        off = DemoSAC.load(
            str(checkpoint), env=venv, device="cpu", demo_library=None, demo_fraction=0.0
        )
        assert off.demo_library_sha256 is None
        assert "demo_library_sha256" not in _model_info(off)
        other, _ = _synthetic_library(tmp_path / "other.pkl", venv, trajectories=7, seed=3)
        moved = DemoSAC.load(str(checkpoint), env=venv, device="cpu", demo_library=other)
        # The override's digest is banked AT LOAD (before any learn()),
        # so a probe in between pairs the new path with the new digest.
        with open(other, "rb") as f:
            other_digest = hashlib.sha256(f.read()).hexdigest()
        assert moved.demo_buffer is None
        assert moved.demo_library_sha256 == other_digest
        probe = _model_info(moved)
        assert probe["hyperparameters"]["demo_library"] == other
        assert probe["demo_library_sha256"] == other_digest
        moved._ensure_demo_loaded()
        assert moved.demo_library_sha256 == other_digest
        # Same path, different bytes: the banked provenance no longer
        # describes the file — refused, not silently re-hashed.
        with open(path, "rb") as f:
            library = pickle.load(f)
        library["note"] = "edited after the checkpoint banked its digest"
        with open(path, "wb") as f:
            pickle.dump(library, f)
        stale = DemoSAC.load(str(checkpoint), env=venv, device="cpu")
        with pytest.raises(ValueError, match="changed under its recorded provenance"):
            stale.learn(total_timesteps=8)

    def test_train_banks_the_digest_before_learning(self, venv, tmp_path):
        """SD2 end to end: the trainer writes config.json before learn()
        starts, so the consumed-library digest has to exist at
        construction — it reaches resolved_model on a real train() run.
        So does the train-transition count, which only exists once the
        (lazy) demo buffer is built: config.json used to record
        ``demo_transitions: 0`` on every DemoSAC run. The count must
        already be on disk when learning starts (a run that dies
        mid-learn keeps the config.json written at launch), and the
        demo kwargs — the construction-time sha pin included — pass
        the frozen-plan validator the campaign notebook runs."""
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        log_dir = tmp_path / "run"

        class _RecordedAtTrainingStart(BaseCallback):
            def __init__(self):
                super().__init__()
                self.demo_transitions = None

            def _on_training_start(self):
                config = json.loads((log_dir / "config.json").read_text())
                hyperparameters = config["resolved_model"]["hyperparameters"]
                self.demo_transitions = hyperparameters["demo_transitions"]

            def _on_step(self):
                return True

        at_start = _RecordedAtTrainingStart()
        demo_kwargs = {
            "demo_library": path,
            "demo_library_sha256": digest[:16],
            "demo_fraction": 0.25,
            "demo_bc_coef": 0.1,
            "demo_bc_filter": "none",
            "demo_window": "point",
        }
        cfg = TrainConfig(
            env_fn=lambda: BallBalanceEnv(),
            algo="DEMOSAC",
            total_timesteps=8,
            log_dir=str(log_dir),
            n_envs=1,
            seed=0,
            eval_freq=10_000,
            checkpoint_freq=0,
            video_freq=0,
            record_video=False,
            info_dict_eval=False,
            n_eval_episodes=1,
            normalize_obs=True,
            extra_callbacks=(at_start,),
            model_kwargs={
                **demo_kwargs,
                "batch_size": 32,
                "buffer_size": 64,
                "learning_starts": 1_000,
            },
        )
        model = train(cfg)
        config = json.loads((log_dir / "config.json").read_text())
        assert config["resolved_model"]["demo_library_sha256"] == digest
        assert config["resolved_model"]["hyperparameters"]["demo_fraction"] == 0.25
        assert config["resolved_model"]["hyperparameters"]["demo_library"] == path
        # The recorded count is the model's, and it is the real one.
        assert isinstance(model, DemoSAC)
        assert model.demo_transitions == n_train > 0
        recorded = config["resolved_model"]["hyperparameters"]["demo_transitions"]
        assert recorded == model.demo_transitions
        assert at_start.demo_transitions == model.demo_transitions
        # The campaign notebook's leg plan for this run validates clean...
        plan = {
            "algo": "DemoSAC",
            "model_kwargs": demo_kwargs,
            "demo_library_sha256": demo_kwargs["demo_library_sha256"],
        }
        validate_run_config_against_plan(log_dir / "config.json", plan)
        # ...and a plan pinning a different library does not.
        with pytest.raises(RunConfigPlanMismatch, match="demo_library_sha256"):
            validate_run_config_against_plan(
                log_dir / "config.json",
                {**plan, "demo_library_sha256": "0123abcd"},
            )

    def test_train_with_the_surface_off_records_zero_transitions(self, tmp_path):
        """The eager pre-learn load the trainer does for DemoSAC is a
        no-op with the surface off: nothing to load, 0 recorded."""
        log_dir = tmp_path / "run"
        cfg = TrainConfig(
            env_fn=lambda: BallBalanceEnv(),
            algo="DemoSAC",
            total_timesteps=8,
            log_dir=str(log_dir),
            n_envs=1,
            seed=0,
            eval_freq=10_000,
            checkpoint_freq=0,
            video_freq=0,
            record_video=False,
            info_dict_eval=False,
            n_eval_episodes=1,
            model_kwargs={"buffer_size": 64, "learning_starts": 1_000},
        )
        model = train(cfg)
        assert model.demo_buffer is None
        config = json.loads((log_dir / "config.json").read_text())
        assert config["resolved_model"]["hyperparameters"]["demo_transitions"] == 0
        assert "demo_library_sha256" not in config["resolved_model"]

    def test_model_probe_records_the_warmup_mode(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC(
            "MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL
        )
        hyperparameters = _model_info(model)["hyperparameters"]
        # Who drives the learning_starts warmup is resolved provenance
        # (train() turns use_sde_at_warmup on for gSDE warm starts).
        assert hyperparameters["use_sde"] is False
        assert hyperparameters["use_sde_at_warmup"] is False

    def test_model_probe_records_the_consumed_digest(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        # Probed as the trainer probes it: before any learn() call.
        info = _model_info(model)
        assert info["algo_class"] == "DemoSAC"
        assert info["demo_library_sha256"] == model.demo_library_sha256
        assert info["hyperparameters"]["demo_fraction"] == 0.25
        assert info["hyperparameters"]["demo_library"] == path
        plain = SAC("MlpPolicy", venv, **_SMALL)
        assert "demo_library_sha256" not in _model_info(plain)


def _rewrite_header(path, **header):
    with open(path, "rb") as f:
        library = pickle.load(f)
    library.update(header)
    with open(path, "wb") as f:
        pickle.dump(library, f)


class TestObservationFingerprint:
    """Review §7.2 (*new*): the demo library was checked by shape only, so
    a same-width observation meaning change (world-frame spin, scaled
    counters) would load the LD1′ library onto a different task."""

    def test_matching_fingerprint_loads(self, venv, tmp_path, capsys):
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        names = list(BallBalanceEnv.observation_names)
        _rewrite_header(
            path,
            observation_names=names,
            observation_names_sha256=observation_names_sha256(names),
        )
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        model._ensure_demo_loaded()
        assert model.demo_transitions == n_train
        assert "fingerprint" not in capsys.readouterr().out

    def test_layout_mismatch_is_refused_naming_the_index(self, venv, tmp_path):
        path, _ = _synthetic_library(tmp_path / "lib.pkl", venv)
        names = list(BallBalanceEnv.observation_names)
        renamed = [*names[:5], "ball_vz_scaled", *names[6:]]
        _rewrite_header(
            path,
            observation_names=renamed,
            observation_names_sha256=observation_names_sha256(renamed),
        )
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        with pytest.raises(
            ValueError,
            match=r"observation index 5 is 'ball_vz_scaled' in the recorded "
            r"layout but 'ball_vz' in this env",
        ):
            model.learn(total_timesteps=8)
        # A digest-only header still refuses, naming both digests.
        _rewrite_header(path, observation_names=None)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        with pytest.raises(ValueError, match="observation_names_sha256"):
            model._ensure_demo_loaded()
        _rewrite_header(path, observation_names_sha256=123)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        with pytest.raises(ValueError, match="malformed observation_names_sha256"):
            model._ensure_demo_loaded()

    def test_pre_fingerprint_library_still_loads_with_a_notice(
        self, venv, tmp_path, capsys
    ):
        """The format every library harvested so far carries (schema +
        trajectories, no fingerprint header) keeps loading on the shape
        checks, with a one-line notice."""
        path, n_train = _synthetic_library(tmp_path / "lib.pkl", venv)
        model = DemoSAC("MlpPolicy", venv, demo_library=path, demo_fraction=0.25, **_SMALL)
        model._ensure_demo_loaded()
        assert model.demo_transitions == n_train
        out = capsys.readouterr().out
        assert "carries no observation fingerprint" in out
        assert len(out.strip().splitlines()) == 1

    def test_harvest_tool_writes_the_fingerprint_the_checks_read(
        self, tmp_path, monkeypatch
    ):
        """New libraries carry the header DemoSAC checks, encoded exactly
        as config.json's env probe encodes the recipe env's layout."""
        import sys

        from courtside_dynamics.envs import PaddleTennisEnv
        from courtside_dynamics.training.artifacts import _probe_env_fn
        from tools import paddle_tennis_k2_demo_harvest as harvest

        layout = harvest._observation_layout()
        assert layout == list(PaddleTennisEnv.observation_names)
        probed = _probe_env_fn(lambda: PaddleTennisEnv(**harvest.ENV_KWARGS))
        assert probed["observation_names_sha256"] == observation_names_sha256(
            layout
        )

        # main() end to end, with the (drill-library-bound) harvest stubbed.
        source = tmp_path / "k2_library.pkl"
        source.write_bytes(b"stub drill library")
        out = tmp_path / "demos.pkl"
        monkeypatch.setattr(
            harvest,
            "harvest_library",
            lambda path, **kwargs: ([], harvest._empty_counts()),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["harvest", "--library", str(source), "--out", str(out)],
        )
        harvest.main()
        with open(out, "rb") as f:
            library = pickle.load(f)
        assert library["schema"] == DEMO_LIBRARY_SCHEMA
        assert library["observation_names"] == layout
        assert library["observation_names_sha256"] == observation_names_sha256(
            layout
        )
