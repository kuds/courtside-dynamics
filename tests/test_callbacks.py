"""Tests for ``VideoRecordCallback`` and ``InfoDictEvalCallback``.

The video recorder is the main entry point for per-step CSV logging, and
its auto-detection of scalar ``info`` keys is what gives WallBall runs
their bounce/hit-count diagnostics "for free". These tests verify:

1. Scalar-only filtering (arrays and non-numeric keys are skipped).
2. Explicit ``info_row_fn`` still wins over auto-detection.
3. The CSV header is emitted before any row, and row length matches.
4. ``InfoDictEvalCallback`` aggregates per-episode info scalars with the
   right derived metrics (mean, last, max, phase fractions).
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnvWrapper

from courtside_dynamics.callbacks.video_record import (
    _scalar_info_keys,
)


class _PassthroughVideoRecorder(VecEnvWrapper):
    """Drop-in replacement for ``VecVideoRecorder`` that skips encoding.

    The real recorder requires ``moviepy``/``rgb_array`` rendering, neither
    of which is needed to exercise the CSV/TensorBoard logging path. This
    stub preserves the callback's call pattern (wrap ``venv``, forward
    ``reset``/``step``, expose ``close``) and ignores everything else.
    """

    def __init__(self, venv, video_folder, record_video_trigger,
                 video_length, name_prefix):
        super().__init__(venv)

    def reset(self):  # type: ignore[override]
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def close(self) -> None:
        self.venv.close()


@pytest.fixture
def _stub_video_recorder(monkeypatch):
    """Stub the VecVideoRecorder so tests don't require moviepy."""
    monkeypatch.setattr(
        "courtside_dynamics.callbacks.video_record.VecVideoRecorder",
        _PassthroughVideoRecorder,
    )


def test_scalar_info_keys_filters_arrays_and_non_numeric():
    info = {
        "phase": 0,
        "rally_count": 3,
        "paddle_touch": 12.5,
        "ball_velocity": np.array([1.0, 2.0, 3.0]),  # array -> skip
        "zero_d_array": np.array(1.0),  # 0-D -> keep
        "label": "approach_paddle",  # string -> skip
        "is_serving": True,  # bool -> keep
        # SB3/gymnasium wrapper injections -> skip.
        "TimeLimit.truncated": False,
        "terminal_observation": np.zeros(4),
        "episode": {"r": 1.0, "l": 100},
    }
    keys = _scalar_info_keys(info)
    assert set(keys) == {
        "phase", "rally_count", "paddle_touch",
        "zero_d_array", "is_serving",
    }
    # Ensure determinism.
    assert keys == sorted(keys)


def test_scalar_info_keys_empty_dict():
    assert _scalar_info_keys({}) == []


class _FakeLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value) -> None:
        self.records[key] = value


class _FakeModel:
    """Stand-in for an SB3 algorithm.

    SB3's ``BaseCallback.logger`` proxies to ``self.model.logger``; tests
    use ``_FakeLogger`` so we can assert which TB tags were emitted.
    """

    def __init__(self, action_dim: int = 1) -> None:
        self.action_dim = action_dim
        self.logger = _FakeLogger()

    def predict(self, obs, deterministic: bool = False):
        # VecEnv expects shape (n_envs, action_dim); the recorder and
        # eval callback both run a single env so n_envs == 1. The
        # ``deterministic`` kwarg mirrors ``BaseAlgorithm.predict`` so the
        # eval callback can forward it without special-casing.
        del deterministic
        return np.zeros((1, self.action_dim), dtype=np.float32), None


def _run_callback_once(callback) -> None:
    """Drive a single recording pass without a real training loop.

    We bypass ``_on_step``'s schedule check by pre-setting ``n_calls`` so
    the modulo trigger fires immediately.
    """
    callback.n_calls = callback.save_freq  # trigger on first _on_step
    callback.num_timesteps = callback.save_freq
    callback._on_step()


def test_auto_log_populates_csv_and_tensorboard(tmp_path, _stub_video_recorder):
    """End-to-end: WallBall's info keys land in CSV + TB on auto-detect."""
    from courtside_dynamics.callbacks.video_record import VideoRecordCallback
    from courtside_dynamics.envs import WallBallEnv

    cb = VideoRecordCallback(
        env_fn=lambda: WallBallEnv(),
        save_path=str(tmp_path),
        video_length=20,
        save_freq=1,
        name_prefix="test",
    )
    cb.model = _FakeModel(action_dim=3)
    _run_callback_once(cb)

    # CSV: the scalar info keys + default reward columns should be in the
    # header, and every row should match that width.
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1
    with open(csv_files[0]) as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2  # header + at least one data row
    header = rows[0]
    for required in (
        "bounce_count",
        "wall_contact_count",
        "paddle_hit_count",
        "paddle_touch",
        "wall_touch",
        "reward",
        "total_reward",
        "done",
    ):
        assert required in header, f"missing column: {required}"
    for data_row in rows[1:]:
        assert len(data_row) == len(header), (
            f"row width {len(data_row)} != header width {len(header)}"
        )

    # TensorBoard: mean scalars under videorecord/<key>_mean plus the
    # total-reward summary.
    tb = cb.model.logger.records
    assert "videorecord/total_reward" in tb
    assert "videorecord/episode_length" in tb
    assert "videorecord/bounce_count_mean" in tb
    assert "videorecord/paddle_touch_mean" in tb


def test_explicit_info_row_fn_overrides_auto_detection(tmp_path, _stub_video_recorder):
    from courtside_dynamics.callbacks.video_record import VideoRecordCallback
    from courtside_dynamics.envs import WallBallEnv

    cb = VideoRecordCallback(
        env_fn=lambda: WallBallEnv(),
        save_path=str(tmp_path),
        video_length=5,
        save_freq=1,
        name_prefix="explicit",
        csv_header=["bounce_count", "reward"],
        info_row_fn=lambda info, r, tr, d: [info["bounce_count"], r],
    )
    cb.model = _FakeModel(action_dim=3)
    _run_callback_once(cb)

    csv_files = list(tmp_path.glob("*.csv"))
    with open(csv_files[0]) as f:
        rows = list(csv.reader(f))
    # User header wins; auto-detection doesn't inject extra columns.
    assert rows[0] == ["bounce_count", "reward"]
    for data_row in rows[1:]:
        assert len(data_row) == 2

    # Auto-detection was skipped, so no info-mean tags should appear.
    tb_keys = set(cb.model.logger.records)
    assert not any(k.endswith("_mean") for k in tb_keys)


def test_auto_log_handles_empty_info(tmp_path, _stub_video_recorder):
    """BallBalance returns ``{}`` — auto-logger should degrade gracefully."""
    from courtside_dynamics.callbacks.video_record import VideoRecordCallback
    from courtside_dynamics.envs import BallBalanceEnv

    cb = VideoRecordCallback(
        env_fn=lambda: BallBalanceEnv(),
        save_path=str(tmp_path),
        video_length=10,
        save_freq=1,
        name_prefix="empty",
    )
    cb.model = _FakeModel(action_dim=6)
    _run_callback_once(cb)

    csv_files = list(tmp_path.glob("*.csv"))
    with open(csv_files[0]) as f:
        rows = list(csv.reader(f))
    # Header falls back to the default reward triple.
    assert rows[0] == ["reward", "total_reward", "done"]


def test_video_recorder_continues_past_episode_ends(
    tmp_path, _stub_video_recorder
):
    """The recorder must not stop at the first episode end: run
    20260721_004722's best-model 'video' was one 104-step episode against
    a 10,000-step budget, an n=1 behavioral record. With auto-resetting
    VecEnvs the rollout continues into the next episode until
    ``max_episodes`` or the ``video_length`` cap."""
    import gymnasium as gym
    from gymnasium import spaces

    from courtside_dynamics.callbacks.video_record import VideoRecordCallback

    class _ShortEpisodeEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                -1.0, 1.0, (1,), dtype=np.float32
            )
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            self._t = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            del action
            self._t += 1
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                self._t >= 3,
                False,
                {},
            )

    cb = VideoRecordCallback(
        env_fn=_ShortEpisodeEnv,
        save_path=str(tmp_path),
        video_length=50,
        save_freq=1,
        name_prefix="multi",
        max_episodes=3,
    )
    cb.model = _FakeModel(action_dim=1)
    _run_callback_once(cb)

    csv_files = list(tmp_path.glob("*.csv"))
    with open(csv_files[0]) as f:
        rows = list(csv.reader(f))
    data_rows = rows[1:]
    # Three 3-step episodes, then the recorder stops.
    assert len(data_rows) == 9
    done_flags = [row[-1] == "True" for row in data_rows]
    assert done_flags == [False, False, True] * 3
    # total_reward restarts with each episode.
    totals = [float(row[-2]) for row in data_rows]
    assert totals == [1.0, 2.0, 3.0] * 3

    tb = cb.model.logger.records
    assert tb["videorecord/episodes"] == 3
    assert tb["videorecord/total_reward"] == pytest.approx(3.0)
    assert tb["videorecord/episode_length"] == 3

    # The video_length cap still binds when episodes are long: an
    # unbounded max_episodes stops at the cap.
    cb_capped = VideoRecordCallback(
        env_fn=_ShortEpisodeEnv,
        save_path=str(tmp_path / "capped"),
        video_length=7,
        save_freq=1,
        name_prefix="capped",
        max_episodes=None,
    )
    cb_capped.model = _FakeModel(action_dim=1)
    _run_callback_once(cb_capped)
    with open(next((tmp_path / "capped").glob("*.csv"))) as f:
        capped_rows = list(csv.reader(f))
    assert len(capped_rows[1:]) == 7  # 2 full episodes + 1 partial step
    assert cb_capped.model.logger.records["videorecord/episodes"] == 2


def test_video_callback_save_freq_zero_disables_recording(tmp_path):
    """``save_freq=0`` must be a no-op (matching InfoDictEvalCallback's
    ``eval_freq <= 0`` contract), not a ZeroDivisionError in the modulo
    schedule check."""
    from courtside_dynamics.callbacks.video_record import VideoRecordCallback
    from courtside_dynamics.envs import BallBalanceEnv

    cb = VideoRecordCallback(
        env_fn=lambda: BallBalanceEnv(),
        save_path=str(tmp_path),
        video_length=5,
        save_freq=0,
    )
    # No model attached: reaching past the guard would crash, so a clean
    # True return proves the early exit fired.
    cb.n_calls = 1
    assert cb._on_step() is True
    assert not list(tmp_path.iterdir()), "recording artifacts were written"


def test_video_callback_preserves_selective_normalizer_contract(
    tmp_path,
    monkeypatch,
):
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.video_record import VideoRecordCallback
    from courtside_dynamics.envs import BallBalanceEnv
    from courtside_dynamics.training import SelectiveVecNormalize

    captured: dict[str, object] = {}

    class _CaptureRecorder(_PassthroughVideoRecorder):
        def __init__(
            self,
            venv,
            video_folder,
            record_video_trigger,
            video_length,
            name_prefix,
        ):
            captured["normalizer"] = venv
            super().__init__(
                venv,
                video_folder,
                record_video_trigger,
                video_length,
                name_prefix,
            )

    monkeypatch.setattr(
        "courtside_dynamics.callbacks.video_record.VecVideoRecorder",
        _CaptureRecorder,
    )
    train_env = SelectiveVecNormalize(
        make_vec_env(lambda: BallBalanceEnv(episode_len=5), n_envs=1),
        norm_obs=True,
        norm_reward=False,
        normalize_obs_excluded_indices=(0, 2),
    )
    model = _FakeModel(action_dim=6)
    model.get_vec_normalize_env = lambda: train_env  # type: ignore[attr-defined]
    model.get_env = lambda: train_env  # type: ignore[attr-defined]
    callback = VideoRecordCallback(
        env_fn=lambda: BallBalanceEnv(episode_len=5),
        save_path=str(tmp_path),
        video_length=2,
        save_freq=1,
    )
    callback.model = model  # type: ignore[assignment]
    try:
        _run_callback_once(callback)
        recorder_normalizer = captured["normalizer"]
        assert isinstance(recorder_normalizer, SelectiveVecNormalize)
        assert recorder_normalizer.normalize_obs_excluded_indices == (0, 2)
    finally:
        train_env.close()


class TestSaveVecNormalizeOnNewBest:
    """The new-best snapshot is what makes ``best_model.zip`` replayable.

    Without it, ``vec_normalize.pkl`` is only written at end of training,
    so its running stats reflect the final train_env state — not the
    moment best_model was saved. ``record_best_model_video`` would then
    feed best_model obs normalized by mismatched stats.
    """

    @staticmethod
    def _build_vec_normalize():
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize

        from courtside_dynamics.envs import BallBalanceEnv

        venv = make_vec_env(lambda: BallBalanceEnv(), n_envs=1)
        return VecNormalize(venv, norm_obs=True, norm_reward=False)

    def test_writes_paired_file_when_vec_normalize_present(self, tmp_path):
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecNormalize,
        )

        from courtside_dynamics.envs import BallBalanceEnv
        from courtside_dynamics.training.train import (
            _SaveVecNormalizeOnNewBest,
        )

        vec_norm = self._build_vec_normalize()
        # Step a few times so obs_rms has a non-default running mean —
        # otherwise a "passed" test could mean we saved the SB3 default.
        vec_norm.reset()
        for _ in range(5):
            vec_norm.step(np.zeros((1, 6), dtype=np.float32))

        model = _FakeModel(action_dim=6)
        model.get_vec_normalize_env = lambda: vec_norm  # type: ignore[attr-defined]

        save_path = tmp_path / "best_vec_normalize.pkl"
        cb = _SaveVecNormalizeOnNewBest(str(save_path))
        cb.model = model  # type: ignore[assignment]
        assert cb._on_step() is True
        assert save_path.exists()

        # Loading should reconstruct an equivalent VecNormalize. obs_rms
        # mean must match the one we just trained, not the default zeros.
        dummy = DummyVecEnv([lambda: BallBalanceEnv()])
        loaded = VecNormalize.load(str(save_path), dummy)
        np.testing.assert_allclose(loaded.obs_rms.mean, vec_norm.obs_rms.mean)
        np.testing.assert_allclose(loaded.obs_rms.var, vec_norm.obs_rms.var)
        dummy.close()
        vec_norm.close()

    def test_noop_when_model_has_no_vec_normalize(self, tmp_path):
        """A model trained without VecNormalize must not raise or write."""
        from courtside_dynamics.training.train import (
            _SaveVecNormalizeOnNewBest,
        )

        model = _FakeModel(action_dim=6)
        model.get_vec_normalize_env = lambda: None  # type: ignore[attr-defined]

        save_path = tmp_path / "best_vec_normalize.pkl"
        cb = _SaveVecNormalizeOnNewBest(str(save_path))
        cb.model = model  # type: ignore[assignment]
        assert cb._on_step() is True
        assert not save_path.exists()


def test_info_dict_eval_callback_aggregates(tmp_path):
    """End-to-end: eval callback records per-episode hit/bounce metrics.

    We don't train a real model — ``_FakeModel`` returns a fixed action,
    so we just need to verify the aggregation pipeline produces the
    expected TB tags with sane values for an env without a phase key.
    """
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
    from courtside_dynamics.envs import WallBallEnv

    eval_env = make_vec_env(lambda: WallBallEnv(episode_len=30), n_envs=1)
    cb = InfoDictEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=2,
        eval_freq=1,
        log_prefix="eval_info",
    )
    cb.model = _FakeModel(action_dim=3)
    cb.n_calls = cb.eval_freq
    cb.num_timesteps = cb.eval_freq
    cb._on_step()
    eval_env.close()

    tb = cb.model.logger.records
    # Per-episode aggregates: final + max for counter-style keys.
    assert "eval_info/bounce_count_final" in tb
    assert "eval_info/bounce_count_max" in tb
    assert "eval_info/paddle_hit_count_final" in tb
    assert "eval_info/wall_contact_count_final" in tb
    # Mean metrics for continuous values.
    assert "eval_info/paddle_touch_mean" in tb
    assert "eval_info/wall_touch_mean" in tb
    assert "eval_info/sensor_data_mean" in tb
    assert "eval_info/episode_length" in tb
    # No phase key configured -> no phase fractions.
    assert not any(k.startswith("eval_info/phase_frac") for k in tb)


def test_info_dict_eval_success_rate_and_ep_mean(tmp_path):
    """With a ``success_key`` set, the callback logs a success rate plus
    per-episode terminal means (``<key>_ep_mean``)."""
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
    from courtside_dynamics.envs import WallBallEnv

    eval_env = make_vec_env(lambda: WallBallEnv(episode_len=30), n_envs=1)
    cb = InfoDictEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=2,
        eval_freq=1,
        log_prefix="eval_info",
        success_key="bounce_count",
        success_threshold=1.0,
    )
    cb.model = _FakeModel(action_dim=3)
    cb.n_calls = cb.eval_freq
    cb.num_timesteps = cb.eval_freq
    cb._on_step()
    eval_env.close()

    tb = cb.model.logger.records
    assert "eval_info/success_rate" in tb
    assert 0.0 <= tb["eval_info/success_rate"] <= 1.0
    # Per-episode terminal mean of a counter key.
    assert "eval_info/bounce_count_ep_mean" in tb
    # The no-op fake policy never completes a rally, so success rate is 0.
    assert tb["eval_info/success_rate"] == 0.0


def test_info_dict_eval_success_rate_omitted_when_key_absent(tmp_path):
    """A success_key that the env never emits must omit success_rate
    entirely (a visible gap) rather than report a confident 0.0."""
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
    from courtside_dynamics.envs import WallBallEnv

    eval_env = make_vec_env(lambda: WallBallEnv(episode_len=30), n_envs=1)
    cb = InfoDictEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=1,
        eval_freq=1,
        log_prefix="eval_info",
        success_key="not_a_real_key",  # WallBall never emits this
        success_threshold=1.0,
    )
    cb.model = _FakeModel(action_dim=3)
    cb.n_calls = cb.eval_freq
    cb.num_timesteps = cb.eval_freq
    cb._on_step()
    eval_env.close()

    assert "eval_info/success_rate" not in cb.model.logger.records


def test_info_dict_eval_compact_schema_and_episode_distribution(tmp_path):
    """Allowlisted metrics stay compact and report rally distributions."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    class _OneStepMetricEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -1.0, 1.0, (1,), dtype=np.float32
            )
            self.episode_index = -1

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.episode_index = (self.episode_index + 1) % 4
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            del action
            rally_count = (0, 1, 3, 4)[self.episode_index]
            info = {
                "rally_count": rally_count,
                "term_ball_net": self.episode_index in {0, 2},
                "debug_contact_peak": 999.0,
            }
            return np.zeros(1, dtype=np.float32), 0.0, True, False, info

    eval_env = make_vec_env(_OneStepMetricEnv, n_envs=1)
    cb = InfoDictEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=4,
        eval_freq=1,
        info_keys=("rally_count",),
        terminal_info_keys=("term_ball_net",),
        episode_distribution_keys=("rally_count",),
        episode_survival_thresholds={"rally_count": (2, 3, 5)},
        csv_path=str(tmp_path / "eval_info.csv"),
    )
    cb.model = _FakeModel(action_dim=1)
    cb.n_calls = cb.eval_freq
    cb.num_timesteps = cb.eval_freq
    cb._on_step()
    eval_env.close()

    tb = cb.model.logger.records
    assert tb["eval_info/term_ball_net_ep_mean"] == pytest.approx(0.5)
    assert tb["eval_info/rally_count_ep_min"] == 0.0
    assert tb["eval_info/rally_count_ep_p50"] == pytest.approx(2.0)
    assert tb["eval_info/rally_count_ep_p90"] == pytest.approx(3.7)
    assert tb["eval_info/rally_count_ep_max"] == 4.0
    assert tb["eval_info/rally_count_ep_ge_2_rate"] == pytest.approx(0.5)
    assert tb["eval_info/rally_count_ep_ge_3_rate"] == pytest.approx(0.5)
    assert tb["eval_info/rally_count_ep_ge_5_rate"] == 0.0
    assert not any("debug_contact_peak" in key for key in tb)
    assert "eval_info/term_ball_net_mean" not in tb
    assert "eval_info/term_ball_net_max" not in tb
    assert "eval_info/term_ball_net_final" not in tb

    with open(tmp_path / "eval_info.csv") as f:
        rows = list(csv.DictReader(f))
    metric_names = {row["metric"] for row in rows}
    assert "rally_count_ep_p50" in metric_names
    assert "rally_count_ep_ge_2_rate" in metric_names
    assert "rally_count_ep_ge_5_rate" in metric_names
    assert "term_ball_net_ep_mean" in metric_names
    assert not any("debug_contact_peak" in name for name in metric_names)


class _FakeSavableModel(_FakeModel):
    """_FakeModel plus SB3's ``save`` contract (appends ``.zip``)."""

    def __init__(self, action_dim: int = 1) -> None:
        super().__init__(action_dim)
        self.save_calls: list[str] = []

    def save(self, path) -> None:
        path = str(path)
        if not path.endswith(".zip"):
            path += ".zip"
        with open(path, "wb") as f:
            f.write(b"fake-model")
        self.save_calls.append(path)


def test_info_dict_eval_task_metric_selection_and_early_stop(tmp_path):
    """Selection compares metrics lexicographically; patience stops training.

    Regression scope: run 20260712_190054's best_model.zip was chosen by
    mean eval reward and completed zero rallies. Selection must follow
    the task metric, with reward only breaking ties, and the early stop
    must count evaluations without improvement of that same score.
    """
    import hashlib
    import json

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),  # never touched by the selection logic
        best_metric_keys=("bounce_count_ep_mean", "episode_reward_mean"),
        best_model_save_path=str(tmp_path),
        early_stop_patience=2,
        early_stop_min_evals=0,
    )
    cb.model = _FakeSavableModel(action_dim=3)

    class FakeNormalizer:
        @staticmethod
        def save(path) -> None:
            with open(path, "wb") as handle:
                handle.write(b"fake-normalizer")

    cb.model.get_vec_normalize_env = lambda: FakeNormalizer()  # type: ignore[attr-defined]

    def meta():
        with open(tmp_path / "best_model_meta.json") as f:
            return json.load(f)

    # First eval: always a new best.
    cb.num_timesteps = 100
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0, "episode_reward_mean": 5.0}
    )
    assert (tmp_path / "best_model.zip").exists()
    assert meta()["timestep"] == 100
    assert meta()["artifacts"]["best_model.zip"]["sha256"] == hashlib.sha256(
        b"fake-model"
    ).hexdigest()
    assert meta()["artifacts"]["best_vec_normalize.pkl"][
        "sha256"
    ] == hashlib.sha256(b"fake-normalizer").hexdigest()

    # Tied primary metric, better reward tie-break: an improvement.
    cb.num_timesteps = 200
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0, "episode_reward_mean": 6.0}
    )
    assert meta()["timestep"] == 200
    assert meta()["selection_values"]["episode_reward_mean"] == 6.0

    # Higher reward must NOT outrank a lower task metric.
    cb.num_timesteps = 300
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 0.5, "episode_reward_mean": 99.0}
    )
    assert meta()["timestep"] == 200, "reward overrode the task metric"

    # Second consecutive non-improvement hits patience=2: stop.
    cb.num_timesteps = 400
    assert not cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 0.5, "episode_reward_mean": 99.0}
    )
    assert meta()["timestep"] == 200


def test_info_dict_eval_min_delta_ignores_noise_improvements(tmp_path):
    """Sub-delta differences neither crown a best nor reset the patience.

    Regression scope: run 20260714_211111's best_model.zip was selected
    by a ~1e-8 ``episode_reward_mean`` difference while every task
    metric was 0.0, and that noise event reset the early-stop patience,
    extending a provably dead run by 225k steps.
    """
    import json

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),  # never touched by the selection logic
        best_metric_keys=("bounce_count_ep_mean", "episode_reward_mean"),
        best_model_save_path=str(tmp_path),
        early_stop_patience=2,
        early_stop_min_evals=0,
        # Half the granularity of a 30-episode mean: one real episode
        # (1/30 ~= 0.033) clears it, float noise never does.
        best_metric_min_delta=0.5 / 30,
    )
    cb.model = _FakeSavableModel(action_dim=3)
    cb.model.get_vec_normalize_env = lambda: None  # type: ignore[attr-defined]

    def best_timestep() -> int:
        with open(tmp_path / "best_model_meta.json") as f:
            return json.load(f)["timestep"]

    cb.num_timesteps = 100
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 0.0, "episode_reward_mean": -1.0}
    )
    assert best_timestep() == 100

    # Float-noise "improvement" on the reward tie-break: not a new best.
    cb.num_timesteps = 200
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 0.0, "episode_reward_mean": -1.0 + 1e-8}
    )
    assert best_timestep() == 100

    # A real one-episode change (1/30) clears the threshold.
    cb.num_timesteps = 300
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0 / 30, "episode_reward_mean": -1.0}
    )
    assert best_timestep() == 300

    # Two flat evaluations exhaust patience=2: noise can no longer
    # keep the run alive.
    cb.num_timesteps = 400
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0 / 30, "episode_reward_mean": -1.0 + 1e-8}
    )
    cb.num_timesteps = 500
    assert not cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0 / 30, "episode_reward_mean": -1.0 - 1e-9}
    )
    assert best_timestep() == 300


def test_info_dict_eval_warmup_does_not_count_against_patience():
    """Earliest stop is eval ``min_evals + patience``.

    Regression scope: the documented "at least 2N evaluations" contract
    was violated because evaluations during the warm-up accrued
    ``_evals_since_best``, allowing a stop at eval ``min_evals + 1``.
    """
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=("bounce_count_ep_mean",),
        early_stop_patience=2,
        early_stop_min_evals=3,
    )
    cb.model = _FakeSavableModel(action_dim=3)

    flat = {"bounce_count_ep_mean": 0.0}
    # Eval 1 sets the first best; evals 2-3 are warm-up and must not
    # count toward the patience.
    for step in (1, 2, 3, 4):
        cb.num_timesteps = step
        assert cb._update_best_and_maybe_stop(flat) is True, f"eval {step}"
    # Eval 5 = min_evals (3) + patience (2): the stop may now fire.
    cb.num_timesteps = 5
    assert cb._update_best_and_maybe_stop(flat) is False


def test_info_dict_eval_degenerate_signal_stops_dead_runs():
    """Flat selection score + zero ball contact ends the run at the guard
    window, long before patience.

    Regression scope: run 20260714_211111 was provably dead (reward
    exactly -1.0, zero paddle contact) from its second evaluation yet
    burned 750k steps before the patience fired.
    """
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=("bounce_count_ep_mean",),
        early_stop_patience=20,
        degenerate_stop_evals=3,
        degenerate_guard_keys=("paddle_hit_count_ep_mean",),
    )
    cb.model = _FakeSavableModel(action_dim=3)

    dead = {"bounce_count_ep_mean": 0.0, "paddle_hit_count_ep_mean": 0.0}
    cb.num_timesteps = 1
    assert cb._update_best_and_maybe_stop(dead) is True
    cb.num_timesteps = 2
    assert cb._update_best_and_maybe_stop(dead) is True
    cb.num_timesteps = 3
    assert cb._update_best_and_maybe_stop(dead) is False


def test_info_dict_eval_degenerate_guard_needs_zero_contact():
    """A flat score with nonzero guard metrics (the policy touches the
    ball) or a missing guard key must NOT trip the degenerate stop."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    for metrics in (
        # Flat score, but the policy interacts with the ball.
        {"bounce_count_ep_mean": 0.0, "paddle_hit_count_ep_mean": 0.5},
        # Flat score, guard key absent: cannot conclude the run is dead.
        {"bounce_count_ep_mean": 0.0},
    ):
        cb = InfoDictEvalCallback(
            eval_env=object(),
            best_metric_keys=("bounce_count_ep_mean",),
            degenerate_stop_evals=3,
            degenerate_guard_keys=("paddle_hit_count_ep_mean",),
        )
        cb.model = _FakeSavableModel(action_dim=3)
        for step in (1, 2, 3, 4, 5):
            cb.num_timesteps = step
            assert cb._update_best_and_maybe_stop(metrics) is True, metrics


def test_info_dict_eval_degenerate_guard_tolerates_float_noise():
    """A float-noise selection key must not disarm the degenerate stop.

    Regression scope: run 20260714_211111's episode_reward_mean jittered
    by ~1e-8 between evaluations of a dead policy; a guard requiring
    bitwise-identical scores would never fire on any key set that keeps
    a continuous reward tie-break.
    """
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=("bounce_count_ep_mean", "episode_reward_mean"),
        best_metric_min_delta=0.5 / 30,
        degenerate_stop_evals=3,
        degenerate_guard_keys=("paddle_hit_count_ep_mean",),
    )
    cb.model = _FakeSavableModel(action_dim=3)

    outcomes = []
    for step, reward in enumerate((-1.0, -1.0 + 1e-8, -1.0 - 2e-8), 1):
        cb.num_timesteps = step
        outcomes.append(
            cb._update_best_and_maybe_stop(
                {
                    "bounce_count_ep_mean": 0.0,
                    "episode_reward_mean": reward,
                    "paddle_hit_count_ep_mean": 0.0,
                }
            )
        )
    assert outcomes == [True, True, False]


def test_info_dict_eval_degenerate_guard_warmup():
    """Evaluations inside ``degenerate_min_evals`` never enter the guard
    window, so a curriculum run cannot be killed while its schedule still
    holds the start distribution (where a dead full-difficulty eval is
    expected by design)."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=("bounce_count_ep_mean",),
        degenerate_stop_evals=2,
        degenerate_guard_keys=("paddle_hit_count_ep_mean",),
        degenerate_min_evals=3,
    )
    cb.model = _FakeSavableModel(action_dim=3)

    dead = {"bounce_count_ep_mean": 0.0, "paddle_hit_count_ep_mean": 0.0}
    # Evals 1-3 are warm-up; evals 4-5 fill the window; stop at eval 5
    # = degenerate_min_evals + degenerate_stop_evals.
    for step in (1, 2, 3, 4):
        cb.num_timesteps = step
        assert cb._update_best_and_maybe_stop(dead) is True, f"eval {step}"
    cb.num_timesteps = 5
    assert cb._update_best_and_maybe_stop(dead) is False


def test_info_dict_eval_confirm_best_banks_weaker_by_delta_order(
    monkeypatch,
):
    """The banked best uses the same delta-tolerant ordering as
    ``_improves``: when the first key ties within delta and the tie-break
    key decides, the sample with the lower tie-break is the weaker one --
    even if its first key is numerically larger, where a raw tuple
    ``min`` would bank the stronger sample and ratchet the best up to a
    lucky high-water mark."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=(
            "bounce_count_ep_mean",
            "bounce_count_ep_ge_2_rate",
        ),
        confirm_best=True,
        best_metric_min_delta=1.0 / 60,
    )
    cb.model = _FakeSavableModel(action_dim=3)

    cb.num_timesteps = 100
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.0, "bounce_count_ep_ge_2_rate": 0.2}
    )

    # Primary (1.010, 0.90) vs confirmation (1.001, 0.95): key 0 ties
    # within delta, key 1 says the confirmation is stronger, so the
    # primary must be banked. Raw tuple min would pick (1.001, 0.95).
    monkeypatch.setattr(
        cb,
        "_collect_metrics",
        lambda: {
            "bounce_count_ep_mean": 1.001,
            "bounce_count_ep_ge_2_rate": 0.95,
        },
    )
    cb.num_timesteps = 200
    assert cb._update_best_and_maybe_stop(
        {"bounce_count_ep_mean": 1.010, "bounce_count_ep_ge_2_rate": 0.90}
    )
    assert cb._best_score == (1.010, 0.90)


def test_info_dict_eval_confirm_best_requires_second_batch(
    tmp_path, monkeypatch
):
    """A candidate best must also win an independent confirmation batch.

    Regression scope: run 20260714_050506's best checkpoint (600k) beat
    its plateau by exactly one lucky 2-bounce episode in 30 -- a
    single-batch fluke that a second sample would have rejected.
    """
    import json

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    cb = InfoDictEvalCallback(
        eval_env=object(),
        best_metric_keys=("bounce_count_ep_mean",),
        best_model_save_path=str(tmp_path),
        confirm_best=True,
        best_metric_min_delta=0.5 / 30,
    )
    cb.model = _FakeSavableModel(action_dim=3)
    cb.model.get_vec_normalize_env = lambda: None  # type: ignore[attr-defined]

    def meta() -> dict:
        with open(tmp_path / "best_model_meta.json") as f:
            return json.load(f)

    # The first evaluation is accepted without a confirmation pass:
    # there is no best to defend yet.
    cb.num_timesteps = 100
    assert cb._update_best_and_maybe_stop({"bounce_count_ep_mean": 1.0})
    assert meta()["timestep"] == 100
    assert "confirmation_values" not in meta()

    # Candidate improvement whose confirmation batch ties the old best:
    # rejected, the fluke does not become best_model.zip.
    monkeypatch.setattr(
        cb, "_collect_metrics", lambda: {"bounce_count_ep_mean": 1.0}
    )
    cb.num_timesteps = 200
    assert cb._update_best_and_maybe_stop({"bounce_count_ep_mean": 1.1})
    assert meta()["timestep"] == 100

    # Confirmed improvement: accepted, both batches recorded, and the
    # stored best score is the weaker of the two samples.
    monkeypatch.setattr(
        cb, "_collect_metrics", lambda: {"bounce_count_ep_mean": 1.2}
    )
    cb.num_timesteps = 300
    assert cb._update_best_and_maybe_stop({"bounce_count_ep_mean": 1.3})
    assert meta()["timestep"] == 300
    assert meta()["selection_values"] == {"bounce_count_ep_mean": 1.3}
    assert meta()["confirmation_values"] == {"bounce_count_ep_mean": 1.2}
    assert cb._best_score == (1.2,)


def test_info_dict_eval_selection_writes_artifacts_end_to_end(tmp_path):
    """A live eval pass under ``best_metric_keys`` writes model + meta and
    logs ``episode_reward_mean`` alongside the info aggregates."""
    import json

    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback
    from courtside_dynamics.envs import WallBallEnv

    eval_env = make_vec_env(lambda: WallBallEnv(episode_len=30), n_envs=1)
    cb = InfoDictEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=2,
        eval_freq=1,
        success_key="bounce_count",
        best_metric_keys=(
            "bounce_count_ep_mean",
            "success_rate",
            "episode_reward_mean",
        ),
        best_model_save_path=str(tmp_path),
    )
    cb.model = _FakeSavableModel(action_dim=3)
    cb.n_calls = cb.eval_freq
    cb.num_timesteps = 12345
    cb._on_step()
    eval_env.close()

    assert "eval_info/episode_reward_mean" in cb.model.logger.records
    assert (tmp_path / "best_model.zip").exists()
    with open(tmp_path / "best_model_meta.json") as f:
        meta = json.load(f)
    assert meta["timestep"] == 12345
    assert meta["selection_keys"] == [
        "bounce_count_ep_mean",
        "success_rate",
        "episode_reward_mean",
    ]
    assert set(meta["selection_values"]) == set(meta["selection_keys"])
    assert set(meta["artifacts"]) == {"best_model.zip"}


def test_info_dict_eval_archive_best_copies_the_best_triple(tmp_path):
    """``archive_best`` snapshots whatever ``_save_best`` wrote -- the
    stage-gated curriculum calls it just before ``reset_selection_state``
    would let the next stage's first eval overwrite the champion."""
    import json

    from gymnasium.envs.classic_control import PendulumEnv
    from stable_baselines3.common.env_util import make_vec_env

    from courtside_dynamics.callbacks.info_dict_eval import (
        InfoDictEvalCallback,
    )

    eval_env = make_vec_env(PendulumEnv, n_envs=1)
    try:
        save_dir = tmp_path / "model"
        save_dir.mkdir()
        callback = InfoDictEvalCallback(
            eval_env, best_model_save_path=str(save_dir)
        )

        # Nothing saved yet: archiving is a clean no-op.
        empty_dest = tmp_path / "stage_bests" / "stage_00"
        assert callback.archive_best(str(empty_dest)) is None
        assert not empty_dest.exists()

        # Simulate a saved best triple (contents are opaque bytes to the
        # archive; only the meta json is parsed).
        (save_dir / "best_model.zip").write_bytes(b"model-bytes")
        (save_dir / "best_vec_normalize.pkl").write_bytes(b"norm-bytes")
        meta = {"timestep": 123, "context": {"curriculum_stage_index": 1.0}}
        (save_dir / "best_model_meta.json").write_text(
            json.dumps(meta) + "\n"
        )

        dest = tmp_path / "stage_bests" / "stage_01"
        returned = callback.archive_best(str(dest))
        assert returned == meta
        assert (dest / "best_model.zip").read_bytes() == b"model-bytes"
        assert (dest / "best_vec_normalize.pkl").read_bytes() == b"norm-bytes"
        assert json.loads((dest / "best_model_meta.json").read_text()) == meta
        # Originals stay in place: the archive is a copy, not a move.
        assert (save_dir / "best_model.zip").exists()

        # Without a normalizer file (non-VecNormalize runs), the model
        # and meta still archive.
        (save_dir / "best_vec_normalize.pkl").unlink()
        dest2 = tmp_path / "stage_bests" / "stage_02"
        assert callback.archive_best(str(dest2)) == meta
        assert (dest2 / "best_model.zip").exists()
        assert not (dest2 / "best_vec_normalize.pkl").exists()

        # No save path configured: always a no-op.
        callback_no_path = InfoDictEvalCallback(eval_env)
        assert callback_no_path.archive_best(str(tmp_path / "x")) is None
    finally:
        eval_env.close()
