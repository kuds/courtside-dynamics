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
from pathlib import Path

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


def _run_callback_once(callback, tmp_path: Path) -> None:
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
    cb.model = _FakeModel(action_dim=5)
    _run_callback_once(cb, tmp_path)

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
    cb.model = _FakeModel(action_dim=5)
    _run_callback_once(cb, tmp_path)

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
    _run_callback_once(cb, tmp_path)

    csv_files = list(tmp_path.glob("*.csv"))
    with open(csv_files[0]) as f:
        rows = list(csv.reader(f))
    # Header falls back to the default reward triple.
    assert rows[0] == ["reward", "total_reward", "done"]


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
    cb.model = _FakeModel(action_dim=5)
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
    cb.model = _FakeModel(action_dim=5)
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
