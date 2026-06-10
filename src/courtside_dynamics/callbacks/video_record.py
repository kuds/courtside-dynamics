"""Unified video recording callback.

The notebooks in the original repository each carried a near-identical
copy of this callback, with small divergences in how they formatted the
per-step CSV log. This version takes a pluggable ``info_row_fn`` so each
environment can decide which pieces of its ``info`` dict to log without
forking the callback itself.

When neither ``info_row_fn`` nor ``csv_header`` is supplied, the callback
auto-detects the scalar keys of the env's first ``info`` dict and logs
them alongside the reward columns, both to the CSV and to TensorBoard
under ``videorecord/<key>``.
"""
from __future__ import annotations

import csv
import os
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

# ``_scalar_info_keys`` lives in the neutral ``_info`` module so this
# callback and ``InfoDictEvalCallback`` can share it without one importing
# a private name from the other. Re-exported here for back-compat: existing
# code (and tests) import it from this module.
from courtside_dynamics.callbacks._info import _scalar_info_keys

InfoRowFn = Callable[[dict, float, float, bool], Sequence[object]]
"""Signature: ``(info, reward, total_reward, done) -> row values``."""


def _default_info_row_fn(
    info: dict, reward: float, total_reward: float, done: bool
) -> Sequence[object]:
    """Fallback row formatter when an env doesn't supply one."""
    return [reward, total_reward, done]


def _default_csv_header() -> Sequence[str]:
    return ["reward", "total_reward", "done"]


class VideoRecordCallback(BaseCallback):
    """Record a video and CSV rollout of the current policy on a schedule.

    Parameters
    ----------
    env_fn:
        Factory that produces a fresh (unwrapped) Gymnasium environment
        matching the one the model is training on. The callback builds its
        own single-env vector env from this so that recording doesn't
        interfere with the training env.
    save_path:
        Directory where videos and CSVs are written. Created if missing.
    video_length:
        Maximum number of steps to roll out when recording.
    save_freq:
        Record every ``save_freq`` training steps.
    name_prefix:
        Filename prefix for both the video and CSV output.
    csv_header:
        Optional CSV header row. If ``None`` *and* ``info_row_fn`` is also
        ``None``, the header is derived at recording time from the scalar
        keys of the env's first ``info`` dict.
    info_row_fn:
        Optional callable returning the per-step CSV row. If ``None``, the
        callback auto-logs scalar info keys.
    tb_log_prefix:
        TensorBoard tag prefix for auto-logged scalar info keys. Ignored
        when the user supplies their own ``info_row_fn``.
    seed:
        Optional seed for the throwaway recording env, so the replayed
        rollout is reproducible across runs. ``None`` leaves it unseeded.
    deterministic:
        Passed to ``model.predict``. Defaults to ``True`` so the video
        shows the same deterministic policy ``EvalCallback`` scores --
        a stochastic rollout (the old behavior) understates a trained
        SAC policy and made videos hard to reconcile with eval curves.
    """

    def __init__(
        self,
        env_fn: Callable,
        save_path: str,
        video_length: int,
        save_freq: int = 5_000,
        name_prefix: str = "rl_model",
        csv_header: Iterable[str] | None = None,
        info_row_fn: InfoRowFn | None = None,
        tb_log_prefix: str = "videorecord",
        seed: int | None = None,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.env_fn = env_fn
        self.save_path = save_path
        self.video_length = video_length
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.tb_log_prefix = tb_log_prefix
        self.seed = seed
        self.deterministic = deterministic

        self._user_info_row_fn = info_row_fn
        self._user_csv_header = list(csv_header) if csv_header else None
        # Resolved lazily on first record so auto-detection can see a real
        # ``info`` dict. ``info_row_fn``/``csv_header`` are re-derived per
        # rollout only when the user didn't pin them.
        self.csv_header: list[str] = self._user_csv_header or list(
            _default_csv_header()
        )
        self.info_row_fn: InfoRowFn = (
            info_row_fn if info_row_fn is not None else _default_info_row_fn
        )

    def _auto_configure(self, info: Mapping) -> list[str]:
        """Populate ``csv_header``/``info_row_fn`` from a first info dict.

        Returns the list of scalar info keys being auto-logged (possibly
        empty). Only runs when the user didn't pin an explicit formatter.
        """
        if self._user_info_row_fn is not None:
            return []

        info_keys = _scalar_info_keys(info)
        if self._user_csv_header is None:
            self.csv_header = list(info_keys) + list(_default_csv_header())

        def _row(
            info: dict, reward: float, total_reward: float, done: bool
        ) -> Sequence[object]:
            return [info.get(k) for k in info_keys] + [reward, total_reward, done]

        self.info_row_fn = _row
        return info_keys

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        os.makedirs(self.save_path, exist_ok=True)
        name_prefix = f"{self.name_prefix}_{self.num_timesteps}"

        rec_env = make_vec_env(self.env_fn, n_envs=1, seed=self.seed)

        # If the policy was trained with VecNormalize, wrap the recording
        # env in a frozen-stats VecNormalize so the policy sees obs on
        # the same scale it trained on. Otherwise the rollout will
        # diverge wildly from the eval curve.
        get_vec_norm = getattr(self.model, "get_vec_normalize_env", None)
        train_vec_norm = get_vec_norm() if get_vec_norm is not None else None
        if train_vec_norm is not None:
            from stable_baselines3.common.vec_env import (
                VecNormalize,
                sync_envs_normalization,
            )

            rec_env = VecNormalize(
                rec_env,
                training=False,
                norm_obs=train_vec_norm.norm_obs,
                norm_reward=False,
                clip_obs=train_vec_norm.clip_obs,
            )
            try:
                sync_envs_normalization(self.training_env, rec_env)
            except (AttributeError, AssertionError):
                pass

        rec_env = VecVideoRecorder(
            rec_env,
            self.save_path,
            video_length=self.video_length,
            record_video_trigger=lambda x: x == 0,
            name_prefix=name_prefix,
        )

        try:
            obs = rec_env.reset()
            # SB3 VecEnv.reset() is typed as ndarray | dict | tuple, but
            # ``VecVideoRecorder`` over a Box obs space always yields ndarray.
            assert not isinstance(obs, tuple)
            session_length = 0
            total_reward = 0.0
            csv_path = os.path.join(self.save_path, f"{name_prefix}.csv")
            auto_keys: list[str] = []
            auto_sums: dict[str, float] = {}
            header_written = False

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                for _ in range(self.video_length):
                    session_length += 1
                    action, _ = self.model.predict(
                        obs, deterministic=self.deterministic
                    )
                    obs, rewards, dones, infos = rec_env.step(action)
                    assert not isinstance(obs, tuple)
                    total_reward += float(rewards[0])
                    info = infos[0]

                    if not header_written:
                        auto_keys = self._auto_configure(info)
                        auto_sums = {k: 0.0 for k in auto_keys}
                        writer.writerow(self.csv_header)
                        header_written = True

                    row = self.info_row_fn(
                        info,
                        float(rewards[0]),
                        float(total_reward),
                        bool(dones[0]),
                    )
                    writer.writerow(_flatten_row(row))

                    for k in auto_keys:
                        value = info.get(k)
                        if value is None:
                            continue
                        try:
                            auto_sums[k] += float(value)
                        except (TypeError, ValueError):
                            # Non-numeric scalar slipped through; skip TB average.
                            auto_sums.pop(k, None)

                    rec_env.render()
                    if dones[0]:
                        break

            # TensorBoard: emit the final-step snapshot plus a rollout mean
            # per scalar info key. Also log the total reward so eyeballing
            # the curve in TB matches the CSV.
            self.logger.record(
                f"{self.tb_log_prefix}/total_reward", float(total_reward)
            )
            self.logger.record(
                f"{self.tb_log_prefix}/episode_length", int(session_length)
            )
            if session_length > 0:
                for k, total in auto_sums.items():
                    self.logger.record(
                        f"{self.tb_log_prefix}/{k}_mean",
                        total / session_length,
                    )

            if self.verbose:
                print(
                    f"[VideoRecordCallback] step={self.num_timesteps} "
                    f"len={session_length} total_reward={total_reward:.2f}"
                )
        finally:
            rec_env.close()

        return True


def _flatten_row(row: Sequence[object]) -> list:
    """Flatten numpy arrays inside a row into scalar values."""
    out: list = []
    for item in row:
        if isinstance(item, np.ndarray):
            out.extend(np.asarray(item).ravel().tolist())
        else:
            out.append(item)
    return out
