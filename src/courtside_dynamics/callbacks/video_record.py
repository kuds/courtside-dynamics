"""Unified video recording callback.

The notebooks in the original repository each carried a near-identical
copy of this callback, with small divergences in how they formatted the
per-step CSV log. This version takes a pluggable ``info_row_fn`` so each
environment can decide which pieces of its ``info`` dict to log without
forking the callback itself.
"""
from __future__ import annotations

import csv
import os
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

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
        Optional CSV header row; if ``None``, uses ``_default_csv_header``.
    info_row_fn:
        Optional callable returning the per-step CSV row; if ``None``,
        uses ``_default_info_row_fn``.
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
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.env_fn = env_fn
        self.save_path = save_path
        self.video_length = video_length
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.csv_header = list(csv_header) if csv_header else list(_default_csv_header())
        self.info_row_fn = info_row_fn or _default_info_row_fn

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        os.makedirs(self.save_path, exist_ok=True)
        name_prefix = f"{self.name_prefix}_{self.num_timesteps}"

        rec_env = make_vec_env(self.env_fn, n_envs=1)
        rec_env = VecVideoRecorder(
            rec_env,
            self.save_path,
            video_length=self.video_length,
            record_video_trigger=lambda x: x == 0,
            name_prefix=name_prefix,
        )

        try:
            obs = rec_env.reset()
            session_length = 0
            total_reward = 0.0
            csv_path = os.path.join(self.save_path, f"{name_prefix}.csv")

            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.csv_header)

                for _ in range(self.video_length):
                    session_length += 1
                    action, _ = self.model.predict(obs)
                    obs, rewards, dones, infos = rec_env.step(action)
                    total_reward += float(rewards[0])
                    row = self.info_row_fn(
                        infos[0],
                        float(rewards[0]),
                        float(total_reward),
                        bool(dones[0]),
                    )
                    writer.writerow(_flatten_row(row))
                    rec_env.render()
                    if dones[0]:
                        break

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
