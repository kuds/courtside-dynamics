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
from typing import Any

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
        Maximum number of steps to roll out when recording (the hard cap
        across all recorded episodes).
    max_episodes:
        Number of episodes to record before stopping, within the
        ``video_length`` step cap. The recorder historically stopped at
        the FIRST episode end regardless of ``video_length`` (run
        20260721_004722's best-model video is 104 frames against a
        10,000-step budget), leaving an n=1 behavioral record per
        milestone. Default 3; ``None`` records episodes until
        ``video_length`` is exhausted.
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
        max_episodes: int | None = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if max_episodes is not None and (
            isinstance(max_episodes, bool)
            or not isinstance(max_episodes, int)
            or max_episodes < 1
        ):
            raise ValueError(
                "max_episodes must be None or a positive integer"
            )
        if csv_header and info_row_fn is None:
            # The auto row formatter writes the env's scalar info keys
            # plus the reward columns; under a user-pinned header of a
            # different width every data row silently misaligns. (A TOML
            # can set csv_header but never info_row_fn -- callbacks are
            # code -- so this pairing must fail loudly here.)
            raise ValueError(
                "csv_header without info_row_fn: the auto-detected row "
                "formatter would write rows keyed to the env's scalar "
                "info keys under your fixed header; supply info_row_fn "
                "with the matching row shape, or drop csv_header to "
                "auto-derive both"
            )
        self.env_fn = env_fn
        self.save_path = save_path
        self.video_length = video_length
        self.max_episodes = max_episodes
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
        # ``save_freq <= 0`` disables recording (same contract as
        # InfoDictEvalCallback's eval_freq) instead of raising
        # ZeroDivisionError on the modulo below.
        if self.save_freq <= 0 or self.n_calls % self.save_freq != 0:
            return True
        # Recording is an optional diagnostic: a missing GL backend or
        # video codec surfacing here must not propagate out of
        # model.learn() and cost the run its final artifacts
        # (KeyboardInterrupt still propagates -- the salvage path owns
        # it).
        try:
            self._record_rollout()
        except Exception as error:
            print(
                f"[VideoRecordCallback] recording at step "
                f"{self.num_timesteps} failed and was skipped: {error!r}"
            )
        return True

    def _record_rollout(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        name_prefix = f"{self.name_prefix}_{self.num_timesteps}"

        rec_env = make_vec_env(self.env_fn, n_envs=1, seed=self.seed)
        # From here on the env must be closed on every exit: the
        # wrap/recorder constructors can raise (and with recording
        # isolated, training then continues past the failure), so a
        # bare raise would leak one MuJoCo env per attempt.
        # ``rec_env`` always names the outermost constructed layer,
        # so closing it closes the whole stack (and finalizes the
        # video on success).
        try:
            # If the policy was trained with VecNormalize, wrap the recording
            # env in a frozen-stats VecNormalize so the policy sees obs on
            # the same scale it trained on. Otherwise the rollout will
            # diverge wildly from the eval curve.
            get_vec_norm = getattr(self.model, "get_vec_normalize_env", None)
            train_vec_norm = get_vec_norm() if get_vec_norm is not None else None
            if train_vec_norm is not None:
                from stable_baselines3.common.vec_env import sync_envs_normalization

                normalizer_kwargs: dict[str, Any] = {}
                excluded_indices = getattr(
                    train_vec_norm,
                    "normalize_obs_excluded_indices",
                    None,
                )
                if excluded_indices is not None:
                    normalizer_kwargs["normalize_obs_excluded_indices"] = (
                        excluded_indices
                    )
                rec_env = type(train_vec_norm)(
                    rec_env,
                    training=False,
                    norm_obs=train_vec_norm.norm_obs,
                    norm_reward=False,
                    clip_obs=train_vec_norm.clip_obs,
                    clip_reward=train_vec_norm.clip_reward,
                    gamma=train_vec_norm.gamma,
                    epsilon=train_vec_norm.epsilon,
                    **normalizer_kwargs,
                )
                try:
                    sync_envs_normalization(self.training_env, rec_env)
                except (AttributeError, AssertionError) as error:
                    # A wrapper-stack mismatch means the replay runs on the
                    # constructor's frozen stats instead of the live
                    # training stats. Keep recording -- the video is still
                    # evidence -- but never silently: a policy replayed on
                    # the wrong obs scale looks broken and sends the run
                    # review chasing a phantom regression.
                    print(
                        f"[VideoRecordCallback] could not sync normalization "
                        f"stats from {type(self.training_env).__name__} to "
                        f"the recording env: {error!r}; the video will use "
                        f"the training env's last-synced statistics"
                    )

            rec_env = VecVideoRecorder(
                rec_env,
                self.save_path,
                video_length=self.video_length,
                record_video_trigger=lambda x: x == 0,
                name_prefix=name_prefix,
            )

            obs = rec_env.reset()
            # SB3 VecEnv.reset() is typed as ndarray | dict | tuple, but
            # ``VecVideoRecorder`` over a Box obs space always yields ndarray.
            assert not isinstance(obs, tuple)
            session_length = 0
            total_reward = 0.0
            # Per-episode records: the VecEnv auto-resets on done, so the
            # rollout continues straight into the next episode until
            # ``max_episodes`` (or the ``video_length`` step cap). The CSV's
            # ``done`` column marks the boundaries; ``total_reward`` restarts
            # with each episode.
            episode_rewards: list[float] = []
            episode_lengths: list[int] = []
            episode_start = 0
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
                        # Once a key is dropped, stay dropped: the
                        # subscript load in ``auto_sums[k] += ...`` runs
                        # before float(), so a revisit after pop() would
                        # raise an uncaught KeyError mid-recording.
                        if k not in auto_sums:
                            continue
                        value = info.get(k)
                        if value is None:
                            continue
                        try:
                            auto_sums[k] += float(value)
                        except (TypeError, ValueError):
                            # Non-numeric scalar slipped through; skip TB average.
                            auto_sums.pop(k, None)

                    # No explicit render here: VecVideoRecorder.step_wait
                    # already captured this step's frame, so a second
                    # env.render() would only double the offscreen cost.
                    if dones[0]:
                        episode_rewards.append(total_reward)
                        episode_lengths.append(session_length - episode_start)
                        episode_start = session_length
                        total_reward = 0.0
                        if (
                            self.max_episodes is not None
                            and len(episode_rewards) >= self.max_episodes
                        ):
                            break

            # TensorBoard: per-episode reward/length means over the completed
            # episodes (falling back to the running totals when the cap cut
            # the first episode short), a completed-episode count, and a
            # whole-rollout mean per scalar info key.
            if episode_rewards:
                reported_reward = sum(episode_rewards) / len(episode_rewards)
                reported_length = sum(episode_lengths) / len(episode_lengths)
            else:
                reported_reward = total_reward
                reported_length = session_length
            self.logger.record(
                f"{self.tb_log_prefix}/total_reward", float(reported_reward)
            )
            self.logger.record(
                f"{self.tb_log_prefix}/episode_length", int(reported_length)
            )
            self.logger.record(
                f"{self.tb_log_prefix}/episodes", len(episode_rewards)
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
                    f"len={session_length} "
                    f"episodes={len(episode_rewards)} "
                    f"mean_episode_reward={reported_reward:.2f}"
                )
        finally:
            rec_env.close()


def _flatten_row(row: Sequence[object]) -> list:
    """Flatten numpy arrays inside a row into scalar values."""
    out: list = []
    for item in row:
        if isinstance(item, np.ndarray):
            out.extend(np.asarray(item).ravel().tolist())
        else:
            out.append(item)
    return out
