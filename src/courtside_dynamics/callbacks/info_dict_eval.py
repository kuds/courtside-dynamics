"""Per-episode ``info`` dict aggregation for evaluation rollouts.

SB3's ``EvalCallback`` logs mean reward across eval episodes, but it
ignores everything in the ``info`` dict. For WallBall that's most of
the useful signal: without per-episode bounce counts, paddle/wall hit
counts, and (for envs that have one) time spent in each phase, the
TensorBoard curves don't tell you whether the agent is learning to
rally or just drifting.

``InfoDictEvalCallback`` rolls ``n_eval_episodes`` on a separate eval
``VecEnv``, collects every scalar ``info`` key it sees, and emits three
classes of aggregate to TensorBoard:

* ``<prefix>/<key>_mean`` — time-averaged scalar (for continuous values
  like ``paddle_touch``).
* ``<prefix>/<key>_final`` and ``<prefix>/<key>_max`` — terminal and
  peak values (for monotone counters like ``rally_count``).
* ``<prefix>/phase_frac_<label>`` — fraction of eval steps spent in each
  phase, for envs that expose a categorical ``phase_key``.

The metric aggregation is stateless across invocations; every
``_on_step`` trigger produces a fresh sample. When ``best_metric_keys``
is set the callback additionally owns best-model selection and early
stopping (see the parameter docs below): run ``20260712_190054`` proved
that selecting checkpoints by mean eval reward crowns whatever policy
maximizes the reward — including a degenerate one — while the task
metric collapses to zero, so selection must key on the task metric
itself.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from courtside_dynamics.callbacks._info import _scalar_info_keys


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InfoDictEvalCallback(BaseCallback):
    """Evaluate the policy and log per-episode ``info`` aggregates.

    Parameters
    ----------
    eval_env:
        A single-env ``VecEnv`` matching the training env's spec. The
        callback resets + steps this env directly; it doesn't share with
        SB3's ``EvalCallback``.
    n_eval_episodes:
        Number of episodes to roll out per evaluation.
    eval_freq:
        Trigger the evaluation every ``eval_freq`` training steps.
    log_prefix:
        TensorBoard tag prefix.
    phase_key:
        If provided, ``info[phase_key]`` is treated as a categorical
        integer and logged as ``phase_frac_<label>`` per label. Omit
        when the env has no phase concept.
    phase_labels:
        Mapping from phase integer to human-readable suffix. If omitted,
        labels default to ``"0"``, ``"1"`` etc.
    deterministic:
        Passed to ``model.predict``. Evaluation is almost always
        deterministic, which is the default.
    success_key / success_threshold:
        When ``success_key`` is set, an episode counts as a success if
        its terminal ``info[success_key]`` is ``>= success_threshold``,
        and the fraction of successful episodes is logged as
        ``<prefix>/success_rate``. For WallBall, ``success_key=
        "bounce_count"`` with threshold 1 yields "fraction of eval
        episodes that completed at least one full rally" -- the real task
        metric that mean reward (shaping-dominated) obscures.
    info_keys:
        Optional allowlist of scalar keys to aggregate on every step. The
        default, ``None``, preserves the original all-scalars behaviour.
    terminal_info_keys:
        Scalar keys sampled only from terminal ``info`` snapshots. Each is
        reported as ``<key>_ep_mean``; boolean ``term_*`` flags therefore
        become a fault-type distribution across evaluation episodes.
    episode_distribution_keys:
        Terminal scalar keys for which episode-level min, median, 90th
        percentile, and max metrics are emitted. ``rally_count`` is the
        intended use.
    episode_survival_thresholds:
        Mapping from a terminal counter key to positive integer thresholds.
        For every threshold, the callback logs the fraction of evaluation
        episodes whose terminal counter reached it as
        ``<key>_ep_ge_<threshold>_rate``. For example, WallBall configures
        ``{"bounce_count": (2, 3, 5)}`` so a one-return plateau remains
        visible even when the mean counter improves through rare outliers.
    csv_path:
        Optional path. When set, every evaluation appends one row per
        metric in long format (``timestep,metric,value``) so the data
        survives outside TensorBoard. The header is written on the
        first call.
    best_metric_keys:
        Ordered metric names (as they appear in this callback's metric
        dict, e.g. ``("bounce_count_ep_mean", "success_rate",
        "episode_reward_mean")``) compared lexicographically to decide
        whether an evaluation is a new best. Empty (the default)
        disables selection entirely. A metric absent from an
        evaluation compares as ``-inf``.
    best_model_save_path:
        Directory that receives ``best_model.zip``, the paired
        ``best_vec_normalize.pkl`` (when the model trains under
        ``VecNormalize``), and ``best_model_meta.json`` (the selection
        step, keys, and values — so "which checkpoint is this and why"
        survives on disk) every time ``best_metric_keys`` improves.
    early_stop_patience:
        When set to N, stop training after N consecutive evaluations
        without a new best under ``best_metric_keys`` — the task-metric
        counterpart of SB3's ``StopTrainingOnNoModelImprovement``.
    early_stop_min_evals:
        Number of evaluations to complete before the stop may fire.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 25_000,
        log_prefix: str = "eval_info",
        phase_key: str | None = None,
        phase_labels: Mapping[int, str] | None = None,
        deterministic: bool = True,
        success_key: str | None = None,
        success_threshold: float = 1.0,
        info_keys: Sequence[str] | None = None,
        terminal_info_keys: Sequence[str] = (),
        episode_distribution_keys: Sequence[str] = (),
        csv_path: str | None = None,
        best_metric_keys: Sequence[str] = (),
        best_model_save_path: str | None = None,
        early_stop_patience: int | None = None,
        early_stop_min_evals: int = 0,
        verbose: int = 0,
        episode_survival_thresholds: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_prefix = log_prefix
        self.phase_key = phase_key
        self.phase_labels = dict(phase_labels or {})
        self.deterministic = deterministic
        self.success_key = success_key
        self.success_threshold = success_threshold
        self.info_keys = (
            None if info_keys is None else tuple(dict.fromkeys(info_keys))
        )
        self.terminal_info_keys = tuple(dict.fromkeys(terminal_info_keys))
        self.episode_distribution_keys = tuple(
            dict.fromkeys(episode_distribution_keys)
        )
        self.episode_survival_thresholds: dict[str, tuple[int, ...]] = {}
        for key, configured_thresholds in (
            episode_survival_thresholds or {}
        ).items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "episode survival metric keys must be non-empty strings"
                )
            thresholds = tuple(dict.fromkeys(configured_thresholds))
            if any(
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold <= 0
                for threshold in thresholds
            ):
                raise ValueError(
                    "episode survival thresholds must be positive integers"
                )
            self.episode_survival_thresholds[key] = tuple(sorted(thresholds))
        self.csv_path = csv_path
        self.best_metric_keys = tuple(dict.fromkeys(best_metric_keys))
        self.best_model_save_path = best_model_save_path
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_evals = int(early_stop_min_evals)
        self._best_score: tuple[float, ...] | None = None
        self._evals_since_best = 0
        self._eval_count = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        # If the model wraps its training env in VecNormalize, copy the
        # current obs_rms/ret_rms into our eval env so the policy sees
        # observations normalized to the same scale it just trained on.
        get_vec_norm = getattr(self.model, "get_vec_normalize_env", None)
        if get_vec_norm is not None and get_vec_norm() is not None:
            try:
                from stable_baselines3.common.vec_env import (
                    sync_envs_normalization,
                )

                sync_envs_normalization(self.training_env, self.eval_env)
            except (AttributeError, AssertionError):
                pass

        # Per-key accumulators across all eval episodes.
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        finals: dict[str, float] = {}
        maxes: dict[str, float] = {}
        phase_counts: dict[int, int] = defaultdict(int)
        # One terminal snapshot per finished episode, so we can aggregate
        # at episode granularity (e.g. mean terminal rally count, success
        # rate) rather than the step-weighted means above.
        episode_finals: list[dict[str, float]] = []
        # Per-episode reward totals (the eval env runs with
        # norm_reward=False, so these are the env's true returns). Kept
        # so reward can serve as the final tie-break in
        # ``best_metric_keys`` without a second rollout.
        episode_rewards: list[float] = []
        current_ep_reward = 0.0
        total_steps = 0
        total_episodes = 0

        obs = self.eval_env.reset()
        assert not isinstance(obs, tuple)
        last_info: dict | None = None

        while total_episodes < self.n_eval_episodes:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic
            )
            obs, rewards, dones, infos = self.eval_env.step(action)
            assert not isinstance(obs, tuple)
            total_steps += 1
            current_ep_reward += float(rewards[0])
            info = infos[0]
            last_info = info

            # Scalar keys are stable within a step; compute once and reuse
            # for both the running stats and the terminal snapshot below.
            scalar_keys = _scalar_info_keys(info)
            if self.info_keys is None:
                step_keys = scalar_keys
            else:
                scalar_key_set = set(scalar_keys)
                step_keys = [
                    key for key in self.info_keys if key in scalar_key_set
                ]
            for key in step_keys:
                value = float(info[key])
                sums[key] += value
                counts[key] += 1
                prev_max = maxes.get(key)
                if prev_max is None or value > prev_max:
                    maxes[key] = value

            if self.phase_key is not None and self.phase_key in info:
                try:
                    phase_counts[int(info[self.phase_key])] += 1
                except (TypeError, ValueError):
                    pass

            if bool(dones[0]):
                # Capture terminal scalars so monotone counters (rally,
                # hit counts) get their final value logged. On done,
                # VecEnvs reset automatically and the next ``obs`` is
                # already post-reset, so ``info`` holds the last step of
                # the just-finished episode.
                scalar_key_set = set(scalar_keys)
                terminal_requested = (
                    *self.terminal_info_keys,
                    *self.episode_distribution_keys,
                    *self.episode_survival_thresholds,
                )
                if self.success_key is not None:
                    terminal_requested = (*terminal_requested, self.success_key)
                terminal_keys = tuple(
                    dict.fromkeys(
                        (
                            *step_keys,
                            *(
                                key
                                for key in terminal_requested
                                if key in scalar_key_set
                            ),
                        )
                    )
                )
                ep_terminal = {
                    key: float(info[key]) for key in terminal_keys
                }
                finals.update({key: ep_terminal[key] for key in step_keys})
                episode_finals.append(ep_terminal)
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                total_episodes += 1

        # Fall back to the last seen step if a rollout hit video_length
        # without termination and ``finals`` is empty.
        if not finals and last_info is not None:
            scalar_keys = _scalar_info_keys(last_info)
            if self.info_keys is not None:
                scalar_key_set = set(scalar_keys)
                scalar_keys = [
                    key for key in self.info_keys if key in scalar_key_set
                ]
            for key in scalar_keys:
                finals[key] = float(last_info[key])

        metrics: dict[str, float] = {
            "episode_length": total_steps / max(1, total_episodes),
        }
        if episode_rewards:
            metrics["episode_reward_mean"] = sum(episode_rewards) / len(
                episode_rewards
            )
        for key, total in sums.items():
            if counts[key] > 0:
                metrics[f"{key}_mean"] = total / counts[key]
            metrics[f"{key}_max"] = maxes[key]
        for key, value in finals.items():
            metrics[f"{key}_final"] = value

        # Per-episode terminal means: average each key's terminal value
        # across episodes. For monotone counters this is the mean
        # end-of-episode total (e.g. avg rallies completed per episode);
        # for the env's ``term_*`` flags it is the fraction of episodes
        # that ended that way (OOB / stall / timeout breakdown).
        if episode_finals:
            ep_keys: set[str] = set()
            for snap in episode_finals:
                ep_keys.update(snap)
            for key in ep_keys:
                vals = [s[key] for s in episode_finals if key in s]
                if vals:
                    metrics[f"{key}_ep_mean"] = sum(vals) / len(vals)
            if self.success_key is not None:
                # Only score episodes whose terminal info actually carried
                # the key. If it never appears (misconfigured success_key),
                # omit success_rate entirely -- a visible gap -- rather than
                # reporting a confident 0.0 indistinguishable from real 0%.
                present = [
                    s[self.success_key]
                    for s in episode_finals
                    if self.success_key in s
                ]
                if present:
                    successes = [
                        1.0 if v >= self.success_threshold else 0.0
                        for v in present
                    ]
                    metrics["success_rate"] = sum(successes) / len(successes)

            for key in self.episode_distribution_keys:
                values = [
                    snapshot[key]
                    for snapshot in episode_finals
                    if key in snapshot
                ]
                if values:
                    metrics[f"{key}_ep_min"] = min(values)
                    metrics[f"{key}_ep_p50"] = float(
                        np.quantile(values, 0.5)
                    )
                    metrics[f"{key}_ep_p90"] = float(
                        np.quantile(values, 0.9)
                    )
                    metrics[f"{key}_ep_max"] = max(values)

            for key, thresholds in self.episode_survival_thresholds.items():
                values = [
                    snapshot[key]
                    for snapshot in episode_finals
                    if key in snapshot
                ]
                if values:
                    for threshold in thresholds:
                        metrics[f"{key}_ep_ge_{threshold}_rate"] = sum(
                            value >= threshold for value in values
                        ) / len(values)

        if self.phase_key is not None and phase_counts and total_steps > 0:
            for phase_int, count in phase_counts.items():
                label = self.phase_labels.get(phase_int, str(phase_int))
                metrics[f"phase_frac_{label}"] = count / total_steps
            # Also surface any declared labels that never appeared so
            # plots don't have surprise gaps.
            for phase_int, label in self.phase_labels.items():
                if phase_int not in phase_counts:
                    metrics.setdefault(f"phase_frac_{label}", 0.0)

        logger = self.logger
        for name, value in metrics.items():
            logger.record(f"{self.log_prefix}/{name}", value)

        if self.csv_path is not None:
            self._append_csv(metrics, self.num_timesteps)

        if self.verbose:
            print(
                f"[InfoDictEvalCallback] step={self.num_timesteps} "
                f"episodes={total_episodes} total_steps={total_steps} "
                f"finals={finals}"
            )

        return self._update_best_and_maybe_stop(metrics)

    def _update_best_and_maybe_stop(
        self, metrics: Mapping[str, float]
    ) -> bool:
        """Task-metric best-model selection + patience-based early stop.

        Scores are compared lexicographically over ``best_metric_keys``
        with strict improvement, mirroring ``EvalCallback``'s strict
        ``>`` on mean reward. Returns ``False`` (stopping training)
        once ``early_stop_patience`` consecutive evaluations pass
        without a new best, after ``early_stop_min_evals`` warm-up
        evaluations.
        """
        if not self.best_metric_keys:
            return True
        self._eval_count += 1
        score = tuple(
            float(metrics.get(key, float("-inf")))
            for key in self.best_metric_keys
        )
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._evals_since_best = 0
            self._save_best(metrics)
        else:
            self._evals_since_best += 1

        if (
            self.early_stop_patience is not None
            and self.early_stop_patience > 0
            and self._eval_count > self.early_stop_min_evals
            and self._evals_since_best >= self.early_stop_patience
        ):
            # A run-ending decision is always worth one line, verbose
            # or not (mirrors SB3's StopTrainingOnNoModelImprovement).
            print(
                f"Stopping training: no improvement in "
                f"{'/'.join(self.best_metric_keys)} for the last "
                f"{self._evals_since_best} evaluations."
            )
            return False
        return True

    def _save_best(self, metrics: Mapping[str, float]) -> None:
        """Write best_model.zip + paired normalizer stats + meta json."""
        if self.best_model_save_path is None:
            return
        os.makedirs(self.best_model_save_path, exist_ok=True)
        model_path = os.path.join(self.best_model_save_path, "best_model.zip")
        self.model.save(os.path.splitext(model_path)[0])
        artifact_paths = {"best_model.zip": model_path}
        get_vec_norm = getattr(self.model, "get_vec_normalize_env", None)
        vec_norm = get_vec_norm() if get_vec_norm is not None else None
        if vec_norm is not None:
            # Paired stats snapshot: replaying best_model against
            # end-of-training stats is a silent obs-distribution
            # mismatch (see _SaveVecNormalizeOnNewBest in train.py).
            normalizer_path = os.path.join(
                self.best_model_save_path, "best_vec_normalize.pkl"
            )
            vec_norm.save(normalizer_path)
            artifact_paths["best_vec_normalize.pkl"] = normalizer_path
        meta = {
            "timestep": int(self.num_timesteps),
            "selection_keys": list(self.best_metric_keys),
            "selection_values": {
                key: float(metrics[key])
                for key in self.best_metric_keys
                if key in metrics
            },
            # Bind the checkpoint and its observation statistics to this
            # selection event. A same-shape normalizer from another run would
            # otherwise load successfully while silently changing every input.
            "artifacts": {
                name: {"sha256": _sha256_file(path)}
                for name, path in artifact_paths.items()
            },
        }
        meta_path = os.path.join(
            self.best_model_save_path, "best_model_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    def _append_csv(self, metrics: Mapping[str, float], timestep: int) -> None:
        """Append one (timestep, metric, value) row per metric in long format."""
        path = self.csv_path
        if path is None:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestep", "metric", "value"])
            for name in sorted(metrics):
                writer.writerow([timestep, name, float(metrics[name])])
