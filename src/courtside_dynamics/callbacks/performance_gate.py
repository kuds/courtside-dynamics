"""Performance-gated curriculum stages for environment attributes.

Where :class:`LinearEnvAttrScheduleCallback` moves one attribute on a
timestep clock, this callback advances through a discrete ladder of
*stages* -- each a mapping of environment attributes to values -- and only
steps forward when an evaluation metric has cleared a threshold for a
sustained number of consecutive evaluations. Run 20260714_211111's
timestep schedule annealed difficulty regardless of whether the policy had
learned anything; a gated ladder advances exactly when the current stage
is mastered, and the sustain requirement keeps a single lucky evaluation
batch from promoting a policy that then faces the next stage from
weakness.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback


class PerformanceGatedEnvStagesCallback(BaseCallback):
    """Advance env attributes through stages as eval performance allows.

    Parameters
    ----------
    stages:
        Ordered ladder of stage definitions. Each stage is a mapping of
        attribute name to value, applied with
        ``env_method("set_wrapper_attr", ..., force=False)`` to every
        target environment, so a typo'd attribute fails loudly instead of
        shadow-writing a wrapper. Stage 0 is applied at training start.
    metric_key:
        Key into the driving evaluator's aggregated metrics (e.g.
        ``"bounce_count_ep_mean"``).
    threshold:
        The stage is considered mastered while
        ``metrics[metric_key] >= threshold``.
    sustain_evals:
        Consecutive evaluations that must clear the threshold before
        advancing. Also absorbs the transient dip expected right after a
        stage change (off-policy learners replay some transitions from
        the previous stage's fence/dynamics).
    info_eval:
        The :class:`InfoDictEvalCallback` whose evaluations drive the
        gate. Its eval environment receives every stage application too,
        keeping the selection/guard metrics *matched* to the training
        stage. Order this callback AFTER it in the callback list so a
        trigger sees the same trigger's fresh metrics.
    extra_target_envs:
        Additional vectorized environments to keep in sync (the training
        env from ``self.training_env`` and ``info_eval.eval_env`` are
        always synced; final-config evaluators must NOT be listed here).
    """

    def __init__(
        self,
        stages: Sequence[Mapping[str, Any]],
        *,
        metric_key: str,
        threshold: float,
        sustain_evals: int,
        info_eval: InfoDictEvalCallback,
        extra_target_envs: Sequence[VecEnv] = (),
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        resolved_stages = tuple(dict(stage) for stage in stages)
        if not resolved_stages:
            raise ValueError("stages must be a non-empty sequence")
        for stage in resolved_stages:
            if not stage:
                raise ValueError("each stage must set at least one attribute")
            for name in stage:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "stage attribute names must be non-empty strings"
                    )
        if not isinstance(metric_key, str) or not metric_key.strip():
            raise ValueError("metric_key must be a non-empty string")
        if not isinstance(threshold, numbers.Real) or not np.isfinite(
            threshold
        ):
            raise ValueError("threshold must be a finite number")
        if (
            isinstance(sustain_evals, bool)
            or not isinstance(sustain_evals, int)
            or sustain_evals < 1
        ):
            raise ValueError("sustain_evals must be a positive integer")
        self.stages: tuple[dict[str, Any], ...] = resolved_stages
        self.metric_key: str = metric_key
        self.threshold: float = float(threshold)
        self.sustain_evals: int = sustain_evals
        self.info_eval: InfoDictEvalCallback = info_eval
        self.extra_target_envs: tuple[VecEnv, ...] = tuple(extra_target_envs)
        self._stage_index: int = 0
        self._streak: int = 0
        self._seen_evals: int = 0

    @property
    def stage_index(self) -> int:
        """Index of the currently applied stage."""
        return self._stage_index

    @property
    def stages_metadata(self) -> list[dict[str, Any]]:
        """JSON-serializable copy of the stage ladder."""
        return [dict(stage) for stage in self.stages]

    def _target_envs(self) -> tuple[VecEnv, ...]:
        return (
            self.training_env,
            self.info_eval.eval_env,
            *self.extra_target_envs,
        )

    def _apply_stage(self) -> None:
        stage = self.stages[self._stage_index]
        for env in self._target_envs():
            for attr_name, value in stage.items():
                applied = env.env_method(
                    "set_wrapper_attr", attr_name, value, force=False
                )
                if not all(applied):
                    raise AttributeError(
                        f"no environment exposes attribute '{attr_name}' "
                        f"(stage {self._stage_index}, applied per worker: "
                        f"{applied}); the curriculum would silently stall"
                    )
        self._record()

    def _record(self) -> None:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return
        logger.record("curriculum/stage_index", self._stage_index)
        logger.record("curriculum/stage_streak", self._streak)
        for attr_name, value in self.stages[self._stage_index].items():
            if isinstance(value, numbers.Real):
                logger.record(f"curriculum/{attr_name}", float(value))

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _on_step(self) -> bool:
        if self.info_eval.completed_evals == self._seen_evals:
            return True
        self._seen_evals = self.info_eval.completed_evals
        metrics = self.info_eval.last_metrics or {}
        value = metrics.get(self.metric_key)
        if value is not None and float(value) >= self.threshold:
            self._streak += 1
        else:
            # A miss -- or an evaluation that never produced the metric --
            # restarts the sustain window: advancement must be earned by
            # consecutive competent evaluations, not accumulated luck.
            self._streak = 0
        if (
            self._streak >= self.sustain_evals
            and self._stage_index < len(self.stages) - 1
        ):
            self._stage_index += 1
            self._streak = 0
            self._apply_stage()
            # Scores from different stage distributions are not
            # comparable: without this reset, a best banked on an easier
            # stage permanently bars genuinely better later policies
            # (measured ~0.6-0.7 bounce_count_ep_mean inflation between
            # the narrowest and the full serve, ~40x the selection
            # min_delta) and stale patience/flatness windows leak across
            # the boundary.
            self.info_eval.reset_selection_state()
            print(
                f"[PerformanceGatedEnvStagesCallback] advancing to stage "
                f"{self._stage_index}/{len(self.stages) - 1} at "
                f"{self.num_timesteps} steps: "
                f"{self.metric_key} held >= {self.threshold} for "
                f"{self.sustain_evals} evaluations."
            )
        else:
            self._record()
        return True
