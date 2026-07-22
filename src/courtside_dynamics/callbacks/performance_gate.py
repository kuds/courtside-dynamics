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

import json
import numbers
import os
import shutil
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

if TYPE_CHECKING:
    from stable_baselines3.common.off_policy_algorithm import (
        OffPolicyAlgorithm,
    )

_PROMOTION_RULES = ("consecutive", "window_mean")

#: Name under which the current stage index is attached to the driving
#: evaluator's metrics, so every eval_info.csv row batch, TensorBoard
#: eval scalar set, and best_model_meta.json records which geometry it
#: was measured at (run 20260721_004722's stage timeline had to be
#: reconstructed by joining serve-speed logger columns by timestep).
STAGE_CONTEXT_METRIC = "curriculum_stage_index"


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
        Number of evaluations the promotion decision is based on. Under
        the ``"consecutive"`` rule, that many consecutive evaluations
        must each clear the threshold. Under ``"window_mean"``, the mean
        of the last ``sustain_evals`` evaluations must clear it. Either
        way the window restarts on a stage advance, so it also absorbs
        the transient dip expected right after a stage change
        (off-policy learners replay some transitions from the previous
        stage's fence/dynamics).
    promotion_rule:
        ``"consecutive"`` (default, the historical rule) requires every
        evaluation in the sustain window to clear the threshold
        independently; a single miss restarts the window.
        ``"window_mean"`` promotes when the *mean* of the window clears
        the threshold: with 30-episode batches the per-eval standard
        error (~0.4 bounces near the WallBall gate bar) makes
        "twice consecutively >= 3.0" a coin flip for a policy whose true
        mean sits at the bar -- run 20260721_004722 cleared 3.0 in four
        separate stage-2 evaluations without ever being promoted. The
        mean of the window estimates the same quantity with the
        variance of a batch ``sustain_evals`` times larger and does not
        lower the bar. An evaluation that lacks the metric clears the
        window under both rules: advancement must be certified by
        evidence, not by absent data.
    info_eval:
        The :class:`InfoDictEvalCallback` whose evaluations drive the
        gate. Its eval environment receives every stage application too,
        keeping the selection/guard metrics *matched* to the training
        stage, and its metrics gain a ``curriculum_stage_index`` context
        entry so CSV/TensorBoard/best-model artifacts record the stage
        they were measured at. Order this callback AFTER it in the
        callback list so a trigger sees the same trigger's fresh
        metrics.
    advance_update_pause_steps:
        When ``> 0``, every stage advance pauses gradient updates for
        this many timesteps by setting the model's ``gradient_steps`` to
        0 (restored automatically afterwards), so the learner collects
        on-policy data at the new geometry before resuming updates.
        Requires an off-policy model exposing ``gradient_steps``
        (checked loudly at training start). This is half of the
        promotion warm-up package motivated by run 20260721_004722's
        promotion shocks (matched eval 3.37 -> 1.43 and 3.10 -> 1.70,
        each costing ~0.5-1M steps of recovery).
    clear_replay_buffer_on_advance:
        When ``True``, every stage advance drops the replay buffer. On
        promotion day the buffer is otherwise ~100% previous-stage
        data, and a stage's ``paddle_x_fence`` participates in dynamics
        (target clamping), so replayed old-stage transitions train the
        critic on physics that no longer hold. Requires
        ``advance_update_pause_steps > 0`` -- clearing without a pause
        would have SAC immediately optimizing against a near-empty
        buffer -- and a model exposing ``replay_buffer`` (both checked
        loudly).
    extra_target_envs:
        Additional vectorized environments to keep in sync (the training
        env from ``self.training_env`` and ``info_eval.eval_env`` are
        always synced; final-config evaluators must NOT be listed here).
    stage_bests_dir:
        When set, every stage's best artifacts are archived to
        ``<stage_bests_dir>/stage_NN/`` -- on each advance for the
        departing stage (immediately before ``reset_selection_state``
        lets the next stage's first evaluation overwrite them) and at
        training end for the final stage. Without this, per-stage
        champions are destroyed by design: runs 20260721_142121 and
        20260722_002913 each needed a stage champion afterwards, and
        only luck (the stage-entry eval becoming the next stage's first
        best) preserved one of them.
    stage_history_path:
        When set, a per-stage history report is written here -- refreshed
        atomically on every stage close (so completed stages survive a
        hard runtime death) and finalized at training end or interrupt:
        entry/exit timesteps, evaluation counts, promotion windows, and
        each stage's archived best selection values -- the table every
        campaign review previously reconstructed by hand from
        ``eval_info.csv``.
    run_config_path:
        Optional path to the run's ``config.json``; when set, it is
        copied into each archived stage directory, which makes
        ``stage_bests/stage_NN/`` a valid legacy-flat-layout
        ``WarmStartConfig.source_run_dir`` -- continuations really can
        seed from any stage's champion without hand-assembling files.
    """

    def __init__(
        self,
        stages: Sequence[Mapping[str, Any]],
        *,
        metric_key: str,
        threshold: float,
        sustain_evals: int,
        info_eval: InfoDictEvalCallback,
        promotion_rule: str = "consecutive",
        advance_update_pause_steps: int = 0,
        clear_replay_buffer_on_advance: bool = False,
        extra_target_envs: Sequence[VecEnv] = (),
        stage_bests_dir: str | None = None,
        stage_history_path: str | None = None,
        run_config_path: str | None = None,
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
        if promotion_rule not in _PROMOTION_RULES:
            raise ValueError(
                f"promotion_rule must be one of {_PROMOTION_RULES}, "
                f"got {promotion_rule!r}"
            )
        if (
            isinstance(advance_update_pause_steps, bool)
            or not isinstance(advance_update_pause_steps, int)
            or advance_update_pause_steps < 0
        ):
            raise ValueError(
                "advance_update_pause_steps must be a nonnegative integer"
            )
        if not isinstance(clear_replay_buffer_on_advance, bool):
            raise ValueError(
                "clear_replay_buffer_on_advance must be a boolean"
            )
        if clear_replay_buffer_on_advance and advance_update_pause_steps <= 0:
            raise ValueError(
                "clear_replay_buffer_on_advance requires "
                "advance_update_pause_steps > 0: without a pause the "
                "learner's next update would sample a near-empty buffer"
            )
        for name, value in (
            ("stage_bests_dir", stage_bests_dir),
            ("stage_history_path", stage_history_path),
            ("run_config_path", run_config_path),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be None or a non-empty string")
        self.stages: tuple[dict[str, Any], ...] = resolved_stages
        self.metric_key: str = metric_key
        self.threshold: float = float(threshold)
        self.sustain_evals: int = sustain_evals
        self.promotion_rule: str = promotion_rule
        self.advance_update_pause_steps: int = advance_update_pause_steps
        self.clear_replay_buffer_on_advance: bool = (
            clear_replay_buffer_on_advance
        )
        self.info_eval: InfoDictEvalCallback = info_eval
        self.extra_target_envs: tuple[VecEnv, ...] = tuple(extra_target_envs)
        self._stage_index: int = 0
        self._streak: int = 0
        self._seen_evals: int = 0
        # Rolling metric window for the "window_mean" rule; cleared on
        # every advance and on any evaluation missing the metric.
        self._metric_window: deque[float] = deque(maxlen=sustain_evals)
        # End of the current post-advance update pause (None = not
        # paused) and the gradient_steps value to restore when it ends.
        self._update_pause_until: int | None = None
        self._resume_gradient_steps: int | None = None
        self.stage_bests_dir: str | None = stage_bests_dir
        self.stage_history_path: str | None = stage_history_path
        self.run_config_path: str | None = run_config_path
        self._finalized: bool = False
        self._stage_entry_timestep: int = 0
        self._stage_eval_count: int = 0
        self._stage_history: list[dict[str, Any]] = []

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
        # Stamp the stage onto the driving evaluator's metrics so every
        # downstream artifact records the geometry it measured.
        self.info_eval.set_context_metric(
            STAGE_CONTEXT_METRIC, float(self._stage_index)
        )
        self._record()

    def _record(self) -> None:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return
        logger.record("curriculum/stage_index", self._stage_index)
        logger.record("curriculum/stage_streak", self._streak)
        if self.promotion_rule == "window_mean" and self._metric_window:
            logger.record(
                "curriculum/gate_window_mean",
                sum(self._metric_window) / len(self._metric_window),
            )
        if self._update_pause_until is not None:
            logger.record(
                "curriculum/update_pause_until",
                float(self._update_pause_until),
            )
        for attr_name, value in self.stages[self._stage_index].items():
            if isinstance(value, numbers.Real):
                logger.record(f"curriculum/{attr_name}", float(value))

    def _on_training_start(self) -> None:
        # Fail before any training if the warm-up package is configured
        # against a model that cannot honor it -- a silently ignored
        # pause/clear is exactly the no-op failure mode this repo bans.
        if self.advance_update_pause_steps > 0:
            gradient_steps = getattr(self.model, "gradient_steps", None)
            if gradient_steps is None or isinstance(gradient_steps, bool):
                raise TypeError(
                    "advance_update_pause_steps requires an off-policy "
                    "model exposing an integer 'gradient_steps' attribute"
                )
            self._resume_gradient_steps = int(gradient_steps)
        if self.clear_replay_buffer_on_advance:
            buffer = getattr(self.model, "replay_buffer", None)
            if buffer is None or not callable(getattr(buffer, "reset", None)):
                raise TypeError(
                    "clear_replay_buffer_on_advance requires a model with "
                    "a resettable 'replay_buffer'"
                )
        self._stage_entry_timestep = int(self.num_timesteps)
        self._apply_stage()

    def _close_stage_record(self, *, promoted: bool) -> None:
        """Archive the departing/final stage's best and append its history row.

        Must run BEFORE ``reset_selection_state`` on an advance: the
        reset lets the next stage's first evaluation overwrite
        ``best_model.zip``, and the departing stage's champion would be
        gone. The archived ``best_model_meta.json`` payload (selection
        values, timestep, sha256s, stage context) rides on the history
        row so ``curriculum_stages.json`` is self-contained.
        """
        best_meta = None
        # Zero stage evaluations means any on-disk best triple predates
        # this stage (the reset leaves the previous champion on disk
        # until the next evaluation overwrites it): archiving it here
        # would file the PREVIOUS stage's champion under this stage's
        # label. One-or-more evaluations guarantees the triple was
        # written in-stage -- the first post-reset eval always saves.
        if self.stage_bests_dir is not None and self._stage_eval_count > 0:
            destination = os.path.join(
                self.stage_bests_dir, f"stage_{self._stage_index:02d}"
            )
            best_meta = self.info_eval.archive_best(destination)
            if best_meta is not None and self.run_config_path is not None:
                # With config.json alongside the triple, the stage dir
                # is a valid legacy-flat-layout warm-start source.
                if os.path.isfile(self.run_config_path):
                    shutil.copy2(
                        self.run_config_path,
                        os.path.join(destination, "config.json"),
                    )
        self._stage_history.append(
            {
                "stage_index": self._stage_index,
                "stage": dict(self.stages[self._stage_index]),
                "entry_timestep": self._stage_entry_timestep,
                "exit_timestep": int(self.num_timesteps),
                "evals": self._stage_eval_count,
                "promoted": promoted,
                "promotion_window": [float(v) for v in self._metric_window],
                "streak": self._streak,
                "best": best_meta,
            }
        )
        # Refresh the report on every stage close: completed-stage rows
        # then survive even a hard runtime death that skips both
        # _on_training_end and train()'s interrupt salvage.
        self._write_stage_history()

    def _write_stage_history(self) -> None:
        if self.stage_history_path is None:
            return
        payload = {
            "metric_key": self.metric_key,
            "threshold": self.threshold,
            "sustain_evals": self.sustain_evals,
            "promotion_rule": self.promotion_rule,
            "advance_update_pause_steps": self.advance_update_pause_steps,
            "clear_replay_buffer_on_advance": (
                self.clear_replay_buffer_on_advance
            ),
            "stage_count": len(self.stages),
            "final_stage_index": self._stage_index,
            "stages": self._stage_history,
        }
        parent = os.path.dirname(self.stage_history_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = self.stage_history_path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, self.stage_history_path)

    def finalize(self) -> None:
        """Close out the current stage and write the history -- once.

        Called from ``_on_training_end`` on a normal ``learn()`` exit
        and from train()'s KeyboardInterrupt salvage path (SB3 does not
        run ``on_training_end`` when an exception propagates out of the
        training loop, and the whole point of the salvage path is that
        the run dir stays self-describing). Idempotent so the interrupt
        path plus a later normal close cannot append a duplicate row.

        The final stage's best is also the run-level canonical best;
        duplicating it under ``stage_bests/`` keeps the per-stage index
        complete (skipped when the final stage recorded no evaluation:
        the on-disk best would belong to the previous stage), so a
        continuation can seed from ANY archived stage's champion
        without special-casing the last one.
        """
        if self._finalized:
            return
        self._finalized = True
        self._close_stage_record(promoted=False)
        self._write_stage_history()

    def _on_training_end(self) -> None:
        self.finalize()

    def _maybe_resume_updates(self) -> None:
        """Restore ``gradient_steps`` once the post-advance pause ends."""
        if (
            self._update_pause_until is not None
            and self.num_timesteps >= self._update_pause_until
        ):
            assert self._resume_gradient_steps is not None
            # Duck-typed off-policy access, validated at training start.
            cast("OffPolicyAlgorithm", self.model).gradient_steps = (
                self._resume_gradient_steps
            )
            self._update_pause_until = None
            if self.verbose:
                print(
                    f"[PerformanceGatedEnvStagesCallback] resuming gradient "
                    f"updates at {self.num_timesteps} steps "
                    f"(gradient_steps={self._resume_gradient_steps})."
                )

    def _promotion_earned(self, metrics: Mapping[str, Any]) -> bool:
        """Update the sustain state with a fresh eval; True when mastered."""
        value = metrics.get(self.metric_key)
        if value is None:
            # An evaluation that never produced the metric cannot certify
            # mastery under either rule: restart the evidence window.
            self._streak = 0
            self._metric_window.clear()
            return False
        resolved = float(value)
        # The streak is maintained under both rules -- it drives the
        # "consecutive" decision and stays a logged diagnostic under
        # "window_mean". The metric window is likewise kept under both:
        # under "consecutive" it never drives promotion, but at a stage
        # close it holds the last ``sustain_evals`` measurements (all
        # threshold-clearing at a promotion, since the streak spans
        # them), so curriculum_stages.json records promotion evidence
        # for consecutive-rule gates too.
        self._streak = self._streak + 1 if resolved >= self.threshold else 0
        self._metric_window.append(resolved)
        if self.promotion_rule == "consecutive":
            return self._streak >= self.sustain_evals
        if len(self._metric_window) < self.sustain_evals:
            return False
        window_mean = sum(self._metric_window) / len(self._metric_window)
        return window_mean >= self.threshold

    def _on_step(self) -> bool:
        self._maybe_resume_updates()
        if self.info_eval.completed_evals == self._seen_evals:
            return True
        self._seen_evals = self.info_eval.completed_evals
        self._stage_eval_count += 1
        metrics = self.info_eval.last_metrics or {}
        if (
            self._promotion_earned(metrics)
            and self._stage_index < len(self.stages) - 1
        ):
            # Archive the departing stage's champion and close its
            # history row while the window/streak still describe the
            # promotion evidence and best_model.zip still holds it.
            self._close_stage_record(promoted=True)
            self._stage_index += 1
            self._stage_entry_timestep = int(self.num_timesteps)
            self._stage_eval_count = 0
            self._streak = 0
            self._metric_window.clear()
            self._apply_stage()
            # Scores from different stage distributions are not
            # comparable: without this reset, a best banked on an easier
            # stage permanently bars genuinely better later policies
            # (measured ~0.6-0.7 bounce_count_ep_mean inflation between
            # the narrowest and the full serve, ~40x the selection
            # min_delta) and stale patience/flatness windows leak across
            # the boundary.
            self.info_eval.reset_selection_state()
            # Promotion warm-up: drop stale-stage replay data and hold
            # gradient updates until fresh-frontier transitions exist.
            # Clearing is validated to imply a pause at construction;
            # both attributes were validated at training start.
            off_policy_model = cast("OffPolicyAlgorithm", self.model)
            if self.clear_replay_buffer_on_advance:
                assert off_policy_model.replay_buffer is not None
                off_policy_model.replay_buffer.reset()
            if self.advance_update_pause_steps > 0:
                off_policy_model.gradient_steps = 0
                self._update_pause_until = (
                    self.num_timesteps + self.advance_update_pause_steps
                )
            rule = (
                f"held >= {self.threshold} for {self.sustain_evals} "
                f"consecutive evaluations"
                if self.promotion_rule == "consecutive"
                else f"averaged >= {self.threshold} over the last "
                f"{self.sustain_evals} evaluations"
            )
            warmup = (
                f" Warm-up: replay buffer "
                f"{'cleared' if self.clear_replay_buffer_on_advance else 'kept'}, "
                f"updates paused until {self._update_pause_until} steps."
                if self.advance_update_pause_steps > 0
                else ""
            )
            print(
                f"[PerformanceGatedEnvStagesCallback] advancing to stage "
                f"{self._stage_index}/{len(self.stages) - 1} at "
                f"{self.num_timesteps} steps: {self.metric_key} {rule}."
                f"{warmup}"
            )
            self._record()
        else:
            self._record()
        return True
