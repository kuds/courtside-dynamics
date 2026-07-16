"""Training-step schedules for numeric environment attributes.

The callback is intentionally agnostic to the environment and attribute being
scheduled.  A training recipe can therefore use it for curriculum probabilities,
difficulty scales, or any other numeric value exposed by every vectorized
environment instance.
"""

from __future__ import annotations

import math
import numbers

from stable_baselines3.common.callbacks import BaseCallback


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be nonnegative")
    return resolved


class LinearEnvAttrScheduleCallback(BaseCallback):
    """Linearly schedule one numeric attribute on all training envs.

    The attribute stays at ``start_value`` through
    ``hold_until_timesteps``.  It then interpolates linearly and reaches
    ``end_value`` at ``end_timesteps``; later calls remain clamped to the end
    value.  Timing uses :attr:`BaseCallback.num_timesteps`, which is SB3's
    global environment-transition count and therefore already accounts for
    the number of parallel environments.

    Values are applied to every worker with
    ``env_method("set_wrapper_attr", ...)``, which walks each worker's
    wrapper stack to the environment that owns ``attr_name``.  Consequently,
    each wrapped environment must expose a writable attribute named
    ``attr_name``.  The applied value is also recorded to the SB3 logger as
    ``schedule/<attr_name>`` so the realized curriculum is auditable from
    TensorBoard / ``progress.csv``.
    """

    def __init__(
        self,
        attr_name: str,
        *,
        start_value: float,
        end_value: float,
        hold_until_timesteps: int,
        end_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if not isinstance(attr_name, str) or not attr_name.strip():
            raise ValueError("attr_name must be a non-empty string")

        self.attr_name = attr_name
        self.start_value = _finite_number("start_value", start_value)
        self.end_value = _finite_number("end_value", end_value)
        self.hold_until_timesteps = _nonnegative_int(
            "hold_until_timesteps", hold_until_timesteps
        )
        self.end_timesteps = _nonnegative_int("end_timesteps", end_timesteps)
        if self.end_timesteps <= self.hold_until_timesteps:
            raise ValueError("end_timesteps must be greater than hold_until_timesteps")

        self._last_value: float | None = None

    @property
    def schedule_metadata(self) -> dict[str, str | float | int]:
        """Return the JSON-serializable schedule configuration."""
        return {
            "schedule": "linear_after_hold",
            "attr_name": self.attr_name,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "hold_until_timesteps": self.hold_until_timesteps,
            "end_timesteps": self.end_timesteps,
        }

    @property
    def last_value(self) -> float | None:
        """Most recent value sent to the vectorized training environment."""
        return self._last_value

    def value_at(self, timesteps: int) -> float:
        """Return the clamped schedule value at global ``timesteps``."""
        if timesteps <= self.hold_until_timesteps:
            return self.start_value
        if timesteps >= self.end_timesteps:
            return self.end_value

        progress = (timesteps - self.hold_until_timesteps) / (
            self.end_timesteps - self.hold_until_timesteps
        )
        return self.start_value + progress * (self.end_value - self.start_value)

    def _apply_current_value(self) -> None:
        value = self.value_at(self.num_timesteps)
        # ``VecEnv.set_attr`` does a plain ``setattr`` on each worker's
        # outermost wrapper (usually SB3's ``Monitor``), and gymnasium
        # wrappers do not forward attribute writes -- the value would land
        # as a shadow attribute on the wrapper and never reach the
        # scheduled environment (run 20260714_211111 trained its entire
        # budget at the schedule's start value this way).
        # ``set_wrapper_attr`` walks the wrapper stack to the attribute's
        # owner; ``force=False`` makes it report failure instead of
        # recreating the silent shadow write when no layer owns the
        # attribute (e.g. a typo'd ``attr_name``).
        applied = self.training_env.env_method(
            "set_wrapper_attr", self.attr_name, value, force=False
        )
        if not all(applied):
            raise AttributeError(
                f"no environment in the training VecEnv exposes attribute "
                f"'{self.attr_name}' (applied per worker: {applied}); the "
                f"schedule would silently do nothing"
            )
        self._last_value = value
        logger = getattr(self.model, "logger", None)
        if logger is not None:
            logger.record(f"schedule/{self.attr_name}", value)

    def _on_training_start(self) -> None:
        # A fresh SB3 run starts at timestep zero, so this installs the
        # configured start value before the first action.  Using the current
        # global timestep also makes resumed ``learn(reset_num_timesteps=False)``
        # calls continue the schedule instead of jumping backwards.
        self._apply_current_value()

    def _on_step(self) -> bool:
        self._apply_current_value()
        return True
