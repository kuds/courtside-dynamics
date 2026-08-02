"""Shared plumbing for the Courtside Dynamics MuJoCo environments.

Every env in the curriculum renders the same way (100 fps, the standard
mode trio), truncates on an internal ``episode_len`` counter, applies the
same small uniform state noise on reset, and builds its observation from
interleaved per-joint ``qpos``/``qvel`` blocks. Those pieces live here so
each env file only contains what is actually unique to it: the reward
function, the termination rule, and the observation layout.

Also here, because every env (and every future env) needs them and each
had started growing its own copy: the constructor validators, the
piecewise normalized-action mapping and its inverse, and the
last-finite-observation guard that keeps a blown-up simulation from
feeding NaNs into ``VecNormalize``'s running statistics.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


def finite_nonnegative(name: str, value: float) -> float:
    """Validate a constructor scalar: finite and ``>= 0``."""
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def finite_positive(name: str, value: float) -> float:
    """Validate a constructor scalar: finite and ``> 0``."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def piecewise_targets(
    normalized: np.ndarray,
    low: np.ndarray,
    home: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map normalized ``[-1, 1]`` actions to targets around a home pose.

    Each half of the action range scales independently, so action zero
    is exactly ``home`` even when the physical limits are asymmetric
    around it -- the calibrated mapping both the wall-ball paddle and
    the humanoid PD targets use. Callers clip/validate ``normalized``
    first; this is the pure mapping.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    positive_span = np.asarray(high, dtype=np.float64) - home
    negative_span = np.asarray(home, dtype=np.float64) - low
    return np.where(
        normalized >= 0.0,
        home + normalized * positive_span,
        home + normalized * negative_span,
    )


def invert_piecewise_target(
    target: float,
    low: float,
    home: float,
    high: float,
    *,
    span_floor: float = 1e-9,
) -> float:
    """Invert :func:`piecewise_targets` for one scalar world target.

    ``span_floor`` keeps the inversion finite when a caller legally
    pivots ``home`` onto a mapping endpoint (zeroing one half-span);
    the result is clipped to the action range. Scripted controllers
    and certification probes use this instead of hand-inverting the
    env's mapping -- a hand inversion against a stale mapping is
    exactly the silent miscalibration behind the retired-map oracle
    result in review 20260727_233859.
    """
    span = (high - home) if target >= home else (home - low)
    return float(
        np.clip((target - home) / max(span, span_floor), -1.0, 1.0)
    )


class CourtsideMujocoEnv(MujocoEnv):
    """Base class for the curriculum envs.

    Subclasses implement ``step`` / ``reset_model`` / ``_get_obs`` as
    usual, call :meth:`_noisy_init_state` from ``reset_model``, and can
    use :meth:`_joints_obs` to assemble the per-joint part of the
    observation vector.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        model_path: str,
        episode_len: int,
        obs_dim: int,
        frame_skip: int = 5,
        **kwargs: Any,
    ) -> None:
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=observation_space,
            **kwargs,
        )
        self.episode_len = int(episode_len)
        self.step_number = 0
        # Most recent fully finite observation, echoed back if the
        # simulator ever produces a nonfinite state: a single NaN
        # observation ingested by VecNormalize's running statistics
        # silently corrupts every subsequent normalized step.
        self._last_finite_observation: np.ndarray | None = None

    def _physics_state_is_finite(self) -> bool:
        """Whether the core MuJoCo state arrays are fully finite.

        Checked BEFORE stepping physics on the failure path: MuJoCo
        reacts to NaN qpos/ctrl by warning and resetting its state, so
        a post-step check would see a silently teleported-but-finite
        episode instead of the blow-up. Envs with extra bodies to watch
        (specific geoms' xpos/xmat) override with a superset.
        """
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.ctrl,
            self.data.act,
            self.data.sensordata,
        )
        return all(bool(np.isfinite(values).all()) for values in arrays)

    def _remember_finite_observation(self, observation: np.ndarray) -> None:
        """Record ``observation`` as the fallback for the NaN echo.

        Call from ``reset_model`` after building the (finite) reset
        observation so the first step of an episode always has a
        fallback.
        """
        self._last_finite_observation = np.asarray(observation).copy()

    def _record_or_echo_observation(
        self, observation: np.ndarray, nonfinite: bool
    ) -> np.ndarray:
        """The step-time half of the last-finite-observation guard.

        A healthy observation becomes the new fallback and passes
        through unchanged. When the caller has detected a nonfinite
        state (by whatever definition the env uses -- raw observation,
        qpos/qvel, contact forces), the last finite observation is
        echoed instead; before any finite observation exists (a reset
        that immediately blew up) zeros are echoed rather than the
        poisoned values, and the fallback is never overwritten with
        nonfinite data.
        """
        if nonfinite:
            if self._last_finite_observation is None:
                return np.zeros_like(np.asarray(observation))
            return self._last_finite_observation.copy()
        self._last_finite_observation = np.asarray(observation).copy()
        return observation

    def _noisy_init_state(
        self, noise: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos, qvel)``: the model keyframe plus uniform noise."""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-noise, high=noise
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-noise, high=noise
        )
        return qpos, qvel

    def _joints_obs(self, *names: str) -> np.ndarray:
        """Concatenate ``qpos``/``qvel`` pairs for each named joint, in order."""
        parts: list[np.ndarray] = []
        for name in names:
            joint = self.data.joint(name)
            parts.append(np.asarray(joint.qpos))
            parts.append(np.asarray(joint.qvel))
        return np.concatenate(parts)
