"""Deterministic held-out evaluation for the humanoid-tennis curriculum.

This module deliberately does not alter an environment, training run, or
curriculum level.  It evaluates a policy on an explicit, mirrored suite of
seeded launch conditions and returns immutable evidence that a caller can use
to make a *manual* promotion decision.

Routine SB3 evaluation remains useful for learning curves, but its reset RNG
continues between evaluations and its best checkpoint is selected by mean
reward.  Neither property is suitable for a curriculum gate.  The helpers
below reset every episode with its exact seed and serving side, score the
terminal ``stage_success`` flag, and aggregate return conversion as a ratio of
sums rather than a mean of per-episode ratios.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from courtside_dynamics.envs.tennis_curriculum import (
    TENNIS_CURRICULUM_PRESETS,
    CurriculumStage,
)
from courtside_dynamics.envs.tennis_rules import CourtSide


@dataclass(frozen=True, slots=True)
class HeldOutCondition:
    """One exact reset condition in a held-out launch suite."""

    seed: int
    serving_side: CourtSide
    local_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_velocity_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("held-out seeds must be non-negative integers")
        if not isinstance(self.seed, int):
            raise TypeError("held-out seeds must be integers")
        if not isinstance(self.serving_side, CourtSide):
            raise TypeError("serving_side must be a CourtSide")
        for name in ("local_position_offset", "local_velocity_offset"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "seed": self.seed,
            "serving_side": self.serving_side.label,
            "local_position_offset": list(self.local_position_offset),
            "local_velocity_offset": list(self.local_velocity_offset),
        }


def _van_der_corput(index: int, base: int) -> float:
    """Return one deterministic low-discrepancy value in ``[0, 1)``."""
    value = 0.0
    denominator = 1.0
    remaining = index + 1
    while remaining:
        remaining, remainder = divmod(remaining, base)
        denominator *= base
        value += remainder / denominator
    return value


def _scaled_offset(
    index: int,
    maxima: tuple[float, float, float],
    bases: tuple[int, int, int],
) -> tuple[float, float, float]:
    return (
        maxima[0] * (2.0 * _van_der_corput(index, bases[0]) - 1.0),
        maxima[1] * (2.0 * _van_der_corput(index, bases[1]) - 1.0),
        maxima[2] * (2.0 * _van_der_corput(index, bases[2]) - 1.0),
    )


@dataclass(frozen=True, slots=True)
class HeldOutSeedSuite:
    """Generate disjoint, paired launch conditions for promotion evaluation.

    With the defaults, 50 distinct launch seeds are each evaluated from both
    court orientations, producing 100 episodes.  Mirroring the same seed
    isolates side-specific policy failures from launch-distribution noise.
    When ``mirror_each`` is false, one orientation is used per seed and sides
    alternate to keep the suite balanced to within one episode.
    """

    base_seed: int = 50_000
    launch_count: int = 50
    mirror_each: bool = True
    max_position_offset: tuple[float, float, float] = (0.03, 0.08, 0.03)
    max_velocity_offset: tuple[float, float, float] = (0.20, 0.10, 0.20)

    def __post_init__(self) -> None:
        for name, value in (
            ("base_seed", self.base_seed),
            ("launch_count", self.launch_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if self.launch_count < 1:
            raise ValueError("launch_count must be at least one")
        if not isinstance(self.mirror_each, bool):
            raise TypeError("mirror_each must be a bool")
        for name in ("max_position_offset", "max_velocity_offset"):
            values = tuple(getattr(self, name))
            if (
                len(values) != 3
                or not all(math.isfinite(value) and value >= 0.0 for value in values)
            ):
                raise ValueError(f"{name} must be a non-negative finite three-vector")
            object.__setattr__(self, name, values)

    @property
    def conditions(self) -> tuple[HeldOutCondition, ...]:
        """Return conditions in stable seed-major, A-then-B order."""
        conditions: list[HeldOutCondition] = []
        for offset in range(self.launch_count):
            seed = self.base_seed + offset
            position_offset = _scaled_offset(
                offset,
                self.max_position_offset,
                (2, 3, 5),
            )
            velocity_offset = _scaled_offset(
                offset,
                self.max_velocity_offset,
                (7, 11, 13),
            )
            if self.mirror_each:
                conditions.extend(
                    (
                        HeldOutCondition(
                            seed,
                            CourtSide.A,
                            position_offset,
                            velocity_offset,
                        ),
                        HeldOutCondition(
                            seed,
                            CourtSide.B,
                            position_offset,
                            velocity_offset,
                        ),
                    )
                )
            else:
                side = CourtSide.A if offset % 2 == 0 else CourtSide.B
                conditions.append(
                    HeldOutCondition(
                        seed,
                        side,
                        position_offset,
                        velocity_offset,
                    )
                )
        return tuple(conditions)

    @property
    def episode_count(self) -> int:
        """Number of episodes in the generated suite."""
        return self.launch_count * (2 if self.mirror_each else 1)


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """Thresholds for an advisory curriculum-promotion decision.

    ``prior_replay_probability`` records the recommended rehearsal share for
    a future curriculum-aware observation/API. It does not enable hidden
    reset-time stage mixing in v0; prior stages are protected by explicit
    held-out retention gates here.
    """

    success_threshold: float = 0.80
    prior_retention_threshold: float = 0.75
    min_episodes: int = 100
    min_unique_initial_states: int = 100
    prior_replay_probability: float = 0.20
    require_zero_unsafe: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("success_threshold", self.success_threshold),
            ("prior_retention_threshold", self.prior_retention_threshold),
            ("prior_replay_probability", self.prior_replay_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        for name, value in (
            ("min_episodes", self.min_episodes),
            ("min_unique_initial_states", self.min_unique_initial_states),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least one")
        if not isinstance(self.require_zero_unsafe, bool):
            raise TypeError("require_zero_unsafe must be a bool")


@dataclass(frozen=True, slots=True)
class RateSummary:
    """Immutable count and episode rate for one diagnostic."""

    name: str
    count: int
    rate: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rate-summary names must not be empty")
        if self.count < 0:
            raise ValueError("rate-summary counts must be non-negative")
        if not math.isfinite(self.rate) or not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate-summary rates must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    """Immutable binding between evidence, task, policy, and normalization."""

    curriculum_stage: int
    curriculum_stage_name: str
    curriculum_fingerprint: str
    evaluation_suite_fingerprint: str
    canonical_evaluation_spec_id: str | None
    environment_class: str
    policy_id: str
    policy_state_sha256: str | None
    policy_artifact_sha256: str | None
    normalization_id: str
    normalization_artifact_sha256: str | None
    observation_transform_name: str | None
    observation_transform_sha256: str | None
    package_version: str
    package_source_sha256: str
    git_sha: str | None
    runtime_versions: tuple[tuple[str, str], ...]
    runtime_fingerprint: str

    def __post_init__(self) -> None:
        if self.curriculum_stage < 0:
            raise ValueError("curriculum_stage must be non-negative")
        for name in (
            "curriculum_stage_name",
            "curriculum_fingerprint",
            "evaluation_suite_fingerprint",
            "environment_class",
            "policy_id",
            "normalization_id",
            "package_version",
            "package_source_sha256",
            "runtime_fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "curriculum_fingerprint",
            "evaluation_suite_fingerprint",
            "policy_state_sha256",
            "policy_artifact_sha256",
            "normalization_artifact_sha256",
            "observation_transform_sha256",
            "package_source_sha256",
            "runtime_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            self.canonical_evaluation_spec_id is not None
            and not self.canonical_evaluation_spec_id.strip()
        ):
            raise ValueError("canonical_evaluation_spec_id cannot be empty")
        if self.normalization_id == "raw" and any(
            value is not None
            for value in (
                self.normalization_artifact_sha256,
                self.observation_transform_name,
                self.observation_transform_sha256,
            )
        ):
            raise ValueError("raw observations cannot carry normalization artifacts")
        if not self.runtime_versions:
            raise ValueError("runtime_versions must not be empty")
        runtime_names = tuple(name for name, _value in self.runtime_versions)
        if tuple(sorted(runtime_names)) != runtime_names or len(set(runtime_names)) != len(
            runtime_names
        ):
            raise ValueError("runtime_versions must have sorted, unique names")
        if any(not name or not value for name, value in self.runtime_versions):
            raise ValueError("runtime version names and values must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_stage": self.curriculum_stage,
            "curriculum_stage_name": self.curriculum_stage_name,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "evaluation_suite_fingerprint": self.evaluation_suite_fingerprint,
            "canonical_evaluation_spec_id": self.canonical_evaluation_spec_id,
            "environment_class": self.environment_class,
            "policy_id": self.policy_id,
            "policy_state_sha256": self.policy_state_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha256,
            "normalization_id": self.normalization_id,
            "normalization_artifact_sha256": (
                self.normalization_artifact_sha256
            ),
            "observation_transform_name": self.observation_transform_name,
            "observation_transform_sha256": self.observation_transform_sha256,
            "package_version": self.package_version,
            "package_source_sha256": self.package_source_sha256,
            "git_sha": self.git_sha,
            "runtime_versions": dict(self.runtime_versions),
            "runtime_fingerprint": self.runtime_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CurriculumEpisodeResult:
    """Terminal evidence from one exact held-out episode."""

    condition: HeldOutCondition
    steps: int
    total_reward: float
    initial_ball_qpos: tuple[float, ...]
    initial_ball_qvel: tuple[float, ...]
    stage_success: bool
    rally_count: int
    valid_return_count: int
    return_opportunity_count: int
    terminated: bool
    truncated: bool
    termination_reason: str
    fault_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("episode steps must be at least one")
        if not math.isfinite(self.total_reward):
            raise ValueError("episode reward must be finite")
        if len(self.initial_ball_qpos) != 7 or not all(
            math.isfinite(value) for value in self.initial_ball_qpos
        ):
            raise ValueError("initial_ball_qpos must be a finite seven-vector")
        if len(self.initial_ball_qvel) != 6 or not all(
            math.isfinite(value) for value in self.initial_ball_qvel
        ):
            raise ValueError("initial_ball_qvel must be a finite six-vector")
        for name, value in (
            ("rally_count", self.rally_count),
            ("valid_return_count", self.valid_return_count),
            ("return_opportunity_count", self.return_opportunity_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.valid_return_count > self.return_opportunity_count:
            raise ValueError(
                "valid_return_count cannot exceed return_opportunity_count"
            )
        if not (self.terminated or self.truncated):
            raise ValueError("held-out episodes must end before being summarized")
        if tuple(sorted(set(self.fault_flags))) != self.fault_flags:
            raise ValueError("fault_flags must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            **self.condition.to_dict(),
            "steps": self.steps,
            "total_reward": self.total_reward,
            "initial_ball_qpos": list(self.initial_ball_qpos),
            "initial_ball_qvel": list(self.initial_ball_qvel),
            "stage_success": self.stage_success,
            "rally_count": self.rally_count,
            "valid_return_count": self.valid_return_count,
            "return_opportunity_count": self.return_opportunity_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "fault_flags": list(self.fault_flags),
        }


@dataclass(frozen=True, slots=True)
class CurriculumEvaluationSummary:
    """Immutable aggregate of one stage's held-out episode results."""

    stage_name: str
    provenance: EvaluationProvenance
    episodes: tuple[CurriculumEpisodeResult, ...]
    unique_initial_state_count: int
    success_count: int
    success_rate: float
    success_rate_ci95: tuple[float, float]
    valid_return_count: int
    return_opportunity_count: int
    valid_return_rate: float
    rally_min: float
    rally_p50: float
    rally_p90: float
    rally_max: float
    fault_rates: tuple[RateSummary, ...]
    termination_rates: tuple[RateSummary, ...]

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if self.stage_name != self.provenance.curriculum_stage_name:
            raise ValueError("stage_name contradicts evaluation provenance")
        if not self.episodes:
            raise ValueError("an evaluation summary needs at least one episode")
        if self.success_count < 0 or self.success_count > len(self.episodes):
            raise ValueError("success_count is inconsistent with episode count")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must lie in [0, 1]")
        low, high = self.success_rate_ci95
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("success-rate confidence interval is invalid")
        if self.return_opportunity_count < 0 or self.valid_return_count < 0:
            raise ValueError("return totals must be non-negative")
        if not 0.0 <= self.valid_return_rate <= 1.0:
            raise ValueError("valid_return_rate must lie in [0, 1]")
        if not 1 <= self.unique_initial_state_count <= len(self.episodes):
            raise ValueError("unique initial-state count is inconsistent")

    @property
    def conditions(self) -> tuple[HeldOutCondition, ...]:
        """Exact seed/side schedule represented by this summary."""
        return tuple(episode.condition for episode in self.episodes)

    def fault_rate(self, name: str) -> float:
        """Return one fault rate, raising when the schema lacks ``name``."""
        for summary in self.fault_rates:
            if summary.name == name:
                return summary.rate
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "stage_name": self.stage_name,
            "provenance": self.provenance.to_dict(),
            "episode_count": len(self.episodes),
            "unique_initial_state_count": self.unique_initial_state_count,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "success_rate_ci95": list(self.success_rate_ci95),
            "valid_return_count": self.valid_return_count,
            "return_opportunity_count": self.return_opportunity_count,
            "valid_return_rate": self.valid_return_rate,
            "rally_distribution": {
                "min": self.rally_min,
                "p50": self.rally_p50,
                "p90": self.rally_p90,
                "max": self.rally_max,
            },
            "fault_rates": {
                item.name: {"count": item.count, "rate": item.rate}
                for item in self.fault_rates
            },
            "termination_rates": {
                item.name: {"count": item.count, "rate": item.rate}
                for item in self.termination_rates
            },
        }


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Advisory result; callers must explicitly perform any promotion."""

    current: CurriculumEvaluationSummary
    prior_stages: tuple[CurriculumEvaluationSummary, ...]
    config: PromotionConfig
    current_stage_passed: bool
    prior_retention_passed: bool
    eligible_for_manual_promotion: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = self.current_stage_passed and self.prior_retention_passed
        if self.eligible_for_manual_promotion is not expected:
            raise ValueError("promotion eligibility is inconsistent with gate results")
        if not self.reasons:
            raise ValueError("promotion report must explain its decision")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "advisory_only": True,
            "eligible_for_manual_promotion": self.eligible_for_manual_promotion,
            "current_stage_passed": self.current_stage_passed,
            "prior_retention_passed": self.prior_retention_passed,
            "reasons": list(self.reasons),
            "thresholds": {
                "success": self.config.success_threshold,
                "prior_retention": self.config.prior_retention_threshold,
                "min_episodes": self.config.min_episodes,
                "min_unique_initial_states": (
                    self.config.min_unique_initial_states
                ),
                "prior_replay_probability": (
                    self.config.prior_replay_probability
                ),
                "require_zero_unsafe": self.config.require_zero_unsafe,
            },
            "current": self.current.to_dict(),
            "prior_stages": [stage.to_dict() for stage in self.prior_stages],
        }


_CANONICAL_FAULT_NAMES = (
    "ball_net",
    "second_bounce",
    "out_of_bounds",
    "humanoid_net",
    "racket_net",
    "target_miss",
    "unsafe",
)


def _required_bool(info: Mapping[str, Any], key: str) -> bool:
    if key not in info:
        raise KeyError(f"terminal info is missing required key {key!r}")
    value = info[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise TypeError(f"terminal info {key!r} must be boolean")


def _optional_bool(info: Mapping[str, Any], key: str) -> bool:
    if key not in info:
        return False
    return _required_bool(info, key)


def _required_count(info: Mapping[str, Any], key: str) -> int:
    if key not in info:
        raise KeyError(f"terminal info is missing required key {key!r}")
    value = info[key]
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"terminal info {key!r} must be an integer count")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"terminal info {key!r} must be an integer count"
        ) from error
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        raise ValueError(
            f"terminal info {key!r} must be a non-negative integer count"
        )
    return int(number)


def _fault_flags(info: Mapping[str, Any]) -> tuple[str, ...]:
    flags: set[str] = set()
    ball_net_count = _required_count(info, "event_count_ball_net")
    if ball_net_count > 0 or _optional_bool(info, "term_ball_net"):
        flags.add("ball_net")
    if _optional_bool(info, "term_second_bounce"):
        flags.add("second_bounce")
    if _optional_bool(info, "term_out_of_bounds"):
        flags.add("out_of_bounds")
    if any(
        _optional_bool(info, key)
        for key in (
            "term_humanoid_net",
            "term_humanoid_a_net",
            "term_humanoid_b_net",
        )
    ):
        flags.add("humanoid_net")
    if any(
        _optional_bool(info, key)
        for key in (
            "term_racket_net",
            "term_racket_a_net",
            "term_racket_b_net",
        )
    ):
        flags.add("racket_net")
    if _optional_bool(info, "term_target_miss"):
        flags.add("target_miss")
    if any(
        _optional_bool(info, key)
        for key in ("term_unsafe", "term_nonfinite_state", "term_unsafe_physics")
    ):
        flags.add("unsafe")
    return tuple(sorted(flags))


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval without a scipy dependency."""
    if total < 1:
        raise ValueError("Wilson interval needs at least one trial")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _file_sha256(path: str | Path | None, *, label: str) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must identify an existing file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_parameter_value(digest: Any, value: Any) -> None:
    """Deterministically hash nested SB3 parameter/state-dict values."""
    if isinstance(value, Mapping):
        digest.update(b"mapping[")
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode("utf-8"))
            _hash_parameter_value(digest, value[key])
        digest.update(b"]")
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"sequence:{len(value)}[".encode())
        for item in value:
            _hash_parameter_value(digest, item)
        digest.update(b"]")
        return
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object-valued policy parameters cannot be fingerprinted")
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
        return
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        digest.update(f"{type(value).__qualname__}:{value!r}".encode())
        return
    raise TypeError(f"unsupported policy parameter type: {type(value).__qualname__}")


def _policy_state_sha256(policy: Any) -> str | None:
    get_parameters = getattr(policy, "get_parameters", None)
    if not callable(get_parameters):
        return None
    parameters = get_parameters()
    digest = hashlib.sha256()
    _hash_parameter_value(digest, parameters)
    return digest.hexdigest()


def _verified_policy_artifact_sha256(
    policy: Any,
    path: str | Path | None,
    live_state_sha256: str | None,
) -> str | None:
    artifact_sha256 = _file_sha256(path, label="policy_artifact_path")
    if path is None:
        return None
    if live_state_sha256 is None:
        raise TypeError(
            "policy_artifact_path verification requires policy.get_parameters()"
        )
    loader = getattr(type(policy), "load", None)
    if not callable(loader):
        raise TypeError("policy_artifact_path verification requires policy.load()")
    artifact_policy = loader(str(Path(path).expanduser().resolve()), device="cpu")
    artifact_state_sha256 = _policy_state_sha256(artifact_policy)
    if artifact_state_sha256 != live_state_sha256:
        raise ValueError("policy artifact parameters do not match the evaluated policy")
    return artifact_sha256


def _callable_sha256(function: Callable[[Any], Any] | None) -> str | None:
    if function is None:
        return None
    target = getattr(function, "__func__", function)
    code = getattr(target, "__code__", None)
    if code is None and callable(target):
        call = type(target).__call__
        code = getattr(call, "__code__", None)
    if code is None:
        raise TypeError("observation_transform code cannot be fingerprinted")
    digest = hashlib.sha256()
    digest.update(str(getattr(target, "__module__", "")).encode("utf-8"))
    digest.update(str(getattr(target, "__qualname__", type(target).__qualname__)).encode())
    digest.update(code.co_code)
    digest.update(repr(code.co_consts).encode("utf-8"))
    digest.update(repr(code.co_names).encode("utf-8"))
    return digest.hexdigest()


def _load_normalization_transform(
    path: str | Path,
) -> Callable[[Any], Any]:
    """Load the exact frozen SB3 VecNormalize snapshot used for inference."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"normalization_artifact_path must identify an existing file: {resolved}"
        )
    with resolved.open("rb") as stream:
        normalizer = pickle.load(stream)
    normalizer_class = f"{type(normalizer).__module__}.{type(normalizer).__qualname__}"
    if normalizer_class != (
        "stable_baselines3.common.vec_env.vec_normalize.VecNormalize"
    ):
        raise TypeError(
            "normalization_artifact_path must contain an SB3 VecNormalize snapshot"
        )
    transform = getattr(normalizer, "normalize_obs", None)
    if not callable(transform):
        raise TypeError("VecNormalize snapshot does not expose normalize_obs")
    return transform


def _package_source_sha256() -> str:
    """Hash installed/source package bytes, including physics assets."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions() -> tuple[tuple[str, str], ...]:
    versions = {
        "machine": platform.machine() or "unknown",
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for label, distribution in (
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
        ("stable_baselines3", "stable-baselines3"),
        ("torch", "torch"),
    ):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return tuple(sorted(versions.items()))


def _git_sha() -> str | None:
    package_path = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(package_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _suite_fingerprint(suite: HeldOutSeedSuite) -> str:
    return _metadata_fingerprint(
        {
            "base_seed": suite.base_seed,
            "launch_count": suite.launch_count,
            "mirror_each": suite.mirror_each,
            "max_position_offset": suite.max_position_offset,
            "max_velocity_offset": suite.max_velocity_offset,
            "conditions": [condition.to_dict() for condition in suite.conditions],
        }
    )


_CANONICAL_EPISODE_LENGTHS = {0: 150, 1: 300, 2: 300}
_CANONICAL_ENVIRONMENT_CLASS = (
    "courtside_dynamics.envs.humanoid_tennis.HumanoidTennisCoopEnv"
)


def _canonical_evaluation_spec_id(
    metadata: Mapping[str, Any],
    suite: HeldOutSeedSuite,
    environment_class: str,
) -> str | None:
    """Identify exact, approved promotion conditions; return None otherwise."""
    try:
        stage = int(metadata["curriculum_stage"])
        preset = TENNIS_CURRICULUM_PRESETS[CurriculumStage(stage)]
    except (KeyError, TypeError, ValueError):
        return None
    if stage not in _CANONICAL_EPISODE_LENGTHS:
        return None
    rules = preset.config.rally_rules()
    expected = {
        "curriculum_enabled": True,
        "curriculum_stage_name": preset.config.name,
        "curriculum_matches_canonical_preset": True,
        "robot_model": "unitree_g1",
        "episode_len": _CANONICAL_EPISODE_LENGTHS[stage],
        "frame_skip": 10,
        "model_timestep": 0.001,
        "control_timestep": 0.01,
        "action_size": 58,
        "observation_size": 299,
        "rally_target": preset.config.rally_target,
        "effective_ball_net_is_fault": rules.ball_net_is_fault,
        "effective_strict_reverse_crossing": rules.strict_reverse_crossing,
        "effective_strict_double_hit": rules.strict_double_hit,
        "stage_success_reward": preset.config.stage_success_reward,
        "terminate_on_success": True,
        "valid_return_reward": 1.0,
        "valid_hit_shaping": 0.0,
        "fault_penalty": 1.0,
        "unsafe_physics_penalty": 2.0,
        "sampler_min_contact_force": 0.0,
        "sampler_release_substeps": 2,
        "sampler_crossing_deadband": 1e-5,
        "safety_max_ball_linear_speed": 120.0,
        "safety_max_abs_qvel": 2_000.0,
        "safety_max_abs_qacc": 10_000_000.0,
        "curriculum_launch_overridden": False,
    }
    if environment_class != _CANONICAL_ENVIRONMENT_CLASS:
        return None
    if suite != HeldOutSeedSuite():
        return None
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return f"humanoid-tennis-stage-{stage}-promotion-v1"


def summarize_curriculum_episodes(
    stage_name: str,
    episodes: Sequence[CurriculumEpisodeResult],
    *,
    provenance: EvaluationProvenance,
) -> CurriculumEvaluationSummary:
    """Aggregate immutable episode evidence for one curriculum stage."""
    episode_tuple = tuple(episodes)
    if not stage_name:
        raise ValueError("stage_name must not be empty")
    if not episode_tuple:
        raise ValueError("at least one episode is required")

    successes = sum(episode.stage_success for episode in episode_tuple)
    count = len(episode_tuple)
    valid_returns = sum(episode.valid_return_count for episode in episode_tuple)
    opportunities = sum(
        episode.return_opportunity_count for episode in episode_tuple
    )
    rally_counts = np.asarray(
        [episode.rally_count for episode in episode_tuple], dtype=np.float64
    )
    unique_initial_states = len(
        {
            (episode.initial_ball_qpos, episode.initial_ball_qvel)
            for episode in episode_tuple
        }
    )

    fault_rates = tuple(
        RateSummary(
            name=name,
            count=sum(name in episode.fault_flags for episode in episode_tuple),
            rate=sum(name in episode.fault_flags for episode in episode_tuple)
            / count,
        )
        for name in _CANONICAL_FAULT_NAMES
    )
    termination_names = sorted(
        {
            episode.termination_reason
            for episode in episode_tuple
            if episode.termination_reason not in {"", "none", "stage_success"}
        }
    )
    termination_rates = tuple(
        RateSummary(
            name=name,
            count=sum(
                episode.termination_reason == name for episode in episode_tuple
            ),
            rate=sum(
                episode.termination_reason == name for episode in episode_tuple
            )
            / count,
        )
        for name in termination_names
    )
    return CurriculumEvaluationSummary(
        stage_name=stage_name,
        provenance=provenance,
        episodes=episode_tuple,
        unique_initial_state_count=unique_initial_states,
        success_count=int(successes),
        success_rate=float(successes / count),
        success_rate_ci95=_wilson_interval(int(successes), count),
        valid_return_count=valid_returns,
        return_opportunity_count=opportunities,
        valid_return_rate=(valid_returns / opportunities if opportunities else 0.0),
        rally_min=float(np.min(rally_counts)),
        rally_p50=float(np.quantile(rally_counts, 0.50)),
        rally_p90=float(np.quantile(rally_counts, 0.90)),
        rally_max=float(np.max(rally_counts)),
        fault_rates=fault_rates,
        termination_rates=termination_rates,
    )


def _policy_action(policy: Any, observation: Any) -> Any:
    predict = getattr(policy, "predict", None)
    if callable(predict):
        prediction = predict(observation, deterministic=True)
        if isinstance(prediction, tuple):
            if not prediction:
                raise TypeError("policy.predict returned an empty tuple")
            return prediction[0]
        return prediction
    if callable(policy):
        return policy(observation)
    raise TypeError("policy must be callable or expose predict(observation, ...)")


def _required_vector(
    info: Mapping[str, Any],
    key: str,
    size: int,
) -> tuple[float, ...]:
    if key not in info:
        raise KeyError(f"reset info is missing required key {key!r}")
    values = tuple(float(value) for value in info[key])
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"reset info {key!r} must be a finite {size}-vector")
    return values


def _reset_for_condition(
    env: Any,
    condition: HeldOutCondition,
) -> tuple[Any, Mapping[str, Any]]:
    options: dict[str, Any] = {"serve_side": condition.serving_side.label}
    result = env.reset(seed=condition.seed, options=options)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("environment reset must return (observation, info)")
    observation, info = result
    if not isinstance(info, Mapping):
        raise TypeError("reset info must be a mapping")

    if any(condition.local_position_offset) or any(
        condition.local_velocity_offset
    ):
        position = np.asarray(_required_vector(info, "initial_ball_qpos", 7)[:3])
        velocity = np.asarray(_required_vector(info, "initial_ball_qvel", 6)[:3])
        position_offset = np.asarray(condition.local_position_offset).copy()
        velocity_offset = np.asarray(condition.local_velocity_offset).copy()
        if condition.serving_side is CourtSide.B:
            position_offset[:2] *= -1.0
            velocity_offset[:2] *= -1.0
        options.update(
            {
                "ball_position": position + position_offset,
                "ball_velocity": velocity + velocity_offset,
            }
        )
        result = env.reset(seed=condition.seed, options=options)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("environment reset must return (observation, info)")
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("reset info must be a mapping")
    return observation, info


def evaluate_curriculum_stage(
    policy: Any,
    env_fn: Callable[[], Any],
    *,
    policy_id: str,
    normalization_id: str,
    policy_artifact_path: str | Path | None = None,
    normalization_artifact_path: str | Path | None = None,
    stage_name: str | None = None,
    suite: HeldOutSeedSuite | None = None,
    max_steps_per_episode: int = 10_000,
) -> CurriculumEvaluationSummary:
    """Evaluate ``policy`` on a fresh env and an exact held-out schedule.

    Each reset passes both ``seed`` and ``options={"serve_side": ...}``.
    Promotion-eligible evidence also needs an SB3 policy whose live parameters
    can be hashed, its checkpoint path, and (for normalized observations) the
    frozen normalization snapshot path. The evaluator loads that snapshot and
    applies its exact ``VecNormalize.normalize_obs`` transform itself, binding
    inference to the artifact hash. Human-readable IDs are labels only; the
    promotion gate compares the derived SHA-256 identities.
    The environment must end every episode within ``max_steps_per_episode``
    and expose terminal ``stage_success``, ``rally_count``,
    ``valid_return_count``, ``return_opportunity_count``, and
    ``event_count_ball_net`` keys.  Missing metrics fail loudly instead of
    silently producing a confident zero-rate report.
    """
    if not isinstance(policy_id, str) or not isinstance(normalization_id, str):
        raise TypeError("policy_id and normalization_id must be strings")
    if not policy_id.strip() or not normalization_id.strip():
        raise ValueError("policy_id and normalization_id must not be empty")
    if normalization_id == "raw" and normalization_artifact_path is not None:
        raise ValueError("raw observations cannot use a normalization artifact")
    if normalization_id == "raw":
        observation_transform = None
    else:
        if normalization_artifact_path is None:
            raise ValueError(
                "non-raw normalization requires normalization_artifact_path"
            )
        observation_transform = _load_normalization_transform(
            normalization_artifact_path
        )
    if isinstance(max_steps_per_episode, bool) or not isinstance(
        max_steps_per_episode, int
    ):
        raise TypeError("max_steps_per_episode must be an integer")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least one")
    resolved_suite = suite or HeldOutSeedSuite()

    env = env_fn()
    episode_results: list[CurriculumEpisodeResult] = []
    try:
        unwrapped = getattr(env, "unwrapped", env)
        metadata = getattr(unwrapped, "curriculum_metadata", None)
        if not isinstance(metadata, Mapping) or not metadata.get(
            "curriculum_enabled"
        ):
            raise ValueError("held-out evaluation requires a curriculum environment")
        curriculum_stage = int(metadata["curriculum_stage"])
        resolved_stage_name = str(metadata["curriculum_stage_name"])
        if stage_name is not None and stage_name != resolved_stage_name:
            raise ValueError(
                f"stage_name={stage_name!r} contradicts environment stage "
                f"{resolved_stage_name!r}"
            )
        from courtside_dynamics import __version__

        environment_class = (
            f"{type(unwrapped).__module__}.{type(unwrapped).__qualname__}"
        )
        policy_state_sha256 = _policy_state_sha256(policy)
        runtime_versions = _runtime_versions()
        transform_name = None
        if observation_transform is not None:
            transform_name = getattr(
                observation_transform,
                "__qualname__",
                type(observation_transform).__qualname__,
            )
        provenance = EvaluationProvenance(
            curriculum_stage=curriculum_stage,
            curriculum_stage_name=resolved_stage_name,
            curriculum_fingerprint=_metadata_fingerprint(metadata),
            evaluation_suite_fingerprint=_suite_fingerprint(resolved_suite),
            canonical_evaluation_spec_id=_canonical_evaluation_spec_id(
                metadata,
                resolved_suite,
                environment_class,
            ),
            environment_class=environment_class,
            policy_id=policy_id,
            policy_state_sha256=policy_state_sha256,
            policy_artifact_sha256=_verified_policy_artifact_sha256(
                policy,
                policy_artifact_path,
                policy_state_sha256,
            ),
            normalization_id=normalization_id,
            normalization_artifact_sha256=_file_sha256(
                normalization_artifact_path,
                label="normalization_artifact_path",
            ),
            observation_transform_name=transform_name,
            observation_transform_sha256=_callable_sha256(
                observation_transform
            ),
            package_version=__version__,
            package_source_sha256=_package_source_sha256(),
            git_sha=_git_sha(),
            runtime_versions=runtime_versions,
            runtime_fingerprint=_metadata_fingerprint(dict(runtime_versions)),
        )
        for condition in resolved_suite.conditions:
            observation, reset_info = _reset_for_condition(env, condition)
            if int(reset_info.get("curriculum_stage", -1)) != curriculum_stage or str(
                reset_info.get("curriculum_stage_name", "")
            ) != resolved_stage_name:
                raise ValueError("reset curriculum metadata changed during evaluation")
            initial_qpos = _required_vector(reset_info, "initial_ball_qpos", 7)
            initial_qvel = _required_vector(reset_info, "initial_ball_qvel", 6)
            total_reward = 0.0
            terminal_info: Mapping[str, Any] | None = None
            terminated = False
            truncated = False
            steps = 0
            for loop_step in range(1, max_steps_per_episode + 1):
                steps = loop_step
                policy_observation = (
                    observation
                    if observation_transform is None
                    else observation_transform(observation)
                )
                action = _policy_action(policy, policy_observation)
                step_result = env.step(action)
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    raise TypeError(
                        "environment step must return "
                        "(observation, reward, terminated, truncated, info)"
                    )
                observation, reward, terminated, truncated, info = step_result
                reward_value = float(reward)
                if not math.isfinite(reward_value):
                    raise ValueError("held-out evaluation received non-finite reward")
                total_reward += reward_value
                if terminated or truncated:
                    if not isinstance(info, Mapping):
                        raise TypeError("terminal info must be a mapping")
                    terminal_info = info
                    break
            if terminal_info is None:
                raise RuntimeError(
                    "held-out episode did not terminate or truncate within "
                    f"{max_steps_per_episode} steps (seed={condition.seed}, "
                    f"side={condition.serving_side.label})"
                )

            episode_results.append(
                CurriculumEpisodeResult(
                    condition=condition,
                    steps=steps,
                    total_reward=total_reward,
                    initial_ball_qpos=initial_qpos,
                    initial_ball_qvel=initial_qvel,
                    stage_success=_required_bool(terminal_info, "stage_success"),
                    rally_count=_required_count(terminal_info, "rally_count"),
                    valid_return_count=_required_count(
                        terminal_info, "valid_return_count"
                    ),
                    return_opportunity_count=_required_count(
                        terminal_info, "return_opportunity_count"
                    ),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    termination_reason=str(
                        terminal_info.get("termination_reason_name", "none")
                    ),
                    fault_flags=_fault_flags(terminal_info),
                )
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return summarize_curriculum_episodes(
        resolved_stage_name,
        episode_results,
        provenance=provenance,
    )


def assess_curriculum_promotion(
    current: CurriculumEvaluationSummary,
    *,
    prior_stages: Sequence[CurriculumEvaluationSummary] = (),
    config: PromotionConfig | None = None,
) -> PromotionReport:
    """Apply current-stage and prior-retention gates without changing state."""
    resolved = config or PromotionConfig()
    priors = tuple(prior_stages)
    all_summaries = (current, *priors)
    names = [summary.stage_name for summary in all_summaries]
    if len(set(names)) != len(names):
        raise ValueError("current and prior stage names must be unique")
    stage_ids = [
        summary.provenance.curriculum_stage for summary in all_summaries
    ]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("current and prior curriculum stage ids must be unique")
    for summary in all_summaries:
        provenance = summary.provenance
        expected_spec_id = (
            f"humanoid-tennis-stage-{provenance.curriculum_stage}-promotion-v1"
        )
        if provenance.canonical_evaluation_spec_id != expected_spec_id:
            raise ValueError(
                f"stage {summary.stage_name!r} was not evaluated with canonical "
                f"promotion spec {expected_spec_id!r}"
            )
        if (
            provenance.policy_state_sha256 is None
            or provenance.policy_artifact_sha256 is None
        ):
            raise ValueError(
                "promotion requires a fingerprintable live SB3 policy and its "
                "checkpoint artifact"
            )
        if provenance.normalization_id != "raw" and (
            provenance.normalization_artifact_sha256 is None
            or provenance.observation_transform_sha256 is None
        ):
            raise ValueError(
                "normalized promotion evidence requires a frozen normalization "
                "artifact and fingerprintable observation transform"
            )
    current_stage = current.provenance.curriculum_stage
    required_prior_ids = {
        int(stage)
        for stage, preset in TENNIS_CURRICULUM_PRESETS.items()
        if preset.training_recipe_available and int(stage) < current_stage
    }
    provided_prior_ids = {
        summary.provenance.curriculum_stage for summary in priors
    }
    if provided_prior_ids != required_prior_ids:
        missing = sorted(required_prior_ids - provided_prior_ids)
        unexpected = sorted(provided_prior_ids - required_prior_ids)
        raise ValueError(
            "prior-stage evidence must cover every implemented predecessor; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(
        summary.provenance.policy_id != current.provenance.policy_id
        or summary.provenance.policy_state_sha256
        != current.provenance.policy_state_sha256
        or summary.provenance.policy_artifact_sha256
        != current.provenance.policy_artifact_sha256
        or summary.provenance.normalization_id
        != current.provenance.normalization_id
        or summary.provenance.normalization_artifact_sha256
        != current.provenance.normalization_artifact_sha256
        or summary.provenance.observation_transform_sha256
        != current.provenance.observation_transform_sha256
        or summary.provenance.evaluation_suite_fingerprint
        != current.provenance.evaluation_suite_fingerprint
        or summary.provenance.package_version != current.provenance.package_version
        or summary.provenance.package_source_sha256
        != current.provenance.package_source_sha256
        or summary.provenance.git_sha != current.provenance.git_sha
        or summary.provenance.runtime_fingerprint
        != current.provenance.runtime_fingerprint
        or summary.provenance.environment_class
        != current.provenance.environment_class
        for summary in priors
    ):
        raise ValueError(
            "current and prior evidence must use the same policy, "
            "checkpoint, normalization, evaluation suite, environment class, "
            "package source, git revision, and runtime stack"
        )
    for summary in all_summaries:
        if len(summary.episodes) < resolved.min_episodes:
            raise ValueError(
                f"stage {summary.stage_name!r} has {len(summary.episodes)} "
                f"episodes; promotion requires at least {resolved.min_episodes}"
            )
        if summary.unique_initial_state_count < resolved.min_unique_initial_states:
            raise ValueError(
                f"stage {summary.stage_name!r} has only "
                f"{summary.unique_initial_state_count} unique initial states; "
                "promotion requires at least "
                f"{resolved.min_unique_initial_states}"
            )

    current_success_passed = current.success_rate >= resolved.success_threshold
    current_safe = (
        not resolved.require_zero_unsafe or current.fault_rate("unsafe") == 0.0
    )
    current_passed = current_success_passed and current_safe
    retention_passed = all(
        stage.success_rate >= resolved.prior_retention_threshold
        and (
            not resolved.require_zero_unsafe
            or stage.fault_rate("unsafe") == 0.0
        )
        for stage in priors
    )
    reasons: list[str] = []
    if current_success_passed:
        reasons.append(
            f"current stage success {current.success_rate:.3f} meets "
            f"{resolved.success_threshold:.3f}"
        )
    else:
        reasons.append(
            f"current stage success {current.success_rate:.3f} is below "
            f"{resolved.success_threshold:.3f}"
        )
    if resolved.require_zero_unsafe:
        relation = "is zero" if current_safe else "must be zero"
        reasons.append(
            f"current unsafe rate {current.fault_rate('unsafe'):.3f} {relation}"
        )
    if not priors:
        reasons.append("no prior-stage retention gates apply")
    for stage in priors:
        relation = "meets" if (
            stage.success_rate >= resolved.prior_retention_threshold
        ) else "is below"
        reasons.append(
            f"prior stage {stage.stage_name!r} success "
            f"{stage.success_rate:.3f} {relation} "
            f"{resolved.prior_retention_threshold:.3f}"
        )
        if resolved.require_zero_unsafe:
            unsafe_rate = stage.fault_rate("unsafe")
            unsafe_relation = "is zero" if unsafe_rate == 0.0 else "must be zero"
            reasons.append(
                f"prior stage {stage.stage_name!r} unsafe rate "
                f"{unsafe_rate:.3f} {unsafe_relation}"
            )
    reasons.append("advisory only; promotion must be performed explicitly")

    return PromotionReport(
        current=current,
        prior_stages=priors,
        config=resolved,
        current_stage_passed=current_passed,
        prior_retention_passed=retention_passed,
        eligible_for_manual_promotion=current_passed and retention_passed,
        reasons=tuple(reasons),
    )


__all__ = [
    "CurriculumEpisodeResult",
    "CurriculumEvaluationSummary",
    "EvaluationProvenance",
    "HeldOutCondition",
    "HeldOutSeedSuite",
    "PromotionConfig",
    "PromotionReport",
    "RateSummary",
    "assess_curriculum_promotion",
    "evaluate_curriculum_stage",
    "summarize_curriculum_episodes",
]
