"""Startup certification for performance-gated wall-ball ladders.

Run ``20260727_233859`` trained for 17 hours on a five-stage ladder that
had never been swept: ``tools/depth_stage_sweep.py`` still carried the
retired 0.21-era stage tables, so the 0.22.0 geometry shipped without
the calibration evidence the sweep exists to provide, and the mismatch
was only discovered by a post-hoc review. This module closes that gap
by deriving the swept ladder from the *live run configuration* — the
same ``performance_gate["stages"]`` the gate will apply and the same
``env_fn`` the workers will build — and running a compact scripted
sweep at ``train()`` startup, writing the verdict into the run's own
artifacts (``reports/ladder_certification.json``). There is no second
stage table to go stale.

What one certification measures, per stage:

- **Stage-application integrity** (new, blocking): every stage key must
  name an attribute that already exists on the constructed env and must
  read back with the value written. This is exactly the failure class
  behind the pre-0.22.0 ``paddle_home_x`` silent no-op, where
  ``set_wrapper_attr(..., force=False)`` "succeeded" against a plain
  attribute the control map never re-read.
- The sweep's scripted reference cells — parked hold-at-start, crude
  full swing, stage-calibrated oracle — with the controller action map
  inverted through the env's *own* ``paddle_home_x`` and
  ``paddle_x_target_range`` after the stage is applied (the standalone
  sweep tool hardcodes a fixed pivot, which is wrong for ladders that
  move it).
- The policy-independent serve-landing distribution (parked paddle,
  first floor bounce), because review 20260727_233859 found the
  stage-0 serve landing exactly on the paddle while stage 1 lands
  +0.34 m in front — a receive-skill discontinuity no rally cell
  surfaces.
- The sweep's blocking criteria (hold-start invariance, monotonicity,
  feasibility, learnability, difficulty inversion, telemetry
  integrity) plus advisory warnings: no scripted reference reaching
  the promotion bar, and adjacent-stage serve-landing jumps beyond the
  ~0.25 m approach-room threshold measured in the serve-alignment
  campaign.

Certification is **advisory by default**: failures are printed loudly
and recorded in the report, but training proceeds unless the spec sets
``enforce``. The stock probes are known to under-perform on fresh
geometry (they are calibrated per stage index, not per geometry), so a
hard gate would brick legitimate runs; the report exists so nobody has
to discover a miscalibrated ladder from a 20-hour learning curve again.

Scope: the controllers and telemetry checks assume ``WallBallEnv``
observations and info counters. Recipes for other environments should
not set a ``ladder_certification`` spec.
"""

from __future__ import annotations

import json
import os
import platform
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

import courtside_dynamics

_GRAVITY = 9.81

#: Reserved scripted-probe seed block. Distinct from every range the
#: experiment ledger has burned for policy or probe evaluation
#: (0-199, 1000-1199, 2000-2199, 3000-3199, 5000-5029, 6000-6039,
#: 10000-10049, 20000-20199) so startup certification never contaminates
#: a held-out comparison.
DEFAULT_SEED_START = 30_000
DEFAULT_EPISODES = 10

#: Oracle feasibility floor and difficulty-inversion ratio, mirrored
#: from ``tools/depth_stage_sweep.py`` so the startup verdict and the
#: standalone campaign tool grade the same criteria.
FEASIBILITY_GE2_FLOOR = 0.90
INVERSION_RATIO = 1.5

#: Adjacent stages whose policy-independent serve landing moves by more
#: than this are flagged: the serve-alignment campaign measured a sharp
#: ~0.25 m approach-room threshold in oracle completions (Phase B), and
#: review 20260727_233859 tied the stage-0->1 stall to a +0.35 m jump.
LANDING_JUMP_WARNING = 0.25

_WILSON_95_Z = 1.959963984540054

_TELEMETRY_KEYS = (
    "pre_bounce_legal_paddle_hit_count",
    "post_bounce_legal_paddle_hit_count",
    "opening_volley_count",
    "post_bounce_completed_return_count",
)

_TERMINATION_KEYS = (
    "term_oob",
    "term_double_bounce",
    "term_stall",
    "term_timeout",
    "term_nonfinite",
)

_SPEC_KEYS = frozenset(
    {
        "episodes",
        "seed_start",
        "enforce",
        "oracle_probes",
        "max_episode_steps",
        "feasibility_ge2_floor",
    }
)

_POLICIES = ("parked", "crude", "oracle")


def _map_action_x(
    target_x: float, home: float, mapping: tuple[float, float]
) -> float:
    """Invert the env's asymmetric x action map for a world-x target.

    A stage may legally pivot ``paddle_home_x`` onto a mapping endpoint
    (the env accepts it and degrades gracefully), which zeroes one
    half-span; the floor keeps the inversion finite so certification
    measures such a ladder instead of crashing on it.
    """
    span = (mapping[1] - home) if target_x >= home else (home - mapping[0])
    return float(np.clip((target_x - home) / max(span, 1e-9), -1.0, 1.0))


def _parked_action(
    start_x: float, home: float, mapping: tuple[float, float]
) -> np.ndarray:
    """Hold the paddle at its configured start instead of drifting home."""
    return np.array(
        [_map_action_x(start_x, home, mapping), 0.0, 0.0], dtype=np.float32
    )


def _crude_action(
    obs: np.ndarray,
    fence: tuple[float, float],
    home: float,
    mapping: tuple[float, float],
) -> np.ndarray:
    """Placement-blind full swing: retreat pre-bounce, slam to the front."""
    floor_bounce_count = obs[13]
    z = obs[2] - 1.2
    ay = float(np.clip(obs[1] / 3.0, -1, 1))
    az = float(np.clip(z / (2.0 if z >= 0 else 0.9), -1, 1))
    incoming = obs[3] < -0.1
    target_x = fence[1] if (incoming and floor_bounce_count >= 1.0) else fence[0]
    return np.array(
        [_map_action_x(target_x, home, mapping), ay, az], dtype=np.float32
    )


def _oracle_action(
    obs: np.ndarray,
    fence: tuple[float, float],
    home: float,
    mapping: tuple[float, float],
    run_up: float | None,
    charge_gap: float | None,
    lead_charge: float | None = None,
) -> np.ndarray:
    """Stage-calibrated feasibility probe (see the sweep tool's docstring).

    ``run_up`` pre-positions behind the predicted landing point and
    charges with a ballistic lead; ``charge_gap`` waits at the fence
    back and commits once the ball is within that world-x gap, tracking
    the ball's current y/z; ``lead_charge`` uses the same commit trigger
    but aims the charge with the ballistic y/z lead at the
    closing-speed intercept. Exactly one must be configured per stage.

    ``lead_charge`` exists because review 20260727_233859 measured the
    y/z-tracking charge arriving laterally wrong on the constant-width
    geometry (stock probes fail the 90% feasibility floor at stages 0
    and 3 of the 0.22.0 ladder); leading the intercept recovers >=90%
    at every stage and is what the recalibrated recipe probes use.
    """
    if (run_up, charge_gap, lead_charge).count(None) != 2:
        raise ValueError(
            "exactly one of run_up, charge_gap, and lead_charge must "
            "configure the oracle"
        )
    ball_x, ball_y, ball_z = obs[0:3]
    vx, vy, vz = obs[3:6]
    floor_bounce_count = obs[13]
    paddle_x = ball_x - obs[14]
    incoming = vx < -0.1

    if lead_charge is not None:
        should_charge = (
            incoming
            and floor_bounce_count >= 1.0
            and ball_x - paddle_x <= lead_charge
        )
        if should_charge:
            target_x = fence[1]
            t_hit = float(
                np.clip((ball_x - paddle_x) / max(0.5, -vx + 2.5), 0.0, 1.2)
            )
            target_y = ball_y + vy * t_hit
            target_z = max(
                0.25, ball_z + vz * t_hit - 0.5 * _GRAVITY * t_hit**2
            )
        else:
            target_x = fence[0]
            target_y = ball_y
            target_z = max(0.25, ball_z)
    elif charge_gap is not None:
        should_charge = (
            incoming
            and floor_bounce_count >= 1.0
            and ball_x - paddle_x <= charge_gap
        )
        target_x = fence[1] if should_charge else fence[0]
        target_y = ball_y
        target_z = max(0.25, ball_z)
    elif incoming and floor_bounce_count >= 1.0:
        target_x = fence[1]
        t_hit = float(
            np.clip((ball_x - paddle_x) / max(0.5, -vx + 2.5), 0.0, 1.2)
        )
        target_y = ball_y + vy * t_hit
        target_z = ball_z + vz * t_hit - 0.5 * _GRAVITY * t_hit**2
        target_z = max(target_z, 0.25)
    elif incoming and run_up is not None:
        drop = max(0.0, ball_z - 0.12)
        t_land = (vz + np.sqrt(vz**2 + 2.0 * _GRAVITY * drop)) / _GRAVITY
        land_x = ball_x + vx * t_land
        target_x = float(
            np.clip(land_x - run_up, fence[0], fence[1] - 0.3)
        )
        target_y = ball_y + vy * t_land
        target_z = max(0.6, ball_z - 0.4)
    else:
        target_x = fence[0]
        target_y = ball_y
        target_z = ball_z

    ay = float(np.clip(target_y / 3.0, -1, 1))
    qz = target_z - 1.2
    az = float(np.clip(qz / (2.0 if qz >= 0 else 0.9), -1, 1))
    return np.array(
        [_map_action_x(target_x, home, mapping), ay, az], dtype=np.float32
    )


def validate_ladder_geometry(stages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return blocking static-geometry failures for ``stages``.

    Mirrors the standalone sweep's checks: every start inside its fence,
    adjacent fences overlapping, and no x interval shared by every stage
    (the all-stage refuge run ``20260722_124613`` exposed).
    """
    if not stages:
        return ["ladder has no stages"]

    # Gate stages are attribute-arbitrary (a serve-only ladder like
    # WallBallBootstrap's never mentions the fence); geometry checks
    # apply only to stages that actually declare geometry, and the
    # cross-stage checks only when every stage does.
    failures = []
    fences = []
    for i, stage in enumerate(stages):
        fence = stage.get("paddle_x_fence")
        start = stage.get("paddle_start_x")
        if fence is None or start is None:
            continue
        back, front = fence
        fences.append((back, front))
        if not back <= start <= front:
            failures.append(
                f"stage {i}: start {start} is outside fence ({back}, {front})"
            )
    if len(fences) != len(stages):
        return failures

    for i, (left, right) in enumerate(zip(fences[:-1], fences[1:], strict=True)):
        if max(left[0], right[0]) >= min(left[1], right[1]):
            failures.append(
                f"stages {i}/{i + 1}: fences do not overlap with positive "
                f"width ({left} / {right})"
            )

    # With two stages, "adjacent fences overlap" and "all stages share
    # an interval" are the same statement, and the former is required —
    # the refuge criterion only separates from it at three stages.
    if len(fences) > 2:
        common_back = max(fence[0] for fence in fences)
        common_front = min(fence[1] for fence in fences)
        if common_back <= common_front:
            failures.append(
                "all stages share a reachable x interval "
                f"({common_back}, {common_front})"
            )
    return failures


def _values_match(read_back: Any, expected: Any) -> bool:
    try:
        return bool(
            np.allclose(
                np.asarray(read_back, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            )
        )
    except (TypeError, ValueError):
        return bool(read_back == expected)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(v) for v in value]
    return value


def apply_stage(env: Any, stage: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """Apply ``stage`` to ``env`` the way the gate does, and verify it took.

    The gate applies stage attributes with
    ``set_wrapper_attr(..., force=False)``, whose contract is "the
    attribute must already exist". This helper enforces the same
    contract on the unwrapped env and additionally requires the value to
    round-trip through ``getattr``, which is what the gate can never
    check — a property whose setter feeds a derived quantity (the
    control-map pivot) passes here only if the property actually exists
    and stores what was written.
    """
    target = env.unwrapped
    applied: dict[str, Any] = {}
    failures: list[str] = []
    for key, value in stage.items():
        if not hasattr(target, key):
            failures.append(
                f"stage attribute {key!r} does not exist on "
                f"{type(target).__name__} — the gate's set_wrapper_attr "
                "would raise on advance (or silently no-op pre-0.20 style)"
            )
            continue
        try:
            setattr(target, key, value)
            read_back = getattr(target, key)
        except Exception as exc:  # noqa: BLE001 — a rejecting property
            # setter is a *finding* about this ladder, not a reason to
            # abort the advisory report ("never raises on a failed
            # criterion").
            failures.append(
                f"stage attribute {key!r} rejected value {value!r}: {exc}"
            )
            continue
        applied[key] = _jsonable(read_back)
        if not _values_match(read_back, value):
            failures.append(
                f"stage attribute {key!r} did not round-trip: wrote "
                f"{value!r}, read back {read_back!r}"
            )
    return applied, failures


def _wilson_interval(
    successes: int, total: int, z: float = _WILSON_95_Z
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if total < 1:
        raise ValueError("total must be positive")
    proportion = successes / total
    z_squared = z**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def _episode_telemetry(info: Mapping[str, Any]):
    """Extract and validate the cumulative episode counters."""
    missing = tuple(key for key in _TELEMETRY_KEYS if key not in info)
    if missing:
        return None, missing, ()

    values = {key: int(info[key]) for key in _TELEMETRY_KEYS}
    errors = []
    if any(value < 0 for value in values.values()):
        errors.append("negative telemetry counter")
    if (
        values["pre_bounce_legal_paddle_hit_count"]
        + values["post_bounce_legal_paddle_hit_count"]
        != int(info["legal_paddle_hit_count"])
    ):
        errors.append("pre + post legal hits != legal hits")
    if (
        values["opening_volley_count"]
        > values["pre_bounce_legal_paddle_hit_count"]
    ):
        errors.append("opening volleys > pre-bounce legal hits")
    if values["post_bounce_completed_return_count"] > int(info["bounce_count"]):
        errors.append("post-bounce completed returns > completed returns")
    if (
        values["post_bounce_completed_return_count"]
        > values["post_bounce_legal_paddle_hit_count"]
    ):
        errors.append("post-bounce completed returns > post-bounce legal hits")
    return values, (), tuple(errors)


def _stage_geometry(env: Any) -> dict[str, Any]:
    """Snapshot the applied geometry the controllers will plan against."""
    target = env.unwrapped
    return {
        "paddle_x_fence": [float(v) for v in target.paddle_x_fence],
        "paddle_start_x": float(target.paddle_start_x),
        "paddle_home_x": float(target.paddle_home_x),
        "paddle_x_target_range": [
            float(v) for v in target.paddle_x_target_range
        ],
        "serve_start_x": float(target.serve_start_x),
        "serve_speed": float(target.serve_speed),
    }


def _run_cell(
    env_fn: Callable[[], Any],
    stage: Mapping[str, Any],
    policy: str,
    probe: Mapping[str, float],
    episodes: int,
    seed_start: int,
    max_episode_steps: int | None,
) -> dict[str, Any]:
    """Roll one (stage, policy) cell and summarize it sweep-style."""
    env = env_fn()
    try:
        apply_stage(env, stage)
        geometry = _stage_geometry(env)
        fence = tuple(geometry["paddle_x_fence"])
        home = geometry["paddle_home_x"]
        mapping = tuple(geometry["paddle_x_target_range"])
        parked = _parked_action(geometry["paddle_start_x"], home, mapping)
        run_up = probe.get("run_up")
        charge_gap = probe.get("charge_gap")
        lead_charge = probe.get("lead_charge")
        step_budget = max_episode_steps or env.unwrapped.episode_len

        counts: list[int] = []
        rewards: list[float] = []
        causes: Counter[str] = Counter()
        contact_eps = 0
        telemetry_episodes = 0
        telemetry_missing: Counter[str] = Counter()
        telemetry_errors: Counter[str] = Counter()
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            info: Mapping[str, Any] = {}
            for _ in range(step_budget):
                if policy == "parked":
                    action = parked
                elif policy == "crude":
                    action = _crude_action(obs, fence, home, mapping)
                elif policy == "oracle":
                    action = _oracle_action(
                        obs, fence, home, mapping, run_up, charge_gap,
                        lead_charge,
                    )
                else:
                    raise ValueError(f"unknown policy {policy!r}")
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward)
                if term or trunc:
                    break
            counts.append(int(info["bounce_count"]))
            rewards.append(total)
            if info["paddle_hit_count"] > 0:
                contact_eps += 1
            telemetry, missing, errors = _episode_telemetry(info)
            if missing:
                telemetry_missing.update(missing)
            else:
                telemetry_episodes += 1
                telemetry_errors.update(errors)
            for key in _TERMINATION_KEYS:
                if info.get(key):
                    causes[key.removeprefix("term_")] += 1
    finally:
        env.close()

    c = np.asarray(counts)
    ge2_successes = int((c >= 2).sum())
    ge2_low, ge2_high = _wilson_interval(ge2_successes, episodes)
    return {
        "policy": policy,
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)),
        "bounce_mean": float(c.mean()),
        "bounce_counts": [int(v) for v in counts],
        "ge2_rate": ge2_successes / episodes,
        "ge2_ci": [ge2_low, ge2_high],
        "contact_rate": contact_eps / episodes,
        "telemetry_episodes": telemetry_episodes,
        "telemetry_missing": dict(telemetry_missing),
        "telemetry_errors": dict(telemetry_errors),
        "termination_causes": dict(causes),
    }


def _run_landing_probe(
    env_fn: Callable[[], Any],
    stage: Mapping[str, Any],
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    """Measure the first serve landing with a parked paddle."""
    env = env_fn()
    try:
        apply_stage(env, stage)
        geometry = _stage_geometry(env)
        home = geometry["paddle_home_x"]
        mapping = tuple(geometry["paddle_x_target_range"])
        start_x = geometry["paddle_start_x"]
        parked = _parked_action(start_x, home, mapping)
        target = env.unwrapped

        landings: list[float] = []
        pre_bounce_contacts = 0
        for seed in range(seed_start, seed_start + episodes):
            env.reset(seed=seed)
            contacted = False
            for _ in range(target.episode_len):
                _, _, term, trunc, info = env.step(parked)
                events = target._substep_contact_events
                first_floor = min(
                    (event for event in events if event[2] == "floor"),
                    default=None,
                )
                contacted = contacted or any(
                    event[2] == "paddle"
                    and (first_floor is None or event < first_floor)
                    for event in events
                )
                if info["event_floor_bounce"]:
                    state = target._substep_floor_bounce_state
                    if state is not None:
                        landings.append(float(state[0][0]))
                    break
                if term or trunc:
                    break
            if contacted:
                pre_bounce_contacts += 1
    finally:
        env.close()

    offsets = [x - start_x for x in landings]
    return {
        "episodes": episodes,
        "observed": len(landings),
        "pre_bounce_contact_episodes": pre_bounce_contacts,
        "landing_mean": float(np.mean(landings)) if landings else None,
        "offset_from_start_mean": (
            float(np.mean(offsets)) if offsets else None
        ),
        "offset_from_start_std": float(np.std(offsets)) if offsets else None,
    }


def certify_ladder(
    env_fn: Callable[[], Any],
    stages: Sequence[Mapping[str, Any]],
    *,
    episodes: int = DEFAULT_EPISODES,
    seed_start: int = DEFAULT_SEED_START,
    oracle_probes: Sequence[Mapping[str, float]],
    gate_metric_key: str | None = None,
    gate_threshold: float | None = None,
    max_episode_steps: int | None = None,
    feasibility_ge2_floor: float = FEASIBILITY_GE2_FLOOR,
) -> dict[str, Any]:
    """Certify a gated ladder against the envs the run will actually train.

    Returns a JSON-serializable report; ``report["failures"]`` holds the
    blocking findings and ``report["warnings"]`` the advisory ones.
    Never raises on a failed criterion — enforcement is the caller's
    policy decision.

    ``feasibility_ge2_floor`` is the oracle >=2-return rate below which
    a stage is declared infeasible. The 0.90 default expresses "a
    scripted reference completes two exchanges almost every serve";
    a recipe whose task is deliberately harder than any scripted
    reference can play (e.g. the 0.25.0 true-baseline era, oracle band
    67%) may declare a lower floor calibrated against a pre-measured
    reference band — the resolved floor is recorded in the report so a
    lowered bar is always visible in the run's artifacts.
    """
    if episodes < 1:
        raise ValueError("certification episodes must be a positive integer")
    if seed_start < 0:
        raise ValueError("certification seed_start must be non-negative")
    if not (0.0 < feasibility_ge2_floor <= 1.0):
        raise ValueError(
            "feasibility_ge2_floor must be in (0, 1], got "
            f"{feasibility_ge2_floor}"
        )
    if len(oracle_probes) != len(stages):
        raise ValueError(
            f"oracle_probes must cover every stage: got {len(oracle_probes)} "
            f"probes for {len(stages)} stages"
        )
    for index, probe in enumerate(oracle_probes):
        if set(probe) not in ({"run_up"}, {"charge_gap"}, {"lead_charge"}):
            raise ValueError(
                f"oracle_probes[{index}] must set exactly one of 'run_up', "
                f"'charge_gap', or 'lead_charge', got {dict(probe)!r}"
            )

    failures = list(validate_ladder_geometry(stages))
    warnings: list[str] = []
    stage_reports: list[dict[str, Any]] = []
    prev_oracle_mean: float | None = None
    prev_landing_offset: float | None = None

    for index, stage in enumerate(stages):
        # Application integrity gets its own env so a bad key is
        # reported once, not once per cell.
        probe_env = env_fn()
        try:
            applied, apply_failures = apply_stage(probe_env, stage)
            geometry = _stage_geometry(probe_env)
        finally:
            probe_env.close()
        stage_failures = [f"stage {index}: {f}" for f in apply_failures]

        cells = {
            policy: _run_cell(
                env_fn,
                stage,
                policy,
                oracle_probes[index],
                episodes,
                seed_start,
                max_episode_steps,
            )
            for policy in _POLICIES
        }
        landing = _run_landing_probe(env_fn, stage, episodes, seed_start)

        parked, crude, oracle = (
            cells["parked"], cells["crude"], cells["oracle"]
        )
        if parked["contact_rate"] != 0.0 or parked["bounce_mean"] != 0.0:
            stage_failures.append(
                f"stage {index}: hold-start controller touched the ball "
                f"(contact {parked['contact_rate']:.0%}, "
                f"bounces {parked['bounce_mean']:.2f})"
            )
        if not (
            parked["reward_mean"] < crude["reward_mean"] < oracle["reward_mean"]
        ):
            stage_failures.append(
                f"stage {index}: reference ladder not monotone "
                f"(parked {parked['reward_mean']:.2f} / crude "
                f"{crude['reward_mean']:.2f} / oracle "
                f"{oracle['reward_mean']:.2f})"
            )
        if oracle["ge2_rate"] < feasibility_ge2_floor:
            stage_failures.append(
                f"stage {index}: oracle >=2-return rate "
                f"{oracle['ge2_rate']:.0%} < {feasibility_ge2_floor:.0%}"
            )
        if crude["ge2_rate"] <= 0.0:
            stage_failures.append(
                f"stage {index}: crude never completes a second exchange"
            )
        if (
            prev_oracle_mean is not None
            and oracle["bounce_mean"] > prev_oracle_mean * INVERSION_RATIO
        ):
            stage_failures.append(
                f"stage {index}: oracle bounces jumped "
                f"{prev_oracle_mean:.2f} -> {oracle['bounce_mean']:.2f} "
                "(difficulty inversion)"
            )
        prev_oracle_mean = oracle["bounce_mean"]

        for policy, cell in cells.items():
            if cell["telemetry_episodes"] != episodes:
                stage_failures.append(
                    f"stage {index}/{policy}: telemetry present in "
                    f"{cell['telemetry_episodes']}/{episodes} episodes; "
                    f"missing={cell['telemetry_missing']}"
                )
            if cell["telemetry_errors"]:
                stage_failures.append(
                    f"stage {index}/{policy}: telemetry identities failed "
                    f"{cell['telemetry_errors']}"
                )
        if landing["observed"] != episodes:
            stage_failures.append(
                f"stage {index}: first serve landing observed in "
                f"{landing['observed']}/{episodes} episodes"
            )
        if landing["pre_bounce_contact_episodes"]:
            stage_failures.append(
                f"stage {index}: parked paddle contacted the serve before "
                f"its first bounce in "
                f"{landing['pre_bounce_contact_episodes']} episodes"
            )

        stage_warnings: list[str] = []
        if gate_threshold is not None and (
            max(oracle["bounce_mean"], crude["bounce_mean"]) < gate_threshold
        ):
            stage_warnings.append(
                f"stage {index}: no scripted reference reaches the "
                f"promotion bar ({gate_metric_key} >= {gate_threshold}); "
                f"best reference mean "
                f"{max(oracle['bounce_mean'], crude['bounce_mean']):.2f}"
            )
        offset = landing["offset_from_start_mean"]
        if (
            offset is not None
            and prev_landing_offset is not None
            and abs(offset - prev_landing_offset) > LANDING_JUMP_WARNING
        ):
            stage_warnings.append(
                f"stage {index}: serve landing offset jumped "
                f"{prev_landing_offset:+.2f} -> {offset:+.2f} m from the "
                f"previous stage (> {LANDING_JUMP_WARNING} m approach-room "
                "threshold; serve-receipt skill may not transfer)"
            )
        if offset is not None:
            prev_landing_offset = offset

        failures.extend(stage_failures)
        warnings.extend(stage_warnings)
        stage_reports.append(
            {
                "index": index,
                "spec": {k: _jsonable(v) for k, v in dict(stage).items()},
                "applied": applied,
                "geometry": geometry,
                "oracle_probe": dict(oracle_probes[index]),
                "cells": cells,
                "landing": landing,
                "failures": stage_failures,
                "warnings": stage_warnings,
            }
        )

    return {
        "schema": 1,
        "provenance": {
            "courtside_dynamics": courtside_dynamics.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "episodes_per_cell": episodes,
            "seed_start": seed_start,
            "feasibility_ge2_floor": feasibility_ge2_floor,
        },
        "gate": {"metric_key": gate_metric_key, "threshold": gate_threshold},
        "stages": stage_reports,
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Render the human-readable certification summary."""
    lines = [
        "LADDER CERTIFICATION "
        f"({report['provenance']['episodes_per_cell']} eps/cell, "
        f"seeds from {report['provenance']['seed_start']}):",
        f"{'stage':>5} {'parked':>7} {'crude':>7} {'oracle':>7} "
        f"{'ge2':>5} {'land off':>9}  verdict",
    ]
    for stage in report["stages"]:
        cells = stage["cells"]
        offset = stage["landing"]["offset_from_start_mean"]
        verdict = "FAIL" if stage["failures"] else (
            "warn" if stage["warnings"] else "ok"
        )
        lines.append(
            f"{stage['index']:>5} "
            f"{cells['parked']['bounce_mean']:>7.2f} "
            f"{cells['crude']['bounce_mean']:>7.2f} "
            f"{cells['oracle']['bounce_mean']:>7.2f} "
            f"{cells['oracle']['ge2_rate']:>5.0%} "
            f"{offset if offset is None else format(offset, '+9.2f')}  "
            f"{verdict}"
        )
    for failure in report["failures"]:
        lines.append(f"  FAIL: {failure}")
    for warning in report["warnings"]:
        lines.append(f"  warn: {warning}")
    lines.append(
        "LADDER CERTIFICATION: "
        + ("all blocking criteria passed" if report["passed"] else
           f"{len(report['failures'])} blocking failure(s) recorded in "
           "the certification report")
    )
    return "\n".join(lines)


def _write_report(output_path: str, report: Mapping[str, Any]) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output_path)


def _skipped_report(
    reason: str,
    gate: Mapping[str, Any],
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    """A report shell for a certification that could not run at all."""
    return {
        "schema": 1,
        "skipped": True,
        "provenance": {
            "courtside_dynamics": courtside_dynamics.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "episodes_per_cell": episodes,
            "seed_start": seed_start,
        },
        "gate": {
            "metric_key": gate.get("metric_key"),
            "threshold": gate.get("threshold"),
        },
        "stages": [],
        "failures": [reason],
        "warnings": [],
        "passed": False,
    }


def run_startup_certification(cfg: Any, output_path: str) -> dict[str, Any]:
    """Certify ``cfg``'s gated ladder and persist the report.

    Reads the stage table from ``cfg.performance_gate`` — the exact
    mapping the gate will apply — and builds envs with ``cfg.env_fn``,
    so there is no second copy of the ladder to drift. Raises only on a
    malformed spec, or on blocking failures when the spec sets
    ``enforce`` (default: advisory).
    """
    spec = dict(cfg.ladder_certification)
    unknown = sorted(set(spec) - _SPEC_KEYS)
    if unknown:
        raise ValueError(
            f"unknown ladder_certification key(s) {unknown}; expected a "
            f"subset of {sorted(_SPEC_KEYS)}"
        )
    gate = cfg.performance_gate
    if not gate or not gate.get("stages"):
        raise ValueError(
            "ladder_certification requires a performance_gate with stages"
        )
    if "oracle_probes" not in spec:
        raise ValueError(
            "ladder_certification spec must set oracle_probes (one "
            '{"run_up": x}, {"charge_gap": y}, or {"lead_charge": z} '
            "table per gate stage), or "
            'be set to "none" in the TOML to disable certification'
        )

    episodes = int(spec.get("episodes", DEFAULT_EPISODES))
    seed_start = int(spec.get("seed_start", DEFAULT_SEED_START))
    stages = [dict(stage) for stage in gate["stages"]]
    probes = [dict(p) for p in spec["oracle_probes"]]
    if len(probes) != len(stages):
        # The documented continuation workflow replaces the gate
        # wholesale with only the remaining stages while the recipe's
        # full probe table survives the merge. That is a legitimate
        # config, not a typo — record the skip instead of refusing to
        # train ("advisory by default" must hold for shape mismatches).
        report = _skipped_report(
            f"certification skipped: {len(probes)} oracle probe(s) for "
            f"{len(stages)} gate stage(s); align "
            "[train.ladder_certification] oracle_probes with the "
            'overridden gate, or set ladder_certification = "none"',
            gate,
            episodes,
            seed_start,
        )
    else:
        report = certify_ladder(
            cfg.env_fn,
            stages,
            episodes=episodes,
            seed_start=seed_start,
            oracle_probes=probes,
            gate_metric_key=gate.get("metric_key"),
            gate_threshold=gate.get("threshold"),
            max_episode_steps=spec.get("max_episode_steps"),
            feasibility_ge2_floor=float(
                spec.get("feasibility_ge2_floor", FEASIBILITY_GE2_FLOOR)
            ),
        )

    _write_report(output_path, report)
    print(format_report(report))
    if spec.get("enforce") and not report["passed"]:
        raise RuntimeError(
            "ladder certification failed with enforce=true: "
            + "; ".join(report["failures"])
        )
    return report
