"""Scripted-ladder calibration sweep for the WallBallDepthCurriculum stages.

Design: docs/design_wall_ball_depth_curriculum.md. Every release stage
is scored with a scripted ladder under the open rally style: parked
hold-at-start / crude full-swing (placement-blind — the historical
learnability bar) / a stage-calibrated oracle (landing-point run-up in
broad shallow windows, timed charge in narrow deep windows).
An uncalibrated pre-bounce chase probe reports whether the geometry still
permits an opening volley, but does not block the sweep.

Pass criteria (all blocking except where noted):

1. Static geometry: every start is inside its fence, every adjacent pair
   overlaps, and there is no position shared by every stage.
2. Serve alignment: for the aligned candidate, the policy-independent
   first-floor-bounce distribution has an absolute mean offset from the
   stage's declared landing target of <=0.10 m, with >=95% of landings
   within +/-0.30 m. The target is the paddle start for the fully
   aligned ladder and ``(1 - lambda) * (1.0 - aligned)`` metres in front
   of it under ``--serve-origin-blend``. Note this makes the criterion a
   self-consistency check -- "the serve lands where this configuration
   says it will" -- not a test of alignment itself: it passes equally at
   blend 0.0 and 1.0, so it cannot be cited as evidence for one blend
   over another. ``mean_offset_from_start`` carries the physical
   distance. The known-misaligned baseline is measured as a non-blocking
   comparison.
3. Hold-start invariance: the parked controller makes no ball contact and
   completes no return.
4. Within-stage monotonicity: parked < crude < oracle mean reward.
5. Feasibility: oracle >=2 returns from >=90% of serves per stage.
6. Learnability: crude completes a second exchange in >0% of episodes.
7. No stage dramatically easier than its predecessor: the oracle
   bounce mean must not exceed the previous stage's by more than 1.5x
   (the U-shape detector that killed the one-bounce depth ladder;
   mild dips and rises inside that band are tolerated and reported).
8. Telemetry integrity: legal hits split exactly into pre- and post-bounce
   hits, opening volleys are a subset of pre-bounce hits, and post-bounce
   completed returns are a subset of all completed returns.

Run: python tools/depth_stage_sweep.py [episodes-per-cell] [json-output]
     [--seed-start N] [--ladder {aligned,baseline}]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import numpy as np

MAPPING = (-4.7, 0.3)
MAP_HOME = -1.7

BASE = dict(
    rally_style="open",
    min_force=20.0,
    serve_speed_jitter=0.5,
    serve_vy_min=0.8,
    serve_vy_max=2.0,
    paddle_hit_bonus=0.25,
    track_shaping_scale=0.5,
    out_of_bounds_penalty=1.0,
    double_bounce_penalty=1.0,
    stall_penalty=1.0,
    style_violation_penalty=1.0,
    paddle_home_x=MAP_HOME,
    paddle_x_target_range=MAPPING,
)

#: Aligned candidate ladder (design doc table); rerun the sweep after
#: changing any geometry, serve, or probe parameter.
#: ``oracle_run_up`` and ``oracle_charge_gap`` are probe-only parameters,
#: not env kwargs; they are stripped before env construction.
ALIGNED_STAGES = [
    dict(paddle_x_fence=(-2.7, 0.3), paddle_start_x=-1.6, serve_start_x=1.0,
         serve_speed=5.2, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.1),
    dict(paddle_x_fence=(-3.2, -0.8), paddle_start_x=-2.1, serve_start_x=0.69,
         serve_speed=5.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.0),
    dict(paddle_x_fence=(-3.7, -1.6), paddle_start_x=-2.7, serve_start_x=0.34,
         serve_speed=6.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.0),
    dict(paddle_x_fence=(-4.2, -2.4), paddle_start_x=-3.3, serve_start_x=-0.01,
         serve_speed=6.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.8),
    dict(paddle_x_fence=(-4.7, -3.0), paddle_start_x=-3.9, serve_start_x=-0.35,
         serve_speed=7.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.7),
]

#: Fixed-origin production baseline. Its stage geometry, speeds, and
#: controller probes are identical to the candidate so common episode seeds
#: isolate serve origin. Stage 0 is exactly equal across both variants.
BASELINE_STAGES = [
    {**stage, "serve_start_x": 1.0}
    for stage in ALIGNED_STAGES
]

#: The aligned ladder's serve origins, i.e. the far end of the blend that
#: ``--serve-origin-blend`` interpolates from the fixed origin 1.0.
_ALIGNED_SERVE_ORIGINS: tuple[float, ...] = tuple(
    cast(float, stage["serve_start_x"]) for stage in ALIGNED_STAGES
)

# Backward-compatible alias for imports that predate explicit variants.
STAGES = ALIGNED_STAGES
LADDERS = {
    "aligned": ALIGNED_STAGES,
    "baseline": BASELINE_STAGES,
}

#: ``--oracle-probe`` mode names mapped to the stage keys they replace.
#: ``_oracle`` accepts exactly one of the two, so an override swaps the
#: stage's probe key outright rather than setting a value beside it.
_ORACLE_PROBE_KEYS = {
    "run_up": "oracle_run_up",
    "charge_gap": "oracle_charge_gap",
}


def _parse_oracle_probe(values, stage_count):
    """Parse ``--oracle-probe STAGE=MODE:VALUE`` into ``{stage: (mode, value)}``.

    Exists so a calibration grid can be driven from the CLI without
    editing the stage tables: the probe parameters ship hardcoded per
    stage, and the aligned ladder inherited them unchanged from the
    fixed-origin geometry they were tuned against.
    """
    overrides: dict[int, tuple[str, float]] = {}
    for raw in values or ():
        stage_part, separator, spec = raw.partition("=")
        mode, mode_separator, value_part = spec.partition(":")
        if not (separator and mode_separator) or mode not in _ORACLE_PROBE_KEYS:
            raise SystemExit(
                "--oracle-probe expects STAGE=run_up:VALUE or "
                f"STAGE=charge_gap:VALUE, got {raw!r}"
            )
        try:
            stage_index = int(stage_part)
            value = float(value_part)
        except ValueError:
            raise SystemExit(
                "--oracle-probe needs an integer stage and a float value, "
                f"got {raw!r}"
            ) from None
        if not 0 <= stage_index < stage_count:
            raise SystemExit(
                f"--oracle-probe stage {stage_index} is outside the ladder's "
                f"0..{stage_count - 1}"
            )
        if stage_index in overrides:
            raise SystemExit(
                f"--oracle-probe given more than once for stage {stage_index}"
            )
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(
                f"--oracle-probe value must be positive and finite, got {raw!r}"
            )
        overrides[stage_index] = (mode, value)
    return overrides


def _apply_oracle_probe(stages, overrides):
    """Return ``stages`` with probe overrides applied, tables left intact."""
    if not overrides:
        return stages
    resolved = []
    for index, stage in enumerate(stages):
        stage = dict(stage)
        if index in overrides:
            mode, value = overrides[index]
            for key in _ORACLE_PROBE_KEYS.values():
                stage.pop(key, None)
            stage[_ORACLE_PROBE_KEYS[mode]] = value
        resolved.append(stage)
    return resolved


#: Stage keys that configure the sweep rather than the environment. They
#: are stripped before ``WallBallEnv`` construction.
PROBE_ONLY_STAGE_KEYS = (
    "oracle_run_up",
    "oracle_charge_gap",
    "landing_target_offset",
)


def _env_kwargs(stage):
    """Strip sweep-only keys so the rest can construct ``WallBallEnv``."""
    kwargs = dict(stage)
    for key in PROBE_ONLY_STAGE_KEYS:
        kwargs.pop(key, None)
    return kwargs


def _blend_serve_origins(stages, blend):
    """Interpolate serve origins between the fixed origin and the aligned ladder.

    ``serve_start_x(lambda) = 1.0 + lambda * (aligned - 1.0)``, so
    ``lambda = 0`` reproduces the fixed-origin baseline and ``lambda = 1``
    the aligned ladder. The serve landing moves one-for-one with the
    origin, so the declared landing target for an interior blend is
    ``(1 - lambda) * (1.0 - aligned)`` metres in front of the paddle
    start -- the alignment contract is measured against that target
    rather than against ``paddle_start_x``, which only describes the
    fully aligned arm.
    """
    if blend is None:
        return stages
    resolved = []
    for stage, aligned_origin in zip(
        stages, _ALIGNED_SERVE_ORIGINS, strict=True
    ):
        stage = dict(stage)
        # Written as a lerp rather than 1.0 + blend * (aligned - 1.0) so the
        # endpoints are exact: blend 1.0 must reproduce the aligned ladder
        # bit-for-bit, and blend 0.0 the fixed origin.
        stage["serve_start_x"] = (1.0 - blend) * 1.0 + blend * aligned_origin
        stage["landing_target_offset"] = (1.0 - blend) * (1.0 - aligned_origin)
        resolved.append(stage)
    return resolved

POLICIES = ("parked", "crude", "oracle", "volley_probe")
LANDING_MEAN_TOLERANCE = 0.10
LANDING_WINDOW = 0.30
LANDING_REQUIRED_FRACTION = 0.95
WILSON_95_Z = 1.959963984540054
TELEMETRY_KEYS = (
    "pre_bounce_legal_paddle_hit_count",
    "post_bounce_legal_paddle_hit_count",
    "opening_volley_count",
    "post_bounce_completed_return_count",
)

_GRAVITY = 9.81


def _validate_ladder(stages):
    """Return blocking static-geometry failures for ``stages``."""
    if not stages:
        return ["ladder has no stages"]

    failures = []
    fences = []
    for i, stage in enumerate(stages):
        back, front = stage["paddle_x_fence"]
        start = stage["paddle_start_x"]
        fences.append((back, front))
        if not back <= start <= front:
            failures.append(
                f"stage {i}: start {start} is outside fence ({back}, {front})"
            )

    for i, (left, right) in enumerate(zip(fences[:-1], fences[1:], strict=True)):
        overlap_back = max(left[0], right[0])
        overlap_front = min(left[1], right[1])
        if overlap_back >= overlap_front:
            failures.append(
                f"stages {i}/{i + 1}: fences do not overlap with positive width "
                f"({left} / {right})"
            )

    common_back = max(fence[0] for fence in fences)
    common_front = min(fence[1] for fence in fences)
    if common_back <= common_front:
        failures.append(
            "all stages share a reachable x interval "
            f"({common_back}, {common_front})"
        )
    return failures


def _map_x(target_x):
    span = (MAPPING[1] - MAP_HOME) if target_x >= MAP_HOME else (
        MAP_HOME - MAPPING[0]
    )
    return float(np.clip((target_x - MAP_HOME) / span, -1, 1))


def _parked(paddle_start_x):
    """Hold the paddle at its configured start instead of mapping to home."""
    return np.array(
        [_map_x(paddle_start_x), 0.0, 0.0],
        dtype=np.float32,
    )


def _oracle(obs, fence, run_up=None, charge_gap=None):
    """Stage-calibrated feasibility controller for the sliding ladder.

    The fence-projected baseline oracle waits until the ball is within
    0.9 m before advancing — fine in the narrow baseline lane, fatal
    across a 3+ m fence (the charge starts too late; double bounce).
    This probe keeps the crude policy's proven x behaviour (commit the
    full charge the moment the ball bounces) and adds what an informed
    player knows: a ready position ``run_up`` metres behind the serve's
    predicted landing point (run-up length is swing power at contact),
    and a ballistic y/z lead at the closing-speed intercept while
    charging.

    In the narrow deep windows that strategy prepares too close to the
    front and charges too early: the servo reaches its target and
    decelerates before impact. ``charge_gap`` selects the calibrated
    alternative: wait at the fence back until after the bounce and
    until the ball is within that world-x gap, then charge to the front
    while tracking current y/z. Exactly one probe parameter must be
    configured for every stage.
    """
    if (run_up is None) == (charge_gap is None):
        raise ValueError(
            "exactly one of run_up and charge_gap must configure the oracle"
        )
    ball_x, ball_y, ball_z = obs[0:3]
    vx, vy, vz = obs[3:6]
    floor_bounce_count = obs[13]
    paddle_x = ball_x - obs[14]
    incoming = vx < -0.1

    if charge_gap is not None:
        should_charge = (
            incoming
            and floor_bounce_count >= 1.0
            and ball_x - paddle_x <= charge_gap
        )
        target_x = fence[1] if should_charge else fence[0]
        target_y = ball_y
        target_z = max(0.25, ball_z)
    elif incoming and floor_bounce_count >= 1.0:
        # Charge the fence front; lead the intercept using the closing
        # speed of ball and servo (~2.5 m/s effective).
        target_x = fence[1]
        t_hit = float(np.clip(
            (ball_x - paddle_x) / max(0.5, -vx + 2.5), 0.0, 1.2,
        ))
        target_y = ball_y + vy * t_hit
        target_z = ball_z + vz * t_hit - 0.5 * _GRAVITY * t_hit**2
        target_z = max(target_z, 0.25)
    elif incoming:
        # Pre-position behind the predicted landing point, laterally on
        # the landing y. Charging from a stand-still produces weak
        # blocks, so the ready depth trades power against arrival time.
        drop = max(0.0, ball_z - 0.12)
        t_land = (vz + np.sqrt(vz**2 + 2.0 * _GRAVITY * drop)) / _GRAVITY
        land_x = ball_x + vx * t_land
        target_x = float(np.clip(land_x - float(run_up), fence[0],
                                 fence[1] - 0.3))
        target_y = ball_y + vy * t_land
        target_z = max(0.6, ball_z - 0.4)
    else:
        target_x = fence[0]
        target_y = ball_y
        target_z = ball_z

    ay = float(np.clip(target_y / 3.0, -1, 1))
    qz = target_z - 1.2
    az = float(np.clip(qz / (2.0 if qz >= 0 else 0.9), -1, 1))
    return np.array([_map_x(target_x), ay, az], dtype=np.float32)


def _crude(obs, fence):
    """Placement-blind full swing: retreat pre-bounce, slam to the front."""
    floor_bounce_count = obs[13]
    z = obs[2] - 1.2
    ay = float(np.clip(obs[1] / 3.0, -1, 1))
    az = float(np.clip(z / (2.0 if z >= 0 else 0.9), -1, 1))
    incoming = obs[3] < -0.1
    target_x = fence[1] if (incoming and floor_bounce_count >= 1.0) else fence[0]
    span = (MAPPING[1] - MAP_HOME) if target_x >= MAP_HOME else (
        MAP_HOME - MAPPING[0]
    )
    ax = float(np.clip((target_x - MAP_HOME) / span, -1, 1))
    return np.array([ax, ay, az], dtype=np.float32)


def _volley_probe(obs, fence):
    """Uncalibrated diagnostic that chases the serve before its first bounce."""
    ball_x, ball_y, ball_z = obs[0:3]
    vx, vy, vz = obs[3:6]
    floor_bounce_count = obs[13]
    incoming = vx < -0.1

    if incoming and floor_bounce_count < 1.0:
        # A small ballistic lead keeps this a genuine interception attempt
        # instead of merely following the ball's previous frame.
        lead = 0.06
        target_x = float(np.clip(ball_x + vx * lead, fence[0], fence[1]))
        target_y = ball_y + vy * lead
        target_z = max(
            0.25,
            ball_z + vz * lead - 0.5 * _GRAVITY * lead**2,
        )
    else:
        # The cell answers one question only: can this fence reach the
        # opening flight? Do not intentionally mount a post-bounce rally.
        target_x = fence[0]
        target_y = 0.0
        target_z = 0.25

    ay = float(np.clip(target_y / 3.0, -1, 1))
    qz = target_z - 1.2
    az = float(np.clip(qz / (2.0 if qz >= 0 else 0.9), -1, 1))
    return np.array([_map_x(target_x), ay, az], dtype=np.float32)


def _episode_telemetry(info):
    """Extract and validate the new cumulative episode counters."""
    missing = tuple(key for key in TELEMETRY_KEYS if key not in info)
    if missing:
        return None, missing, ()

    values = {key: int(info[key]) for key in TELEMETRY_KEYS}
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


def _landing_statistics(landing_x, paddle_start_x, target_offset=0.0):
    """Summarize first-bounce x offsets from the stage's declared target.

    ``target_offset`` is how far in front of ``paddle_start_x`` the
    landing is *meant* to fall. It is 0.0 for the fully aligned ladder,
    which is what the contract was originally written against; an
    interior ``--serve-origin-blend`` declares a positive target so the
    same tolerance band can gate it. ``mean_offset_from_start`` keeps the
    physical picture (distance from the paddle) regardless of target.
    """
    values = np.asarray(landing_x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("landing_x must be one-dimensional")
    if not bool(np.isfinite(values).all()):
        raise ValueError("landing_x must contain only finite values")
    if not np.isfinite(paddle_start_x):
        raise ValueError("paddle_start_x must be finite")
    if not np.isfinite(target_offset):
        raise ValueError("target_offset must be finite")

    count = int(values.size)
    if count == 0:
        return dict(
            observed=0,
            landing_mean=None,
            target_offset=float(target_offset),
            mean_offset=None,
            mean_offset_from_start=None,
            offset_std=None,
            offset_p05=None,
            offset_median=None,
            offset_p95=None,
            offset_min=None,
            offset_max=None,
            offset_abs_max=None,
            within_window=None,
        )

    offsets = values - (float(paddle_start_x) + float(target_offset))
    p05, median, p95 = np.quantile(offsets, (0.05, 0.50, 0.95))
    return dict(
        observed=count,
        landing_mean=float(values.mean()),
        target_offset=float(target_offset),
        mean_offset=float(offsets.mean()),
        mean_offset_from_start=float(
            (values - float(paddle_start_x)).mean()
        ),
        offset_std=float(offsets.std()),
        offset_p05=float(p05),
        offset_median=float(median),
        offset_p95=float(p95),
        offset_min=float(offsets.min()),
        offset_max=float(offsets.max()),
        offset_abs_max=float(np.abs(offsets).max()),
        within_window=float((np.abs(offsets) <= LANDING_WINDOW).mean()),
    )


def _wilson_interval(successes, total, z=WILSON_95_Z):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 1:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be a positive finite number")

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


def _landing_alignment_failures(row, *, require_alignment=True):
    """Return blocking failures for one landing-distribution result row.

    Landing observation and pre-bounce-contact integrity apply to both
    variants. The alignment thresholds apply only to the aligned candidate:
    the fixed-origin baseline is measured on the same seeds precisely because
    its deeper-stage offsets are the comparison condition.
    """
    stage = row["stage"]
    failures = []
    if row["observed"] != row["episodes"]:
        failures.append(
            f"stage {stage}: first serve landing observed in "
            f"{row['observed']}/{row['episodes']} episodes"
        )
    if row["pre_bounce_contact_episodes"]:
        failures.append(
            f"stage {stage}: parked paddle contacted the serve before its "
            f"first bounce in {row['pre_bounce_contact_episodes']} episodes"
        )
    if (
        not require_alignment
        or row["mean_offset"] is None
        or row["within_window"] is None
    ):
        return failures
    if abs(row["mean_offset"]) > LANDING_MEAN_TOLERANCE:
        failures.append(
            f"stage {stage}: mean serve-landing offset "
            f"{row['mean_offset']:+.3f} m exceeds "
            f"+/-{LANDING_MEAN_TOLERANCE:.2f} m"
        )
    if row["within_window"] < LANDING_REQUIRED_FRACTION:
        failures.append(
            f"stage {stage}: {row['within_window']:.1%} of serve landings "
            f"within +/-{LANDING_WINDOW:.2f} m < "
            f"{LANDING_REQUIRED_FRACTION:.0%}"
        )
    return failures


def _paddle_contact_before_first_floor(events):
    """Whether raw substep events contain a paddle contact before the floor."""
    first_floor = min(
        (event for event in events if event[2] == "floor"),
        default=None,
    )
    return any(
        event[2] == "paddle"
        and (first_floor is None or event < first_floor)
        for event in events
    )


def _parked_invariant_failures(row):
    """Return failures when hold-at-start earns a return or touches the ball."""
    stage = row["stage"]
    failures = []
    if row["contact"] != 0.0:
        failures.append(
            f"stage {stage}: hold-start controller contacted the ball in "
            f"{row['contact']:.1%} of episodes"
        )
    if row["mean"] != 0.0 or row["ge2_successes"] != 0:
        failures.append(
            f"stage {stage}: hold-start controller scored returns "
            f"(mean={row['mean']:.3f}, ge2={row['ge2_successes']})"
        )
    return failures


def _job(args):
    stage_idx, policy, episodes, seed_start, stage_spec = args
    from courtside_dynamics.envs import WallBallEnv

    stage = dict(stage_spec)
    run_up = stage.pop("oracle_run_up", None)
    charge_gap = stage.pop("oracle_charge_gap", None)
    env = WallBallEnv(**{**BASE, **_env_kwargs(stage)})
    fence = stage["paddle_x_fence"]
    parked_action = _parked(stage["paddle_start_x"])
    counts: list[int] = []
    rewards: list[float] = []
    causes: Counter[str] = Counter()
    contact_eps = 0
    telemetry_totals: Counter[str] = Counter()
    telemetry_episodes = 0
    opening_volley_episodes = 0
    telemetry_missing: Counter[str] = Counter()
    telemetry_errors: Counter[str] = Counter()
    episode_results: list[dict] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            info = {}
            for _ in range(env.episode_len):
                if policy == "oracle":
                    action = _oracle(
                        obs,
                        fence,
                        run_up=run_up,
                        charge_gap=charge_gap,
                    )
                elif policy == "crude":
                    action = _crude(obs, fence)
                elif policy == "volley_probe":
                    action = _volley_probe(obs, fence)
                elif policy == "parked":
                    action = parked_action
                else:
                    raise ValueError(f"unknown policy {policy!r}")
                obs, reward, term, trunc, info = env.step(action)
                total += reward
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
                telemetry_totals.update(telemetry)
                if telemetry["opening_volley_count"] > 0:
                    opening_volley_episodes += 1
                telemetry_errors.update(errors)
            termination_causes = []
            for key in ("term_oob", "term_double_bounce", "term_stall",
                        "term_timeout", "term_nonfinite"):
                if info.get(key):
                    cause = key.removeprefix("term_")
                    causes[cause] += 1
                    termination_causes.append(cause)
            episode_results.append(
                dict(
                    seed=seed,
                    reward=float(total),
                    bounce_count=int(info["bounce_count"]),
                    paddle_hit_count=int(info["paddle_hit_count"]),
                    legal_paddle_hit_count=int(info["legal_paddle_hit_count"]),
                    pre_bounce_legal_paddle_hit_count=(
                        int(info["pre_bounce_legal_paddle_hit_count"])
                        if "pre_bounce_legal_paddle_hit_count" in info
                        else None
                    ),
                    post_bounce_legal_paddle_hit_count=(
                        int(info["post_bounce_legal_paddle_hit_count"])
                        if "post_bounce_legal_paddle_hit_count" in info
                        else None
                    ),
                    opening_volley_count=(
                        int(info["opening_volley_count"])
                        if "opening_volley_count" in info
                        else None
                    ),
                    post_bounce_completed_return_count=(
                        int(info["post_bounce_completed_return_count"])
                        if "post_bounce_completed_return_count" in info
                        else None
                    ),
                    terminated=bool(term),
                    truncated=bool(trunc),
                    termination_causes=termination_causes,
                    telemetry_missing=list(missing),
                    telemetry_errors=list(errors),
                )
            )
    finally:
        env.close()
    c = np.array(counts)
    ge2_successes = int((c >= 2).sum())
    ge2_ci_low, ge2_ci_high = _wilson_interval(ge2_successes, episodes)
    return dict(
        stage=stage_idx, policy=policy,
        reward=float(np.mean(rewards)),
        mean=float(c.mean()),
        ge2=ge2_successes / episodes,
        ge2_successes=ge2_successes,
        ge2_ci_low=ge2_ci_low,
        ge2_ci_high=ge2_ci_high,
        contact=contact_eps / episodes,
        pre_hits=(
            telemetry_totals["pre_bounce_legal_paddle_hit_count"]
            / telemetry_episodes
            if telemetry_episodes
            else None
        ),
        post_hits=(
            telemetry_totals["post_bounce_legal_paddle_hit_count"]
            / telemetry_episodes
            if telemetry_episodes
            else None
        ),
        opening_volley_rate=(
            opening_volley_episodes / telemetry_episodes
            if telemetry_episodes
            else None
        ),
        post_bounce_returns=(
            telemetry_totals["post_bounce_completed_return_count"]
            / telemetry_episodes
            if telemetry_episodes
            else None
        ),
        telemetry_episodes=telemetry_episodes,
        telemetry_missing=dict(telemetry_missing),
        telemetry_errors=dict(telemetry_errors),
        causes=dict(causes),
        samples=episode_results,
    )


def _landing_job(args):
    """Measure the first serve landing without a rally policy in the loop."""
    stage_idx, episodes, seed_start, stage_spec = args
    from courtside_dynamics.envs import WallBallEnv

    stage = dict(stage_spec)
    target_offset = stage.get("landing_target_offset", 0.0)
    env = WallBallEnv(**{**BASE, **_env_kwargs(stage)})
    parked_action = _parked(stage["paddle_start_x"])
    landing_x: list[float] = []
    episode_results: list[dict] = []
    pre_bounce_contact_episodes = 0
    try:
        for seed in range(seed_start, seed_start + episodes):
            _, _ = env.reset(seed=seed)
            contacted_pre_bounce = False
            episode_landing_x = None
            info = {}
            term = False
            trunc = False
            for _ in range(env.episode_len):
                _, _, term, trunc, info = env.step(parked_action)
                contacted_pre_bounce = (
                    contacted_pre_bounce
                    or _paddle_contact_before_first_floor(
                        env._substep_contact_events
                    )
                )
                if info["event_floor_bounce"]:
                    state = env._substep_floor_bounce_state
                    if state is not None:
                        episode_landing_x = float(state[0][0])
                        landing_x.append(episode_landing_x)
                    if contacted_pre_bounce:
                        pre_bounce_contact_episodes += 1
                    break
                if term or trunc:
                    if contacted_pre_bounce:
                        pre_bounce_contact_episodes += 1
                    break
            termination_causes = [
                key.removeprefix("term_")
                for key in (
                    "term_oob",
                    "term_double_bounce",
                    "term_stall",
                    "term_timeout",
                    "term_nonfinite",
                )
                if info.get(key)
            ]
            episode_results.append(
                dict(
                    seed=seed,
                    landing_x=episode_landing_x,
                    offset=(
                        episode_landing_x - stage["paddle_start_x"]
                        if episode_landing_x is not None
                        else None
                    ),
                    pre_bounce_contact=contacted_pre_bounce,
                    terminated=bool(term),
                    truncated=bool(trunc),
                    termination_causes=termination_causes,
                )
            )
    finally:
        env.close()

    return dict(
        stage=stage_idx,
        episodes=episodes,
        pre_bounce_contact_episodes=pre_bounce_contact_episodes,
        **_landing_statistics(
            landing_x, stage["paddle_start_x"], target_offset
        ),
        samples=episode_results,
    )


def _write_json_report(
    path,
    episodes,
    seed_start,
    rows,
    landing_rows,
    failures,
    *,
    ladder_name="aligned",
    stages=None,
    require_alignment=True,
    serve_origin_blend=None,
    calibration_seed_start=None,
    calibration_episodes=None,
    calibration_artifact=None,
    provenance=None,
):
    """Persist exact per-seed outcomes alongside aggregate sweep results."""
    if stages is None:
        stages = STAGES
    calibration = None
    if calibration_seed_start is not None and calibration_episodes is not None:
        calibration = dict(
            seed_range=dict(
                start=calibration_seed_start,
                stop=calibration_seed_start + calibration_episodes,
                stop_is_exclusive=True,
            ),
            artifact=calibration_artifact,
        )
    payload = dict(
        schema_version=3,
        ladder=dict(
            name=ladder_name,
            serve_origin_blend=serve_origin_blend,
            paired_variant=(
                "baseline" if ladder_name == "aligned" else "aligned"
            ),
            pairing_key=f"wall-ball-serve-alignment:{seed_start}:"
                        f"{seed_start + episodes}",
            stage_zero_equivalent=(
                ALIGNED_STAGES[0] == BASELINE_STAGES[0]
            ),
            alignment_thresholds_blocking=require_alignment,
        ),
        episodes_per_cell=episodes,
        seed_range=dict(
            start=seed_start,
            stop=seed_start + episodes,
            stop_is_exclusive=True,
        ),
        base=BASE,
        stages=stages,
        landing_contract=dict(
            absolute_mean_offset_max=LANDING_MEAN_TOLERANCE,
            absolute_offset_window=LANDING_WINDOW,
            fraction_within_window_min=LANDING_REQUIRED_FRACTION,
            alignment_thresholds_blocking=require_alignment,
        ),
        calibration=calibration,
        provenance=provenance,
        controller_cells=rows,
        serve_landings=landing_rows,
        failures=failures,
        passed=not failures,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_provenance():
    """Describe the exact invocation and local source state for the report."""
    repo_root = Path(__file__).resolve().parents[1]

    def _git(*args):
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        return result.stdout if result.returncode == 0 else b""

    status = _git("status", "--porcelain=v1")
    diff = _git("diff", "--binary", "HEAD", "--", ".")
    script_bytes = Path(__file__).read_bytes()
    head = _git("rev-parse", "HEAD").decode("utf-8").strip() or None
    return dict(
        generated_at_utc=datetime.now(UTC).isoformat(),
        command=shlex.join(sys.argv),
        working_directory=str(Path.cwd()),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        git_head=head,
        git_dirty=bool(status),
        git_status_sha256=hashlib.sha256(status).hexdigest(),
        git_tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
        script_sha256=hashlib.sha256(script_bytes).hexdigest(),
    )


def _format_mean(value):
    return f"{value:>7.2f}" if value is not None else f"{'n/a':>7}"


def _format_rate(value):
    return f"{value:>7.0%}" if value is not None else f"{'n/a':>7}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the WallBall depth-stage calibration sweep.",
    )
    parser.add_argument(
        "episodes",
        nargs="?",
        type=int,
        default=200,
        help="episodes per stage/controller cell (default: 200)",
    )
    parser.add_argument(
        "json_output",
        nargs="?",
        help="optional path for exact per-seed JSON results",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed in the contiguous held-out seed range (default: 0)",
    )
    parser.add_argument(
        "--ladder",
        choices=tuple(LADDERS),
        default="aligned",
        help=(
            "serve-origin variant to measure; run both variants with the "
            "same seed range for the paired comparison (default: aligned)"
        ),
    )
    parser.add_argument(
        "--serve-origin-blend",
        type=float,
        metavar="LAMBDA",
        help=(
            "interpolate serve origins between the fixed origin (0.0) and "
            "the aligned ladder (1.0), i.e. 1.0 + LAMBDA * (aligned - 1.0). "
            "The landing contract is then measured against the target the "
            "blend declares, (1 - LAMBDA) * (1.0 - aligned) metres in front "
            "of the paddle start, instead of against paddle_start_x. "
            "Requires --ladder aligned"
        ),
    )
    parser.add_argument(
        "--oracle-probe",
        action="append",
        metavar="STAGE=MODE:VALUE",
        help=(
            "override one stage's oracle probe, e.g. '3=charge_gap:1.4' or "
            "'4=run_up:0.9'. The oracle takes exactly one of run_up / "
            "charge_gap, so this replaces the stage's probe key outright. "
            "Repeatable, at most once per stage; the resolved values are "
            "recorded in the JSON report's stages block"
        ),
    )
    parser.add_argument(
        "--calibration-seed-start",
        type=int,
        help="first seed used to tune controller parameters, if any",
    )
    parser.add_argument(
        "--calibration-episodes",
        type=int,
        help="number of calibration seeds used, if any",
    )
    parser.add_argument(
        "--calibration-artifact",
        help="optional path or identifier for a separate calibration artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    episodes = args.episodes
    seed_start = args.seed_start
    json_output = args.json_output
    ladder_name = args.ladder
    stages = LADDERS[ladder_name]
    blend = args.serve_origin_blend
    if blend is not None:
        if ladder_name != "aligned":
            raise SystemExit(
                "--serve-origin-blend interpolates from the aligned ladder; "
                "use --ladder aligned (blend 0.0 is the baseline)"
            )
        if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise SystemExit(
                "--serve-origin-blend must be between 0.0 and 1.0 inclusive"
            )
        stages = _blend_serve_origins(stages, blend)
    stages = _apply_oracle_probe(
        stages, _parse_oracle_probe(args.oracle_probe, len(stages))
    )
    # A blend declares its own landing target, so the contract stays
    # blocking for it rather than being switched off for anything that is
    # not the fully aligned ladder.
    require_alignment = ladder_name == "aligned"
    if episodes < 1:
        raise SystemExit("episodes-per-cell must be a positive integer")
    if seed_start < 0:
        raise SystemExit("--seed-start must be a non-negative integer")
    calibration_values = (
        args.calibration_seed_start,
        args.calibration_episodes,
    )
    if (calibration_values[0] is None) != (calibration_values[1] is None):
        raise SystemExit(
            "--calibration-seed-start and --calibration-episodes must be "
            "provided together"
        )
    if args.calibration_artifact is not None and calibration_values[0] is None:
        raise SystemExit(
            "--calibration-artifact requires a declared calibration seed range"
        )
    if calibration_values[0] is not None:
        calibration_start, calibration_count = calibration_values
        if calibration_start < 0 or calibration_count < 1:
            raise SystemExit(
                "calibration seed start must be non-negative and episodes "
                "must be positive"
            )
        calibration_stop = calibration_start + calibration_count
        held_out_stop = seed_start + episodes
        if max(calibration_start, seed_start) < min(
            calibration_stop, held_out_stop
        ):
            raise SystemExit(
                "calibration and held-out seed ranges must be disjoint"
            )
    if ALIGNED_STAGES[0] != BASELINE_STAGES[0]:
        raise SystemExit(
            "aligned and baseline stage 0 must be exactly equivalent"
        )

    static_failures = _validate_ladder(stages)
    if static_failures:
        print("STATIC LADDER FAILURES:")
        for failure in static_failures:
            print(" -", failure)
        return 1

    jobs = [
        (i, policy, episodes, seed_start, stage)
        for i, stage in enumerate(stages)
        for policy in POLICIES
    ]
    landing_jobs = [
        (i, episodes, seed_start, stage)
        for i, stage in enumerate(stages)
    ]
    with Pool(min(8, len(jobs))) as pool:
        rows = pool.map(_job, jobs)
        landing_rows = pool.map(_landing_job, landing_jobs)
    by = {(r["stage"], r["policy"]): r for r in rows}
    landing_by = {r["stage"]: r for r in landing_rows}

    print(f"LADDER: {ladder_name}")
    print(
        f"{'stage':>5} {'policy':>12} {'reward':>8} {'bounces':>8} "
        f"{'ge2':>6} {'95% CI':>13} {'contact':>8} {'pre/ep':>7} {'post/ep':>7} "
        f"{'open%':>7} {'pbret/ep':>8}  causes"
    )
    for i in range(len(stages)):
        for p in POLICIES:
            r = by[(i, p)]
            print(
                f"{i:>5} {p:>12} {r['reward']:>8.2f} {r['mean']:>8.2f} "
                f"{r['ge2']:>6.0%} "
                f"[{r['ge2_ci_low']:>4.0%}, {r['ge2_ci_high']:>4.0%}] "
                f"{r['contact']:>8.0%} "
                f"{_format_mean(r['pre_hits'])} "
                f"{_format_mean(r['post_hits'])} "
                f"{_format_rate(r['opening_volley_rate'])} "
                f"{_format_mean(r['post_bounce_returns']):>8}  {r['causes']}"
            )

    print()
    print(
        f"{'stage':>5} {'n':>5} {'landing':>9} {'mean off':>9} "
        f"{'std':>8} {'p05':>8} {'median':>8} {'p95':>8} "
        f"{'abs max':>8} {'within':>8}"
    )
    for i in range(len(stages)):
        row = landing_by[i]
        if row["mean_offset"] is None:
            print(f"{i:>5} {row['observed']:>5} {'n/a':>9}")
            continue
        print(
            f"{i:>5} {row['observed']:>5} {row['landing_mean']:>9.3f} "
            f"{row['mean_offset']:>+9.3f} {row['offset_std']:>8.3f} "
            f"{row['offset_p05']:>+8.3f} {row['offset_median']:>+8.3f} "
            f"{row['offset_p95']:>+8.3f} {row['offset_abs_max']:>8.3f} "
            f"{row['within_window']:>8.1%}"
        )

    failures = []
    for row in landing_rows:
        failures.extend(
            _landing_alignment_failures(
                row,
                require_alignment=require_alignment,
            )
        )
    for i in range(len(stages)):
        for policy in POLICIES:
            row = by[(i, policy)]
            if row["telemetry_episodes"] != episodes:
                failures.append(
                    f"stage {i}/{policy}: telemetry present in "
                    f"{row['telemetry_episodes']}/{episodes} episodes; "
                    f"missing={row['telemetry_missing']}"
                )
            if row["telemetry_errors"]:
                failures.append(
                    f"stage {i}/{policy}: telemetry identities failed "
                    f"{row['telemetry_errors']}"
                )

    prev_oracle = None
    for i in range(len(stages)):
        parked, crude, oracle = (by[(i, p)] for p in
                                 ("parked", "crude", "oracle"))
        failures.extend(_parked_invariant_failures(parked))
        if not (parked["reward"] < crude["reward"] < oracle["reward"]):
            failures.append(f"stage {i}: ladder not monotone "
                            f"({parked['reward']:.2f} / "
                            f"{crude['reward']:.2f} / "
                            f"{oracle['reward']:.2f})")
        if oracle["ge2"] < 0.90:
            failures.append(f"stage {i}: oracle ge2 {oracle['ge2']:.0%} < 90%")
        if crude["ge2"] <= 0.0:
            failures.append(f"stage {i}: crude never completes a 2nd "
                            f"exchange")
        if prev_oracle is not None:
            # A stage must not be dramatically easier than its
            # predecessor for the oracle (U-shape detector); checked
            # for every adjacent pair including stage 0 -> 1.
            if oracle["mean"] > prev_oracle * 1.5:
                failures.append(f"stage {i}: oracle bounces jumped "
                                f"{prev_oracle:.2f} -> {oracle['mean']:.2f} "
                                f"(difficulty inversion)")
        prev_oracle = oracle["mean"]

    print()
    if json_output is not None:
        _write_json_report(
            json_output,
            episodes,
            seed_start,
            rows,
            landing_rows,
            failures,
            ladder_name=ladder_name,
            stages=stages,
            require_alignment=require_alignment,
            serve_origin_blend=blend,
            calibration_seed_start=args.calibration_seed_start,
            calibration_episodes=args.calibration_episodes,
            calibration_artifact=args.calibration_artifact,
            provenance=_run_provenance(),
        )
        print(f"RAW RESULTS: {json_output}")
    if failures:
        print("SWEEP FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("ALL STAGES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
