"""Scripted-ladder calibration sweep for the WallBallDepthCurriculum stages.

Design: docs/design_wall_ball_depth_curriculum.md. Every release stage
is scored with a scripted ladder under the open rally style: parked /
crude full-swing (placement-blind — the historical learnability bar) /
a stage-calibrated oracle (landing-point run-up in broad shallow
windows, timed charge in narrow deep windows).
An uncalibrated pre-bounce chase probe reports whether the geometry still
permits an opening volley, but does not block the sweep.

Pass criteria (all blocking):

1. Static geometry: every start is inside its fence, every adjacent pair
   overlaps, and there is no position shared by every stage.
2. Within-stage monotonicity: parked < crude < oracle mean reward.
3. Feasibility: oracle >=2 returns from >=90% of serves per stage.
4. Learnability: crude completes a second exchange in >0% of episodes.
5. No stage dramatically easier than its predecessor: the oracle
   bounce mean must not exceed the previous stage's by more than 1.5x
   (the U-shape detector that killed the one-bounce depth ladder;
   mild dips and rises inside that band are tolerated and reported).
6. Telemetry integrity: legal hits split exactly into pre- and post-bounce
   hits, opening volleys are a subset of pre-bounce hits, and post-bounce
   completed returns are a subset of all completed returns.

Run: python tools/depth_stage_sweep.py [episodes-per-cell]
"""
from __future__ import annotations

import sys
from collections import Counter
from multiprocessing import Pool

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

#: Release ladder (design doc table); rerun the sweep after changing any
#: geometry, serve, or probe parameter.
#: ``oracle_run_up`` and ``oracle_charge_gap`` are probe-only parameters,
#: not env kwargs; they are stripped before env construction.
STAGES = [
    dict(paddle_x_fence=(-2.7, 0.3), paddle_start_x=-1.6, serve_start_x=1.0,
         serve_speed=5.2, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.1),
    dict(paddle_x_fence=(-3.2, -0.8), paddle_start_x=-2.1, serve_start_x=1.0,
         serve_speed=5.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.0),
    dict(paddle_x_fence=(-3.7, -1.6), paddle_start_x=-2.7, serve_start_x=1.0,
         serve_speed=6.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.0),
    dict(paddle_x_fence=(-4.2, -2.4), paddle_start_x=-3.3, serve_start_x=1.0,
         serve_speed=6.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.8),
    dict(paddle_x_fence=(-4.7, -3.0), paddle_start_x=-3.9, serve_start_x=1.0,
         serve_speed=7.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_charge_gap=1.7),
]

POLICIES = ("parked", "crude", "oracle", "volley_probe")
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


def _job(args):
    stage_idx, policy, episodes = args
    from courtside_dynamics.envs import WallBallEnv

    stage = dict(STAGES[stage_idx])
    run_up = stage.pop("oracle_run_up", None)
    charge_gap = stage.pop("oracle_charge_gap", None)
    env = WallBallEnv(**{**BASE, **stage})
    fence = stage["paddle_x_fence"]
    counts: list[int] = []
    rewards: list[float] = []
    causes: Counter[str] = Counter()
    contact_eps = 0
    telemetry_totals: Counter[str] = Counter()
    telemetry_episodes = 0
    opening_volley_episodes = 0
    telemetry_missing: Counter[str] = Counter()
    telemetry_errors: Counter[str] = Counter()
    try:
        for seed in range(episodes):
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
                else:
                    action = np.zeros(3, dtype=np.float32)
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
            for key in ("term_oob", "term_double_bounce", "term_stall",
                        "term_timeout", "term_nonfinite"):
                if info.get(key):
                    causes[key.removeprefix("term_")] += 1
    finally:
        env.close()
    c = np.array(counts)
    return dict(
        stage=stage_idx, policy=policy,
        reward=float(np.mean(rewards)),
        mean=float(c.mean()),
        ge2=float((c >= 2).mean()),
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
    )


def _format_mean(value):
    return f"{value:>7.2f}" if value is not None else f"{'n/a':>7}"


def _format_rate(value):
    return f"{value:>7.0%}" if value is not None else f"{'n/a':>7}"


def main() -> int:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if episodes < 1:
        raise SystemExit("episodes-per-cell must be a positive integer")

    static_failures = _validate_ladder(STAGES)
    if static_failures:
        print("STATIC LADDER FAILURES:")
        for failure in static_failures:
            print(" -", failure)
        return 1

    jobs = [(i, p, episodes) for i in range(len(STAGES))
            for p in POLICIES]
    with Pool(min(8, len(jobs))) as pool:
        rows = pool.map(_job, jobs)
    by = {(r["stage"], r["policy"]): r for r in rows}

    print(
        f"{'stage':>5} {'policy':>12} {'reward':>8} {'bounces':>8} "
        f"{'ge2':>6} {'contact':>8} {'pre/ep':>7} {'post/ep':>7} "
        f"{'open%':>7} {'pbret/ep':>8}  causes"
    )
    for i in range(len(STAGES)):
        for p in POLICIES:
            r = by[(i, p)]
            print(
                f"{i:>5} {p:>12} {r['reward']:>8.2f} {r['mean']:>8.2f} "
                f"{r['ge2']:>6.0%} {r['contact']:>8.0%} "
                f"{_format_mean(r['pre_hits'])} "
                f"{_format_mean(r['post_hits'])} "
                f"{_format_rate(r['opening_volley_rate'])} "
                f"{_format_mean(r['post_bounce_returns']):>8}  {r['causes']}"
            )

    failures = []
    for i in range(len(STAGES)):
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
    for i in range(len(STAGES)):
        parked, crude, oracle = (by[(i, p)] for p in
                                 ("parked", "crude", "oracle"))
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
    if failures:
        print("SWEEP FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("ALL STAGES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
