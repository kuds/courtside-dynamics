"""Scripted-ladder calibration sweep for the WallBallDepthCurriculum stages.

Design: docs/design_wall_ball_depth_curriculum.md. Every candidate stage
is scored with a scripted ladder under the open rally style: parked /
crude full-swing (placement-blind — the historical learnability bar) /
charge-and-lead oracle (ballistic lead + landing-point pre-positioning).
Pass criteria (all blocking):

1. Within-stage monotonicity: parked < crude < oracle mean reward.
2. Feasibility: oracle >=2 returns from >=90% of serves per stage.
3. Learnability: crude completes a second exchange in >0% of episodes.
4. No cross-stage difficulty inversion for the oracle bounce mean.

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

#: Candidate ladder (design doc table); the sweep's job is to confirm or
#: force retuning of these values before they ship in the recipe.
#: ``oracle_run_up`` is a probe parameter (ready-position depth behind
#: the predicted landing point), not an env kwarg; it is stripped before
#: env construction.
STAGES = [
    dict(paddle_x_fence=(-2.7, 0.3), paddle_start_x=-1.6, serve_start_x=1.0,
         serve_speed=5.2, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.1),
    dict(paddle_x_fence=(-3.2, -0.6), paddle_start_x=-2.1, serve_start_x=1.0,
         serve_speed=5.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.1),
    dict(paddle_x_fence=(-3.7, -1.0), paddle_start_x=-2.7, serve_start_x=1.0,
         serve_speed=6.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.4),
    dict(paddle_x_fence=(-4.2, -1.2), paddle_start_x=-3.3, serve_start_x=1.0,
         serve_speed=6.5, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.4),
    dict(paddle_x_fence=(-4.7, -1.2), paddle_start_x=-3.9, serve_start_x=1.0,
         serve_speed=7.0, serve_lob=0.0, paddle_joint_damping=8.0,
         oracle_run_up=1.4),
]


_GRAVITY = 9.81


def _map_x(target_x):
    span = (MAPPING[1] - MAP_HOME) if target_x >= MAP_HOME else (
        MAP_HOME - MAPPING[0]
    )
    return float(np.clip((target_x - MAP_HOME) / span, -1, 1))


def _oracle(obs, fence, run_up):
    """Charge-and-lead controller for wide fences.

    The fence-projected baseline oracle waits until the ball is within
    0.9 m before advancing — fine in the narrow baseline lane, fatal
    across a 3+ m fence (the charge starts too late; double bounce).
    This probe keeps the crude policy's proven x behaviour (commit the
    full charge the moment the ball bounces) and adds what an informed
    player knows: a ready position ``run_up`` metres behind the serve's
    predicted landing point (run-up length is swing power at contact),
    and a ballistic y/z lead at the closing-speed intercept while
    charging.
    """
    ball_x, ball_y, ball_z = obs[0:3]
    vx, vy, vz = obs[3:6]
    floor_bounce_count = obs[13]
    paddle_x = ball_x - obs[14]
    incoming = vx < -0.1

    if incoming and floor_bounce_count >= 1.0:
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
        target_x = float(np.clip(land_x - run_up, fence[0],
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


def _job(args):
    stage_idx, policy, episodes = args
    from courtside_dynamics.envs import WallBallEnv

    stage = dict(STAGES[stage_idx])
    run_up = stage.pop("oracle_run_up")
    env = WallBallEnv(**{**BASE, **stage})
    fence = stage["paddle_x_fence"]
    counts: list[int] = []
    rewards: list[float] = []
    causes: Counter[str] = Counter()
    contact_eps = 0
    try:
        for seed in range(episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            info = {}
            for _ in range(env.episode_len):
                if policy == "oracle":
                    action = _oracle(obs, fence, run_up)
                elif policy == "crude":
                    action = _crude(obs, fence)
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
            for key in ("term_oob", "term_double_bounce", "term_stall",
                        "term_timeout"):
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
        causes=dict(causes),
    )


def main() -> int:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    jobs = [(i, p, episodes) for i in range(len(STAGES))
            for p in ("parked", "crude", "oracle")]
    with Pool(min(8, len(jobs))) as pool:
        rows = pool.map(_job, jobs)
    by = {(r["stage"], r["policy"]): r for r in rows}

    print(f"{'stage':>5} {'policy':>8} {'reward':>8} {'bounces':>8} "
          f"{'ge2':>6} {'contact':>8}  causes")
    for i in range(len(STAGES)):
        for p in ("parked", "crude", "oracle"):
            r = by[(i, p)]
            print(f"{i:>5} {p:>8} {r['reward']:>8.2f} {r['mean']:>8.2f} "
                  f"{r['ge2']:>6.0%} {r['contact']:>8.0%}  {r['causes']}")

    failures = []
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
        if prev_oracle is not None and i >= 2:
            # No inversion: a stage must not be dramatically easier than
            # its predecessor for the oracle (U-shape detector).
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
