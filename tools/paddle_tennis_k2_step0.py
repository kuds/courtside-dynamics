"""k=2 drill step-0 rows — both D2 arms scored on a harvested library.

Scores a policy, with zero training, on every entry of a
``paddle_tennis_k2_harvest.py`` library under the two launch arms of
``docs/design_paddle_tennis_k2_drill.md`` §2 D2:

- **feed** (arm a): the harvested physics relaunched as a fresh feed —
  full qpos/qvel restored, fresh ``RallyStateMachine(serving_side=B)``,
  event sampler re-primed. The rally-context observation flags read as a
  serve receive.
- **full** (arm b): the harvested instant restored exactly — physics,
  rules machine, event sampler, counters, escrow state — with the launch
  observation checked against the harvest-recorded observation (the
  KD1-style fidelity report) and the continuation checked against the
  harvest-recorded ball track and outcome (deterministic policy + physics
  should reproduce the recorded play until the point ends; the env RNG is
  not restored, so divergence is expected only after a point boundary).

Per arm it reports: touch rate (any side-A racket contact), legal-hit
rate, paddle-to-bounce distance at the return's first side-A bounce, and
the ender taxonomy. These are the step-0 baselines the design's §4 KD2
banks for the maintainer's D2 fork.

Usage::

    MUJOCO_GL=disable python tools/paddle_tennis_k2_step0.py \
        --library k2_library.pkl \
        --model .../best_model.zip --vec-normalize .../best_vec_normalize.pkl \
        [--arm both] [--policy checkpoint|oracle] [--max-steps 400] \
        [--reset-seed 9147] [--json OUT]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess

import numpy as np

from courtside_dynamics.envs._paddle_court import (
    PADDLE_COURT,
    scripted_ground_opponent,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import CourtSide, RallyStateMachine
from courtside_dynamics.training.paddle_diagnosis import native_checkpoint_policy


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        root = __file__.rsplit("/tools/", 1)[0]
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _make_env() -> PaddleTennisEnv:
    return PaddleTennisEnv(
        episode_len=1500,
        court_style="diagnostic",
        volley_rule="fault",
        contact_shaping=0.25,
        reach_shaping=0.25,
        reach_shaping_radius=3.0,
        points_per_episode=None,
    )


def _restore_sampler(entry: dict, env: PaddleTennisEnv):
    sampler = copy.deepcopy(entry["sampler"])
    sampler.model = env.model
    return sampler


def _launch_feed(env: PaddleTennisEnv, entry: dict) -> np.ndarray:
    """Arm (a): harvested physics as a fresh feed from side B."""
    env.set_state(entry["qpos"].copy(), entry["qvel"].copy())
    if "head_a" in entry:
        ball = entry["qpos"][env._ball_qposadr : env._ball_qposadr + 3]
        for key in ("head_a", "head_b"):
            assert (
                float(np.linalg.norm(entry[key] - ball))
                >= env._SERVE_CLEARANCE
            ), f"drill launch violates clearance vs {key}"
    env._serving_side = CourtSide.B
    env._rules = RallyStateMachine(
        serving_side=CourtSide.B, rules=env._rally_rules, court=PADDLE_COURT
    )
    env._event_sampler.reset(env.data, ball_side=CourtSide.B)
    env._latest_event_batch = None
    env._last_transition = None
    ball = entry["qpos"][env._ball_qposadr : env._ball_qposadr + 3]
    vel = entry["qvel"][env._ball_dofadr : env._ball_dofadr + 3]
    env._last_serve_state = (ball.copy(), vel.copy())
    obs = env._get_obs()
    env._remember_finite_observation(obs)
    return obs


def _launch_full(env: PaddleTennisEnv, entry: dict) -> np.ndarray:
    """Arm (b): restore the harvested instant exactly."""
    env.set_state(entry["qpos"].copy(), entry["qvel"].copy())
    if "qacc_warmstart" in entry:
        env.data.qacc_warmstart[:] = entry["qacc_warmstart"]
        env.data.time = entry["time"]
    env._rules = copy.deepcopy(entry["rules"])
    env._event_sampler = _restore_sampler(entry, env)
    env._serving_side = entry["serving_side"]
    if "next_serving_side" in entry:
        env._next_serving_side = entry["next_serving_side"]
    env.step_number = entry["step_number"]
    env._crossings = entry["crossings"]
    env._crossings_base = entry["crossings_base"]
    env._points_played = entry["points_played"]
    env._pending_shaping = entry["pending_shaping"]
    env._pending_reach = entry["pending_reach"]
    env._pending_hold = entry["pending_hold"]
    env._hold_anchor_xy = (
        None
        if entry["hold_anchor_xy"] is None
        else entry["hold_anchor_xy"].copy()
    )
    env._hold_travel = entry["hold_travel"]
    if entry["last_serve_state"] is not None:
        env._last_serve_state = (
            entry["last_serve_state"][0].copy(),
            entry["last_serve_state"][1].copy(),
        )
    env._latest_event_batch = None
    env._last_transition = None
    obs = env._get_obs()
    env._remember_finite_observation(obs)
    return obs


def run_arm(arm: str, library: dict, policy, max_steps: int, reset_seed: int) -> dict:
    env = _make_env()
    rows: list[dict] = []
    try:
        for entry in library["entries"]:
            env.reset(seed=reset_seed)  # draw discarded by the launch below
            if arm == "feed":
                obs = _launch_feed(env, entry)
            else:
                obs = _launch_full(env, entry)
            obs_fidelity = float(np.max(np.abs(obs - entry["obs"])))
            row = dict(
                touch=False,
                legal_hit=False,
                bounce_dist=None,
                ender=None,
                obs_fidelity=obs_fidelity,
                cont_divergence=None,
                cont_divergence_at10=None,
                outcome_match=None,
            )
            ball_adr = env._ball_qposadr
            track: list[np.ndarray] = []
            for _ in range(max_steps):
                action = policy(obs)
                obs, _, term, trunc, info = env.step(action)
                points = int(info["points_played"])
                tr = env._last_transition
                if tr is not None:
                    for ev in tr.processed_events:
                        if ev.kind.name == "BALL_RACKET_A":
                            row["touch"] = True
                        if (
                            ev.kind.name == "BALL_COURT_A"
                            and row["bounce_dist"] is None
                        ):
                            pos = np.asarray(ev.position, dtype=np.float64)
                            head = np.asarray(
                                env.data.body("player_a_head").xpos,
                                dtype=np.float64,
                            )
                            row["bounce_dist"] = float(
                                np.hypot(head[0] - pos[0], head[1] - pos[1])
                            )
                if bool(info["event_valid_racket_hit_a"]):
                    row["legal_hit"] = True
                track.append(env.data.qpos[ball_adr : ball_adr + 3].copy())
                boundary = points != (
                    entry["points_played"] if arm == "full" else 0
                )
                if term or trunc or boundary:
                    name = str(info["termination_reason_name"])
                    if boundary and not (term or trunc):
                        row["ender"] = f"point_boundary/{name}"
                    elif trunc and not term and name == "none":
                        row["ender"] = "episode_truncated"
                    else:
                        row["ender"] = name
                    break
            if arm == "full":
                rec = entry["cont_ball_track"]
                # alignment by absolute step number: replay track[i] is
                # the state at step_number snap+i+1, harvest rec[k] the
                # state at cont_first_step+k; the recorded boundary step
                # is trimmed because the point-end relaunch consumes env
                # RNG (not restored) and teleports the ball by
                # construction
                offset = (
                    int(entry["cont_first_step"]) - int(entry["step_number"]) - 1
                    if entry.get("cont_first_step") is not None
                    else 1
                )
                n = min(len(rec) - 1, len(track) - offset)
                if n > 0 and offset >= 0:
                    diff = np.linalg.norm(
                        np.asarray(track[offset : offset + n])[:, :2]
                        - rec[:n, :2],
                        axis=1,
                    )
                    row["cont_divergence"] = float(diff.max())
                    row["cont_divergence_at10"] = float(
                        diff[: min(10, n)].max()
                    )
                else:
                    row["cont_divergence_at10"] = None
                row["outcome_match"] = bool(
                    row["legal_hit"] == entry["cont_legal_hit"]
                )
            if row["ender"] is None:
                row["ender"] = "budget_exhausted"
            rows.append(row)
    finally:
        env.close()

    n = len(rows)
    bd = [r["bounce_dist"] for r in rows if r["bounce_dist"] is not None]
    enders: dict[str, int] = {}
    for r in rows:
        enders[str(r["ender"])] = enders.get(str(r["ender"]), 0) + 1
    summary = {
        "arm": arm,
        "entries": n,
        "touch_rate": sum(r["touch"] for r in rows) / max(n, 1),
        "legal_hit_rate": sum(r["legal_hit"] for r in rows) / max(n, 1),
        "bounce_n": len(bd),
        "bounce_dist_mean": float(np.mean(bd)) if bd else None,
        "bounce_within_1m": float(np.mean([d <= 1.0 for d in bd])) if bd else None,
        "obs_fidelity_max": float(max(r["obs_fidelity"] for r in rows)),
        "truncation_censored": sum(
            r["ender"] == "episode_truncated" for r in rows
        ),
        "enders": enders,
    }
    if arm == "full":
        div = [r["cont_divergence"] for r in rows if r["cont_divergence"] is not None]
        d10 = [
            r["cont_divergence_at10"]
            for r in rows
            if r["cont_divergence_at10"] is not None
        ]
        summary["cont_divergence_max"] = float(max(div)) if div else None
        summary["cont_divergence_p50"] = float(np.median(div)) if div else None
        summary["cont_divergence_at10_max"] = float(max(d10)) if d10 else None
        summary["outcome_match_rate"] = sum(
            bool(r["outcome_match"]) for r in rows
        ) / max(n, 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument("--arm", choices=("feed", "full", "both"), default="both")
    parser.add_argument(
        "--policy", choices=("checkpoint", "oracle"), default="checkpoint"
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--reset-seed", type=int, default=9147)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    with open(args.library, "rb") as f:
        library = pickle.load(f)
    if library.get("schema") != "k2-drill-library-v0":
        raise SystemExit(f"unknown library schema: {library.get('schema')!r}")
    if args.policy == "oracle":
        policy = scripted_ground_opponent
    else:
        if not (args.model and args.vec_normalize):
            raise SystemExit("--model and --vec-normalize required for checkpoint policy")
        policy = native_checkpoint_policy(args.model, args.vec_normalize)

    arms = ("feed", "full") if args.arm == "both" else (args.arm,)
    results = []
    for arm in arms:
        summary = run_arm(arm, library, policy, args.max_steps, args.reset_seed)
        results.append(summary)
        print(f"\n=== arm {arm} ({args.policy}) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    if args.json:
        payload = {
            "library_schema": library["schema"],
            "library_path": args.library,
            "library_sha256": _sha256(args.library),
            "library_git_sha": library.get("git_sha"),
            "model_sha256": _sha256(args.model) if args.model else None,
            "vec_normalize_sha256": (
                _sha256(args.vec_normalize) if args.vec_normalize else None
            ),
            "tool_git_sha": _git_sha(),
            "policy": args.policy,
            "max_steps": args.max_steps,
            "reset_seed": args.reset_seed,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwritten -> {args.json}")


if __name__ == "__main__":
    main()
