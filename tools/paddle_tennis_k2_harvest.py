"""k=2 drill harvest — capture real second-ball scenarios from a checkpoint.

The harvest instrument of ``docs/design_paddle_tennis_k2_drill.md`` §2 D1:
replays a named checkpoint deterministically and records, at the **last control
step with ball x > 0.05 m before the net crossing** of every
k=2-opportunity return (own valid hit → opponent valid strike, same
point → return heading to side A; at most one entry per point — a
later same-point opportunity from a converted k=2 is not captured,
by convention), a full scenario tuple:

- the complete physics state (full qpos/qvel — the paddle court is exactly
  ball + six slides, nq=13/nv=12, so this is exact);
- deep copies of the rules machine and event sampler (the full-context
  drill arm restores them; the sampler's model reference is shared out of
  the copy and reattached at restore);
- the observation the policy saw at that step (the restore-fidelity
  target), plus env counters/escrow state needed for exact continuation;
- the recorded continuation (outcome + ball track) for restore validation;
- provenance: repo git sha, artifact sha256s, seed, step.

Seeds must come from the scratch/workpaper range proposed in the design's
§7: harvesting from calibration 5200+ would make the diagnosis
instrument's own scenarios training data, and every reserved/burned
ledger block plus the consumed probe seeds are refused outright. The
default range (9030, 70 episodes) reproduces the §3a registered
library; a harvest of a DIFFERENT checkpoint must draw fresh seeds
from the unconsumed scratch remainder (9168+) — consumed harvest
ranges may be reused only to reproduce their own library.

The library is a single pickle file (schema ``k2-drill-library-v0``).

Usage::

    MUJOCO_GL=disable python tools/paddle_tennis_k2_harvest.py \
        --model .../best_model.zip --vec-normalize .../best_vec_normalize.pkl \
        --out k2_library.pkl [--seed-start 9030] [--episodes 70] \
        [--continuation-steps 200]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import pickle
import subprocess

import numpy as np

from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import CourtSide
from courtside_dynamics.training.paddle_diagnosis import native_checkpoint_policy

SCHEMA = "k2-drill-library-v0"

# Every reserved/burned ledger block, the sealed gate, the diagnosis
# calibration block (train-on-test refusal), and already-consumed
# scratch ranges. See docs/README.md's ledger and the drill design §7.
_REFUSED_BLOCKS = (
    (3000, 3099),
    (3100, 3199),
    (4000, 4099),
    (4100, 4199),
    (4200, 4299),
    (4300, 4399),
    (5000, 5099),
    (5100, 5199),
    (5200, 5299),  # diagnosis calibration: harvesting here would be train-on-test
    (5300, 5399),
    (5400, 5499),
    (5500, 5599),
    (5600, 6199),
    (6200, 6299),
    (6300, 6399),
    (9000, 9029),  # 2026-08-30 feasibility probe (consumed scratch)
    (9100, 9146),  # 2026-08-30 review probes (consumed scratch)
    (9147, 9147),  # 2026-08-30 step-0 replay reset seed (consumed scratch)
)


def _refuse_reserved(seed_start: int, episodes: int) -> None:
    span = range(seed_start, seed_start + episodes)
    for low, high in _REFUSED_BLOCKS:
        if any(low <= seed <= high for seed in span):
            raise SystemExit(
                f"seed range [{seed_start}, {seed_start + episodes}) intersects "
                f"refused block {low}-{high}; refuse to harvest"
            )


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
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _snapshot_env(env: PaddleTennisEnv, obs: np.ndarray) -> dict:
    """Deep-copy everything a full-context restore needs, model shared out."""
    sampler = env._event_sampler
    sampler_copy = copy.deepcopy(sampler, memo={id(sampler.model): None})
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "qacc_warmstart": env.data.qacc_warmstart.copy(),
        "time": float(env.data.time),
        "obs": np.asarray(obs, dtype=np.float64).copy(),
        "rules": copy.deepcopy(env._rules),
        "sampler": sampler_copy,
        "serving_side": env._serving_side,
        "next_serving_side": env._next_serving_side,
        "head_a": np.asarray(
            env._paddle_position(CourtSide.A), dtype=np.float64
        ).copy(),
        "head_b": np.asarray(
            env._paddle_position(CourtSide.B), dtype=np.float64
        ).copy(),
        "step_number": int(env.step_number),
        "crossings": int(env._crossings),
        "crossings_base": int(env._crossings_base),
        "points_played": int(env._points_played),
        "pending_shaping": float(env._pending_shaping),
        "pending_reach": float(env._pending_reach),
        "pending_hold": float(env._pending_hold),
        "hold_anchor_xy": None
        if env._hold_anchor_xy is None
        else np.asarray(env._hold_anchor_xy, dtype=np.float64).copy(),
        "hold_travel": float(env._hold_travel),
        "last_serve_state": None
        if env._last_serve_state is None
        else (
            env._last_serve_state[0].copy(),
            env._last_serve_state[1].copy(),
        ),
    }


def harvest(
    model_path: str,
    vec_normalize_path: str,
    seed_start: int,
    episodes: int,
    continuation_steps: int,
) -> dict:
    policy = native_checkpoint_policy(model_path, vec_normalize_path)
    env = PaddleTennisEnv(
        episode_len=1500,
        court_style="diagnostic",
        volley_rule="fault",
        contact_shaping=0.25,
        reach_shaping=0.25,
        reach_shaping_radius=3.0,
        points_per_episode=None,
    )
    entries: list[dict] = []
    clearance_dropped = 0
    crossings_unarmed = 0  # crossing observed with no armed candidate
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            ball_adr = env._ball_qposadr
            ball_dof = env._ball_dofadr
            last_hit_a_point: int | None = None
            opp_struck_point: int | None = None
            candidate: dict | None = None
            open_entry: dict | None = None
            prev_points = 0
            while True:
                action = policy(obs)
                obs, _, term, trunc, info = env.step(action)
                points = int(info["points_played"])
                ended = bool(term or trunc)
                ball_x = float(env.data.qpos[ball_adr])
                ball_vx = float(env.data.qvel[ball_dof])
                if points != prev_points:
                    # absorbed point boundary: no chain state survives it
                    # (guards the same-control-step hit+fault edge)
                    last_hit_a_point = None
                    opp_struck_point = None
                    candidate = None
                prev_points = points

                if open_entry is not None:
                    e = open_entry
                    tr = env._last_transition
                    if tr is not None:
                        for ev in tr.processed_events:
                            if ev.kind.name == "BALL_RACKET_A":
                                e["cont_touch"] = True
                            if (
                                ev.kind.name == "BALL_COURT_A"
                                and e["cont_bounce_xy"] is None
                            ):
                                pos = np.asarray(ev.position, dtype=np.float64)
                                e["cont_bounce_xy"] = pos[:2].copy()
                    if bool(info["event_valid_racket_hit_a"]):
                        e["cont_legal_hit"] = True
                    if len(e["cont_ball_track"]) < continuation_steps:
                        if not e["cont_ball_track"]:
                            e["cont_first_step"] = int(env.step_number)
                        e["cont_ball_track"].append(
                            env.data.qpos[ball_adr : ball_adr + 3].copy()
                        )
                    if points != e["point"] or ended:
                        e["cont_ender"] = (
                            str(info["termination_reason_name"])
                            if ended
                            else "point_boundary"
                        )
                        e["cont_ball_track"] = np.asarray(e["cont_ball_track"])
                        entries.append(e)
                        open_entry = None

                if bool(info["event_valid_racket_hit_a"]):
                    last_hit_a_point = points
                    opp_struck_point = None
                    candidate = None
                if bool(info["event_valid_racket_hit_b"]):
                    if last_hit_a_point == points:
                        opp_struck_point = points
                        candidate = None
                if (
                    opp_struck_point == points
                    and open_entry is None
                    and not ended
                ):
                    if ball_x > 0.05 and ball_vx < -1.0:
                        candidate = _snapshot_env(env, obs)
                        candidate["seed"] = seed
                        candidate["point"] = points
                    elif ball_x <= 0.0 and candidate is None:
                        crossings_unarmed += 1
                        opp_struck_point = None
                    elif ball_x <= 0.0 and candidate is not None:
                        ball_pos = candidate["qpos"][
                            ball_adr : ball_adr + 3
                        ]
                        clear = all(
                            float(np.linalg.norm(candidate[key] - ball_pos))
                            >= env._SERVE_CLEARANCE
                            for key in ("head_a", "head_b")
                        )
                        if not clear:
                            clearance_dropped += 1
                            candidate = None
                            opp_struck_point = None
                            continue
                        open_entry = candidate
                        open_entry.update(
                            cont_first_step=None,
                            cont_touch=False,
                            cont_legal_hit=False,
                            cont_bounce_xy=None,
                            cont_ender=None,
                            cont_ball_track=[],
                        )
                        candidate = None
                        opp_struck_point = None
                if ended:
                    break
            print(f"seed {seed}: {len(entries)} entries so far")
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "git_sha": _git_sha(),
        "model_path": model_path,
        "model_sha256": _sha256(model_path),
        "vec_normalize_sha256": _sha256(vec_normalize_path),
        "seed_start": seed_start,
        "episodes": episodes,
        "continuation_steps": continuation_steps,
        "clearance_dropped": clearance_dropped,
        "crossings_unarmed": crossings_unarmed,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-start", type=int, default=9030)
    parser.add_argument("--episodes", type=int, default=70)
    parser.add_argument("--continuation-steps", type=int, default=200)
    args = parser.parse_args()
    _refuse_reserved(args.seed_start, args.episodes)
    library = harvest(
        args.model,
        args.vec_normalize,
        args.seed_start,
        args.episodes,
        args.continuation_steps,
    )
    with open(args.out, "wb") as f:
        pickle.dump(library, f)
    n = len(library["entries"])
    print(
        f"library written: {args.out}  ({n} entries from {args.episodes} "
        f"episodes, {n / max(args.episodes, 1):.2f}/episode; "
        f"sha {_sha256(args.out)[:12]})"
    )


if __name__ == "__main__":
    main()
