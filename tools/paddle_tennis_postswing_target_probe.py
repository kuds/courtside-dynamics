"""PT1 — post-swing action-target re-diagnosis (instrument-side).

The question the hold-escrow line closed on
(``docs/design_paddle_tennis_postswing_hold.md`` §4c): the policy
travels 7.5+ m after its own legal hit even when paid to hold — is
that wander **commanded** (the policy's position targets themselves
move away) or **drift** (targets hold, the paddle moves anyway)?
Three reward designs were aimed without this datum; the two fixes it
separates live on different sides of the stack (action/observation
versus dynamics).

The paddle is a position servo: each step's ``data.ctrl`` for the
side-A actuators IS the commanded paddle position, in the same joint
frame as the slide-joint ``qpos``. So the probe replays a checkpoint
deterministically (the diagnosis instrument's loader — checkpoint
behind its own normalizer) and records, for every step of every
post-swing window (side-A legal hit → the opponent's next legal
strike, point boundary, or episode end):

- the commanded target XY (world frame, via the interface's x origin
  and the measured y origin);
- the actual paddle XY (slide qpos, same frame);
- the ball XY (world).

Per window, split at the instrument's ``FOLLOW_THROUGH_STEPS`` (the
swing itself is not "wandering"), it aggregates:

- ``cmd_travel`` / ``act_travel``: XY path length of the command and
  of the paddle — commanded wander moves the first, drift only the
  second;
- ``servo_gap``: mean |command − paddle| — a small gap means the
  paddle is where it was told to be;
- ``cmd_from_hit``: mean/max displacement of the command from the
  paddle's position at the hit — how far away the policy *asks* to go;
- ``cmd_to_ball`` / ``cmd_to_home`` / ``cmd_to_hit``: mean XY
  distance of the command from the (departing) ball, from the
  paddle's home, and from the hit point — which attractor the
  command tracks;
- ``act_sat_frac``: fraction of steps with the raw x/y action beyond
  |0.9| — saturated commands are deliberate, not noise;
- ``cmd_step``: mean per-step command motion (saccade size).

Windows that end in the opponent's return strike (the hold escrow's
paying windows) are also aggregated separately.

Diagnosis-side only: no bars, no verdict — the numbers route the next
design. Seeds default to the calibration block **5200+** (the
diagnosis convention); reserved/burned blocks are refused.

Usage::

    python tools/paddle_tennis_postswing_target_probe.py \
        --model .../best_model.zip \
        --vec-normalize .../best_vec_normalize.pkl \
        [--episodes 30] [--seed-start 5200] [--label NAME] \
        [--hold-shaping 0.0] [--hold-shaping-travel 4.0] [--json OUT]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable

import numpy as np

from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import CourtSide
from courtside_dynamics.training.paddle_diagnosis import (
    FOLLOW_THROUGH_STEPS,
    native_checkpoint_policy,
)

PROBE_SEED_START = 5200
PROBE_EPISODES = 30
WINDOW_CAP_STEPS = 300

_RESERVED_BLOCKS = (
    (4100, 4199),
    (4300, 4399),
    (5300, 5399),
    (5400, 5499),
    (5500, 5599),
    (5600, 6199),
    (6200, 6299),
    (6300, 6399),
)


def _refuse_reserved(seed_start: int, episodes: int) -> None:
    span = range(seed_start, seed_start + episodes)
    for low, high in _RESERVED_BLOCKS:
        if any(low <= seed <= high for seed in span):
            raise SystemExit(
                f"seed range [{seed_start}, {seed_start + episodes}) intersects "
                f"reserved/burned block {low}-{high}; refuse to run"
            )


@dataclasses.dataclass(slots=True)
class WindowRecord:
    """One post-swing window's per-step series (post-follow-through)."""

    end: str  # opponent_strike | boundary | episode_end | cap
    steps: int
    cmd_travel: float
    act_travel: float
    servo_gap: float
    cmd_from_hit_mean: float
    cmd_from_hit_max: float
    cmd_to_ball: float
    cmd_to_home: float
    cmd_to_hit: float
    act_sat_frac: float
    cmd_step: float


class _SideAFrames:
    """World-frame views of side A's command, paddle, and the ball."""

    def __init__(self, env: PaddleTennisEnv) -> None:
        iface = env._paddles[CourtSide.A]
        self._env = env
        self._actuator_indices = iface._actuator_indices
        self._qposadrs = tuple(
            int(env.model.joint(name).qposadr[0])
            for name in (
                "player_a_slide_x",
                "player_a_slide_y",
                "player_a_slide_z",
            )
        )
        # Joint-frame -> world-frame origins, resolved the same way the
        # interface resolves x: body position minus joint qpos at the
        # current (reset) pose.
        head = np.asarray(env.data.body("player_a_head").xpos, dtype=np.float64)
        qpos = np.array(
            [float(env.data.qpos[adr]) for adr in self._qposadrs]
        )
        self._origin_xy = np.array(
            [head[0] - qpos[0], head[1] - qpos[1]], dtype=np.float64
        )
        self.home_xy = self._origin_xy + np.array(
            [0.0, 0.0]
        )  # zero action targets control_home = joint 0

    def cmd_xy(self) -> np.ndarray:
        ctrl = self._env.data.ctrl[self._actuator_indices]
        return self._origin_xy + np.array(
            [float(ctrl[0]), float(ctrl[1])], dtype=np.float64
        )

    def paddle_xy(self) -> np.ndarray:
        return self._origin_xy + np.array(
            [
                float(self._env.data.qpos[self._qposadrs[0]]),
                float(self._env.data.qpos[self._qposadrs[1]]),
            ],
            dtype=np.float64,
        )

    def ball_xy(self) -> np.ndarray:
        return np.asarray(
            self._env._ball_position()[:2], dtype=np.float64
        ).copy()


def _close_window(
    end: str,
    hit_xy: np.ndarray,
    home_xy: np.ndarray,
    cmds: list[np.ndarray],
    pads: list[np.ndarray],
    balls: list[np.ndarray],
    sats: list[bool],
) -> WindowRecord | None:
    """Aggregate one window's post-follow-through series."""
    start = FOLLOW_THROUGH_STEPS
    if len(cmds) <= start + 1:
        return None
    cmd = np.asarray(cmds[start:])
    pad = np.asarray(pads[start:])
    ball = np.asarray(balls[start:])
    sat = np.asarray(sats[start:], dtype=bool)
    dc = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
    da = np.linalg.norm(np.diff(pad, axis=0), axis=1)
    from_hit = np.linalg.norm(cmd - hit_xy[None, :], axis=1)
    return WindowRecord(
        end=end,
        steps=int(len(cmd)),
        cmd_travel=float(dc.sum()),
        act_travel=float(da.sum()),
        servo_gap=float(np.linalg.norm(cmd - pad, axis=1).mean()),
        cmd_from_hit_mean=float(from_hit.mean()),
        cmd_from_hit_max=float(from_hit.max()),
        cmd_to_ball=float(np.linalg.norm(cmd - ball, axis=1).mean()),
        cmd_to_home=float(np.linalg.norm(cmd - home_xy[None, :], axis=1).mean()),
        cmd_to_hit=float(from_hit.mean()),
        act_sat_frac=float(sat.mean()),
        cmd_step=float(dc.mean()),
    )


def run_probe(
    model_path: str | None,
    vec_normalize_path: str | None,
    episodes: int,
    seed_start: int,
    *,
    hold_shaping: float = 0.0,
    hold_shaping_travel: float = 4.0,
) -> list[WindowRecord]:
    policy: Callable[[np.ndarray], np.ndarray]
    if model_path is None:
        from courtside_dynamics.envs._paddle_court import (
            scripted_ground_opponent,
        )

        policy = scripted_ground_opponent
    else:
        if vec_normalize_path is None:
            raise ValueError("vec_normalize_path is required alongside model_path")
        policy = native_checkpoint_policy(model_path, vec_normalize_path)
    env = PaddleTennisEnv(
        episode_len=1500,
        court_style="diagnostic",
        volley_rule="fault",
        contact_shaping=0.25,
        reach_shaping=0.25,
        reach_shaping_radius=3.0,
        points_per_episode=None,
        hold_shaping=hold_shaping,
        hold_shaping_travel=hold_shaping_travel,
    )
    windows: list[WindowRecord] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            frames = _SideAFrames(env)
            open_window: dict | None = None
            last_points = 0
            while True:
                action = policy(obs)
                obs, _, term, trunc, info = env.step(action)
                cmd = frames.cmd_xy()
                pad = frames.paddle_xy()
                ball = frames.ball_xy()
                saturated = bool(np.max(np.abs(action[:2])) > 0.9)
                points = int(info["points_played"])
                boundary = points != last_points
                last_points = points
                ended = bool(term or trunc)

                if open_window is not None:
                    w = open_window
                    if bool(info["event_valid_racket_hit_b"]):
                        end = "opponent_strike"
                    elif boundary:
                        end = "boundary"
                    elif ended:
                        end = "episode_end"
                    elif len(w["cmds"]) >= WINDOW_CAP_STEPS:
                        end = "cap"
                    else:
                        end = None
                    if end is None:
                        w["cmds"].append(cmd)
                        w["pads"].append(pad)
                        w["balls"].append(ball)
                        w["sats"].append(saturated)
                    else:
                        record = _close_window(
                            end,
                            w["hit_xy"],
                            frames.home_xy,
                            w["cmds"],
                            w["pads"],
                            w["balls"],
                            w["sats"],
                        )
                        if record is not None:
                            windows.append(record)
                        open_window = None

                if bool(info["event_valid_racket_hit_a"]) and not ended:
                    open_window = {
                        "hit_xy": pad.copy(),
                        "cmds": [cmd],
                        "pads": [pad],
                        "balls": [ball],
                        "sats": [saturated],
                    }
                if ended:
                    break
        return windows
    finally:
        env.close()


def _summarize(name: str, group: list[WindowRecord]) -> dict:
    if not group:
        return {"group": name, "windows": 0}
    def mean(key: str) -> float:
        return float(np.mean([getattr(w, key) for w in group]))
    return {
        "group": name,
        "windows": len(group),
        "steps_mean": mean("steps"),
        "cmd_travel_mean": mean("cmd_travel"),
        "act_travel_mean": mean("act_travel"),
        "servo_gap_mean": mean("servo_gap"),
        "cmd_from_hit_mean": mean("cmd_from_hit_mean"),
        "cmd_from_hit_max_mean": mean("cmd_from_hit_max"),
        "cmd_to_ball_mean": mean("cmd_to_ball"),
        "cmd_to_home_mean": mean("cmd_to_home"),
        "act_sat_frac_mean": mean("act_sat_frac"),
        "cmd_step_mean": mean("cmd_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vec-normalize", default=None)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="probe the scripted ground oracle instead of a checkpoint "
        "(the calibration row; --model must be omitted)",
    )
    parser.add_argument("--episodes", type=int, default=PROBE_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PROBE_SEED_START)
    parser.add_argument("--label", default="checkpoint")
    parser.add_argument("--hold-shaping", type=float, default=0.0)
    parser.add_argument("--hold-shaping-travel", type=float, default=4.0)
    parser.add_argument("--json", default=None, help="write summaries to this path")
    args = parser.parse_args()
    if args.oracle == (args.model is not None):
        raise SystemExit("pass exactly one of --oracle or --model/--vec-normalize")
    if args.model is not None and args.vec_normalize is None:
        raise SystemExit("--model requires --vec-normalize")
    _refuse_reserved(args.seed_start, args.episodes)

    windows = run_probe(
        args.model,
        args.vec_normalize,
        args.episodes,
        args.seed_start,
        hold_shaping=args.hold_shaping,
        hold_shaping_travel=args.hold_shaping_travel,
    )
    groups = {
        "all": windows,
        "hold(opponent_strike)": [w for w in windows if w.end == "opponent_strike"],
        "boundary": [w for w in windows if w.end == "boundary"],
        "episode_end/cap": [w for w in windows if w.end in ("episode_end", "cap")],
    }
    summaries = [_summarize(name, group) for name, group in groups.items()]

    print(
        f"PT1 post-swing action targets — {args.label} "
        f"(episodes={args.episodes}, seeds {args.seed_start}+)"
    )
    header = (
        f"  {'group':<22} {'n':>4} {'steps':>6} {'cmd trv':>8} {'act trv':>8} "
        f"{'servo':>6} {'|cmd-hit|':>9} {'max':>6} {'to ball':>8} "
        f"{'to home':>8} {'sat%':>5} {'step':>6}"
    )
    print(header)
    for s in summaries:
        if s["windows"] == 0:
            print(f"  {s['group']:<22} {0:>4}")
            continue
        print(
            f"  {s['group']:<22} {s['windows']:>4d} {s['steps_mean']:>6.1f} "
            f"{s['cmd_travel_mean']:>8.2f} {s['act_travel_mean']:>8.2f} "
            f"{s['servo_gap_mean']:>6.2f} {s['cmd_from_hit_mean']:>9.2f} "
            f"{s['cmd_from_hit_max_mean']:>6.2f} {s['cmd_to_ball_mean']:>8.2f} "
            f"{s['cmd_to_home_mean']:>8.2f} {100 * s['act_sat_frac_mean']:>5.1f} "
            f"{s['cmd_step_mean']:>6.3f}"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"label": args.label, "summaries": summaries}, handle, indent=2)
        print(f"  summaries written to {args.json}")


if __name__ == "__main__":
    main()
