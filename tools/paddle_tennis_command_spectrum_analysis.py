"""PT2 — command-spectrum and k=2-geometry analysis of captured streams.

Analyzes the per-step command streams captured for
``docs/paddle_tennis_command_spectrum_20260830.md`` (PT2): for each
subject's ``<name>_streams.npz`` + ``<name>_events.json`` pair this
computes, inside PT1-convention post-swing windows,

- the per-step command-delta distribution (F1: jump-then-dwell);
- an open-loop simulation of the command-rate design's D1 limiter at
  several deltas, with a per-axis plant-rate follower on both arms
  (F2 — see the snapshot's honest instrument limit: the proxy cannot
  discriminate at delta >= the plant rate);
- the oracle-tail percentiles (F4);
- the k=2 opportunity geometry with the frozen-at-hit / hit+30 /
  parked-at-home counterfactual anchors (F3).

This is an analysis tool over recorded data, not an env probe: it
burns no seeds itself. The capture convention (subjects, seeds
5200-5299, env shape, recorded keys) is documented in the snapshot's
section 1.

Usage::

    python tools/paddle_tennis_command_spectrum_analysis.py \
        --streams-dir path/to/pt1_ctrl_streams_20260829 \
        [--subjects registered_2p4M oracle] [--deltas 0.03 0.05 0.1 0.15 0.2]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

FOLLOW_THROUGH_STEPS = 30
WINDOW_CAP_STEPS = 300
PLANT_RATE = 0.125  # m/step per-axis terminal-velocity ceiling (design section 3)
DEFAULT_DELTAS = (0.03, 0.05, 0.10, 0.15, 0.20)
DEFAULT_SUBJECTS = ("registered_2p4M", "lh1c_2p325M", "lt1_1M", "oracle")


def _episodes(npz):
    seeds = sorted(
        {
            int(k.split("_")[0][2:])
            for k in npz.files
            if k.startswith("ep") and k.split("_")[0][2:].isdigit()
        }
    )
    for s in seeds:
        yield s, {
            k: npz[f"ep{s}_{k}"]
            for k in (
                "actions",
                "ctrl",
                "qpos",
                "ball",
                "hit_a",
                "hit_b",
                "points",
                "origin_xy",
            )
        }


def _windows(ep):
    """Yield (end_kind, (open_step, close_step)) per the PT1 probe."""
    n = len(ep["ctrl"])
    open_start = None
    last_points = 0
    for t in range(n):
        points = int(ep["points"][t])
        boundary = points != last_points
        last_points = points
        ended = t == n - 1
        if open_start is not None:
            if bool(ep["hit_b"][t]):
                yield "opponent_strike", (open_start, t)
                open_start = None
            elif boundary:
                yield "boundary", (open_start, t)
                open_start = None
            elif ended:
                yield "episode_end", (open_start, t)
                open_start = None
            elif t - open_start >= WINDOW_CAP_STEPS:
                yield "cap", (open_start, t)
                open_start = None
        if bool(ep["hit_a"][t]) and not ended:
            open_start = t
        if ended:
            break


def _rate_limit(target, seed_pos, delta):
    """The design's D1 per-axis update rule, run over a whole episode."""
    out = np.empty_like(target, dtype=np.float64)
    eff = seed_pos.astype(np.float64).copy()
    for t in range(len(target)):
        raw = target[t].astype(np.float64)
        d = raw - eff
        eff = np.where(np.abs(d) <= delta, raw, eff + np.clip(d, -delta, delta))
        out[t] = eff
    return out


def _xy_path(a):
    return float(np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1).sum())


def analyze_subject(streams_dir, name, deltas):
    npz = np.load(os.path.join(streams_dir, f"{name}_streams.npz"))
    with open(os.path.join(streams_dir, f"{name}_events.json")) as f:
        events = json.load(f)
    eps = list(_episodes(npz))
    print(f"\n{'=' * 72}\nSUBJECT {name}  ({len(eps)} episodes)")

    step_deltas, strike_rows, strike_rows_30 = [], [], []
    off_actual, off_sim = [], []
    sim = {d: [] for d in deltas}
    drift = {d: [] for d in deltas}
    eff_axis_fast = eff_axis_total = 0  # |d eff|/axis >= plant rate at Delta=0.15
    for seed, ep in eps:
        ctrl, qpos, act = ep["ctrl"], ep["qpos"], ep["actions"]
        effs = {d: _rate_limit(ctrl, qpos[0], d) for d in deltas}
        f_raw = _rate_limit(ctrl, qpos[0], PLANT_RATE)
        f_eff = {d: _rate_limit(effs[d], qpos[0], PLANT_RATE) for d in deltas}
        for end, (a, b) in _windows(ep):
            s0, s1 = a + FOLLOW_THROUGH_STEPS, b + 1
            if s1 - a <= FOLLOW_THROUGH_STEPS + 1:
                continue
            c = ctrl[s0:s1, :2].astype(np.float64)
            dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
            step_deltas.extend(dc.tolist())
            if end == "opponent_strike":
                sat = float((np.abs(act[s0:s1, :2]).max(axis=1) > 0.9).mean())
                row = (sat, float(dc.mean()), float(dc.sum()),
                       _xy_path(qpos[s0:s1]))
                strike_rows.append(row)
                if seed < 5230:  # PT1's published seed subset
                    strike_rows_30.append(row)
            off_actual.append(_xy_path(qpos[s0:s1]))
            off_sim.append(_xy_path(f_raw[s0:s1]))
            if 0.15 in effs:
                de = np.abs(np.diff(effs[0.15][s0:s1, :2], axis=0))
                eff_axis_fast += int((de >= PLANT_RATE).sum())
                eff_axis_total += int(de.size)
            for d in deltas:
                sim[d].append(_xy_path(f_eff[d][s0:s1]))
                e = effs[d][s0:s1, :2]
                drift[d].append(float(np.linalg.norm(e[-1] - e[0])))

    for label, rows in (("strike-ended windows", strike_rows),
                        ("  PT1-subset (seeds 5200-5229)", strike_rows_30)):
        if rows:
            sats, css, cts, ats = zip(*rows)
            print(
                f"{label} {len(rows)}: sat {np.mean(sats):.1%}  "
                f"cmd_step {np.mean(css):.3f}  cmd_travel {np.mean(cts):.2f}  "
                f"act_travel {np.mean(ats):.2f}"
            )
    sd = np.asarray(step_deltas)
    print(
        f"per-step |dcmd| XY: p50 {np.percentile(sd, 50):.4f}  "
        f"p90 {np.percentile(sd, 90):.3f}  p99 {np.percentile(sd, 99):.3f}  "
        f"max {sd.max():.2f}  (<=0.01: {(sd <= 0.01).mean():.1%}  "
        f">1.0: {(sd > 1.0).mean():.1%})"
    )
    print(
        f"plant-proxy validation: sim/actual window travel = "
        f"{np.sum(off_sim) / max(np.sum(off_actual), 1e-9):.3f}"
    )
    if eff_axis_total:
        print(
            f"eff command at Delta=0.15: per-axis steps >= plant rate: "
            f"{eff_axis_fast / eff_axis_total:.1%}"
        )
    for d in deltas:
        ratio = np.mean(sim[d]) / max(np.mean(off_sim), 1e-9)
        print(
            f"  Delta={d:.2f}: ON/OFF sim travel {ratio:.0%}  "
            f"net drift/window {np.mean(drift[d]):.2f} m"
        )

    # ---- F3: k=2 geometry with counterfactual anchors ----
    stats = {k: [] for k in ("actual", "hit", "hit30", "home")}
    strict_hit = []  # frozen-at-hit, strict k=2 (first own hit of the point)
    opps = conv = 0
    for e_ep, (_, ep) in zip(events["episodes"], eps):
        qpos, pts = ep["qpos"], ep["points"]
        ha = np.flatnonzero(ep["hit_a"])
        hb = np.flatnonzero(ep["hit_b"])
        origin = ep["origin_xy"]
        bounces = [
            (t, x, y) for (t, k, x, y) in e_ep["events"] if k == "BALL_COURT_A"
        ]
        first_hit_of_point = {}
        for t0 in ha:
            first_hit_of_point.setdefault(int(pts[t0]), t0)
        for t0 in ha:
            nxt = hb[hb > t0]
            if not len(nxt) or pts[nxt[0]] != pts[t0]:
                continue
            t1 = nxt[0]
            opps += 1
            if len(ha[ha > t1]) and pts[ha[ha > t1][0]] == pts[t0]:
                conv += 1
            cand = [
                (t, x, y)
                for (t, x, y) in bounces
                if t > t1 and pts[min(t, len(pts) - 1)] == pts[t0]
            ]
            if not cand:
                continue
            t2, bx, by = cand[0]
            t2i = min(t2, len(qpos) - 1)
            t30 = min(t0 + FOLLOW_THROUGH_STEPS, len(qpos) - 1)
            for key, xy in (
                ("actual", origin + qpos[t2i, :2]),
                ("hit", origin + qpos[t0, :2]),
                ("hit30", origin + qpos[t30, :2]),
                ("home", origin),
            ):
                stats[key].append(float(np.hypot(xy[0] - bx, xy[1] - by)))
            if first_hit_of_point.get(int(pts[t0])) == t0:
                hx, hy = origin + qpos[t0, :2]
                strict_hit.append(float(np.hypot(hx - bx, hy - by)))
    print(f"k=2: opportunities {opps}  conversions {conv}")
    for key, label in (
        ("actual", "actual paddle @ bounce"),
        ("hit", "frozen-at-hit anchor  "),
        ("hit30", "frozen-at-hit+30 anchor"),
        ("home", "parked-at-home anchor "),
    ):
        if stats[key]:
            a = np.asarray(stats[key])
            print(
                f"  {label}: mean {a.mean():.2f} m  "
                f"<=1.0m {(a <= 1.0).mean():.1%}  (n={len(a)})"
            )
    if strict_hit:
        a = np.asarray(strict_hit)
        print(
            f"  frozen-at-hit, strict k=2 (first own hit of point): "
            f"mean {a.mean():.2f} m  <=1.0m {(a <= 1.0).mean():.1%}  (n={len(a)})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--streams-dir", required=True)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS)
    )
    args = parser.parse_args()
    for name in args.subjects:
        analyze_subject(args.streams_dir, name, tuple(args.deltas))


if __name__ == "__main__":
    main()
