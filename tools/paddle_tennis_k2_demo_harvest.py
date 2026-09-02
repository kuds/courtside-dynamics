"""k=2 demonstration harvest — oracle completions from the policy's own
failure states (LD1′ decision D-E, docs/design_paddle_tennis_demo_injection.md).

For every entry of one or more k=2 failure-state libraries
(``k2-drill-library-v0``, the states the trained policy actually
faces and fails), launch the state through the shipped full-context
drill arm (real rally flags, real physics), let the scripted ground
oracle play side A, and record the SB3 replay tuple at every control
step until the point ends (or a step cap; cap-ended trajectories are
kept, counted, and end on a non-terminal row). A trajectory is KEPT only
if the oracle converted the k=2 ball AND the conversion was confirmed
inside the recording — the +return_reward is asserted present at the
confirmation step (fail-loud, cardinal rule 1), so "the demos carry
the success signal" is true by construction. Rejected trajectories
are counted, never silently dropped.

Every kept trajectory carries a train/held-out split (every
``--holdout-every``-th source entry is held out; held-out states are
the D-C ordering-metric material and never enter the demo buffer).

Two launch details keep the demos on the policy's own distribution:
the failure state's episode clock is restored after the launch (so
``episode_remaining_fraction`` reads what the policy saw, not a
fresh episode's), and the drill is switched OFF after the library
loads, so the point-boundary relaunch inside a kept row is a plain
drawn serve exactly as the drill-off pilot env would produce.

Schema ``k2-demo-library-v0``: a pickle with header provenance
(sources with sha256, git sha, env kwargs, oracle name, counts) and
``trajectories`` = list of dicts with float64 arrays ``obs``
(T×48), ``actions`` (T×3), ``next_obs`` (T×48), ``rewards`` (T),
``terminated``/``truncated`` (T, bool), plus ``hit_step``,
``confirm_step``, ``ender``, ``split``, ``source``, ``entry``.

Usage::

    MUJOCO_GL=disable python tools/paddle_tennis_k2_demo_harvest.py \
        --library k2_library_registered.pkl [--library more.pkl ...] \
        --out k2_demos.pkl [--max-steps 300] [--holdout-every 5]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import subprocess
from typing import Any

import numpy as np

from courtside_dynamics.envs._paddle_court import scripted_ground_opponent
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import CourtSide

SCHEMA = "k2-demo-library-v0"
ORACLE = "scripted_ground_opponent"
#: The step-0 tool's discarded-draw reset convention: the reset's
#: serve draw is overwritten by the launch, so the seed carries no
#: statistical content and is reused by every launch instrument.
RESET_SEED = 9147

#: The standing training-recipe env shape (the demos' reward stream
#: must be the stream the pilot trains on).
ENV_KWARGS: dict[str, Any] = dict(
    episode_len=1500,
    court_style="diagnostic",
    volley_rule="fault",
    contact_shaping=0.25,
    reach_shaping=0.25,
    reach_shaping_radius=3.0,
    points_per_episode=None,
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
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout
        if not status.strip():
            return sha
        # A dirty tree is identified by the CONTENT of its
        # modifications, so the start/end comparison in main() also
        # catches a dirty tree whose edits changed during the run.
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout
        tag = hashlib.sha256((status + diff).encode()).hexdigest()[:8]
        return f"{sha}-dirty-{tag}"
    except Exception:
        return "unknown"


def _empty_counts() -> dict[str, int]:
    return {
        "launched": 0,
        "kept": 0,
        "oracle_miss": 0,
        "unconfirmed": 0,
        "kept_point_ended": 0,
        "kept_cap_ended": 0,
        "kept_episode_truncated": 0,
    }


def harvest_library(
    library_path: str,
    *,
    max_steps: int,
    holdout_every: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Harvest one source library; returns (trajectories, its own counts)."""
    counts = _empty_counts()
    env = PaddleTennisEnv(
        **ENV_KWARGS,
        drill_library=library_path,
        drill_fraction=1.0,
        drill_context="full",
    )
    # The drill kwargs above are only the loader; the pilot trains with
    # the drill OFF (D-F), so the point-boundary relaunch recorded in a
    # kept row's next_obs must be a plain drawn serve, not another drill.
    env.drill_fraction = 0.0
    source_sha = _sha256(library_path)
    trajectories: list[dict[str, Any]] = []
    entries = env._drill_entries
    assert entries is not None  # the drill kwargs above loaded the library
    try:
        for index, entry in enumerate(entries):
            env.reset(seed=RESET_SEED)
            env._serving_side = CourtSide.B
            # The relaunch after the demo point serves from the
            # harvested rally's next server (the step-0 tool's full
            # arm restores the same field), not from reset()'s
            # alternation parity.
            env._next_serving_side = entry["next_serving_side"]
            env._launch_drill(entry, index)
            # The launch never touches the episode clock; restore the
            # harvested one so the demo observations carry the failure
            # state's own episode_remaining_fraction (and truncate where
            # that episode would have).
            env.step_number = int(entry["step_number"])
            obs = env._get_obs()
            env._remember_finite_observation(obs)
            obs_list, act_list, next_list = [], [], []
            rew_list, term_list, trunc_list = [], [], []
            hit_step = confirm_step = None
            ender = None
            for step in range(max_steps):
                action = scripted_ground_opponent(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                obs_list.append(obs.copy())
                act_list.append(np.asarray(action, dtype=np.float64).copy())
                next_list.append(next_obs.copy())
                rew_list.append(float(reward))
                term_list.append(bool(term))
                trunc_list.append(bool(trunc))
                if hit_step is None and bool(info["event_valid_racket_hit_a"]):
                    hit_step = step
                if (
                    hit_step is not None
                    and confirm_step is None
                    and bool(info["event_valid_return_a"])
                ):
                    confirm_step = step
                obs = next_obs
                boundary = int(info["points_played"]) > 0
                if term or trunc or boundary:
                    if boundary:
                        # This row's next_obs is the relaunch: it must
                        # be the drill-off pilot env's plain serve.
                        assert not env._drill_point_active, (
                            f"entry {index}: boundary relaunch drilled"
                        )
                    name = str(info["termination_reason_name"])
                    if boundary and not (term or trunc):
                        ender = f"point_boundary/{name}"
                    elif trunc and not term:
                        ender = "episode_truncated"
                    else:
                        ender = name
                    break
            if ender is None:
                ender = "cap_ended"
            counts["launched"] += 1
            if hit_step is None:
                counts["oracle_miss"] += 1
                continue
            if confirm_step is None:
                counts["unconfirmed"] += 1
                continue
            rewards = np.asarray(rew_list, dtype=np.float64)
            # The conversion payment must be IN the recorded stream:
            # the confirmation step pays +return_reward (cooperative
            # +1 per confirmed return), never less.
            assert rewards[confirm_step] >= env.return_reward, (
                f"entry {index}: confirmation step {confirm_step} paid "
                f"{rewards[confirm_step]}, below return_reward "
                f"{env.return_reward}"
            )
            trajectories.append(
                {
                    "source": source_sha,
                    "entry": int(index),
                    "split": "heldout" if index % holdout_every == 0 else "train",
                    "obs": np.asarray(obs_list, dtype=np.float64),
                    "actions": np.asarray(act_list, dtype=np.float64),
                    "next_obs": np.asarray(next_list, dtype=np.float64),
                    "rewards": rewards,
                    "terminated": np.asarray(term_list, dtype=bool),
                    "truncated": np.asarray(trunc_list, dtype=bool),
                    "hit_step": int(hit_step),
                    "confirm_step": int(confirm_step),
                    "ender": ender,
                }
            )
            counts["kept"] += 1
            if ender == "cap_ended":
                counts["kept_cap_ended"] += 1
            elif ender == "episode_truncated":
                counts["kept_episode_truncated"] += 1
            else:
                counts["kept_point_ended"] += 1
    finally:
        env.close()
    return trajectories, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--library", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--holdout-every", type=int, default=5)
    args = parser.parse_args()
    if args.holdout_every < 2:
        raise SystemExit("--holdout-every must be >= 2 (some entries must train)")

    # Provenance is the tree the harvest STARTED on; a tree that
    # changes underneath a running harvest is refused at the end
    # rather than recorded as if it had been stable.
    git_sha = _git_sha()
    counts = _empty_counts()
    trajectories: list[dict[str, Any]] = []
    sources = []
    for path in args.library:
        got, local = harvest_library(
            path,
            max_steps=args.max_steps,
            holdout_every=args.holdout_every,
        )
        for key, value in local.items():
            counts[key] += value
        trajectories.extend(got)
        sources.append(
            {
                "file": os.path.basename(path),
                "sha256": _sha256(path),
                "kept": len(got),
                "counts": local,
            }
        )
        print(f"{path}: {local}")
    if _git_sha() != git_sha:
        raise SystemExit(
            f"working tree changed during the harvest ({git_sha} -> "
            f"{_git_sha()}); re-run on a stable tree"
        )
    library = {
        "schema": SCHEMA,
        "git_sha": git_sha,
        "oracle": ORACLE,
        "env_kwargs": dict(ENV_KWARGS),
        "reset_seed": RESET_SEED,
        "max_steps": args.max_steps,
        "holdout_every": args.holdout_every,
        "sources": sources,
        "counts": counts,
        "trajectories": trajectories,
    }
    with open(args.out, "wb") as f:
        pickle.dump(library, f)
    n_train = sum(t["split"] == "train" for t in trajectories)
    steps = sum(len(t["rewards"]) for t in trajectories)
    print(
        f"demo library written: {args.out}  ({len(trajectories)} trajectories, "
        f"{n_train} train / {len(trajectories) - n_train} held-out, "
        f"{steps} transitions; counts {counts}; sha {_sha256(args.out)[:12]})"
    )


if __name__ == "__main__":
    main()
