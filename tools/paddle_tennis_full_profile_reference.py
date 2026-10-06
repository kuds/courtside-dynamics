"""Record the PaddleTennis full-layout reference streams (b9585e2).

The ``"full"`` observation profile -- the default -- is promised
bit-identical to the pre-profile env (b9585e2): observation, reward,
terminated, truncated and info, for every existing kwarg combination
(``CHANGELOG.md``, ``docs/paddle_tennis_physical_pilot_20261005.md``).
A default-vs-explicit ``"full"`` comparison cannot hold that promise:
both envs run the same code, so the layout itself could drift and the
comparison would still pass. This tool records the streams **from
b9585e2's own source**, and
``tests/test_paddle_tennis.py::TestFullProfileReference`` replays them
on the current tree -- the default env, an explicit ``"full"`` env, and
(for everything except the policy observation) a ``"physical"`` env.

Each case resets with a seed, plays ``steps`` side-A actions (seeded
uniform actions, or the ground oracle reading
``observation_for_side(A)``), resets unseeded at every episode end and
once mid-case, and folds every reset and step into one chained sha256
per field:

- ``obs``: the policy observation reset/step return;
- ``view_a`` / ``view_b``: ``observation_for_side(A)`` / ``(B)`` (the
  full side-local layout the opponent, oracles and instruments read);
- ``reward`` (exact, ``float.hex``), ``terminated``, ``truncated``;
- ``info``: canonical JSON of the reset and step info dicts.

A checkpoint digest of every chain is kept every ``checkpoint_every``
steps, so a failure names the first window that diverged.

The file pins the MuJoCo version and machine it was recorded on, and
the test skips elsewhere: physics bits are comparable only on the same
MuJoCo build. To re-record (e.g. after a MuJoCo upgrade), run this
script against b9585e2's source, never the current tree::

    git worktree add /tmp/cd-b9585e2 b9585e2
    PYTHONPATH=/tmp/cd-b9585e2/src python \\
        tools/paddle_tennis_full_profile_reference.py \\
        --recorded-from b9585e2 \\
        --out tests/data/paddle_tennis_full_profile_b9585e2.json
    git worktree remove /tmp/cd-b9585e2

The script imports only APIs b9585e2 already had (``PaddleTennisEnv``,
``observation_for_side``, ``scripted_ground_opponent``), and refuses to
record from a tree whose env has ``observation_profile`` unless
``--allow-profile-tree`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

SCHEMA = "paddle-tennis-full-profile-reference-v1"
CHECKPOINT_EVERY = 25
FIELDS = (
    "obs",
    "view_a",
    "view_b",
    "reward",
    "terminated",
    "truncated",
    "info",
)

_RECIPE_KWARGS: dict[str, Any] = {
    "points_per_episode": None,
    "contact_shaping": 0.25,
    "reach_shaping": 0.25,
}

#: name -> (env kwargs, side-A player, reset seed, steps). Kwargs cover
#: the frozen one-point default, the adopted PaddleTennis recipe's
#: n-point escrow task, the volley-legal profile, and a short-episode
#: n-point variant with the hold escrow on, so truncation (and its
#: escrow clawback) lands inside the recorded window.
CASES: dict[str, tuple[dict[str, Any], str, int, int]] = {
    "default_random": ({}, "random", 1000, 400),
    "default_oracle": ({}, "oracle", 1001, 400),
    "recipe_random": (dict(_RECIPE_KWARGS), "random", 1002, 400),
    "recipe_oracle": (dict(_RECIPE_KWARGS), "oracle", 1000, 400),
    "volley_legal_random": ({"volley_rule": "legal"}, "random", 1001, 400),
    "volley_legal_oracle": ({"volley_rule": "legal"}, "oracle", 1002, 400),
    "recipe_short_hold_oracle": (
        {**_RECIPE_KWARGS, "hold_shaping": 0.5, "episode_len": 140},
        "oracle",
        1003,
        300,
    ),
}


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def info_bytes(info: Mapping[str, Any]) -> bytes:
    """One exact encoding of an info dict (floats via their shortest
    round-trip repr, keys sorted, numpy values unwrapped)."""
    return json.dumps(_canonical(info), sort_keys=True).encode()


def array_bytes(array: np.ndarray) -> bytes:
    array = np.asarray(array)
    header = f"{array.dtype.str}{array.shape}|".encode()
    return header + np.ascontiguousarray(array).tobytes()


class _Chains:
    def __init__(self) -> None:
        self._hashes = {field: hashlib.sha256() for field in FIELDS}
        self.checkpoints: dict[str, list[str]] = {field: [] for field in FIELDS}

    def feed(self, field: str, payload: bytes) -> None:
        self._hashes[field].update(len(payload).to_bytes(8, "little"))
        self._hashes[field].update(payload)

    def checkpoint(self) -> None:
        for field, digest in self._hashes.items():
            self.checkpoints[field].append(digest.hexdigest()[:16])


def record_case(
    env_factory: Callable[[dict[str, Any]], Any],
    env_kwargs: Mapping[str, Any],
    player: str,
    seed: int,
    steps: int,
    *,
    fields: tuple[str, ...] = FIELDS,
) -> dict[str, Any]:
    """Play one case and return its checkpointed chain digests.

    ``fields`` limits which chains are fed (the others stay at the
    empty digest), so a ``"physical"`` replay can skip ``obs``.
    """
    from courtside_dynamics.envs._paddle_court import scripted_ground_opponent
    from courtside_dynamics.envs.tennis_rules import CourtSide

    if player not in ("random", "oracle"):
        raise ValueError(f"player must be random|oracle, got {player!r}")
    chains = _Chains()
    counts = {"resets": 0, "terminations": 0, "truncations": 0}
    rng = np.random.default_rng(seed)
    env = env_factory(dict(env_kwargs))

    def feed_views() -> None:
        if "view_a" in fields:
            chains.feed("view_a", array_bytes(env.observation_for_side(CourtSide.A)))
        if "view_b" in fields:
            chains.feed("view_b", array_bytes(env.observation_for_side(CourtSide.B)))

    def do_reset(reset_seed: int | None) -> None:
        obs, info = env.reset(seed=reset_seed)
        counts["resets"] += 1
        if "obs" in fields:
            chains.feed("obs", b"reset|" + array_bytes(obs))
        if "info" in fields:
            chains.feed("info", b"reset|" + info_bytes(info))
        feed_views()

    try:
        do_reset(seed)
        for step in range(1, steps + 1):
            if player == "random":
                action = rng.uniform(-1.0, 1.0, size=3)
            else:
                action = scripted_ground_opponent(
                    env.observation_for_side(CourtSide.A)
                )
            obs, reward, terminated, truncated, info = env.step(action)
            if "obs" in fields:
                chains.feed("obs", array_bytes(obs))
            if "reward" in fields:
                chains.feed("reward", float(reward).hex().encode())
            if "terminated" in fields:
                chains.feed("terminated", b"1" if terminated else b"0")
            if "truncated" in fields:
                chains.feed("truncated", b"1" if truncated else b"0")
            if "info" in fields:
                chains.feed("info", info_bytes(info))
            feed_views()
            counts["terminations"] += int(bool(terminated))
            counts["truncations"] += int(bool(truncated))
            if step % CHECKPOINT_EVERY == 0 or step == steps:
                chains.checkpoint()
            if step == steps:
                break
            if terminated or truncated or step == steps // 2:
                do_reset(None)
    finally:
        env.close()
    return {"checkpoints": chains.checkpoints, **counts}


def record_reference(
    env_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    import mujoco

    from courtside_dynamics.envs import PaddleTennisEnv

    factory = env_factory or (lambda kwargs: PaddleTennisEnv(**kwargs))
    cases: dict[str, Any] = {}
    for name, (env_kwargs, player, seed, steps) in CASES.items():
        cases[name] = {
            "env_kwargs": env_kwargs,
            "player": player,
            "seed": seed,
            "steps": steps,
            **record_case(factory, env_kwargs, player, seed, steps),
        }
    return {
        "schema": SCHEMA,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "checkpoint_every": CHECKPOINT_EVERY,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--recorded-from",
        required=True,
        help="the commit whose source PYTHONPATH points at (b9585e2)",
    )
    parser.add_argument(
        "--allow-profile-tree",
        action="store_true",
        help="record from a tree that already has observation_profile",
    )
    args = parser.parse_args(argv)

    import courtside_dynamics
    from courtside_dynamics.envs import PaddleTennisEnv

    has_profile = (
        "observation_profile" in inspect.signature(PaddleTennisEnv).parameters
    )
    if has_profile and not args.allow_profile_tree:
        parser.error(
            "this tree's PaddleTennisEnv already has observation_profile; "
            "point PYTHONPATH at b9585e2's src (module docstring) or pass "
            "--allow-profile-tree"
        )
    reference = {"recorded_from": args.recorded_from, **record_reference()}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"recorded from {courtside_dynamics.__file__} -> {args.out}")
    for name, case in reference["cases"].items():
        print(
            f"  {name}: resets {case['resets']}, terminations "
            f"{case['terminations']}, truncations {case['truncations']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
