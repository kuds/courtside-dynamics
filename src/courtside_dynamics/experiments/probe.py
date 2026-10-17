"""Probe CLI: score a scripted witness or a saved model under a protocol.

``docs/design_campaign_structure_20261016.md`` step L2 ("a generic probe
CLI") and section 3's probe step: the oracle, random and camper
witnesses, and any snapshot, measured by the same evaluation code as
the in-loop evaluator, with no bespoke probe tool::

    python -m courtside_dynamics.experiments.probe --recipe PaddleTennis \\
        --policy oracle --protocol paddle-npoint@v2 --seed-start 5500 \\
        [--episodes 100] [--env points_per_episode=1 ...] [--json]
    python -m courtside_dynamics.experiments.probe --recipe PaddleTennis \\
        --policy run/snapshots/snapshot_25000_steps.zip \\
        --normalizer run/snapshots/snapshot_vecnormalize_25000_steps.pkl \\
        --protocol paddle-npoint@v3

``--policy`` is ``zero``, ``random`` (``np.random.default_rng`` of the
block's first seed), ``oracle`` or ``camper`` -- the golden digests'
frozen definitions (``digests.scripted_policy``), acting on raw
observations -- or a model zip, loaded through its run's recorded
algorithm (``--algo`` outside a run; outside a run it also needs
``--normalizer``, or ``--raw-observations`` for a model trained
without ``VecNormalize``: nothing else says which). ``--env KEY=VALUE`` (repeatable,
the value read as JSON when it parses) and ``--episodes`` derive an
unregistered protocol from ``--protocol``; its fingerprint is printed
with the metrics, so a probe result names exactly what it measured. A
protocol that scores on a run's own seed block needs ``--seed-start``
unless the model sits in a run directory.

The lever "audit mode" (a reward term logged, not paid) arrives with
the levers (design L6); a probe scores the task env as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from courtside_dynamics.experiments.protocols import (
    EvalProtocol,
    get_protocol,
    resolve_seed_block,
)

#: The scripted witnesses ``--policy`` accepts (``digests.POLICIES``).
SCRIPTED_POLICIES = ("zero", "random", "oracle", "camper")


class _ScriptedModel:
    """A scripted ``obs -> action`` policy behind SB3's ``predict`` signature.

    The evaluation VecEnv has one worker: worker 0's raw observation
    goes in, one action comes out (no clipping, no normalizer).
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        action = np.asarray(self.policy(np.asarray(obs)[0]))
        return action[None, ...], None


def score_scripted(
    policy: str,
    protocol: EvalProtocol,
    *,
    seed_start: int | None = None,
    block: str = "standard",
) -> dict[str, float]:
    """A scripted witness's metrics row under ``protocol``.

    The env is built as the harness builds it (one ``make_vec_env``
    worker), without a normalizer: the witnesses read raw observations.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    from courtside_dynamics.callbacks.info_dict_eval import collect_episode_metrics
    from courtside_dynamics.digests import scripted_policy

    if policy not in SCRIPTED_POLICIES:
        raise ValueError(f"policy must be one of {SCRIPTED_POLICIES}, got {policy!r}")
    seeds = resolve_seed_block(protocol, seed_start=seed_start)
    start = seeds.for_block(protocol, block)
    venv = make_vec_env(protocol.make_env, n_envs=1, seed=start)
    try:
        assert isinstance(venv, DummyVecEnv)
        model = _ScriptedModel(scripted_policy(policy, venv.envs[0], start))
        batch = collect_episode_metrics(
            model,
            venv,
            seed_base=start if protocol.paired else None,
            **protocol.measurement_kwargs(),
        )
    finally:
        venv.close()
    return batch.metrics


def _parse_env_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--env takes KEY=VALUE, got {pair!r}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key] = value
    return overrides


def probe_protocol(
    base: EvalProtocol,
    *,
    env_overrides: Mapping[str, Any] | None = None,
    episodes: int | None = None,
) -> EvalProtocol:
    """``base``, or the unregistered protocol ``--env`` / ``--episodes`` make."""
    if not env_overrides and episodes is None:
        return base
    parts = [f"{key}={json.dumps(value)}" for key, value in sorted((env_overrides or {}).items())]
    if episodes is not None:
        parts.append(f"episodes={episodes}")
    return base.derive(
        f"{base.id}+probe({','.join(parts)})".replace(" ", ""),
        env_overrides=env_overrides,
        description=f"{base.id} with probe overrides {', '.join(parts)}",
        **({"n_episodes": episodes} if episodes is not None else {}),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m courtside_dynamics.experiments.probe",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--recipe", required=True, help="the recipe whose env is probed")
    parser.add_argument(
        "--policy",
        required=True,
        help=f"one of {', '.join(SCRIPTED_POLICIES)}, or a model .zip",
    )
    parser.add_argument("--normalizer", help="the model's VecNormalize .pkl")
    parser.add_argument("--algo", help="the model's algorithm, for a model outside a run")
    parser.add_argument(
        "--raw-observations",
        action="store_true",
        help="score a model outside a run without a normalizer (it trained "
        "without VecNormalize)",
    )
    parser.add_argument("--protocol", required=True, help="a registered protocol id")
    parser.add_argument("--episodes", type=int, help="episodes (derives a probe protocol)")
    parser.add_argument("--seed-start", type=int, help="the block's first seed")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an eval-env kwarg (derives a probe protocol)",
    )
    parser.add_argument(
        "--block", choices=("standard", "confirm"), default="standard"
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)

    from courtside_dynamics.recipes import RECIPES

    try:
        if args.recipe not in RECIPES:
            raise ValueError(f"unknown recipe {args.recipe!r}; recipes: {sorted(RECIPES)}")
        base = get_protocol(args.protocol)
        if RECIPES[args.recipe].env_cls is not base.env_cls:
            raise ValueError(
                f"protocol {base.id} evaluates {base.env_cls.__name__}, not "
                f"{args.recipe}'s {RECIPES[args.recipe].env_cls.__name__}"
            )
        protocol = probe_protocol(
            base,
            env_overrides=_parse_env_overrides(args.env),
            episodes=args.episodes,
        )
        if args.policy in SCRIPTED_POLICIES:
            if args.normalizer is not None or args.raw_observations:
                raise ValueError(
                    "a scripted policy reads raw observations: drop "
                    "--normalizer and --raw-observations"
                )
            metrics = score_scripted(
                args.policy, protocol, seed_start=args.seed_start, block=args.block
            )
            seed_start = resolve_seed_block(protocol, seed_start=args.seed_start).start
        else:
            from courtside_dynamics.experiments.harness import score_model

            scored = score_model(
                args.policy,
                args.normalizer,
                protocol,
                algo=args.algo,
                blocks=(args.block,),
                seed_start=args.seed_start,
                raw_observations=args.raw_observations,
            )
            metrics = dict(scored.metrics[args.block])
            seed_start = scored.seed_block.start
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"probe: {error}", file=sys.stderr)
        return 2
    report = {
        "recipe": args.recipe,
        "policy": args.policy,
        "protocol": protocol.id,
        "protocol_fingerprint": protocol.fingerprint,
        "episodes": protocol.n_episodes,
        "block": args.block,
        "seed_start": seed_start,
        "metrics": metrics,
    }
    if args.json:
        print(json.dumps(report, sort_keys=True))
        return 0
    print(f"protocol {protocol.id}")
    print(f"  fingerprint {protocol.fingerprint}")
    print(
        f"policy {args.policy}; {protocol.n_episodes} episodes, {args.block} "
        f"block from seed {seed_start}"
    )
    width = max(len(name) for name in metrics)
    for name in sorted(metrics):
        print(f"  {name:<{width}}  {metrics[name]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
