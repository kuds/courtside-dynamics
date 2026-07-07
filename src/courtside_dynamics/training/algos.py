"""Canonical SB3 algorithm registry shared across the training code.

A single source of truth for "which ``algo`` string maps to which SB3
class" keeps :mod:`courtside_dynamics.training.train` and the notebook
replay helpers from drifting -- they previously each carried their own
``{"SAC": SAC, "PPO": PPO}`` literal.
"""
from __future__ import annotations

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

#: Maps the ``algo`` string used throughout the project to its SB3 class.
ALGOS: dict[str, type[BaseAlgorithm]] = {"SAC": SAC, "PPO": PPO}

#: Algorithms that learn off-policy from a replay buffer. For these the
#: number of gradient updates per rollout is decoupled from the number of
#: env steps collected, so a vectorised env silently starves the policy of
#: updates unless we compensate (see ``train._build_algo``).
OFF_POLICY_ALGOS = frozenset({"SAC"})


def resolve_algo(name: str) -> type[BaseAlgorithm]:
    """Look up an algorithm class by name, case-insensitively.

    The rest of the project compares ``algo`` strings with ``.upper()``
    (reward-normalization default, off-policy check, model loading), so
    the registry lookup must be equally forgiving -- otherwise
    ``algo="sac"`` would pass every other branch and still crash on the
    registry. Raises ``ValueError`` for unknown names so callers can
    validate the config before doing any expensive setup.
    """
    try:
        return ALGOS[name.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algo '{name}'. Expected one of {sorted(ALGOS)}."
        ) from exc
