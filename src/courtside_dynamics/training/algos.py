"""Canonical SB3 algorithm registry shared across the training code.

A single source of truth for "which ``algo`` string maps to which SB3
class" keeps :mod:`courtside_dynamics.training.train` and the notebook
replay helpers from drifting -- they previously each carried their own
``{"SAC": SAC, "PPO": PPO}`` literal.
"""
from __future__ import annotations

import difflib
import inspect
from collections.abc import Mapping
from typing import Any

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

#: Maps the ``algo`` string used throughout the project to its SB3 class.
ALGOS: dict[str, type[BaseAlgorithm]] = {"SAC": SAC, "PPO": PPO}

#: Algorithms that learn off-policy from a replay buffer. For these the
#: number of gradient updates per rollout is decoupled from the number of
#: env steps collected, so a vectorised env silently starves the policy of
#: updates unless we compensate (see ``train._build_algo``).
OFF_POLICY_ALGOS = frozenset({"SAC"})


#: Constructor arguments the trainer supplies itself (``train._build_algo``
#: passes ``policy`` and ``env`` positionally and ``tensorboard_log``
#: explicitly), so a recipe/config value for them would collide at
#: construction with a confusing ``TypeError``.
_RESERVED_MODEL_KWARGS = frozenset({"policy", "env", "tensorboard_log"})


def validate_model_kwargs(
    algo_name: str, model_kwargs: Mapping[str, Any]
) -> None:
    """Reject ``model_kwargs`` the resolved algorithm cannot accept.

    Without this, a cross-algorithm key (``buffer_size`` on PPO, say)
    only raises inside the SB3 constructor -- after the vectorised env
    fleet is built and run artifacts are written -- and a string
    ``ent_coef`` such as ``"auto_0.02"`` even *constructs* a PPO model
    successfully, crashing at the first gradient update, a full
    ``n_steps * n_envs`` env steps into the run. Both violate the
    fail-loud-and-early rule the env-kwarg probe already enforces.
    """
    cls = resolve_algo(algo_name)
    parameters = inspect.signature(cls.__init__).parameters
    reserved = sorted(_RESERVED_MODEL_KWARGS & set(model_kwargs))
    if reserved:
        raise ValueError(
            f"model_kwargs may not set {reserved}: the trainer supplies "
            f"these to {cls.__name__} itself"
        )
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        accepted = set(parameters) - {"self"} - _RESERVED_MODEL_KWARGS
        for key in model_kwargs:
            if key in accepted:
                continue
            suggestions = difflib.get_close_matches(key, sorted(accepted), n=1)
            hint = f"; did you mean '{suggestions[0]}'?" if suggestions else ""
            raise ValueError(
                f"model_kwargs key '{key}' is not accepted by "
                f"{cls.__name__}{hint}"
            )
    if algo_name.upper() not in OFF_POLICY_ALGOS and "ent_coef" in model_kwargs:
        ent_coef = model_kwargs["ent_coef"]
        if isinstance(ent_coef, bool) or not isinstance(ent_coef, (int, float)):
            raise ValueError(
                f"{cls.__name__} requires a numeric ent_coef, got "
                f"{ent_coef!r}: the 'auto'/'auto_<value>' strings are "
                f"SAC-only, and a non-numeric value would not fail until "
                f"the first gradient update"
            )


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
