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
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from courtside_dynamics.training.demo_sac import DemoSAC

#: Maps the ``algo`` string used throughout the project to its SB3 class.
#: ``DemoSAC`` is SAC plus default-off demonstration injection
#: (docs/design_paddle_tennis_demo_injection.md) — a distinct name so a
#: run's algo string alone says whether the injection surface was in
#: play. Its off-policy membership and its SAC family below follow from
#: its class, not from a hand-kept list.
ALGOS: dict[str, type[BaseAlgorithm]] = {"SAC": SAC, "PPO": PPO, "DEMOSAC": DemoSAC}

#: Algorithms that learn off-policy from a replay buffer. For these the
#: number of gradient updates per rollout is decoupled from the number of
#: env steps collected, so a vectorised env silently starves the policy of
#: updates unless we compensate (see ``train._build_algo``). Derived from
#: the registered classes: a hand-kept set was the silent trap here, since
#: an off-policy registration missing from it kept SB3's
#: ``gradient_steps=1`` against a vectorised ``train_freq``.
OFF_POLICY_ALGOS = frozenset(
    name for name, cls in ALGOS.items() if issubclass(cls, OffPolicyAlgorithm)
)

#: Algorithms that learn on-policy: each update consumes one fresh
#: rollout of ``n_steps * n_envs`` transitions and then discards it, so
#: their guards and cadences are counted in rollouts (PPO).
ON_POLICY_ALGOS = frozenset(
    name for name, cls in ALGOS.items() if issubclass(cls, OnPolicyAlgorithm)
)

_UNCLASSIFIED_ALGOS = sorted(set(ALGOS) - OFF_POLICY_ALGOS - ON_POLICY_ALGOS)
if _UNCLASSIFIED_ALGOS:
    # Every branch in the trainer asks "off-policy or on-policy?"; an
    # algorithm that is neither would silently take whichever branch
    # tests the other set.
    raise TypeError(
        f"registered algorithm(s) {_UNCLASSIFIED_ALGOS} subclass neither "
        f"OffPolicyAlgorithm nor OnPolicyAlgorithm"
    )


def _family_root(name: str) -> str:
    """The registered algorithm that names ``name``'s config family.

    An algorithm's family is its most general registered ancestor
    (itself included): ``DemoSAC`` subclasses ``SAC``, so it reads SAC's
    hyperparameter block and warm-starts across the SAC family. Two
    unrelated registered ancestors would make the family ambiguous, so
    that refuses at import.
    """
    cls = ALGOS[name]
    ancestors = [other for other, base in ALGOS.items() if issubclass(cls, base)]
    roots = [
        other
        for other in ancestors
        if not any(
            ALGOS[other] is not ALGOS[base]
            and issubclass(ALGOS[other], ALGOS[base])
            for base in ancestors
        )
    ]
    if len(roots) != 1:
        raise TypeError(
            f"registered algorithm {name!r} has an ambiguous config family: "
            f"it subclasses the unrelated registered algorithms {roots}"
        )
    return roots[0]


#: Each registered algorithm's config family (``"SAC"`` for SAC and
#: DemoSAC, ``"PPO"`` for PPO). A recipe's per-algorithm block
#: (``Recipe.algo_cfg``) and a run file's ``[train.algo.<family>]`` table
#: are keyed by family, so DemoSAC inherits SAC's calibrated settings.
ALGO_FAMILIES: dict[str, str] = {name: _family_root(name) for name in ALGOS}

#: The family names: the only keys ``Recipe.algo_cfg`` may use.
FAMILIES = frozenset(ALGO_FAMILIES.values())


def algo_family(name: str) -> str:
    """The config family of ``name``, case-insensitively (``"SAC"``/``"PPO"``).

    Raises ``ValueError`` for an unknown algorithm, like
    :func:`resolve_algo`. Every behaviour that depends on the algorithm
    tests its family (or the derived on/off-policy sets) instead of
    comparing hand-written strings: ``algo.upper() == "SAC"`` silently
    treated DemoSAC as PPO.
    """
    resolve_algo(name)
    return ALGO_FAMILIES[name.upper()]


#: Constructor arguments SAC and PPO both accept but read differently,
#: so ``validate_model_kwargs`` cannot catch one carried from one family
#: to the other: ``n_steps`` (SAC 2.9's n-step returns vs PPO's rollout
#: length), ``batch_size`` (a replay sample vs a rollout minibatch),
#: ``ent_coef`` (SAC's ``"auto"`` temperature vs PPO's fixed loss
#: weight), ``use_sde`` and ``sde_sample_freq`` (gSDE noise reset per
#: off-policy ``train_freq`` collection vs per on-policy rollout). A
#: flat recipe or run-file key from this set is refused under the
#: family the recipe does not default to
#: (``recipes._refuse_divergent_flat_kwargs``).
DIVERGENT_SHARED_KWARGS = frozenset(
    {"n_steps", "batch_size", "ent_coef", "use_sde", "sde_sample_freq"}
)


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
    reserved = sorted(_RESERVED_MODEL_KWARGS & set(model_kwargs))
    if reserved:
        raise ValueError(
            f"model_kwargs may not set {reserved}: the trainer supplies "
            f"these to {cls.__name__} itself"
        )
    accepted, open_ended = _accepted_parameters(cls)
    if not open_ended:
        accepted = accepted - _RESERVED_MODEL_KWARGS
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


def _accepted_parameters(cls: type) -> tuple[set[str], bool]:
    """Explicit constructor parameters along the MRO.

    A subclass that adds its own keywords and forwards ``**kwargs`` to
    its parent (``DemoSAC`` -> ``SAC``) still has a closed parameter
    set: the union of every explicit signature down to the first
    ``__init__`` without a ``**kwargs`` catch-all. Returns
    ``(accepted, open_ended)``; ``open_ended`` is True only when every
    level forwards ``**kwargs``, in which case nothing can be rejected.
    """
    accepted: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters
        accepted |= {
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return accepted - {"self"}, False
    return accepted - {"self"}, True


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
