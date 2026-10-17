"""Canonical SB3 algorithm registry shared across the training code.

A single source of truth for "which ``algo`` string maps to which SB3
class" keeps :mod:`courtside_dynamics.training.train` and the notebook
replay helpers from drifting -- they previously each carried their own
``{"SAC": SAC, "PPO": PPO}`` literal.
"""
from __future__ import annotations

import difflib
import inspect
import json
import os
import warnings
import zipfile
from collections.abc import Mapping
from typing import Any

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import json_to_data

from courtside_dynamics.training.artifacts import RUN_LAYOUT, locate_artifact
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


#: Policy constructor arguments the algorithm supplies itself (its
#: ``_setup_model`` passes the spaces and the learning-rate schedule,
#: and ``use_sde`` from its own argument), so a ``policy_kwargs`` value
#: for one collides at construction or is silently overwritten.
_RESERVED_POLICY_KWARGS = frozenset(
    {"observation_space", "action_space", "lr_schedule", "use_sde"}
)

#: The ``net_arch`` dict keys SB3's actor-critic policies read (the
#: policy and value networks); anything else is silently ignored.
_ACTOR_CRITIC_NET_ARCH_KEYS = frozenset({"pi", "vf"})


def validate_policy_kwargs(
    algo_name: str, policy: Any, policy_kwargs: Any
) -> None:
    """Reject ``policy_kwargs`` the algorithm's policy class cannot accept.

    SB3 hands them to the policy constructor inside the algorithm's
    ``_setup_model`` -- after the env fleet is built and ``config.json``
    is written -- so an unknown key (a typo, or ``ortho_init`` meant for
    PPO's ``ActorCriticPolicy``) used to fail there. ``policy`` is one
    of the algorithm's aliases (``"MlpPolicy"``) or a policy class. An
    actor-critic policy's ``net_arch`` dict may only name ``"pi"`` and
    ``"vf"``: SB3 ignores any other key, SAC's ``"qf"`` included.
    """
    cls = resolve_algo(algo_name)
    if not isinstance(policy_kwargs, Mapping):
        raise TypeError(
            f"policy_kwargs must be a mapping, got {policy_kwargs!r}"
        )
    if isinstance(policy, str):
        policy_class = cls.policy_aliases.get(policy)
        if policy_class is None:
            raise ValueError(
                f"policy {policy!r} is not one of {cls.__name__}'s "
                f"policies {sorted(cls.policy_aliases)}"
            )
    else:
        policy_class = policy
    reserved = sorted(_RESERVED_POLICY_KWARGS & set(policy_kwargs))
    if reserved:
        raise ValueError(
            f"policy_kwargs may not set {reserved}: {cls.__name__} supplies "
            f"these to {policy_class.__name__} itself (gSDE is the "
            f"algorithm's model_kwargs['use_sde'])"
        )
    accepted, open_ended = _accepted_parameters(policy_class)
    if not open_ended:
        accepted = accepted - _RESERVED_POLICY_KWARGS
        for key in policy_kwargs:
            if key in accepted:
                continue
            suggestions = difflib.get_close_matches(key, sorted(accepted), n=1)
            hint = f"; did you mean '{suggestions[0]}'?" if suggestions else ""
            raise ValueError(
                f"policy_kwargs key '{key}' is not accepted by "
                f"{policy_class.__name__}{hint}"
            )
    net_arch = policy_kwargs.get("net_arch")
    if (
        isinstance(policy_class, type)
        and issubclass(policy_class, ActorCriticPolicy)
        and isinstance(net_arch, Mapping)
    ):
        # SB3's MlpExtractor reads only "pi" and "vf" and ignores every
        # other key, so SAC's {"pi": ..., "qf": ...} built a PPO value
        # network with no hidden layer and said nothing.
        unknown = sorted(set(net_arch) - _ACTOR_CRITIC_NET_ARCH_KEYS)
        if unknown:
            qf_hint = (
                " ('qf' is SAC's critic key; the actor-critic's value "
                "network is 'vf')"
                if "qf" in unknown
                else ""
            )
            raise ValueError(
                f"policy_kwargs net_arch keys {unknown} are not read by "
                f"{policy_class.__name__}, which builds its networks from "
                f"'pi' and 'vf' only{qf_hint}"
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


def _model_directories() -> tuple[tuple[str, ...], ...]:
    """Where a run keeps model files, relative to the run directory.

    The run root itself (the pre-0.14 flat layout), the layout's model
    folder (``model/``), ``checkpoints/``, the evaluation snapshots
    (``snapshots/``), and one stage directory under the performance
    gate's stage archive (``model/stage_bests/``, pre-0.14
    ``stage_bests/``). Derived from :data:`RUN_LAYOUT`, so a moved
    artifact moves the search with it.
    """
    model_dir = os.path.dirname(RUN_LAYOUT["best_model"])
    stage_archive = RUN_LAYOUT["stage_bests_dir"]
    shapes = {
        (),
        tuple(model_dir.split("/")),
        tuple(RUN_LAYOUT["checkpoints_dir"].split("/")),
        tuple(RUN_LAYOUT["snapshots_dir"].split("/")),
        (*stage_archive.split("/"), "*"),
        (os.path.basename(stage_archive), "*"),
    }
    return tuple(sorted(shapes, key=lambda shape: (len(shape), shape)))


def locate_run_config(path: str | os.PathLike[str]) -> str | None:
    """The ``config.json`` of the run that ``path`` belongs to.

    ``path`` is a run directory or a model file inside one
    (``model/best_model.zip``, a ``checkpoints/`` snapshot, a stage
    archive's ``best_model.zip``, or a pre-0.14 run's root-level file;
    an SB3 path without its ``.zip`` suffix works too). Only the
    directories the run layout puts a model in are searched, nearest
    first, so a checkpoint copied out of its run does not borrow a
    ``config.json`` from an arbitrary folder above it. ``None`` when no
    run configuration is found.
    """
    target = os.path.abspath(os.fspath(path))
    if os.path.isdir(target):
        return locate_artifact(target, "config")
    here = os.path.dirname(target)
    for shape in _model_directories():
        root = here
        matched = True
        for part in reversed(shape):
            if part != "*" and os.path.basename(root) != part:
                matched = False
                break
            root = os.path.dirname(root)
        if not matched:
            continue
        candidate = os.path.join(root, RUN_LAYOUT["config"])
        if os.path.isfile(candidate):
            return candidate
    return None


def recorded_run_algo(path: str | os.PathLike[str]) -> str | None:
    """The algorithm ``path``'s run recorded in ``config.json``, upper-cased.

    ``path`` is anything :func:`locate_run_config` accepts. ``None``
    when no run configuration is found. A ``config.json`` that cannot
    be read, or records no ``train_config.algo``, raises instead: the
    record exists and says nothing, so falling back to a guess would
    load the checkpoint through whichever class the guess names.
    """
    config_path = locate_run_config(path)
    if config_path is None:
        return None
    try:
        with open(config_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read the run configuration {config_path} "
            f"({error!r}), so the algorithm that trained {os.fspath(path)} "
            f"is unknown"
        ) from error
    train_config = payload.get("train_config") if isinstance(payload, dict) else None
    algo = train_config.get("algo") if isinstance(train_config, dict) else None
    if not isinstance(algo, str) or not algo:
        raise ValueError(
            f"{config_path} records no train_config.algo, so the algorithm "
            f"that trained {os.fspath(path)} is unknown"
        )
    return algo.upper()


def _checkpoint_zip_path(path: str | os.PathLike[str]) -> str | None:
    """The file SB3's ``load`` would open for ``path`` (``.zip`` optional)."""
    target = os.fspath(path)
    for candidate in (target, f"{target}.zip"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _checkpoint_policy_class(
    path: str | os.PathLike[str],
    custom_objects: Mapping[str, Any] | None = None,
) -> type | None:
    """The policy class an SB3 model zip records in its ``data`` entry.

    Deserialized the way SB3's ``load`` does it (``json_to_data``, so a
    ``custom_objects["policy_class"]`` replaces the stored one).
    ``None`` when the file, its ``data`` entry or the class cannot be
    read: ``load`` then reports the problem itself.
    """
    zip_path = _checkpoint_zip_path(path)
    if zip_path is None:
        return None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            data = json.loads(archive.read("data"))
        entry = data["policy_class"]
        with warnings.catch_warnings():
            # SB3 warns (and drops the key) when an object does not
            # deserialize; returning None leaves that to load().
            warnings.simplefilter("ignore")
            decoded = json_to_data(
                json.dumps({"policy_class": entry}),
                custom_objects=dict(custom_objects) if custom_objects else None,
            )
    except Exception:
        # Unreadable here is unreadable for load() too, which raises
        # its own (more specific) error; this check only adds one.
        return None
    policy_class = decoded.get("policy_class")
    return policy_class if isinstance(policy_class, type) else None


def _policy_family(policy_class: type) -> str | None:
    """The registered family whose policies ``policy_class`` subclasses."""
    for family in sorted(FAMILIES):
        bases = tuple(set(resolve_algo(family).policy_aliases.values()))
        if issubclass(policy_class, bases):
            return family
    return None


def _check_checkpoint_policy_family(
    path: str | os.PathLike[str],
    algo: str,
    custom_objects: Mapping[str, Any] | None,
    *,
    recorded: bool = False,
    label: str | None = None,
) -> None:
    """Refuse an ``algo`` the checkpoint's own policy contradicts.

    A checkpoint outside any run has no ``config.json`` to check the
    caller's ``algo`` against, but its zip records the policy class:
    ``SAC.load`` of a PPO checkpoint used to fail deep inside SB3
    (``'ActorCriticPolicy' object has no attribute 'actor'``). With
    ``recorded``, ``algo`` is what the run's ``config.json`` records,
    and a contradiction means the record is wrong (``PPO.load`` of an
    SAC zip raised a ``TypeError`` about ``use_sde``). ``label`` names
    the file in the message (default: ``path``).
    """
    policy_class = _checkpoint_policy_class(path, custom_objects)
    if policy_class is None:
        return
    expected = algo_family(algo)
    found = _policy_family(policy_class)
    if found == expected:
        return
    trained_by = (
        f"a {found}-family policy"
        if found is not None
        else "a policy of no registered algorithm family"
    )
    remedy = (
        "its run's config.json records the wrong algorithm"
        if recorded
        else "pass the algorithm that trained it"
    )
    raise ValueError(
        f"{label or os.fspath(path)} holds a {policy_class.__module__}."
        f"{policy_class.__qualname__}, {trained_by}, so it cannot load as "
        f"{algo.upper()} (family {expected}); {remedy}"
    )


def checkpoint_family_mismatch(
    path: str | os.PathLike[str], algo: str, *, label: str | None = None
) -> str | None:
    """Why ``path`` cannot load as its run's recorded ``algo``, or ``None``.

    :func:`load_run_model`'s family check, for a caller that checks
    every artifact before loading any (``experiments.harness``).
    """
    try:
        _check_checkpoint_policy_family(path, algo, None, recorded=True, label=label)
    except ValueError as error:
        return str(error)
    return None


def load_run_model(
    path: str | os.PathLike[str],
    algo: str | None = None,
    **load_kwargs: Any,
) -> BaseAlgorithm:
    """Load a saved model through the class its run recorded.

    ``path`` is the model file (``best_model.zip``, a checkpoint, a
    stage archive's best); the algorithm comes from the run's
    ``config.json`` (:func:`recorded_run_algo`). ``algo``, when given,
    must name the recorded algorithm (case-insensitively), else this
    raises: a caller that expected SAC is about to score a PPO run.
    Without a run configuration, ``algo`` is required, and the
    caller vouches for it.

    Each family loads through its root class: SAC and DemoSAC through
    ``SAC.load``, which is DemoSAC's designed inference path (its demo
    buffers are rebuilt only for training, so the library file need
    not exist), and PPO through ``PPO.load``. Loaders used to
    hard-code ``SAC.load``, which cannot read a PPO checkpoint.
    ``load_kwargs`` go to ``load`` unchanged (``device="cpu"``, say).
    The policy class the zip itself records must belong to the
    family of the algorithm used, recorded or given
    (:func:`_check_checkpoint_policy_family`): a mismatch raises
    ``ValueError`` naming it, never a ``TypeError`` from inside SB3.
    """
    recorded = recorded_run_algo(path)
    if recorded is None:
        if algo is None:
            raise FileNotFoundError(
                f"no run configuration (config.json) records the algorithm "
                f"that trained {os.fspath(path)}; pass the algorithm "
                f"explicitly (algo=..., or the tool's --algo)"
            )
        name = algo
        _check_checkpoint_policy_family(
            path, name, load_kwargs.get("custom_objects")
        )
    else:
        if algo is not None and algo.upper() != recorded:
            raise ValueError(
                f"requested algorithm {algo.upper()} does not match the "
                f"algorithm {recorded} that the run of {os.fspath(path)} "
                f"recorded"
            )
        name = recorded
        _check_checkpoint_policy_family(
            path, name, load_kwargs.get("custom_objects"), recorded=True
        )
    return resolve_algo(algo_family(name)).load(os.fspath(path), **load_kwargs)
