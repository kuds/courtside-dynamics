"""Evaluation protocols: what an offline score is a score of.

``docs/design_env_ladder_and_structure_20261016.md`` section 4 (the
eval-protocol layer) and M3; ``docs/design_campaign_structure_20261016.md``
sections 2 and 5 ("Eras") and step L2. An era is the pair (task
fingerprint, evaluation protocol). A protocol fixes everything an
evaluation batch measures besides the policy: the evaluation env (its
class and every constructor argument), the episode count, the seed
blocks and reset options, the info keys and aggregates, the selection
keys with their per-key deltas, the confirmation block, and the device.
A protocol change is a version bump, which means re-scoring the saved
snapshots (``experiments.harness.rescore``), not retraining.

:class:`EvalProtocol` is a frozen dataclass with a :attr:`fingerprint`
(the sha256 of its canonical content, id and description excluded), so
two protocols with the same fingerprint score identically and a score
row names the content it was scored under, not only an id.

Registered protocols (:func:`get_protocol`):

``paddle-npoint@v2``
    The PaddleTennis in-loop selection evaluation of the 2026-10-05
    instruments: 30 paired episodes on the run's own seed block with
    alternating serves, the recipe's evaluation env as it was through
    2026-10-17 (drill off, the contact and reach escrows on), the k>=2
    headline, ``success_rate`` and reward with per-key deltas, and a
    confirmation block 100,000 seeds on. Declared literally, so it
    outlived the recipe's move to @v3 and stays scorable.
``paddle-npoint@v3``
    @v2 with the evaluation env's training aids stripped (design L3):
    ``contact_shaping``, ``reach_shaping`` and ``hold_shaping`` are 0,
    so ``episode_reward_mean``, the last selection tie-break, is task
    reward only. Nothing else moves: PaddleTennis observations hold no
    escrow state, so a policy sees the same observations and acts the
    same under both, and every non-reward metric (the k>=2 headline,
    ``success_rate``) is identical. The PaddleTennis recipe's in-loop
    selection evaluation since 2026-10-17; a test pins that the recipe
    still equals it.
``<recipe>-inloop@v1``
    One per other recipe: its in-loop info-dict evaluator as
    ``train()`` wires it today (``RECIPE_PROTOCOLS``), so the harness is
    not paddle-only. These are derived from the live recipe and
    checked against fingerprints pinned at registration
    (``_PINNED_FINGERPRINTS``): once a recipe's in-loop evaluation
    changes, its protocol refuses to load rather than silently score
    something else. Declare the old version literally (as
    ``paddle-npoint@v2`` is) before changing such a recipe.

**The seed block.** Today's paired evaluation derives its block from
the run (``train.resolve_eval_seed``: the run's ``eval_seed``, else
``seed + 1,000,000``), so the registered protocols carry
``seed_start=None``, "the run's own block", resolved by
:func:`resolve_seed_block` from the run's ``config.json``: a scored row
records the block it used. A probe or a model outside any run passes
the block explicitly. The confirmation block starts
``confirm_seed_offset`` after the standard one. An unpaired protocol
(most of the legacy recipes) opens one fresh env seeded at the block
start and lets its auto-resets continue: offline that is reproducible
and the same feed for every artifact, but it reproduces only the
*first* in-loop evaluation of an unpaired run, whose stream continues
across evaluations (which is why pairing exists). A performance gate
rescales its matched evaluator per stage; a protocol scores the
evaluation env as constructed.

**What a run records.** ``config.json``'s ``comparability_key`` and the
run record's ``task`` carry ``eval_protocol`` (:func:`eval_protocol_record`):
the in-loop selection protocol's fingerprint and, when it is a
registered protocol of the run's recipe, its id.

Building a protocol imports MuJoCo (the env classes), never
Stable-Baselines3 or torch; :func:`from_train_config` and the derived
registry entries import the training stack lazily.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
import types
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from courtside_dynamics.experiments.records import read_json_object
from courtside_dynamics.experiments.select import SelectionRule

#: ``callbacks.info_dict_eval.CONFIRMATION_SEED_OFFSET`` and
#: ``training.train.EVAL_SEED_OFFSET``, written as data here: a
#: protocol is frozen, and a test pins these to the code's values.
CONFIRMATION_SEED_OFFSET = 100_000
RUN_SEED_OFFSET = 1_000_000

#: A registered protocol id: ``<kebab-name>@v<N>``.
PROTOCOL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*@v[1-9][0-9]*$")

#: The pseudo-protocol "each run's own in-loop evaluation"
#: (:func:`protocol_for_run`), for checking offline scores against a
#: run's in-loop rows.
INLOOP = "inloop"

#: A recorded constructor kwarg that ``config.json`` could only store as
#: an object's repr (``json.dump(default=repr)``): it cannot be rebuilt.
_REPR_STRING = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*\(.*\)$", re.DOTALL)


def _qualname(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def canonical(value: Any, path: str = "$") -> Any:
    """``value`` as JSON-ready data for the fingerprint.

    Tuples and lists are both arrays (a JSON round trip turns one into
    the other, and the env receives the same values); dataclasses,
    enums, classes and numpy values carry a tag; non-finite floats are
    spelled out. Anything else refuses, so the fingerprint never
    depends on a ``repr`` that could change silently.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return {"__enum__": _qualname(type(value)), "name": value.name}
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, np.generic):
        return {"__numpy__": value.dtype.str, "value": canonical(value.item(), path)}
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.dtype.str,
            "shape": list(value.shape),
            "data": canonical(value.ravel().tolist(), path),
        }
    if isinstance(value, (tuple, list)):
        return [canonical(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": _qualname(type(value)),
            "fields": {
                f.name: canonical(getattr(value, f.name), f"{path}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: canonical(item, f"{path}.{key}") for key, item in value.items()}
        return {
            "__items__": sorted(
                (
                    [canonical(key, f"{path}<key>"), canonical(item, f"{path}[{key!r}]")]
                    for key, item in value.items()
                ),
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, type):
        return {"__class__": _qualname(value)}
    raise TypeError(f"{path}: cannot canonicalize {type(value).__name__}")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    keys = tuple(dict.fromkeys(value))
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError(f"{name} must hold non-empty strings, got {value!r}")
    return keys


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class EvalProtocol:
    """One evaluation protocol (see the module docstring).

    The measurement fields are ``InfoDictEvalCallback``'s parameters of
    the same meaning (``n_episodes`` is its ``n_eval_episodes``,
    ``distribution_keys`` its ``episode_distribution_keys``,
    ``survival_thresholds`` its ``episode_survival_thresholds``), and
    the harness hands them to the callback's own measurement core
    (``callbacks.info_dict_eval.collect_episode_metrics``).

    - ``env_cls`` / ``env_kwargs``: the evaluation env, every
      constructor argument as the env resolves it (its EzPickle
      kwargs, what ``config.json`` records as
      ``evaluation_env.constructor_kwargs``).
    - ``paired``: every episode reseeded (``seed_start + i``, options
      cycled), or one fresh unpaired stream from ``seed_start``.
    - ``seed_start``: the standard block's first seed; ``None`` is the
      run's own block (:func:`resolve_seed_block`; a paired run's
      recorded block, else its ``eval_seed``, else ``seed +
      run_seed_offset``).
    - ``confirm_seed_offset``: where the confirmation block starts,
      relative to the standard one.
    - ``selection_keys`` / ``min_deltas`` / ``confirm``: the
      lexicographic rule (:meth:`selection_rule`); empty keys for a
      reward-selected recipe, whose run selected on SB3's reward
      evaluator, not this stream.
    - ``device``: where the policy runs. Scores on another device are
      not this protocol's (float32 kernels differ by device).
    """

    id: str
    recipe: str | None
    env_cls: type
    env_kwargs: Mapping[str, Any]
    n_episodes: int
    paired: bool
    seed_start: int | None = None
    run_seed_offset: int = RUN_SEED_OFFSET
    confirm_seed_offset: int = CONFIRMATION_SEED_OFFSET
    reset_options: tuple[Mapping[str, Any], ...] | None = None
    deterministic: bool = True
    info_keys: tuple[str, ...] | None = None
    terminal_info_keys: tuple[str, ...] = ()
    distribution_keys: tuple[str, ...] = ()
    survival_thresholds: Mapping[str, tuple[int, ...]] = dataclasses.field(
        default_factory=dict
    )
    episode_sum_keys: tuple[str, ...] | None = None
    success_key: str | None = None
    success_threshold: float = 1.0
    phase_key: str | None = None
    phase_labels: Mapping[int, str] = dataclasses.field(default_factory=dict)
    selection_keys: tuple[str, ...] = ()
    min_deltas: tuple[float, ...] = ()
    confirm: bool = False
    device: str = "cpu"
    description: str = ""

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if (
            not isinstance(self.id, str)
            or not self.id
            or any(c.isspace() or c in "/\\" for c in self.id)
        ):
            raise ValueError(f"protocol id must be a file-name-safe string: {self.id!r}")
        if not isinstance(self.env_cls, type):
            raise TypeError(f"env_cls must be a class, got {self.env_cls!r}")
        if not isinstance(self.env_kwargs, Mapping) or any(
            not isinstance(key, str) for key in self.env_kwargs
        ):
            raise TypeError("env_kwargs must be a mapping with string keys")
        put("env_kwargs", types.MappingProxyType(dict(self.env_kwargs)))
        put("n_episodes", _int(self.n_episodes, "n_episodes", minimum=1))
        if not isinstance(self.paired, bool) or not isinstance(self.deterministic, bool):
            raise TypeError("paired and deterministic must be bools")
        if self.seed_start is not None:
            put("seed_start", _int(self.seed_start, "seed_start"))
        put("run_seed_offset", _int(self.run_seed_offset, "run_seed_offset"))
        put(
            "confirm_seed_offset",
            _int(self.confirm_seed_offset, "confirm_seed_offset", minimum=1),
        )
        if self.reset_options is not None:
            if not self.paired:
                raise ValueError(
                    "reset_options require a paired protocol: an unpaired "
                    "stream never applies them"
                )
            if isinstance(self.reset_options, Mapping) or not self.reset_options:
                raise ValueError("reset_options must be a non-empty sequence of mappings")
            options = []
            for option in self.reset_options:
                if not isinstance(option, Mapping) or any(
                    not isinstance(key, str) for key in option
                ):
                    raise TypeError(f"a reset option must be a mapping: {option!r}")
                options.append(types.MappingProxyType(dict(option)))
            put("reset_options", tuple(options))
        if self.info_keys is not None:
            put("info_keys", _string_tuple(self.info_keys, "info_keys"))
        put(
            "terminal_info_keys",
            _string_tuple(self.terminal_info_keys, "terminal_info_keys"),
        )
        put(
            "distribution_keys",
            _string_tuple(self.distribution_keys, "distribution_keys"),
        )
        thresholds: dict[str, tuple[int, ...]] = {}
        for key, values in dict(self.survival_thresholds).items():
            if not isinstance(key, str) or not key:
                raise TypeError("survival threshold keys must be non-empty strings")
            thresholds[key] = tuple(
                sorted(
                    dict.fromkeys(
                        _int(v, f"survival_thresholds[{key!r}]", minimum=1)
                        for v in values
                    )
                )
            )
        put("survival_thresholds", types.MappingProxyType(thresholds))
        if self.episode_sum_keys is not None:
            put(
                "episode_sum_keys",
                _string_tuple(self.episode_sum_keys, "episode_sum_keys"),
            )
        labels = {
            _int(phase, "phase label key"): str(label)
            for phase, label in dict(self.phase_labels).items()
        }
        put("phase_labels", types.MappingProxyType(labels))
        put("success_threshold", float(self.success_threshold))
        keys = (
            _string_tuple(self.selection_keys, "selection_keys")
            if self.selection_keys
            else ()
        )
        put("selection_keys", keys)
        deltas = tuple(float(delta) for delta in self.min_deltas)
        if len(deltas) != len(keys):
            raise ValueError(
                f"{len(deltas)} min_deltas for {len(keys)} selection keys"
            )
        if any(not math.isfinite(delta) or delta < 0.0 for delta in deltas):
            raise ValueError(f"min_deltas must be finite and nonnegative: {deltas}")
        put("min_deltas", deltas)
        if not isinstance(self.confirm, bool):
            raise TypeError("confirm must be a bool")
        if self.confirm and not keys:
            raise ValueError("a confirmation block needs selection keys")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string: {self.device!r}")

    # -- identity ------------------------------------------------------------

    def canonical(self) -> dict[str, Any]:
        """Every field but ``id``, ``recipe`` and ``description``, canonical.

        ``recipe`` names where the env came from; the env itself is
        ``env_cls`` and ``env_kwargs``, so two recipes that evaluate
        identically share a fingerprint.
        """
        skipped = {"id", "recipe", "description"}
        return {
            f.name: canonical(getattr(self, f.name), f"$.{f.name}")
            for f in dataclasses.fields(self)
            if f.name not in skipped
        }

    @property
    def fingerprint(self) -> str:
        """sha256 of :meth:`canonical`'s sorted, compact JSON."""
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def selection_rule(self) -> SelectionRule:
        """The protocol's lexicographic rule (``experiments.select``)."""
        if not self.selection_keys:
            raise ValueError(
                f"protocol {self.id} has no selection keys: its recipe selects "
                f"on SB3's reward evaluator, not on this stream; pass select() "
                f"an explicit SelectionRule"
            )
        return SelectionRule(self.selection_keys, self.min_deltas, self.confirm)

    # -- use -----------------------------------------------------------------

    def make_env(self) -> Any:
        """A fresh evaluation env (the kwargs are copied, never shared)."""
        return self.env_cls(**deepcopy(dict(self.env_kwargs)))

    def measurement_kwargs(self) -> dict[str, Any]:
        """``collect_episode_metrics``'s settings (all but the seeds)."""
        return {
            "n_eval_episodes": self.n_episodes,
            "deterministic": self.deterministic,
            "reset_options": (
                None
                if self.reset_options is None
                else tuple(dict(option) for option in self.reset_options)
            ),
            "info_keys": self.info_keys,
            "terminal_info_keys": self.terminal_info_keys,
            "episode_distribution_keys": self.distribution_keys,
            "episode_survival_thresholds": dict(self.survival_thresholds),
            "episode_sum_keys": self.episode_sum_keys,
            "success_key": self.success_key,
            "success_threshold": self.success_threshold,
            "phase_key": self.phase_key,
            "phase_labels": dict(self.phase_labels),
        }

    def derive(
        self,
        id: str,
        *,
        env_overrides: Mapping[str, Any] | None = None,
        description: str | None = None,
        **changes: Any,
    ) -> EvalProtocol:
        """A new protocol from this one: other env kwargs or fields.

        For probes (``experiments.probe``'s ``--env``): the result is not
        registered, and its fingerprint says what it scores.
        """
        unknown = sorted(set(env_overrides or {}) - set(self.env_kwargs))
        if unknown:
            raise ValueError(
                f"{self.id}'s env takes no {unknown}; its kwargs are "
                f"{sorted(self.env_kwargs)}"
            )
        kwargs = {**self.env_kwargs, **(env_overrides or {})}
        return dataclasses.replace(
            self,
            id=id,
            env_kwargs=kwargs,
            description=self.description if description is None else description,
            **changes,
        )


# -- seed blocks ---------------------------------------------------------------


@dataclass(frozen=True)
class SeedBlock:
    """Where a scored batch's seeds start, and why there.

    ``source`` is ``"explicit"`` (the caller's), ``"protocol"`` (a fixed
    block), ``"run_paired_block"`` (the paired block the run recorded),
    ``"run_eval_seed"`` (its configured ``eval_seed``),
    ``"run_seed_offset"`` (``seed + run_seed_offset``, how today's
    recipe derives a block, for a run that evaluated unpaired) or
    ``"run_info_eval_env"`` (``seed + n_envs + 1``, the seed ``train()``
    gives the info evaluator's env, for an unpaired protocol).
    """

    start: int
    source: str

    def for_block(self, protocol: EvalProtocol, block: str) -> int:
        """The first seed of ``block`` (``"standard"`` or ``"confirm"``)."""
        if block == "standard":
            return self.start
        if block == "confirm":
            return self.start + protocol.confirm_seed_offset
        raise ValueError(f"block must be 'standard' or 'confirm', got {block!r}")


def resolve_seed_block(
    protocol: EvalProtocol,
    *,
    run_config: Mapping[str, Any] | None = None,
    seed_start: int | None = None,
) -> SeedBlock:
    """The standard block a scored batch starts at (see :class:`SeedBlock`).

    An explicit ``seed_start`` wins, then the protocol's own; otherwise
    the protocol scores on the run's block, read from ``run_config``
    (the run's ``config.json`` payload). Raises ``ValueError`` when
    neither says, rather than picking a block.
    """
    if seed_start is not None:
        return SeedBlock(_int(seed_start, "seed_start"), "explicit")
    if protocol.seed_start is not None:
        return SeedBlock(protocol.seed_start, "protocol")
    if run_config is None:
        raise ValueError(
            f"protocol {protocol.id} scores on the run's own seed block: pass "
            f"seed_start, or a model inside a run directory (its config.json "
            f"names the block)"
        )
    train_config = run_config.get("train_config")
    train_config = train_config if isinstance(train_config, Mapping) else {}
    seed = train_config.get("seed")
    seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    if protocol.paired:
        seeding = run_config.get("evaluation_seeding")
        if isinstance(seeding, Mapping) and seeding.get("paired") is True:
            start = seeding.get("selection_batch_seed_start")
            if isinstance(start, int) and not isinstance(start, bool):
                return SeedBlock(start, "run_paired_block")
        eval_seed = train_config.get("eval_seed")
        if isinstance(eval_seed, int) and not isinstance(eval_seed, bool):
            return SeedBlock(eval_seed, "run_eval_seed")
        if seed is not None:
            return SeedBlock(seed + protocol.run_seed_offset, "run_seed_offset")
    else:
        n_envs = train_config.get("n_envs")
        if seed is not None and isinstance(n_envs, int) and not isinstance(n_envs, bool):
            return SeedBlock(seed + n_envs + 1, "run_info_eval_env")
    raise ValueError(
        f"protocol {protocol.id} scores on the run's own seed block, and the "
        f"run's config.json records no seed to derive it from (an unseeded "
        f"run); pass seed_start"
    )


# -- derivation from a run's configuration ---------------------------------------


def _inloop_protocol(
    *,
    id: str,
    recipe: str | None,
    env_cls: type,
    env_kwargs: Mapping[str, Any],
    settings: Mapping[str, Any],
    paired: bool,
    description: str,
) -> EvalProtocol:
    """The protocol of the info-dict evaluator ``train()`` wires for ``settings``.

    ``settings`` holds ``TrainConfig`` field values by name (from a
    config or a ``config.json``'s ``train_config``). Mirrors
    ``_train_in_log_dir``'s wiring: under headline selection the
    evaluator rolls ``n_eval_episodes`` and selects
    (``train._selection_metric_keys``, the per-key deltas, the
    confirmation block); otherwise it reports on a quarter of them and
    selects nothing.
    """
    if not settings.get("info_dict_eval", True):
        raise ValueError(
            "the run has no info-dict evaluator (info_dict_eval=False), so no "
            "in-loop protocol to derive"
        )
    headline_key = settings.get("headline_key")
    success_key = settings.get("success_key")
    n_eval_episodes = int(settings.get("n_eval_episodes", 30))
    keys: tuple[str, ...] = ()
    deltas: tuple[float, ...] = ()
    if headline_key:
        configured = settings.get("best_metric_keys")
        keys = (
            _string_tuple(configured, "best_metric_keys")
            if configured
            else (
                f"{headline_key}_ep_mean",
                *(("success_rate",) if success_key else ()),
                "episode_reward_mean",
            )
        )
        raw = settings.get("best_metric_min_delta", 0.0)
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - set(keys))
            if unknown:
                raise ValueError(f"best_metric_min_delta names {unknown}, not selection keys")
            deltas = tuple(float(raw.get(key, 0.0)) for key in keys)
        else:
            deltas = tuple(float(raw) for _ in keys)
    options = settings.get("eval_reset_options")
    survival = settings.get("info_eval_survival_thresholds") or {}
    return EvalProtocol(
        id=id,
        recipe=recipe,
        env_cls=env_cls,
        env_kwargs=env_kwargs,
        n_episodes=n_eval_episodes if headline_key else max(1, n_eval_episodes // 4),
        paired=paired,
        seed_start=None,
        reset_options=(
            tuple(options) if paired and options is not None else None
        ),
        info_keys=settings.get("info_eval_keys"),
        terminal_info_keys=tuple(settings.get("info_eval_terminal_keys") or ()),
        distribution_keys=tuple(settings.get("info_eval_distribution_keys") or ()),
        survival_thresholds={key: tuple(values) for key, values in dict(survival).items()},
        success_key=success_key,
        success_threshold=float(settings.get("success_threshold", 1.0)),
        phase_key=settings.get("phase_key"),
        phase_labels={int(k): v for k, v in dict(settings.get("phase_labels") or {}).items()},
        selection_keys=keys,
        min_deltas=deltas,
        confirm=bool(headline_key) and bool(settings.get("confirm_best_eval", False)),
        description=description,
    )


#: The ``TrainConfig`` fields :func:`_inloop_protocol` reads.
_INLOOP_FIELDS = (
    "info_dict_eval",
    "headline_key",
    "success_key",
    "success_threshold",
    "n_eval_episodes",
    "best_metric_keys",
    "best_metric_min_delta",
    "confirm_best_eval",
    "eval_reset_options",
    "info_eval_keys",
    "info_eval_terminal_keys",
    "info_eval_distribution_keys",
    "info_eval_survival_thresholds",
    "phase_key",
    "phase_labels",
)


def from_train_config(
    cfg: Any,
    *,
    id: str,
    description: str = "",
    env_cls: type | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
) -> EvalProtocol:
    """The in-loop selection protocol ``train(cfg)`` would evaluate under.

    Builds one evaluation env (``cfg.eval_env_fn``, else ``cfg.env_fn``)
    to read its class and resolved constructor kwargs, as
    ``config.json`` records them; a caller that already built one
    (``write_run_config``) passes its ``env_cls`` and ``env_kwargs``
    instead. Pairing follows ``train.resolve_eval_seed``. The seed
    block stays the run's own.
    """
    from courtside_dynamics.training.train import resolve_eval_seed

    kwargs: Any = env_kwargs
    if env_cls is None:
        if env_kwargs is not None:
            raise ValueError("env_kwargs describe an env: pass its env_cls too")
        env = (cfg.eval_env_fn or cfg.env_fn)()
        try:
            env_cls = type(env)
            kwargs = getattr(env, "_ezpickle_kwargs", None)
        finally:
            env.close()
    if not isinstance(kwargs, Mapping):
        raise TypeError(
            f"{env_cls.__name__} records no constructor kwargs "
            f"(_ezpickle_kwargs), so a protocol cannot pin its env"
        )
    return _inloop_protocol(
        id=id,
        recipe=cfg.recipe_name,
        env_cls=env_cls,
        env_kwargs=kwargs,
        settings={name: getattr(cfg, name) for name in _INLOOP_FIELDS},
        paired=resolve_eval_seed(cfg) is not None,
        description=description,
    )


def _env_class_named(name: str, recipe: str | None) -> type:
    from courtside_dynamics.recipes import RECIPES

    if recipe in RECIPES and RECIPES[recipe].env_cls.__name__ == name:
        return RECIPES[recipe].env_cls
    found = {r.env_cls for r in RECIPES.values() if r.env_cls.__name__ == name}
    if len(found) != 1:
        raise ValueError(
            f"config.json's evaluation env class {name!r} is not one recipe "
            f"env class, so it cannot be rebuilt"
        )
    return found.pop()


def _rebuildable(value: Any, path: str) -> Any:
    """A recorded kwarg as the env receives it: arrays become tuples."""
    if isinstance(value, list):
        return tuple(_rebuildable(item, f"{path}[{i}]") for i, item in enumerate(value))
    if isinstance(value, dict):
        return {key: _rebuildable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, str) and _REPR_STRING.match(value):
        raise ValueError(
            f"config.json records {path} as an object's repr ({value[:60]!r}), "
            f"so its evaluation env cannot be rebuilt from the record; score "
            f"it under its recipe's registered protocol"
        )
    return value


def from_run_config(
    config: Mapping[str, Any], *, id: str, description: str = ""
) -> EvalProtocol:
    """The in-loop selection protocol a run's ``config.json`` records.

    The same derivation as :func:`from_train_config`, from the
    recorded ``train_config`` and ``evaluation_env`` blocks (the env's
    class by name, its recorded constructor kwargs with JSON arrays
    read as tuples). Pairing is the recorded ``evaluation_seeding``,
    else ``train.resolve_eval_seed``'s rule on the recorded fields. A
    field an older ``config.json`` lacks takes ``TrainConfig``'s
    default. Raises ``ValueError`` for a record it cannot rebuild.
    """
    train_config = config.get("train_config")
    if not isinstance(train_config, Mapping):
        raise ValueError("config.json has no train_config object")
    evaluation_env = config.get("evaluation_env") or config.get("env")
    if not isinstance(evaluation_env, Mapping):
        raise ValueError("config.json has no evaluation_env object")
    kwargs = evaluation_env.get("constructor_kwargs")
    name = evaluation_env.get("class")
    if not isinstance(kwargs, Mapping) or not isinstance(name, str):
        raise ValueError(
            "config.json records no evaluation env class and constructor kwargs"
        )
    recipe = config.get("recipe_name")
    recipe = recipe if isinstance(recipe, str) else None
    seeding = config.get("evaluation_seeding")
    if isinstance(seeding, Mapping) and isinstance(seeding.get("paired"), bool):
        paired = bool(seeding["paired"])
    else:
        paired = train_config.get("eval_seed") is not None or (
            train_config.get("seed") is not None
            and train_config.get("eval_reset_options") is not None
        )
    settings = {name_: train_config[name_] for name_ in _INLOOP_FIELDS if name_ in train_config}
    return _inloop_protocol(
        id=id,
        recipe=recipe,
        env_cls=_env_class_named(name, recipe),
        env_kwargs={
            key: _rebuildable(value, f"evaluation_env.constructor_kwargs.{key}")
            for key, value in kwargs.items()
        },
        settings=settings,
        paired=paired,
        description=description,
    )


def protocol_for_run(run_dir: str | os.PathLike[str]) -> EvalProtocol:
    """``inloop:<run id>``: the run's own in-loop evaluation, from its config.json."""
    from courtside_dynamics.training.artifacts import locate_artifact

    root = Path(run_dir)
    config_path = locate_artifact(root, "config")
    if config_path is None:
        raise FileNotFoundError(f"{root} has no config.json")
    config = read_json_object(config_path, label=f"{root.name}/config.json")
    return from_run_config(
        config,
        id=f"{INLOOP}:{root.resolve().name}",
        description=f"the in-loop evaluation {root.resolve().name}'s config.json records",
    )


# -- the registry ------------------------------------------------------------------


def _paddle_npoint_v2() -> EvalProtocol:
    from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv

    return EvalProtocol(
        id="paddle-npoint@v2",
        recipe="PaddleTennis",
        env_cls=PaddleTennisEnv,
        # The recipe's evaluation env through 2026-10-17 (contact and
        # reach escrows on, continuous n-point play, the drill off),
        # every argument as the env resolves it.
        env_kwargs={
            "episode_len": 1500,
            "serve_config": None,
            "opponent_controller": None,
            "court_style": "diagnostic",
            "volley_rule": "fault",
            "return_reward": 1.0,
            "fault_penalty": 1.0,
            "unsafe_physics_penalty": 2.0,
            "contact_shaping": 0.25,
            "reach_shaping": 0.25,
            "reach_shaping_radius": 3.0,
            "hold_shaping": 0.0,
            "hold_shaping_travel": 4.0,
            "points_per_episode": None,
            "drill_library": None,
            "drill_fraction": 0.0,
            "drill_context": "feed",
            "deep_contact_depth": 5.5,
            "render_mode": "rgb_array",
        },
        n_episodes=30,
        paired=True,
        seed_start=None,
        run_seed_offset=RUN_SEED_OFFSET,
        confirm_seed_offset=CONFIRMATION_SEED_OFFSET,
        reset_options=({"serve_side": "a"}, {"serve_side": "b"}),
        deterministic=True,
        info_keys=(
            "crossings",
            "rally_count",
            "legal_hit_count",
            "legal_hit_count_a",
            "bounce_count",
            "episode_legal_hit_count_a",
            "episode_valid_return_count_a",
            "episode_rally_returns_a",
            "contact_depth_a",
            "episode_mean_contact_depth_a",
            "episode_deep_returns_a",
            "rew_return",
            "rew_fault",
            "rew_unsafe",
            "rew_shaping",
            "rew_shaping_clawback",
            "rew_reach",
            "rew_reach_clawback",
            "rew_hold",
            "rew_hold_clawback",
        ),
        terminal_info_keys=(
            "duplicate_contact_suppressed_count",
            "duplicate_event_suppressed_count",
            "stale_event_suppressed_count",
            "serve_side_is_policy",
            "term_out_of_bounds",
            "term_ball_net",
            "term_second_bounce",
            "term_failed_to_cross",
            "term_illegal_hit",
            "term_net_touch",
            "term_volley",
            "term_nonfinite",
            "term_timeout",
            "points_played",
            "completed_point_crossings",
            "point_serve_nudged",
            "point_end_out_of_bounds",
            "point_end_ball_net",
            "point_end_second_bounce",
            "point_end_failed_to_cross",
            "point_end_illegal_hit",
            "point_end_net_touch",
            "point_end_volley",
            "episode_legal_hit_count_a",
            "episode_valid_return_count_a",
            "episode_rally_returns_a",
            "episode_deep_returns_a",
            "episode_contact_depth_sum_a",
            "episode_mean_contact_depth_a",
        ),
        distribution_keys=(
            "crossings",
            "episode_rally_returns_a",
            "episode_legal_hit_count_a",
        ),
        survival_thresholds={},
        episode_sum_keys=None,
        success_key="episode_rally_returns_a",
        success_threshold=1.0,
        phase_key="rally_phase",
        phase_labels={
            0: "initial_feed",
            1: "awaiting_return",
            2: "return_in_flight",
            3: "terminal",
        },
        selection_keys=(
            "episode_rally_returns_a_ep_mean",
            "success_rate",
            "episode_reward_mean",
        ),
        min_deltas=(0.5 / 30, 0.5 / 30, 0.25),
        confirm=True,
        device="cpu",
        description=(
            "PaddleTennis in-loop selection evaluation of the 2026-10-05 "
            "instrument batch: 30 paired episodes on the run's seed block "
            "(eval_seed, else seed + 1,000,000) with alternating serves, "
            "selected lexicographically on the k>=2 headline "
            "(episode_rally_returns_a_ep_mean), success_rate and reward "
            "(deltas 1/60, 1/60, 0.25), confirmed on the block 100,000 on; "
            "the eval env carries the contact and reach escrows"
        ),
    )


#: The training aids @v3 strips from @v2's evaluation env: the escrowed
#: shaping terms (``docs/design_campaign_structure_20261016.md`` section
#: 5, "Eras"). Their radius and travel parameters stay as they were;
#: with the scale at 0 they pay nothing.
PADDLE_TRAINING_AIDS_OFF: Mapping[str, float] = types.MappingProxyType(
    {"contact_shaping": 0.0, "reach_shaping": 0.0, "hold_shaping": 0.0}
)


def _paddle_npoint_v3() -> EvalProtocol:
    return _paddle_npoint_v2().derive(
        "paddle-npoint@v3",
        env_overrides=PADDLE_TRAINING_AIDS_OFF,
        description=(
            "paddle-npoint@v2 with the evaluation env's training aids "
            "stripped (contact, reach and hold shaping 0), so "
            "episode_reward_mean, the last selection tie-break, is task "
            "reward only: 30 paired episodes on the run's seed block "
            "(eval_seed, else seed + 1,000,000) with alternating serves, "
            "selected lexicographically on the k>=2 headline "
            "(episode_rally_returns_a_ep_mean), success_rate and task "
            "reward (deltas 1/60, 1/60, 0.25), confirmed on the block "
            "100,000 on. PaddleTennis observations carry no escrow "
            "state, so every non-reward metric equals @v2's"
        ),
    )


#: Recipe -> the protocol of its in-loop selection evaluation.
RECIPE_PROTOCOLS: dict[str, str] = {
    "BallBalance": "ball-balance-inloop@v1",
    "BallBounce": "ball-bounce-inloop@v1",
    "HumanoidTennisCoopSmoke": "humanoid-tennis-coop-smoke-inloop@v1",
    "HumanoidTennisStage0Intercept": "humanoid-tennis-stage0-intercept-inloop@v1",
    "HumanoidTennisStage1AnchoredReturn": (
        "humanoid-tennis-stage1-anchored-return-inloop@v1"
    ),
    "HumanoidTennisStage2RandomizedReturn": (
        "humanoid-tennis-stage2-randomized-return-inloop@v1"
    ),
    # paddle-npoint@v2 until 2026-10-17 (design L3's era boundary).
    "PaddleTennis": "paddle-npoint@v3",
    "WallBall": "wall-ball-inloop@v1",
    "WallBallBaseline": "wall-ball-baseline-inloop@v1",
    "WallBallBootstrap": "wall-ball-bootstrap-inloop@v1",
    "WallBallDepthCurriculum": "wall-ball-depth-curriculum-inloop@v1",
    "WallBallDepthCurriculumAligned": "wall-ball-depth-curriculum-aligned-inloop@v1",
    "WallBallGoalRally": "wall-ball-goal-rally-inloop@v1",
    "WallBallTrueBaseline": "wall-ball-true-baseline-inloop@v1",
    "WallBallVolley": "wall-ball-volley-inloop@v1",
}

#: The fingerprint each derived protocol had when it was registered.
#: A recipe whose in-loop evaluation changes no longer derives it, and
#: :func:`get_protocol` refuses: declare the old version literally
#: before changing the recipe, and register the new one as ``@v2``.
_PINNED_FINGERPRINTS: dict[str, str] = {
    "ball-balance-inloop@v1": (
        "fe5c579495647297fdcdf6ad7cc256d36756145d45497ad22936950804b7c25c"
    ),
    "ball-bounce-inloop@v1": (
        "049645eafc421195d3a82ab4802b2760c27372e8b261e1ecaa3a6b65e850c2cf"
    ),
    "humanoid-tennis-coop-smoke-inloop@v1": (
        "03aa1eee91c986073962ff31185764ee160e8c4e71a1c7b8dd30ebb61456d513"
    ),
    "humanoid-tennis-stage0-intercept-inloop@v1": (
        "f28b3802abd4c5ea598fb29c890370fc0425af75eec67cbc23313f48d24f3762"
    ),
    "humanoid-tennis-stage1-anchored-return-inloop@v1": (
        "5610f630569f3cddadd54fd6b299de2bcab3ccd22b3d9fd0226412ee296721eb"
    ),
    "humanoid-tennis-stage2-randomized-return-inloop@v1": (
        "b16361080411a4d6ed725914fb5627e23860a0721e3dff6d8d4cc3e8900c2937"
    ),
    "wall-ball-inloop@v1": (
        "e72a2238f8b7c3e1ed4e9903128cc429c66600baaf1946868cc35cfaf062a3b3"
    ),
    "wall-ball-baseline-inloop@v1": (
        "2b386484cacbe2c669c2efeee7264c79fdf9e91f15b324ef0f9e16af94b845d2"
    ),
    "wall-ball-bootstrap-inloop@v1": (
        "565ce6241dc1430f3d3e84caaa518d3d690399671c71acc57f1c32a5c03a3207"
    ),
    "wall-ball-depth-curriculum-inloop@v1": (
        "74322b2773af39a8a6e35ffcc98798e8cb8a410ddf0803e35cfee489384bb612"
    ),
    "wall-ball-depth-curriculum-aligned-inloop@v1": (
        "f1d601e4f69186929afd66ca1078a9ed7dc30d4044e197744c57aed289acaf5f"
    ),
    # The same evaluation env and settings as the depth curriculum's.
    "wall-ball-goal-rally-inloop@v1": (
        "74322b2773af39a8a6e35ffcc98798e8cb8a410ddf0803e35cfee489384bb612"
    ),
    "wall-ball-true-baseline-inloop@v1": (
        "6cd06c805e48b824d54276729066afc0dc0826086f0b5e9bd66f1276df5f13f8"
    ),
    "wall-ball-volley-inloop@v1": (
        "b4d730b7e42af16ffa5e0f5222d71b390b9a83165e7dfa232050473d9918ea07"
    ),
}

#: Literal (recipe-independent) protocols.
_LITERAL: dict[str, Callable[[], EvalProtocol]] = {
    "paddle-npoint@v2": _paddle_npoint_v2,
    "paddle-npoint@v3": _paddle_npoint_v3,
}

_CACHE: dict[str, EvalProtocol] = {}


def registered_protocols() -> tuple[str, ...]:
    """Every registered protocol id, sorted."""
    return tuple(sorted({*_LITERAL, *RECIPE_PROTOCOLS.values()}))


def _derive_registered(protocol_id: str) -> EvalProtocol:
    import warnings

    from courtside_dynamics.recipes import build_train_config

    recipe = next(name for name, pid in RECIPE_PROTOCOLS.items() if pid == protocol_id)
    with warnings.catch_warnings():
        # The build only resolves the configuration; nothing trains.
        warnings.simplefilter("ignore")
        cfg = build_train_config(recipe, log_dir="protocol-registry", seed=0)
    return from_train_config(
        cfg,
        id=protocol_id,
        description=(
            f"{recipe}'s in-loop info-dict evaluation as train() wires it "
            f"(derived from the recipe; fingerprint pinned at registration)"
        ),
    )


def get_protocol(protocol_id: str) -> EvalProtocol:
    """A registered protocol by id; ``KeyError`` names the registered ones.

    A derived protocol whose recipe no longer reproduces its pinned
    fingerprint raises ``ValueError``: the id would otherwise silently
    name another evaluation.
    """
    if protocol_id in _CACHE:
        return _CACHE[protocol_id]
    if protocol_id in _LITERAL:
        protocol = _LITERAL[protocol_id]()
    elif protocol_id in RECIPE_PROTOCOLS.values():
        protocol = _derive_registered(protocol_id)
        pinned = _PINNED_FINGERPRINTS.get(protocol_id)
        if pinned != protocol.fingerprint:
            raise ValueError(
                f"protocol {protocol_id} was registered with fingerprint "
                f"{pinned}, but its recipe now derives {protocol.fingerprint}: "
                f"the recipe's in-loop evaluation changed. Declare "
                f"{protocol_id} literally and register the new evaluation as "
                f"a new version; never re-pin"
            )
    else:
        raise KeyError(
            f"unknown evaluation protocol {protocol_id!r}; registered: "
            f"{list(registered_protocols())}"
        )
    _CACHE[protocol_id] = protocol
    return protocol


def _registered_for_recipe(recipe: str | None) -> list[str]:
    """The registered ids a run of ``recipe`` may evaluate under.

    The recipe's in-loop protocol first, then the literal protocols
    declared for it (an earlier era, ``paddle-npoint@v2``).
    """
    if recipe is None:
        return []
    ids = [RECIPE_PROTOCOLS[recipe]] if recipe in RECIPE_PROTOCOLS else []
    ids.extend(
        protocol_id
        for protocol_id in sorted(_LITERAL)
        if protocol_id not in ids and get_protocol(protocol_id).recipe == recipe
    )
    return ids


def eval_protocol_record(
    cfg: Any,
    *,
    env_cls: type | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """``config.json``'s ``eval_protocol``: what ``train(cfg)`` selects under.

    ``{"id": ..., "fingerprint": ...}``. The fingerprint is the in-loop
    selection protocol's (:func:`from_train_config`, which takes
    ``env_cls`` and ``env_kwargs`` the same way). The id is the
    registered protocol of the run's recipe with that fingerprint:
    ``paddle-npoint@v3`` for a stock PaddleTennis run,
    ``paddle-npoint@v2`` for one whose ``[eval_env]`` table restores
    the escrows. It is ``None`` when no registered protocol is this
    evaluation (a quick test, other episodes or env settings, an env
    outside the recipes); the fingerprint still names it, so two runs
    compare only when both fields match. The whole record is ``None``
    for a run without an info-dict evaluator, and, with a warning, for
    one whose evaluation env records no constructor kwargs a
    fingerprint can read.
    """
    import warnings

    if not cfg.info_dict_eval:
        return None
    try:
        fingerprint = from_train_config(
            cfg, id="in-loop", env_cls=env_cls, env_kwargs=env_kwargs
        ).fingerprint
    except (TypeError, ValueError) as error:
        warnings.warn(
            f"no evaluation protocol recorded for this run: {error}", stacklevel=2
        )
        return None
    registered: str | None = None
    for protocol_id in _registered_for_recipe(cfg.recipe_name):
        try:
            candidate = get_protocol(protocol_id)
        except ValueError as error:
            # A derived protocol whose recipe moved names nothing now.
            warnings.warn(str(error), stacklevel=2)
            continue
        if candidate.fingerprint == fingerprint:
            registered = protocol_id
            break
    return {"id": registered, "fingerprint": fingerprint}
