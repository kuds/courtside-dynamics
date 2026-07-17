"""TOML run-configuration files for :func:`recipes.build_train_config`.

One file describes one experiment (docs/run_config_file_spec.md). Three
optional top-level tables:

- ``[train]``: :class:`~courtside_dynamics.training.train.TrainConfig`
  fields. Mapping-valued fields (``model_kwargs``, ``phase_labels``,
  ``info_eval_survival_thresholds``) deep-merge one level onto the
  recipe's values; ``performance_gate`` replaces wholesale (an ordered
  stage ladder has no unambiguous element-wise merge); everything else
  replaces.
- ``[env]``: environment constructor kwargs, applied to the training
  AND evaluation environments (like the recipe's own ``env_kwargs``
  layer, so a physics tweak cannot silently split the two).
- ``[eval_env]``: evaluation-only kwargs, applied after the recipe's
  ``eval_env_overrides`` re-assert the canonical evaluation setup.

Precedence overall: recipe < file < ``quick_test`` presets < explicit
``build_train_config`` keyword arguments.

TOML has no ``None``; the sentinel string ``"none"`` (any case) converts
to ``None`` recursively throughout every table (including nested tables
such as ``model_kwargs.policy_kwargs`` and ``performance_gate`` stages).
Two exceptions: ``phase_labels`` values are display text (a phase
legitimately named ``"none"`` stays one; its keys -- TOML forces
strings -- convert strictly to ``int``, and values must be strings),
and a ``[train]`` key whose ``TrainConfig`` field is not Optional
(``n_envs``, ``model_kwargs``, ``record_video``, ...) rejects the
sentinel outright -- the smuggled ``None`` would only crash
mid-``train()`` or, worse for falsy-checked flags, silently disable
the feature while ``config.json`` records ``null``.
The quoted strings ``"true"``/``"false"`` are rejected recursively --
as Python strings both are truthy, silently enabling exactly what the
file tried to disable.

Everything fails loudly -- this repository's run history is a catalog of
silent no-ops: unknown tables and unknown/forbidden ``[train]`` keys
raise with did-you-mean suggestions rather than being ignored.
"""

from __future__ import annotations

import difflib
import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_ALLOWED_TABLES = ("train", "env", "eval_env")

# TrainConfig fields a file may never set: callables and runtime-only
# wiring belong to code, and builder-owned identity fields are written
# into the kwargs BEFORE the file merges, so a file value would silently
# invert the documented "file < explicit arguments" precedence and
# record false provenance (an algo= run relabeled, artifacts redirected
# away from the mandatory log_dir=, config.json naming a recipe that
# never ran). warm_start is data but needs a from-mapping constructor
# first (spec: v2).
_REJECTED_TRAIN_KEYS = {
    "env_fn": "environment factories are code; use [env] for kwargs",
    "eval_env_fn": "evaluation factories are code; use [eval_env]",
    "extra_callbacks": "callbacks are code and cannot be described in TOML",
    "info_row_fn": "callbacks are code and cannot be described in TOML",
    "warm_start": "not file-configurable until v2 grows a from-mapping "
    "constructor",
    "run_config_file": "set by the loader itself",
    "algo": "pass algo= to build_train_config; the recipe's name_prefix "
    "derives from it before the file merges",
    "log_dir": "pass log_dir= to build_train_config; a file must not "
    "silently redirect run artifacts",
    "recipe_name": "recorded from the recipe actually built",
}

# Mapping-valued TrainConfig fields that deep-merge one level onto the
# recipe's value. performance_gate is deliberately absent: it replaces
# wholesale.
DEEP_MERGE_TRAIN_KEYS = (
    "model_kwargs",
    "phase_labels",
    "info_eval_survival_thresholds",
)


@dataclass(frozen=True)
class RunFileConfig:
    """A parsed run-configuration file plus its provenance identity."""

    path: str
    sha256: str
    text: str
    train: dict[str, Any]
    env: dict[str, Any]
    eval_env: dict[str, Any]
    raw: dict[str, Any]


def _convert_none_sentinels(value: Any) -> Any:
    """Recursively convert the ``"none"`` sentinel to ``None``.

    Descends into tables and into tables nested inside arrays (e.g.
    ``performance_gate`` stages); scalar array elements are left alone
    -- a tuple-valued env kwarg has no meaningful ``None`` element.
    Always builds fresh containers, so ``raw`` keeps the file verbatim.
    """
    if isinstance(value, str) and value.lower() == "none":
        return None
    if isinstance(value, dict):
        return {key: _convert_none_sentinels(v) for key, v in value.items()}
    if isinstance(value, list):
        return [
            _convert_none_sentinels(v) if isinstance(v, (dict, list)) else v
            for v in value
        ]
    return value


def _reject_quoted_booleans(value: Any, path: str, context: str) -> None:
    # ``normalize_reward = "false"`` is a truthy string that would
    # silently enable the exact feature the user tried to disable.
    # Recursive for the same reason the sentinel is: a nested
    # ``policy_kwargs.ortho_init = "false"`` is the same footgun.
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item.lower() in ("true", "false"):
                raise ValueError(
                    f"{path}: [{context}] key '{key}' is the quoted "
                    f"string {item!r}; use a bare TOML boolean "
                    f"({item.lower()})"
                )
            _reject_quoted_booleans(item, path, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_quoted_booleans(item, path, f"{context}[{index}]")


def _convert_phase_labels(labels: Any, path: str) -> dict[int, Any]:
    """Convert TOML's string keys to the env's integer phase ids.

    Label *values* are exempt from the ``"none"`` sentinel and the
    quoted-boolean rejection: a label is a display string, and a phase
    legitimately named ``"none"`` (or ``"false"``) must stay one.
    """
    if not isinstance(labels, dict):
        raise ValueError(f"{path}: [train.phase_labels] must be a table")
    converted: dict[int, Any] = {}
    for key, value in labels.items():
        # Strict digits only: bare int() also accepts "1_0" (-> 10),
        # "+2", and " 3 ", silently relabeling a different phase.
        if not (isinstance(key, str) and key.isascii() and key.isdigit()):
            raise ValueError(
                f"{path}: [train.phase_labels] key {key!r} is not an "
                f"integer phase id"
            )
        int_key = int(key)
        if not isinstance(value, str):
            # A non-string (a mistyped dotted key making a sub-table, a
            # number) would end up interpolated into TensorBoard tag
            # names as garbage instead of failing here.
            raise ValueError(
                f"{path}: [train.phase_labels] value for phase id "
                f"{int_key} must be a string label, got {value!r}"
            )
        if int_key in converted:
            raise ValueError(
                f"{path}: [train.phase_labels] keys collide at phase id "
                f"{int_key} after integer conversion"
            )
        converted[int_key] = value
    return converted


_GATE_REQUIRED_KEYS = ("metric_key", "threshold", "sustain_evals", "stages")


def _validate_performance_gate(gate: Any, path: str) -> None:
    """Check the gate so a broken one fails here, with the file named,
    instead of inside ``train()`` after environments and loggers have
    spun up. ``train()`` reads exactly the four required keys, so an
    unknown key is a silent no-op -- rejected like unknown ``[train]``
    keys. (Deep semantic validation -- metric existence, stage attrs --
    stays with ``PerformanceGatedEnvStagesCallback``.)"""
    if not isinstance(gate, dict):
        raise ValueError(
            f"{path}: [train.performance_gate] must be a table (or the "
            f'"none" sentinel to disable the recipe\'s gate)'
        )
    missing = [key for key in _GATE_REQUIRED_KEYS if key not in gate]
    if missing:
        raise ValueError(
            f"{path}: [train.performance_gate] replaces the recipe's gate "
            f"wholesale and must define {missing}"
        )
    unknown = sorted(set(gate) - set(_GATE_REQUIRED_KEYS))
    if unknown:
        suggestions = difflib.get_close_matches(
            unknown[0], _GATE_REQUIRED_KEYS, n=3
        )
        hint = (
            f"; did you mean {', '.join(map(repr, suggestions))}?"
            if suggestions
            else ""
        )
        raise ValueError(
            f"{path}: unknown [train.performance_gate] key(s) "
            f"{unknown}{hint} (train() reads exactly "
            f"{list(_GATE_REQUIRED_KEYS)}, so anything else would be "
            f"silently ignored)"
        )
    if not isinstance(gate["metric_key"], str):
        raise ValueError(
            f"{path}: [train.performance_gate] metric_key must be a "
            f"string, got {gate['metric_key']!r}"
        )
    threshold = gate["threshold"]
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise ValueError(
            f"{path}: [train.performance_gate] threshold must be a "
            f"number, got {threshold!r}"
        )
    sustain = gate["sustain_evals"]
    if isinstance(sustain, bool) or not isinstance(sustain, int):
        raise ValueError(
            f"{path}: [train.performance_gate] sustain_evals must be an "
            f"integer, got {sustain!r}"
        )
    stages = gate["stages"]
    if (
        not isinstance(stages, list)
        or not stages
        or not all(isinstance(stage, dict) and stage for stage in stages)
    ):
        raise ValueError(
            f"{path}: [train.performance_gate.stages] must be a non-empty "
            f"array of non-empty tables"
        )


def _train_field_names() -> tuple[str, ...]:
    # Imported lazily: train.py pulls in SB3/torch, which a config-only
    # caller (or a docs build) should not pay for at module import time.
    from courtside_dynamics.training.train import TrainConfig

    return tuple(field.name for field in fields(TrainConfig))


def _optional_train_fields() -> frozenset[str]:
    """TrainConfig fields whose annotation admits ``None``.

    Only these may take the ``"none"`` sentinel: for any other field a
    smuggled ``None`` either crashes mid-``train()`` with the file
    unnamed (``dict(None)``, ``range(None)``) or -- worse, for flags
    read as ``if cfg.record_video`` -- silently disables the feature
    while ``config.json`` records ``null``.
    """
    import types
    import typing

    from courtside_dynamics.training.train import TrainConfig

    optional = set()
    for name, hint in typing.get_type_hints(TrainConfig).items():
        origin = typing.get_origin(hint)
        if origin in (typing.Union, types.UnionType) and type(
            None
        ) in typing.get_args(hint):
            optional.add(name)
    return frozenset(optional)


def _validate_train_table(table: dict[str, Any], path: str) -> None:
    valid = _train_field_names()
    settable = [name for name in valid if name not in _REJECTED_TRAIN_KEYS]
    for key in table:
        if key in _REJECTED_TRAIN_KEYS:
            raise ValueError(
                f"{path}: [train] key '{key}' is not file-configurable "
                f"({_REJECTED_TRAIN_KEYS[key]})"
            )
        if key not in valid:
            suggestions = difflib.get_close_matches(key, settable, n=3)
            hint = (
                f"; did you mean {', '.join(map(repr, suggestions))}?"
                if suggestions
                else ""
            )
            raise ValueError(
                f"{path}: unknown [train] key '{key}'{hint} (valid keys "
                f"are TrainConfig fields)"
            )


def load_run_config(path: str | Path) -> RunFileConfig:
    """Parse and validate a run-configuration file.

    Raises ``FileNotFoundError`` for a missing file (never silently
    skipped), ``tomllib.TOMLDecodeError`` wrapped with the path for
    malformed TOML, and ``ValueError`` for unknown tables or unknown /
    forbidden ``[train]`` keys.
    """
    resolved = Path(path)
    data = resolved.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{resolved}: not valid UTF-8 (TOML files must be UTF-8): "
            f"{error}"
        ) from error
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        # TOMLDecodeError's constructor signature differs across Python
        # versions; wrap with the path in a plain ValueError (its base
        # class) and chain the original for the parse position.
        raise ValueError(f"{resolved}: invalid TOML: {error}") from error

    unknown_tables = sorted(set(raw) - set(_ALLOWED_TABLES))
    if unknown_tables:
        raise ValueError(
            f"{resolved}: unknown table(s) {unknown_tables}; a run config "
            f"may only contain {list(_ALLOWED_TABLES)}"
        )
    for name in _ALLOWED_TABLES:
        if name in raw and not isinstance(raw[name], dict):
            raise ValueError(f"{resolved}: [{name}] must be a table")

    train = dict(raw.get("train", {}))
    _validate_train_table(train, str(resolved))
    # phase_labels is handled apart from the recursive passes: its
    # values are display text, exempt from both the sentinel and the
    # quoted-boolean rejection.
    has_phase_labels = "phase_labels" in train
    phase_labels = train.pop("phase_labels", None)
    train = _convert_none_sentinels(train)
    _reject_quoted_booleans(train, str(resolved), "train")
    optional_fields = _optional_train_fields()
    for key, value in train.items():
        if value is None and key not in optional_fields:
            raise ValueError(
                f'{resolved}: [train] {key} = "none" is not supported: '
                f"TrainConfig.{key} is never None, so the smuggled None "
                f"would only crash or silently misconfigure "
                f"mid-train(); omit the key to keep the recipe's value"
            )
    for key in DEEP_MERGE_TRAIN_KEYS:
        if key == "phase_labels" or key not in train:
            continue
        if not isinstance(train[key], dict):
            raise ValueError(f"{resolved}: [train.{key}] must be a table")
    thresholds = train.get("info_eval_survival_thresholds")
    if isinstance(thresholds, dict):
        for tkey, tval in thresholds.items():
            if tval is None:
                # A threshold value is a sequence of step counts; None
                # cannot remove a recipe key under deep-merge and would
                # only fail inside train().
                raise ValueError(
                    f"{resolved}: [train.info_eval_survival_thresholds] "
                    f'{tkey} = "none" is not supported; thresholds are '
                    f"step-count arrays"
                )
    if has_phase_labels:
        if (
            isinstance(phase_labels, str)
            and phase_labels.lower() == "none"
        ):
            train["phase_labels"] = None
        else:
            train["phase_labels"] = _convert_phase_labels(
                phase_labels, str(resolved)
            )
    gate = train.get("performance_gate")
    if gate is not None:
        _validate_performance_gate(gate, str(resolved))

    env = _convert_none_sentinels(dict(raw.get("env", {})))
    _reject_quoted_booleans(env, str(resolved), "env")
    eval_env = _convert_none_sentinels(dict(raw.get("eval_env", {})))
    _reject_quoted_booleans(eval_env, str(resolved), "eval_env")

    return RunFileConfig(
        path=str(resolved),
        sha256=hashlib.sha256(data).hexdigest(),
        text=text,
        train=train,
        env=env,
        eval_env=eval_env,
        raw=raw,
    )


def merge_train_overrides(
    base: dict[str, Any], file_train: dict[str, Any]
) -> dict[str, Any]:
    """Apply a file's ``[train]`` table onto accumulated config kwargs.

    Scalars and sequences replace; the mapping fields in
    :data:`DEEP_MERGE_TRAIN_KEYS` merge one level so a file can tweak a
    single ``model_kwargs`` key without silently discarding the rest of
    a recipe's calibrated bundle -- the exact footgun explicit keyword
    arguments have (and keep, for backward compatibility).
    """
    merged = dict(base)
    for key, value in file_train.items():
        if (
            key in DEEP_MERGE_TRAIN_KEYS
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
