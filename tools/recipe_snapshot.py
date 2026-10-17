"""Recipe snapshot: every resolved TrainConfig, frozen as canonical JSON.

Part 1 of the SAC freeze (docs/design_ppo_first_class_20261016.md,
Step 0). For every recipe in ``RECIPES`` x {its default algo, ``"SAC"``,
``"DEMOSAC"``, ``"PPO"``} x ``quick_test`` {False, True}, the
``TrainConfig`` that ``build_train_config`` resolves is serialized
canonically and compared with ``tests/golden/recipe_snapshot.json``.
The SAC-family entries freeze the SAC paths; the PPO entries (every
recipe's, since per-family config blocks gave each recipe a PPO
resolution) make each deliberate PPO change a reviewable diff. A
combination ``build_train_config`` refuses is recorded as its refusal,
so a recipe that stops building is a diff, never a silent skip.

Canonical form (no env is constructed, so the whole matrix takes well
under a second):

- the env and eval-env factories are recovered from their closures as
  ``{"__env_factory__", "env_cls", "kwargs"}`` -- the constructor
  kwargs exactly as the factory will pass them;
- other callables and classes are stored by qualified name;
- tuples are JSON arrays; lists, sets, numpy values, dataclasses, enums
  and non-string-keyed mappings carry a ``"__<kind>__"`` tag, so a
  tuple turning into a list (``train_freq``!) is a diff;
- ``log_dir`` is dropped, and any other absolute path refuses to
  serialize (a snapshot must not depend on the machine).

Each build is unseeded, as ``build_train_config``'s defaults resolve it,
with its warnings recorded; where a seeded build (``seed=0``) resolves
differently beyond the seed itself (paired evaluation keeps the recipe's
``eval_reset_options``), that build is recorded too under ``|seed=0``.

The check compares the canonical JSON text of each entry, so a
type-only change (``8`` to ``8.0``, ``True`` to ``1``) is a diff even
though Python's ``==`` would call the values equal. Like the golden
digests, the tool refuses to run when ``courtside_dynamics`` imports
from anywhere but this checkout's ``src/``.

Usage::

    python tools/recipe_snapshot.py            # check (exit 1 on any diff)
    python tools/recipe_snapshot.py --write    # re-record, deliberately
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import json
import os
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools._checkout import checkout_package_mismatch
except ModuleNotFoundError:  # run as a script: tools/ itself is on sys.path
    from _checkout import checkout_package_mismatch  # type: ignore[no-redef]

#: The checkout this tool belongs to.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The committed snapshot.
SNAPSHOT_PATH = REPO_ROOT / "tests" / "golden" / "recipe_snapshot.json"

#: Format tag of the snapshot file; bump it with any canonical-form change.
SNAPSHOT_FORMAT = "courtside-recipe-snapshot/1"

#: TrainConfig fields left out of the snapshot: machine-specific paths.
_DROPPED_FIELDS = frozenset({"log_dir"})

#: The algorithms every recipe is resolved under, besides its default.
_ALGOS = ("SAC", "DEMOSAC", "PPO")


def _qualname(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _env_factory(factory: Callable[..., Any], path: str) -> dict[str, Any]:
    """Recover a recipe env factory's class and constructor kwargs."""
    names = factory.__code__.co_freevars
    if names != ("kwargs", "recipe") or factory.__closure__ is None:
        raise TypeError(
            f"{path}: {_qualname(factory)} is not a recipe env factory "
            f"(free variables {names}); cannot recover its env kwargs"
        )
    cells = dict(zip(names, factory.__closure__, strict=True))
    return {
        "__env_factory__": _qualname(factory),
        "env_cls": _qualname(cells["recipe"].cell_contents.env_cls),
        "kwargs": canonical(cells["kwargs"].cell_contents, f"{path}.kwargs"),
    }


def canonical(value: Any, path: str = "$") -> Any:
    """Serialize ``value`` to canonical, JSON-ready data (see module doc)."""
    if value is None or isinstance(value, bool):
        return value
    # Before str/int: IntEnum and StrEnum members are both.
    if isinstance(value, enum.Enum):
        return {"__enum__": _qualname(type(value)), "name": value.name}
    if isinstance(value, str):
        if os.path.isabs(value):
            raise ValueError(
                f"{path}: absolute path {value!r} would make the snapshot "
                f"machine-specific"
            )
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, np.generic):
        return {"__numpy__": value.dtype.str, "value": canonical(value.item(), path)}
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.dtype.str,
            "shape": list(value.shape),
            "data": canonical(tuple(value.ravel().tolist()), path),
        }
    if isinstance(value, tuple):
        return [canonical(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, list):
        return {
            "__list__": [canonical(item, f"{path}[{i}]") for i, item in enumerate(value)]
        }
    if isinstance(value, (set, frozenset)):
        items = [canonical(item, f"{path}{{}}") for item in value]
        return {"__set__": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
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
            body: Any = {
                key: canonical(item, f"{path}.{key}") for key, item in value.items()
            }
        else:
            pairs = [
                [canonical(key, f"{path}<key>"), canonical(item, f"{path}[{key!r}]")]
                for key, item in value.items()
            ]
            body = {
                "__items__": sorted(
                    pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True)
                )
            }
        if type(value) is not dict:
            return {"__mapping__": _qualname(type(value)), "items": body}
        return body
    if isinstance(value, type):
        return {"__class__": _qualname(value)}
    if isinstance(value, functools.partial):
        return {
            "__partial__": canonical(value.func, f"{path}.func"),
            "args": canonical(value.args, f"{path}.args"),
            "keywords": canonical(value.keywords, f"{path}.keywords"),
        }
    if callable(value) and hasattr(value, "__qualname__"):
        return {"__callable__": _qualname(value)}
    raise TypeError(f"{path}: cannot canonicalize {type(value).__name__}")


def canonical_train_config(cfg: Any) -> dict[str, Any]:
    """Every TrainConfig field but ``log_dir``, canonically serialized."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        if f.name in _DROPPED_FIELDS:
            continue
        value = getattr(cfg, f.name)
        path = f"$.{f.name}"
        if f.name in ("env_fn", "eval_env_fn") and value is not None:
            out[f.name] = _env_factory(value, path)
        else:
            out[f.name] = canonical(value, path)
    return out


def _resolve(
    env_name: str, algo: str, quick_test: bool, seed: int | None
) -> dict[str, Any]:
    from courtside_dynamics.recipes import build_train_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cfg = build_train_config(
                env_name,
                algo=algo,
                log_dir="snapshot-log-dir",
                quick_test=quick_test,
                seed=seed,
            )
        except (ValueError, TypeError, KeyError) as error:
            return {"refused": f"{type(error).__name__}: {error}"}
    return {
        "train_config": canonical_train_config(cfg),
        "warnings": [
            f"{w.category.__name__}: {w.message}" for w in caught
        ],
    }


def build_snapshot() -> dict[str, Any]:
    """The live snapshot entries, keyed ``recipe|algo|quick_test=...``."""
    from courtside_dynamics.recipes import RECIPES

    entries: dict[str, Any] = {}
    for env_name in sorted(RECIPES):
        algos = dict.fromkeys((RECIPES[env_name].default_algo.upper(), *_ALGOS))
        for algo in algos:
            for quick_test in (False, True):
                key = f"{env_name}|{algo}|quick_test={quick_test}"
                unseeded = _resolve(env_name, algo, quick_test, None)
                entries[key] = unseeded
                seeded = _resolve(env_name, algo, quick_test, 0)
                comparable = json.loads(json.dumps(seeded))
                if "train_config" in comparable:
                    comparable["train_config"]["seed"] = None
                if comparable != json.loads(json.dumps(unseeded)):
                    entries[f"{key}|seed=0"] = seeded
    return entries


def _same(expected: Any, actual: Any) -> bool:
    """Equal as canonical JSON text: ``8`` is not ``8.0``, ``True`` not ``1``."""
    return json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)


def _first_differences(
    expected: Any, actual: Any, path: str = "$", limit: int = 8
) -> list[str]:
    if _same(expected, actual):
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if len(out) >= limit:
                break
            if key not in expected:
                out.append(f"{path}.{key}: added {actual[key]!r}")
            elif key not in actual:
                out.append(f"{path}.{key}: removed (was {expected[key]!r})")
            else:
                out += _first_differences(
                    expected[key], actual[key], f"{path}.{key}", limit - len(out)
                )
        return out[:limit]
    return [f"{path}: expected {expected!r}, got {actual!r}"]


def compare(recorded: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    """One readable block per differing snapshot key (empty = identical)."""
    problems: list[str] = []
    for key in sorted(set(recorded) | set(live)):
        if key not in live:
            problems.append(f"{key}: recorded but no longer resolved")
        elif key not in recorded:
            problems.append(f"{key}: resolved but never recorded")
        elif not _same(recorded[key], live[key]):
            lines = _first_differences(recorded[key], live[key])
            problems.append("\n".join([f"{key}:", *(f"  {line}" for line in lines)]))
    return problems


#: Printed with every snapshot diff, by the tool and the test.
DIFF_EXPLANATION = (
    "The resolved TrainConfig of a recipe changed. Every SAC training path "
    "must stay bit-identical, and a PPO path may change only where a step "
    "declares it: if it is intended, name it in the CHANGELOG (and "
    "docs/DECISIONS.md for a design decision) and re-record in the same "
    "commit with `python tools/recipe_snapshot.py --write`. Never "
    "re-record to make this check pass."
)


def load_snapshot(path: Path) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(
            f"{path}: format {snapshot.get('format')!r}, expected {SNAPSHOT_FORMAT!r}"
        )
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare (default)")
    mode.add_argument("--write", action="store_true", help="re-record")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args(argv)
    problem = checkout_package_mismatch(REPO_ROOT)
    if problem is not None:
        raise SystemExit(f"recipe_snapshot: {problem}")
    live = build_snapshot()
    if args.write:
        old = (
            load_snapshot(args.snapshot)["entries"] if args.snapshot.exists() else {}
        )
        for problem in compare(old, live):
            print(f"re-recorded {problem}")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(
            json.dumps(
                {"format": SNAPSHOT_FORMAT, "note": DIFF_EXPLANATION, "entries": live},
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(live)} entries to {args.snapshot}")
        return 0
    problems = compare(load_snapshot(args.snapshot)["entries"], live)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(DIFF_EXPLANATION, file=sys.stderr)
        return 1
    print(f"{len(live)} recipe resolutions match {args.snapshot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
