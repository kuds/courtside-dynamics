"""Shared ``info``-dict filtering for the logging callbacks.

Both :class:`~courtside_dynamics.callbacks.video_record.VideoRecordCallback`
and :class:`~courtside_dynamics.callbacks.info_dict_eval.InfoDictEvalCallback`
need to decide which keys of a step ``info`` dict are env-authored scalars
worth logging. Keeping that logic in one neutral module avoids one callback
importing a private helper from the other.
"""
from __future__ import annotations

import numbers
from collections.abc import Mapping

import numpy as np

#: Keys that SB3 / gymnasium wrappers inject into ``info`` and that we
#: don't want showing up as training diagnostics. Anything with a ``.``
#: in its name is also filtered out (e.g. ``TimeLimit.truncated``).
_WRAPPER_INFO_KEYS = frozenset({"terminal_observation", "episode"})


def _scalar_info_keys(info: Mapping) -> list[str]:
    """Return the sorted env-authored scalar keys of ``info``.

    Scalars are Python numbers/booleans or numpy scalar arrays (0-D). The
    set excludes arrays/sequences so the auto-logger doesn't emit
    unbounded-width rows, and wrapper-injected keys (e.g.
    ``TimeLimit.truncated``) so the diagnostics only surface metrics the
    env itself emits. Keys are returned in sorted order so CSV column
    ordering is deterministic across runs.
    """
    keys: list[str] = []
    for key, value in info.items():
        name = str(key)
        if name in _WRAPPER_INFO_KEYS or "." in name:
            continue
        if isinstance(value, (bool, numbers.Number)):
            keys.append(name)
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            keys.append(name)
    return sorted(keys)
