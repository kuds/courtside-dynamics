"""Helpers for reading SB3 ``Monitor`` CSV logs from a vectorised run.

When ``stable_baselines3.common.env_util.make_vec_env`` is given a
``monitor_dir``, every parallel worker writes its own ``<i>.monitor.csv``
file. The episodes inside one file are in chronological order *for that
worker*, but the per-worker files share no timeline with each other:
naively concatenating them ("file 0 first, then file 1, ...") produces a
training-reward curve that looks like the agent learns, collapses, and
re-learns once per worker — it's a plotting artifact, not a training
collapse.

``load_monitor_episodes`` reads every monitor file under a directory and
returns a single ``DataFrame`` sorted by wall-clock time. Each worker's
``t`` column holds *seconds elapsed since that worker's monitor opened*
(``t_start`` in the JSON header). Workers may be launched at different
real-world times, so the correct ordering key is ``t_start + t``, stored
in the ``t_abs`` column. Downstream plotting code can index by ``t_abs``
to get a curve that reflects how the agents actually progressed in time.

.. note:: ``cumulative_episode_steps`` counts only the steps in episodes
   that have already *finished*. For a vectorised run with N workers,
   SB3's internal ``num_timesteps`` also includes in-progress steps on
   other workers, so ``cumulative_episode_steps`` will under-count the
   real training-step axis by up to ``(N-1) * max_episode_length``. Use
   ``cumulative_episode_steps`` for plotting purposes only; do not treat
   it as equivalent to SB3's ``num_timesteps``.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class MonitorBundle:
    """A flattened view of every monitor CSV under a directory.

    Attributes
    ----------
    episodes:
        Per-episode rows (one per finished episode, across all workers),
        sorted by ``t_abs`` (absolute wall-clock seconds, ``t_start + t``).
        Columns include ``r`` (return), ``l`` (length), ``t``,
        ``t_abs``, plus derived ``worker_id``,
        ``cumulative_episode_steps``, and
        ``rolling_mean_r`` / ``rolling_mean_l`` columns.

        ``cumulative_episode_steps`` is the cumulative sum of ``l`` over
        finished episodes in wall-clock order. It is *not* equivalent to
        SB3's ``num_timesteps`` (which also counts in-progress steps on
        other workers); use it for plotting only.
    headers:
        Per-worker JSON headers as written by SB3's ``Monitor`` (one per
        file). Useful for recovering the env id and the per-worker start
        time if needed.
    """

    episodes: pd.DataFrame
    headers: list[dict[str, Any]]


def load_monitor_episodes(
    monitor_dir: str,
    rolling_window: int = 25,
) -> MonitorBundle:
    """Load every ``*.monitor.csv`` under ``monitor_dir`` into one frame.

    Parameters
    ----------
    monitor_dir:
        Directory containing one ``<n>.monitor.csv`` per worker, as
        produced by ``stable_baselines3.common.monitor.Monitor`` /
        ``make_vec_env(monitor_dir=...)``.
    rolling_window:
        Window (in episodes, ordered by wall-clock time) used for the
        derived ``rolling_mean_r`` and ``rolling_mean_l`` columns.

    Returns
    -------
    MonitorBundle
        Episodes sorted by ``t_abs`` (absolute wall-clock time,
        ``t_start + t``) plus the per-worker JSON headers.
    """
    paths = sorted(glob.glob(os.path.join(monitor_dir, "*.monitor.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No *.monitor.csv files found under {monitor_dir!r}"
        )

    frames: list[pd.DataFrame] = []
    headers: list[dict[str, Any]] = []
    for worker_id, path in enumerate(paths):
        with open(path) as fh:
            first = fh.readline().lstrip("#")
            try:
                header = json.loads(first)
            except json.JSONDecodeError:
                header = {}
            headers.append(header)
        t_start: float = float(header.get("t_start", 0.0))
        df = pd.read_csv(path, skiprows=1)
        df["worker_id"] = worker_id
        # Compute absolute wall-clock time so that workers launched at
        # different real-world times are sorted correctly. SB3 writes
        # ``t`` as seconds since *this worker's* monitor opened, so two
        # workers with different ``t_start`` values cannot be compared
        # on raw ``t`` alone.
        df["t_abs"] = t_start + df["t"]
        frames.append(df)

    episodes = pd.concat(frames, ignore_index=True)
    episodes = episodes.sort_values("t_abs", kind="mergesort").reset_index(drop=True)
    if "l" in episodes.columns:
        # cumulative_episode_steps counts only finished-episode steps —
        # NOT SB3's num_timesteps which also includes in-progress steps
        # on other workers. Use this column for plotting only.
        episodes["cumulative_episode_steps"] = (
            episodes["l"].cumsum().astype("int64")
        )
    if "r" in episodes.columns:
        episodes["rolling_mean_r"] = (
            episodes["r"]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
        )
    if "l" in episodes.columns:
        episodes["rolling_mean_l"] = (
            episodes["l"]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
        )
    return MonitorBundle(episodes=episodes, headers=headers)
