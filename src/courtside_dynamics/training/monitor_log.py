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
returns a single ``DataFrame`` sorted by ``t`` (the wall-clock seconds
since the run started, recorded by SB3 for every episode). Downstream
plotting code can index by ``t`` (or by cumulative ``timestep``) and get
a curve that reflects how the agents actually progressed in time.
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
        sorted by ``t`` (wall-clock seconds since the worker started).
        Columns include ``r`` (return), ``l`` (length), ``t``, plus
        derived ``worker_id``, ``cumulative_timestep``, and
        ``rolling_mean_r`` / ``rolling_mean_l`` columns.
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
        Episodes sorted by ``t`` plus the per-worker JSON headers.
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
                headers.append(json.loads(first))
            except json.JSONDecodeError:
                headers.append({})
        df = pd.read_csv(path, skiprows=1)
        df["worker_id"] = worker_id
        frames.append(df)

    episodes = pd.concat(frames, ignore_index=True)
    if "t" not in episodes.columns:
        raise ValueError(
            f"Monitor CSVs under {monitor_dir!r} are missing the 't' column; "
            "cannot interleave by wall-clock time."
        )
    episodes = episodes.sort_values("t", kind="mergesort").reset_index(drop=True)
    if "l" in episodes.columns:
        episodes["cumulative_timestep"] = episodes["l"].cumsum().astype("int64")
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
