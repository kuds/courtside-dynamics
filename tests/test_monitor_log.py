"""Tests for ``load_monitor_episodes``.

The loader exists so plotting code can produce training-reward curves
that interleave the per-worker monitor files by wall-clock time, instead
of the naive concatenation that produces a fake "collapse and re-learn"
artifact (see review of the SAC WallBall run). Key behaviours to lock
down:

1. Episodes from N worker files are interleaved by absolute wall-clock
   time (``t_start + t``), not by file index *or* raw ``t``.  Raw ``t``
   is relative to each worker's own monitor start, so workers launched
   at different times must be offset by their ``t_start`` before
   sorting.
2. The ``cumulative_episode_steps`` column is the cumsum of episode
   lengths in time order.  It counts *finished* episode steps only and
   is NOT equivalent to SB3's ``num_timesteps``.
3. The rolling-mean column smooths over the time-ordered series.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from courtside_dynamics.training import load_monitor_episodes


def _write_monitor_csv(
    path: Path,
    rows: list[tuple[float, int, float]],
    env_id: str = "Test-v0",
    t_start: float = 0.0,
) -> None:
    """Write a synthetic SB3-style monitor.csv at ``path``.

    Format mirrors ``stable_baselines3.common.monitor.Monitor``:
    a leading ``# {json header}`` line followed by a CSV with the
    ``r,l,t`` columns SB3 writes.
    """
    with open(path, "w", newline="") as fh:
        fh.write("#" + json.dumps({"t_start": t_start, "env_id": env_id}) + "\n")
        writer = csv.writer(fh)
        writer.writerow(["r", "l", "t"])
        for r, length, t in rows:
            writer.writerow([r, length, t])


def test_episodes_interleaved_by_wall_clock(tmp_path: Path):
    # Worker 0 finishes its episodes at t = 1, 4, 7 (t_start = 0).
    # Worker 1 finishes its episodes at t = 2, 3, 6 (t_start = 0).
    # Both workers have the same t_start, so t_abs == t.
    # Naive file-order concat would produce 1,4,7,2,3,6 — i.e. the
    # plot's "collapses every N episodes" artifact. The loader must
    # instead produce the time-sorted order 1,2,3,4,6,7.
    _write_monitor_csv(
        tmp_path / "0.monitor.csv",
        rows=[(1.0, 100, 1.0), (2.0, 100, 4.0), (3.0, 100, 7.0)],
        t_start=0.0,
    )
    _write_monitor_csv(
        tmp_path / "1.monitor.csv",
        rows=[(0.5, 50, 2.0), (1.5, 50, 3.0), (2.5, 50, 6.0)],
        t_start=0.0,
    )

    bundle = load_monitor_episodes(str(tmp_path))
    df = bundle.episodes
    assert list(df["t_abs"]) == [1.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    assert list(df["worker_id"]) == [0, 1, 1, 0, 1, 0]
    assert list(df["r"]) == [1.0, 0.5, 1.5, 2.0, 2.5, 3.0]


def test_staggered_start_interleaving(tmp_path: Path):
    """Workers with different t_start values must be sorted by t_abs = t_start + t.

    Worker 0 starts at real time 0 and finishes episodes at raw t = 1, 3.
    Worker 1 starts at real time 5 and finishes episodes at raw t = 0.5, 2.

    In absolute time:
      worker 0: t_abs = 1.0, 3.0
      worker 1: t_abs = 5.5, 7.0

    If the loader sorted on raw ``t``, worker 1's t=0.5 would sort before
    worker 0's t=1.0, giving the wrong order (worker 1 first).  Sorting
    by t_abs gives the correct order: worker 0 both episodes, then
    worker 1 both episodes.
    """
    _write_monitor_csv(
        tmp_path / "0.monitor.csv",
        rows=[(1.0, 100, 1.0), (2.0, 100, 3.0)],
        t_start=0.0,
    )
    _write_monitor_csv(
        tmp_path / "1.monitor.csv",
        rows=[(3.0, 50, 0.5), (4.0, 50, 2.0)],
        t_start=5.0,
    )

    bundle = load_monitor_episodes(str(tmp_path))
    df = bundle.episodes
    # t_abs values: 0+1=1.0, 0+3=3.0, 5+0.5=5.5, 5+2=7.0
    assert list(df["t_abs"]) == [1.0, 3.0, 5.5, 7.0]
    assert list(df["worker_id"]) == [0, 0, 1, 1]
    assert list(df["r"]) == [1.0, 2.0, 3.0, 4.0]


def test_cumulative_episode_steps_and_rolling_mean(tmp_path: Path):
    _write_monitor_csv(
        tmp_path / "0.monitor.csv",
        rows=[(1.0, 10, 0.1), (2.0, 20, 0.3)],
    )
    _write_monitor_csv(
        tmp_path / "1.monitor.csv",
        rows=[(3.0, 30, 0.2), (4.0, 40, 0.4)],
    )

    bundle = load_monitor_episodes(str(tmp_path), rolling_window=2)
    df = bundle.episodes

    # Time-ordered by t_abs (t_start=0 for both): t = 0.1, 0.2, 0.3, 0.4
    # with l = 10, 30, 20, 40.
    assert list(df["cumulative_episode_steps"]) == [10, 40, 60, 100]
    # Rolling window of 2 over r = [1, 3, 2, 4] -> [1.0, 2.0, 2.5, 3.0].
    assert list(df["rolling_mean_r"]) == [1.0, 2.0, 2.5, 3.0]


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_monitor_episodes(str(tmp_path / "does-not-exist"))


def test_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_monitor_episodes(str(tmp_path))
