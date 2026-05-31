"""Tests for ``load_monitor_episodes``.

The loader exists so plotting code can produce training-reward curves
that interleave the per-worker monitor files by wall-clock time, instead
of the naive concatenation that produces a fake "collapse and re-learn"
artifact. Key behaviours to lock down:

1. Episodes from N worker files are interleaved by absolute wall-clock
   time (``t_start + t``), not by file index *or* raw ``t``. Raw ``t``
   is relative to each worker's own monitor start, so workers launched
   at different times must be offset by their ``t_start`` before
   sorting.
2. ``cumulative_episode_steps`` is the cumsum of episode lengths in
   time order. It counts *finished* episode steps only and is NOT
   equivalent to SB3's ``num_timesteps``.
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
    """Write a synthetic SB3-style monitor.csv at ``path``."""
    with open(path, "w", newline="") as fh:
        fh.write("#" + json.dumps({"t_start": t_start, "env_id": env_id}) + "\n")
        writer = csv.writer(fh)
        writer.writerow(["r", "l", "t"])
        for r, length, t in rows:
            writer.writerow([r, length, t])


def test_episodes_interleaved_by_wall_clock(tmp_path: Path):
    # Worker 0 finishes at t = 1, 4, 7. Worker 1 at t = 2, 3, 6.
    # Naive file-order concat gives 1,4,7,2,3,6 — the artifact. The
    # loader must produce time-sorted 1,2,3,4,6,7.
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
    """Workers with different t_start values must be sorted by t_abs.

    If the loader sorted on raw ``t`` instead, worker 1's t=0.5 would
    sort before worker 0's t=1.0. Sorting by t_abs = t_start + t gives
    the correct order.
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

    # Time-ordered by t_abs: t = 0.1, 0.2, 0.3, 0.4 with l = 10, 30, 20, 40.
    assert list(df["cumulative_episode_steps"]) == [10, 40, 60, 100]
    # Rolling window 2 over r = [1, 3, 2, 4] -> [1.0, 2.0, 2.5, 3.0].
    assert list(df["rolling_mean_r"]) == [1.0, 2.0, 2.5, 3.0]


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_monitor_episodes(str(tmp_path / "does-not-exist"))


def test_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_monitor_episodes(str(tmp_path))


def test_header_only_files_raise_filenotfound(tmp_path: Path):
    """A monitor file that exists but holds no episode rows (run inspected
    before the first episode finished) must be skipped, not crash with a
    pandas EmptyDataError. With no usable data it reads as 'no logs'."""
    # Only the SB3 ``#{json}`` metadata line, no column header or rows.
    (tmp_path / "0.monitor.csv").write_text(
        "#" + json.dumps({"t_start": 0.0, "env_id": "Test-v0"}) + "\n"
    )
    with pytest.raises(FileNotFoundError):
        load_monitor_episodes(str(tmp_path))


def test_header_only_file_skipped_when_another_has_data(tmp_path: Path):
    """A header-only worker file is skipped; a sibling with real episodes
    still loads."""
    (tmp_path / "0.monitor.csv").write_text(
        "#" + json.dumps({"t_start": 0.0}) + "\n"
    )
    _write_monitor_csv(
        tmp_path / "1.monitor.csv",
        rows=[(1.0, 10, 0.1), (2.0, 20, 0.2)],
        t_start=0.0,
    )
    bundle = load_monitor_episodes(str(tmp_path))
    assert list(bundle.episodes["r"]) == [1.0, 2.0]


def test_read_monitor_rewards_lengths(tmp_path: Path):
    """The shared helper returns time-ordered (rewards, lengths) lists, and
    empty lists (not an exception) when no usable logs exist."""
    from courtside_dynamics.training.monitor_log import (
        read_monitor_rewards_lengths,
    )

    assert read_monitor_rewards_lengths(str(tmp_path)) == ([], [])

    _write_monitor_csv(
        tmp_path / "0.monitor.csv",
        rows=[(1.0, 10, 0.2)],
        t_start=0.0,
    )
    _write_monitor_csv(
        tmp_path / "1.monitor.csv",
        rows=[(3.0, 30, 0.1)],
        t_start=0.0,
    )
    rewards, lengths = read_monitor_rewards_lengths(str(tmp_path))
    # Time-ordered by t_abs: worker 1's t=0.1 before worker 0's t=0.2.
    assert rewards == [3.0, 1.0]
    assert lengths == [30, 10]
