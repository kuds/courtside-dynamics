"""Evaluation snapshots: the policy every in-loop evaluation scored.

``docs/design_campaign_structure_20261016.md`` step L2 ("snapshots at
every eval cadence"). With ``TrainConfig.snapshot_evals`` on, ``train()``
saves the model and its ``VecNormalize`` statistics at every
evaluation, in the format ``best_model.zip`` / ``best_vec_normalize.pkl``
use (``model.save`` and ``VecNormalize.save``; no replay buffer):

``snapshots/snapshot_<N>_steps.zip``
    The policy at ``N`` env steps.
``snapshots/snapshot_vecnormalize_<N>_steps.pkl``
    Its observation statistics (absent for a run without
    ``VecNormalize``). The ``<prefix>_<N>_steps.zip`` /
    ``<prefix>_vecnormalize_<N>_steps.pkl`` naming is SB3's
    ``CheckpointCallback``'s, so one parser reads both layouts.
``snapshots/index.json`` (:data:`SNAPSHOT_INDEX_SCHEMA`)
    One entry per snapshot binding the pair by sha256, plus any
    snapshot that could not be written.

Every snapshot is written the way the best model is
(``records.ReplaceTogether``): both files and the new index go to
temporaries, are hashed, and are renamed into place, the index last,
so the index only ever names complete files; an interrupt leaves the
previous index whole. An interrupt between the renames leaves complete
snapshot files the index does not bind (and, at the first snapshot,
no index at all): :func:`unindexed_files` names them, the run record's
``snapshots`` block lists them, ``verify`` calls the directory not
clean, and the harness refuses to score them. The offline harness
(``experiments.harness.rescore``) re-scores any of them under any
protocol, so a new instrument means re-scoring, not retraining.

Stable-Baselines3 and torch are never imported here: the writer takes
the two save calls as callables.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtside_dynamics.experiments.records import (
    ReplaceTogether,
    describe_read_error,
    is_temporary_path,
    read_json_object,
    sha256_file,
    write_json_atomic,
)

#: ``snapshots/index.json``'s schema tag.
SNAPSHOT_INDEX_SCHEMA = "courtside-snapshots/1"

#: The directory at the run root (``training.artifacts.RUN_LAYOUT``'s
#: ``snapshots_dir``; a test pins the agreement) and its index.
SNAPSHOTS_DIR_NAME = "snapshots"
SNAPSHOT_INDEX_NAME = "index.json"

#: The file-name prefix of every snapshot pair.
SNAPSHOT_PREFIX = "snapshot"

#: ``<prefix>_<N>_steps.zip``: a snapshot's, or a periodic checkpoint's.
MODEL_FILE_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<step>\d+)_steps\.zip$")


def model_file_name(timestep: int, prefix: str = SNAPSHOT_PREFIX) -> str:
    """``<prefix>_<timestep>_steps.zip`` (SB3's checkpoint naming)."""
    return f"{prefix}_{int(timestep)}_steps.zip"


def normalizer_file_name(timestep: int, prefix: str = SNAPSHOT_PREFIX) -> str:
    """``<prefix>_vecnormalize_<timestep>_steps.pkl`` (SB3's naming)."""
    return f"{prefix}_vecnormalize_{int(timestep)}_steps.pkl"


@dataclass(frozen=True)
class SnapshotEntry:
    """One snapshot pair as ``index.json`` binds it.

    ``model`` and ``normalizer`` are file names inside the snapshots
    directory; ``normalizer`` is ``None`` for a run without
    ``VecNormalize``. ``n_updates`` is SB3's ``_n_updates`` when the
    snapshot was taken (an evaluation runs before its rollout's update
    on-policy, so a PPO snapshot at 0 is the initial policy).
    """

    timestep: int
    n_updates: int | None
    model: str
    model_sha256: str
    normalizer: str | None
    normalizer_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "n_updates": self.n_updates,
            "model": {"path": self.model, "sha256": self.model_sha256},
            "normalizer": (
                None
                if self.normalizer is None
                else {"path": self.normalizer, "sha256": self.normalizer_sha256}
            ),
        }


def _index_payload(
    run_id: str | None,
    entries: list[SnapshotEntry],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_INDEX_SCHEMA,
        "run_id": run_id,
        "snapshots": [entry.to_dict() for entry in entries],
        "failures": list(failures),
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class SnapshotWriter:
    """Write evaluation snapshots and keep ``index.json`` current.

    Refuses a directory that already holds an index (a previous
    attempt's snapshots: ``train()`` rotates those aside under
    ``reuse_log_dir`` and refuses the directory otherwise), so every
    entry is this run's.
    """

    def __init__(self, directory: str | os.PathLike[str], *, run_id: str | None) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.entries: list[SnapshotEntry] = []
        self.failures: list[dict[str, Any]] = []

    @property
    def index_path(self) -> Path:
        return self.directory / SNAPSHOT_INDEX_NAME

    def open(self) -> None:
        """Create the directory; refuse one that holds another run's index."""
        if self.index_path.exists():
            raise FileExistsError(
                f"{self.index_path} already exists: the snapshots of a "
                f"previous attempt would be listed as this run's. Use a "
                f"fresh log_dir, or reuse_log_dir=True to rotate them aside"
            )
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        timestep: int,
        *,
        save_model: Callable[[str], None],
        save_normalizer: Callable[[str], None] | None = None,
        n_updates: int | None = None,
    ) -> SnapshotEntry:
        """Save one snapshot pair and the index binding it, atomically.

        ``save_model(path)`` and ``save_normalizer(path)`` write the files
        (``model.save`` and ``VecNormalize.save``); each writes to a
        temporary beside its target, the temporaries are hashed, and the
        model, the normalizer and then the index are renamed into place.
        A failure leaves the previous index and no temporary; it raises,
        and the caller decides whether a snapshot is worth a run
        (``train()`` logs an ``OSError`` and continues, :meth:`note_failure`).
        """
        timestep = int(timestep)
        if any(entry.timestep == timestep for entry in self.entries):
            raise ValueError(f"a snapshot at {timestep} steps was already written")
        model_name = model_file_name(timestep)
        normalizer_name = (
            None if save_normalizer is None else normalizer_file_name(timestep)
        )
        with ReplaceTogether() as staged:
            model_temporary = staged.stage(self.directory / model_name)
            save_model(model_temporary)
            normalizer_temporary: str | None = None
            if save_normalizer is not None:
                assert normalizer_name is not None
                normalizer_temporary = staged.stage(
                    self.directory / normalizer_name
                )
                save_normalizer(normalizer_temporary)
            entry = SnapshotEntry(
                timestep=timestep,
                n_updates=n_updates,
                model=model_name,
                model_sha256=sha256_file(model_temporary),
                normalizer=normalizer_name,
                normalizer_sha256=(
                    None
                    if normalizer_temporary is None
                    else sha256_file(normalizer_temporary)
                ),
            )
            _write_json(
                staged.stage(self.index_path),
                _index_payload(self.run_id, [*self.entries, entry], self.failures),
            )
            staged.commit()
        self.entries.append(entry)
        return entry

    def note_failure(self, timestep: int, error: BaseException) -> None:
        """Record a snapshot that could not be written, in the index too.

        The index is rewritten (atomically) with the failure listed, so
        a missing snapshot is never silent; if even that write fails,
        the failure is printed and the next successful snapshot's index
        carries it.
        """
        self.failures.append(
            {"timestep": int(timestep), "error": describe_read_error(error)}
        )
        try:
            write_json_atomic(
                self.index_path,
                _index_payload(self.run_id, self.entries, self.failures),
            )
        except OSError as index_error:
            print(
                f"[snapshots] could not record the failed snapshot at "
                f"{timestep} steps in {self.index_path}: {index_error!r}"
            )


def _entry_from_dict(raw: Any, where: str) -> SnapshotEntry:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: an entry is not an object: {raw!r}")
    timestep = raw.get("timestep")
    n_updates = raw.get("n_updates")
    model = raw.get("model")
    normalizer = raw.get("normalizer")
    if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
        raise ValueError(f"{where}: malformed timestep {timestep!r}")
    if n_updates is not None and (
        isinstance(n_updates, bool) or not isinstance(n_updates, int)
    ):
        raise ValueError(f"{where}: malformed n_updates {n_updates!r}")

    def _file(value: Any, label: str) -> tuple[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{where}: {label} at {timestep} is not an object")
        path, digest = value.get("path"), value.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or "/" in path
            or "\\" in path
            or path in (".", "..")
        ):
            raise ValueError(f"{where}: {label} path {path!r} is not a file name")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{where}: {label} at {timestep} has no sha256")
        return path, digest

    model_path, model_digest = _file(model, "model")
    normalizer_path: str | None = None
    normalizer_digest: str | None = None
    if normalizer is not None:
        normalizer_path, normalizer_digest = _file(normalizer, "normalizer")
    return SnapshotEntry(
        timestep=timestep,
        n_updates=n_updates,
        model=model_path,
        model_sha256=model_digest,
        normalizer=normalizer_path,
        normalizer_sha256=normalizer_digest,
    )


@dataclass(frozen=True)
class SnapshotIndex:
    """A parsed ``snapshots/index.json``."""

    directory: Path
    run_id: str | None
    entries: tuple[SnapshotEntry, ...]
    failures: tuple[dict[str, Any], ...]


def read_snapshot_index(directory: str | os.PathLike[str]) -> SnapshotIndex:
    """Parse ``<directory>/index.json``; ``ValueError`` on anything malformed.

    ``directory`` is the snapshots directory (``<run>/snapshots``).
    Entries come back in timestep order; a repeated timestep is
    malformed.
    """
    root = Path(directory)
    where = f"{SNAPSHOTS_DIR_NAME}/{SNAPSHOT_INDEX_NAME}"
    payload = read_json_object(root / SNAPSHOT_INDEX_NAME, label=where)
    if payload.get("schema") != SNAPSHOT_INDEX_SCHEMA:
        raise ValueError(
            f"{where}: schema {payload.get('schema')!r}, expected "
            f"{SNAPSHOT_INDEX_SCHEMA!r}"
        )
    raw_entries = payload.get("snapshots")
    raw_failures = payload.get("failures", [])
    if not isinstance(raw_entries, list) or not isinstance(raw_failures, list):
        raise ValueError(f"{where}: 'snapshots' and 'failures' must be lists")
    entries = sorted(
        (_entry_from_dict(raw, where) for raw in raw_entries),
        key=lambda entry: entry.timestep,
    )
    steps = [entry.timestep for entry in entries]
    if len(set(steps)) != len(steps):
        raise ValueError(f"{where}: a timestep is listed twice")
    run_id = payload.get("run_id")
    return SnapshotIndex(
        directory=root,
        run_id=run_id if isinstance(run_id, str) else None,
        entries=tuple(entries),
        failures=tuple(
            dict(failure) for failure in raw_failures if isinstance(failure, Mapping)
        ),
    )


def snapshot_unbound(directory: str | os.PathLike[str], entry: SnapshotEntry) -> str | None:
    """Why ``entry``'s files are not the ones the index binds, or ``None``."""
    problems: list[str] = []
    for name, digest in (
        (entry.model, entry.model_sha256),
        (entry.normalizer, entry.normalizer_sha256),
    ):
        if name is None:
            continue
        path = Path(directory) / name
        if not path.is_file():
            problems.append(f"{name} (missing)")
        elif sha256_file(path) != digest:
            problems.append(f"{name} (sha256 differs)")
    if problems:
        return (
            f"{SNAPSHOT_INDEX_NAME} does not bind the snapshot at "
            f"{entry.timestep} steps: " + ", ".join(problems)
        )
    return None


def unindexed_files(
    directory: str | os.PathLike[str], index: SnapshotIndex | None
) -> list[str]:
    """The files in ``directory`` that ``index`` does not bind, sorted.

    Snapshot files an interrupt between a snapshot's renames left
    outside the index (every file but the index itself when ``index``
    is ``None``). A writer's temporary leftover is not listed here:
    ``verify`` fails the directory for it already.
    """
    root = Path(directory)
    bound = {SNAPSHOT_INDEX_NAME}
    for entry in () if index is None else index.entries:
        bound.add(entry.model)
        if entry.normalizer is not None:
            bound.add(entry.normalizer)
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.name not in bound
        and not is_temporary_path(path.name)
    )


def snapshot_summary(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The run record's ``snapshots`` block, or ``None`` without snapshots.

    ``{"index", "count", "timesteps", "failures", "unindexed"}`` read
    from the index as written (the manifest hashes the files
    themselves); ``unindexed`` names the files the index does not bind
    (:func:`unindexed_files`). An index that does not parse gives
    ``{"index", "problem"}``, and a directory holding snapshot files but
    no index (the first snapshot interrupted between its renames)
    ``{"index", "problem", "unindexed"}``: never a crash, never silence.
    """
    directory = Path(run_dir) / SNAPSHOTS_DIR_NAME
    relative = f"{SNAPSHOTS_DIR_NAME}/{SNAPSHOT_INDEX_NAME}"
    if not directory.is_dir():
        return None
    if not (directory / SNAPSHOT_INDEX_NAME).is_file():
        orphans = unindexed_files(directory, None)
        if not orphans:
            return None
        return {
            "index": relative,
            "problem": (
                f"no {relative}, though {SNAPSHOTS_DIR_NAME}/ holds "
                f"{len(orphans)} file(s): the first snapshot was interrupted "
                f"between its renames, and none is bound"
            ),
            "unindexed": orphans,
        }
    try:
        index = read_snapshot_index(directory)
    except ValueError as error:
        return {"index": relative, "problem": str(error)}
    return {
        "index": relative,
        "count": len(index.entries),
        "timesteps": [entry.timestep for entry in index.entries],
        "failures": [failure.get("timestep") for failure in index.failures],
        "unindexed": unindexed_files(directory, index),
    }
