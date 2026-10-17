"""Local-then-sync: mirror a run directory and seal the mirror.

``TrainConfig.sync_dir`` (default ``None``) splits a run's two homes:
``train()`` writes under ``log_dir`` on local disk and mirrors it to
``sync_dir`` (a Google Drive folder on Colab) at every checkpoint and at
the end. Writing a multi-hour run straight onto the Drive mount is how
evidence got lost (``repo_review_20261016`` R3, critic-4): unflushed
writes when the VM was torn down, a run with no videos and no trace of
why. A mirror is only trusted through its manifest:

- each changed file is streamed into a temporary file beside its
  destination while being hashed, renamed into place, and read back:
  the copy must re-hash to the digest of the bytes that were read;
- files the source no longer has are removed from the mirror;
- only then is the mirror's ``sync_manifest.json`` written, atomically
  and last.

A failure anywhere raises :class:`SyncError` before the new manifest is
written, so the mirror keeps its previous manifest (or none) and
:func:`~courtside_dynamics.experiments.records.verify_run_dir` reports
it incomplete: a changed file no longer matches, and a run record that
never got its terminal copy still says ``"running"``.

The mirror target must be absent or empty when a run starts (checked
before any output by :func:`validate_sync_target`), so every file in it
is this run's.

Stable-Baselines3 and torch are never imported here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from courtside_dynamics.experiments.records import (
    SYNC_MANIFEST_NAME,
    manifest_payload,
    run_files,
    sha256_file,
    write_json_atomic,
)

_COPY_CHUNK = 1024 * 1024


class SyncError(RuntimeError):
    """A mirror pass failed; the mirror keeps its previous manifest."""


def validate_sync_target(
    log_dir: str | os.PathLike[str], sync_dir: Any
) -> Path | None:
    """Refuse a ``sync_dir`` that cannot hold a clean mirror of ``log_dir``.

    ``None`` passes (no mirror). Otherwise it must be a path that is not
    ``log_dir``, not inside it and not around it; whose parent directory
    exists (an unmounted Drive path fails here instead of filling a
    local folder of the same name); and that is absent or an empty
    directory, so the mirror never mixes with another run's files.
    Returns the resolved path.
    """
    if sync_dir is None:
        return None
    if not isinstance(sync_dir, (str, os.PathLike)) or not str(sync_dir).strip():
        raise TypeError(f"sync_dir must be a non-empty path or None, got {sync_dir!r}")
    target = Path(sync_dir).expanduser().resolve()
    source = Path(log_dir).expanduser().resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError(
            f"sync_dir {target} must be a separate directory from log_dir "
            f"{source} (neither may contain the other)"
        )
    if not target.parent.is_dir():
        raise ValueError(
            f"sync_dir's parent {target.parent} does not exist; on Colab, "
            f"mount Drive first (mount_drive()) -- a missing mount would "
            f"otherwise mirror into a local folder of the same name"
        )
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(
            f"sync_dir {target} already exists and is not an empty "
            f"directory; a mirror must start empty so every file in it is "
            f"this run's. Use a fresh directory"
        )
    return target


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    """Stream ``source`` into ``destination``; return what was copied.

    The bytes are hashed as they are read, written to a temporary file
    beside the destination, flushed to the device and renamed into
    place, then read back: a copy whose bytes do not re-hash to what was
    read raises :class:`SyncError`. Files still being written during
    training (TensorBoard events, the monitor CSVs) are mirrored as the
    bytes read here, a consistent snapshot of that moment.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.sync-tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with open(source, "rb") as reader, open(temporary, "wb") as writer:
            for chunk in iter(lambda: reader.read(_COPY_CHUNK), b""):
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    expected = digest.hexdigest()
    copied_size = destination.stat().st_size
    copied = sha256_file(destination)
    if copied_size != size or copied != expected:
        raise SyncError(
            f"{destination} does not re-hash to the bytes copied from "
            f"{source} ({copied_size} bytes, sha256 {copied[:12]}; expected "
            f"{size} bytes, {expected[:12]})"
        )
    return size, expected


class RunDirMirror:
    """Mirror one run directory into another and seal the copy.

    Keeps, per file, the source's (size, mtime) at its last verified
    copy, so a pass re-copies only what changed since; a file whose copy
    failed is forgotten and re-copied next time. :attr:`passes` and
    :attr:`failures` count :meth:`sync` calls that completed and raised.
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        run_id: str | None = None,
    ) -> None:
        self.source = Path(source)
        self.destination = Path(destination)
        self.run_id = run_id or self.source.name
        self._copied: dict[str, tuple[int, int, int, str]] = {}
        self.passes = 0
        self.failures = 0

    def sync(self) -> dict[str, Any]:
        """One mirror pass; returns the manifest it wrote.

        Raises :class:`SyncError` (or the underlying ``OSError``) without
        writing a manifest when any copy, re-hash or cleanup fails.
        """
        try:
            manifest = self._sync()
        except BaseException:
            self.failures += 1
            raise
        self.passes += 1
        return manifest

    def _sync(self) -> dict[str, Any]:
        if not self.source.is_dir():
            raise SyncError(f"the run directory {self.source} does not exist")
        self.destination.mkdir(parents=True, exist_ok=True)
        entries: dict[str, tuple[int, str]] = {}
        files = run_files(self.source)
        for relative in files:
            source = self.source / relative
            destination = self.destination / relative
            stat = source.stat()
            cached = self._copied.get(relative)
            if (
                cached is not None
                and cached[:2] == (stat.st_size, stat.st_mtime_ns)
                and destination.is_file()
                and destination.stat().st_size == cached[2]
            ):
                entries[relative] = (cached[2], cached[3])
                continue
            self._copied.pop(relative, None)
            size, digest = _copy_and_hash(source, destination)
            self._copied[relative] = (stat.st_size, stat.st_mtime_ns, size, digest)
            entries[relative] = (size, digest)
        keep = set(files)
        for relative in run_files(self.destination):
            if relative not in keep:
                (self.destination / relative).unlink()
        payload = manifest_payload(self.run_id, entries)
        write_json_atomic(self.destination / SYNC_MANIFEST_NAME, payload)
        return payload


def sync_run_dir(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> dict[str, Any]:
    """Mirror ``source`` into ``destination`` once (every file re-copied).

    The manual re-sync a failed end-of-run mirror asks for, and the
    notebooks' last step after their post-training cells add plots and
    a replay video to the local run directory. Returns the manifest.
    """
    return RunDirMirror(source, destination).sync()
