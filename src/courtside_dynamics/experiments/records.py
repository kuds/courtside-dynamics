"""Run records and sync manifests: what a run directory says about itself.

``docs/design_campaign_structure_20261016.md`` step L1. Seven of about
48 archived run directories were stubs that looked live, one run lost
all its videos without a trace, and a Drive copy could not be told from
a partial one (``repo_review_20261016.md`` R3). Two files close that:

``run_record.json`` (:data:`RUN_RECORD_SCHEMA`)
    One JSON object per run: its identity, budget, algorithm, task,
    comparability key, seeds, git state, library versions, warm start,
    selection, wall clock, and how it ended. ``train()`` writes it with
    status ``"running"`` right after ``config.json`` and rewrites it on
    every terminal path (:data:`TERMINAL_STATUSES`).
``sync_manifest.json`` (:data:`SYNC_MANIFEST_SCHEMA`)
    The relative path, size and sha256 of every other file in the run
    directory. ``train()`` writes it last, after the final run record,
    once every file is closed; a Drive mirror gets its own, written only
    after every copied file re-hashes correctly
    (:mod:`courtside_dynamics.experiments.sync`).

Both are written atomically: a temporary file in the same directory is
renamed over the target, so a reader never sees half of one.

A run directory is **complete** when its manifest verifies and the run
record it lists carries a terminal status (:func:`verify_run_dir`).
Complete means the directory is exactly what ``train()`` left when the
run ended -- a crashed or interrupted run included, its status says
how it ended -- not that the run succeeded. Without a manifest (a
SIGKILL, a dead Colab VM, a run still training, a directory from before
run records), or with one that does not verify, it is incomplete by
definition.

Stable-Baselines3 and torch are never imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

#: ``run_record.json``'s schema tag; bump it with any incompatible change.
RUN_RECORD_SCHEMA = "courtside-run-record/1"

#: ``sync_manifest.json``'s schema tag.
SYNC_MANIFEST_SCHEMA = "courtside-sync-manifest/1"

#: File names at the run-directory root (``training.artifacts.RUN_LAYOUT``
#: carries the same two names; a test pins the agreement).
RUN_RECORD_NAME = "run_record.json"
SYNC_MANIFEST_NAME = "sync_manifest.json"

#: Written at ``train()`` start; never terminal.
RUNNING = "running"
#: Trained the whole budget. A guard whose condition held only at the
#: final scheduled evaluation ends here too, and its ``stop_reason``
#: says so (``repo_review_20261016`` train-3).
COMPLETED = "completed"
#: A guard ended training before the budget: early-stop patience (the
#: headline or the reward path), the degenerate-signal guard, or a
#: performance gate's stage budget. ``stop_reason`` names which.
EARLY_STOP = "early_stop"
#: ``KeyboardInterrupt`` (a stopped Colab cell): the salvage path saved
#: what it could.
INTERRUPTED = "interrupted"
#: An exception: the crash salvage, or a failure before or after
#: ``learn()`` once ``config.json`` existed. ``stop_reason`` carries the
#: error.
CRASHED = "crashed"

RUN_STATUSES = (RUNNING, COMPLETED, EARLY_STOP, INTERRUPTED, CRASHED)
TERMINAL_STATUSES = frozenset({COMPLETED, EARLY_STOP, INTERRUPTED, CRASHED})

_HASH_CHUNK = 1024 * 1024


def utc_now() -> str:
    """UTC ISO-8601 timestamp to the second (``config.json``'s format)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """sha256 of ``value``'s canonical JSON (sorted keys, no whitespace)."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write ``payload`` as indented JSON via a temporary file and a rename.

    The temporary file sits beside the target, so ``os.replace`` is a
    same-directory rename and a reader sees the old file or the new one,
    never a torn write. A failed write removes its temporary file and
    raises.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=repr)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def _kwargs_sha256(env_profile: Mapping[str, Any]) -> str | None:
    kwargs = env_profile.get("constructor_kwargs")
    return canonical_sha256(kwargs) if isinstance(kwargs, Mapping) else None


def _warm_start_section(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The warm start's source run (its directory name) and artifact digests.

    ``config.json``'s ``initialization`` block (written once the model is
    built) carries the verified sha256s; before it exists the configured
    ``train_config.warm_start`` names the source.
    """
    train_config = config.get("train_config") or {}
    configured = train_config.get("warm_start") or {}
    initialization = config.get("initialization") or {}
    if not configured and not initialization:
        return None
    source = initialization.get("source_run_dir") or configured.get(
        "source_run_dir"
    )
    source_artifacts = initialization.get("source_artifacts") or {}
    digests = {
        str(name): entry.get("sha256")
        for name, entry in source_artifacts.items()
        if isinstance(entry, Mapping)
    }
    return {
        "source_run_dir_name": (
            os.path.basename(os.path.normpath(str(source))) if source else None
        ),
        "artifact_sha256": digests or None,
    }


def selection_from_best_model_meta(
    meta: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The record's ``selection`` block from ``best_model_meta.json``.

    ``None`` without a meta file: a reward-selected run's
    ``best_model.zip`` is SB3's reward argmax, which records no
    selection event. ``n_updates`` is the SB3 update count behind the
    selecting evaluation (on-policy runs record it; ``None`` otherwise).
    """
    if not isinstance(meta, Mapping):
        return None
    return {
        "timestep": meta.get("timestep"),
        "keys": list(meta.get("selection_keys") or []),
        "values": dict(meta.get("selection_values") or {}),
        "n_updates": meta.get("n_updates"),
    }


def build_run_record(
    config: Mapping[str, Any],
    *,
    run_id: str,
    created_at: str,
    status: str,
    finished_at: str | None = None,
    stop_reason: str | None = None,
    trained_timesteps: int | None = None,
    n_updates: int | None = None,
    git_dirty: bool | None = None,
    best_model_meta: Mapping[str, Any] | None = None,
    wall_clock: Mapping[str, Any] | None = None,
    preflight_warnings: Sequence[str] = (),
    artifacts: Sequence[Mapping[str, str]] = (),
    sync: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One run's record, from its ``config.json`` payload and its outcome.

    Every field the run's ``config.json`` already records is copied from
    it (one source, no second derivation): the algorithm, the recipe,
    the task's env classes and observation fingerprint, the
    comparability key, the seeds, the git sha and the library versions.
    The task's constructor kwargs are recorded as the sha256 of their
    canonical JSON. ``eval_protocol`` stays ``None`` until evaluation
    protocols are versioned, and the ``experiment`` block holds the
    placeholders the spec system will fill.
    """
    if status not in RUN_STATUSES:
        raise ValueError(
            f"run status must be one of {list(RUN_STATUSES)}, got {status!r}"
        )
    train_config = config.get("train_config") or {}
    env = config.get("env") or {}
    evaluation_env = config.get("evaluation_env") or {}
    algorithm = config.get("algorithm") or {}
    return {
        "schema": RUN_RECORD_SCHEMA,
        "run_id": run_id,
        "created_at": created_at,
        "finished_at": finished_at,
        "status": status,
        "stop_reason": stop_reason,
        "budget": {
            "total_timesteps": train_config.get("total_timesteps"),
            "trained_timesteps": trained_timesteps,
            "n_updates": n_updates,
        },
        "progress": dict(progress) if progress else None,
        "algo": {
            "name": algorithm.get("name"),
            "family": algorithm.get("family"),
            "impl": algorithm.get("implementation"),
        },
        "recipe": config.get("recipe_name"),
        "task": {
            "env_class": env.get("class"),
            "constructor_kwargs_sha256": _kwargs_sha256(env),
            "evaluation_env_class": evaluation_env.get("class"),
            "evaluation_constructor_kwargs_sha256": _kwargs_sha256(
                evaluation_env
            ),
            "observation_names_sha256": env.get("observation_names_sha256"),
            "eval_protocol": None,
        },
        "comparability_key": config.get("comparability_key"),
        "seeds": {
            "seed": train_config.get("seed"),
            "evaluation_seeding": config.get("evaluation_seeding"),
        },
        "git": {"sha": config.get("git_sha"), "dirty": git_dirty},
        "versions": config.get("versions"),
        "warm_start": _warm_start_section(config),
        "selection": selection_from_best_model_meta(best_model_meta),
        "wall_clock": dict(wall_clock) if wall_clock else None,
        "preflight_warnings": list(preflight_warnings),
        "sync": dict(sync) if sync else None,
        "experiment": {"id": None, "parent": None, "lever": None, "kind": None},
        "artifacts": [dict(artifact) for artifact in artifacts],
    }


def write_run_record(
    run_dir: str | os.PathLike[str], record: Mapping[str, Any]
) -> Path:
    """Atomically (re)write ``run_dir/run_record.json``."""
    return write_json_atomic(Path(run_dir) / RUN_RECORD_NAME, dict(record))


def read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON object; ``ValueError`` (naming the file) on anything else."""
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is unreadable: {error!r}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


def run_files(run_dir: str | os.PathLike[str]) -> list[str]:
    """Every file under ``run_dir`` but its manifest, as sorted POSIX paths."""
    root = Path(run_dir)
    found: list[str] = []
    for directory, _subdirs, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative != SYNC_MANIFEST_NAME:
                found.append(relative)
    return sorted(found)


def manifest_payload(
    run_id: str | None, entries: Mapping[str, tuple[int, str]]
) -> dict[str, Any]:
    """A manifest over ``entries`` (relative path -> (bytes, sha256))."""
    files = [
        {"path": path, "bytes": int(size), "sha256": digest}
        for path, (size, digest) in sorted(entries.items())
    ]
    return {
        "schema": SYNC_MANIFEST_SCHEMA,
        "run_id": run_id,
        "written_at": utc_now(),
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }


def write_sync_manifest(
    run_dir: str | os.PathLike[str], *, run_id: str | None = None
) -> dict[str, Any]:
    """Hash every file in ``run_dir`` and atomically write its manifest.

    Call it last: a file written afterwards is not listed (see
    :func:`verify_run_dir` for how unlisted files are reported).
    ``run_id`` defaults to the directory's name.
    """
    root = Path(run_dir)
    entries = {
        relative: ((root / relative).stat().st_size, sha256_file(root / relative))
        for relative in run_files(root)
    }
    payload = manifest_payload(run_id or root.name, entries)
    write_json_atomic(root / SYNC_MANIFEST_NAME, payload)
    return payload


@dataclass(frozen=True)
class ManifestMismatch:
    """One listed file that is not what the manifest recorded."""

    path: str
    problem: str  # "missing", "bytes" or "sha256"
    expected: str | int | None
    actual: str | int | None


@dataclass(frozen=True)
class RunDirVerification:
    """:func:`verify_run_dir`'s verdict on one run directory.

    ``manifest`` is ``"verified"`` (every listed file matches),
    ``"mismatch"``, ``"missing"`` or ``"unreadable"``. ``status`` is the
    run record's status when one is readable (the listed one when the
    manifest verifies). ``unlisted`` names files on disk the manifest
    does not list: written after it (a notebook's plots and replay
    video), or by a sync that failed before its new manifest.
    ``problems`` says, in words, why the directory is incomplete.
    """

    run_dir: str
    complete: bool
    manifest: str
    status: str | None
    files_checked: int
    mismatches: tuple[ManifestMismatch, ...] = ()
    unlisted: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_relative(path: Any) -> str | None:
    """``path`` when it is a plain relative POSIX path inside the run dir."""
    if not isinstance(path, str) or not path or "\\" in path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        return None
    return pure.as_posix()


def _manifest_entries(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[int, str]]:
    if manifest.get("schema") != SYNC_MANIFEST_SCHEMA:
        raise ValueError(
            f"schema {manifest.get('schema')!r}, expected {SYNC_MANIFEST_SCHEMA!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("no 'files' list")
    entries: dict[str, tuple[int, str]] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError(f"malformed entry {entry!r}")
        path = _safe_relative(entry.get("path"))
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            path is None
            or path == SYNC_MANIFEST_NAME
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"malformed entry {dict(entry)!r}")
        if path in entries:
            raise ValueError(f"{path} is listed twice")
        entries[path] = (size, digest)
    return entries


def _record_status(path: Path) -> str | None:
    try:
        status = read_json_object(path).get("status")
    except ValueError:
        return None
    return status if isinstance(status, str) else None


def verify_run_dir(
    run_dir: str | os.PathLike[str], *, strict: bool = False
) -> RunDirVerification:
    """Check a run directory against its ``sync_manifest.json``.

    Complete when the manifest exists, every file it lists is on disk
    with its recorded size and sha256, it lists ``run_record.json``, and
    that record's status is terminal. Files the manifest does not list
    are reported in ``unlisted``; they make the directory incomplete
    only under ``strict`` (by default a notebook's post-training plots
    beside a sealed run do not, while every file ``train()`` wrote is
    still checked). A mid-run Drive mirror whose manifest verifies is
    an intact snapshot of an unfinished run: its listed record says
    ``"running"``, so it is incomplete too.
    """
    root = Path(run_dir)
    record_path = root / RUN_RECORD_NAME
    if not root.is_dir():
        return RunDirVerification(
            run_dir=str(root),
            complete=False,
            manifest="missing",
            status=None,
            files_checked=0,
            problems=(f"{root} is not a directory",),
        )
    manifest_path = root / SYNC_MANIFEST_NAME
    if not manifest_path.is_file():
        record_status = (
            _record_status(record_path) if record_path.is_file() else None
        )
        cause = (
            f"its run record says {record_status!r}"
            if record_status is not None
            else "there is no readable run_record.json either"
        )
        return RunDirVerification(
            run_dir=str(root),
            complete=False,
            manifest="missing",
            status=record_status,
            files_checked=0,
            problems=(
                f"no {SYNC_MANIFEST_NAME} ({cause}): the run never reached a "
                f"terminal path (still training, killed, a disconnected "
                f"runtime) or predates run records",
            ),
        )
    try:
        entries = _manifest_entries(read_json_object(manifest_path))
    except ValueError as error:
        return RunDirVerification(
            run_dir=str(root),
            complete=False,
            manifest="unreadable",
            status=None,
            files_checked=0,
            problems=(f"{SYNC_MANIFEST_NAME} is unreadable: {error}",),
        )
    mismatches: list[ManifestMismatch] = []
    for relative, (size, digest) in entries.items():
        path = root / relative
        if not path.is_file():
            mismatches.append(ManifestMismatch(relative, "missing", digest, None))
            continue
        actual_size = path.stat().st_size
        if actual_size != size:
            mismatches.append(ManifestMismatch(relative, "bytes", size, actual_size))
            continue
        actual_digest = sha256_file(path)
        if actual_digest != digest:
            mismatches.append(
                ManifestMismatch(relative, "sha256", digest, actual_digest)
            )
    unlisted = tuple(
        relative for relative in run_files(root) if relative not in entries
    )
    problems: list[str] = []
    if mismatches:
        problems.append(
            f"{len(mismatches)} listed file(s) differ from {SYNC_MANIFEST_NAME}: "
            + ", ".join(f"{m.path} ({m.problem})" for m in mismatches[:5])
            + (" ..." if len(mismatches) > 5 else "")
        )
    status: str | None = None
    if RUN_RECORD_NAME not in entries:
        problems.append(f"{SYNC_MANIFEST_NAME} does not list {RUN_RECORD_NAME}")
    elif record_path.is_file():
        status = _record_status(record_path)
        if status is None:
            problems.append(f"{RUN_RECORD_NAME} is unreadable or has no status")
        elif status not in TERMINAL_STATUSES:
            problems.append(
                f"{RUN_RECORD_NAME} status is {status!r}: the listed files are "
                f"a snapshot of a run that had not finished"
            )
    if strict and unlisted:
        problems.append(
            f"{len(unlisted)} file(s) are not in {SYNC_MANIFEST_NAME} "
            f"(strict): " + ", ".join(unlisted[:5])
            + (" ..." if len(unlisted) > 5 else "")
        )
    return RunDirVerification(
        run_dir=str(root),
        complete=not problems,
        manifest="mismatch" if mismatches else "verified",
        status=status,
        files_checked=len(entries),
        mismatches=tuple(mismatches),
        unlisted=unlisted,
        problems=tuple(problems),
    )
