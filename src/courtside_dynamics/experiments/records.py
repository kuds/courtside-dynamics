"""Run records and sync manifests: what a run directory says about itself.

``docs/design_campaign_structure_20261016.md`` step L1. Seven of about
48 archived run directories were stubs that looked live, one run lost
all its videos without a trace, and a Drive copy could not be told from
a partial one (``repo_review_20261016.md`` R3). Two files close that:

``run_record.json`` (:data:`RUN_RECORD_SCHEMA`)
    One JSON object per run: its identity, budget, algorithm, task,
    comparability key, seeds, git state, library versions, warm start,
    selection, evaluation snapshots, wall clock, and how it ended. ``train()`` writes it with
    status ``"running"`` right after ``config.json`` and rewrites it on
    every terminal path (:data:`TERMINAL_STATUSES`).
``sync_manifest.json`` (:data:`SYNC_MANIFEST_SCHEMA`)
    The relative path, size and sha256 of every other file in the run
    directory. ``train()`` writes it last, after the final run record,
    once every file is closed; a Drive mirror gets its own, written only
    after every copied file re-hashes correctly
    (:mod:`courtside_dynamics.experiments.sync`).

Both are written atomically: a temporary file in the same directory is
renamed over the target, so a reader never sees half of one. The
best-model triple is written the same way (:class:`ReplaceTogether`),
and the record's ``selection`` says whether ``best_model_meta.json``'s
sha256s bind the files beside it (:func:`best_model_meta_unbound`).

A run directory is **complete** when its manifest verifies, the run
record it lists carries a terminal status, every file it does not list
is one a notebook writes after training, and it holds no writer's
temporary leftover, listed or not (:func:`verify_run_dir`). Complete
means the directory is exactly what ``train()`` left when the run
ended -- a crashed or interrupted run included, its status says how it
ended -- not that the run succeeded. Without a manifest (a SIGKILL, a
dead Colab VM, a run still training, a directory from before run
records), or with one that does not verify, it is incomplete by
definition.

Stable-Baselines3 and torch are never imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
#: Trained the whole budget (``num_timesteps >= total_timesteps``). A
#: guard that fires once the run has trained its whole budget -- at the
#: final evaluation of a budget that is a multiple of ``eval_freq``, say
#: -- ends here too, and its ``stop_reason`` says so
#: (``repo_review_20261016`` train-3); for an on-policy run that reason
#: says the budget was collected and the last rollout's update skipped,
#: since SB3 stops before that update. A guard that fires at the final
#: scheduled evaluation of a budget that is *not* such a multiple left
#: steps untrained: ``early_stop``.
COMPLETED = "completed"
#: A guard ended training before the budget: early-stop patience (the
#: headline or the reward path), the degenerate-signal guard, or a
#: performance gate's stage budget. ``stop_reason`` names which.
EARLY_STOP = "early_stop"
#: ``KeyboardInterrupt`` (a stopped Colab cell): the salvage path saved
#: what it could.
INTERRUPTED = "interrupted"
#: An exception: the crash salvage, or a failure before or after
#: ``learn()`` once ``config.json`` existed -- including one while the
#: run was closing after it had ended another way (an env or the SB3
#: logger failing to close after a completed run: ``train()`` raises,
#: so the record must not read ``completed``). ``stop_reason`` carries
#: the error, and the earlier outcome when there was one.
CRASHED = "crashed"

RUN_STATUSES = (RUNNING, COMPLETED, EARLY_STOP, INTERRUPTED, CRASHED)
TERMINAL_STATUSES = frozenset({COMPLETED, EARLY_STOP, INTERRUPTED, CRASHED})

#: Where the notebooks write after ``train()`` returned and sealed the
#: run directory: the plots, audits and gate reports under ``reports/``
#: and the replay video (``training.artifacts.RUN_LAYOUT``'s
#: ``learning_curve``, ``eval_*``, ``training_health_plot``,
#: ``best_long_eval*`` and ``best_model_video``; a test pins the
#: agreement), and the offline harness's ``scores/<protocol>.jsonl``
#: (``experiments.harness.rescore``). :func:`verify_run_dir` reports an unlisted file there
#: and fails it only under ``strict``; an unlisted file anywhere else
#: fails the directory.
POST_TRAINING_DIRS = ("reports/", "scores/")
POST_TRAINING_FILES = frozenset({"media/best_model.mp4"})

#: Basename suffixes of the temporary files this repository's writers
#: rename into place (:func:`temporary_path`'s ``.<name>.tmp-<pid>``,
#: the mirror's ``.<name>.sync-tmp``, the callbacks' ``<name>.tmp`` and
#: ``<name>.tmp.npz``, ``NamedTemporaryFile``'s ``tmp*.tmp``). One left
#: behind is an interrupted write: it fails verification whether or not
#: a manifest lists it.
_TEMPORARY_SUFFIX = re.compile(r"\.(tmp(-\d+)?|sync-tmp|tmp\.npz)$")

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


def temporary_path(target: str | os.PathLike[str]) -> Path:
    """The temporary file a writer here renames over ``target``.

    ``.<name>.tmp-<pid>`` beside the target, so the rename is a
    same-directory ``os.replace`` and a leftover is recognized
    (:func:`is_temporary_path`).
    """
    target = Path(target)
    return target.with_name(f".{target.name}.tmp-{os.getpid()}")


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write ``payload`` as indented JSON via a temporary file and a rename.

    The temporary file sits beside the target, so ``os.replace`` is a
    same-directory rename and a reader sees the old file or the new one,
    never a torn write. A failed write removes its temporary file and
    raises.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(target)
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


class ReplaceTogether:
    """Write files under temporary names, then rename them all into place.

    :meth:`stage` returns the temporary path (:func:`temporary_path`) to
    write a target's new bytes to; :meth:`commit` renames every staged
    file over its target, in staging order. Used as a context manager:
    leaving the block without a commit -- an exception, or a
    ``KeyboardInterrupt`` (a stopped Colab cell) inside a write --
    removes every temporary still on disk, so the previous files stand
    whole and nothing is left behind. Each rename is atomic, the set is
    not: an interrupt between two renames leaves the earlier targets new
    and the later ones old. The best-model triple therefore stages
    ``best_model_meta.json`` last, binding the others by sha256, and
    :func:`best_model_meta_unbound` catches such a split.
    """

    def __init__(self) -> None:
        self._staged: list[tuple[Path, Path]] = []

    def __enter__(self) -> ReplaceTogether:
        return self

    def stage(self, target: str | os.PathLike[str]) -> str:
        """The temporary path to write ``target``'s new content to."""
        temporary = temporary_path(target)
        self._staged.append((temporary, Path(target)))
        return str(temporary)

    def commit(self) -> None:
        """Rename every staged file over its target, in staging order."""
        for temporary, target in self._staged:
            os.replace(temporary, target)

    def __exit__(self, *_exc_info: object) -> None:
        for temporary, _target in self._staged:
            try:
                temporary.unlink()
            except OSError:  # renamed already, or never written
                pass


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a JSON object; otherwise read as missing."""
    return value if isinstance(value, Mapping) else {}


def _kwargs_sha256(env_profile: Mapping[str, Any]) -> str | None:
    kwargs = env_profile.get("constructor_kwargs")
    return canonical_sha256(kwargs) if isinstance(kwargs, Mapping) else None


def _warm_start_section(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The warm start's source run (its directory name) and artifact digests.

    ``config.json``'s ``initialization`` block (written once the model is
    built) carries the verified sha256s; before it exists the configured
    ``train_config.warm_start`` names the source.
    """
    configured = _mapping(_mapping(config.get("train_config")).get("warm_start"))
    initialization = _mapping(config.get("initialization"))
    if not configured and not initialization:
        return None
    source = initialization.get("source_run_dir") or configured.get(
        "source_run_dir"
    )
    source_artifacts = _mapping(initialization.get("source_artifacts"))
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


#: ``selection["bindings"]``: whether ``best_model_meta.json``'s
#: sha256s were checked against the files beside it and match
#: (``"verified"``), do not match (``"unbound"``), or were not checked
#: (``"not_checked"``: the index's backfill of runs from before run
#: records, or a meta written before metas bound their files, which has
#: no ``artifacts`` block to check).
BINDINGS_VERIFIED = "verified"
BINDINGS_UNBOUND = "unbound"
BINDINGS_NOT_CHECKED = "not_checked"

#: The best-model triple's ``training.artifacts.RUN_LAYOUT`` names (a
#: test pins the agreement). An unbound selection leaves all three out
#: of the record's ``artifacts``: the files no longer form one save.
BEST_MODEL_ARTIFACTS = frozenset(
    {"best_model", "best_vec_normalize", "best_model_meta"}
)


def best_model_meta_unbound(
    meta: Mapping[str, Any], directory: str | os.PathLike[str]
) -> str | None:
    """Why ``best_model_meta.json`` does not bind the files beside it.

    ``None`` when it does: its ``artifacts`` names ``best_model.zip``
    (and the normalizer, when the run had one) with a sha256, and every
    file it names is in ``directory`` with that digest -- the check the
    loaders apply before replaying a best model
    (``notebook_utils.verify_best_checkpoint_pair``). Also ``None`` for a
    meta written before metas bound their files (no ``artifacts`` at
    all), which the loaders accept as legacy: there is nothing to check.
    Otherwise the reason: a best-model save interrupted between its
    renames (:class:`ReplaceTogether`) or a file replaced since leaves a
    meta describing a selection that is not the files on disk.
    """
    bound = meta.get("artifacts")
    if bound is None:
        return None
    if not isinstance(bound, Mapping) or not bound:
        return "best_model_meta.json binds no files (an empty 'artifacts')"
    if "best_model.zip" not in bound:
        return "best_model_meta.json does not bind best_model.zip"
    stale: list[str] = []
    for name, entry in sorted(bound.items()):
        expected = entry.get("sha256") if isinstance(entry, Mapping) else None
        if _safe_relative(name) != name or "/" in name:
            stale.append(f"{name!r} (not a file name)")
            continue
        path = Path(directory) / name
        if not path.is_file():
            stale.append(f"{name} (missing)")
        elif sha256_file(path) != expected:
            stale.append(f"{name} (sha256 differs)")
    if stale:
        return (
            "best_model_meta.json does not bind the files beside it: "
            + ", ".join(stale)
            + " (a best-model save interrupted between its renames, or a "
            "file replaced since)"
        )
    return None


def selection_from_best_model_meta(
    meta: Mapping[str, Any] | None,
    *,
    directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """The record's ``selection`` block from ``best_model_meta.json``.

    ``None`` without a meta file: a reward-selected run's
    ``best_model.zip`` is SB3's reward argmax, which records no
    selection event. ``n_updates`` is the SB3 update count behind the
    selecting evaluation (on-policy runs record it; ``None`` otherwise).

    With ``directory`` (the meta's own), the meta's sha256 bindings are
    checked against the files there (:func:`best_model_meta_unbound`).
    When they do not bind them, ``bindings`` is ``"unbound"`` with the
    reason, and ``timestep``, ``values`` and ``n_updates`` are ``None``:
    the meta then describes another save (the previous best, when a
    save was interrupted), never the files on disk. Without
    ``directory``, or for a meta that binds nothing (written before
    metas bound their files), the bindings are ``"not_checked"``. A
    field of the wrong type is read as missing (:func:`shape_problems`
    names it).
    """
    if not isinstance(meta, Mapping):
        return None
    keys = meta.get("selection_keys")
    values = meta.get("selection_values")
    checked = False
    unbound: str | None = None
    if directory is not None and meta.get("artifacts") is not None:
        checked = True
        unbound = best_model_meta_unbound(meta, directory)
    if unbound is not None:
        return {
            "timestep": None,
            "keys": list(keys) if isinstance(keys, list) else [],
            "values": None,
            "n_updates": None,
            "bindings": BINDINGS_UNBOUND,
            "unbound_reason": unbound,
        }
    return {
        "timestep": meta.get("timestep"),
        "keys": list(keys) if isinstance(keys, list) else [],
        "values": dict(values) if isinstance(values, Mapping) else {},
        "n_updates": meta.get("n_updates"),
        "bindings": BINDINGS_VERIFIED if checked else BINDINGS_NOT_CHECKED,
        "unbound_reason": None,
    }


#: The ``config.json`` objects :func:`build_run_record` reads fields
#: from, as dotted paths.
_CONFIG_OBJECTS = (
    "train_config",
    "train_config.warm_start",
    "env",
    "env.constructor_kwargs",
    "evaluation_env",
    "evaluation_env.constructor_kwargs",
    "algorithm",
    "comparability_key",
    "comparability_key.eval_protocol",
    "initialization",
    "initialization.source_artifacts",
    "versions",
)

#: The ``best_model_meta.json`` fields the selection block reads, with
#: the type each must have.
_META_FIELDS: tuple[tuple[str, type, str], ...] = (
    ("selection_keys", list, "a list"),
    ("selection_values", Mapping, "an object"),
)


def _json_kind(value: Any) -> str:
    """``value``'s type in JSON's words ("a string", "a list" ...)."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, Mapping):
        return "an object"
    return f"a {type(value).__name__}"


def shape_problems(
    config: Mapping[str, Any] | None = None,
    best_model_meta: Mapping[str, Any] | None = None,
) -> list[str]:
    """The fields :func:`build_run_record` reads whose type is unexpected.

    :func:`build_run_record` reads each as missing rather than failing;
    the index reports them as row problems, naming the file and field
    (``config.json train_config.warm_start is a string, not an object``).
    ``null`` is never a problem.
    """
    problems: list[str] = []
    for dotted in _CONFIG_OBJECTS:
        value: Any = config or {}
        for key in dotted.split("."):
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is not None and not isinstance(value, Mapping):
            problems.append(
                f"config.json {dotted} is {_json_kind(value)}, not an object; "
                f"read as missing"
            )
    for key, wanted, kind in _META_FIELDS:
        value = (best_model_meta or {}).get(key)
        if value is not None and not isinstance(value, wanted):
            problems.append(
                f"best_model_meta.json {key} is {_json_kind(value)}, not "
                f"{kind}; read as missing"
            )
    return problems


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
    best_model_dir: str | os.PathLike[str] | None = None,
    wall_clock: Mapping[str, Any] | None = None,
    preflight_warnings: Sequence[str] = (),
    artifacts: Sequence[Mapping[str, str]] = (),
    sync: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    snapshots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One run's record, from its ``config.json`` payload and its outcome.

    Every field the run's ``config.json`` already records is copied from
    it (one source, no second derivation): the algorithm, the recipe,
    the task's env classes and observation fingerprint, the
    comparability key, the seeds, the git sha and the library versions.
    The task's constructor kwargs are recorded as the sha256 of their
    canonical JSON. ``task.eval_protocol`` is the comparability key's
    (the in-loop selection protocol's id and fingerprint,
    ``experiments.protocols.eval_protocol_record``), ``None`` for a run
    recorded before protocols were versioned. The ``experiment`` block
    holds the placeholders the spec system will fill. A ``config.json``
    object of the wrong type is read as missing (:func:`shape_problems`).

    ``best_model_dir`` is the directory holding ``best_model_meta.json``
    (``train()`` passes it): the selection is then checked against the
    files the meta binds (:func:`selection_from_best_model_meta`), and
    an unbound selection leaves the best-model triple out of
    ``artifacts`` (:data:`BEST_MODEL_ARTIFACTS`): those files are on
    disk, but they are not one save.

    ``snapshots`` is ``experiments.snapshots.snapshot_summary``'s block
    (the evaluation snapshots ``snapshots/index.json`` binds), ``None``
    for a run that took none.
    """
    if status not in RUN_STATUSES:
        raise ValueError(
            f"run status must be one of {list(RUN_STATUSES)}, got {status!r}"
        )
    train_config = _mapping(config.get("train_config"))
    env = _mapping(config.get("env"))
    evaluation_env = _mapping(config.get("evaluation_env"))
    algorithm = _mapping(config.get("algorithm"))
    eval_protocol = _mapping(config.get("comparability_key")).get("eval_protocol")
    versions = config.get("versions")
    selection = selection_from_best_model_meta(
        best_model_meta, directory=best_model_dir
    )
    if selection is not None and selection["bindings"] == BINDINGS_UNBOUND:
        artifacts = [
            artifact
            for artifact in artifacts
            if artifact.get("name") not in BEST_MODEL_ARTIFACTS
        ]
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
            "eval_protocol": (
                dict(eval_protocol) if isinstance(eval_protocol, Mapping) else None
            ),
        },
        "comparability_key": config.get("comparability_key"),
        "seeds": {
            "seed": train_config.get("seed"),
            "evaluation_seeding": config.get("evaluation_seeding"),
        },
        "git": {"sha": config.get("git_sha"), "dirty": git_dirty},
        "versions": dict(versions) if isinstance(versions, Mapping) else None,
        "warm_start": _warm_start_section(config),
        "selection": selection,
        "wall_clock": dict(wall_clock) if wall_clock else None,
        "preflight_warnings": list(preflight_warnings),
        "sync": dict(sync) if sync else None,
        "snapshots": dict(snapshots) if snapshots else None,
        "experiment": {"id": None, "parent": None, "lever": None, "kind": None},
        "artifacts": [dict(artifact) for artifact in artifacts],
    }


def write_run_record(
    run_dir: str | os.PathLike[str], record: Mapping[str, Any]
) -> Path:
    """Atomically (re)write ``run_dir/run_record.json``."""
    return write_json_atomic(Path(run_dir) / RUN_RECORD_NAME, dict(record))


def describe_read_error(error: BaseException) -> str:
    """An I/O or decode error's type and message, without a path or bytes.

    ``repr(UnicodeDecodeError)`` embeds the whole file's bytes and an
    ``OSError``'s message its absolute path; neither belongs in a row,
    a record or a problem line.
    """
    if isinstance(error, OSError) and error.strerror:
        return f"{type(error).__name__}: {error.strerror}"
    return f"{type(error).__name__}: {error}"


def read_json_object(
    path: str | os.PathLike[str], *, label: str | None = None
) -> dict[str, Any]:
    """Load a JSON object; ``ValueError`` (naming the file) on anything else.

    The error names the file as ``label`` (default: ``path`` itself), so
    a caller can keep machine paths out of what it reports.
    """
    name = str(path) if label is None else label
    try:
        # utf-8-sig: a byte-order mark (a Windows editor's) is not damage.
        with open(path, encoding="utf-8-sig") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{name} is unreadable: {describe_read_error(error)}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} does not hold a JSON object")
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
    manifest verifies). ``unlisted`` names every file on disk the
    manifest does not list: written after it (a notebook's plots and
    replay video, :func:`is_post_training_path`), by a sync that failed
    before its new manifest, or by anything else. ``problems`` says, in
    words, why the directory is incomplete.
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


def is_temporary_path(relative: str) -> bool:
    """Whether ``relative`` names a writer's temporary file (a leftover)."""
    return bool(_TEMPORARY_SUFFIX.search(PurePosixPath(relative).name))


def is_post_training_path(relative: str) -> bool:
    """Whether a notebook writes ``relative`` after ``train()`` sealed the run.

    The post-training plots, audits and gate reports under ``reports/``
    and the replay video (:data:`POST_TRAINING_DIRS`,
    :data:`POST_TRAINING_FILES`); never a temporary file.
    """
    if is_temporary_path(relative):
        return False
    return relative in POST_TRAINING_FILES or relative.startswith(
        POST_TRAINING_DIRS
    )


def verify_run_dir(
    run_dir: str | os.PathLike[str], *, strict: bool = False
) -> RunDirVerification:
    """Check a run directory against its ``sync_manifest.json``.

    Complete when the manifest exists, every file it lists is on disk
    with its recorded size and sha256, it lists ``run_record.json``, that
    record's status is terminal, every file it does not list is a
    post-training file (:func:`is_post_training_path`: a notebook's
    plots, audits and replay video beside a sealed run), and no file on
    disk or in the manifest is a writer's temporary leftover
    (:func:`is_temporary_path`: ``.x.tmp-<pid>``, ``.x.sync-tmp``,
    ``x.tmp.npz`` ...). A leftover is an interrupted write: it fails the
    directory whether the manifest lists it (it was there when the run
    was sealed) or not, even under ``reports/``. Every unlisted file is
    reported in ``unlisted``; one outside the post-training locations
    fails the directory, and under ``strict`` every unlisted file does.
    A mid-run Drive mirror whose manifest verifies is an intact snapshot
    of an unfinished run: its listed record says ``"running"``, so it is
    incomplete too. A directory whose record is terminal but which has
    no manifest was never sealed: a mirror pass that did not finish, or
    a run that died while sealing itself.
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
        if record_status in TERMINAL_STATUSES:
            problem = (
                f"no {SYNC_MANIFEST_NAME}, though its run record says "
                f"{record_status!r}: the directory was never sealed -- a "
                f"mirror pass that did not finish (re-run "
                f"experiments.sync.sync_run_dir from the sealed local run "
                f"directory), or a run that died while sealing itself"
            )
        else:
            cause = (
                f"its run record says {record_status!r}"
                if record_status is not None
                else "there is no readable run_record.json either"
            )
            problem = (
                f"no {SYNC_MANIFEST_NAME} ({cause}): the run never reached a "
                f"terminal path (still training, killed, a disconnected "
                f"runtime) or predates run records"
            )
        return RunDirVerification(
            run_dir=str(root),
            complete=False,
            manifest="missing",
            status=record_status,
            files_checked=0,
            problems=(problem,),
        )
    try:
        entries = _manifest_entries(
            read_json_object(manifest_path, label=SYNC_MANIFEST_NAME)
        )
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
    on_disk = run_files(root)
    unlisted = tuple(relative for relative in on_disk if relative not in entries)
    leftovers = sorted(
        relative
        for relative in set(on_disk) | set(entries)
        if is_temporary_path(relative)
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
    if leftovers:
        problems.append(
            f"{len(leftovers)} temporary file(s) left by an interrupted "
            f"write, listed in {SYNC_MANIFEST_NAME} or not: "
            + ", ".join(leftovers[:5])
            + (" ..." if len(leftovers) > 5 else "")
        )
    refused = tuple(
        path
        for path in unlisted
        if not is_temporary_path(path)
        and (strict or not is_post_training_path(path))
    )
    if refused:
        where = (
            "(strict)"
            if strict
            else "outside the post-training locations (reports/, "
            "media/best_model.mp4)"
        )
        problems.append(
            f"{len(refused)} file(s) are not in {SYNC_MANIFEST_NAME} "
            f"{where}: " + ", ".join(refused[:5])
            + (" ..." if len(refused) > 5 else "")
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
