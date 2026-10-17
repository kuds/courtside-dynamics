"""Regenerate the run index: one normalized row per run directory.

``docs/design_campaign_structure_20261016.md`` step L1, second half.
Each run directory's own files are the source of truth; the index is
derived from them, rebuilt whole on every call (never appended to), and
byte-identical while the run directories are unchanged.

Usage::

    python -m courtside_dynamics.experiments.index --root <runs root>
    python -m courtside_dynamics.experiments.index --root <runs root> \\
        --out runs.jsonl --legacy-levers my_levers.toml --skip-verify

writes ``<root>/index/runs.jsonl`` (or ``--out``): one JSON object per
line, sorted by the run directory's path under the root, keys sorted.
What becomes a row:

- A directory holding ``run_record.json`` (``train()`` since design L1):
  the record as written, plus its manifest verdict
  (:func:`~courtside_dynamics.experiments.records.verify_run_dir`;
  ``--skip-verify`` leaves it unchecked, which avoids re-reading every
  checkpoint on a Drive mount). A record that still says ``"running"``
  is a run that is training or died without reaching a terminal path
  (a killed VM): the row says so in ``index.problems`` -- with or
  without verification -- and is a stub when no final model exists,
  as a backfilled launch would be.
- A directory holding a courtside ``config.json`` but no record (every
  run before run records): backfilled into the record's shape from
  ``config.json`` (git sha, recipe, algorithm, seeds, env kwargs, warm
  start, budget), ``stage_summary.txt`` (status, stop reason, steps
  trained, best and headline lines), ``best_model_meta.json`` (the
  selection) and which model, checkpoint and video files exist;
  ``index.version_inferred`` is true and ``index.inferred`` names the
  fields that were derived rather than copied. Without
  ``stage_summary.txt`` the run never reached a terminal path: status
  ``"aborted"``, and a *stub* when no final model was saved either. A
  summary that exists but cannot be read leaves the status ``None``
  with a problem: the run may well have finished.
- A directory named like a run (``YYYYMMDD_HHMMSS``) whose
  ``config.json`` cannot be read or is not a courtside run config (a
  zero-byte file from a partial Drive sync): backfilled without the
  config's fields, the config problem in ``index.problems``.
- An empty directory named like a run: an aborted stub, a launch that
  died before writing ``config.json``.

Nothing else is parsed. A directory that holds files but none of
these, or whose ``config.json`` is not a courtside run config, is
listed under ``skipped`` with the reason, and its contents are never
read as a run. A run directory is not searched for further runs unless
a subdirectory holds a run marker (``config.json``, ``run_record.json``
or a run-named directory): then the scan goes on into it and the outer
row (or skip) names the nested directories, because a stray
``config.json`` in a recipe folder would otherwise hide every run
beneath it.

Every row joins the curated campaign map (``legacy_levers.toml``,
package data: run id -> campaign, lever, verdict, doc) under
``campaign``, ``None`` for a run the map does not name. A backfilled
row holds no machine path: the warm start is named by its source
directory's basename, the run config file by its basename,
``index.path`` is relative to the root, and every problem names files
relative to the run directory.

Stable-Baselines3 and torch are never imported here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from courtside_dynamics.experiments.records import (
    COMPLETED,
    CRASHED,
    EARLY_STOP,
    INTERRUPTED,
    RUN_RECORD_NAME,
    RUNNING,
    SYNC_MANIFEST_NAME,
    TERMINAL_STATUSES,
    build_run_record,
    describe_read_error,
    read_json_object,
    verify_run_dir,
)

#: Tag of a row's ``index`` block; bump it with any incompatible change.
RUN_INDEX_SCHEMA = "courtside-run-index/1"

#: ``<root>/index/runs.jsonl`` unless ``--out`` says otherwise. The
#: index directory is never scanned for runs.
INDEX_DIR_NAME = "index"
INDEX_FILE_NAME = "runs.jsonl"

#: Package data: run id -> campaign, lever, verdict, doc.
LEGACY_LEVERS_NAME = "legacy_levers.toml"

#: A legacy directory that never wrote ``stage_summary.txt``: the run
#: ended before any terminal path (a dead runtime, a false start). Only
#: the index uses it; a run record's own statuses are
#: ``records.RUN_STATUSES``.
ABORTED = "aborted"

#: Run-dir-relative locations read here, as (0.14 layout, pre-0.14 flat
#: layout). ``training.artifacts.RUN_LAYOUT`` and its legacy fallback
#: hold the same paths; a test pins the agreement, because importing the
#: training package would import Stable-Baselines3.
_LAYOUT: dict[str, tuple[str, str]] = {
    "config": ("config.json", "config.json"),
    "stage_summary": ("stage_summary.txt", "stage_summary.txt"),
    "best_model": ("model/best_model.zip", "best_model.zip"),
    "best_model_meta": ("model/best_model_meta.json", "best_model_meta.json"),
    "final_model": ("model/final_model.zip", "final_model.zip"),
    "checkpoints_dir": ("checkpoints", "checkpoints"),
    "videos_dir": ("media/videos", "videos"),
    "best_model_video": ("media/best_model.mp4", "best_model.mp4"),
}

#: ``config.json``'s ``algorithm`` block arrived with first-class PPO;
#: older runs name only ``train_config.algo``, whose family this maps
#: (``training.algos.algo_family``; a test pins the agreement).
_ALGO_FAMILIES = {"SAC": "SAC", "DemoSAC": "SAC", "PPO": "PPO"}

_RUN_ID = re.compile(r"\d{8}_\d{6}")
_CHECKPOINT_STEP = re.compile(r"_(\d+)_steps\.zip$")
_SUMMARY_LINE = re.compile(r"([A-Z][A-Za-z0-9_ ]*):\s*(.*)")
_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
#: "(at 4,096 steps)", or an on-policy run's "(at 4,096 steps, SB3
#: n_updates 0)".
_STEPS = re.compile(r"\(at ([\d,]+) steps[,)]")


@dataclass(frozen=True)
class SkippedDir:
    """A directory under the root that the index did not read as a run."""

    path: str
    reason: str


@dataclass(frozen=True)
class RunIndex:
    """:func:`build_index`'s result: the rows and what was passed over."""

    rows: tuple[dict[str, Any], ...]
    skipped: tuple[SkippedDir, ...] = ()
    campaign_entries: int = 0

    def lines(self) -> list[str]:
        """``runs.jsonl``'s lines: sorted keys, one row per line."""
        return [json.dumps(row, sort_keys=True) for row in self.rows]


@dataclass
class _Scan:
    runs: list[tuple[Path, str]] = field(default_factory=list)
    skipped: list[SkippedDir] = field(default_factory=list)
    # Row problems found while scanning (nested run directories).
    notes: dict[Path, list[str]] = field(default_factory=dict)


def _locate(run_dir: Path, name: str) -> Path | None:
    for relative in _LAYOUT[name]:
        path = run_dir / relative
        if path.exists():
            return path
    return None


def _has_files(directory: Path) -> bool:
    return any(files for _root, _dirs, files in os.walk(directory))


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _budget(value: Any) -> int | None:
    """A ``total_timesteps`` as an int: an int, or an integral float
    (``TrainConfig(total_timesteps=1e6)`` records ``1000000.0``)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _courtside_config(config: Any) -> bool:
    """Whether a ``config.json`` payload is a courtside run config."""
    if not isinstance(config, dict):
        return False
    train_config = config.get("train_config")
    return (
        isinstance(train_config, dict)
        and isinstance(train_config.get("algo"), str)
        and _budget(train_config.get("total_timesteps")) is not None
    )


_NOT_COURTSIDE = (
    "config.json is not a courtside run config (no train_config with algo "
    "and an integral total_timesteps)"
)


def _read_json(run_dir: Path, relative: str) -> dict[str, Any]:
    """A JSON object from ``run_dir/relative``; errors name ``relative``."""
    return read_json_object(run_dir / relative, label=relative)


def _nested_runs(here: Path, subdirs: list[str]) -> list[str]:
    """The subdirectories of ``here`` that hold a run marker at any depth.

    A run-named directory, or one holding ``config.json`` or
    ``run_record.json`` somewhere below. A real run directory's own
    folders (``model/``, ``metrics/``, ``checkpoints/`` ...) hold none.
    """
    nested = []
    for name in subdirs:
        if _RUN_ID.fullmatch(name) or any(
            RUN_RECORD_NAME in files or "config.json" in files
            for _root, _dirs, files in os.walk(here / name)
        ):
            nested.append(name)
    return nested


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() or "."


def _scan(root: Path, out: Path | None) -> _Scan:
    """Walk ``root`` in sorted order; classify run directories."""
    scan = _Scan()
    index_dir = root / INDEX_DIR_NAME
    written = out.resolve() if out is not None else None
    for directory, subdirs, files in os.walk(root):
        here = Path(directory)
        subdirs.sort()
        if here == index_dir:
            subdirs[:] = []
            continue
        names = {
            name
            for name in files
            if written is None
            or name != written.name
            or (here / name).resolve() != written
        }
        rel = _relative(here, root)
        run_named = here != root and bool(_RUN_ID.fullmatch(here.name))
        if RUN_RECORD_NAME in names or "config.json" in names:
            # Only runs nested below (a stray config.json in a recipe
            # folder) keep the walk going; a run's own folders do not.
            subdirs[:] = _nested_runs(here, subdirs)
            nested = (
                f"holds nested run directories ({', '.join(subdirs)}), "
                f"indexed as rows of their own"
                if subdirs
                else None
            )
            reason: str | None = None
            if RUN_RECORD_NAME not in names:
                try:
                    config = _read_json(here, "config.json")
                except ValueError as error:
                    reason = str(error)
                else:
                    if not _courtside_config(config):
                        reason = _NOT_COURTSIDE
            if reason is None or run_named:
                # A run-named directory with a broken config.json is a
                # dead launch (a zero-byte file from a partial sync), not
                # something else: a row, the config problem among its
                # problems.
                kind = "record" if RUN_RECORD_NAME in names else "legacy"
                scan.runs.append((here, kind))
                if nested:
                    scan.notes.setdefault(here, []).append(nested)
            else:
                scan.skipped.append(
                    SkippedDir(rel, reason + (f"; it {nested}" if nested else ""))
                )
            continue
        if run_named:
            subdirs[:] = _nested_runs(here, subdirs)
            if _has_files(here):
                scan.skipped.append(
                    SkippedDir(
                        rel,
                        "named like a run but holds no config.json or "
                        f"{RUN_RECORD_NAME}",
                    )
                )
            else:
                scan.runs.append((here, "empty"))
            continue
        if names and here != root:
            scan.skipped.append(
                SkippedDir(rel, "holds files but is not a run directory")
            )
    return scan


def parse_stage_summary(text: str) -> dict[str, Any]:
    """The head block of a ``stage_summary.txt`` as typed fields.

    Reads the ``Label: value`` lines before the first section; a field
    whose line is absent (older summaries lack several) is ``None``.
    ``budget_timesteps`` is set only by the ``X of Y budget`` form,
    written when a run trained less than its budget or is on-policy.
    """
    head: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("-" * 10):
            break
        match = _SUMMARY_LINE.fullmatch(line.rstrip())
        if match and match.group(1) not in head:
            head[match.group(1)] = match.group(2).strip()

    def number_at(label: str, pattern: str) -> re.Match[str] | None:
        value = head.get(label)
        return re.match(pattern, value) if value is not None else None

    parsed: dict[str, Any] = {
        "status": head.get("Status"),
        "stop_reason": head.get("Stop reason"),
        "date": head.get("Date"),
        "git_sha": None,
        "run_config": head.get("Run config"),
        "trained_timesteps": None,
        "budget_timesteps": None,
        "n_updates": None,
        "duration_seconds": None,
        "best_eval_step": None,
        "best_model_step": None,
        "headline": head.get("Headline"),
        "headline_final": None,
        "headline_best": None,
        "headline_best_step": None,
    }
    sha = head.get("Git SHA")
    if sha and sha != "unknown":
        parsed["git_sha"] = sha
    if match := number_at("Timesteps", r"([\d,]+)(?: of ([\d,]+) budget)?"):
        parsed["trained_timesteps"] = _int(match.group(1))
        if match.group(2):
            parsed["budget_timesteps"] = _int(match.group(2))
    if match := number_at("SB3 n_updates", r"([\d,]+)"):
        parsed["n_updates"] = _int(match.group(1))
    if match := number_at("Duration", r"(\d+)h (\d+)m (\d+)s"):
        hours, minutes, seconds = (int(part) for part in match.groups())
        parsed["duration_seconds"] = hours * 3600 + minutes * 60 + seconds
    if (value := head.get("Best eval")) and (match := _STEPS.search(value)):
        parsed["best_eval_step"] = _int(match.group(1))
    if match := number_at("Best model", r"step ([\d,]+)"):
        parsed["best_model_step"] = _int(match.group(1))
    if match := number_at("Headline final", f"({_NUMBER})"):
        parsed["headline_final"] = float(match.group(1))
    if match := number_at("Headline best", rf"({_NUMBER}) \(at ([\d,]+) steps\)"):
        parsed["headline_best"] = float(match.group(1))
        parsed["headline_best_step"] = _int(match.group(2))
    return parsed


def _files(run_dir: Path) -> dict[str, Any]:
    """Which artifacts exist (either layout); checkpoints and videos counted.

    ``videos`` counts the periodic training videos; ``best_model_video``
    is the replay a notebook records after training (``best_model.mp4``
    at the root of a pre-0.14 flat-layout run).
    """
    present: dict[str, Any] = {
        name: _locate(run_dir, name) is not None
        for name in (
            "config",
            "stage_summary",
            "best_model",
            "best_model_meta",
            "final_model",
            "best_model_video",
        )
    }
    present["run_record"] = (run_dir / RUN_RECORD_NAME).is_file()
    present["sync_manifest"] = (run_dir / SYNC_MANIFEST_NAME).is_file()
    checkpoints = _locate(run_dir, "checkpoints_dir")
    zips = sorted(checkpoints.glob("*.zip")) if checkpoints else []
    steps = [
        int(match.group(1))
        for path in zips
        if (match := _CHECKPOINT_STEP.search(path.name))
    ]
    videos = _locate(run_dir, "videos_dir")
    present["checkpoints"] = len(zips)
    present["latest_checkpoint_step"] = max(steps) if steps else None
    present["videos"] = (
        sum(1 for path in videos.rglob("*.mp4") if path.is_file()) if videos else 0
    )
    return present


def _run_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The run config file a run loaded, by basename and sha256."""
    entry = (config or {}).get("run_config_file")
    if not isinstance(entry, Mapping):
        return None
    path = entry.get("path")
    return {
        "name": os.path.basename(str(path)) if path else None,
        "sha256": entry.get("sha256"),
    }


def _legacy_outcome(
    summary: Mapping[str, Any] | None,
    budget: Any,
    problems: list[str],
    *,
    summary_present: bool,
) -> tuple[str | None, str | None, int | None]:
    """(status, stop_reason, trained) in run-record terms.

    ``stage_summary.txt`` says ``completed`` for an early stop too; the
    ``X of Y budget (stopped early)`` timesteps form tells them apart. A
    stop reason on a run that trained its whole budget is the train-3
    label (the guard fired once the budget was trained): ``completed``
    with a ``full_budget`` reason, exactly as ``train()`` records it now.
    Only a *missing* summary means no terminal path was reached
    (``aborted``); one that exists but cannot be read leaves the status
    unknown (``None``; the caller recorded why). Without a budget in
    ``config.json`` the summary's ``of Y budget`` form names it, and a
    summary without that form trained exactly its budget.
    """
    if summary is None:
        return (None if summary_present else ABORTED), None, None
    raw_status = summary["status"]
    raw_reason = summary["stop_reason"]
    trained = summary["trained_timesteps"]
    summary_budget = summary["budget_timesteps"]
    configured = _budget(budget)
    if summary_budget is not None and configured is not None and (
        summary_budget != configured
    ):
        problems.append(
            f"stage_summary.txt budget {summary_budget:,} differs from "
            f"config.json total_timesteps {budget!r}"
        )
    if raw_status in (INTERRUPTED, CRASHED):
        return raw_status, raw_reason, trained
    if raw_status != COMPLETED:
        problems.append(f"stage_summary.txt has no known Status ({raw_status!r})")
        return None, raw_reason, trained
    if trained is None:
        problems.append("stage_summary.txt has no readable Timesteps line")
        return COMPLETED, raw_reason, trained
    if configured is not None:
        budget = configured
    elif summary_budget is not None:
        budget = summary_budget
    else:
        budget = trained
    if trained >= budget:
        if raw_reason is None:
            return COMPLETED, None, trained
        return (
            COMPLETED,
            f"full_budget: trained all {budget:,} steps; the stop condition "
            f"also held at the final evaluation ({raw_reason})",
            trained,
        )
    if raw_reason is None:
        raw_reason = (
            f"stopped_early: trained {trained:,} of {budget:,} steps; the "
            f"summary records no stop reason"
        )
    return EARLY_STOP, raw_reason, trained


def _read_summary(run_dir: Path, problems: list[str]) -> dict[str, Any] | None:
    """The parsed summary head; ``None`` when missing or unreadable.

    An unreadable one (bytes that are not UTF-8, written under another
    locale before the summary was written as UTF-8) is a problem,
    described by its error type and position, never its contents.
    """
    path = _locate(run_dir, "stage_summary")
    if path is None:
        return None
    try:
        return parse_stage_summary(path.read_text("utf-8-sig"))
    except (OSError, UnicodeDecodeError) as error:
        problems.append(
            f"stage_summary.txt is unreadable ({describe_read_error(error)})"
        )
        return None


def _index_block(
    rel: str,
    *,
    files: Mapping[str, Any],
    source: str,
    config: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    inferred: list[str],
    stub: bool,
    verification: Mapping[str, Any] | None,
    problems: list[str],
) -> dict[str, Any]:
    return {
        "schema": RUN_INDEX_SCHEMA,
        "path": rel,
        "source": source,
        "version_inferred": source == "legacy",
        "inferred": sorted(inferred),
        "stub": stub,
        "verification": dict(verification) if verification else None,
        "files": dict(files),
        "run_config": _run_config(config),
        "summary": dict(summary) if summary else None,
        "problems": list(problems),
    }


def _legacy_row(
    run_dir: Path, rel: str, config: Mapping[str, Any] | None, problems: list[str]
) -> dict[str, Any]:
    """Backfill a pre-record run directory into the run record's shape.

    :func:`~courtside_dynamics.experiments.records.build_run_record`
    copies every ``config.json`` field (one derivation for old and new
    runs); the outcome comes from ``stage_summary.txt``.
    """
    config = dict(config or {})
    summary = _read_summary(run_dir, problems)
    meta: dict[str, Any] | None = None
    meta_path = _locate(run_dir, "best_model_meta")
    if meta_path is not None:
        try:
            meta = _read_json(run_dir, meta_path.relative_to(run_dir).as_posix())
        except ValueError as error:
            problems.append(f"the selection is omitted: {error}")
    train_config = config.get("train_config") or {}
    status, stop_reason, trained = _legacy_outcome(
        summary,
        train_config.get("total_timesteps"),
        problems,
        summary_present=_locate(run_dir, "stage_summary") is not None,
    )
    # build_run_record accepts only run-record statuses and a creation
    # time; "aborted", an unknown status and a missing time are set after.
    record = build_run_record(
        config,
        run_id=run_dir.name,
        created_at="",
        status=RUNNING,
        stop_reason=stop_reason,
        trained_timesteps=trained,
        n_updates=summary["n_updates"] if summary else None,
        best_model_meta=meta,
    )
    record["created_at"] = config.get("timestamp_utc")
    record["status"] = status
    record["artifacts"] = None
    inferred = ["status"]
    if stop_reason is not None:
        inferred.append("stop_reason")
    if trained is not None:
        inferred.append("budget.trained_timesteps")
    if not config.get("algorithm") and train_config.get("algo"):
        name = train_config["algo"]
        versions = config.get("versions") or {}
        sb3 = versions.get("stable_baselines3")
        record["algo"] = {
            "name": name,
            "family": _ALGO_FAMILIES.get(name),
            "impl": f"sb3=={sb3}" if sb3 else None,
        }
        inferred.append("algo")
    if summary and summary["duration_seconds"] is not None:
        record["wall_clock"] = {
            "total_seconds": summary["duration_seconds"],
            "periodic_eval_seconds": None,
            "closing_eval_seconds": None,
            "eval_share": None,
        }
        inferred.append("wall_clock")
    files = _files(run_dir)
    record["index"] = _index_block(
        rel,
        files=files,
        source="legacy",
        config=config or None,
        summary=summary,
        inferred=inferred,
        stub=status == ABORTED and not files["final_model"],
        verification=None,
        problems=problems,
    )
    return record


def _record_row(
    run_dir: Path, rel: str, *, verify: bool, problems: list[str]
) -> dict[str, Any] | None:
    """A run record as written, plus its manifest verdict.

    A record that is not terminal adds a problem whether or not the
    manifest is verified (``--skip-verify`` is the Colab default): one
    that still says ``"running"`` without a manifest is a run that is
    training or died (the R3 stub that looked live), and with one it is
    a mid-run mirror snapshot. Either is a stub when no final model
    exists, as a backfilled launch would be.
    """
    try:
        record = _read_json(run_dir, RUN_RECORD_NAME)
    except ValueError as error:
        problems.append(f"{error}; backfilled from config.json instead")
        return None
    config: dict[str, Any] | None = None
    if _locate(run_dir, "config") is not None:
        try:
            config = _read_json(run_dir, "config.json")
        except ValueError as error:
            problems.append(str(error))
    if verify:
        result = verify_run_dir(run_dir)
        verification: dict[str, Any] = {
            "manifest": result.manifest,
            "complete": result.complete,
            "files_checked": result.files_checked,
            "problems": list(result.problems),
            "unlisted": list(result.unlisted),
        }
    else:
        verification = {
            "manifest": "not_checked",
            "complete": None,
            "files_checked": 0,
            "problems": [],
            "unlisted": [],
        }
    files = _files(run_dir)
    status = record.get("status")
    terminal = status in TERMINAL_STATUSES
    if status == RUNNING and files["sync_manifest"]:
        problems.append(
            f"{RUN_RECORD_NAME} says 'running' and {SYNC_MANIFEST_NAME} "
            f"lists it: a mid-run mirror snapshot, not a finished run"
        )
    elif status == RUNNING:
        problems.append(
            f"{RUN_RECORD_NAME} says 'running' and there is no "
            f"{SYNC_MANIFEST_NAME}: the run is still training, or died "
            f"without reaching a terminal path"
        )
    elif not terminal:
        problems.append(f"{RUN_RECORD_NAME} has no known status ({status!r})")
    row = dict(record)
    row["index"] = _index_block(
        rel,
        files=files,
        source="run_record",
        config=config,
        summary=_read_summary(run_dir, problems),
        inferred=[],
        stub=not terminal and not files["final_model"],
        verification=verification,
        problems=problems,
    )
    return row


_CAMPAIGN_KEYS = frozenset({"campaign", "lever", "verdict", "doc", "note"})
_CAMPAIGN_REQUIRED = frozenset({"campaign", "lever", "verdict"})


def load_legacy_levers(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, str]]:
    """The campaign map: run id -> {campaign, lever, verdict, doc[, note]}.

    ``path=None`` reads the packaged ``legacy_levers.toml``. Raises
    ``ValueError`` (naming the file and run) on an unknown key, a
    missing required key, a non-string value, or an entry with neither
    ``doc`` nor ``note``.
    """
    if path is None:
        source = LEGACY_LEVERS_NAME
        text = (
            resources.files("courtside_dynamics.experiments")
            .joinpath(LEGACY_LEVERS_NAME)
            .read_text("utf-8")
        )
    else:
        source = os.fspath(path)
        text = Path(path).read_text("utf-8")
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{source} is not valid TOML: {error}") from error
    if set(payload) - {"runs"}:
        raise ValueError(
            f"{source}: unknown top-level key(s) {sorted(set(payload) - {'runs'})}; "
            f"only [runs.<run id>] tables are allowed"
        )
    runs = payload.get("runs") or {}
    campaign_map: dict[str, dict[str, str]] = {}
    for run_id, entry in runs.items():
        where = f"{source} [runs.{run_id!r}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a table")
        unknown = set(entry) - _CAMPAIGN_KEYS
        missing = _CAMPAIGN_REQUIRED - set(entry)
        if unknown or missing:
            raise ValueError(
                f"{where}: unknown key(s) {sorted(unknown)}, missing key(s) "
                f"{sorted(missing)}"
            )
        if not all(isinstance(value, str) and value for value in entry.values()):
            raise ValueError(f"{where}: every value must be a non-empty string")
        if "doc" not in entry and "note" not in entry:
            raise ValueError(
                f"{where}: without a doc, a note must say why no doc names the run"
            )
        campaign_map[str(run_id)] = dict(entry)
    return campaign_map


def build_index(
    root: str | os.PathLike[str],
    *,
    legacy_levers: Mapping[str, Mapping[str, str]] | None = None,
    verify: bool = True,
    out: str | os.PathLike[str] | None = None,
) -> RunIndex:
    """Index every run directory under ``root`` (nothing is written).

    ``legacy_levers`` defaults to the packaged campaign map
    (:func:`load_legacy_levers`). ``verify=False`` skips the manifest
    check of run-record directories. ``out``, when it lies under
    ``root``, is ignored by the scan.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"runs root {base} is not a directory")
    campaign_map = (
        load_legacy_levers() if legacy_levers is None else dict(legacy_levers)
    )
    scan = _scan(base, Path(out) if out is not None else None)
    rows: list[dict[str, Any]] = []
    for run_dir, kind in scan.runs:
        rel = _relative(run_dir, base)
        problems: list[str] = list(scan.notes.get(run_dir, ()))
        row: dict[str, Any] | None = None
        if kind == "record":
            row = _record_row(run_dir, rel, verify=verify, problems=problems)
        if row is None:
            config: dict[str, Any] | None = None
            if kind != "empty" and _locate(run_dir, "config") is not None:
                try:
                    config = _read_json(run_dir, "config.json")
                except ValueError as error:
                    problems.append(f"{error}; the row has none of its fields")
                else:
                    if not _courtside_config(config):
                        problems.append(
                            f"{_NOT_COURTSIDE}; the row has none of its fields"
                        )
                        config = None
            row = _legacy_row(run_dir, rel, config, problems)
        entry = campaign_map.get(str(row.get("run_id")))
        row["campaign"] = dict(entry) if entry is not None else None
        rows.append(row)
    rows.sort(key=lambda row: row["index"]["path"])
    return RunIndex(
        rows=tuple(rows),
        skipped=tuple(scan.skipped),
        campaign_entries=len(campaign_map),
    )


def write_index(index: RunIndex, path: str | os.PathLike[str]) -> Path:
    """Write ``runs.jsonl`` atomically (temporary file, then a rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
            for line in index.lines():
                stream.write(line + "\n")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", required=True, help="the runs root (e.g. Drive training_runs/)"
    )
    parser.add_argument(
        "--out",
        help=f"output file (default: <root>/{INDEX_DIR_NAME}/{INDEX_FILE_NAME})",
    )
    parser.add_argument(
        "--legacy-levers",
        help=f"campaign map TOML (default: the packaged {LEGACY_LEVERS_NAME})",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="do not re-hash run-record directories against their manifests",
    )
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.root) / INDEX_DIR_NAME / INDEX_FILE_NAME
    index = build_index(
        args.root,
        legacy_levers=load_legacy_levers(args.legacy_levers),
        verify=not args.skip_verify,
        out=out,
    )
    write_index(index, out)
    rows = index.rows
    records = sum(1 for row in rows if row["index"]["source"] == "run_record")
    running = sum(
        1
        for row in rows
        if row["index"]["source"] == "run_record" and row["status"] == RUNNING
    )
    aborted = sum(1 for row in rows if row["status"] == ABORTED)
    stubs = sum(1 for row in rows if row["index"]["stub"])
    joined = sum(1 for row in rows if row["campaign"] is not None)
    print(
        f"{out}: {len(rows)} run(s): {records} from run records ({running} "
        f"still say running: training, or dead), {len(rows) - records} "
        f"backfilled ({aborted} aborted); {stubs} stub(s); {joined} joined "
        f"to the campaign map ({index.campaign_entries} entries); "
        f"{len(index.skipped)} director"
        f"{'y' if len(index.skipped) == 1 else 'ies'} skipped"
    )
    for skipped in index.skipped:
        print(f"  skipped {skipped.path}: {skipped.reason}")
    for row in rows:
        for problem in row["index"]["problems"]:
            print(f"  WARNING {row['index']['path']}: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
