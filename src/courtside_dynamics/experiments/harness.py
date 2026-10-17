"""The offline evaluation harness: score any saved policy under any protocol.

``docs/design_env_ladder_and_structure_20261016.md`` section 4 ("Harness")
and M3; ``docs/design_campaign_structure_20261016.md`` step L2.

- :func:`evaluate` scores one model (and its ``VecNormalize``
  statistics) under an :class:`~courtside_dynamics.experiments.protocols.EvalProtocol`
  and returns the metrics row. It runs the in-loop evaluator's own
  measurement core (``callbacks.info_dict_eval.collect_episode_metrics``)
  on an evaluation env built the way ``train()`` builds the info
  evaluator's (``make_vec_env``, one worker, Monitor-wrapped, the
  normalizer frozen with raw rewards). So, on the same device, a
  snapshot's row under the run's own protocol (``inloop``) equals that
  evaluation's ``eval_info.csv`` row bit for bit for an ungated run:
  every evaluation's when the run evaluated paired, only the first
  one's when it evaluated unpaired (its in-loop stream continues
  across evaluations, and its confirmation block is a fresh block with
  no in-loop counterpart). A performance-gated run has no such
  protocol (``protocols.GatedEvaluationError``).
- :func:`rescore` scores a run's saved artifacts -- its best model, its
  evaluation snapshots (``snapshots/``), its periodic checkpoints,
  including the pre-snapshot layout of the August runs
  (``checkpoints/<prefix>_<N>_steps.zip`` with
  ``<prefix>_vecnormalize_<N>_steps.pkl``) -- and writes one row per
  artifact to ``<run_dir>/scores/<protocol>.jsonl`` (or one combined
  file under ``out_dir``). Models load through the class the run
  recorded (``training.algos.load_run_model``). Before anything is
  scored, the best and snapshot pairs are checked against the sha256s
  that bind them (and the best model against the run record's
  verdict), every model's zip against the run's recorded algorithm,
  and checkpoints are hashed and recorded: nothing binds a checkpoint,
  and SB3's ``CheckpointCallback`` writes it in place, so a checkpoint
  torn by an interrupt fails only when it is loaded.
- ``experiments.select.select`` picks a best from those rows.

CLI: ``python -m courtside_dynamics.experiments.rescore --protocol
<id> <run_dir>... [--which best,snapshots,checkpoints] [--steps N,...]
[--out DIR]``; ``notebooks/rescore_runs.ipynb`` drives it over a Drive
runs folder.

Not here yet: the lever "audit mode" (scoring with a reward term
logged but not paid) arrives with the levers themselves (design L6).
"""

from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courtside_dynamics.experiments.protocols import (
    INLOOP,
    EvalProtocol,
    SeedBlock,
    get_protocol,
    protocol_for_run,
    resolve_seed_block,
)
from courtside_dynamics.experiments.records import (
    BINDINGS_UNBOUND,
    best_model_meta_unbound,
    read_json_object,
    sha256_file,
    temporary_path,
)
from courtside_dynamics.experiments.snapshots import (
    MODEL_FILE_PATTERN,
    SNAPSHOT_INDEX_NAME,
    SNAPSHOTS_DIR_NAME,
    normalizer_file_name,
    read_snapshot_index,
    snapshot_unbound,
    unindexed_files,
)

#: A score row's schema tag.
SCORE_SCHEMA = "courtside-score/1"

#: The directory rescored rows go to (``records.POST_TRAINING_DIRS``
#: tolerates it beside a sealed run).
SCORES_DIR_NAME = "scores"

#: The evaluation blocks a protocol defines.
BLOCKS = ("standard", "confirm")

#: The artifact kinds :func:`rescore` reads, in this order.
ARTIFACT_KINDS = ("best", "snapshots", "checkpoints")

#: The env step in a snapshot or checkpoint file name (``..._<N>_steps.zip``
#: or ``.pkl``).
_STEP_IN_NAME = re.compile(r"_(?P<step>\d+)_steps\.[a-z]+$")


@dataclass(frozen=True)
class ScoredModel:
    """One model scored on one or more blocks of a protocol."""

    metrics: Mapping[str, Mapping[str, float]]
    seed_block: SeedBlock
    algo: str


def _run_config_for(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The ``config.json`` of the run ``path`` (a model file) belongs to."""
    from courtside_dynamics.training.algos import locate_run_config

    config_path = locate_run_config(path)
    if config_path is None:
        return None
    return read_json_object(config_path, label="config.json")


def _check_observation_layout(
    protocol: EvalProtocol, run_config: Mapping[str, Any] | None, venv: Any
) -> None:
    """Refuse a protocol env whose observation layout the run did not train on.

    Same width is not the same layout (``artifacts.observation_names_sha256``);
    a run that recorded no fingerprint (before 0.25) is checked by shape only.
    """
    from courtside_dynamics.training.artifacts import (
        observation_fingerprint_mismatch,
        vec_env_observation_names,
    )

    recorded = (run_config or {}).get("env")
    if not isinstance(recorded, Mapping):
        return
    digest = recorded.get("observation_names_sha256")
    names = vec_env_observation_names(venv)
    if not isinstance(digest, str) or names is None:
        return
    recorded_names = recorded.get("observation_names")
    problem = observation_fingerprint_mismatch(
        recorded_sha256=digest,
        recorded_names=recorded_names if isinstance(recorded_names, list) else None,
        actual_names=names,
    )
    if problem is not None:
        raise ValueError(
            f"protocol {protocol.id}'s env has another observation layout than "
            f"the run trained on: {problem}"
        )


def _eval_env(protocol: EvalProtocol, normalizer_path: str | None, seed: int) -> Any:
    """The evaluation VecEnv, built as ``train()`` builds the info evaluator's.

    One worker through ``make_vec_env`` (Monitor-wrapped, seeded so the
    first reset is the block's; a paired batch reseeds every episode
    anyway), wrapped in the saved normalizer frozen as the in-loop
    evaluator's is: ``training=False``, raw rewards.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    venv = make_vec_env(protocol.make_env, n_envs=1, seed=seed)
    if normalizer_path is None:
        return venv
    try:
        normalizer = VecNormalize.load(normalizer_path, venv)
    except BaseException:
        venv.close()
        raise
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def score_model(
    model_path: str | os.PathLike[str],
    normalizer_path: str | os.PathLike[str] | None,
    protocol: EvalProtocol,
    *,
    algo: str | None = None,
    device: str = "cpu",
    blocks: Sequence[str] = ("standard",),
    seed_start: int | None = None,
    raw_observations: bool = False,
) -> ScoredModel:
    """Load ``model_path`` once and score it on each of ``blocks``.

    The seed block is :func:`~courtside_dynamics.experiments.protocols.resolve_seed_block`'s:
    ``seed_start``, the protocol's own, or the block of the run the
    model belongs to. ``device`` must be the protocol's. A policy scored
    on raw observations is not that policy, so without a normalizer:
    a model whose run recorded observation normalization is refused,
    and one outside any run (no ``config.json`` to say) is refused
    unless ``raw_observations=True`` vouches that it trained without
    ``VecNormalize`` (the zip cannot say).
    """
    from stable_baselines3.common.utils import check_for_correct_spaces

    from courtside_dynamics.callbacks.info_dict_eval import collect_episode_metrics
    from courtside_dynamics.training.algos import load_run_model, recorded_run_algo

    if device != protocol.device:
        raise ValueError(
            f"protocol {protocol.id} scores on {protocol.device!r}; a score on "
            f"{device!r} is not this protocol's (float32 kernels differ by device)"
        )
    unknown = [block for block in blocks if block not in BLOCKS]
    if unknown or not blocks:
        raise ValueError(f"blocks must be drawn from {BLOCKS}, got {list(blocks)}")
    if "confirm" in blocks and not protocol.confirm:
        raise ValueError(f"protocol {protocol.id} defines no confirmation block")
    model_path = os.fspath(model_path)
    run_config = _run_config_for(model_path)
    if normalizer_path is not None and raw_observations:
        raise ValueError(
            "raw_observations=True scores without a normalizer: drop one or "
            "the other"
        )
    if normalizer_path is None and run_config is not None:
        train_config = run_config.get("train_config")
        if isinstance(train_config, Mapping) and train_config.get("normalize_obs"):
            raise ValueError(
                f"{model_path}'s run recorded normalize_obs=True: pass its "
                f"normalizer, or the policy is scored on raw observations"
            )
    elif normalizer_path is None and not raw_observations:
        raise ValueError(
            f"{model_path} is outside any run directory (no config.json says "
            f"whether it trained on normalized observations): pass its "
            f"normalizer, or raw_observations=True (--raw-observations) for "
            f"a model trained without VecNormalize"
        )
    seeds = resolve_seed_block(protocol, run_config=run_config, seed_start=seed_start)
    model = load_run_model(model_path, algo=algo, device=device)
    resolved_algo = (recorded_run_algo(model_path) or algo or "").upper()
    metrics: dict[str, dict[str, float]] = {}
    for block in blocks:
        start = seeds.for_block(protocol, block)
        venv = _eval_env(
            protocol,
            None if normalizer_path is None else os.fspath(normalizer_path),
            start,
        )
        try:
            check_for_correct_spaces(venv, model.observation_space, model.action_space)
            _check_observation_layout(protocol, run_config, venv)
            batch = collect_episode_metrics(
                model,
                venv,
                seed_base=start if protocol.paired else None,
                **protocol.measurement_kwargs(),
            )
        finally:
            venv.close()
        metrics[block] = batch.metrics
    return ScoredModel(metrics=metrics, seed_block=seeds, algo=resolved_algo)


def evaluate(
    model_path: str | os.PathLike[str],
    normalizer_path: str | os.PathLike[str] | None,
    protocol: EvalProtocol | str,
    *,
    algo: str | None = None,
    device: str = "cpu",
    block: str = "standard",
    seed_start: int | None = None,
    raw_observations: bool = False,
) -> dict[str, float]:
    """One model's metrics row under ``protocol`` on ``block``.

    ``protocol`` is an :class:`EvalProtocol` or a registered id. The
    row is what ``InfoDictEvalCallback`` computes for an evaluation
    (its ``eval_info.csv`` row without the context metrics and the
    on-policy ``n_updates`` row). ``block`` is ``"standard"`` or
    ``"confirm"`` (the confirmation block, ``confirm_seed_offset`` on).
    See :func:`score_model` for the seed block, ``algo``, ``device`` and
    ``raw_observations``.
    """
    resolved = get_protocol(protocol) if isinstance(protocol, str) else protocol
    scored = score_model(
        model_path,
        normalizer_path,
        resolved,
        algo=algo,
        device=device,
        blocks=(block,),
        seed_start=seed_start,
        raw_observations=raw_observations,
    )
    return dict(scored.metrics[block])


# -- run artifacts ---------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifact:
    """One scorable model of a run, its normalizer and their digests.

    ``kind`` is ``"best"``, ``"snapshot"`` or ``"checkpoint"``; ``step``
    the env steps it was saved at (``None`` for a reward-selected best,
    which records no selection step). Paths are relative to the run.
    """

    kind: str
    step: int | None
    model: str
    model_sha256: str
    normalizer: str | None
    normalizer_sha256: str | None
    n_updates: int | None = None


def series_steps(run_dir: str | os.PathLike[str], kind: str) -> list[int]:
    """The env steps of a run's ``"snapshots"`` or ``"checkpoints"``, sorted.

    Read from the snapshot index or the checkpoint file names, nothing
    hashed: for choosing ``steps`` (:func:`run_artifacts`) before any
    file is read. A malformed snapshot index raises ``ValueError``.
    """
    from courtside_dynamics.training.artifacts import locate_artifact

    root = Path(run_dir)
    if kind == "snapshots":
        directory = root / SNAPSHOTS_DIR_NAME
        if not directory.is_dir():
            return []
        return sorted(entry.timestep for entry in read_snapshot_index(directory).entries)
    if kind == "checkpoints":
        found = locate_artifact(root, "checkpoints_dir")
        if found is None:
            return []
        return sorted(
            int(match.group("step"))
            for path in Path(found).iterdir()
            if (match := MODEL_FILE_PATTERN.match(path.name)) and path.is_file()
        )
    raise ValueError(f"kind must be 'snapshots' or 'checkpoints', got {kind!r}")


def _needs_normalizer(config: Mapping[str, Any]) -> bool:
    train_config = config.get("train_config")
    return isinstance(train_config, Mapping) and bool(train_config.get("normalize_obs"))


def run_artifacts(
    run_dir: str | os.PathLike[str],
    which: Iterable[str] = ARTIFACT_KINDS,
    *,
    config: Mapping[str, Any] | None = None,
    steps: Collection[int] | None = None,
) -> tuple[list[RunArtifact], list[str]]:
    """The run's scorable artifacts of the kinds in ``which``, and problems.

    ``steps`` keeps only the snapshots and checkpoints saved at those
    env steps (the best model is kept whatever its step); a step that
    names no snapshot or checkpoint of the run is a problem, so a
    subset is never smaller than asked without saying so.

    - ``best``: ``best_model.zip`` and ``best_vec_normalize.pkl`` (the
      0.14 ``model/`` layout or the flat root); its step is
      ``best_model_meta.json``'s. A meta that does not bind the pair
      (``records.best_model_meta_unbound``) is a problem, as the loaders
      refuse it, and so is a run record whose selection is ``unbound``
      (the trainer's own check, which also covers a reward-selected
      pair, whose saves write no meta) or that cannot be read.
    - ``snapshots``: ``snapshots/index.json``'s entries, each checked
      against its sha256s; a snapshot file the index does not bind, or
      a ``snapshots/`` holding files but no index (a snapshot interrupted
      between its renames), is a problem, never silently skipped.
    - ``checkpoints``: ``checkpoints/<prefix>_<N>_steps.zip`` with
      ``<prefix>_vecnormalize_<N>_steps.pkl`` (SB3's naming; the August
      runs' ``paddle_tennis_sac_<N>_steps.zip`` are these), hashed and
      recorded; nothing binds them.

    A kind the run does not have yields nothing (the August runs have
    no snapshots); a pair whose normalizer is missing although the run
    normalized observations is a problem, never a raw-observation score;
    so is a model whose zip holds a policy of another algorithm family
    than the run recorded (``algos.checkpoint_family_mismatch``: the
    load would fail inside SB3).
    """
    from courtside_dynamics.training.algos import checkpoint_family_mismatch
    from courtside_dynamics.training.artifacts import locate_artifact

    root = Path(run_dir)
    if config is None:
        config_path = locate_artifact(root, "config")
        if config_path is None:
            return [], [f"{root.name}: no config.json"]
        config = read_json_object(config_path, label=f"{root.name}/config.json")
    needs_normalizer = _needs_normalizer(config)
    recorded = config.get("train_config")
    recorded_algo = (
        recorded.get("algo") if isinstance(recorded, Mapping) else None
    )
    kinds = tuple(dict.fromkeys(which))
    unknown = [kind for kind in kinds if kind not in ARTIFACT_KINDS]
    if unknown:
        raise ValueError(f"which must be drawn from {ARTIFACT_KINDS}, got {unknown}")
    if steps is not None and not {"snapshots", "checkpoints"} & set(kinds):
        raise ValueError("steps select snapshots or checkpoints; which names neither")
    artifacts: list[RunArtifact] = []
    problems: list[str] = []
    wanted = None if steps is None else {int(step) for step in steps}
    seen_steps: set[int] = set()

    def kept(step: int) -> bool:
        """Whether a snapshot or checkpoint at ``step`` is scored."""
        if wanted is None:
            return True
        if step in wanted:
            seen_steps.add(step)
            return True
        return False

    def relative(path: str | Path) -> str:
        return Path(path).relative_to(root).as_posix()

    def pair(
        kind: str,
        step: int | None,
        model: Path,
        normalizer: Path | None,
        n_updates: int | None = None,
    ) -> None:
        if normalizer is None and needs_normalizer:
            problems.append(
                f"{root.name}: {relative(model)} has no normalizer, and the run "
                f"normalized observations"
            )
            return
        if not isinstance(recorded_algo, str) or not recorded_algo:
            problems.append(
                f"{root.name}: config.json records no train_config.algo, so "
                f"{relative(model)} cannot load through its run's class"
            )
            return
        mismatch = checkpoint_family_mismatch(
            model, recorded_algo, label=relative(model)
        )
        if mismatch is not None:
            problems.append(f"{root.name}: {mismatch}")
            return
        artifacts.append(
            RunArtifact(
                kind=kind,
                step=step,
                model=relative(model),
                model_sha256=sha256_file(model),
                normalizer=None if normalizer is None else relative(normalizer),
                normalizer_sha256=None if normalizer is None else sha256_file(normalizer),
                n_updates=n_updates,
            )
        )

    if "best" in kinds:
        model = locate_artifact(root, "best_model")
        if model is not None:
            normalizer = locate_artifact(root, "best_vec_normalize")
            meta_path = locate_artifact(root, "best_model_meta")
            step: int | None = None
            n_updates: int | None = None
            unbound: str | None = None
            if meta_path is not None:
                try:
                    meta = read_json_object(meta_path, label="best_model_meta.json")
                except ValueError as error:
                    unbound = str(error)
                else:
                    unbound = best_model_meta_unbound(meta, Path(meta_path).parent)
                    raw_step = meta.get("timestep")
                    step = raw_step if isinstance(raw_step, int) else None
                    raw_updates = meta.get("n_updates")
                    n_updates = raw_updates if isinstance(raw_updates, int) else None
            record_path = locate_artifact(root, "run_record")
            if unbound is None and record_path is not None:
                try:
                    record = read_json_object(record_path, label="run_record.json")
                except ValueError as error:
                    unbound = f"{error}, so whether it is one save is unknown"
                else:
                    selection = record.get("selection")
                    if (
                        isinstance(selection, Mapping)
                        and selection.get("bindings") == BINDINGS_UNBOUND
                    ):
                        unbound = (
                            f"the run record marks it unbound "
                            f"({selection.get('unbound_reason')})"
                        )
            if unbound is not None:
                problems.append(f"{root.name}: best model not scored: {unbound}")
            else:
                pair(
                    "best",
                    step,
                    Path(model),
                    None if normalizer is None else Path(normalizer),
                    n_updates,
                )
    if "snapshots" in kinds:
        directory = root / SNAPSHOTS_DIR_NAME
        if directory.is_dir() and not (directory / SNAPSHOT_INDEX_NAME).is_file():
            orphans = unindexed_files(directory, None)
            if orphans:
                problems.append(
                    f"{root.name}: {SNAPSHOTS_DIR_NAME}/ holds {len(orphans)} "
                    f"file(s) but no {SNAPSHOT_INDEX_NAME} (the first snapshot "
                    f"was interrupted between its renames): none is bound, so "
                    f"none is scored; rescore without snapshots"
                )
        elif directory.is_dir():
            try:
                index = read_snapshot_index(directory)
            except ValueError as error:
                problems.append(f"{root.name}: {error}")
            else:
                for name in unindexed_files(directory, index):
                    match = _STEP_IN_NAME.search(name)
                    if match is not None and not kept(int(match.group("step"))):
                        continue
                    problems.append(
                        f"{root.name}: {SNAPSHOTS_DIR_NAME}/{name} is not in "
                        f"{SNAPSHOT_INDEX_NAME} (a snapshot interrupted between "
                        f"its renames): not bound, so not scored; pass steps "
                        f"the index binds, or rescore without snapshots"
                    )
                for entry in index.entries:
                    if not kept(entry.timestep):
                        continue
                    unbound = snapshot_unbound(directory, entry)
                    if unbound is not None:
                        problems.append(f"{root.name}: {unbound}")
                        continue
                    pair(
                        "snapshot",
                        entry.timestep,
                        directory / entry.model,
                        None if entry.normalizer is None else directory / entry.normalizer,
                        entry.n_updates,
                    )
    if "checkpoints" in kinds:
        directory_path = locate_artifact(root, "checkpoints_dir")
        if directory_path is not None:
            directory = Path(directory_path)
            found = []
            for path in directory.iterdir():
                match = MODEL_FILE_PATTERN.match(path.name)
                if match and path.is_file():
                    found.append((int(match.group("step")), match.group("prefix"), path))
            for step_value, prefix, path in sorted(found):
                if not kept(step_value):
                    continue
                normalizer_path = directory / normalizer_file_name(step_value, prefix)
                pair(
                    "checkpoint",
                    step_value,
                    path,
                    normalizer_path if normalizer_path.is_file() else None,
                )
    if wanted is not None and wanted - seen_steps:
        problems.append(
            f"{root.name}: no {'/'.join(k for k in kinds if k != 'best')} "
            f"artifact at step(s) {sorted(wanted - seen_steps)}"
        )
    if not artifacts and not problems:
        problems.append(f"{root.name}: no {'/'.join(kinds)} artifact to score")
    return artifacts, problems


# -- rescoring ---------------------------------------------------------------------


def _versions() -> dict[str, str]:
    import mujoco
    import numpy
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as JSON lines, atomically (temporary file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return path


def scores_file_name(protocol: EvalProtocol) -> str:
    """``<protocol id>.jsonl``; ``inloop.jsonl`` for a run's own protocol."""
    stem = INLOOP if protocol.id.startswith(f"{INLOOP}:") else protocol.id
    return f"{stem}.jsonl"


def rescore(
    run_dirs: Iterable[str | os.PathLike[str]],
    protocol: EvalProtocol | str,
    *,
    which: Iterable[str] = ARTIFACT_KINDS,
    out_dir: str | os.PathLike[str] | None = None,
    blocks: Sequence[str] | None = None,
    device: str = "cpu",
    seed_start: int | None = None,
    steps: Collection[int] | None = None,
) -> list[Path]:
    """Score every artifact of ``run_dirs`` under ``protocol``; write the rows.

    ``protocol`` is an :class:`EvalProtocol`, a registered id, or
    ``"inloop"``: each run's own in-loop evaluation, rebuilt from its
    ``config.json`` (``protocols.protocol_for_run``). Under it a paired
    run's snapshot rows equal its ``eval_info.csv`` rows, an unpaired
    run's only at the first evaluation (see the module docstring), and a
    performance-gated run is refused: no protocol is its staged
    evaluation. ``blocks`` defaults to both
    blocks when the protocol confirms, so ``select()`` can confirm
    within block. ``steps`` scores only the snapshots and checkpoints
    saved at those env steps, besides the best model (a quick look; see
    :func:`run_artifacts`). Every run's artifacts are enumerated and
    checked against their digests before anything is scored, and any
    problem raises ``ValueError`` naming all of them, so a rescore never
    writes a partial set silently.

    Rows go to ``<run_dir>/scores/<protocol>.jsonl`` (one file per run,
    rewritten whole), or, with ``out_dir``, to one
    ``<out_dir>/scores/<protocol>.jsonl`` holding every run's rows. Each
    row: the run id, the artifact (kind, step, relative paths and
    sha256s, SB3 update count), the protocol id and fingerprint, the
    seed block, the algorithm, the device and library versions, and
    ``metrics`` (standard block) / ``confirm_metrics`` (or ``None``).
    Returns the files written.
    """
    roots = [Path(run_dir) for run_dir in run_dirs]
    if not roots:
        raise ValueError("rescore needs at least one run directory")
    kinds = tuple(which)
    planned: list[tuple[Path, EvalProtocol, list[RunArtifact]]] = []
    problems: list[str] = []
    for root in roots:
        try:
            if isinstance(protocol, EvalProtocol):
                resolved = protocol
            elif protocol == INLOOP:
                resolved = protocol_for_run(root)
            else:
                resolved = get_protocol(protocol)
        except (OSError, ValueError) as error:
            problems.append(f"{root.name}: {error}")
            continue
        if blocks is not None and (
            "standard" not in blocks
            or any(block not in BLOCKS for block in blocks)
            or ("confirm" in blocks and not resolved.confirm)
        ):
            problems.append(
                f"{root.name}: blocks {list(blocks)} must hold 'standard' and "
                f"only blocks of protocol {resolved.id} "
                f"({'standard, confirm' if resolved.confirm else 'standard'})"
            )
        artifacts, found = run_artifacts(root, kinds, steps=steps)
        problems.extend(found)
        planned.append((root, resolved, artifacts))
    if problems:
        raise ValueError(
            "rescore refused before scoring anything:\n  " + "\n  ".join(problems)
        )
    versions = _versions()
    written: list[Path] = []
    combined: list[dict[str, Any]] = []
    combined_protocol: EvalProtocol | None = None
    for root, resolved, artifacts in planned:
        run_blocks = tuple(
            blocks
            if blocks is not None
            else (BLOCKS if resolved.confirm else ("standard",))
        )
        rows: list[dict[str, Any]] = []
        for artifact in artifacts:
            scored = score_model(
                root / artifact.model,
                None if artifact.normalizer is None else root / artifact.normalizer,
                resolved,
                device=device,
                blocks=run_blocks,
                seed_start=seed_start,
            )
            rows.append(
                {
                    "schema": SCORE_SCHEMA,
                    "run_id": root.resolve().name,
                    "artifact": artifact.kind,
                    "step": artifact.step,
                    "n_updates": artifact.n_updates,
                    "model": {"path": artifact.model, "sha256": artifact.model_sha256},
                    "normalizer": (
                        None
                        if artifact.normalizer is None
                        else {
                            "path": artifact.normalizer,
                            "sha256": artifact.normalizer_sha256,
                        }
                    ),
                    "algo": scored.algo,
                    "protocol": resolved.id,
                    "protocol_fingerprint": resolved.fingerprint,
                    "device": device,
                    "seed_start": scored.seed_block.start,
                    "seed_source": scored.seed_block.source,
                    "versions": versions,
                    "metrics": dict(scored.metrics["standard"]),
                    "confirm_metrics": (
                        dict(scored.metrics["confirm"])
                        if "confirm" in scored.metrics
                        else None
                    ),
                }
            )
        if out_dir is None:
            written.append(
                _write_rows(root / SCORES_DIR_NAME / scores_file_name(resolved), rows)
            )
        else:
            combined.extend(rows)
            combined_protocol = combined_protocol or resolved
    if out_dir is not None:
        assert combined_protocol is not None
        written.append(
            _write_rows(
                Path(out_dir) / SCORES_DIR_NAME / scores_file_name(combined_protocol),
                combined,
            )
        )
    return written


def read_scores(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The rows of a ``scores/<protocol>.jsonl`` file."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != SCORE_SCHEMA:
                raise ValueError(f"{path}:{number} is not a {SCORE_SCHEMA} row")
            rows.append(row)
    return rows
