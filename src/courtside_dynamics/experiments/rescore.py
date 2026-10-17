"""Re-score runs' saved policies under an evaluation protocol (CLI).

``python -m courtside_dynamics.experiments.rescore --protocol <id>
<run_dir>... [--which best,snapshots,checkpoints] [--steps N,...]
[--out DIR]`` runs
``experiments.harness.rescore``: every best model, evaluation snapshot
and periodic checkpoint of each run (the August runs' legacy
``checkpoints/`` layout included) is scored under the protocol on CPU,
and the rows go to ``<run_dir>/scores/<protocol>.jsonl`` (or one
combined ``<DIR>/scores/<protocol>.jsonl``). ``--protocol inloop``
re-derives each run's own in-loop evaluation from its ``config.json``:
a paired run's snapshot rows must equal its ``eval_info.csv``, an
unpaired run's only at the first evaluation (the summary says so), and
a performance-gated run is refused (no protocol is its staged
evaluation).

Each run's summary lists the scored artifacts with the protocol's
selection keys, and, when the snapshots were scored, any snapshot the
run failed to save (a gap in the series) and the offline ``select()``
pick next to the run's in-loop best (``best_model_meta.json``). Exit
status 0, or 2 when the rescore is refused (an unbound pair, a missing
normalizer, an unreadable model zip, an unknown protocol, an in-loop
protocol of another device).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from courtside_dynamics.experiments.harness import (
    ARTIFACT_KINDS,
    BLOCKS,
    read_scores,
    rescore,
    snapshot_gaps,
)
from courtside_dynamics.experiments.protocols import (
    INLOOP,
    get_protocol,
    protocol_for_run,
    registered_protocols,
)
from courtside_dynamics.experiments.select import select


def _csv(value: str, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [item for item in items if item not in allowed]
    if unknown or not items:
        raise argparse.ArgumentTypeError(
            f"{name} takes a comma-separated subset of {', '.join(allowed)}"
        )
    return items


def _steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError:
        steps = ()
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError(
            "--steps takes comma-separated nonnegative env steps"
        )
    return steps


def _in_loop_best(run_dir: Path) -> Any:
    from courtside_dynamics.experiments.records import read_json_object
    from courtside_dynamics.training.artifacts import locate_artifact

    meta = locate_artifact(run_dir, "best_model_meta")
    if meta is None:
        return None
    try:
        return read_json_object(meta).get("timestep")
    except ValueError:
        return None


def _summarize(rows: Sequence[Mapping[str, Any]], run_dir: Path | None) -> None:
    if not rows:
        return
    first = rows[0]
    print(
        f"{first['run_id']}: {len(rows)} artifact(s) under {first['protocol']} "
        f"({first['protocol_fingerprint'][:12]}), seeds from "
        f"{first['seed_start']} ({first['seed_source']})"
    )
    if str(first["protocol"]).startswith(f"{INLOOP}:"):
        protocol = protocol_for_run(run_dir) if run_dir is not None else None
    else:
        protocol = get_protocol(str(first["protocol"]))
    if protocol is not None and str(first["protocol"]).startswith(f"{INLOOP}:"):
        if not protocol.paired:
            print(
                "  note: the run evaluated unpaired, so only its first "
                "evaluation's row reproduces eval_info.csv (the in-loop stream "
                "continues across evaluations; the confirmation block has no "
                "in-loop counterpart)"
            )
    keys = () if protocol is None else protocol.selection_keys
    rule = protocol.selection_rule() if protocol is not None and keys else None
    for row in rows:
        values = ", ".join(
            f"{key}={row['metrics'][key]:.4g}" for key in keys if key in row["metrics"]
        )
        print(f"  {row['artifact']:<10} step {row['step']!s:>10}  {values}")
    snapshots = [row for row in rows if row["artifact"] == "snapshot"]
    if snapshots and run_dir is not None:
        for gap in snapshot_gaps(run_dir):
            print(f"  note: {gap}")
    if rule is not None and snapshots:
        try:
            chosen = select(snapshots, rule)
        except ValueError as error:
            print(f"  select(): {error}")
        else:
            in_loop = _in_loop_best(run_dir) if run_dir is not None else None
            print(
                f"  select() over the snapshots: step {chosen.step}"
                + (f"; in-loop best_model: step {in_loop}" if in_loop is not None else "")
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m courtside_dynamics.experiments.rescore",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help=f"a registered protocol ({', '.join(registered_protocols())}) or "
        f"'{INLOOP}' (each run's own in-loop evaluation)",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run directories")
    parser.add_argument(
        "--which",
        type=lambda value: _csv(value, ARTIFACT_KINDS, "--which"),
        default=ARTIFACT_KINDS,
        help="artifact kinds to score (default: all)",
    )
    parser.add_argument(
        "--blocks",
        type=lambda value: _csv(value, BLOCKS, "--blocks"),
        default=None,
        help="evaluation blocks (default: both when the protocol confirms)",
    )
    parser.add_argument(
        "--steps",
        type=_steps,
        default=None,
        help="score only the snapshots and checkpoints at these env steps "
        "(comma-separated; the best model is scored as well)",
    )
    parser.add_argument("--out", type=Path, help="write one combined file here")
    parser.add_argument("--seed-start", type=int, help="override the seed block")
    args = parser.parse_args(argv)
    try:
        written = rescore(
            args.run_dirs,
            args.protocol,
            which=args.which,
            out_dir=args.out,
            blocks=args.blocks,
            seed_start=args.seed_start,
            steps=args.steps,
        )
    except (OSError, KeyError, ValueError) as error:
        print(f"rescore: {error}", file=sys.stderr)
        return 2
    for path in written:
        rows = read_scores(path)
        if args.out is None:
            _summarize(rows, path.parent.parent)
        else:
            for run_dir in args.run_dirs:
                name = run_dir.resolve().name
                _summarize([row for row in rows if row["run_id"] == name], run_dir)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
