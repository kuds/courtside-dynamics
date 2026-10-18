"""Check run directories against their sync manifests.

Usage::

    python -m courtside_dynamics.experiments.verify <run_dir> [<run_dir> ...]
    python -m courtside_dynamics.experiments.verify --strict --json <run_dir>
    python -m courtside_dynamics.experiments.verify --allow-warnings <run_dir>

Prints one verdict per directory (``--json``: one JSON object per line,
with ``complete`` and ``clean``) and exits 0 only when every directory
is complete and clean
(:func:`~courtside_dynamics.experiments.records.verify_run_dir`). A file
the manifest does not list fails the directory unless it is a
post-training plot, audit, score file or replay video (``reports/``,
``scores/``, ``media/best_model.mp4``); ``--strict`` fails those too. A
writer's temporary leftover (``.x.tmp-<pid>``, ``x.tmp.npz`` ...) fails
it whether the manifest lists it or not. What the sealed run record
reports wrong inside a complete directory (an unbound best model,
snapshot files the index does not bind) makes it complete but not
clean: listed, and failed unless ``--allow-warnings`` accepts it (a
faithful copy of a damaged run is still exactly what ``train()`` left);
``--strict`` counts it as incomplete.

Exit status: 0 every directory complete and clean (or, with
``--allow-warnings``, complete); 1 a directory is incomplete; 2 every
directory is complete but one is not clean.
"""

from __future__ import annotations

import argparse
import json
import sys

from courtside_dynamics.experiments.records import RunDirVerification, verify_run_dir

#: Exit status when every directory is complete but one carries warnings.
EXIT_NOT_CLEAN = 2


def _describe(result: RunDirVerification, *, allow_warnings: bool) -> list[str]:
    verified = (
        f"status {result.status}, {result.files_checked} file(s) verified"
    )
    if result.complete and (result.clean or allow_warnings):
        lines = [f"{result.run_dir}: complete ({verified})"]
        lines.extend(f"  WARNING: {warning}" for warning in result.warnings)
    elif result.complete:
        lines = [
            f"{result.run_dir}: NOT CLEAN -- complete ({verified}), but its "
            f"run record reports damage inside it (--allow-warnings accepts "
            f"a faithful copy of a damaged run):"
        ]
        lines.extend(f"  - {warning}" for warning in result.warnings)
    else:
        lines = [f"{result.run_dir}: INCOMPLETE"]
        lines.extend(f"  - {problem}" for problem in result.problems)
        lines.extend(f"  WARNING: {warning}" for warning in result.warnings)
    if result.unlisted:
        lines.append(
            f"  {len(result.unlisted)} file(s) not in the manifest: "
            + ", ".join(result.unlisted[:10])
            + (" ..." if len(result.unlisted) > 10 else "")
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=(
            "exit status: 0 every directory complete and clean; 1 one is "
            "incomplete; 2 every one is complete but one is not clean (its "
            "run record reports an unbound best model or unindexed snapshot "
            "files) and --allow-warnings was not given"
        ),
    )
    parser.add_argument("run_dirs", nargs="+", help="run directories to check")
    leniency = parser.add_mutually_exclusive_group()
    leniency.add_argument(
        "--strict",
        action="store_true",
        help=(
            "also fail post-training files (reports/, scores/, "
            "media/best_model.mp4) the manifest does not list, and count what "
            "the run record reports wrong (an unbound best model, unindexed "
            "snapshot files) as incomplete"
        ),
    )
    leniency.add_argument(
        "--allow-warnings",
        action="store_true",
        help=(
            "accept a complete directory whose run record reports damage "
            "inside it (an unbound best model, unindexed snapshot files): "
            "print it as WARNING lines and exit 0, for checking a faithful "
            "copy of a damaged run"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="one JSON verdict per line"
    )
    args = parser.parse_args(argv)
    every_complete = True
    every_clean = True
    for run_dir in args.run_dirs:
        result = verify_run_dir(run_dir, strict=args.strict)
        every_complete = every_complete and result.complete
        every_clean = every_clean and result.clean
        if args.json:
            print(json.dumps(result.to_dict(), sort_keys=True))
        else:
            print("\n".join(_describe(result, allow_warnings=args.allow_warnings)))
    if not every_complete:
        return 1
    if not every_clean and not args.allow_warnings:
        return EXIT_NOT_CLEAN
    return 0


if __name__ == "__main__":
    sys.exit(main())
