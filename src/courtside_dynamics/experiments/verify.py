"""Check run directories against their sync manifests.

Usage::

    python -m courtside_dynamics.experiments.verify <run_dir> [<run_dir> ...]
    python -m courtside_dynamics.experiments.verify --strict --json <run_dir>

Prints one verdict per directory (``--json``: one JSON object per line)
and exits 0 only when every directory is complete
(:func:`~courtside_dynamics.experiments.records.verify_run_dir`), 1
otherwise. ``--strict`` also fails files the manifest does not list.
"""

from __future__ import annotations

import argparse
import json
import sys

from courtside_dynamics.experiments.records import verify_run_dir


def _describe(result) -> list[str]:
    if result.complete:
        lines = [
            f"{result.run_dir}: complete (status {result.status}, "
            f"{result.files_checked} file(s) verified)"
        ]
    else:
        lines = [f"{result.run_dir}: INCOMPLETE"]
        lines.extend(f"  - {problem}" for problem in result.problems)
    if result.unlisted:
        lines.append(
            f"  {len(result.unlisted)} file(s) not in the manifest (written "
            f"after it): " + ", ".join(result.unlisted[:10])
            + (" ..." if len(result.unlisted) > 10 else "")
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="run directories to check")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail files the manifest does not list",
    )
    parser.add_argument(
        "--json", action="store_true", help="one JSON verdict per line"
    )
    args = parser.parse_args(argv)
    every_complete = True
    for run_dir in args.run_dirs:
        result = verify_run_dir(run_dir, strict=args.strict)
        every_complete = every_complete and result.complete
        if args.json:
            print(json.dumps(result.to_dict(), sort_keys=True))
        else:
            print("\n".join(_describe(result)))
    return 0 if every_complete else 1


if __name__ == "__main__":
    sys.exit(main())
