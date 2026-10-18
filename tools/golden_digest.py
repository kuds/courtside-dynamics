"""Golden env-stream digests: check them (default) or re-record them.

The gate every env/rules refactor must pass (design M1 + L0): each case
in :mod:`courtside_dynamics.digests` is rolled and its stream and info
digests compared with ``tests/golden/env_digests.json``.

- ``--check`` (the default) exits 1 on any mismatch, printing the case,
  the expected and actual values, the versions the golden was recorded
  under and the versions running now. A mismatch is an era break or
  physics drift that must be declared -- never re-recorded silently.
- ``--write`` re-records the selected set under the running versions
  and prints, case by case, what changed against the file it replaces.
  A case with a declared drift slot for the running MuJoCo
  (``digests.DECLARED_DRIFT``) is written into that slot; its primary
  digest must be recorded under a MuJoCo without declared drift.
- ``--full`` selects the larger local set (five spaced seeds per case;
  two for the humanoid case, and the NaN witnesses keep their one
  calibrated seed) instead of the CI subset; ``--case NAME``
  (repeatable) narrows either.

The tool refuses to run when ``courtside_dynamics`` imports from
anywhere but this checkout's ``src/`` (a git worktree shares the main
checkout's editable install): set ``PYTHONPATH=src`` there.

Usage::

    OMP_NUM_THREADS=1 python tools/golden_digest.py            # CI set, check
    OMP_NUM_THREADS=1 python tools/golden_digest.py --full     # local gate
    OMP_NUM_THREADS=1 python tools/golden_digest.py --write [--full]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools._checkout import checkout_package_mismatch
except ModuleNotFoundError:  # run as a script: tools/ itself is on sys.path
    from _checkout import checkout_package_mismatch  # type: ignore[no-redef]

from courtside_dynamics.digests import (
    CASE_SETS,
    DECLARED_DRIFT,
    MISMATCH_EXPLANATION,
    StreamCase,
    digest_mismatches,
    drift_label,
    expected_digest,
    format_mismatch,
    stream_digest,
    version_info,
)

#: The checkout this tool belongs to.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The committed golden file.
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "env_digests.json"

#: Format tag of the golden file; bump it with any layout or hashing
#: change. /2: every info value hashed (non-scalars recursively), every
#: info key in the census, plus interface_sha256 and, for PaddleTennis,
#: reward_order_sensitive_steps. stream_sha256 is unchanged from /1.
GOLDEN_FORMAT = "courtside-env-digests/2"


def load_golden(path: Path) -> dict[str, Any]:
    """Read a golden file, refusing an unknown format loudly."""
    golden = json.loads(path.read_text(encoding="utf-8"))
    if golden.get("format") != GOLDEN_FORMAT:
        raise ValueError(
            f"{path}: format {golden.get('format')!r}, expected {GOLDEN_FORMAT!r}"
        )
    return golden


def _select(set_name: str, names: list[str] | None) -> tuple[StreamCase, ...]:
    cases = CASE_SETS[set_name]
    if not names:
        return cases
    known = {case.name: case for case in cases}
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise SystemExit(
            f"golden_digest: unknown case(s) {unknown} in set {set_name!r}; "
            f"choose from {sorted(known)}"
        )
    return tuple(known[name] for name in names)


def check(golden: dict[str, Any], set_name: str, cases: tuple[StreamCase, ...]) -> int:
    """Roll and compare ``cases``; print a report; return the exit code."""
    live = version_info()
    recorded = golden["sets"].get(set_name, {})
    failures: list[str] = []
    matched = 0
    started = time.monotonic()
    for case in cases:
        tick = time.monotonic()
        entry = recorded.get(case.name)
        if entry is None:
            failures.append(
                f"GOLDEN DIGEST MISSING: {set_name}/{case.name} has no "
                f"recorded digest; record it deliberately with --write."
            )
            print(f"MISSING   {case.name}")
            continue
        expected, versions, drift = expected_digest(entry, case.name, live)
        actual = stream_digest(case)
        lines = digest_mismatches(expected, actual)
        took = time.monotonic() - tick
        note = f"  [declared drift: {drift}]" if drift else ""
        if lines:
            failures.append(
                format_mismatch(set_name, case.name, lines, versions, live, drift)
            )
            print(f"MISMATCH  {case.name}  ({took:.1f}s){note}")
        else:
            matched += 1
            print(f"ok        {case.name}  ({took:.1f}s){note}")
    if not failures and len(cases) == len(CASE_SETS[set_name]):
        stale = sorted(set(recorded) - {case.name for case in cases})
        if stale:
            failures.append(
                f"GOLDEN DIGEST STALE: {set_name} records {stale}, which no "
                f"case defines any more; re-record the set deliberately."
            )
    print(
        f"{set_name}: {matched}/{len(cases)} cases match "
        f"in {time.monotonic() - started:.1f}s under {live}"
    )
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


def write(
    golden: dict[str, Any] | None,
    path: Path,
    set_name: str,
    cases: tuple[StreamCase, ...],
    *,
    whole_set: bool,
) -> int:
    """Re-record ``cases`` under the running versions; report every change.

    An entry whose digest reproduces exactly is left untouched, versions
    included, so a re-record never churns provenance: ``versions`` names
    the environment that last *changed* the entry.
    """
    live = version_info()
    label = drift_label(live)
    golden = golden or {"format": GOLDEN_FORMAT, "sets": {}}
    golden["note"] = MISMATCH_EXPLANATION
    old_set: dict[str, Any] = dict(golden["sets"].get(set_name, {}))
    new_set: dict[str, Any] = {} if whole_set else dict(old_set)
    for case in cases:
        digest = stream_digest(case)
        old = old_set.get(case.name, {})
        entry = dict(old)
        reason = DECLARED_DRIFT.get(label, {}).get(case.name)
        if reason is not None:
            slots = dict(old.get("declared_drift", {}))
            previous = slots.get(label)
            where = f"drift slot {label!r}"
        else:
            previous = old if "stream_sha256" in old else None
            where = "primary"
        changed = (
            None
            if previous is None
            else digest_mismatches(
                {k: v for k, v in previous.items() if k in digest}, digest
            )
        )
        if changed is None or changed:
            if reason is not None:
                slots[label] = {"reason": reason, **digest, "versions": live}
                entry["declared_drift"] = slots
            else:
                entry.update(digest, versions=live)
        new_set[case.name] = entry
        if changed is None:
            status = "new"
        elif changed:
            fields = ", ".join(line.split(":")[0] for line in changed)
            status = f"CHANGED ({fields})"
        else:
            status = "unchanged"
        print(f"{status:<32} {case.name}  [{where}]")
        if "stream_sha256" not in entry:
            print(
                f"  warning: {case.name} has no primary digest yet; record it "
                f"under a MuJoCo version without a declared drift slot",
                file=sys.stderr,
            )
    if whole_set:
        for name in sorted(set(old_set) - set(new_set)):
            print(f"{'REMOVED':<32} {name}")
    golden["sets"][set_name] = dict(sorted(new_set.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {set_name} set to {path} under {live}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare (default)")
    mode.add_argument("--write", action="store_true", help="re-record")
    parser.add_argument(
        "--full", action="store_true", help="use the full local set, not the CI set"
    )
    parser.add_argument(
        "--case", action="append", dest="cases", metavar="NAME",
        help="only this case (repeatable)",
    )
    parser.add_argument(
        "--golden", type=Path, default=GOLDEN_PATH, help="golden file to use"
    )
    args = parser.parse_args(argv)
    problem = checkout_package_mismatch(REPO_ROOT)
    if problem is not None:
        raise SystemExit(f"golden_digest: {problem}")
    set_name = "full" if args.full else "ci"
    cases = _select(set_name, args.cases)
    if args.write:
        golden = load_golden(args.golden) if args.golden.exists() else None
        return write(golden, args.golden, set_name, cases, whole_set=not args.cases)
    return check(load_golden(args.golden), set_name, cases)


if __name__ == "__main__":
    sys.exit(main())
