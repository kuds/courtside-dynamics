"""Refuse to gate one checkout's goldens against another checkout's code.

The editable install points ``courtside_dynamics`` at the checkout that
ran ``pip install -e``. In a ``git worktree`` (or a second clone sharing
that environment) the package therefore imports from the *other*
checkout's ``src/``, while the golden files, tools and tests come from
this one: a gate run there compares this checkout's goldens with the
other checkout's code, and a real change made here passes silently. The
golden-digest and recipe-snapshot tools and the test suite call
:func:`checkout_package_mismatch` first and stop loudly instead;
``tools/trainer_digest.py`` makes the same check per job.
"""
from __future__ import annotations

from pathlib import Path


def checkout_package_mismatch(repo_root: Path) -> str | None:
    """Why ``courtside_dynamics`` is not ``repo_root``'s ``src/`` copy, if so.

    Returns ``None`` when the imported package lives under
    ``repo_root/src``, else a message naming both paths and the fix.
    """
    import courtside_dynamics

    package = Path(courtside_dynamics.__file__).resolve().parent
    src = (repo_root / "src").resolve()
    if package == src / "courtside_dynamics":
        return None
    return (
        f"courtside_dynamics was imported from {package}, not from this "
        f"checkout's {src}: its goldens would be checked against another "
        f"checkout's code, and a change made here would pass silently. "
        f"Run with PYTHONPATH={src} (a git worktree shares the main "
        f"checkout's editable install), or reinstall with "
        f"`pip install -e {repo_root.resolve()}`."
    )
