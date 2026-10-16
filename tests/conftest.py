"""Suite-wide pytest configuration.

No fixtures yet -- shared plain-function helpers (ball-state
teleports, contact inspection, event-batch injection, the
``get_env``-shaped fake model) live in :mod:`tests._helpers`, imported
explicitly by the files that use them so each test's dependencies stay
visible at its imports. Add fixtures here only when they need pytest's
lifecycle (caching, teardown, parametrization); a helper a test calls
with arguments belongs in ``_helpers``.

The one hook here refuses to run the suite against another checkout's
package: in a git worktree the editable install still imports
``courtside_dynamics`` from the main checkout, so the golden gates
would compare this checkout's goldens with that checkout's code and a
real change made here would pass (``tools/_checkout.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools._checkout import checkout_package_mismatch

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config: pytest.Config) -> None:
    problem = checkout_package_mismatch(REPO_ROOT)
    if problem is not None:
        raise pytest.UsageError(problem)
